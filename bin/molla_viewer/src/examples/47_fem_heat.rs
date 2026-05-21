// of the unit cube is held by `FemHeatExampleState`; each frame we
// step backward-Euler once via `molla::fem::solve_transient`. The
// tet wireframe is drawn as gizmo lines coloured by the per-vertex
// temperature (blue → red).
//
// The scene runs entirely outside the standard solver pipeline —
// `step_simulation` short-circuits on `Scene::FemHeat`, and
// `build_fem_heat_model` returns an empty `Model` with no bodies or
// particles.

const FEM_HEAT_GRID: usize = 16;
const FEM_HEAT_DEFAULT_ALPHA: f64 = 1.0;
const FEM_HEAT_DEFAULT_DT: f64 = 0.01;
// Origin offset for the unit cube — keep it just above the ground
// plane so the gizmo wireframe doesn't clip the grid.
const FEM_HEAT_ORIGIN: bevy::math::Vec3 = bevy::math::Vec3::new(-0.5, 0.05, -0.5);

// Persistent runtime state for `Scene::FemHeat`. Built lazily on
// first entry into the scene; `apply_scene_switch` stamps
// `needs_reset` so re-entering re-initialises `u` to zero.
#[derive(Resource)]
struct FemHeatExampleState {
    mesh: fem::FemMesh,
    /// Mass matrix M (consistent, P1).
    mass: molla::math::sparse::BsrMatrix,
    /// Stiffness matrix K = α · ∫ ∇φᵢ·∇φⱼ dx. Rebuilt whenever `α`
    /// changes via the right-rail slider.
    stiff: molla::math::sparse::BsrMatrix,
    /// Current α — when this drifts from `last_alpha_baked` we
    /// rebuild `stiff` next frame.
    alpha: f64,
    last_alpha_baked: f64,
    /// Backward-Euler step.
    dt: f64,
    /// Dirichlet BCs (top + bottom faces).
    bcs: Vec<fem::BoundaryCondition>,
    /// Current per-vertex temperature.
    u: Vec<f64>,
    /// Last CG residual (for the readout).
    last_residual: f64,
    /// Last CG iteration count (for the readout).
    last_iters: usize,
    /// Set by `apply_scene_switch` and cleared by the update system on
    /// re-init.
    needs_reset: bool,
    /// Ribbon-button-driven flag — `true` means the user clicked
    /// "Reset to T=0" on the right-rail pane. Cleared after the
    /// solver re-initialises.
    user_reset: bool,
    /// GPU stepper — `Some` once the user has enabled GPU mode.
    /// Lazily built (Default has no Engine access). Invalidated
    /// whenever α or dt change.
    gpu_stepper: Option<fem::TransientStepperGpu>,
    use_gpu: bool,
    last_dt_baked: f64,
}

impl Default for FemHeatExampleState {
    fn default() -> Self {
        let mesh =
            fem::FemMesh::tet_grid_3d(FEM_HEAT_GRID, FEM_HEAT_GRID, FEM_HEAT_GRID, 1.0, 1.0, 1.0);
        let space = fem::FunctionSpace::P1Nodal;
        let mass = fem::assemble_global(&mesh, space, fem::compute_element_mass)
            .expect("FEM heat: assemble mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, FEM_HEAT_DEFAULT_ALPHA)
        })
        .expect("FEM heat: assemble stiffness");
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.z.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            } else if (p.z - 1.0).abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 1.0,
                });
            }
        }
        let n = mesh.vert_count();
        let mut u = vec![0.0_f64; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh,
            mass,
            stiff,
            alpha: FEM_HEAT_DEFAULT_ALPHA,
            last_alpha_baked: FEM_HEAT_DEFAULT_ALPHA,
            dt: FEM_HEAT_DEFAULT_DT,
            bcs,
            u,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: false,
            user_reset: false,
            gpu_stepper: None,
            use_gpu: false,
            last_dt_baked: FEM_HEAT_DEFAULT_DT,
        }
    }
}

impl FemHeatExampleState {
    fn rebuild_stiffness(&mut self) {
        let space = fem::FunctionSpace::P1Nodal;
        self.stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, self.alpha)
        })
        .expect("FEM heat: rebuild stiffness");
        self.last_alpha_baked = self.alpha;
    }

    fn reset_temperature_field(&mut self) {
        let n = self.mesh.vert_count();
        self.u = vec![0.0_f64; n];
        for bc in &self.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                self.u[*vertex as usize] = *value;
            }
        }
        self.last_residual = 0.0;
        self.last_iters = 0;
    }
}

// Empty molla model — the FEM heat demo carries its own mesh +
// solver in `FemHeatExampleState`; the standard pipeline is
// short-circuited in `step_simulation`.
fn build_fem_heat_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM heat: empty model"), Vec::new())
}

// Per-frame system: rebuilds the stiffness matrix when α has
// changed, applies any pending reset, and runs one backward-Euler
// step.
fn update_fem_heat_example(
    sim: Res<Sim>,
    mut state: ResMut<FemHeatExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::FemHeat {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_temperature_field();
        state.needs_reset = false;
        state.user_reset = false;
        // Push the fresh u to the GPU stepper if it's already built.
        // Clone first so we can split the borrow.
        let u_snapshot = state.u.clone();
        if let Some(s) = state.gpu_stepper.as_mut() {
            if let Err(e) = s.write_u(&u_snapshot) {
                warn!("FemHeat: GPU write_u after reset failed ({e})");
                state.gpu_stepper = None;
            }
        }
    }
    let alpha_changed = (state.alpha - state.last_alpha_baked).abs() > 1e-12;
    let dt_changed = (state.dt - state.last_dt_baked).abs() > 1e-9;
    if alpha_changed {
        state.rebuild_stiffness();
        state.gpu_stepper = None;
    }
    if dt_changed {
        state.gpu_stepper = None;
        state.last_dt_baked = state.dt;
    }
    if run.paused {
        return;
    }
    let dt = state.dt.max(1e-6);
    let n = state.mesh.vert_count();
    let f = vec![0.0_f64; n];

    // Lazily build GPU stepper if requested.
    if state.use_gpu && state.gpu_stepper.is_none() {
        match fem::TransientStepperGpu::new(
            &sim.engine,
            &state.mesh,
            &state.mass,
            &state.stiff,
            dt,
            &state.bcs,
            &state.u,
        ) {
            Ok(s) => state.gpu_stepper = Some(s),
            Err(e) => {
                warn!("FemHeat: GPU stepper init failed ({e}); falling back to CPU");
                state.use_gpu = false;
            }
        }
    }

    if state.use_gpu && state.gpu_stepper.is_some() {
        // Slice H — `step` updates the GPU-resident `u` in place;
        // read it back per render frame so the gizmo system sees
        // current data. The source `f` is zero in this scene; we
        // skip `write_f` because the constructor already zeroed it.
        let _ = f; // keep the binding live so future non-zero f flows compile
        let stepper = state.gpu_stepper.as_mut().expect("checked above");
        let step_result = stepper.step();
        let read_result = match step_result {
            Ok(stats) => match stepper.read_u() {
                Ok(u) => Ok((stats, u)),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match read_result {
            Ok((stats, u)) => {
                state.last_residual = stats.residual;
                state.last_iters = stats.iters;
                state.u = u;
                return;
            }
            Err(e) => {
                warn!("FemHeat: GPU step/read failed ({e}); falling back to CPU this frame");
            }
        }
    }
    let result = molla::fem::solver::solve_transient_with_initial(
        &state.mass,
        &state.stiff,
        &f,
        &state.u.clone(),
        dt,
        &state.bcs,
        &state.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    state.u = result.x;
    state.last_residual = result.residual;
    state.last_iters = result.iters;
}

// Map a temperature value `t ∈ [0, 1]` to a colour (blue → cyan →
// yellow → red). Out-of-range values are clamped.
fn fem_heat_colour(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0) as f32;
    // Two-segment ramp: 0..0.5 = blue → green-ish, 0.5..1 = → red.
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

// Draw the FEM heat-equation tet wireframe with per-edge colour
// derived from the average of its endpoints' temperatures.
fn draw_fem_heat_gizmos(sim: Res<Sim>, state: Res<FemHeatExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::FemHeat {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 4);
    let origin = FEM_HEAT_ORIGIN;
    for cell in &state.mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let pa = state.mesh.vertices[a as usize];
                let pb = state.mesh.vertices[b as usize];
                let t_mid = 0.5 * (state.u[a as usize] + state.u[b as usize]);
                let colour = fem_heat_colour(t_mid);
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                gizmos.line(av, bv, colour);
            }
        }
    }
}

// Right-rail pane for `Scene::FemHeat`. Sliders for diffusivity α +
// timestep dt, residual readout, and the "Reset to T=0" button.
fn fem_heat_pane(ui: &mut egui::Ui, state: &mut FemHeatExampleState) {
    ui.label("FEM heat equation — Stage 36 v1");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            FEM_HEAT_GRID,
            state.mesh.vert_count()
        ),
    );
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    ui.separator();
    ui.label("Thermal diffusivity α");
    let _ = ui.add(egui::Slider::new(&mut state.alpha, 0.05..=10.0).logarithmic(true));
    ui.label("Time step dt");
    let _ = ui.add(egui::Slider::new(&mut state.dt, 0.001..=0.1).logarithmic(true));
    ui.separator();
    readout(ui, "Last residual", &format!("{:.3e}", state.last_residual));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    ui.separator();
    if ui.button("Reset to T = 0").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU CG (Stage 36.5 Slice F)");
    if was_gpu != state.use_gpu {
        state.gpu_stepper = None;
    }
    let backend = if state.use_gpu && state.gpu_stepper.is_some() {
        "GPU resident CG"
    } else if state.use_gpu {
        "GPU (init pending / failed)"
    } else {
        "CPU"
    };
    readout(ui, "Backend", backend);
    ui.small("BCs: T(z=0) = 0, T(z=1) = 1; sides insulating.");
    ui.small("Steady state: T(z) = z (linear gradient).");
}


