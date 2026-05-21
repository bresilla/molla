// ─── Stage 36.4 — Differentiable FEM beam ─────────────────────────────
//
// Same beam geometry as `Scene::FemBeam`, but instead of just solving
// K · u = f for a fixed material, the scene runs gradient descent on
// Young's modulus E to drive the tip-deflection toward a user-supplied
// target. Each frame:
//   1. Forward-solve the linear elasticity system at the current E.
//   2. Read the tip deflection.
//   3. Compute the loss `L = ½ (δ − δ_target)²`.
//   4. Adjoint-solve `K · λ = ∂L/∂u` (the seed has a single non-zero
//      component, the average of (δ − δ_target) / N_tip across the
//      tip-face DoFs).
//   5. Compute `∂L/∂E = compute_grad_youngs(...)`, step
//      `E ← E − η · ∂L/∂E`.
//
// Convergence is fastest for an adaptive step ∝ E²/L; we keep a
// fixed-but-relative `lr_factor` slider so the user can play.

#[derive(Resource)]
struct FemDiffBeamExampleState {
    mesh: fem::FemMesh,
    bcs: Vec<fem::BoundaryCondition>,
    /// Tip-face vertex indices — used to read the tip deflection +
    /// build the adjoint seed.
    tip_face_verts: Vec<u32>,
    /// External-load vector — fixed for this scene (the scene
    /// optimises material, not load).
    f_ext: Vec<f64>,
    /// Current displacement field.
    u: Vec<f64>,
    /// Current Young's modulus (the optimisation variable).
    youngs: f64,
    /// Fixed Poisson's ratio.
    poisson: f64,
    /// Fixed total tip-load (downward).
    load_z: f64,
    /// Target tip deflection (slider, m).
    target_deflection: f64,
    /// Current tip deflection.
    tip_deflection: f64,
    /// Last computed `∂L/∂E`.
    last_grad: f64,
    /// Per-step learning-rate factor (multiplies the adaptive E²/|L|
    /// scale). 0 ⇒ paused.
    lr_factor: f64,
    /// Step counter (resets on entry / when target changes).
    iter_count: usize,
    /// Run / pause toggle for the descent loop.
    run_descent: bool,
    /// Set by `apply_scene_switch`; cleared on the first update tick.
    needs_reset: bool,
    /// Set by the pane "Reset" button.
    user_reset: bool,
    /// Last loss value (for the readout).
    last_loss: f64,
}

impl Default for FemDiffBeamExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            FEM_BEAM_LEN_X,
            FEM_BEAM_DEPTH_Y,
            FEM_BEAM_HEIGHT_Z,
        );
        let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::DirichletVector {
                    vertex: i as u32,
                    values: vec![0.0, 0.0, 0.0],
                });
            }
        }
        let tip_face_verts: Vec<u32> = mesh
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if (p.x - FEM_BEAM_LEN_X).abs() < 1e-9 {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();
        let n = mesh.vert_count() * 3;
        let f_ext = build_fem_beam_traction(&mesh, FEM_BEAM_DEFAULT_LOAD);
        Self {
            mesh,
            bcs,
            tip_face_verts,
            f_ext,
            u: vec![0.0; n],
            youngs: FEM_BEAM_DEFAULT_E,
            poisson: FEM_BEAM_DEFAULT_NU,
            load_z: FEM_BEAM_DEFAULT_LOAD,
            // Default target: 80% of the analytic Bernoulli deflection
            // at E0 (so the optimiser can move E either up or down to
            // hit it depending on starting conditions).
            target_deflection: -2.0e-3,
            tip_deflection: 0.0,
            last_grad: 0.0,
            lr_factor: 0.2,
            iter_count: 0,
            run_descent: true,
            needs_reset: true,
            user_reset: false,
            last_loss: 0.0,
        }
    }
}

fn build_fem_diff_beam_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM diff beam: empty model"), Vec::new())
}

// Per-frame system: drives the gradient-descent loop on Young's
// modulus when the scene is active.
fn update_fem_diff_beam_example(sim: Res<Sim>, mut state: ResMut<FemDiffBeamExampleState>) {
    if sim.scene != Scene::FemDiffBeam {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_optim();
        // Always run one step on entry so the deformed shape is visible
        // from frame 0.
        state.step_once();
        state.needs_reset = false;
        state.user_reset = false;
    }
    if state.run_descent {
        state.step_once();
    }
}

impl FemStokesExampleState {
    fn rebuild_bcs(&mut self) {
        self.bcs = fem::build_cavity_bcs_2d(&self.mesh, &self.th, 1.0, [self.lid_u, 0.0]);
        self.last_lid_u_baked = self.lid_u;
    }

    fn solve_now(&mut self) {
        let f = vec![0.0; self.th.ndof_velocity()];
        let cfg = fem::StokesSimpleConfig {
            alpha_p: 0.7,
            max_outer_iters: 200,
            tol_outer: 1.0e-6,
            max_inner_iters: 1500,
            tol_inner: 1.0e-9,
        };
        let res =
            fem::solve_stokes_steady(&self.mesh, &self.th, &f, self.viscosity, &self.bcs, &cfg);
        match res {
            Ok(r) => {
                self.u = r.u;
                self.p = r.p;
                self.last_outer_iters = r.outer_iters;
                self.last_div_residual = r.div_residual;
            }
            Err(e) => {
                eprintln!("FemStokes solve failed: {e}");
            }
        }
        self.last_viscosity_baked = self.viscosity;
    }
}

// Empty molla model — the FEM Stokes demo carries its own mesh +
// solver in `FemStokesExampleState`; the standard pipeline is

fn draw_fem_diff_beam_gizmos(
    sim: Res<Sim>,
    state: Res<FemDiffBeamExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::FemDiffBeam {
        return;
    }
    use std::collections::HashSet;
    let origin = FEM_BEAM_ORIGIN;
    let pinned: HashSet<u32> = state
        .bcs
        .iter()
        .filter_map(|bc| match bc {
            fem::BoundaryCondition::DirichletVector { vertex, .. } => Some(*vertex),
            _ => None,
        })
        .collect();
    let pos = |v: u32| -> bevy::math::Vec3 {
        let p = state.mesh.vertices[v as usize];
        let base = (v as usize) * 3;
        let ux = state.u[base];
        let uy = state.u[base + 1];
        let uz = state.u[base + 2];
        origin + bevy::math::Vec3::new((p.x + ux) as f32, (p.z + uz) as f32, (p.y + uy) as f32)
    };
    let edge_colour = |a: u32, b: u32| -> Color {
        if pinned.contains(&a) && pinned.contains(&b) {
            Color::srgb(0.95, 0.85, 0.30)
        } else {
            Color::srgb(0.40, 0.95, 0.55)
        }
    };
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
                gizmos.line(pos(a), pos(b), edge_colour(a, b));
            }
        }
    }
}

// Map a pressure value `p` (signed) to a colour (blue = low, red =
// high) by normalising against `p_abs_max`.
fn fem_stokes_pressure_colour(p: f64, p_abs_max: f64) -> Color {
    let denom = p_abs_max.max(1e-12);
    let t = ((p / denom) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

// Draw the cavity: triangle wireframe coloured by averaged pressure,

fn fem_diff_beam_pane(ui: &mut egui::Ui, state: &mut FemDiffBeamExampleState) {
    ui.label("Differentiable FEM — Stage 36.4");
    ui.separator();
    readout(ui, "Iteration", &format!("{}", state.iter_count));
    readout(ui, "Loss", &format!("{:.3e}", state.last_loss));
    readout(ui, "E (Young's)", &format!("{:.3e} Pa", state.youngs));
    readout(
        ui,
        "Tip deflection",
        &format!("{:.4e} m", state.tip_deflection),
    );
    readout(ui, "Target", &format!("{:.4e} m", state.target_deflection));
    ui.separator();
    ui.label("Target deflection (m)");
    let _ = ui.add(egui::Slider::new(
        &mut state.target_deflection,
        -0.05..=0.05,
    ));
    ui.label("Learning rate factor");
    let _ = ui.add(egui::Slider::new(&mut state.lr_factor, 0.01..=1.0));
    ui.separator();
    ui.checkbox(&mut state.run_descent, "Run gradient descent");
    if ui.button("Reset to E = 1e6").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    ui.small("Adjoint method: solve K λ = ∂J/∂u once, then");
    ui.small("∂J/∂E = -(λᵀ K u) / E. Pure-CPU CG inner solve.");
}

// Right-rail pane for `Scene::FemStokes`. Material + lid sliders +

