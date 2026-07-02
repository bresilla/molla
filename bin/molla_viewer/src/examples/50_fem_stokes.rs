// ─── Stage 36.2 — FEM Stokes (lid-driven cavity, Taylor-Hood) ──────

const FEM_STOKES_GRID: usize = 16;
const FEM_STOKES_DEFAULT_VISCOSITY: f64 = 1.0;
const FEM_STOKES_DEFAULT_LID_U: f64 = 1.0;
// Origin offset for the unit cavity — sits just above the ground
// plane so the gizmo wireframe + arrows don't clip.
const FEM_STOKES_ORIGIN: bevy::math::Vec3 = bevy::math::Vec3::new(-0.5, 0.05, -0.5);

// Persistent state for `Scene::FemStokes`. The Taylor-Hood mesh +
// matrices survive scene switches (rebuilding 16² P2 meshes is cheap
// but pointless on every entry); only the velocity / pressure fields
// reset on entry.
#[derive(Resource)]
struct FemStokesExampleState {
    mesh: fem::FemMesh,
    th: fem::MixedTaylorHood,
    bcs: Vec<fem::StokesBc>,
    viscosity: f64,
    lid_u: f64,
    /// Latest velocity field (P2 vector, interleaved layout).
    u: Vec<f64>,
    /// Latest pressure field (P1 nodal).
    p: Vec<f64>,
    /// Last solver telemetry.
    last_outer_iters: usize,
    last_div_residual: f64,
    /// Apply_scene_switch sets this on entry so the next update re-runs
    /// the solve (otherwise re-entering shows a stale field).
    needs_reset: bool,
    /// Right-rail "Recompute" button.
    user_reset: bool,
    // Cached values to detect slider changes.
    last_viscosity_baked: f64,
    last_lid_u_baked: f64,
}

impl Default for FemStokesExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(FEM_STOKES_GRID, FEM_STOKES_GRID, 1.0, 1.0);
        let th = fem::MixedTaylorHood::build(&mesh, 2);
        let bcs = fem::build_cavity_bcs_2d(&mesh, &th, 1.0, [FEM_STOKES_DEFAULT_LID_U, 0.0]);
        let n_v = th.ndof_velocity();
        let n_p = th.ndof_pressure();
        Self {
            mesh,
            th,
            bcs,
            viscosity: FEM_STOKES_DEFAULT_VISCOSITY,
            lid_u: FEM_STOKES_DEFAULT_LID_U,
            u: vec![0.0; n_v],
            p: vec![0.0; n_p],
            last_outer_iters: 0,
            last_div_residual: 0.0,
            needs_reset: true,
            user_reset: false,
            last_viscosity_baked: FEM_STOKES_DEFAULT_VISCOSITY,
            last_lid_u_baked: FEM_STOKES_DEFAULT_LID_U,
        }
    }
}

impl FemDiffBeamExampleState {
    /// Run one step of (forward solve → adjoint → gradient descent).
    fn step_once(&mut self) {
        // Re-assemble K(E).
        let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        let stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM diff beam: assemble stiffness");

        // Forward solve.
        let mut k_red = stiff.clone();
        let mut f_red = self.f_ext.clone();
        fem::apply_bcs(&mut k_red, &mut f_red, &self.bcs, &self.mesh.vertices, 3);
        let cg_res = molla::math::iterative::cg_solve(
            |x| {
                let mut y = vec![0.0; k_red.nrows()];
                k_red.mv(x, &mut y).unwrap();
                y
            },
            &f_red,
            &vec![0.0; f_red.len()],
            1.0e-10,
            molla::fem::solver::DEFAULT_MAX_ITERS,
        );
        self.u = cg_res.x;

        // Tip deflection (avg of tip-face z-components).
        let n_tip = self.tip_face_verts.len().max(1);
        let mut sum = 0.0;
        for &v in &self.tip_face_verts {
            sum += self.u[(v as usize) * 3 + 2];
        }
        self.tip_deflection = sum / n_tip as f64;
        let err = self.tip_deflection - self.target_deflection;
        self.last_loss = 0.5 * err * err;

        // Adjoint: ∂L/∂u_v = (err / n_tip) at tip-face z-DoFs (chain
        // rule of the average reduction); zero everywhere else.
        let n_dof = self.u.len();
        let mut dj_du = vec![0.0_f64; n_dof];
        let coeff = err / n_tip as f64;
        for &v in &self.tip_face_verts {
            dj_du[(v as usize) * 3 + 2] = coeff;
        }
        let lambda = fem::solve_adjoint_with(
            &stiff,
            &dj_du,
            &self.bcs,
            &self.mesh.vertices,
            3,
            molla::fem::solver::DEFAULT_MAX_ITERS,
            1.0e-10,
        );
        let g = fem::compute_grad_youngs(
            &self.mesh,
            &self.mesh.vertices,
            &self.u,
            &lambda,
            self.poisson,
            self.youngs,
        );
        self.last_grad = g;

        // Adaptive step proportional to E²/|L|, scaled by lr_factor.
        if self.run_descent && self.lr_factor > 0.0 {
            let lr = self.lr_factor * self.youngs * self.youngs / self.last_loss.max(1e-30);
            let e_next = (self.youngs - lr * g).max(1.0e3).min(1.0e10);
            self.youngs = e_next;
        }
        self.iter_count += 1;
    }

    fn reset_optim(&mut self) {
        let n = self.mesh.vert_count() * 3;
        self.u = vec![0.0; n];
        self.youngs = FEM_BEAM_DEFAULT_E;
        self.tip_deflection = 0.0;
        self.last_grad = 0.0;
        self.iter_count = 0;
        self.last_loss = 0.0;
        self.f_ext = build_fem_beam_traction(&self.mesh, self.load_z);
    }
}

// Empty molla model — the diff-FEM beam scene carries its own mesh +
// solver in `FemDiffBeamExampleState`; the standard pipeline is

fn build_fem_stokes_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM stokes: empty model"), Vec::new())
}

// Per-frame system: re-solves the steady-state Stokes system whenever
// material / lid sliders change, or on scene re-entry.
fn update_fem_stokes_example(sim: Res<Sim>, mut state: ResMut<FemStokesExampleState>) {
    if sim.scene != Scene::FemStokes {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.rebuild_bcs();
        state.solve_now();
        state.needs_reset = false;
        state.user_reset = false;
    }
}

// Draw the deformed tet wireframe + tip-face overlay highlighting how

fn draw_fem_stokes_gizmos(sim: Res<Sim>, state: Res<FemStokesExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::FemStokes {
        return;
    }
    let origin = FEM_STOKES_ORIGIN;
    let p_abs_max = state.p.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 3);
    let pos2d = |v: usize| -> bevy::math::Vec3 {
        let p = state.mesh.vertices[v];
        origin + bevy::math::Vec3::new(p.x as f32, 0.0, p.y as f32)
    };
    for cell in &state.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let p_avg = 0.5 * (state.p[a as usize] + state.p[b as usize]);
                let colour = fem_stokes_pressure_colour(p_avg, p_abs_max);
                gizmos.line(pos2d(a as usize), pos2d(b as usize), colour);
            }
        }
    }
    // Velocity arrows at every vertex (P2 vector layout: vertex DoFs
    // come first, components interleaved).
    let dim = state.th.dim;
    let scale = 0.08;
    for (i, p) in state.mesh.vertices.iter().enumerate() {
        let ux = state.u[i * dim];
        let uy = state.u[i * dim + 1];
        let mag = (ux * ux + uy * uy).sqrt();
        if mag < 1e-6 {
            continue;
        }
        let from = origin + bevy::math::Vec3::new(p.x as f32, 0.01, p.y as f32);
        let to = from + bevy::math::Vec3::new((ux * scale) as f32, 0.0, (uy * scale) as f32);
        gizmos.line(from, to, Color::srgb(0.95, 0.95, 0.95));
    }
}

// Right-rail pane for `Scene::FemDiffBeam`. Target deflection slider,

fn fem_stokes_pane(ui: &mut egui::Ui, state: &mut FemStokesExampleState) {
    ui.label("FEM Stokes — Stage 36.2 (lid-driven cavity)");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0} P2-P1 tris ({1} verts, {2} edges)",
            FEM_STOKES_GRID,
            state.mesh.vert_count(),
            state.th.map.edge_id.len(),
        ),
    );
    readout(
        ui,
        "Velocity DoFs",
        &format!("{}", state.th.ndof_velocity()),
    );
    readout(
        ui,
        "Pressure DoFs",
        &format!("{}", state.th.ndof_pressure()),
    );
    ui.separator();
    ui.label("Viscosity μ (Pa·s)");
    let _ = ui.add(egui::Slider::new(&mut state.viscosity, 1.0e-2..=1.0e2).logarithmic(true));
    ui.label("Lid velocity U (m/s)");
    let _ = ui.add(egui::Slider::new(&mut state.lid_u, 0.0..=5.0));
    ui.separator();
    // Reynolds number Re = ρ U L / μ, with ρ = 1, L = 1.
    let re = state.lid_u / state.viscosity.max(1e-12);
    readout(ui, "Reynolds Re", &format!("{:.3e}", re));
    readout(ui, "Outer-CG iters", &format!("{}", state.last_outer_iters));
    readout(
        ui,
        "Div residual",
        &format!("{:.3e}", state.last_div_residual),
    );
    ui.separator();
    if ui.button("Recompute").clicked() {
        state.user_reset = true;
    }
    ui.small("Solver: Uzawa-CG on the pressure-Schur complement.");
    ui.small("Pure Stokes (Re→0). Re slider is informational only.");
}

// ── Warp-port scenes ────────────────────────────────────────────────
//
// Ports of NVIDIA Warp's `xtra/warp/warp/examples/` library. Each
// scene wraps a molla primitive (raycaster, FEM, MPM, autograd, ...)
// in a small host-driven demo with sliders + readouts; physics is
// short-circuited in `step_simulation` for the host-only ones, while
// MPM-backed ones (Dem/Fluid/Sph) reuse the existing MpmExampleState
// pipeline.

// Empty molla model — used by Warp scenes that own their own state /
// primitive and don't need bodies to participate in the standard

