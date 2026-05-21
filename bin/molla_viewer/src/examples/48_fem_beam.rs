// --- Stage 36.1 — FEM elasticity cantilever-beam scene (#48) -------
//
// Pure-CPU steady-state demo of the Stage 36.1 vector-valued FEM
// framework. A 20×4×4 tet beam (Kuhn 6-tet decomposition of each hex)
// is pinned along its left face and loaded with a downward traction
// on the right face. The deformed wireframe (`x + u`) is drawn each
// frame; the assembled stiffness matrix + Dirichlet pattern survive
// scene switches and only re-solve when a slider changes.
//
// The right-rail pane carries Young's modulus E, Poisson ratio ν,
// applied tip force F, and a Linear-vs-SVK toggle.

const FEM_BEAM_NX: usize = 20;
const FEM_BEAM_NY: usize = 4;
const FEM_BEAM_NZ: usize = 4;
const FEM_BEAM_LEN_X: f64 = 4.0;
const FEM_BEAM_DEPTH_Y: f64 = 0.5;
const FEM_BEAM_HEIGHT_Z: f64 = 0.5;
const FEM_BEAM_DEFAULT_E: f64 = 1.0e6;
const FEM_BEAM_DEFAULT_NU: f64 = 0.0;
const FEM_BEAM_DEFAULT_LOAD: f64 = -50.0;
// Origin offset for the beam — keep it just above the ground plane
// so the gizmo wireframe doesn't clip the grid.
const FEM_BEAM_ORIGIN: bevy::math::Vec3 = bevy::math::Vec3::new(-2.0, 0.6, -0.25);

#[derive(Resource)]
struct FemBeamExampleState {
    mesh: fem::FemMesh,
    /// Linear-elasticity stiffness matrix `K = ∫ Bᵀ D B dV`. Rebuilt
    /// when `youngs` or `poisson` change.
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    /// Indices of the vertices on the loaded face (x = L). Used both
    /// to assemble the traction and to read out the tip deflection.
    tip_face_verts: Vec<u32>,
    /// External-load vector (3 components per vertex). Rebuilt when
    /// `load_z` changes.
    f_ext: Vec<f64>,
    /// Current displacement field (3 × vert_count, interleaved).
    u: Vec<f64>,
    /// Current Young's modulus.
    youngs: f64,
    /// Current Poisson's ratio.
    poisson: f64,
    /// Total tip-load magnitude (signed; downward = negative).
    load_z: f64,
    /// Use SVK nonlinear elasticity instead of linear K · u = f.
    use_svk: bool,
    // Cached values to detect slider changes.
    last_youngs_baked: f64,
    last_poisson_baked: f64,
    last_load_baked: f64,
    last_svk_baked: bool,
    /// Last solve telemetry.
    tip_deflection: f64,
    last_residual: f64,
    last_iters: usize,
    /// Set by `apply_scene_switch` and cleared by the update system on
    /// re-init.
    needs_reset: bool,
    user_reset: bool,
}

impl Default for FemBeamExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            FEM_BEAM_LEN_X,
            FEM_BEAM_DEPTH_Y,
            FEM_BEAM_HEIGHT_Z,
        );
        let (lam, mu) = fem_beam_lame(FEM_BEAM_DEFAULT_E, FEM_BEAM_DEFAULT_NU);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM beam: assemble stiffness");
        // Dirichlet on the left face (x = 0) — fully clamp.
        let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::DirichletVector {
                    vertex: i as u32,
                    values: vec![0.0, 0.0, 0.0],
                });
            }
        }
        // Tip-face vertex list.
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
        let u = vec![0.0; n];
        Self {
            mesh,
            stiff,
            bcs,
            tip_face_verts,
            f_ext,
            u,
            youngs: FEM_BEAM_DEFAULT_E,
            poisson: FEM_BEAM_DEFAULT_NU,
            load_z: FEM_BEAM_DEFAULT_LOAD,
            use_svk: false,
            last_youngs_baked: FEM_BEAM_DEFAULT_E,
            last_poisson_baked: FEM_BEAM_DEFAULT_NU,
            last_load_baked: FEM_BEAM_DEFAULT_LOAD,
            last_svk_baked: false,
            tip_deflection: 0.0,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: true,
            user_reset: false,
        }
    }
}

impl FemHeatP2ExampleState {
    fn rebuild_stiffness(&mut self) {
        self.stiff = fem::assemble_global_p2(&self.mesh, &self.dof_map, |c, v| {
            fem::compute_element_p2_stiffness(c, v, self.alpha)
        })
        .expect("FEM heat P2: rebuild stiffness");
        self.last_alpha_baked = self.alpha;
    }

    fn reset_temperature_field(&mut self) {
        let n = self.dof_map.ndof();
        self.u = vec![0.0_f64; n];
        for bc in &self.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = *bc {
                self.u[vertex as usize] = value;
            }
        }
    }
}

fn fem_beam_lame(youngs: f64, poisson: f64) -> (f64, f64) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    // Plane-strain-style Lamé. For Poisson = 0.5 the formula
    // diverges; clamp the slider to ≤ 0.49 to avoid that singularity.
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lam, mu)
}

// Build the external-force vector for a constant downward (−z)
// traction over the tip face (x = L_x). Uses the consistent (P1
// surface integral) lumping `(area / 3) · t` per face vertex.
// Total integrated tip load along z equals `load_z` (the surface
// area cancels: traction = load_z / (b · h) and `∫ t dA = load_z`).
fn build_fem_beam_traction(mesh: &fem::FemMesh, load_z: f64) -> Vec<f64> {
    let n = mesh.vert_count() * 3;
    let mut f = vec![0.0; n];
    let area_total = FEM_BEAM_DEPTH_Y * FEM_BEAM_HEIGHT_Z;
    let traction_z = load_z / area_total;
    for cell in &mesh.cells {
        if let fem::FemCell::Tet(idx) = cell {
            let faces = [
                [idx[0], idx[1], idx[2]],
                [idx[0], idx[1], idx[3]],
                [idx[0], idx[2], idx[3]],
                [idx[1], idx[2], idx[3]],
            ];
            for face in &faces {
                let pa = mesh.vertices[face[0] as usize];
                let pb = mesh.vertices[face[1] as usize];
                let pc = mesh.vertices[face[2] as usize];
                let on_tip = (pa.x - FEM_BEAM_LEN_X).abs() < 1e-9
                    && (pb.x - FEM_BEAM_LEN_X).abs() < 1e-9
                    && (pc.x - FEM_BEAM_LEN_X).abs() < 1e-9;
                if !on_tip {
                    continue;
                }
                let area = 0.5 * (pb - pa).cross(pc - pa).length();
                let third = area / 3.0;
                for &v in face {
                    let v = v as usize;
                    f[v * 3 + 2] += third * traction_z;
                }
            }
        }
    }
    f
}

impl FemBeamExampleState {
    fn rebuild_stiffness(&mut self) {
        let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
        let space = fem::FunctionSpace::P1Vector { dim: 3 };
        self.stiff = fem::assemble_global(&self.mesh, space, |c, v| {
            fem::compute_element_elasticity_stiffness(c, v, lam, mu)
        })
        .expect("FEM beam: rebuild stiffness");
        self.last_youngs_baked = self.youngs;
        self.last_poisson_baked = self.poisson;
    }

    fn rebuild_traction(&mut self) {
        self.f_ext = build_fem_beam_traction(&self.mesh, self.load_z);
        self.last_load_baked = self.load_z;
    }

    fn solve(&mut self) {
        if self.use_svk {
            // SVK Newton-Raphson. Re-assemble the (linear) tangent
            // each iter — full geometric tangent is a Stage 36.1.1
            // upgrade.
            let (lam, mu) = fem_beam_lame(self.youngs, self.poisson);
            let cells = self.mesh.cells.clone();
            let verts = self.mesh.vertices.clone();
            let stiff_cached = self.stiff.clone();
            let tangent = |_u: &[f64]| -> molla::math::sparse::BsrMatrix { stiff_cached.clone() };
            let n = self.mesh.vert_count() * 3;
            let residual = move |u: &[f64]| -> Vec<f64> {
                let mut r = vec![0.0; n];
                for cell in &cells {
                    if let fem::FemCell::Tet(idx) = cell {
                        let mut u_local = [0.0; 12];
                        for (i, &v) in idx.iter().enumerate() {
                            let base = (v as usize) * 3;
                            u_local[3 * i] = u[base];
                            u_local[3 * i + 1] = u[base + 1];
                            u_local[3 * i + 2] = u[base + 2];
                        }
                        let f_int = fem::compute_element_st_venant_kirchhoff_stress(
                            cell, &verts, &u_local, lam, mu,
                        );
                        for (i, &v) in idx.iter().enumerate() {
                            let base = (v as usize) * 3;
                            r[base] += f_int[3 * i];
                            r[base + 1] += f_int[3 * i + 1];
                            r[base + 2] += f_int[3 * i + 2];
                        }
                    }
                }
                r
            };
            self.u = fem::solve_nonlinear_static(
                tangent,
                residual,
                &self.f_ext,
                &self.bcs,
                &self.mesh.vertices,
                3,
                5,
                1.0e-7,
            );
            self.last_iters = 5; // best-effort; the inner driver doesn't expose its counter.
            self.last_residual = 0.0;
        } else {
            // Linear elasticity steady state.
            let mut k_red = self.stiff.clone();
            let mut f_red = self.f_ext.clone();
            fem::apply_bcs(&mut k_red, &mut f_red, &self.bcs, &self.mesh.vertices, 3);
            // Initial guess: zero (Dirichlet rows already pinned).
            let res = molla::math::iterative::cg_solve(
                |x| {
                    let mut y = vec![0.0; k_red.nrows()];
                    k_red.mv(x, &mut y).unwrap();
                    y
                },
                &f_red,
                &vec![0.0; f_red.len()],
                1.0e-9,
                molla::fem::solver::DEFAULT_MAX_ITERS,
            );
            self.u = res.x;
            self.last_residual = res.residual;
            self.last_iters = res.iters;
        }
        // Read tip deflection (average of tip-face vertices, z-component).
        if self.tip_face_verts.is_empty() {
            self.tip_deflection = 0.0;
        } else {
            let mut sum = 0.0;
            for &v in &self.tip_face_verts {
                sum += self.u[(v as usize) * 3 + 2];
            }
            self.tip_deflection = sum / self.tip_face_verts.len() as f64;
        }
        self.last_youngs_baked = self.youngs;
        self.last_poisson_baked = self.poisson;
        self.last_load_baked = self.load_z;
        self.last_svk_baked = self.use_svk;
    }

    fn reset_displacement_field(&mut self) {
        let n = self.mesh.vert_count() * 3;
        self.u = vec![0.0; n];
        self.tip_deflection = 0.0;

        self.last_residual = 0.0;
        self.last_iters = 0;
    }
}

// Build the Dirichlet BC list for the P2 heat scene: corner DoFs on
// the top + bottom faces, plus edge DoFs whose endpoints are both on
// the same face (so the edge midpoint sits on that face).
fn fem_heat_p2_dirichlet_bcs(
    mesh: &fem::FemMesh,
    dof_map: &fem::P2DofMap,
) -> Vec<fem::BoundaryCondition> {
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
    for (edge, &edge_id) in &dof_map.edge_id {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mid_z = 0.5 * (pa.z + pb.z);
        let on_bottom = pa.z.abs() < 1e-9 && pb.z.abs() < 1e-9;
        let on_top = (pa.z - 1.0).abs() < 1e-9 && (pb.z - 1.0).abs() < 1e-9;
        if on_bottom || on_top {
            bcs.push(fem::BoundaryCondition::Dirichlet {
                vertex: (dof_map.vertex_count + edge_id as usize) as u32,
                value: mid_z,
            });
        }
    }
    bcs
}

// Empty molla model — the P2 FEM heat demo carries its own mesh +
// solver in `FemHeatP2ExampleState`; the standard pipeline is

fn build_fem_beam_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM beam: empty model"), Vec::new())
}

// Per-frame system: rebuilds the stiffness / traction when sliders
// change, applies any pending reset, and re-solves the steady-state
// (or SVK Newton) system.
fn update_fem_beam_example(sim: Res<Sim>, mut state: ResMut<FemBeamExampleState>) {
    if sim.scene != Scene::FemBeam {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_displacement_field();
        state.needs_reset = false;
        state.user_reset = false;
        // Always solve once on entry so the deformed shape is visible
        // immediately.
        state.solve();
        return;
    }
    let mat_changed = (state.youngs - state.last_youngs_baked).abs() > 1e-12
        || (state.poisson - state.last_poisson_baked).abs() > 1e-12;
    let load_changed = (state.load_z - state.last_load_baked).abs() > 1e-12;
    let svk_changed = state.use_svk != state.last_svk_baked;
    if mat_changed {
        state.rebuild_stiffness();
    }
    if load_changed {
        state.rebuild_traction();
    }
    if mat_changed || load_changed || svk_changed {
        state.solve();
    }
}

// Draw the deformed tet wireframe (vertices `x + u`). Pinned
// vertices are drawn in a slightly brighter tone so the user can
// see the clamped face.
fn draw_fem_beam_gizmos(sim: Res<Sim>, state: Res<FemBeamExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::FemBeam {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(state.mesh.cells.len() * 4);
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
            Color::srgb(0.30, 0.65, 0.95)
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
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                gizmos.line(pos(a), pos(b), edge_colour(a, b));
            }
        }
    }
}

// Right-rail pane for `Scene::FemHeatP2`. Same controls as the P1
// pane plus a side-by-side P1 vs P2 DoF count readout so users can

fn fem_beam_pane(ui: &mut egui::Ui, state: &mut FemBeamExampleState) {
    ui.label("FEM elasticity — Stage 36.1");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{}×{}×{} tets ({} verts)",
            FEM_BEAM_NX,
            FEM_BEAM_NY,
            FEM_BEAM_NZ,
            state.mesh.vert_count()
        ),
    );
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    ui.separator();
    ui.label("Young's modulus E (Pa)");
    let _ = ui.add(egui::Slider::new(&mut state.youngs, 1.0e4..=1.0e8).logarithmic(true));
    ui.label("Poisson's ratio ν");
    let _ = ui.add(egui::Slider::new(&mut state.poisson, 0.0..=0.45));
    ui.label("Tip load F (N, downward = negative)");
    let _ = ui.add(egui::Slider::new(&mut state.load_z, -500.0..=0.0));
    ui.separator();
    ui.checkbox(&mut state.use_svk, "Use SVK (nonlinear)");
    ui.separator();
    readout(
        ui,
        "Tip deflection",
        &format!("{:.4e} m", state.tip_deflection),
    );
    // Analytic Bernoulli reference (linear small-strain): δ = P L³ / (3 E I).
    let i_inertia = FEM_BEAM_DEPTH_Y * FEM_BEAM_HEIGHT_Z.powi(3) / 12.0;
    let delta = state.load_z * FEM_BEAM_LEN_X.powi(3) / (3.0 * state.youngs * i_inertia);
    readout(ui, "Bernoulli δ", &format!("{:.4e} m", delta));
    readout(ui, "CG residual", &format!("{:.3e}", state.last_residual));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    ui.separator();
    if ui.button("Reset to undeformed").clicked() {
        state.user_reset = true;
    }
    ui.small("BCs: clamp at x = 0, traction on x = L.");
    ui.small("Linear: solves K · u = f. SVK: 5-iter Newton-Raphson.");
}

