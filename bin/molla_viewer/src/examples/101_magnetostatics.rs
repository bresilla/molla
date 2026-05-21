// ═══ Scene 101: Magnetostatics (Stage 46) ═══════════════════════════════
//
// A short single-turn current loop (the simplest "solenoid") sitting
// in the centre of a tet block — really an annulus of cells carrying
// `J` tangential to the loop axis. We call
// `solve_magnetostatics_steady` lazily when the user changes a slider
// and render a per-cell magnitude proxy on the z = 0.5 cut plane.

const MAG_GRID: usize = 6;
const MAG_PLANE_HALF: f32 = 0.50;

struct MagnetostaticsState {
    mesh: fem::FemMesh,
    cell_bz: Vec<f64>,
    cell_centroid: Vec<MVec3>,
    current_a: f64,
    loop_radius: f64,
    last_solved_a: f64,
    last_solved_r: f64,
    paused: bool,
    user_reset: bool,
    last_residual: f64,
    last_iters: usize,
    last_solve_ms: f64,
}

impl Default for MagnetostaticsState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(MAG_GRID, MAG_GRID, MAG_GRID, 1.0, 1.0, 1.0);
        let mut me = Self {
            mesh,
            cell_bz: Vec::new(),
            cell_centroid: Vec::new(),
            current_a: 1.0,
            loop_radius: 0.18,
            last_solved_a: -1.0,
            last_solved_r: -1.0,
            paused: false,
            user_reset: false,
            last_residual: 0.0,
            last_iters: 0,
            last_solve_ms: 0.0,
        };
        me.solve();
        me
    }
}

impl MagnetostaticsState {
    fn solve(&mut self) {
        let mesh = &self.mesh;
        let mut j_source: Vec<molla::math::Vec3> = vec![molla::math::Vec3::ZERO; mesh.cells.len()];
        let mut centroids: Vec<MVec3> = Vec::with_capacity(mesh.cells.len());
        for (ci, cell) in mesh.cells.iter().enumerate() {
            if let fem::FemCell::Tet(v) = cell {
                let a = mesh.vertices[v[0] as usize];
                let b = mesh.vertices[v[1] as usize];
                let c = mesh.vertices[v[2] as usize];
                let d = mesh.vertices[v[3] as usize];
                let c0 = (a + b + c + d) / 4.0;
                centroids.push(c0);
                let dx = c0.x - 0.5;
                let dy = c0.y - 0.5;
                let dz = (c0.z - 0.5).abs();
                let r = (dx * dx + dy * dy).sqrt();
                let in_band = (r - self.loop_radius).abs() < 0.04 && dz < 0.05;
                if in_band {
                    let inv_r = 1.0 / r.max(1.0e-9);
                    let tx = -dy * inv_r;
                    let ty = dx * inv_r;
                    j_source[ci] =
                        molla::math::Vec3::new(self.current_a * tx, self.current_a * ty, 0.0);
                }
            } else {
                centroids.push(MVec3::ZERO);
            }
        }
        let nu_per_cell = vec![1.0_f64; mesh.cells.len()];
        let t0 = std::time::Instant::now();
        match fem::solve_magnetostatics_steady(mesh, &j_source, &nu_per_cell, &[]) {
            Ok(result) => {
                self.last_residual = result.residual;
                self.last_iters = result.iters;
                let map_built = molla::fem::NedelecDofMap::build(mesh);
                let n_dof = map_built.ndof();
                let a_dof = if result.a.len() == n_dof {
                    result.a
                } else {
                    vec![0.0; n_dof]
                };
                let mut cell_b = vec![0.0_f64; mesh.cells.len()];
                for (ci, cell) in mesh.cells.iter().enumerate() {
                    if let fem::FemCell::Tet(v) = cell {
                        let edges = [
                            (v[0], v[1]),
                            (v[0], v[2]),
                            (v[0], v[3]),
                            (v[1], v[2]),
                            (v[1], v[3]),
                            (v[2], v[3]),
                        ];
                        let mut s = 0.0;
                        for &(a, b) in &edges {
                            let d = map_built.edge_dof(a, b);
                            s += a_dof[d].abs();
                        }
                        cell_b[ci] = s;
                    }
                }
                self.cell_bz = cell_b;
                self.cell_centroid = centroids;
            }
            Err(_) => {
                self.cell_bz.clear();
                self.cell_centroid.clear();
            }
        }
        self.last_solve_ms = t0.elapsed().as_micros() as f64 * 1.0e-3;
        self.last_solved_a = self.current_a;
        self.last_solved_r = self.loop_radius;
    }
}

fn update_magnetostatics(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::Magnetostatics {
        return;
    }
    let st = &mut warp.magnetostatics;
    if st.user_reset {
        *st = MagnetostaticsState::default();
        return;
    }
    if run.paused || st.paused {
        return;
    }
    if (st.current_a - st.last_solved_a).abs() > 1.0e-6
        || (st.loop_radius - st.last_solved_r).abs() > 1.0e-3
    {
        st.solve();
    }
}

fn draw_magnetostatics_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::Magnetostatics {
        return;
    }
    let st = &warp.magnetostatics;
    let half = MAG_PLANE_HALF;
    let outline = Color::srgb(0.3, 0.3, 0.35);
    let corners = [
        Vec3::new(-half, 0.0, -half),
        Vec3::new(half, 0.0, -half),
        Vec3::new(half, 0.0, half),
        Vec3::new(-half, 0.0, half),
    ];
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], outline);
    }
    let mx = st
        .cell_bz
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(1.0e-9);
    for (ci, c) in st.cell_centroid.iter().enumerate() {
        if (c.z - 0.5).abs() > 0.10 {
            continue;
        }
        let s = (st.cell_bz[ci] / mx).clamp(0.0, 1.0) as f32;
        let col = Color::srgb(0.15 + 0.80 * s, 0.30 + 0.30 * (1.0 - s), 0.95 - 0.80 * s);
        let pos = Vec3::new(
            -half + (c.x as f32) * (2.0 * half),
            (s * 0.40) as f32,
            -half + (c.y as f32) * (2.0 * half),
        );
        gizmos.sphere(pos, 0.018, col);
    }
    let loop_y = 0.001_f32;
    let loop_col = Color::srgb(1.0, 0.85, 0.20);
    let r = st.loop_radius as f32;
    for k in 0..64 {
        let a0 = (k as f32 / 64.0) * std::f32::consts::TAU;
        let a1 = ((k + 1) as f32 / 64.0) * std::f32::consts::TAU;
        gizmos.line(
            Vec3::new(r * a0.cos(), loop_y, r * a0.sin()),
            Vec3::new(r * a1.cos(), loop_y, r * a1.sin()),
            loop_col,
        );
    }
}

fn magnetostatics_pane(ui: &mut egui::Ui, st: &mut MagnetostaticsState) {
    ui.label("Stage 46 — magnetostatics (current loop, Ned-0)");
    ui.separator();
    readout(ui, "Mesh", &format!("{0}×{0}×{0} tets", MAG_GRID));
    // `cell_bz` is one summed |A| sample per tet — this is the cell
    // count, not the Nédélec edge-DoF count of the linear system.
    readout(ui, "Cells", &format!("{}", st.cell_bz.len()));
    readout(ui, "CG residual", &format!("{:.2e}", st.last_residual));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "Solve", &format!("{:.2} ms", st.last_solve_ms));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.current_a, 0.0..=5.0).text("I (A)"));
    let _ = ui.add(egui::Slider::new(&mut st.loop_radius, 0.05..=0.35).text("loop radius"));
    let _ = ui.checkbox(&mut st.paused, "Pause / freeze re-solve");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "The yellow ring is the single-turn current loop on z = 0.5. \
         Sphere colour ≈ Σ |A_e| over the cell's six Ned-0 edges — a \
         proxy for `‖B‖` at the cell centre. A pure B_z slice needs \
         the curl operator on Ned-0 elements which Stage 46 v2 will \
         surface publicly.",
    );
}

