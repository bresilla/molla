// ═══ Scene 100: AmrHeat (Stage 45 v2) ════════════════════════════════
//
// Steady-state heat equation `-∇²T = f` on the unit square with a
// moving Gaussian heat source. After each frame we re-refine cells
// whose centroid lies close to the source (a simple radius-based
// criterion), flatten the AMR mesh, re-assemble K, and solve via the
// CPU FEM CG. The refined cells + temperature colormap are drawn.

const AMR_HEAT_BASE_N: usize = 8;

struct AmrHeatState {
    base_mesh: fem::FemMesh,
    amr: fem::AmrMesh,
    t: Vec<f64>,
    flat: fem::FemMesh,
    refine_radius: f64,
    source_speed: f64,
    sim_t: f64,
    paused: bool,
    user_reset: bool,
    leaf_count: usize,
    last_residual: f64,
    last_iters: usize,
}

impl Default for AmrHeatState {
    fn default() -> Self {
        let base_mesh = fem::FemMesh::tri_grid_2d(AMR_HEAT_BASE_N, AMR_HEAT_BASE_N, 1.0, 1.0);
        let amr = fem::AmrMesh::new(base_mesh.clone());
        let flat = amr.flatten();
        let n = flat.vert_count();
        Self {
            base_mesh,
            amr,
            t: vec![0.0; n],
            flat,
            refine_radius: 0.18,
            source_speed: 0.6,
            sim_t: 0.0,
            paused: false,
            user_reset: false,
            leaf_count: 0,
            last_residual: 0.0,
            last_iters: 0,
        }
    }
}

fn amr_heat_source_pos(t: f64) -> (f64, f64) {
    let cx = 0.5 + 0.25 * t.cos();
    let cy = 0.5 + 0.25 * t.sin();
    (cx, cy)
}

fn update_amr_heat(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::AmrHeat {
        return;
    }
    let st = &mut warp.amr_heat;
    if st.user_reset {
        *st = AmrHeatState::default();
        return;
    }
    if run.paused || st.paused {
        return;
    }
    st.sim_t += 1.0 / 60.0 * st.source_speed;
    let (cx, cy) = amr_heat_source_pos(st.sim_t);

    st.amr = fem::AmrMesh::new(st.base_mesh.clone());
    let radius = st.refine_radius;
    st.amr.refine_where(
        move |cell: &fem::FemCell, verts: &[molla::math::Vec3]| -> bool {
            if let fem::FemCell::Tri(v) = cell {
                let a = verts[v[0] as usize];
                let b = verts[v[1] as usize];
                let c = verts[v[2] as usize];
                let mx = (a.x + b.x + c.x) / 3.0;
                let my = (a.y + b.y + c.y) / 3.0;
                let d = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
                return d <= radius;
            }
            false
        },
    );
    st.flat = st.amr.flatten();

    let space = fem::FunctionSpace::P1Nodal;
    let mut k = match fem::assemble_global(&st.flat, space, |c, v| {
        fem::compute_element_stiffness(c, v, 1.0)
    }) {
        Ok(k) => k,
        Err(_) => return,
    };
    let n = st.flat.vert_count();
    let mut f = vec![0.0_f64; n];
    let sigma2 = 0.04_f64 * 0.04;
    for (i, p) in st.flat.vertices.iter().enumerate() {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let r2 = dx * dx + dy * dy;
        f[i] = (-r2 / (2.0 * sigma2)).exp();
    }
    let mut bcs = Vec::new();
    for (i, p) in st.flat.vertices.iter().enumerate() {
        if p.x <= 1.0e-9 || p.x >= 1.0 - 1.0e-9 || p.y <= 1.0e-9 || p.y >= 1.0 - 1.0e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }
    fem::apply_bcs(&mut k, &mut f, &bcs, &st.flat.vertices, 1);
    let k_clone = k.clone();
    let result = molla::math::iterative::cg_solve(
        |x| {
            let mut y = vec![0.0; x.len()];
            let _ = k_clone.mv(x, &mut y);
            y
        },
        &f,
        &vec![0.0; n],
        1.0e-8,
        500,
    );
    st.t = result.x;
    st.last_residual = result.residual;
    st.last_iters = result.iters;
    st.leaf_count = st.amr.leaf_count();
}

fn draw_amr_heat_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::AmrHeat {
        return;
    }
    let st = &warp.amr_heat;
    let half = 0.50_f32;
    let to_world = |x: f64, y: f64| -> Vec3 {
        Vec3::new(
            -half + x as f32 * (2.0 * half),
            0.0,
            -half + y as f32 * (2.0 * half),
        )
    };
    let tmax = st.t.iter().cloned().fold(0.0_f64, f64::max).max(1.0e-6);
    for cell in &st.flat.cells {
        if let fem::FemCell::Tri(v) = cell {
            let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
            for &(a, b) in &edges {
                let pa = st.flat.vertices[a as usize];
                let pb = st.flat.vertices[b as usize];
                let ta = st.t[a as usize];
                let tb = st.t[b as usize];
                let tmid = (ta + tb) * 0.5;
                let t = ((tmid / tmax).clamp(0.0, 1.0)) as f32;
                let col = Color::srgb(0.10 + 0.85 * t, 0.30 + 0.25 * t, 0.95 - 0.85 * t);
                gizmos.line(to_world(pa.x, pa.y), to_world(pb.x, pb.y), col);
            }
        }
    }
    let (cx, cy) = amr_heat_source_pos(st.sim_t);
    let src = to_world(cx, cy);
    let marker_col = Color::srgb(1.0, 1.0, 0.30);
    let r = 0.03;
    for k in 0..16 {
        let a0 = k as f32 / 16.0 * std::f32::consts::TAU;
        let a1 = (k + 1) as f32 / 16.0 * std::f32::consts::TAU;
        gizmos.line(
            src + Vec3::new(r * a0.cos(), 0.001, r * a0.sin()),
            src + Vec3::new(r * a1.cos(), 0.001, r * a1.sin()),
            marker_col,
        );
    }
}

fn amr_heat_pane(ui: &mut egui::Ui, st: &mut AmrHeatState) {
    ui.label("Stage 45 v2 — AMR heat (moving Gaussian source)");
    ui.separator();
    readout(ui, "Base mesh", &format!("{0}×{0}", AMR_HEAT_BASE_N));
    readout(ui, "Leaf cells", &format!("{}", st.leaf_count));
    readout(ui, "DoF", &format!("{}", st.flat.vert_count()));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "CG residual", &format!("{:.2e}", st.last_residual));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.refine_radius, 0.05..=0.40).text("refine radius"));
    let _ = ui.add(egui::Slider::new(&mut st.source_speed, 0.0..=2.0).text("source ω"));
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Cells inside the refine radius around the moving source are \
         red-refined; AMR enforces 2:1 balance through propagation. \
         Edge colour = local temperature.",
    );
}

