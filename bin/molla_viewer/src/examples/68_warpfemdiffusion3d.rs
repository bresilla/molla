// ── Scene 68: WarpFemDiffusion3D (animated BC) ────────────────────
struct WarpFemDiffusion3DState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    /// Bottom + top vertex indices (BCs are time-varying).
    bottom: Vec<u32>,
    top: Vec<u32>,
    u: Vec<f64>,
    alpha: f64,
    /// Diffusivity the `stiff` matrix was last assembled with; the
    /// stiffness is rebuilt when the `alpha` slider moves off it.
    last_alpha: f64,
    dt: f64,
    sim_t: f64,
    user_reset: bool,
}
impl Default for WarpFemDiffusion3DState {
    fn default() -> Self {
        let n = 8;
        let mesh = fem::FemMesh::tet_grid_3d(n, n, n, 1.0, 1.0, 1.0);
        let space = fem::FunctionSpace::P1Nodal;
        let mass = fem::assemble_global(&mesh, space, fem::compute_element_mass)
            .expect("warp diff3d mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, 1.0)
        })
        .expect("warp diff3d stiff");
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.z.abs() < 1e-9 {
                bottom.push(i as u32);
            }
            if (p.z - 1.0).abs() < 1e-9 {
                top.push(i as u32);
            }
        }
        Self {
            u: vec![0.0; mesh.vert_count()],
            mesh,
            mass,
            stiff,
            bottom,
            top,
            alpha: 1.0,
            last_alpha: 1.0,
            dt: 0.02,
            sim_t: 0.0,
            user_reset: false,
        }
    }
}

fn update_warp_fem_diffusion_3d(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemDiffusion3D {
        return;
    }
    let st = &mut warp.fem_diff3d;
    if st.user_reset {
        st.u = vec![0.0; st.mesh.vert_count()];
        st.sim_t = 0.0;
        st.user_reset = false;
    }
    // Rebuild the stiffness matrix when the diffusivity slider moves —
    // `compute_element_stiffness` bakes α in, so a stale matrix would
    // leave the α slider inert (it was assembled once at α = 1.0).
    if (st.alpha - st.last_alpha).abs() > 1e-12 {
        let space = fem::FunctionSpace::P1Nodal;
        st.stiff = fem::assemble_global(&st.mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, st.alpha)
        })
        .expect("warp diff3d rebuild");
        st.last_alpha = st.alpha;
    }
    if run.paused {
        return;
    }
    st.sim_t += time.delta_secs() as f64;
    let bottom_v = (st.sim_t * 1.5).sin() * 0.5 + 0.5;
    let top_v = (st.sim_t * 1.5).cos() * 0.5 + 0.5;
    let mut bcs = Vec::with_capacity(st.bottom.len() + st.top.len());
    for &v in &st.bottom {
        bcs.push(fem::BoundaryCondition::Dirichlet {
            vertex: v,
            value: bottom_v,
        });
    }
    for &v in &st.top {
        bcs.push(fem::BoundaryCondition::Dirichlet {
            vertex: v,
            value: top_v,
        });
    }
    let f = vec![0.0; st.mesh.vert_count()];
    let res = molla::fem::solver::solve_transient_with_initial(
        &st.mass,
        &st.stiff,
        &f,
        &st.u.clone(),
        st.dt,
        &bcs,
        &st.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    st.u = res.x;
}

fn draw_warp_fem_diffusion_3d_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemDiffusion3D {
        return;
    }
    let st = &warp.fem_diff3d;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for cell in &st.mesh.cells {
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
                let pa = st.mesh.vertices[a as usize];
                let pb = st.mesh.vertices[b as usize];
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                let t = 0.5 * (st.u[a as usize] + st.u[b as usize]);
                gizmos.line(av, bv, fem_heat_colour(t));
            }
        }
    }
}

fn warp_fem_diffusion_3d_pane(ui: &mut egui::Ui, st: &mut WarpFemDiffusion3DState) {
    ui.label("Warp FEM 3D diffusion — port of fem/example_diffusion_3d.py");
    ui.separator();
    readout(ui, "Tets", &format!("{}", st.mesh.cells.len()));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    ui.label("α");
    let _ = ui.add(egui::Slider::new(&mut st.alpha, 0.1..=10.0).logarithmic(true));
    ui.label("dt");
    let _ = ui.add(egui::Slider::new(&mut st.dt, 0.001..=0.2).logarithmic(true));
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("Time-varying Dirichlet BCs (sin/cos on top + bottom faces).");
}

