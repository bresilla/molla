// ── Scene 67: WarpFemDiffusion (2D triangular) ────────────────────
struct WarpFemDiffusionState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    u: Vec<f64>,
    alpha: f64,
    last_alpha: f64,
    dt: f64,
    user_reset: bool,
    last_iters: usize,
    last_residual: f64,
}
impl Default for WarpFemDiffusionState {
    fn default() -> Self {
        let nx = 16;
        let ny = 16;
        let mesh = fem::FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
        let space = fem::FunctionSpace::P1Nodal;
        let mass =
            fem::assemble_global(&mesh, space, fem::compute_element_mass).expect("warp diff mass");
        let stiff = fem::assemble_global(&mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, 1.0)
        })
        .expect("warp diff stiff");
        // Dirichlet on left + right vertical edges.
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            } else if (p.x - 1.0).abs() < 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 1.0,
                });
            }
        }
        let n = mesh.vert_count();
        let mut u = vec![0.0; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh,
            mass,
            stiff,
            bcs,
            u,
            alpha: 1.0,
            last_alpha: 1.0,
            dt: 0.02,
            user_reset: false,
            last_iters: 0,
            last_residual: 0.0,
        }
    }
}

fn update_warp_fem_diffusion(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemDiffusion {
        return;
    }
    let st = &mut warp.fem_diff;
    if st.user_reset {
        let n = st.mesh.vert_count();
        st.u = vec![0.0; n];
        for bc in &st.bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                st.u[*vertex as usize] = *value;
            }
        }
        st.user_reset = false;
    }
    if (st.alpha - st.last_alpha).abs() > 1e-12 {
        let space = fem::FunctionSpace::P1Nodal;
        st.stiff = fem::assemble_global(&st.mesh, space, |c, v| {
            fem::compute_element_stiffness(c, v, st.alpha)
        })
        .expect("warp diff rebuild");
        st.last_alpha = st.alpha;
    }
    if run.paused {
        return;
    }
    let n = st.mesh.vert_count();
    let f = vec![0.0_f64; n];
    let res = molla::fem::solver::solve_transient_with_initial(
        &st.mass,
        &st.stiff,
        &f,
        &st.u.clone(),
        st.dt.max(1e-6),
        &st.bcs,
        &st.u.clone(),
        molla::fem::solver::DEFAULT_MAX_ITERS,
        molla::fem::solver::DEFAULT_TOL,
    );
    st.u = res.x;
    st.last_iters = res.iters;
    st.last_residual = res.residual;
}

fn draw_warp_fem_diffusion_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFemDiffusion {
        return;
    }
    let st = &warp.fem_diff;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let pa = st.mesh.vertices[v[0] as usize];
            let pb = st.mesh.vertices[v[1] as usize];
            let pc = st.mesh.vertices[v[2] as usize];
            let to = |p: MVec3, vi: u32| {
                origin + Vec3::new(p.x as f32, st.u[vi as usize] as f32 * 0.4, p.y as f32)
            };
            let a = to(pa, v[0]);
            let b = to(pb, v[1]);
            let c = to(pc, v[2]);
            let avg = (st.u[v[0] as usize] + st.u[v[1] as usize] + st.u[v[2] as usize]) / 3.0;
            let col = fem_heat_colour(avg);
            gizmos.line(a, b, col);
            gizmos.line(b, c, col);
            gizmos.line(c, a, col);
        }
    }
}

fn warp_fem_diffusion_pane(ui: &mut egui::Ui, st: &mut WarpFemDiffusionState) {
    ui.label("Warp FEM 2D diffusion — port of fem/example_diffusion.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.mesh.cells.len()));
    readout(ui, "Verts", &format!("{}", st.mesh.vert_count()));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.2e}", st.last_residual));
    ui.label("Diffusivity α");
    let _ = ui.add(egui::Slider::new(&mut st.alpha, 0.05..=10.0).logarithmic(true));
    ui.label("dt");
    let _ = ui.add(egui::Slider::new(&mut st.dt, 0.001..=0.2).logarithmic(true));
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("BCs: T(x=0) = 0, T(x=1) = 1; Neumann on top/bottom.");
}

