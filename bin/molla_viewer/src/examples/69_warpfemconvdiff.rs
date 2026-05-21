// ── Scene 69: WarpFemConvDiff (steady advection-diffusion) ────────
//
// Stage 37.2 retrofit — the analytic-stub Laplacian was replaced by a
// real convection-diffusion operator `(α∇² − b·∇)` with SUPG
// stabilisation (`τ = h / (2|b|)`). Convection direction is +x; left
// edge is the inflow Dirichlet (1.0) and right edge is the outflow
// Dirichlet (0.0). The non-symmetric system is solved via BiCGSTAB.
struct WarpFemConvDiffState {
    mesh: fem::FemMesh,
    /// Last steady-state solution.
    u: Vec<f64>,
    alpha: f64,
    bx: f64,
    user_reset: bool,
    last_iters: usize,
    last_residual: f64,
}
impl Default for WarpFemConvDiffState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(20, 20, 1.0, 1.0);
        Self {
            u: vec![0.0; mesh.vert_count()],
            mesh,
            alpha: 0.05,
            bx: 1.0,
            user_reset: true,
            last_iters: 0,
            last_residual: 0.0,
        }
    }
}

fn update_warp_fem_convdiff(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemConvDiff {
        return;
    }
    let st = &mut warp.fem_convdiff;
    if !st.user_reset {
        return;
    }
    // Assemble A = α·K + C + S_SUPG. The convection block uses the
    // user's velocity in +x; the SUPG block uses the cell length scale
    // `h = 1/nx` for τ = h/(2|b|).
    let nx = 20;
    let h = 1.0 / nx as f64;
    let b = molla::math::Vec3::new(st.bx, 0.0, 0.0);
    let alpha = st.alpha;
    let space = fem::FunctionSpace::P1Nodal;
    let a = fem::assemble_global(&st.mesh, space, |cell, verts| {
        let kk = fem::compute_element_stiffness(cell, verts, alpha);
        let cc = fem::compute_element_convection(cell, verts, b);
        let ss = fem::compute_element_supg_convection(cell, verts, b, 1.0, h);
        let n = cell.vert_count();
        (0..n)
            .map(|i| (0..n).map(|j| kk[i][j] + cc[i][j] + ss[i][j]).collect())
            .collect()
    })
    .expect("warp convdiff assemble");

    let mut bcs = Vec::new();
    for (i, p) in st.mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        } else if (p.x - 1.0).abs() < 1e-9 {
            bcs.push(fem::BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }

    // Apply Dirichlet to (A, f) and solve via BiCGSTAB — CG can't
    // handle the non-symmetric advection block.
    let mut a_red = a;
    let mut f = vec![0.0_f64; st.mesh.vert_count()];
    fem::apply_dirichlet(&mut a_red, &mut f, &bcs);
    let n = f.len();
    let mut x0 = vec![0.0; n];
    for bc in &bcs {
        if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }
    let res = molla::math::iterative::bicgstab_solve(
        |x| {
            let mut y = vec![0.0; n];
            a_red.mv(x, &mut y).expect("BsrMatrix mv");
            y
        },
        &f,
        &x0,
        molla::fem::solver::DEFAULT_TOL,
        molla::fem::solver::DEFAULT_MAX_ITERS,
    );
    st.u = res.x;
    st.last_iters = res.iters;
    st.last_residual = res.residual;
    st.user_reset = false;
}

fn draw_warp_fem_convdiff_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFemConvDiff {
        return;
    }
    let st = &warp.fem_convdiff;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            let pa = st.mesh.vertices[v[0] as usize];
            let pb = st.mesh.vertices[v[1] as usize];
            let pc = st.mesh.vertices[v[2] as usize];
            let to = |p: MVec3| origin + Vec3::new(p.x as f32, 0.0, p.y as f32);
            let avg = (st.u[v[0] as usize] + st.u[v[1] as usize] + st.u[v[2] as usize]) / 3.0;
            let col = fem_heat_colour(avg);
            gizmos.line(to(pa), to(pb), col);
            gizmos.line(to(pb), to(pc), col);
            gizmos.line(to(pc), to(pa), col);
        }
    }
}

fn warp_fem_convdiff_pane(ui: &mut egui::Ui, st: &mut WarpFemConvDiffState) {
    ui.label("Warp FEM convection-diffusion — port of fem/example_convection_diffusion.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.mesh.cells.len()));
    readout(ui, "CG iters", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.2e}", st.last_residual));
    ui.label("Diffusivity α");
    if ui
        .add(egui::Slider::new(&mut st.alpha, 0.001..=1.0).logarithmic(true))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Recompute").clicked() {
        st.user_reset = true;
    }
    ui.small("Steady (α∇² − b·∇)u = 0 — Stage 37.2 SUPG-stabilised");
    ui.small("convection-diffusion; non-symmetric solve via BiCGSTAB.");
}

