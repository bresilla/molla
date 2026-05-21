const FEM_HEAT_P2_GRID: usize = 8;

#[derive(Resource)]
struct FemHeatP2ExampleState {
    mesh: fem::FemMesh,
    /// P2 DoF map (vertex DoFs + edge DoFs).
    dof_map: fem::P2DofMap,
    /// Mass matrix M (consistent, P2).
    mass: molla::math::sparse::BsrMatrix,
    /// Stiffness matrix K (P2).
    stiff: molla::math::sparse::BsrMatrix,
    alpha: f64,
    last_alpha_baked: f64,
    dt: f64,
    /// Dirichlet BCs over both corner DoFs (top + bottom faces) and
    /// the matching edge DoFs (edges with both endpoints on a face).
    bcs: Vec<fem::BoundaryCondition>,
    /// Current temperature field, length = `dof_map.ndof()`.
    u: Vec<f64>,
    last_residual: f64,
    last_iters: usize,
    needs_reset: bool,
    user_reset: bool,
    /// P1-baseline DoF count at the same mesh — for the side-by-side
    /// pane readout. Cached at construction since the mesh is fixed.
    p1_dof_count: usize,
}

impl Default for FemHeatP2ExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tet_grid_3d(
            FEM_HEAT_P2_GRID,
            FEM_HEAT_P2_GRID,
            FEM_HEAT_P2_GRID,
            1.0,
            1.0,
            1.0,
        );
        let dof_map = fem::P2DofMap::build(&mesh);
        let mass = fem::assemble_global_p2(&mesh, &dof_map, fem::compute_element_p2_mass)
            .expect("FEM heat P2: assemble mass");
        let stiff = fem::assemble_global_p2(&mesh, &dof_map, |c, v| {
            fem::compute_element_p2_stiffness(c, v, FEM_HEAT_DEFAULT_ALPHA)
        })
        .expect("FEM heat P2: assemble stiffness");
        let bcs = fem_heat_p2_dirichlet_bcs(&mesh, &dof_map);
        let n = dof_map.ndof();
        let mut u = vec![0.0_f64; n];
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = *bc {
                u[vertex as usize] = value;
            }
        }
        let p1_dof_count = mesh.vert_count();
        Self {
            mesh,
            dof_map,
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
            p1_dof_count,
        }
    }
}

fn build_fem_heat_p2_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("FEM heat P2: empty model"), Vec::new())
}

// Per-frame system: mirrors `update_fem_heat_example` but for the
// P2 scene.
fn update_fem_heat_p2_example(
    sim: Res<Sim>,
    mut state: ResMut<FemHeatP2ExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::FemHeatP2 {
        return;
    }
    if state.needs_reset || state.user_reset {
        state.reset_temperature_field();
        state.needs_reset = false;
        state.user_reset = false;
    }
    if (state.alpha - state.last_alpha_baked).abs() > 1e-12 {
        state.rebuild_stiffness();
    }
    if run.paused {
        return;
    }
    let dt = state.dt.max(1e-6);
    let n = state.dof_map.ndof();
    let f = vec![0.0_f64; n];
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

// Draw the P2 FEM heat tet wireframe with per-edge colour from the
// edge-midpoint DoF temperature (which is what P2 actually solves
// for — a true mid-edge sample, not an interpolation of corners).
fn draw_fem_heat_p2_gizmos(sim: Res<Sim>, state: Res<FemHeatP2ExampleState>, gizmos: Gizmos) {
    if sim.scene != Scene::FemHeatP2 {
        return;
    }
    let _ = (state, gizmos);
}

// Empty molla model — the FEM beam demo carries its own mesh +
// solver in `FemBeamExampleState`; the standard pipeline is

fn fem_heat_p2_pane(ui: &mut egui::Ui, state: &mut FemHeatP2ExampleState) {
    ui.label("FEM heat equation — Stage 36.3 P2");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            FEM_HEAT_P2_GRID,
            state.mesh.vert_count()
        ),
    );
    let p2_dofs = state.u.len();
    let p1_dofs = state.p1_dof_count;
    let edge_dofs = p2_dofs - p1_dofs;
    readout(ui, "P1 DoFs (verts)", &format!("{p1_dofs}"));
    readout(
        ui,
        "P2 DoFs (verts + edges)",
        &format!("{p2_dofs}  (+{edge_dofs} edges)"),
    );
    let ratio = if p1_dofs > 0 {
        p2_dofs as f32 / p1_dofs as f32
    } else {
        0.0
    };
    readout(ui, "P2 / P1 DoF ratio", &format!("{ratio:.2}×"));
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
    ui.small("Same problem as #47, but P2 (quadratic) basis.");
    ui.small("Edges carry their own DoF — colour reflects mid-edge T.");
}

// Right-rail pane for `Scene::FemBeam`. Material + load sliders,

