// ── Scene 74: WarpFemDistortion ───────────────────────────────────
//
// Stage 37.4 retrofit — runs the real symmetric-Dirichlet energy
// kernel + Newton minimisation on a small 2D mesh. The state holds
// both the initial-energy reading (from a small perturbation of the
// rest mesh) and the post-Newton energy so the pane can show the
// solver pulling the mesh back to E = 0.
struct WarpFemDistortionState {
    /// Initial perturbed energy.
    initial_energy: f64,
    /// Final energy after Newton.
    energy: f64,
    /// Newton iteration cap.
    max_iter: usize,
    user_reset: bool,
}
impl Default for WarpFemDistortionState {
    fn default() -> Self {
        Self {
            initial_energy: 0.0,
            energy: 0.0,
            max_iter: 10,
            user_reset: true,
        }
    }
}

fn update_warp_fem_distortion(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemDistortion {
        return;
    }
    let st = &mut warp.fem_distortion;
    if !st.user_reset {
        return;
    }
    // Build a small 4×4 tri grid and perturb the interior vertices
    // by a deterministic ±2e-2 offset. Hold the boundary fixed so
    // Newton has a unique minimum to walk towards.
    let mesh = fem::FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let rest = mesh.vertices.clone();
    let mut init = rest.clone();
    for (i, p) in init.iter_mut().enumerate() {
        let on_boundary = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if !on_boundary {
            let h1 = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
            let h2 = ((i as u64).wrapping_mul(40503)) as f64 / u64::MAX as f64;
            p.x += (2.0 * h1 - 1.0) * 2.0e-2;
            p.y += (2.0 * h2 - 1.0) * 2.0e-2;
        }
    }
    let mut e0: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &init);
        e0 += e;
    }
    st.initial_energy = e0;
    let final_def = fem::solve_distortion_minimization(&mesh, &init, st.max_iter, 1e-12);
    let mut ef: f64 = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = fem::compute_element_distortion_energy(cell, &rest, &final_def);
        ef += e;
    }
    st.energy = ef;
    st.user_reset = false;
}

fn warp_fem_distortion_pane(ui: &mut egui::Ui, st: &mut WarpFemDistortionState) {
    ui.label("Warp FEM distortion energy — port of fem/example_distortion_energy.py");
    ui.separator();
    readout(
        ui,
        "Initial perturbed E",
        &format!("{:.4e}", st.initial_energy),
    );
    readout(ui, "Final E (post-Newton)", &format!("{:.4e}", st.energy));
    if ui.button("Recompute").clicked() {
        st.user_reset = true;
    }
    ui.small("Stage 37.4 — symmetric Dirichlet E = ½ tr(FᵀF + (FᵀF)⁻¹) − 2");
    ui.small("on 4×4 tri grid; Newton drives perturbed mesh back to E = 0.");
}

