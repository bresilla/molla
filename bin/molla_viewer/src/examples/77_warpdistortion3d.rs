// ── Scene 77: WarpDistortion3D ────────────────────────────────────
//
// Stage 37.4 retrofit — 3D companion to `WarpFemDistortion`. Builds a
// 3×3×3 tet grid, perturbs the interior vertices, and runs the same
// Newton minimisation on the symmetric Dirichlet energy.
struct WarpDistortion3DState {
    initial_energy: f64,
    energy: f64,
    max_iter: usize,
    user_reset: bool,
}
impl Default for WarpDistortion3DState {
    fn default() -> Self {
        Self {
            initial_energy: 0.0,
            energy: 0.0,
            max_iter: 10,
            user_reset: true,
        }
    }
}

fn update_warp_distortion_3d(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpDistortion3D {
        return;
    }
    let st = &mut warp.distortion_3d;
    if !st.user_reset {
        return;
    }
    let mesh = fem::FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let rest = mesh.vertices.clone();
    let mut init = rest.clone();
    for (i, p) in init.iter_mut().enumerate() {
        let on_boundary = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9
            || p.z.abs() < 1e-9
            || (p.z - 1.0).abs() < 1e-9;
        if !on_boundary {
            let h1 = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
            let h2 = ((i as u64).wrapping_mul(40503)) as f64 / u64::MAX as f64;
            let h3 = ((i as u64).wrapping_mul(2246822519)) as f64 / u64::MAX as f64;
            p.x += (2.0 * h1 - 1.0) * 1.0e-2;
            p.y += (2.0 * h2 - 1.0) * 1.0e-2;
            p.z += (2.0 * h3 - 1.0) * 1.0e-2;
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

fn warp_distortion_3d_pane(ui: &mut egui::Ui, st: &mut WarpDistortion3DState) {
    ui.label("Warp distortion 3D (bonus, fem/example_distortion_energy.py extension)");
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
    ui.small("Stage 37.4 — symmetric Dirichlet E = ½ tr(FᵀF + (FᵀF)⁻¹) − 3");
    ui.small("on 3×3×3 tet grid; Newton minimises energy back to E ≈ 0.");
}

