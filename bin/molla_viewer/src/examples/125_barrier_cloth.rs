// Scene 125 — barrier-Newton cloth (roadmap C0): a cloth grid draped
// over a sphere by the penetration-free IPC-family backend. The
// solver guarantees no interpenetration at ANY Newton iterate —
// watch the resting gap sit exactly inside the barrier band instead
// of relying on contact stiffness.

fn build_barrier_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();

    let n = 14u32;
    let size = 0.42_f64;
    let origin = MVec3::new(-size * 0.5, 0.32, -size * 0.5);
    let _ = b.add_cloth_grid(
        origin,
        MVec3::new(size, 0.0, 0.0),
        MVec3::new(0.0, 0.0, size),
        n,
        n,
        0.02,
        0.005,
        2.5e-3, // → k = 400
        1.5,
        mp::WorldId(0),
    );

    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.12,
        mp::Transform::from_translation(MVec3::new(0.0, 0.12, 0.0)),
    )
    .expect("sphere shape");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    let model = b.build().expect("model");
    (model, Vec::new())
}

fn barrier_cloth_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    ui.label("Barrier-Newton contact (IPC-family, roadmap C0).");
    ui.label("Backend: GPU (f32 kernel, device-resident trajectory).");
    ui.label("Interpenetration is impossible at every Newton iterate:");
    ui.label("the CCD-truncated line search cannot cross a surface.");
    ui.separator();
    if let SolverBackend::Barrier(s) = &sim.solver {
        ui.label(format!("newton iters (last step): {}", s.last_newton_iters));
        ui.label(format!("min gap (invariant): {:.3e} m", s.last_min_gap));
        ui.label(format!("barrier band d̂: {:.1e} m", s.params.dhat));
    }
    if let SolverBackend::Barrier(s) = &mut sim.solver {
        let mut mu = s.params.friction_mu as f32;
        if ui
            .add(egui::Slider::new(&mut mu, 0.0..=1.0).text("friction μ_k"))
            .changed()
        {
            s.params.friction_mu = mu as f64;
        }
        let mut scale = s.params.friction_mu_s_scale as f32;
        if ui
            .add(egui::Slider::new(&mut scale, 1.0..=2.0).text("μ_s / μ_k"))
            .changed()
        {
            s.params.friction_mu_s_scale = scale as f64;
        }
    }
}
