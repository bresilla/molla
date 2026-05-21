fn build_fast_soft_impact_model() -> (mp::Model, Vec<BodyVisual>) {
    use mp::Vec3 as MVec3;
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.20);

    // Horizontal pinned cloth strip at y = 1.0.
    let cloth_n = 16u32;
    let cloth_w: f64 = 0.6;
    let cloth_y: f64 = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        MVec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        MVec3::new(cloth_w, 0.0, 0.0),
        MVec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        mp::WorldId(0),
    );
    // Pin all four edges so the strip is taut.
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
        b.pin_particle(cloth_first + ix * cloth_n);
        b.pin_particle(cloth_first + ix * cloth_n + cloth_n - 1);
    }

    // Tet cube above the cloth — 3×3×3 verts, 5cm cell size.
    let nx = 3u32;
    let dx: f64 = 0.05;
    let cube_lo_y: f64 = cloth_y + 0.30; // 30 cm above cloth.
    let cube_origin = MVec3::new(
        -dx * (nx as f64 - 1.0) * 0.5,
        cube_lo_y,
        -dx * (nx as f64 - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin + MVec3::new(ii as f64 * dx, jj as f64 * dx, kk as f64 * dx);
                b.add_particle(p, 0.05, dx * 0.4, mp::WorldId(0));
            }
        }
    }
    let cloth_count = cloth_n * cloth_n;
    let v_at =
        |i: u32, j: u32, k: u32| -> u32 { cloth_first + cloth_count + (k * nx + j) * nx + i };
    let mut tets = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                let cell_tets = jelly_cube_to_six_tets(corners);
                for tet in cell_tets {
                    tets.push(tet);
                }
            }
        }
    }
    let materials = vec![mp::MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add tet mesh");

    // Floor (so the cube has somewhere to land in the no-CCD case).
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("fast soft impact model"), Vec::new())
}

// Stage 33.5 SoftBrake — soft tet block sliding down a soft tet
// ramp. Both registered as soft surfaces with adjustable friction

fn fast_soft_impact_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );

    let mut ccd_on: bool = false;
    let mut vt_count: usize = 0;
    let mut ccd_inv: usize = 0;
    let mut ccd_snaps: usize = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        ccd_on = s.ccd_enabled;
        vt_count = s.last_vt_contact_count();
        ccd_inv = s.last_ccd_invocations();
        ccd_snaps = s.last_ccd_snaps();
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    readout(ui, "CCD invocations", &format!("{ccd_inv}"));
    readout(ui, "CCD snaps", &format!("{ccd_snaps}"));
    ui.separator();

    let mut speed = fast_soft_impact_speed() as f32;
    ui.label("Impact speed (m/s)");
    if ui.add(egui::Slider::new(&mut speed, 0.5..=15.0)).changed() {
        set_fast_soft_impact_speed(speed);
    }

    if ui
        .checkbox(&mut ccd_on, "CCD enabled (Stage 33.4)")
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.ccd_enabled = ccd_on;
        }
    }

    if ui.button("Re-launch impact").clicked() {
        sim.reset();
    }

    ui.separator();
    ui.small("Toggle CCD off + click Re-launch to see the cube tunnel.");
    ui.small("Toggle CCD on + Re-launch: the cube stops at the cloth.");
    ui.small("Stage 33.4 ships VT-CCD only; EE-CCD is deferred.");
}

