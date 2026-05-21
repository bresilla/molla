fn build_soft_brake_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.05);

    // Tilt the WORLD gravity (Newton-style "tilt the world" trick).
    let ramp_tilt: f64 = 25.0_f64.to_radians();
    b.set_world_gravity(
        mp::WorldId(0),
        MVec3::new(9.81 * ramp_tilt.sin(), -9.81 * ramp_tilt.cos(), 0.0),
    );

    // Ramp: a thick (1.5 × 0.10 × 0.6 m) kinematic tet slab at y = 1.0.
    let ramp_lo = MVec3::new(-0.75, 1.0, -0.30);
    let ramp_hi = MVec3::new(0.75, 1.10, 0.30);
    let mut ramp_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[ramp_lo.z, ramp_hi.z] {
        for &y in &[ramp_lo.y, ramp_hi.y] {
            for &x in &[ramp_lo.x, ramp_hi.x] {
                let id = b.add_particle(MVec3::new(x, y, z), 0.0, 0.02, mp::WorldId(0));
                ramp_verts.push(id);
            }
        }
    }
    let rv: [u32; 8] = ramp_verts.as_slice().try_into().unwrap();
    let ramp_tets = jelly_cube_to_six_tets(rv);
    let ramp_id = b
        .add_tet_mesh(&ramp_tets, 1.0e6, 0.3, None)
        .expect("ramp tet block");
    b.set_tet_mesh_friction(ramp_id, 0.5);

    // Block: a 3×3×3 jelly cube placed just above the ramp.
    let bnx: u32 = 3;
    let bdx: f64 = 0.10;
    let block_origin = MVec3::new(-0.50, 1.10 + 0.005, -bdx * (bnx as f64 - 1.0) * 0.5);
    let block_first = 8u32;
    for kk in 0..bnx {
        for jj in 0..bnx {
            for ii in 0..bnx {
                let p =
                    block_origin + MVec3::new(ii as f64 * bdx, jj as f64 * bdx, kk as f64 * bdx);
                b.add_particle(p, 0.05, bdx * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at_b = |i: u32, j: u32, k: u32| -> u32 { block_first + (k * bnx + j) * bnx + i };
    let mut block_tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..bnx - 1 {
        for jj in 0..bnx - 1 {
            for ii in 0..bnx - 1 {
                let corners = [
                    v_at_b(ii, jj, kk),
                    v_at_b(ii + 1, jj, kk),
                    v_at_b(ii, jj + 1, kk),
                    v_at_b(ii + 1, jj + 1, kk),
                    v_at_b(ii, jj, kk + 1),
                    v_at_b(ii + 1, jj, kk + 1),
                    v_at_b(ii, jj + 1, kk + 1),
                    v_at_b(ii + 1, jj + 1, kk + 1),
                ];
                block_tets.extend_from_slice(&jelly_cube_to_six_tets(corners));
            }
        }
    }
    let materials = vec![mp::MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; block_tets.len()];
    let block_id = b
        .add_tet_mesh_per_material(&block_tets, &materials, &mat_idx, None)
        .expect("block tet mesh");
    b.set_tet_mesh_friction(block_id, 0.5);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("soft brake model"), Vec::new())
}

// ── Per-scene state stubs for Newton-port scenes (defaults populated
// once the scene is implemented). These are insert_resource'd in main()
// and bundled into SceneStates / PaneSceneStates.


fn soft_brake_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );
    let mut vt_count: usize = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        vt_count = s.last_vt_contact_count();
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));

    let inv_mass = sim
        .model
        .particle_inv_mass
        .host()
        .ok()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let mut block_idx: Option<usize> = None;
    for (i, s) in sim.model.soft_surfaces.iter().enumerate() {
        let any_dynamic = (s.vertex_offset..s.vertex_offset + s.vertex_count)
            .any(|v| inv_mass.get(v as usize).copied().unwrap_or(0.0) > 0.0);
        if any_dynamic {
            block_idx = Some(i);
            break;
        }
    }
    ui.separator();

    if let Some(bi) = block_idx {
        let mut friction = sim.model.soft_surfaces[bi].friction as f32;
        ui.label("Block friction μ");
        if ui
            .add(egui::Slider::new(&mut friction, 0.0_f32..=1.0_f32))
            .changed()
        {
            sim.model.soft_surfaces[bi].friction = friction as mp::Real;
        }
        ui.separator();
        let cur_friction = sim.model.soft_surfaces[bi].friction;
        readout(ui, "block μ", &format!("{cur_friction:.2}"));
    }
    ui.separator();
    ui.small("Tet block on a tilted (25°) tet ramp.");
    ui.small("μ = 1.0 → sticks; μ = 0.5 → slips slowly;");
    ui.small("μ = 0.0 → slides freely.");
}

