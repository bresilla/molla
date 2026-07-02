const SOFT_DROP_CLOTH_W: f64 = 1.6;
fn build_softbody_drop_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    // 1. Cloth grid hanging at Y ≈ 1.0, pinned at left + right edges.
    let origin = MVec3::new(-SOFT_DROP_CLOTH_W * 0.5, 1.0, -SOFT_DROP_CLOTH_W * 0.5);
    let u_axis = MVec3::new(SOFT_DROP_CLOTH_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, SOFT_DROP_CLOTH_W);
    let (cloth_first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        SOFT_DROP_CLOTH_NX,
        SOFT_DROP_CLOTH_NX,
        0.05,   // particle mass
        0.04,   // particle radius (larger so contact lands)
        5.0e-6, // stretch compliance
        0.05,   // max stretch
        mp::WorldId(0),
    );
    // Pin two opposite edges (iy = 0 row and iy = NX-1 row).
    for ix in 0..SOFT_DROP_CLOTH_NX {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (SOFT_DROP_CLOTH_NX - 1) * SOFT_DROP_CLOTH_NX + ix);
    }

    // 2. Tet jelly cube above the cloth at Y ≈ 2.5.
    let cube_origin = MVec3::new(
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NX as f64 - 1.0) * 0.5,
        2.5,
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NZ as f64 - 1.0) * 0.5,
    );
    for kk in 0..SOFT_DROP_TET_NZ {
        for jj in 0..SOFT_DROP_TET_NY {
            for ii in 0..SOFT_DROP_TET_NX {
                let p = cube_origin
                    + MVec3::new(
                        ii as f64 * SOFT_DROP_TET_DX,
                        jj as f64 * SOFT_DROP_TET_DX,
                        kk as f64 * SOFT_DROP_TET_DX,
                    );
                b.add_particle(p, 0.05, SOFT_DROP_TET_DX * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        // Tet particle ids start AFTER the cloth's (cloth_first + nx² particles).
        let cloth_n = SOFT_DROP_CLOTH_NX * SOFT_DROP_CLOTH_NX;
        cloth_first + cloth_n + (k * SOFT_DROP_TET_NY + j) * SOFT_DROP_TET_NX + i
    };
    let mut tets = Vec::new();
    for kk in 0..SOFT_DROP_TET_NZ - 1 {
        for jj in 0..SOFT_DROP_TET_NY - 1 {
            for ii in 0..SOFT_DROP_TET_NX - 1 {
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
    let cube_tet_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add tet mesh");
    // Stage 33.5 — friction = 0.5 keeps the cube STUCK on the cloth
    // (the Stage 33 hand-off note's "cube slowly drifts laterally
    // and slips off the cloth edge in the viewer's 2-edge-pinned
    // scene at >2 s sim time" was specifically about the
    // friction-less case).
    b.set_tet_mesh_friction(cube_tet_id, 0.5);

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("softbody drop model"), Vec::new())
}

// Stage 33.3 — ClothCollar scene constants.
const COLLAR_NX: u32 = 24;
const COLLAR_W: f64 = 0.6;
const COLLAR_TOP_Y: f64 = 1.6;
// Stage 33.3 — pinned cloth tube collapsing onto itself. The cloth
// is built as a 24×24 grid that is then pinned along its iy=0 row
// (acts as the "tube floor") and along its iy=COLLAR_NX-1 row
// (acts as the kinematic puller). The puller is driven down by
// `update_cloth_collar_example` — that brings the top edge near the
// bottom edge and the cloth folds onto itself. Without self-
// collision the layers interpenetrate; with it on (the default for

fn softbody_drop_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(
        ui,
        "Soft surfaces",
        &format!("{}", sim.model.soft_surfaces.len()),
    );
    // Stage 33 — surface the live VT contact count + k_contact slider
    // so users can verify the contact pipeline is engaged. Reads
    // from the live VBD solver instance.
    let mut k_contact: f64 = 0.0;
    let mut vt_count: usize = 0;
    let mut sweeps: i32 = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        k_contact = s.contact_stiffness;
        vt_count = s.last_vt_contact_count();
        sweeps = s.vt_contact_sweeps as i32;
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    ui.separator();

    ui.label("k_contact (N/m)");
    if ui
        .add(egui::Slider::new(&mut k_contact, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
        && let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = k_contact;
        }
    ui.label("VT contact sweeps");
    if ui.add(egui::Slider::new(&mut sweeps, 1..=16)).changed()
        && let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vt_contact_sweeps = sweeps.max(1) as usize;
        }
    ui.separator();
    ui.small("Tet jelly cube falls onto a hanging VBD cloth.");
    ui.small("Stage 33 cross-mesh VT contact: the jelly should");
    ui.small("rest on the cloth instead of passing through.");
}

// Stage 33.3 — ClothCollar right-rail pane. Surfaces the live VT
// contact count + a self-collision toggle (default ON for this
