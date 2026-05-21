// Stage 33.3 — ClothCollar pane / update-system shared state. Since
// SystemParam bundles are at the 16-tuple ceiling, we route per-
// scene state through static atomics (the same pattern Stage 32's
// arm-mimic toggle uses). The pane writes; the update system + the
// scene rebuild read.
static COLLAR_SELF_COLL_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
// Reset signal — the pane sets this when the user wants to rewind
// the puller height. The update system checks + clears it each
// frame.
static COLLAR_RESET_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
// Puller height (m, current Y of the iy=NX-1 row). The update
// system mutates this each frame; the pane reads it for display.
static COLLAR_PULLER_Y: std::sync::atomic::Ordering = std::sync::atomic::Ordering::Relaxed;
// We store the puller Y as bits because AtomicF64 is unstable.
static COLLAR_PULLER_Y_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn collar_get_puller_y() -> f64 {
    f64::from_bits(COLLAR_PULLER_Y_BITS.load(COLLAR_PULLER_Y))
}
fn collar_set_puller_y(y: f64) {
    COLLAR_PULLER_Y_BITS.store(y.to_bits(), COLLAR_PULLER_Y);
}

fn build_cloth_collar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let cy: f64 = COLLAR_TOP_Y;
    let origin = MVec3::new(-COLLAR_W * 0.5, cy, -COLLAR_W * 0.5);
    let u_axis = MVec3::new(COLLAR_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, COLLAR_W);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        COLLAR_NX,
        COLLAR_NX,
        0.05,   // particle mass
        0.02,   // particle radius
        5.0e-6, // stretch compliance
        0.05,   // max stretch
        mp::WorldId(0),
    );
    // Pin two opposite long edges: iy=0 row stays fixed; iy=NX-1 row
    // is the kinematic puller (pinned + advected by the update sys).
    for ix in 0..COLLAR_NX {
        b.pin_particle(first + ix);
        b.pin_particle(first + (COLLAR_NX - 1) * COLLAR_NX + ix);
    }
    // Stage 33.3 — opt this cloth into self-collision (default ON for
    // the scene; the pane lets the user toggle it off for A/B).
    b.set_cloth_mesh_self_collision(0, true);

    // Floor plane so the cloth has something to settle against.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth collar model"), Vec::new())
}

// Stage 33.4 — FastSoftImpact constants. Cloth grid is 16x16; the
// "small horizontal cloth strip" of the spec is honoured by pinning
// all four edges so the central area takes the impact.
const FAST_SOFT_IMPACT_CLOTH_NX: u32 = 16;

// Atomic-backed slider state for impact speed (m/s). Default 5 m/s
// (the spec-stated impact speed). The right-rail pane writes here and
// the `finalize` / `reset` paths read it to (re)set the cube's qd.
// We store the f32 bit pattern in the AtomicU32. A pristine zero
// value (haven't written yet) is treated as "use the default".
static FAST_SOFT_IMPACT_SPEED_M_S: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

fn fast_soft_impact_speed() -> f64 {
    let bits = FAST_SOFT_IMPACT_SPEED_M_S.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        return 5.0;
    }
    let stored = f32::from_bits(bits) as f64;
    if stored.is_finite() && stored > 0.0 {
        stored
    } else {
        5.0
    }
}

fn set_fast_soft_impact_speed(v: f32) {
    FAST_SOFT_IMPACT_SPEED_M_S.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

// Stage 33.4 — FastSoftImpact scene. A small (3×3×3 vert) tet cube
// is launched downward at ~5 m/s into a horizontal pinned cloth strip
// (16×4). With CCD on (default for this scene) the cube stops at the
// cloth; with CCD off it tunnels through. Right-rail pane carries

fn update_cloth_collar_example(time: Res<Time>, mut sim: ResMut<Sim>) {
    if sim.scene != Scene::ClothCollar {
        return;
    }
    if sim.model.particle_count == 0 {
        return;
    }
    // Pinned-row indices: iy=NX-1, ix∈[0, NX). Particles are laid out
    // row-major over (ix, iy) starting at particle 0 (this scene has
    // no other particles before the cloth).
    let nx = COLLAR_NX as usize;
    let row_start = (nx - 1) * nx;
    // Initialise puller Y on first frame (when the bits are still 0,
    // f64::from_bits(0) == 0.0; we treat that as "uninitialised").
    let mut y = collar_get_puller_y();
    if y == 0.0 {
        y = COLLAR_TOP_Y;
        collar_set_puller_y(y);
    }
    if COLLAR_RESET_REQUESTED.swap(false, std::sync::atomic::Ordering::Relaxed) {
        y = COLLAR_TOP_Y;
        collar_set_puller_y(y);
        // Snap puller particles back up.
        if let Ok(q) = sim.state_a.particle_q.host_mut() {
            for ix in 0..nx {
                let idx = row_start + ix;
                if idx < q.len() {
                    let p = q[idx];
                    q[idx] = MVec3::new(p.x, y, p.z);
                }
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.particle_q.ensure_device(&device);
            let _ = sim.state_a.particle_q.upload(&queue);
        }
        return;
    }
    // Descend at 0.04 m/s — slow enough that the cloth folds without
    // the puller punching through (the contact stiffness needs time
    // to engage). Stop at the bottom edge's nominal Y (the iy=0 row
    // sits at COLLAR_TOP_Y exactly when the cloth is flat — but it
    // sags slightly under gravity; we descend to 0.02 m above the
    // top floor so the layers visibly stack).
    let descend_speed: f64 = 0.04;
    let target_y: f64 = 0.02;
    if y > target_y {
        y -= descend_speed * time.delta_secs() as f64;
        if y < target_y {
            y = target_y;
        }
        collar_set_puller_y(y);
        if let Ok(q) = sim.state_a.particle_q.host_mut() {
            for ix in 0..nx {
                let idx = row_start + ix;
                if idx < q.len() {
                    let p = q[idx];
                    q[idx] = MVec3::new(p.x, y, p.z);
                }
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.particle_q.ensure_device(&device);
            let _ = sim.state_a.particle_q.upload(&queue);
        }
    }
}

fn cloth_collar_pane(ui: &mut egui::Ui, sim: &mut Sim) {
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(
        ui,
        "Cloth tris",
        &format!(
            "{}",
            sim.model
                .soft_surfaces
                .first()
                .map(|s| s.tri_count)
                .unwrap_or(0)
        ),
    );
    readout(ui, "Puller Y (m)", &format!("{:.3}", collar_get_puller_y()));

    let mut vt_count: usize = 0;
    let mut k_contact: f64 = 0.0;
    let mut sweeps: i32 = 0;
    if let SolverBackend::Vbd(s) = &sim.solver {
        k_contact = s.contact_stiffness;
        vt_count = s.last_vt_contact_count();
        sweeps = s.vt_contact_sweeps as i32;
    }
    readout(ui, "VT contacts", &format!("{vt_count}"));
    ui.separator();

    // Self-collision toggle. Mirrors the static atomic AND the live
    // model so the gather sees the change immediately.
    let mut self_coll = COLLAR_SELF_COLL_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    if ui
        .checkbox(&mut self_coll, "Self-collision (Stage 33.3)")
        .changed()
    {
        COLLAR_SELF_COLL_ENABLED.store(self_coll, std::sync::atomic::Ordering::Relaxed);
        // Apply live to the active model — the next gather picks it up.
        sim.model.set_soft_surface_self_collision(0, self_coll);
    }

    ui.label("k_contact (N/m)");
    if ui
        .add(egui::Slider::new(&mut k_contact, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = k_contact;
        }
    }
    ui.label("VT contact sweeps");
    if ui.add(egui::Slider::new(&mut sweeps, 1..=16)).changed() {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vt_contact_sweeps = sweeps.max(1) as usize;
        }
    }
    ui.separator();
    if ui.button("Reset puller height").clicked() {
        COLLAR_RESET_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    ui.separator();
    ui.small("24×24 cloth, two long edges pinned. The top edge is");
    ui.small("kinematically pulled down onto the bottom edge so the");
    ui.small("cloth folds onto itself. With Stage 33.3 self-collision");
    ui.small("ON the layers stack; OFF, they interpenetrate.");
}

// Stage 33.5 SoftBrake pane — Coulomb friction sweep over the

