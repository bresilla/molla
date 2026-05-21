// ═══ Scene 86: CableTwist (Newton port) ════════════════════════════════
//
// One CABLE polyline pinned at both ends. The bottom-right corner of the
// path is rotated about the cable's local tangent each frame, injecting
// twist that propagates along the cable. The polyline carries a 90°
// bend mid-route (path goes +X then +Y) so the writhe is visually
// obvious where the twist energy concentrates at the elbow.
//
// Newton ref: `xtra/newton/newton/examples/cable/example_cable_twist.py`.

const TWIST_PARTICLES: u32 = 48;
const TWIST_LEG_LEN: f64 = 0.9;
const TWIST_RADIUS: f64 = 0.012;
const TWIST_PARTICLE_MASS: f64 = 0.02;
const TWIST_STIFFNESS: f64 = 1.0e6;
const TWIST_BEND_STIFFNESS: f64 = 5.0e2;
const TWIST_DRIVEN_END_Y: f64 = 0.20;

struct CableTwistState {
    /// Accumulated twist angle of the driven endpoint (radians).
    twist_angle: f32,
    /// Twist rate (rad/s) — controlled from the pane.
    twist_rate: f32,
    /// Pause just the spin (not the physics).
    paused: bool,
    /// Live readout: mean particle Y at the elbow, surfaced so the
    /// user can confirm the cable is actually writhing.
    elbow_y: f64,
    user_reset: bool,
}

impl Default for CableTwistState {
    fn default() -> Self {
        Self {
            twist_angle: 0.0,
            twist_rate: 1.5,
            paused: false,
            elbow_y: 0.0,
            user_reset: false,
        }
    }
}

// Build a single L-shaped cable: leg 1 goes +X from origin, leg 2 goes
// +Y from the corner. Endpoint pins (first + last) are added via
// `pin_particle`. The driven endpoint (last particle) gets re-written
// each frame by `update_cable_twist`.
fn build_cable_twist_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);
    b.set_angular_damping(0.10);

    let n = TWIST_PARTICLES;
    let half = n / 2;
    let start = MVec3::new(-TWIST_LEG_LEN, TWIST_DRIVEN_END_Y, 0.0);
    let corner = MVec3::new(0.0, TWIST_DRIVEN_END_Y, 0.0);
    let end = MVec3::new(0.0, TWIST_DRIVEN_END_Y + TWIST_LEG_LEN, 0.0);

    // Lay out particles along the two legs by hand so we get a clean
    // 90° elbow. We share the elbow particle between the legs.
    let mut first: u32 = u32::MAX;
    let mut last: u32 = 0;
    for i in 0..half {
        let t = i as f64 / half as f64;
        let p = start + (corner - start) * t;
        let id = b.add_particle(p, TWIST_PARTICLE_MASS, TWIST_RADIUS, mp::WorldId(0));
        if first == u32::MAX {
            first = id;
        }
        last = id;
    }
    // Corner + leg 2.
    let leg2 = n - half;
    for i in 0..=leg2 {
        let t = i as f64 / leg2 as f64;
        let p = corner + (end - corner) * t;
        let id = b.add_particle(p, TWIST_PARTICLE_MASS, TWIST_RADIUS, mp::WorldId(0));
        last = id;
    }
    let ids: Vec<u32> = (first..=last).collect();
    let cable_id = b.add_cable(&ids, TWIST_STIFFNESS, 0.02);
    b.set_cable_bending_stiffness(cable_id, TWIST_BEND_STIFFNESS);

    // Pin both endpoints. The driven (last) end is rewritten each
    // frame by `update_cable_twist`; the first stays at `start`.
    b.pin_particle(first);
    b.pin_particle(last);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cable twist model"), Vec::new())
}

fn update_cable_twist(
    mut sim: ResMut<Sim>,
    time: Res<Time>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::CableTwist {
        return;
    }
    let st = &mut warp.cable_twist;
    if st.user_reset {
        st.twist_angle = 0.0;
        st.user_reset = false;
    }
    if !run.paused && !st.paused {
        st.twist_angle += st.twist_rate * time.delta_secs();
    }
    let n = sim.model.particle_count;
    if n < 4 {
        return;
    }
    // The driven endpoint is the LAST particle. We rotate it around
    // the tangent (which at the end of leg 2 points +Y) so the
    // displacement traces a tiny circle in the XZ plane around the
    // anchor. The cable's bending energy carries the twist back along
    // the leg toward the elbow.
    let r = 0.04_f64; // wiggle radius
    let anchor = MVec3::new(0.0, TWIST_DRIVEN_END_Y + TWIST_LEG_LEN, 0.0);
    let a = st.twist_angle as f64;
    let driven = anchor + MVec3::new(r * a.cos(), 0.0, r * a.sin());
    if let Ok(pq) = sim.state_a.particle_q.host_mut() {
        if let Some(last) = pq.last_mut() {
            *last = driven;
        }
        // Sample the elbow Y for the readout. The elbow particle is
        // roughly at index `TWIST_PARTICLES / 2` (where leg 1 meets
        // leg 2 — small off-by-one is fine, it's a readout).
        let elbow_idx = (TWIST_PARTICLES / 2) as usize;
        if elbow_idx < pq.len() {
            st.elbow_y = pq[elbow_idx].y;
        }
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.particle_q.ensure_device(&device);
        let _ = sim.state_a.particle_q.upload(&queue);
    }
}

fn draw_cable_twist_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::CableTwist {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let n = sim.model.particle_count;
    if n < 2 {
        return;
    }
    // Cable polyline — warm yellow.
    let cable_col = Color::srgb(0.95, 0.80, 0.20);
    for i in 0..n - 1 {
        let a = pq[i];
        let b = pq[i + 1];
        gizmos.line(
            Vec3::new(a.x as f32, a.y as f32, a.z as f32),
            Vec3::new(b.x as f32, b.y as f32, b.z as f32),
            cable_col,
        );
    }
    // Pin markers at endpoints.
    let pin_col = Color::srgb(1.0, 0.4, 0.4);
    for &i in [0_usize, n - 1].iter() {
        let p = pq[i];
        gizmos.sphere(
            Vec3::new(p.x as f32, p.y as f32, p.z as f32),
            0.025,
            pin_col,
        );
    }
    // Frame triad at the driven endpoint — three short axes so the
    // user can see the twist rotation directly.
    let st = &warp.cable_twist;
    let p = pq[n - 1];
    let centre = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
    let a = st.twist_angle;
    let r = 0.08;
    // The driven-end tangent points +Y; "right" rotates in the XZ
    // plane with `twist_angle`.
    let right = Vec3::new(a.cos() * r, 0.0, a.sin() * r);
    let up = Vec3::new(-a.sin() * r, 0.0, a.cos() * r);
    let fwd = Vec3::new(0.0, r, 0.0);
    gizmos.line(centre, centre + right, Color::srgb(1.0, 0.3, 0.3));
    gizmos.line(centre, centre + up, Color::srgb(0.3, 1.0, 0.3));
    gizmos.line(centre, centre + fwd, Color::srgb(0.3, 0.5, 1.0));
}

fn cable_twist_pane(ui: &mut egui::Ui, st: &mut CableTwistState) {
    ui.label("Newton port — single CABLE, driven endpoint rotated.");
    ui.separator();
    readout(ui, "Particles", &format!("{}", TWIST_PARTICLES));
    readout(
        ui,
        "Stretch stiff.",
        &format!("{:.1e} N/m", TWIST_STIFFNESS),
    );
    readout(
        ui,
        "Bend stiff.",
        &format!("{:.1e} N·m", TWIST_BEND_STIFFNESS),
    );
    readout(ui, "Twist angle", &format!("{:+.2} rad", st.twist_angle));
    readout(ui, "Elbow Y", &format!("{:.3} m", st.elbow_y));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.twist_rate, -6.0..=6.0).text("twist rate (rad/s)"));
    let _ = ui.checkbox(&mut st.paused, "Pause twist");
    if ui.button("Reset twist").clicked() {
        st.user_reset = true;
    }
    ui.small("Driven endpoint traces a tiny circle around its anchor;");
    ui.small("the bending energy carries twist back across the elbow.");
}
