const CLOTH_ROLLERS_GRID_X: u32 = 26;
const CLOTH_ROLLERS_GRID_Z: u32 = 14;
const CLOTH_ROLLERS_SIZE_X: f64 = 1.45;
const CLOTH_ROLLERS_SIZE_Z: f64 = 0.72;
const CLOTH_ROLLERS_TOP_Y: f64 = 0.72;
const CLOTH_ROLLER_COUNT: usize = 3;
const CLOTH_ROLLER_RADIUS: f64 = 0.105;
const CLOTH_ROLLER_HALF: f64 = 0.48;
const CLOTH_ROLLER_Y: f64 = 0.36;
const CLOTH_ROLLER_SPEED: f64 = 4.0;
const CLOTH_ROLLER_XS: [f64; CLOTH_ROLLER_COUNT] = [-0.44, 0.0, 0.44];

fn cloth_roller_pose(i: usize, phase: f64) -> mp::Transform {
    let x = CLOTH_ROLLER_XS[i];
    mp::Transform::new(
        MVec3::new(x, CLOTH_ROLLER_Y, 0.0),
        MQuat::from_axis_angle(MVec3::X, std::f64::consts::FRAC_PI_2)
            * MQuat::from_axis_angle(MVec3::Z, phase),
    )
}

fn build_cloth_rollers_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.04);
    b.set_linear_damping(0.35);

    for i in 0..CLOTH_ROLLER_COUNT {
        let roller = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 1.0,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(1.0, CLOTH_ROLLER_HALF, CLOTH_ROLLER_RADIUS),
                })
                .with_pose(cloth_roller_pose(i, 0.0))
                .with_flags(mp::BodyFlags::KINEMATIC),
        );
        b.add_shape_cylinder(
            roller,
            CLOTH_ROLLER_HALF,
            CLOTH_ROLLER_RADIUS,
            mp::Transform::IDENTITY,
        )
        .expect("cloth roller cylinder");
    }

    let origin = MVec3::new(
        -CLOTH_ROLLERS_SIZE_X * 0.5,
        CLOTH_ROLLERS_TOP_Y,
        -CLOTH_ROLLERS_SIZE_Z * 0.5,
    );
    let _ = b.add_cloth_grid(
        origin,
        MVec3::new(CLOTH_ROLLERS_SIZE_X, 0.0, 0.0),
        MVec3::new(0.0, 0.0, CLOTH_ROLLERS_SIZE_Z),
        CLOTH_ROLLERS_GRID_X,
        CLOTH_ROLLERS_GRID_Z,
        0.025,
        0.018,
        4.0e-6,
        0.045,
        mp::WorldId(0),
    );
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("cloth rollers plane");

    (b.build().expect("cloth rollers model"), Vec::new())
}

struct ClothRollersState {
    roller_speed: f64,
    phase: f64,
    last_tangent_speed: f64,
    support_gap: f64,
}

impl Default for ClothRollersState {
    fn default() -> Self {
        Self {
            roller_speed: CLOTH_ROLLER_SPEED,
            phase: 0.0,
            last_tangent_speed: 0.0,
            support_gap: CLOTH_ROLLERS_TOP_Y - (CLOTH_ROLLER_Y + CLOTH_ROLLER_RADIUS),
        }
    }
}

fn cloth_roller_surface_velocity(omega: f64, radius: f64) -> MVec3 {
    // Rollers are oriented along +Z. At the top contact point (+Y from the
    // axis), tangential velocity is w × r = (0,0,ω) × (0,r,0).
    MVec3::new(-omega * radius, 0.0, 0.0)
}

fn cloth_rollers_support_gap(cloth_y: f64, roller_y: f64, roller_radius: f64) -> f64 {
    cloth_y - (roller_y + roller_radius)
}

fn update_cloth_rollers(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::ClothRollers {
        return;
    }
    let st = &mut warp.cloth_rollers;
    let dt = time.delta_secs() as f64;
    st.phase += st.roller_speed * dt;
    st.last_tangent_speed =
        cloth_roller_surface_velocity(st.roller_speed, CLOTH_ROLLER_RADIUS).length();
    st.support_gap =
        cloth_rollers_support_gap(CLOTH_ROLLERS_TOP_Y, CLOTH_ROLLER_Y, CLOTH_ROLLER_RADIUS);

    let Ok(body_q) = sim.state_a.body_q.host_mut() else {
        return;
    };
    for i in 0..CLOTH_ROLLER_COUNT.min(body_q.len()) {
        body_q[i] = cloth_roller_pose(i, st.phase * if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    let Ok(body_qd) = sim.state_a.body_qd.host_mut() else {
        return;
    };
    for i in 0..CLOTH_ROLLER_COUNT.min(body_qd.len()) {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        body_qd[i].angular = MVec3::Z * (st.roller_speed * sign);
        body_qd[i].linear = MVec3::ZERO;
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.body_q.ensure_device(&device);
        let _ = sim.state_a.body_q.upload(&queue);
        let _ = sim.state_a.body_qd.ensure_device(&device);
        let _ = sim.state_a.body_qd.upload(&queue);
    }
}

fn draw_cloth_rollers_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::ClothRollers {
        return;
    }
    let st = &warp.cloth_rollers;
    let col = Color::srgb(0.95, 0.60, 0.25);
    let axis_col = Color::srgb(0.25, 0.75, 1.0);
    for i in 0..CLOTH_ROLLER_COUNT {
        let c = Vec3::new(CLOTH_ROLLER_XS[i] as f32, CLOTH_ROLLER_Y as f32, 0.0);
        let half = CLOTH_ROLLER_HALF as f32;
        let r = CLOTH_ROLLER_RADIUS as f32;
        gizmos.line(c + Vec3::Z * -half, c + Vec3::Z * half, axis_col);
        for s in 0..40 {
            let t0 = s as f32 / 40.0 * std::f32::consts::TAU + st.phase as f32;
            let t1 = (s + 1) as f32 / 40.0 * std::f32::consts::TAU + st.phase as f32;
            let a = c + Vec3::new(r * t0.cos(), r * t0.sin(), -half);
            let b = c + Vec3::new(r * t1.cos(), r * t1.sin(), -half);
            let aa = c + Vec3::new(r * t0.cos(), r * t0.sin(), half);
            let bb = c + Vec3::new(r * t1.cos(), r * t1.sin(), half);
            gizmos.line(a, b, col);
            gizmos.line(aa, bb, col);
            if s % 10 == 0 {
                gizmos.line(a, aa, col);
            }
        }
        let tangent = cloth_roller_surface_velocity(
            st.roller_speed * if i % 2 == 0 { 1.0 } else { -1.0 },
            CLOTH_ROLLER_RADIUS,
        );
        let top = c + Vec3::Y * r;
        gizmos.line(
            top,
            top + Vec3::new(tangent.x as f32, 0.0, 0.0) * 0.22,
            Color::srgb(0.3, 1.0, 0.35),
        );
    }
}

fn cloth_rollers_pane(ui: &mut egui::Ui, st: &mut ClothRollersState) {
    ui.label("Cloth rollers — reduced Newton cloth/roller port");
    ui.separator();
    readout(ui, "Rollers", &format!("{}", CLOTH_ROLLER_COUNT));
    readout(
        ui,
        "Cloth grid",
        &format!("{}×{}", CLOTH_ROLLERS_GRID_X, CLOTH_ROLLERS_GRID_Z),
    );
    readout(
        ui,
        "Tangential |v|",
        &format!("{:.3} m/s", st.last_tangent_speed),
    );
    readout(
        ui,
        "Initial support gap",
        &format!("{:.3} m", st.support_gap),
    );
    ui.add(
        egui::Slider::new(&mut st.roller_speed, -8.0..=8.0)
            .text("roller ω")
            .step_by(0.05),
    );
    if ui.button("Reset roller phase").clicked() {
        st.phase = 0.0;
    }
    ui.small(
        "Forward cloth uses the VBD cloth lane; roller tangential velocity is explicit telemetry.",
    );
    ui.small(
        "Full frictional cloth/roller coupling remains partial until solver parity is stronger.",
    );
}
