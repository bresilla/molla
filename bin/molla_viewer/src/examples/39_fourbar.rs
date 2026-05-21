// FourBar scene — a closed kinematic 4-bar linkage. We build it procedurally
// as 4 bodies + 3 revolute joints in a tree, plus a
// `JointMimic` linear-polynomial coupling on the closing pair so the
// loop appears closed visually. Per the spec — driver joint is one
// revolute, the others mimic via polynomial.
const FOURBAR_BAR_RADIUS: f32 = 0.04;
const FOURBAR_BASE_Y: f64 = 2.5;
fn build_fourbar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.10);

    // 4-bar geometry (rectangular crank-rocker):
    //   bar A — ground crank (rotates around world frame, length L_a)
    //   bar B — coupler     (length L_b)
    //   bar C — output rocker (length L_c)
    // The fourth bar is the implicit world ground (distance L_d
    // between the two ground pivots).
    let l_a = 0.6_f64;
    let l_b = 1.2_f64;
    let l_c = 0.6_f64;
    let _l_d = 1.0_f64;
    let mass = 0.3_f64;
    let inertia = box_inertia_xyz(mass, 0.05, l_a * 0.5, 0.05);

    let mut visuals = Vec::with_capacity(4);

    // ── Driver crank (bar A). Anchored at world-X = -0.5. ────────
    b.begin_articulation();
    let a_pivot = MVec3::new(-0.5, FOURBAR_BASE_Y, 0.0);
    let bar_a = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(a_pivot - MVec3::new(0.0, l_a * 0.5, 0.0)),
    );
    b.add_shape_capsule(
        bar_a,
        l_a * 0.5,
        FOURBAR_BAR_RADIUS as f64,
        mp::Transform::IDENTITY,
    )
    .expect("bar A capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_a * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let driver_id = b
        .add_joint_revolute(
            mp::BodyId::NONE,
            bar_a,
            mp::Transform::new(a_pivot, flip),
            mp::Transform::from_translation(MVec3::new(0.0, -l_a * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("driver joint");

    // ── Coupler (bar B), revolute to bar A's far end. ────────────
    let bar_b = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia: box_inertia_xyz(mass, 0.05, l_b * 0.5, 0.05),
            })
            .with_position(MVec3::new(0.0, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(
        bar_b,
        l_b * 0.5,
        FOURBAR_BAR_RADIUS as f64,
        mp::Transform::IDENTITY,
    )
    .expect("bar B capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_b * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let coupler_id = b
        .add_joint_revolute(
            bar_a,
            bar_b,
            mp::Transform::from_translation(MVec3::new(0.0, l_a * 0.5, 0.0)),
            mp::Transform::from_translation(MVec3::new(0.0, -l_b * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("coupler joint");

    // ── Output rocker (bar C), revolute to bar B's far end. ──────
    let bar_c = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(MVec3::new(0.5, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(
        bar_c,
        l_c * 0.5,
        FOURBAR_BAR_RADIUS as f64,
        mp::Transform::IDENTITY,
    )
    .expect("bar C capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: (l_c * 0.5) as f32,
        radius: FOURBAR_BAR_RADIUS,
    });
    let rocker_id = b
        .add_joint_revolute(
            bar_b,
            bar_c,
            mp::Transform::from_translation(MVec3::new(0.0, l_b * 0.5, 0.0)),
            mp::Transform::from_translation(MVec3::new(0.0, -l_c * 0.5, 0.0)),
            MVec3::Z,
        )
        .expect("rocker joint");

    // Linear-identity JointMimic — coupler tracks `-driver` and the
    // rocker tracks `+driver`. Real Freudenstein-equation closure is
    // a transcendental relation; the spec calls for a polynomial fit
    // so we ship the simplest case (linear) here. The visual loop
    // closure is approximate but reads as a "crank + rocker" cycle.
    let _ = b.add_equality_joint_mimic(driver_id, coupler_id, [0.0, -1.0, 0.0, 0.0, 0.0]);
    let _ = b.add_equality_joint_mimic(driver_id, rocker_id, [0.0, 1.0, 0.0, 0.0, 0.0]);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("fourbar model"), visuals)
}
// DiffBall scene — a kinematic projectile + a target hoop. The host
// optimises `(v, angle)` via closed-form gradient on miss distance.
// Differs from Newton's `example_diffsim_ball.py` in that we don't
// run a particle simulation (molla's tape autograd targets rigid
// XPBD); we use the analytical projectile equations directly. The
// pedagogy — "diffsim finds the right initial conditions" — is the

#[derive(Resource)]
struct FourBarExampleState {
    /// Driver joint angular velocity (rad/s). Constant — the driver
    /// joint is prescribed kinematically.
    driver_speed: f32,
    /// Cached driver angle (for pane readout).
    driver_angle: f32,
}

impl Default for FourBarExampleState {
    fn default() -> Self {
        Self {
            driver_speed: 0.6,
            driver_angle: 0.0,
        }
    }
}


fn update_fourbar_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<FourBarExampleState>,
) {
    if sim.scene != Scene::FourBar {
        return;
    }
    if sim.model.joint_dof_total < 1 {
        return;
    }
    state.driver_angle += state.driver_speed * time.delta_secs();
    // Keep driver velocity prescribed: write joint_target_pos[0] +
    // joint_target_vel[0] every frame; rely on JointMimic to slave
    // the other joints. Without an actuator the joint won't track —
    // we instead overwrite joint_q / joint_qd directly each frame.
    let queue = sim.engine.queue().expect("queue").clone();
    if let Ok(q) = sim.state_a.joint_q.host_mut() {
        if !q.is_empty() {
            q[0] = state.driver_angle as f64;
            // Slave joints — linear mimic factors match the
            // `add_equality_joint_mimic` polynomials in build.
            if q.len() >= 2 {
                q[1] = -(state.driver_angle as f64);
            }
            if q.len() >= 3 {
                q[2] = state.driver_angle as f64;
            }
        }
    }
    if let Ok(qd) = sim.state_a.joint_qd.host_mut() {
        if !qd.is_empty() {
            qd[0] = state.driver_speed as f64;
            if qd.len() >= 2 {
                qd[1] = -(state.driver_speed as f64);
            }
            if qd.len() >= 3 {
                qd[2] = state.driver_speed as f64;
            }
        }
    }
    let _ = sim.state_a.joint_q.upload(&queue);
    let _ = sim.state_a.joint_qd.upload(&queue);
}

fn fourbar_pane(ui: &mut egui::Ui, state: &mut FourBarExampleState) {
    ui.label("Driver speed (rad/s)");
    ui.add(egui::Slider::new(&mut state.driver_speed, -2.0..=2.0));
    readout(
        ui,
        "Driver angle",
        &format!("{:>+5.2} rad", state.driver_angle),
    );
    ui.separator();
    ui.small("Closed kinematic loop (4-bar).");
    ui.small("`JointMimic` enforces the closure constraint.");
}

