// ── Pendulum scene ─────────────────────────────────────────────────
//
// Two articulations side-by-side:
//   * A single 1-DoF revolute pendulum (rod + weight) at -X.
//   * A 3-link chain pendulum at +X.
// Featherstone-stepped — same dynamics path as `Scene::Arm`.

const PENDULUM_BASE_Y: f64 = 4.0;
const PENDULUM_LINK_LEN: f64 = 0.8;
const PENDULUM_LINK_RADIUS: f64 = 0.06;
const PENDULUM_LINK_MASS: f64 = 0.6;

fn build_pendulum_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let half_h = PENDULUM_LINK_LEN * 0.5;
    let radius = PENDULUM_LINK_RADIUS;
    let inertia = capsule_inertia(PENDULUM_LINK_MASS, half_h, radius);

    let mut visuals = Vec::new();

    // ── Single 1-DoF pendulum at -X. ───────────────────────────────
    b.begin_articulation();
    let single_anchor = MVec3::new(-1.5, PENDULUM_BASE_Y, 0.0);
    let single = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: PENDULUM_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(single_anchor - MVec3::new(0.0, half_h, 0.0)),
    );
    b.add_shape_capsule(single, half_h, radius, mp::Transform::IDENTITY)
        .expect("single pendulum shape");
    visuals.push(BodyVisual::Capsule {
        half_h: half_h as f32,
        radius: radius as f32,
    });
    // Hang straight down at q=0: rotate joint frame 180° around X so
    // joint-local +Y → world -Y. Matches the `Arm` scene's trick.
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let x_p_single = mp::Transform::new(single_anchor, flip);
    let x_c_single = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(mp::BodyId::NONE, single, x_p_single, x_c_single, MVec3::Z)
        .expect("single pendulum joint");

    // ── 3-link chain pendulum at +X. ───────────────────────────────
    b.begin_articulation();
    let chain_anchor = MVec3::new(1.5, PENDULUM_BASE_Y, 0.0);
    let mut prev = mp::BodyId::NONE;
    let mut next_pos = chain_anchor - MVec3::new(0.0, half_h, 0.0);
    for i in 0..3 {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: PENDULUM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("chain link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = if i == 0 {
            mp::Transform::new(chain_anchor, flip)
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0))
        };
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_revolute(prev, body, x_p, x_c, MVec3::Z)
            .expect("chain joint");
        prev = body;
        next_pos = next_pos - MVec3::new(0.0, PENDULUM_LINK_LEN, 0.0);
    }

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor plane");

    (b.build().expect("pendulum model"), visuals)
}

// Read joint angles + angular velocities for the pendulum readout.
#[derive(Resource, Default)]
struct PendulumExampleState {
    /// Single-pendulum angle (rad) and angular velocity (rad/s).
    single_q: f32,
    single_qd: f32,
    /// 3-link chain joint angles and rates.
    chain_q: [f32; 3],
    chain_qd: [f32; 3],
}

fn update_pendulum_example(sim: Res<Sim>, mut state: ResMut<PendulumExampleState>) {
    if sim.scene != Scene::Pendulum {
        return;
    }
    let q = match sim.state_a.joint_q.host() {
        Ok(q) => q,
        Err(_) => return,
    };
    let qd = match sim.state_a.joint_qd.host() {
        Ok(q) => q,
        Err(_) => return,
    };
    if q.len() >= 4 && qd.len() >= 4 {
        state.single_q = q[0] as f32;
        state.single_qd = qd[0] as f32;
        state.chain_q = [q[1] as f32, q[2] as f32, q[3] as f32];
        state.chain_qd = [qd[1] as f32, qd[2] as f32, qd[3] as f32];
    }
}

fn pendulum_pane(ui: &mut egui::Ui, state: &PendulumExampleState) {
    ui.label("Two pendulums in one scene:");
    ui.separator();
    ui.label("Single (-X)");
    readout(ui, "  · angle", &format!("{:>+5.2} rad", state.single_q));
    readout(ui, "  · ω", &format!("{:>+5.2} rad/s", state.single_qd));
    ui.separator();
    ui.label("3-link chain (+X)");
    for i in 0..3 {
        readout(
            ui,
            &format!("  · q{} | ω{}", i, i),
            &format!(
                "{:>+5.2} rad | {:>+5.2} rad/s",
                state.chain_q[i], state.chain_qd[i]
            ),
        );
    }
    ui.separator();
    ui.small("Featherstone GPU solver. Drag to swing.");
    ui.small("Reset (R) to restart from straight-down rest.");
}

