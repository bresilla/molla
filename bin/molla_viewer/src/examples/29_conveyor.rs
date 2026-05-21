const CONVEYOR_BELT_LEN: f64 = 6.0;
const CONVEYOR_BELT_HALF_WIDTH: f64 = 0.6;
const CONVEYOR_BELT_HALF_THICK: f64 = 0.05;
const CONVEYOR_BELT_Y: f64 = 1.0;
fn build_conveyor_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals = Vec::new();

    // Belt — body 0, kinematic, translated each frame in
    // `update_conveyor_example`.
    let belt_half = MVec3::new(
        CONVEYOR_BELT_LEN * 0.5,
        CONVEYOR_BELT_HALF_THICK,
        CONVEYOR_BELT_HALF_WIDTH,
    );
    let belt = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, CONVEYOR_BELT_Y, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(belt, belt_half, mp::Transform::IDENTITY)
        .expect("belt shape");
    visuals.push(BodyVisual::Box(Vec3::new(
        belt_half.x as f32,
        belt_half.y as f32,
        belt_half.z as f32,
    )));

    // Side rails — static, just keep boxes from rolling off the belt.
    for &z_off in &[
        -CONVEYOR_BELT_HALF_WIDTH - 0.1,
        CONVEYOR_BELT_HALF_WIDTH + 0.1,
    ] {
        b.add_shape_box(
            mp::BodyId::NONE,
            MVec3::new(CONVEYOR_BELT_LEN * 0.5, 0.15, 0.05),
            mp::Transform::from_translation(MVec3::new(0.0, CONVEYOR_BELT_Y + 0.15, z_off)),
        )
        .expect("rail shape");
    }

    // Dropped boxes — 8 of them, scattered along the belt.
    for i in 0..8 {
        let x = -CONVEYOR_BELT_LEN * 0.5 + (i as f64 + 0.5) * (CONVEYOR_BELT_LEN / 8.0);
        let half = MVec3::splat(0.15);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 0.5,
                    com: MVec3::ZERO,
                    inertia: box_inertia(0.5, 0.15),
                })
                .with_position(MVec3::new(
                    x,
                    CONVEYOR_BELT_Y + CONVEYOR_BELT_HALF_THICK + 0.20 + (i as f64) * 0.05,
                    0.0,
                )),
        );
        b.add_shape_box(body, half, mp::Transform::IDENTITY)
            .expect("box shape");
        visuals.push(BodyVisual::Box(Vec3::splat(0.15)));
    }

    // Floor.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");

    (b.build().expect("conveyor model"), visuals)
}

// V6 — drivable wheeled-vehicle scene. Five kinematic bodies (a chassis
// box + four wheel spheres); their poses are written every frame by

#[derive(Resource)]
struct ConveyorExampleState {
    /// Belt linear speed (m/s) at the top surface.
    belt_speed: f32,
}

impl Default for ConveyorExampleState {
    fn default() -> Self {
        Self { belt_speed: 0.6 }
    }
}


fn update_conveyor_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    state: ResMut<ConveyorExampleState>,
) {
    if sim.scene != Scene::Conveyor {
        return;
    }
    if sim.model.body_count == 0 {
        return;
    }
    // Belt is body 0. Translate its kinematic pose along +X by
    // belt_speed each viewer frame. Wrap when the belt has shifted
    // by half the belt length so it never disappears off-screen.
    let dt = time.delta_secs() as f64;
    let v = state.belt_speed as f64;
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut pose = body_q[0];
    pose.position.x += v * dt;
    if pose.position.x > CONVEYOR_BELT_LEN * 0.5 {
        pose.position.x -= CONVEYOR_BELT_LEN;
    } else if pose.position.x < -CONVEYOR_BELT_LEN * 0.5 {
        pose.position.x += CONVEYOR_BELT_LEN;
    }
    body_q[0] = pose;
    // Push belt linear velocity into body_qd[0] so contact friction
    // gets the right tangential surface velocity.
    let body_qd = match sim.state_a.body_qd.host_mut() {
        Ok(qd) => qd,
        Err(_) => return,
    };
    body_qd[0].linear = MVec3::new(v, 0.0, 0.0);

    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.body_q.ensure_device(&device);
        let _ = sim.state_a.body_q.upload(&queue);
        let _ = sim.state_a.body_qd.ensure_device(&device);
        let _ = sim.state_a.body_qd.upload(&queue);
    }
}

fn conveyor_pane(ui: &mut egui::Ui, state: &mut ConveyorExampleState) {
    ui.label("Conveyor belt");
    ui.label("Belt speed (m/s)");
    ui.add(egui::Slider::new(&mut state.belt_speed, -2.0..=2.0));
    ui.separator();
    ui.small("Boxes are dropped onto a long kinematic box.");
    ui.small("Belt pose is advected each frame; boxes settle on it.");
    ui.colored_label(
        egui::Color32::YELLOW,
        "Caveat: XPBD's Coulomb friction is gated by per-substep \
         normal impulse |λ_n|, which is ~0 at steady state. The \
         box receives only a tiny tangential push from the belt — \
         the belt itself slides away beneath. Visualises belt \
         motion + box settling, not steady-state belt-drag.",
    );
}

