fn build_vehicle_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let chassis = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, 0.786, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(
        chassis,
        MVec3::new(1.5, 0.35, 0.85),
        mp::Transform::IDENTITY,
    )
    .expect("vehicle chassis shape");
    visuals.push(BodyVisual::Box(Vec3::new(1.5, 0.35, 0.85)));
    for &(x, z) in &[(1.3, 0.75), (1.3, -0.75), (-1.3, 0.75), (-1.3, -0.75)] {
        let wheel = b.add_body(
            mp::BodyParams::new()
                .with_mass(0.0)
                .with_inertia(molla::math::Mat3::ZERO)
                .with_position(MVec3::new(x, 0.32, z))
                .with_flags(mp::BodyFlags::KINEMATIC),
        );
        b.add_shape_sphere(wheel, 0.32, mp::Transform::IDENTITY)
            .expect("vehicle wheel shape");
        visuals.push(BodyVisual::Sphere(0.32));
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("vehicle floor");
    (b.build().expect("vehicle model"), visuals)
}

// Per-frame state for the drivable [`Scene::Vehicle`] scene.
#[derive(Resource, Default)]
struct VehicleExampleState {
    vehicle: Option<molla_vehicle::Vehicle>,
    powertrain: molla_vehicle::Powertrain,
    was_active: bool,
}

// V6 — advance the `molla_vehicle::Vehicle` under arrow-key driver
// input and write the chassis + wheel poses into the host body buffer.
// `step_simulation` skips `Scene::Vehicle`, so `sync_transforms` renders
// straight from these host poses.
fn update_vehicle_example(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<VehicleExampleState>,
) {
    if sim.scene != Scene::Vehicle {
        state.was_active = false;
        return;
    }
    if sim.model.body_count < 5 {
        return;
    }
    if !state.was_active {
        // Fresh car on every (re)entry to the scene.
        state.vehicle = Some(molla_vehicle::Vehicle::car());
        state.powertrain = molla_vehicle::Powertrain::default();
        state.was_active = true;
    }
    let throttle = if keys.pressed(KeyCode::ArrowUp) {
        1.0
    } else {
        0.0
    };
    let brake = if keys.pressed(KeyCode::ArrowDown) {
        1.0
    } else {
        0.0
    };
    let steer = ((keys.pressed(KeyCode::ArrowLeft) as i8)
        - (keys.pressed(KeyCode::ArrowRight) as i8)) as f64
        * 0.45;
    let input = molla_vehicle::DriverInput {
        throttle,
        brake,
        steer,
    };
    let terrain = molla_vehicle::FlatTerrain::new(0.0, 1.0);
    // Sub-step at 1 ms for a stable vehicle integration.
    let frame = (time.delta_secs() as f64).clamp(0.0, 1.0 / 20.0);
    let steps = ((frame / 1.0e-3).round() as u32).clamp(1, 64);
    let powertrain = state.powertrain.clone();
    {
        let v = state.vehicle.as_mut().unwrap();
        for _ in 0..steps {
            v.apply_driver(&powertrain, &input);
            v.step(&terrain, 1.0e-3);
        }
    }
    let v = state.vehicle.as_ref().unwrap();
    let chassis_pos = MVec3::new(v.pos[0], v.pos[1], v.pos[2]);
    let chassis_rot = MQuat::from_rotation_y(v.yaw);
    let wheels: Vec<(MVec3, MQuat)> = (0..4)
        .map(|i| {
            let p = v.corner_world_pos(i);
            (
                MVec3::new(p[0], v.corners[i].wheel_radius, p[2]),
                MQuat::from_rotation_y(v.yaw + v.corners[i].steer),
            )
        })
        .collect();
    if let Ok(body_q) = sim.state_a.body_q.host_mut() {
        if body_q.len() >= 5 {
            body_q[0].position = chassis_pos;
            body_q[0].rotation = chassis_rot;
            for (i, w) in wheels.iter().enumerate() {
                body_q[1 + i].position = w.0;
                body_q[1 + i].rotation = w.1;
            }
        }
    }
}

