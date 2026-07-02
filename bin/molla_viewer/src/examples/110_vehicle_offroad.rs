// ── V3-V5 Vehicle off-road (heightfield) ───────────────────────────

// Per-frame state for [`Scene::VehicleOffroad`]. Carries the vehicle
// + the heightfield it drives on. The vehicle is rebuilt on every
// scene entry (same pattern as [`VehicleExampleState`]).
#[derive(Resource)]
#[derive(Default)]
struct VehicleOffroadExampleState {
    vehicle: Option<molla_vehicle::Vehicle>,
    powertrain: molla_vehicle::Powertrain,
    terrain: Option<molla_vehicle::HeightfieldTerrain>,
    /// Pre-built `(p0, p1)` heightfield wireframe segments for the
    /// gizmo system — recomputed once when the terrain is built.
    grid_segments: Vec<(Vec3, Vec3)>,
    was_active: bool,
}


fn build_offroad_heightfield() -> molla_vehicle::HeightfieldTerrain {
    // 60×60 m terrain, 1.5 m spacing → 41×41 grid. Smooth-ish
    // rolling hills via summed sinusoids; gentle enough that a
    // street car can drive over them without flipping.
    let nx = 41usize;
    let nz = 41usize;
    let spacing = 1.5_f64;
    let origin = [-(nx as f64) * spacing * 0.5, -(nz as f64) * spacing * 0.5];
    let mut heights = Vec::with_capacity(nx * nz);
    for iz in 0..nz {
        let z = origin[1] + iz as f64 * spacing;
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            let h = 0.55 * (0.18 * x).sin() * (0.18 * z).cos()
                + 0.30 * (0.42 * x + 0.7).sin()
                + 0.20 * (0.61 * z - 0.4).cos();
            heights.push(h);
        }
    }
    molla_vehicle::HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 1.0)
}

// Cache the wireframe-grid line segments so the gizmo system isn't
// re-running the heightfield evaluator every frame.
fn build_offroad_grid_segments(terrain: &molla_vehicle::HeightfieldTerrain) -> Vec<(Vec3, Vec3)> {
    use molla_vehicle::Terrain;
    let mut out = Vec::new();
    // Skip every other line for clarity at this density.
    let stride = 2usize;
    for iz in (0..terrain.nz).step_by(stride) {
        let z = terrain.origin[1] + iz as f64 * terrain.spacing;
        for ix in 0..(terrain.nx - 1) {
            let x0 = terrain.origin[0] + ix as f64 * terrain.spacing;
            let x1 = x0 + terrain.spacing;
            let h0 = terrain.height(x0, z);
            let h1 = terrain.height(x1, z);
            out.push((
                Vec3::new(x0 as f32, h0 as f32, z as f32),
                Vec3::new(x1 as f32, h1 as f32, z as f32),
            ));
        }
    }
    for ix in (0..terrain.nx).step_by(stride) {
        let x = terrain.origin[0] + ix as f64 * terrain.spacing;
        for iz in 0..(terrain.nz - 1) {
            let z0 = terrain.origin[1] + iz as f64 * terrain.spacing;
            let z1 = z0 + terrain.spacing;
            let h0 = terrain.height(x, z0);
            let h1 = terrain.height(x, z1);
            out.push((
                Vec3::new(x as f32, h0 as f32, z0 as f32),
                Vec3::new(x as f32, h1 as f32, z1 as f32),
            ));
        }
    }
    out
}

// Drive the `Vehicle` over the heightfield. Mirrors the
// `update_vehicle_example` pattern; the only structural difference
// is the `HeightfieldTerrain` swap and the terrain build on first
// entry.
fn update_vehicle_offroad_example(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<VehicleOffroadExampleState>,
) {
    if sim.scene != Scene::VehicleOffroad {
        state.was_active = false;
        return;
    }
    if sim.model.body_count < 5 {
        return;
    }
    if !state.was_active {
        let terrain = build_offroad_heightfield();
        state.grid_segments = build_offroad_grid_segments(&terrain);
        state.terrain = Some(terrain);
        // Fresh car positioned over the centre of the terrain (the
        // ground at (0, 0) is ≈ 0.55 · sin(0) · cos(0) + … so the
        // starting body_y is set generously in Vehicle::car()).
        state.vehicle = Some(molla_vehicle::Vehicle::car_with(Box::new(
            molla_vehicle::RealBrushTireModel::offroad_soft().with_nominal_speed(8.0),
        )));
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
    let frame = (time.delta_secs() as f64).clamp(0.0, 1.0 / 20.0);
    let steps = ((frame / 1.0e-3).round() as u32).clamp(1, 64);
    // Clone across the mutable-vehicle borrow, then write back so the
    // auto-shifter's selected gear persists frame to frame.
    let mut powertrain = state.powertrain.clone();
    // Take ownership of the terrain across the mutable-vehicle borrow,
    // then put it back. Cheap: `HeightfieldTerrain` is `Vec`-of-`f64`
    // and the swap doesn't touch the heap.
    let terrain = state.terrain.take().expect("offroad terrain");
    {
        let v = state.vehicle.as_mut().unwrap();
        for _ in 0..steps {
            v.apply_driver(&mut powertrain, &input);
            v.step(&terrain, 1.0e-3);
        }
    }
    state.terrain = Some(terrain);
    state.powertrain = powertrain;
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
    if let Ok(body_q) = sim.state_a.body_q.host_mut()
        && body_q.len() >= 5 {
            body_q[0].position = chassis_pos;
            body_q[0].rotation = chassis_rot;
            for (i, w) in wheels.iter().enumerate() {
                body_q[1 + i].position = w.0;
                body_q[1 + i].rotation = w.1;
            }
        }
}

// Draw the heightfield grid + (when the wheels are in contact) a
// small marker at each wheel's terrain contact point.
fn draw_vehicle_offroad_gizmos(
    sim: Res<Sim>,
    state: Res<VehicleOffroadExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::VehicleOffroad {
        return;
    }
    let grid_color = Color::srgb(0.40, 0.45, 0.55);
    for (a, b) in &state.grid_segments {
        gizmos.line(*a, *b, grid_color);
    }
    if let (Some(v), Some(t)) = (state.vehicle.as_ref(), state.terrain.as_ref()) {
        use molla_vehicle::Terrain;
        let contact_color = Color::srgb(0.95, 0.65, 0.20);
        for i in 0..4 {
            let p = v.corner_world_pos(i);
            let gh = t.height(p[0], p[2]);
            gizmos.sphere(
                Vec3::new(p[0] as f32, gh as f32, p[2] as f32),
                0.07,
                contact_color,
            );
        }
    }
}

