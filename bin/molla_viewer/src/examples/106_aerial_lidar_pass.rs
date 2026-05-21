fn build_aerial_lidar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(1.2, 1.2, 1.2))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(body, MVec3::splat(0.8), mp::Transform::IDENTITY)
        .expect("aerial obstacle");
    visuals.push(BodyVisual::Box(Vec3::splat(0.8)));
    let mesh = mp::TriMesh {
        vertices: vec![
            MVec3::new(-8.0, 0.0, -8.0),
            MVec3::new(8.0, 0.0, -8.0),
            MVec3::new(8.0, 0.0, 8.0),
            MVec3::new(-8.0, 0.0, 8.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    };
    let mid = b.add_mesh(mesh);
    b.add_shape_trimesh(mp::BodyId::NONE, mid, mp::Transform::IDENTITY)
        .expect("aerial ground");
    (b.build().expect("aerial lidar model"), visuals)
}

// Per-frame state for the H7 aerial-LiDAR scene.
#[derive(Resource, Default)]
struct AerialLidarExampleState {
    aerial: Option<AerialLidar>,
    hits: Vec<Vec3>,
    origins: Vec<Vec3>,
    was_active: bool,
}

// Drives [`Scene::AerialLidarPass`]: runs the aerial pass once on
// entry and caches the hit cloud + pulse origins for the gizmo
// system; re-entering the scene rebuilds.
fn update_aerial_lidar_example(sim: Res<Sim>, mut state: ResMut<AerialLidarExampleState>) {
    if sim.scene != Scene::AerialLidarPass {
        state.was_active = false;
        state.hits.clear();
        state.origins.clear();
        return;
    }
    if state.was_active {
        return;
    }
    let meta = AerialScanMetadata::new(
        MVec3::new(-4.0, 8.0, -4.0),
        MVec3::new(4.0, 8.0, 4.0),
        2.0, // ≈128 pulses over 64 m² (rounded)
    );
    let budget = (meta.ray_count() + 64).max(64);
    if state.aerial.is_none() {
        state.aerial = Some(AerialLidar::new(&sim.engine, budget).expect("AerialLidar init"));
    }
    let scan =
        match state
            .aerial
            .as_mut()
            .unwrap()
            .cast(&sim.engine, &sim.model, &sim.state_a, &meta)
        {
            Ok(s) => s,
            Err(e) => {
                warn!("aerial lidar cast failed: {e}");
                return;
            }
        };
    state.origins.clear();
    state.hits.clear();
    for o in &scan.origins {
        state
            .origins
            .push(Vec3::new(o.x as f32, o.y as f32, o.z as f32));
    }
    for h in &scan.hits {
        if h.shape_id.is_some() {
            state.hits.push(Vec3::new(
                h.point.x as f32,
                h.point.y as f32,
                h.point.z as f32,
            ));
        }
    }
    state.was_active = true;
}

// Pulse origins drawn as dim dots at altitude; hits drawn as warm
// spheres on the ground.
fn draw_aerial_lidar_gizmos(
    sim: Res<Sim>,
    state: Res<AerialLidarExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::AerialLidarPass {
        return;
    }
    let orig_color = Color::srgb(0.65, 0.65, 0.75);
    for o in &state.origins {
        gizmos.sphere(*o, 0.05, orig_color);
    }
    let hit_color = Color::srgb(0.95, 0.62, 0.18);
    for p in &state.hits {
        gizmos.sphere(*p, 0.05, hit_color);
    }
}

// ── Stage H13 SolarSky viewer wiring ───────────────────────────────

// A few labelled blocks on a flat ground — a tiny outdoor scene

