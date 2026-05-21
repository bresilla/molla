// ── Stage H4-H6 LiDAR v2 viewer wiring ─────────────────────────────

// A small scene of analytical primitives plus a TriMesh ground —
// chosen so the [`Scene::LidarV2`] scan exercises both the new H4
// `Lidar`/`ScanMetadata` path and the H1 mesh-BVH raycast.
fn build_lidar_v2_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let sphere_body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(-0.9, 0.5, 0.4))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(sphere_body, 0.4, mp::Transform::IDENTITY)
        .expect("lidar_v2 sphere");
    visuals.push(BodyVisual::Sphere(0.4));
    let box_body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.8, 0.4, -0.3))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(box_body, MVec3::splat(0.4), mp::Transform::IDENTITY)
        .expect("lidar_v2 box");
    visuals.push(BodyVisual::Box(Vec3::splat(0.4)));
    // TriMesh ground quad exercises the H1 wiring.
    let mesh = mp::TriMesh {
        vertices: vec![
            MVec3::new(-3.0, 0.0, -3.0),
            MVec3::new(3.0, 0.0, -3.0),
            MVec3::new(3.0, 0.0, 3.0),
            MVec3::new(-3.0, 0.0, 3.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    };
    let mid = b.add_mesh(mesh);
    b.add_shape_trimesh(mp::BodyId::NONE, mid, mp::Transform::IDENTITY)
        .expect("lidar_v2 ground");
    (b.build().expect("lidar_v2 model"), visuals)
}

// Per-frame state for the H4-H6 LiDAR v2 viewer scene.
#[derive(Resource, Default)]
struct LidarV2ExampleState {
    /// Built lazily on first entry.
    lidar: Option<Lidar>,
    /// Settled hit cloud: `(world_point, t)` per real return.
    hits: Vec<(Vec3, f32)>,
    /// Scanner origin in world (for the marker gizmo).
    origin: Vec3,
}

// Drives [`Scene::LidarV2`]: lazily builds the [`Lidar`], casts a
// scan against the procedural scene every frame, and stashes the
// returns for [`draw_lidar_v2_gizmos`].
fn update_lidar_v2_example(
    sim: Res<Sim>,
    mut state: ResMut<LidarV2ExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::LidarV2 {
        state.hits.clear();
        return;
    }
    if run.paused {
        return;
    }
    // 24 × 64 × 3 sub-rays ≈ 4600 rays / frame — modest, fast.
    let n_theta: u32 = 24;
    let n_phi: u32 = 64;
    let subray_count: u32 = 3;
    let budget = n_theta * n_phi * subray_count + 64;
    if state.lidar.is_none() {
        state.lidar = Some(Lidar::new(&sim.engine, budget).expect("LidarV2 init"));
    }
    let origin = MVec3::new(0.0, 1.7, -2.2);
    state.origin = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
    let mut meta = ScanMetadata::new(origin, n_theta, n_phi);
    // Downward hemisphere band, looking at the scene.
    meta.theta_range = (
        std::f64::consts::FRAC_PI_2 + 0.2,
        std::f64::consts::PI - 0.05,
    );
    meta.phi_range = (-0.7, 0.7);
    meta.subray_count = subray_count;
    meta.beam_divergence = 0.006;
    meta.max_distance = 25.0;
    let scan =
        match state
            .lidar
            .as_mut()
            .unwrap()
            .cast(&sim.engine, &sim.model, &sim.state_a, &meta)
        {
            Ok(s) => s,
            Err(e) => {
                warn!("lidar v2 cast failed: {e}");
                return;
            }
        };
    state.hits.clear();
    state.hits.reserve(scan.hits.len());
    for h in &scan.hits {
        if h.shape_id.is_some() {
            let p = Vec3::new(h.point.x as f32, h.point.y as f32, h.point.z as f32);
            state.hits.push((p, h.t as f32));
        }
    }
}

// Renders the [`Scene::LidarV2`] hit cloud as small spheres coloured
// by `t` (near = warm, far = cool) plus a scanner-origin marker.
fn draw_lidar_v2_gizmos(sim: Res<Sim>, state: Res<LidarV2ExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::LidarV2 {
        return;
    }
    for (p, t) in &state.hits {
        let frac = (t / 6.0).clamp(0.0, 1.0);
        let color = Color::srgb(0.30 + 0.65 * (1.0 - frac), 0.55, 0.40 + 0.55 * frac);
        gizmos.sphere(*p, 0.018, color);
    }
    gizmos.sphere(state.origin, 0.08, Color::WHITE);
}

// ── Stage H7 AerialLidar viewer wiring ─────────────────────────────


