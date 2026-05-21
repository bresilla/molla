fn build_synth_annotation_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("synth floor");
    // Class 1 — box.
    let body_box = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(-1.6, 0.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(body_box, MVec3::splat(0.5), mp::Transform::IDENTITY)
        .expect("synth box");
    visuals.push(BodyVisual::Box(Vec3::splat(0.5)));
    // Class 2 — sphere.
    let body_sph = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.4, 0.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(body_sph, 0.5, mp::Transform::IDENTITY)
        .expect("synth sphere");
    visuals.push(BodyVisual::Sphere(0.5));
    // Class 3 — capsule.
    let body_cap = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(2.3, 0.55, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(body_cap, 0.3, 0.25, mp::Transform::IDENTITY)
        .expect("synth capsule");
    visuals.push(BodyVisual::Capsule {
        half_h: 0.3,
        radius: 0.25,
    });
    (b.build().expect("synth annotation model"), visuals)
}

// Per-frame state for [`Scene::SyntheticAnnotation`].
#[derive(Resource, Default)]
struct SynthAnnotationExampleState {
    camera: Option<TiledCamera>,
    dataset: Dataset,
    boxes: Vec<BoundingBox2D>,
    /// Pose-set sentinel so we only `set_camera_look_at` once.
    pose_set: bool,
}

// Render the labelled scene through a [`TiledCamera`], then convert
// the SHAPE_INDEX image to per-instance bounding boxes via the H9
// [`Dataset`] post-process. Bboxes are cached for the gizmo system.
fn update_synth_annotation_example(sim: Res<Sim>, mut state: ResMut<SynthAnnotationExampleState>) {
    if sim.scene != Scene::SyntheticAnnotation {
        state.boxes.clear();
        state.pose_set = false;
        return;
    }
    if state.camera.is_none() {
        let cfg = TiledCameraConfig {
            width: 192,
            height: 192,
            fov_y_radians: 45f32.to_radians(),
            channels: CameraChannels::ALL,
            ambient: MVec3::new(0.18, 0.18, 0.22),
            light_dir: MVec3::new(-0.4, -0.85, -0.35).normalize(),
            light_color: MVec3::new(1.0, 0.96, 0.88),
            max_distance: 30.0,
            near: 0.05,
            optics: Default::default(),
        };
        state.camera = Some(TiledCamera::new(&sim.engine, cfg, 1).expect("synth tc init"));
        // Labels match the build order: shape 0 is the plane (unlabelled),
        // 1/2/3 are box/sphere/capsule.
        state.dataset.label_shapes(&[1], 1);
        state.dataset.label_shapes(&[2], 2);
        state.dataset.label_shapes(&[3], 3);
    }
    // Pose set in its own scope so the camera borrow is released
    // before we touch `state.pose_set`.
    if !state.pose_set {
        let pose_ok = {
            let cam = state.camera.as_mut().unwrap();
            cam.set_camera_look_at(
                0,
                MVec3::new(0.0, 2.6, 5.0),
                MVec3::new(0.3, 0.5, 0.0),
                MVec3::Y,
            )
            .is_ok()
        };
        if pose_ok {
            state.pose_set = true;
        }
    }
    // Dispatch + download in another scope. We pull `cfg` out so the
    // host-side `annotate` call doesn't keep the camera borrowed.
    let (image, cfg) = {
        let cam = state.camera.as_mut().unwrap();
        if let Err(e) = cam.dispatch(&sim.engine, &sim.model, &sim.state_a) {
            warn!("synth annotation camera dispatch failed: {e}");
            return;
        }
        let cfg = *cam.config();
        let image: Vec<u32> = match cam.download_shape_index(&sim.engine) {
            Ok(s) => s.to_vec(),
            Err(e) => {
                warn!("synth annotation shape index download failed: {e}");
                return;
            }
        };
        (image, cfg)
    };
    let frame = state.dataset.annotate(
        0,
        &image[..(cfg.width * cfg.height) as usize],
        cfg.width,
        cfg.height,
    );
    state.boxes = frame.bboxes;
}

// Draw a wireframe 3-D bounding box around each labelled body so
// the detection is visible in the scene view. The per-class colour
// matches the dataset class id.
fn draw_synth_annotation_gizmos(
    sim: Res<Sim>,
    state: Res<SynthAnnotationExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::SyntheticAnnotation {
        return;
    }
    // Recover each labelled body's world AABB from the model's shape
    // metadata. The cube uses `Box.half_extents`, the sphere uses its
    // radius, the capsule uses (half_h + radius) along Y.
    let shape_body = match sim.model.shape_body.host() {
        Ok(s) => s,
        Err(_) => return,
    };
    let body_q = match sim.state_a.body_q.host() {
        Ok(s) => s,
        Err(_) => return,
    };
    for b in &state.boxes {
        let s_idx = b.shape_id as usize;
        let body_idx = shape_body.get(s_idx).copied().unwrap_or(-1);
        if body_idx < 0 {
            continue;
        }
        let pose = body_q[body_idx as usize];
        let centre = Vec3::new(
            pose.position.x as f32,
            pose.position.y as f32,
            pose.position.z as f32,
        );
        // Conservative 1×1×1-ish AABB — each body in this scene fits
        // inside ±0.55 m of its centre, so a single half-extent works
        // for all three classes.
        let half = Vec3::splat(0.55);
        let lo = centre - half;
        let hi = centre + half;
        let color = match b.class_id {
            1 => Color::srgb(0.95, 0.45, 0.35),
            2 => Color::srgb(0.35, 0.85, 0.55),
            3 => Color::srgb(0.45, 0.55, 0.95),
            _ => Color::srgb(0.7, 0.7, 0.7),
        };
        draw_aabb_edges(&mut gizmos, lo, hi, color);
    }
}

// ── V1-V2 Tire-model lab ───────────────────────────────────────────

// Trivial model — just the floor plane. The tire-model curves are

