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

#[derive(Resource, Default)]
struct SynthAnnotationExampleState {
    runtime: Option<AnnotationPreview>,
    revision: Option<u64>,
    error: Option<String>,
    boxes: Vec<BoundingBox2D>,
}

fn synth_annotation_config() -> TiledCameraConfig {
    TiledCameraConfig {
        width: 192,
        height: 192,
        fov_y_radians: 45f32.to_radians(),
        channels: CameraChannels::SHAPE_INDEX,
        ambient: MVec3::new(0.18, 0.18, 0.22),
        light_dir: MVec3::new(-0.4, -0.85, -0.35).normalize(),
        light_color: MVec3::new(1.0, 0.96, 0.88),
        max_distance: 30.0,
        near: 0.05,
        optics: Default::default(),
    }
}

fn update_synth_annotation_example(sim: Res<Sim>, mut state: ResMut<SynthAnnotationExampleState>) {
    if sim.scene != Scene::SyntheticAnnotation {
        *state = SynthAnnotationExampleState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = SynthAnnotationExampleState {
            revision: Some(sim.sensor_clock.revision),
            ..Default::default()
        };
    }
    if let Some(runtime) = &mut state.runtime {
        runtime.display.now = sim.sensor_clock.time();
    }
    if state.error.is_none()
        && let Err(error) = state.sample(&sim)
    {
        warn!("annotation preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

impl SynthAnnotationExampleState {
    fn sample(&mut self, sim: &Sim) -> molla::core::Result<()> {
        if self.runtime.is_none() {
            self.runtime = Some(AnnotationPreview::new(sim)?);
        }
        if let Some(frame) = self.runtime.as_mut().expect("runtime").update(sim)? {
            self.boxes.clear();
            self.boxes.extend(
                frame
                    .channel_iter::<molla::sensors::DeviceBoundingBox>(0)?
                    .filter_map(molla::sensors::DeviceBoundingBox::to_cpu),
            );
        }
        Ok(())
    }
}

fn synth_annotation_pane(ui: &mut egui::Ui, state: &SynthAnnotationExampleState) {
    readout(ui, "Pipeline", "GPU camera → instance boxes");
    readout(ui, "Resolution", "192 × 192");
    readout(
        ui,
        "Visible labelled instances",
        &state.boxes.len().to_string(),
    );
    sensor_preview_pane(
        ui,
        state.runtime.as_ref().map(|r| &r.display),
        state.error.as_deref(),
    );
    ui.small("Pixel bounds below; viewport wireframes mark detected bodies, not projected boxes.");
    for bbox in &state.boxes {
        ui.label(format!(
            "Shape {} · class {} · {} pixels",
            bbox.shape_id, bbox.class_id, bbox.pixel_count
        ));
        ui.small(format!(
            "Inclusive bounds: ({}, {})–({}, {})",
            bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
        ));
    }
}

// Approximate world-space markers for bodies with a completed pixel detection.
fn draw_synth_annotation_gizmos(
    sim: Res<Sim>,
    state: Res<SynthAnnotationExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::SyntheticAnnotation {
        return;
    }
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
        let Some(pose) = body_q.get(body_idx as usize) else {
            continue;
        };
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
