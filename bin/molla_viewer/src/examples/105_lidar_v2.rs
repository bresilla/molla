// LiDAR scan-grid scene with analytical primitives and a triangle ground.
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
    runtime: Option<LidarPreview>,
    revision: Option<u64>,
    error: Option<String>,
    /// Settled hit cloud: `(world_point, t)` per real return.
    hits: Vec<(Vec3, f32)>,
    /// Scanner origin in world (for the marker gizmo).
    origin: Vec3,
}

// Submits a scan per new simulation frame and polls displayed hits.
fn update_lidar_v2_example(sim: Res<Sim>, mut state: ResMut<LidarV2ExampleState>) {
    if sim.scene != Scene::LidarV2 {
        *state = LidarV2ExampleState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = LidarV2ExampleState {
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
        warn!("LiDAR v2 preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

fn lidar_v2_metadata() -> ScanMetadata {
    let origin = MVec3::new(0.0, 1.7, -2.2);
    let mut meta = ScanMetadata::new(origin, 24, 64);
    // Downward hemisphere band, looking at the scene.
    meta.theta_range = (
        std::f64::consts::FRAC_PI_2 + 0.2,
        std::f64::consts::PI - 0.05,
    );
    meta.phi_range = (-0.7, 0.7);
    meta.subray_count = 3;
    meta.beam_divergence = 0.006;
    meta.max_distance = 25.0;
    meta
}

impl LidarV2ExampleState {
    fn sample(&mut self, sim: &Sim) -> molla::core::Result<()> {
        let meta = lidar_v2_metadata();
        meta.validate()?;
        self.origin = meta.origin.as_vec3();
        if self.runtime.is_none() {
            self.runtime = Some(LidarPreview::new(
                sim,
                meta.checked_ray_count()?,
                false,
                false,
            )?);
        }
        let scan = molla::sensors::LidarScan {
            sensor_id: 105,
            pattern: molla::sensors::LidarPattern::Terrestrial(meta),
        };
        if let Some(frame) = self.runtime.as_mut().expect("runtime").update(sim, scan)? {
            self.hits.clear();
            for ((t, shape), point) in frame
                .channel_iter::<f64>(0)?
                .zip(frame.channel_iter::<i32>(1)?)
                .zip(frame.channel_iter::<MVec3>(2)?)
            {
                if shape >= 0 {
                    self.hits.push((point.as_vec3(), t as f32));
                }
            }
        }
        Ok(())
    }
}

fn lidar_v2_pane(ui: &mut egui::Ui, state: &LidarV2ExampleState) {
    readout(ui, "Pattern", "24 × 64 pulses; 3 sub-rays/pulse");
    readout(ui, "Rays / range", "4608 / 25 m");
    readout(ui, "Beam half-angle", "0.006 rad");
    readout(ui, "Displayed hits", &state.hits.len().to_string());
    sensor_preview_pane(
        ui,
        state.runtime.as_ref().map(|r| &r.display),
        state.error.as_deref(),
    );
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
