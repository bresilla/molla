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
    runtime: Option<LidarPreview>,
    revision: Option<u64>,
    error: Option<String>,
    hits: Vec<Vec3>,
    origins: Vec<Vec3>,
}

// Submits once per scene/reset and polls the hit cloud and pulse origins.
fn update_aerial_lidar_example(sim: Res<Sim>, mut state: ResMut<AerialLidarExampleState>) {
    if sim.scene != Scene::AerialLidarPass {
        *state = AerialLidarExampleState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = AerialLidarExampleState {
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
        warn!("aerial LiDAR preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

fn aerial_lidar_metadata() -> AerialScanMetadata {
    AerialScanMetadata::new(
        MVec3::new(-4.0, 8.0, -4.0),
        MVec3::new(4.0, 8.0, 4.0),
        2.0, // ≈128 pulses over 64 m² (rounded)
    )
}

impl AerialLidarExampleState {
    fn sample(&mut self, sim: &Sim) -> molla::core::Result<()> {
        let meta = aerial_lidar_metadata();
        meta.validate()?;
        if self.runtime.is_none() {
            self.runtime = Some(LidarPreview::new(
                sim,
                meta.checked_ray_count()?,
                true,
                true,
            )?);
        }
        let scan = molla::sensors::LidarScan {
            sensor_id: 106,
            pattern: molla::sensors::LidarPattern::Aerial(meta),
        };
        if let Some(frame) = self.runtime.as_mut().expect("runtime").update(sim, scan)? {
            self.origins.clear();
            self.hits.clear();
            self.origins.extend(
                frame
                    .channel_iter::<MVec3>(0)?
                    .step_by(meta.effective_subray_count() as usize)
                    .map(|p| p.as_vec3()),
            );
            for (shape, point) in frame
                .channel_iter::<i32>(1)?
                .zip(frame.channel_iter::<MVec3>(2)?)
            {
                if shape >= 0 {
                    self.hits.push(point.as_vec3());
                }
            }
        }
        Ok(())
    }
}

fn aerial_lidar_pane(ui: &mut egui::Ui, state: &AerialLidarExampleState) {
    let meta = aerial_lidar_metadata();
    readout(ui, "Coverage / density", "8 × 8 m / 2 pulses per m²");
    readout(ui, "Rays", &meta.ray_count().to_string());
    readout(ui, "Displayed hits", &state.hits.len().to_string());
    sensor_preview_pane(
        ui,
        state.runtime.as_ref().map(|r| &r.display),
        state.error.as_deref(),
    );
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
