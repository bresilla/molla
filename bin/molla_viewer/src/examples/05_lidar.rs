#[derive(Resource, Default)]
struct LidarState {
    runtime: Option<LidarPreview>,
    revision: Option<u64>,
    error: Option<String>,
    rotation_tick: u64,
    request_id: u64,
    requested: Option<(u64, LidarConfig)>,
    pending: std::collections::VecDeque<(u64, LidarConfig)>,
    displayed: Option<LidarConfig>,
    rays: Vec<LidarRay>,
}

#[derive(Clone, Copy)]
struct LidarRay {
    origin: Vec3,
    direction: Vec3,
    hit: Option<Vec3>,
}

fn build_lidar_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();

    let kin_at = |pos: MVec3| {
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
            .with_flags(mp::BodyFlags::KINEMATIC)
    };

    // Big sphere off to the right.
    {
        let body = b.add_body(kin_at(MVec3::new(4.5, 1.2, 0.0)));
        b.add_shape_sphere(body, 1.2, mp::Transform::IDENTITY)
            .expect("sphere");
        visuals.push(BodyVisual::Sphere(1.2));
    }
    // Tall box on the left.
    {
        let body = b.add_body(kin_at(MVec3::new(-3.5, 1.5, 2.0)));
        let h = MVec3::new(0.7, 1.5, 0.7);
        b.add_shape_box(body, h, mp::Transform::IDENTITY)
            .expect("box");
        visuals.push(BodyVisual::Box(Vec3::new(0.7, 1.5, 0.7)));
    }
    // Wide low box behind.
    {
        let body = b.add_body(kin_at(MVec3::new(0.0, 0.6, 6.0)));
        let h = MVec3::new(3.0, 0.6, 0.4);
        b.add_shape_box(body, h, mp::Transform::IDENTITY)
            .expect("wall");
        visuals.push(BodyVisual::Box(Vec3::new(3.0, 0.6, 0.4)));
    }
    // Capsule pillar on the right-back.
    {
        let body = b.add_body(kin_at(MVec3::new(3.0, 1.0, -3.5)));
        b.add_shape_capsule(body, 1.0, 0.4, mp::Transform::IDENTITY)
            .expect("capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: 1.0,
            radius: 0.4,
        });
    }
    // Cylinder on the left-back.
    {
        let body = b.add_body(kin_at(MVec3::new(-3.0, 0.8, -3.0)));
        b.add_shape_cylinder(body, 0.8, 0.5, mp::Transform::IDENTITY)
            .expect("cylinder");
        // BodyVisual doesn't have a cylinder variant — render as a
        // capsule of the same dimensions, close enough for the demo.
        visuals.push(BodyVisual::Capsule {
            half_h: 0.8,
            radius: 0.5,
        });
    }

    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("model"), visuals)
}

fn lidar_metadata(config: &LidarConfig) -> molla::core::Result<ScanMetadata> {
    let nh = config.ray_count_h.max(1);
    let nv = config.ray_count_v.max(1);
    let half_h = if nh == 1 {
        0.0
    } else {
        f64::from(config.fov_h) * 0.5
    };
    let half_v = if nv == 1 {
        0.0
    } else {
        f64::from(config.fov_v) * 0.5
    };
    let mut meta = ScanMetadata::new(config.origin.as_dvec3(), nv, nh);
    meta.forward = -MVec3::Z;
    meta.up = MVec3::Y;
    // Negative theta with forward -Z preserves ascending pitch and yaw row order.
    let theta = f64::from(config.pitch) - std::f64::consts::FRAC_PI_2;
    meta.theta_range = (theta - half_v, theta + half_v);
    meta.phi_range = (
        f64::from(config.yaw) - half_h,
        f64::from(config.yaw) + half_h,
    );
    meta.max_distance = f64::from(config.max_range);
    meta.validate()?;
    meta.checked_ray_count()?;
    Ok(meta)
}

fn update_lidar(sim: Res<Sim>, mut config: ResMut<LidarConfig>, mut state: ResMut<LidarState>) {
    if sim.scene != Scene::Lidar {
        *state = LidarState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = LidarState {
            revision: Some(sim.sensor_clock.revision),
            ..Default::default()
        };
    }
    let elapsed = sim.sensor_clock.tick.saturating_sub(state.rotation_tick) as f64 * FIXED_DT;
    state.rotation_tick = sim.sensor_clock.tick;
    if config.auto_rotate_speed.abs() > 1e-6 && elapsed > 0.0 {
        config.yaw = (f64::from(config.yaw) + f64::from(config.auto_rotate_speed) * elapsed)
            .rem_euclid(std::f64::consts::TAU) as f32;
    }
    if let Some(runtime) = &mut state.runtime {
        runtime.display.now = sim.sensor_clock.time();
    }
    if state.error.is_none()
        && let Err(error) = state.sample(&sim, *config)
    {
        warn!("LiDAR preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

impl LidarState {
    fn sample(&mut self, sim: &Sim, config: LidarConfig) -> molla::core::Result<()> {
        let meta = lidar_metadata(&config)?;
        let count = meta.checked_ray_count()?;
        if self.runtime.is_none() {
            self.runtime = Some(LidarPreview::ray_fan(sim, count)?);
        }
        let runtime = self.runtime.as_mut().expect("runtime");
        if runtime.grow(sim, count)? {
            self.pending.clear();
        }
        if let Some(frame) = runtime.display.poll_latest(sim.sensor_clock)? {
            let sampled = self
                .pending
                .iter()
                .find(|(id, _)| *id == frame.stamp.sample_id)
                .map(|(_, config)| *config)
                .ok_or_else(|| {
                    molla::core::Error::Storage("LiDAR preview metadata missing".into())
                })?;
            self.pending.retain(|(id, _)| *id > frame.stamp.sample_id);
            self.displayed = Some(sampled);
            self.rays.clear();
            for (((origin, shape), point), direction) in frame
                .channel_iter::<MVec3>(0)?
                .zip(frame.channel_iter::<i32>(1)?)
                .zip(frame.channel_iter::<MVec3>(2)?)
                .zip(frame.channel_iter::<MVec3>(3)?)
            {
                self.rays.push(LidarRay {
                    origin: origin.as_vec3(),
                    direction: direction.as_vec3(),
                    hit: (shape >= 0).then_some(point.as_vec3()),
                });
            }
        }
        let request = (sim.sensor_clock.tick, config);
        if self.requested != Some(request) {
            self.request_id = self.request_id.checked_add(1).ok_or_else(|| {
                molla::core::Error::Storage("LiDAR request identity exhausted".into())
            })?;
            self.requested = Some(request);
        }
        let scan = molla::sensors::LidarScan {
            sensor_id: 5,
            pattern: molla::sensors::LidarPattern::Terrestrial(meta),
        };
        if runtime.request(sim, scan, self.request_id)? {
            self.pending.push_back((self.request_id, config));
        }
        Ok(())
    }
}

fn draw_lidar_gizmos(
    sim: Res<Sim>,
    state: Res<LidarState>,
    config: Res<LidarConfig>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::Lidar {
        return;
    }

    let sampled = state.displayed.as_ref().unwrap_or(&config);
    gizmos.sphere(sampled.origin, 0.06, Color::srgb(1.0, 0.95, 0.2));

    let hit_color = Color::srgb(1.0, 0.25, 0.20);
    let miss_color = Color::srgba(0.30, 0.75, 0.85, 0.18);
    let cross_color = Color::srgb(1.0, 1.0, 1.0);
    let cross_half = 0.06_f32;

    for ray in state.rays.iter() {
        let end = match ray.hit {
            Some(p) => p,
            None => ray.origin + ray.direction * sampled.max_range,
        };
        let color = if ray.hit.is_some() {
            hit_color
        } else {
            miss_color
        };
        gizmos.line(ray.origin, end, color);
        if let Some(p) = ray.hit {
            gizmos.line(
                p - Vec3::X * cross_half,
                p + Vec3::X * cross_half,
                cross_color,
            );
            gizmos.line(
                p - Vec3::Y * cross_half,
                p + Vec3::Y * cross_half,
                cross_color,
            );
            gizmos.line(
                p - Vec3::Z * cross_half,
                p + Vec3::Z * cross_half,
                cross_color,
            );
        }
    }
}

fn lidar_pane(ui: &mut egui::Ui, config: &mut LidarConfig, state: &LidarState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU layout + shared scene");
    readout(ui, "Rays", &format!("{}", state.rays.len()));
    let n_hits = state.rays.iter().filter(|r| r.hit.is_some()).count();
    readout(ui, "Hits", &format!("{n_hits}"));
    sensor_preview_pane(
        ui,
        state.runtime.as_ref().map(|r| &r.display),
        state.error.as_deref(),
    );
    ui.small("Paused edits request a fresh scan; auto-spin follows simulation time.");
    ui.separator();
    ui.push_id("lidar_settings", |ui| lidar_settings(ui, config));
}

/// Editable scan settings. Kept in their own id scope: the preview above
/// changes its widget count while a scan rebuilds, which would otherwise
/// shift these sliders' automatic ids and hand an active drag to a neighbour.
fn lidar_settings(ui: &mut egui::Ui, config: &mut LidarConfig) {
    ui.label("Origin");
    ui.add(
        egui::DragValue::new(&mut config.origin.x)
            .speed(0.05)
            .prefix("x "),
    );
    ui.add(
        egui::DragValue::new(&mut config.origin.y)
            .speed(0.05)
            .range(0.05..=10.0)
            .prefix("y "),
    );
    ui.add(
        egui::DragValue::new(&mut config.origin.z)
            .speed(0.05)
            .prefix("z "),
    );
    ui.separator();

    ui.label("Heading");
    ui.add(egui::Slider::new(&mut config.yaw, 0.0..=std::f32::consts::TAU).text("yaw (rad)"));
    ui.add(
        egui::Slider::new(
            &mut config.pitch,
            -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
        )
        .text("pitch (rad)"),
    );
    ui.add(egui::Slider::new(&mut config.auto_rotate_speed, 0.0..=4.0).text("auto-spin (rad/s)"));
    ui.separator();

    ui.label("FOV");
    let mut fov_h_deg = config.fov_h.to_degrees();
    let mut fov_v_deg = config.fov_v.to_degrees();
    if ui
        .add(egui::Slider::new(&mut fov_h_deg, 5.0..=360.0).text("horizontal °"))
        .changed()
    {
        config.fov_h = fov_h_deg.to_radians();
    }
    if ui
        .add(egui::Slider::new(&mut fov_v_deg, 1.0..=120.0).text("vertical °"))
        .changed()
    {
        config.fov_v = fov_v_deg.to_radians();
    }
    ui.separator();

    ui.label("Resolution");
    let mut nh = config.ray_count_h as i32;
    if ui
        .add(egui::Slider::new(&mut nh, 8..=512).text("rays h"))
        .changed()
    {
        config.ray_count_h = nh.max(1) as u32;
    }
    let mut nv = config.ray_count_v as i32;
    if ui
        .add(egui::Slider::new(&mut nv, 1..=64).text("rays v"))
        .changed()
    {
        config.ray_count_v = nv.max(1) as u32;
    }
    ui.separator();
    ui.add(egui::Slider::new(&mut config.max_range, 1.0..=100.0).text("range (m)"));
}
