#[derive(Resource, Default)]
struct SensorLabState {
    settings: sensor_lab::LabSettings,
    runtime: Option<sensor_lab::LabRuntime>,
    revision: Option<u64>,
    motion_tick: Option<u64>,
    error: Option<String>,
    image: Option<Handle<Image>>,
    texture: Option<egui::TextureId>,
    image_valid: bool,
}

fn build_sensor_lab_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_position(MVec3::new(0.0, 1.0, 1.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    let vertices = [
        [-0.7, -0.8, -0.35],
        [0.7, -0.8, -0.35],
        [0.7, 0.8, -0.35],
        [-0.7, 0.8, -0.35],
        [-0.7, -0.8, 0.35],
        [0.7, -0.8, 0.35],
        [0.7, 0.8, 0.35],
        [-0.7, 0.8, 0.35],
    ]
    .map(MVec3::from_array)
    .to_vec();
    let mesh = b.add_mesh(mp::TriMesh {
        vertices,
        indices: vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
    });
    let moving = b
        .add_shape_trimesh(body, mesh, mp::Transform::IDENTITY)
        .expect("sensor lab mesh");
    b.set_shape_visual_material(moving, MVec3::new(0.9, 0.25, 0.12), 0.0, 0.5)
        .expect("mesh material");
    let sphere = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_position(MVec3::new(0.0, 1.0, -1.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    let sphere = b
        .add_shape_sphere(sphere, 0.7, mp::Transform::IDENTITY)
        .expect("sensor lab sphere");
    b.set_shape_visual_material(sphere, MVec3::new(0.15, 0.8, 0.3), 0.0, 0.5)
        .expect("sphere material");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("sensor lab floor");
    (
        b.build().expect("sensor lab model"),
        vec![
            BodyVisual::Box(Vec3::new(0.7, 0.8, 0.35)),
            BodyVisual::Sphere(0.7),
        ],
    )
}

fn sensor_lab_pose(time: f64) -> mp::Transform {
    mp::Transform::new(
        MVec3::new(
            1.6 * (0.8 * time).sin(),
            1.0,
            1.0 + 0.2 * (0.3 * time).sin(),
        ),
        MQuat::from_rotation_y(0.6 * time),
    )
}

fn sensor_lab_velocity(time: f64) -> MVec3 {
    MVec3::new(1.28 * (0.8 * time).cos(), 0.0, 0.06 * (0.3 * time).cos())
}

fn update_sensor_lab_motion(mut sim: ResMut<Sim>, mut state: ResMut<SensorLabState>) {
    if sim.scene != Scene::SensorLab {
        state.runtime = None;
        state.revision = None;
        state.motion_tick = None;
        state.error = None;
        state.image_valid = false;
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        state.runtime = None;
        state.revision = Some(sim.sensor_clock.revision);
        state.motion_tick = None;
        state.error = None;
        state.image_valid = false;
    }
    if state.error.is_some() || state.motion_tick == Some(sim.sensor_clock.tick) {
        return;
    }
    let result = (|| -> molla::core::Result<()> {
        let time = sim.sensor_clock.time();
        sim.state_a.body_q.host_mut()?[0] = sensor_lab_pose(time);
        let velocity = &mut sim.state_a.body_qd.host_mut()?[0];
        velocity.linear = sensor_lab_velocity(time);
        velocity.angular = MVec3::Y * 0.6;
        let queue = sim.engine.queue().expect("GPU").clone();
        sim.state_a.body_q.upload(&queue)?;
        sim.state_a.body_qd.upload(&queue)?;
        Ok(())
    })();
    match result {
        Ok(()) => state.motion_tick = Some(sim.sensor_clock.tick),
        Err(error) => state.error = Some(error.to_string()),
    }
}

fn update_sensor_lab(
    sim: Res<Sim>,
    mut state: ResMut<SensorLabState>,
    mut images: ResMut<Assets<Image>>,
    mut textures: ResMut<EguiUserTextures>,
    ribbon: Res<RibbonOpen>,
    placement: Res<RibbonPlacement>,
) {
    if sim.scene != Scene::SensorLab || state.error.is_some() {
        return;
    }
    let result = (|| -> molla::core::Result<()> {
        if state.runtime.is_none() {
            state.runtime = Some(sensor_lab::LabRuntime::new(&sim, state.settings)?);
        }
        let settings = state.settings;
        let visible = find_item(RIBBON_ITEMS, PANE_SCENE).is_some_and(|item| {
            let (rail, _, _) = placement.resolve(item);
            ribbon.is_open(rail, PANE_SCENE)
        });
        if let Some(color) = state
            .runtime
            .as_mut()
            .expect("runtime")
            .update(&sim, settings, visible)?
        {
            if state.image.is_none() {
                let (image, texture) = alloc_sensor_image(
                    &mut images,
                    &mut textures,
                    sensor_lab::CAMERA_WIDTH,
                    sensor_lab::CAMERA_HEIGHT,
                );
                state.image = Some(image);
                state.texture = Some(texture);
            }
            if let Some(image) = state.image.as_ref().and_then(|h| images.get_mut(h)) {
                write_rgba_u32_into_image(image, &color);
                state.image_valid = true;
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        warn!("sensor lab failed: {error}");
        state.error = Some(error.to_string());
    }
}

fn draw_sensor_lab(sim: Res<Sim>, state: Res<SensorLabState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::SensorLab {
        return;
    }
    gizmos.sphere(Vec3::new(0.0, 1.0, 5.0), 0.08, Color::srgb(1.0, 0.9, 0.1));
    if let Some(runtime) = &state.runtime {
        for &(point, shape) in &runtime.cloud {
            if shape >= 0 {
                let color = match shape {
                    0 => Color::srgb(1.0, 0.35, 0.1),
                    1 => Color::srgb(0.2, 1.0, 0.4),
                    _ => Color::srgb(0.3, 0.7, 1.0),
                };
                gizmos.sphere(point.as_vec3(), 0.025, color);
            }
        }
    }
}

fn sensor_lab_pane(ui: &mut egui::Ui, state: &mut SensorLabState) {
    ui.label("Shared-scene sensor lab");
    ui.small("Prescribed motion; the occluder is a body-attached triangle mesh.");
    ui.small(
        "One snapshot per viewer step. Missed deadlines and deferred outputs are not replayed.",
    );
    if let Some(error) = &state.error {
        ui.colored_label(egui::Color32::LIGHT_RED, error);
        ui.small("Reset or switch scenes to retry.");
    }
    // Own id scopes: the error label and previews change the widget count.
    ui.push_id("lab_settings", |ui| {
        ui.add(
            egui::Slider::new(&mut state.settings.columns, 9..=129).text("LiDAR columns (5 rows)"),
        );
        ui.add(egui::Slider::new(&mut state.settings.lidar_fov, 0.3..=2.4).text("LiDAR FOV (rad)"));
        ui.add(egui::Slider::new(&mut state.settings.range, 2.0..=50.0).text("LiDAR range (m)"));
        ui.add(
            egui::Slider::new(&mut state.settings.camera_fov, 0.4..=1.7).text("Camera FOV (rad)"),
        );
    });
    for (index, name) in ["LiDAR", "Camera", "IMU"].into_iter().enumerate() {
        ui.separator();
        ui.label(name);
        ui.push_id(("lab_rate", index), |ui| {
            ui.add(egui::Slider::new(&mut state.settings.rates[index], 1..=60).text("Hz (sim)"))
        });
        if let Some(runtime) = &state.runtime {
            readout(ui, "Skipped deadlines", &runtime.skipped[index].to_string());
            sensor_preview_pane(ui, Some(&runtime.displays[index]), None);
        }
        if index == 0 {
            readout(
                ui,
                "Rays / sample",
                &(5 * state.settings.columns).to_string(),
            );
        } else if index == 1 {
            readout(
                ui,
                "Rays / sample",
                &(sensor_lab::CAMERA_WIDTH * sensor_lab::CAMERA_HEIGHT).to_string(),
            );
            if state.image_valid
                && let Some(texture) = state.texture
            {
                let width = ui.available_width().max(64.0);
                ui.image((texture, egui::vec2(width, width * 0.75)));
            }
        } else if let Some(reading) = state.runtime.as_ref().and_then(|r| r.reading) {
            readout(
                ui,
                "Specific force (m/s²)",
                &format!(
                    "{:.2}, {:.2}, {:.2}",
                    reading.accelerometer.x, reading.accelerometer.y, reading.accelerometer.z
                ),
            );
            readout(
                ui,
                "Gyro (rad/s)",
                &format!(
                    "{:.2}, {:.2}, {:.2}",
                    reading.gyroscope.x, reading.gyroscope.y, reading.gyroscope.z
                ),
            );
        } else {
            ui.small("IMU at moving-body origin; waiting for velocity history.");
        }
    }
}
