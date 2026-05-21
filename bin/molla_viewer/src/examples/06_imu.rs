// IMU example runtime state. Lazy-initialised when entering the IMU
// scene; reset on scene switch.
#[derive(Resource, Default)]
struct ImuExampleState {
    sensor: Option<ImuSensor>,
    /// Most recent reading (for the right-rail readout). `None` until
    /// the sensor primes.
    last: Option<ImuReading>,
    /// Smoothed magnitude (for the simple bar visualization).
    smoothed_accel_mag: f32,
    smoothed_gyro_mag: f32,
}

fn build_imu_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.0); // let the bounces stay lively

    let body = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, 0.4))
            .with_position(MVec3::new(0.0, 4.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.4, mp::Transform::IDENTITY)
        .expect("imu sphere");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("imu plane");

    (b.build().expect("model"), vec![BodyVisual::Sphere(0.4)])
}

fn update_imu_example(sim: ResMut<Sim>, mut state: ResMut<ImuExampleState>) {
    if sim.scene != Scene::Imu {
        if state.sensor.is_some() {
            state.sensor = None;
            state.last = None;
            state.smoothed_accel_mag = 0.0;
            state.smoothed_gyro_mag = 0.0;
        }
        return;
    }

    if state.sensor.is_none() {
        // Body 0 is the bouncing sphere we built in `build_imu_model`.
        let mounts = [ImuMount {
            body: mp::BodyId::from_index(0),
            local_pose: mp::Transform::IDENTITY,
        }];
        match ImuSensor::new(&sim.engine, &sim.model, &mounts) {
            Ok(s) => state.sensor = Some(s),
            Err(e) => {
                warn!("imu sensor init failed: {e}");
                return;
            }
        }
    }

    // dt for the IMU = the wall-clock time between consecutive frames'
    // body_qd snapshots, which is `SUBSTEPS_PER_FRAME * FIXED_DT`
    // (the integrator advances `body_qd` by that much per frame).
    let frame_dt = (SUBSTEPS_PER_FRAME as f64) * FIXED_DT;
    let Sim {
        ref engine,
        ref model,
        ref state_a,
        ..
    } = *sim;
    let sensor = state.sensor.as_mut().expect("sensor");
    let readings = match sensor.update(engine, model, state_a, frame_dt) {
        Ok(r) => r,
        Err(e) => {
            warn!("imu update failed: {e}");
            return;
        }
    };
    if let Some(&r) = readings.first() {
        state.last = Some(r);
        let alpha = 0.2_f32;
        state.smoothed_accel_mag =
            (1.0 - alpha) * state.smoothed_accel_mag + alpha * r.accelerometer.length() as f32;
        state.smoothed_gyro_mag =
            (1.0 - alpha) * state.smoothed_gyro_mag + alpha * r.gyroscope.length() as f32;
    }
}

// GPU IK arm scene — runs `IkSolver::solve_gpu` (or its CPU mirror
// when `state.use_gpu` is false) every frame to drive the arm toward
// the user-controlled XYZ target, then writes `joint_q` and `body_q`
// on the host. `step_simulation` short-circuits for this scene so

fn imu_pane(ui: &mut egui::Ui, state: &ImuExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU IMU");
    ui.separator();

    match state.last {
        None => {
            ui.label("priming…");
            ui.small("(first frame snapshots body_qd; reading shows next frame)");
        }
        Some(r) => {
            ui.label("Accelerometer (sensor frame)");
            readout(ui, "  ax", &format!("{:>8.3} m/s²", r.accelerometer.x));
            readout(ui, "  ay", &format!("{:>8.3} m/s²", r.accelerometer.y));
            readout(ui, "  az", &format!("{:>8.3} m/s²", r.accelerometer.z));
            readout(
                ui,
                "  |a|",
                &format!("{:>8.3} m/s²", r.accelerometer.length()),
            );
            ui.separator();
            ui.label("Gyroscope (sensor frame)");
            readout(ui, "  ωx", &format!("{:>8.3} rad/s", r.gyroscope.x));
            readout(ui, "  ωy", &format!("{:>8.3} rad/s", r.gyroscope.y));
            readout(ui, "  ωz", &format!("{:>8.3} rad/s", r.gyroscope.z));
            readout(ui, "  |ω|", &format!("{:>8.3} rad/s", r.gyroscope.length()));
            ui.separator();
            let accel_norm = (state.smoothed_accel_mag / 50.0).clamp(0.0, 1.0);
            let gyro_norm = (state.smoothed_gyro_mag / 20.0).clamp(0.0, 1.0);
            ui.label("|a| (0..50 m/s²)");
            ui.add(egui::ProgressBar::new(accel_norm).show_percentage());
            ui.label("|ω| (0..20 rad/s)");
            ui.add(egui::ProgressBar::new(gyro_norm).show_percentage());
        }
    }

    ui.add_space(4.0);
    ui.small(
        "At rest: |a| ≈ 9.81. Free fall: |a| ≈ 0. Bounce impacts \
         produce sharp accel spikes.",
    );
}

// Body for the right-rail pane in the Camera scene. Each of the
// 5 GPU output channels lives in its own `bevy_frost::Normal`
// container so the user can fold/unfold them individually; the
// thumbnails inside use the pane's available width, so dragging

