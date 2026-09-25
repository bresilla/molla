// IMU example runtime state. Lazy-initialised when entering the IMU
// scene; reset on scene switch.
#[derive(Resource, Default)]
struct ImuExampleState {
    sensor: Option<ImuSensor>,
    preview: Option<SensorPreview>,
    revision: Option<u64>,
    error: Option<String>,
    last_device: Option<molla::sensors::SensorFrame>,
    last_enqueued: Option<u64>,
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

fn update_imu_example(
    sim: Res<Sim>,
    mut state: ResMut<ImuExampleState>,
    ribbon: Res<RibbonOpen>,
    placement: Res<RibbonPlacement>,
) {
    if sim.scene != Scene::Imu {
        *state = ImuExampleState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = ImuExampleState {
            revision: Some(sim.sensor_clock.revision),
            ..Default::default()
        };
    }
    if let Some(preview) = &mut state.preview {
        preview.now = sim.sensor_clock.time();
    }
    let display = find_item(RIBBON_ITEMS, PANE_SCENE).is_some_and(|item| {
        let (rail, _, _) = placement.resolve(item);
        ribbon.is_open(rail, PANE_SCENE)
    });
    if state.error.is_none()
        && let Err(error) = state.sample(&sim, display)
    {
        warn!("imu preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

impl ImuExampleState {
    fn sample(&mut self, sim: &Sim, display: bool) -> molla::core::Result<()> {
        if self.sensor.is_none() {
            let sensor = ImuSensor::new(
                &sim.engine,
                &sim.model,
                &[ImuMount {
                    body: mp::BodyId::from_index(0),
                    local_pose: mp::Transform::IDENTITY,
                }],
            )?;
            let pool = sensor.output_pool(&sim.engine, preview_pool_config(32)?)?;
            self.preview = Some(SensorPreview::new(&sim.engine, pool, 32)?);
            self.sensor = Some(sensor);
        }
        let preview = self.preview.as_mut().expect("initialized preview");
        if let Some(frame) = preview.poll_latest(sim.sensor_clock)? {
            let accel = frame.channel_iter::<MVec3>(0)?.next();
            let gyro = frame.channel_iter::<MVec3>(1)?.next();
            if let Some((accelerometer, gyroscope)) = accel.zip(gyro) {
                self.last = Some(ImuReading {
                    accelerometer,
                    gyroscope,
                });
                self.smoothed_accel_mag =
                    0.8 * self.smoothed_accel_mag + 0.2 * accelerometer.length() as f32;
                self.smoothed_gyro_mag =
                    0.8 * self.smoothed_gyro_mag + 0.2 * gyroscope.length() as f32;
            }
        }
        let clock = sim.sensor_clock;
        if preview.begin_sample(clock.tick)? {
            let start = std::time::Instant::now();
            let mut batch = SensorCommandBatch::new(&sim.engine)?;
            let timed = display && preview.begin_gpu(&mut batch, &sim.engine)?;
            if let Some(sample) = self.sensor.as_mut().expect("sensor").encode(
                &mut batch,
                &sim.engine,
                &sim.model,
                &sim.state_a,
                preview.sample_dt(clock.tick),
                &mut preview.pool,
                clock.stamp(6),
            )? {
                preview.end_gpu(&mut batch, &sim.engine, timed)?;
                let submission = batch.submit()?;
                preview.submitted(clock.tick, start);
                preview.collect_gpu(clock.stamp(6), timed)?;
                if !sample.priming {
                    self.last_device = Some(submission.frame(&sample.frame)?.clone());
                }
            } else {
                preview.deferred += 1;
            }
        }
        if display
            && let Some(frame) = &self.last_device
            && self.last_enqueued != Some(frame.stamp().sample_id)
            && preview.readback.available_slots() > 0
            && preview.enqueue(
                &[
                    frame.channel::<MVec3>(0)?.readback_source(0..1)?,
                    frame.channel::<MVec3>(1)?.readback_source(0..1)?,
                ],
                frame.stamp(),
            )?
        {
            self.last_enqueued = Some(frame.stamp().sample_id);
        }
        Ok(())
    }
}

// GPU IK arm scene — runs `IkSolver::solve_gpu` (or its CPU mirror
// when `state.use_gpu` is false) every frame to drive the arm toward
// the user-controlled XYZ target, then writes `joint_q` and `body_q`
// on the host. `step_simulation` short-circuits for this scene so

fn imu_pane(ui: &mut egui::Ui, state: &ImuExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU IMU");
    sensor_preview_pane(ui, state.preview.as_ref(), state.error.as_deref());
    ui.separator();

    match state.last {
        None => {
            ui.label("priming…");
            ui.small("First sample snapshots velocity; later readings arrive asynchronously.");
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
