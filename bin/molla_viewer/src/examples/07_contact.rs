#[derive(Default, Reflect, GizmoConfigGroup)]
struct ContactNormalGizmos;

#[derive(Resource, Default)]
struct ContactExampleState {
    sensor: Option<ContactSensor>,
    preview: Option<SensorPreview>,
    revision: Option<u64>,
    error: Option<String>,
    readings: Vec<BodyContactReading>,
}

fn build_contact_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.1);
    b.set_linear_damping(0.5);

    let mut visuals = Vec::new();

    // Bottom box (large flat-ish plinth).
    {
        let h = MVec3::new(1.0, 0.25, 1.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 4.0,
                    com: MVec3::ZERO,
                    inertia: box_inertia(4.0, 0.5),
                })
                .with_position(MVec3::new(0.0, 0.4, 0.0)),
        );
        b.add_shape_box(body, h, mp::Transform::IDENTITY)
            .expect("box");
        visuals.push(BodyVisual::Box(Vec3::new(1.0, 0.25, 1.0)));
    }
    // Middle sphere.
    {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia::sphere(1.5, 0.4))
                .with_position(MVec3::new(0.2, 1.6, 0.0)),
        );
        b.add_shape_sphere(body, 0.4, mp::Transform::IDENTITY)
            .expect("sphere");
        visuals.push(BodyVisual::Sphere(0.4));
    }
    // Top capsule.
    {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 1.0,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(1.0, 0.4, 0.25),
                })
                .with_position(MVec3::new(-0.1, 2.6, 0.05)),
        );
        b.add_shape_capsule(body, 0.4, 0.25, mp::Transform::IDENTITY)
            .expect("capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: 0.4,
            radius: 0.25,
        });
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("model"), visuals)
}

fn update_contact_example(sim: Res<Sim>, mut state: ResMut<ContactExampleState>) {
    if sim.scene != Scene::Contact {
        *state = ContactExampleState::default();
        return;
    }
    if state.revision != Some(sim.sensor_clock.revision) {
        *state = ContactExampleState {
            revision: Some(sim.sensor_clock.revision),
            ..Default::default()
        };
    }
    if let Some(preview) = &mut state.preview {
        preview.now = sim.sensor_clock.time();
    }
    if state.error.is_none()
        && let Err(error) = state.sample(&sim)
    {
        warn!("contact preview failed: {error}");
        state.error = Some(error.to_string());
    }
}

impl ContactExampleState {
    fn sample(&mut self, sim: &Sim) -> molla::core::Result<()> {
        if self.sensor.is_none() {
            let sensor = ContactSensor::new(&sim.engine, &sim.model)?;
            let bytes = (sim.model.body_count as u64)
                .checked_mul(40)
                .ok_or_else(|| {
                    molla::core::Error::Storage("contact preview capacity overflow".into())
                })?;
            let pool = sensor.output_pool(&sim.engine, preview_pool_config(bytes)?)?;
            self.preview = Some(SensorPreview::new(&sim.engine, pool, bytes)?);
            self.sensor = Some(sensor);
        }
        let preview = self.preview.as_mut().expect("initialized preview");
        if let Some(frame) = preview.poll_latest(sim.sensor_clock)? {
            self.readings.clear();
            for (((count, max_depth), normal_sum), centroid) in frame
                .channel_iter::<u32>(0)?
                .zip(frame.channel_iter::<f64>(1)?)
                .zip(frame.channel_iter::<MVec3>(2)?)
                .zip(frame.channel_iter::<MVec3>(3)?)
            {
                self.readings.push(BodyContactReading {
                    count,
                    max_depth,
                    normal_sum,
                    centroid,
                });
            }
        }
        let clock = sim.sensor_clock;
        if clock.tick == 0 || !preview.begin_sample(clock.tick)? {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = preview.begin_gpu(&mut batch, &sim.engine)?;
        let Some(sample) = self.sensor.as_mut().expect("sensor").encode(
            &mut batch,
            &sim.engine,
            &sim.contacts,
            &mut preview.pool,
            clock.stamp(7),
        )?
        else {
            preview.deferred += 1;
            return Ok(());
        };
        preview.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        preview.submitted(clock.tick, start);
        preview.collect_gpu(clock.stamp(7), timed)?;
        let frame = submission.frame(&sample)?;
        let n = sim.model.body_count;
        preview.enqueue(
            &[
                frame.channel::<u32>(0)?.readback_source(0..n)?,
                frame.channel::<f64>(1)?.readback_source(0..n)?,
                frame.channel::<MVec3>(2)?.readback_source(0..n)?,
                frame.channel::<MVec3>(3)?.readback_source(0..n)?,
            ],
            frame.stamp(),
        )?;
        Ok(())
    }
}

fn draw_contact_gizmos(
    sim: Res<Sim>,
    state: Res<ContactExampleState>,
    mut gizmos: Gizmos<ContactNormalGizmos>,
) {
    if sim.scene != Scene::Contact {
        return;
    }
    let arrow_color = Color::srgb(1.0, 0.85, 0.20);
    let dot_color = Color::srgb(1.0, 0.95, 0.55);
    for r in state.readings.iter() {
        if r.count == 0 {
            continue;
        }
        let centroid = Vec3::new(
            r.centroid.x as f32,
            r.centroid.y as f32,
            r.centroid.z as f32,
        );
        let normal_sum = Vec3::new(
            r.normal_sum.x as f32,
            r.normal_sum.y as f32,
            r.normal_sum.z as f32,
        );
        let len = normal_sum.length();
        if len < 1e-6 {
            // Opposing contact normals cancel at the centroid.
            gizmos.sphere(centroid, 0.04, dot_color);
            continue;
        }
        let dir = normal_sum / len;
        // Arrow length: 0.15m base + scaled by depth (capped).
        let arrow_len = 0.15 + (r.max_depth as f32 * 5.0).min(0.4);
        let tip = centroid + dir * arrow_len;
        gizmos
            .arrow(centroid, tip, arrow_color)
            .with_tip_length(arrow_len * 0.35);
        // Tiny "+" cross at the tail.
        let cross = 0.04;
        gizmos.line(
            centroid - Vec3::X * cross,
            centroid + Vec3::X * cross,
            dot_color,
        );
        gizmos.line(
            centroid - Vec3::Y * cross,
            centroid + Vec3::Y * cross,
            dot_color,
        );
    }
}

fn contact_pane(ui: &mut egui::Ui, state: &ContactExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU contact aggregate");
    sensor_preview_pane(ui, state.preview.as_ref(), state.error.as_deref());
    let total_contacts: u32 = state.readings.iter().map(|r| r.count).sum();
    let in_contact: usize = state.readings.iter().filter(|r| r.count > 0).count();
    let max_depth: f64 = state
        .readings
        .iter()
        .map(|r| r.max_depth)
        .fold(0.0_f64, f64::max);
    readout(ui, "Sampled bodies", &format!("{}", state.readings.len()));
    if state.readings.is_empty() && state.error.is_none() {
        ui.small("Waiting for a physics sample.");
    }
    readout(ui, "In contact", &format!("{}", in_contact));
    readout(ui, "Σ contacts", &format!("{}", total_contacts));
    readout(ui, "Max depth", &format!("{max_depth:.4} m"));
    ui.separator();

    ui.label("Per-body");
    egui::ScrollArea::vertical()
        .max_height(360.0)
        .show(ui, |ui| {
            for (i, r) in state.readings.iter().enumerate() {
                let label = format!("body {i:>2}");
                let txt = if r.count == 0 {
                    "—".to_string()
                } else {
                    format!(
                        "n={} d={:.4} normals=({:>5.2},{:>5.2},{:>5.2})",
                        r.count, r.max_depth, r.normal_sum.x, r.normal_sum.y, r.normal_sum.z,
                    )
                };
                readout(ui, &label, &txt);
            }
        });
    ui.separator();
    ui.small(
        "Yellow overlay arrows show summed signed contact normals at the centroid, \
         including through bodies. \
         Length scales with penetration depth; this is not a force measurement.",
    );
}
