// Contact example runtime state. Lazy-initialised when entering the
// Contact scene; reset on scene switch. Stores the latest per-body
// readings for the right-rail readout and the gizmo system.
#[derive(Resource, Default)]
struct ContactExampleState {
    sensor: Option<ContactSensor>,
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

fn update_contact_example(sim: ResMut<Sim>, mut state: ResMut<ContactExampleState>) {
    if sim.scene != Scene::Contact {
        if state.sensor.is_some() {
            state.sensor = None;
            state.readings.clear();
        }
        return;
    }

    if state.sensor.is_none() {
        match ContactSensor::new(&sim.engine, &sim.model) {
            Ok(s) => state.sensor = Some(s),
            Err(e) => {
                warn!("contact sensor init failed: {e}");
                return;
            }
        }
    }

    let Sim {
        ref engine,
        ref contacts,
        ..
    } = *sim;
    let sensor = state.sensor.as_mut().expect("contact sensor");
    match sensor.update(engine, contacts) {
        Ok(r) => state.readings = r,
        Err(e) => warn!("contact sensor update failed: {e}"),
    }
}

// Camera-sensor demo — orbits a single `TiledCamera` around the
// scene each frame and downloads every channel. The right-rail pane
// (`camera_pane`) reads from the cached arrays. No-op when the
// scene isn't active.

fn draw_contact_gizmos(sim: Res<Sim>, state: Res<ContactExampleState>, mut gizmos: Gizmos) {
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
            // Sandwiched body — net force ~0. Just mark the centroid.
            gizmos.sphere(centroid, 0.04, dot_color);
            continue;
        }
        let dir = normal_sum / len;
        // Arrow length: 0.15m base + scaled by depth (capped).
        let arrow_len = 0.15 + (r.max_depth as f32 * 5.0).min(0.4);
        let tip = centroid + dir * arrow_len;
        gizmos.line(centroid, tip, arrow_color);
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

// Draw the Stage-23 diff-arm scene's overlays:
//   - Yellow sphere at the target XYZ.
//   - When `show_trajectory` is on, a polyline through the
//     per-iteration EE positions, fading from cool (early iter) to
//     warm (late iter) so the GD progression reads at a glance.
//   - Small dot at the live EE position (last body[1] tip in the

fn contact_pane(ui: &mut egui::Ui, state: &ContactExampleState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU contact aggregate");
    let total_contacts: u32 = state.readings.iter().map(|r| r.count).sum();
    let in_contact: usize = state.readings.iter().filter(|r| r.count > 0).count();
    let max_depth: f64 = state
        .readings
        .iter()
        .map(|r| r.max_depth)
        .fold(0.0_f64, f64::max);
    readout(ui, "Bodies", &format!("{}", state.readings.len()));
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
                        "n={} d={:.4} push=({:>5.2},{:>5.2},{:>5.2})",
                        r.count, r.max_depth, r.normal_sum.x, r.normal_sum.y, r.normal_sum.z,
                    )
                };
                readout(ui, &label, &txt);
            }
        });
    ui.separator();
    ui.small(
        "Yellow arrows in the viewport point along the net push \
         direction at each body's contact centroid; length scales \
         with penetration depth.",
    );
}


