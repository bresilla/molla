// Lazily-initialised lidar runtime state — the GPU `Raycaster` and
// the most recent set of (origin, dir, hit) triplets used by the
// gizmo system. Reset on scene switch.
#[derive(Resource, Default)]
struct LidarState {
    raycaster: Option<Raycaster>,
    /// Capacity the raycaster was built for; rebuild if the config
    /// asks for more rays than this.
    raycaster_capacity: u32,
    /// Last frame's rays + hit results (one entry per ray).
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

fn update_lidar(
    time: Res<Time>,
    sim: ResMut<Sim>,
    mut config: ResMut<LidarConfig>,
    mut state: ResMut<LidarState>,
) {
    if sim.scene != Scene::Lidar {
        if !state.rays.is_empty() {
            state.rays.clear();
        }
        return;
    }

    // Auto-rotate yaw if configured. Wraps in [0, 2π).
    if config.auto_rotate_speed.abs() > 1e-6 {
        config.yaw += config.auto_rotate_speed * time.delta_secs();
        let two_pi = std::f32::consts::TAU;
        if config.yaw > two_pi {
            config.yaw -= two_pi;
        }
        if config.yaw < 0.0 {
            config.yaw += two_pi;
        }
    }

    let nh = config.ray_count_h.max(1);
    let nv = config.ray_count_v.max(1);
    let n_rays = (nh * nv) as usize;

    // Build (or rebuild) the GPU raycaster if capacity changed.
    if state.raycaster.is_none() || state.raycaster_capacity < n_rays as u32 {
        // Round up to next multiple of 64 to play nicely with the
        // kernel's workgroup size.
        let cap = (n_rays as u32).div_ceil(64) * 64;
        state.raycaster =
            Some(Raycaster::new(&sim.engine, cap.max(64)).expect("lidar Raycaster init"));
        state.raycaster_capacity = cap.max(64);
    }

    // Build N ray origins + directions on host. yaw/pitch convention:
    // yaw = 0 looks at +Z (forward), increases toward +X (right);
    // pitch = 0 horizontal, positive looks up.
    let mut origins = Vec::with_capacity(n_rays);
    let mut directions = Vec::with_capacity(n_rays);

    let half_h = config.fov_h * 0.5;
    let half_v = config.fov_v * 0.5;
    let dh = if nh > 1 {
        config.fov_h / (nh - 1) as f32
    } else {
        0.0
    };
    let dv = if nv > 1 {
        config.fov_v / (nv - 1) as f32
    } else {
        0.0
    };

    for iy in 0..nv {
        let pitch_off = if nv > 1 {
            -half_v + iy as f32 * dv
        } else {
            0.0
        };
        let pitch = config.pitch + pitch_off;
        for ix in 0..nh {
            let yaw_off = if nh > 1 {
                -half_h + ix as f32 * dh
            } else {
                0.0
            };
            let yaw = config.yaw + yaw_off;
            // Spherical → Cartesian. Forward is +Z when yaw = pitch = 0.
            let cp = pitch.cos();
            let dir = Vec3::new(yaw.sin() * cp, pitch.sin(), yaw.cos() * cp);
            origins.push(MVec3::new(
                config.origin.x as f64,
                config.origin.y as f64,
                config.origin.z as f64,
            ));
            directions.push(MVec3::new(dir.x as f64, dir.y as f64, dir.z as f64));
        }
    }

    // Need to make sure model + state are uploaded. The pile/tower/etc.
    // pipeline uploads model once; the lidar scene model was uploaded
    // in `Sim::finalize` and `state.body_q` in `step_simulation`.
    let raycaster = state.raycaster.as_mut().expect("raycaster");
    let max_range = config.max_range as f64;
    let hits = match raycaster.cast(
        &sim.engine,
        &sim.model,
        &sim.state_a,
        &origins,
        &directions,
        max_range,
    ) {
        Ok(h) => h,
        Err(e) => {
            warn!("lidar raycast failed: {e}");
            return;
        }
    };

    // Stash for the gizmo system. Convert from molla f64 vecs to
    // bevy f32 along the way.
    state.rays.clear();
    state.rays.reserve(n_rays);
    for (i, hit) in hits.iter().enumerate() {
        let o = origins[i];
        let d = directions[i];
        let origin = Vec3::new(o.x as f32, o.y as f32, o.z as f32);
        let direction = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        let hit_world: Option<Vec3> = hit_to_world(hit);
        state.rays.push(LidarRay {
            origin,
            direction,
            hit: hit_world,
        });
    }
}

// IMU example — runs the GPU `ImuSensor` against the bouncing
// sphere each frame and stashes the latest reading for the right-rail

fn draw_lidar_gizmos(
    sim: Res<Sim>,
    state: Res<LidarState>,
    config: Res<LidarConfig>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::Lidar {
        return;
    }

    // Sensor body indicator.
    gizmos.sphere(config.origin, 0.06, Color::srgb(1.0, 0.95, 0.2));

    let hit_color = Color::srgb(1.0, 0.25, 0.20);
    let miss_color = Color::srgba(0.30, 0.75, 0.85, 0.18);
    let cross_color = Color::srgb(1.0, 1.0, 1.0);
    let cross_half = 0.06_f32;

    for ray in state.rays.iter() {
        let end = match ray.hit {
            Some(p) => p,
            None => ray.origin + ray.direction * config.max_range,
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

// Render the CABLE polyline as a connected gizmo line strip. The
// physics representation is just particles + segment springs — no
// triangle mesh — so we draw it directly each frame from the host
// `particle_q` mirror that `step_simulation` keeps fresh.
//
// Colours: orange line strip for the cable; yellow markers on pinned

fn lidar_pane(ui: &mut egui::Ui, config: &mut LidarConfig, state: &LidarState) {
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU raycast");
    readout(ui, "Rays", &format!("{}", state.rays.len()));
    let n_hits = state.rays.iter().filter(|r| r.hit.is_some()).count();
    readout(ui, "Hits", &format!("{n_hits}"));
    ui.separator();

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


