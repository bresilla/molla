fn build_solar_sky_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("solar floor");
    for &(x, z) in &[(-2.0, 0.0), (0.5, 1.5), (2.4, -1.2)] {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_mass(0.0)
                .with_inertia(molla::math::Mat3::ZERO)
                .with_position(MVec3::new(x, 0.5, z))
                .with_flags(mp::BodyFlags::KINEMATIC),
        );
        b.add_shape_box(body, MVec3::splat(0.5), mp::Transform::IDENTITY)
            .expect("solar box");
        visuals.push(BodyVisual::Box(Vec3::splat(0.5)));
    }
    (b.build().expect("solar sky model"), visuals)
}

// Per-frame state for [`Scene::SolarSky`]. The scene sweeps
// simulated time through one day on a loop; date is fixed at
// summer solstice so the sun arcs at its highest.
#[derive(Resource)]
struct SolarSkyExampleState {
    /// Scene latitude in degrees (`+N`).
    lat_deg: f64,
    /// Scene longitude in degrees (`+E`).
    lon_deg: f64,
    /// Sim seconds elapsed within the simulated day (`0..86400`).
    sim_seconds: f64,
    /// Sim-seconds per real-second. `1800` → one simulated day in 48 s.
    pub time_scale: f64,
    /// Computed sun direction in world frame (LTP: `+X` east, `+Y`
    /// up, `+Z` north). Initialised pointing up.
    sun_dir: Vec3,
}

impl Default for SolarSkyExampleState {
    fn default() -> Self {
        Self {
            lat_deg: 40.0,
            lon_deg: 0.0,
            sim_seconds: 6.0 * 3600.0, // start at 06:00 local
            time_scale: 1800.0,
            sun_dir: Vec3::Y,
        }
    }
}

// Advance the sim-day clock and recompute the sun direction.
fn update_solar_sky_example(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<SolarSkyExampleState>,
) {
    if sim.scene != Scene::SolarSky {
        return;
    }
    if !run.paused {
        state.sim_seconds += time.delta_secs() as f64 * state.time_scale;
        let day = 86_400.0_f64;
        state.sim_seconds = state.sim_seconds.rem_euclid(day);
    }
    let total = state.sim_seconds;
    let hour = (total / 3600.0).floor() as u32;
    let minute = ((total - hour as f64 * 3600.0) / 60.0).floor() as u32;
    let second = total - hour as f64 * 3600.0 - minute as f64 * 60.0;
    let dt = molla::math::CivilDateTime::new(2024, 6, 21, hour, minute, second);
    let s = molla::math::solar_position(state.lat_deg, state.lon_deg, dt, 0.0);
    state.sun_dir = Vec3::new(
        s.sun_direction[0] as f32,
        s.sun_direction[1] as f32,
        s.sun_direction[2] as f32,
    );
}

// Draw a sun marker at the computed direction plus a shaft from the
// scene origin. Below-horizon sun is dimmed.
fn draw_solar_sky_gizmos(sim: Res<Sim>, state: Res<SolarSkyExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::SolarSky {
        return;
    }
    let centre = Vec3::new(0.0, 0.5, 0.0);
    let sun_pos = centre + state.sun_dir * 14.0;
    let above = state.sun_dir.y > 0.0;
    let sun_color = if above {
        Color::srgb(1.0, 0.95, 0.45)
    } else {
        Color::srgb(0.25, 0.30, 0.55)
    };
    gizmos.sphere(sun_pos, 0.55, sun_color);
    gizmos.line(centre, sun_pos, sun_color);
    // Horizon ring for context.
    let ring_color = Color::srgb(0.40, 0.40, 0.50);
    let segs = 64;
    for k in 0..segs {
        let a0 = (k as f32) / segs as f32 * std::f32::consts::TAU;
        let a1 = ((k + 1) as f32) / segs as f32 * std::f32::consts::TAU;
        let r = 14.0_f32;
        let p0 = centre + Vec3::new(r * a0.cos(), 0.0, r * a0.sin());
        let p1 = centre + Vec3::new(r * a1.cos(), 0.0, r * a1.sin());
        gizmos.line(p0, p1, ring_color);
    }
}

// ── Stage H9 SyntheticAnnotation viewer wiring ─────────────────────

// Three labelled bodies on a ground for the annotation preview.
// Class assignment is the shape's index in `model.shape_geometry`

