fn build_cable_bundle_hysteresis_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.35);

    let strand_count = 7usize;
    let particles_per_strand: u32 = 20;
    let start_x = -0.75_f64;
    let end_x = 0.75_f64;
    let y0 = 2.2_f64;
    for i in 0..strand_count {
        let centered = i as f64 - (strand_count as f64 - 1.0) * 0.5;
        let z = centered * 0.045;
        let y = y0 + (centered * 0.018).sin() * 0.035;
        let (first, n, _) = b.add_cable_polyline(
            MVec3::new(start_x, y, z),
            MVec3::new(end_x, y, z),
            particles_per_strand,
            0.035,
            0.018,
            8.0e5,
            0.015,
            mp::WorldId(0),
        );
        b.pin_particle(first);
        b.pin_particle(first + n - 1);
    }

    // Two light guide rails give the bundle something to rest against if the
    // user lowers damping/stiffness in the pane.
    for z in [-0.22_f64, 0.22_f64] {
        b.add_shape_cylinder(
            mp::BodyId::NONE,
            0.85,
            0.035,
            mp::Transform::new(
                MVec3::new(0.0, 1.95, z),
                MQuat::from_axis_angle(MVec3::Z, std::f64::consts::FRAC_PI_2),
            ),
        )
        .expect("bundle rail");
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (
        b.build().expect("cable bundle hysteresis model"),
        Vec::new(),
    )
}

const NUT_BOLT_SOCKET_WALLS: usize = 6;
const NUT_BOLT_THREAD_RINGS: usize = 3; // retained: helix spans the same 3-ring band
const NUT_BOLT_THREAD_PITCH: f64 = 0.20; // was the ring spacing — now the helix pitch
const NUT_BOLT_BOLT_RADIUS: f64 = 0.24;
const NUT_BOLT_BOLT_HALF_HEIGHT: f64 = 0.32;
const NUT_BOLT_SOCKET_RADIUS: f64 = 0.34;
const NUT_BOLT_THREAD_MAJOR: f64 = 0.30;
const NUT_BOLT_THREAD_MINOR: f64 = 0.035;

// Procedural dense nut/bolt SDF contact scene. It deliberately avoids any
// threaded mesh/importer path: the "nut" is a six-wall kinematic socket and
// the SDF contribution is a stack of analytic torus rings that behave like

