// Pyramid scene — 10-layer 4-sided box pyramid struck by a heavy ball
// from above. Larger and more dramatic than the existing `Tower`.
// Port of `xtra/newton/newton/examples/contacts/example_pyramid.py`.
const PYRAMID_LEVELS: usize = 10;
const PYRAMID_BOX_HALF: f64 = 0.30;
// 2.05× spacing matches the Newton example — small gap so the
// neighbouring boxes don't collide on frame 0.
const PYRAMID_SPACING: f64 = 2.05 * PYRAMID_BOX_HALF;
const PYRAMID_BALL_RADIUS: f64 = 1.2;
const PYRAMID_BALL_MASS: f64 = 80.0;
const PYRAMID_BALL_HEIGHT: f64 = 14.0;
fn build_pyramid_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.20);

    let half = PYRAMID_BOX_HALF;
    let spacing = PYRAMID_SPACING;
    let inertia = box_inertia(1.0, half);
    let half_v = MVec3::splat(half);

    let mut visuals = Vec::new();
    let layer_y0 = half + 0.001;

    // 4-sided pyramid: layer L contains (PYRAMID_LEVELS - L)² boxes.
    for level in 0..PYRAMID_LEVELS {
        let row_count = PYRAMID_LEVELS - level;
        let y = layer_y0 + (level as f64) * (PYRAMID_BOX_HALF * 2.0 + 0.005);
        for ix in 0..row_count {
            for iz in 0..row_count {
                let x = (ix as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia {
                            mass: 1.0,
                            com: MVec3::ZERO,
                            inertia,
                        })
                        .with_position(MVec3::new(x, y, z)),
                );
                b.add_shape_box(body, half_v, mp::Transform::IDENTITY)
                    .expect("pyramid box");
                visuals.push(BodyVisual::Box(Vec3::splat(half as f32)));
            }
        }
    }

    // Heavy wrecking ball.
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(
                PYRAMID_BALL_MASS,
                PYRAMID_BALL_RADIUS,
            ))
            // Drop slightly off-centre so the impact knocks the
            // pyramid sideways rather than punching straight through.
            .with_position(MVec3::new(1.5, PYRAMID_BALL_HEIGHT, 0.0)),
    );
    b.add_shape_sphere(ball, PYRAMID_BALL_RADIUS, mp::Transform::IDENTITY)
        .expect("ball");
    visuals.push(BodyVisual::Sphere(PYRAMID_BALL_RADIUS as f32));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("pyramid model"), visuals)
}
