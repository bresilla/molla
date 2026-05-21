// --- Tower scene ----------------------------------------------------
const TOWER_LAYERS: usize = 8;
const TOWER_GRID: usize = 4;
const TOWER_BRICK_HALF: f32 = 0.30;
// Brick is 0.6m on a side. Spacing 0.65 = 0.05m gap so neighbours
// don't kiss each other on frame 0 and trigger jitter.
const TOWER_BRICK_SPACING: f32 = 0.65;
// Vertical: brick height = 0.6, gap 0.005, so 0.605m per layer.
const TOWER_LAYER_HEIGHT: f32 = 0.605;
const TOWER_BALL_RADIUS: f32 = 1.0;
const TOWER_BALL_MASS: f64 = 50.0;
const TOWER_BALL_HEIGHT: f32 = 14.0;

fn build_tower_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Higher angular damping than the other scenes — bricks shouldn't
    // spin gracefully in mid-air, they should drop and stop.
    b.set_angular_damping(0.20);
    // Linear damping is set globally by `Sim::finalize` from
    // `LINEAR_DAMPING`; the per-scene call would be overwritten.

    let mut visuals = Vec::with_capacity(TOWER_LAYERS * TOWER_GRID * TOWER_GRID + 1);
    let half = TOWER_BRICK_HALF as f64;
    let half_v = MVec3::splat(half);
    let inertia = box_inertia(1.0, half);

    let layer_y0 = half + 0.001; // tiny lift to avoid frame-0 plane overlap
    for layer in 0..TOWER_LAYERS {
        let y = layer_y0 + (layer as f64) * (TOWER_LAYER_HEIGHT as f64);
        for ix in 0..TOWER_GRID {
            for iz in 0..TOWER_GRID {
                let x = (ix as f64 - (TOWER_GRID as f64 - 1.0) * 0.5) * TOWER_BRICK_SPACING as f64;
                let z = (iz as f64 - (TOWER_GRID as f64 - 1.0) * 0.5) * TOWER_BRICK_SPACING as f64;
                let pose = mp::Transform::new(MVec3::new(x, y, z), MQuat::IDENTITY);
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_spatial_inertia(mp::SpatialInertia {
                            mass: 1.0,
                            com: MVec3::ZERO,
                            inertia,
                        })
                        .with_pose(pose),
                );
                b.add_shape_box(body, half_v, mp::Transform::IDENTITY)
                    .expect("brick shape");
                visuals.push(BodyVisual::Box(Vec3::splat(TOWER_BRICK_HALF)));
            }
        }
    }

    // Heavy sphere wrecking ball.
    let ball_radius = TOWER_BALL_RADIUS as f64;
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(TOWER_BALL_MASS, ball_radius))
            .with_position(MVec3::new(0.0, TOWER_BALL_HEIGHT as f64, 0.0)),
    );
    b.add_shape_sphere(ball, ball_radius, mp::Transform::IDENTITY)
        .expect("ball shape");
    visuals.push(BodyVisual::Sphere(TOWER_BALL_RADIUS));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

// Horizontal cloth dropping onto a static sphere.
//
// The cloth starts a few metres above the sphere with no pins, so it
// falls under gravity and drapes around the sphere. Particle-vs-rigid
// collision (sphere SDF projection) is handled by the WGSL
// `particle_vs_shapes` pass in the GPU XPBD pipeline. Mouse grab still

