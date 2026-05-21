// --- Rope scene -----------------------------------------------------
const ROPE_LINKS: usize = 18;
const ROPE_LINK_RADIUS: f32 = 0.12;
const ROPE_LINK_SPACING: f32 = 0.26;
const ROPE_ANCHOR_HEIGHT: f32 = 6.0;

fn build_rope_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.10);

    let mut visuals = Vec::with_capacity(ROPE_LINKS);
    let mut bodies = Vec::with_capacity(ROPE_LINKS);

    for i in 0..ROPE_LINKS {
        let y = ROPE_ANCHOR_HEIGHT as f64 - i as f64 * ROPE_LINK_SPACING as f64;
        // Slight horizontal offset so the rope starts to swing instead
        // of sitting in unstable vertical equilibrium.
        let x = 0.001 * i as f64;
        let pos = MVec3::new(x, y, 0.0);

        let mut params = mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.4, ROPE_LINK_RADIUS as f64))
            .with_position(pos);

        if i == 0 {
            // Kinematic anchor: zero mass + KINEMATIC flag so the
            // integrator passes its pose through unchanged.
            params = params
                .with_mass(0.0)
                .with_inertia(molla::math::Mat3::ZERO)
                .with_flags(mp::BodyFlags::KINEMATIC);
        }

        let id = b.add_body(params);
        b.add_shape_sphere(id, ROPE_LINK_RADIUS as f64, mp::Transform::IDENTITY)
            .expect("rope sphere");
        bodies.push(id);
        visuals.push(BodyVisual::Sphere(ROPE_LINK_RADIUS));
    }

    // Pin every consecutive pair at the midpoint between their COMs.
    // Anchor offsets are body-local, so half a link spacing along ±Y.
    let half = (ROPE_LINK_SPACING as f64) * 0.5;
    for i in 1..ROPE_LINKS {
        let _ = b.add_equality_connect(
            bodies[i - 1],
            bodies[i],
            MVec3::new(0.0, -half, 0.0),
            MVec3::new(0.0, half, 0.0),
        );
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

// 8-layer 4×4 brick tower. A heavy sphere drops onto it from above —
// shows stacking stability, contact reduction, and impact impulses.
//
// Tried 45°-alternated layers initially (real masonry is bonded
// half-brick offsets); rotating 0.6 m bricks by 45° gives a 0.85 m
// diagonal, which overlaps neighbours within a layer and explodes the
// stack on frame 0. Now: identical pose every layer, straight columns.

