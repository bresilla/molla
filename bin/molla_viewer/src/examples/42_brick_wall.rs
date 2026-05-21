const BRICK_COLS: usize = 10;
fn build_brick_wall_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.20);

    let mass = 0.6_f64;
    let h2 = BRICK_HALF * BRICK_HALF;
    let m_third = mass / 3.0;
    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));

    let mut visuals = Vec::new();

    let layer_h = BRICK_HALF.y * 2.0 + BRICK_GAP;
    let col_w = BRICK_HALF.x * 2.0 + BRICK_GAP;
    for row in 0..BRICK_ROWS {
        // Half-brick offset on every other row (real masonry).
        let stagger = if row % 2 == 0 { 0.0 } else { BRICK_HALF.x };
        for col in 0..BRICK_COLS {
            let x = (col as f64 - (BRICK_COLS as f64 - 1.0) * 0.5) * col_w + stagger;
            let y = BRICK_HALF.y + 0.001 + (row as f64) * layer_h;
            let body = b.add_body(
                mp::BodyParams::new()
                    .with_spatial_inertia(mp::SpatialInertia {
                        mass,
                        com: MVec3::ZERO,
                        inertia,
                    })
                    .with_position(MVec3::new(x, y, 0.0)),
            );
            b.add_shape_box(body, BRICK_HALF, mp::Transform::IDENTITY)
                .expect("brick");
            visuals.push(BodyVisual::Box(Vec3::new(
                BRICK_HALF.x as f32,
                BRICK_HALF.y as f32,
                BRICK_HALF.z as f32,
            )));
        }
    }

    // Heavy projectile dropped from the +Z side. Aim it diagonally so
    // gravity gives it a downward + side-on impact on the wall.
    let proj_radius = 0.45_f64;
    let proj_mass = 25.0_f64;
    let projectile = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(proj_mass, proj_radius))
            .with_position(MVec3::new(0.0, 12.0, 3.5)),
    );
    b.add_shape_sphere(projectile, proj_radius, mp::Transform::IDENTITY)
        .expect("projectile");
    visuals.push(BodyVisual::Sphere(proj_radius as f32));

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("brick wall model"), visuals)
}
// SoftbodyDrop scene — a 6×6×3 tet jelly cube dropped onto a hanging
// VBD cloth. Tests soft-on-soft contact at the engine layer. Port of
// `xtra/newton/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py`.
const SOFT_DROP_TET_NX: u32 = 6;
const SOFT_DROP_TET_NY: u32 = 6;
const SOFT_DROP_TET_NZ: u32 = 3;
const SOFT_DROP_TET_DX: f64 = 0.10;
const SOFT_DROP_CLOTH_NX: u32 = 24;
