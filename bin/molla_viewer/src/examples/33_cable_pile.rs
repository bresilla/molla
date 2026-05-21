fn build_cable_pile_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);

    // Bowl: floor + 4 vertical walls.
    let bowl_size = 1.6_f64;
    let wall_h = 0.8_f64;
    let wall_t = 0.05_f64;
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(bowl_size, wall_h, wall_t),
        mp::Transform::from_translation(MVec3::new(0.0, wall_h, bowl_size)),
    )
    .expect("wall +Z");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(bowl_size, wall_h, wall_t),
        mp::Transform::from_translation(MVec3::new(0.0, wall_h, -bowl_size)),
    )
    .expect("wall -Z");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(wall_t, wall_h, bowl_size),
        mp::Transform::from_translation(MVec3::new(bowl_size, wall_h, 0.0)),
    )
    .expect("wall +X");
    b.add_shape_box(
        mp::BodyId::NONE,
        MVec3::new(wall_t, wall_h, bowl_size),
        mp::Transform::from_translation(MVec3::new(-bowl_size, wall_h, 0.0)),
    )
    .expect("wall -X");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    // 8 cables, alternating horizontal axis, dropped from staggered
    // heights. No pins — they just fall and pile.
    let cable_len = 1.4_f64;
    let particles_per_cable: u32 = 24;
    let mut rng_state: u64 = 0xFEEDFACECAFEBEEF;
    fn rng(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    for i in 0..8 {
        let h = 1.5 + (i as f64) * 0.30;
        let yaw = (i as f64) * 0.7 + rng(&mut rng_state) * 0.2;
        let dx = yaw.cos() * cable_len * 0.5;
        let dz = yaw.sin() * cable_len * 0.5;
        let cx = 0.6 * rng(&mut rng_state);
        let cz = 0.6 * rng(&mut rng_state);
        let p0 = MVec3::new(cx - dx, h, cz - dz);
        let p1 = MVec3::new(cx + dx, h, cz + dz);
        let _ = b.add_cable_polyline(
            p0,
            p1,
            particles_per_cable,
            0.04,  // particle mass
            0.025, // particle radius
            5.0e5, // stiffness
            0.02,  // hard cap on stretch
            mp::WorldId(0),
        );
    }

    (b.build().expect("cable pile model"), Vec::new())
}
// CableYJunction scene — three cables meeting at a Y junction. The
// first particle of each cable is welded to the same world point via
// a kinematic anchor body and `CONNECT` equalities. One cable is
// kinematic-pinned at its free end (top); the other two sag from
// gravity. Port of

