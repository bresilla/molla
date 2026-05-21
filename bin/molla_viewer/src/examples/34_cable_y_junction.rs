fn build_cable_y_junction_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let junction = MVec3::new(0.0, 2.5, 0.0);
    let branch_len = 1.2_f64;
    let particles_per_branch: u32 = 16;
    let dirs = [
        MVec3::new(1.0, 0.7, 0.0).normalize(),
        MVec3::new(-0.5, 0.7, 0.866).normalize(),
        MVec3::new(-0.5, 0.7, -0.866).normalize(),
    ];
    // 3 cable polylines all starting at junction, fanning outward.
    let mut firsts = Vec::with_capacity(3);
    for d in &dirs {
        let p_far = junction + *d * branch_len;
        let (first, _, _) = b.add_cable_polyline(
            junction,
            p_far,
            particles_per_branch,
            0.04,
            0.025,
            5.0e5,
            0.02,
            mp::WorldId(0),
        );
        firsts.push(first);
    }

    // Pin the very first particle of each cable so they share the
    // junction point (Newton uses add_rod_graph; we approximate by
    // pinning all three start particles in place).
    for f in &firsts {
        b.pin_particle(*f);
    }
    // The first cable's far endpoint stays kinematic (top hanger);
    // the other two are free.
    b.pin_particle(firsts[0] + particles_per_branch - 1);

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cable y junction model"), Vec::new())
}

// CableBundleHysteresis scene — several parallel cable strands in a compact
// bundle. This is the procedural viewer-side scaffold for Newton's
// `example_cable_bundle_hysteresis.py`; the contact/friction hysteresis proof
// currently lives in the Dahl oracle tests rather than in a fully coupled

