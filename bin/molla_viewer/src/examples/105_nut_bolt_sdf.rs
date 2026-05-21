fn build_nut_bolt_sdf_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.12);
    b.set_linear_damping(0.08);
    b.set_default_friction(0.85).expect("default friction");
    b.set_default_restitution(0.0).expect("default restitution");

    let mut visuals = Vec::new();
    let wall_half = MVec3::new(0.24, 0.42, 0.045);

    for i in 0..NUT_BOLT_SOCKET_WALLS {
        let angle = i as f64 * std::f64::consts::TAU / NUT_BOLT_SOCKET_WALLS as f64;
        let radial = MVec3::new(angle.cos(), 0.0, angle.sin());
        let wall = b.add_body(
            mp::BodyParams::new()
                .with_mass(0.0)
                .with_pose(mp::Transform::new(
                    radial * NUT_BOLT_SOCKET_RADIUS + MVec3::new(0.0, wall_half.y, 0.0),
                    MQuat::from_axis_angle(MVec3::Y, -angle),
                ))
                .with_flags(mp::BodyFlags::DYNAMIC | mp::BodyFlags::KINEMATIC),
        );
        b.add_shape_box(wall, wall_half, mp::Transform::IDENTITY)
            .expect("nut socket wall");
        visuals.push(BodyVisual::Box(Vec3::new(
            wall_half.x as f32,
            wall_half.y as f32,
            wall_half.z as f32,
        )));
    }

    // Thread-like analytic SDF lips. The current SDF narrow phase is CPU-only,
    // so the scene audit names GPU reduction/grid lanes separately instead of
    // pretending the analytic SDF contact itself is GPU resident.
    for i in 0..NUT_BOLT_THREAD_RINGS {
        let sdf_id = b.add_sdf(mp::SdfShape::Torus {
            major: NUT_BOLT_THREAD_MAJOR,
            minor: NUT_BOLT_THREAD_MINOR,
        });
        let y = 0.30 + i as f64 * 0.20;
        b.add_shape_sdf(
            mp::BodyId::NONE,
            sdf_id,
            mp::Transform::from_translation(MVec3::new(0.0, y, 0.0)),
        )
        .expect("nut thread SDF ring");
    }

    let bolt_mass = 1.0;
    let bolt_inertia = molla::math::Mat3::from_diagonal(MVec3::new(
        bolt_mass
            * (3.0 * NUT_BOLT_BOLT_RADIUS * NUT_BOLT_BOLT_RADIUS
                + (2.0 * NUT_BOLT_BOLT_HALF_HEIGHT).powi(2))
            / 12.0,
        0.5 * bolt_mass * NUT_BOLT_BOLT_RADIUS * NUT_BOLT_BOLT_RADIUS,
        bolt_mass
            * (3.0 * NUT_BOLT_BOLT_RADIUS * NUT_BOLT_BOLT_RADIUS
                + (2.0 * NUT_BOLT_BOLT_HALF_HEIGHT).powi(2))
            / 12.0,
    ));
    let bolt = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: bolt_mass,
                com: MVec3::ZERO,
                inertia: bolt_inertia,
            })
            .with_pose(mp::Transform::new(
                MVec3::new(0.09, 1.55, -0.06),
                MQuat::from_axis_angle(MVec3::Z, 0.18),
            )),
    );
    b.add_shape_cylinder(
        bolt,
        NUT_BOLT_BOLT_HALF_HEIGHT,
        NUT_BOLT_BOLT_RADIUS,
        mp::Transform::IDENTITY,
    )
    .expect("bolt cylinder");
    // Render the cylinder with the existing capsule visual surrogate used by
    // the primitive-mix scene; collision remains an actual cylinder.
    visuals.push(BodyVisual::Capsule {
        half_h: NUT_BOLT_BOLT_HALF_HEIGHT as f32,
        radius: NUT_BOLT_BOLT_RADIUS as f32,
    });

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("socket floor plane");

    let mut model = b.build().expect("nut bolt SDF model");
    model.contact_margin = 0.02;
    (model, visuals)
}

