// ── Newton-port scene builders (Group A-E). Each block defines its
//    own constants near the top, the build_*_model() function, and
//    (optionally) a state struct, an update_*_example() pre-step
//    system, and a *_pane() right-rail panel.
// Joint-sampler scene: 3 columns side-by-side, each showing one joint
// kind in motion. Port of `xtra/newton/newton/examples/basic/example_basic_joints.py`.
//   * Column 0 (-2): revolute pendulum about +Z (rotating arm).
//   * Column 1 ( 0): prismatic slider along +Y.
//   * Column 2 (+2): ball joint (3D pendulum).
//
// We omit Newton's 4th "distance" column because molla's Featherstone
// solver doesn't enforce XPBD `CONNECT` equalities, and XPBD doesn't
// dispatch revolute/ball joint constraints — so a single scene can't
// show all four under one solver. The `Rope` scene already covers the
// distance-equality case via XPBD CONNECT.
fn build_joint_sampler_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let base_y = 4.0_f64;
    let half_h = 0.4_f64;
    let radius = 0.07_f64;
    let mass = 0.6_f64;
    let inertia = capsule_inertia(mass, half_h, radius);
    let mut visuals = Vec::new();

    // Helper: hang a capsule from a revolute/prismatic anchor at the
    // given column x-offset. Joint flip rotates joint-local +Y onto
    // world -Y so the rest pose hangs straight down (matches
    // `build_arm_model_with`).
    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);

    // ── Column 0: revolute about +Z. ──────────────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(-2.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(anchor - MVec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("rev cap");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = mp::Transform::new(anchor, flip);
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_revolute(mp::BodyId::NONE, body, x_p, x_c, MVec3::Z)
            .expect("revolute joint");
    }

    // ── Column 1: prismatic along +Y. ─────────────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(0.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia: box_inertia_xyz(mass, 0.15, 0.15, 0.15),
                })
                .with_position(anchor - MVec3::new(0.0, 0.5, 0.0)),
        );
        b.add_shape_box(body, MVec3::splat(0.15), mp::Transform::IDENTITY)
            .expect("prism box");
        visuals.push(BodyVisual::Box(Vec3::splat(0.15)));
        let x_p = mp::Transform::from_translation(anchor);
        let x_c = mp::Transform::IDENTITY;
        b.add_joint_prismatic(mp::BodyId::NONE, body, x_p, x_c, MVec3::Y)
            .expect("prismatic joint");
    }

    // ── Column 2: ball joint (3D pendulum). ───────────────────────
    {
        b.begin_articulation();
        let anchor = MVec3::new(2.0, base_y, 0.0);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(anchor - MVec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("ball cap");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let x_p = mp::Transform::new(anchor, flip);
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        b.add_joint_ball(mp::BodyId::NONE, body, x_p, x_c)
            .expect("ball joint");
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("joint sampler model"), visuals)
}
// Heightfield scene — 32×32 sinusoidal heightfield with ~30 mixed
// primitives dropped on top. Port of `example_basic_heightfield.py`.

