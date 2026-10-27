//! Stage 7 — equality constraints (CONNECT / WELD).

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Quat, Real, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, Contacts, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn box_inertia(mass: Real, half: Vec3) -> Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    Mat3::from_diagonal(Vec3::new(
        s * (h2.y + h2.z),
        s * (h2.x + h2.z),
        s * (h2.x + h2.y),
    ))
}

#[test]
fn connect_pins_two_bodies_under_gravity() {
    // Two unit-mass bodies connected by a CONNECT constraint at
    // their respective origins. One is kinematic (anchored to the
    // world), the other free. Under gravity, the free body should
    // hang directly below the anchor.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(1.0, 5.0, 0.0)),
    );
    // Anchor in `anchor`'s body frame at (0, 0, 0).
    // Anchor in `free`'s body frame at (0, 0, 0). After projection
    // they should coincide → free's origin = anchor's origin.
    b.add_equality_connect(anchor, free, Vec3::ZERO, Vec3::ZERO);

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(0);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..1000 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    eprintln!(
        "connect: anchor at {:?}, free at {:?}, dist {dist:.4}",
        q[0].position, q[1].position
    );
    // CONNECT should pin them to ~zero distance.
    assert!(dist < 0.05, "CONNECT failed: bodies {:.3} m apart", dist);
}

/// Stage-7 DoD: humanoid hip ball-joint demo. A "pelvis" body is
/// kinematically anchored in space; a "thigh" body hangs from it via
/// a CONNECT equality constraint at the hip anchor point. The thigh
/// starts pinned (no initial constraint violation) with a small
/// angular kick. Under gravity it swings as a 3-DoF pendulum about
/// the hip anchor — exactly what a ball joint provides.
#[test]
fn humanoid_ball_joint_swings_under_gravity() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let pelvis = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    // Thigh COM starts directly below pelvis at (0, 4.5, 0), with
    // identity rotation. Hip anchor in thigh-local frame at +Y =
    // (0, 0.5, 0); world hip anchor = (0, 5, 0) = pelvis origin.
    // Constraint already satisfied (no initial violation), so
    // velocity recovery doesn't blow up.
    let thigh = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.083,
            })
            .with_pose(Transform::new(Vec3::new(0.0, 4.5, 0.0), Quat::IDENTITY)),
    );
    b.add_equality_connect(pelvis, thigh, Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0));

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Give the thigh an initial angular velocity about Z (kicks it
    // sideways from rest).
    s_a.body_qd.host_mut().unwrap()[1] = SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = XpbdSolver::new().with_iterations(16);

    let dt = 0.002;
    let mut min_x = Real::INFINITY;
    let mut max_x = -Real::INFINITY;
    for _ in 0..2000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.body_q.host().unwrap();
        let thigh_x = q[1].position.x;
        if thigh_x < min_x {
            min_x = thigh_x;
        }
        if thigh_x > max_x {
            max_x = thigh_x;
        }
    }

    let q = s_a.body_q.host().unwrap();
    let pelvis_pos = q[0].position;
    let thigh_xf = q[1];
    let hip_world = thigh_xf.transform_point(Vec3::new(0.0, 0.5, 0.0));
    let pin_err = (hip_world - pelvis_pos).length();
    eprintln!("ball joint demo: pin_err={pin_err:.4} thigh x range=[{min_x:.3}, {max_x:.3}]");
    assert!(pin_err < 0.05, "CONNECT failed to pin hip: |Δ|={pin_err}");
    // The thigh should have swung sideways enough to register.
    assert!(
        max_x - min_x > 0.05,
        "thigh didn't swing — x range = {:.3}",
        max_x - min_x
    );
    // Thigh COM must stay below pelvis (it's a pendulum).
    assert!(
        thigh_xf.position.y < pelvis_pos.y,
        "thigh ended above pelvis"
    );
}

#[test]
fn weld_keeps_orientation_aligned() {
    // Two bodies welded together. The first is kinematic at the
    // world origin with identity rotation. The second is dynamic,
    // initially translated and rotated. After settling, orientation
    // and position should both match.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::IDENTITY)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            })
            .with_pose(Transform::new(
                Vec3::new(0.5, 0.0, 0.0),
                Quat::from_rotation_z(0.5),
            )),
    );
    b.add_equality_weld(anchor, free, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY);

    // No gravity to make the test deterministic.
    let mut model = b.build().unwrap();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = XpbdSolver::new().with_iterations(20);

    for _ in 0..200 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    let dot = q[1].rotation.x * q[0].rotation.x
        + q[1].rotation.y * q[0].rotation.y
        + q[1].rotation.z * q[0].rotation.z
        + q[1].rotation.w * q[0].rotation.w;
    eprintln!("weld: dist={dist:.4} rot-dot={dot:.4}");
    assert!(dist < 0.05, "WELD position drifted: dist={dist}");
    assert!(dot.abs() > 0.95, "WELD orientation drifted: dot={dot}");
}

#[test]
fn four_bar_closed_loop_keeps_pivots_bounded_under_gravity() {
    // Procedural closed-chain gate: a four-bar linkage made only from
    // rigid bodies plus CONNECT equalities. The left and right ground
    // pivots are kinematic, while three dynamic rod bodies form a
    // closed loop. Gravity and an initial angular kick load the loop;
    // all pivot anchors must remain coincident and finite.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.04);

    let ground_left_pos = Vec3::new(0.0, 2.0, 0.0);
    let ground_right_pos = Vec3::new(1.2, 2.0, 0.0);
    let upper_left = Vec3::new(0.3, 2.519_615_242_270_663, 0.0);
    let upper_right = Vec3::new(0.9, 2.519_615_242_270_663, 0.0);

    let ground_left = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(ground_left_pos))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let ground_right = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(ground_right_pos))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );

    let rod_half_thickness = 0.035;
    let add_rod = |builder: &mut ModelBuilder, a: Vec3, c: Vec3| {
        let d = c - a;
        let len = d.length();
        let center = (a + c) * 0.5;
        let angle = d.y.atan2(d.x);
        let half = Vec3::new(0.5 * len, rod_half_thickness, rod_half_thickness);
        let body = builder.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.35,
                    com: Vec3::ZERO,
                    inertia: box_inertia(0.35, half),
                })
                .with_pose(Transform::new(center, Quat::from_rotation_z(angle))),
        );
        builder
            .add_shape_box(body, half, Transform::IDENTITY)
            .unwrap();
        (
            body,
            Vec3::new(-0.5 * len, 0.0, 0.0),
            Vec3::new(0.5 * len, 0.0, 0.0),
        )
    };

    let (left_link, left_a, left_b) = add_rod(&mut b, ground_left_pos, upper_left);
    let (top_link, top_a, top_b) = add_rod(&mut b, upper_left, upper_right);
    let (right_link, right_a, right_b) = add_rod(&mut b, upper_right, ground_right_pos);

    b.add_equality_connect(ground_left, left_link, Vec3::ZERO, left_a);
    b.add_equality_connect(left_link, top_link, left_b, top_a);
    b.add_equality_connect(top_link, right_link, top_b, right_a);
    b.add_equality_connect(right_link, ground_right, right_b, Vec3::ZERO);

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[top_link.index()] =
        SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.9));
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = XpbdSolver::new().with_iterations(24);
    let dt = 0.002;
    let mut max_err: Real = 0.0;
    let mut min_top_y: Real = Real::INFINITY;
    let mut max_top_y: Real = -Real::INFINITY;

    for _ in 0..1500 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let q = s_a.body_q.host().unwrap();
        let gl = q[ground_left.index()].transform_point(Vec3::ZERO);
        let gr = q[ground_right.index()].transform_point(Vec3::ZERO);
        let l0 = q[left_link.index()].transform_point(left_a);
        let l1 = q[left_link.index()].transform_point(left_b);
        let t0 = q[top_link.index()].transform_point(top_a);
        let t1 = q[top_link.index()].transform_point(top_b);
        let r0 = q[right_link.index()].transform_point(right_a);
        let r1 = q[right_link.index()].transform_point(right_b);
        max_err = max_err
            .max((gl - l0).length())
            .max((l1 - t0).length())
            .max((t1 - r0).length())
            .max((r1 - gr).length());
        let top_y = q[top_link.index()].position.y;
        min_top_y = min_top_y.min(top_y);
        max_top_y = max_top_y.max(top_y);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    for body in [left_link, top_link, right_link] {
        let pose = q[body.index()];
        let vel = qd[body.index()];
        assert!(
            pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
            "four-bar body {body:?} produced non-finite state: pose={pose:?}, vel={vel:?}"
        );
    }
    eprintln!(
        "four-bar: max pivot error={max_err:.4e}, top y range=[{min_top_y:.4}, {max_top_y:.4}]"
    );
    assert!(
        max_err < 0.035,
        "four-bar loop closure drifted too far: max pivot error {max_err}"
    );
    assert!(
        max_top_y - min_top_y > 0.01,
        "four-bar did not move under gravity/kick; top y range was only {}",
        max_top_y - min_top_y
    );
}

#[test]
fn double_four_bar_shared_link_keeps_multi_loop_closure_bounded() {
    // Tier 3.1 multi-loop gate: two adjacent four-bar loops share the
    // middle descending link and the middle ground pivot. The graph is
    // intentionally branched at the shared upper pivot, so constraint
    // projection must keep both loops closed at once rather than solving one
    // isolated four-bar.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let g0 = Vec3::new(0.0, 2.0, 0.0);
    let g1 = Vec3::new(1.0, 2.0, 0.0);
    let g2 = Vec3::new(2.0, 2.0, 0.0);
    let u0 = Vec3::new(0.28, 2.46, 0.0);
    let u1 = Vec3::new(1.0, 2.58, 0.0);
    let u2 = Vec3::new(1.72, 2.46, 0.0);

    let ground0 = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(g0))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let ground1 = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(g1))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let ground2 = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(g2))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );

    let rod_half_thickness = 0.03;
    let add_rod = |builder: &mut ModelBuilder, a: Vec3, c: Vec3| {
        let d = c - a;
        let len = d.length();
        let center = (a + c) * 0.5;
        let angle = d.y.atan2(d.x);
        let half = Vec3::new(0.5 * len, rod_half_thickness, rod_half_thickness);
        let body = builder.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.28,
                    com: Vec3::ZERO,
                    inertia: box_inertia(0.28, half),
                })
                .with_pose(Transform::new(center, Quat::from_rotation_z(angle))),
        );
        builder
            .add_shape_box(body, half, Transform::IDENTITY)
            .unwrap();
        (
            body,
            Vec3::new(-0.5 * len, 0.0, 0.0),
            Vec3::new(0.5 * len, 0.0, 0.0),
        )
    };

    let (left_riser, left_riser_a, left_riser_b) = add_rod(&mut b, g0, u0);
    let (left_top, left_top_a, left_top_b) = add_rod(&mut b, u0, u1);
    let (shared_middle, shared_middle_a, shared_middle_b) = add_rod(&mut b, u1, g1);
    let (right_top, right_top_a, right_top_b) = add_rod(&mut b, u1, u2);
    let (right_riser, right_riser_a, right_riser_b) = add_rod(&mut b, u2, g2);

    b.add_equality_connect(ground0, left_riser, Vec3::ZERO, left_riser_a);
    b.add_equality_connect(left_riser, left_top, left_riser_b, left_top_a);
    b.add_equality_connect(left_top, shared_middle, left_top_b, shared_middle_a);
    b.add_equality_connect(shared_middle, ground1, shared_middle_b, Vec3::ZERO);
    b.add_equality_connect(shared_middle, right_top, shared_middle_a, right_top_a);
    b.add_equality_connect(right_top, right_riser, right_top_b, right_riser_a);
    b.add_equality_connect(right_riser, ground2, right_riser_b, Vec3::ZERO);

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[left_top.index()] =
        SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.75));
    s_a.body_qd.host_mut().unwrap()[right_top.index()] =
        SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -0.65));
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = XpbdSolver::new().with_iterations(28);
    let dt: Real = 0.002;
    let mut max_err: Real = 0.0;
    let mut max_speed: Real = 0.0;
    let mut min_shared_y: Real = Real::INFINITY;
    let mut max_shared_y: Real = Real::NEG_INFINITY;

    for _ in 0..1800 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let q = s_a.body_q.host().unwrap();
        let qd = s_a.body_qd.host().unwrap();
        let p_g0 = q[ground0.index()].transform_point(Vec3::ZERO);
        let p_g1 = q[ground1.index()].transform_point(Vec3::ZERO);
        let p_g2 = q[ground2.index()].transform_point(Vec3::ZERO);
        let p_lr_a = q[left_riser.index()].transform_point(left_riser_a);
        let p_lr_b = q[left_riser.index()].transform_point(left_riser_b);
        let p_lt_a = q[left_top.index()].transform_point(left_top_a);
        let p_lt_b = q[left_top.index()].transform_point(left_top_b);
        let p_sm_a = q[shared_middle.index()].transform_point(shared_middle_a);
        let p_sm_b = q[shared_middle.index()].transform_point(shared_middle_b);
        let p_rt_a = q[right_top.index()].transform_point(right_top_a);
        let p_rt_b = q[right_top.index()].transform_point(right_top_b);
        let p_rr_a = q[right_riser.index()].transform_point(right_riser_a);
        let p_rr_b = q[right_riser.index()].transform_point(right_riser_b);
        max_err = max_err
            .max((p_g0 - p_lr_a).length())
            .max((p_lr_b - p_lt_a).length())
            .max((p_lt_b - p_sm_a).length())
            .max((p_sm_b - p_g1).length())
            .max((p_sm_a - p_rt_a).length())
            .max((p_rt_b - p_rr_a).length())
            .max((p_rr_b - p_g2).length());
        for body in [left_riser, left_top, shared_middle, right_top, right_riser] {
            max_speed = max_speed.max(qd[body.index()].linear.length());
        }
        let shared_y = q[shared_middle.index()].position.y;
        min_shared_y = min_shared_y.min(shared_y);
        max_shared_y = max_shared_y.max(shared_y);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    for body in [left_riser, left_top, shared_middle, right_top, right_riser] {
        let pose = q[body.index()];
        let vel = qd[body.index()];
        assert!(
            pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
            "multi-loop body {body:?} produced non-finite state: pose={pose:?}, vel={vel:?}"
        );
        assert!(
            pose.position.y > 1.2 && pose.position.y < 3.0,
            "multi-loop body {body:?} escaped expected height band: y={}",
            pose.position.y
        );
    }
    eprintln!(
        "double four-bar: max pivot error={max_err:.4e}, max speed={max_speed:.4}, shared y range=[{min_shared_y:.4}, {max_shared_y:.4}]"
    );
    assert!(
        max_err < 0.04,
        "double four-bar loop closure drifted too far: max pivot error {max_err}"
    );
    assert!(
        max_speed < 8.0,
        "double four-bar generated unstable speeds: max |v|={max_speed}"
    );
    assert!(
        max_shared_y - min_shared_y > 0.008,
        "shared link did not respond to gravity/kicks; y range was only {}",
        max_shared_y - min_shared_y
    );
}

#[test]
fn equality_connect_preserves_restitution_bounce_velocity() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    b.set_linear_damping(0.0);
    b.set_default_friction(0.0).unwrap();
    b.set_default_restitution(1.0).unwrap();
    let half = Vec3::splat(0.15);
    let inertia = box_inertia(1.0, half);
    let bottom = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(Vec3::new(0.0, half.y + 0.05, 0.0)),
    );
    b.add_shape_box(bottom, half, Transform::IDENTITY).unwrap();
    let top = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(Vec3::new(0.0, 3.0 * half.y + 0.05, 0.0)),
    );
    b.add_shape_box(top, half, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.add_equality_connect(
        bottom,
        top,
        Vec3::new(0.0, half.y, 0.0),
        Vec3::new(0.0, -half.y, 0.0),
    );

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(256);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt: Real = 1.0 / 240.0;
    let mut max_contacts = 0_u32;
    for _ in 0..30 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    assert!(
        max_contacts > 0,
        "test did not exercise restitution contact"
    );
    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let bottom_top = q[bottom.index()].transform_point(Vec3::new(0.0, half.y, 0.0));
    let top_bottom = q[top.index()].transform_point(Vec3::new(0.0, -half.y, 0.0));
    let closure_err = (bottom_top - top_bottom).length();
    let min_y = q[bottom.index()].position.y.min(q[top.index()].position.y);
    eprintln!(
        "equality restitution: max_contacts={max_contacts}, closure_err={closure_err:.4e}, bottom_vy={:.5}, top_vy={:.5}, min_y={min_y:.5}",
        qd[bottom.index()].linear.y,
        qd[top.index()].linear.y,
    );
    assert!(
        closure_err < 0.03,
        "CONNECT closure drifted while restitution was active: {closure_err}"
    );
    assert!(
        min_y > half.y - 0.02,
        "connected stack tunneled through floor during restitution: min_y={min_y}"
    );
    assert!(
        qd[bottom.index()].linear.y > 0.10 && qd[top.index()].linear.y > 0.10,
        "restitution bounce velocity was lost after equality recovery: bottom_vy={}, top_vy={}",
        qd[bottom.index()].linear.y,
        qd[top.index()].linear.y,
    );
}
