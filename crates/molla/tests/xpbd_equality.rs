//! Stage 7 — equality constraints (CONNECT / WELD).

use molla::geometry::CollisionPipeline;
use molla::prelude::*;

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
    assert!(
        dist < 0.05,
        "CONNECT failed: bodies {:.3} m apart",
        dist
    );
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
            .with_pose(Transform::new(
                Vec3::new(0.0, 4.5, 0.0),
                Quat::IDENTITY,
            )),
    );
    b.add_equality_connect(pelvis, thigh, Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0));

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Give the thigh an initial angular velocity about Z (kicks it
    // sideways from rest).
    s_a.body_qd.host_mut().unwrap()[1] =
        SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
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
    eprintln!(
        "ball joint demo: pin_err={pin_err:.4} thigh x range=[{min_x:.3}, {max_x:.3}]"
    );
    assert!(
        pin_err < 0.05,
        "CONNECT failed to pin hip: |Δ|={pin_err}"
    );
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
