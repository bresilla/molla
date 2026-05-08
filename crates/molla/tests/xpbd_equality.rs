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
