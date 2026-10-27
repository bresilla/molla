//! Stage 6.1 — capsule integration tests under XPBD with the
//! closed-form capsule narrow phase.

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

/// Solid capsule principal moments (uniform density), oriented along
/// local +Y. Approximation: treat as a cylinder of length
/// `2·half_height` plus two hemispheres. For our small test cases the
/// `cylinder + 2 hemispheres` formula isn't crucial — we just need a
/// reasonable inertia tensor so the body responds to torque.
fn capsule_inertia(mass: Real, half_height: Real, radius: Real) -> Mat3 {
    let r2 = radius * radius;
    let h2 = half_height * half_height;
    // Treat as a cylinder + 2 hemispheres for the principal axes.
    // Cylinder mass fraction: m_c = m · (cyl_volume / total_volume).
    // For simplicity we just use a cylinder approximation here —
    // tests don't exercise inertia precision.
    let i_y = 0.5 * mass * r2; // about capsule axis
    let i_perp = mass * (h2 / 3.0 + r2 / 4.0);
    Mat3::from_diagonal(Vec3::new(i_perp, i_y, i_perp))
}

#[test]
fn capsule_settles_on_plane() {
    // Drop a capsule lying on its side onto a horizontal plane.
    // The capsule axis is rotated from +Y to +X via a Z rotation
    // (-π/2). End-cap centres are along ±X. Both should rest on
    // the plane.
    let half_h = 0.5;
    let radius = 0.2;
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: capsule_inertia(1.0, half_h, radius),
            })
            .with_pose(Transform::new(
                Vec3::new(0.0, 4.0, 0.0),
                Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
            )),
    );
    b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.6;
    model.default_restitution = 0.0;
    model.contact_margin = 0.02;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
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
    let qd = s_a.body_qd.host().unwrap();
    let y = q[0].position.y;
    let v = qd[0].linear.length();
    eprintln!("capsule lying: y={y:.4} |v|={v:.4}");
    // Capsule centre when resting on its side: y = radius.
    assert!(
        (y - radius).abs() < 0.05,
        "capsule didn't settle: y={y}, expected ~{radius}"
    );
    assert!(v < 0.2, "capsule still moving: |v|={v}");
}

#[test]
fn capsule_stack_two() {
    // Two horizontal capsules: one rests on the plane, the other
    // stacks on top. Both axes along +X (rotated −π/2 about Z).
    let half_h = 0.5;
    let radius = 0.2;
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    for i in 0..2 {
        let y = radius + i as Real * (2.0 * radius + 0.05); // small gap
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.0,
                    com: Vec3::ZERO,
                    inertia: capsule_inertia(1.0, half_h, radius),
                })
                .with_pose(Transform::new(
                    Vec3::new(0.0, y + 1.0, 0.0), // start above settle point
                    Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
                )),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.8;
    model.default_restitution = 0.0;
    model.contact_margin = 0.02;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(16);

    for _ in 0..1500 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let y0 = q[0].position.y;
    let y1 = q[1].position.y;
    let v0 = qd[0].linear.length();
    let v1 = qd[1].linear.length();
    eprintln!("capsule stack: y0={y0:.4} y1={y1:.4} v0={v0:.4} v1={v1:.4}");
    eprintln!("capsule stack poses={q:?} velocities={qd:?}");
    // Bottom capsule resting on plane → y ≈ radius.
    assert!(
        (y0 - radius).abs() < 0.05,
        "bottom capsule at y={y0}, expected ~{radius}"
    );
    // Top capsule stacked on bottom → y ≈ 3·radius (touching).
    assert!(
        (y1 - 3.0 * radius).abs() < 0.10,
        "top capsule at y={y1}, expected ~{}",
        3.0 * radius
    );
    assert!(v0 < 0.2, "bottom capsule moving: |v|={v0}");
    assert!(v1 < 0.3, "top capsule moving: |v|={v1}");
}

#[test]
fn capsule_capsule_parallel_no_overlap_after_settling() {
    // Two side-by-side capsules — both settle on the plane next to
    // each other and end up resting without interpenetrating.
    let half_h = 0.5;
    let radius = 0.2;
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    for &x in &[-0.3, 0.3] {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.0,
                    com: Vec3::ZERO,
                    inertia: capsule_inertia(1.0, half_h, radius),
                })
                .with_pose(Transform::new(
                    Vec3::new(x, 3.0, 0.0),
                    Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
                )),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.8;
    model.default_restitution = 0.0;
    model.contact_margin = 0.02;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
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
    let dx = (q[0].position.x - q[1].position.x).abs();
    eprintln!("two parallel capsules: dx={dx:.4}");
    // Distance between centres on X must be at least 2·radius
    // (allowing 5mm contact slop).
    assert!(
        dx >= 2.0 * radius - 5e-3,
        "capsules interpenetrating: dx={dx}, expected ≥ {}",
        2.0 * radius
    );
}
