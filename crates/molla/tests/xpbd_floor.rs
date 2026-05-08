//! Stage 4.0 sanity test: drop a sphere on a static plane and check
//! that it settles above the plane after enough simulation time.

use molla::geometry::collide;
use molla::prelude::*;

#[test]
fn sphere_settles_on_plane() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();

    // Static plane at y = 0 with +Y normal, attached to the world.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();

    let mut solver = XpbdSolver::new();
    let dt = 0.005;

    // Run 4 seconds — generous enough to bounce-and-settle.
    for _ in 0..800 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let y = q[0].position.y;
    let v = qd[0].linear.y;

    // Sphere of radius 0.5 should sit at y >= 0.5 (with slight slack
    // tolerance from XPBD's contact slop).
    assert!(
        (0.5 - 1e-3..0.6).contains(&y),
        "sphere settled at y={y}, expected ~0.5"
    );
    // And its vertical velocity should be near zero.
    assert!(v.abs() < 0.1, "sphere still moving, vy={v}");
}

#[test]
fn two_spheres_dont_overlap() {
    // Two spheres dropped side-by-side; check they don't penetrate.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(-0.4, 5.0, 0.0)),
    );
    b.add_shape_sphere(a, 0.5, Transform::IDENTITY).unwrap();

    let c = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.4, 5.0, 0.0)),
    );
    b.add_shape_sphere(c, 0.5, Transform::IDENTITY).unwrap();

    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..800 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let dist = (q[0].position - q[1].position).length();
    assert!(
        dist >= 1.0 - 5e-3,
        "spheres overlap: distance {dist}, expected >= 1.0"
    );
}
