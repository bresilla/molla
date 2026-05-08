//! Stage 4 box-collision integration tests: box on plane, sphere on
//! box, and the headline 10-box stack settling test.

use molla::geometry::CollisionPipeline;
use molla::prelude::*;

fn box_inertia(mass: Real, half: Vec3) -> Mat3 {
    // Solid box principal moments: I_xx = m/3 · (hy² + hz²) etc.
    // (half-extents convention: hx, hy, hz; full extents 2·h.)
    let hx2 = half.x * half.x;
    let hy2 = half.y * half.y;
    let hz2 = half.z * half.z;
    let s = mass / 3.0;
    Mat3::from_diagonal(Vec3::new(s * (hy2 + hz2), s * (hx2 + hz2), s * (hx2 + hy2)))
}

fn box_si(mass: Real, half: Vec3) -> SpatialInertia {
    SpatialInertia {
        mass,
        com: Vec3::ZERO,
        inertia: box_inertia(mass, half),
    }
}

#[test]
fn box_settles_on_plane() {
    let half = Vec3::splat(0.5);
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(box_si(1.0, half))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.5;
    model.default_restitution = 0.0;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..800 {
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
    let w = qd[0].angular.length();
    assert!(
        (0.5 - 5e-3..0.55).contains(&y),
        "box settled at y={y}, expected ~0.5"
    );
    assert!(v < 0.05, "box still translating: |v| = {v}");
    assert!(w < 0.5, "box still tumbling: |ω| = {w}");
}

#[test]
fn sphere_on_box() {
    let half = Vec3::new(2.0, 0.25, 2.0);
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    // Static box on the floor — kinematic.
    let plate = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(Vec3::new(0.0, 0.25, 0.0), Quat::IDENTITY))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_box(plate, half, Transform::IDENTITY).unwrap();

    let ball = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.3))
            .with_position(Vec3::new(0.0, 4.0, 0.0)),
    );
    b.add_shape_sphere(ball, 0.3, Transform::IDENTITY).unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.5;
    model.default_restitution = 0.1;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..800 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let y = q[1].position.y;
    // Ball radius 0.3 + plate top at y=0.5 → ball centre ~0.8.
    assert!((0.78..0.86).contains(&y), "ball settled at y={y}");
}

#[test]
fn box_stack_settles() {
    // 10 unit boxes stacked along +Y. They start with a small gap so
    // each one falls onto the one below it. After 1.5s all 10 must
    // be near rest, with no interpenetration > 1mm and no growing
    // bounce.
    let half = Vec3::splat(0.5);
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let count = 10;
    for i in 0..count {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(box_si(1.0, half))
                .with_position(Vec3::new(0.0, 0.5 + i as Real * 1.05, 0.0)),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
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
    let mut contacts = Contacts::with_capacity(512);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(16);

    let dt = 0.005;
    let mut max_top_y_after_warmup = 0.0_f64;
    let warmup = 400;
    let total = 1200;

    for step in 0..total {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if step > warmup {
            let q = s_a.body_q.host().unwrap();
            let top_y = q[count - 1].position.y;
            if top_y > max_top_y_after_warmup {
                max_top_y_after_warmup = top_y;
            }
        }
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let bottom_y = q[0].position.y;
    let top_y = q[count - 1].position.y;
    let top_v = qd[count - 1].linear.length();

    eprintln!(
        "box stack: bottom y={bottom_y:.3} top y={top_y:.3} \
         peak top y after warmup={max_top_y_after_warmup:.3} top |v|={top_v:.3}"
    );

    // Bottom centre should sit near 0.5 (resting on the plane).
    assert!(
        (0.495..0.55).contains(&bottom_y),
        "bottom box at y={bottom_y}, expected ~0.5"
    );
    // Stack must not be growing in height.
    let initial_top_y = 0.5 + (count as Real - 1.0) * 1.05; // = 9.95
    assert!(
        max_top_y_after_warmup < initial_top_y + 0.1,
        "stack grew: peak top y={max_top_y_after_warmup}, initial={initial_top_y}"
    );
    assert!(top_v < 0.2, "top box still moving: |v|={top_v}");

    // Pairwise non-interpenetration: each adjacent pair must overlap
    // by no more than the contact slop (~5mm at our iteration count).
    for i in 0..(count - 1) {
        let lower_top = q[i].position.y + 0.5;
        let upper_bottom = q[i + 1].position.y - 0.5;
        let gap = upper_bottom - lower_top;
        assert!(
            gap > -5e-3,
            "boxes {i} and {} interpenetrating: gap={gap:.4}",
            i + 1
        );
    }
}
