//! Stage 6.x — cylinder integration tests using the GJK+EPA fallback.
//! Cylinders use no closed-form narrow phase in molla; every contact
//! pair flows through `gjk` → `epa`.

use molla::geometry::CollisionPipeline;
use molla::prelude::*;

fn cylinder_inertia(mass: Real, half_height: Real, radius: Real) -> Mat3 {
    let r2 = radius * radius;
    let h_full = 2.0 * half_height;
    let h2 = h_full * h_full;
    let i_y = 0.5 * mass * r2;
    let i_perp = mass * (h2 / 12.0 + r2 / 4.0);
    Mat3::from_diagonal(Vec3::new(i_perp, i_y, i_perp))
}

#[test]
fn cylinder_settles_on_plane() {
    // Vertical cylinder dropped onto Y=0 plane.
    let half_h = 0.4;
    let radius = 0.3;
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: cylinder_inertia(1.0, half_h, radius),
            })
            .with_position(Vec3::new(0.0, 4.0, 0.0)),
    );
    b.set_default_friction(0.6).unwrap();
    b.set_default_restitution(0.0).unwrap();
    b.add_shape_cylinder(body, half_h, radius, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY).unwrap();

    let mut model = b.build().unwrap();
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
    eprintln!("vertical cylinder: y={y:.4} |v|={v:.4}");
    // Vertical cylinder resting flat on plane → centre y = half_h.
    assert!(
        (y - half_h).abs() < 0.05,
        "cylinder didn't settle: y={y}, expected ~{half_h}"
    );
    assert!(v < 0.2, "cylinder still moving: |v|={v}");
}

#[test]
fn capsule_on_box_via_gjk() {
    // Static box on the floor, capsule dropped on top. Capsule-box
    // has no closed-form path → uses GJK+EPA.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let plate = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_box(plate, Vec3::new(2.0, 0.5, 2.0), Transform::IDENTITY)
        .unwrap();

    let half_h = 0.3;
    let radius = 0.2;
    let cap = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.05,
            })
            .with_pose(Transform::new(
                Vec3::new(0.0, 4.0, 0.0),
                Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
            )),
    );
    b.set_default_friction(0.5).unwrap();
    b.set_default_restitution(0.1).unwrap();
    b.add_shape_capsule(cap, half_h, radius, Transform::IDENTITY).unwrap();

    let model = b.build().unwrap();

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
    eprintln!("capsule-on-box: y={y:.4}");
    // Capsule lying on box-top (y=1.0) at radius height: y ≈ 1.0 + r.
    assert!(
        (y - (1.0 + radius)).abs() < 0.1,
        "capsule didn't settle on box: y={y}"
    );
}
