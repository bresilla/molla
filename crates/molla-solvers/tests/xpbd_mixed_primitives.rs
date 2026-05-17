//! Tier-4 hard-contact CPU stress for mixed primitive shapes.
//!
//! Individual primitive tests cover exact settle heights. This one is
//! intentionally broader: a mixed set of supported rigid primitives
//! must remain finite and above the floor when solved together.

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, Contacts, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn inertia(mass: Real) -> SpatialInertia {
    SpatialInertia {
        mass,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(0.03)),
    }
}

#[test]
fn mixed_primitive_drop_stays_finite_and_above_floor() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.12);
    b.set_linear_damping(0.05);
    b.set_default_friction(0.7).unwrap();
    b.set_default_restitution(0.0).unwrap();

    let mk_body = |b: &mut ModelBuilder, x: Real, y: Real, rot: Quat| {
        b.add_body(
            BodyParams::new()
                .with_spatial_inertia(inertia(1.0))
                .with_pose(Transform::new(Vec3::new(x, y, 0.0), rot)),
        )
    };

    let sphere = mk_body(&mut b, -1.25, 2.2, Quat::IDENTITY);
    b.add_shape_sphere(sphere, 0.18, Transform::IDENTITY)
        .unwrap();

    let bx = mk_body(&mut b, -0.75, 2.8, Quat::from_axis_angle(Vec3::Z, 0.25));
    b.add_shape_box(bx, Vec3::new(0.18, 0.12, 0.16), Transform::IDENTITY)
        .unwrap();

    let capsule = mk_body(
        &mut b,
        -0.25,
        3.4,
        Quat::from_axis_angle(Vec3::Z, std::f64::consts::FRAC_PI_2),
    );
    b.add_shape_capsule(capsule, 0.22, 0.10, Transform::IDENTITY)
        .unwrap();

    let cylinder = mk_body(&mut b, 0.25, 4.0, Quat::from_axis_angle(Vec3::X, 0.35));
    b.add_shape_cylinder(cylinder, 0.20, 0.12, Transform::IDENTITY)
        .unwrap();

    let cone = mk_body(&mut b, 0.75, 4.6, Quat::from_axis_angle(Vec3::Z, -0.35));
    b.add_shape_cone(cone, 0.22, 0.14, Transform::IDENTITY)
        .unwrap();

    let ellipsoid = mk_body(&mut b, 1.25, 5.2, Quat::from_axis_angle(Vec3::X, 0.45));
    b.add_shape_ellipsoid(ellipsoid, Vec3::new(0.22, 0.14, 0.12), Transform::IDENTITY)
        .unwrap();

    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = b.build().unwrap();
    model.contact_margin = 0.025;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(512);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(10);
    let dt: Real = 1.0 / 240.0;
    let mut max_contacts = 0usize;

    for _ in 0..720 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count as usize);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    assert!(
        max_contacts > 0,
        "mixed primitive scene never generated contacts"
    );
    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    for (i, pose) in q.iter().enumerate() {
        assert!(
            pose.position.is_finite(),
            "body {i} produced non-finite position {:?}",
            pose.position
        );
        assert!(
            pose.position.y > -0.20,
            "body {i} sank below floor: y={}",
            pose.position.y
        );
        assert!(
            qd[i].linear.length() < 20.0 && qd[i].angular.length() < 100.0,
            "body {i} velocity exploded: qd={:?}",
            qd[i]
        );
    }
}

#[test]
fn procedural_socket_bolt_dense_contact_stays_finite_and_caged() {
    // Tier 4.1 nut/bolt-like dense contact without assets: a dynamic
    // cylinder "bolt" falls into a six-wall kinematic socket. The socket is
    // intentionally slightly oversized and the bolt starts off-centre so the
    // solver exercises simultaneous cylinder/floor and cylinder/wall contacts.
    let bolt_radius: Real = 0.24;
    let bolt_half_height: Real = 0.32;
    let socket_radius: Real = 0.34;
    let wall_half = Vec3::new(0.24, 0.42, 0.045);

    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.12);
    b.set_linear_damping(0.08);
    b.set_default_friction(0.85).unwrap();
    b.set_default_restitution(0.0).unwrap();

    for i in 0..6 {
        let angle = i as Real * std::f64::consts::TAU / 6.0;
        let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
        let wall = b.add_body(
            BodyParams::new()
                .with_mass(0.0)
                .with_pose(Transform::new(
                    radial * socket_radius + Vec3::new(0.0, wall_half.y, 0.0),
                    Quat::from_axis_angle(Vec3::Y, -angle),
                ))
                .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
        );
        b.add_shape_box(wall, wall_half, Transform::IDENTITY)
            .unwrap();
    }

    let bolt = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(inertia(1.0))
            .with_pose(Transform::new(
                Vec3::new(0.09, 1.55, -0.06),
                Quat::from_axis_angle(Vec3::Z, 0.18),
            )),
    );
    b.add_shape_cylinder(bolt, bolt_half_height, bolt_radius, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.contact_margin = 0.02;
    let mut s_a = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[bolt.index()].linear = Vec3::new(-0.20, 0.0, 0.15);
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(14);
    let dt: Real = 1.0 / 240.0;
    let mut max_contacts = 0_u32;
    let mut max_lateral: Real = 0.0;
    for _ in 0..900 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let p = s_a.body_q.host().unwrap()[bolt.index()].position;
        max_lateral = max_lateral.max(Vec3::new(p.x, 0.0, p.z).length());
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let pose = q[bolt.index()];
    let vel = qd[bolt.index()];
    let lateral = Vec3::new(pose.position.x, 0.0, pose.position.z).length();
    eprintln!(
        "socket/bolt dense contact: max_contacts={max_contacts}, final_pos={:?}, lateral={lateral:.4}, max_lateral={max_lateral:.4}, |v|={:.4}, |w|={:.4}",
        pose.position,
        vel.linear.length(),
        vel.angular.length()
    );
    assert!(
        max_contacts >= 5,
        "socket/bolt did not exercise dense contact: max_contacts={max_contacts}"
    );
    assert!(
        pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
        "socket/bolt produced non-finite state: pose={pose:?}, vel={vel:?}"
    );
    assert!(
        pose.position.y > bolt_half_height - 0.06,
        "bolt tunneled through socket floor plane: y={}",
        pose.position.y
    );
    assert!(
        max_lateral < socket_radius + bolt_radius,
        "bolt escaped the procedural socket: max_lateral={max_lateral}"
    );
    assert!(
        lateral < socket_radius + 0.08,
        "bolt did not remain caged by socket walls: lateral={lateral}"
    );
    assert!(
        vel.linear.length() < 1.2 && vel.angular.length() < 12.0,
        "socket/bolt did not settle enough: qd={vel:?}"
    );
}
