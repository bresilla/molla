//! Rolling + torsional friction tests. Both are CPU-only for now —
//! GPU XPBD will pick them up when the WGSL kernel is updated.
//!
//! Coulomb tangential friction already damps any angular velocity
//! whose contact-point projection is non-zero (via the lever-arm
//! effect: tangential impulse at offset `r_contact` produces a
//! torque about the COM). The tests below isolate rolling /
//! torsional damping by **comparing two simulations** that differ
//! only in the friction-mode coefficient — anything that's down to
//! Coulomb appears in both runs and cancels out.

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, SpatialInertia, SpatialVector, Transform, Vec3 as V3};
use molla_sim::{BodyFlags, BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

fn box_inertia(mass: f64, half: V3) -> Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    Mat3::from_diagonal(V3::new(
        s * (h2.y + h2.z),
        s * (h2.x + h2.z),
        s * (h2.x + h2.y),
    ))
}

fn build_sphere_on_plane(coulomb: f64, rolling: f64, torsional: f64) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    let radius = 0.1_f64;
    let mass = 1.0_f64;
    let inertia = sphere_inertia(mass, radius);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia,
            })
            .with_position(V3::new(0.0, radius, 0.0)),
    );
    let sphere = b
        .add_shape_sphere(body, radius, Transform::IDENTITY)
        .unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(sphere, coulomb, 0.0).unwrap();
    b.set_shape_material(plane, coulomb, 0.0).unwrap();
    b.set_shape_friction_modes(sphere, rolling, torsional)
        .unwrap();
    b.build().unwrap()
}

fn build_box_on_plane(friction: f64) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.02);
    let half = V3::new(0.16, 0.10, 0.14);
    let mass = 1.0_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia: box_inertia(mass, half),
            })
            .with_position(V3::new(0.0, half.y, 0.0)),
    );
    let box_shape = b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(box_shape, friction, 0.0).unwrap();
    b.set_shape_material(plane, friction, 0.0).unwrap();
    b.build().unwrap()
}

fn build_box_on_inclined_plane(friction: f64, angle_rad: f64) -> (Model, usize, V3) {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.05);
    let half = V3::new(0.16, 0.10, 0.14);
    let mass = 1.0_f64;
    let normal = V3::new(-angle_rad.sin(), angle_rad.cos(), 0.0).normalize();
    let downhill = V3::new(-angle_rad.cos(), -angle_rad.sin(), 0.0).normalize();
    let support = half.x * normal.x.abs() + half.y * normal.y.abs() + half.z * normal.z.abs();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia: box_inertia(mass, half),
            })
            .with_position(normal * (support + 0.005)),
    );
    let box_shape = b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, normal, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(box_shape, friction, 0.0).unwrap();
    b.set_shape_material(plane, friction, 0.0).unwrap();
    (b.build().unwrap(), body.0 as usize, downhill)
}

fn run_steady(model: &Model, w0: V3, n_steps: usize) -> V3 {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[0] = SpatialVector::new(V3::ZERO, w0);
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..n_steps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.body_qd.host().unwrap()[0].angular
}

fn run_sliding_body(model: &Model, v0: V3, n_steps: usize) -> (f64, f64) {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[0] = SpatialVector::new(v0, V3::ZERO);
    let x0 = s_a.body_q.host().unwrap()[0].position.x;
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..n_steps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    (q[0].position.x - x0, qd[0].linear.x)
}

fn run_incline_box(friction: f64) -> (f64, f64, u32) {
    let angle = 12.0_f64.to_radians();
    let (model, body_idx, downhill) = build_box_on_inclined_plane(friction, angle);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let p0 = s_a.body_q.host().unwrap()[body_idx].position;
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(12);
    let dt = 0.005_f64;
    let mut max_contacts = 0_u32;
    for _ in 0..360 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let displacement = (q[body_idx].position - p0).dot(downhill);
    let downhill_speed = qd[body_idx].linear.dot(downhill);
    (displacement, downhill_speed, max_contacts)
}

fn build_connected_sphere_pair_on_plane(friction: f64) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    let radius = 0.1_f64;
    let mass = 1.0_f64;
    let inertia = sphere_inertia(mass, radius);
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia,
            })
            .with_position(V3::new(-0.15, radius, 0.0)),
    );
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia,
            })
            .with_position(V3::new(0.15, radius, 0.0)),
    );
    let sphere_a = b
        .add_shape_sphere(body_a, radius, Transform::IDENTITY)
        .unwrap();
    let sphere_b = b
        .add_shape_sphere(body_b, radius, Transform::IDENTITY)
        .unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(sphere_a, friction, 0.0).unwrap();
    b.set_shape_material(sphere_b, friction, 0.0).unwrap();
    b.set_shape_material(plane, friction, 0.0).unwrap();

    // The connect anchors coincide at world origin in the initial state:
    // body_a local +X anchor and body_b local -X anchor form a dumbbell
    // resting on the plane.
    b.add_equality_connect(
        body_a,
        body_b,
        V3::new(0.15, 0.0, 0.0),
        V3::new(-0.15, 0.0, 0.0),
    );
    b.build().unwrap()
}

fn build_sphere_on_kinematic_conveyor(friction: f64) -> (Model, usize) {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    let belt = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_position(V3::new(0.0, -0.05, 0.0))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let sphere = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(V3::new(0.0, 0.095, 0.0)),
    );
    let belt_shape = b
        .add_shape_box(belt, V3::new(2.0, 0.05, 2.0), Transform::IDENTITY)
        .unwrap();
    let sphere_shape = b
        .add_shape_sphere(sphere, 0.1, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(belt_shape, friction, 0.0).unwrap();
    b.set_shape_material(sphere_shape, friction, 0.0).unwrap();
    (b.build().unwrap(), sphere.0 as usize)
}

/// With Coulomb = 0, μ_rolling > 0 is the ONLY thing that can damp
/// horizontal-axis spin. Without rolling friction the sphere spins
/// forever (Coulomb is the only existing damping path; we've zeroed it).
#[test]
fn rolling_friction_damps_horizontal_axis_spin_when_coulomb_zero() {
    let m_zero = build_sphere_on_plane(0.0, 0.0, 0.0);
    let m_with = build_sphere_on_plane(0.0, 0.5, 0.0);
    let w_zero = run_steady(&m_zero, V3::new(10.0, 0.0, 0.0), 60);
    let w_with = run_steady(&m_with, V3::new(10.0, 0.0, 0.0), 60);
    eprintln!(
        "horizontal spin |ω| after 0.3 s w/ Coulomb=0: roll=0 → {:.4}, roll=0.5 → {:.4}",
        w_zero.length(),
        w_with.length()
    );
    assert!(
        w_zero.length() > 9.5,
        "no Coulomb + no rolling: should keep spin, got |ω|={}",
        w_zero.length()
    );
    assert!(
        w_with.length() < w_zero.length() - 1.0,
        "rolling friction did nothing: zero={}, with={}",
        w_zero.length(),
        w_with.length()
    );
}

/// μ_torsional > 0 damps normal-axis spin. Coulomb has no purchase
/// on this DOF (the contact-point projection of pure normal-axis
/// rotation is zero), so this test stays clean even with non-zero
/// Coulomb friction.
#[test]
fn torsional_friction_damps_spin_along_normal() {
    let m_zero = build_sphere_on_plane(0.7, 0.0, 0.0);
    let m_with = build_sphere_on_plane(0.7, 0.0, 0.5);
    let w_zero = run_steady(&m_zero, V3::new(0.0, 10.0, 0.0), 60);
    let w_with = run_steady(&m_with, V3::new(0.0, 10.0, 0.0), 60);
    eprintln!(
        "normal-axis spin |ω.y| after 0.3 s: μ_tors=0 → {:.4}, μ_tors=0.5 → {:.4}",
        w_zero.y.abs(),
        w_with.y.abs()
    );
    assert!(
        w_zero.y.abs() > 9.5,
        "Coulomb shouldn't damp normal-axis spin, but |ω.y|={}",
        w_zero.y.abs()
    );
    assert!(
        w_with.y.abs() < w_zero.y.abs() - 0.3,
        "torsional friction not stronger than μ=0 baseline"
    );
}

#[test]
fn coulomb_friction_arrests_sliding_box_more_than_zero_friction() {
    let m_zero = build_box_on_plane(0.0);
    let m_high = build_box_on_plane(0.9);
    let (dx_zero, vx_zero) = run_sliding_body(&m_zero, V3::new(1.0, 0.0, 0.0), 300);
    let (dx_high, vx_high) = run_sliding_body(&m_high, V3::new(1.0, 0.0, 0.0), 300);
    eprintln!(
        "sliding box: μ=0 dx={dx_zero:.4}, vx={vx_zero:.4}; μ=0.9 dx={dx_high:.4}, vx={vx_high:.4}"
    );
    assert!(
        dx_zero > 1.20 && vx_zero > 0.80,
        "zero-friction sphere should keep sliding, got dx={dx_zero}, vx={vx_zero}"
    );
    assert!(
        dx_high < dx_zero * 0.90,
        "high Coulomb friction did not reduce sliding distance enough: μ=0 dx={dx_zero}, μ=0.9 dx={dx_high}"
    );
    assert!(
        vx_high.abs() < vx_zero.abs() * 0.85,
        "high Coulomb friction did not reduce tangent velocity enough: μ=0 vx={vx_zero}, μ=0.9 vx={vx_high}"
    );
}

#[test]
fn static_friction_holds_box_on_subcritical_incline() {
    let (dx_free, speed_free, contacts_free) = run_incline_box(0.0);
    let (dx_stick, speed_stick, contacts_stick) = run_incline_box(0.8);
    eprintln!(
        "inclined box static friction: μ=0 dx={dx_free:.4}, v={speed_free:.4}, contacts={contacts_free}; \
         μ=0.8 dx={dx_stick:.4}, v={speed_stick:.4}, contacts={contacts_stick}"
    );
    assert!(
        contacts_free > 0 && contacts_stick > 0,
        "incline test did not exercise contacts: free={contacts_free}, stick={contacts_stick}"
    );
    assert!(
        dx_free > 0.20,
        "frictionless box should slide down the incline, got dx={dx_free}"
    );
    assert!(
        dx_stick < dx_free * 0.35,
        "high static friction did not hold below-threshold incline enough: μ=0 dx={dx_free}, μ=0.8 dx={dx_stick}"
    );
    assert!(
        speed_stick.abs() < speed_free.abs() * 0.35,
        "high static friction did not suppress downhill speed enough: μ=0 v={speed_free}, μ=0.8 v={speed_stick}"
    );
}

/// Contact velocity recovery must not erase tangent motion that is unrelated to
/// the normal/friction solve. This is a regression gate for the interaction
/// between equality projection (CONNECT dumbbell) and contact velocity
/// projection: with μ=0, the connected pair should keep sliding along the
/// plane while the connect anchor remains tight.
#[test]
fn equality_contact_recovery_preserves_tangential_slide_when_friction_zero() {
    let model = build_connected_sphere_pair_on_plane(0.0);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let qd = s_a.body_qd.host_mut().unwrap();
        qd[0] = SpatialVector::new(V3::new(1.0, 0.0, 0.0), V3::ZERO);
        qd[1] = SpatialVector::new(V3::new(1.0, 0.0, 0.0), V3::ZERO);
    }
    let x0 = {
        let q = s_a.body_q.host().unwrap();
        0.5 * (q[0].position.x + q[1].position.x)
    };
    let control = model.control();
    let mut contacts = Contacts::with_capacity(16);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(12);
    let dt = 0.005_f64;

    for _ in 0..200 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let x1 = 0.5 * (q[0].position.x + q[1].position.x);
    let vx = 0.5 * (qd[0].linear.x + qd[1].linear.x);
    let anchor_a = q[0].transform_point(V3::new(0.15, 0.0, 0.0));
    let anchor_b = q[1].transform_point(V3::new(-0.15, 0.0, 0.0));
    let connect_error = (anchor_a - anchor_b).length();

    eprintln!(
        "equality+contact μ=0 slide: dx={:.4}, vx={:.4}, connect_error={:.6}, y=({:.4}, {:.4})",
        x1 - x0,
        vx,
        connect_error,
        q[0].position.y,
        q[1].position.y,
    );
    assert!(
        x1 - x0 > 0.50,
        "tangent slide was erased by contact/equality velocity recovery: dx={}",
        x1 - x0,
    );
    assert!(
        vx > 0.40,
        "tangent velocity should remain positive with zero friction, got vx={vx}",
    );
    assert!(
        connect_error < 0.02,
        "CONNECT anchor drifted while sliding: error={connect_error}",
    );
    assert!(
        q[0].position.y > 0.08 && q[1].position.y > 0.08,
        "spheres should remain above the plane, y=({}, {})",
        q[0].position.y,
        q[1].position.y,
    );
}

/// Kinematic contact tangential velocity should enter the Coulomb solve. This
/// is the procedural equivalent of a conveyor belt: the belt body itself stays
/// kinematic, but its body velocity should drag the dynamic sphere along +X.
#[test]
fn kinematic_conveyor_friction_drags_sphere() {
    let (model, sphere_idx) = build_sphere_on_kinematic_conveyor(0.8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let x0 = s_a.body_q.host().unwrap()[sphere_idx].position.x;
    let control = model.control();
    let mut contacts = Contacts::with_capacity(16);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(12);
    let dt = 0.005_f64;

    for _ in 0..240 {
        // Kinematic bodies are not pose-integrated by XPBD, but their
        // prescribed qd is part of the contact-point relative velocity.
        s_a.body_qd.host_mut().unwrap()[0] = SpatialVector::new(V3::new(1.0, 0.0, 0.0), V3::ZERO);
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let dx = q[sphere_idx].position.x - x0;
    let vx = qd[sphere_idx].linear.x;
    eprintln!(
        "kinematic conveyor: sphere dx={dx:.4}, vx={vx:.4}, y={:.4}",
        q[sphere_idx].position.y
    );
    assert!(
        dx > 0.05,
        "conveyor friction did not drag the sphere far enough: dx={dx}",
    );
    assert!(
        vx.is_finite(),
        "conveyor friction produced non-finite tangent velocity: vx={vx}",
    );
    assert!(
        q[sphere_idx].position.y > 0.08,
        "sphere should stay above the conveyor, y={}",
        q[sphere_idx].position.y,
    );
}
