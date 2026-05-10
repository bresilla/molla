//! GPU equality-constraint projection — golden / parity tests for
//! the WGSL `XpbdEqualityKernel` that replaced the host-side
//! `apply_equality_post_gpu` shim. `#[ignore]` because they need a
//! real wgpu adapter; run with `make test-equality-gpu`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla::prelude::*;

/// Mirror of `xpbd_equality.rs::connect_pins_two_bodies_under_gravity`
/// but driven by the **GPU** solver. Two bodies (one kinematic
/// anchor, one free) joined by a CONNECT — after settling, the free
/// body should hang directly under the anchor. The GPU equality
/// kernel is what makes this work end-to-end on device.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_connect_pins_two_bodies_under_gravity() {
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
    b.add_equality_connect(anchor, free, Vec3::ZERO, Vec3::ZERO);
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(8);
    let mut pipeline = CollisionPipeline::new();

    let dt: Real = 0.005;
    for _ in 0..1000 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_q.download(&device, &queue).unwrap();
    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    eprintln!(
        "GPU connect: anchor at {:?}, free at {:?}, dist {dist:.4}",
        q[0].position, q[1].position
    );
    assert!(
        dist < 0.05,
        "GPU CONNECT failed: bodies {dist:.3} m apart"
    );
}

/// Mirror of `weld_keeps_orientation_aligned`, GPU-driven.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_weld_keeps_orientation_aligned() {
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
    let mut model = b.build().unwrap();
    // No gravity so the test stays deterministic.
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(20);

    // Run a single no-op collide pass so body_contact_offsets gets
    // sized for body_count and gains a GPU device buffer. Without
    // this the XPBD kernel's contact bind group fails to bind.
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let dt: Real = 0.005;
    for _ in 0..200 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_q.download(&device, &queue).unwrap();
    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    let dot = q[1].rotation.x * q[0].rotation.x
        + q[1].rotation.y * q[0].rotation.y
        + q[1].rotation.z * q[0].rotation.z
        + q[1].rotation.w * q[0].rotation.w;
    eprintln!("GPU weld: dist={dist:.4} rot-dot={dot:.4}");
    assert!(dist < 0.05, "GPU WELD position drifted: dist={dist}");
    assert!(
        dot.abs() > 0.95,
        "GPU WELD orientation drifted: dot={dot}"
    );
}

/// Velocity recovery sanity check: a free body pinned to a
/// stationary kinematic anchor must end the step with near-zero
/// linear velocity (the constraint snapped the position back, the
/// recovery pass must reflect that as ≈ zero net displacement / dt).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_equality_velocity_recovery_zeros_drift() {
    let mut b = ModelBuilder::new();
    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::IDENTITY)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(0.001, 0.0, 0.0)), // tiny offset
    );
    b.add_equality_connect(anchor, free, Vec3::ZERO, Vec3::ZERO);
    let mut model = b.build().unwrap();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO; // no gravity → equilibrium

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(20);

    // No-op collide to size body_contact_offsets — see comment in
    // the WELD test above.
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    // Several steps so the constraint converges; velocity should
    // settle back to zero.
    for _ in 0..50 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_qd.download(&device, &queue).unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let v = qd[1].linear.length();
    eprintln!("GPU equality v_post: {v:.4}");
    assert!(v < 0.5, "expected ≈ 0 velocity at equilibrium, got {v}");
}
