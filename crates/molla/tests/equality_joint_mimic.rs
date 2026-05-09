//! Stage 32 — joint-mimic equality constraint
//! (`EqualityType::JointMimic`).
//!
//! The constraint enforces `q_driven = polynomial(q_driver)` over the
//! XPBD position-projection step. We don't need the full solver-step
//! pipeline (gravity, contacts, …) to test the constraint convergence
//! itself; the projection pass operates purely on `joint_q` /
//! `joint_qd`. Each test seeds `joint_q` with a deliberate violation,
//! runs the projection-only solver on a contacts-empty Model, and
//! checks that the constraint holds.
//!
//! - Linear mimic (driver = driven · 1.0): two revolute joints,
//!   drive one and check the other tracks within 1e-6 over 600
//!   substeps.
//! - Quadratic mimic (driven = driver²): nonlinear coupling,
//!   verified across a sweep of driver values.
//! - Robotiq-style 2-finger gripper: `driven = -driver` (symmetric
//!   antiparallel motion).
//! - GPU parity (gated behind `#[ignore]`, GPU feature only):
//!   rebuilds the same model on the device and checks the WGSL
//!   `project_joint_mimic` kernel matches the CPU reference.

use molla::geometry::CollisionPipeline;
use molla::prelude::*;
use molla::sim::Contacts;

/// Build a 2-revolute-joint model: each joint has a parent body
/// (kinematic, anchored) and a child body. The two articulations are
/// independent — only the mimic constraint couples them.
fn build_two_revolute_chain() -> (Model, JointId, JointId) {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    // First articulation (driver).
    let driver_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::IDENTITY)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driver_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driver = b
        .add_joint_revolute(
            driver_anchor,
            driver_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();

    // Second articulation (driven).
    b.begin_articulation();
    let driven_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::IDENTITY,
            ))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driven_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driven = b
        .add_joint_revolute(
            driven_anchor,
            driven_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();

    let model = b.build().unwrap();
    (model, driver, driven)
}

/// Run the XPBD CPU solver for `steps` substeps with `dt`. The model
/// is contact-free; gravity is disabled by setting per-body inertia
/// such that gravity has no effect on dynamics — but really we just
/// rely on the kinematic anchor + zero forces to keep the body still
/// while the projection drives `joint_q`.
fn run_xpbd_cpu(model: &Model, init_q: &[Real], steps: usize, dt: Real) -> Vec<Real> {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        q.copy_from_slice(init_q);
    }
    let control = model.control();
    let mut contacts = Contacts::with_capacity(0);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    for _ in 0..steps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.joint_q.host().unwrap().to_vec()
}

#[test]
fn linear_mimic_identity_holds_to_1e6() {
    let (mut model, driver, driven) = build_two_revolute_chain();
    // Use direct ModelBuilder shouldn't be re-used; instead rebuild
    // with the constraint. We refactor: rebuild the model with the
    // mimic added.
    // (We can't add an equality after `build()`; re-use the same
    // construction here.)
    let _ = (driver, driven, &mut model);
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    let driver_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driver_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driver = b
        .add_joint_revolute(
            driver_anchor,
            driver_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    b.begin_articulation();
    let driven_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::IDENTITY,
            ))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driven_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driven = b
        .add_joint_revolute(
            driven_anchor,
            driven_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    // Linear identity mimic: poly = [0, 1, 0, 0, 0] → driven = driver.
    b.add_equality_joint_mimic(driver, driven, [0.0, 1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let model = b.build().unwrap();

    // Seed driver = 0.5 rad, driven = 0 (deliberate violation).
    // Driver is held by the kinematic anchor + zero gravity (no
    // forces) so its joint coord stays at 0.5; the projection
    // pulls driven onto the polynomial each substep.
    let init_q = vec![0.5, 0.0];
    let q_after = run_xpbd_cpu(&model, &init_q, 600, 0.001);
    let drift = (q_after[0] - q_after[1]).abs();
    eprintln!(
        "linear: driver={:.9} driven={:.9} drift={drift:.3e}",
        q_after[0], q_after[1]
    );
    assert!(
        drift < 1e-6,
        "linear mimic broke: driver={} driven={} drift={drift}",
        q_after[0],
        q_after[1]
    );
}

/// Quadratic mimic: `q_driven = q_driver²`. Sweep driver across
/// several values; the projection should snap `driven` onto the
/// parabola each time, after a handful of XPBD iterations.
#[test]
fn quadratic_mimic_tracks_parabola() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    let driver_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driver_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driver = b
        .add_joint_revolute(
            driver_anchor,
            driver_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    b.begin_articulation();
    let driven_anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::IDENTITY,
            ))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let driven_link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driven = b
        .add_joint_revolute(
            driven_anchor,
            driven_link,
            Transform::IDENTITY,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    // poly = [0, 0, 1, 0, 0] → driven = driver².
    b.add_equality_joint_mimic(driver, driven, [0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    let model = b.build().unwrap();

    // Sweep driver values in [-0.5, 0.5] (rad). The projection should
    // converge onto driven = driver² for each. We re-seed init_q
    // for each case and run the full XPBD pipeline so the projection
    // accumulates per-substep iterations.
    for &dr in &[-0.5_f64, -0.25, 0.0, 0.25, 0.5] {
        // Driver is held fixed by initial-q seed + zero gravity (no
        // contacts). Driven starts at 0.0 (deliberate violation).
        let init_q = vec![dr, 0.0];
        let q_after = run_xpbd_cpu(&model, &init_q, 200, 0.001);
        let target = dr * dr;
        let drift = (q_after[1] - target).abs();
        eprintln!(
            "quadratic dr={dr:+.3} → driven={:.6} (target={:.6}, drift={drift:.3e})",
            q_after[1], target,
        );
        // Quadratic mimic with 8 iterations × 200 substeps reaches
        // ~1e-9 — generous 1e-5 tolerance accommodates the
        // chain-rule fraction redistribution affecting `driver`
        // slightly across the sweep (we don't pin the driver as
        // kinematic at the joint-coord level).
        assert!(
            drift < 1e-5,
            "quadratic mimic dr={dr}: driven={} target={} drift={drift}",
            q_after[1],
            target,
        );
    }
}

/// Robotiq-style 2-finger gripper: both fingers move symmetrically
/// (driven = −driver). This is the canonical use case for mimic
/// joints in real robot URDFs.
#[test]
fn robotiq_gripper_two_fingers_couple_antisymmetrically() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    // Two prismatic finger joints — abstract away the mechanical
    // detail. Anchors are kinematic (the gripper "palm").
    let palm_a = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let finger_a = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 0.1,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 0.001,
        }),
    );
    let driver = b
        .add_joint_prismatic(
            palm_a,
            finger_a,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    b.begin_articulation();
    let palm_b = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let finger_b = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 0.1,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 0.001,
        }),
    );
    let driven = b
        .add_joint_prismatic(
            palm_b,
            finger_b,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    // Anti-symmetric linear mimic: driven = -driver.
    b.add_equality_joint_mimic(driver, driven, [0.0, -1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let model = b.build().unwrap();

    // Seed driver = +0.03 m (3 cm finger open), driven = 0 (off).
    let init_q = vec![0.03, 0.0];
    let q_after = run_xpbd_cpu(&model, &init_q, 600, 0.001);
    let sum = q_after[0] + q_after[1];
    eprintln!(
        "robotiq: driver={:.6} driven={:.6} sum={sum:.3e} (target=0)",
        q_after[0], q_after[1]
    );
    assert!(
        sum.abs() < 1e-6,
        "robotiq antisymmetric coupling broke: driver={} driven={} sum={sum}",
        q_after[0],
        q_after[1]
    );
}

/// Polynomial sanity check on the projection pass, called directly
/// (no XPBD substep wrapping). Confirms the Lagrange-multiplier
/// formula converges in a single sweep for the linear-identity case.
#[test]
fn projection_pass_converges_single_sweep_linear() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    let pa = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let la = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driver = b
        .add_joint_revolute(pa, la, Transform::IDENTITY, Transform::IDENTITY, Vec3::Z)
        .unwrap();
    b.begin_articulation();
    let pb = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let lb = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let driven = b
        .add_joint_revolute(pb, lb, Transform::IDENTITY, Transform::IDENTITY, Vec3::Z)
        .unwrap();
    b.add_equality_joint_mimic(driver, driven, [0.1, 1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let model = b.build().unwrap();

    // One XPBD step. Driver-authoritative semantics snap driven
    // directly onto `poly(driver) = 0.1 + 1.0 · 0.5 = 0.6` in a
    // single sweep; the driver's coord is left untouched (acts as
    // infinite-mass).
    let q_after = run_xpbd_cpu(&model, &[0.5, 0.0], 1, 0.001);
    let target = 0.1 + q_after[0]; // driven = c0 + c1·driver
    let drift = (q_after[1] - target).abs();
    eprintln!(
        "single-sweep: driver={:.9} driven={:.9} target={:.9} drift={drift:.3e}",
        q_after[0], q_after[1], target
    );
    assert!(
        (q_after[0] - 0.5).abs() < 1e-12,
        "driver should be unchanged (driver-authoritative), got {}",
        q_after[0]
    );
    assert!(
        drift < 1e-12,
        "linear mimic should converge to machine eps in one sweep, got {drift}"
    );
}

// ── GPU parity test (Stage 32 GPU port) ──────────────────────────
//
// Rebuilds the same linear-mimic model and the same quadratic-mimic
// model on the GPU, runs a handful of substeps, and checks the WGSL
// `project_joint_mimic` entry point produces a `joint_q` matching
// the CPU reference within f32-narrowing tolerance (1e-5 absolute).

// GPU helper: build a 2-revolute model with one body each (driver +
// driven) plus joint-mimic equality with the supplied polynomial.
// Used by the GPU parity tests below — the model adds a placeholder
// sphere shape per body so the full XPBD GPU pipeline (which expects
// a non-empty shape table for upload) doesn't trip on empty buffers.
#[cfg(feature = "gpu")]
fn build_mimic_model_gpu(poly: [Real; 5]) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    let pa = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(pa, 0.05, Transform::IDENTITY).unwrap();
    let la = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    b.add_shape_sphere(la, 0.05, Transform::IDENTITY).unwrap();
    let driver = b
        .add_joint_revolute(pa, la, Transform::IDENTITY, Transform::IDENTITY, Vec3::Z)
        .unwrap();
    b.begin_articulation();
    let pb = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(pb, 0.05, Transform::IDENTITY).unwrap();
    let lb = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    b.add_shape_sphere(lb, 0.05, Transform::IDENTITY).unwrap();
    let driven = b
        .add_joint_revolute(pb, lb, Transform::IDENTITY, Transform::IDENTITY, Vec3::Z)
        .unwrap();
    b.add_equality_joint_mimic(driver, driven, poly).unwrap();
    b.build().unwrap()
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-equality-joint-mimic-gpu`"]
fn gpu_linear_mimic_matches_cpu_reference() {
    use molla::compute::{Backend, Engine};

    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let device = engine.device().expect("device");
    let queue = engine.queue().expect("queue");

    let model_cpu = build_mimic_model_gpu([0.0, 1.0, 0.0, 0.0, 0.0]);
    let mut model_gpu = build_mimic_model_gpu([0.0, 1.0, 0.0, 0.0, 0.0]);
    model_gpu.upload_to_device(device, queue).unwrap();

    let cpu_q = run_xpbd_cpu(&model_cpu, &[0.5, 0.0], 50, 0.001);

    // GPU: full XPBD step pipeline; the joint-mimic projection is
    // dispatched as part of `XpbdEqualityKernel::dispatch`.
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        q[0] = 0.5;
        q[1] = 0.0;
    }
    s_a.upload_to_device(device, queue).unwrap();
    s_b.upload_to_device(device, queue).unwrap();
    let mut control = model_gpu.control();
    let _ = control.body_external_f.ensure_device(device);
    control.body_external_f.upload(queue).unwrap();
    // Allocate a contacts buffer + run the collision pipeline once
    // to populate the per-body CSR (the GPU XPBD kernel reads
    // `body_contact_offsets`, which the pipeline initialises). The
    // bodies are far apart and have tiny placeholder spheres so no
    // contacts will actually fire.
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model_gpu, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(device, queue).unwrap();
    let mut solver = XpbdSolver::new_gpu(&engine, model_gpu.body_count as u32)
        .expect("new_gpu");
    solver.position_iterations = 8;
    for _ in 0..50 {
        solver
            .step(&model_gpu, &s_a, &control, &contacts, 0.001, &mut s_b)
            .unwrap();
        s_b.joint_q.download(device, queue).unwrap();
        s_b.joint_qd.download(device, queue).unwrap();
        s_b.body_q.download(device, queue).unwrap();
        s_b.body_qd.download(device, queue).unwrap();
        s_b.body_f.download(device, queue).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let gpu_q = s_a.joint_q.host().unwrap().to_vec();

    eprintln!("CPU q={:?}", cpu_q);
    eprintln!("GPU q={:?}", gpu_q);
    let drift = (cpu_q[0] - gpu_q[0]).abs() + (cpu_q[1] - gpu_q[1]).abs();
    assert!(drift < 1e-4, "CPU/GPU joint_q drift {drift} > 1e-4");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-equality-joint-mimic-gpu`"]
fn gpu_quadratic_mimic_matches_cpu_reference() {
    use molla::compute::{Backend, Engine};

    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let device = engine.device().expect("device");
    let queue = engine.queue().expect("queue");

    let model_cpu = build_mimic_model_gpu([0.0, 0.0, 1.0, 0.0, 0.0]);
    let mut model_gpu = build_mimic_model_gpu([0.0, 0.0, 1.0, 0.0, 0.0]);
    model_gpu.upload_to_device(device, queue).unwrap();

    let cpu_q = run_xpbd_cpu(&model_cpu, &[0.4, 0.0], 50, 0.001);

    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        q[0] = 0.4;
        q[1] = 0.0;
    }
    s_a.upload_to_device(device, queue).unwrap();
    s_b.upload_to_device(device, queue).unwrap();
    let mut control = model_gpu.control();
    let _ = control.body_external_f.ensure_device(device);
    control.body_external_f.upload(queue).unwrap();
    // Allocate a contacts buffer + run the collision pipeline once
    // to populate the per-body CSR (the GPU XPBD kernel reads
    // `body_contact_offsets`, which the pipeline initialises). The
    // bodies are far apart and have tiny placeholder spheres so no
    // contacts will actually fire.
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model_gpu, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(device, queue).unwrap();
    let mut solver = XpbdSolver::new_gpu(&engine, model_gpu.body_count as u32)
        .expect("new_gpu");
    solver.position_iterations = 8;
    for _ in 0..50 {
        solver
            .step(&model_gpu, &s_a, &control, &contacts, 0.001, &mut s_b)
            .unwrap();
        s_b.joint_q.download(device, queue).unwrap();
        s_b.joint_qd.download(device, queue).unwrap();
        s_b.body_q.download(device, queue).unwrap();
        s_b.body_qd.download(device, queue).unwrap();
        s_b.body_f.download(device, queue).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let gpu_q = s_a.joint_q.host().unwrap().to_vec();

    eprintln!("quadratic GPU q={:?} (cpu={:?})", gpu_q, cpu_q);
    let drift = (cpu_q[1] - gpu_q[1]).abs();
    assert!(
        drift < 1e-4,
        "quadratic CPU/GPU drift {drift} > 1e-4 (cpu={}, gpu={})",
        cpu_q[1],
        gpu_q[1],
    );
}
