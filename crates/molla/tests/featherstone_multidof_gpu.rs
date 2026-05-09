//! Stage 17 — multi-DOF Featherstone parity tests on GPU. Covers Ball,
//! Free, Distance, D6 joints. `#[ignore]` because the GPU backend
//! requires a real wgpu adapter — run via:
//!
//! ```bash
//! nixVulkan cargo test -p molla --features gpu --test featherstone_multidof_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Mat3, Quat, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

fn try_engine() -> Option<Engine> {
    match Engine::new(Backend::AutoGpu) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

/// Spherical (BALL) pendulum — body hangs from a ball joint, gets
/// nudged sideways, must oscillate as a real pendulum (energy
/// bounded, body stays under the anchor).
#[test]
#[ignore = "needs a real wgpu adapter"]
fn ball_spherical_pendulum_falls_under_gravity_gpu() {
    let Some(engine) = try_engine() else { return };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.05,
    }));
    b.add_joint_ball(
        BodyId::NONE,
        body,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
    )
    .unwrap();
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Start tilted ~17° about world +Z.
    let tilt = Quat::from_axis_angle(V3::Z, 0.3);
    let q = s_a.joint_q.host_mut().unwrap();
    q[0] = tilt.x;
    q[1] = tilt.y;
    q[2] = tilt.z;
    q[3] = tilt.w;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_damping(0.1);

    let dt = 0.001_f64;
    let n_steps = 2000usize;
    let mut max_y = -f64::INFINITY;
    let mut max_dist_from_anchor = 0.0_f64;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.body_q.download(&device, &queue);
        let p = s_a.body_q.host().unwrap()[0].position;
        if p.y > max_y {
            max_y = p.y;
        }
        // The COM should remain a fixed distance (≈0.5 m) from the
        // anchor at world origin; it's a constrained pendulum.
        let dist = p.length();
        if (dist - 0.5).abs() > max_dist_from_anchor {
            max_dist_from_anchor = (dist - 0.5).abs();
        }
    }
    eprintln!(
        "BALL pendulum after {n_steps} steps: max_y={max_y:.4}, max |dist-0.5|={max_dist_from_anchor:.4}"
    );
    // Without energy injection, the COM should stay ≈ in the lower
    // hemisphere (y < 0) — the tilt should never lift it past horizon.
    assert!(
        max_y < 0.05,
        "BALL pendulum gained energy and swung up: max_y={max_y:.4}"
    );
    // Distance from anchor should remain near 0.5 m.
    assert!(
        max_dist_from_anchor < 0.05,
        "BALL pendulum stretched: max |dist-0.5|={max_dist_from_anchor:.4}"
    );
}

/// Free-floating body under gravity falls at g (no ground).
#[test]
#[ignore = "needs a real wgpu adapter"]
fn free_joint_body_falls_at_g_gpu() {
    let Some(engine) = try_engine() else { return };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.1,
    }));
    b.add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();

    let dt = 0.001_f64;
    let n_steps = 200usize;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let _ = s_a.body_q.download(&device, &queue);
    let t = dt * n_steps as f64;
    let expected_y = -0.5 * molla::math::GRAVITY * t * t;
    let actual_y = s_a.joint_q.host().unwrap()[1];
    let actual_body_y = s_a.body_q.host().unwrap()[0].position.y;
    eprintln!(
        "FREE body fall after {t:.3}s: joint_q.y={actual_y:.4} body.y={actual_body_y:.4} expected={expected_y:.4}"
    );
    let err = (actual_y - expected_y).abs();
    assert!(
        err < 0.02,
        "free body fall: expected y={expected_y:.4}, got {actual_y:.4} (Δ={err:.4})"
    );
}

/// D6 with all DOFs locked behaves like a fixed joint — the body
/// stays at its initial pose under gravity.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn d6_pretensioned_holds_under_gravity_gpu() {
    let Some(engine) = try_engine() else { return };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.1,
    }));
    let j = b
        .add_joint_d6(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    // Lock all 6 DOFs (lower==upper==0). This is the "pretensioned
    // cable" setup: D6 acts as a fixed weld.
    b.set_joint_limits(j, &[0.0; 6], &[0.0; 6]).unwrap();
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();

    for _ in 0..500 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.body_q.download(&device, &queue);
    let pos = s_a.body_q.host().unwrap()[0].position;
    eprintln!("D6 locked body after 0.5s under gravity: pos={pos:?}");
    assert!(
        pos.length() < 1e-2,
        "D6 locked body drifted under gravity: pos={pos:?}"
    );
}

/// CPU/GPU parity: free-joint body under gravity follows the same
/// trajectory (within f32 tolerance) on both backends.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn free_joint_cpu_gpu_match() {
    let Some(engine) = try_engine() else { return };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    fn build() -> Model {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.1,
        }));
        b.add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        b.build().unwrap()
    }

    let dt = 0.005_f64;
    let n_steps = 100;

    let model_cpu = build();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let ctl = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &ctl, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap().to_vec();

    let mut model_gpu = build();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut ctl = model_gpu.control();
    ctl.upload_to_device(&device, &queue).unwrap();
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &ctl, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap().to_vec();

    eprintln!(
        "FREE parity: cpu y={:.6} gpu y={:.6}",
        cpu_q[1], gpu_q[1]
    );
    for i in 0..7 {
        let d = (cpu_q[i] - gpu_q[i]).abs();
        assert!(
            d < 5e-3,
            "FREE joint coord {i} diverged: cpu={} gpu={} |Δ|={d}",
            cpu_q[i],
            gpu_q[i]
        );
    }
}
