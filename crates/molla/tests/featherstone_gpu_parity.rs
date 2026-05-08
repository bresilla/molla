//! Featherstone CPU vs GPU parity test. `#[ignore]` because the GPU
//! backend requires a real wgpu adapter — run via:
//!
//! ```bash
//! make test-featherstone-gpu
//! # or:
//! cargo test -p molla --test featherstone_gpu_parity --features gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

fn build_pendulum() -> Model {
    let mut b = ModelBuilder::new();
    let link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    b.add_joint_revolute(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.build().unwrap()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn pendulum_cpu_gpu_match_within_1e_3_rad() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let dt = 0.005_f64;
    let n_steps = 200usize;
    let initial_angle = 0.5_f64; // ~28°

    // ── CPU run ──
    let model_cpu = build_pendulum();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap()[0] = initial_angle;
    let control = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    let mut cpu_q = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        cpu_q.push(s_a.joint_q.host().unwrap()[0]);
    }

    // ── GPU run ──
    let mut model_gpu = build_pendulum();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap()[0] = initial_angle;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model_gpu.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);

    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    let mut gpu_q = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        // GPU buffers chain across iterations: kernel's `copy_buffer`
        // moves state_in.GPU → state_out.GPU then mutates state_out.
        // Re-uploading from host would clobber the chain with stale
        // (post-swap host = pre-step host) data.
        solver_gpu
            .step(&model_gpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let _ = s_a.joint_q.download(&device, &queue);
        gpu_q.push(s_a.joint_q.host().unwrap()[0]);
    }

    // Print first 10 steps for diagnosis.
    eprintln!("step | cpu_q     | gpu_q     | |Δ|");
    for i in 0..10.min(cpu_q.len()) {
        let c = cpu_q[i];
        let g = gpu_q[i];
        eprintln!("{:4} | {:.6} | {:.6} | {:.3e}", i, c, g, (c - g).abs());
    }

    let mut max_diff = 0.0_f64;
    for (i, (c, g)) in cpu_q.iter().zip(gpu_q.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff >= 1e-2 {
            panic!(
                "step {i}: CPU q = {c:.6}, GPU q = {g:.6}, |Δ| = {diff:.6e}"
            );
        }
    }
    eprintln!(
        "ok: pendulum CPU/GPU match over {n_steps} steps; max |Δq| = {max_diff:.6e}"
    );
}

fn build_pd_pendulum_with_limits() -> (Model, molla::core::JointId) {
    let mut b = ModelBuilder::new();
    let link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let joint = b
        .add_joint_revolute(
            BodyId::NONE,
            link,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
    // Limit + friction.
    b.set_joint_limits(joint, &[-1.0], &[1.0]).unwrap();
    b.set_joint_friction(joint, &[0.5]).unwrap();
    (b.build().unwrap(), joint)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn pd_pendulum_with_limits_cpu_gpu_match() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let dt = 0.005_f64;
    let n_steps = 400usize;

    // Same scene on both backends; a PD drive holds the pendulum at
    // q=0.7 (well inside the [-1, 1] limit), with friction damping.
    let configure_control = |control: &mut Control| {
        control.joint_target_mode.host_mut().unwrap()[0] =
            JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[0] = 0.7;
        control.joint_target_kp.host_mut().unwrap()[0] = 80.0;
        control.joint_target_kd.host_mut().unwrap()[0] = 8.0;
    };

    let (model_cpu, _) = build_pd_pendulum_with_limits();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let mut control = model_cpu.control();
    configure_control(&mut control);
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap()[0];

    let (mut model_gpu, _) = build_pd_pendulum_with_limits();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model_gpu.control();
    configure_control(&mut control);
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap()[0];

    eprintln!("PD+limit pendulum: cpu_q={cpu_q:.6} gpu_q={gpu_q:.6}");
    let diff = (cpu_q - gpu_q).abs();
    assert!(
        diff < 5e-3,
        "CPU/GPU diverged: cpu={cpu_q} gpu={gpu_q} |Δ|={diff}"
    );
    // And both should sit close to the target.
    assert!(
        (cpu_q - 0.7).abs() < 0.1,
        "CPU PD drive missed target: cpu_q={cpu_q}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn joint_limits_clamp_on_gpu() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let (mut model, _) = build_pd_pendulum_with_limits();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Start at the upper limit with a velocity that would otherwise
    // push past it.
    s_a.joint_q.host_mut().unwrap()[0] = 1.0;
    s_a.joint_qd.host_mut().unwrap()[0] = 5.0;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    let mut max_q = -10.0_f64;
    for _ in 0..1000 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        let q = s_a.joint_q.host().unwrap()[0];
        if q > max_q {
            max_q = q;
        }
    }
    assert!(
        max_q <= 1.0 + 1e-3,
        "GPU joint exceeded upper limit: max_q={max_q}"
    );
}
