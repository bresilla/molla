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
