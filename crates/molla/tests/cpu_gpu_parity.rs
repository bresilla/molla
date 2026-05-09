//! CPU/GPU parity: the same scene run on both backends should produce
//! the same trajectory within an f32-narrowing tolerance.
//!
//! `#[ignore]` because the GPU backend requires a real wgpu adapter.
//! Run via:
//!
//! ```bash
//! make test-gpu
//! # or:
//! cargo test -p molla --test cpu_gpu_parity --features gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::compute::{Backend, Engine};
use molla::prelude::*;

fn build_scene(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    for i in 0..n {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(i as f64 * 1.5, 10.0, (i as f64 * 0.3).sin())),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    }
    b.build().unwrap()
}

fn run_cpu(model: &Model, steps: usize, dt: Real) -> Vec<Vec<Transform>> {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = SemiImplicitSolver::new_cpu();

    let mut traj = Vec::with_capacity(steps);
    for _ in 0..steps {
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        traj.push(s_a.body_q.host().unwrap().to_vec());
    }
    traj
}

fn run_gpu(engine: &Engine, model: &Model, steps: usize, dt: Real) -> Vec<Vec<Transform>> {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut model_mut = clone_model_for_gpu(model);
    model_mut.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model_mut.state().unwrap();
    let mut s_b = model_mut.state().unwrap();
    let mut control = model_mut.control();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();

    let contacts = Contacts::with_capacity(0);
    let mut solver = SemiImplicitSolver::new_gpu(engine).unwrap();

    let mut traj = Vec::with_capacity(steps);
    for _ in 0..steps {
        solver
            .step(&model_mut, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        s_a.download_from_device(&device, &queue).unwrap();
        traj.push(s_a.body_q.host().unwrap().to_vec());
    }
    traj
}

/// Rebuild a fresh `Model` matching `model`. The original Model holds
/// `Buffer`s that don't `Clone` (and we want a clean device-side state
/// for the GPU run anyway).
fn clone_model_for_gpu(model: &Model) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(model.angular_damping);
    let q0 = model.initial_body_q.host().unwrap();
    let mass = model.body_mass.host().unwrap();
    let inertia = model.body_inertia.host().unwrap();
    let com = model.body_com.host().unwrap();
    let flags = model.body_flags.host().unwrap();
    for i in 0..model.body_count {
        let body = b.add_body(
            BodyParams::new()
                .with_mass(mass[i])
                .with_inertia(inertia[i])
                .with_com(com[i])
                .with_pose(q0[i])
                .with_flags(flags[i]),
        );
        // Sphere with a placeholder radius — kernel doesn't read shapes.
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    }
    b.build().unwrap()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu`"]
fn parity_100_spheres_within_1e_3() {
    let engine = Engine::new(Backend::AutoGpu).expect("GPU adapter required for this test (gated by #[ignore])");
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let n = 100usize;
    let steps = 100usize;
    let dt = 0.01;

    let model = build_scene(n);
    let cpu_traj = run_cpu(&model, steps, dt);
    let gpu_traj = run_gpu(&engine, &model, steps, dt);

    for (step_idx, (cpu_step, gpu_step)) in cpu_traj.iter().zip(gpu_traj.iter()).enumerate() {
        for (j, (cpu, gpu)) in cpu_step.iter().zip(gpu_step.iter()).enumerate() {
            let dx = (cpu.position - gpu.position).length();
            assert!(
                dx < 1e-3,
                "step {step_idx} body {j}: CPU/GPU drift {dx} > 1e-3 \
                 (cpu={:?}, gpu={:?})",
                cpu.position,
                gpu.position
            );
        }
    }
}
