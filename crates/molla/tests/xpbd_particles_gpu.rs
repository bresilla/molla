//! GPU particle XPBD — parity test: take a small cloth grid, run
//! N substeps on the CPU oracle (`step_particles`) and on the GPU
//! kernel (`XpbdParticlesKernel` via the GPU solver path), and
//! confirm the per-particle positions match.
//!
//! `#[ignore]` because it needs a real wgpu adapter; run with
//! `make test-particles-gpu`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Real, Transform, Vec3};
use molla::prelude::*;
use molla::solvers::Solver as _;

/// Build a small 6×6 cloth pinned at the four corners.
fn build_cloth() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.1);
    let n = 6_u32;
    let size = 1.0_f64;
    let origin = Vec3::new(-0.5, 1.5, -0.5);
    let u_axis = Vec3::new(size, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, size);
    let (first, _, _) = b.add_cloth_grid(
        origin, u_axis, v_axis, n, n, 0.05, 0.02, 5.0e-5, 0.05, WorldId(0),
    );
    // Pin the four corners.
    b.pin_particle(first);
    b.pin_particle(first + n - 1);
    b.pin_particle(first + (n - 1) * n);
    b.pin_particle(first + (n - 1) * n + n - 1);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().unwrap()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-particles-gpu"]
fn cpu_gpu_cloth_match() {
    let model_cpu = build_cloth();
    let model_gpu = build_cloth();

    // ── CPU run ─────────────────────────────────────────────────────
    let mut s_a_cpu = model_cpu.state().unwrap();
    let mut s_b_cpu = model_cpu.state().unwrap();
    let control_cpu = model_cpu.control();
    let mut contacts_cpu = Contacts::with_capacity(64);
    let mut pipeline_cpu = CollisionPipeline::new();
    let mut solver_cpu = XpbdSolver::new_cpu().with_iterations(8);
    let dt: Real = 0.005;
    let n_steps = 60;
    for _ in 0..n_steps {
        pipeline_cpu
            .collide(&model_cpu, &s_a_cpu, &mut contacts_cpu)
            .unwrap();
        solver_cpu
            .step(&model_cpu, &s_a_cpu, &control_cpu, &contacts_cpu, dt, &mut s_b_cpu)
            .unwrap();
        std::mem::swap(&mut s_a_cpu, &mut s_b_cpu);
    }
    let cpu_positions = s_a_cpu.particle_q.host().unwrap().to_vec();

    // ── GPU run ─────────────────────────────────────────────────────
    let Ok(engine) = Engine::new_gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model_gpu = model_gpu;
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    let mut control = model_gpu.control();
    let mut contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model_gpu.body_count.max(1) as u32)
        .unwrap()
        .with_iterations(8);
    let mut pipeline = CollisionPipeline::new();
    for _ in 0..n_steps {
        pipeline.collide(&model_gpu, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model_gpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let gpu_positions = s_a.particle_q.host().unwrap().to_vec();

    // Per-particle position parity. f32 GPU vs f64 CPU + Gauss-Seidel
    // ordering means we expect roughly mm-scale deviation per particle
    // after 60 steps. Tighten if needed.
    assert_eq!(cpu_positions.len(), gpu_positions.len());
    let mut max_err = 0.0_f64;
    for (i, (c, g)) in cpu_positions.iter().zip(gpu_positions.iter()).enumerate() {
        let d = (*c - *g).length();
        if d > max_err {
            max_err = d;
        }
        assert!(
            d < 5.0e-2,
            "particle {i} diverged: cpu={c:?} gpu={g:?} |Δ|={d:.4}"
        );
    }
    eprintln!("max CPU/GPU position error after {n_steps} steps: {max_err:.5} m");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-particles-gpu"]
fn pinned_cloth_corners_stay_put() {
    // Sanity check: a cloth pinned at corners should keep those
    // corners exactly where we placed them, on both CPU and GPU.
    let model = build_cloth();
    let Ok(engine) = Engine::new_gpu() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model = model;
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let initial_q = s_a.particle_q.host().unwrap().to_vec();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count.max(1) as u32)
        .unwrap()
        .with_iterations(8);
    let mut pipeline = CollisionPipeline::new();
    for _ in 0..100 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    let final_q = s_a.particle_q.host().unwrap();

    // Indices of the four pinned corners in the 6×6 grid.
    let n = 6_u32;
    let corners = [0, n - 1, (n - 1) * n, (n - 1) * n + n - 1];
    for c in corners {
        let i = c as usize;
        let drift = (final_q[i] - initial_q[i]).length();
        assert!(
            drift < 1.0e-4,
            "pinned corner {i} drifted by {drift:.5} m"
        );
    }
}
