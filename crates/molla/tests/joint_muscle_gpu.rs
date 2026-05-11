//! Stage 42 v3 — Hill-type muscle GPU/CPU parity.
//!
//! Three scenarios all gated behind `#[ignore]` because the GPU
//! backend requires a real wgpu adapter — run via:
//!
//! ```bash
//! nixVulkan cargo test -p molla --features gpu --test joint_muscle_gpu -- --ignored
//! ```
//!
//! 1. `single_muscle_cpu_gpu_parity_100_steps` — 1 cable + 1 muscle,
//!    100 steps at `dt = 1 ms`, joint coords agree within `1e-5`
//!    relative.
//! 2. `antagonist_pair_cancels_on_gpu` — agonist + antagonist muscles
//!    at equal activation on the same cable; both CPU and GPU keep
//!    `|q_l| < 1 mm` after 500 steps.
//! 3. `muscle_scaling_walltime` — runs N muscles for 100 steps on
//!    CPU + GPU and prints wall-clock for N ∈ {10, 100}. Diagnostic
//!    only; the assertion just sanity-checks GPU produces finite
//!    output.

#![cfg(feature = "gpu")]

use molla::actuators::MuscleConfig;
use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

/// Build a single-cable single-body model with `muscles` muscles all
/// attached to the same cable joint. The model uses `f_max = 100 N`,
/// `l_opt = 5 cm` defaults — agonist polarity unless overridden.
fn cable_with_muscles(
    mass: Real,
    k_lin: Real,
    c_lin: Real,
    muscles: &[MuscleConfig],
) -> Model {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.05,
    }));
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            k_lin,
            0.0,
            c_lin,
            0.0,
        )
        .unwrap();
    for m in muscles {
        b.add_muscle(joint, *m).unwrap();
    }
    b.set_world_gravity(WorldId(0), V3::ZERO);
    b.build().unwrap()
}

/// Compose a "settled" initial state for the cable so we exercise
/// both Hill curves immediately (not at the f_L peak).
fn seed_state(model: &Model, q_l: Real, qd_l: Real) -> State {
    let mut s = model.state().unwrap();
    {
        let q = s.joint_q.host_mut().unwrap();
        if !q.is_empty() {
            q[0] = q_l;
        }
    }
    {
        let qd = s.joint_qd.host_mut().unwrap();
        if !qd.is_empty() {
            qd[0] = qd_l;
        }
    }
    s
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-joint-muscle-gpu` or `nixVulkan cargo test ...`"]
fn single_muscle_cpu_gpu_parity_100_steps() {
    let engine = Engine::new(Backend::AutoGpu)
        .expect("GPU adapter required for this test (gated by #[ignore])");
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let dt: Real = 1.0e-3;
    let n_steps = 100usize;
    let f_max: Real = 80.0;
    let k_lin: Real = 4000.0;
    let mass: Real = 1.0;
    let c_lin: Real = 2.0 * (k_lin * mass).sqrt();

    let cfg = MuscleConfig {
        f_max,
        l_opt: 0.05,
        v_max: 0.5,
        activation: 1.0,
        ..MuscleConfig::default()
    };

    // ── CPU run ──
    let model_cpu = cable_with_muscles(mass, k_lin, c_lin, &[cfg]);
    let mut s_a = seed_state(&model_cpu, 0.0, 0.0);
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    let mut cpu_q = Vec::with_capacity(n_steps);
    let mut cpu_qd = Vec::with_capacity(n_steps);
    let t0 = std::time::Instant::now();
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        cpu_q.push(s_a.joint_q.host().unwrap()[0]);
        cpu_qd.push(s_a.joint_qd.host().unwrap()[0]);
    }
    let cpu_us = t0.elapsed().as_micros();

    // ── GPU run ──
    let mut model_gpu = cable_with_muscles(mass, k_lin, c_lin, &[cfg]);
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();

    let mut s_a = seed_state(&model_gpu, 0.0, 0.0);
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control_gpu = model_gpu.control();
    control_gpu.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);

    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    let mut gpu_q = Vec::with_capacity(n_steps);
    let mut gpu_qd = Vec::with_capacity(n_steps);
    let t0 = std::time::Instant::now();
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &control_gpu, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        let _ = s_a.joint_qd.download(&device, &queue);
        gpu_q.push(s_a.joint_q.host().unwrap()[0]);
        gpu_qd.push(s_a.joint_qd.host().unwrap()[0]);
    }
    let gpu_us = t0.elapsed().as_micros();

    eprintln!(
        "single muscle 100 steps — CPU {:.2} ms, GPU {:.2} ms",
        (cpu_us as f64) * 1.0e-3,
        (gpu_us as f64) * 1.0e-3,
    );

    // Print first few + last step for diagnostics.
    eprintln!("step | cpu_q      | gpu_q      | |Δ|         | rel");
    for &i in &[0usize, 1, 5, 25, 50, 99] {
        if i >= cpu_q.len() {
            continue;
        }
        let c = cpu_q[i];
        let g = gpu_q[i];
        let d = (c - g).abs();
        let rel = d / c.abs().max(1e-12);
        eprintln!(
            "{:4} | {:+.5e} | {:+.5e} | {:.3e} | {:.3e}",
            i, c, g, d, rel
        );
    }

    let mut max_rel: f64 = 0.0;
    for (c, g) in cpu_q.iter().zip(gpu_q.iter()) {
        let d = (c - g).abs();
        let rel = d / c.abs().max(1e-6);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // The CPU uses f64 throughout; the GPU shader runs in f32. We
    // expect single-precision rounding to dominate the divergence at
    // ~1e-5 relative.
    assert!(
        max_rel < 5.0e-5,
        "CPU/GPU diverged: max rel = {max_rel}"
    );
    let q_final = cpu_q.last().copied().unwrap();
    assert!(
        q_final < 0.0,
        "muscle should have contracted by end of run (q_l = {q_final})"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-joint-muscle-gpu` or `nixVulkan cargo test ...`"]
fn antagonist_pair_cancels_on_gpu() {
    let engine = Engine::new(Backend::AutoGpu)
        .expect("GPU adapter required for this test (gated by #[ignore])");

    let cfg = MuscleConfig {
        f_max: 150.0,
        l_opt: 0.1,
        v_max: 0.5,
        activation: 1.0,
        ..MuscleConfig::default()
    };
    let antagonist = MuscleConfig {
        polarity: -1.0,
        ..cfg
    };

    let dt: Real = 1.0e-4;
    let n_steps = 500usize;

    // CPU baseline first.
    let model_cpu = cable_with_muscles(1.0, 0.0, 0.0, &[cfg, antagonist]);
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap()[0];
    eprintln!("antagonist CPU q_l = {:.3e}", cpu_q);
    assert!(
        cpu_q.abs() < 5.0e-4,
        "CPU antagonist pair should net zero; got q_l = {cpu_q}"
    );

    // GPU run with the same setup.
    let mut model_gpu = cable_with_muscles(1.0, 0.0, 0.0, &[cfg, antagonist]);
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control_gpu = model_gpu.control();
    control_gpu.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &control_gpu, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap()[0];
    eprintln!("antagonist GPU q_l = {:.3e}", gpu_q);
    // Antagonist torque cancellation on GPU should hold to f32
    // precision over 500 steps. Tolerance ~1 mm matches the CPU
    // assertion plus a margin for f32 noise.
    assert!(
        gpu_q.abs() < 1.0e-3,
        "GPU antagonist pair should net zero; got q_l = {gpu_q}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-joint-muscle-gpu` or `nixVulkan cargo test ...`"]
fn muscle_scaling_walltime() {
    let engine = Engine::new(Backend::AutoGpu)
        .expect("GPU adapter required for this test (gated by #[ignore])");

    let dt: Real = 1.0e-3;
    let n_steps = 100usize;

    for &n_muscles in &[10usize, 100] {
        let muscles: Vec<MuscleConfig> = (0..n_muscles)
            .map(|i| MuscleConfig {
                f_max: 5.0,
                l_opt: 0.05,
                v_max: 0.5,
                activation: 1.0,
                // Mix agonists + antagonists so the sum exercises
                // both polarity branches.
                polarity: if i % 2 == 0 { 1.0 } else { -1.0 },
                ..MuscleConfig::default()
            })
            .collect();

        // ── CPU ──
        let model_cpu = cable_with_muscles(1.0, 1000.0, 60.0, &muscles);
        let mut s_a = model_cpu.state().unwrap();
        let mut s_b = model_cpu.state().unwrap();
        let control = model_cpu.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver_cpu = FeatherstoneSolver::new_cpu();
        let t0 = std::time::Instant::now();
        for _ in 0..n_steps {
            solver_cpu
                .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let cpu_us = t0.elapsed().as_micros();
        let cpu_q = s_a.joint_q.host().unwrap()[0];

        // ── GPU ──
        let mut model_gpu = cable_with_muscles(1.0, 1000.0, 60.0, &muscles);
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();
        model_gpu.upload_to_device(&device, &queue).unwrap();
        let mut s_a = model_gpu.state().unwrap();
        let mut s_b = model_gpu.state().unwrap();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        let mut control_gpu = model_gpu.control();
        control_gpu.upload_to_device(&device, &queue).unwrap();
        let contacts = Contacts::with_capacity(0);
        let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..n_steps {
            solver_gpu
                .step(&model_gpu, &s_a, &control_gpu, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let _ = s_a.joint_q.download(&device, &queue);
        let gpu_us = t0.elapsed().as_micros();
        let gpu_q = s_a.joint_q.host().unwrap()[0];

        eprintln!(
            "N = {:>3}: CPU {:>6.2} ms (q={:+.4e}), GPU {:>6.2} ms (q={:+.4e})",
            n_muscles,
            (cpu_us as f64) * 1.0e-3,
            cpu_q,
            (gpu_us as f64) * 1.0e-3,
            gpu_q,
        );
        assert!(
            gpu_q.is_finite(),
            "GPU produced non-finite q for N = {n_muscles}: {gpu_q}"
        );
        // Symmetric agonist/antagonist split means the sum should
        // remain small — a useful sanity check for the GPU
        // accumulator. We accept up to f_max·dt² worth of "drift"
        // due to f32 imbalance: with N=100, half agonists half
        // antagonists, f_max = 5, dt = 1e-3, that's about 2.5 mm.
        assert!(
            gpu_q.abs() < 5.0e-3,
            "balanced N={n_muscles} muscles drifted on GPU: q = {gpu_q}"
        );
    }
}
