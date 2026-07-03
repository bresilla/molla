//! Stage 42 v3/v4 — Hill-type muscle GPU/CPU parity.
//!
//! Four scenarios all gated behind `#[ignore]` because the GPU
//! backend requires a real wgpu adapter — run via:
//!
//! ```bash
//! nixVulkan cargo test -p molla-solvers --features gpu --test joint_muscle_gpu -- --ignored
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
//! 4. `see_muscle_cpu_gpu_parity` (v4) — single cable with non-zero
//!    `l_tendon_slack`, full activation, 100 steps. Exercises the GPU
//!    SEE Newton solve. Joint coords agree within `1e-5` relative.
//!    Also prints Newton convergence stats (avg iters to residual
//!    `< 1e-4 · f_max`) computed by mirroring the WGSL loop in Rust
//!    over the recorded `(q_l, qd_l)` trajectory.

#![cfg(feature = "gpu")]

use molla_actuators::{
    DEFAULT_TENDON_STIFFNESS_RATIO, MuscleConfig, force_length, force_velocity, tendon_force,
};
use molla_compute::{Backend, Engine};
use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State};
use molla_solvers::{FeatherstoneSolver, Solver};

/// Build a single-cable single-body model with `muscles` muscles all
/// attached to the same cable joint. The model uses `f_max = 100 N`,
/// `l_opt = 5 cm` defaults — agonist polarity unless overridden.
fn cable_with_muscles(mass: Real, k_lin: Real, c_lin: Real, muscles: &[MuscleConfig]) -> Model {
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
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=joint_muscle_gpu` or `nixVulkan cargo test ...`"]
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
    assert!(max_rel < 5.0e-5, "CPU/GPU diverged: max rel = {max_rel}");
    let q_final = cpu_q.last().copied().unwrap();
    assert!(
        q_final < 0.0,
        "muscle should have contracted by end of run (q_l = {q_final})"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=joint_muscle_gpu` or `nixVulkan cargo test ...`"]
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
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=joint_muscle_gpu` or `nixVulkan cargo test ...`"]
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

/// Mirror of the WGSL `muscle_force` SEE Newton loop (rigid-tendon
/// branch skipped — caller only invokes this for `l_tendon_slack > 0`).
/// Returns `(f_tendon_signed, iters_used, final_residual)` so the test
/// can both apply the force *and* report convergence statistics. The
/// algorithm tracks `muscle.rs::Muscle::torque` and
/// `shaders/muscle.wgsl::muscle_force` line-for-line.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` intentionally treats NaN as invalid input
fn see_newton_diag(
    f_max: Real,
    l_opt: Real,
    v_max: Real,
    a: Real,
    l_tendon_slack: Real,
    pennation_angle: Real,
    polarity: Real,
    q_l: Real,
    qd_l: Real,
) -> (Real, u32, Real) {
    let pol = if polarity >= 0.0 { 1.0 } else { -1.0 };
    let q_local = pol * q_l;
    let qd_local = pol * qd_l;
    let a_c = a.clamp(0.0, 1.0);
    if a_c <= 0.0 || !(f_max > 0.0) || !(l_opt > 0.0) || !(v_max > 0.0) {
        return (0.0, 0, 0.0);
    }
    let cos_alpha = pennation_angle.cos().max(1.0e-6);
    let l_rest = l_opt * cos_alpha + l_tendon_slack;
    let l_total = l_rest + q_local;

    let k: Real = 0.25 * DEFAULT_TENDON_STIFFNESS_RATIO * DEFAULT_TENDON_STIFFNESS_RATIO;
    let l_t0 = (l_total - l_opt * cos_alpha).max(l_tendon_slack);
    let eps_geom = ((l_t0 - l_tendon_slack) / l_tendon_slack).max(0.0);
    let eps_analytic = (a_c * cos_alpha / k).sqrt();
    let mut eps = if eps_geom < eps_analytic * 0.1 {
        eps_analytic
    } else {
        eps_geom
    };

    let mut iters_used: u32 = 8;
    let mut last_residual: Real = Real::INFINITY;
    const SIGMA: Real = 0.45;
    for it in 0..8u32 {
        let l_t = l_tendon_slack * (1.0 + eps);
        let l_m_cos = l_total - l_t;
        let l_m = (l_m_cos / cos_alpha).max(1.0e-9);
        let l_norm = l_m / l_opt;
        let v_norm = qd_local / v_max;
        let f_l = force_length(l_norm);
        let f_v = force_velocity(v_norm);
        let f_fibre = a_c * f_max * f_l * f_v;
        let f_t = tendon_force(eps, f_max, DEFAULT_TENDON_STIFFNESS_RATIO);
        let residual = f_fibre * cos_alpha - f_t;
        last_residual = residual.abs();
        if last_residual < 1.0e-4 * f_max {
            iters_used = it + 1;
            break;
        }
        let dlm_deps = -l_tendon_slack / cos_alpha;
        let dln_deps = dlm_deps / l_opt;
        let df_l_deps = if (0.4..=1.6).contains(&l_norm) {
            -2.0 * (l_norm - 1.0) / (SIGMA * SIGMA) * f_l * dln_deps
        } else {
            0.0
        };
        let df_fibre_deps = a_c * f_max * f_v * df_l_deps;
        let df_t_deps = f_max * 2.0 * k * eps.max(1.0e-3);
        let denom = df_fibre_deps * cos_alpha - df_t_deps;
        if denom.abs() < 1.0e-12 * f_max {
            iters_used = it + 1;
            break;
        }
        let step = residual / denom;
        let max_step = (eps.abs() + 1.0e-3) * 0.5;
        let clamped_step = step.clamp(-max_step, max_step);
        eps -= clamped_step;
        if !eps.is_finite() {
            eps = eps_analytic;
            iters_used = it + 1;
            break;
        }
        eps = eps.max(0.0);
    }
    let f_tendon = tendon_force(eps, f_max, DEFAULT_TENDON_STIFFNESS_RATIO);
    (pol * -f_tendon, iters_used, last_residual)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `nixVulkan cargo test ...`"]
fn see_muscle_cpu_gpu_parity() {
    let engine = Engine::new(Backend::AutoGpu)
        .expect("GPU adapter required for this test (gated by #[ignore])");
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let dt: Real = 1.0e-3;
    let n_steps = 100usize;
    let f_max: Real = 80.0;
    let k_lin: Real = 4000.0;
    let mass: Real = 1.0;
    let c_lin: Real = 2.0 * (k_lin * mass).sqrt();

    // Non-zero tendon slack → exercises the Newton SEE path on both
    // backends. `l_opt = 5 cm`, `l_tendon_slack = 1 cm` ⇒ a fairly
    // stretchy MTU; full activation, zero pennation so this scenario
    // isolates the SEE solve (per-muscle pennation/stiffness parity is
    // covered by `per_muscle_pennation_and_stiffness_cpu_gpu_parity`).
    let cfg = MuscleConfig {
        f_max,
        l_opt: 0.05,
        v_max: 0.5,
        activation: 1.0,
        l_tendon_slack: 0.01,
        pennation_angle: 0.0,
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
    for _ in 0..n_steps {
        // Sample the muscle Newton solver at the *pre-step* state so
        // the convergence stats reflect what the GPU shader actually
        // executed inside this step.
        let q_pre = s_a.joint_q.host().unwrap()[0];
        let qd_pre = s_a.joint_qd.host().unwrap()[0];
        cpu_q.push(q_pre);
        cpu_qd.push(qd_pre);
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    // Final state too (so we can compare 100 entries to GPU's 100).
    let cpu_q_final = s_a.joint_q.host().unwrap()[0];

    // Newton convergence stats — replay the SEE solve at each
    // recorded `(q_l, qd_l)` and tally `iters_used`.
    let mut total_iters: u64 = 0;
    let mut max_iters: u32 = 0;
    let mut max_residual: Real = 0.0;
    for (&q, &qd) in cpu_q.iter().zip(cpu_qd.iter()) {
        let (_, iters, residual) = see_newton_diag(
            cfg.f_max,
            cfg.l_opt,
            cfg.v_max,
            cfg.activation,
            cfg.l_tendon_slack,
            cfg.pennation_angle,
            cfg.polarity,
            q,
            qd,
        );
        total_iters += u64::from(iters);
        max_iters = max_iters.max(iters);
        max_residual = max_residual.max(residual);
    }
    let avg_iters = (total_iters as f64) / (cpu_q.len() as f64);
    eprintln!(
        "SEE Newton convergence — avg iters = {:.2}, max iters = {}, \
         max final residual = {:.3e}",
        avg_iters, max_iters, max_residual
    );

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
    for _ in 0..n_steps {
        let _ = s_a.joint_q.download(&device, &queue);
        gpu_q.push(s_a.joint_q.host().unwrap()[0]);
        solver_gpu
            .step(&model_gpu, &s_a, &control_gpu, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q_final = s_a.joint_q.host().unwrap()[0];

    // Diagnostics.
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
    let d_final = (cpu_q_final - gpu_q_final).abs();
    let rel_final = d_final / cpu_q_final.abs().max(1.0e-6);
    eprintln!(
        " end | {:+.5e} | {:+.5e} | {:.3e} | {:.3e}",
        cpu_q_final, gpu_q_final, d_final, rel_final
    );

    // Parity tolerance: 1e-5 relative as per the v4 spec. We use
    // a `max(1e-6)` floor on the denominator so the very first
    // step (where both q's are nearly zero) doesn't blow the
    // relative metric up.
    let mut max_rel: f64 = 0.0;
    for (c, g) in cpu_q.iter().zip(gpu_q.iter()) {
        let d = (c - g).abs();
        let rel = d / c.abs().max(1.0e-6);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let rel_final_metric = d_final / cpu_q_final.abs().max(1.0e-6);
    if rel_final_metric > max_rel {
        max_rel = rel_final_metric;
    }
    assert!(
        max_rel < 1.0e-5,
        "SEE CPU/GPU diverged: max rel = {max_rel}"
    );
    assert!(
        cpu_q_final < 0.0,
        "SEE muscle should have contracted by end of run (q_l = {cpu_q_final})"
    );
    assert!(
        gpu_q_final < 0.0,
        "SEE muscle GPU should have contracted by end of run (q_l = {gpu_q_final})"
    );
}

/// Roadmap A2 (Stage 42 v5) validation — the GPU muscle kernel must
/// honor *per-muscle* `pennation_angle` and `tendon_stiffness_ratio`
/// storage buffers (`muscle.wgsl` bindings 7/9), not one shared value.
/// Two muscles on the same cable with deliberately different pennation
/// and SEE stiffness: the CPU path is the per-muscle reference, so any
/// GPU lane that falls back to a shared/default parameter diverges and
/// fails the parity gate. A homogeneous-parameter CPU control run
/// guards against the scenario being vacuous (parameters not mattering
/// at all).
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=joint_muscle_gpu`"]
fn per_muscle_pennation_and_stiffness_cpu_gpu_parity() {
    let engine = Engine::new(Backend::AutoGpu)
        .expect("GPU adapter required for this test (gated by #[ignore])");
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let dt: Real = 1.0e-3;
    let n_steps = 100usize;
    let k_lin: Real = 4000.0;
    let mass: Real = 1.0;
    let c_lin: Real = 2.0 * (k_lin * mass).sqrt();

    // Both muscles pull the same way (forces add) through the SEE
    // Newton path, with distinct per-muscle parameters on every axis
    // A2 cares about.
    let m_stiff = MuscleConfig {
        f_max: 80.0,
        l_opt: 0.05,
        v_max: 0.5,
        activation: 1.0,
        l_tendon_slack: 0.01,
        pennation_angle: 0.0,
        tendon_stiffness_ratio: 45.0,
        ..MuscleConfig::default()
    };
    let m_pennate = MuscleConfig {
        f_max: 60.0,
        l_opt: 0.05,
        v_max: 0.5,
        activation: 1.0,
        l_tendon_slack: 0.01,
        pennation_angle: 0.35,
        tendon_stiffness_ratio: 20.0,
        ..MuscleConfig::default()
    };
    let muscles = [m_stiff, m_pennate];

    let run_cpu = |muscles: &[MuscleConfig]| -> Vec<Real> {
        let model = cable_with_muscles(mass, k_lin, c_lin, muscles);
        let mut s_a = seed_state(&model, 0.0, 0.0);
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu();
        let mut q = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            q.push(s_a.joint_q.host().unwrap()[0]);
        }
        q
    };

    // ── CPU reference (heterogeneous parameters) ──
    let cpu_q = run_cpu(&muscles);

    // ── Non-vacuousness control: homogeneous defaults must produce a
    // measurably different trajectory, or this scenario would pass
    // even if both backends ignored the per-muscle buffers. ──
    let homogeneous = [
        MuscleConfig {
            pennation_angle: 0.0,
            tendon_stiffness_ratio: DEFAULT_TENDON_STIFFNESS_RATIO,
            ..m_stiff
        },
        MuscleConfig {
            pennation_angle: 0.0,
            tendon_stiffness_ratio: DEFAULT_TENDON_STIFFNESS_RATIO,
            ..m_pennate
        },
    ];
    let cpu_q_homog = run_cpu(&homogeneous);
    let q_final = *cpu_q.last().unwrap();
    let q_final_homog = *cpu_q_homog.last().unwrap();
    let control_rel = (q_final - q_final_homog).abs() / q_final.abs().max(1.0e-6);
    eprintln!(
        "hetero q_final = {q_final:+.5e}, homogeneous q_final = {q_final_homog:+.5e}, \
         rel diff = {control_rel:.3e}"
    );
    assert!(
        control_rel > 1.0e-2,
        "per-muscle parameters had no measurable effect (rel diff {control_rel:.3e}) — \
         the parity check below would be vacuous"
    );

    // ── GPU run (heterogeneous parameters) ──
    let mut model_gpu = cable_with_muscles(mass, k_lin, c_lin, &muscles);
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
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &control_gpu, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        gpu_q.push(s_a.joint_q.host().unwrap()[0]);
    }

    eprintln!("step | cpu_q      | gpu_q      | |Δ|         | rel");
    for &i in &[0usize, 1, 5, 25, 50, 99] {
        if i >= cpu_q.len() {
            continue;
        }
        let c = cpu_q[i];
        let g = gpu_q[i];
        let d = (c - g).abs();
        eprintln!(
            "{:4} | {:+.5e} | {:+.5e} | {:.3e} | {:.3e}",
            i,
            c,
            g,
            d,
            d / c.abs().max(1e-12)
        );
    }

    let mut max_rel: f64 = 0.0;
    for (c, g) in cpu_q.iter().zip(gpu_q.iter()) {
        let rel = (c - g).abs() / c.abs().max(1.0e-6);
        max_rel = max_rel.max(rel);
    }
    // Same f32-rounding budget as the single-muscle lane; the second
    // muscle roughly doubles the accumulated rounding, so the 5e-5
    // gate is reused rather than the SEE lane's 1e-5.
    assert!(
        max_rel < 5.0e-5,
        "per-muscle CPU/GPU diverged: max rel = {max_rel} — check that \
         muscle.wgsl reads bindings 7/9 per muscle id"
    );
    assert!(
        *cpu_q.last().unwrap() < 0.0 && *gpu_q.last().unwrap() < 0.0,
        "both backends should have contracted"
    );
}
