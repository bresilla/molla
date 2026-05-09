//! Quick perf measurement for MPM v1.
//!
//! Stage 24d.2 added [`MpmSimulation::step_n`] (persistent-buffer
//! batched dispatcher) — the `mpm_step_n_beats_individual_steps`
//! test below asserts the empirical win.
#![cfg(feature = "gpu")]

use molla::compute::{Engine, MpmConfig, MpmSimulation};
use std::time::Instant;

#[test]
#[ignore = "perf measurement"]
fn mpm_perf_10k_32grid() {
    let Some(engine) = Engine::new_gpu().ok() else {
        eprintln!("no GPU");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.max_particles = 16384;
    let mut sim = MpmSimulation::new(&engine, cfg).expect("new");
    // 10k particles
    let mut particles = Vec::with_capacity(10000);
    let per_axis: i32 = 22; // 22^3 = 10648
    for ix in 0..per_axis {
        for iy in 0..per_axis {
            for iz in 0..per_axis {
                particles.push([
                    -0.3 + 0.027 * ix as f32,
                    0.2 + 0.027 * iy as f32,
                    -0.3 + 0.027 * iz as f32,
                ]);
                if particles.len() >= 10000 { break; }
            }
            if particles.len() >= 10000 { break; }
        }
        if particles.len() >= 10000 { break; }
    }
    sim.set_particles(&queue, &particles).expect("set");
    // Warmup
    for _ in 0..10 { sim.step(&engine, 1.0/240.0).expect("step"); }

    let n = 100;
    let t0 = Instant::now();
    for _ in 0..n {
        sim.step(&engine, 1.0/240.0).expect("step");
    }
    // Force submit completion
    let _ = sim.read_positions_with(&device, &queue).expect("read");
    let dt = t0.elapsed().as_secs_f64();
    let hz = n as f64 / dt;
    eprintln!("MPM perf (step×{n}):    {} particles, 32^3 grid: {:.1} Hz ({:.2} ms/step)",
              particles.len(), hz, dt * 1000.0 / n as f64);

    // Stage 24d.2 — same workload via batched dispatcher.  One submit,
    // not n.  Readback amortised at the end of the frame.
    let t0 = Instant::now();
    sim.step_n(&engine, 1.0/240.0, n as usize).expect("step_n");
    let _ = sim.read_positions_with(&device, &queue).expect("read");
    let dt = t0.elapsed().as_secs_f64();
    let hz = n as f64 / dt;
    eprintln!("MPM perf (step_n={n}):  {} particles, 32^3 grid: {:.1} Hz ({:.2} ms/step)",
              particles.len(), hz, dt * 1000.0 / n as f64);
}

#[test]
#[ignore = "perf measurement"]
fn mpm_perf_50k_64grid() {
    let Some(engine) = Engine::new_gpu().ok() else {
        eprintln!("no GPU");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 64;
    cfg.cell_size = 0.025;
    cfg.grid_origin = [-0.8, 0.0, -0.8];
    cfg.max_particles = 65536;
    let mut sim = MpmSimulation::new(&engine, cfg).expect("new");
    let mut particles = Vec::with_capacity(50000);
    let per_axis: i32 = 37; // 37^3 ~ 50k
    for ix in 0..per_axis {
        for iy in 0..per_axis {
            for iz in 0..per_axis {
                particles.push([
                    -0.3 + 0.016 * ix as f32,
                    0.2 + 0.016 * iy as f32,
                    -0.3 + 0.016 * iz as f32,
                ]);
                if particles.len() >= 50000 { break; }
            }
            if particles.len() >= 50000 { break; }
        }
        if particles.len() >= 50000 { break; }
    }
    sim.set_particles(&queue, &particles).expect("set");
    for _ in 0..5 { sim.step(&engine, 1.0/240.0).expect("step"); }

    let n = 30;
    let t0 = Instant::now();
    for _ in 0..n {
        sim.step(&engine, 1.0/240.0).expect("step");
    }
    let _ = sim.read_positions_with(&device, &queue).expect("read");
    let dt = t0.elapsed().as_secs_f64();
    let hz = n as f64 / dt;
    eprintln!("MPM perf (step×{n}):    {} particles, 64^3 grid: {:.1} Hz ({:.2} ms/step)",
              particles.len(), hz, dt * 1000.0 / n as f64);

    // Stage 24d.2 batched dispatcher.
    let t0 = Instant::now();
    sim.step_n(&engine, 1.0/240.0, n as usize).expect("step_n");
    let _ = sim.read_positions_with(&device, &queue).expect("read");
    let dt = t0.elapsed().as_secs_f64();
    let hz = n as f64 / dt;
    eprintln!("MPM perf (step_n={n}):  {} particles, 64^3 grid: {:.1} Hz ({:.2} ms/step)",
              particles.len(), hz, dt * 1000.0 / n as f64);
}

/// Stage 24d.2 — empirical proof that the batched dispatcher beats N
/// individual-step calls.  Asserts a ≥ 2× speedup at 60 substeps; we
/// typically observe 5-15× depending on the GPU's submit-queue latency.
///
/// The workload is held identical between the two paths (same
/// particles, same dt, same warmed-up sim).  We do a single readback
/// per phase so the frame-end sync isn't double-counted in either
/// timing.
#[test]
#[ignore = "perf measurement"]
fn mpm_step_n_beats_individual_steps() {
    let Some(engine) = Engine::new_gpu().ok() else {
        eprintln!("no GPU");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.max_particles = 16384;
    let mut sim = MpmSimulation::new(&engine, cfg).expect("new");

    let mut particles = Vec::with_capacity(8000);
    let per_axis: i32 = 20; // 20^3 = 8000
    for ix in 0..per_axis {
        for iy in 0..per_axis {
            for iz in 0..per_axis {
                particles.push([
                    -0.25 + 0.025 * ix as f32,
                    0.20 + 0.025 * iy as f32,
                    -0.25 + 0.025 * iz as f32,
                ]);
            }
        }
    }
    sim.set_particles(&queue, &particles).expect("set");

    // Warmup — exercise both code paths so the GPU has a chance to
    // compile pipelines and prime caches.
    for _ in 0..10 {
        sim.step(&engine, 1.0/240.0).expect("warmup step");
    }
    sim.step_n(&engine, 1.0/240.0, 10).expect("warmup step_n");
    let _ = sim.read_positions_with(&device, &queue).expect("warmup read");

    const SUBSTEPS: usize = 60;
    const REPEATS: usize = 3;

    // Phase A: SUBSTEPS calls to the legacy step() per repeat.
    let mut t_step_total = 0.0_f64;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        for _ in 0..SUBSTEPS {
            sim.step(&engine, 1.0/240.0).expect("step");
        }
        // One frame-end readback.
        let _ = sim.read_positions_with(&device, &queue).expect("read");
        t_step_total += t0.elapsed().as_secs_f64();
    }
    let t_step_avg = t_step_total / REPEATS as f64;

    // Phase B: one batched step_n(SUBSTEPS) per repeat.
    let mut t_stepn_total = 0.0_f64;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        sim.step_n(&engine, 1.0/240.0, SUBSTEPS).expect("step_n");
        let _ = sim.read_positions_with(&device, &queue).expect("read");
        t_stepn_total += t0.elapsed().as_secs_f64();
    }
    let t_stepn_avg = t_stepn_total / REPEATS as f64;

    let speedup = t_step_avg / t_stepn_avg.max(1.0e-9);
    eprintln!(
        "step×{SUBSTEPS} avg = {:.2} ms; step_n({SUBSTEPS}) avg = {:.2} ms; speedup = {:.2}×",
        t_step_avg * 1000.0,
        t_stepn_avg * 1000.0,
        speedup
    );
    assert!(
        speedup >= 2.0,
        "step_n was only {speedup:.2}× faster than {SUBSTEPS} individual steps; \
         expected ≥ 2× — investigate dispatcher batching"
    );
}
