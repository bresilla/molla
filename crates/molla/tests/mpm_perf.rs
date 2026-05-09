//! Quick perf measurement for MPM v1.
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
    eprintln!("MPM perf: {} particles, 32^3 grid: {:.1} Hz ({:.2} ms/step)",
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
    eprintln!("MPM perf: {} particles, 64^3 grid: {:.1} Hz ({:.2} ms/step)",
              particles.len(), hz, dt * 1000.0 / n as f64);
}
