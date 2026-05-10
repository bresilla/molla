//! Implicit-time-step MPM particles (Jacobi-PCG over the grid).
//! GPU only.
//!
//! ```bash
//! cargo bench -p molla --bench bench_mpm_implicit --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Engine, MpmConfig, MpmSimulation, MpmTimeStep};

const SUBSTEPS: usize = 30;
const DT: f32 = 1.0 / 240.0;

fn fill_particles(n: usize) -> Vec<[f32; 3]> {
    let per_axis = ((n as f64).cbrt().ceil() as i32).max(1);
    let mut out = Vec::with_capacity(n);
    let spacing = 0.025_f32;
    for ix in 0..per_axis {
        for iy in 0..per_axis {
            for iz in 0..per_axis {
                if out.len() >= n {
                    return out;
                }
                out.push([
                    -0.5 + spacing * ix as f32,
                    0.2 + spacing * iy as f32,
                    -0.5 + spacing * iz as f32,
                ]);
            }
        }
    }
    out
}

fn bench_mpm_implicit(c: &mut Criterion) {
    let engine = match Engine::new_gpu() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_mpm_implicit — no GPU: {e}");
            return;
        }
    };
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("mpm_implicit_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(25));
    for &n in &[1_000usize, 10_000, 50_000] {
        let particles = fill_particles(n);
        let mut cfg = MpmConfig::sand();
        cfg.grid_size = 64;
        cfg.cell_size = 0.025;
        cfg.grid_origin = [-0.8, 0.0, -0.8];
        cfg.max_particles = (particles.len() as u32).max(1024);
        cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 3 };
        let mut sim = MpmSimulation::new(&engine, cfg).expect("MpmSimulation::new");
        sim.set_particles(&queue, &particles).expect("set_particles");
        sim.step(&engine, DT).expect("warmup");

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                sim.step_n(black_box(&engine), black_box(DT), SUBSTEPS)
                    .expect("step_n");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mpm_implicit);
criterion_main!(benches);
