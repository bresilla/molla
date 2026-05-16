//! VBD cloth N×N grid — CPU vs GPU side-by-side, 30 substeps.
//!
//! ```bash
//! cargo bench -p molla --bench bench_vbd_cloth_cpu_vs_gpu --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::{Backend, Engine};
use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};

const SUBSTEPS: usize = 30;
const DT: Real = 1.0 / 240.0;

fn build_cloth(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let (first, _, _) = b.add_cloth_grid(
        Vec3::new(-0.5, 1.0, -0.5),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        n,
        n,
        0.05,
        0.02,
        1.0e-5,
        0.05,
        WorldId(0),
    );
    for ix in 0..n {
        b.pin_particle(first + ix);
    }
    b.build().unwrap()
}

fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("vbd_cloth_cpu_vs_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));

    for &n in &[16u32, 32, 64] {
        // ── CPU ────────────────────────────────────────────────────
        {
            let model = build_cloth(n);
            let mut s_a = model.state().unwrap();
            let mut s_b = model.state().unwrap();
            let control = model.control();
            let contacts = Contacts::with_capacity(0);
            let mut solver = VbdSolver::new_cpu().with_iterations(8);
            group.bench_function(BenchmarkId::new("cpu", n), |bencher| {
                bencher.iter(|| {
                    for _ in 0..SUBSTEPS {
                        Solver::step(
                            &mut solver,
                            black_box(&model),
                            black_box(&s_a),
                            black_box(&control),
                            black_box(&contacts),
                            black_box(DT),
                            &mut s_b,
                        )
                        .unwrap();
                        std::mem::swap(&mut s_a, &mut s_b);
                    }
                });
            });
        }

        // ── GPU ────────────────────────────────────────────────────
        {
            let mut model = build_cloth(n);
            model.upload_to_device(&device, &queue).unwrap();
            let mut s_a = model.state().unwrap();
            let mut s_b = model.state().unwrap();
            s_a.upload_to_device(&device, &queue).unwrap();
            s_b.upload_to_device(&device, &queue).unwrap();
            let mut control = model.control();
            control.upload_to_device(&device, &queue).unwrap();
            let contacts = Contacts::with_capacity(0);
            let mut solver = VbdSolver::new_gpu(&engine, &model)
                .expect("VbdSolver::new_gpu")
                .with_iterations(8);
            group.bench_function(BenchmarkId::new("gpu", n), |bencher| {
                bencher.iter(|| {
                    for _ in 0..SUBSTEPS {
                        Solver::step(
                            &mut solver,
                            black_box(&model),
                            black_box(&s_a),
                            black_box(&control),
                            black_box(&contacts),
                            black_box(DT),
                            &mut s_b,
                        )
                        .unwrap();
                        std::mem::swap(&mut s_a, &mut s_b);
                    }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cpu_vs_gpu);
criterion_main!(benches);
