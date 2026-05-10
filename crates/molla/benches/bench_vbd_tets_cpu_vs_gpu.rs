//! VBD tet jelly cube — CPU vs GPU side-by-side, 30 substeps.
//!
//! ```bash
//! cargo bench -p molla --bench bench_vbd_tets_cpu_vs_gpu --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::core::WorldId;
use molla::prelude::*;

const SUBSTEPS: usize = 30;
const DT: Real = 1.0 / 240.0;

fn cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    [
        [c[0], c[1], c[3], c[7]],
        [c[0], c[3], c[2], c[7]],
        [c[0], c[2], c[6], c[7]],
        [c[0], c[6], c[4], c[7]],
        [c[0], c[4], c[5], c[7]],
        [c[0], c[5], c[1], c[7]],
    ]
}

fn build_jelly(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let dx: Real = 0.10;
    let origin = Vec3::new(
        -dx * (n as Real - 1.0) * 0.5,
        1.0,
        -dx * (n as Real - 1.0) * 0.5,
    );
    let first = 0u32;
    for kk in 0..n {
        for jj in 0..n {
            for ii in 0..n {
                let p = origin
                    + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { first + (k * n + j) * n + i };
    let mut tets = Vec::new();
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let c = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(c));
            }
        }
    }
    b.add_tet_mesh(&tets, 2.0e5, 0.3, None).unwrap();
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

    let mut group = c.benchmark_group("vbd_tets_cpu_vs_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    for &n in &[3u32, 5, 8] {
        // ── CPU ────────────────────────────────────────────────────
        {
            let model = build_jelly(n);
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
            let mut model = build_jelly(n);
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
