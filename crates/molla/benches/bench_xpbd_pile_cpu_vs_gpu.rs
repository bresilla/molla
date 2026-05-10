//! XPBD rigid-body pile — CPU vs GPU side-by-side.
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_pile_cpu_vs_gpu --features gpu
//! ```
//!
//! Both backends step a falling pile of N spheres on a plane through
//! 60 substeps per measurement. Sizes chosen so the CPU side stays
//! tractable (caps at ~10k); the GPU side runs the same sizes plus a
//! 100k+ stress.

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::geometry::CollisionPipeline;
use molla::prelude::*;

const SUBSTEPS: usize = 60;
const DT: Real = 1.0 / 240.0;

fn build_pile(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.05);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let side = ((n as f64).cbrt().ceil() as usize).max(1);
    let r: Real = 0.25;
    let spacing: Real = r * 2.2;
    let mut placed = 0;
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if placed >= n {
                    break 'outer;
                }
                let p = Vec3::new(
                    ix as Real * spacing,
                    1.0 + iy as Real * spacing,
                    iz as Real * spacing,
                );
                let body = b.add_body(
                    BodyParams::new()
                        .with_spatial_inertia(SpatialInertia::sphere(1.0, r))
                        .with_position(p),
                );
                b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
                placed += 1;
            }
        }
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

    let mut group = c.benchmark_group("xpbd_pile_cpu_vs_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    for &n in &[100usize, 1_000, 10_000] {
        let model = build_pile(n);

        // ── CPU ────────────────────────────────────────────────────
        {
            let mut s_a = model.state().unwrap();
            let mut s_b = model.state().unwrap();
            let control = model.control();
            let mut contacts = Contacts::with_capacity((n * 4).max(1024));
            let mut pipeline = CollisionPipeline::new();
            pipeline.collide(&model, &s_a, &mut contacts).unwrap();
            let mut solver = XpbdSolver::new_cpu().with_iterations(2);
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
            let mut model = build_pile(n);
            model.upload_to_device(&device, &queue).unwrap();
            let mut s_a = model.state().unwrap();
            let mut s_b = model.state().unwrap();
            s_a.upload_to_device(&device, &queue).unwrap();
            s_b.upload_to_device(&device, &queue).unwrap();
            let mut control = model.control();
            control.upload_to_device(&device, &queue).unwrap();
            let mut contacts = Contacts::with_capacity((n * 4).clamp(1024, 256_000));
            let mut pipeline = CollisionPipeline::new();
            pipeline.collide(&model, &s_a, &mut contacts).unwrap();
            contacts.upload_to_device(&device, &queue).unwrap();
            let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
                .expect("XpbdSolver::new_gpu");
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
