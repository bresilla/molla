//! Pile of N falling spheres on GPU XPBD. Mirrors `bench_xpbd_pile`
//! but runs the solver on the GPU.
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_pile_gpu --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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

fn bench_pile_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_xpbd_pile_gpu — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("xpbd_pile_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));
    for &n in &[1_000usize, 10_000, 100_000] {
        let mut model = build_pile(n);
        model.upload_to_device(&device, &queue).unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        let mut control = model.control();
        control.upload_to_device(&device, &queue).unwrap();
        // Pre-allocate contacts + run a single collide pass so the
        // body-contact CSR (offsets/indices) gets sized to the model
        // before the GPU bind group is built. Without this the XPBD
        // GPU solver fails with "binding 6: no device mirror".
        let mut contacts = Contacts::with_capacity((n * 8).max(1024));
        let mut pipeline = CollisionPipeline::new();
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
            .expect("XpbdSolver::new_gpu");

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
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
    group.finish();
}

criterion_group!(benches, bench_pile_gpu);
criterion_main!(benches);
