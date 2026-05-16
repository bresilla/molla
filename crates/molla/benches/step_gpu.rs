//! Stage 2 GPU benchmark: cost per `SemiImplicitSolver::step` on the GPU.
//!
//! ```bash
//! make bench-gpu
//! # or:
//! cargo bench -p molla --bench step_gpu --features gpu
//! ```
//!
//! The DoD for stage 2 is 1k spheres at >= 1 kHz on a 2020-era discrete
//! GPU, i.e. < 1 ms per step.

#![cfg(feature = "gpu")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla::compute::{Backend, Engine};
use molla::prelude::*;

fn build_scene(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    for i in 0..n {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(i as f64 * 1.5, 10.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    }
    b.build().unwrap()
}

fn bench_step(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("semi_implicit_step_gpu");
    for &n in &[100usize, 1_000, 10_000] {
        let mut model = build_scene(n);
        model.upload_to_device(&device, &queue).unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let mut control = model.control();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        control.upload_to_device(&device, &queue).unwrap();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new_gpu(&engine).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                solver
                    .step(
                        black_box(&model),
                        black_box(&s_a),
                        black_box(&control),
                        black_box(&contacts),
                        black_box(0.01),
                        &mut s_b,
                    )
                    .unwrap();
                std::mem::swap(&mut s_a, &mut s_b);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
