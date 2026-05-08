//! Stage 1 benchmark: cost per `SemiImplicitSolver::step` on the CPU.
//!
//! ```bash
//! cargo bench -p molla --bench step_cpu
//! ```
//!
//! The DoD for stage 1 is < 100µs per step for 100 spheres.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::prelude::*;

fn build_n_spheres(n: usize) -> Model {
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
    let mut group = c.benchmark_group("semi_implicit_step_cpu");
    for &n in &[1usize, 10, 100, 1_000] {
        let model = build_n_spheres(n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new();

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
