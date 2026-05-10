//! N-particle cable falling onto a plane via XPBD distance + bending.
//!
//! ```bash
//! cargo bench -p molla --bench bench_cable_drop
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::core::{BodyId, WorldId};
use molla::geometry::CollisionPipeline;
use molla::prelude::*;

const SUBSTEPS: usize = 30;
const DT: Real = 1.0 / 240.0;

fn build_cable(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let length: Real = (n as Real) * 0.05;
    let (_first, _, cable_id) = b.add_cable_polyline(
        Vec3::new(-length * 0.5, 2.0, 0.0),
        Vec3::new(length * 0.5, 2.0, 0.0),
        n,
        0.05,
        0.02,
        1.0e5,
        0.05,
        WorldId(0),
    );
    // Light bending stiffness so the cable resists straightening.
    b.set_cable_bending_stiffness(cable_id, 0.01);
    b.build().unwrap()
}

fn bench_cable(c: &mut Criterion) {
    let mut group = c.benchmark_group("cable_drop_cpu");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    for &n in &[32u32, 128, 512, 2048] {
        let model = build_cable(n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let mut contacts = Contacts::with_capacity(0);
        let mut pipeline = CollisionPipeline::new();
        let mut solver = XpbdSolver::new_cpu().with_iterations(4).with_substeps(1);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                for _ in 0..SUBSTEPS {
                    pipeline
                        .collide(black_box(&model), black_box(&s_a), &mut contacts)
                        .unwrap();
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

criterion_group!(benches, bench_cable);
criterion_main!(benches);
