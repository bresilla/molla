//! N×N×N box pyramid settling on a ground plane. CPU XPBD.
//! Stresses persistent box-vs-box manifolds and stacked impulses.
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_stacking
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

const SUBSTEPS: usize = 30;
const DT: Real = 1.0 / 240.0;

fn build_stack(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.1);
    b.set_linear_damping(0.05);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let half = Vec3::splat(0.25);
    let spacing: Real = 0.55;
    let inertia = SpatialInertia::cuboid(1.0, half * 2.0);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let p = Vec3::new(
                    ix as Real * spacing,
                    half.y + iy as Real * spacing,
                    iz as Real * spacing,
                );
                let body = b.add_body(
                    BodyParams::new()
                        .with_spatial_inertia(inertia)
                        .with_position(p),
                );
                b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
            }
        }
    }
    b.build().unwrap()
}

fn bench_stacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("xpbd_stacking_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &n in &[4usize, 8, 16] {
        let model = build_stack(n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let mut contacts = Contacts::with_capacity(model.body_count * 4);
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

criterion_group!(benches, bench_stacking);
criterion_main!(benches);
