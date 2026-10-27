//! Pile of N falling spheres + plane, run 60 substeps per sample.
//! CPU XPBD. Stresses both the constraint projection inner loop and
//! the contact-pipeline narrow-phase against a single ground plane.
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_pile
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

const SUBSTEPS: usize = 60;
const DT: Real = 1.0 / 240.0;

fn build_pile(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.05);
    // Ground plane.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    // Pile particles arranged on a cube grid above the plane.
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

fn bench_pile(c: &mut Criterion) {
    let mut group = c.benchmark_group("xpbd_pile_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &n in &[100usize, 1_000, 10_000, 50_000] {
        let model = build_pile(n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let mut contacts = Contacts::with_capacity(n * 2);
        let mut pipeline = CollisionPipeline::new();
        let mut solver = XpbdSolver::new_cpu().with_iterations(2).with_substeps(1);

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

criterion_group!(benches, bench_pile);
criterion_main!(benches);
