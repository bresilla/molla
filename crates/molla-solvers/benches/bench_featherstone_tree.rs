//! Featherstone forward dynamics on a balanced binary tree of revolute
//! joints, depth ∈ {3, 5, 7}. CPU only — exercises the multi-child
//! traversal path that the chain bench misses.
//!
//! ```bash
//! cargo bench -p molla --bench bench_featherstone_tree
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

const DT: Real = 0.005;

fn build_tree_recursive(b: &mut ModelBuilder, parent: BodyId, depth: usize, max_depth: usize) {
    if depth == max_depth {
        return;
    }
    for i in 0..2 {
        let child = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
        let dx = if i == 0 { -0.2 } else { 0.2 };
        b.add_joint_revolute(
            parent,
            child,
            Transform::from_translation(if parent == BodyId::NONE {
                V3::ZERO
            } else {
                V3::new(dx, -0.5, 0.0)
            }),
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        build_tree_recursive(b, child, depth + 1, max_depth);
    }
}

fn build_tree(depth: usize) -> Model {
    let mut b = ModelBuilder::new();
    build_tree_recursive(&mut b, BodyId::NONE, 0, depth);
    b.build().unwrap()
}

fn bench_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("featherstone_tree_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &depth in &[3usize, 5, 7] {
        let model = build_tree(depth);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu();
        let label = format!("depth={depth}_links={}", model.body_count);
        group.bench_with_input(BenchmarkId::from_parameter(label), &depth, |bencher, _| {
            bencher.iter(|| {
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
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tree);
criterion_main!(benches);
