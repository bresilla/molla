//! Stage 33.3 — soft self-collision gather. Cloth grid with self-
//! collision enabled, no folding required: BVH query is what dominates.
//!
//! ```bash
//! cargo bench -p molla --bench bench_soft_self_collision
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::WorldId;
use molla_geometry::SoftBvh;
use molla_math::{Real, Vec3};
use molla_sim::{Model, ModelBuilder};
use molla_solvers::gather_vt_contacts;

const MARGIN: Real = 0.05;

fn build_cloth_self(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let (_first, _, cloth_id) = (
        b.add_cloth_grid(
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
        )
        .0,
        (),
        0usize,
    );
    let _ = _first;
    let _ = cloth_id;
    // Enable self-collision on the freshly-added cloth (id 0).
    b.set_cloth_mesh_self_collision(0, true);
    b.build().unwrap()
}

fn bench_self(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_self_collision_cpu");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(15));
    for &n in &[16u32, 32, 64] {
        let model = build_cloth_self(n);
        let state = model.state().unwrap();
        let bvh = SoftBvh::build(&model, &state);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let contacts = gather_vt_contacts(
                    black_box(&model),
                    black_box(&state),
                    black_box(&bvh),
                    black_box(MARGIN),
                );
                black_box(contacts.len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_self);
criterion_main!(benches);
