//! Stage 33.1 — soft edge-edge contact gather. Two cloths crossing
//! each other so the edges are within the contact margin.
//!
//! ```bash
//! cargo bench -p molla --bench bench_soft_ee_contact
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::WorldId;
use molla_geometry::SoftBvh;
use molla_math::{Real, Vec3};
use molla_sim::{Model, ModelBuilder};
use molla_solvers::gather_ee_contacts;

const MARGIN: Real = 0.05;

fn build_two_cloths(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    // Cloth A in XZ plane.
    let (a_first, _, _) = b.add_cloth_grid(
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
    );
    // Cloth B in XY plane, intersecting cloth A.
    let (_b_first, _, _) = b.add_cloth_grid(
        Vec3::new(-0.5, 0.5, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        n,
        n,
        0.05,
        0.02,
        1.0e-5,
        0.05,
        WorldId(0),
    );
    let _ = a_first;
    b.build().unwrap()
}

fn bench_ee(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_ee_contact_cpu");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(15));
    for &n in &[16u32, 32, 64] {
        let model = build_two_cloths(n);
        let state = model.state().unwrap();
        let bvh = SoftBvh::build(&model, &state);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let contacts = gather_ee_contacts(
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

criterion_group!(benches, bench_ee);
criterion_main!(benches);
