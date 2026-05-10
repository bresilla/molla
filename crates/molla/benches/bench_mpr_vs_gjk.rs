//! MPR vs GJK+EPA on a deeply penetrating box pair.
//!
//! ```bash
//! cargo bench -p molla --bench bench_mpr_vs_gjk
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::geometry::{epa, gjk, mpr_collide, GjkResult, ShapeSupport};
use molla::math::{Quat, Real, Transform, Vec3};
use molla::sim::ShapeGeometry;

fn box_geom(half: Real) -> ShapeGeometry {
    ShapeGeometry::Box {
        half_extents: Vec3::splat(half),
    }
}

fn bench_mpr_vs_gjk(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpr_vs_gjk_deep_penetration");
    group.sample_size(20).measurement_time(Duration::from_secs(10));

    // Deep penetration: two unit cubes overlapping by 50% along X with
    // a small rotation so the contact features are non-trivial.
    let geom_a = box_geom(0.5);
    let geom_b = box_geom(0.5);
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform {
        position: Vec3::new(0.5, 0.0, 0.0),
        rotation: Quat::from_rotation_y(0.3),
    };

    group.bench_with_input(BenchmarkId::from_parameter("mpr"), &(), |bencher, _| {
        let support_a = ShapeSupport::new(geom_a, &xf_a);
        let support_b = ShapeSupport::new(geom_b, &xf_b);
        bencher.iter(|| {
            let m = mpr_collide(black_box(&support_a), black_box(&support_b), 1e-6, 64);
            black_box(m.is_some())
        });
    });
    group.bench_with_input(BenchmarkId::from_parameter("gjk_epa"), &(), |bencher, _| {
        bencher.iter(|| {
            match gjk(geom_a, &xf_a, geom_b, &xf_b) {
                GjkResult::Overlap(simplex) => {
                    let r = epa(geom_a, &xf_a, geom_b, &xf_b, &simplex);
                    black_box(r.is_some())
                }
                GjkResult::Disjoint(_) => black_box(false),
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_mpr_vs_gjk);
criterion_main!(benches);
