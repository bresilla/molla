//! `pair_contact` per shape-vs-shape pair (sphere/box/capsule/cylinder/
//! plane). Each pair runs 1000 invocations per measurement so the
//! per-call cost is easy to read off the criterion graph.
//!
//! ```bash
//! cargo bench -p molla --bench bench_narrow_phase_pairs
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::core::BodyId;
use molla::geometry::{pair_contact, ContactManifold, ShapePair};
use molla::math::SpatialInertia;
use molla::prelude::*;

const REPEATS: usize = 1000;

/// Build a 2-body model with shapes `a` and `b` placed near the origin
/// so their AABBs overlap (we want the deep narrow-phase paths to
/// actually fire).
fn build_pair(
    add_a: impl FnOnce(&mut ModelBuilder, BodyId) -> ShapeId,
    add_b: impl FnOnce(&mut ModelBuilder, BodyId) -> ShapeId,
    pos_a: Vec3,
    pos_b: Vec3,
) -> Model {
    let mut b = ModelBuilder::new();
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(pos_a),
    );
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(pos_b),
    );
    let _ = add_a(&mut b, body_a);
    let _ = add_b(&mut b, body_b);
    b.build().unwrap()
}

fn run_pair(c: &mut Criterion, label: &str, model: Model) {
    let state = model.state().unwrap();
    let mut out: Vec<ContactManifold> = Vec::with_capacity(8);
    let pair = ShapePair { a: 0, b: 1 };
    let mut group = c.benchmark_group("narrow_phase_pair_contact");
    group.sample_size(20).measurement_time(Duration::from_secs(8));
    group.bench_with_input(BenchmarkId::from_parameter(label), &pair, |bencher, p| {
        bencher.iter(|| {
            for _ in 0..REPEATS {
                out.clear();
                pair_contact(
                    black_box(&model),
                    black_box(&state),
                    black_box(*p),
                    0.5,
                    0.0,
                    &mut out,
                );
            }
            black_box(out.len())
        });
    });
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    // sphere vs sphere
    let m = build_pair(
        |b, body| b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap(),
        |b, body| b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap(),
        Vec3::ZERO,
        Vec3::new(0.6, 0.0, 0.0),
    );
    run_pair(c, "sphere_sphere", m);

    // sphere vs box
    let m = build_pair(
        |b, body| b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap(),
        |b, body| {
            b.add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
                .unwrap()
        },
        Vec3::ZERO,
        Vec3::new(0.7, 0.0, 0.0),
    );
    run_pair(c, "sphere_box", m);

    // box vs box
    let m = build_pair(
        |b, body| {
            b.add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
                .unwrap()
        },
        |b, body| {
            b.add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
                .unwrap()
        },
        Vec3::ZERO,
        Vec3::new(0.7, 0.0, 0.0),
    );
    run_pair(c, "box_box", m);

    // box vs plane
    let mut bb = ModelBuilder::new();
    bb.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = bb.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 0.4, 0.0)),
    );
    bb.add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    run_pair(c, "plane_box", bb.build().unwrap());

    // capsule vs capsule
    let m = build_pair(
        |b, body| {
            b.add_shape_capsule(body, 0.4, 0.2, Transform::IDENTITY)
                .unwrap()
        },
        |b, body| {
            b.add_shape_capsule(body, 0.4, 0.2, Transform::IDENTITY)
                .unwrap()
        },
        Vec3::ZERO,
        Vec3::new(0.3, 0.0, 0.0),
    );
    run_pair(c, "capsule_capsule", m);

    // cylinder vs cylinder
    let m = build_pair(
        |b, body| {
            b.add_shape_cylinder(body, 0.4, 0.3, Transform::IDENTITY)
                .unwrap()
        },
        |b, body| {
            b.add_shape_cylinder(body, 0.4, 0.3, Transform::IDENTITY)
                .unwrap()
        },
        Vec3::ZERO,
        Vec3::new(0.4, 0.0, 0.0),
    );
    run_pair(c, "cylinder_cylinder", m);

    // sphere vs plane
    let mut bb = ModelBuilder::new();
    bb.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = bb.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 0.3, 0.0)),
    );
    bb.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    run_pair(c, "plane_sphere", bb.build().unwrap());
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
