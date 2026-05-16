//! Stage 19 — `find_pairs` over an N-body rigid sphere pile (full
//! GPU LBVH dispatch including pair emission).
//!
//! ```bash
//! cargo bench -p molla --bench bench_broadphase_pairs --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_geometry::LbvhBroadPhase;
use molla_math::{Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Model, ModelBuilder};

fn build_pile(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    b.set_linear_damping(0.0);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let side = ((n as f64).cbrt().ceil() as usize).max(1);
    let r: Real = 0.25;
    let spacing: Real = r * 1.5; // dense overlap
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

fn bench_pairs(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_broadphase_pairs — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("broadphase_find_pairs_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    for &n in &[1_000usize, 10_000, 100_000] {
        let mut model = build_pile(n);
        model.upload_to_device(&device, &queue).unwrap();
        let state = model.state().unwrap();
        let mut lbvh = LbvhBroadPhase::new(
            device.clone(),
            queue.clone(),
            (model.shape_count + 1).next_power_of_two().max(2048),
            (n * 16).max(32_768),
        )
        .expect("LbvhBroadPhase::new");
        // Seed topology so the per-iter run takes the cached path.
        lbvh.dispatch_from_model(&model, &state, model.contact_margin)
            .expect("seed");

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let pairs = lbvh
                    .refit_from_model(black_box(&model), black_box(&state), model.contact_margin)
                    .expect("refit_from_model");
                black_box(pairs.a.len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pairs);
criterion_main!(benches);
