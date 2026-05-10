//! `TiledCamera` 5-channel raytrace at 256×256, 512×512, and 720p.
//! GPU only.
//!
//! ```bash
//! cargo bench -p molla --bench bench_sensor_camera --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Real, SpatialInertia, Transform, Vec3};
use molla::prelude::*;
use molla::sensors::{TiledCamera, TiledCameraConfig};

fn build_scene() -> Model {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let r: Real = 0.5;
    for ix in 0..6 {
        for iz in 0..6 {
            let p = Vec3::new(
                ix as Real - 3.0,
                0.5,
                iz as Real - 3.0,
            );
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia::sphere(1.0, r))
                    .with_position(p),
            );
            b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
        }
    }
    b.build().unwrap()
}

fn bench_camera(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_sensor_camera — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut model = build_scene();
    model.upload_to_device(&device, &queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(&device, &queue).unwrap();

    let mut group = c.benchmark_group("sensor_camera_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    for &(w, h, label) in &[
        (256u32, 256u32, "256x256"),
        (512, 512, "512x512"),
        (1280, 720, "720p"),
    ] {
        let cfg = TiledCameraConfig {
            width: w,
            height: h,
            ..Default::default()
        };
        let mut camera = TiledCamera::new(&engine, cfg, 1).expect("TiledCamera::new");
        camera
            .set_camera_look_at(0, Vec3::new(0.0, 5.0, 8.0), Vec3::ZERO, Vec3::Y)
            .unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &w, |bencher, _| {
            bencher.iter(|| {
                camera
                    .dispatch(black_box(&engine), black_box(&model), black_box(&state))
                    .expect("dispatch");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_camera);
criterion_main!(benches);
