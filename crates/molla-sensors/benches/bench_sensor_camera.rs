//! GPU camera completion/collection and homogeneous/heterogeneous batch benchmarks.
//!
//! ```bash
//! make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera'
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_math::{Real, SpatialInertia, Transform, Vec3};
use molla_sensors::{
    CameraBatch, CameraBatchLimits, CameraSpec, SensorCommandBatch, SensorFramePoolConfig,
    SensorScene, TiledCamera, TiledCameraConfig,
};
use molla_sim::{BodyParams, Model, ModelBuilder};

fn build_scene() -> Model {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let r: Real = 0.5;
    for ix in 0..6 {
        for iz in 0..6 {
            let p = Vec3::new(ix as Real - 3.0, 0.5, iz as Real - 3.0);
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
    let engine = Engine::new(Backend::AutoGpu).expect("sensor benchmark requires a GPU");
    eprintln!("sensor camera adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut model = build_scene();
    model.upload_to_device(&device, &queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(&device, &queue).unwrap();

    let mut group = c.benchmark_group("sensor_camera_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));

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

        group.bench_with_input(BenchmarkId::new("completed", label), &w, |bencher, _| {
            bencher.iter(|| {
                camera
                    .dispatch(black_box(&engine), black_box(&model), black_box(&state))
                    .expect("dispatch");
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU completion");
            });
        });
        group.bench_with_input(
            BenchmarkId::new("host_channels", label),
            &w,
            |bencher, _| {
                bencher.iter(|| {
                    camera
                        .dispatch(black_box(&engine), black_box(&model), black_box(&state))
                        .expect("dispatch");
                    black_box(camera.download_depth(&engine).expect("depth"));
                    black_box(camera.download_color(&engine).expect("color"));
                    black_box(camera.download_normal(&engine).expect("normal"));
                    black_box(camera.download_albedo(&engine).expect("albedo"));
                    black_box(camera.download_shape_index(&engine).expect("shape index"));
                });
            },
        );
    }
    group.finish();
}

fn bench_camera_batch(c: &mut Criterion) {
    let engine = Engine::new(Backend::AutoGpu).expect("sensor benchmark requires a GPU");
    eprintln!("sensor camera batch adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model = build_scene();
    model.upload_to_device(&device, &queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(&device, &queue).unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let complete = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("GPU completion");
    };
    let mut group = c.benchmark_group("sensor_camera_batch_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for heterogeneous in [false, true] {
        for count in [1, 4, 16] {
            let specs: Vec<_> = (0..count)
                .map(|i| {
                    let kind = if heterogeneous { i % 4 } else { 0 };
                    let (width, height) = [(128, 96), (256, 192), (320, 240), (160, 120)][kind];
                    let mut config = TiledCameraConfig {
                        width,
                        height,
                        ..Default::default()
                    };
                    if kind == 1 {
                        config.optics.lens_diameter_m = 0.03;
                        config.optics.focal_distance_m = 4.0;
                        config.optics.aperture_samples = 4;
                    }
                    CameraSpec {
                        sensor_id: (i as u64) << 32 | 7,
                        config,
                        pose: Transform::IDENTITY,
                    }
                })
                .collect();
            let mut cameras =
                CameraBatch::new(&engine, &specs, CameraBatchLimits::default()).unwrap();
            for (i, spec) in specs.iter().enumerate() {
                cameras
                    .set_camera_look_at(
                        spec.sensor_id,
                        Vec3::new(i as f64 * 0.05, 5.0, 8.0),
                        Vec3::ZERO,
                        Vec3::Y,
                    )
                    .unwrap();
            }
            let mut pool = cameras
                .output_pool(
                    &engine,
                    SensorFramePoolConfig {
                        slots: 1,
                        max_total_bytes: 64 << 20,
                    },
                )
                .unwrap();
            let layout = if heterogeneous {
                "heterogeneous"
            } else {
                "homogeneous"
            };
            let label = format!("{layout}_{count}");
            cameras.dispatch_scene(&engine, &scene, &state).unwrap();
            complete();
            group.bench_function(BenchmarkId::new("legacy_completed", &label), |bencher| {
                bencher.iter(|| {
                    let frame = cameras.dispatch_scene(&engine, &scene, &state).unwrap();
                    black_box(frame.stamp());
                    complete();
                });
            });
            for _ in 0..2 {
                let mut batch = SensorCommandBatch::new(&engine).unwrap();
                cameras
                    .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
                    .unwrap()
                    .expect("available output slots");
                let submitted = batch.submit().unwrap();
                complete();
                drop(submitted);
            }
            group.bench_function(BenchmarkId::new("direct_completed", &label), |bencher| {
                bencher.iter(|| {
                    let mut batch = SensorCommandBatch::new(&engine).unwrap();
                    let frame = cameras
                        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
                        .unwrap()
                        .expect("available output slots");
                    let submitted = batch.submit().unwrap();
                    black_box(&frame);
                    complete();
                    black_box(submitted);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_camera, bench_camera_batch);
criterion_main!(benches);
