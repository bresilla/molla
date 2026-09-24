//! N rays into a scene of M shapes via the GPU `Raycaster`.
//!
//! ```bash
//! make bench BENCH_ARGS='-p molla-sensors --bench bench_sensor_raycast --features gpu'
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_math::{Real, SpatialInertia, Transform, Vec3};
use molla_sensors::{Raycaster, SensorScene, SensorSceneOptions};
use molla_sim::{BodyParams, Model, ModelBuilder, TriMesh};

fn build_scene(m: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let side = ((m as f64).cbrt().ceil() as usize).max(1);
    let r: Real = 0.25;
    let spacing: Real = 1.0;
    let mut placed = 0;
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if placed >= m {
                    break 'outer;
                }
                let p = Vec3::new(
                    ix as Real * spacing - 5.0,
                    1.0 + iy as Real * spacing,
                    iz as Real * spacing - 5.0,
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

fn bench_raycast(c: &mut Criterion) {
    let engine = Engine::new(Backend::AutoGpu).expect("sensor benchmark requires a GPU");
    eprintln!("sensor raycast adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("sensor_raycast_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    for &(n_rays, m_shapes) in &[(1_000usize, 100usize), (10_000, 1_000)] {
        let mut model = build_scene(m_shapes);
        model.upload_to_device(&device, &queue).unwrap();
        let mut state = model.state().unwrap();
        state.upload_to_device(&device, &queue).unwrap();

        // Top-down rays in a square pattern over the scene.
        let side = (n_rays as f64).sqrt().ceil() as usize;
        let mut origins = Vec::with_capacity(n_rays);
        let mut dirs = Vec::with_capacity(n_rays);
        for iy in 0..side {
            for ix in 0..side {
                if origins.len() >= n_rays {
                    break;
                }
                let x = (ix as f64 / side as f64 - 0.5) * 12.0;
                let z = (iy as f64 / side as f64 - 0.5) * 12.0;
                origins.push(Vec3::new(x, 10.0, z));
                dirs.push(Vec3::new(0.0, -1.0, 0.0));
            }
        }
        origins.truncate(n_rays);
        dirs.truncate(n_rays);
        let mut sensor = Raycaster::new(&engine, n_rays as u32).expect("Raycaster::new");

        let label = format!("rays={n_rays}_shapes={m_shapes}");
        group.bench_with_input(BenchmarkId::from_parameter(label), &n_rays, |bencher, _| {
            bencher.iter(|| {
                let hits = sensor
                    .cast(
                        black_box(&engine),
                        black_box(&model),
                        black_box(&state),
                        black_box(&origins),
                        black_box(&dirs),
                        100.0,
                    )
                    .expect("cast");
                black_box(hits.len())
            });
        });
    }
    group.finish();
}

fn bench_mesh_raycast(c: &mut Criterion) {
    let engine = Engine::new(Backend::AutoGpu).expect("sensor benchmark requires a GPU");
    eprintln!("sensor mesh raycast adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let origins: Vec<_> = (0..1_000)
        .map(|i| {
            Vec3::new(
                (i % 32) as f64 / 3.0 - 5.0,
                10.0,
                (i / 32) as f64 / 3.0 - 5.0,
            )
        })
        .collect();
    let directions = vec![Vec3::NEG_Y; origins.len()];
    let mut group = c.benchmark_group("sensor_raycast_mesh_gpu");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for count in [1, 16, 64] {
        let mut builder = ModelBuilder::new();
        let mesh = builder.add_mesh(TriMesh {
            vertices: vec![
                Vec3::new(-4.0, 0.0, -4.0),
                Vec3::new(4.0, 0.0, -4.0),
                Vec3::new(4.0, 0.0, 4.0),
                Vec3::new(-4.0, 0.0, 4.0),
            ],
            indices: vec![[0, 1, 2], [0, 2, 3]],
        });
        for index in 0..count {
            builder
                .add_shape_trimesh(
                    BodyId::NONE,
                    mesh,
                    Transform::new(Vec3::Y * (index as f64 * 0.01), molla_math::Quat::IDENTITY),
                )
                .unwrap();
        }
        let mut model = builder.build().unwrap();
        model.upload_to_device(device, queue).unwrap();
        let mut state = model.state().unwrap();
        state.upload_to_device(device, queue).unwrap();
        let mut sensor = Raycaster::new(&engine, origins.len() as u32).unwrap();
        sensor
            .cast(&engine, &model, &state, &origins, &directions, 100.0)
            .unwrap();
        assert_eq!(sensor.mesh_cache_len(), 1);
        let scene =
            SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("scene_build_completed", count),
            &count,
            |bencher, _| {
                bencher.iter(|| {
                    let scene =
                        SensorScene::from_model(&engine, &model, SensorSceneOptions::default())
                            .unwrap();
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(30)),
                        })
                        .unwrap();
                    black_box(scene);
                });
            },
        );
        for mode in ["host_channels", "device_completed", "owned_scene_host"] {
            group.bench_with_input(BenchmarkId::new(mode, count), &count, |bencher, _| {
                bencher.iter(|| {
                    if mode == "owned_scene_host" {
                        black_box(
                            sensor
                                .cast_scene(&engine, &scene, &state, &origins, &directions, 100.0)
                                .unwrap(),
                        );
                    } else if mode == "host_channels" {
                        black_box(
                            sensor
                                .cast(&engine, &model, &state, &origins, &directions, 100.0)
                                .unwrap(),
                        );
                    } else {
                        black_box(
                            sensor
                                .dispatch(&engine, &model, &state, &origins, &directions, 100.0)
                                .unwrap(),
                        );
                        device
                            .poll(wgpu::PollType::Wait {
                                submission_index: None,
                                timeout: Some(Duration::from_secs(30)),
                            })
                            .expect("completed sensor dispatch");
                    }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_raycast, bench_mesh_raycast);
criterion_main!(benches);
