//! Completed mixed-scene queries, including host ray upload and optional collection.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Transform, Vec3};
use molla_sensors::{Raycaster, SceneRebuildPolicy, SensorScene};
use molla_sim::{ModelBuilder, TriMesh};
use std::time::Duration;

fn bench_queries(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("scene query benchmark requires GPU");
    eprintln!("scene query adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let wait = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap()
    };
    let mut group = c.benchmark_group("scene_queries_1024_rays");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for n in [1_000, 10_000] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        let mesh = builder.add_mesh(TriMesh {
            vertices: vec![-Vec3::X, Vec3::X, Vec3::Z],
            indices: vec![[0, 1, 2]],
        });
        for i in 0..n {
            let pose = Transform::from_translation(Vec3::new(
                (i % 100) as f64 * 3.0,
                0.0,
                (i / 100) as f64 * 3.0,
            ));
            if i % 4 == 0 {
                builder.add_shape_sphere(body, 0.5, pose).unwrap();
            } else {
                builder.add_shape_trimesh(body, mesh, pose).unwrap();
            }
        }
        builder
            .add_shape_plane(
                BodyId::NONE,
                Vec3::Y,
                Transform::from_translation(Vec3::NEG_Y * 2.0),
            )
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let origins: Vec<_> = (0..1024)
            .map(|i| {
                let target = (i * n / 1024).min(n - 1);
                Vec3::new(
                    (target % 100) as f64 * 3.0 + 0.1,
                    10.0,
                    (target / 100) as f64 * 3.0 + 0.1,
                )
            })
            .collect();
        let directions = vec![Vec3::NEG_Y; origins.len()];
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        for (name, policy) in [
            ("linear_per_instance", None),
            ("hierarchy", Some(SceneRebuildPolicy::default())),
            (
                "hierarchy_fixed",
                Some(SceneRebuildPolicy {
                    interval: 0,
                    ..Default::default()
                }),
            ),
            (
                "hierarchy_once",
                Some(SceneRebuildPolicy {
                    interval: u32::MAX,
                    ..Default::default()
                }),
            ),
        ] {
            let accelerated = policy.is_some();
            if matches!(name, "linear_per_instance" | "hierarchy") {
                group.bench_function(
                    BenchmarkId::new(format!("{name}_construct_and_collect"), n),
                    |b| {
                        b.iter(|| {
                            let mut query = Raycaster::new(&engine, 1024).unwrap();
                            let hits = if accelerated {
                                let scene =
                                    SensorScene::from_model(&engine, &model, Default::default())
                                        .unwrap();
                                query
                                    .cast_scene(
                                        &engine,
                                        &scene,
                                        &state,
                                        &origins,
                                        &directions,
                                        30.0,
                                    )
                                    .unwrap()
                            } else {
                                query
                                    .cast(&engine, &model, &state, &origins, &directions, 30.0)
                                    .unwrap()
                            };
                            std::hint::black_box(hits);
                        })
                    },
                );
            }
            let mut query = Raycaster::new(&engine, 1024).unwrap();
            if let Some(policy) = policy {
                query.set_scene_rebuild_policy(policy).unwrap();
            }
            let warm_hits = if accelerated {
                query
                    .cast_scene(&engine, &scene, &state, &origins, &directions, 30.0)
                    .unwrap()
            } else {
                query
                    .cast(&engine, &model, &state, &origins, &directions, 30.0)
                    .unwrap()
            };
            for (i, hit) in warm_hits.iter().enumerate() {
                let target = (i * n / 1024).min(n - 1);
                let expected = if target % 4 == 0 {
                    10.0 - 0.23_f64.sqrt()
                } else {
                    10.0
                };
                assert_eq!(hit.shape_id, Some(target as u32));
                assert!((hit.t - expected).abs() < 1e-4);
            }
            group.bench_function(BenchmarkId::new(format!("{name}_warm_completed"), n), |b| {
                b.iter(|| {
                    if accelerated {
                        query
                            .dispatch_scene(&engine, &scene, &state, &origins, &directions, 30.0)
                            .unwrap();
                    } else {
                        query
                            .dispatch(&engine, &model, &state, &origins, &directions, 30.0)
                            .unwrap();
                    }
                    wait();
                })
            });
            group.bench_function(BenchmarkId::new(format!("{name}_warm_collect"), n), |b| {
                b.iter(|| {
                    let hits = if accelerated {
                        query
                            .cast_scene(&engine, &scene, &state, &origins, &directions, 30.0)
                            .unwrap()
                    } else {
                        query
                            .cast(&engine, &model, &state, &origins, &directions, 30.0)
                            .unwrap()
                    };
                    std::hint::black_box(hits);
                })
            });
            let stats = query.query_stats();
            if let Some(policy) = policy {
                assert_eq!(stats.scene_builds, 1);
                assert_eq!(stats.scene_preparations, 1);
                assert_eq!(stats.refit_preparations, 1);
                assert_eq!(
                    stats.scene_rebuilds,
                    if policy.interval == 0 {
                        0
                    } else {
                        stats.encoded_queries.div_ceil(u64::from(policy.interval))
                    }
                );
            }
            eprintln!("{name}/{n}: {stats:?}");
        }
    }
    group.finish();
}

fn bench_gpu_pose_rebuild(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("scene rebuild benchmark requires GPU");
    eprintln!("GPU-pose rebuild adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut group = c.benchmark_group("scene_rebuild_gpu_poses_1024_rays");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for n in [1_000, 10_000] {
        let mut builder = ModelBuilder::new();
        let mesh = builder.add_mesh(TriMesh {
            vertices: vec![-Vec3::X, Vec3::X, Vec3::Z],
            indices: vec![[0, 1, 2]],
        });
        for i in 0..n {
            let body = builder.add_body(Default::default());
            if i % 4 == 0 {
                builder
                    .add_shape_sphere(body, 0.5, Transform::IDENTITY)
                    .unwrap();
            } else {
                builder
                    .add_shape_trimesh(body, mesh, Transform::IDENTITY)
                    .unwrap();
            }
        }
        builder
            .add_shape_plane(
                BodyId::NONE,
                Vec3::Y,
                Transform::from_translation(Vec3::NEG_Y * 2.0),
            )
            .unwrap();
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state.upload_to_device(device, queue).unwrap();
        let poses: Vec<GpuTransform> = (0..n)
            .map(|i| {
                let slot = i * 997 % n;
                Transform::from_translation(Vec3::new(
                    (slot % 100) as f64 * 3.0,
                    0.0,
                    (slot / 100) as f64 * 3.0,
                ))
                .into()
            })
            .collect();
        queue.write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&poses),
        );
        state.body_q.mark_device_dirty();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let origins: Vec<_> = (0..1024)
            .map(|i| {
                let slot = (i * n / 1024).min(n - 1);
                Vec3::new(
                    (slot % 100) as f64 * 3.0 + 0.1,
                    10.0,
                    (slot / 100) as f64 * 3.0 + 0.1,
                )
            })
            .collect();
        let directions = vec![Vec3::NEG_Y; 1024];
        for (name, interval, quality_ratio_percent) in [
            ("fixed_local_seed", 0, None),
            ("periodic_64", 64, None),
            ("every_query", 1, None),
            ("quality_64", 64, Some(125)),
            ("quality_every_query", 1, Some(125)),
        ] {
            let mut query = Raycaster::new(&engine, 1024).unwrap();
            query
                .set_scene_rebuild_policy(SceneRebuildPolicy {
                    interval,
                    quality_ratio_percent,
                    ..Default::default()
                })
                .unwrap();
            let hits = query
                .cast_scene(&engine, &scene, &state, &origins, &directions, 30.0)
                .unwrap();
            assert!(
                hits.iter()
                    .all(|hit| hit.shape_id.is_some() && hit.t > 9.0 && hit.t <= 10.0)
            );
            group.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| {
                    query
                        .dispatch_scene(&engine, &scene, &state, &origins, &directions, 30.0)
                        .unwrap();
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(30)),
                        })
                        .unwrap();
                })
            });
            assert!(state.body_q.host().is_err());
            eprintln!("{name}/{n}: {:?}", query.query_stats());
        }
    }
    group.finish();
}

criterion_group!(benches, bench_queries, bench_gpu_pose_rebuild);
criterion_main!(benches);
