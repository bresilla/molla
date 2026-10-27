#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{GpuTransform, Real, Transform, Vec3};
use molla_sensors::{
    CameraBatch, CameraSpec, LidarBatch, LidarPattern, LidarScan, Raycaster, ScanMetadata,
    SceneRebuildPolicy, SceneRebuildStorage, SensorCommandBatch, SensorFrame,
    SensorFramePoolConfig, SensorScene, TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackRing},
};
use molla_sim::{ModelBuilder, State};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires GPU"]
fn rebuilt_mixed_mesh_primitive_ties_follow_dense_identity_not_spatial_order() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for quality_ratio_percent in [None, Some(125)] {
        let mut builder = ModelBuilder::new();
        let mesh = builder.add_mesh(molla_sim::TriMesh {
            vertices: vec![
                Vec3::new(-2.0, 1.0, -2.0),
                Vec3::new(2.0, 1.0, -2.0),
                Vec3::new(0.0, 1.0, 2.0),
            ],
            indices: vec![[0, 1, 2]],
        });
        for i in 0..129 {
            let body = builder.add_body(Default::default());
            if i % 2 == 0 {
                builder
                    .add_shape_trimesh(body, mesh, Transform::IDENTITY)
                    .unwrap();
            } else {
                builder
                    .add_shape_sphere(body, 1.0, Transform::IDENTITY)
                    .unwrap();
            }
        }
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        model
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 1).unwrap();
        query
            .set_scene_rebuild_policy(SceneRebuildPolicy {
                interval: 1,
                quality_ratio_percent,
                ..Default::default()
            })
            .unwrap();
        let mut raw = Raycaster::new(&engine, 1).unwrap();
        for phase in 0..3 {
            let poses: Vec<GpuTransform> = (0..129)
                .map(|i| {
                    let x = if phase == 0 {
                        ((i * 73) % 129) as f64 * 4.0
                    } else if phase == 2 && i % 2 == 1 {
                        1000.0
                    } else {
                        0.0
                    };
                    Transform::from_translation(Vec3::X * x).into()
                })
                .collect();
            engine.queue().unwrap().write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&poses),
            );
            state.body_q.mark_device_dirty();
            let actual = query
                .cast_scene(
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                )
                .unwrap();
            let reference = raw
                .cast(
                    &engine,
                    &model,
                    &state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                )
                .unwrap();
            assert_eq!(actual[0].shape_id, Some(if phase == 1 { 1 } else { 0 }));
            assert_eq!(actual[0].shape_id, reference[0].shape_id);
            assert_eq!(actual[0].t, 9.0);
            assert_eq!(actual[0].t, reference[0].t);
            assert_eq!(actual[0].point, reference[0].point);
            assert_eq!(actual[0].normal, reference[0].normal);
            assert!(state.body_q.host().is_err());
        }
        assert_eq!(
            query.query_stats().scene_rebuilds,
            if quality_ratio_percent.is_some() {
                0
            } else {
                3
            }
        );
        assert_eq!(
            query.query_stats().scene_quality_checks,
            if quality_ratio_percent.is_some() {
                3
            } else {
                0
            }
        );
        assert_eq!(query.query_stats().scene_builds, 1);
        assert_eq!(query.query_stats().scene_preparations, 1);
        assert_eq!(query.query_stats().refit_preparations, 1);
    }
}

fn fixture(engine: &Engine) -> (State, SensorScene) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    for i in 0..65 {
        builder
            .add_shape_sphere(
                body,
                1.0,
                Transform::from_translation(Vec3::X * i as f64 * 3.0),
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (
        state,
        SensorScene::from_model(engine, &model, Default::default()).unwrap(),
    )
}

fn pool_config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 20,
    }
}

fn distance(engine: &Engine, frame: &SensorFrame) -> Real {
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 8,
            max_total_bytes: 8,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(
            &[frame
                .channel::<Real>(2)
                .unwrap()
                .readback_source(0..1)
                .unwrap()],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30))
        .unwrap()
        .channel::<Real>(0)
        .unwrap()[0]
}

#[test]
#[ignore = "requires GPU"]
fn periodic_rebuilds_cancel_safely_preserve_leases_and_reject_policy_changes_transactionally() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for quality_ratio_percent in [None, Some(125)] {
        let (mut state, scene) = fixture(&engine);
        let (mut other, _) = fixture(&engine);
        let mut query = Raycaster::new(&engine, 1).unwrap();
        let policy = SceneRebuildPolicy {
            interval: 2,
            min_leaves: 2,
            max_device_bytes: if quality_ratio_percent.is_some() {
                SceneRebuildStorage::for_quality_leaves(65)
                    .unwrap()
                    .device_bytes
            } else {
                SceneRebuildStorage::for_leaves(65).unwrap().device_bytes
            },
            quality_ratio_percent,
        };
        query.set_scene_rebuild_policy(policy).unwrap();
        let mut pool = query.output_pool(&engine, pool_config(3)).unwrap();
        {
            let mut cancelled = SensorCommandBatch::new(&engine).unwrap();
            query
                .encode_scene(
                    &mut cancelled,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
        }
        let encoded_count = |stats: molla_sensors::RayQueryStats| {
            if quality_ratio_percent.is_some() {
                stats.scene_quality_checks
            } else {
                stats.scene_rebuilds
            }
        };
        assert_eq!(encoded_count(query.query_stats()), 1);
        assert!(
            query
                .set_scene_rebuild_policy(SceneRebuildPolicy {
                    quality_ratio_percent: Some(100),
                    ..policy
                })
                .is_err()
        );
        assert_eq!(query.scene_rebuild_policy(), policy);
        assert!(
            query
                .set_scene_rebuild_policy(SceneRebuildPolicy {
                    max_device_bytes: policy.max_device_bytes - 1,
                    ..policy
                })
                .is_err()
        );
        assert_eq!(query.scene_rebuild_policy(), policy);
        let mut held = Vec::new();
        for frame in 0..3 {
            let state = if frame % 2 == 0 {
                &mut state
            } else {
                &mut other
            };
            let pose = GpuTransform::from(Transform::from_translation(Vec3::Y * frame as f64));
            engine.queue().unwrap().write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::bytes_of(&pose),
            );
            state.body_q.mark_device_dirty();
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let out = query
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            assert_eq!(
                submitted.stats.kernel_dispatches,
                if frame == 1 {
                    if quality_ratio_percent.is_some() {
                        36
                    } else {
                        31
                    }
                } else {
                    9
                }
            );
            held.push(submitted.frame(&out.frame).unwrap().clone());
        }
        assert!(state.body_q.host().is_err());
        assert!(other.body_q.host().is_err());
        let stats = query.query_stats();
        assert_eq!(stats.encoded_queries, 4);
        assert_eq!(stats.scene_builds, 1);
        assert_eq!(encoded_count(stats), 2);
        assert_eq!(stats.refit_preparations, 2);
        let mut blocked = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            query
                .encode_scene(
                    &mut blocked,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut pool,
                    Default::default()
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(blocked.submit().unwrap().stats.kernel_dispatches, 0);
        assert_eq!(query.query_stats(), stats);
        query
            .set_scene_rebuild_policy(SceneRebuildPolicy {
                interval: 0,
                max_device_bytes: 0,
                ..policy
            })
            .unwrap();
        for (i, frame) in held.iter().enumerate() {
            assert_eq!(distance(&engine, frame), 9.0 - i as f64);
        }
        drop(held);
        let hit = query
            .cast_scene(
                &engine,
                &scene,
                &state,
                &[Vec3::Y * 10.0],
                &[Vec3::NEG_Y],
                20.0,
            )
            .unwrap();
        assert_eq!(hit[0].t, 7.0);
        assert_eq!(encoded_count(query.query_stats()), 2);
        assert_eq!(query.query_stats().scene_builds, 2);
        assert_eq!(query.readback_stats().submissions, 1);
    }
}

#[test]
#[ignore = "requires GPU"]
fn rebuild_budget_preflight_covers_ray_lidar_camera_and_independent_camera_groups() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for quality_ratio_percent in [None, Some(125)] {
        let (state, scene) = fixture(&engine);
        let too_small = SceneRebuildPolicy {
            interval: 1,
            min_leaves: 2,
            max_device_bytes: 0,
            quality_ratio_percent,
        };
        let mut query = Raycaster::new(&engine, 1).unwrap();
        query.set_scene_rebuild_policy(too_small).unwrap();
        let mut qp = query.output_pool(&engine, pool_config(1)).unwrap();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            query
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y * 10.0],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut qp,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.submit().unwrap().stats.input_upload_bytes, 0);
        assert_eq!(query.query_stats().scene_builds, 0);
        assert_eq!(qp.stats().reserved_frames, 0);
        let mut lidar = LidarBatch::new(&engine, 1, 1).unwrap();
        lidar.set_scene_rebuild_policy(too_small).unwrap();
        let mut lp = lidar.output_pool(&engine, pool_config(1)).unwrap();
        let scans = [LidarScan {
            sensor_id: 5,
            pattern: LidarPattern::Terrestrial(ScanMetadata::new(Vec3::Y * 10.0, 1, 1)),
        }];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            lidar
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &scans,
                    &mut lp,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.submit().unwrap().stats.input_upload_bytes, 0);
        assert_eq!(lidar.query_stats().scene_builds, 0);
        assert_eq!(lp.stats().reserved_frames, 0);
        let config = TiledCameraConfig {
            width: 1,
            height: 1,
            ..Default::default()
        };
        let mut camera = TiledCamera::new(&engine, config, 1).unwrap();
        camera.set_scene_rebuild_policy(too_small).unwrap();
        let mut cp = camera.output_pool(&engine, pool_config(1)).unwrap();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            camera
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &mut cp,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.submit().unwrap().stats.input_upload_bytes, 0);
        assert_eq!(camera.query_stats().scene_builds, 0);
        assert_eq!(cp.stats().reserved_frames, 0);
        let specs = [
            CameraSpec {
                sensor_id: 1,
                config,
                pose: Transform::IDENTITY,
            },
            CameraSpec {
                sensor_id: 2,
                config: TiledCameraConfig { width: 2, ..config },
                pose: Transform::IDENTITY,
            },
        ];
        let mut manager = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
        assert_eq!(manager.group_count(), 2);
        manager.set_scene_rebuild_policy(1, too_small).unwrap();
        assert_eq!(
            manager.scene_rebuild_policy(0),
            Some(SceneRebuildPolicy::default())
        );
        assert_eq!(manager.scene_rebuild_policy(1), Some(too_small));
        assert!(manager.set_scene_rebuild_policy(2, too_small).is_err());
        let mut mp = manager.output_pool(&engine, pool_config(1)).unwrap();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            manager
                .encode_scene(&mut batch, &engine, &scene, &state, &mut mp, None)
                .is_err()
        );
        assert_eq!(batch.submit().unwrap().stats.input_upload_bytes, 0);
        assert_eq!(manager.query_stats(0).unwrap().scene_builds, 0);
        assert_eq!(manager.query_stats(1).unwrap().scene_builds, 0);
        let valid = SceneRebuildPolicy {
            max_device_bytes: 1 << 20,
            ..too_small
        };
        manager.set_scene_rebuild_policy(1, valid).unwrap();
        for _ in 0..2 {
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let sample = manager
                .encode_scene(&mut batch, &engine, &scene, &state, &mut mp, None)
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            drop(sample);
            drop(submitted);
            engine
                .device()
                .unwrap()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                })
                .unwrap();
        }
        assert_eq!(manager.query_stats(0).unwrap().scene_rebuilds, 1);
        let second = manager.query_stats(1).unwrap();
        assert_eq!(
            second.scene_quality_checks,
            if quality_ratio_percent.is_some() {
                2
            } else {
                0
            }
        );
        assert_eq!(
            second.scene_rebuilds,
            if quality_ratio_percent.is_some() {
                0
            } else {
                2
            }
        );
    }
}
