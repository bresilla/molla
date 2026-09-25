#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Real, Transform, Vec3};
use molla_sensors::{
    CameraChannels, INVALID_SCENE_DISTANCE, INVALID_SCENE_INDEX, INVALID_SCENE_SHAPE_ID,
    LidarBatch, LidarPattern, LidarScan, PhysicalOptics, Raycaster, ScanMetadata,
    SceneRebuildPolicy, SensorCommandBatch, SensorFrame, SensorFramePoolConfig, SensorScene,
    TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource},
};
use molla_sim::{ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn write_pose(engine: &Engine, state: &mut State, pose: GpuTransform) {
    engine.queue().unwrap().write_buffer(
        state.body_q.device_buffer().unwrap(),
        0,
        bytemuck::bytes_of(&pose),
    );
    state.body_q.mark_device_dirty();
}

fn invalid_poses() -> Vec<GpuTransform> {
    let mut poses = Vec::new();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut pose: GpuTransform = Transform::IDENTITY.into();
        pose.position[0] = value;
        poses.push(pose);
    }
    for rotation in [[0.0; 4], [0.0, 0.0, 0.0, 2.0], [f32::NAN, 0.0, 0.0, 1.0]] {
        let mut pose: GpuTransform = Transform::IDENTITY.into();
        pose.rotation = rotation;
        poses.push(pose);
    }
    poses
}

#[test]
#[ignore = "requires GPU"]
fn invalid_live_primitive_mesh_and_plane_poses_fail_queries_and_recover() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for kind in 0..3 {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        match kind {
            0 => builder
                .add_shape_sphere(body, 1.0, Transform::IDENTITY)
                .unwrap(),
            1 => {
                let mesh = builder.add_mesh(TriMesh {
                    vertices: vec![-Vec3::X, Vec3::X, Vec3::Y],
                    indices: vec![[0, 1, 2]],
                });
                builder
                    .add_shape_trimesh(body, mesh, Transform::IDENTITY)
                    .unwrap()
            }
            _ => builder
                .add_shape_plane(body, Vec3::Y, Transform::IDENTITY)
                .unwrap(),
        };
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 2).unwrap();
        let origins = [Vec3::Y * 5.0, Vec3::X * 100.0];
        let directions = [Vec3::NEG_Y, Vec3::X];
        for pose in invalid_poses() {
            write_pose(&engine, &mut state, pose);
            let result = query.cast_scene(&engine, &scene, &state, &origins, &directions, 10.0);
            assert!(
                result.is_err(),
                "kind {kind}: invalid pose returned {result:?}"
            );
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("invalid scene pose")
            );
            write_pose(&engine, &mut state, Transform::IDENTITY.into());
            let hits = query
                .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
                .unwrap();
            assert!(hits[1].shape_id.is_none());
            if kind != 1 {
                assert_eq!(hits[0].t, if kind == 0 { 4.0 } else { 5.0 });
            }
        }
        assert!(state.body_q.host().is_err());
        assert_eq!(query.query_stats().scene_builds, 1);
    }
}

#[test]
#[ignore = "requires GPU"]
fn invalid_leaf_poisoning_survives_parent_refits_and_spatial_rebuilds() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for quality_ratio_percent in [None, Some(125)] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        builder
            .add_shape_sphere(body, 1.0, Transform::IDENTITY)
            .unwrap();
        for i in 0..65 {
            builder
                .add_shape_sphere(
                    BodyId::NONE,
                    1.0,
                    Transform::from_translation(Vec3::X * (10.0 + i as f64 * 3.0)),
                )
                .unwrap();
        }
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
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
        for pose in invalid_poses() {
            write_pose(&engine, &mut state, pose);
            assert!(
                query
                    .cast_scene(
                        &engine,
                        &scene,
                        &state,
                        &[Vec3::X * 10.0 + Vec3::Y * 5.0],
                        &[Vec3::NEG_Y],
                        10.0
                    )
                    .is_err()
            );
            write_pose(&engine, &mut state, Transform::IDENTITY.into());
            let hits = query
                .cast_scene(
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::X * 10.0 + Vec3::Y * 5.0],
                    &[Vec3::NEG_Y],
                    10.0,
                )
                .unwrap();
            assert_eq!(hits[0].shape_id, Some(1));
            assert_eq!(hits[0].t, 4.0);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn only_included_geometry_invalidates_queries_and_scene_replacement_clears_flags() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    write_pose(&engine, &mut state, invalid_poses()[0]);
    let mut scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut query = Raycaster::new(&engine, 1).unwrap();
    let origins = [Vec3::Z * 5.0];
    let directions = [Vec3::NEG_Z];
    assert_eq!(
        query
            .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
            .unwrap()[0]
            .t,
        4.0
    );
    let mut builder = ModelBuilder::new();
    assert_eq!(builder.add_body(Default::default()), body);
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_plane(body, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    scene
        .replace_model(&engine, &builder.build().unwrap())
        .unwrap();
    assert!(
        query
            .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
            .is_err()
    );
    assert!(
        query
            .cast_scene(&engine, &scene, &state, &[], &[], 10.0)
            .unwrap()
            .is_empty()
    );
    let mut empty = ModelBuilder::new();
    empty.add_body(Default::default());
    scene
        .replace_model(&engine, &empty.build().unwrap())
        .unwrap();
    assert!(
        query
            .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
            .unwrap()[0]
            .shape_id
            .is_none()
    );
}

fn read(engine: &Engine, sources: &[ReadbackSource<'_>]) -> ReadbackFrame {
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1 << 16,
            max_total_bytes: 1 << 16,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(sources, Default::default())
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn read_query(engine: &Engine, frame: &SensorFrame, lidar: bool) -> ReadbackFrame {
    let t = frame.channel::<Real>(2).unwrap();
    let id = frame.channel::<i32>(3).unwrap();
    let p = frame.channel::<Vec3>(4).unwrap();
    let n = frame.channel::<Vec3>(5).unwrap();
    let mut sources = vec![
        t.readback_source(0..t.len()).unwrap(),
        id.readback_source(0..id.len()).unwrap(),
        p.readback_source(0..p.len()).unwrap(),
        n.readback_source(0..n.len()).unwrap(),
    ];
    let first;
    let last;
    let count;
    if lidar {
        first = frame.channel::<i32>(6).unwrap();
        last = frame.channel::<i32>(7).unwrap();
        count = frame.channel::<u32>(8).unwrap();
        sources.extend([
            first.readback_source(0..first.len()).unwrap(),
            last.readback_source(0..last.len()).unwrap(),
            count.readback_source(0..count.len()).unwrap(),
        ]);
    }
    read(engine, &sources)
}

fn read_camera(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let depth = frame.channel::<Real>(0).unwrap();
    let color = frame.channel::<u32>(1).unwrap();
    let normal = frame.channel::<Vec3>(2).unwrap();
    let albedo = frame.channel::<u32>(3).unwrap();
    let shape = frame.channel::<u32>(4).unwrap();
    read(
        engine,
        &[
            depth.readback_source(0..depth.len()).unwrap(),
            color.readback_source(0..color.len()).unwrap(),
            normal.readback_source(0..normal.len()).unwrap(),
            albedo.readback_source(0..albedo.len()).unwrap(),
            shape.readback_source(0..shape.len()).unwrap(),
        ],
    )
}

fn assert_invalid_query(frame: &ReadbackFrame, lidar: bool) {
    assert!(
        frame
            .channel::<Real>(0)
            .unwrap()
            .iter()
            .all(|&v| v == INVALID_SCENE_DISTANCE)
    );
    assert!(
        frame
            .channel::<i32>(1)
            .unwrap()
            .iter()
            .all(|&v| v == INVALID_SCENE_INDEX)
    );
    for c in [2, 3] {
        assert!(
            frame
                .channel::<Vec3>(c)
                .unwrap()
                .iter()
                .all(|&v| v == Vec3::ZERO)
        );
    }
    if lidar {
        for c in [4, 5] {
            assert_eq!(frame.channel::<i32>(c).unwrap(), [INVALID_SCENE_INDEX; 2]);
        }
        assert_eq!(frame.channel::<u32>(6).unwrap(), [0; 2]);
    }
}

#[test]
#[ignore = "requires GPU"]
fn direct_and_standalone_lidar_camera_keep_invalid_flags_and_historical_frames() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for plane in [false, true] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        if plane {
            builder
                .add_shape_plane(body, Vec3::Z, Transform::IDENTITY)
                .unwrap();
        } else {
            builder
                .add_shape_sphere(body, 1.0, Transform::IDENTITY)
                .unwrap();
        }
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 2).unwrap();
        let mut lidar = LidarBatch::new(&engine, 2, 6).unwrap();
        let mut camera = TiledCamera::new(
            &engine,
            TiledCameraConfig {
                width: 3,
                height: 3,
                channels: CameraChannels::ALL,
                optics: PhysicalOptics {
                    lens_diameter_m: 0.04,
                    aperture_samples: 3,
                    ..Default::default()
                },
                ..Default::default()
            },
            1,
        )
        .unwrap();
        let config = SensorFramePoolConfig {
            slots: 3,
            max_total_bytes: 1 << 20,
        };
        let mut ray_pool = query.output_pool(&engine, config).unwrap();
        let mut lidar_pool = lidar.output_pool(&engine, config).unwrap();
        let mut camera_pool = camera.output_pool(&engine, config).unwrap();
        let scans = [10.0, 0.5].map(|max_distance| LidarScan {
            sensor_id: max_distance as u64,
            pattern: LidarPattern::Terrestrial(ScanMetadata {
                forward: Vec3::NEG_Z,
                theta_range: (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                subray_count: 3,
                max_distance,
                ..ScanMetadata::new(Vec3::ZERO, 1, 1)
            }),
        });
        let valid: GpuTransform = Transform::from_translation(Vec3::NEG_Z * 5.0).into();
        let mut saved = Vec::new();
        for (sample, pose) in [valid, invalid_poses()[0], valid].into_iter().enumerate() {
            write_pose(&engine, &mut state, pose);
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let ray = query
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::ZERO, Vec3::X * 100.0],
                    &[Vec3::NEG_Z, Vec3::Z],
                    10.0,
                    &mut ray_pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            let scan = lidar
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &scans,
                    &mut lidar_pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            let image = camera
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &mut camera_pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            saved.push((
                sample,
                submitted.frame(&ray.frame).unwrap().clone(),
                submitted.frame(&scan.frame).unwrap().clone(),
                submitted.frame(&image.frame).unwrap().clone(),
            ));
        }
        for (sample, ray, scan, image) in &saved {
            let ray = read_query(&engine, ray, false);
            let scan = read_query(&engine, scan, true);
            let image = read_camera(&engine, image);
            if *sample == 1 {
                assert_invalid_query(&ray, false);
                assert_invalid_query(&scan, true);
                assert_eq!(
                    image.channel::<Real>(0).unwrap(),
                    [INVALID_SCENE_DISTANCE; 9]
                );
                assert_eq!(image.channel::<u32>(1).unwrap(), [0; 9]);
                assert_eq!(image.channel::<Vec3>(2).unwrap(), [Vec3::ZERO; 9]);
                assert_eq!(image.channel::<u32>(3).unwrap(), [0; 9]);
                assert_eq!(
                    image.channel::<u32>(4).unwrap(),
                    [INVALID_SCENE_SHAPE_ID; 9]
                );
            } else {
                let t = if plane { 5.0 } else { 4.0 };
                assert_eq!(ray.channel::<Real>(0).unwrap(), [t, -1.0]);
                assert_eq!(ray.channel::<i32>(1).unwrap(), [0, -1]);
                assert_eq!(
                    scan.channel::<Real>(0).unwrap(),
                    [t, t, t, -1.0, -1.0, -1.0]
                );
                assert_eq!(scan.channel::<i32>(4).unwrap(), [0, -1]);
                assert_eq!(scan.channel::<i32>(5).unwrap(), [2, -1]);
                assert_eq!(scan.channel::<u32>(6).unwrap(), [3, 0]);
                assert_eq!(image.channel::<Real>(0).unwrap()[4], t);
                assert_eq!(image.channel::<u32>(4).unwrap()[4], 0);
            }
        }
        for pose in invalid_poses() {
            write_pose(&engine, &mut state, pose);
            let scan = lidar
                .dispatch_scene(&engine, &scene, &state, &scans)
                .unwrap();
            let cpu = read(
                &engine,
                &[
                    scan.hits.distance.readback_source(0..6).unwrap(),
                    scan.hits.shape.readback_source(0..6).unwrap(),
                    scan.hits.point.readback_source(0..6).unwrap(),
                    scan.hits.normal.readback_source(0..6).unwrap(),
                    scan.first_ray.readback_source(0..2).unwrap(),
                    scan.last_ray.readback_source(0..2).unwrap(),
                    scan.hit_count.readback_source(0..2).unwrap(),
                ],
            );
            assert_invalid_query(&cpu, true);
            camera.dispatch_scene(&engine, &scene, &state).unwrap();
            assert_eq!(
                camera.download_depth(&engine).unwrap(),
                [INVALID_SCENE_DISTANCE; 9]
            );
            assert_eq!(
                camera.download_shape_index(&engine).unwrap(),
                [INVALID_SCENE_SHAPE_ID; 9]
            );
        }
        write_pose(&engine, &mut state, valid);
        camera.dispatch_scene(&engine, &scene, &state).unwrap();
        assert_eq!(camera.download_shape_index(&engine).unwrap()[4], 0);
        assert!(state.body_q.host().is_err());
    }
}
