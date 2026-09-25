#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    FrameTarget, FrameTransformSensor, Raycaster, SensorCommandBatch, SensorFrame,
    SensorFramePoolConfig, SensorScene, SensorSceneOptions,
};
use molla_sim::{BodyParams, Heightfield, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires GPU"]
fn converted_terrain_and_triangle_hits_match_independent_planes() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let terrain = builder.add_heightfield(Heightfield {
        nx: 2,
        nz: 2,
        cell_size: 1.0,
        cell_size_z: 1.0,
        origin: Vec3::ZERO,
        heights: vec![0.0, 1.0, 2.0, 4.0],
    });
    builder
        .add_shape_heightfield(BodyId::NONE, terrain, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_triangle(
            BodyId::NONE,
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Transform::new(Vec3::new(5.0, 3.0, 0.0), Quat::IDENTITY),
        )
        .unwrap();
    let (model, state) = upload(&engine, builder);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut sensor = Raycaster::new(&engine, 4).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    let origins = [
        Vec3::new(0.25, 10.0, 0.25),
        Vec3::new(0.75, 10.0, 0.75),
        Vec3::new(5.0, 10.0, 0.0),
        Vec3::new(10.0, 10.0, 0.0),
    ];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &origins,
            &[Vec3::NEG_Y * 7.0; 4],
            20.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().kernel_dispatches, 3);
    let submitted = batch.submit().unwrap();
    let cpu = collect(&engine, submitted.frame(&receipt.frame).unwrap());
    assert_eq!(cpu.channel::<i32>(3).unwrap(), [0, 0, 1, -1]);
    let distances = cpu.channel::<Real>(2).unwrap();
    let points = cpu.channel::<Vec3>(4).unwrap();
    let normals = cpu.channel::<Vec3>(5).unwrap();
    for (i, (distance, normal)) in [
        (9.25, Vec3::new(-1.0, 1.0, -2.0).normalize()),
        (7.25, Vec3::new(-2.0, 1.0, -3.0).normalize()),
        (7.0, Vec3::Y),
    ]
    .into_iter()
    .enumerate()
    {
        assert!((distances[i] - distance).abs() < 1e-5);
        assert!((points[i] - (origins[i] - Vec3::Y * distance)).length() < 1e-5);
        assert!((normals[i] - normal).length() < 1e-5);
    }
    assert_eq!(distances[3], -1.0);
    assert_eq!(scene.mesh_count(), 2);
    assert_eq!(sensor.mesh_build_count(), 0);
}

fn quad() -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-4.0, 0.0, -4.0),
            Vec3::new(4.0, 0.0, -4.0),
            Vec3::new(4.0, 0.0, 4.0),
            Vec3::new(-4.0, 0.0, 4.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

fn upload(engine: &Engine, builder: ModelBuilder) -> (Model, State) {
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state)
}

fn fixture(engine: &Engine) -> (Model, State, SensorScene, BodyId) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 2.0));
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Y,
            Transform::new(-Vec3::Y * 2.0, Quat::IDENTITY),
        )
        .unwrap();
    let mesh = builder.add_mesh(quad());
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_trimesh(body, mesh, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_sphere(
            BodyId::NONE,
            1.0,
            Transform::new(Vec3::new(6.0, 2.0, 0.0), Quat::IDENTITY),
        )
        .unwrap();
    let (model, state) = upload(engine, builder);
    let scene = SensorScene::from_model(engine, &model, SensorSceneOptions::default()).unwrap();
    (model, state, scene, body)
}

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 20,
    }
}

fn drain(engine: &Engine) {
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
}

fn collect(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let n = frame.channel::<Vec3>(0).unwrap().len();
    let bytes = (n as u64 * 72).max(8);
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: bytes,
        },
    )
    .unwrap();
    let sources = [
        frame
            .channel::<Vec3>(0)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
        frame
            .channel::<Vec3>(1)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
        frame
            .channel::<Real>(2)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
        frame
            .channel::<i32>(3)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
        frame
            .channel::<Vec3>(4)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
        frame
            .channel::<Vec3>(5)
            .unwrap()
            .readback_source(0..n)
            .unwrap(),
    ];
    let ticket = ring.try_enqueue(&sources, frame.stamp()).unwrap().unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

#[test]
#[ignore = "requires GPU"]
fn mixed_primitive_moving_mesh_queries_compose_with_frames_and_match_legacy() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, scene, body) = fixture(&engine);
    let mut sensor = Raycaster::new(&engine, 65).unwrap();
    let mut second = Raycaster::new(&engine, 9).unwrap();
    let mut oracle = Raycaster::new(&engine, 65).unwrap();
    let mut transforms = FrameTransformSensor::new(
        &engine,
        &model,
        None,
        &[FrameTarget {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let mut rp = sensor.output_pool(&engine, config(3)).unwrap();
    let mut sp = second.output_pool(&engine, config(3)).unwrap();
    let mut tp = transforms.output_pool(&engine, config(3)).unwrap();
    let origins: Vec<_> = (0..65)
        .map(|i| Vec3::new(-6.0 + i as f64 * 0.2, 10.0, 0.3))
        .collect();
    let directions: Vec<_> = (0..65)
        .map(|i| Vec3::NEG_Y * [2.0, 1e-300, 1e300][i % 3])
        .collect();
    let mut captured = Vec::new();
    for step in 0..3 {
        let pose = Transform::new(
            Vec3::Y * (2.0 + step as f64),
            Quat::from_rotation_z(step as f64 * 0.1),
        );
        let packed: molla_math::GpuTransform = pose.into();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&packed),
        );
        state.body_q.mark_device_dirty();
        assert!(state.body_q.host().is_err());
        let expected = oracle
            .cast_scene(&engine, &scene, &state, &origins, &directions, 20.0)
            .unwrap();
        let stamp = ReadbackStamp {
            sensor_id: u64::MAX,
            sample_id: step,
            scene_revision: 42,
            simulation_time: Some(step as f64),
        };
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let a = sensor
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &state,
                &origins,
                &directions,
                20.0,
                &mut rp,
                stamp,
            )
            .unwrap()
            .unwrap();
        let b = second
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &state,
                &[Vec3::Y * 10.0; 9],
                &[Vec3::Y; 9],
                5.0,
                &mut sp,
                Default::default(),
            )
            .unwrap()
            .unwrap();
        transforms
            .encode(&mut batch, &engine, &state, &mut tp, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(a.ray_count, 65);
        assert_eq!(a.scene_stamp, scene.stamp());
        assert_eq!(batch.stats().submissions, 0);
        assert_eq!(batch.stats().input_upload_bytes, (65 + 9) * 32);
        assert_eq!(batch.stats().kernel_dispatches, 9);
        assert_eq!(batch.stats().output_clears, 12);
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.submissions, 1);
        captured.push((submitted, a, b, expected, stamp));
    }
    for (submitted, query, misses, expected, stamp) in captured {
        let frame = submitted.frame(&query.frame).unwrap();
        assert_eq!(frame.stamp(), stamp);
        let cpu = collect(&engine, frame);
        assert_eq!(cpu.byte_len(), 65 * 72);
        let actual_origins = cpu.channel::<Vec3>(0).unwrap();
        let actual_dirs = cpu.channel::<Vec3>(1).unwrap();
        let distances = cpu.channel::<Real>(2).unwrap();
        let shapes = cpu.channel::<i32>(3).unwrap();
        let points = cpu.channel::<Vec3>(4).unwrap();
        let normals = cpu.channel::<Vec3>(5).unwrap();
        for i in 0..65 {
            assert!((actual_origins[i] - origins[i]).length() < 1e-5);
            assert_eq!(actual_dirs[i], Vec3::NEG_Y);
            assert_eq!(
                shapes[i],
                expected[i].shape_id.map(|id| id as i32).unwrap_or(-1)
            );
            assert!((distances[i] - expected[i].t).abs() < 1e-4);
            assert!((points[i] - expected[i].point).length() < 1e-4);
            assert!((normals[i] - expected[i].normal).length() < 1e-4);
        }
        let miss = collect(&engine, submitted.frame(&misses.frame).unwrap());
        assert_eq!(miss.channel::<i32>(3).unwrap(), vec![-1; 9]);
        assert_eq!(miss.channel::<Real>(2).unwrap(), vec![-1.0; 9]);
        assert_eq!(miss.channel::<Vec3>(4).unwrap(), vec![Vec3::ZERO; 9]);
        assert_eq!(miss.channel::<Vec3>(5).unwrap(), vec![Vec3::Y; 9]);
    }
    assert_eq!(scene.mesh_count(), 1);
    assert_eq!(sensor.mesh_build_count(), 0);
    assert_eq!(sensor.query_stats().submissions, 0);
    assert_eq!(sensor.query_stats().encoded_queries, 3);
    assert_eq!(sensor.readback_stats().copied_bytes, 0);
    assert_eq!(sensor.readback_stats().blocking_waits, 0);
}

#[test]
#[ignore = "requires GPU"]
fn mesh_only_empty_scenes_range_boundaries_and_ties_preserve_miss_channels() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad());
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let (model, state) = upload(&engine, builder);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let empty_scene =
        SensorScene::from_model(&engine, &Model::empty(), SensorSceneOptions::default()).unwrap();
    let mut tie_builder = ModelBuilder::new();
    let mesh = tie_builder.add_mesh(quad());
    tie_builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    tie_builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let (tie_model, _) = upload(&engine, tie_builder);
    let tie_scene =
        SensorScene::from_model(&engine, &tie_model, SensorSceneOptions::default()).unwrap();
    let mut sensor = Raycaster::new(&engine, 3).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    for (scene, distance, expected) in [
        (&scene, 10.0, 0),
        (&scene, 5.0, -1),
        (&empty_scene, 10.0, -1),
        (&tie_scene, 10.0, 1),
    ] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let receipt = sensor
            .encode_scene(
                &mut batch,
                &engine,
                scene,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y * 7.0],
                distance,
                &mut pool,
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        let cpu = collect(&engine, submitted.frame(&receipt.frame).unwrap());
        assert_eq!(cpu.channel::<i32>(3).unwrap(), [expected]);
        assert_eq!(
            cpu.channel::<Real>(2).unwrap(),
            [if expected < 0 { -1.0 } else { 5.0 }]
        );
        assert_eq!(cpu.channel::<Vec3>(5).unwrap(), [Vec3::Y]);
        drop(submitted);
        drain(&engine);
        assert_eq!(pool.available_slots().unwrap(), 1);
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &[],
            &[],
            10.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().input_upload_bytes, 0);
    assert_eq!(batch.stats().kernel_dispatches, 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(
        collect(&engine, submitted.frame(&receipt.frame).unwrap()).byte_len(),
        0
    );
}

#[test]
#[ignore = "requires GPU"]
fn reset_dropped_uploads_backpressure_and_scene_replacement_keep_old_frames_immutable() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (mut model, state, mut scene, _) = fixture(&engine);
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    let mut pool = sensor.output_pool(&engine, config(2)).unwrap();
    {
        let mut cancelled = SensorCommandBatch::new(&engine).unwrap();
        sensor
            .encode_scene(
                &mut cancelled,
                &engine,
                &scene,
                &state,
                &[Vec3::Y * 123.0],
                &[Vec3::X],
                2.0,
                &mut pool,
                Default::default(),
            )
            .unwrap()
            .unwrap();
    }
    assert_eq!(pool.available_slots().unwrap(), 2);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let old_receipt = sensor
        .encode_scene(
            &mut batch,
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
    pool.reset().unwrap();
    let old_submission = batch.submit().unwrap();
    let old = old_submission.frame(&old_receipt.frame).unwrap().clone();
    assert!(!pool.is_current(&old));
    drop(old_submission);
    for vertex in &mut model.meshes[0].vertices {
        vertex.y += 3.0;
    }
    scene.replace_model(&engine, &model).unwrap();
    assert!(scene.resolve_body(old_receipt.scene_stamp, 2).is_err());
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let next = sensor
        .encode_scene(
            &mut batch,
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
    assert_ne!(next.scene_stamp, old_receipt.scene_stamp);
    let submitted = batch.submit().unwrap();
    let stats = sensor.query_stats();
    let mut blocked = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        sensor
            .encode_scene(
                &mut blocked,
                &engine,
                &scene,
                &state,
                &[Vec3::Y],
                &[Vec3::NEG_Y],
                20.0,
                &mut pool,
                Default::default()
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(blocked.stats().input_upload_bytes, 0);
    assert_eq!(blocked.stats().output_clears, 0);
    assert_eq!(blocked.submit().unwrap().stats.submissions, 0);
    assert_eq!(sensor.query_stats(), stats);
    assert_eq!(collect(&engine, &old).channel::<Real>(2).unwrap(), [8.0]);
    assert_eq!(
        collect(&engine, submitted.frame(&next.frame).unwrap())
            .channel::<Real>(2)
            .unwrap(),
        [5.0]
    );
    drop(pool);
    drop(sensor);
    drop(scene);
    assert_eq!(
        collect(&engine, &old).channel::<Vec3>(0).unwrap(),
        [Vec3::Y * 10.0]
    );
}

#[test]
#[ignore = "requires GPU"]
fn four_slot_ping_pong_scene_queries_reuse_all_warmed_bindings() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, scene, _) = fixture(&engine);
    let mut second = model.state().unwrap();
    second.body_q.host_mut().unwrap()[0].position = Vec3::Y * 4.0;
    second
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut sensor = Raycaster::new(&engine, 65).unwrap();
    let mut pool = sensor.output_pool(&engine, config(4)).unwrap();
    for round in 0..6 {
        let mut frames = Vec::new();
        for slot in 0..4 {
            let n = [1, 65, 3, 17][slot];
            let state = if (slot + round) % 2 == 0 {
                &state
            } else {
                &second
            };
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            sensor
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    state,
                    &vec![Vec3::Y * 10.0; n],
                    &vec![Vec3::NEG_Y; n],
                    20.0,
                    &mut pool,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(batch.stats().input_upload_bytes, n as u64 * 32);
            let submitted = batch.submit().unwrap();
            frames.extend(submitted.into_frames());
        }
        assert_eq!(pool.available_slots().unwrap(), 0);
        let stats = sensor.query_stats();
        assert_eq!(stats.primitive_preparations, 0);
        assert_eq!(stats.mesh_preparations, 0);
        assert_eq!(stats.merge_preparations, 0);
        assert_eq!(stats.scene_builds, 1);
        assert_eq!(stats.scene_preparations, if round == 0 { 4 } else { 8 });
        assert_eq!(stats.refit_preparations, 2);
        drain(&engine);
        drop(frames);
        assert_eq!(pool.available_slots().unwrap(), 4);
    }
    assert_eq!(pool.stats().buffer_allocations, 24);
    assert_eq!(sensor.query_stats().encoded_queries, 24);
    assert_eq!(sensor.query_stats().submissions, 0);
    assert_eq!(sensor.mesh_build_count(), 0);
    assert_eq!(sensor.readback_stats().copied_bytes, 0);
}

#[test]
#[ignore = "requires GPU"]
fn invalid_rays_foreign_inputs_and_device_loss_reject_before_direct_query_work() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (model, mut state, scene, _) = fixture(&engine);
    let (_, foreign_state, foreign_scene, _) = fixture(&foreign);
    let mut raw_state = model.state().unwrap();
    raw_state
        .upload_to_device(engine.device().unwrap().as_ref(), engine.queue().unwrap())
        .unwrap();
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    for (origins, directions, range) in [
        (vec![Vec3::Y], vec![], 2.0),
        (vec![Vec3::Y; 2], vec![Vec3::X; 2], 2.0),
        (vec![Vec3::Y], vec![Vec3::ZERO], 2.0),
        (vec![Vec3::splat(f64::MAX)], vec![Vec3::X], 2.0),
        (vec![Vec3::Y], vec![Vec3::splat(f64::NAN)], 2.0),
        (vec![Vec3::Y], vec![Vec3::X], 0.0),
        (vec![Vec3::Y], vec![Vec3::X], f64::INFINITY),
    ] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &origins,
                    &directions,
                    range,
                    &mut pool,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.stats().input_upload_bytes, 0);
    }
    for (scene, state) in [
        (&scene, &foreign_state),
        (&foreign_scene, &state),
        (&scene, &raw_state),
    ] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode_scene(
                    &mut batch,
                    &engine,
                    scene,
                    state,
                    &[Vec3::Y],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut pool,
                    Default::default()
                )
                .is_err()
        );
    }
    state.body_q.host_mut().unwrap()[0].position = Vec3::Y;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut pool,
                    Default::default()
                )
                .is_err()
        );
    }
    assert_eq!(pool.stats().reserved_frames, 0);
    assert_eq!(sensor.query_stats().encoded_queries, 0);
    assert!(sensor.output_pool(&foreign, config(1)).is_err());
    assert!(
        sensor
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 1,
                    max_total_bytes: 71
                }
            )
            .is_err()
    );
    let foreign_sensor = Raycaster::new(&foreign, 1).unwrap();
    let mut foreign_pool = foreign_sensor.output_pool(&foreign, config(1)).unwrap();
    state.body_q.upload(engine.queue().unwrap()).unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &[Vec3::Y],
                    &[Vec3::NEG_Y],
                    20.0,
                    &mut foreign_pool,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.stats().input_upload_bytes, 0);
        assert_eq!(foreign_pool.stats().reserved_frames, 0);
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    sensor
        .encode_scene(
            &mut batch,
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
    engine.destroy_device().unwrap();
    assert!(batch.submit().is_err());
    assert!(pool.available_slots().is_err());
    assert!(sensor.output_pool(&engine, config(1)).is_err());
}
