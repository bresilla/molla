#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{
    Raycaster, SensorScene, SensorSceneOptions,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::{BodyParams, Model, ModelBuilder, State, TriMesh};
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires a GPU"]
fn retained_query_bindings_handle_ping_pong_and_geometry_replacement() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y));
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Y,
            Transform::new(-Vec3::Y, molla_math::Quat::IDENTITY),
        )
        .unwrap();
    let mesh = builder.add_mesh(quad(0.0));
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_trimesh(body, mesh, Transform::IDENTITY)
        .unwrap();
    let (mut model, state_a) = upload(&engine, builder);
    let mut state_b = model.state().unwrap();
    state_b.body_q.host_mut().unwrap()[0].position = Vec3::Y * 3.0;
    state_b
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut scene =
        SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut sensor = Raycaster::new(&engine, 65).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 520,
            max_total_bytes: 520,
        },
    )
    .unwrap();
    for phase in 0..2 {
        if phase == 1 {
            for vertex in &mut model.meshes[0].vertices {
                vertex.y += 2.0;
            }
            scene.replace_model(&engine, &model).unwrap();
        }
        let mut warm = None;
        for sample in 0..8 {
            let n = [1, 65, 3, 17][sample % 4];
            let range = if sample % 3 == 0 { 8.0 } else { 20.0 };
            let state = if sample % 2 == 0 { &state_a } else { &state_b };
            let expected = 10.0 - if sample % 2 == 0 { 1.0 } else { 3.0 } - phase as f64 * 2.0;
            let output = sensor
                .dispatch_scene(
                    &engine,
                    &scene,
                    state,
                    &vec![Vec3::Y * 10.0; n],
                    &vec![Vec3::NEG_Y; n],
                    range,
                )
                .unwrap();
            let ticket = ring
                .try_enqueue(
                    &[
                        output.distance.readback_source(0..n).unwrap(),
                        output.shape.readback_source(0..n).unwrap(),
                    ],
                    ReadbackStamp::default(),
                )
                .unwrap()
                .unwrap();
            let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
            for (distance, shape) in frame
                .channel_iter::<Real>(0)
                .unwrap()
                .zip(frame.channel_iter::<i32>(1).unwrap())
            {
                if expected > range {
                    assert_eq!(shape, -1);
                    assert_eq!(distance, -1.0);
                } else {
                    assert_eq!(shape, 2);
                    assert!((distance - expected).abs() < 1e-5);
                }
            }
            let stats = sensor.query_stats();
            if sample == 1 {
                warm = Some(stats);
            }
            if sample > 1 {
                let warm = warm.unwrap();
                assert_eq!(stats.primitive_preparations, warm.primitive_preparations);
                assert_eq!(stats.mesh_preparations, warm.mesh_preparations);
                assert_eq!(stats.merge_preparations, warm.merge_preparations);
                assert_eq!(stats.refit_preparations, warm.refit_preparations);
            }
        }
    }
    let stats = sensor.query_stats();
    assert_eq!(stats.encoded_queries, 16);
    assert_eq!(stats.submissions, 16);
    assert_eq!(stats.primitive_preparations, 0);
    assert_eq!(stats.mesh_preparations, 0);
    assert_eq!(stats.merge_preparations, 0);
    assert_eq!(stats.scene_builds, 2);
    assert_eq!(stats.scene_preparations, 14);
    assert_eq!(stats.refit_preparations, 4);
    assert_eq!(sensor.readback_stats().submissions, 0);
    assert_eq!(scene.mesh_count(), 1);
}

fn quad(y: f64) -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-4.0, y, -4.0),
            Vec3::new(4.0, y, -4.0),
            Vec3::new(4.0, y, 4.0),
            Vec3::new(-4.0, y, 4.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

fn upload(engine: &Engine, builder: ModelBuilder) -> (Model, State) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    (model, state)
}

fn scene(engine: &Engine) -> (Model, State) {
    let mut builder = ModelBuilder::new();
    for y in [0.0, 1.0] {
        let mesh = builder.add_mesh(quad(y));
        builder
            .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
            .unwrap();
    }
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 2.0));
    builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    upload(engine, builder)
}

#[test]
#[ignore = "requires a GPU"]
fn device_hits_chain_and_snapshots_survive_reuse_without_internal_collection() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (model, state) = scene(&engine);
    let mut sensor = Raycaster::new(&engine, 128).unwrap();
    let origins: Vec<_> = (0..65)
        .map(|i| Vec3::new([0.0, 2.0, 10.0][i % 3], 5.0, 0.0))
        .collect();
    let directions = vec![Vec3::NEG_Y * 7.0; 65];
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 65 * 44,
            max_total_bytes: 2 * 65 * 44 + 8,
        },
    )
    .unwrap();
    let mut doubled = DeviceBuffer::from_host(
        &engine,
        vec![0.0 as Real; 65],
        BufferUsage::StorageReadWrite,
    )
    .unwrap();
    let hits = sensor
        .dispatch(&engine, &model, &state, &origins, &directions, 20.0)
        .unwrap();
    assert_eq!(hits.ray_count, 65);
    assert!(hits.distance.host().is_err());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raycast device consumer"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> input: array<f32>;
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;
             @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                 if (id.x < 65u) { output[id.x] = 2.0 * input[id.x]; }
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: hits.distance.device_buffer().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: doubled.device_buffer().unwrap().as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(2, 1, 1);
    }
    queue.submit([encoder.finish()]);
    doubled.mark_device_dirty();
    let ticket = ring
        .try_enqueue(
            &[
                hits.distance.readback_source(0..65).unwrap(),
                hits.shape.readback_source(0..65).unwrap(),
                hits.point.readback_source(0..65).unwrap(),
                hits.normal.readback_source(0..65).unwrap(),
                doubled.readback_source(0..65).unwrap(),
            ],
            ReadbackStamp {
                sample_id: 1,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let clipped = sensor
        .dispatch(&engine, &model, &state, &origins, &directions, 3.0)
        .unwrap();
    let second = ring
        .try_enqueue(
            &[clipped.shape.readback_source(0..65).unwrap()],
            ReadbackStamp {
                sample_id: 2,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let frame2 = ring.wait(&second, Duration::from_secs(10)).unwrap();
    assert_eq!(frame2.stamp.sample_id, 2);
    for (i, shape) in frame2.channel_iter::<i32>(0).unwrap().enumerate() {
        assert_eq!(shape, if i % 3 == 0 { 2 } else { -1 });
    }
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.stamp.sample_id, 1);
    assert_eq!(frame.byte_len(), 65 * 44);
    let distances = frame.channel::<Real>(0).unwrap();
    let shapes = frame.channel::<i32>(1).unwrap();
    let points = frame.channel::<Vec3>(2).unwrap();
    let normals = frame.channel::<Vec3>(3).unwrap();
    let doubled = frame.channel::<Real>(4).unwrap();
    for i in 0..65 {
        assert_eq!(shapes[i], [2, 1, -1][i % 3]);
        let expected = [2.5, 4.0, -1.0][i % 3];
        assert!((distances[i] - expected).abs() < 1e-4);
        assert!((doubled[i] - expected * 2.0).abs() < 1e-4);
        assert!((normals[i] - Vec3::Y).length() < 1e-4);
        let expected_point = if i % 3 == 2 {
            Vec3::ZERO
        } else {
            origins[i] + Vec3::NEG_Y * expected
        };
        assert!((points[i] - expected_point).length() < 1e-4);
    }
    assert_eq!(sensor.mesh_cache_len(), 2);
    let stats = sensor.readback_stats();
    assert_eq!(
        (
            stats.staging_allocations,
            stats.submissions,
            stats.copied_bytes,
            stats.blocking_waits
        ),
        (0, 0, 0, 0)
    );
    for n in [65, 1, 0, 3] {
        let collected = sensor
            .cast(
                &engine,
                &model,
                &state,
                &origins[..n],
                &directions[..n],
                20.0,
            )
            .unwrap();
        assert_eq!(collected.len(), n);
    }
    let stats = sensor.readback_stats();
    assert_eq!(stats.staging_allocations, 1);
    assert_eq!(stats.submissions, 3);
    assert_eq!(stats.copied_bytes, (65 + 1 + 3) * 40);
}

#[test]
#[ignore = "requires a GPU"]
fn mesh_only_empty_and_equal_distance_queries_preserve_legacy_ordering() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad(1.0));
    for _ in 0..2 {
        builder
            .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
            .unwrap();
    }
    let (mesh_model, mesh_state) = upload(&engine, builder);
    let mut sensor = Raycaster::new(&engine, 2).unwrap();
    for _ in 0..4 {
        let hits = sensor
            .cast(
                &engine,
                &mesh_model,
                &mesh_state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0,
            )
            .unwrap();
        assert_eq!(hits[0].shape_id, Some(0));
        assert!((hits[0].t - 4.0).abs() < 1e-4);
    }
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad(1.0));
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 1.0, Transform::IDENTITY)
        .unwrap();
    let (model, state) = upload(&engine, builder);
    sensor.clear_mesh_cache();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::Y * 5.0],
            &[Vec3::NEG_Y],
            10.0,
        )
        .unwrap();
    assert_eq!(hits[0].shape_id, Some(1));
    let (empty, state) = upload(&engine, ModelBuilder::new());
    sensor.clear_mesh_cache();
    let hits = sensor
        .cast(
            &engine,
            &empty,
            &state,
            &[Vec3::Y * 5.0],
            &[Vec3::NEG_Y],
            10.0,
        )
        .unwrap();
    assert_eq!(hits[0].shape_id, None);
    assert_eq!(hits[0].t, Real::INFINITY);
    assert_eq!(hits[0].point, Vec3::ZERO);
    assert_eq!(hits[0].normal, Vec3::Y);
    assert_eq!(
        sensor
            .dispatch(&engine, &empty, &state, &[], &[], 1.0)
            .unwrap()
            .ray_count,
        0
    );
}

#[test]
#[ignore = "requires a GPU"]
fn foreign_engine_and_host_dirty_model_are_rejected_before_query() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let other = Engine::new_gpu().expect("second GPU engine required");
    let (mut model, state) = scene(&engine);
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    assert!(
        sensor
            .dispatch(&other, &model, &state, &[], &[], 1.0)
            .is_err()
    );
    model.shape_transform.host_mut().unwrap()[2].position = Vec3::X;
    assert!(
        sensor
            .dispatch(
                &engine,
                &model,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .is_err()
    );
    model
        .shape_transform
        .upload(engine.queue().unwrap())
        .unwrap();
    assert!(
        sensor
            .dispatch(
                &engine,
                &model,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .is_ok()
    );
}

#[test]
#[ignore = "requires a GPU"]
fn ordered_mesh_reductions_keep_each_shapes_identity_and_partial_occlusion() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad(0.0));
    for i in 0..17 {
        builder
            .add_shape_trimesh(
                BodyId::NONE,
                mesh,
                Transform::new(
                    Vec3::new(i as f64 * 10.0, (i % 3) as f64, 0.0),
                    molla_math::Quat::IDENTITY,
                ),
            )
            .unwrap();
    }
    let (model, state) = upload(&engine, builder);
    let origins: Vec<_> = (0..17)
        .map(|i| Vec3::new(i as f64 * 10.0, 5.0, 0.0))
        .collect();
    let directions = vec![Vec3::NEG_Y; 17];
    let mut sensor = Raycaster::new(&engine, 32).unwrap();
    for _ in 0..3 {
        let hits = sensor
            .cast(&engine, &model, &state, &origins, &directions, 10.0)
            .unwrap();
        for (i, hit) in hits.iter().enumerate() {
            assert_eq!(hit.shape_id, Some(i as u32));
            assert!((hit.t - (5.0 - (i % 3) as f64)).abs() < 1e-4);
        }
    }
    assert_eq!(sensor.readback_stats().submissions, 3);
    assert_eq!(sensor.readback_stats().copied_bytes, 3 * 17 * 40);
}
