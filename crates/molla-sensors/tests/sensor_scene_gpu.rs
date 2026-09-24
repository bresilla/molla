#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    LidarBatch, LidarPattern, LidarScan, Raycaster, ScanMetadata, SensorScene, SensorSceneOptions,
    UnsupportedGeometryPolicy,
};
use molla_sim::{BodyParams, Control, Heightfield, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn quad(y: f64) -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-2.0, y, -2.0),
            Vec3::new(2.0, y, -2.0),
            Vec3::new(2.0, y, 2.0),
            Vec3::new(-2.0, y, 2.0),
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

fn mesh_builder(y: f64) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad(y));
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    builder
}

fn oracle(origin: Vec3, dir: Vec3, vertices: [Vec3; 3]) -> Option<f64> {
    let e1 = vertices[1] - vertices[0];
    let e2 = vertices[2] - vertices[0];
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let offset = origin - vertices[0];
    let u = offset.dot(p) / det;
    let q = offset.cross(e1);
    let v = dir.dot(q) / det;
    let t = e2.dot(q) / det;
    (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0).then_some(t)
}

#[test]
#[ignore = "requires GPU"]
fn moving_rotated_instances_share_geometry_and_match_triangle_oracle() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad(0.25));
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y));
    let local = Transform::new(Vec3::new(0.2, 0.0, 0.0), Quat::from_rotation_x(0.23));
    builder.add_shape_trimesh(body, mesh, local).unwrap();
    builder
        .add_shape_trimesh(
            BodyId::NONE,
            mesh,
            Transform::new(Vec3::new(-6.0, 0.0, 0.0), Quat::IDENTITY),
        )
        .unwrap();
    let (model, mut state) = upload(&engine, builder);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    assert_eq!(scene.mesh_count(), 1);
    let mut sensor = Raycaster::new(&engine, 100).unwrap();
    let mut legacy = Raycaster::new(&engine, 100).unwrap();
    let origins: Vec<_> = (0..100)
        .map(|i| {
            Vec3::new(
                (i % 10) as f64 * 0.7 - 3.0,
                8.0,
                (i / 10) as f64 * 0.6 - 2.8,
            )
        })
        .collect();
    let direction = Vec3::new(0.02, -1.0, 0.01).normalize();
    for step in 0..4 {
        let pose = Transform::new(
            Vec3::new(step as f64 * 0.3, 1.0 + step as f64 * 0.2, 0.0),
            Quat::from_rotation_z(0.1 + step as f64 * 0.11),
        );
        let packed: molla_math::GpuTransform = pose.into();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&packed),
        );
        state.body_q.mark_device_dirty();
        let hits = sensor
            .cast_scene(
                &engine,
                &scene,
                &state,
                &origins,
                &vec![direction * 4.0; 100],
                20.0,
            )
            .unwrap();
        let legacy_hits = legacy
            .cast(
                &engine,
                &model,
                &state,
                &origins,
                &vec![direction; 100],
                20.0,
            )
            .unwrap();
        let mesh = &model.meshes[0];
        let vertices: Vec<_> = mesh
            .vertices
            .iter()
            .map(|&v| pose.transform_point(local.transform_point(v)))
            .collect();
        for (i, hit) in hits.iter().enumerate() {
            let expected = mesh
                .indices
                .iter()
                .filter_map(|tri| oracle(origins[i], direction, tri.map(|j| vertices[j as usize])))
                .min_by(f64::total_cmp);
            assert_eq!(hit.shape_id, legacy_hits[i].shape_id);
            assert_eq!(
                hit.shape_id.is_some(),
                expected.is_some(),
                "step {step}, ray {i}"
            );
            if let Some(t) = expected {
                assert_eq!(hit.shape_id, Some(0));
                assert_eq!(hit.body_id, body);
                assert!((hit.t - t).abs() < 1e-4);
                assert!((hit.point - (origins[i] + direction * t)).length() < 1e-4);
                let expected_normal = pose.rotation * (local.rotation * Vec3::Y);
                assert!((hit.normal - expected_normal).length() < 1e-4);
            }
        }
        assert!(state.body_q.host().is_err());
    }
    assert_eq!(legacy.mesh_build_count(), 1);
    assert_eq!(legacy.mesh_cache_len(), 1);
    assert_eq!(sensor.mesh_build_count(), 0);
    let other_hits = Raycaster::new(&engine, 1)
        .unwrap()
        .cast_scene(
            &engine,
            &scene,
            &state,
            &[Vec3::new(-6.0, 5.0, 0.0)],
            &[Vec3::NEG_Y],
            10.0,
        )
        .unwrap();
    assert_eq!(other_hits[0].shape_id, Some(1));
    assert!((other_hits[0].t - 4.75).abs() < 1e-4);
}

#[test]
#[ignore = "requires GPU"]
fn gpu_integrated_mesh_pose_updates_without_geometry_rebuild_or_pose_download() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 3.0));
    let mesh = builder.add_mesh(quad(0.0));
    builder
        .add_shape_trimesh(body, mesh, Transform::IDENTITY)
        .unwrap();
    let (model, mut current) = upload(&engine, builder);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut next = model.state().unwrap();
    next.upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut control = Control::for_model(&model);
    control
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let force = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    let integrate = molla_kernels::IntegrateBodiesKernel::new(engine.device().unwrap());
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    for step in 1..=5 {
        integrate
            .dispatch(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                &model,
                &current,
                &control,
                &force,
                &mut next,
                0.01,
            )
            .unwrap();
        assert!(next.body_q.host().is_err());
        let hits = sensor
            .cast_scene(
                &engine,
                &scene,
                &next,
                &[Vec3::Y * 6.0],
                &[Vec3::NEG_Y],
                10.0,
            )
            .unwrap();
        let expected = 3.0 + 9.81 * 0.01 * 0.01 * (step * (step + 1) / 2) as f64;
        assert!((hits[0].t - expected).abs() < 1e-4);
        assert!(next.body_q.host().is_err());
        std::mem::swap(&mut current, &mut next);
    }
    assert_eq!(sensor.mesh_build_count(), 0);
    assert_eq!(scene.mesh_count(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn legacy_edits_and_owned_revisions_cannot_reuse_stale_mesh_results() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (mut model, state) = upload(&engine, mesh_builder(0.0));
    let mut scene =
        SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let old = scene.stamp();
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    assert_eq!(
        sensor
            .cast(
                &engine,
                &model,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .t,
        5.0
    );
    for vertex in &mut model.meshes[0].vertices {
        vertex.y = 2.0;
    }
    assert_eq!(
        sensor
            .cast(
                &engine,
                &model,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .t,
        3.0
    );
    assert_eq!(sensor.mesh_build_count(), 2);
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 8,
            max_total_bytes: 8,
        },
    )
    .unwrap();
    let view = sensor
        .dispatch_scene(
            &engine,
            &scene,
            &state,
            &[Vec3::Y * 5.0],
            &[Vec3::NEG_Y],
            10.0,
        )
        .unwrap();
    assert_eq!(view.scene_stamp, Some(old));
    let ticket = ring
        .try_enqueue(
            &[view.distance.readback_source(0..1).unwrap()],
            ReadbackStamp {
                sensor_id: old.scene_id,
                scene_revision: old.geometry_revision,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    scene.replace_model(&engine, &model).unwrap();
    assert_eq!(scene.stamp().scene_id, old.scene_id);
    assert_eq!(scene.stamp().geometry_revision, old.geometry_revision + 1);
    assert!(scene.resolve_body(old, 0).is_err());
    assert_eq!(scene.resolve_body(scene.stamp(), 0).unwrap(), BodyId::NONE);
    assert_eq!(
        sensor
            .cast_scene(
                &engine,
                &scene,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .t,
        3.0
    );
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.channel::<f64>(0).unwrap(), [5.0]);
    assert_eq!(frame.stamp.scene_revision, old.geometry_revision);
    let current_stamp = scene.stamp();
    model.meshes[0].indices[0][0] = 999;
    assert!(scene.replace_model(&engine, &model).is_err());
    assert_eq!(scene.stamp(), current_stamp);
    assert_eq!(
        sensor
            .cast_scene(
                &engine,
                &scene,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .t,
        3.0
    );
    assert!(
        sensor
            .cast(
                &engine,
                &model,
                &state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .is_err()
    );
    let (another, another_state) = upload(&engine, mesh_builder(-1.0));
    assert_eq!(
        sensor
            .cast(
                &engine,
                &another,
                &another_state,
                &[Vec3::Y * 5.0],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .t,
        6.0
    );
    let (replacement, empty_state) = upload(&engine, ModelBuilder::new());
    scene.replace_model(&engine, &replacement).unwrap();
    assert!(scene.resolve_body(current_stamp, 0).is_err());
    assert!(scene.resolve_body(scene.stamp(), 0).is_err());
    assert!(
        sensor
            .cast_scene(
                &engine,
                &scene,
                &empty_state,
                &[Vec3::Y],
                &[Vec3::NEG_Y],
                10.0
            )
            .unwrap()[0]
            .shape_id
            .is_none()
    );
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    assert!(
        sensor
            .dispatch_scene(&foreign, &scene, &empty_state, &[], &[], 1.0)
            .is_err()
    );
}

#[test]
#[ignore = "requires GPU"]
fn terrain_triangles_capabilities_and_lidar_scene_stamps() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let terrain = Heightfield {
        nx: 2,
        nz: 2,
        cell_size: 1.0,
        cell_size_z: 1.0,
        origin: Vec3::ZERO,
        heights: vec![0.0, 1.0, 2.0, 4.0],
    };
    let mut builder = ModelBuilder::new();
    let field = builder.add_heightfield(terrain.clone());
    builder
        .add_shape_heightfield(BodyId::NONE, field, Transform::IDENTITY)
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
    assert!(
        SensorScene::from_model(
            &engine,
            &model,
            SensorSceneOptions {
                max_triangles: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    assert_eq!(scene.mesh_count(), 2);
    let origins = [
        Vec3::new(0.2, 8.0, 0.2),
        Vec3::new(0.8, 8.0, 0.8),
        Vec3::new(5.0, 8.0, 0.0),
    ];
    let hits = Raycaster::new(&engine, 3)
        .unwrap()
        .cast_scene(&engine, &scene, &state, &origins, &[Vec3::NEG_Y; 3], 20.0)
        .unwrap();
    for i in 0..2 {
        let (height, normal) = terrain
            .height_and_normal(origins[i].x, origins[i].z)
            .unwrap();
        assert_eq!(hits[i].shape_id, Some(0));
        assert!((hits[i].t - (8.0 - height)).abs() < 1e-4);
        assert!((hits[i].normal - normal).length() < 1e-4);
    }
    assert_eq!(hits[2].shape_id, Some(1));
    assert_eq!(hits[2].t, 5.0);
    let mut batch = LidarBatch::new(&engine, 1, 1).unwrap();
    let mut scan = ScanMetadata::new(origins[2], 1, 1);
    scan.forward = Vec3::NEG_Y;
    scan.up = Vec3::Z;
    scan.theta_range = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    let result = batch
        .dispatch_scene(
            &engine,
            &scene,
            &state,
            &[LidarScan {
                sensor_id: 4,
                pattern: LidarPattern::Terrestrial(scan),
            }],
        )
        .unwrap();
    assert_eq!(result.hits.scene_stamp, Some(scene.stamp()));
    let mut builder = ModelBuilder::new();
    let hull = builder.add_hull(vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]);
    builder
        .add_shape_convex_hull(BodyId::NONE, hull, Transform::IDENTITY)
        .unwrap();
    let (unsupported, _) = upload(&engine, builder);
    assert!(SensorScene::from_model(&engine, &unsupported, Default::default()).is_err());
    let partial = SensorScene::from_model(
        &engine,
        &unsupported,
        SensorSceneOptions {
            unsupported: UnsupportedGeometryPolicy::ReportAndSkip,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(partial.unsupported_shapes().len(), 1);
    assert_eq!(partial.unsupported_shapes()[0].kind, "ConvexHull");
    assert_eq!(partial.unsupported_shapes()[0].shape_index, 0);
}
