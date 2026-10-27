#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sensors::{Raycaster, SensorScene};
use molla_sim::{ModelBuilder, TriMesh};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

fn grid() -> TriMesh {
    let mut mesh = TriMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for z in 0..6 {
        for x in 0..6 {
            mesh.vertices.push(Vec3::new(
                x as f64 * 0.4 - 1.0,
                x as f64 * z as f64 * 0.003,
                z as f64 * 0.4 - 1.0,
            ));
        }
    }
    for z in 0..5 {
        for x in 0..5 {
            let i = z * 6 + x;
            mesh.indices
                .extend([[i, i + 6, i + 1], [i + 1, i + 6, i + 7]]);
        }
    }
    mesh
}

fn triangle_t(origin: Vec3, direction: Vec3, v: [Vec3; 3]) -> Option<f64> {
    let a = v[1] - v[0];
    let b = v[2] - v[0];
    let h = direction.cross(b);
    let det = a.dot(h);
    if det.abs() < 1e-12 {
        return None;
    }
    let s = origin - v[0];
    let u = s.dot(h) / det;
    let q = s.cross(a);
    let w = direction.dot(q) / det;
    let t = b.dot(q) / det;
    (u >= 0.0 && w >= 0.0 && u + w <= 1.0 && t > 0.0).then_some(t)
}

#[test]
#[ignore = "requires GPU"]
fn many_instances_device_pose_refits_match_raw_and_double_precision_triangles() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mesh = grid();
    let mut builder = ModelBuilder::new();
    let a = builder.add_mesh(mesh.clone());
    let b = builder.add_mesh(TriMesh {
        vertices: vec![-Vec3::X, Vec3::X, Vec3::Z],
        indices: vec![[0, 1, 2]],
    });
    for i in 0..129 {
        let body = builder.add_body(Default::default());
        builder
            .add_shape_trimesh(body, if i % 3 == 0 { b } else { a }, Transform::IDENTITY)
            .unwrap();
    }
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Y,
            Transform::from_translation(Vec3::NEG_Y * 4.0),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    assert_eq!(scene.mesh_count(), 2);
    let mut accelerated = Raycaster::new(&engine, 260).unwrap();
    let mut raw = Raycaster::new(&engine, 260).unwrap();
    for frame in 0..3 {
        let poses: Vec<_> = (0..129)
            .map(|i| {
                Transform::new(
                    Vec3::new(
                        (i % 13) as f64 * 4.0 - 24.0 + frame as f64 * 0.13,
                        (i % 4) as f64 * 0.1,
                        (i / 13) as f64 * 4.0 - 20.0,
                    ),
                    Quat::from_rotation_y(i as f64 * 0.13)
                        * Quat::from_rotation_x(frame as f64 * 0.07),
                )
            })
            .collect();
        let gpu: Vec<GpuTransform> = poses.iter().copied().map(Into::into).collect();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&gpu),
        );
        state.body_q.mark_device_dirty();
        let mut origins = Vec::new();
        for pose in &poses {
            origins.push(pose.position + Vec3::new(0.13, 10.0, 0.11));
            origins.push(pose.position + Vec3::new(1.7, 10.0, 1.8));
        }
        origins.extend([Vec3::new(-100.0, 10.0, 0.0), Vec3::new(100.0, 10.0, 0.0)]);
        let directions = vec![Vec3::NEG_Y * 7.0; origins.len()];
        let actual = accelerated
            .cast_scene(&engine, &scene, &state, &origins, &directions, 30.0)
            .unwrap();
        let old = raw
            .cast(&engine, &model, &state, &origins, &directions, 30.0)
            .unwrap();
        assert!(state.body_q.host().is_err());
        for ((origin, hit), old) in origins.iter().zip(&actual).zip(old) {
            let mut expected = (origin.y + 4.0, 129u32);
            for (id, pose) in poses.iter().enumerate() {
                let source = if id % 3 == 0 {
                    &model.meshes[1]
                } else {
                    &model.meshes[0]
                };
                for tri in &source.indices {
                    let v = tri.map(|i| pose.transform_point(source.vertices[i as usize]));
                    if let Some(t) = triangle_t(*origin, Vec3::NEG_Y, v).filter(|&t| t < expected.0)
                    {
                        expected = (t, id as u32);
                    }
                }
            }
            assert_eq!(hit.shape_id, Some(expected.1));
            assert!((hit.t - expected.0).abs() < 5e-5, "{hit:?} vs {expected:?}");
            assert_eq!(hit.shape_id, old.shape_id);
            assert!((hit.t - old.t).abs() < 5e-5);
            assert!((hit.point - old.point).length() < 1e-4);
            assert!((hit.normal - old.normal).length() < 1e-5);
        }
    }
    assert_eq!(accelerated.query_stats().scene_builds, 1);
    assert_eq!(accelerated.query_stats().refit_preparations, 1);
    assert_eq!(accelerated.query_stats().scene_preparations, 1);
    assert_eq!(accelerated.query_stats().mesh_preparations, 0);
    assert_eq!(accelerated.mesh_build_count(), 0);
    assert_eq!(raw.mesh_build_count(), 2);
}

#[test]
#[ignore = "requires GPU"]
fn primitive_mesh_plane_ties_range_and_empty_queries_keep_legacy_semantics() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for primitive in [false, true] {
        let mut builder = ModelBuilder::new();
        let mesh = builder.add_mesh(TriMesh {
            vertices: vec![
                Vec3::new(-2.0, 1.0, -2.0),
                Vec3::new(2.0, 1.0, -2.0),
                Vec3::new(0.0, 1.0, 2.0),
            ],
            indices: vec![[0, 1, 2]],
        });
        for _ in 0..2 {
            builder
                .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
                .unwrap();
        }
        if primitive {
            builder
                .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
                .unwrap();
        }
        builder
            .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::from_translation(Vec3::Y))
            .unwrap();
        let mut model = builder.build().unwrap();
        let state = model.state().unwrap();
        model
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 3).unwrap();
        let origins = [Vec3::Y * 10.0, Vec3::Y * 10.0 + Vec3::X * 20.0, Vec3::ZERO];
        let directions = [Vec3::NEG_Y, Vec3::NEG_Y, Vec3::NEG_Y];
        let actual = query
            .cast_scene(&engine, &scene, &state, &origins, &directions, 20.0)
            .unwrap();
        let reference = query
            .cast(&engine, &model, &state, &origins, &directions, 20.0)
            .unwrap();
        assert_eq!(actual[0].shape_id, Some(2));
        for (hit, old) in actual.iter().zip(reference) {
            assert_eq!(hit.shape_id, old.shape_id);
            assert_eq!(hit.t, old.t);
            assert_eq!(hit.point, old.point);
            assert_eq!(hit.normal, old.normal);
        }
        assert!(
            query
                .cast_scene(
                    &engine,
                    &scene,
                    &state,
                    &origins[..1],
                    &directions[..1],
                    9.0
                )
                .unwrap()[0]
                .shape_id
                .is_none()
        );
        let before = query.query_stats();
        assert!(
            query
                .cast_scene(&engine, &scene, &state, &[], &[], 20.0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(query.query_stats().encoded_queries, before.encoded_queries);
    }
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let mut scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut query = Raycaster::new(&engine, 1).unwrap();
    assert_eq!(
        query
            .cast_scene(&engine, &scene, &state, &[Vec3::Y], &[Vec3::NEG_Y], 10.0)
            .unwrap()[0]
            .t,
        1.0
    );
    let empty = ModelBuilder::new().build().unwrap();
    scene.replace_model(&engine, &empty).unwrap();
    assert!(
        query
            .cast_scene(&engine, &scene, &state, &[Vec3::Y], &[Vec3::NEG_Y], 10.0)
            .unwrap()[0]
            .shape_id
            .is_none()
    );
}

#[test]
#[ignore = "requires GPU"]
fn all_finite_primitives_rotations_large_coordinates_and_axis_parallel_rays_match_raw() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let mut origins = Vec::new();
    let mut directions = Vec::new();
    for x in [-32768.0, -128.0, 0.0, 8192.0, 32768.0] {
        for kind in 0..6 {
            let position = Vec3::new(x, 0.0, kind as f64 * 8.0);
            let pose = Transform::new(
                position,
                Quat::from_rotation_z(0.17) * Quat::from_rotation_y(0.3),
            );
            match kind {
                0 => builder.add_shape_sphere(BodyId::NONE, 0.7, pose),
                1 => builder.add_shape_box(BodyId::NONE, Vec3::new(0.5, 0.8, 0.6), pose),
                2 => builder.add_shape_capsule(BodyId::NONE, 0.5, 0.3, pose),
                3 => builder.add_shape_cylinder(BodyId::NONE, 0.6, 0.4, pose),
                4 => builder.add_shape_cone(BodyId::NONE, 0.8, 0.5, pose),
                _ => builder.add_shape_ellipsoid(BodyId::NONE, Vec3::new(0.5, 0.8, 0.6), pose),
            }
            .unwrap();
            for (offset, direction) in [
                (Vec3::Y * 8.0, Vec3::NEG_Y),
                (Vec3::X * 8.0, Vec3::NEG_X),
                (Vec3::ZERO, Vec3::Z),
                (
                    Vec3::Y * 8.0 + Vec3::X * 0.1,
                    Vec3::new(1e-20, -1.0, -1e-20),
                ),
            ] {
                origins.push(position + offset);
                directions.push(direction);
            }
        }
    }
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut query = Raycaster::new(&engine, origins.len() as u32).unwrap();
    let actual = query
        .cast_scene(&engine, &scene, &state, &origins, &directions, 20.0)
        .unwrap();
    let expected = query
        .cast(&engine, &model, &state, &origins, &directions, 20.0)
        .unwrap();
    for (i, (hit, old)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(hit.shape_id, old.shape_id, "ray {i}");
        assert_eq!(hit.t, old.t, "ray {i}");
        assert!((hit.point - old.point).length() < 1e-5, "ray {i}");
        assert!((hit.normal - old.normal).length() < 1e-5, "ray {i}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn packed_geometry_budget_is_exact_and_replacement_retains_limit() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(grid());
    for i in 0..65 {
        builder
            .add_shape_trimesh(
                BodyId::NONE,
                mesh,
                Transform::from_translation(Vec3::X * i as f64 * 4.0),
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let reference = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let bytes = reference.query_device_bytes();
    let mut scene =
        SensorScene::from_model_with_query_budget(&engine, &model, Default::default(), bytes)
            .unwrap();
    assert!(
        SensorScene::from_model_with_query_budget(&engine, &model, Default::default(), bytes - 1)
            .is_err()
    );
    scene.replace_model(&engine, &model).unwrap();
    assert_eq!(scene.query_device_bytes(), bytes);
    let empty = ModelBuilder::new().build().unwrap();
    let mut tiny =
        SensorScene::from_model_with_query_budget(&engine, &empty, Default::default(), 100)
            .unwrap();
    assert_eq!(tiny.query_device_bytes(), 100);
    let stamp = tiny.stamp();
    assert!(tiny.replace_model(&engine, &model).is_err());
    assert_eq!(tiny.stamp(), stamp);
    assert_eq!(tiny.query_device_bytes(), 100);
}
