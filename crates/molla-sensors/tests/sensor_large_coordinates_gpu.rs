#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_kernels::CameraTraceKernel;
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sensors::{
    CameraChannels, Raycaster, SceneRebuildPolicy, SensorScene, TiledCamera, TiledCameraConfig,
};
use molla_sim::{ModelBuilder, TriMesh};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());
const DISTANCE_ERROR: f64 = 2e-4;
const NORMAL_ERROR: f64 = 4e-4;

fn encoded_point(point: Vec3) -> Vec3 {
    point.as_vec3().as_dvec3()
}

fn point_error(point: Vec3) -> f64 {
    let magnitude = point.abs().max_element().max(1.0) as f32;
    2.0 * f64::from(magnitude.next_up() - magnitude) + DISTANCE_ERROR
}

fn check_rotated_large_surfaces(owned: bool) {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    let vertices = [
        Vec3::new(-2.0, -0.25, 3.0),
        Vec3::new(2.0, 0.2, 3.0),
        Vec3::new(0.0, 0.1, -3.0),
    ]
    .map(encoded_point);
    let mesh_point = vertices.iter().copied().sum::<Vec3>() / 3.0;
    let mesh_normal = (vertices[1] - vertices[0])
        .cross(vertices[2] - vertices[0])
        .normalize();
    let surfaces = [
        (Vec3::new(0.42, 0.56, 0.0), Vec3::new(0.6, 0.8, 0.0)),
        (Vec3::new(0.2, -0.4, 0.0), Vec3::Z),
        (Vec3::new(0.8, 0.15, -0.1), Vec3::X),
        (Vec3::new(0.24, 0.92, 0.0), Vec3::new(0.6, 0.8, 0.0)),
        (Vec3::X * 0.4, Vec3::X),
        (Vec3::X * 0.5, Vec3::new(1.0, 0.5, 0.0).normalize()),
        (
            Vec3::new(0.48, 0.96, 0.0),
            Vec3::new(0.48 / 0.8f64.powi(2), 0.96 / 1.2f64.powi(2), 0.0).normalize(),
        ),
        (mesh_point, mesh_normal),
    ];
    for (kind, (point, normal)) in surfaces.into_iter().enumerate() {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        let local = Transform::IDENTITY;
        match kind {
            0 => builder.add_shape_sphere(body, 0.7, local),
            1 => builder.add_shape_plane(body, Vec3::Z, local),
            2 => builder.add_shape_box(body, Vec3::new(0.8, 0.6, 0.7), local),
            3 => builder.add_shape_capsule(body, 0.6, 0.4, local),
            4 => builder.add_shape_cylinder(body, 0.6, 0.4, local),
            5 => builder.add_shape_cone(body, 1.0, 1.0, local),
            6 => builder.add_shape_ellipsoid(body, Vec3::new(0.8, 1.2, 0.6), local),
            _ => {
                let mesh = builder.add_mesh(TriMesh {
                    vertices: vertices.to_vec(),
                    indices: vec![[0, 1, 2]],
                });
                builder.add_shape_trimesh(body, mesh, local)
            }
        }
        .unwrap();
        let mut model = builder.build().unwrap();
        model.upload_to_device(device, queue).unwrap();
        let mut state = model.state().unwrap();
        state.upload_to_device(device, queue).unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 4).unwrap();
        for offset in [0.0, -1024.0, 8192.0, -32768.0, -1_000_000.0, 1_000_000.0] {
            let packed = GpuTransform::from(Transform::new(
                Vec3::new(offset, -0.75 * offset, 0.5 * offset),
                Quat::from_rotation_y(0.61) * Quat::from_rotation_x(0.29),
            ));
            queue.write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::bytes_of(&packed),
            );
            state.body_q.mark_device_dirty();
            let mut pose = Transform::from(packed);
            pose.rotation = pose.rotation.normalize();
            let target = pose.transform_point(point);
            let outward = pose.rotation * normal;
            let outside = encoded_point(target + outward * 3.25);
            let mut origins = vec![outside; 3];
            let mut directions: Vec<_> = [1e-250, 1.0, 1e250]
                .map(|scale| (target - outside).normalize() * scale)
                .into();
            let mut normals = vec![outward; 3];
            if kind == 1 || kind == 7 {
                let back = encoded_point(target - outward * 3.25);
                origins.push(back);
                directions.push((target - back).normalize());
                normals.push(if kind == 7 { -outward } else { outward });
            } else {
                origins.push(pose.position);
                directions.push((target - pose.position).normalize());
                normals.push(outward);
            }
            let hits = if owned {
                query.cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
            } else {
                query.cast(&engine, &model, &state, &origins, &directions, 10.0)
            }
            .unwrap();
            for (i, hit) in hits.iter().enumerate() {
                let label = format!("kind={kind}, offset={offset}, owned={owned}, ray={i}");
                let expected_t = (target - origins[i]).length();
                assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
                assert_eq!(hit.body_id, body, "{label}");
                assert!(
                    (hit.t - expected_t).abs() < DISTANCE_ERROR,
                    "{label}: distance {} vs {expected_t}",
                    hit.t
                );
                assert!(
                    (hit.point - target).length() < point_error(target),
                    "{label}: point {:?} vs {target:?}",
                    hit.point
                );
                assert!(
                    (hit.normal - normals[i]).length() < NORMAL_ERROR,
                    "{label}: normal {:?} vs {:?}",
                    hit.normal,
                    normals[i]
                );
                assert!((hit.normal.length() - 1.0).abs() < 2e-6, "{label}");
            }
            assert!(state.body_q.host().is_err());
        }
        assert_eq!(query.query_stats().scene_builds, u64::from(owned));
    }
}

#[test]
#[ignore = "requires GPU"]
fn raw_rotated_large_world_surfaces_match_independent_metric_references() {
    check_rotated_large_surfaces(false);
}

#[test]
#[ignore = "requires GPU"]
fn owned_rotated_large_world_surfaces_match_independent_metric_references() {
    check_rotated_large_surfaces(true);
}

fn buffer<T: BufferElement>(engine: &Engine, value: T) -> Buffer<T> {
    let mut buffer = Buffer::from_vec(vec![value], BufferUsage::StorageReadWrite);
    buffer.ensure_device(engine.device().unwrap()).unwrap();
    buffer.upload(engine.queue().unwrap()).unwrap();
    buffer
}

#[test]
#[ignore = "requires GPU"]
fn large_world_camera_depth_matches_an_independent_plane_reference() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder
        .add_shape_plane(body, Vec3::Z, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut camera = TiledCamera::new(
        &engine,
        TiledCameraConfig {
            width: 1,
            height: 1,
            max_distance: 10.0,
            channels: CameraChannels::ALL,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let trace = CameraTraceKernel::new(device);
    for offset in [0.0, -1024.0, 8192.0, -32768.0, -1_000_000.0, 1_000_000.0] {
        let packed = GpuTransform::from(Transform::new(
            Vec3::new(offset, -0.75 * offset, 0.5 * offset),
            Quat::from_rotation_y(0.61) * Quat::from_rotation_x(0.29),
        ));
        queue.write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&packed),
        );
        state.body_q.mark_device_dirty();
        let mut pose = Transform::from(packed);
        pose.rotation = pose.rotation.normalize();
        let normal = pose.rotation * Vec3::Z;
        let target = pose.transform_point(Vec3::new(0.2, -0.4, 0.0));
        let origin = encoded_point(target + normal * 3.25);
        let direction = (target - origin).normalize();
        let expected = normal.dot(pose.position - origin) / normal.dot(direction);
        camera
            .set_camera_look_at(0, origin, target, Vec3::Y)
            .unwrap();
        for owned in [false, true] {
            if owned {
                camera.dispatch_scene(&engine, &scene, &state)
            } else {
                camera.dispatch(&engine, &model, &state)
            }
            .unwrap();
            let actual = camera.download_depth(&engine).unwrap()[0];
            assert!(
                (actual - expected).abs() < DISTANCE_ERROR,
                "offset={offset}, owned={owned}: {actual} vs {expected}"
            );
            assert_eq!(camera.download_shape_index(&engine).unwrap(), &[0]);
            assert!((camera.download_normal(&engine).unwrap()[0] - normal).length() < NORMAL_ERROR);
        }
        let origins = buffer(&engine, origin);
        let directions = buffer(&engine, direction);
        let mut depth = buffer(&engine, 0.0f64);
        let mut color = buffer(&engine, 0u32);
        let mut normals = buffer(&engine, Vec3::ZERO);
        let mut albedo = buffer(&engine, 0u32);
        let mut ids = buffer(&engine, u32::MAX);
        trace
            .dispatch(
                device,
                queue,
                &model,
                state.body_q.device_buffer_current().unwrap(),
                &origins,
                &directions,
                &mut depth,
                &mut color,
                &mut normals,
                &mut albedo,
                &mut ids,
                1,
                1,
                1,
                10.0,
                CameraChannels::ALL.bits(),
                Vec3::ONE,
                Vec3::NEG_Y,
                Vec3::ONE,
            )
            .unwrap();
        depth.download(device, queue).unwrap();
        normals.download(device, queue).unwrap();
        ids.download(device, queue).unwrap();
        assert!(
            (depth.host().unwrap()[0] - expected).abs() < DISTANCE_ERROR,
            "low-level camera, offset={offset}: {} vs {expected}",
            depth.host().unwrap()[0]
        );
        assert!((normals.host().unwrap()[0] - normal).length() < NORMAL_ERROR);
        assert_eq!(ids.host().unwrap(), &[0]);
        assert!(state.body_q.host().is_err());
    }
}

fn triangle_reference(origin: Vec3, vertices: [Vec3; 3]) -> Option<(f64, Vec3)> {
    let direction = Vec3::NEG_Y;
    let a = vertices[1] - vertices[0];
    let b = vertices[2] - vertices[0];
    let h = direction.cross(b);
    let determinant = a.dot(h);
    if determinant.abs() < 1e-12 {
        return None;
    }
    let s = origin - vertices[0];
    let u = s.dot(h) / determinant;
    let q = s.cross(a);
    let v = direction.dot(q) / determinant;
    let t = b.dot(q) / determinant;
    if u < 0.0 || v < 0.0 || u + v > 1.0 || t <= 0.0 {
        return None;
    }
    let mut normal = a.cross(b).normalize();
    if normal.dot(direction) > 0.0 {
        normal = -normal;
    }
    Some((t, normal))
}

#[test]
fn triangle_oracle_and_world_point_budgets_have_independent_known_values() {
    let vertices = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert_eq!(
        triangle_reference(Vec3::Y * 2.0, vertices),
        Some((2.0, Vec3::Y))
    );
    assert_eq!(
        triangle_reference(Vec3::Y * 2.0, [vertices[2], vertices[1], vertices[0]]),
        Some((2.0, Vec3::Y))
    );
    assert!(triangle_reference(Vec3::new(3.0, 2.0, 0.0), vertices).is_none());
    assert!(triangle_reference(Vec3::NEG_Y, vertices).is_none());
    assert!(triangle_reference(Vec3::Y, [Vec3::ZERO; 3]).is_none());
    assert!((point_error(Vec3::splat(1_000_000.0)) - 0.1252).abs() < 1e-15);
    assert_eq!(
        point_error(Vec3::splat(1_000_000.0)),
        point_error(Vec3::splat(-1_000_000.0))
    );
}

#[test]
#[ignore = "requires GPU"]
fn large_world_moving_instances_and_rebuilds_match_brute_force_triangles() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    let mut mesh = TriMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for z in 0..3 {
        for x in 0..3 {
            mesh.vertices.push(encoded_point(Vec3::new(
                x as f64 - 1.0,
                (x * z) as f64 * 0.07,
                z as f64 - 1.0,
            )));
        }
    }
    for z in 0..2 {
        for x in 0..2 {
            let i = z * 3 + x;
            mesh.indices
                .extend([[i, i + 3, i + 1], [i + 1, i + 3, i + 4]]);
        }
    }
    let mut builder = ModelBuilder::new();
    let mesh_id = builder.add_mesh(mesh.clone());
    for _ in 0..66 {
        let body = builder.add_body(Default::default());
        builder
            .add_shape_trimesh(body, mesh_id, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    assert_eq!(scene.mesh_count(), 1);
    for quality_ratio_percent in [None, Some(125)] {
        let mut query = Raycaster::new(&engine, 132).unwrap();
        query
            .set_scene_rebuild_policy(SceneRebuildPolicy {
                interval: 1,
                min_leaves: 2,
                quality_ratio_percent,
                ..Default::default()
            })
            .unwrap();
        for (frame, offset) in [0.0, -32768.0, -1_000_000.0, 1_000_000.0]
            .into_iter()
            .enumerate()
        {
            let packed: Vec<GpuTransform> = (0..66)
                .map(|i| {
                    let position = (i * (frame * 2 + 1)) % 67;
                    Transform::new(
                        Vec3::new(
                            offset + (position % 11) as f64 * 4.0,
                            -offset * 0.75 + (i % 4) as f64 * 0.1,
                            offset * 0.5 + (position / 11) as f64 * 4.0,
                        ),
                        Quat::from_rotation_y(i as f64 * 0.13)
                            * Quat::from_rotation_x(frame as f64 * 0.07),
                    )
                    .into()
                })
                .collect();
            queue.write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&packed),
            );
            state.body_q.mark_device_dirty();
            let poses: Vec<_> = packed
                .into_iter()
                .map(|packed| {
                    let mut pose = Transform::from(packed);
                    pose.rotation = pose.rotation.normalize();
                    pose
                })
                .collect();
            let origins: Vec<_> = poses
                .iter()
                .flat_map(|pose| {
                    [Vec3::new(0.125, 4.0, 0.25), Vec3::new(1.75, 4.0, 1.75)]
                        .map(|delta| encoded_point(pose.position + delta))
                })
                .collect();
            let directions = vec![Vec3::NEG_Y * 7.0; origins.len()];
            let actual = query
                .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
                .unwrap();
            let mut hits = 0;
            for (ray, (origin, hit)) in origins.iter().zip(&actual).enumerate() {
                let mut reference = None;
                let mut distance = 10.0;
                for (id, pose) in poses.iter().enumerate() {
                    for triangle in &mesh.indices {
                        let vertices =
                            triangle.map(|i| pose.transform_point(mesh.vertices[i as usize]));
                        if let Some((t, normal)) = triangle_reference(*origin, vertices) {
                            if t < distance {
                                distance = t;
                                reference = Some((id as u32, normal));
                            }
                        }
                    }
                }
                let label = format!("frame={frame}, quality={quality_ratio_percent:?}, ray={ray}");
                if let Some((id, normal)) = reference {
                    hits += 1;
                    let point = *origin - Vec3::Y * distance;
                    assert_eq!(hit.shape_id, Some(id), "{label}");
                    assert_eq!(hit.body_id, BodyId(id), "{label}");
                    assert!(
                        (hit.t - distance).abs() < DISTANCE_ERROR,
                        "{label}: {hit:?} vs {distance}"
                    );
                    assert!((hit.normal - normal).length() < NORMAL_ERROR, "{label}");
                    assert!((hit.point - point).length() < point_error(point), "{label}");
                } else {
                    assert_eq!(hit.shape_id, None, "{label}");
                    assert_eq!(hit.body_id, BodyId::NONE, "{label}");
                    assert_eq!(hit.t, f64::INFINITY, "{label}");
                    assert_eq!(hit.point, Vec3::ZERO, "{label}");
                    assert_eq!(hit.normal, Vec3::Y, "{label}");
                }
            }
            assert_eq!(hits, 66);
            assert!(state.body_q.host().is_err());
        }
        assert_eq!(query.query_stats().scene_builds, 1);
        assert_eq!(query.query_stats().mesh_preparations, 0);
        assert_eq!(query.mesh_build_count(), 0);
        if quality_ratio_percent.is_none() {
            assert_eq!(query.query_stats().scene_rebuilds, 4);
        } else {
            assert_eq!(query.query_stats().scene_quality_checks, 4);
        }
    }
}
