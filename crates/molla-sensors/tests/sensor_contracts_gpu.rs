#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sensors::{
    CameraChannels, LidarBatch, LidarPattern, LidarScan, Raycaster, ScanMetadata, SensorScene,
    TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackRing},
};
use molla_sim::{ModelBuilder, TriMesh};

const LENGTH_TOLERANCE: f64 = 2e-4;
const NORMAL_TOLERANCE: f64 = 4e-4;

#[test]
#[ignore = "requires GPU"]
fn capsule_hits_only_exposed_hemispheres_including_inside_and_sphere_limit() {
    let engine = Engine::new_gpu().unwrap();
    for half_height in [0.0, 0.6] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        builder
            .add_shape_capsule(body, half_height, 0.4, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        model
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let cases = [
            (Vec3::ZERO, Vec3::Y, half_height + 0.4),
            (Vec3::ZERO, Vec3::NEG_Y, half_height + 0.4),
            (Vec3::Y * half_height, Vec3::NEG_Y, 2.0 * half_height + 0.4),
            (Vec3::Y * (half_height + 0.2), Vec3::Y, 0.2),
            (Vec3::Y * 2.0, Vec3::NEG_Y, 1.6 - half_height),
            (Vec3::NEG_Y * 2.0, Vec3::Y, 1.6 - half_height),
            (Vec3::ZERO, Vec3::X, 0.4),
            (
                Vec3::new(2.0, half_height + 0.2, 0.0),
                Vec3::NEG_X,
                2.0 - 0.12f64.sqrt(),
            ),
        ];
        let origins: Vec<_> = cases.iter().map(|c| c.0).collect();
        let directions: Vec<_> = cases.iter().map(|c| c.1).collect();
        let mut rays = Raycaster::new(&engine, cases.len() as u32).unwrap();
        let owned = rays
            .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
            .unwrap();
        let raw = rays
            .cast(&engine, &model, &state, &origins, &directions, 10.0)
            .unwrap();
        for hits in [owned, raw] {
            for (hit, &(origin, direction, expected)) in hits.iter().zip(&cases) {
                assert_eq!(hit.shape_id, Some(0));
                assert!(
                    (hit.t - expected).abs() < 2e-5,
                    "half-height {half_height}, {origin:?}, {direction:?}, {} vs {expected}",
                    hit.t
                );
                let p = origin + direction * expected;
                let axis_point = Vec3::Y * p.y.clamp(-half_height, half_height);
                let normal = (p - axis_point).normalize();
                assert!((hit.point - p).length() < 2e-5);
                assert!((hit.normal - normal).length() < 2e-5);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn rigid_primitive_and_baked_mesh_surfaces_match_independent_metric_normal_references() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let vertices = [
        Vec3::new(-1.0, -0.5, -1.0),
        Vec3::new(1.0, 0.4, -1.0),
        Vec3::new(0.0, 0.2, 1.0),
    ]
    .map(|v| v * Vec3::new(2.0, 0.5, -3.0));
    let mesh_point = vertices.iter().copied().sum::<Vec3>() / 3.0;
    let mesh_normal = (vertices[1] - vertices[0])
        .cross(vertices[2] - vertices[0])
        .normalize();
    let cases = [
        (Vec3::new(0.6, 0.8, 0.0) * 0.7, Vec3::new(0.6, 0.8, 0.0)),
        (Vec3::new(0.2, -0.4, 0.0), Vec3::Z),
        (Vec3::new(0.8, 0.15, -0.1), Vec3::X),
        (
            Vec3::Y * 0.6 + Vec3::new(0.6, 0.8, 0.0) * 0.4,
            Vec3::new(0.6, 0.8, 0.0),
        ),
        (Vec3::X * 0.4, Vec3::X),
        (Vec3::X * 0.5, Vec3::new(1.0, 0.5, 0.0).normalize()),
        (
            Vec3::new(0.48, 0.96, 0.0),
            Vec3::new(0.48 / 0.8f64.powi(2), 0.96 / 1.2f64.powi(2), 0.0).normalize(),
        ),
        (mesh_point, mesh_normal),
    ];
    for (kind, (point, normal)) in cases.into_iter().enumerate() {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        let local = Transform::new(
            Vec3::new(0.4, -0.3, 0.2),
            Quat::from_rotation_z(0.23) * Quat::from_rotation_x(-0.31),
        );
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
        let mut rays = Raycaster::new(&engine, 4).unwrap();
        for step in 0..2 {
            let pose = Transform::new(
                Vec3::new(-10.0 + step as f64 * 21.0, 3.0, -7.0),
                Quat::from_rotation_y(0.61 + step as f64 * 0.17) * Quat::from_rotation_x(0.29),
            );
            let packed = GpuTransform::from(pose);
            queue.write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::bytes_of(&packed),
            );
            state.body_q.mark_device_dirty();
            let target = pose.transform_point(local.transform_point(point));
            let world_normal = pose.rotation * (local.rotation * normal);
            let mut origins = vec![target + world_normal * 3.25; 3];
            let mut directions: Vec<_> = [1e-250, 1.0, 1e250]
                .into_iter()
                .map(|s| -world_normal * s)
                .collect();
            let mut expected_distances = vec![3.25; 3];
            let mut expected_normals = vec![world_normal; 3];
            if kind == 1 || kind == 7 {
                origins.push(target - world_normal * 3.25);
                directions.push(world_normal);
                expected_distances.push(3.25);
                expected_normals.push(if kind == 7 {
                    -world_normal
                } else {
                    world_normal
                });
            } else {
                let center = pose.transform_point(local.position);
                origins.push(center);
                directions.push((target - center).normalize());
                expected_distances.push(point.length());
                expected_normals.push(world_normal);
            }
            let owned = rays
                .cast_scene(&engine, &scene, &state, &origins, &directions, 10.0)
                .unwrap();
            let raw = rays
                .cast(&engine, &model, &state, &origins, &directions, 10.0)
                .unwrap();
            for (path, hits) in [("owned", owned), ("raw", raw)] {
                for (i, hit) in hits.iter().enumerate() {
                    let label = format!("{path}, kind {kind}, pose {step}, ray {i}");
                    assert_eq!(hit.shape_id, Some(0), "{label}");
                    assert_eq!(hit.body_id, body, "{label}");
                    assert!(
                        (hit.t - expected_distances[i]).abs() < LENGTH_TOLERANCE,
                        "{label}: t={}",
                        hit.t
                    );
                    assert!(
                        (hit.point - target).length() < LENGTH_TOLERANCE,
                        "{label}: point={:?}",
                        hit.point
                    );
                    assert!(
                        (hit.normal - expected_normals[i]).length() < NORMAL_TOLERANCE,
                        "{label}: normal={:?}",
                        hit.normal
                    );
                    assert!((hit.normal.length() - 1.0).abs() < 2e-6, "{label}");
                }
            }
            assert!(state.body_q.host().is_err());
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn invalid_shape_poses_reject_transactionally_instead_of_encoding_scale() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    let mut scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let stamp = scene.stamp();
    for pose in [
        Transform::from_translation(Vec3::splat(f64::NAN)),
        Transform::from_translation(Vec3::splat(f64::INFINITY)),
        Transform::from_translation(Vec3::splat(f64::MAX)),
        Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        Transform::new(Vec3::ZERO, Quat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0)),
    ] {
        model.shape_transform.host_mut().unwrap()[0] = pose;
        assert!(scene.replace_model(&engine, &model).is_err());
        assert!(SensorScene::from_model(&engine, &model, Default::default()).is_err());
        assert_eq!(scene.stamp(), stamp);
        let hits = Raycaster::new(&engine, 1)
            .unwrap()
            .cast_scene(
                &engine,
                &scene,
                &state,
                &[Vec3::Z * 5.0],
                &[Vec3::NEG_Z],
                10.0,
            )
            .unwrap();
        assert_eq!(hits[0].shape_id, Some(0));
        assert!((hits[0].t - 4.0).abs() < 1e-6);
    }
}

#[test]
#[ignore = "requires GPU"]
fn camera_depth_is_radial_distance_with_strict_far_range_and_informational_near() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Z,
            Transform::from_translation(Vec3::NEG_Z * 5.0),
        )
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut config = TiledCameraConfig {
        width: 9,
        height: 9,
        channels: CameraChannels::ALL,
        near: 100.0,
        max_distance: 20.0,
        ..Default::default()
    };
    let mut camera = TiledCamera::new(&engine, config, 1).unwrap();
    camera.set_camera_pose(0, Transform::IDENTITY).unwrap();
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    let depth = camera.download_depth(&engine).unwrap();
    let tangent = (f64::from(config.fov_y_radians) * 0.5).tan();
    for y in 0..9 {
        for x in 0..9 {
            let u = 2.0 * (x as f64 + 0.5) / 9.0 - 1.0;
            let v = 1.0 - 2.0 * (y as f64 + 0.5) / 9.0;
            let expected = 5.0 * (1.0 + (u * tangent).powi(2) + (v * tangent).powi(2)).sqrt();
            assert!((depth[y * 9 + x] - expected).abs() < 2e-5);
        }
    }
    assert!(depth[0] > depth[40] + 0.5);
    assert!(
        camera
            .download_normal(&engine)
            .unwrap()
            .iter()
            .all(|&n| (n - Vec3::Z).length() < 1e-6)
    );
    config.max_distance = 5.0;
    camera.set_config(config).unwrap();
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    assert!(
        camera
            .download_depth(&engine)
            .unwrap()
            .iter()
            .all(|&t| t == -1.0)
    );
    assert!(
        camera
            .download_shape_index(&engine)
            .unwrap()
            .iter()
            .all(|&id| id == u32::MAX)
    );
    assert!(
        camera
            .download_normal(&engine)
            .unwrap()
            .iter()
            .all(|&n| n == Vec3::ZERO)
    );
}

#[test]
#[ignore = "requires GPU"]
fn terrestrial_azimuth_uses_up_cross_forward_on_cpu_and_gpu() {
    let engine = Engine::new_gpu().unwrap();
    let model = ModelBuilder::new().build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut lidar = LidarBatch::new(&engine, 1, 5).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 5 * 16,
            max_total_bytes: 5 * 16,
        },
    )
    .unwrap();
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.3),
    ] {
        let mut scan = ScanMetadata::new(Vec3::ZERO, 1, 5);
        scan.forward = rotation * Vec3::Z;
        scan.up = rotation * Vec3::Y;
        scan.theta_range = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let result = lidar
            .dispatch_scene(
                &engine,
                &scene,
                &state,
                &[LidarScan {
                    sensor_id: 1,
                    pattern: LidarPattern::Terrestrial(scan),
                }],
            )
            .unwrap();
        let ticket = ring
            .try_enqueue(
                &[result.hits.directions.readback_source(0..5).unwrap()],
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let frame = ring
            .wait(&ticket, std::time::Duration::from_secs(30))
            .unwrap();
        for (i, (gpu, axis)) in frame
            .channel_iter::<Vec3>(0)
            .unwrap()
            .zip([Vec3::Z, Vec3::X, Vec3::NEG_Z, Vec3::NEG_X, Vec3::Z])
            .enumerate()
        {
            let expected = rotation * axis;
            assert!((gpu - expected).length() < 2e-6);
            assert!((scan.rc_to_direction(0, i as u32) - expected).length() < 1e-12);
        }
    }
}
