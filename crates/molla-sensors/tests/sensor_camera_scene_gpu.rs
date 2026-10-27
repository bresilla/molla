#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::readback::{
    ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp, ReadbackTicket,
};
use molla_sensors::{
    CameraChannels as Ch, SensorScene, SensorSceneOptions, TiledCamera, TiledCameraConfig,
};
use molla_sim::{BodyParams, Heightfield, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[derive(Clone)]
enum Object {
    Triangle([Vec3; 3], u32),
    Sphere(Vec3, Real, u32),
    Plane(Real, u32),
}

fn intersect(
    origin: Vec3,
    direction: Vec3,
    objects: &[Object],
    max: Real,
) -> Option<(Real, Vec3, u32)> {
    let mut best = None;
    let mut limit = max;
    for object in objects {
        let (t, normal, id) = match *object {
            Object::Triangle(vertices, id) => {
                let e1 = vertices[1] - vertices[0];
                let e2 = vertices[2] - vertices[0];
                let p = direction.cross(e2);
                let det = e1.dot(p);
                if det.abs() < 1e-12 {
                    continue;
                }
                let offset = origin - vertices[0];
                let q = offset.cross(e1);
                let u = offset.dot(p) / det;
                let v = direction.dot(q) / det;
                if u < 0.0 || v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let mut normal = e1.cross(e2).normalize();
                if normal.dot(direction) > 0.0 {
                    normal = -normal;
                }
                (e2.dot(q) / det, normal, id)
            }
            Object::Sphere(center, radius, id) => {
                let delta = origin - center;
                let b = delta.dot(direction);
                let discriminant = b * b - delta.length_squared() + radius * radius;
                if discriminant < 0.0 {
                    continue;
                }
                let near = -b - discriminant.sqrt();
                let far = -b + discriminant.sqrt();
                let t = if near > 0.0 { near } else { far };
                (t, (origin + direction * t - center).normalize(), id)
            }
            Object::Plane(z, id) => {
                if direction.z.abs() < 1e-12 {
                    continue;
                }
                ((z - origin.z) / direction.z, Vec3::Z, id)
            }
        };
        if t > 0.0 && t < limit {
            best = Some((t, normal, id));
            limit = t;
        }
    }
    best
}

fn quad() -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-2.0, -2.0, 0.0),
            Vec3::new(2.0, -2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(-2.0, 2.0, 0.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

fn mesh_objects(mesh: &TriMesh, pose: Transform, id: u32) -> Vec<Object> {
    mesh.indices
        .iter()
        .map(|indices| {
            Object::Triangle(
                indices.map(|i| pose.position + pose.rotation * mesh.vertices[i as usize]),
                id,
            )
        })
        .collect()
}

fn upload(engine: &Engine, builder: ModelBuilder, colors: &[Vec3]) -> (Model, State) {
    let mut model = builder.build().unwrap();
    model
        .shape_color
        .host_mut()
        .unwrap()
        .copy_from_slice(colors);
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state)
}

fn camera(engine: &Engine, width: u32, height: u32, count: usize) -> TiledCamera {
    TiledCamera::new(
        engine,
        TiledCameraConfig {
            width,
            height,
            ambient: Vec3::ONE,
            light_color: Vec3::ZERO,
            ..Default::default()
        },
        count,
    )
    .unwrap()
}

fn ring(engine: &Engine, pixels: usize) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: pixels as u64 * 32,
            max_total_bytes: pixels as u64 * 64,
        },
    )
    .unwrap()
}

fn enqueue(camera: &TiledCamera, ring: &mut ReadbackRing, sample: u64) -> ReadbackTicket {
    let n = camera.config().width as usize
        * camera.config().height as usize
        * camera.num_cameras() as usize;
    let sources: Vec<_> = [
        Ch::DEPTH,
        Ch::COLOR,
        Ch::NORMAL,
        Ch::ALBEDO,
        Ch::SHAPE_INDEX,
    ]
    .map(|ch| camera.readback_source(ch, 0..n).unwrap())
    .into();
    ring.try_enqueue(
        &sources,
        ReadbackStamp {
            sample_id: sample,
            scene_revision: camera
                .channel_scene_stamp(Ch::DEPTH)
                .unwrap()
                .map_or(0, |stamp| stamp.geometry_revision),
            ..Default::default()
        },
    )
    .unwrap()
    .unwrap()
}

fn packed(color: Vec3) -> u32 {
    let c = color.clamp(Vec3::ZERO, Vec3::ONE) * 255.0;
    0xff000000 | c.x.round() as u32 | ((c.y.round() as u32) << 8) | ((c.z.round() as u32) << 16)
}

fn check(frame: &ReadbackFrame, camera: &TiledCamera, objects: &[Object], colors: &[Vec3]) {
    let cfg = camera.config();
    let w = cfg.width;
    let h = cfg.height;
    let depth = frame.channel::<Real>(0).unwrap();
    let color = frame.channel::<u32>(1).unwrap();
    let normals = frame.channel::<Vec3>(2).unwrap();
    let albedo = frame.channel::<u32>(3).unwrap();
    let ids = frame.channel::<u32>(4).unwrap();
    for (cam, pose) in camera.poses_host().unwrap().iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let i = cam * (w * h) as usize + (y * w + x) as usize;
                let u = 2.0 * (x as Real + 0.5) / w as Real - 1.0;
                let v = 1.0 - 2.0 * (y as Real + 0.5) / h as Real;
                let tangent = (cfg.fov_y_radians as Real * 0.5).tan();
                let direction = (pose.rotation
                    * Vec3::new(u * tangent * w as Real / h as Real, v * tangent, -1.0)
                        .normalize())
                .normalize();
                match intersect(pose.position, direction, objects, cfg.max_distance as Real) {
                    Some((t, n, id)) => {
                        assert_eq!(ids[i], id, "pixel {i}");
                        assert!(
                            (depth[i] - t).abs() < 5e-4,
                            "pixel {i}: {} vs {t}",
                            depth[i]
                        );
                        assert!(
                            (normals[i] - n).length() < 5e-4,
                            "pixel {i}: {:?} vs {n:?}",
                            normals[i]
                        );
                        assert_eq!(color[i], packed(colors[id as usize]));
                        assert_eq!(albedo[i], color[i]);
                    }
                    None => {
                        assert_eq!(ids[i], u32::MAX);
                        assert_eq!(depth[i], -1.0);
                        assert_eq!(normals[i], Vec3::ZERO);
                        assert_eq!(color[i], 0);
                        assert_eq!(albedo[i], 0);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn moving_meshes_primitive_occlusion_and_two_cameras_match_independent_oracle() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let moving = builder.add_body(BodyParams::new());
    let sphere_body = builder.add_body(BodyParams::new().with_position(Vec3::new(-0.7, 0.2, 1.0)));
    let mesh = quad();
    let mesh_id = builder.add_mesh(mesh.clone());
    let local = Transform::new(Vec3::new(0.1, 0.0, 0.0), Quat::from_rotation_x(0.13));
    let static_pose = Transform::new(Vec3::new(4.5, 0.0, -1.0), Quat::from_rotation_y(-0.17));
    builder.add_shape_trimesh(moving, mesh_id, local).unwrap();
    builder
        .add_shape_trimesh(BodyId::NONE, mesh_id, static_pose)
        .unwrap();
    builder
        .add_shape_sphere(sphere_body, 0.45, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Z,
            Transform::new(Vec3::new(0.0, 0.0, -3.0), Quat::IDENTITY),
        )
        .unwrap();
    let colors = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.2, 0.4, 0.8)];
    let (model, state) = upload(&engine, builder, &colors);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    assert_eq!(scene.mesh_count(), 1);
    let mut second = model.state().unwrap();
    second
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut states = [state, second];
    let mut camera = camera(&engine, 19, 13, 2);
    camera
        .set_camera_look_at(0, Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    camera
        .set_camera_look_at(
            1,
            Vec3::new(4.5, 0.3, 6.0),
            Vec3::new(4.5, 0.0, -1.0),
            Vec3::Y,
        )
        .unwrap();
    let mut ring = ring(&engine, 19 * 13 * 2);
    let mut pending = Vec::new();
    for sample in 0..6 {
        let state = &mut states[sample % 2];
        let pose = Transform::new(
            Vec3::new(sample as Real * 0.07, 0.0, sample as Real * 0.15),
            Quat::from_rotation_y(sample as Real * 0.09),
        );
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&pose.to_gpu()),
        );
        state.body_q.mark_device_dirty();
        let mut objects = mesh_objects(&mesh, pose.compose(&local), 0);
        objects.extend(mesh_objects(&mesh, static_pose, 1));
        objects.push(Object::Sphere(Vec3::new(-0.7, 0.2, 1.0), 0.45, 2));
        objects.push(Object::Plane(-3.0, 3));
        camera.dispatch_scene(&engine, &scene, state).unwrap();
        assert!(state.body_q.host().is_err());
        assert!(camera.color.host().is_err());
        pending.push((enqueue(&camera, &mut ring, sample as u64), objects));
        if pending.len() == 2 {
            while let Some((ticket, objects)) = pending.pop() {
                let result = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
                check(&result, &camera, &objects, &colors);
            }
        }
    }
    assert_eq!(camera.mesh_build_count(), 0);
    assert_eq!(camera.query_stats().mesh_preparations, 0);
    assert_eq!(camera.query_stats().primitive_preparations, 0);
    assert_eq!(camera.query_stats().scene_builds, 1);
    assert_eq!(camera.query_stats().scene_preparations, 2);
    assert_eq!(camera.query_stats().refit_preparations, 2);
    assert_eq!(camera.query_stats().submissions, 0);
    assert_eq!(camera.dispatch_stats().submissions, 6);
    assert_eq!(camera.dispatch_stats().trace_preparations, 1);
    assert_eq!(
        camera.channel_scene_stamp(Ch::SHAPE_INDEX).unwrap(),
        Some(scene.stamp())
    );
    let owned_ticket = enqueue(&camera, &mut ring, 6);
    camera.dispatch(&engine, &model, &states[1]).unwrap();
    let raw_ticket = enqueue(&camera, &mut ring, 7);
    let raw = ring.wait(&raw_ticket, Duration::from_secs(10)).unwrap();
    let owned = ring.wait(&owned_ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(
        raw.channel::<Real>(0).unwrap(),
        owned.channel::<Real>(0).unwrap()
    );
    assert_eq!(
        raw.channel::<Vec3>(2).unwrap(),
        owned.channel::<Vec3>(2).unwrap()
    );
    for channel in [1, 3, 4] {
        assert_eq!(
            raw.channel::<u32>(channel).unwrap(),
            owned.channel::<u32>(channel).unwrap()
        );
    }
    assert_eq!(camera.mesh_build_count(), 1);
    assert_eq!(camera.channel_scene_stamp(Ch::SHAPE_INDEX).unwrap(), None);
}

#[test]
#[ignore = "needs a real GPU"]
fn camera_heightfield_and_triangle_visibility_match_cpu_triangles() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let field = Heightfield {
        nx: 3,
        nz: 3,
        cell_size: 1.0,
        cell_size_z: 1.0,
        origin: Vec3::new(-1.0, 0.0, -1.0),
        heights: vec![0.0, 0.2, 0.3, 0.1, 0.7, 0.4, 0.2, 0.3, 0.6],
    };
    let mut builder = ModelBuilder::new();
    let id = builder.add_heightfield(field.clone());
    builder
        .add_shape_heightfield(BodyId::NONE, id, Transform::IDENTITY)
        .unwrap();
    let tri = [
        Vec3::new(-0.7, 1.0, -0.7),
        Vec3::new(0.8, 1.0, -0.7),
        Vec3::new(0.0, 2.0, 0.8),
    ];
    builder
        .add_shape_triangle(
            BodyId::NONE,
            tri[0],
            tri[1],
            tri[2],
            Transform::new(Vec3::X * 4.0, Quat::IDENTITY),
        )
        .unwrap();
    let colors = [Vec3::new(0.2, 0.4, 0.8), Vec3::new(0.8, 0.4, 0.2)];
    let (model, state) = upload(&engine, builder, &colors);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut camera = camera(&engine, 19, 13, 2);
    for cam in 0..2 {
        camera
            .set_camera_pose(
                cam,
                Transform::new(
                    Vec3::new(cam as Real * 4.0 + 0.037, 5.0, 0.023),
                    Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
                ),
            )
            .unwrap();
    }
    let mut objects = vec![Object::Triangle(tri.map(|v| v + Vec3::X * 4.0), 1)];
    let vertex = |x: u32, z: u32| {
        field.origin + Vec3::new(x as Real, field.heights[(z * 3 + x) as usize], z as Real)
    };
    for z in 0..2 {
        for x in 0..2 {
            objects.push(Object::Triangle(
                [vertex(x, z), vertex(x, z + 1), vertex(x + 1, z)],
                0,
            ));
            objects.push(Object::Triangle(
                [vertex(x + 1, z), vertex(x, z + 1), vertex(x + 1, z + 1)],
                0,
            ));
        }
    }
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    let mut ring = ring(&engine, 19 * 13 * 2);
    let ticket = enqueue(&camera, &mut ring, 0);
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    check(&frame, &camera, &objects, &colors);
    let ids = frame.channel::<u32>(4).unwrap();
    assert!(ids[..19 * 13].contains(&0));
    assert!(ids[19 * 13..].contains(&1));
    assert!(ids.contains(&u32::MAX));
    assert_eq!(scene.mesh_count(), 2);
    assert_eq!(camera.mesh_build_count(), 0);
    assert_eq!(camera.query_stats().primitive_preparations, 0);
    assert_eq!(camera.query_stats().mesh_preparations, 0);
    assert_eq!(camera.query_stats().scene_preparations, 1);
    assert_eq!(camera.query_stats().refit_preparations, 1);

    let mut edge_camera = TiledCamera::new(
        &engine,
        TiledCameraConfig {
            width: 1,
            height: 1,
            ..*camera.config()
        },
        1,
    )
    .unwrap();
    edge_camera
        .set_camera_pose(
            0,
            Transform::new(
                Vec3::new(0.0, 5.0, -0.75),
                Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
            ),
        )
        .unwrap();
    edge_camera.dispatch_scene(&engine, &scene, &state).unwrap();
    let ticket = enqueue(&edge_camera, &mut ring, 1);
    let edge = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(edge.channel::<u32>(4).unwrap(), [0]);
    assert!((edge.channel::<Real>(0).unwrap()[0] - 4.675).abs() < 5e-4);
    let normal = edge.channel::<Vec3>(2).unwrap()[0];
    assert!(
        [
            Vec3::new(-0.6, 1.0, -0.5).normalize(),
            Vec3::new(-0.1, 1.0, -0.5).normalize()
        ]
        .iter()
        .any(|candidate| (normal - *candidate).length() < 5e-4)
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn camera_scene_revisions_partial_channels_range_and_empty_scene_are_explicit() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    let mesh = builder.add_mesh(quad());
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let (mut model, state) = upload(&engine, builder, &[Vec3::X]);
    let mut scene =
        SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut camera = camera(&engine, 1, 1, 1);
    camera
        .set_camera_pose(0, Transform::new(Vec3::Z * 5.0, Quat::IDENTITY))
        .unwrap();
    let mut ring = ring(&engine, 1);
    for (distance, expected) in [(4.999, -1.0), (5.0, -1.0), (5.0001, 5.0)] {
        let mut cfg = *camera.config();
        cfg.max_distance = distance;
        cfg.near = 10.0;
        camera.set_config(cfg).unwrap();
        camera.dispatch_scene(&engine, &scene, &state).unwrap();
        let ticket = enqueue(&camera, &mut ring, 0);
        let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        assert_eq!(frame.channel::<Real>(0).unwrap(), [expected]);
        assert_eq!(
            frame.channel::<u32>(4).unwrap(),
            [if expected < 0.0 { u32::MAX } else { 0 }]
        );
    }
    let old = scene.stamp();
    let old_ticket = enqueue(&camera, &mut ring, 1);
    for vertex in &mut model.meshes[0].vertices {
        vertex.z = 1.0;
    }
    model.shape_color.host_mut().unwrap()[0] = Vec3::Z;
    scene.replace_model(&engine, &model).unwrap();
    let new = scene.stamp();
    let mut cfg = *camera.config();
    cfg.channels = Ch::DEPTH;
    camera.set_config(cfg).unwrap();
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    assert_eq!(camera.channel_scene_stamp(Ch::DEPTH).unwrap(), Some(new));
    assert_eq!(
        camera.channel_scene_stamp(Ch::SHAPE_INDEX).unwrap(),
        Some(old)
    );
    assert_eq!(camera.channel_scene_stamp(Ch::COLOR).unwrap(), Some(old));
    let new_ticket = enqueue(&camera, &mut ring, 2);
    let current = ring.wait(&new_ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(current.channel::<Real>(0).unwrap(), [4.0]);
    assert_eq!(current.channel::<u32>(1).unwrap(), [packed(Vec3::X)]);
    let historical = ring.wait(&old_ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(historical.channel::<Real>(0).unwrap(), [5.0]);
    assert_eq!(historical.stamp.scene_revision, old.geometry_revision);
    assert!(scene.resolve_body(old, 0).is_err());
    assert_eq!(scene.resolve_body(new, 0).unwrap(), BodyId::NONE);
    cfg.channels = Ch::ALL;
    cfg.optics.lens_diameter_m = 0.6;
    cfg.optics.focal_distance_m = 2.0;
    cfg.optics.aperture_samples = 7;
    camera.set_config(cfg).unwrap();
    let before = camera.dispatch_stats();
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    assert_eq!(camera.dispatch_stats().submissions, before.submissions + 1);
    let ticket = enqueue(&camera, &mut ring, 3);
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.channel::<u32>(1).unwrap(), [packed(Vec3::Z)]);
    assert_eq!(frame.channel::<Real>(0).unwrap(), [4.0]);
    for channel in [
        Ch::DEPTH,
        Ch::COLOR,
        Ch::NORMAL,
        Ch::ALBEDO,
        Ch::SHAPE_INDEX,
    ] {
        assert_eq!(camera.channel_scene_stamp(channel).unwrap(), Some(new));
    }
    model.shape_color.host_mut().unwrap()[0] = Vec3::splat(f64::NAN);
    assert!(scene.replace_model(&engine, &model).is_err());
    assert_eq!(scene.stamp(), new);
    model.shape_color.host_mut().unwrap()[0] = Vec3::Z;
    model.meshes[0].indices[0][0] = 999;
    assert!(scene.replace_model(&engine, &model).is_err());
    assert_eq!(scene.stamp(), new);
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    let ticket = enqueue(&camera, &mut ring, 4);
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.channel::<u32>(1).unwrap(), [packed(Vec3::Z)]);
    let empty = Model::empty();
    scene.replace_model(&engine, &empty).unwrap();
    camera
        .dispatch_scene(&engine, &scene, &empty.state().unwrap())
        .unwrap();
    let ticket = enqueue(&camera, &mut ring, 5);
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.channel::<Real>(0).unwrap(), [-1.0]);
    assert_eq!(frame.channel::<u32>(1).unwrap(), [0]);
    assert_eq!(frame.channel::<Vec3>(2).unwrap(), [Vec3::ZERO]);
    assert_eq!(frame.channel::<u32>(4).unwrap(), [u32::MAX]);
    assert_eq!(
        camera.channel_scene_stamp(Ch::SHAPE_INDEX).unwrap(),
        Some(scene.stamp())
    );
    assert!(camera.channel_scene_stamp(Ch::ALL).is_err());
}
