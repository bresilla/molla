#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    SceneAcceleration, SceneAccelerationStorage, SceneAccelerationView, SensorScene,
};
use molla_sim::{ModelBuilder, State, TriMesh};
use std::time::Duration;

fn collect(
    engine: &Engine,
    view: SceneAccelerationView<'_>,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<Transform>, Vec<u32>) {
    let bytes = (view.node_count * 32 + view.leaf_count * 36 + 8) as u64;
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: (bytes + 7) / 8 * 8,
        },
    )
    .unwrap();
    let sources = [
        view.lower.readback_source(0..view.node_count).unwrap(),
        view.upper.readback_source(0..view.node_count).unwrap(),
        view.world_poses
            .readback_source(0..view.leaf_count)
            .unwrap(),
        view.status.readback_source(0..view.leaf_count).unwrap(),
    ];
    let ticket = ring
        .try_enqueue(&sources, ReadbackStamp::default())
        .unwrap()
        .unwrap();
    let frame = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    (
        frame.channel::<Vec3>(0).unwrap().to_vec(),
        frame.channel::<Vec3>(1).unwrap().to_vec(),
        frame.channel::<Transform>(2).unwrap().to_vec(),
        frame.channel::<u32>(3).unwrap().to_vec(),
    )
}

fn write_poses(engine: &Engine, state: &mut State, poses: &[Transform]) {
    let packed: Vec<GpuTransform> = poses.iter().copied().map(Into::into).collect();
    engine.queue().unwrap().write_buffer(
        state.body_q.device_buffer().unwrap(),
        0,
        bytemuck::cast_slice(&packed),
    );
    state.body_q.mark_device_dirty();
}

#[test]
#[ignore = "requires GPU"]
fn mixed_instances_refit_from_device_poses_and_enclose_independent_corner_bounds() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder.add_body(Default::default());
    builder.add_body(Default::default());
    let lo = Vec3::new(-0.25, -0.7, -0.4);
    let hi = Vec3::new(0.9, 0.3, 0.6);
    let mesh = builder.add_mesh(TriMesh {
        vertices: vec![lo, hi, Vec3::new(hi.x, lo.y, lo.z)],
        indices: vec![[0, 1, 2]],
    });
    let mut reference = Vec::new();
    for i in 0..129 {
        let body = if i % 3 == 0 {
            BodyId::NONE
        } else {
            BodyId(i % 2)
        };
        let local = Transform::new(
            Vec3::new((i % 13) as f64 * 3.0 - 18.0, 0.2, (i / 13) as f64 * -2.0),
            Quat::from_rotation_y(i as f64 * 0.03),
        );
        let (min, max) = match i % 3 {
            0 => {
                builder
                    .add_shape_box(body, Vec3::new(1.0, 2.0, 0.5), local)
                    .unwrap();
                (-Vec3::new(1.0, 2.0, 0.5), Vec3::new(1.0, 2.0, 0.5))
            }
            1 => {
                builder.add_shape_capsule(body, 0.5, 0.3, local).unwrap();
                (-Vec3::new(0.3, 0.8, 0.3), Vec3::new(0.3, 0.8, 0.3))
            }
            _ => {
                builder.add_shape_trimesh(body, mesh, local).unwrap();
                (lo, hi)
            }
        };
        reference.push((body, local, min, max));
    }
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut states = [model.state().unwrap(), model.state().unwrap()];
    for state in &mut states {
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
    }
    let seeds = [
        Transform::from_translation(Vec3::X * 20.0),
        Transform::from_translation(Vec3::X * -20.0),
    ];
    let mut acceleration =
        SceneAcceleration::new(&engine, &scene, Some(&seeds), Default::default()).unwrap();
    assert_eq!(acceleration.planes(), [129]);
    assert_eq!(acceleration.storage().leaves, 129);
    assert_eq!(acceleration.storage().nodes, 257);
    let topology: Vec<_> = acceleration
        .topology()
        .iter()
        .map(|n| (n.left, n.right, n.shape, n.height))
        .collect();
    for frame in 0..8 {
        let poses = [
            Transform::new(
                Vec3::new(frame as f64 * 2.0, 3.0, 1.0),
                Quat::from_rotation_z(frame as f64 * 0.17),
            ),
            Transform::new(
                Vec3::new(-10000.0 + frame as f64, -2.0, -6.0),
                Quat::from_rotation_x(frame as f64 * 0.12),
            ),
        ];
        let state = &mut states[frame % 2];
        write_poses(&engine, state, &poses);
        let view = acceleration.dispatch(&engine, &scene, state).unwrap();
        assert_eq!(view.root, Some(256));
        let (lower, upper, world, status) = collect(&engine, view);
        assert!(state.body_q.host().is_err());
        assert!(status.iter().all(|&s| s == 0));
        for (i, &(body, local, lo, hi)) in reference.iter().enumerate() {
            let pose = if body == BodyId::NONE {
                local
            } else {
                poses[body.index()].compose(&local)
            };
            let mut expected_lo = Vec3::splat(f64::INFINITY);
            let mut expected_hi = Vec3::splat(f64::NEG_INFINITY);
            for corner in 0..8 {
                let p = pose.transform_point(Vec3::new(
                    if corner & 1 == 0 { lo.x } else { hi.x },
                    if corner & 2 == 0 { lo.y } else { hi.y },
                    if corner & 4 == 0 { lo.z } else { hi.z },
                ));
                expected_lo = expected_lo.min(p);
                expected_hi = expected_hi.max(p);
                assert!(
                    lower[i].cmple(p).all() && upper[i].cmpge(p).all(),
                    "leaf {i}, frame {frame}: {:?} <= {p:?} <= {:?}",
                    lower[i],
                    upper[i]
                );
            }
            let tolerance = (pose.position.abs().max_element() + 100.0) * 1e-5;
            assert!((lower[i] - expected_lo).abs().max_element() < tolerance);
            assert!((upper[i] - expected_hi).abs().max_element() < tolerance);
            assert!((world[i].position - pose.position).length() < tolerance);
            assert!((world[i].rotation * Vec3::X - pose.rotation * Vec3::X).length() < 2e-6);
        }
        for (i, &(left, right, _, height)) in topology.iter().enumerate() {
            if height != 0 {
                assert_eq!(lower[i], lower[left as usize].min(lower[right as usize]));
                assert_eq!(upper[i], upper[left as usize].max(upper[right as usize]));
            }
        }
    }
    assert_eq!(
        acceleration
            .topology()
            .iter()
            .map(|n| (n.left, n.right, n.shape, n.height))
            .collect::<Vec<_>>(),
        topology
    );
    let stats = acceleration.stats();
    assert_eq!(stats.encoded_refits, 8);
    assert_eq!(stats.submissions, 8);
    assert_eq!(stats.binding_preparations, 2);
    assert_eq!(
        stats.kernel_dispatches,
        8 * acceleration.storage().levels as u64
    );
}

#[test]
#[ignore = "requires GPU"]
fn invalid_live_poses_are_flagged_conservatively_and_valid_refits_clear_status() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder
        .add_shape_sphere(body, 1.0, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut acceleration =
        SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
    for (pose, code) in [
        (
            Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
            1,
        ),
        (Transform::from_translation(Vec3::splat(f64::NAN)), 1),
        (Transform::from_translation(Vec3::splat(f32::MAX as f64)), 2),
        (Transform::from_translation(Vec3::X * -2.0), 0),
    ] {
        write_poses(&engine, &mut state, &[pose]);
        let (lo, hi, _, status) = collect(
            &engine,
            acceleration.dispatch(&engine, &scene, &state).unwrap(),
        );
        assert_eq!(status, [code]);
        if code != 0 {
            assert_eq!(lo, [Vec3::splat(-(f32::MAX as f64))]);
            assert_eq!(hi, [Vec3::splat(f32::MAX as f64)]);
        } else {
            assert!(lo[0].x <= -3.0 && hi[0].x >= -1.0);
        }
    }
    let before = acceleration.stats();
    let unuploaded = model.state().unwrap();
    assert!(acceleration.dispatch(&engine, &scene, &unuploaded).is_err());
    assert_eq!(acceleration.stats(), before);
    let other = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    assert!(acceleration.dispatch(&engine, &other, &state).is_err());
    scene.replace_model(&engine, &model).unwrap();
    assert!(acceleration.dispatch(&engine, &scene, &state).is_err());
    assert_eq!(acceleration.stats(), before);
    assert!(SceneAcceleration::new(&engine, &scene, Some(&[]), Default::default()).is_err());
    assert!(
        SceneAcceleration::new(
            &engine,
            &scene,
            Some(&[Transform::from_translation(Vec3::splat(f64::NAN))]),
            Default::default()
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires GPU"]
fn empty_and_plane_only_scenes_have_no_finite_dispatches_and_budgets_are_enforced() {
    let engine = Engine::new_gpu().unwrap();
    for plane in [false, true] {
        let mut builder = ModelBuilder::new();
        if plane {
            builder
                .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
                .unwrap();
        }
        let model = builder.build().unwrap();
        let state = model.state().unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let usage = SceneAccelerationStorage::for_leaves(0).unwrap();
        assert!(
            SceneAcceleration::new(
                &engine,
                &scene,
                None,
                molla_sensors::SceneAccelerationConfig {
                    max_device_bytes: usage.device_bytes - 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        let mut acceleration = SceneAcceleration::new(
            &engine,
            &scene,
            None,
            molla_sensors::SceneAccelerationConfig {
                max_device_bytes: usage.device_bytes,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(acceleration.planes().len(), usize::from(plane));
        let view = acceleration.dispatch(&engine, &scene, &state).unwrap();
        assert_eq!(view.root, None);
        let (lo, hi, poses, flags) = collect(&engine, view);
        assert!(lo.is_empty() && hi.is_empty() && poses.is_empty() && flags.is_empty());
        assert_eq!(acceleration.stats().kernel_dispatches, 0);
        assert_eq!(acceleration.stats().binding_preparations, 0);
    }
}
