#![cfg(feature = "gpu")]

#[path = "../benches/support/sensor_physics.rs"]
mod physics;
#[path = "../../molla-sensors/benches/support/timing.rs"]
#[allow(dead_code)]
mod timing;

use molla_compute::Engine;
use molla_math::{Real, Vec3};
use molla_sensors::{
    Raycaster, SensorCommandBatch, SensorFramePoolConfig, SensorScene,
    readback::{ReadbackConfig, ReadbackRing},
};
use std::time::Duration;

#[test]
#[ignore = "requires GPU"]
fn batched_host_rays_match_scalar_queries_after_gpu_step() {
    use molla_sensors::ScanMetadata;
    let engine = timing::engine(false);
    let mut physics = physics::Physics::new(&engine, 1, true);
    let scene = SensorScene::from_model(&engine, &physics.model, Default::default()).unwrap();
    let mut query = Raycaster::new(&engine, 2048).unwrap();
    let scan = ScanMetadata {
        forward: Vec3::NEG_Y,
        up: Vec3::Z,
        theta_range: (1.2, 1.9),
        phi_range: (-0.4, 0.4),
        max_distance: 30.0,
        ..ScanMetadata::new(Vec3::new(0.5, 12.0, 0.0), 32, 32)
    };
    physics.step(&engine, 1);
    let directions: Vec<_> = (0..1024)
        .map(|i| scan.rc_to_direction(i / 32, i % 32))
        .collect();
    let reference: Vec<_> = directions
        .iter()
        .map(|&direction| {
            let hit = query
                .cast_scene(
                    &engine,
                    &scene,
                    &physics.state,
                    &[scan.origin],
                    &[direction],
                    30.0,
                )
                .unwrap()[0];
            assert!(hit.t.is_finite() && hit.t > 0.0);
            hit
        })
        .collect();
    let sphere_ray = reference
        .iter()
        .position(|hit| hit.shape_id.is_some_and(|shape| shape > 0))
        .unwrap();
    assert_eq!(reference[0].shape_id, Some(0));
    let mut check = |indices: Vec<usize>| {
        let rays: Vec<_> = indices.iter().map(|&i| directions[i]).collect();
        let hits = query
            .cast_scene(
                &engine,
                &scene,
                &physics.state,
                &vec![scan.origin; indices.len()],
                &rays,
                30.0,
            )
            .unwrap();
        for (hit, &index) in hits.iter().zip(&indices) {
            let expected = reference[index];
            assert_eq!(hit.shape_id, expected.shape_id, "ray {index}");
            assert!((hit.t - expected.t).abs() < 1e-5, "ray {index}");
            assert!((hit.point - expected.point).length() < 2e-5, "ray {index}");
            assert!(
                (hit.normal - expected.normal).length() < 2e-5,
                "ray {index}"
            );
        }
    };
    check(vec![0, sphere_ray]);
    check(vec![0; 1024]);
    check(vec![sphere_ray; 1024]);
    check(
        (0..1024)
            .map(|i| if i % 2 == 0 { 0 } else { sphere_ray })
            .collect(),
    );
    for count in [
        1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024,
    ] {
        check((0..count).collect());
    }
    assert!(physics.state.body_q.host().is_err());
}

#[test]
#[ignore = "requires GPU"]
fn articulation_unseeded_hierarchy_has_valid_live_bounds() {
    use molla_math::Transform;
    use molla_sensors::SceneAcceleration;
    let engine = timing::engine(false);
    let mut cpu = physics::Physics::new(&engine, 1, false);
    let mut gpu = physics::Physics::new(&engine, 1, true);
    cpu.step(&engine, 1);
    gpu.step(&engine, 1);
    let scene = SensorScene::from_model(&engine, &gpu.model, Default::default()).unwrap();
    let mut acceleration =
        SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
    let view = acceleration.dispatch(&engine, &scene, &gpu.state).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 4096,
            max_total_bytes: 4096,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(
            &[
                view.lower.readback_source(0..view.node_count).unwrap(),
                view.upper.readback_source(0..view.node_count).unwrap(),
                view.world_poses
                    .readback_source(0..view.leaf_count)
                    .unwrap(),
                view.status.readback_source(0..view.leaf_count).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let result = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    let lower = result.channel::<Vec3>(0).unwrap();
    let upper = result.channel::<Vec3>(1).unwrap();
    let poses = result.channel::<Transform>(2).unwrap();
    assert!(result.channel::<u32>(3).unwrap().iter().all(|&x| x == 0));
    for (i, node) in view.topology.iter().enumerate() {
        assert!(lower[i].is_finite() && upper[i].is_finite());
        assert!(lower[i].cmplt(upper[i]).all());
        if node.height == 0 {
            let center = cpu.state.body_q.host().unwrap()[node.shape as usize - 1].position;
            assert!((poses[i].position - center).length() < 2e-4);
            assert!(lower[i].cmple(center - Vec3::splat(physics::RADIUS)).all());
            assert!(upper[i].cmpge(center + Vec3::splat(physics::RADIUS)).all());
        } else {
            for child in [node.left as usize, node.right as usize] {
                assert!(child < i);
                assert!(lower[i].cmple(lower[child]).all());
                assert!(upper[i].cmpge(upper[child]).all());
            }
        }
    }
    assert!(gpu.state.body_q.host().is_err());
}

#[test]
#[ignore = "requires GPU"]
fn external_device_gpu_physics_lidar_without_an_intermediate_wait() {
    use molla_sensors::{LidarBatch, LidarPattern, LidarScan, ScanMetadata};
    let engine = timing::engine(false);
    let mut reference = physics::Physics::new(&engine, 1, false);
    let mut physics = physics::Physics::new(&engine, 1, true);
    let scene = SensorScene::from_model(&engine, &physics.model, Default::default()).unwrap();
    let mut lidar = LidarBatch::new(&engine, 1, 1024).unwrap();
    let mut pool = lidar
        .output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 1 << 20,
            },
        )
        .unwrap();
    let scans = [LidarScan {
        sensor_id: 1,
        pattern: LidarPattern::Terrestrial(ScanMetadata {
            forward: Vec3::NEG_Y,
            up: Vec3::Z,
            theta_range: (1.2, 1.9),
            phi_range: (-0.4, 0.4),
            max_distance: 30.0,
            ..ScanMetadata::new(Vec3::new(0.5, 12.0, 0.0), 32, 32)
        }),
    }];
    let LidarPattern::Terrestrial(scan) = &scans[0].pattern else {
        unreachable!()
    };
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 8192,
            max_total_bytes: 8192,
        },
    )
    .unwrap();
    engine.queue().unwrap().submit([]);
    timing::wait(&engine);
    for _ in 0..4 {
        reference.step(&engine, 1);
        physics.step(&engine, 1);
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let output = lidar
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &physics.state,
                &scans,
                &mut pool,
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        timing::wait(&engine);
        let frame = submitted.frame(&output.frame).unwrap();
        let depth = frame.channel::<Real>(2).unwrap();
        let shapes = frame.channel::<i32>(3).unwrap();
        let ticket = ring
            .try_enqueue(
                &[
                    depth.readback_source(0..1024).unwrap(),
                    shapes.readback_source(0..1024).unwrap(),
                ],
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let results = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
        let mut sphere_hits = 0;
        let mut ground_hits = 0;
        for (i, (distance, shape)) in results
            .channel_iter::<Real>(0)
            .unwrap()
            .zip(results.channel_iter::<i32>(1).unwrap())
            .enumerate()
        {
            let direction = scan
                .rc_to_direction(i as u32 / 32, i as u32 % 32)
                .normalize();
            let mut expected = scan.origin.y / -direction.y;
            let mut expected_shape = 0;
            for (body, pose) in reference.state.body_q.host().unwrap().iter().enumerate() {
                let delta = scan.origin - pose.position;
                let b = delta.dot(direction);
                let discriminant = b * b - delta.length_squared() + physics::RADIUS.powi(2);
                if discriminant >= 0.0 {
                    let near = -b - discriminant.sqrt();
                    if near > 0.0 && near < expected {
                        expected = near;
                        expected_shape = body as i32 + 1;
                    }
                }
            }
            assert_eq!(shape, expected_shape, "ray {i}");
            assert!(
                (distance - expected).abs() < 1e-3,
                "ray {i}: {distance} != {expected}"
            );
            if shape == 0 {
                ground_hits += 1;
            } else {
                sphere_hits += 1;
            }
        }
        assert!(sphere_hits > 0 && ground_hits > 0);
        assert!(physics.state.body_q.host().is_err());
    }
    assert!(physics.state.body_q.host().is_err());
}

#[test]
#[ignore = "requires GPU"]
fn cpu_and_gpu_articulations_feed_current_poses_to_sensors_without_intermediate_readback() {
    let engine = Engine::new_gpu().unwrap();
    for arts in [1, 4] {
        let mut cpu = physics::Physics::new(&engine, arts, false);
        let mut gpu = physics::Physics::new(&engine, arts, true);
        let cpu_scene = SensorScene::from_model(&engine, &cpu.model, Default::default()).unwrap();
        let gpu_scene = SensorScene::from_model(&engine, &gpu.model, Default::default()).unwrap();
        let mut query = Raycaster::new(&engine, 1).unwrap();
        let mut pool = query
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 1,
                    max_total_bytes: 4096,
                },
            )
            .unwrap();
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 128,
                max_total_bytes: 128,
            },
        )
        .unwrap();
        let initial = cpu.state.body_q.host().unwrap()[0].position;
        for step in 0..24 {
            let n = if step % 2 == 0 { 1 } else { 10 };
            cpu.step(&engine, n);
            gpu.step(&engine, n);
            let center = cpu.state.body_q.host().unwrap()[0].position;
            for (state, scene) in [(&cpu.state, &cpu_scene), (&gpu.state, &gpu_scene)] {
                let mut batch = SensorCommandBatch::new(&engine).unwrap();
                let query = query
                    .encode_scene(
                        &mut batch,
                        &engine,
                        scene,
                        state,
                        &[center + Vec3::X * 3.0],
                        &[Vec3::NEG_X],
                        10.0,
                        &mut pool,
                        Default::default(),
                    )
                    .unwrap()
                    .unwrap();
                let submitted = batch.submit().unwrap();
                let frame = submitted.frame(&query.frame).unwrap();
                let distance = frame.channel::<Real>(2).unwrap();
                let shape = frame.channel::<i32>(3).unwrap();
                let point = frame.channel::<Vec3>(4).unwrap();
                let ticket = ring
                    .try_enqueue(
                        &[
                            distance.readback_source(0..1).unwrap(),
                            shape.readback_source(0..1).unwrap(),
                            point.readback_source(0..1).unwrap(),
                        ],
                        Default::default(),
                    )
                    .unwrap()
                    .unwrap();
                let result = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
                assert_eq!(result.channel::<i32>(1).unwrap(), [1]);
                assert!(
                    (result.channel::<Real>(0).unwrap()[0] - (3.0 - physics::RADIUS)).abs() < 2e-4
                );
                assert!(
                    (result.channel::<Vec3>(2).unwrap()[0] - (center + Vec3::X * physics::RADIUS))
                        .length()
                        < 2e-4
                );
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
            assert!(gpu.state.body_q.host().is_err());
        }
        assert!((cpu.state.body_q.host().unwrap()[0].position - initial).length() > 0.005);
    }
}
