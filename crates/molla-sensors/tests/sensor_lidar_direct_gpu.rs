#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{
    AerialScanMetadata, CloudPoint, CloudProcessor, CloudProcessorConfig, EncodedLidarBatch,
    LidarBatch, LidarPattern, LidarScan, PendingCloudInput, PendingCloudJob, ScanMetadata,
    SensorCommandBatch, SensorFrame, SensorFramePoolConfig, SensorScene, SensorSceneOptions,
    readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource, ReadbackStamp},
};
use molla_sim::{BodyParams, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn pool_config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 128 << 20,
    }
}

fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX - 1,
        scene_revision: 987,
        simulation_time: Some(1.5),
    }
}

fn scene(engine: &Engine) -> (Model, State, SensorScene) {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = b.add_body(BodyParams::new().with_position(Vec3::Y * 2.0));
    let mesh = b.add_mesh(TriMesh {
        vertices: vec![
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(-2.0, 0.0, 2.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    });
    b.add_shape_trimesh(body, mesh, Transform::IDENTITY)
        .unwrap();
    let mut model = b.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(engine, &model, SensorSceneOptions::default()).unwrap();
    (model, state, scene)
}

fn scan(id: u64, origin: Vec3, pulses: u32, subrays: u32, max: Real) -> LidarScan {
    let mut s = ScanMetadata::new(origin, 1, pulses);
    s.forward = Vec3::NEG_Y;
    s.up = Vec3::Z;
    s.theta_range = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    s.phi_range = (0.0, 0.0);
    s.subray_count = subrays;
    s.max_distance = max;
    LidarScan {
        sensor_id: id,
        pattern: LidarPattern::Terrestrial(s),
    }
}

fn read(engine: &Engine, sources: &[ReadbackSource<'_>]) -> ReadbackFrame {
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1 << 20,
            max_total_bytes: 1 << 20,
        },
    )
    .unwrap();
    let ticket = ring.try_enqueue(sources, stamp(0)).unwrap().unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn collect(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let o = frame.channel::<Vec3>(0).unwrap();
    let d = frame.channel::<Vec3>(1).unwrap();
    let t = frame.channel::<Real>(2).unwrap();
    let s = frame.channel::<i32>(3).unwrap();
    let p = frame.channel::<Vec3>(4).unwrap();
    let n = frame.channel::<Vec3>(5).unwrap();
    let first = frame.channel::<i32>(6).unwrap();
    let last = frame.channel::<i32>(7).unwrap();
    let count = frame.channel::<u32>(8).unwrap();
    read(
        engine,
        &[
            o.readback_source(0..o.len()).unwrap(),
            d.readback_source(0..d.len()).unwrap(),
            t.readback_source(0..t.len()).unwrap(),
            s.readback_source(0..s.len()).unwrap(),
            p.readback_source(0..p.len()).unwrap(),
            n.readback_source(0..n.len()).unwrap(),
            first.readback_source(0..first.len()).unwrap(),
            last.readback_source(0..last.len()).unwrap(),
            count.readback_source(0..count.len()).unwrap(),
        ],
    )
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

fn verify(cpu: &ReadbackFrame, out: &EncodedLidarBatch, mesh_y: Real) {
    let origins = cpu.channel::<Vec3>(0).unwrap();
    let directions = cpu.channel::<Vec3>(1).unwrap();
    let distances = cpu.channel::<Real>(2).unwrap();
    let shapes = cpu.channel::<i32>(3).unwrap();
    let points = cpu.channel::<Vec3>(4).unwrap();
    let normals = cpu.channel::<Vec3>(5).unwrap();
    let first = cpu.channel::<i32>(6).unwrap();
    let last = cpu.channel::<i32>(7).unwrap();
    let counts = cpu.channel::<u32>(8).unwrap();
    for range in &out.scans {
        let max = match range.scan.pattern {
            LidarPattern::Terrestrial(s) => s.max_distance,
            LidarPattern::Aerial(s) => s.max_distance,
        };
        for j in 0..range.ray_count as usize {
            let i = range.ray_offset as usize + j;
            assert!((directions[i].length() - 1.0).abs() < 2e-6);
            let mesh_t = (mesh_y - origins[i].y) / directions[i].y;
            let mesh_p = origins[i] + mesh_t * directions[i];
            let mesh_hit = mesh_t > 0.0 && mesh_p.x.abs() < 2.0 && mesh_p.z.abs() < 2.0;
            let expected = if mesh_hit {
                mesh_t
            } else {
                -origins[i].y / directions[i].y
            };
            if expected > 0.0 && expected < max {
                assert!((distances[i] - expected).abs() < 2e-5, "ray {i}");
                assert_eq!(shapes[i], if mesh_hit { 1 } else { 0 });
                assert!((points[i] - (origins[i] + expected * directions[i])).length() < 2e-5);
                assert!((normals[i] - Vec3::Y).length() < 2e-5);
            } else {
                assert_eq!(distances[i], -1.0);
                assert_eq!(shapes[i], -1);
                assert_eq!(points[i], Vec3::ZERO);
                assert_eq!(normals[i], Vec3::Y);
            }
            if let LidarPattern::Terrestrial(s) = range.scan.pattern {
                assert!((origins[i] - s.origin).length() < 1e-6);
                if s.beam_divergence == 0.0 {
                    let pulse = j as u32 / range.subray_count;
                    assert!(
                        (directions[i] - s.rc_to_direction(pulse / s.n_phi, pulse % s.n_phi))
                            .length()
                            < 2e-6
                    );
                }
            }
        }
        for j in 0..range.pulse_count as usize {
            let pulse = range.pulse_offset as usize + j;
            let start = range.ray_offset as usize + j * range.subray_count as usize;
            let mut nearest = -1;
            let mut farthest = -1;
            let mut min = max;
            let mut far = 0.0;
            let mut count = 0;
            for i in start..start + range.subray_count as usize {
                if distances[i] > 0.0 {
                    count += 1;
                    if distances[i] < min {
                        min = distances[i];
                        nearest = i as i32;
                    }
                    if distances[i] >= far {
                        far = distances[i];
                        farthest = i as i32;
                    }
                }
            }
            assert_eq!(first[pulse], nearest);
            assert_eq!(last[pulse], farthest);
            assert_eq!(counts[pulse], count);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn independent_mixed_scans_feed_clouds_in_one_submission_and_match_legacy() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = scene(&engine);
    let mut aerial =
        AerialScanMetadata::new(Vec3::new(8.0, 6.0, -1.0), Vec3::new(10.0, 6.0, 1.0), 2.0);
    aerial.subray_count = 3;
    aerial.cone_angle = 0.1;
    aerial.beam_divergence = 0.02;
    let scans = [
        scan(u64::MAX, Vec3::Y * 6.0, 65, 2, 20.0),
        LidarScan {
            sensor_id: 0x1234567800000001,
            pattern: LidarPattern::Aerial(aerial),
        },
        scan(0xabcdef0000000001, Vec3::Y * 6.0, 3, 1, 3.0),
    ];
    let second_scans = [scan(19, Vec3::new(9.0, 6.0, 0.0), 3, 4, 20.0)];
    let mut a = LidarBatch::new(&engine, 3, 256).unwrap();
    let mut b = LidarBatch::new(&engine, 1, 32).unwrap();
    let mut cloud = CloudProcessor::new(
        &engine,
        CloudProcessorConfig {
            max_points: 256,
            max_jobs: 3,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut pa = a.output_pool(&engine, pool_config(1)).unwrap();
    let mut pb = b.output_pool(&engine, pool_config(1)).unwrap();
    let mut pc = cloud.output_pool(&engine, pool_config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let ra = a
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &scans,
            &mut pa,
            stamp(1),
        )
        .unwrap()
        .unwrap();
    let rb = b
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &second_scans,
            &mut pb,
            stamp(2),
        )
        .unwrap()
        .unwrap();
    assert!(PendingCloudInput::from_lidar(&ra, 3).is_err());
    let jobs: Vec<_> = scans
        .iter()
        .enumerate()
        .map(|(i, scan)| PendingCloudJob {
            input: PendingCloudInput::from_lidar(&ra, i).unwrap(),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: stamp(scan.sensor_id),
        })
        .collect();
    let rc = cloud
        .encode_pending(&mut batch, &jobs, &mut pc, stamp(3))
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().input_upload_bytes, 4 * 128);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 13);
    assert_eq!(submitted.stats.output_clears, 20);
    assert_eq!(ra.scene_stamp, scene.stamp());
    let fa = submitted.frame(&ra.frame).unwrap().clone();
    let fb = submitted.frame(&rb.frame).unwrap().clone();
    let fc = submitted.frame(&rc.frame).unwrap().clone();
    let cpu_a = collect(&engine, &fa);
    verify(&cpu_a, &ra, 2.0);
    verify(&collect(&engine, &fb), &rb, 2.0);
    assert_eq!(a.stats().submissions, 0);
    assert_eq!(a.readback_stats().copied_bytes, 0);
    assert_eq!(a.readback_stats().blocking_waits, 0);
    let mut legacy = LidarBatch::new(&engine, 3, 256).unwrap();
    let old = legacy
        .dispatch_scene(&engine, &scene, &state, &scans)
        .unwrap();
    let n = ra.ray_count;
    let p = ra.pulse_count;
    let cpu_old = read(
        &engine,
        &[
            old.hits.origins.readback_source(0..n).unwrap(),
            old.hits.directions.readback_source(0..n).unwrap(),
            old.hits.distance.readback_source(0..n).unwrap(),
            old.hits.shape.readback_source(0..n).unwrap(),
            old.hits.point.readback_source(0..n).unwrap(),
            old.hits.normal.readback_source(0..n).unwrap(),
            old.first_ray.readback_source(0..p).unwrap(),
            old.last_ray.readback_source(0..p).unwrap(),
            old.hit_count.readback_source(0..p).unwrap(),
        ],
    );
    for channel in [0, 1, 4, 5] {
        assert_eq!(
            cpu_a.channel::<Vec3>(channel).unwrap(),
            cpu_old.channel::<Vec3>(channel).unwrap()
        );
    }
    assert_eq!(
        cpu_a.channel::<Real>(2).unwrap(),
        cpu_old.channel::<Real>(2).unwrap()
    );
    for channel in [3, 6, 7] {
        assert_eq!(
            cpu_a.channel::<i32>(channel).unwrap(),
            cpu_old.channel::<i32>(channel).unwrap()
        );
    }
    assert_eq!(
        cpu_a.channel::<u32>(8).unwrap(),
        cpu_old.channel::<u32>(8).unwrap()
    );
    drop((submitted, a, b, cloud, pa, pb, pc));
    let points = fc.channel::<CloudPoint>(0).unwrap();
    let counts = fc.channel::<u32>(1).unwrap();
    let cpu = read(
        &engine,
        &[
            points.readback_source(0..points.len()).unwrap(),
            counts.readback_source(0..counts.len()).unwrap(),
        ],
    );
    let points = cpu.channel::<CloudPoint>(0).unwrap();
    let counts = cpu.channel::<u32>(1).unwrap();
    for (i, range) in ra.scans.iter().enumerate() {
        let start = counts[1 + i * 5] as usize;
        let count = counts[2 + i * 5] as usize;
        assert_eq!(count, if i == 2 { 0 } else { range.ray_count as usize });
        for (j, point) in points[start..start + count].iter().enumerate() {
            assert_eq!(point.sensor_id(), range.scan.sensor_id);
            assert_eq!(point.sample_id(), u64::MAX - 1);
            assert_eq!(point.grid[3], range.ray_offset + j as u32);
        }
    }
    assert!(
        points[counts[0] as usize..]
            .iter()
            .all(|p| *p == CloudPoint::default())
    );
}

#[test]
#[ignore = "requires GPU"]
fn warm_output_slots_follow_current_body_poses_without_rebuilding_bindings() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, scene) = scene(&engine);
    let mut other = model.state().unwrap();
    other
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut lidar = LidarBatch::new(&engine, 1, 257).unwrap();
    let mut pool = lidar.output_pool(&engine, pool_config(4)).unwrap();
    for round in 0..6 {
        let mut frames = Vec::new();
        for n in [1, 65, 257, 3] {
            let state = if round % 2 == 0 {
                &mut state
            } else {
                &mut other
            };
            state.body_q.host_mut().unwrap()[0].position.y = 1.0 + round as Real * 0.25;
            state
                .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
                .unwrap();
            let scans = [scan(
                round as u64,
                Vec3::Y * 6.0,
                n,
                1,
                if round == 5 { 2.0 } else { 20.0 },
            )];
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let out = lidar
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    state,
                    &scans,
                    &mut pool,
                    stamp(round as u64),
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            frames.push((submitted.frame(&out.frame).unwrap().clone(), out));
        }
        pool.reset().unwrap();
        for (frame, out) in &frames {
            assert!(!pool.is_current(frame));
            verify(&collect(&engine, frame), out, 1.0 + round as Real * 0.25);
        }
        assert_eq!(pool.available_slots().unwrap(), 0);
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            lidar
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &[],
                    &mut pool,
                    stamp(0)
                )
                .unwrap()
                .is_none()
        );
        assert!(batch.submit().unwrap().index.is_none());
        drop(frames);
        drain(&engine);
    }
    assert_eq!(lidar.stats().binding_preparations, 5);
    assert_eq!(lidar.query_stats().primitive_preparations, 8);
    assert_eq!(lidar.query_stats().mesh_preparations, 8);
    assert_eq!(pool.stats().buffer_allocations, 36);
    assert_eq!(lidar.stats().encoded_batches, 24);
    assert_eq!(lidar.stats().submissions, 0);
}

#[test]
#[ignore = "requires GPU"]
fn full_scan_and_subray_limits_preserve_independent_ranges_and_tie_order() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = scene(&engine);
    let mut lidar = LidarBatch::new(&engine, 1024, 1024).unwrap();
    let mut pool = lidar.output_pool(&engine, pool_config(2)).unwrap();
    let scans: Vec<_> = (0..1024)
        .map(|i| {
            scan(
                (i as u64) << 32 | 1,
                Vec3::new(9.0, 6.0, 0.0),
                1,
                1,
                if i % 2 == 0 { 5.0 } else { 20.0 },
            )
        })
        .collect();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let many = lidar
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &scans,
            &mut pool,
            stamp(1),
        )
        .unwrap()
        .unwrap();
    let first = batch.submit().unwrap();
    verify(
        &collect(&engine, first.frame(&many.frame).unwrap()),
        &many,
        2.0,
    );
    for (i, range) in many.scans.iter().enumerate() {
        assert_eq!(range.scan.sensor_id, (i as u64) << 32 | 1);
        assert_eq!(range.ray_offset, i as u32);
        assert_eq!(range.pulse_offset, i as u32);
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = lidar
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &[scan(u64::MAX, Vec3::new(9.0, 6.0, 0.0), 1, 1024, 20.0)],
            &mut pool,
            stamp(2),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let cpu = collect(&engine, submitted.frame(&out.frame).unwrap());
    verify(&cpu, &out, 2.0);
    assert_eq!(cpu.channel::<i32>(6).unwrap(), [0]);
    assert_eq!(cpu.channel::<i32>(7).unwrap(), [1023]);
    assert_eq!(cpu.channel::<u32>(8).unwrap(), [1024]);
}

#[test]
#[ignore = "requires GPU"]
fn empty_cancellation_preflight_and_known_loss_do_not_publish_stale_lidar() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (_foreign_model, foreign_state, _) = scene(&foreign);
    let (_model, state, scene) = scene(&engine);
    let mut lidar = LidarBatch::new(&engine, 1, 4).unwrap();
    let mut pool = lidar.output_pool(&engine, pool_config(1)).unwrap();
    assert!(lidar.output_pool(&foreign, pool_config(1)).is_err());
    assert!(
        lidar
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 1,
                    max_total_bytes: 4 * 84 - 1
                }
            )
            .is_err()
    );
    let scans = [scan(1, Vec3::Y * 6.0, 1, 1, 20.0)];
    for case in 0..9 {
        let mut scans = scans.to_vec();
        let mut frame_stamp = stamp(0);
        if let LidarPattern::Terrestrial(s) = &mut scans[0].pattern {
            match case {
                0 => s.n_phi = 5,
                1 => s.origin.x = Real::NAN,
                2 => s.max_distance = 1e-100,
                3 => s.subray_count = 1025,
                4 => s.n_theta = u32::MAX,
                5 => s.forward = Vec3::ZERO,
                6 => frame_stamp.simulation_time = Some(-1.0),
                _ => {}
            }
        }
        if case == 7 {
            scans.push(scans[0]);
        }
        let state = if case == 8 { &foreign_state } else { &state };
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            lidar
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    state,
                    &scans,
                    &mut pool,
                    frame_stamp
                )
                .is_err(),
            "case {case}"
        );
        assert!(batch.submit().unwrap().index.is_none());
        assert_eq!(pool.stats().reserved_frames, 0);
    }
    assert_eq!(lidar.stats().descriptor_upload_bytes, 0);
    assert_eq!(lidar.stats().binding_preparations, 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let canceled = lidar
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &scans,
            &mut pool,
            stamp(1),
        )
        .unwrap()
        .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = lidar
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &[],
            &mut pool,
            stamp(2),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert!(submitted.frame(&canceled.frame).is_err());
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    assert_eq!(submitted.stats.input_upload_bytes, 0);
    let empty = collect(&engine, submitted.frame(&out.frame).unwrap());
    assert_eq!(empty.byte_len(), 0);
    drop(submitted);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = lidar
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &scans,
            &mut pool,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    assert!(
        batch
            .encode_with_pending(&[&out.frame], |_, _| Err(molla_core::Error::Solver(
                "poison".into()
            )))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    engine.destroy_device().unwrap();
    assert!(
        lidar
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &state,
                &scans,
                &mut pool,
                stamp(4)
            )
            .is_err()
    );
    assert!(batch.submit().is_err());
}

#[test]
#[ignore = "requires GPU"]
fn scene_replacement_and_empty_geometry_keep_historical_outputs_readable() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, mut scene) = scene(&engine);
    let mut lidar = LidarBatch::new(&engine, 1, 1).unwrap();
    let mut pool = lidar.output_pool(&engine, pool_config(3)).unwrap();
    let scans = [scan(1, Vec3::Y * 6.0, 1, 1, 20.0)];
    let mut held = Vec::new();
    for version in 0..3 {
        if version != 0 {
            let mut builder = ModelBuilder::new();
            if version == 1 {
                builder
                    .add_shape_plane(
                        BodyId::NONE,
                        Vec3::Y,
                        Transform::new(Vec3::Y, molla_math::Quat::IDENTITY),
                    )
                    .unwrap();
            }
            let model = builder.build().unwrap();
            scene.replace_model(&engine, &model).unwrap();
        }
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let out = lidar
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &state,
                &scans,
                &mut pool,
                stamp(version),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        held.push((
            submitted.frame(&out.frame).unwrap().clone(),
            out.scene_stamp,
        ));
    }
    assert_ne!(held[0].1, held[1].1);
    assert_ne!(held[1].1, held[2].1);
    pool.reset().unwrap();
    assert_eq!(pool.available_slots().unwrap(), 0);
    drop((lidar, pool, scene));
    for (version, (frame, _)) in held.iter().enumerate() {
        let cpu = collect(&engine, frame);
        assert_eq!(
            cpu.channel::<Real>(2).unwrap(),
            [[4.0], [5.0], [-1.0]][version]
        );
        assert_eq!(cpu.channel::<i32>(3).unwrap(), [[1], [0], [-1]][version]);
        assert_eq!(cpu.channel::<u32>(8).unwrap(), [[1], [1], [0]][version]);
        assert_eq!(cpu.channel::<i32>(6).unwrap(), [[0], [0], [-1]][version]);
        assert_eq!(cpu.channel::<i32>(7).unwrap(), [[0], [0], [-1]][version]);
    }
}
