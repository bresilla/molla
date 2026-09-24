#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    AerialLidar, AerialScanMetadata, Lidar, LidarBatch, LidarPattern, LidarScan, ScanMetadata,
};
use molla_sim::{BodyParams, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires a GPU"]
fn million_ray_dispatch_covers_boundary_indices_and_final_pulse() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut scan = nadir(Vec3::Y * 10.0, 1);
    scan.n_theta = 1000;
    scan.n_phi = 1000;
    scan.theta_range = (0.6, 2.5);
    scan.phi_range = (-0.3, 0.3);
    let mut batch = LidarBatch::new(&engine, 1, 1_000_000).unwrap();
    let view = batch
        .dispatch(
            &engine,
            &model,
            &state,
            &[LidarScan {
                sensor_id: 5,
                pattern: LidarPattern::Terrestrial(scan),
            }],
        )
        .unwrap();
    let probes = [0, 63, 64, 999, 1000, 65_535, 65_536, 999_999];
    let mut sources = Vec::new();
    for i in probes {
        sources.push(view.hits.distance.readback_source(i..i + 1).unwrap());
        sources.push(view.hits.shape.readback_source(i..i + 1).unwrap());
        sources.push(view.first_ray.readback_source(i..i + 1).unwrap());
        sources.push(view.last_ray.readback_source(i..i + 1).unwrap());
    }
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 128,
            max_total_bytes: 128,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(&sources, ReadbackStamp::default())
        .unwrap()
        .unwrap();
    let frame = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    for (probe, i) in probes.iter().enumerate() {
        let expected = 10.0 / -scan.rc_to_direction(*i as u32 / 1000, *i as u32 % 1000).y;
        assert!((frame.channel::<Real>(probe * 4).unwrap()[0] - expected).abs() < 1e-4);
        assert_eq!(frame.channel::<i32>(probe * 4 + 1).unwrap(), [0]);
        assert_eq!(frame.channel::<i32>(probe * 4 + 2).unwrap(), [*i as i32]);
        assert_eq!(frame.channel::<i32>(probe * 4 + 3).unwrap(), [*i as i32]);
    }
    assert_eq!(frame.byte_len(), 128);
    assert_eq!(batch.stats().descriptor_upload_bytes, 128);
    assert_eq!(batch.stats().submissions, 1);
    assert_eq!(batch.query_stats().encoded_queries, 1);
    assert_eq!(batch.query_stats().submissions, 0);
}

#[test]
#[ignore = "requires a GPU"]
fn maximum_subray_budget_matches_cone_oracle_and_invalid_budget_is_rejected() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state) = scene(&engine);
    let mut scan = nadir(Vec3::new(-10.0, 1.0, 0.0), 1024);
    scan.beam_divergence = 0.1;
    let mut batch = LidarBatch::new(&engine, 1, 1024).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1024 * 16,
            max_total_bytes: 1024 * 16,
        },
    )
    .unwrap();
    let view = batch
        .dispatch(
            &engine,
            &model,
            &state,
            &[LidarScan {
                sensor_id: 0,
                pattern: LidarPattern::Terrestrial(scan),
            }],
        )
        .unwrap();
    let ticket = ring
        .try_enqueue(
            &[view.hits.directions.readback_source(0..1024).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    for (i, direction) in frame.channel_iter::<Vec3>(0).unwrap().enumerate() {
        assert!((direction.length() - 1.0).abs() < 2e-6);
        assert!(
            (direction - cone(Vec3::NEG_Y, 0.1, 1024, i)).length() < 3e-5,
            "subray {i}"
        );
    }
    scan.subray_count = 1025;
    assert!(
        batch
            .dispatch(
                &engine,
                &model,
                &state,
                &[LidarScan {
                    sensor_id: 0,
                    pattern: LidarPattern::Terrestrial(scan)
                }]
            )
            .is_err()
    );
    assert_eq!(batch.stats().batches, 1);
    assert!(LidarBatch::new(&engine, 0, 1).is_err());
    assert!(LidarBatch::new(&engine, 1025, 1).is_err());
    assert!(LidarBatch::new(&engine, 1, u32::MAX).is_err());
}

fn scene(engine: &Engine) -> (Model, State) {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mesh = b.add_mesh(TriMesh {
        vertices: vec![
            Vec3::new(-1.0, 2.0, -1.0),
            Vec3::new(1.0, 2.0, -1.0),
            Vec3::new(1.0, 2.0, 1.0),
            Vec3::new(-1.0, 2.0, 1.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    });
    b.add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(2.0, 2.0, 0.0)));
    b.add_shape_sphere(body, 0.7, Transform::IDENTITY).unwrap();
    let mut model = b.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state)
}

fn nadir(origin: Vec3, subrays: u32) -> ScanMetadata {
    let mut s = ScanMetadata::new(origin, 1, 1);
    s.forward = Vec3::NEG_Y;
    s.up = Vec3::Z;
    s.theta_range = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    s.phi_range = (0.0, 0.0);
    s.subray_count = subrays;
    s
}

fn cone(axis: Vec3, angle: f64, count: usize, i: usize) -> Vec3 {
    let axis = axis.normalize();
    if i == 0 || count == 1 || angle == 0.0 {
        return axis;
    }
    let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let e1 = axis.cross(helper).normalize();
    let e2 = axis.cross(e1).normalize();
    let a = angle * (i as f64 / (count - 1) as f64).sqrt();
    let b = i as f64 * std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (a.cos() * axis + a.sin() * (b.cos() * e1 + b.sin() * e2)).normalize()
}

#[test]
#[ignore = "requires a GPU"]
fn mixed_gpu_layouts_and_subray_selection_match_legacy_cpu_layouts() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state) = scene(&engine);
    let mut terrestrial = nadir(Vec3::new(0.0, 6.0, -0.3), 7);
    terrestrial.n_theta = 3;
    terrestrial.n_phi = 5;
    terrestrial.theta_range = (1.0, 2.2);
    terrestrial.phi_range = (-0.35, 0.35);
    terrestrial.forward = Vec3::new(0.2, -1.0, 0.0);
    terrestrial.beam_divergence = 0.05;
    let mut aerial =
        AerialScanMetadata::new(Vec3::new(-3.0, 6.0, -3.0), Vec3::new(3.0, 6.0, 3.0), 0.5);
    aerial.cone_angle = 0.17;
    aerial.beam_divergence = 0.03;
    aerial.subray_count = 3;
    let mut clipped = terrestrial;
    clipped.max_distance = 0.5;
    clipped.subray_count = 1;
    let scans = [
        LidarScan {
            sensor_id: 91,
            pattern: LidarPattern::Terrestrial(terrestrial),
        },
        LidarScan {
            sensor_id: 7,
            pattern: LidarPattern::Aerial(aerial),
        },
        LidarScan {
            sensor_id: 123,
            pattern: LidarPattern::Terrestrial(clipped),
        },
        LidarScan {
            sensor_id: 42,
            pattern: LidarPattern::Terrestrial(nadir(Vec3::new(-2.0, 6.0, 0.0), 4)),
        },
    ];
    let mut batch = LidarBatch::new(&engine, 4, 512).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 512 * 84,
            max_total_bytes: 512 * 84,
        },
    )
    .unwrap();
    let view = batch.dispatch(&engine, &model, &state, &scans).unwrap();
    assert!(view.hits.origins.host().is_err());
    assert!(view.hits.directions.host().is_err());
    let ranges = view.scans.to_vec();
    let n = view.hits.ray_count;
    let p = view.pulse_count;
    let ticket = ring
        .try_enqueue(
            &[
                view.hits.origins.readback_source(0..n).unwrap(),
                view.hits.directions.readback_source(0..n).unwrap(),
                view.hits.distance.readback_source(0..n).unwrap(),
                view.hits.shape.readback_source(0..n).unwrap(),
                view.hits.point.readback_source(0..n).unwrap(),
                view.hits.normal.readback_source(0..n).unwrap(),
                view.first_ray.readback_source(0..p).unwrap(),
                view.last_ray.readback_source(0..p).unwrap(),
                view.hit_count.readback_source(0..p).unwrap(),
            ],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.byte_len(), n * 72 + p * 12);
    let origins = frame.channel::<Vec3>(0).unwrap();
    let directions = frame.channel::<Vec3>(1).unwrap();
    let distances = frame.channel::<Real>(2).unwrap();
    let shapes = frame.channel::<i32>(3).unwrap();
    let points = frame.channel::<Vec3>(4).unwrap();
    let normals = frame.channel::<Vec3>(5).unwrap();
    let first = frame.channel::<i32>(6).unwrap();
    let last = frame.channel::<i32>(7).unwrap();
    let count = frame.channel::<u32>(8).unwrap();
    let mut legacy = Lidar::new(&engine, 512).unwrap();
    let mut legacy_aerial = AerialLidar::new(&engine, 512).unwrap();
    for (s, range) in scans.iter().zip(&ranges) {
        assert_eq!(s.sensor_id, range.scan.sensor_id);
        let (hits, pulse_origins, pulse_dirs, divergence) = match s.pattern {
            LidarPattern::Terrestrial(scan) => {
                let result = legacy.cast(&engine, &model, &state, &scan).unwrap();
                let dirs: Vec<_> = (0..scan.cell_count())
                    .map(|i| scan.rc_to_direction(i / scan.n_phi, i % scan.n_phi))
                    .collect();
                (
                    result.hits,
                    vec![scan.origin; scan.cell_count() as usize],
                    dirs,
                    scan.beam_divergence,
                )
            }
            LidarPattern::Aerial(scan) => {
                let result = legacy_aerial.cast(&engine, &model, &state, &scan).unwrap();
                (
                    result.hits,
                    result.origins,
                    result.directions,
                    scan.beam_divergence,
                )
            }
        };
        let k = range.subray_count as usize;
        for (local, reference) in hits.iter().enumerate() {
            let i = range.ray_offset as usize + local;
            assert!((origins[i] - pulse_origins[local / k]).length() < 2e-5);
            let expected_dir = cone(pulse_dirs[local / k], divergence, k, local % k);
            assert!(
                (directions[i] - expected_dir).length() < 2e-5,
                "ray {i}: {:?} vs {:?}",
                directions[i],
                expected_dir
            );
            assert_eq!(
                shapes[i],
                reference.shape_id.map(|v| v as i32).unwrap_or(-1),
                "ray {i}"
            );
            if reference.shape_id.is_some() {
                assert!((distances[i] - reference.t).abs() < 1e-3, "ray {i}");
                assert!((points[i] - reference.point).length() < 1e-3);
                assert!((normals[i] - reference.normal).length() < 1e-3);
            } else {
                assert_eq!(distances[i], -1.0);
                assert_eq!(points[i], Vec3::ZERO);
                assert_eq!(normals[i], Vec3::Y);
            }
        }
        for pulse in 0..range.pulse_count as usize {
            let begin = range.ray_offset as usize + pulse * k;
            let valid: Vec<_> = (begin..begin + k).filter(|&i| shapes[i] >= 0).collect();
            let expected_first = valid
                .iter()
                .min_by(|&&a, &&b| distances[a].total_cmp(&distances[b]))
                .map(|&i| i as i32)
                .unwrap_or(-1);
            let expected_last = valid
                .iter()
                .max_by(|&&a, &&b| distances[a].total_cmp(&distances[b]))
                .map(|&i| i as i32)
                .unwrap_or(-1);
            let index = range.pulse_offset as usize + pulse;
            assert_eq!(first[index], expected_first);
            assert_eq!(last[index], expected_last);
            assert_eq!(count[index], valid.len() as u32);
        }
    }
    assert_eq!(batch.stats().descriptor_upload_bytes, 4 * 128);
    assert_eq!(batch.readback_stats().copied_bytes, 0);
    assert_eq!(batch.readback_stats().blocking_waits, 0);
    let final_pulse = p - 1;
    assert_eq!(first[final_pulse], n as i32 - 4);
    assert_eq!(last[final_pulse], n as i32 - 1);
}

#[test]
#[ignore = "requires a GPU"]
fn many_independent_mounts_snapshots_empty_batches_and_capacity_rejection() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state) = scene(&engine);
    let scans: Vec<_> = (0..70)
        .map(|i| {
            let mut scan = nadir(
                Vec3::new(-10.0, 3.0 + i as f64 * 0.1, 0.0),
                if i % 2 == 0 { 0 } else { 2 },
            );
            scan.max_distance = if i % 3 == 0 { 0.5 } else { 100.0 };
            LidarScan {
                sensor_id: i + 900,
                pattern: LidarPattern::Terrestrial(scan),
            }
        })
        .collect();
    let mut batch = LidarBatch::new(&engine, 70, 105).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 512,
            max_total_bytes: 1024,
        },
    )
    .unwrap();
    let result = batch.dispatch(&engine, &model, &state, &scans).unwrap();
    assert_eq!((result.hits.ray_count, result.pulse_count), (105, 70));
    let ticket1 = ring
        .try_enqueue(
            &[result.hit_count.readback_source(0..70).unwrap()],
            ReadbackStamp {
                sample_id: 1,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let all_clipped: Vec<_> = scans
        .iter()
        .copied()
        .map(|mut s| {
            if let LidarPattern::Terrestrial(ref mut metadata) = s.pattern {
                metadata.max_distance = 0.25;
            }
            s
        })
        .collect();
    let result = batch
        .dispatch(&engine, &model, &state, &all_clipped)
        .unwrap();
    let ticket2 = ring
        .try_enqueue(
            &[result.first_ray.readback_source(0..70).unwrap()],
            ReadbackStamp {
                sample_id: 2,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let frame2 = ring.wait(&ticket2, Duration::from_secs(10)).unwrap();
    assert!(frame2.channel::<i32>(0).unwrap().iter().all(|&i| i == -1));
    let frame1 = ring.wait(&ticket1, Duration::from_secs(10)).unwrap();
    for (i, count) in frame1.channel_iter::<u32>(0).unwrap().enumerate() {
        assert_eq!(
            count,
            if i % 3 == 0 {
                0
            } else if i % 2 == 0 {
                1
            } else {
                2
            }
        );
    }
    let before = batch.stats();
    let empty = batch.dispatch(&engine, &model, &state, &[]).unwrap();
    assert_eq!(empty.hits.ray_count, 0);
    assert_eq!(empty.pulse_count, 0);
    assert!(empty.scans.is_empty());
    assert_eq!(
        batch.stats().descriptor_upload_bytes,
        before.descriptor_upload_bytes
    );
    let mut too_many = scans.clone();
    too_many.push(scans[0]);
    assert!(batch.dispatch(&engine, &model, &state, &too_many).is_err());
    if let LidarPattern::Terrestrial(ref mut s) = too_many[0].pattern {
        s.subray_count = 4;
    }
    assert!(
        batch
            .dispatch(&engine, &model, &state, &too_many[..70])
            .is_err()
    );
    assert_eq!(batch.stats().batches, 2);
    assert_eq!(batch.stats().submissions, 2);
    let warm = batch.query_stats();
    assert!(batch.dispatch(&engine, &model, &state, &scans[..1]).is_ok());
    let reused = batch.query_stats();
    assert_eq!(reused.primitive_preparations, warm.primitive_preparations);
    assert_eq!(reused.mesh_preparations, warm.mesh_preparations);
    assert_eq!(reused.merge_preparations, warm.merge_preparations);
    assert_eq!(reused.submissions, 0);
    assert_eq!(reused.encoded_queries, 3);
    assert_eq!(batch.stats().submissions, 3);
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    assert!(batch.dispatch(&foreign, &model, &state, &[]).is_err());
}
