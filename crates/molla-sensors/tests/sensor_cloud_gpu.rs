#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::{
    AerialScanMetadata, CloudFilter, CloudFrame, CloudInput, CloudJob, CloudPoint, CloudProcessor,
    CloudProcessorConfig, CloudRaster, CloudSummary, Compare, DeviceSnapshotConfig,
    DeviceSnapshotPool, LidarBatch, LidarPattern, LidarScan, ScanMetadata, SensorScene,
    SensorSceneOptions,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_sim::ModelBuilder;
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn processor(engine: &Engine, n: usize, jobs: u32) -> CloudProcessor {
    CloudProcessor::new(
        engine,
        CloudProcessorConfig {
            max_points: n.max(1) as u32,
            max_jobs: jobs,
            max_buffer_bytes: 512 * 1024 * 1024,
        },
    )
    .unwrap()
}
fn source<T: molla_storage::BufferElement>(engine: &Engine, data: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, data, BufferUsage::StorageReadWrite).unwrap()
}
fn ring(engine: &Engine, n: usize, jobs: usize, slots: usize) -> ReadbackRing {
    let bytes = (n as u64 * 112 + jobs as u64 * 24 + 16).next_multiple_of(8);
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots,
            bytes_per_slot: bytes,
            max_total_bytes: bytes * slots as u64,
        },
    )
    .unwrap()
}
fn collect(frame: &CloudFrame<'_>, ring: &mut ReadbackRing) -> ReadbackFrame {
    let mut sources = vec![
        frame.count_source().unwrap(),
        frame.points_source(0..frame.input_count()).unwrap(),
    ];
    for i in 0..frame.layouts().len() {
        sources.push(frame.summary_source(i).unwrap());
    }
    let ticket = ring
        .try_enqueue(&sources, Default::default())
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(10)).unwrap()
}
fn vec3(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real)
}
fn point(i: u32) -> CloudPoint {
    let mut p = CloudPoint::new(
        Vec3::new(
            (i % 71) as Real * 0.125 - 4.0,
            (i % 11) as Real * 0.25,
            (i % 5) as Real,
        ),
        0xfedcba9876543210,
        0x123456789abcdef0,
    );
    p.position[3] = (i % 8) as f32 / 8.0;
    p.normal = [0.0, 1.0, 0.0, i as f32 + 1.0];
    p.direction = [0.0, -1.0, 0.0, (i % 7) as f32];
    p.grid = [i / 13, i % 13, i % 3, i];
    p.extra = [i, 0x80c0a010, 15, 0];
    p.angles = [0.25, 1.5, 0.0, 0.0];
    p
}

fn oracle(
    input: &[CloudPoint],
    transform: Transform,
    filter: CloudFilter,
) -> (Vec<CloudPoint>, [u32; 3]) {
    let mut out = Vec::new();
    let mut dropped = [0; 3];
    for p in input {
        if p.extra[2] & 1 == 0 {
            dropped[0] += 1;
            continue;
        }
        let valid = vec3(p.position).is_finite()
            && vec3(p.normal).is_finite()
            && vec3(p.direction).is_finite()
            && p.position[3].is_finite()
            && p.normal[3].is_finite()
            && p.normal[3] >= 0.0
            && (p.extra[2] & 4 == 0 || p.direction[3].is_finite())
            && (p.extra[2] & 8 == 0 || (p.angles[0].is_finite() && p.angles[1].is_finite()));
        if !valid {
            dropped[1] += 1;
            continue;
        }
        let mut p = *p;
        let position = transform.transform_point(vec3(p.position));
        let normal = transform.transform_vector(vec3(p.normal));
        let direction = transform.transform_vector(vec3(p.direction));
        if !position.as_vec3().is_finite()
            || !normal.as_vec3().is_finite()
            || !direction.as_vec3().is_finite()
        {
            dropped[1] += 1;
            continue;
        }
        let distance = filter
            .distance
            .is_none_or(|(origin, radius)| position.distance(origin) <= radius);
        let bounds = filter.bounds.is_none_or(|(min, max, keep)| {
            (position.cmpge(min).all() && position.cmple(max).all()) == keep
        });
        let intensity = filter.min_intensity.is_none_or(|min| p.position[3] >= min);
        let scalar = filter.scalar.is_none_or(|(op, value)| {
            p.extra[2] & 4 != 0
                && match op {
                    Compare::Lt => p.direction[3] < value,
                    Compare::Le => p.direction[3] <= value,
                    Compare::Gt => p.direction[3] > value,
                    Compare::Ge => p.direction[3] >= value,
                    Compare::Eq => (p.direction[3] - value).abs() < 1e-6,
                }
        });
        if !(distance && bounds && intensity && scalar) {
            dropped[2] += 1;
            continue;
        }
        for (target, value) in [
            (&mut p.position, position),
            (&mut p.normal, normal),
            (&mut p.direction, direction),
        ] {
            target[..3].copy_from_slice(&[value.x as f32, value.y as f32, value.z as f32]);
        }
        out.push(p);
    }
    (out, dropped)
}

fn check_point(a: &CloudPoint, e: &CloudPoint) {
    assert_eq!(a.ids, e.ids);
    assert_eq!(a.grid, e.grid);
    assert_eq!(a.extra, e.extra);
    for (a, e) in a
        .position
        .iter()
        .chain(&a.normal)
        .chain(&a.direction)
        .chain(&a.angles)
        .zip(
            e.position
                .iter()
                .chain(&e.normal)
                .chain(&e.direction)
                .chain(&e.angles),
        )
    {
        assert!((a - e).abs() <= 3e-5 * e.abs().max(1.0), "{a} != {e}");
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn stable_compaction_transform_filters_and_diagnostics_match_cpu() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut proc = processor(&engine, 70_000, 8);
    let mut ring = ring(&engine, 70_000, 8, 1);
    let transform = Transform::new(Vec3::new(0.3, -0.2, 0.4), Quat::from_rotation_y(0.37));
    for n in [0, 1, 255, 256, 257, 65_537] {
        let mut values: Vec<_> = (0..n as u32).map(point).collect();
        for (i, p) in values.iter_mut().enumerate() {
            if i % 19 == 0 {
                p.extra[2] &= !1;
            } else if i % 31 == 0 {
                p.position[0] = f32::NAN;
            } else if i % 43 == 0 {
                p.direction[3] = f32::INFINITY;
            } else if i % 47 == 0 {
                p.extra[2] &= !4;
            }
        }
        let filters = [
            CloudFilter::default(),
            CloudFilter {
                distance: Some((Vec3::ZERO, 3.123)),
                ..Default::default()
            },
            CloudFilter {
                bounds: Some((Vec3::splat(-2.25), Vec3::splat(2.25), false)),
                min_intensity: Some(0.5),
                scalar: Some((Compare::Ge, 2.0)),
                ..Default::default()
            },
        ];
        let split = n / 3;
        let ranges = [0..split, split..split, split..2 * split, 2 * split..n];
        let input = source(&engine, values.clone());
        let jobs: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(i, range)| CloudJob {
                input: CloudInput::Points(input.readback_source(range.clone()).unwrap()),
                transform,
                filter: filters[i % 3],
                stamp: ReadbackStamp {
                    sensor_id: i as u64,
                    sample_id: n as u64,
                    ..Default::default()
                },
            })
            .collect();
        let out = proc.process(&jobs).unwrap();
        let result = collect(&out, &mut ring);
        let points = result.channel::<CloudPoint>(1).unwrap();
        let mut offset = 0;
        for (i, range) in ranges.iter().enumerate() {
            let (expected, dropped) = oracle(&values[range.clone()], transform, filters[i % 3]);
            let summary = CloudSummary::from_words(&result.channel::<u32>(i + 2).unwrap()).unwrap();
            assert_eq!(
                summary,
                CloudSummary {
                    start: offset as u32,
                    count: expected.len() as u32,
                    misses: dropped[0],
                    invalid: dropped[1],
                    filtered: dropped[2]
                }
            );
            for (a, e) in points[offset..offset + expected.len()]
                .iter()
                .zip(&expected)
            {
                check_point(a, e);
            }
            assert_eq!(out.layouts()[i].stamp.sample_id, n as u64);
            offset += expected.len();
        }
        assert_eq!(result.channel::<u32>(0).unwrap(), [offset as u32]);
        assert!(points[offset..].iter().all(|p| *p == CloudPoint::default()));
    }
    assert_eq!(proc.stats().frames, 6);
    assert_eq!(proc.stats().submissions, 6);
    assert_eq!(proc.stats().kernel_dispatches, 15);
    assert_eq!(proc.stats().buffer_allocations, 8);
    assert_eq!(proc.stats().binding_preparations, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn lidar_clouds_preserve_mount_grid_subray_shape_and_full_width_ids() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut meta = ScanMetadata::new(Vec3::new(0.0, 2.0, 0.0), 5, 7);
    meta.theta_range = (2.0, 2.8);
    meta.phi_range = (-1.0, 1.0);
    meta.subray_count = 3;
    meta.beam_divergence = 0.1;
    let mut up = meta;
    up.origin.x = 4.0;
    up.theta_range = (0.1, 0.5);
    let mut aerial =
        AerialScanMetadata::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 3.0, 1.0), 2.0);
    aerial.subray_count = 2;
    let scans = [
        LidarScan {
            sensor_id: u64::MAX,
            pattern: LidarPattern::Terrestrial(meta),
        },
        LidarScan {
            sensor_id: 0xabcdef0123456789,
            pattern: LidarPattern::Terrestrial(up),
        },
        LidarScan {
            sensor_id: 77,
            pattern: LidarPattern::Aerial(aerial),
        },
    ];
    let mut lidar = LidarBatch::new(&engine, 3, 226).unwrap();
    let view = lidar
        .dispatch_scene(&engine, &scene, &state, &scans)
        .unwrap();
    assert!(CloudInput::from_lidar(&view, 3).is_err());
    let jobs: Vec<_> = scans
        .iter()
        .enumerate()
        .map(|(i, scan)| CloudJob {
            input: CloudInput::from_lidar(&view, i).unwrap(),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: ReadbackStamp {
                sensor_id: scan.sensor_id,
                sample_id: 0x1234567890abcdef,
                scene_revision: scene.stamp().geometry_revision,
                simulation_time: Some(0.5),
            },
        })
        .collect();
    let mut proc = processor(&engine, 226, 3);
    let out = proc.process(&jobs).unwrap();
    let mut ring = ring(&engine, 226, 3, 2);
    let result = collect(&out, &mut ring);
    assert_eq!(result.channel::<u32>(0).unwrap(), [121]);
    let points = result.channel::<CloudPoint>(1).unwrap();
    let reference = ring
        .try_enqueue(
            &[
                view.hits.point.readback_source(0..105).unwrap(),
                view.hits.normal.readback_source(0..105).unwrap(),
                view.hits.directions.readback_source(0..105).unwrap(),
                view.hits.distance.readback_source(0..105).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let reference = ring.wait(&reference, Duration::from_secs(10)).unwrap();
    let positions = reference.channel::<Vec3>(0).unwrap();
    let normals = reference.channel::<Vec3>(1).unwrap();
    let directions = reference.channel::<Vec3>(2).unwrap();
    let distances = reference.channel::<Real>(3).unwrap();
    for (i, p) in points[105..121].iter().enumerate() {
        assert_eq!(p.sensor_id(), 77);
        assert_eq!(p.sample_id(), 0x1234567890abcdef);
        assert_eq!(p.grid, [0, (i / 2) as u32, (i % 2) as u32, 210 + i as u32]);
        assert_eq!(p.extra, [0, u32::MAX, 1, 0]);
        assert_eq!(p.angles, [0.0; 4]);
        assert_eq!(p.normal[3], 3.0);
    }
    assert_eq!(
        CloudSummary::from_words(&result.channel::<u32>(4).unwrap()).unwrap(),
        CloudSummary {
            start: 105,
            count: 16,
            misses: 0,
            invalid: 0,
            filtered: 0
        }
    );
    for (i, p) in points[..105].iter().enumerate() {
        assert_eq!(p.sensor_id(), u64::MAX);
        assert_eq!(p.sample_id(), 0x1234567890abcdef);
        assert_eq!(
            p.grid,
            [
                (i / 21) as u32,
                ((i / 3) % 7) as u32,
                (i % 3) as u32,
                i as u32
            ]
        );
        assert_eq!(p.extra, [0, u32::MAX, 9, 0]);
        assert_eq!(vec3(p.position), positions[i]);
        assert_eq!(vec3(p.normal), normals[i]);
        assert_eq!(vec3(p.direction), directions[i]);
        assert_eq!(p.normal[3] as Real, distances[i]);
        assert!((p.angles[0] as Real - (2.0 + 0.8 * p.grid[0] as Real / 4.0)).abs() < 1e-6);
        assert!((p.angles[1] as Real - (-1.0 + 2.0 * p.grid[1] as Real / 6.0)).abs() < 1e-6);
    }
    assert_eq!(
        CloudSummary::from_words(&result.channel::<u32>(3).unwrap()).unwrap(),
        CloudSummary {
            start: 105,
            count: 0,
            misses: 105,
            invalid: 0,
            filtered: 0
        }
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn snapshots_queued_frames_all_rejected_and_empty_outputs_remain_distinct() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let values: Vec<_> = (0..257).map(point).collect();
    let input = source(&engine, values.clone());
    let mut proc = processor(&engine, 257, 1);
    let mut other = processor(&engine, 257, 1);
    let mut pool = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 257 * 112,
            max_total_bytes: 257 * 112,
        },
    )
    .unwrap();
    let mut ring = ring(&engine, 257, 1, 3);
    let out = proc
        .process(&[CloudJob {
            input: CloudInput::Points(input.readback_source(0..257).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default(),
        }])
        .unwrap();
    let snapshot = pool
        .try_capture(&[out.points_source(0..257).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let old = ring
        .try_enqueue(
            &[
                out.count_source().unwrap(),
                out.points_source(0..257).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let out = other
        .process(&[CloudJob {
            input: CloudInput::Points(
                snapshot
                    .channel::<CloudPoint>(0)
                    .unwrap()
                    .readback_source(0..257)
                    .unwrap(),
            ),
            transform: Transform::from_translation(Vec3::X),
            filter: Default::default(),
            stamp: Default::default(),
        }])
        .unwrap();
    let transformed = ring
        .try_enqueue(
            &[
                out.count_source().unwrap(),
                out.points_source(0..257).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    drop(snapshot);
    pool.reset().unwrap();
    let out = proc
        .process(&[CloudJob {
            input: CloudInput::Points(input.readback_source(0..257).unwrap()),
            transform: Transform::IDENTITY,
            filter: CloudFilter {
                min_intensity: Some(2.0),
                ..Default::default()
            },
            stamp: Default::default(),
        }])
        .unwrap();
    let newest = ring
        .try_enqueue(
            &[
                out.count_source().unwrap(),
                out.points_source(0..257).unwrap(),
                out.summary_source(0).unwrap(),
            ],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let newest = ring.wait(&newest, Duration::from_secs(10)).unwrap();
    assert_eq!(newest.channel::<u32>(0).unwrap(), [0]);
    assert!(
        newest
            .channel::<CloudPoint>(1)
            .unwrap()
            .iter()
            .all(|p| *p == CloudPoint::default())
    );
    assert_eq!(
        CloudSummary::from_words(&newest.channel::<u32>(2).unwrap())
            .unwrap()
            .filtered,
        257
    );
    let old = ring.wait(&old, Duration::from_secs(10)).unwrap();
    assert_eq!(old.channel::<CloudPoint>(1).unwrap(), values);
    let transformed = ring.wait(&transformed, Duration::from_secs(10)).unwrap();
    let (expected, _) = oracle(
        &values,
        Transform::from_translation(Vec3::X),
        Default::default(),
    );
    for (a, e) in transformed
        .channel::<CloudPoint>(1)
        .unwrap()
        .iter()
        .zip(&expected)
    {
        check_point(a, e);
    }
    let out = proc.process(&[]).unwrap();
    assert!(out.layouts().is_empty());
    assert!(out.summary_source(0).is_err());
    assert!(out.points_source(0..1).is_err());
    assert_eq!(collect(&out, &mut ring).channel::<u32>(0).unwrap(), [0]);
    assert_eq!(pool.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn invalid_job_descriptions_do_not_submit_partial_work() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let input = source(&engine, vec![point(0), point(1)]);
    let mut proc = processor(&engine, 2, 1);
    let stats = proc.stats();
    for filter in [
        CloudFilter {
            distance: Some((Vec3::ZERO, -1.0)),
            ..Default::default()
        },
        CloudFilter {
            bounds: Some((Vec3::ONE, Vec3::ZERO, true)),
            ..Default::default()
        },
        CloudFilter {
            min_intensity: Some(f32::NAN),
            ..Default::default()
        },
        CloudFilter {
            scalar: Some((Compare::Eq, f32::INFINITY)),
            ..Default::default()
        },
    ] {
        assert!(
            proc.process(&[CloudJob {
                input: CloudInput::Points(input.readback_source(0..2).unwrap()),
                transform: Transform::IDENTITY,
                filter,
                stamp: Default::default()
            }])
            .is_err()
        );
    }
    let wrong = source(&engine, vec![1u32]);
    assert!(
        proc.process(&[CloudJob {
            input: CloudInput::Points(wrong.readback_source(0..1).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default()
        }])
        .is_err()
    );
    for transform in [
        Transform::from_translation(Vec3::splat(f64::MAX)),
        Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
    ] {
        assert!(
            proc.process(&[CloudJob {
                input: CloudInput::Points(input.readback_source(0..1).unwrap()),
                transform,
                filter: Default::default(),
                stamp: Default::default()
            }])
            .is_err()
        );
    }
    let p = source(&engine, vec![Vec3::ZERO]);
    let n = source(&engine, vec![Vec3::Y]);
    let d = source(&engine, vec![Vec3::Z]);
    let t = source(&engine, vec![1.0 as Real]);
    let s = source(&engine, vec![0i32]);
    for raster in [
        CloudRaster {
            sensor_id: 1,
            rows: 1,
            columns: 1,
            subrays: 1,
            source_offset: 0,
            angles: None,
        },
        CloudRaster {
            sensor_id: 0,
            rows: u32::MAX,
            columns: 2,
            subrays: 1,
            source_offset: 0,
            angles: None,
        },
        CloudRaster {
            sensor_id: 0,
            rows: 1,
            columns: 1,
            subrays: 1,
            source_offset: u32::MAX,
            angles: None,
        },
    ] {
        assert!(
            proc.process(&[CloudJob {
                input: CloudInput::Rays {
                    channels: [
                        p.readback_source(0..1).unwrap(),
                        n.readback_source(0..1).unwrap(),
                        d.readback_source(0..1).unwrap(),
                        t.readback_source(0..1).unwrap(),
                        s.readback_source(0..1).unwrap()
                    ],
                    raster
                },
                transform: Transform::IDENTITY,
                filter: Default::default(),
                stamp: Default::default()
            }])
            .is_err()
        );
    }
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    let foreign = source(&foreign, vec![point(0)]);
    assert!(
        proc.process(&[CloudJob {
            input: CloudInput::Points(foreign.readback_source(0..1).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default()
        }])
        .is_err()
    );
    assert_eq!(proc.stats(), stats);
}

#[test]
#[ignore = "needs a real GPU"]
fn million_point_global_prefix_keeps_exact_order_across_all_blocks() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let n = 1_000_003usize;
    let input = source(
        &engine,
        (0..n as u32)
            .map(|i| {
                let mut p = point(i);
                if i % 3 != 1 {
                    p.extra[2] &= !1;
                }
                p
            })
            .collect(),
    );
    let mut proc = processor(&engine, n, 1);
    let frame = proc
        .process(&[CloudJob {
            input: CloudInput::Points(input.readback_source(0..n).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default(),
        }])
        .unwrap();
    let count = (n + 1) / 3;
    let indices = [0, 255, 256, 32768, count - 1];
    let mut sources = vec![
        frame.count_source().unwrap(),
        frame.summary_source(0).unwrap(),
    ];
    for i in indices {
        sources.push(frame.points_source(i..i + 1).unwrap());
    }
    let mut ring = ring(&engine, indices.len(), 1, 1);
    let ticket = ring
        .try_enqueue(&sources, Default::default())
        .unwrap()
        .unwrap();
    let result = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(result.channel::<u32>(0).unwrap(), [count as u32]);
    assert_eq!(
        CloudSummary::from_words(&result.channel::<u32>(1).unwrap()).unwrap(),
        CloudSummary {
            start: 0,
            count: count as u32,
            misses: (n - count) as u32,
            invalid: 0,
            filtered: 0,
        }
    );
    for (channel, i) in indices.into_iter().enumerate() {
        assert_eq!(
            result.channel::<CloudPoint>(channel + 2).unwrap(),
            [point((i * 3 + 1) as u32)]
        );
    }
    assert_eq!(proc.stats().kernel_dispatches, 3);
    assert_eq!(proc.stats().submissions, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn scalar_comparators_inclusive_bounds_zero_radius_and_invalid_attributes() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut values = vec![CloudPoint::new(Vec3::ZERO, 11, 22); 10];
    values[1].direction[3] = 2.0;
    values[1].extra[2] |= 4;
    values[2].direction[3] = 3.0;
    values[2].extra[2] |= 4;
    values[3].direction[3] = 4.0;
    values[3].extra[2] |= 4;
    values[4].normal[0] = f32::NAN;
    values[5].normal[3] = -1.0;
    values[6].position[3] = f32::INFINITY;
    values[7].extra[2] |= 8;
    values[7].angles[0] = f32::NAN;
    values[8].position[0] = 1.0;
    values[9].direction[0] = f32::INFINITY;
    for (i, p) in values.iter_mut().enumerate() {
        p.grid[3] = i as u32;
    }
    let input = source(&engine, values.clone());
    let mut proc = processor(&engine, 10, 1);
    let mut ring = ring(&engine, 10, 1, 1);
    let mut filters: Vec<_> = [
        Compare::Lt,
        Compare::Le,
        Compare::Gt,
        Compare::Ge,
        Compare::Eq,
    ]
    .into_iter()
    .map(|op| CloudFilter {
        scalar: Some((op, 3.0)),
        ..Default::default()
    })
    .collect();
    filters.push(CloudFilter {
        distance: Some((Vec3::ZERO, 0.0)),
        ..Default::default()
    });
    filters.push(CloudFilter {
        bounds: Some((Vec3::ZERO, Vec3::X, true)),
        ..Default::default()
    });
    for filter in filters {
        let frame = proc
            .process(&[CloudJob {
                input: CloudInput::Points(input.readback_source(0..10).unwrap()),
                transform: Transform::IDENTITY,
                filter,
                stamp: Default::default(),
            }])
            .unwrap();
        let result = collect(&frame, &mut ring);
        let (expected, dropped) = oracle(&values, Transform::IDENTITY, filter);
        let summary = CloudSummary::from_words(&result.channel::<u32>(2).unwrap()).unwrap();
        assert_eq!(summary.count, expected.len() as u32);
        assert_eq!([summary.misses, summary.invalid, summary.filtered], dropped);
        assert_eq!(
            &result.channel::<CloudPoint>(1).unwrap()[..expected.len()],
            expected
        );
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn finite_large_angle_interpolation_does_not_overflow_and_transform_overflow_is_reported() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let p = source(&engine, vec![Vec3::ZERO; 3]);
    let n = source(&engine, vec![Vec3::Y; 3]);
    let d = source(&engine, vec![Vec3::Z; 3]);
    let t = source(&engine, vec![1.0 as Real; 3]);
    let s = source(&engine, vec![0i32; 3]);
    let mut proc = processor(&engine, 3, 1);
    let mut ring = ring(&engine, 3, 1, 1);
    let frame = proc
        .process(&[CloudJob {
            input: CloudInput::Rays {
                channels: [
                    p.readback_source(0..3).unwrap(),
                    n.readback_source(0..3).unwrap(),
                    d.readback_source(0..3).unwrap(),
                    t.readback_source(0..3).unwrap(),
                    s.readback_source(0..3).unwrap(),
                ],
                raster: CloudRaster {
                    sensor_id: 0,
                    rows: 1,
                    columns: 3,
                    subrays: 1,
                    source_offset: 0,
                    angles: Some([(0.0, 0.0), (0.0, 3e38)]),
                },
            },
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default(),
        }])
        .unwrap();
    let result = collect(&frame, &mut ring);
    assert_eq!(result.channel::<u32>(0).unwrap(), [3]);
    for (i, point) in result.channel::<CloudPoint>(1).unwrap().iter().enumerate() {
        assert!(point.angles[1].is_finite());
        let expected = 3e38_f32 * (i as f32 / 2.0);
        assert_eq!(point.angles[1], expected);
    }
    let overflow = source(
        &engine,
        vec![CloudPoint::new(Vec3::X * f32::MAX as f64, 0, 0)],
    );
    let frame = proc
        .process(&[CloudJob {
            input: CloudInput::Points(overflow.readback_source(0..1).unwrap()),
            transform: Transform::from_translation(Vec3::X * f32::MAX as f64),
            filter: Default::default(),
            stamp: Default::default(),
        }])
        .unwrap();
    let result = collect(&frame, &mut ring);
    assert_eq!(result.channel::<u32>(0).unwrap(), [0]);
    assert_eq!(
        CloudSummary::from_words(&result.channel::<u32>(2).unwrap())
            .unwrap()
            .invalid,
        1
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn extreme_finite_transforms_and_distance_filters_guard_overflow_before_arithmetic() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut proc = processor(&engine, 1, 1);
    let mut ring = ring(&engine, 1, 1, 1);
    let max = f32::MAX as f64;
    let mut point = CloudPoint::new(Vec3::ZERO, 0, 0);
    let cases = [
        (Vec3::X, Vec3::X, 0.0, None, [0, 1, 0], Vec3::ZERO),
        (-Vec3::X, -Vec3::X, 0.0, None, [0, 1, 0], Vec3::ZERO),
        (Vec3::X, -Vec3::X, 0.0, None, [1, 0, 0], Vec3::ZERO),
        (Vec3::X, Vec3::ZERO, 0.0, None, [1, 0, 0], Vec3::X),
        (
            Vec3::X,
            Vec3::ZERO,
            std::f64::consts::FRAC_PI_2,
            None,
            [1, 0, 0],
            Vec3::Y,
        ),
        (
            Vec3::X + Vec3::Y,
            Vec3::ZERO,
            std::f64::consts::FRAC_PI_4,
            None,
            [0, 1, 0],
            Vec3::ZERO,
        ),
        (
            Vec3::X + Vec3::Y,
            -Vec3::Y,
            std::f64::consts::FRAC_PI_4,
            None,
            [1, 0, 0],
            Vec3::Y * (2.0_f64.sqrt() - 1.0),
        ),
        (
            Vec3::X,
            Vec3::ZERO,
            0.0,
            Some((-Vec3::X * max, max)),
            [0, 0, 1],
            Vec3::ZERO,
        ),
        (
            Vec3::X + Vec3::Y,
            Vec3::ZERO,
            0.0,
            Some((Vec3::ZERO, max)),
            [0, 0, 1],
            Vec3::ZERO,
        ),
        (
            Vec3::X,
            Vec3::ZERO,
            0.0,
            Some((Vec3::ZERO, max)),
            [1, 0, 0],
            Vec3::X,
        ),
        (
            Vec3::splat(0.1),
            Vec3::ZERO,
            0.0,
            Some((Vec3::ZERO, max)),
            [1, 0, 0],
            Vec3::splat(0.1),
        ),
    ];
    for (index, (position, translation, angle, distance, counts, expected)) in
        cases.into_iter().enumerate()
    {
        let input = source(&engine, vec![CloudPoint::new(position * max, 0, 0)]);
        let frame = proc
            .process(&[CloudJob {
                input: CloudInput::Points(input.readback_source(0..1).unwrap()),
                transform: Transform::new(translation * max, Quat::from_rotation_z(angle)),
                filter: CloudFilter {
                    distance,
                    ..Default::default()
                },
                stamp: Default::default(),
            }])
            .unwrap();
        let out = collect(&frame, &mut ring);
        let summary = CloudSummary::from_words(&out.channel::<u32>(2).unwrap()).unwrap();
        assert_eq!(
            [summary.count, summary.invalid, summary.filtered],
            counts,
            "case {index}"
        );
        if counts[0] == 1 {
            let point = out.channel::<CloudPoint>(1).unwrap()[0];
            let actual = vec3(point.position) / max;
            assert!(
                actual.is_finite() && (actual - expected).length() < 3e-6,
                "case {index}: {actual:?}"
            );
        }
    }
    for normal in [true, false] {
        let target = if normal {
            &mut point.normal
        } else {
            &mut point.direction
        };
        target[..3].copy_from_slice(&[f32::MAX, f32::MAX, 0.0]);
        let input = source(&engine, vec![point]);
        let frame = proc
            .process(&[CloudJob {
                input: CloudInput::Points(input.readback_source(0..1).unwrap()),
                transform: Transform::new(
                    Vec3::ZERO,
                    Quat::from_rotation_z(std::f64::consts::FRAC_PI_4),
                ),
                filter: Default::default(),
                stamp: Default::default(),
            }])
            .unwrap();
        let out = collect(&frame, &mut ring);
        let summary = CloudSummary::from_words(&out.channel::<u32>(2).unwrap()).unwrap();
        assert_eq!(
            [summary.count, summary.invalid, summary.filtered],
            [0, 1, 0]
        );
        point = CloudPoint::new(Vec3::ZERO, 0, 0);
    }
}
