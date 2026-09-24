#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Transform, Vec3};
use molla_sensors::{
    CloudFilter, CloudGrid, CloudGridCell, CloudGridConfig, CloudGridDiagnostics, CloudGridFrame,
    CloudGridSpec, CloudInput, CloudJob, CloudPoint, CloudProcessor, CloudProcessorConfig,
    DeviceSnapshotConfig, DeviceSnapshotPool,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_storage::{BufferElement, BufferUsage};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn source<T: BufferElement>(engine: &Engine, data: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, data, BufferUsage::StorageReadWrite).unwrap()
}
fn grid(engine: &Engine, n: usize, c: u32) -> CloudGrid {
    CloudGrid::new(
        engine,
        CloudGridConfig {
            max_points: n.max(1) as u32,
            max_cells: c,
            max_buffer_bytes: 256 * 1024 * 1024,
        },
    )
    .unwrap()
}
fn spec(dimensions: [u32; 3]) -> CloudGridSpec {
    CloudGridSpec {
        minimum: [-1.; 3],
        maximum: [1.; 3],
        dimensions,
    }
}
fn ring(engine: &Engine, n: usize, c: usize, slots: usize) -> ReadbackRing {
    let bytes = (n as u64 * 116 + c as u64 * 64 + 32).next_multiple_of(8);
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
fn enqueue(
    frame: &CloudGridFrame<'_>,
    ring: &mut ReadbackRing,
) -> molla_sensors::readback::ReadbackTicket {
    ring.try_enqueue(
        &[
            frame.cells_source(0..frame.cell_count()).unwrap(),
            frame.indices_source(0..frame.input_count()).unwrap(),
            frame.diagnostics_source().unwrap(),
            frame.points_source(0..frame.input_count()).unwrap(),
        ],
        frame.stamp,
    )
    .unwrap()
    .unwrap()
}
fn verify(frame: &ReadbackFrame, points: &[CloudPoint], spec: CloudGridSpec) {
    let cells = frame.channel::<CloudGridCell>(0).unwrap();
    let indices = frame.channel::<u32>(1).unwrap();
    let diagnostic = CloudGridDiagnostics::from_words(&frame.channel::<u32>(2).unwrap()).unwrap();
    let copied = frame.channel::<CloudPoint>(3).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&copied),
        bytemuck::cast_slice::<_, u8>(points)
    );
    let mut expected = vec![Vec::new(); spec.dimensions.iter().product::<u32>() as usize];
    let mut misses = 0;
    let mut invalid = 0;
    let mut outside = 0;
    for (i, p) in points.iter().enumerate() {
        if p.extra[2] & 1 == 0 {
            misses += 1;
            continue;
        }
        if p.position.iter().any(|v| !v.is_finite()) {
            invalid += 1;
            continue;
        }
        if (0..3).any(|j| p.position[j] < spec.minimum[j] || p.position[j] >= spec.maximum[j]) {
            outside += 1;
            continue;
        }
        let mut xyz = [0usize; 3];
        for j in 0..3 {
            let fraction = (p.position[j] as f64 - spec.minimum[j] as f64)
                / (spec.maximum[j] as f64 - spec.minimum[j] as f64);
            xyz[j] = ((fraction * spec.dimensions[j] as f64).floor() as usize)
                .min(spec.dimensions[j] as usize - 1);
        }
        let cell =
            (xyz[2] * spec.dimensions[1] as usize + xyz[1]) * spec.dimensions[0] as usize + xyz[0];
        expected[cell].push(i as u32);
    }
    let accepted = expected.iter().map(Vec::len).sum::<usize>();
    assert_eq!(
        diagnostic,
        CloudGridDiagnostics {
            accepted: accepted as u32,
            misses,
            invalid,
            outside,
            ..Default::default()
        }
    );
    assert_eq!(cells.len(), expected.len());
    assert_eq!(indices.len(), points.len());
    let mut start = 0;
    for (cell, expected) in cells.iter().zip(expected) {
        assert_eq!(cell.header, [start as u32, expected.len() as u32, 0, 0]);
        let mut actual = indices[start..start + expected.len()].to_vec();
        actual.sort_unstable();
        assert_eq!(actual, expected);
        start += expected.len();
        if expected.is_empty() {
            assert_eq!(cell.mean, [0.; 4]);
            assert_eq!(cell.minimum, [0.; 4]);
            assert_eq!(cell.maximum, [0.; 4]);
        } else {
            for j in 0..4 {
                let mean = expected
                    .iter()
                    .map(|&i| points[i as usize].position[j] as f64)
                    .sum::<f64>()
                    / expected.len() as f64;
                let min = expected
                    .iter()
                    .map(|&i| points[i as usize].position[j])
                    .fold(f32::INFINITY, f32::min);
                let max = expected
                    .iter()
                    .map(|&i| points[i as usize].position[j])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    (cell.mean[j] as f64 - mean).abs() < 2e-4 * mean.abs().max(1.0),
                    "{} != {mean}",
                    cell.mean[j]
                );
                assert_eq!(cell.minimum[j], min);
                assert_eq!(cell.maximum[j], max);
            }
        }
    }
    assert!(indices[accepted..].iter().all(|&i| i == u32::MAX));
}

fn point(i: usize) -> CloudPoint {
    let mut p = CloudPoint::new(Vec3::ZERO, 0xfedcba9876543210, i as u64);
    p.position = [
        (i % 31) as f32 / 16. - 0.9375,
        (i % 13) as f32 / 8. - 0.75,
        (i % 7) as f32 / 4. - 0.75,
        (i % 17) as f32 / 16.,
    ];
    p.grid[3] = i as u32;
    p
}

#[test]
#[ignore = "requires GPU"]
fn grid_matches_cpu_boundaries_rejections_and_warm_reuse() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut grid = grid(&engine, 65_537, 66049);
    let mut ring = ring(&engine, 65_537, 66049, 1);
    let mut calls = 0;
    for n in [0, 1, 255, 256, 257, 65_537] {
        let mut points: Vec<_> = (0..n).map(point).collect();
        if n > 7 {
            points[0].extra[2] = 0;
            points[1].position[0] = f32::NAN;
            points[2].position[3] = f32::INFINITY;
            points[3].position = [-1., -1., -1., 1.];
            points[4].position[0] = 1.;
            points[5].position[1] = -1.0001;
            points[6].normal[0] = f32::NAN;
            points[7].position = [0.; 4];
        }
        let input = source(&engine, points.clone());
        for dimensions in [[1, 1, 1], [8, 4, 2], [257, 257, 1]] {
            let spec = spec(dimensions);
            let frame = grid
                .process(
                    input.readback_source(0..n).unwrap(),
                    spec,
                    Default::default(),
                )
                .unwrap();
            assert!(frame.indices_source(0..n + 1).is_err());
            assert!(frame.cells_source(0..frame.cell_count() + 1).is_err());
            let ticket = enqueue(&frame, &mut ring);
            let read = ring.wait(&ticket, Duration::from_secs(20)).unwrap();
            verify(&read, &points, spec);
            calls += 1;
        }
    }
    assert_eq!(grid.stats().frames, calls);
    assert_eq!(grid.stats().submissions, calls);
    assert_eq!(grid.stats().kernel_dispatches, (calls - 3) * 5);
    assert_eq!(grid.stats().buffer_allocations, 8);
    assert_eq!(grid.stats().binding_preparations, 1);
}

#[test]
#[ignore = "requires GPU"]
fn grid_dense_million_points_keeps_every_member() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let n = 1_000_003;
    let points: Vec<_> = (0..n).map(point).collect();
    let input = source(&engine, points.clone());
    let mut grid = grid(&engine, n, 1);
    let mut ring = ring(&engine, n, 1, 1);
    let spec = spec([1; 3]);
    let frame = grid
        .process(
            input.readback_source(0..n).unwrap(),
            spec,
            Default::default(),
        )
        .unwrap();
    let ticket = enqueue(&frame, &mut ring);
    verify(
        &ring.wait(&ticket, Duration::from_secs(30)).unwrap(),
        &points,
        spec,
    );
}

#[test]
#[ignore = "requires GPU"]
fn grid_invalid_requests_do_not_submit() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut grid = grid(&engine, 2, 8);
    let input = source(&engine, vec![point(0); 3]);
    let wrong = source(&engine, vec![0u32]);
    let spec = spec([2; 3]);
    assert!(
        grid.process(
            wrong.readback_source(0..1).unwrap(),
            spec,
            Default::default()
        )
        .is_err()
    );
    assert!(
        grid.process(
            input.readback_source(0..3).unwrap(),
            spec,
            Default::default()
        )
        .is_err()
    );
    assert!(
        grid.process(
            input.readback_source(0..1).unwrap(),
            CloudGridSpec {
                dimensions: [3; 3],
                ..spec
            },
            Default::default()
        )
        .is_err()
    );
    for time in [f64::NAN, f64::INFINITY, -1.] {
        assert!(
            grid.process(
                input.readback_source(0..1).unwrap(),
                spec,
                ReadbackStamp {
                    simulation_time: Some(time),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    let foreign = Engine::new_gpu().unwrap();
    let other = source(&foreign, vec![point(0)]);
    assert!(
        grid.process(
            other.readback_source(0..1).unwrap(),
            spec,
            Default::default()
        )
        .is_err()
    );
    assert_eq!(grid.stats().submissions, 0);
}

#[test]
#[ignore = "requires GPU"]
fn grid_extreme_reductions_and_all_rejected_reset() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut points = vec![point(0); 23];
    for p in &mut points {
        p.position = [-0.5, 0., 0., f32::MAX];
    }
    let mut input = source(&engine, points);
    let mut grid = grid(&engine, 23, 2);
    let mut ring = ring(&engine, 23, 2, 1);
    let spec = spec([2, 1, 1]);
    let frame = grid
        .process(
            input.readback_source(0..23).unwrap(),
            spec,
            Default::default(),
        )
        .unwrap();
    let ticket = enqueue(&frame, &mut ring);
    let read = ring.wait(&ticket, Duration::from_secs(20)).unwrap();
    let cells = read.channel::<CloudGridCell>(0).unwrap();
    let diag = CloudGridDiagnostics::from_words(&read.channel::<u32>(2).unwrap()).unwrap();
    assert_eq!(diag.accepted, 23);
    assert_eq!(diag.index_overflow, 0);
    assert_eq!(diag.numeric_overflow_cells, cells[0].header[2]);
    assert_eq!(cells[0].header[..2], [0, 23]);
    assert!(cells[0].mean.iter().all(|v| v.is_finite()));
    if cells[0].header[2] == 0 {
        assert!((cells[0].mean[3] / f32::MAX - 1.).abs() < 1e-5);
    } else {
        assert_eq!(cells[0].header[2], 1);
        assert_eq!(cells[0].mean, [0.; 4]);
    }
    assert_eq!(cells[0].minimum, [-0.5, 0., 0., f32::MAX]);
    assert_eq!(cells[0].maximum, cells[0].minimum);
    for p in input.host_mut().unwrap() {
        p.extra[2] = 0;
    }
    input.upload().unwrap();
    let frame = grid
        .process(
            input.readback_source(0..23).unwrap(),
            spec,
            Default::default(),
        )
        .unwrap();
    let ticket = enqueue(&frame, &mut ring);
    verify(
        &ring.wait(&ticket, Duration::from_secs(20)).unwrap(),
        input.host().unwrap(),
        spec,
    );
}

#[test]
#[ignore = "requires GPU"]
fn grid_device_chain_snapshot_and_pending_copies_survive_reuse() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let points: Vec<_> = (0..257).map(point).collect();
    let input = source(&engine, points.clone());
    let mut cloud = CloudProcessor::new(
        &engine,
        CloudProcessorConfig {
            max_points: 257,
            max_jobs: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let frame = cloud
        .process(&[CloudJob {
            input: CloudInput::Points(input.readback_source(0..257).unwrap()),
            transform: Transform::IDENTITY,
            filter: CloudFilter {
                min_intensity: Some(0.5),
                ..Default::default()
            },
            stamp: Default::default(),
        }])
        .unwrap();
    let mut snapshots = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 257 * 112,
            max_total_bytes: 257 * 112,
        },
    )
    .unwrap();
    let snapshot = snapshots
        .try_capture(&[frame.points_source(0..257).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let channel = snapshot.channel::<CloudPoint>(0).unwrap();
    let mut grid = grid(&engine, 257, 64);
    let mut ring = ring(&engine, 257, 64, 3);
    let spec = spec([4; 3]);
    let stamp = ReadbackStamp {
        sensor_id: 9,
        sample_id: 7,
        scene_revision: 4,
        simulation_time: Some(1.25),
    };
    let frame = grid
        .process(channel.readback_source(0..257).unwrap(), spec, stamp)
        .unwrap();
    let first = enqueue(&frame, &mut ring);
    drop(channel);
    drop(snapshot);
    snapshots.reset().unwrap();
    let frame = grid
        .process(
            input.readback_source(17..257).unwrap(),
            spec,
            Default::default(),
        )
        .unwrap();
    let second = enqueue(&frame, &mut ring);
    let empty = source::<CloudPoint>(&engine, vec![]);
    let frame = grid
        .process(
            empty.readback_source(0..0).unwrap(),
            spec,
            Default::default(),
        )
        .unwrap();
    let third = enqueue(&frame, &mut ring);
    verify(
        &ring.wait(&third, Duration::from_secs(20)).unwrap(),
        &[],
        spec,
    );
    verify(
        &ring.wait(&second, Duration::from_secs(20)).unwrap(),
        &points[17..],
        spec,
    );
    let mut expected: Vec<_> = points
        .into_iter()
        .filter(|p| p.position[3] >= 0.5)
        .collect();
    expected.resize(257, CloudPoint::default());
    let read = ring.wait(&first, Duration::from_secs(20)).unwrap();
    assert_eq!(read.stamp, stamp);
    verify(&read, &expected, spec);
    assert_eq!(snapshots.available_slots().unwrap(), 1);
}
