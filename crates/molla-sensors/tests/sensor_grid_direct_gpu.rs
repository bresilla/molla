#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_sensors::{
    CloudGrid, CloudGridCell, CloudGridConfig, CloudGridDiagnostics, CloudGridSpec, CloudPoint,
    DeviceSnapshotConfig, DeviceSnapshotPool, PendingCloudGridInput, SensorChannelSpec,
    SensorCommandBatch, SensorFrame, SensorFramePool, SensorFramePoolConfig,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 128 << 20,
    }
}

fn grid(engine: &Engine, points: u32, cells: u32) -> CloudGrid {
    CloudGrid::new(
        engine,
        CloudGridConfig {
            max_points: points,
            max_cells: cells,
            max_buffer_bytes: 128 << 20,
        },
    )
    .unwrap()
}

fn spec(dimensions: [u32; 3]) -> CloudGridSpec {
    CloudGridSpec {
        minimum: [-1.0; 3],
        maximum: [1.0; 3],
        dimensions,
    }
}

fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX,
        scene_revision: u64::MAX - 1,
        simulation_time: Some(4.5),
    }
}

fn point(i: usize) -> CloudPoint {
    let mut p = CloudPoint::new(Vec3::ZERO, u64::MAX - 1, i as u64 | (1 << 40));
    p.position = [
        (i % 31) as f32 / 16.0 - 0.9375,
        (i % 13) as f32 / 8.0 - 0.75,
        (i % 7) as f32 / 4.0 - 0.75,
        (i % 17) as f32 / 16.0,
    ];
    p.grid = [0x80000001, 0xabcdef01, i as u32, i as u32];
    p.extra[3] = 0x12345678;
    p.angles[3] = f32::from_bits(0x7fc01234);
    p
}

fn source(engine: &Engine, points: Vec<CloudPoint>) -> DeviceBuffer<CloudPoint> {
    DeviceBuffer::from_host(engine, points, BufferUsage::StorageReadWrite).unwrap()
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

fn collect(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let p = frame.channel::<CloudPoint>(0).unwrap();
    let i = frame.channel::<u32>(1).unwrap();
    let c = frame.channel::<CloudGridCell>(2).unwrap();
    let d = frame.channel::<u32>(3).unwrap();
    let bytes = (p.len() as u64 * 116 + c.len() as u64 * 64 + 32).next_multiple_of(8);
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: bytes,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(
            &[
                p.readback_source(0..p.len()).unwrap(),
                i.readback_source(0..i.len()).unwrap(),
                c.readback_source(0..c.len()).unwrap(),
                d.readback_source(0..6).unwrap(),
            ],
            frame.stamp(),
        )
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn verify(cpu: &ReadbackFrame, points: &[CloudPoint], spec: CloudGridSpec) {
    let actual = cpu.channel::<CloudPoint>(0).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&actual),
        bytemuck::cast_slice::<_, u8>(points)
    );
    let indices = cpu.channel::<u32>(1).unwrap();
    let cells = cpu.channel::<CloudGridCell>(2).unwrap();
    let diagnostics = CloudGridDiagnostics::from_words(&cpu.channel::<u32>(3).unwrap()).unwrap();
    let mut expected = vec![Vec::<u32>::new(); spec.dimensions.iter().product::<u32>() as usize];
    let mut counts = CloudGridDiagnostics::default();
    for (i, point) in points.iter().enumerate() {
        if point.extra[2] & 1 == 0 {
            counts.misses += 1;
            continue;
        }
        if point.position.iter().any(|v| !v.is_finite()) {
            counts.invalid += 1;
            continue;
        }
        if (0..3)
            .any(|j| point.position[j] < spec.minimum[j] || point.position[j] >= spec.maximum[j])
        {
            counts.outside += 1;
            continue;
        }
        let xyz: [u32; 3] = std::array::from_fn(|j| {
            let fraction = (point.position[j] as f64 - spec.minimum[j] as f64)
                / (spec.maximum[j] as f64 - spec.minimum[j] as f64);
            ((fraction * spec.dimensions[j] as f64).floor() as u32).min(spec.dimensions[j] - 1)
        });
        let cell = (xyz[2] * spec.dimensions[1] + xyz[1]) * spec.dimensions[0] + xyz[0];
        expected[cell as usize].push(i as u32);
        counts.accepted += 1;
    }
    assert_eq!(diagnostics, counts);
    assert_eq!(cells.len(), expected.len());
    assert_eq!(indices.len(), points.len());
    let mut start = 0;
    for (cell, members) in cells.iter().zip(expected) {
        assert_eq!(cell.header, [start as u32, members.len() as u32, 0, 0]);
        let mut got = indices[start..start + members.len()].to_vec();
        got.sort_unstable();
        assert_eq!(got, members);
        start += members.len();
        if members.is_empty() {
            assert_eq!(cell.mean, [0.0; 4]);
            assert_eq!(cell.minimum, [0.0; 4]);
            assert_eq!(cell.maximum, [0.0; 4]);
        } else {
            for j in 0..4 {
                let mean = members
                    .iter()
                    .map(|&i| points[i as usize].position[j] as f64)
                    .sum::<f64>()
                    / members.len() as f64;
                let minimum = members
                    .iter()
                    .map(|&i| points[i as usize].position[j])
                    .fold(f32::INFINITY, f32::min);
                let maximum = members
                    .iter()
                    .map(|&i| points[i as usize].position[j])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!((cell.mean[j] as f64 - mean).abs() < 2e-4 * mean.abs().max(1.0));
                assert_eq!(cell.minimum[j], minimum);
                assert_eq!(cell.maximum[j], maximum);
            }
        }
    }
    assert!(indices[start..].iter().all(|&i| i == u32::MAX));
}

#[test]
#[ignore = "requires GPU"]
fn independent_grids_chain_typed_clouds_with_offsets_and_large_cell_dispatch() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut points: Vec<_> = (0..300).map(point).collect();
    points[5].extra[2] = 0;
    points[6].position[0] = f32::from_bits(0x7fc04321);
    points[7].position[3] = f32::INFINITY;
    points[8].position[0] = 1.0;
    points[9].position[1] = -1.01;
    points[10].position = [-1.0, -1.0, -1.0, 0.5];
    points[11].normal[0] = f32::NAN;
    let mut input = source(&engine, points.clone());
    input.mark_device_dirty();
    assert!(input.host().is_err());
    let mut a = grid(&engine, 257, 64);
    let mut b = grid(&engine, 257, 66049);
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let specs = [SensorChannelSpec::of::<CloudPoint>(260)];
    let mut shifted_pool = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(
            &mut batch,
            input.readback_source(5..262).unwrap(),
            spec([8, 4, 2]),
            &mut pool_a,
            stamp(u64::MAX),
        )
        .unwrap()
        .unwrap();
    let shifted = batch
        .encode_from_pending(
            &[&first.frame],
            &mut shifted_pool,
            &specs,
            std::slice::from_ref(&(3..260)),
            stamp(2),
            |encoder, inputs, outputs| {
                let input = inputs.channel::<CloudPoint>(0, 0)?;
                encoder.copy_buffer_to_buffer(
                    input.binding()?.buffer,
                    0,
                    &outputs[0],
                    3 * 112,
                    257 * 112,
                );
                Ok(0)
            },
        )
        .unwrap()
        .unwrap();
    let second = b
        .encode_pending(
            &mut batch,
            PendingCloudGridInput {
                frame: &shifted,
                channel: 0,
                points: 2..255,
            },
            spec([257, 257, 1]),
            &mut pool_b,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().input_upload_bytes, 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 3);
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 10);
    assert_eq!(first.input_count, 257);
    assert_eq!(second.input_count, 253);
    assert_eq!(second.cell_count, 66049);
    assert_eq!(a.stats().submissions, 0);
    assert_eq!(b.stats().submissions, 0);
    let fa = submitted.frame(&first.frame).unwrap().clone();
    let fb = submitted.frame(&second.frame).unwrap().clone();
    assert_eq!(fa.stamp().sensor_id, u64::MAX);
    drop(submitted);
    drop(a);
    drop(b);
    drop(pool_a);
    drop(pool_b);
    drop(shifted_pool);
    verify(&collect(&engine, &fa), &points[5..262], first.spec);
    verify(&collect(&engine, &fb), &points[7..260], second.spec);
}

#[test]
#[ignore = "requires GPU"]
fn empty_rejected_and_extreme_reductions_keep_distinct_fresh_channels() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut rejected = vec![point(0); 3];
    rejected[0].extra[2] = 0;
    rejected[0].position[0] = f32::NAN;
    rejected[1].position[3] = f32::INFINITY;
    rejected[2].position[0] = 1.0;
    let input = source(&engine, rejected.clone());
    let mut p = grid(&engine, 4, 8);
    let mut pool = p.output_pool(&engine, config(3)).unwrap();
    let mut held = Vec::new();
    for count in [3, 0] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let result = p
            .encode(
                &mut batch,
                input.readback_source(0..count).unwrap(),
                spec([2, 2, 2]),
                &mut pool,
                stamp(count as u64),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(
            submitted.stats.kernel_dispatches,
            if count == 0 { 0 } else { 5 }
        );
        held.push((
            submitted.frame(&result.frame).unwrap().clone(),
            result.spec,
            count,
        ));
    }
    let mut huge = point(0);
    huge.position = [0.0, 0.0, 0.0, f32::MAX];
    let overflow = source(&engine, vec![huge; 4]);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let result = p
        .encode(
            &mut batch,
            overflow.readback_source(0..4).unwrap(),
            spec([1, 1, 1]),
            &mut pool,
            stamp(9),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let cpu = collect(&engine, submitted.frame(&result.frame).unwrap());
    assert_eq!(
        CloudGridDiagnostics::from_words(&cpu.channel::<u32>(3).unwrap()).unwrap(),
        CloudGridDiagnostics {
            accepted: 4,
            numeric_overflow_cells: 0,
            ..Default::default()
        }
    );
    let cell = cpu.channel::<CloudGridCell>(2).unwrap()[0];
    assert_eq!(cell.header, [0, 4, 0, 0]);
    assert_eq!(cell.mean, huge.position);
    assert_eq!(cell.minimum, huge.position);
    assert_eq!(cell.maximum, huge.position);
    let mut indices = cpu.channel::<u32>(1).unwrap();
    indices.sort_unstable();
    assert_eq!(indices, [0, 1, 2, 3]);
    for (frame, spec, count) in held {
        verify(&collect(&engine, &frame), &rejected[..count], spec);
    }
}

#[test]
#[ignore = "requires GPU"]
fn warm_four_slot_grids_reuse_bindings_across_specs_and_snapshot_inputs() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let points: Vec<_> = (0..257).map(point).collect();
    let input = source(&engine, points.clone());
    let mut snapshots = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 32768,
            max_total_bytes: 32768,
        },
    )
    .unwrap();
    let snapshot = snapshots
        .try_capture(
            std::slice::from_ref(&(input.readback_source(0..257).unwrap())),
            stamp(1),
        )
        .unwrap()
        .unwrap();
    let mut p = grid(&engine, 257, 257);
    let mut pool = p.output_pool(&engine, config(4)).unwrap();
    let cases = [
        (1, [1, 1, 1]),
        (65, [8, 4, 2]),
        (257, [257, 1, 1]),
        (3, [2, 2, 2]),
    ];
    for round in 0..6 {
        let mut held = Vec::new();
        for &(count, dims) in &cases {
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let result = p
                .encode(
                    &mut batch,
                    snapshot
                        .channel::<CloudPoint>(0)
                        .unwrap()
                        .readback_source(0..count)
                        .unwrap(),
                    spec(dims),
                    &mut pool,
                    stamp(round),
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            held.push((submitted.frame(&result.frame).unwrap().clone(), result));
        }
        assert_eq!(p.stats().binding_preparations, 5);
        assert_eq!(pool.stats().buffer_allocations, 16);
        for (frame, result) in held {
            verify(
                &collect(&engine, &frame),
                &points[..result.input_count],
                result.spec,
            );
        }
        drain(&engine);
    }
    assert_eq!(p.stats().buffer_allocations, 8);
    assert_eq!(p.stats().encoded_frames, 24);
    assert_eq!(p.stats().submissions, 0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let result = p
        .encode(
            &mut batch,
            snapshot
                .channel::<CloudPoint>(0)
                .unwrap()
                .readback_source(0..257)
                .unwrap(),
            spec([1, 1, 1]),
            &mut pool,
            stamp(7),
        )
        .unwrap()
        .unwrap();
    snapshots.reset().unwrap();
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 0);
    let submitted = batch.submit().unwrap();
    drop(snapshot);
    verify(
        &collect(&engine, submitted.frame(&result.frame).unwrap()),
        &points,
        result.spec,
    );
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn cancellation_reset_backpressure_and_poisoned_pending_work_preserve_grid_versions() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let points: Vec<_> = (0..65).map(point).collect();
    let input = source(&engine, points.clone());
    let mut p = grid(&engine, 65, 8);
    let mut downstream = grid(&engine, 65, 8);
    let mut pool = p.output_pool(&engine, config(2)).unwrap();
    let mut downstream_pool = downstream.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    p.encode(
        &mut batch,
        input.readback_source(0..65).unwrap(),
        spec([1, 1, 1]),
        &mut pool,
        stamp(1),
    )
    .unwrap()
    .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 2);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = p
        .encode(
            &mut batch,
            input.readback_source(0..65).unwrap(),
            spec([2, 2, 2]),
            &mut pool,
            stamp(2),
        )
        .unwrap()
        .unwrap();
    pool.reset().unwrap();
    let submitted = batch.submit().unwrap();
    let old = submitted.frame(&first.frame).unwrap().clone();
    assert!(!pool.is_current(&old));
    drop(submitted);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let empty = p
        .encode(
            &mut batch,
            input.readback_source(0..0).unwrap(),
            spec([1, 1, 1]),
            &mut pool,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    let pending = downstream
        .encode_pending(
            &mut batch,
            PendingCloudGridInput {
                frame: &empty.frame,
                channel: 0,
                points: 0..0,
            },
            spec([2, 2, 2]),
            &mut downstream_pool,
            stamp(4),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().kernel_dispatches, 0);
    let submitted = batch.submit().unwrap();
    let empty_frame = submitted.frame(&empty.frame).unwrap().clone();
    verify(
        &collect(&engine, submitted.frame(&pending.frame).unwrap()),
        &[],
        pending.spec,
    );
    drop(submitted);
    drain(&engine);
    let before = p.stats();
    let mut blocked = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        p.encode(
            &mut blocked,
            input.readback_source(0..65).unwrap(),
            spec([1, 1, 1]),
            &mut pool,
            stamp(0)
        )
        .unwrap()
        .is_none()
    );
    assert!(blocked.submit().unwrap().index.is_none());
    assert_eq!(p.stats(), before);
    verify(&collect(&engine, &old), &points, first.spec);
    verify(&collect(&engine, &empty_frame), &[], empty.spec);
    drop(old);
    drop(empty_frame);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = p
        .encode(
            &mut batch,
            input.readback_source(0..65).unwrap(),
            spec([2, 2, 2]),
            &mut pool,
            stamp(5),
        )
        .unwrap()
        .unwrap();
    let final_grid = downstream
        .encode_pending(
            &mut batch,
            PendingCloudGridInput {
                frame: &first.frame,
                channel: 0,
                points: 0..65,
            },
            spec([1, 1, 1]),
            &mut downstream_pool,
            stamp(6),
        )
        .unwrap()
        .unwrap();
    assert!(
        batch
            .encode_with_pending(&[&final_grid.frame], |_, _| Err(molla_core::Error::Gpu(
                "injected downstream failure".into()
            )))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pool.available_slots().unwrap(), 2);
    assert_eq!(downstream_pool.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn invalid_pending_ranges_specs_sources_and_loss_fail_before_reservation() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let values: Vec<_> = (0..17).map(point).collect();
    let input = source(&engine, values.clone());
    let foreign_input = source(&foreign, vec![point(0)]);
    let wrong_type =
        DeviceBuffer::from_host(&engine, vec![0u32], BufferUsage::StorageReadWrite).unwrap();
    let mut a = grid(&engine, 17, 8);
    let mut b = grid(&engine, 16, 8);
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    assert!(b.output_pool(&foreign, config(1)).is_err());
    assert!(
        b.output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 116 * 16 + 64 * 8 + 24 - 1
            }
        )
        .is_err()
    );
    for case in 0..10 {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let first = a
            .encode(
                &mut batch,
                input.readback_source(0..17).unwrap(),
                spec([2, 2, 2]),
                &mut pool_a,
                stamp(1),
            )
            .unwrap()
            .unwrap();
        let mut source = PendingCloudGridInput {
            frame: &first.frame,
            channel: 0,
            points: 0..1,
        };
        let mut grid_spec = spec([2, 2, 2]);
        let mut stamp = stamp(2);
        match case {
            0 => source.points = 0..18,
            1 => source.points = 0..17,
            2 => source.channel = 1,
            3 => grid_spec.dimensions = [0, 1, 1],
            4 => grid_spec.dimensions = [u32::MAX; 3],
            5 => grid_spec.dimensions = [3, 3, 1],
            6 => grid_spec.maximum = [f32::NAN; 3],
            7 => {
                grid_spec.minimum = [0.0; 3];
                grid_spec.maximum = [f32::MIN_POSITIVE; 3];
            }
            8 => stamp.simulation_time = Some(f64::NAN),
            _ => stamp.simulation_time = Some(-1.0),
        }
        assert!(
            b.encode_pending(&mut batch, source, grid_spec, &mut pool_b, stamp)
                .is_err(),
            "case {case}"
        );
        assert_eq!(pool_b.stats().reserved_frames, 0);
        let submitted = batch.submit().unwrap();
        verify(
            &collect(&engine, submitted.frame(&first.frame).unwrap()),
            &values,
            first.spec,
        );
        drop(submitted);
        drain(&engine);
    }
    for source in [
        foreign_input.readback_source(0..1).unwrap(),
        wrong_type.readback_source(0..1).unwrap(),
    ] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            b.encode(&mut batch, source, spec([1, 1, 1]), &mut pool_b, stamp(0))
                .is_err()
        );
        assert!(batch.submit().unwrap().index.is_none());
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(
            &mut batch,
            input.readback_source(0..17).unwrap(),
            spec([1, 1, 1]),
            &mut pool_a,
            stamp(0),
        )
        .unwrap()
        .unwrap();
    for engine in [&engine, &foreign] {
        let mut wrong = SensorCommandBatch::new(engine).unwrap();
        assert!(
            b.encode_pending(
                &mut wrong,
                PendingCloudGridInput {
                    frame: &first.frame,
                    channel: 0,
                    points: 0..1
                },
                spec([1, 1, 1]),
                &mut pool_b,
                stamp(0)
            )
            .is_err()
        );
        assert!(wrong.submit().unwrap().index.is_none());
    }
    engine.destroy_device().unwrap();
    assert!(
        b.encode_pending(
            &mut batch,
            PendingCloudGridInput {
                frame: &first.frame,
                channel: 0,
                points: 0..1
            },
            spec([1, 1, 1]),
            &mut pool_b,
            stamp(0)
        )
        .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pool_b.stats().reserved_frames, 0);
    assert_eq!(b.stats().binding_preparations, 1);
    assert_eq!(b.stats().encoded_frames, 0);
}
