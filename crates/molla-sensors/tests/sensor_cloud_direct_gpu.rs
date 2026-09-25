#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::{
    CloudFilter, CloudGrid, CloudGridConfig, CloudGridSpec, CloudInput, CloudJob, CloudPoint,
    CloudProcessor, CloudProcessorConfig, CloudRaster, DeviceSnapshotConfig, DeviceSnapshotPool,
    EncodedSensorFrame, PendingCloudGridInput, PendingCloudInput, PendingCloudJob,
    PendingCloudSource, Raycaster, SensorChannelSpec, SensorCommandBatch, SensorFrame,
    SensorFramePool, SensorFramePoolConfig, SensorScene, SensorSceneOptions,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_sim::ModelBuilder;
use molla_storage::{BufferElement, BufferUsage};
use std::{ops::Range, sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 128 << 20,
    }
}

fn processor(engine: &Engine, n: u32, jobs: u32) -> CloudProcessor {
    CloudProcessor::new(
        engine,
        CloudProcessorConfig {
            max_points: n,
            max_jobs: jobs,
            max_buffer_bytes: 128 << 20,
        },
    )
    .unwrap()
}

fn source<T: BufferElement>(engine: &Engine, values: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, values, BufferUsage::StorageReadWrite).unwrap()
}

fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX - 3,
        scene_revision: 17,
        simulation_time: Some(2.5),
    }
}

fn point(i: usize) -> CloudPoint {
    let mut p = CloudPoint::new(
        Vec3::new(i as Real, 0.0, 0.0),
        u64::MAX,
        i as u64 | (1 << 40),
    );
    p.position[3] = (i % 4) as f32;
    p.extra[3] = 0x12345678;
    p.angles[3] = f32::from_bits(0x7fc01234);
    p
}

fn job(input: CloudInput<'_>) -> CloudJob<'_> {
    CloudJob {
        input,
        transform: Transform::IDENTITY,
        filter: CloudFilter::default(),
        stamp: stamp(7),
    }
}

fn pending(frame: &EncodedSensorFrame, range: Range<usize>) -> PendingCloudJob<'_> {
    PendingCloudJob {
        input: PendingCloudInput::Points(PendingCloudSource {
            frame,
            channel: 0,
            elements: range,
        }),
        transform: Transform::IDENTITY,
        filter: CloudFilter::default(),
        stamp: stamp(8),
    }
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
    let c = frame.channel::<u32>(1).unwrap();
    let bytes = (p.len() as u64 * 112 + c.len() as u64 * 4 + 16).next_multiple_of(8);
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
                c.readback_source(0..c.len()).unwrap(),
            ],
            frame.stamp(),
        )
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn assert_points(actual: &[CloudPoint], expected: &[CloudPoint]) {
    assert!(actual.len() >= expected.len());
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&actual[..expected.len()]),
        bytemuck::cast_slice::<_, u8>(expected)
    );
    assert!(
        actual[expected.len()..]
            .iter()
            .all(|p| bytemuck::bytes_of(p).iter().all(|b| *b == 0))
    );
}

#[test]
#[ignore = "requires GPU"]
fn mixed_external_clouds_and_rays_chain_compaction_and_grid_in_one_submission() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut values: Vec<_> = (0..270).map(point).collect();
    values[7].extra[2] = 0;
    values[8].normal[0] = f32::NAN;
    let input = source(&engine, values.clone());
    let rp = source(&engine, vec![Vec3::new(400.0, 0.0, 0.0); 4]);
    let rn = source(&engine, vec![Vec3::Y; 4]);
    let rd = source(&engine, vec![Vec3::NEG_Y; 4]);
    let rt = source(&engine, vec![3.0 as Real; 4]);
    let ri = source(&engine, vec![-1i32, 13, 14, 15]);
    let mut jobs = [
        job(CloudInput::Points(input.readback_source(5..262).unwrap())),
        job(CloudInput::Rays {
            channels: [
                rp.readback_source(1..4).unwrap(),
                rn.readback_source(1..4).unwrap(),
                rd.readback_source(1..4).unwrap(),
                rt.readback_source(1..4).unwrap(),
                ri.readback_source(1..4).unwrap(),
            ],
            raster: CloudRaster {
                sensor_id: 7,
                rows: 1,
                columns: 3,
                subrays: 1,
                source_offset: 23,
                angles: Some([(0.25, 0.25), (0.0, 1.0)]),
            },
        }),
    ];
    jobs[0].filter.min_intensity = Some(2.0);
    jobs[0].transform.position = Vec3::X;
    let mut expected: Vec<_> = values[5..262]
        .iter()
        .copied()
        .filter(|p| p.extra[2] & 1 != 0 && p.normal[0].is_finite() && p.position[3] >= 2.0)
        .map(|mut p| {
            p.position[0] += 1.0;
            p
        })
        .collect();
    let kept = expected.len();
    for i in 0..3 {
        expected.push(CloudPoint {
            position: [400.0, 0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0, 3.0],
            direction: [0.0, -1.0, 0.0, 0.0],
            ids: [7, 0, (u64::MAX - 3) as u32, u32::MAX],
            grid: [0, i, 0, 23 + i],
            extra: [13 + i, u32::MAX, 9, 0],
            angles: [0.25, i as f32 * 0.5, 0.0, 0.0],
        });
    }
    let mut a = processor(&engine, 260, 2);
    let mut b = processor(&engine, 260, 1);
    let mut pa = a.output_pool(&engine, config(1)).unwrap();
    let mut pb = b.output_pool(&engine, config(1)).unwrap();
    let mut grid = CloudGrid::new(
        &engine,
        CloudGridConfig {
            max_points: 260,
            max_cells: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut pg = grid.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pa, stamp(1))
        .unwrap()
        .unwrap();
    let second = b
        .encode_pending(
            &mut batch,
            &[pending(&first.frame, 0..260)],
            &mut pb,
            stamp(2),
        )
        .unwrap()
        .unwrap();
    let gr = grid
        .encode_pending(
            &mut batch,
            PendingCloudGridInput {
                frame: &second.frame,
                channel: 0,
                points: 0..260,
            },
            CloudGridSpec {
                minimum: [-1.0; 3],
                maximum: [500.0; 3],
                dimensions: [1; 3],
            },
            &mut pg,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().input_upload_bytes, 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 11);
    assert_eq!(submitted.stats.encoded_frames, 3);
    assert_eq!(first.layouts[1].input_range, 257..260);
    let fa = submitted.frame(&first.frame).unwrap().clone();
    let fb = submitted.frame(&second.frame).unwrap().clone();
    let fg = submitted.frame(&gr.frame).unwrap().clone();
    assert_eq!(a.stats().submissions, 0);
    assert_eq!(b.stats().submissions, 0);
    drop((submitted, a, b, grid, pa, pb, pg));
    let ca = collect(&engine, &fa);
    let cb = collect(&engine, &fb);
    assert_points(&ca.channel::<CloudPoint>(0).unwrap(), &expected);
    assert_points(&cb.channel::<CloudPoint>(0).unwrap(), &expected);
    assert_eq!(
        ca.channel::<u32>(1).unwrap(),
        [
            expected.len() as u32,
            0,
            kept as u32,
            1,
            1,
            255 - kept as u32,
            kept as u32,
            3,
            0,
            0,
            0
        ]
    );
    assert_eq!(
        cb.channel::<u32>(1).unwrap(),
        [
            expected.len() as u32,
            0,
            expected.len() as u32,
            260 - expected.len() as u32,
            0,
            0
        ]
    );
    let diagnostics = fg.channel::<u32>(3).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 24,
            max_total_bytes: 24,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(&[diagnostics.readback_source(0..6).unwrap()], stamp(3))
        .unwrap()
        .unwrap();
    let d = ring
        .wait(&ticket, Duration::from_secs(30))
        .unwrap()
        .channel::<u32>(0)
        .unwrap();
    assert_eq!(d[0], expected.len() as u32);
    assert_eq!(d[1], 260 - expected.len() as u32);
}

#[test]
#[ignore = "requires GPU"]
fn pending_scene_rays_construct_clouds_without_intermediate_collection() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_triangle(
            BodyId::NONE,
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, -2.0),
            Vec3::new(0.0, 0.0, 2.0),
            Transform::IDENTITY,
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut rays = Raycaster::new(&engine, 4).unwrap();
    let mut pr = rays.output_pool(&engine, config(1)).unwrap();
    let mut cloud = processor(&engine, 3, 1);
    let mut pc = cloud.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let r = rays
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &[
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, 3.0, 0.0),
                Vec3::new(10.0, 3.0, 0.0),
                Vec3::new(0.0, 4.0, 0.0),
            ],
            &[Vec3::NEG_Y; 4],
            20.0,
            &mut pr,
            stamp(u64::MAX),
        )
        .unwrap()
        .unwrap();
    let jobs = [PendingCloudJob {
        input: PendingCloudInput::Rays {
            channels: [4, 5, 1, 2, 3].map(|channel| PendingCloudSource {
                frame: &r.frame,
                channel,
                elements: 1..4,
            }),
            raster: CloudRaster {
                sensor_id: u64::MAX,
                rows: 1,
                columns: 3,
                subrays: 1,
                source_offset: 1,
                angles: None,
            },
        },
        transform: Transform::new(Vec3::X, Quat::IDENTITY),
        filter: CloudFilter::default(),
        stamp: stamp(u64::MAX),
    }];
    let c = cloud
        .encode_pending(&mut batch, &jobs, &mut pc, stamp(1))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    let cpu = collect(&engine, submitted.frame(&c.frame).unwrap());
    assert_eq!(cpu.channel::<u32>(1).unwrap(), [2, 0, 2, 1, 0, 0]);
    let p = cpu.channel::<CloudPoint>(0).unwrap();
    for (i, distance) in [3.0, 4.0].into_iter().enumerate() {
        assert_eq!(p[i].position, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p[i].normal, [0.0, 1.0, 0.0, distance]);
        assert_eq!(p[i].direction, [0.0, -1.0, 0.0, 0.0]);
        assert_eq!(p[i].sensor_id(), u64::MAX);
        assert_eq!(p[i].sample_id(), u64::MAX - 3);
        assert_eq!(p[i].grid, [0, i as u32 * 2, 0, 1 + i as u32 * 2]);
    }
    assert_eq!(p[2], CloudPoint::default());
}

#[test]
#[ignore = "requires GPU"]
fn deduplicated_pending_jobs_preserve_1024_jobs_and_relative_active_ranges() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let values: Vec<_> = (0..5).map(point).collect();
    let input = source(&engine, values.clone());
    let jobs = [job(CloudInput::Points(
        input.readback_source(0..5).unwrap(),
    ))];
    let mut a = processor(&engine, 5, 1);
    let mut b = processor(&engine, 1024, 1024);
    let mut pa = a.output_pool(&engine, config(1)).unwrap();
    let mut pb = b.output_pool(&engine, config(1)).unwrap();
    let specs = [SensorChannelSpec::of::<CloudPoint>(8)];
    let mut shifted_pool = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pa, stamp(0))
        .unwrap()
        .unwrap();
    let shifted = batch
        .encode_from_pending(
            &[&first.frame],
            &mut shifted_pool,
            &specs,
            &[3..8],
            stamp(0),
            |encoder, inputs, out| {
                encoder.copy_buffer_to_buffer(
                    inputs.channel::<CloudPoint>(0, 0)?.binding()?.buffer,
                    0,
                    &out[0],
                    3 * 112,
                    5 * 112,
                );
                Ok(0)
            },
        )
        .unwrap()
        .unwrap();
    let clones = [shifted.clone(), shifted.clone()];
    let pending_jobs: Vec<_> = (0..1024)
        .map(|i| pending(&clones[i % 2], i % 5..i % 5 + i % 2))
        .collect();
    let last = b
        .encode_pending(&mut batch, &pending_jobs, &mut pb, stamp(1))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 3);
    assert_eq!(submitted.stats.kernel_dispatches, 6);
    let cpu = collect(&engine, submitted.frame(&last.frame).unwrap());
    let expected: Vec<_> = (0..1024)
        .filter(|i| i % 2 == 1)
        .map(|i| values[i % 5])
        .collect();
    assert_points(&cpu.channel::<CloudPoint>(0).unwrap(), &expected);
    let counts = cpu.channel::<u32>(1).unwrap();
    assert_eq!(counts[0], 512);
    for i in 0..1024 {
        assert_eq!(
            &counts[1 + 5 * i..6 + 5 * i],
            &[(i / 2) as u32, (i % 2) as u32, 0, 0, 0]
        );
        assert_eq!(last.layouts[i].input_range, i / 2..i / 2 + i % 2);
    }
}

#[test]
#[ignore = "requires GPU"]
fn warm_slots_cancellation_reset_backpressure_and_zero_tails_are_independent() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let values: Vec<_> = (0..257).map(point).collect();
    let input = source(&engine, values.clone());
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
        .try_capture(&[input.readback_source(0..257).unwrap()], stamp(0))
        .unwrap()
        .unwrap();
    let input = snapshot.channel::<CloudPoint>(0).unwrap();
    let mut p = processor(&engine, 257, 1);
    let mut pool = p.output_pool(&engine, config(4)).unwrap();
    for round in 0..6 {
        let mut frames = Vec::new();
        for n in [1, 65, 257, 3] {
            let mut jobs = [job(CloudInput::Points(
                input.readback_source(0..n).unwrap(),
            ))];
            jobs[0].transform.position = Vec3::X * round as Real;
            jobs[0].filter.min_intensity = Some((round % 5) as f32);
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let out = p
                .encode(&mut batch, &jobs, &mut pool, stamp(round as u64))
                .unwrap()
                .unwrap();
            frames.push(batch.submit().unwrap().frame(&out.frame).unwrap().clone());
        }
        assert_eq!(pool.available_slots().unwrap(), 0);
        assert!(frames.iter().all(|frame| pool.is_current(frame)));
        pool.reset().unwrap();
        assert!(frames.iter().all(|frame| !pool.is_current(frame)));
        let jobs = [job(CloudInput::Points(
            input.readback_source(0..1).unwrap(),
        ))];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            p.encode(&mut batch, &jobs, &mut pool, stamp(0))
                .unwrap()
                .is_none()
        );
        assert!(batch.submit().unwrap().index.is_none());
        for (frame, n) in frames.iter().zip([1, 65, 257, 3]) {
            let expected: Vec<_> = values[..n]
                .iter()
                .copied()
                .filter(|p| p.position[3] >= (round % 5) as f32)
                .map(|mut p| {
                    p.position[0] += round as f32;
                    p
                })
                .collect();
            let cpu = collect(&engine, frame);
            assert_points(&cpu.channel::<CloudPoint>(0).unwrap(), &expected);
            assert_eq!(
                cpu.channel::<u32>(1).unwrap(),
                [
                    expected.len() as u32,
                    0,
                    expected.len() as u32,
                    0,
                    0,
                    (n - expected.len()) as u32
                ]
            );
        }
        drop(frames);
        drain(&engine);
    }
    assert_eq!(p.stats().binding_preparations, 5);
    assert_eq!(p.stats().buffer_allocations, 8);
    assert_eq!(pool.stats().buffer_allocations, 8);
    let jobs = [job(CloudInput::Points(
        input.readback_source(0..257).unwrap(),
    ))];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let canceled = p
        .encode(&mut batch, &jobs, &mut pool, stamp(0))
        .unwrap()
        .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 4);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let empty = p
        .encode_pending(&mut batch, &[], &mut pool, stamp(1))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert!(submitted.frame(&canceled.frame).is_err());
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    let cpu = collect(&engine, submitted.frame(&empty.frame).unwrap());
    assert!(cpu.channel::<CloudPoint>(0).unwrap().is_empty());
    assert_eq!(cpu.channel::<u32>(1).unwrap(), [0]);
    drop(submitted);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = p
        .encode(&mut batch, &jobs, &mut pool, stamp(2))
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
    assert_eq!(pool.available_slots().unwrap(), 4);
    snapshots.reset().unwrap();
    assert_eq!(snapshots.available_slots().unwrap(), 0);
    drop(jobs);
    drop(snapshot);
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn malformed_pending_metadata_and_foreign_resources_fail_before_reservation() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let input = source(&engine, (0..17).map(point).collect());
    let foreign_input = source(&foreign, vec![point(0)]);
    let wrong = source(&engine, vec![0u32]);
    let mut a = processor(&engine, 17, 1);
    let mut b = processor(&engine, 16, 1);
    let mut pa = a.output_pool(&engine, config(1)).unwrap();
    let mut pb = b.output_pool(&engine, config(1)).unwrap();
    let mut foreign_pool = processor(&foreign, 16, 1)
        .output_pool(&foreign, config(1))
        .unwrap();
    assert!(b.output_pool(&foreign, config(1)).is_err());
    assert!(
        b.output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 2,
                max_total_bytes: 2 * (16 * 112 + 24) - 1
            }
        )
        .is_err()
    );
    assert!(
        b.output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 2,
                max_total_bytes: 2 * (16 * 112 + 24)
            }
        )
        .is_ok()
    );
    for case in 0..12 {
        let jobs = [job(CloudInput::Points(
            input.readback_source(0..17).unwrap(),
        ))];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let first = a
            .encode(&mut batch, &jobs, &mut pa, stamp(0))
            .unwrap()
            .unwrap();
        let mut j = pending(&first.frame, 0..1);
        let mut output_stamp = stamp(1);
        match case {
            0 => {
                if let PendingCloudInput::Points(source) = &mut j.input {
                    source.elements = 0..18;
                }
            }
            1 => {
                if let PendingCloudInput::Points(source) = &mut j.input {
                    source.elements = 0..17;
                }
            }
            2 => {
                if let PendingCloudInput::Points(source) = &mut j.input {
                    source.channel = 1;
                }
            }
            3 => {
                if let PendingCloudInput::Points(source) = &mut j.input {
                    source.elements = Range { start: 1, end: 0 };
                }
            }
            4 => j.transform.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            5 => j.transform.position = Vec3::splat(Real::NAN),
            6 => j.filter.distance = Some((Vec3::ZERO, -1.0)),
            7 => j.filter.bounds = Some((Vec3::ONE, Vec3::ZERO, true)),
            8 => j.filter.min_intensity = Some(f32::INFINITY),
            9 => j.stamp.simulation_time = Some(-1.0),
            10 => output_stamp.simulation_time = Some(Real::NAN),
            _ => {}
        }
        let target = if case == 11 {
            &mut foreign_pool
        } else {
            &mut pb
        };
        assert!(
            b.encode_pending(&mut batch, &[j], target, output_stamp)
                .is_err(),
            "case {case}"
        );
        assert_eq!(pb.stats().reserved_frames, 0);
        assert_eq!(foreign_pool.stats().reserved_frames, 0);
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.encoded_frames, 1);
        assert_points(
            &collect(&engine, submitted.frame(&first.frame).unwrap())
                .channel::<CloudPoint>(0)
                .unwrap(),
            &(0..17).map(point).collect::<Vec<_>>(),
        );
        drop(submitted);
        drain(&engine);
    }
    for source in [
        foreign_input.readback_source(0..1).unwrap(),
        wrong.readback_source(0..1).unwrap(),
    ] {
        let jobs = [job(CloudInput::Points(source))];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(b.encode(&mut batch, &jobs, &mut pb, stamp(0)).is_err());
        assert!(batch.submit().unwrap().index.is_none());
    }
    let jobs = [job(CloudInput::Points(
        input.readback_source(0..1).unwrap(),
    ))];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pa, stamp(0))
        .unwrap()
        .unwrap();
    for other_engine in [&engine, &foreign] {
        let mut other = SensorCommandBatch::new(other_engine).unwrap();
        assert!(
            b.encode_pending(
                &mut other,
                &[pending(&first.frame, 0..1)],
                &mut pb,
                stamp(1)
            )
            .is_err()
        );
        assert!(other.submit().unwrap().index.is_none());
    }
    engine.destroy_device().unwrap();
    assert!(
        b.encode_pending(
            &mut batch,
            &[pending(&first.frame, 0..1)],
            &mut pb,
            stamp(1)
        )
        .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pb.stats().reserved_frames, 0);
    assert_eq!(b.stats().binding_preparations, 1);
}

#[test]
#[ignore = "requires GPU"]
fn external_input_bound_counts_each_ray_channel_and_empty_jobs_have_fresh_counts() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![point(0)]);
    let vector = source(&engine, vec![Vec3::Y]);
    let distance = source(&engine, vec![1.0 as Real]);
    let shape = source(&engine, vec![0i32]);
    let mut p = processor(&engine, 1, 65);
    let mut pool = p.output_pool(&engine, config(1)).unwrap();
    let mut jobs: Vec<_> = (0..60)
        .map(|_| job(CloudInput::Points(input.readback_source(0..0).unwrap())))
        .collect();
    jobs.push(job(CloudInput::Rays {
        channels: [
            vector.readback_source(0..1).unwrap(),
            vector.readback_source(0..1).unwrap(),
            vector.readback_source(0..1).unwrap(),
            distance.readback_source(0..1).unwrap(),
            shape.readback_source(0..1).unwrap(),
        ],
        raster: CloudRaster {
            sensor_id: 7,
            rows: 1,
            columns: 1,
            subrays: 1,
            source_offset: 0,
            angles: None,
        },
    }));
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(p.encode(&mut batch, &jobs, &mut pool, stamp(0)).is_err());
    assert!(batch.submit().unwrap().index.is_none());
    assert_eq!(pool.stats().reserved_frames, 0);
    jobs.remove(0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = p
        .encode(&mut batch, &jobs, &mut pool, stamp(1))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let cpu = collect(&engine, submitted.frame(&out.frame).unwrap());
    let counts = cpu.channel::<u32>(1).unwrap();
    assert_eq!(counts[0], 1);
    assert!(counts[1..1 + 59 * 5].iter().all(|v| *v == 0));
    assert_eq!(&counts[1 + 59 * 5..], &[0, 1, 0, 0, 0]);
    drop(submitted);
    drain(&engine);
    let jobs: Vec<_> = (0..64)
        .map(|_| job(CloudInput::Points(input.readback_source(0..0).unwrap())))
        .collect();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = p
        .encode(&mut batch, &jobs, &mut pool, stamp(2))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    let cpu = collect(&engine, submitted.frame(&out.frame).unwrap());
    assert!(cpu.channel::<CloudPoint>(0).unwrap().is_empty());
    assert_eq!(cpu.channel::<u32>(1).unwrap(), vec![0; 321]);
}
