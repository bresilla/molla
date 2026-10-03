#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_sensors::readback::{
    DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp,
};
use molla_sensors::{
    DeviceSnapshotConfig, DeviceSnapshotPool, ImageJob, ImageOp, ImageProcessor,
    ImageProcessorConfig, PendingImageJob, ProcessedImages, SensorChannelSpec, SensorCommandBatch,
    SensorFrame, SensorFramePool, SensorFramePoolConfig,
};
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires GPU"]
fn pending_image_ranges_are_relative_to_nonzero_active_offsets() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let pixels = [0x10102030, 0x80402010, 0xFF102040];
    let input = source(&engine, pixels.to_vec());
    let jobs = [ImageJob {
        source: input.readback_source(0..3).unwrap(),
        operations: &[],
        stamp: stamp(1),
    }];
    let mut a = processor(&engine, 3, 1);
    let mut b = processor(&engine, 2, 1);
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let specs = [SensorChannelSpec::of::<u32>(8)];
    let mut middle_pool = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pool_a, stamp(1))
        .unwrap()
        .unwrap();
    let shifted = batch
        .encode_from_pending(
            &[&first.frame],
            &mut middle_pool,
            &specs,
            std::slice::from_ref(&(3..6)),
            stamp(2),
            |encoder, inputs, outputs| {
                let channel = inputs.channel::<u32>(0, 0)?;
                encoder.copy_buffer_to_buffer(channel.binding()?.buffer, 0, &outputs[0], 12, 12);
                Ok(0)
            },
        )
        .unwrap()
        .unwrap();
    let jobs = [PendingImageJob {
        frame: &shifted,
        channel: 0,
        pixels: 1..3,
        operations: &[ImageOp::Exposure(2.0)],
        stamp: stamp(3),
    }];
    let result = b
        .encode_pending(&mut batch, &jobs, &mut pool_b, stamp(3))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 1);
    assert_eq!(
        collect(&engine, submitted.frame(&result.frame).unwrap())
            .channel::<u32>(0)
            .unwrap(),
        [0x80804020, 0xFF204080]
    );
}

fn processor(engine: &Engine, pixels: u32, images: u32) -> ImageProcessor {
    ImageProcessor::new(
        engine,
        ImageProcessorConfig {
            max_pixels: pixels,
            max_images: images,
            max_operations_per_image: 16,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap()
}

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 22,
    }
}

fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX - id,
        scene_revision: 31,
        simulation_time: Some(7.25),
    }
}

fn source(engine: &Engine, pixels: Vec<u32>) -> DeviceBuffer<u32> {
    DeviceBuffer::from_host(engine, pixels, BufferUsage::StorageReadWrite).unwrap()
}

fn ring(engine: &Engine) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1 << 20,
            max_total_bytes: 1 << 20,
        },
    )
    .unwrap()
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
    let pixels = frame.channel::<u32>(0).unwrap();
    let gains = frame.channel::<Vec3>(1).unwrap();
    let mut ring = ring(engine);
    let ticket = ring
        .try_enqueue(
            &[
                pixels.readback_source(0..pixels.len()).unwrap(),
                gains.readback_source(0..gains.len()).unwrap(),
            ],
            frame.stamp(),
        )
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn collect_legacy(
    engine: &Engine,
    images: &ProcessedImages<'_>,
) -> Vec<(Vec<u32>, Vec<(usize, Vec3)>)> {
    let mut ring = ring(engine);
    images
        .layouts()
        .iter()
        .map(|layout| {
            let image = images.image(layout.stamp.sensor_id).unwrap();
            let mut sources = vec![image.readback_source().unwrap()];
            for &stage in &layout.white_balance_stages {
                sources.push(image.white_balance_source(stage).unwrap());
            }
            let ticket = ring.try_enqueue(&sources, layout.stamp).unwrap().unwrap();
            let cpu = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
            (
                cpu.channel::<u32>(0).unwrap(),
                layout
                    .white_balance_stages
                    .iter()
                    .enumerate()
                    .map(|(i, &stage)| (stage, cpu.channel::<Vec3>(i + 1).unwrap()[0]))
                    .collect(),
            )
        })
        .collect()
}

#[test]
#[ignore = "requires GPU"]
fn independent_image_pipelines_chain_before_submission_with_legacy_pixel_and_gain_parity() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let colors: Vec<_> = (0..520u32)
        .map(|i| {
            ((i * 13) & 255)
                | (((i * 29 + 17) & 255) << 8)
                | (((i * 7 + 43) & 255) << 16)
                | ((i & 255) << 24)
        })
        .collect();
    let mut input = source(&engine, colors);
    input.mark_device_dirty();
    assert!(input.host().is_err());
    let operations = [
        ImageOp::Exposure(0.7),
        ImageOp::Gamma(2.2),
        ImageOp::LinearToSrgb,
        ImageOp::SrgbToLinear,
        ImageOp::Reinhard,
        ImageOp::ReinhardWhitePoint(0.8),
        ImageOp::BrightnessContrast {
            brightness: -0.1,
            contrast: 1.3,
        },
        ImageOp::Saturation(0.4),
        ImageOp::GrayWorld,
        ImageOp::WhitePatch,
        ImageOp::GrayWorld,
    ];
    let counts = [65, 257, 1, 513, 257, 65, 257, 1, 513, 257, 0];
    let jobs: Vec<_> = operations
        .iter()
        .enumerate()
        .map(|(i, op)| ImageJob {
            source: input.readback_source(3..3 + counts[i]).unwrap(),
            operations: std::slice::from_ref(op),
            stamp: stamp(u64::MAX - i as u64),
        })
        .collect();
    let second_ops = [
        ImageOp::Exposure(0.7),
        ImageOp::GrayWorld,
        ImageOp::Gamma(1.7),
        ImageOp::WhitePatch,
    ];
    let mut legacy_a = processor(&engine, 4096, 16);
    let mut legacy_b = processor(&engine, 4096, 16);
    let old_a = legacy_a.process(&jobs).unwrap();
    let expected_a = collect_legacy(&engine, &old_a);
    let second_jobs: Vec<_> = old_a
        .layouts()
        .iter()
        .map(|layout| ImageJob {
            source: old_a
                .image(layout.stamp.sensor_id)
                .unwrap()
                .readback_source()
                .unwrap(),
            operations: &second_ops,
            stamp: layout.stamp,
        })
        .collect();
    let old_b = legacy_b.process(&second_jobs).unwrap();
    let expected_b = collect_legacy(&engine, &old_b);
    let mut a = processor(&engine, 4096, 16);
    let mut b = processor(&engine, 4096, 16);
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pool_a, stamp(11))
        .unwrap()
        .unwrap();
    let pending: Vec<_> = first
        .layouts
        .iter()
        .map(|layout| PendingImageJob {
            frame: &first.frame,
            channel: 0,
            pixels: layout.pixels.clone(),
            operations: &second_ops,
            stamp: layout.stamp,
        })
        .collect();
    let second = b
        .encode_pending(&mut batch, &pending, &mut pool_b, stamp(12))
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().input_upload_bytes, 0);
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().downstream_encodes, 1);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.encoded_frames, 2);
    assert_eq!(
        submitted.stats.kernel_dispatches,
        a.stats().kernel_dispatches + b.stats().kernel_dispatches
    );
    assert_eq!(a.stats().submissions, 0);
    assert_eq!(b.stats().submissions, 0);
    assert_eq!(
        a.stats().copied_bytes,
        counts.iter().sum::<usize>() as u64 * 4
    );
    for (result, expected) in [(&first, &expected_a), (&second, &expected_b)] {
        let cpu = collect(&engine, submitted.frame(&result.frame).unwrap());
        let pixels = cpu.channel::<u32>(0).unwrap();
        let gains = cpu.channel::<Vec3>(1).unwrap();
        for (layout, (expected_pixels, expected_gains)) in result.layouts.iter().zip(expected) {
            assert_eq!(&pixels[layout.pixels.clone()], expected_pixels);
            for (stage, gain) in expected_gains {
                assert_eq!(gains[layout.stages.start + stage], *gain);
            }
            for stage in 0..layout.stages.len() {
                if !layout.white_balance_stages.contains(&stage) {
                    assert_eq!(gains[layout.stages.start + stage], Vec3::ZERO);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn cancelled_uniform_writes_backpressure_reset_and_empty_jobs_preserve_output_versions() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![0x80402010; 257]);
    let mut p = processor(&engine, 257, 2);
    let mut pool = p.output_pool(&engine, config(2)).unwrap();
    let cancelled = [ImageJob {
        source: input.readback_source(0..257).unwrap(),
        operations: &[ImageOp::Exposure(20.0)],
        stamp: stamp(1),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    p.encode(&mut batch, &cancelled, &mut pool, stamp(0))
        .unwrap()
        .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 2);
    let jobs = [ImageJob {
        source: input.readback_source(0..257).unwrap(),
        operations: &[ImageOp::Exposure(2.0)],
        stamp: stamp(1),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = p
        .encode(&mut batch, &jobs, &mut pool, stamp(0))
        .unwrap()
        .unwrap();
    pool.reset().unwrap();
    let submitted = batch.submit().unwrap();
    let old = submitted.frame(&first.frame).unwrap().clone();
    assert!(!pool.is_current(&old));
    drop(submitted);
    let empty_jobs = [ImageJob {
        source: input.readback_source(0..0).unwrap(),
        operations: &[ImageOp::GrayWorld, ImageOp::WhitePatch],
        stamp: stamp(2),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let second = p
        .encode(&mut batch, &empty_jobs, &mut pool, stamp(3))
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().kernel_dispatches, 2);
    let submitted = batch.submit().unwrap();
    let empty = submitted.frame(&second.frame).unwrap().clone();
    drop(submitted);
    let stats = p.stats();
    let mut blocked = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        p.encode(&mut blocked, &jobs, &mut pool, stamp(4))
            .unwrap()
            .is_none()
    );
    assert!(blocked.submit().unwrap().index.is_none());
    assert_eq!(p.stats().encoded_frames, stats.encoded_frames);
    assert_eq!(p.stats().binding_preparations, stats.binding_preparations);
    assert_eq!(p.stats().copied_bytes, stats.copied_bytes);
    assert_eq!(
        collect(&engine, &old).channel::<u32>(0).unwrap(),
        vec![0x80804020; 257]
    );
    let cpu = collect(&engine, &empty);
    assert!(cpu.channel::<u32>(0).unwrap().is_empty());
    assert_eq!(
        cpu.channel::<Vec3>(1).unwrap(),
        vec![Vec3::ONE, Vec3::splat(1e6)]
    );
    drop(old);
    drop(empty);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let none = p
        .encode_pending(&mut batch, &[], &mut pool, stamp(5))
        .unwrap()
        .unwrap();
    assert!(none.layouts.is_empty());
    assert_eq!(batch.stats().kernel_dispatches, 0);
    let submitted = batch.submit().unwrap();
    let frame = submitted.frame(&none.frame).unwrap().clone();
    drop(submitted);
    drop(pool);
    drop(p);
    let cpu = collect(&engine, &frame);
    assert!(cpu.channel::<u32>(0).unwrap().is_empty());
    assert!(cpu.channel::<Vec3>(1).unwrap().is_empty());
}

#[test]
#[ignore = "requires GPU"]
fn snapshot_sources_remain_leased_through_submission_and_warm_slots_reuse_bindings() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![0xFF332211; 257]);
    let mut snapshots = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 2048,
            max_total_bytes: 2048,
        },
    )
    .unwrap();
    let snapshot = snapshots
        .try_capture(
            std::slice::from_ref(&(input.readback_source(0..257).unwrap())),
            stamp(9),
        )
        .unwrap()
        .unwrap();
    let mut p = processor(&engine, 257, 1);
    let mut pool = p.output_pool(&engine, config(4)).unwrap();
    for round in 0..6 {
        let mut held = Vec::new();
        for i in 0..4 {
            let count = [1, 65, 257, 3][i];
            let operations = [ImageOp::Exposure((i + 1) as f32), ImageOp::GrayWorld];
            let jobs = [ImageJob {
                source: snapshot
                    .channel::<u32>(0)
                    .unwrap()
                    .readback_source(0..count)
                    .unwrap(),
                operations: &operations,
                stamp: stamp(i as u64),
            }];
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let result = p
                .encode(&mut batch, &jobs, &mut pool, stamp(round))
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            held.push(submitted.frame(&result.frame).unwrap().clone());
        }
        assert_eq!(p.stats().binding_preparations, 5);
        assert_eq!(pool.stats().buffer_allocations, 8);
        for (i, frame) in held.iter().enumerate() {
            let cpu = collect(&engine, frame);
            assert!(
                cpu.channel::<u32>(0)
                    .unwrap()
                    .iter()
                    .all(|&pixel| pixel == 0xFF000000 | (0x222222 * (i + 1) as u32))
            );
        }
        drop(held);
        drain(&engine);
    }
    let jobs = [ImageJob {
        source: snapshot
            .channel::<u32>(0)
            .unwrap()
            .readback_source(0..257)
            .unwrap(),
        operations: &[],
        stamp: stamp(1),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let result = p
        .encode(&mut batch, &jobs, &mut pool, stamp(1))
        .unwrap()
        .unwrap();
    snapshots.reset().unwrap();
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 0);
    let submitted = batch.submit().unwrap();
    drop(jobs);
    drop(snapshot);
    let cpu = collect(&engine, submitted.frame(&result.frame).unwrap());
    assert_eq!(cpu.channel::<u32>(0).unwrap(), vec![0xFF332211; 257]);
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 1);
    assert_eq!(p.stats().encoded_frames, 25);
    assert_eq!(p.stats().buffer_allocations, 4);
    assert_eq!(p.stats().submissions, 0);
}

#[test]
#[ignore = "requires GPU"]
fn bad_pending_jobs_foreign_sources_and_device_loss_reject_before_output_work() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let other = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![0xFF302010; 32]);
    let foreign = source(&other, vec![0u32; 32]);
    let wrong_type =
        DeviceBuffer::from_host(&engine, vec![Vec3::ZERO; 4], BufferUsage::StorageReadWrite)
            .unwrap();
    let mut producer = processor(&engine, 32, 2);
    let mut target = processor(&engine, 16, 2);
    let mut source_pool = producer.output_pool(&engine, config(1)).unwrap();
    let mut output = target.output_pool(&engine, config(1)).unwrap();
    assert!(target.output_pool(&other, config(1)).is_err());
    assert!(
        target
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 1,
                    max_total_bytes: 16 * 4 + 2 * 16 * 16 - 1
                }
            )
            .is_err()
    );
    let source_jobs = [ImageJob {
        source: input.readback_source(0..32).unwrap(),
        operations: &[],
        stamp: stamp(1),
    }];
    for case in 0..9 {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let first = producer
            .encode(&mut batch, &source_jobs, &mut source_pool, stamp(1))
            .unwrap()
            .unwrap();
        let ops = if case == 0 {
            vec![ImageOp::Gamma(f32::NAN)]
        } else if case == 1 {
            vec![ImageOp::Gamma(1.0); 17]
        } else {
            vec![]
        };
        let mut jobs = vec![PendingImageJob {
            frame: &first.frame,
            channel: 0,
            pixels: 0..4,
            operations: &ops,
            stamp: stamp(2),
        }];
        match case {
            2 => jobs[0].pixels = 0..33,
            3 => jobs[0].channel = 1,
            4 => jobs[0].stamp.simulation_time = Some(f64::NAN),
            5 => jobs[0].pixels = 0..17,
            6 => jobs.push(PendingImageJob {
                frame: &first.frame,
                channel: 0,
                pixels: 0..4,
                operations: &[],
                stamp: stamp(2),
            }),
            7 => {
                jobs[0].stamp.simulation_time = Some(-1.0);
            }
            8 => {
                for id in 3..5 {
                    jobs.push(PendingImageJob {
                        frame: &first.frame,
                        channel: 0,
                        pixels: 0..1,
                        operations: &[],
                        stamp: stamp(id),
                    });
                }
            }
            _ => {}
        }
        assert!(
            target
                .encode_pending(&mut batch, &jobs, &mut output, stamp(0))
                .is_err(),
            "case {case}"
        );
        assert_eq!(output.stats().reserved_frames, 0);
        assert_eq!(batch.stats().encoded_frames, 1);
        let submitted = batch.submit().unwrap();
        assert_eq!(
            collect(&engine, submitted.frame(&first.frame).unwrap())
                .channel::<u32>(0)
                .unwrap(),
            vec![0xFF302010; 32]
        );
        drop(submitted);
        drain(&engine);
    }
    for bad in [
        foreign.readback_source(0..1).unwrap(),
        wrong_type.readback_source(0..1).unwrap(),
    ] {
        let jobs = [ImageJob {
            source: bad,
            operations: &[],
            stamp: stamp(1),
        }];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            target
                .encode(&mut batch, &jobs, &mut output, stamp(0))
                .is_err()
        );
        assert!(batch.submit().unwrap().index.is_none());
        assert_eq!(output.stats().reserved_frames, 0);
    }
    let mut wrong = SensorCommandBatch::new(&other).unwrap();
    assert!(
        target
            .encode(&mut wrong, &source_jobs, &mut output, stamp(1))
            .is_err()
    );
    assert!(wrong.submit().unwrap().index.is_none());
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = producer
        .encode(&mut batch, &source_jobs, &mut source_pool, stamp(0))
        .unwrap()
        .unwrap();
    let jobs = [PendingImageJob {
        frame: &first.frame,
        channel: 0,
        pixels: 0..1,
        operations: &[],
        stamp: stamp(1),
    }];
    let mut wrong = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        target
            .encode_pending(&mut wrong, &jobs, &mut output, stamp(0))
            .is_err()
    );
    assert!(wrong.submit().unwrap().index.is_none());
    engine.destroy_device().unwrap();
    assert!(
        target
            .encode_pending(&mut batch, &jobs, &mut output, stamp(0))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(output.stats().reserved_frames, 0);
    assert_eq!(target.stats().encoded_frames, 0);
    assert_eq!(target.stats().binding_preparations, 1);
}

#[test]
#[ignore = "requires GPU"]
fn device_source_jobs_share_the_batch_input_bound_but_pending_jobs_do_not_use_it() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![0u32]);
    let jobs: Vec<_> = (0..64)
        .map(|id| ImageJob {
            source: input.readback_source(0..0).unwrap(),
            operations: &[],
            stamp: stamp(id),
        })
        .collect();
    let mut a = processor(&engine, 1, 64);
    let mut b = processor(&engine, 1, 1);
    let mut c = processor(&engine, 1, 1);
    let mut pool_a = a.output_pool(&engine, config(2)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let mut pool_c = c.output_pool(&engine, config(1)).unwrap();
    let mut seed = SensorCommandBatch::new(&engine).unwrap();
    let old = a
        .encode(&mut seed, &jobs[..1], &mut pool_a, stamp(0))
        .unwrap()
        .unwrap();
    let submitted = seed.submit().unwrap();
    let old_frame = submitted.frame(&old.frame).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pool_a, stamp(0))
        .unwrap()
        .unwrap();
    assert!(
        batch
            .encode_with_frames(&[old_frame], |_| panic!("input limit"))
            .is_err()
    );
    assert!(
        c.encode(&mut batch, &jobs[..1], &mut pool_c, stamp(0))
            .is_err()
    );
    let pending = [PendingImageJob {
        frame: &first.frame,
        channel: 0,
        pixels: 0..0,
        operations: &[ImageOp::GrayWorld],
        stamp: stamp(1),
    }];
    let result = b
        .encode_pending(&mut batch, &pending, &mut pool_b, stamp(0))
        .unwrap()
        .unwrap();
    let done = batch.submit().unwrap();
    assert_eq!(done.stats.encoded_frames, 2);
    assert_eq!(
        collect(&engine, done.frame(&result.frame).unwrap())
            .channel::<Vec3>(1)
            .unwrap(),
        [Vec3::ONE]
    );
    assert_eq!(pool_c.stats().reserved_frames, 0);
    drop(done);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    batch
        .encode_with_frames(&vec![old_frame; 64], |_| Ok(()))
        .unwrap();
    assert!(
        a.encode(&mut batch, &jobs[..1], &mut pool_a, stamp(0))
            .is_err()
    );
    assert_eq!(batch.submit().unwrap().stats.encoded_frames, 0);
}
