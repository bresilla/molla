#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_sensors::{
    AnnotationConfig, AnnotationCounts, AnnotationJob, AnnotationLayout, AnnotationProcessor,
    Dataset, DeviceBoundingBox, DeviceSnapshotConfig, DeviceSnapshotPool, PendingAnnotationJob,
    SensorChannelSpec, SensorCommandBatch, SensorFrame, SensorFramePool, SensorFramePoolConfig,
    ShapeLabel,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_storage::BufferUsage;
use std::{collections::HashSet, sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 24,
    }
}

fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX - id,
        scene_revision: 27,
        simulation_time: Some(6.25),
    }
}

fn processor(engine: &Engine, pixels: u32, images: u32, labels: u32) -> AnnotationProcessor {
    AnnotationProcessor::new(
        engine,
        AnnotationConfig {
            max_pixels: pixels,
            max_images: images,
            max_labels: labels,
            max_buffer_bytes: 1 << 24,
        },
    )
    .unwrap()
}

fn source(engine: &Engine, values: Vec<u32>) -> DeviceBuffer<u32> {
    DeviceBuffer::from_host(engine, values, BufferUsage::StorageReadWrite).unwrap()
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
    let s = frame.channel::<u32>(0).unwrap();
    let c = frame.channel::<u32>(1).unwrap();
    let b = frame.channel::<DeviceBoundingBox>(2).unwrap();
    let n = frame.channel::<u32>(3).unwrap();
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1 << 22,
            max_total_bytes: 1 << 22,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(
            &[
                s.readback_source(0..s.len()).unwrap(),
                c.readback_source(0..c.len()).unwrap(),
                b.readback_source(0..b.len()).unwrap(),
                n.readback_source(0..n.len()).unwrap(),
            ],
            frame.stamp(),
        )
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn verify(
    cpu: &ReadbackFrame,
    layouts: &[AnnotationLayout],
    shapes: &[u32],
    labels: &[ShapeLabel],
) {
    let mut dataset = Dataset::new();
    let known: HashSet<_> = labels.iter().map(|label| label.shape_id).collect();
    for label in labels {
        dataset.label_shapes(&[label.shape_id], label.class_id);
    }
    let actual_shapes = cpu.channel::<u32>(0).unwrap();
    let classes = cpu.channel::<u32>(1).unwrap();
    let boxes = cpu.channel::<DeviceBoundingBox>(2).unwrap();
    let counts = cpu.channel::<u32>(3).unwrap();
    assert_eq!(actual_shapes, shapes);
    for (i, layout) in layouts.iter().enumerate() {
        let image = &shapes[layout.pixels.clone()];
        let reference = dataset.annotate(i as u32, image, layout.width, layout.height);
        assert_eq!(&classes[layout.pixels.clone()], reference.class_image);
        let read = AnnotationCounts::from_words(&counts[i * 4..i * 4 + 4]).unwrap();
        let missed = image.iter().filter(|&&id| id == u32::MAX).count() as u32;
        let labelled = image.iter().filter(|id| known.contains(id)).count() as u32;
        assert_eq!(
            read,
            AnnotationCounts {
                labelled_pixels: labelled,
                missed_pixels: missed,
                unlabelled_pixels: image.len() as u32 - missed - labelled,
                visible_instances: reference.bboxes.len() as u32
            }
        );
        for (j, bbox) in boxes[layout.boxes.clone()].iter().enumerate() {
            let expected = reference
                .bboxes
                .get(j)
                .map(|b| DeviceBoundingBox {
                    bounds: [b.x_min, b.y_min, b.x_max, b.y_max],
                    shape_id: b.shape_id,
                    class_id: b.class_id,
                    pixel_count: b.pixel_count,
                    reserved: 0,
                })
                .unwrap_or_default();
            assert_eq!(*bbox, expected);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn independent_label_tables_chain_annotations_in_one_submission_and_match_cpu() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let values: Vec<_> = (0..400)
        .map(|i| [0, 7, u32::MAX - 1, u32::MAX, 321][i % 5])
        .collect();
    let mut input = source(&engine, values.clone());
    input.mark_device_dirty();
    assert!(input.host().is_err());
    let labels_a = [
        ShapeLabel {
            shape_id: u32::MAX - 1,
            class_id: u32::MAX,
        },
        ShapeLabel {
            shape_id: 7,
            class_id: 0,
        },
        ShapeLabel {
            shape_id: 0,
            class_id: 6,
        },
        ShapeLabel {
            shape_id: 1_000_000,
            class_id: 9,
        },
    ];
    let labels_b = [
        ShapeLabel {
            shape_id: 7,
            class_id: 17,
        },
        ShapeLabel {
            shape_id: u32::MAX - 1,
            class_id: 0,
        },
        ShapeLabel {
            shape_id: 321,
            class_id: 999,
        },
    ];
    let mut a = processor(&engine, 400, 4, 4);
    let mut b = processor(&engine, 400, 4, 4);
    a.set_labels(&labels_a).unwrap();
    b.set_labels(&[]).unwrap();
    b.set_labels(&labels_b).unwrap();
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let descriptions = [(0, 0, 0), (3, 257, 1), (300, 7, 9), (0, 0, 0)];
    let jobs: Vec<_> = descriptions
        .iter()
        .enumerate()
        .map(|(i, &(offset, width, height))| AnnotationJob {
            shapes: input
                .readback_source(offset..offset + (width * height) as usize)
                .unwrap(),
            width,
            height,
            stamp: stamp(u64::MAX - i as u64),
        })
        .collect();
    let expected: Vec<_> = descriptions
        .iter()
        .flat_map(|&(offset, width, height)| {
            values[offset..offset + (width * height) as usize]
                .iter()
                .copied()
        })
        .collect();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pool_a, stamp(1))
        .unwrap()
        .unwrap();
    let pending: Vec<_> = first
        .layouts
        .iter()
        .map(|layout| PendingAnnotationJob {
            frame: &first.frame,
            channel: 0,
            pixels: layout.pixels.clone(),
            width: layout.width,
            height: layout.height,
            stamp: layout.stamp,
        })
        .collect();
    let second = b
        .encode_pending(&mut batch, &pending, &mut pool_b, stamp(2))
        .unwrap()
        .unwrap();
    assert_eq!(first.label_revision, 1);
    assert_eq!(second.label_revision, 2);
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().input_upload_bytes, 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 6);
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.encoded_frames, 2);
    assert_eq!(a.stats().submissions, 0);
    assert_eq!(b.stats().submissions, 0);
    let old_a = submitted.frame(&first.frame).unwrap().clone();
    let old_b = submitted.frame(&second.frame).unwrap().clone();
    drop(submitted);
    drop(pool_a);
    drop(pool_b);
    drop(a);
    drop(b);
    verify(
        &collect(&engine, &old_a),
        &first.layouts,
        &expected,
        &labels_a,
    );
    verify(
        &collect(&engine, &old_b),
        &second.layouts,
        &expected,
        &labels_b,
    );
}

#[test]
#[ignore = "requires GPU"]
fn full_1024_image_capacity_uses_deduplicated_pending_receipts_and_relative_offsets() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let data: Vec<_> = (0..1024u32)
        .map(|i| [7, u32::MAX - 1, u32::MAX, 321][i as usize % 4])
        .collect();
    let input = source(&engine, data.clone());
    let mut a = processor(&engine, 1024, 1, 2);
    let mut b = processor(&engine, 1024, 1024, 2);
    let labels = [
        ShapeLabel {
            shape_id: 7,
            class_id: 0,
        },
        ShapeLabel {
            shape_id: u32::MAX - 1,
            class_id: u32::MAX,
        },
    ];
    b.set_labels(&labels).unwrap();
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let specs = [SensorChannelSpec::of::<u32>(1027)];
    let mut shifted_pool = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    let jobs = [AnnotationJob {
        shapes: input.readback_source(0..1024).unwrap(),
        width: 32,
        height: 32,
        stamp: stamp(1),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &jobs, &mut pool_a, stamp(1))
        .unwrap()
        .unwrap();
    let shifted = batch
        .encode_from_pending(
            &[&first.frame],
            &mut shifted_pool,
            &specs,
            &[3..1027],
            stamp(2),
            |encoder, inputs, outputs| {
                let view = inputs.channel::<u32>(0, 0)?;
                encoder.copy_buffer_to_buffer(view.binding()?.buffer, 0, &outputs[0], 12, 4096);
                Ok(0)
            },
        )
        .unwrap()
        .unwrap();
    let clones: Vec<_> = (0..1024).map(|_| shifted.clone()).collect();
    let jobs: Vec<_> = clones
        .iter()
        .enumerate()
        .map(|(i, frame)| PendingAnnotationJob {
            frame,
            channel: 0,
            pixels: i..i + 1,
            width: 1,
            height: 1,
            stamp: stamp(u64::MAX - i as u64),
        })
        .collect();
    let result = b
        .encode_pending(&mut batch, &jobs, &mut pool_b, stamp(3))
        .unwrap()
        .unwrap();
    assert_eq!(result.layouts.len(), 1024);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 4);
    assert_eq!(submitted.stats.encoded_frames, 3);
    verify(
        &collect(&engine, submitted.frame(&result.frame).unwrap()),
        &result.layouts,
        &data,
        &labels,
    );
}

#[test]
#[ignore = "requires GPU"]
fn cancelled_descriptors_relabel_reset_backpressure_and_empty_frames_preserve_versions() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![7; 65]);
    let mut p = processor(&engine, 65, 2, 2);
    let mut pool = p.output_pool(&engine, config(2)).unwrap();
    let old_labels = [ShapeLabel {
        shape_id: 7,
        class_id: 5,
    }];
    p.set_labels(&old_labels).unwrap();
    let cancelled = [AnnotationJob {
        shapes: input.readback_source(0..65).unwrap(),
        width: 1,
        height: 65,
        stamp: stamp(1),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    p.encode(&mut batch, &cancelled, &mut pool, stamp(1))
        .unwrap()
        .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 2);
    let jobs = [AnnotationJob {
        shapes: input.readback_source(0..65).unwrap(),
        width: 65,
        height: 1,
        stamp: stamp(2),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = p
        .encode(&mut batch, &jobs, &mut pool, stamp(2))
        .unwrap()
        .unwrap();
    pool.reset().unwrap();
    let submitted = batch.submit().unwrap();
    let old = submitted.frame(&first.frame).unwrap().clone();
    assert!(!pool.is_current(&old));
    drop(submitted);
    p.set_labels(&[]).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let second = p
        .encode(&mut batch, &jobs, &mut pool, stamp(3))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 1);
    let unlabelled = submitted.frame(&second.frame).unwrap().clone();
    drop(submitted);
    let before = p.stats();
    let mut blocked = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        p.encode(&mut blocked, &jobs, &mut pool, stamp(4))
            .unwrap()
            .is_none()
    );
    assert!(blocked.submit().unwrap().index.is_none());
    assert_eq!(p.stats(), before);
    assert_eq!(first.label_revision, 1);
    assert_eq!(second.label_revision, 2);
    verify(
        &collect(&engine, &old),
        &first.layouts,
        &[7; 65],
        &old_labels,
    );
    verify(
        &collect(&engine, &unlabelled),
        &second.layouts,
        &[7; 65],
        &[],
    );
    drop(old);
    drop(unlabelled);
    drain(&engine);
    p.set_labels(&old_labels).unwrap();
    let empty = [AnnotationJob {
        shapes: input.readback_source(0..0).unwrap(),
        width: 0,
        height: 0,
        stamp: stamp(5),
    }];
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let result = p
        .encode(&mut batch, &empty, &mut pool, stamp(5))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 2);
    verify(
        &collect(&engine, submitted.frame(&result.frame).unwrap()),
        &result.layouts,
        &[],
        &old_labels,
    );
    drop(submitted);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let result = p
        .encode_pending(&mut batch, &[], &mut pool, stamp(6))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    verify(
        &collect(&engine, submitted.frame(&result.frame).unwrap()),
        &result.layouts,
        &[],
        &old_labels,
    );
}

#[test]
#[ignore = "requires GPU"]
fn four_slot_warm_relabelling_reuses_bindings_and_snapshot_leases() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let values: Vec<_> = (0..257).map(|i| [0, 7, u32::MAX, 321][i % 4]).collect();
    let input = source(&engine, values.clone());
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
        .try_capture(&[input.readback_source(0..257).unwrap()], stamp(1))
        .unwrap()
        .unwrap();
    let mut p = processor(&engine, 257, 1, 2);
    let mut pool = p.output_pool(&engine, config(4)).unwrap();
    for round in 0..6 {
        let labels = [
            ShapeLabel {
                shape_id: 0,
                class_id: round,
            },
            ShapeLabel {
                shape_id: 7,
                class_id: u32::MAX - round,
            },
        ];
        p.set_labels(&labels).unwrap();
        let mut held = Vec::new();
        for count in [1, 65, 257, 3] {
            let jobs = [AnnotationJob {
                shapes: snapshot
                    .channel::<u32>(0)
                    .unwrap()
                    .readback_source(0..count)
                    .unwrap(),
                width: count as u32,
                height: 1,
                stamp: stamp(count as u64),
            }];
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let result = p
                .encode(&mut batch, &jobs, &mut pool, stamp(round as u64))
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            held.push((
                submitted.frame(&result.frame).unwrap().clone(),
                result.layouts,
                count,
            ));
        }
        assert_eq!(p.stats().binding_preparations, 5);
        assert_eq!(pool.stats().buffer_allocations, 16);
        for (frame, layouts, count) in &held {
            verify(
                &collect(&engine, frame),
                layouts,
                &values[..*count],
                &labels,
            );
        }
        drop(held);
        drain(&engine);
    }
    assert_eq!(p.stats().encoded_frames, 24);
    assert_eq!(p.stats().buffer_allocations, 8);
    assert_eq!(p.stats().submissions, 0);
    let jobs = [AnnotationJob {
        shapes: snapshot
            .channel::<u32>(0)
            .unwrap()
            .readback_source(0..257)
            .unwrap(),
        width: 257,
        height: 1,
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
    verify(
        &collect(&engine, submitted.frame(&result.frame).unwrap()),
        &result.layouts,
        &values,
        &[
            ShapeLabel {
                shape_id: 0,
                class_id: 5,
            },
            ShapeLabel {
                shape_id: 7,
                class_id: u32::MAX - 5,
            },
        ],
    );
    drain(&engine);
    assert_eq!(snapshots.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn malformed_and_foreign_inputs_reject_before_work_and_known_loss_cancels_pending_frames() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![7; 32]);
    let foreign_input = source(&foreign, vec![7; 32]);
    let wrong_type = DeviceBuffer::from_host(
        &engine,
        vec![molla_math::Vec3::ZERO],
        BufferUsage::StorageReadWrite,
    )
    .unwrap();
    let mut a = processor(&engine, 32, 1, 2);
    let mut b = processor(&engine, 16, 2, 2);
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    assert!(b.output_pool(&foreign, config(1)).is_err());
    assert!(
        b.output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 8 * 16 + 32 * 2 * 2 + 16 * 2 - 1
            }
        )
        .is_err()
    );
    let originals = [AnnotationJob {
        shapes: input.readback_source(0..32).unwrap(),
        width: 32,
        height: 1,
        stamp: stamp(0),
    }];
    for case in 0..10 {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let first = a
            .encode(&mut batch, &originals, &mut pool_a, stamp(0))
            .unwrap()
            .unwrap();
        let mut jobs = vec![PendingAnnotationJob {
            frame: &first.frame,
            channel: 0,
            pixels: 0..4,
            width: 4,
            height: 1,
            stamp: stamp(1),
        }];
        match case {
            0 => jobs[0].width = 0,
            1 => {
                jobs[0].width = u32::MAX;
                jobs[0].height = u32::MAX;
            }
            2 => jobs[0].width = 5,
            3 => jobs[0].channel = 2,
            4 => jobs[0].pixels = 0..33,
            5 => jobs[0].stamp.simulation_time = Some(f64::INFINITY),
            6 => jobs[0].stamp.simulation_time = Some(-1.0),
            7 => {
                jobs[0].pixels = 0..32;
                jobs[0].width = 32;
            }
            8 => jobs.push(PendingAnnotationJob {
                frame: &first.frame,
                channel: 0,
                pixels: 0..4,
                width: 4,
                height: 1,
                stamp: stamp(1),
            }),
            _ => {
                for id in 2..4 {
                    jobs.push(PendingAnnotationJob {
                        frame: &first.frame,
                        channel: 0,
                        pixels: 0..1,
                        width: 1,
                        height: 1,
                        stamp: stamp(id),
                    });
                }
            }
        }
        assert!(
            b.encode_pending(&mut batch, &jobs, &mut pool_b, stamp(0))
                .is_err(),
            "case {case}"
        );
        assert_eq!(pool_b.stats().reserved_frames, 0);
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.encoded_frames, 1);
        verify(
            &collect(&engine, submitted.frame(&first.frame).unwrap()),
            &first.layouts,
            &[7; 32],
            &[],
        );
        drop(submitted);
        drain(&engine);
    }
    for bad in [
        foreign_input.readback_source(0..1).unwrap(),
        wrong_type.readback_source(0..1).unwrap(),
    ] {
        let jobs = [AnnotationJob {
            shapes: bad,
            width: 1,
            height: 1,
            stamp: stamp(1),
        }];
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(b.encode(&mut batch, &jobs, &mut pool_b, stamp(0)).is_err());
        assert!(batch.submit().unwrap().index.is_none());
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = a
        .encode(&mut batch, &originals, &mut pool_a, stamp(0))
        .unwrap()
        .unwrap();
    let jobs = [PendingAnnotationJob {
        frame: &first.frame,
        channel: 0,
        pixels: 0..1,
        width: 1,
        height: 1,
        stamp: stamp(1),
    }];
    let mut wrong = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        b.encode_pending(&mut wrong, &jobs, &mut pool_b, stamp(0))
            .is_err()
    );
    assert!(wrong.submit().unwrap().index.is_none());
    let mut wrong = SensorCommandBatch::new(&foreign).unwrap();
    assert!(
        b.encode_pending(&mut wrong, &jobs, &mut pool_b, stamp(0))
            .is_err()
    );
    assert!(wrong.submit().unwrap().index.is_none());
    engine.destroy_device().unwrap();
    assert!(
        b.encode_pending(&mut batch, &jobs, &mut pool_b, stamp(0))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pool_b.stats().reserved_frames, 0);
    assert_eq!(b.stats().encoded_frames, 0);
    assert_eq!(b.stats().binding_preparations, 1);
}

#[test]
#[ignore = "requires GPU"]
fn external_job_bound_is_explicit_and_empty_label_and_image_lists_clear_all_active_channels() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input = source(&engine, vec![7]);
    let mut p = processor(&engine, 64, 128, 2);
    let mut pool = p.output_pool(&engine, config(1)).unwrap();
    let jobs: Vec<_> = (0..65)
        .map(|i| AnnotationJob {
            shapes: input.readback_source(0..0).unwrap(),
            width: 0,
            height: 0,
            stamp: stamp(i),
        })
        .collect();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(p.encode(&mut batch, &jobs, &mut pool, stamp(0)).is_err());
    assert!(batch.submit().unwrap().index.is_none());
    assert_eq!(pool.stats().reserved_frames, 0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let frame = p
        .encode(&mut batch, &jobs[..64], &mut pool, stamp(0))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    verify(
        &collect(&engine, submitted.frame(&frame.frame).unwrap()),
        &frame.layouts,
        &[],
        &[],
    );
    drop(submitted);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let frame = p
        .encode(&mut batch, &[], &mut pool, stamp(0))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    assert_eq!(p.stats().binding_preparations, 1);
    verify(
        &collect(&engine, submitted.frame(&frame.frame).unwrap()),
        &frame.layouts,
        &[],
        &[],
    );
}
