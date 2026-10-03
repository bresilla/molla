#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_sensors::{
    AnnotationConfig, AnnotationCounts, AnnotationDeviceFrame, AnnotationJob, AnnotationProcessor,
    Dataset, DeviceBoundingBox, DeviceSnapshotConfig, DeviceSnapshotPool, ShapeLabel,
    readback::{
        DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp, ReadbackTicket,
    },
};
use molla_storage::{BufferElement, BufferUsage};
use std::{collections::HashSet, sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn source<T: BufferElement>(engine: &Engine, values: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, values, BufferUsage::StorageReadWrite).unwrap()
}
fn processor(engine: &Engine, pixels: usize, images: u32, labels: u32) -> AnnotationProcessor {
    AnnotationProcessor::new(
        engine,
        AnnotationConfig {
            max_pixels: pixels.max(1) as u32,
            max_images: images,
            max_labels: labels.max(1),
            max_buffer_bytes: 256 * 1024 * 1024,
        },
    )
    .unwrap()
}
fn ring(engine: &Engine, pixels: usize, labels: usize, slots: usize) -> ReadbackRing {
    let bytes = (pixels as u64 * 8 + labels as u64 * 32 + 16).next_multiple_of(8);
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
fn job(
    source: &DeviceBuffer<u32>,
    offset: usize,
    width: u32,
    height: u32,
    id: u64,
) -> AnnotationJob<'_> {
    AnnotationJob {
        shapes: source
            .readback_source(offset..offset + (width * height) as usize)
            .unwrap(),
        width,
        height,
        stamp: ReadbackStamp {
            sensor_id: id,
            sample_id: id + 11,
            scene_revision: 42,
            simulation_time: Some(0.25),
        },
    }
}
fn enqueue(
    frame: &AnnotationDeviceFrame<'_>,
    image: usize,
    ring: &mut ReadbackRing,
) -> ReadbackTicket {
    let layout = &frame.layouts()[image];
    ring.try_enqueue(
        &[
            frame.shape_source(image, 0..layout.pixels.len()).unwrap(),
            frame.class_source(image, 0..layout.pixels.len()).unwrap(),
            frame.boxes_source(image, 0..layout.boxes.len()).unwrap(),
            frame.counts_source(image).unwrap(),
        ],
        layout.stamp,
    )
    .unwrap()
    .unwrap()
}
fn verify(read: &ReadbackFrame, shapes: &[u32], width: u32, height: u32, labels: &[ShapeLabel]) {
    let mut dataset = Dataset::new();
    for label in labels {
        dataset.label_shapes(&[label.shape_id], label.class_id);
    }
    let expected = dataset.annotate(0, shapes, width, height);
    assert_eq!(read.channel::<u32>(0).unwrap(), expected.shape_image);
    assert_eq!(read.channel::<u32>(1).unwrap(), expected.class_image);
    let boxes = read.channel::<DeviceBoundingBox>(2).unwrap();
    let counts = AnnotationCounts::from_words(&read.channel::<u32>(3).unwrap()).unwrap();
    let labelled: HashSet<_> = labels.iter().map(|l| l.shape_id).collect();
    let misses = shapes.iter().filter(|&&id| id == u32::MAX).count() as u32;
    let known = shapes.iter().filter(|id| labelled.contains(id)).count() as u32;
    assert_eq!(
        counts,
        AnnotationCounts {
            labelled_pixels: known,
            unlabelled_pixels: shapes.len() as u32 - misses - known,
            missed_pixels: misses,
            visible_instances: expected.bboxes.len() as u32
        }
    );
    for (actual, expected) in boxes.iter().zip(&expected.bboxes) {
        assert_eq!(
            *actual,
            DeviceBoundingBox {
                bounds: [
                    expected.x_min,
                    expected.y_min,
                    expected.x_max,
                    expected.y_max
                ],
                shape_id: expected.shape_id,
                class_id: expected.class_id,
                pixel_count: expected.pixel_count,
                reserved: 0
            }
        );
        let decoded = actual.to_cpu().unwrap();
        assert_eq!(decoded.width(), expected.width());
        assert_eq!(decoded.height(), expected.height());
    }
    assert!(
        boxes[expected.bboxes.len()..]
            .iter()
            .all(|b| *b == DeviceBoundingBox::default())
    );
}

#[test]
#[ignore = "requires GPU"]
fn annotation_matches_cpu_independent_images_sparse_ids_and_empty_ranges() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let labels = [
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
    let data: Vec<u32> = (0..300)
        .map(|i| [0, 7, u32::MAX - 1, u32::MAX, 321][i % 5])
        .collect();
    let input = source(&engine, data.clone());
    let empty = source::<u32>(&engine, vec![]);
    let mut proc = processor(&engine, 300, 5, 4);
    assert_eq!(proc.set_labels(&labels).unwrap(), 1);
    let jobs = [
        job(&empty, 0, 0, 0, 100),
        job(&input, 3, 17, 11, 101),
        job(&empty, 0, 0, 0, 102),
        job(&input, 200, 7, 9, 103),
        job(&empty, 0, 0, 0, 104),
    ];
    let frame = proc.process(&jobs).unwrap();
    assert_eq!(frame.label_revision(), 1);
    let mut ring = ring(&engine, 300, 4, 5);
    let tickets: Vec<_> = (0..5).map(|i| enqueue(&frame, i, &mut ring)).collect();
    assert!(frame.class_source(1, 0..188).is_err());
    assert!(frame.boxes_source(0, 0..5).is_err());
    assert!(frame.counts_source(5).is_err());
    for i in (0..5).rev() {
        let read = ring.wait(&tickets[i], Duration::from_secs(10)).unwrap();
        let (offset, n) = if i == 1 {
            (3, 187)
        } else if i == 3 {
            (200, 63)
        } else {
            (0, 0)
        };
        verify(
            &read,
            &data[offset..offset + n],
            jobs[i].width,
            jobs[i].height,
            &labels,
        );
        assert_eq!(read.stamp, jobs[i].stamp);
    }
    assert_eq!(proc.stats().buffer_allocations, 8);
    assert_eq!(proc.stats().binding_preparations, 1);
    assert_eq!(proc.stats().kernel_dispatches, 3);
}

#[test]
#[ignore = "requires GPU"]
fn annotation_compaction_orders_labels_across_workgroup_chunks() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let n = 65_537u32;
    let labels: Vec<_> = (0..n)
        .rev()
        .map(|i| ShapeLabel {
            shape_id: i * 17,
            class_id: i ^ 0xfedcba98,
        })
        .collect();
    let mut proc = processor(&engine, n as usize * 2, 2, n);
    proc.set_labels(&labels).unwrap();
    let data: Vec<_> = (0..n).map(|i| (n - 1 - i) * 17).collect();
    let input = source(&engine, data.clone());
    let alternate: Vec<_> = data
        .iter()
        .enumerate()
        .map(|(i, &id)| if i % 3 == 0 { u32::MAX } else { id })
        .collect();
    let input2 = source(&engine, alternate.clone());
    let frame = proc
        .process(&[job(&input, 0, n, 1, 1), job(&input2, 0, 1, n, 2)])
        .unwrap();
    let mut ring = ring(&engine, n as usize, n as usize, 2);
    let a = enqueue(&frame, 0, &mut ring);
    let b = enqueue(&frame, 1, &mut ring);
    verify(
        &ring.wait(&b, Duration::from_secs(20)).unwrap(),
        &alternate,
        1,
        n,
        &labels,
    );
    verify(
        &ring.wait(&a, Duration::from_secs(20)).unwrap(),
        &data,
        n,
        1,
        &labels,
    );
}

#[test]
#[ignore = "requires GPU"]
fn annotation_million_pixel_single_instance_has_exact_bounds_and_count() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let n = 1_000_003;
    let data = vec![7u32; n];
    let input = source(&engine, data.clone());
    let mut proc = processor(&engine, n, 1, 1);
    let labels = [ShapeLabel {
        shape_id: 7,
        class_id: u32::MAX,
    }];
    proc.set_labels(&labels).unwrap();
    let frame = proc.process(&[job(&input, 0, n as u32, 1, 1)]).unwrap();
    let mut ring = ring(&engine, n, 1, 1);
    let ticket = enqueue(&frame, 0, &mut ring);
    verify(
        &ring.wait(&ticket, Duration::from_secs(30)).unwrap(),
        &data,
        n as u32,
        1,
        &labels,
    );
}

#[test]
#[ignore = "requires GPU"]
fn annotation_relabel_snapshot_reuse_and_empty_labels_preserve_queued_versions() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let data = vec![7, 7, u32::MAX, 9, 9, 7];
    let input = source(&engine, data.clone());
    let mut proc = processor(&engine, 6, 1, 2);
    let first_labels = [ShapeLabel {
        shape_id: 7,
        class_id: 1,
    }];
    proc.set_labels(&first_labels).unwrap();
    let mut pool = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 24,
            max_total_bytes: 24,
        },
    )
    .unwrap();
    let snapshot = pool
        .try_capture(&[input.readback_source(0..6).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let channel = snapshot.channel::<u32>(0).unwrap();
    let jobs = [AnnotationJob {
        shapes: channel.readback_source(0..6).unwrap(),
        width: 3,
        height: 2,
        stamp: Default::default(),
    }];
    let frame = proc.process(&jobs).unwrap();
    let mut ring = ring(&engine, 6, 2, 3);
    let first = enqueue(&frame, 0, &mut ring);
    drop(jobs);
    drop(snapshot);
    pool.reset().unwrap();
    let second_labels = [
        ShapeLabel {
            shape_id: 9,
            class_id: 77,
        },
        ShapeLabel {
            shape_id: 7,
            class_id: 88,
        },
    ];
    assert_eq!(proc.set_labels(&second_labels).unwrap(), 2);
    let frame = proc.process(&[job(&input, 0, 2, 3, 1)]).unwrap();
    let second = enqueue(&frame, 0, &mut ring);
    assert_eq!(proc.set_labels(&[]).unwrap(), 3);
    let frame = proc.process(&[job(&input, 0, 3, 2, 2)]).unwrap();
    let third = enqueue(&frame, 0, &mut ring);
    verify(
        &ring.wait(&third, Duration::from_secs(10)).unwrap(),
        &data,
        3,
        2,
        &[],
    );
    verify(
        &ring.wait(&second, Duration::from_secs(10)).unwrap(),
        &data,
        2,
        3,
        &second_labels,
    );
    verify(
        &ring.wait(&first, Duration::from_secs(10)).unwrap(),
        &data,
        3,
        2,
        &first_labels,
    );
    assert_eq!(pool.available_slots().unwrap(), 1);
    let frame = proc.process(&[]).unwrap();
    assert!(frame.layouts().is_empty());
    assert!(frame.counts_source(0).is_err());
    assert_eq!(proc.stats().submissions, 4);
    assert_eq!(proc.stats().kernel_dispatches, 7);
    assert_eq!(proc.stats().binding_preparations, 1);
}

#[test]
#[ignore = "requires GPU"]
fn annotation_rejects_bad_inputs_without_submitting_or_replacing_labels() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut proc = processor(&engine, 4, 2, 2);
    let label = ShapeLabel {
        shape_id: 7,
        class_id: 17,
    };
    proc.set_labels(&[label]).unwrap();
    for labels in [
        vec![label, label],
        vec![ShapeLabel {
            shape_id: u32::MAX,
            class_id: 9,
        }],
        vec![label; 3],
    ] {
        assert!(proc.set_labels(&labels).is_err());
        assert_eq!(proc.label_revision(), 1);
    }
    let input = source(&engine, vec![7u32; 5]);
    let wrong = source(&engine, vec![7i32]);
    assert!(
        proc.process(&[AnnotationJob {
            shapes: wrong.readback_source(0..1).unwrap(),
            width: 1,
            height: 1,
            stamp: Default::default()
        }])
        .is_err()
    );
    for (width, height) in [(5, 1), (3, 1), (0, 2), (u32::MAX, u32::MAX)] {
        assert!(
            proc.process(&[AnnotationJob {
                shapes: input.readback_source(0..4).unwrap(),
                width,
                height,
                stamp: Default::default()
            }])
            .is_err()
        );
    }
    assert!(
        proc.process(&[job(&input, 0, 1, 1, 2), job(&input, 0, 1, 1, 2)])
            .is_err()
    );
    assert!(
        proc.process(&[
            job(&input, 0, 1, 1, 1),
            job(&input, 0, 1, 1, 2),
            job(&input, 0, 1, 1, 3)
        ])
        .is_err()
    );
    for time in [f64::NAN, f64::INFINITY, -0.1] {
        let mut job = job(&input, 0, 1, 1, 1);
        job.stamp.simulation_time = Some(time);
        assert!(proc.process(&[job]).is_err());
    }
    let foreign = Engine::new_gpu().unwrap();
    let other = source(&foreign, vec![7u32]);
    assert!(proc.process(&[job(&other, 0, 1, 1, 1)]).is_err());
    assert_eq!(proc.stats().submissions, 0);
    assert_eq!(proc.stats().label_uploads, 1);
    let frame = proc.process(&[job(&input, 0, 2, 2, 1)]).unwrap();
    let mut ring = ring(&engine, 4, 2, 1);
    let ticket = enqueue(&frame, 0, &mut ring);
    verify(
        &ring.wait(&ticket, Duration::from_secs(10)).unwrap(),
        &[7; 4],
        2,
        2,
        &[label],
    );
}
