//! CPU annotation, completed GPU annotation and selected box collection.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_sensors::{
    AnnotationConfig, AnnotationJob, AnnotationProcessor, Dataset, DeviceBoundingBox, ShapeLabel,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_storage::BufferUsage;
use std::time::Duration;

fn bench_annotations(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("annotation benchmark requires GPU");
    eprintln!("sensor annotation adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let complete = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap()
    };
    let mut group = c.benchmark_group("sensor_annotations");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [1024u32, 65_536, 1_000_000] {
        for images in [1u32, 4] {
            for (label, label_count) in [("dense", 1u32), ("distributed", 257)] {
                let labels: Vec<_> = (0..label_count)
                    .map(|i| ShapeLabel {
                        shape_id: i * 17,
                        class_id: i + 1,
                    })
                    .collect();
                let pixels: Vec<_> = (0..n).map(|i| (i % label_count) * 17).collect();
                let input =
                    DeviceBuffer::from_host(&engine, pixels.clone(), BufferUsage::StorageReadWrite)
                        .unwrap();
                let mut proc = AnnotationProcessor::new(
                    &engine,
                    AnnotationConfig {
                        max_pixels: n,
                        max_images: images,
                        max_labels: label_count,
                        max_buffer_bytes: 64 * 1024 * 1024,
                    },
                )
                .unwrap();
                proc.set_labels(&labels).unwrap();
                let size = n / images;
                let jobs: Vec<_> = (0..images)
                    .map(|i| AnnotationJob {
                        shapes: input
                            .readback_source((i * size) as usize..((i + 1) * size) as usize)
                            .unwrap(),
                        width: size,
                        height: 1,
                        stamp: ReadbackStamp {
                            sensor_id: i as u64,
                            ..Default::default()
                        },
                    })
                    .collect();
                let mut dataset = Dataset::new();
                for label in &labels {
                    dataset.label_shapes(&[label.shape_id], label.class_id);
                }
                let bytes = (label_count as u64 * 32).max(8);
                let mut ring = ReadbackRing::new(
                    &engine,
                    ReadbackConfig {
                        slots: 1,
                        bytes_per_slot: bytes,
                        max_total_bytes: bytes,
                    },
                )
                .unwrap();
                let key = format!("{label}/{images}_cameras");
                group.bench_function(BenchmarkId::new(format!("cpu/{key}"), n), |b| {
                    b.iter(|| {
                        for i in 0..images {
                            black_box(dataset.annotate(
                                i,
                                black_box(&pixels[(i * size) as usize..((i + 1) * size) as usize]),
                                size,
                                1,
                            ));
                        }
                    })
                });
                proc.process(&jobs).unwrap();
                complete();
                group.bench_function(BenchmarkId::new(format!("gpu_completed/{key}"), n), |b| {
                    b.iter(|| {
                        proc.process(black_box(&jobs)).unwrap();
                        complete();
                    })
                });
                group.bench_function(
                    BenchmarkId::new(format!("gpu_selected_boxes/{key}"), n),
                    |b| {
                        b.iter(|| {
                            let frame = proc.process(black_box(&jobs)).unwrap();
                            for image in 0..images as usize {
                                let ticket = ring
                                    .try_enqueue(
                                        &[frame.box_count_source(image).unwrap()],
                                        jobs[image].stamp,
                                    )
                                    .unwrap()
                                    .unwrap();
                                let count =
                                    ring.wait(&ticket, Duration::from_secs(30))
                                        .unwrap()
                                        .channel::<u32>(0)
                                        .unwrap()[0] as usize;
                                let ticket = ring
                                    .try_enqueue(
                                        &[frame.boxes_source(image, 0..count).unwrap()],
                                        jobs[image].stamp,
                                    )
                                    .unwrap()
                                    .unwrap();
                                black_box(
                                    ring.wait(&ticket, Duration::from_secs(30))
                                        .unwrap()
                                        .channel::<DeviceBoundingBox>(0)
                                        .unwrap(),
                                );
                            }
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_annotations);
criterion_main!(benches);
