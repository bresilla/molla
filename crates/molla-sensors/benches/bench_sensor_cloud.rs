//! Completed device cloud transforms/compaction, CPU reference and selected-prefix collection.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{
    CloudFilter, CloudInput, CloudJob, CloudPoint, CloudProcessor, CloudProcessorConfig,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing},
};
use molla_storage::BufferUsage;
use std::time::Duration;

fn bench_cloud(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("cloud benchmark requires GPU");
    eprintln!("sensor cloud adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let mut group = c.benchmark_group("sensor_cloud");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [1024usize, 65_536, 1_000_003] {
        let values: Vec<_> = (0..n)
            .map(|i| {
                let mut p = CloudPoint::new(
                    Vec3::new((i % 101) as f64 * 0.1, (i % 37) as f64 * 0.1, 0.5),
                    7,
                    9,
                );
                p.position[3] = (i % 8) as f32 * 0.125;
                p.grid[3] = i as u32;
                p
            })
            .collect();
        let input = DeviceBuffer::from_host(&engine, values.clone(), BufferUsage::StorageReadWrite)
            .unwrap();
        let mut proc = CloudProcessor::new(
            &engine,
            CloudProcessorConfig {
                max_points: n as u32,
                max_jobs: 1,
                max_buffer_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();
        let bytes = (n as u64 * 112).max(8);
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: bytes,
                max_total_bytes: bytes,
            },
        )
        .unwrap();
        let transform = Transform::new(Vec3::new(0.25, 0.5, 0.75), Quat::from_rotation_z(0.1));
        for (label, min_intensity) in [
            ("transform", None),
            ("half_kept", Some(0.5)),
            ("all_rejected", Some(2.0)),
        ] {
            let filter = CloudFilter {
                min_intensity,
                ..Default::default()
            };
            let jobs = [CloudJob {
                input: CloudInput::Points(input.readback_source(0..n).unwrap()),
                transform,
                filter,
                stamp: Default::default(),
            }];
            let mut host = Vec::with_capacity(n);
            group.bench_function(BenchmarkId::new(format!("cpu/{label}"), n), |b| {
                b.iter(|| {
                    host.clear();
                    for p in black_box(&values) {
                        let mut point = *p;
                        let pos = transform.transform_point(Vec3::new(
                            p.position[0] as f64,
                            p.position[1] as f64,
                            p.position[2] as f64,
                        ));
                        let normal = transform.transform_vector(Vec3::new(
                            p.normal[0] as f64,
                            p.normal[1] as f64,
                            p.normal[2] as f64,
                        ));
                        let direction = transform.transform_vector(Vec3::new(
                            p.direction[0] as f64,
                            p.direction[1] as f64,
                            p.direction[2] as f64,
                        ));
                        for (target, v) in [
                            (&mut point.position, pos),
                            (&mut point.normal, normal),
                            (&mut point.direction, direction),
                        ] {
                            target[..3].copy_from_slice(&[v.x as f32, v.y as f32, v.z as f32]);
                        }
                        if min_intensity.is_none_or(|min| point.position[3] >= min) {
                            host.push(point);
                        }
                    }
                    black_box(&host);
                })
            });
            proc.process(&jobs).unwrap();
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .unwrap();
            group.bench_function(BenchmarkId::new(format!("gpu_completed/{label}"), n), |b| {
                b.iter(|| {
                    proc.process(black_box(&jobs)).unwrap();
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(10)),
                        })
                        .unwrap();
                })
            });
            group.bench_function(
                BenchmarkId::new(format!("gpu_selected_host/{label}"), n),
                |b| {
                    b.iter(|| {
                        let frame = proc.process(black_box(&jobs)).unwrap();
                        let ticket = ring
                            .try_enqueue(&[frame.count_source().unwrap()], Default::default())
                            .unwrap()
                            .unwrap();
                        let count = ring
                            .wait(&ticket, Duration::from_secs(10))
                            .unwrap()
                            .channel::<u32>(0)
                            .unwrap()[0] as usize;
                        let ticket = ring
                            .try_enqueue(
                                &[frame.points_source(0..count).unwrap()],
                                Default::default(),
                            )
                            .unwrap()
                            .unwrap();
                        black_box(
                            ring.wait(&ticket, Duration::from_secs(10))
                                .unwrap()
                                .channel::<CloudPoint>(0)
                                .unwrap(),
                        );
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cloud);
criterion_main!(benches);
