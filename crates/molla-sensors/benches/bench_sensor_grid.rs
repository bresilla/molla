//! Completed GPU spatial grouping, CPU membership/reductions and selected summary collection.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_math::Vec3;
use molla_sensors::{
    CloudGrid, CloudGridCell, CloudGridConfig, CloudGridSpec, CloudPoint,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing},
};
use molla_storage::BufferUsage;
use std::time::Duration;

fn bench_grid(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("grid benchmark requires GPU");
    eprintln!("sensor grid adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let complete = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap()
    };
    let mut group = c.benchmark_group("sensor_grid");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [1024usize, 65_536, 1_000_003] {
        let points: Vec<_> = (0..n)
            .map(|i| CloudPoint::new(Vec3::new((i % 997) as f64 / 997., 0.5, 0.5), 7, i as u64))
            .collect();
        let input = DeviceBuffer::from_host(&engine, points.clone(), BufferUsage::StorageReadWrite)
            .unwrap();
        for (label, cells) in [("dense", 1u32), ("distributed", 1024)] {
            let spec = CloudGridSpec {
                minimum: [0.; 3],
                maximum: [1.; 3],
                dimensions: [cells, 1, 1],
            };
            let mut grid = CloudGrid::new(
                &engine,
                CloudGridConfig {
                    max_points: n as u32,
                    max_cells: cells,
                    max_buffer_bytes: 256 * 1024 * 1024,
                },
            )
            .unwrap();
            let bytes = (cells as u64 * 64 + 24).next_multiple_of(8);
            let mut ring = ReadbackRing::new(
                &engine,
                ReadbackConfig {
                    slots: 1,
                    bytes_per_slot: bytes,
                    max_total_bytes: bytes,
                },
            )
            .unwrap();
            let mut members = vec![Vec::<u32>::new(); cells as usize];
            let mut summaries = vec![CloudGridCell::default(); cells as usize];
            group.bench_function(BenchmarkId::new(format!("cpu/{label}"), n), |b| {
                b.iter(|| {
                    for cell in &mut members {
                        cell.clear();
                    }
                    for (i, p) in black_box(&points).iter().enumerate() {
                        members[((p.position[0] * cells as f32).floor() as usize)
                            .min(cells as usize - 1)]
                        .push(i as u32);
                    }
                    let mut start = 0;
                    for (output, cell) in summaries.iter_mut().zip(&members) {
                        *output = CloudGridCell {
                            header: [start, cell.len() as u32, 0, 0],
                            ..Default::default()
                        };
                        start += cell.len() as u32;
                        if !cell.is_empty() {
                            output.minimum = [f32::MAX; 4];
                            output.maximum = [-f32::MAX; 4];
                            for &i in cell {
                                for j in 0..4 {
                                    let v = points[i as usize].position[j];
                                    output.mean[j] += v / cell.len() as f32;
                                    output.minimum[j] = output.minimum[j].min(v);
                                    output.maximum[j] = output.maximum[j].max(v);
                                }
                            }
                        }
                    }
                    black_box((&members, &summaries));
                })
            });
            grid.process(
                input.readback_source(0..n).unwrap(),
                spec,
                Default::default(),
            )
            .unwrap();
            complete();
            group.bench_function(BenchmarkId::new(format!("gpu_completed/{label}"), n), |b| {
                b.iter(|| {
                    grid.process(
                        input.readback_source(0..n).unwrap(),
                        black_box(spec),
                        Default::default(),
                    )
                    .unwrap();
                    complete();
                })
            });
            group.bench_function(
                BenchmarkId::new(format!("gpu_selected_host/{label}"), n),
                |b| {
                    b.iter(|| {
                        let frame = grid
                            .process(
                                input.readback_source(0..n).unwrap(),
                                black_box(spec),
                                Default::default(),
                            )
                            .unwrap();
                        let ticket = ring
                            .try_enqueue(
                                &[
                                    frame.cells_source(0..cells as usize).unwrap(),
                                    frame.diagnostics_source().unwrap(),
                                ],
                                Default::default(),
                            )
                            .unwrap()
                            .unwrap();
                        let read = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
                        black_box(read.channel::<CloudGridCell>(0).unwrap());
                        black_box(read.channel::<u32>(1).unwrap());
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_grid);
criterion_main!(benches);
