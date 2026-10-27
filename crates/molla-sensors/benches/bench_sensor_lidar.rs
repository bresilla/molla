//! Completed terrestrial scans with CPU or GPU ray layout on a static plane.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{Lidar, LidarBatch, LidarPattern, LidarScan, RayHit, ScanMetadata};
use molla_sim::ModelBuilder;
use std::time::Duration;

fn bench_lidar(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("sensor benchmark requires GPU");
    eprintln!("sensor lidar adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mut group = c.benchmark_group("sensor_lidar_layout_gpu");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for (rows, cols) in [(10, 100), (100, 100), (100, 1000), (1000, 1000)] {
        let rays = rows * cols;
        let mut scan = ScanMetadata::new(Vec3::Y * 10.0, rows, cols);
        scan.forward = Vec3::NEG_Y;
        scan.up = Vec3::Z;
        scan.theta_range = (0.6, 2.5);
        scan.phi_range = (-0.3, 0.3);
        let scans = [LidarScan {
            sensor_id: 1,
            pattern: LidarPattern::Terrestrial(scan),
        }];
        let mut legacy = Lidar::new(&engine, rays).unwrap();
        let mut batch = LidarBatch::new(&engine, 1, rays).unwrap();
        let mut collector = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: rays as u64 * 40,
                max_total_bytes: rays as u64 * 40,
            },
        )
        .unwrap();
        for mode in [
            "cpu_layout_host",
            "gpu_layout_host",
            "gpu_layout_device_completed",
        ] {
            group.bench_with_input(BenchmarkId::new(mode, rays), &rays, |b, _| {
                b.iter(|| {
                    if mode == "cpu_layout_host" {
                        black_box(legacy.cast(&engine, &model, &state, &scan).unwrap().hits);
                    } else {
                        let view = batch.dispatch(&engine, &model, &state, &scans).unwrap();
                        if mode == "gpu_layout_host" {
                            let n = view.hits.ray_count;
                            let sources = [
                                view.hits.distance.readback_source(0..n).unwrap(),
                                view.hits.shape.readback_source(0..n).unwrap(),
                                view.hits.point.readback_source(0..n).unwrap(),
                                view.hits.normal.readback_source(0..n).unwrap(),
                            ];
                            let ticket = collector
                                .try_enqueue(&sources, ReadbackStamp::default())
                                .unwrap()
                                .unwrap();
                            let frame = collector.wait(&ticket, Duration::from_secs(30)).unwrap();
                            let hits: Vec<_> = frame
                                .channel_iter::<Real>(0)
                                .unwrap()
                                .zip(frame.channel_iter::<i32>(1).unwrap())
                                .zip(frame.channel_iter::<Vec3>(2).unwrap())
                                .zip(frame.channel_iter::<Vec3>(3).unwrap())
                                .map(|(((t, shape), point), normal)| RayHit {
                                    shape_id: (shape >= 0).then_some(shape as u32),
                                    body_id: BodyId::NONE,
                                    t: if shape >= 0 { t } else { Real::INFINITY },
                                    point,
                                    normal,
                                })
                                .collect();
                            black_box(hits);
                        } else {
                            black_box(view);
                            device
                                .poll(wgpu::PollType::Wait {
                                    submission_index: None,
                                    timeout: Some(Duration::from_secs(30)),
                                })
                                .unwrap();
                        }
                    }
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_lidar);
criterion_main!(benches);
