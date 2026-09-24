//! CPU, completed GPU and GPU-plus-readback image-operation workloads.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_sensors::{
    ImageJob, ImageOp, ImageProcessor, ImageProcessorConfig,
    camera_post::*,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_storage::BufferUsage;
use std::time::Duration;

fn cpu(pixels: &mut [u32], op: ImageOp) {
    match op {
        ImageOp::Exposure(v) => apply_exposure(pixels, v),
        ImageOp::Gamma(v) => apply_gamma(pixels, v),
        ImageOp::LinearToSrgb => apply_gamma_srgb(pixels),
        ImageOp::SrgbToLinear => apply_srgb_to_linear(pixels),
        ImageOp::Reinhard => apply_reinhard_tone_map(pixels),
        ImageOp::ReinhardWhitePoint(v) => apply_reinhard_white_point(pixels, v),
        ImageOp::BrightnessContrast {
            brightness,
            contrast,
        } => apply_brightness_contrast(pixels, brightness, contrast),
        ImageOp::Saturation(v) => apply_saturation(pixels, v),
        ImageOp::GrayWorld => {
            black_box(apply_gray_world_wb(pixels));
        }
        ImageOp::WhitePatch => {
            black_box(apply_white_patch_wb(pixels));
        }
    }
}

fn bench_images(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("image benchmark requires GPU");
    eprintln!("sensor image adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let operations = [
        ("exposure", ImageOp::Exposure(1.3)),
        ("gamma", ImageOp::Gamma(2.2)),
        ("encode_srgb", ImageOp::LinearToSrgb),
        ("decode_srgb", ImageOp::SrgbToLinear),
        ("reinhard", ImageOp::Reinhard),
        ("white_point", ImageOp::ReinhardWhitePoint(0.8)),
        (
            "brightness_contrast",
            ImageOp::BrightnessContrast {
                brightness: 0.05,
                contrast: 1.2,
            },
        ),
        ("saturation", ImageOp::Saturation(0.8)),
        ("gray_world", ImageOp::GrayWorld),
        ("white_patch", ImageOp::WhitePatch),
    ];
    let mut group = c.benchmark_group("sensor_images");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [1024usize, 65_536, 1_048_576] {
        let values: Vec<_> = (0..n)
            .map(|i| (i as u32).wrapping_mul(1664525).wrapping_add(1013904223))
            .collect();
        let input = DeviceBuffer::from_host(&engine, values.clone(), BufferUsage::StorageReadWrite)
            .unwrap();
        let mut processor = ImageProcessor::new(
            &engine,
            ImageProcessorConfig {
                max_pixels: n as u32,
                max_images: 1,
                max_operations_per_image: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: n as u64 * 4,
                max_total_bytes: n as u64 * 4,
            },
        )
        .unwrap();
        for (name, op) in operations {
            let mut host = values.clone();
            group.bench_function(BenchmarkId::new(format!("cpu/{name}"), n), |b| {
                b.iter(|| {
                    host.copy_from_slice(&values);
                    cpu(black_box(&mut host), black_box(op));
                    black_box(&host);
                })
            });
            let ops = [op];
            let jobs = [ImageJob {
                source: input.readback_source(0..n).unwrap(),
                operations: &ops,
                stamp: ReadbackStamp::default(),
            }];
            processor.process(&jobs).unwrap();
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .unwrap();
            group.bench_function(BenchmarkId::new(format!("gpu_completed/{name}"), n), |b| {
                b.iter(|| {
                    processor.process(black_box(&jobs)).unwrap();
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(10)),
                        })
                        .unwrap();
                })
            });
            group.bench_function(BenchmarkId::new(format!("gpu_host/{name}"), n), |b| {
                b.iter(|| {
                    let frame = processor.process(black_box(&jobs)).unwrap();
                    let image = frame.image(0).unwrap();
                    let ticket = ring
                        .try_enqueue(&[image.readback_source().unwrap()], image.layout().stamp)
                        .unwrap()
                        .unwrap();
                    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
                    black_box(frame.channel::<u32>(0).unwrap());
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_images);
criterion_main!(benches);
