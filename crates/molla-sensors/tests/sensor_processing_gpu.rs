#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{
    CameraBatch, CameraChannels, CameraSpec, DeviceSnapshotConfig, DeviceSnapshotPool, ImageJob,
    ImageOp, ImageProcessor, ImageProcessorConfig, SensorScene, SensorSceneOptions,
    TiledCameraConfig,
    camera_post::*,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp},
};
use molla_sim::ModelBuilder;
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn processor(engine: &Engine, pixels: usize, images: u32, operations: u32) -> ImageProcessor {
    ImageProcessor::new(
        engine,
        ImageProcessorConfig {
            max_pixels: pixels.max(1) as u32,
            max_images: images,
            max_operations_per_image: operations,
            max_buffer_bytes: 8 * 1024 * 1024,
        },
    )
    .unwrap()
}

fn ring(engine: &Engine, bytes: u64, slots: usize) -> ReadbackRing {
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

fn source(engine: &Engine, pixels: Vec<u32>) -> DeviceBuffer<u32> {
    DeviceBuffer::from_host(engine, pixels, BufferUsage::StorageReadWrite).unwrap()
}

fn reference(pixels: &mut [u32], ops: &[ImageOp]) -> Vec<(usize, Vec3)> {
    let mut gains = Vec::new();
    for (stage, op) in ops.iter().enumerate() {
        let scale = match *op {
            ImageOp::Exposure(v) => {
                apply_exposure(pixels, v);
                None
            }
            ImageOp::Gamma(v) => {
                apply_gamma(pixels, v);
                None
            }
            ImageOp::LinearToSrgb => {
                apply_gamma_srgb(pixels);
                None
            }
            ImageOp::SrgbToLinear => {
                apply_srgb_to_linear(pixels);
                None
            }
            ImageOp::Reinhard => {
                apply_reinhard_tone_map(pixels);
                None
            }
            ImageOp::ReinhardWhitePoint(v) => {
                apply_reinhard_white_point(pixels, v);
                None
            }
            ImageOp::BrightnessContrast {
                brightness,
                contrast,
            } => {
                apply_brightness_contrast(pixels, brightness, contrast);
                None
            }
            ImageOp::Saturation(v) => {
                apply_saturation(pixels, v);
                None
            }
            ImageOp::GrayWorld => Some(apply_gray_world_wb(pixels)),
            ImageOp::WhitePatch => Some(apply_white_patch_wb(pixels)),
        };
        if let Some(s) = scale {
            gains.push((stage, Vec3::new(s[0] as f64, s[1] as f64, s[2] as f64)));
        }
    }
    gains
}

fn assert_pixels(actual: &[u32], expected: &[u32], tolerance: u32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a >> 24, e >> 24, "alpha at {i}");
        for shift in [0, 8, 16] {
            let av = (a >> shift) & 255;
            let ev = (e >> shift) & 255;
            assert!(
                av.abs_diff(ev) <= tolerance,
                "pixel {i} channel {shift}: {av} != {ev}, tolerance {tolerance}"
            );
        }
    }
}

fn assert_gains(frame: &ReadbackFrame, expected: &[(usize, Vec3)]) {
    for (i, (_, expected)) in expected.iter().enumerate() {
        let actual = frame.channel::<Vec3>(i + 1).unwrap()[0];
        for (a, e) in actual.to_array().into_iter().zip(expected.to_array()) {
            assert!((a - e).abs() <= 2e-5 * e.abs().max(1.0), "gain {a} != {e}");
        }
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn individual_image_operations_match_cpu_and_preserve_alpha_and_ranges() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let operations = [
        ImageOp::Exposure(0.0),
        ImageOp::Exposure(-0.5),
        ImageOp::Exposure(2.0),
        ImageOp::Exposure(f32::MAX),
        ImageOp::Gamma(2.2),
        ImageOp::Gamma(0.0),
        ImageOp::Gamma(1e6),
        ImageOp::Gamma(f32::MAX),
        ImageOp::LinearToSrgb,
        ImageOp::SrgbToLinear,
        ImageOp::Reinhard,
        ImageOp::ReinhardWhitePoint(0.6),
        ImageOp::ReinhardWhitePoint(0.0),
        ImageOp::ReinhardWhitePoint(f32::MAX),
        ImageOp::BrightnessContrast {
            brightness: 0.1,
            contrast: 1.5,
        },
        ImageOp::Saturation(0.0),
        ImageOp::Saturation(2.0),
        ImageOp::Saturation(-1.0),
        ImageOp::GrayWorld,
        ImageOp::WhitePatch,
    ];
    let n = 517;
    let mut values: Vec<u32> = (0..n + 6)
        .map(|i| (i as u32).wrapping_mul(1664525).wrapping_add(1013904223))
        .collect();
    values[3..7].copy_from_slice(&[0, 0xff000000, 0xffffffff, 0x00402010]);
    let input = source(&engine, values.clone());
    let mut proc = processor(&engine, n * operations.len(), operations.len() as u32, 1);
    let jobs: Vec<_> = operations
        .iter()
        .enumerate()
        .map(|(i, op)| ImageJob {
            source: input.readback_source(3..3 + n).unwrap(),
            operations: std::slice::from_ref(op),
            stamp: ReadbackStamp {
                sensor_id: i as u64,
                sample_id: 43,
                scene_revision: 7,
                simulation_time: Some(1.25),
            },
        })
        .collect();
    let mut ring = ring(
        &engine,
        (n as u64 * 4 + 32).next_multiple_of(8),
        operations.len(),
    );
    let out = proc.process(&jobs).unwrap();
    let mut tickets = Vec::new();
    for (i, op) in operations.iter().enumerate() {
        let image = out.image(i as u64).unwrap();
        assert_eq!(image.layout().pixels, i * n..(i + 1) * n);
        let mut sources = vec![image.readback_source().unwrap()];
        let mut expected = values[3..3 + n].to_vec();
        let gains = reference(&mut expected, &[*op]);
        if !gains.is_empty() {
            sources.push(image.white_balance_source(0).unwrap());
        } else {
            assert!(image.white_balance_source(0).is_err());
        }
        let ticket = ring
            .try_enqueue(&sources, image.layout().stamp)
            .unwrap()
            .unwrap();
        tickets.push((ticket, expected, gains));
    }
    for (i, (ticket, expected, gains)) in tickets.into_iter().enumerate().rev() {
        let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        assert_pixels(&frame.channel::<u32>(0).unwrap(), &expected, 1);
        assert_gains(&frame, &gains);
        assert_eq!(frame.stamp.sensor_id, i as u64);
        assert_eq!(frame.stamp.sample_id, 43);
        assert_eq!(frame.stamp.scene_revision, 7);
        assert_eq!(frame.stamp.simulation_time, Some(1.25));
    }
    assert_eq!(input.host().unwrap(), values);
    assert_eq!(proc.stats().submissions, 1);
    assert_eq!(proc.stats().kernel_dispatches, operations.len() as u64 + 4);
    assert_eq!(proc.stats().copied_bytes, (n * operations.len() * 4) as u64);
}

#[test]
#[ignore = "needs a real GPU"]
fn reduction_boundaries_per_image_chains_empty_and_black_images_match_cpu() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let ops = [
        ImageOp::GrayWorld,
        ImageOp::Exposure(0.75),
        ImageOp::WhitePatch,
        ImageOp::Reinhard,
        ImageOp::LinearToSrgb,
        ImageOp::Saturation(0.4),
        ImageOp::BrightnessContrast {
            brightness: -0.03,
            contrast: 0.9,
        },
    ];
    let mut proc = processor(&engine, 70_000, 3, ops.len() as u32);
    let initial_stats = proc.stats();
    let mut ring = ring(&engine, 70_000 * 4 + 64, 3);
    for n in [0usize, 1, 255, 256, 257, 65_537] {
        let values: Vec<_> = (0..n)
            .map(|i| ((i as u32 * 47) & 255) | (((i as u32 * 13) & 127) << 8) | (0x80 << 24))
            .collect();
        let inputs = [
            source(&engine, values.clone()),
            source(&engine, vec![0xff000000; 17]),
            source(&engine, vec![0x004080c0; 31]),
        ];
        let jobs: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| ImageJob {
                source: input.readback_source(0..input.len()).unwrap(),
                operations: &ops,
                stamp: ReadbackStamp {
                    sensor_id: i as u64,
                    sample_id: n as u64,
                    ..Default::default()
                },
            })
            .collect();
        let out = proc.process(&jobs).unwrap();
        let mut tickets = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let image = out.image(i as u64).unwrap();
            let mut expected = input.host().unwrap().to_vec();
            let gains = reference(&mut expected, &ops);
            let sources = [
                image.readback_source().unwrap(),
                image.white_balance_source(0).unwrap(),
                image.white_balance_source(2).unwrap(),
            ];
            assert!(image.white_balance_source(1).is_err());
            assert!(image.white_balance_source(7).is_err());
            tickets.push((
                ring.try_enqueue(&sources, image.layout().stamp)
                    .unwrap()
                    .unwrap(),
                expected,
                gains,
            ));
        }
        for (ticket, expected, gains) in tickets.into_iter().rev() {
            let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
            assert_pixels(&frame.channel::<u32>(0).unwrap(), &expected, 2);
            assert_gains(&frame, &gains);
        }
    }
    assert_eq!(proc.stats().frames, 6);
    assert_eq!(proc.stats().submissions, 6);
    assert_eq!(proc.stats().buffer_bytes, initial_stats.buffer_bytes);
    assert_eq!(proc.stats().buffer_allocations, 4);
    assert_eq!(proc.stats().binding_preparations, 1);
    let stats = proc.stats();
    assert!(proc.process(&[]).unwrap().layouts().is_empty());
    let empty = source(&engine, vec![]);
    let out = proc
        .process(&[ImageJob {
            source: empty.readback_source(0..0).unwrap(),
            operations: &[ImageOp::Exposure(2.0)],
            stamp: Default::default(),
        }])
        .unwrap();
    assert_eq!(out.image(0).unwrap().layout().pixels, 0..0);
    assert_eq!(proc.stats().submissions, stats.submissions);
}

#[test]
#[ignore = "needs a real GPU"]
fn camera_batches_feed_processing_and_snapshot_consumers_without_host_intermediates() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Z, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.shape_color.host_mut().unwrap()[0] = Vec3::new(0.8, 0.4, 0.2);
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let config = TiledCameraConfig {
        width: 7,
        height: 5,
        channels: CameraChannels::COLOR,
        ambient: Vec3::ONE,
        light_color: Vec3::ZERO,
        ..Default::default()
    };
    let specs = [
        CameraSpec {
            sensor_id: 80,
            config,
            pose: Transform::new(Vec3::Z * 2.0, Quat::IDENTITY),
        },
        CameraSpec {
            sensor_id: 2,
            config: TiledCameraConfig {
                width: 9,
                height: 3,
                ambient: Vec3::splat(0.5),
                ..config
            },
            pose: Transform::new(Vec3::Z * 4.0, Quat::IDENTITY),
        },
    ];
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut first = processor(&engine, 64, 2, 2);
    let mut second = processor(&engine, 64, 2, 1);
    let mut pool = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 512,
            max_total_bytes: 512,
        },
    )
    .unwrap();
    let mut ring = ring(&engine, 512, 2);
    let frame = cameras.dispatch_scene(&engine, &scene, &state).unwrap();
    let a = frame.image(80).unwrap();
    let b = frame.image(2).unwrap();
    let jobs = [
        ImageJob {
            source: a.readback_source(CameraChannels::COLOR).unwrap(),
            operations: &[ImageOp::Exposure(0.5), ImageOp::GrayWorld],
            stamp: a.readback_stamp(CameraChannels::COLOR, Some(2.0)).unwrap(),
        },
        ImageJob {
            source: b.readback_source(CameraChannels::COLOR).unwrap(),
            operations: &[ImageOp::WhitePatch, ImageOp::Reinhard],
            stamp: b.readback_stamp(CameraChannels::COLOR, Some(2.0)).unwrap(),
        },
    ];
    let processed = first.process(&jobs).unwrap();
    let a = processed.image(80).unwrap();
    let b = processed.image(2).unwrap();
    let snapshot = pool
        .try_capture(
            &[a.readback_source().unwrap(), b.readback_source().unwrap()],
            a.layout().stamp,
        )
        .unwrap()
        .unwrap();
    let jobs = [
        ImageJob {
            source: snapshot
                .channel::<u32>(0)
                .unwrap()
                .readback_source(0..35)
                .unwrap(),
            operations: &[ImageOp::LinearToSrgb],
            stamp: a.layout().stamp,
        },
        ImageJob {
            source: snapshot
                .channel::<u32>(1)
                .unwrap()
                .readback_source(0..27)
                .unwrap(),
            operations: &[ImageOp::Gamma(2.2)],
            stamp: b.layout().stamp,
        },
    ];
    let processed = second.process(&jobs).unwrap();
    drop(jobs);
    drop(snapshot);
    pool.reset().unwrap();
    let mut tickets = Vec::new();
    for (id, n, raw, ops) in [
        (
            80,
            35,
            [0.8, 0.4, 0.2, 1.0],
            vec![
                ImageOp::Exposure(0.5),
                ImageOp::GrayWorld,
                ImageOp::LinearToSrgb,
            ],
        ),
        (
            2,
            27,
            [0.4, 0.2, 0.1, 1.0],
            vec![ImageOp::WhitePatch, ImageOp::Reinhard, ImageOp::Gamma(2.2)],
        ),
    ] {
        let image = processed.image(id).unwrap();
        let mut expected = vec![pack_rgba8(raw); n];
        reference(&mut expected, &ops);
        tickets.push((
            ring.try_enqueue(&[image.readback_source().unwrap()], image.layout().stamp)
                .unwrap()
                .unwrap(),
            expected,
        ));
    }
    cameras.reset().unwrap();
    for (ticket, expected) in tickets.into_iter().rev() {
        let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        assert_pixels(&frame.channel::<u32>(0).unwrap(), &expected, 1);
        assert_eq!(frame.stamp.simulation_time, Some(2.0));
        assert_eq!(frame.stamp.scene_revision, scene.stamp().geometry_revision);
    }
    assert_eq!(first.stats().submissions, 1);
    assert_eq!(second.stats().submissions, 1);
    assert_eq!(pool.stats().captures, 1);
    assert_eq!(pool.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn invalid_jobs_are_rejected_before_submission_and_copy_only_is_exact() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let input = source(&engine, vec![0x01020304, 0x80000000, 0xffffffff]);
    let mut proc = processor(&engine, 3, 2, 2);
    let mut ring = ring(&engine, 32, 1);
    let out = proc
        .process(&[ImageJob {
            source: input.readback_source(0..3).unwrap(),
            operations: &[],
            stamp: Default::default(),
        }])
        .unwrap();
    let ticket = ring
        .try_enqueue(
            &[out.image(0).unwrap().readback_source().unwrap()],
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let stats = proc.stats();
    for op in [
        ImageOp::Exposure(f32::NAN),
        ImageOp::Gamma(f32::INFINITY),
        ImageOp::ReinhardWhitePoint(f32::NEG_INFINITY),
        ImageOp::Saturation(f32::NAN),
        ImageOp::BrightnessContrast {
            brightness: f32::MAX,
            contrast: f32::MAX,
        },
    ] {
        assert!(
            proc.process(&[
                ImageJob {
                    source: input.readback_source(0..1).unwrap(),
                    operations: &[ImageOp::Exposure(2.0)],
                    stamp: Default::default()
                },
                ImageJob {
                    source: input.readback_source(0..1).unwrap(),
                    operations: &[op],
                    stamp: ReadbackStamp {
                        sensor_id: 1,
                        ..Default::default()
                    }
                }
            ])
            .is_err()
        );
    }
    let other_type =
        DeviceBuffer::from_host(&engine, vec![Vec3::ONE], BufferUsage::StorageReadWrite).unwrap();
    assert!(
        proc.process(&[ImageJob {
            source: other_type.readback_source(0..1).unwrap(),
            operations: &[],
            stamp: Default::default()
        }])
        .is_err()
    );
    assert!(
        proc.process(&[ImageJob {
            source: input.readback_source(0..3).unwrap(),
            operations: &[ImageOp::GrayWorld; 3],
            stamp: Default::default()
        }])
        .is_err()
    );
    assert!(
        proc.process(&[
            ImageJob {
                source: input.readback_source(0..3).unwrap(),
                operations: &[],
                stamp: Default::default()
            },
            ImageJob {
                source: input.readback_source(0..1).unwrap(),
                operations: &[],
                stamp: ReadbackStamp {
                    sensor_id: 1,
                    ..Default::default()
                }
            }
        ])
        .is_err()
    );
    assert!(
        proc.process(&[
            ImageJob {
                source: input.readback_source(0..1).unwrap(),
                operations: &[],
                stamp: Default::default()
            },
            ImageJob {
                source: input.readback_source(0..1).unwrap(),
                operations: &[],
                stamp: Default::default()
            }
        ])
        .is_err()
    );
    for time in [f64::NAN, -1.0, f64::INFINITY] {
        assert!(
            proc.process(&[ImageJob {
                source: input.readback_source(0..1).unwrap(),
                operations: &[],
                stamp: ReadbackStamp {
                    simulation_time: Some(time),
                    ..Default::default()
                }
            }])
            .is_err()
        );
    }
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    let foreign = source(&foreign, vec![1]);
    assert!(
        proc.process(&[ImageJob {
            source: foreign.readback_source(0..1).unwrap(),
            operations: &[],
            stamp: Default::default()
        }])
        .is_err()
    );
    assert_eq!(proc.stats(), stats);
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_pixels(&frame.channel::<u32>(0).unwrap(), input.host().unwrap(), 0);
}

#[test]
#[ignore = "needs a real GPU"]
fn final_uniform_slot_and_pending_frames_do_not_alias_reused_outputs() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut proc = processor(&engine, 1, 64, 64);
    let input = source(&engine, vec![0x7f1e140a]);
    let empty = source(&engine, vec![]);
    let neutral = [ImageOp::Exposure(1.0); 64];
    let mut balanced = neutral;
    balanced[63] = ImageOp::GrayWorld;
    let jobs: Vec<_> = (0..64)
        .map(|i| ImageJob {
            source: if i == 63 {
                input.readback_source(0..1).unwrap()
            } else {
                empty.readback_source(0..0).unwrap()
            },
            operations: if i == 63 { &balanced } else { &neutral },
            stamp: ReadbackStamp {
                sensor_id: i,
                sample_id: 1,
                ..Default::default()
            },
        })
        .collect();
    let mut ring = ring(&engine, 32, 2);
    let out = proc.process(&jobs).unwrap();
    assert_eq!(out.layouts().len(), 64);
    let image = out.image(63).unwrap();
    assert_eq!(image.layout().stages, 4032..4096);
    let old = ring
        .try_enqueue(
            &[
                image.readback_source().unwrap(),
                image.white_balance_source(63).unwrap(),
            ],
            image.layout().stamp,
        )
        .unwrap()
        .unwrap();
    let out = proc
        .process(&[ImageJob {
            source: input.readback_source(0..1).unwrap(),
            operations: &[ImageOp::Exposure(0.0)],
            stamp: ReadbackStamp {
                sensor_id: 63,
                sample_id: 2,
                ..Default::default()
            },
        }])
        .unwrap();
    let image = out.image(63).unwrap();
    assert!(image.white_balance_source(0).is_err());
    assert!(image.white_balance_source(63).is_err());
    let newest = ring
        .try_enqueue(&[image.readback_source().unwrap()], image.layout().stamp)
        .unwrap()
        .unwrap();
    let newest = ring.wait(&newest, Duration::from_secs(10)).unwrap();
    let old = ring.wait(&old, Duration::from_secs(10)).unwrap();
    assert_eq!(newest.channel::<u32>(0).unwrap(), [0x7f000000]);
    assert_eq!(newest.stamp.sample_id, 2);
    let mut expected = input.host().unwrap().to_vec();
    let gains = reference(&mut expected, &balanced);
    assert_pixels(&old.channel::<u32>(0).unwrap(), &expected, 1);
    assert_gains(&old, &gains);
    assert_eq!(old.stamp.sample_id, 1);
    assert_eq!(proc.stats().submissions, 2);
    assert_eq!(proc.stats().kernel_dispatches, 67);
    assert_eq!(proc.stats().buffer_allocations, 4);
}
