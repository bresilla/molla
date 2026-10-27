#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraBatch, CameraBatchLimits, CameraChannels as Ch,
    CameraSpec, EncodedCameraBatch, ImageProcessor, ImageProcessorConfig, PhysicalOptics,
    SensorCommandBatch, SensorFramePoolConfig, SensorScene, SensorSceneOptions, SensorSubmission,
    ShapeLabel, TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource},
};
use molla_sim::{BodyParams, Model, ModelBuilder, State};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());
const CHANNELS: [Ch; 5] = [
    Ch::DEPTH,
    Ch::COLOR,
    Ch::NORMAL,
    Ch::ALBEDO,
    Ch::SHAPE_INDEX,
];

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 16 << 20,
    }
}

fn fixture(engine: &Engine) -> (Model, State, SensorScene) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.7, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Z,
            Transform::new(-Vec3::Z, Quat::IDENTITY),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .shape_color
        .host_mut()
        .unwrap()
        .copy_from_slice(&[Vec3::X, Vec3::Y]);
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(engine, &model, SensorSceneOptions::default()).unwrap();
    (model, state, scene)
}

fn specs() -> Vec<CameraSpec> {
    let a = TiledCameraConfig {
        width: 3,
        height: 3,
        channels: Ch::ALL,
        ambient: Vec3::ONE,
        light_color: Vec3::ZERO,
        ..Default::default()
    };
    let b = TiledCameraConfig {
        width: 5,
        height: 3,
        optics: PhysicalOptics {
            lens_diameter_m: 0.3,
            aperture_samples: 3,
            focal_distance_m: 1.5,
            ..Default::default()
        },
        ..a
    };
    [
        (u64::MAX, a, 0.0),
        (7, b, 0.2),
        (0x12345678ffffffff, a, 1.2),
        (
            8,
            TiledCameraConfig {
                width: 1,
                height: 1,
                channels: Ch::DEPTH,
                ..a
            },
            0.0,
        ),
    ]
    .map(|(sensor_id, config, x)| CameraSpec {
        sensor_id,
        config,
        pose: Transform::new(Vec3::new(x, 0.0, 3.0), Quat::IDENTITY),
    })
    .to_vec()
}

fn read(engine: &Engine, sources: &[ReadbackSource<'_>]) -> ReadbackFrame {
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1 << 20,
            max_total_bytes: 1 << 20,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(sources, Default::default())
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn collect(
    engine: &Engine,
    submitted: &SensorSubmission,
    out: &EncodedCameraBatch,
    id: u64,
) -> ReadbackFrame {
    let (layout, group) = out.image(id).unwrap();
    let frame = submitted.frame(&group.frame).unwrap();
    let ranges = CHANNELS.map(|channel| {
        if layout.channels.contains(channel) {
            layout.pixels.clone()
        } else {
            0..0
        }
    });
    let d = frame.channel::<Real>(0).unwrap();
    let c = frame.channel::<u32>(1).unwrap();
    let n = frame.channel::<Vec3>(2).unwrap();
    let a = frame.channel::<u32>(3).unwrap();
    let s = frame.channel::<u32>(4).unwrap();
    read(
        engine,
        &[
            d.readback_source(ranges[0].clone()).unwrap(),
            c.readback_source(ranges[1].clone()).unwrap(),
            n.readback_source(ranges[2].clone()).unwrap(),
            a.readback_source(ranges[3].clone()).unwrap(),
            s.readback_source(ranges[4].clone()).unwrap(),
        ],
    )
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

#[test]
#[ignore = "requires GPU"]
fn heterogeneous_groups_and_per_image_consumers_share_one_submission() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let specs = specs();
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut pool = cameras.output_pool(&engine, config(2)).unwrap();
    assert_eq!(pool.group_count(), 3);
    let mut images = ImageProcessor::new(
        &engine,
        ImageProcessorConfig {
            max_pixels: 33,
            max_images: 3,
            max_operations_per_image: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut labels = AnnotationProcessor::new(
        &engine,
        AnnotationConfig {
            max_pixels: 33,
            max_images: 3,
            max_labels: 2,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    labels
        .set_labels(&[
            ShapeLabel {
                shape_id: 0,
                class_id: 20,
            },
            ShapeLabel {
                shape_id: 1,
                class_id: 40,
            },
        ])
        .unwrap();
    let mut pi = images.output_pool(&engine, config(1)).unwrap();
    let mut pl = labels.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, Some(2.5))
        .unwrap()
        .unwrap();
    assert_eq!(out.groups.len(), 3);
    assert_eq!(out.images[2].pixels, 9..18);
    assert!(out.image_job(8, &[]).is_err());
    assert!(out.annotation_job(8).is_err());
    assert!(out.image_job(123, &[]).is_err());
    let image_jobs: Vec<_> = specs[..3]
        .iter()
        .map(|s| out.image_job(s.sensor_id, &[]).unwrap())
        .collect();
    let annotation_jobs: Vec<_> = specs[..3]
        .iter()
        .map(|s| out.annotation_job(s.sensor_id).unwrap())
        .collect();
    for (job, spec) in image_jobs.iter().zip(&specs) {
        assert_eq!(job.stamp.sensor_id, spec.sensor_id);
        assert_eq!(job.stamp.sample_id, 0);
        assert_eq!(job.stamp.scene_revision, scene.stamp().geometry_revision);
        assert_eq!(job.stamp.simulation_time, Some(2.5));
    }
    let processed = images
        .encode_pending(&mut batch, &image_jobs, &mut pi, Default::default())
        .unwrap()
        .unwrap();
    let annotated = labels
        .encode_pending(&mut batch, &annotation_jobs, &mut pl, Default::default())
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 5);
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.input_upload_bytes, 4 * 32);
    assert_eq!(cameras.stats().frames, 0);
    assert_eq!(cameras.stats().encoded_frames, 1);
    assert_eq!(cameras.stats().submissions, 0);
    assert!(cameras.is_current(&out.stamp));
    let mut colors = Vec::new();
    let mut classes = Vec::new();
    for spec in &specs {
        let actual = collect(&engine, &submitted, &out, spec.sensor_id);
        let mut reference = TiledCamera::new(&engine, spec.config, 1).unwrap();
        reference.set_camera_pose(0, spec.pose).unwrap();
        reference.dispatch_scene(&engine, &scene, &state).unwrap();
        let n = (spec.config.width * spec.config.height) as usize;
        let expected = read(
            &engine,
            &CHANNELS.map(|ch| {
                reference
                    .readback_source(
                        ch,
                        0..if spec.config.channels.contains(ch) {
                            n
                        } else {
                            0
                        },
                    )
                    .unwrap()
            }),
        );
        assert_eq!(
            actual.channel::<Real>(0).unwrap(),
            expected.channel::<Real>(0).unwrap()
        );
        assert_eq!(
            actual.channel::<Vec3>(2).unwrap(),
            expected.channel::<Vec3>(2).unwrap()
        );
        for i in [1, 3, 4] {
            assert_eq!(
                actual.channel::<u32>(i).unwrap(),
                expected.channel::<u32>(i).unwrap()
            );
        }
        colors.extend(actual.channel::<u32>(1).unwrap());
        classes.extend(
            actual
                .channel::<u32>(4)
                .unwrap()
                .into_iter()
                .map(|id| if id == 0 { 20 } else { 40 }),
        );
    }
    cameras.reset().unwrap();
    assert!(!cameras.is_current(&out.stamp));
    drop((cameras, pool, images, labels, pi, pl));
    let c = submitted
        .frame(&processed.frame)
        .unwrap()
        .channel::<u32>(0)
        .unwrap();
    let l = submitted
        .frame(&annotated.frame)
        .unwrap()
        .channel::<u32>(1)
        .unwrap();
    let cpu = read(
        &engine,
        &[
            c.readback_source(0..33).unwrap(),
            l.readback_source(0..33).unwrap(),
        ],
    );
    assert_eq!(cpu.channel::<u32>(0).unwrap(), colors);
    assert_eq!(cpu.channel::<u32>(1).unwrap(), classes);
    assert!(
        collect(&engine, &submitted, &out, u64::MAX)
            .channel::<Real>(0)
            .unwrap()
            .iter()
            .all(|v| *v > 0.0)
    );
}

#[test]
#[ignore = "requires GPU"]
fn later_group_backpressure_cancellation_and_reconfiguration_are_transactional() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let specs = specs()[..2].to_vec();
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut pool = cameras.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let retained_later = submitted.frame(&first.groups[1].frame).unwrap().clone();
    drop(submitted);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 0);
    let before = cameras.group_stats(0).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        cameras
            .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
            .unwrap()
            .is_none()
    );
    assert!(batch.submit().unwrap().index.is_none());
    assert_eq!(cameras.group_stats(0).unwrap(), before);
    assert_eq!(pool.group_stats(0).unwrap().reserved_frames, 1);
    assert_eq!(pool.group_stats(1).unwrap().reserved_frames, 1);
    drop(retained_later);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let canceled = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    assert_eq!(canceled.stamp.sample_id(), 1);
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 1);
    pool.reset().unwrap();
    cameras.reset().unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let third = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    assert_eq!(third.stamp.sample_id(), 2);
    let submitted = batch.submit().unwrap();
    assert!(submitted.frame(&canceled.groups[0].frame).is_err());
    drop(submitted);
    drain(&engine);
    cameras.reconfigure(&engine, &specs).unwrap();
    assert!(!cameras.is_current(&third.stamp));
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let compatible = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    assert_eq!(compatible.stamp.sample_id(), 3);
    drop(batch);
    let mut changed = specs.clone();
    changed[1].config.width += 1;
    cameras.reconfigure(&engine, &changed).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        cameras
            .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
            .is_err()
    );
    assert!(batch.submit().unwrap().index.is_none());
    let mut replacement = cameras.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut replacement, None)
        .unwrap()
        .unwrap();
    assert_eq!(out.stamp.sample_id(), 4);
    assert!(
        batch
            .encode_with_pending(&[&out.groups[1].frame], |_, _| Err(
                molla_core::Error::Solver("poison".into())
            ))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(replacement.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn maximum_camera_and_group_counts_preserve_ids_and_preflight_output_capacity() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let cfg = TiledCameraConfig {
        width: 1,
        height: 1,
        channels: Ch::SHAPE_INDEX,
        ..Default::default()
    };
    let specs: Vec<_> = (0..1024)
        .map(|i| CameraSpec {
            sensor_id: (i as u64) << 32 | 7,
            config: cfg,
            pose: Transform::new(Vec3::Z * 3.0, Quat::IDENTITY),
        })
        .collect();
    let limits = CameraBatchLimits {
        max_cameras: 1024,
        max_groups: 64,
        max_total_pixels: 1024,
    };
    let mut cameras = CameraBatch::new(&engine, &specs, limits).unwrap();
    let mut pool = cameras.output_pool(&engine, config(1)).unwrap();
    let mut labels = AnnotationProcessor::new(
        &engine,
        AnnotationConfig {
            max_pixels: 1024,
            max_images: 1024,
            max_labels: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    labels
        .set_labels(&[ShapeLabel {
            shape_id: 0,
            class_id: 99,
        }])
        .unwrap();
    let mut pl = labels.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = cameras
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    let jobs: Vec<_> = specs
        .iter()
        .map(|spec| out.annotation_job(spec.sensor_id).unwrap())
        .collect();
    let annotated = labels
        .encode_pending(&mut batch, &jobs, &mut pl, Default::default())
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 2);
    let classes = submitted
        .frame(&annotated.frame)
        .unwrap()
        .channel::<u32>(1)
        .unwrap();
    assert_eq!(
        read(&engine, &[classes.readback_source(0..1024).unwrap()])
            .channel::<u32>(0)
            .unwrap(),
        vec![99; 1024]
    );
    for (i, image) in out.images.iter().enumerate() {
        assert_eq!(image.sensor_id, (i as u64) << 32 | 7);
        assert_eq!(image.pixels, i..i + 1);
    }
    let mut grouped: Vec<_> = specs[..64].to_vec();
    for (i, spec) in grouped.iter_mut().enumerate() {
        spec.config.channels = Ch::NONE;
        spec.config.max_distance = 10.0 + i as f32;
    }
    let mut many = CameraBatch::new(&engine, &grouped, limits).unwrap();
    let mut pm = many.output_pool(&engine, config(1)).unwrap();
    let mut one = TiledCamera::new(
        &engine,
        TiledCameraConfig {
            channels: Ch::NONE,
            ..cfg
        },
        1,
    )
    .unwrap();
    let mut po = one.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let first = one
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &mut po,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert!(
        many.encode_scene(&mut batch, &engine, &scene, &state, &mut pm, None)
            .is_err()
    );
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 1);
    assert!(submitted.frame(&first.frame).is_ok());
    for i in 0..64 {
        assert_eq!(pm.group_stats(i).unwrap().reserved_frames, 0);
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let all = many
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pm, None)
        .unwrap()
        .unwrap();
    assert_eq!(all.stamp.sample_id(), 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 64);
    assert_eq!(submitted.stats.kernel_dispatches, 0);
    assert_eq!(submitted.stats.output_clears, 320);
}

#[test]
#[ignore = "requires GPU"]
fn invalid_later_group_budgets_foreign_pools_empty_batches_and_loss_are_explicit() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (_, foreign_state, _) = fixture(&foreign);
    let (_model, state, scene) = fixture(&engine);
    let mut specs = specs()[..2].to_vec();
    specs[0].config.channels = Ch::NONE;
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let bytes = specs
        .iter()
        .map(|s| s.config.width as u64 * s.config.height as u64 * 32 * 2)
        .sum::<u64>();
    assert!(
        cameras
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 2,
                    max_total_bytes: bytes - 1
                }
            )
            .is_err()
    );
    let mut pool = cameras
        .output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 2,
                max_total_bytes: bytes,
            },
        )
        .unwrap();
    let mut other_pool = CameraBatch::new(&engine, &specs, Default::default())
        .unwrap()
        .output_pool(&engine, config(1))
        .unwrap();
    for case in 0..3 {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let source = if case == 0 { &foreign_state } else { &state };
        let output = if case == 2 {
            &mut other_pool
        } else {
            &mut pool
        };
        assert!(
            cameras
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    source,
                    output,
                    if case == 1 { Some(f64::NAN) } else { None }
                )
                .is_err()
        );
        assert!(batch.submit().unwrap().index.is_none());
        assert_eq!(pool.group_stats(0).unwrap().reserved_frames, 0);
        assert_eq!(pool.group_stats(1).unwrap().reserved_frames, 0);
        assert_eq!(cameras.group_stats(0).unwrap().encoded_frames, 0);
    }
    let mut empty = CameraBatch::new(&engine, &[], Default::default()).unwrap();
    let mut pe = empty
        .output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 0,
            },
        )
        .unwrap();
    assert_eq!(pe.group_count(), 0);
    assert_eq!(pe.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = empty
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pe, None)
        .unwrap()
        .unwrap();
    assert!(out.groups.is_empty());
    assert!(out.images.is_empty());
    assert!(out.image_job(0, &[]).is_err());
    assert!(batch.submit().unwrap().index.is_none());
    assert!(cameras.output_pool(&foreign, config(1)).is_err());
    let mut foreign_pool = CameraBatch::new(&foreign, &specs, Default::default())
        .unwrap()
        .output_pool(&foreign, config(1))
        .unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            cameras
                .encode_scene(&mut batch, &engine, &scene, &state, &mut foreign_pool, None)
                .is_err()
        );
        assert!(batch.submit().unwrap().index.is_none());
    }
    {
        let mut batch = SensorCommandBatch::new(&foreign).unwrap();
        assert!(
            cameras
                .encode_scene(&mut batch, &foreign, &scene, &state, &mut pool, None)
                .is_err()
        );
    }
    for group in 0..2 {
        assert_eq!(foreign_pool.group_stats(group).unwrap().reserved_frames, 0);
        assert_eq!(pool.group_stats(group).unwrap().reserved_frames, 0);
    }
    let mut pending = SensorCommandBatch::new(&engine).unwrap();
    cameras
        .encode_scene(&mut pending, &engine, &scene, &state, &mut pool, None)
        .unwrap()
        .unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    engine.destroy_device().unwrap();
    assert!(
        empty
            .encode_scene(&mut batch, &engine, &scene, &state, &mut pe, None)
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert!(pe.available_slots().is_err());
    assert!(pe.reset().is_err());
    assert!(pending.submit().is_err());
    assert!(pool.available_slots().is_err());
    assert!(pool.reset().is_err());
    let mut batch = SensorCommandBatch::new(&engine);
    if let Ok(batch) = batch.as_mut() {
        assert!(
            cameras
                .encode_scene(batch, &engine, &scene, &state, &mut pool, None)
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn warm_groups_preserve_independent_moving_images_and_fixed_slot_resources() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let specs = specs();
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut pool = cameras.output_pool(&engine, config(4)).unwrap();
    for round in 0..6 {
        let mut held = Vec::new();
        for slot in 0..4 {
            let sample = round * 4 + slot;
            let z = 3.0 + sample as f64 * 0.05;
            for spec in &specs {
                let mut pose = spec.pose;
                pose.position.z = z;
                cameras.set_camera_pose(spec.sensor_id, pose).unwrap();
            }
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let out = cameras
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &mut pool,
                    Some(sample as f64),
                )
                .unwrap()
                .unwrap();
            assert_eq!(out.stamp.sample_id(), sample);
            let submitted = batch.submit().unwrap();
            assert_eq!(submitted.stats.encoded_frames, 3);
            assert_eq!(submitted.stats.kernel_dispatches, 22);
            assert_eq!(submitted.stats.output_clears, 15);
            assert_eq!(submitted.stats.input_upload_bytes, 4 * 32);
            assert_eq!(submitted.stats.submissions, 1);
            held.push((submitted, out, z));
        }
        pool.reset().unwrap();
        cameras.reset().unwrap();
        assert_eq!(pool.available_slots().unwrap(), 0);
        let before: Vec<_> = (0..3).map(|i| cameras.group_stats(i).unwrap()).collect();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            cameras
                .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, None)
                .unwrap()
                .is_none()
        );
        assert!(batch.submit().unwrap().index.is_none());
        for (i, before) in before.into_iter().enumerate() {
            assert_eq!(cameras.group_stats(i).unwrap(), before);
        }
        for (submitted, out, z) in &held {
            assert!(!cameras.is_current(&out.stamp));
            for spec in &specs {
                let cpu = collect(&engine, submitted, out, spec.sensor_id);
                let center = (spec.config.width * spec.config.height / 2) as usize;
                let x = spec.pose.position.x;
                let on_sphere = x.abs() < 0.7;
                let expected = if on_sphere {
                    z - (0.7f64.powi(2) - x * x).sqrt()
                } else {
                    z + 1.0
                };
                assert!((cpu.channel::<Real>(0).unwrap()[center] - expected).abs() < 2e-5);
                if spec.config.channels.contains(Ch::SHAPE_INDEX) {
                    assert_eq!(
                        cpu.channel::<u32>(4).unwrap()[center],
                        if on_sphere { 0 } else { 1 }
                    );
                }
                assert_eq!(
                    out.readback_stamp(spec.sensor_id).unwrap().sensor_id,
                    spec.sensor_id
                );
            }
        }
        drop(held);
        drain(&engine);
    }
    assert_eq!(pool.available_slots().unwrap(), 4);
    assert_eq!(cameras.stats().encoded_frames, 24);
    assert_eq!(cameras.stats().frames, 0);
    assert_eq!(cameras.stats().submissions, 0);
    for i in 0..3 {
        let pool = pool.group_stats(i).unwrap();
        assert_eq!(pool.buffer_allocations, 20);
        assert_eq!(pool.reserved_frames, 24);
        assert_eq!(pool.backpressure, 6);
        let stats = cameras.group_stats(i).unwrap();
        assert_eq!(stats.encoded_frames, 24);
        assert_eq!(stats.rays_preparations, 1);
        assert_eq!(stats.trace_preparations, 4);
        assert_eq!(stats.accumulator_preparations, if i == 1 { 4 } else { 0 });
        assert_eq!(
            stats.accumulator_buffer_allocations,
            if i == 1 { 2 } else { 0 }
        );
        let query = cameras.query_stats(i).unwrap();
        assert_eq!(query.primitive_preparations, 0);
        assert_eq!(query.scene_builds, 1);
        assert_eq!(query.scene_preparations, 1);
        assert_eq!(query.refit_preparations, 1);
    }
}
