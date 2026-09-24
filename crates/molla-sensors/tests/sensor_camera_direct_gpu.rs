#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraChannels as Ch, ImageProcessor,
    ImageProcessorConfig, PendingAnnotationJob, PendingImageJob, PhysicalOptics,
    SensorCommandBatch, SensorFrame, SensorFramePoolConfig, SensorScene, SensorSceneOptions,
    ShapeLabel, TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource, ReadbackStamp},
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

fn pools(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 8 << 20,
    }
}
fn stamp(id: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: id,
        sample_id: u64::MAX,
        scene_revision: 123,
        simulation_time: Some(1.0),
    }
}

fn fixture(engine: &Engine) -> (Model, State, SensorScene) {
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::new(0.8, 0.65, 0.0)));
    builder
        .add_shape_sphere(body, 0.6, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
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

fn config(width: u32, height: u32, samples: u32) -> TiledCameraConfig {
    TiledCameraConfig {
        width,
        height,
        channels: Ch::ALL,
        optics: PhysicalOptics {
            lens_diameter_m: 0.5,
            focal_distance_m: 1.8,
            aperture_samples: samples,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn camera(engine: &Engine, config: TiledCameraConfig, count: usize) -> TiledCamera {
    let mut c = TiledCamera::new(engine, config, count).unwrap();
    for i in 0..count {
        let target = Vec3::X * (i as Real * 4.0);
        c.set_camera_look_at(i, target + Vec3::Y * 3.0, target, Vec3::Z)
            .unwrap();
    }
    c
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
    let ticket = ring.try_enqueue(sources, stamp(0)).unwrap().unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn collect(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let d = frame.channel::<Real>(0).unwrap();
    let c = frame.channel::<u32>(1).unwrap();
    let n = frame.channel::<Vec3>(2).unwrap();
    let a = frame.channel::<u32>(3).unwrap();
    let s = frame.channel::<u32>(4).unwrap();
    read(
        engine,
        &[
            d.readback_source(0..d.len()).unwrap(),
            c.readback_source(0..c.len()).unwrap(),
            n.readback_source(0..n.len()).unwrap(),
            a.readback_source(0..a.len()).unwrap(),
            s.readback_source(0..s.len()).unwrap(),
        ],
    )
}

fn legacy(
    engine: &Engine,
    scene: &SensorScene,
    state: &State,
    c: &mut TiledCamera,
) -> ReadbackFrame {
    c.dispatch_scene(engine, scene, state).unwrap();
    let n = c.config().width as usize * c.config().height as usize * c.num_cameras() as usize;
    read(
        engine,
        &CHANNELS.map(|channel| c.readback_source(channel, 0..n).unwrap()),
    )
}

fn same(a: &ReadbackFrame, b: &ReadbackFrame, mask: Ch) {
    assert_eq!(
        a.channel::<Real>(0).unwrap(),
        if mask.contains(Ch::DEPTH) {
            b.channel::<Real>(0).unwrap()
        } else {
            vec![]
        }
    );
    assert_eq!(
        a.channel::<Vec3>(2).unwrap(),
        if mask.contains(Ch::NORMAL) {
            b.channel::<Vec3>(2).unwrap()
        } else {
            vec![]
        }
    );
    for i in [1, 3, 4] {
        assert_eq!(
            a.channel::<u32>(i).unwrap(),
            if mask.contains(CHANNELS[i]) {
                b.channel::<u32>(i).unwrap()
            } else {
                vec![]
            }
        );
    }
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
fn independent_aperture_cameras_feed_images_and_annotations_without_readback() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let ca = config(9, 7, 7);
    let cb = config(13, 5, 1);
    let mut a = camera(&engine, ca, 2);
    let mut b = camera(&engine, cb, 1);
    let mut images = ImageProcessor::new(
        &engine,
        ImageProcessorConfig {
            max_pixels: 126,
            max_images: 1,
            max_operations_per_image: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut annotations = AnnotationProcessor::new(
        &engine,
        AnnotationConfig {
            max_pixels: 63,
            max_images: 1,
            max_labels: 2,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    annotations
        .set_labels(&[
            ShapeLabel {
                shape_id: 0,
                class_id: 7,
            },
            ShapeLabel {
                shape_id: 1,
                class_id: 9,
            },
        ])
        .unwrap();
    let mut pa = a.output_pool(&engine, pools(1)).unwrap();
    let mut pb = b.output_pool(&engine, pools(1)).unwrap();
    let mut pi = images.output_pool(&engine, pools(1)).unwrap();
    let mut pn = annotations.output_pool(&engine, pools(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let ra = a
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pa, stamp(1))
        .unwrap()
        .unwrap();
    let rb = b
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pb, stamp(2))
        .unwrap()
        .unwrap();
    let ri = images
        .encode_pending(
            &mut batch,
            &[PendingImageJob {
                frame: &ra.frame,
                channel: 1,
                pixels: 0..126,
                operations: &[],
                stamp: stamp(3),
            }],
            &mut pi,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    let rn = annotations
        .encode_pending(
            &mut batch,
            &[PendingAnnotationJob {
                frame: &ra.frame,
                channel: 4,
                pixels: 0..63,
                width: 9,
                height: 7,
                stamp: stamp(4),
            }],
            &mut pn,
            stamp(4),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().input_upload_bytes, 3 * 32);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.encoded_frames, 4);
    assert_eq!(ra.scene_stamp, scene.stamp());
    assert_eq!(ra.camera_count, 2);
    let fa = submitted.frame(&ra.frame).unwrap().clone();
    let fb = submitted.frame(&rb.frame).unwrap().clone();
    let fi = submitted.frame(&ri.frame).unwrap().clone();
    let fn_ = submitted.frame(&rn.frame).unwrap().clone();
    assert_eq!(a.dispatch_stats().submissions, 0);
    assert_eq!(a.dispatch_stats().accumulation_passes, 7);
    assert_eq!(a.channel_scene_stamp(Ch::COLOR).unwrap(), None);
    drop((submitted, a, b, images, annotations, pa, pb, pi, pn));
    let actual = collect(&engine, &fa);
    let mut old_a = camera(&engine, ca, 2);
    let expected = legacy(&engine, &scene, &state, &mut old_a);
    same(&actual, &expected, Ch::ALL);
    same(
        &collect(&engine, &fb),
        &legacy(&engine, &scene, &state, &mut camera(&engine, cb, 1)),
        Ch::ALL,
    );
    let mut pinhole = ca;
    pinhole.optics.aperture_samples = 1;
    let pinhole = legacy(&engine, &scene, &state, &mut camera(&engine, pinhole, 2));
    for channel in [0, 2, 3, 4] {
        match channel {
            0 => assert_eq!(
                actual.channel::<Real>(0).unwrap(),
                pinhole.channel::<Real>(0).unwrap()
            ),
            2 => assert_eq!(
                actual.channel::<Vec3>(2).unwrap(),
                pinhole.channel::<Vec3>(2).unwrap()
            ),
            _ => assert_eq!(
                actual.channel::<u32>(channel).unwrap(),
                pinhole.channel::<u32>(channel).unwrap()
            ),
        }
    }
    assert!((actual.channel::<Real>(0).unwrap()[31] - 3.0).abs() < 1e-5);
    let channel = fi.channel::<u32>(0).unwrap();
    assert_eq!(
        read(&engine, &[channel.readback_source(0..126).unwrap()])
            .channel::<u32>(0)
            .unwrap(),
        actual.channel::<u32>(1).unwrap()
    );
    let channel = fn_.channel::<u32>(1).unwrap();
    let classes = read(&engine, &[channel.readback_source(0..63).unwrap()])
        .channel::<u32>(0)
        .unwrap();
    let ids = actual.channel::<u32>(4).unwrap();
    assert_eq!(
        classes,
        ids[..63]
            .iter()
            .map(|id| if *id == 0 { 7 } else { 9 })
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires GPU"]
fn every_channel_mask_has_explicit_empty_ranges_and_center_geometry() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let mut cfg = config(3, 3, 5);
    let mut camera = camera(&engine, cfg, 1);
    let expected = legacy(&engine, &scene, &state, &mut camera);
    let mut pool = camera.output_pool(&engine, pools(1)).unwrap();
    for mask in 0..32 {
        cfg.channels = Ch(mask);
        camera.set_config(cfg).unwrap();
        let before = camera.dispatch_stats();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let out = camera
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                &state,
                &mut pool,
                stamp(mask as u64),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(out.channels, cfg.channels);
        assert_eq!(
            submitted.stats.kernel_dispatches,
            if mask == 0 {
                0
            } else if cfg.channels.contains(Ch::COLOR) {
                21
            } else {
                3
            }
        );
        same(
            &collect(&engine, submitted.frame(&out.frame).unwrap()),
            &expected,
            cfg.channels,
        );
        assert_eq!(
            camera.dispatch_stats().accumulation_passes - before.accumulation_passes,
            if cfg.channels.contains(Ch::COLOR) {
                5
            } else {
                0
            }
        );
        drop(submitted);
        drain(&engine);
    }
}

#[test]
#[ignore = "requires GPU"]
fn warm_aperture_outputs_keep_history_across_slot_reuse_and_parameter_changes() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = fixture(&engine);
    let mut cfg = config(9, 7, 2);
    let mut c = camera(&engine, cfg, 1);
    let mut reference = camera(&engine, cfg, 1);
    let mut pool = c.output_pool(&engine, pools(4)).unwrap();
    for round in 0..6 {
        let mut held = Vec::new();
        for slot in 0..4 {
            cfg.optics.aperture_samples = 2 + round + slot;
            cfg.optics.focal_distance_m = 1.0 + slot as f32;
            cfg.max_distance = if round == 5 { 0.5 } else { 100.0 };
            c.set_config(cfg).unwrap();
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let out = c
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &mut pool,
                    stamp(slot as u64),
                )
                .unwrap()
                .unwrap();
            held.push((
                batch.submit().unwrap().frame(&out.frame).unwrap().clone(),
                cfg,
            ));
        }
        pool.reset().unwrap();
        assert_eq!(pool.available_slots().unwrap(), 0);
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            c.encode_scene(&mut batch, &engine, &scene, &state, &mut pool, stamp(0))
                .unwrap()
                .is_none()
        );
        assert!(batch.submit().unwrap().index.is_none());
        for (frame, cfg) in &held {
            assert!(!pool.is_current(frame));
            reference.set_config(*cfg).unwrap();
            same(
                &collect(&engine, frame),
                &legacy(&engine, &scene, &state, &mut reference),
                Ch::ALL,
            );
        }
        drop(held);
        drain(&engine);
    }
    assert_eq!(pool.stats().buffer_allocations, 20);
    assert_eq!(c.dispatch_stats().rays_preparations, 1);
    assert_eq!(c.dispatch_stats().trace_preparations, 4);
    assert_eq!(c.dispatch_stats().accumulator_preparations, 4);
    assert_eq!(c.dispatch_stats().accumulator_buffer_allocations, 2);
    assert_eq!(c.dispatch_stats().encoded_frames, 24);
    assert_eq!(c.dispatch_stats().submissions, 0);
}

#[test]
#[ignore = "requires GPU"]
fn cancellation_bad_inputs_empty_scene_and_device_loss_do_not_publish_old_images() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (_, foreign_state, _) = fixture(&foreign);
    let (model, state, mut scene) = fixture(&engine);
    let cfg = config(3, 3, 3);
    let mut c = camera(&engine, cfg, 1);
    let mut pool = c.output_pool(&engine, pools(1)).unwrap();
    assert!(c.output_pool(&foreign, pools(1)).is_err());
    assert!(
        c.output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 9 * 32 - 1
            }
        )
        .is_err()
    );
    let dirty = model.state().unwrap();
    for source in [&foreign_state, &dirty] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            c.encode_scene(&mut batch, &engine, &scene, source, &mut pool, stamp(0))
                .is_err()
        );
        assert!(batch.submit().unwrap().index.is_none());
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let mut bad = stamp(0);
    bad.simulation_time = Some(f64::NAN);
    assert!(
        c.encode_scene(&mut batch, &engine, &scene, &state, &mut pool, bad)
            .is_err()
    );
    assert!(batch.submit().unwrap().index.is_none());
    assert_eq!(pool.stats().reserved_frames, 0);
    assert_eq!(c.dispatch_stats().trace_preparations, 0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let canceled = c
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, stamp(1))
        .unwrap()
        .unwrap();
    drop(batch);
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = c
        .encode_scene(&mut batch, &engine, &scene, &state, &mut pool, stamp(2))
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert!(submitted.frame(&canceled.frame).is_err());
    let old = collect(&engine, submitted.frame(&out.frame).unwrap());
    assert!(old.channel::<Real>(0).unwrap().iter().all(|v| *v > 0.0));
    drop(submitted);
    drain(&engine);
    let empty_model = ModelBuilder::new().build().unwrap();
    scene.replace_model(&engine, &empty_model).unwrap();
    let empty_state = empty_model.state().unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = c
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &empty_state,
            &mut pool,
            stamp(3),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let empty = collect(&engine, submitted.frame(&out.frame).unwrap());
    assert_eq!(empty.channel::<Real>(0).unwrap(), vec![-1.0; 9]);
    assert_eq!(empty.channel::<u32>(4).unwrap(), vec![u32::MAX; 9]);
    assert_eq!(empty.channel::<u32>(1).unwrap(), vec![0; 9]);
    assert_eq!(empty.channel::<Vec3>(2).unwrap(), vec![Vec3::ZERO; 9]);
    drop(submitted);
    drain(&engine);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let out = c
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &empty_state,
            &mut pool,
            stamp(4),
        )
        .unwrap()
        .unwrap();
    assert!(
        batch
            .encode_with_pending(&[&out.frame], |_, _| Err(molla_core::Error::Solver(
                "poison".into()
            )))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    engine.destroy_device().unwrap();
    assert!(
        c.encode_scene(
            &mut batch,
            &engine,
            &scene,
            &empty_state,
            &mut pool,
            stamp(5)
        )
        .is_err()
    );
    assert!(batch.submit().is_err());
}

#[test]
#[ignore = "requires GPU"]
fn moving_triangle_and_terrain_replacement_preserve_leased_geometry_and_strict_range() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y));
    builder
        .add_shape_triangle(
            body,
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Transform::IDENTITY,
        )
        .unwrap();
    let model = builder.build().unwrap();
    let mut scene =
        SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut terrain_builder = ModelBuilder::new();
    let terrain = terrain_builder.add_heightfield(molla_sim::Heightfield {
        nx: 2,
        nz: 2,
        cell_size: 1.0,
        cell_size_z: 1.0,
        origin: Vec3::new(-0.5, 3.0, -0.5),
        heights: vec![0.0; 4],
    });
    terrain_builder
        .add_shape_heightfield(BodyId::NONE, terrain, Transform::IDENTITY)
        .unwrap();
    let terrain_model = terrain_builder.build().unwrap();
    let empty_state = terrain_model.state().unwrap();
    let mut c = camera(&engine, config(1, 1, 4), 1);
    c.set_camera_look_at(0, Vec3::Y * 6.0, Vec3::ZERO, Vec3::Z)
        .unwrap();
    let mut pool = c.output_pool(&engine, pools(4)).unwrap();
    let mut held = Vec::new();
    for version in 0..4 {
        if version == 1 {
            state.body_q.host_mut().unwrap()[0].position.y = 2.0;
            state
                .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
                .unwrap();
        }
        if version == 2 {
            scene.replace_model(&engine, &terrain_model).unwrap();
        }
        if version == 3 {
            let mut cfg = *c.config();
            cfg.max_distance = 3.0;
            c.set_config(cfg).unwrap();
        }
        let input = if version < 2 { &state } else { &empty_state };
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let out = c
            .encode_scene(
                &mut batch,
                &engine,
                &scene,
                input,
                &mut pool,
                stamp(version),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.kernel_dispatches, 25);
        held.push((
            submitted.frame(&out.frame).unwrap().clone(),
            out.scene_stamp,
        ));
    }
    assert_eq!(held[0].1, held[1].1);
    assert_ne!(held[1].1, held[2].1);
    assert_eq!(held[2].1, held[3].1);
    assert_eq!(c.mesh_build_count(), 0);
    pool.reset().unwrap();
    drop((c, pool, scene));
    for (version, (frame, _)) in held.iter().enumerate() {
        let cpu = collect(&engine, frame);
        let depth = cpu.channel::<Real>(0).unwrap()[0];
        if version == 3 {
            assert_eq!(depth, -1.0);
        } else {
            assert!((depth - [5.0, 4.0, 3.0][version]).abs() < 2e-5);
        }
        assert_eq!(
            cpu.channel::<u32>(4).unwrap(),
            if version == 3 {
                vec![u32::MAX]
            } else {
                vec![0]
            }
        );
        let normal = cpu.channel::<Vec3>(2).unwrap()[0];
        assert!((normal - if version == 3 { Vec3::ZERO } else { Vec3::Y }).length() < 2e-5);
    }
}
