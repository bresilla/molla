#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{CameraAccumulateKernel, CameraRaysKernel, CameraTraceKernel};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::readback::{
    DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp,
};
use molla_sensors::{CameraChannels as Ch, PhysicalOptics, TiledCamera, TiledCameraConfig};
use molla_sim::{BodyParams, Model, ModelBuilder, State};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());
const W: u32 = 13;
const H: u32 = 11;
const CAMERAS: u32 = 2;
const PIXELS: usize = (W * H * CAMERAS) as usize;

fn buffer<T: BufferElement>(engine: &Engine, values: Vec<T>) -> Buffer<T> {
    let mut buffer = Buffer::from_vec(values, BufferUsage::StorageReadWrite);
    buffer.ensure_device(engine.device().unwrap()).unwrap();
    buffer.upload(engine.queue().unwrap()).unwrap();
    buffer
}

fn fixture(engine: &Engine) -> (Model, State, TiledCamera) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.65, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_box(
            body,
            Vec3::new(0.2, 0.5, 0.3),
            Transform::new(Vec3::new(1.0, 0.0, -0.5), Quat::IDENTITY),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let config = TiledCameraConfig {
        width: W,
        height: H,
        light_dir: -Vec3::Y,
        optics: PhysicalOptics {
            lens_diameter_m: 0.6,
            focal_distance_m: 1.5,
            aperture_samples: 7,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut camera = TiledCamera::new(engine, config, CAMERAS as usize).unwrap();
    camera
        .set_camera_look_at(0, Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    camera
        .set_camera_look_at(
            1,
            Vec3::new(1.0, 0.5, 3.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::Y,
        )
        .unwrap();
    (model, state, camera)
}

fn ring(engine: &Engine, bytes: u64) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: bytes,
            max_total_bytes: bytes * 2,
        },
    )
    .unwrap()
}

fn collect(camera: &TiledCamera, ring: &mut ReadbackRing) -> ReadbackFrame {
    let sources: Vec<_> = [
        Ch::COLOR,
        Ch::DEPTH,
        Ch::NORMAL,
        Ch::ALBEDO,
        Ch::SHAPE_INDEX,
    ]
    .map(|ch| camera.readback_source(ch, 0..PIXELS).unwrap())
    .into();
    let ticket = ring
        .try_enqueue(&sources, ReadbackStamp::default())
        .unwrap()
        .unwrap();
    ring.wait(&ticket, Duration::from_secs(10)).unwrap()
}

#[derive(Debug)]
struct Reference {
    color: Vec<u32>,
    depth: Vec<Real>,
    normal: Vec<Vec3>,
    albedo: Vec<u32>,
    ids: Vec<u32>,
}

fn reference(engine: &Engine, model: &Model, state: &State, camera: &TiledCamera) -> Reference {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = camera.config();
    let poses = buffer(engine, camera.poses_host().unwrap().to_vec());
    let mut origins = buffer(engine, vec![Vec3::ZERO; PIXELS]);
    let mut directions = buffer(engine, vec![Vec3::ZERO; PIXELS]);
    let mut depth = buffer(engine, vec![0.0 as Real; PIXELS]);
    let mut color = buffer(engine, vec![0u32; PIXELS]);
    let mut normal = buffer(engine, vec![Vec3::ZERO; PIXELS]);
    let mut albedo = buffer(engine, vec![0u32; PIXELS]);
    let mut ids = buffer(engine, vec![u32::MAX; PIXELS]);
    let rays = CameraRaysKernel::new(device);
    let trace = CameraTraceKernel::new(device);
    let samples = if cfg.optics.lens_diameter_m > 0.0 {
        cfg.optics.aperture_samples.max(1)
    } else {
        1
    };
    let mut sums = vec![[0.0f32; 4]; PIXELS];
    let mut first = None;
    for sample in 0..samples {
        let offset = if sample == 0 {
            [0.0, 0.0]
        } else {
            let radius = (sample as f32 / (samples - 1) as f32).sqrt();
            let theta = sample as f32 * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
            [radius * theta.cos(), radius * theta.sin()]
        };
        rays.dispatch(
            device,
            queue,
            &poses,
            &mut origins,
            &mut directions,
            W,
            H,
            CAMERAS,
            cfg.fov_y_radians,
            W as f32 / H as f32,
            cfg.near,
            cfg.optics.lens_diameter_m * 0.5,
            cfg.optics.focal_distance_m,
            offset,
        )
        .unwrap();
        trace
            .dispatch(
                device,
                queue,
                model,
                state.body_q.device_buffer_current().unwrap(),
                &origins,
                &directions,
                &mut depth,
                &mut color,
                &mut normal,
                &mut albedo,
                &mut ids,
                W,
                H,
                CAMERAS,
                cfg.max_distance,
                Ch::ALL.bits(),
                cfg.ambient,
                cfg.light_dir.normalize(),
                cfg.light_color,
            )
            .unwrap();
        color.download(device, queue).unwrap();
        for (sum, packed) in sums.iter_mut().zip(color.host().unwrap()) {
            for (channel, value) in sum.iter_mut().enumerate() {
                *value += ((packed >> (channel * 8)) & 255) as f32;
            }
        }
        if sample == 0 {
            depth.download(device, queue).unwrap();
            normal.download(device, queue).unwrap();
            albedo.download(device, queue).unwrap();
            ids.download(device, queue).unwrap();
            first = Some(Reference {
                color: Vec::new(),
                depth: depth.host().unwrap().to_vec(),
                normal: normal.host().unwrap().to_vec(),
                albedo: albedo.host().unwrap().to_vec(),
                ids: ids.host().unwrap().to_vec(),
            });
        }
    }
    let mut result = first.unwrap();
    let inverse = 1.0 / samples as f32;
    result.color = sums
        .into_iter()
        .map(|sum| {
            sum.into_iter()
                .enumerate()
                .fold(0, |packed, (channel, value)| {
                    packed | ((value * inverse).clamp(0.0, 255.0).round() as u32) << (channel * 8)
                })
        })
        .collect();
    result
}

fn assert_geometry(frame: &ReadbackFrame, expected: &Reference) {
    assert_eq!(frame.channel::<Real>(1).unwrap(), expected.depth);
    assert_eq!(frame.channel::<Vec3>(2).unwrap(), expected.normal);
    assert_eq!(frame.channel::<u32>(3).unwrap(), expected.albedo);
    assert_eq!(frame.channel::<u32>(4).unwrap(), expected.ids);
}

#[test]
#[ignore = "needs a real GPU"]
fn shared_and_legacy_camera_capsule_depth_ignores_hidden_cap_surfaces() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder
        .add_shape_capsule(body, 0.6, 0.4, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut camera = TiledCamera::new(
        &engine,
        TiledCameraConfig {
            width: W,
            height: H,
            channels: Ch::ALL,
            ..Default::default()
        },
        CAMERAS as usize,
    )
    .unwrap();
    camera
        .set_camera_look_at(0, Vec3::ZERO, Vec3::Y, Vec3::Z)
        .unwrap();
    camera
        .set_camera_look_at(1, Vec3::Y * 0.6, Vec3::NEG_Y, Vec3::Z)
        .unwrap();
    let legacy = reference(&engine, &model, &state, &camera);
    camera.dispatch(&engine, &model, &state).unwrap();
    let depth = camera.download_depth(&engine).unwrap();
    for (camera_id, expected) in [1.0, 1.6].into_iter().enumerate() {
        let pixel = camera_id * (W * H) as usize + (H / 2 * W + W / 2) as usize;
        assert!((depth[pixel] - expected).abs() < 2e-5);
        assert!((legacy.depth[pixel] - expected).abs() < 2e-5);
        assert_eq!(legacy.ids[pixel], 0);
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn camera_masks_ping_pong_and_replaced_outputs_preserve_independent_channels() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state, mut camera) = fixture(&engine);
    let mut second = model.state().unwrap();
    second
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut states = [state, second];
    let mut ring = ring(&engine, PIXELS as u64 * 32);
    let mut expected = reference(&engine, &model, &states[0], &camera);
    camera.dispatch(&engine, &model, &states[0]).unwrap();
    for (sample, mask) in [
        Ch::DEPTH,
        Ch::COLOR,
        Ch::NORMAL,
        Ch::ALBEDO,
        Ch::SHAPE_INDEX,
        Ch::ALL,
    ]
    .into_iter()
    .enumerate()
    {
        let state = &mut states[(sample + 1) % 2];
        let pose = Transform::new(Vec3::X * (sample + 1) as Real * 0.13, Quat::IDENTITY).to_gpu();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&pose),
        );
        state.body_q.mark_device_dirty();
        let mut cfg = *camera.config();
        cfg.channels = mask;
        camera.set_config(cfg).unwrap();
        let next = reference(&engine, &model, state, &camera);
        if mask.contains(Ch::COLOR) {
            expected.color = next.color;
        }
        if mask.contains(Ch::DEPTH) {
            expected.depth = next.depth;
        }
        if mask.contains(Ch::NORMAL) {
            expected.normal = next.normal;
        }
        if mask.contains(Ch::ALBEDO) {
            expected.albedo = next.albedo;
        }
        if mask.contains(Ch::SHAPE_INDEX) {
            expected.ids = next.ids;
        }
        camera.dispatch(&engine, &model, state).unwrap();
        assert!(state.body_q.host().is_err());
        let result = collect(&camera, &mut ring);
        assert_eq!(result.channel::<u32>(0).unwrap(), expected.color);
        assert_geometry(&result, &expected);
        assert_eq!(camera.dispatch_stats().trace_preparations, 1);
        assert_eq!(camera.query_stats().primitive_preparations, 2);
    }
    assert_eq!(camera.dispatch_stats().rays_preparations, 1);
    assert_eq!(camera.dispatch_stats().accumulator_preparations, 1);
    camera.color = buffer(&engine, vec![0u32; PIXELS]);
    camera.dispatch(&engine, &model, &states[0]).unwrap();
    let result = collect(&camera, &mut ring);
    assert_eq!(result.channel::<u32>(0).unwrap(), expected.color);
    assert_geometry(&result, &expected);
    assert_eq!(camera.dispatch_stats().trace_preparations, 2);
    assert_eq!(camera.query_stats().primitive_preparations, 2);
    assert_eq!(camera.dispatch_stats().accumulator_preparations, 2);
    assert_eq!(camera.dispatch_stats().accumulator_buffer_allocations, 2);
    assert!(camera.readback_source(Ch::ALL, 0..1).is_err());
    assert!(
        camera
            .readback_source(Ch::COLOR, PIXELS..PIXELS + 1)
            .is_err()
    );
    let config = *camera.config();
    let mut invalid = config;
    invalid.optics.aperture_samples = 1025;
    assert!(camera.set_config(invalid).is_err());
    assert_eq!(
        camera.config().optics.aperture_samples,
        config.optics.aperture_samples
    );
    let before = camera.dispatch_stats();
    let mut dirty = model.state().unwrap();
    assert!(camera.dispatch(&engine, &model, &dirty).is_err());
    dirty
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    dirty.body_q.host_mut().unwrap()[0].position = Vec3::Y;
    assert!(camera.dispatch(&engine, &model, &dirty).is_err());
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    assert!(camera.dispatch(&foreign, &model, &states[0]).is_err());
    assert!(camera.download_depth(&foreign).is_err());
    assert!(camera.download_color(&foreign).is_err());
    assert!(camera.download_normal(&foreign).is_err());
    assert!(camera.download_albedo(&foreign).is_err());
    assert!(camera.download_shape_index(&foreign).is_err());
    assert_eq!(camera.dispatch_stats(), before);
}

#[test]
#[ignore = "needs a real GPU"]
fn camera_optics_matches_host_average_and_center_geometry_without_readback() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state, mut camera) = fixture(&engine);
    let mut ring = ring(&engine, PIXELS as u64 * 32);
    let mut passes = 0;
    for (frame, samples) in [2, 3, 7, 16].into_iter().enumerate() {
        let mut cfg = *camera.config();
        cfg.optics.aperture_samples = samples;
        camera.set_config(cfg).unwrap();
        let expected = reference(&engine, &model, &state, &camera);
        camera.dispatch(&engine, &model, &state).unwrap();
        assert!(camera.color.host().is_err());
        assert!(camera.depth.host().is_err());
        let result = collect(&camera, &mut ring);
        assert_eq!(
            result.channel::<u32>(0).unwrap(),
            expected.color,
            "K={samples}"
        );
        assert_geometry(&result, &expected);
        passes += samples as u64;
        let stats = camera.dispatch_stats();
        assert_eq!(stats.submissions, frame as u64 + 1);
        assert_eq!(stats.ray_passes, passes);
        assert_eq!(stats.trace_passes, passes);
        assert_eq!(stats.accumulation_passes, passes);
        assert_eq!(stats.resolve_passes, frame as u64 + 1);
        assert_eq!(stats.rays_preparations, 1);
        assert_eq!(stats.trace_preparations, 1);
        assert_eq!(stats.accumulator_preparations, 1);
    }
    let before = camera.dispatch_stats();
    let mut cfg = *camera.config();
    cfg.channels = Ch::DEPTH | Ch::NORMAL | Ch::ALBEDO | Ch::SHAPE_INDEX;
    camera.set_config(cfg).unwrap();
    let color = collect(&camera, &mut ring).channel::<u32>(0).unwrap();
    camera.dispatch(&engine, &model, &state).unwrap();
    let result = collect(&camera, &mut ring);
    assert_eq!(result.channel::<u32>(0).unwrap(), color);
    assert_geometry(&result, &reference(&engine, &model, &state, &camera));
    let stats = camera.dispatch_stats();
    assert_eq!(stats.ray_passes, before.ray_passes + 1);
    assert_eq!(stats.trace_passes, before.trace_passes + 1);
    assert_eq!(stats.accumulation_passes, before.accumulation_passes);
    cfg.channels = Ch::NONE;
    camera.set_config(cfg).unwrap();
    camera.dispatch(&engine, &model, &state).unwrap();
    assert_eq!(camera.dispatch_stats(), stats);

    cfg.channels = Ch::ALL;
    cfg.optics.aperture_samples = 1;
    camera.set_config(cfg).unwrap();
    let center = reference(&engine, &model, &state, &camera);
    cfg.optics.aperture_samples = 1024;
    camera.set_config(cfg).unwrap();
    camera.dispatch(&engine, &model, &state).unwrap();
    let result = collect(&camera, &mut ring);
    assert_geometry(&result, &center);
    assert_eq!(camera.dispatch_stats().ray_passes, stats.ray_passes + 1024);
    assert_eq!(camera.dispatch_stats().submissions, stats.submissions + 1);
    assert_eq!(camera.dispatch_stats().rays_preparations, 1);
    assert_eq!(camera.dispatch_stats().trace_preparations, 1);
    assert_eq!(camera.dispatch_stats().accumulator_preparations, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn aperture_resolve_matches_cpu_rounding_through_maximum_sample_budget() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut output =
        DeviceBuffer::from_host(&engine, vec![0u32; 8], BufferUsage::StorageReadWrite).unwrap();
    let kernel = CameraAccumulateKernel::new(
        engine.device().unwrap(),
        output.device_buffer().unwrap(),
        [8, 1, 1],
    )
    .unwrap();
    let mut ring = ring(&engine, 32);
    for count in [1u32, 2, 3, 7, 1024] {
        let samples: Vec<u32> = (0..count)
            .flat_map(|sample| {
                (0..8).map(move |pixel| {
                    if pixel == 0 {
                        sample & 1
                    } else {
                        sample
                            .wrapping_mul(0x632be5ab)
                            .wrapping_add(pixel * 0x192b59)
                    }
                })
            })
            .collect();
        let source =
            DeviceBuffer::from_host(&engine, samples.clone(), BufferUsage::Storage).unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        kernel
            .encode_reset(engine.queue().unwrap(), &mut encoder, count)
            .unwrap();
        for sample in 0..count {
            encoder.copy_buffer_to_buffer(
                source.device_buffer().unwrap(),
                sample as u64 * 32,
                output.device_buffer().unwrap(),
                0,
                32,
            );
            kernel.encode_accumulate(&mut encoder);
        }
        kernel.encode_resolve(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        output.mark_device_dirty();
        let ticket = ring
            .try_enqueue(
                &[output.readback_source(0..8).unwrap()],
                ReadbackStamp::default(),
            )
            .unwrap()
            .unwrap();
        let actual = ring
            .wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<u32>(0)
            .unwrap();
        for pixel in 0..8 {
            let mut expected = 0;
            for channel in 0..4 {
                let sum: f32 = (0..count)
                    .map(|sample| {
                        ((samples[sample as usize * 8 + pixel] >> (channel * 8)) & 255) as f32
                    })
                    .sum();
                expected |= ((sum * (1.0 / count as f32)).round() as u32) << (channel * 8);
            }
            assert_eq!(actual[pixel], expected, "K={count}, pixel={pixel}");
        }
    }
}
