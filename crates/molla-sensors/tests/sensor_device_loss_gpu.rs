#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Transform, Vec3};
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraBatch, CameraBatchLimits, CameraChannels,
    CameraSpec, CloudGrid, CloudGridConfig, CloudGridSpec, CloudPoint, CloudProcessor,
    CloudProcessorConfig, ContactSensor, DeviceCommandBatch, DeviceSnapshotConfig,
    DeviceSnapshotPool, FrameTarget, FrameTransformSensor, ImageProcessor, ImageProcessorConfig,
    ImuMount, ImuSensor, LidarBatch, Raycaster, SensorScene, SensorSceneOptions, TiledCamera,
    TiledCameraConfig,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_storage::BufferUsage;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

static GPU: Mutex<()> = Mutex::new(());
fn ring(engine: &Engine) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 8,
            max_total_bytes: 16,
        },
    )
    .unwrap()
}
fn pool(engine: &Engine) -> DeviceSnapshotPool {
    DeviceSnapshotPool::new(
        engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 8,
            max_total_bytes: 8,
        },
    )
    .unwrap()
}
fn drain(engine: &Engine) {
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
}

#[test]
#[ignore = "requires GPU"]
fn explicit_logical_device_loss_rejects_pending_work_and_fresh_engine_recovers() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut input =
        DeviceBuffer::from_host(&engine, vec![7u32], BufferUsage::StorageReadWrite).unwrap();
    let saved_source = input.readback_source(0..1).unwrap();
    let mut readbacks = ring(&engine);
    let mut snapshots = pool(&engine);
    let snapshot = snapshots
        .try_capture(&[input.readback_source(0..1).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let ticket = readbacks
        .try_enqueue(&[input.readback_source(0..1).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let cancelled = readbacks
        .try_enqueue(&[input.readback_source(0..1).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    readbacks.cancel(&cancelled).unwrap();
    let mut batch = DeviceCommandBatch::new(&engine).unwrap();
    batch
        .encode_with_snapshots(&[&snapshot], |_| Ok(()))
        .unwrap();
    engine.destroy_device().unwrap();
    assert!(engine.check_device().is_err());
    assert!(engine.device_health().unwrap().is_lost());
    assert!(readbacks.is_failed());
    assert_eq!(readbacks.available_slots(), 0);
    assert!(readbacks.poll(&ticket).is_err());
    readbacks.reset();
    assert!(readbacks.try_enqueue(&[], Default::default()).is_err());
    assert!(readbacks.wait(&cancelled, Duration::ZERO).is_err());
    assert!(snapshots.available_slots().is_err());
    assert!(snapshots.try_capture(&[], Default::default()).is_err());
    assert!(snapshots.reset().is_err());
    assert!(!snapshots.is_current(&snapshot));
    let channel = snapshot.channel::<u32>(0).unwrap();
    assert!(channel.binding().is_err());
    assert!(channel.readback_source(0..1).is_err());
    assert!(input.device_buffer().is_err());
    assert!(input.readback_source(0..1).is_err());
    assert!(
        batch
            .encode_with_snapshots(&[&snapshot], |_| panic!("must not encode after loss"))
            .is_err()
    );
    assert!(batch.submit().is_err());
    assert!(DeviceCommandBatch::new(&engine).is_err());
    assert!(
        ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8
            }
        )
        .is_err()
    );
    assert!(
        DeviceSnapshotPool::new(
            &engine,
            DeviceSnapshotConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8
            }
        )
        .is_err()
    );
    assert!(DeviceBuffer::from_host(&engine, vec![1u32], BufferUsage::StorageReadWrite).is_err());
    drain(&engine);
    engine.destroy_device().unwrap();
    assert!(
        engine
            .device_health()
            .unwrap()
            .loss()
            .unwrap()
            .message
            .contains("explicitly")
    );

    let fresh = Engine::new_gpu().unwrap();
    let mut recovered = ring(&fresh);
    assert!(
        recovered
            .try_enqueue(&[saved_source], Default::default())
            .is_err()
    );
    assert!(input.upload().is_err());
    let new_input =
        DeviceBuffer::from_host(&fresh, vec![91u32], BufferUsage::StorageReadWrite).unwrap();
    let ticket = recovered
        .try_enqueue(
            &[new_input.readback_source(0..1).unwrap()],
            ReadbackStamp {
                sample_id: 22,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    let frame = recovered.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(frame.channel::<u32>(0).unwrap(), [91]);
    assert_eq!(frame.stamp.sample_id, 22);
    assert!(fresh.check_device().is_ok());
}

#[test]
#[ignore = "requires GPU"]
fn backend_loss_callback_invalidates_empty_requests_but_collected_host_data_survives() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let input =
        DeviceBuffer::from_host(&engine, vec![12u32], BufferUsage::StorageReadWrite).unwrap();
    let mut readbacks = ring(&engine);
    let ticket = readbacks
        .try_enqueue(&[input.readback_source(0..1).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let collected = readbacks.wait(&ticket, Duration::from_secs(10)).unwrap();
    let empty = readbacks
        .try_enqueue(&[], Default::default())
        .unwrap()
        .unwrap();
    let mut snapshots = pool(&engine);
    let health = engine.device_health().unwrap().clone();
    engine.device().unwrap().destroy();
    drain(&engine);
    assert!(health.is_lost());
    assert!(matches!(
        health.loss().unwrap().reason,
        wgpu::DeviceLostReason::Destroyed
    ));
    assert!(readbacks.poll(&empty).is_err());
    assert!(readbacks.is_failed());
    assert!(snapshots.available_slots().is_err());
    assert_eq!(collected.channel::<u32>(0).unwrap(), [12]);
    assert_eq!(input.host().unwrap(), [12]);
    drop(input);
    drop(readbacks);
    drop(snapshots);
    let weak_device = Arc::downgrade(engine.device().unwrap());
    drop(engine);
    assert!(weak_device.upgrade().is_none());
    assert!(health.is_lost());
}

#[test]
#[ignore = "requires GPU"]
fn known_loss_rejects_sensor_and_processing_work_before_resource_use() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut ray = Raycaster::new(&engine, 4).unwrap();
    let mut lidar = LidarBatch::new(&engine, 1, 4).unwrap();
    let camera_config = TiledCameraConfig {
        width: 2,
        height: 2,
        ..Default::default()
    };
    let mut camera = TiledCamera::new(&engine, camera_config, 1).unwrap();
    let specs = [CameraSpec {
        sensor_id: 1,
        config: camera_config,
        pose: Transform::IDENTITY,
    }];
    let mut cameras = CameraBatch::new(&engine, &specs, CameraBatchLimits::default()).unwrap();
    let mounts = [ImuMount {
        body,
        local_pose: Transform::IDENTITY,
    }];
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let targets = [FrameTarget {
        body,
        local_pose: Transform::IDENTITY,
    }];
    let mut transforms = FrameTransformSensor::new(&engine, &model, None, &targets).unwrap();
    let mut contacts = ContactSensor::new(&engine, &model).unwrap();
    let image_config = ImageProcessorConfig {
        max_pixels: 4,
        max_images: 1,
        max_operations_per_image: 1,
        ..Default::default()
    };
    let cloud_config = CloudProcessorConfig {
        max_points: 4,
        max_jobs: 1,
        ..Default::default()
    };
    let grid_config = CloudGridConfig {
        max_points: 4,
        max_cells: 1,
        ..Default::default()
    };
    let annotation_config = AnnotationConfig {
        max_pixels: 4,
        max_images: 1,
        max_labels: 1,
        ..Default::default()
    };
    let mut images = ImageProcessor::new(&engine, image_config).unwrap();
    let mut cloud = CloudProcessor::new(&engine, cloud_config).unwrap();
    let mut grid = CloudGrid::new(&engine, grid_config).unwrap();
    let mut annotation = AnnotationProcessor::new(&engine, annotation_config).unwrap();
    let points = DeviceBuffer::from_host(
        &engine,
        vec![CloudPoint::new(Vec3::ZERO, 0, 0)],
        BufferUsage::StorageReadWrite,
    )
    .unwrap();
    let grid_input = points.readback_source(0..1).unwrap();
    engine.destroy_device().unwrap();
    assert!(
        ray.dispatch(&engine, &model, &state, &[Vec3::ZERO], &[Vec3::Y], 10.)
            .is_err()
    );
    assert!(
        ray.dispatch_scene(&engine, &scene, &state, &[Vec3::ZERO], &[Vec3::Y], 10.)
            .is_err()
    );
    assert!(lidar.dispatch(&engine, &model, &state, &[]).is_err());
    assert!(lidar.dispatch_scene(&engine, &scene, &state, &[]).is_err());
    assert!(camera.dispatch(&engine, &model, &state).is_err());
    assert!(camera.download_color(&engine).is_err());
    assert!(camera.readback_source(CameraChannels::COLOR, 0..1).is_err());
    assert!(cameras.dispatch(&engine, &model, &state).is_err());
    assert!(cameras.dispatch_scene(&engine, &scene, &state).is_err());
    assert!(cameras.reconfigure(&engine, &specs).is_err());
    assert!(imu.dispatch(&engine, &model, &state, 0.01).is_err());
    assert!(transforms.dispatch(&engine, &state).is_err());
    assert!(
        contacts
            .dispatch(&engine, &Contacts::with_capacity(1))
            .is_err()
    );
    assert!(images.process(&[]).is_err());
    assert!(cloud.process(&[]).is_err());
    assert!(
        grid.process(
            grid_input,
            CloudGridSpec {
                minimum: [-1.; 3],
                maximum: [1.; 3],
                dimensions: [1; 3]
            },
            Default::default()
        )
        .is_err()
    );
    assert!(annotation.process(&[]).is_err());
    assert!(annotation.set_labels(&[]).is_err());
    assert!(Raycaster::new(&engine, 4).is_err());
    assert!(LidarBatch::new(&engine, 1, 4).is_err());
    assert!(TiledCamera::new(&engine, camera_config, 1).is_err());
    assert!(CameraBatch::new(&engine, &specs, CameraBatchLimits::default()).is_err());
    assert!(ImuSensor::new(&engine, &model, &mounts).is_err());
    assert!(FrameTransformSensor::new(&engine, &model, None, &targets).is_err());
    assert!(ContactSensor::new(&engine, &model).is_err());
    assert!(SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).is_err());
    assert!(ImageProcessor::new(&engine, image_config).is_err());
    assert!(CloudProcessor::new(&engine, cloud_config).is_err());
    assert!(CloudGrid::new(&engine, grid_config).is_err());
    assert!(AnnotationProcessor::new(&engine, annotation_config).is_err());
    assert_eq!(images.stats().submissions, 0);
    assert_eq!(cloud.stats().submissions, 0);
    assert_eq!(grid.stats().submissions, 0);
    assert_eq!(annotation.stats().submissions, 0);
    drain(&engine);
}
