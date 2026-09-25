#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Transform, Vec3};
use molla_sensors::{
    ImuMount, ImuSensor,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::{BodyParams, ModelBuilder};
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "needs a real GPU"]
fn auxiliary_commands_submit_transactionally_and_poison_on_failure() {
    use molla_sensors::SensorCommandBatch;
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let buffer =
        DeviceBuffer::from_host(&engine, vec![7u32], BufferUsage::StorageReadWrite).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 8,
            max_total_bytes: 8,
        },
    )
    .unwrap();
    let read = |ring: &mut ReadbackRing| {
        let ticket = ring
            .try_enqueue(
                &[buffer.readback_source(0..1).unwrap()],
                ReadbackStamp::default(),
            )
            .unwrap()
            .unwrap();
        ring.wait(&ticket, Duration::from_secs(30))
            .unwrap()
            .channel::<u32>(0)
            .unwrap()[0]
    };
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        batch
            .encode_auxiliary(|encoder| {
                encoder.clear_buffer(buffer.device_buffer()?, 0, None);
                Err(molla_core::Error::Solver("fixture cancellation".into()))
            })
            .is_err()
    );
    let called = std::cell::Cell::new(false);
    assert!(
        batch
            .encode_auxiliary(|_| {
                called.set(true);
                Ok(())
            })
            .is_err()
    );
    assert!(!called.get());
    assert!(batch.submit().is_err());
    assert_eq!(read(&mut ring), 7);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    batch
        .encode_auxiliary(|encoder| {
            encoder.clear_buffer(buffer.device_buffer()?, 0, None);
            Ok(())
        })
        .unwrap();
    let submission = batch.submit().unwrap();
    assert_eq!(submission.stats.submissions, 1);
    assert_eq!(submission.stats.auxiliary_encodes, 1);
    assert!(submission.into_frames().is_empty());
    assert_eq!(read(&mut ring), 0);
    let empty = SensorCommandBatch::new(&engine).unwrap().submit().unwrap();
    assert!(empty.index.is_none());
    assert_eq!(empty.stats.submissions, 0);
    let mut bounded = SensorCommandBatch::new(&engine).unwrap();
    for _ in 0..128 {
        bounded.encode_auxiliary(|_| Ok(())).unwrap();
    }
    assert!(
        bounded
            .encode_auxiliary(|_| {
                called.set(true);
                Ok(())
            })
            .is_err()
    );
    assert!(!called.get());
    assert_eq!(bounded.submit().unwrap().stats.auxiliary_encodes, 128);
    let mut panicking = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = panicking.encode_auxiliary(|_| panic!("fixture auxiliary panic"));
        }))
        .is_err()
    );
    assert!(panicking.submit().is_err());
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    engine.destroy_device().unwrap();
    assert!(
        batch
            .encode_auxiliary(|_| {
                called.set(true);
                Ok(())
            })
            .is_err()
    );
    assert!(!called.get());
}

#[test]
#[ignore = "needs a real GPU"]
fn imu_gpu_consumer_and_async_snapshots_do_not_download_intermediate_outputs() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.1, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 48,
            max_total_bytes: 96,
        },
    )
    .unwrap();
    assert!(
        imu.dispatch(&engine, &model, &state, 0.01)
            .unwrap()
            .is_none()
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sensor device-only consumer"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var<storage, read> acceleration: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> result: array<vec4<f32>>;
            @compute @workgroup_size(1) fn main() { result[0] = acceleration[0] * 2.0; }
        "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sensor device-only consumer"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut result =
        DeviceBuffer::from_host(&engine, vec![Vec3::ZERO], BufferUsage::StorageReadWrite).unwrap();
    let mut tickets = Vec::new();
    for sample in 1..=2 {
        state.body_qd.host_mut().unwrap()[0].angular = Vec3::Y * sample as f64;
        state.body_qd.upload(queue).unwrap();
        let frame = imu
            .dispatch(&engine, &model, &state, 0.01)
            .unwrap()
            .unwrap();
        assert!(frame.accelerometer.host().is_err());
        assert!(frame.gyroscope.host().is_err());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sensor device-only consumer"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame
                        .accelerometer
                        .device_buffer()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.device_buffer().unwrap().as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        result.mark_device_dirty();
        let sources = [
            frame.accelerometer.readback_source(0..1).unwrap(),
            frame.gyroscope.readback_source(0..1).unwrap(),
            result.readback_source(0..1).unwrap(),
        ];
        tickets.push(
            ring.try_enqueue(
                &sources,
                ReadbackStamp {
                    sample_id: sample,
                    scene_revision: 17,
                    simulation_time: Some(sample as f64 * 0.01),
                    sensor_id: 8,
                },
            )
            .unwrap()
            .unwrap(),
        );
    }
    assert_eq!(imu.readback_stats().staging_allocations, 0);
    assert_eq!(imu.readback_stats().submissions, 0);
    for (i, ticket) in tickets.iter().enumerate() {
        let frame = ring.wait(ticket, Duration::from_secs(10)).unwrap();
        let accel = frame.channel::<Vec3>(0).unwrap()[0];
        assert!((accel - Vec3::Y * 9.806649).length() < 1.0e-3);
        assert!(
            (frame.channel::<Vec3>(1).unwrap()[0] - Vec3::Y * (i + 1) as f64).length() < 1.0e-4
        );
        assert!((frame.channel::<Vec3>(2).unwrap()[0] - accel * 2.0).length() < 1.0e-4);
        assert_eq!(frame.stamp.sample_id, (i + 1) as u64);
        assert_eq!(frame.stamp.scene_revision, 17);
    }
    assert!(result.host().is_err());
    for _ in 0..8 {
        assert_eq!(imu.update(&engine, &model, &state, 0.01).unwrap().len(), 1);
    }
    assert_eq!(imu.readback_stats().staging_allocations, 1);
    assert_eq!(imu.readback_stats().submissions, 8);
    assert_eq!(imu.readback_stats().copied_bytes, 8 * 32);
    assert!(imu.readback_stats().blocking_waits <= 8);
}

#[test]
#[ignore = "needs a real GPU"]
fn imu_rejects_foreign_engine_without_advancing_history() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    assert!(imu.dispatch(&foreign, &model, &state, 0.01).is_err());
    assert!(
        imu.dispatch(&engine, &model, &state, 0.01)
            .unwrap()
            .is_none()
    );
}
