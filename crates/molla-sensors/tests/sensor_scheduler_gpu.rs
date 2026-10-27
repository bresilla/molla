#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_kernels::IntegrateBodiesKernel;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{
    ContactSensor, FrameTarget, FrameTransformSensor, ImuMount, ImuSensor, SensorSchedule,
    SensorScheduler,
    readback::{ReadbackConfig, ReadbackRing},
};
use molla_sim::{BodyParams, Contacts, Control, ModelBuilder};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn schedules() -> [SensorSchedule; 3] {
    [10, 5, 20].map(|ms| SensorSchedule {
        sensor_id: ms,
        period: Duration::from_millis(ms),
        phase: Duration::ZERO,
    })
}

fn run(engine: &Engine, render_every: u64) -> Vec<(u64, Duration, Vec3)> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 3.0));
    builder
        .add_shape_sphere(body, 0.2, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut current = model.state().unwrap();
    let mut next = model.state().unwrap();
    current.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut control = Control::for_model(&model);
    control.upload_to_device(device, queue).unwrap();
    let force = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let integrate = IntegrateBodiesKernel::new(device);
    let mut imu = ImuSensor::new(
        engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let local_pose = Transform::new(Vec3::new(0.2, 0.3, 0.4), Quat::IDENTITY);
    let mut frames =
        FrameTransformSensor::new(engine, &model, None, &[FrameTarget { body, local_pose }])
            .unwrap();
    let mut contact = ContactSensor::new(engine, &model).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut scheduler = SensorScheduler::new(Duration::ZERO, 91, &schedules()).unwrap();
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 32,
            bytes_per_slot: 32,
            max_total_bytes: 1024,
        },
    )
    .unwrap();
    let mut pending = Vec::new();
    for tick in 0..=40 {
        if tick != 0 {
            integrate
                .dispatch(
                    device, queue, &model, &current, &control, &force, &mut next, 0.001,
                )
                .unwrap();
            std::mem::swap(&mut current, &mut next);
            assert!(current.body_q.host().is_err());
            assert!(current.body_qd.host().is_err());
        }
        let time = Duration::from_millis(tick);
        for sample in scheduler.advance(time).unwrap() {
            assert_eq!(sample.simulation_time, sample.scheduled_time);
            assert_eq!(sample.skipped_samples, 0);
            let ticket = match sample.sensor_id {
                10 => {
                    let dt = sample.elapsed.unwrap_or(sample.period).as_secs_f64();
                    let output = imu.dispatch(engine, &model, &current, dt).unwrap();
                    if tick == 0 {
                        assert!(output.is_none());
                    }
                    output.map(|output| {
                        ring.try_enqueue(
                            &[
                                output.accelerometer.readback_source(0..1).unwrap(),
                                output.gyroscope.readback_source(0..1).unwrap(),
                            ],
                            sample.readback_stamp(),
                        )
                        .unwrap()
                        .unwrap()
                    })
                }
                5 => {
                    let output = frames.dispatch(engine, &current).unwrap();
                    Some(
                        ring.try_enqueue(
                            &[output
                                .world_position
                                .readback_source(output.target_range)
                                .unwrap()],
                            sample.readback_stamp(),
                        )
                        .unwrap()
                        .unwrap(),
                    )
                }
                20 => {
                    let output = contact.dispatch(engine, &contacts).unwrap();
                    Some(
                        ring.try_enqueue(
                            &[output.normal_sum.readback_source(0..1).unwrap()],
                            sample.readback_stamp(),
                        )
                        .unwrap()
                        .unwrap(),
                    )
                }
                _ => unreachable!(),
            };
            if let Some(ticket) = ticket {
                pending.push((sample.clone(), ticket));
            }
        }
        if tick % render_every == 0 {
            assert!(scheduler.advance(time).unwrap().is_empty());
        }
    }
    assert_eq!(imu.readback_stats().submissions, 0);
    assert_eq!(frames.readback_stats().submissions, 0);
    assert_eq!(contact.readback_stats().submissions, 0);
    let mut readings = Vec::new();
    for (sample, ticket) in pending.iter().rev() {
        let frame = ring.wait(ticket, Duration::from_secs(10)).unwrap();
        assert!(scheduler.is_current(sample));
        assert!(sample.matches_readback(&frame));
        let value = frame.channel::<Vec3>(0).unwrap()[0];
        if sample.sensor_id == 5 {
            let step = sample.simulation_time.as_millis() as f64;
            let expected = Vec3::new(0.2, 3.3 - 9.81 * 1e-6 * step * (step + 1.0) / 2.0, 0.4);
            assert!(
                (value - expected).length() < 2e-5,
                "{value:?} != {expected:?}"
            );
        } else {
            assert!(
                value.length() < 2e-3,
                "sensor {}: {value:?}",
                sample.sensor_id
            );
        }
        if sample.sensor_id == 10 {
            assert!(frame.channel::<Vec3>(1).unwrap()[0].length() < 1e-6);
        }
        readings.push((sample.sensor_id, sample.simulation_time, value));
    }
    assert_eq!(readings.iter().filter(|r| r.0 == 10).count(), 4);
    assert_eq!(readings.iter().filter(|r| r.0 == 5).count(), 9);
    assert_eq!(readings.iter().filter(|r| r.0 == 20).count(), 3);
    readings
}

#[test]
#[ignore = "needs a real GPU"]
fn physics_time_rates_match_across_render_cadences_without_sensor_readback() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let every_frame = run(&engine, 1);
    let sparse_frames = run(&engine, 13);
    assert_eq!(every_frame.len(), sparse_frames.len());
    for (a, b) in every_frame.iter().zip(sparse_frames) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert!((a.2 - b.2).length() < 1e-6);
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn scene_reset_rejects_pending_generation_and_reprimes_imu() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
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
    let target = FrameTarget {
        body: BodyId::from_index(0),
        local_pose: Transform::IDENTITY,
    };
    let mut frames = FrameTransformSensor::new(&engine, &model, None, &[target]).unwrap();
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let mut scheduler = SensorScheduler::new(Duration::ZERO, 7, &schedules()[..1]).unwrap();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 16,
            max_total_bytes: 32,
        },
    )
    .unwrap();
    let old = scheduler.advance(Duration::ZERO).unwrap()[0].clone();
    assert!(
        imu.dispatch(&engine, &model, &state, old.period.as_secs_f64())
            .unwrap()
            .is_none()
    );
    let output = frames.dispatch(&engine, &state).unwrap();
    let old_ticket = ring
        .try_enqueue(
            &[output.world_position.readback_source(0..1).unwrap()],
            old.readback_stamp(),
        )
        .unwrap()
        .unwrap();
    scheduler.reset(Duration::ZERO, 7).unwrap();
    imu.reset();
    state.body_q.host_mut().unwrap()[0].position = Vec3::Y * 8.0;
    state.body_q.upload(queue).unwrap();
    let new = scheduler.advance(Duration::ZERO).unwrap()[0].clone();
    assert_eq!(new.elapsed, None);
    assert!(new.sample_id > old.sample_id);
    assert!(
        imu.dispatch(&engine, &model, &state, new.period.as_secs_f64())
            .unwrap()
            .is_none()
    );
    let output = frames.dispatch(&engine, &state).unwrap();
    let new_ticket = ring
        .try_enqueue(
            &[output.world_position.readback_source(0..1).unwrap()],
            new.readback_stamp(),
        )
        .unwrap()
        .unwrap();
    let new_frame = ring.wait(&new_ticket, Duration::from_secs(10)).unwrap();
    let old_frame = ring.wait(&old_ticket, Duration::from_secs(10)).unwrap();
    assert!(scheduler.is_current(&new));
    assert!(!scheduler.is_current(&old));
    assert!(new.matches_readback(&new_frame));
    assert!(!new.matches_readback(&old_frame));
    assert_eq!(old_frame.channel::<Vec3>(0).unwrap()[0], Vec3::ZERO);
    assert_eq!(new_frame.channel::<Vec3>(0).unwrap()[0], Vec3::Y * 8.0);
    assert_eq!(imu.readback_stats().submissions, 0);
    assert_eq!(frames.readback_stats().submissions, 0);
}
