#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sensors::{
    ContactSensor, FrameTarget, FrameTransformSensor, ImuMount, ImuSensor,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());
const N: usize = 70;

fn device_write<T: BufferElement>(engine: &Engine, buffer: &mut Buffer<T>, values: &[T]) {
    let gpu: Vec<_> = values.iter().map(BufferElement::to_gpu).collect();
    engine.queue().unwrap().write_buffer(
        buffer.device_buffer().unwrap(),
        0,
        bytemuck::cast_slice(&gpu),
    );
    buffer.mark_device_dirty();
}

fn device_buffer<T: BufferElement>(engine: &Engine, values: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, values, BufferUsage::StorageReadWrite).unwrap()
}

#[test]
#[ignore = "needs a real GPU"]
fn prepared_auxiliary_kernels_share_encoder_with_independent_parameters() {
    use molla_kernels::{ContactAggregateKernel, FrameTransformKernel, ImuKernel};

    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state
        .body_q
        .host_mut()
        .unwrap()
        .fill(Transform::new(Vec3::Y * 3.0, Quat::IDENTITY));
    state
        .body_qd
        .host_mut()
        .unwrap()
        .fill(SpatialVector::new(Vec3::X, Vec3::ZERO));
    state.upload_to_device(device, queue).unwrap();
    let contacts = contacts(&engine);
    let bodies = device_buffer(&engine, (0..N as u32).collect());
    let positions = device_buffer(&engine, vec![Vec3::ZERO; N]);
    let rotations = device_buffer(&engine, vec![Quat::IDENTITY; N]);
    let history = device_buffer(&engine, vec![SpatialVector::ZERO; N]);
    let mut accel = [
        device_buffer(&engine, vec![Vec3::ZERO; N]),
        device_buffer(&engine, vec![Vec3::ZERO; N]),
    ];
    let gyro = [
        device_buffer(&engine, vec![Vec3::ZERO; N]),
        device_buffer(&engine, vec![Vec3::ZERO; N]),
    ];
    let mut world_pos = device_buffer(&engine, vec![Vec3::ZERO; N]);
    let world_rot = device_buffer(&engine, vec![Quat::IDENTITY; N]);
    let relative_pos = device_buffer(&engine, vec![Vec3::ZERO; N]);
    let relative_rot = device_buffer(&engine, vec![Quat::IDENTITY; N]);
    let count = device_buffer(&engine, vec![0u32; N]);
    let mut depth = device_buffer(&engine, vec![0.0 as Real; N]);
    let normal = device_buffer(&engine, vec![Vec3::ZERO; N]);
    let centroid = device_buffer(&engine, vec![Vec3::ZERO; N]);
    let imu_kernel = ImuKernel::new(device);
    let imu_bindings: Vec<_> = (0..2)
        .map(|i| {
            imu_kernel.prepare(
                device,
                [
                    state.body_q.device_buffer_current().unwrap(),
                    state.body_qd.device_buffer_current().unwrap(),
                    history.device_buffer().unwrap(),
                    model.body_com.device_buffer_current().unwrap(),
                    model.body_world.device_buffer_current().unwrap(),
                    model.gravity.device_buffer_current().unwrap(),
                    bodies.device_buffer().unwrap(),
                    positions.device_buffer().unwrap(),
                    rotations.device_buffer().unwrap(),
                    accel[i].device_buffer().unwrap(),
                    gyro[i].device_buffer().unwrap(),
                ],
            )
        })
        .collect();
    let frame_kernel = FrameTransformKernel::new(device);
    let frame_bindings = frame_kernel.prepare(
        device,
        [
            state.body_q.device_buffer_current().unwrap(),
            bodies.device_buffer().unwrap(),
            positions.device_buffer().unwrap(),
            rotations.device_buffer().unwrap(),
            world_pos.device_buffer().unwrap(),
            world_rot.device_buffer().unwrap(),
            relative_pos.device_buffer().unwrap(),
            relative_rot.device_buffer().unwrap(),
        ],
    );
    let contact_kernel = ContactAggregateKernel::new(device);
    let contact_bindings = contact_kernel.prepare(
        device,
        [
            contacts.a_index.device_buffer_current().unwrap(),
            contacts.b_index.device_buffer_current().unwrap(),
            contacts.normal.device_buffer_current().unwrap(),
            contacts.depth.device_buffer_current().unwrap(),
            contacts.point.device_buffer_current().unwrap(),
            contacts
                .body_contact_offsets
                .device_buffer_current()
                .unwrap(),
            contacts
                .body_contact_indices
                .device_buffer_current()
                .unwrap(),
            count.device_buffer().unwrap(),
            depth.device_buffer().unwrap(),
            normal.device_buffer().unwrap(),
            centroid.device_buffer().unwrap(),
        ],
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        imu_kernel
            .encode_prepared(queue, &mut encoder, &imu_bindings[0], N as u32, 0.0)
            .is_err()
    );
    imu_kernel
        .encode_prepared(queue, &mut encoder, &imu_bindings[0], N as u32, 0.5)
        .unwrap();
    imu_kernel
        .encode_prepared(queue, &mut encoder, &imu_bindings[1], N as u32, 0.25)
        .unwrap();
    frame_kernel.encode_prepared(queue, &mut encoder, &frame_bindings, N as u32, false);
    contact_kernel.encode_prepared(queue, &mut encoder, &contact_bindings, N as u32);
    imu_kernel
        .encode_prepared(queue, &mut encoder, &imu_bindings[0], 0, 1.0)
        .unwrap();
    frame_kernel.encode_prepared(queue, &mut encoder, &frame_bindings, 0, true);
    contact_kernel.encode_prepared(queue, &mut encoder, &contact_bindings, 0);
    drop((imu_bindings, frame_bindings, contact_bindings));
    drop((imu_kernel, frame_kernel, contact_kernel));
    queue.submit([encoder.finish()]);
    for buffer in &mut accel {
        buffer.mark_device_dirty();
    }
    world_pos.mark_device_dirty();
    depth.mark_device_dirty();
    let mut ring = ring(&engine, N as u64 * 52);
    let ticket = ring
        .try_enqueue(
            &[
                accel[0].readback_source(0..N).unwrap(),
                accel[1].readback_source(0..N).unwrap(),
                world_pos.readback_source(0..N).unwrap(),
                depth.readback_source(0..N).unwrap(),
            ],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let result = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    for (channel, expected) in [(0, Vec3::X * 2.0), (1, Vec3::X * 4.0), (2, Vec3::Y * 3.0)] {
        for value in result.channel::<Vec3>(channel).unwrap() {
            assert!((value - expected).length() < 1e-6);
        }
    }
    for (body, depth) in result.channel::<Real>(3).unwrap().into_iter().enumerate() {
        assert!((depth - (body + 1).min(N - 1) as Real * 0.001).abs() < 1e-6);
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn auxiliary_bindings_reuse_ping_pong_and_invalidate_replaced_buffers() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut model = model();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut states = [model.state().unwrap(), model.state().unwrap()];
    for state in &mut states {
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
    }
    let mut contacts = [contacts(&engine), contacts(&engine)];
    let targets: Vec<_> = (0..N)
        .map(|i| FrameTarget {
            body: BodyId::from_index(i),
            local_pose: Transform::IDENTITY,
        })
        .collect();
    let mounts: Vec<_> = targets
        .iter()
        .map(|target| ImuMount {
            body: target.body,
            local_pose: target.local_pose,
        })
        .collect();
    let mut frames = FrameTransformSensor::new(&engine, &model, None, &targets).unwrap();
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut contact_sensor = ContactSensor::new(&engine, &model).unwrap();
    assert!(
        imu.dispatch(&engine, &model, &states[0], 0.01)
            .unwrap()
            .is_none()
    );
    let mut ring = ring(&engine, N as u64 * 36);
    let mut pending = Vec::new();
    for sample in 1..=16 {
        let index = sample % 2;
        let state = &mut states[index];
        let contact = &mut contacts[index];
        if sample == 8 {
            state.body_q =
                Buffer::from_vec(vec![Transform::IDENTITY; N], BufferUsage::StorageReadWrite);
            state
                .body_q
                .ensure_device(engine.device().unwrap())
                .unwrap();
            state.body_q.upload(engine.queue().unwrap()).unwrap();
            contact.depth = Buffer::from_vec(vec![0.0; N - 1], BufferUsage::StorageReadWrite);
            contact
                .depth
                .ensure_device(engine.device().unwrap())
                .unwrap();
            contact.depth.upload(engine.queue().unwrap()).unwrap();
        }
        let poses: Vec<_> = (0..N)
            .map(|i| Transform::new(Vec3::new(sample as Real, i as Real, 0.0), Quat::IDENTITY))
            .collect();
        let velocities = vec![SpatialVector::new(Vec3::X * sample as Real * 0.125, Vec3::ZERO); N];
        let depths: Vec<_> = (1..N).map(|i| i as Real * sample as Real * 0.002).collect();
        device_write(&engine, &mut state.body_q, &poses);
        device_write(&engine, &mut state.body_qd, &velocities);
        device_write(&engine, &mut contact.depth, &depths);
        let dt = if sample % 2 == 0 { 0.02 } else { 0.01 };
        let f = frames.dispatch(&engine, state).unwrap();
        let i = imu.dispatch(&engine, &model, state, dt).unwrap().unwrap();
        let c = contact_sensor.dispatch(&engine, contact).unwrap();
        pending.push((
            sample,
            dt,
            ring.try_enqueue(
                &[
                    f.world_position.readback_source(0..N).unwrap(),
                    i.accelerometer.readback_source(0..N).unwrap(),
                    c.max_depth.readback_source(0..N).unwrap(),
                ],
                ReadbackStamp {
                    sample_id: sample as u64,
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap(),
        ));
        let preparations = if sample == 1 {
            1
        } else if sample < 8 {
            2
        } else {
            3
        };
        for stats in [
            frames.dispatch_stats(),
            imu.dispatch_stats(),
            contact_sensor.dispatch_stats(),
        ] {
            assert_eq!(stats.binding_preparations, preparations, "sample {sample}");
            assert_eq!(stats.kernel_dispatches, sample as u64);
        }
        if pending.len() == 2 {
            while let Some((sample, dt, ticket)) = pending.pop() {
                let result = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
                assert_eq!(result.stamp.sample_id, sample as u64);
                let positions = result.channel::<Vec3>(0).unwrap();
                let accelerations = result.channel::<Vec3>(1).unwrap();
                let depths = result.channel::<Real>(2).unwrap();
                for body in 0..N {
                    assert!(
                        (positions[body] - Vec3::new(sample as Real, body as Real, 0.0)).length()
                            < 1e-5
                    );
                    assert!((accelerations[body] - Vec3::X * (0.125 / dt)).length() < 1e-5);
                    assert!(
                        (depths[body] - (body + 1).min(N - 1) as Real * sample as Real * 0.002)
                            .abs()
                            < 1e-5
                    );
                }
            }
        }
    }
    assert_eq!(frames.dispatch_stats().submissions, 16);
    assert_eq!(contact_sensor.dispatch_stats().submissions, 16);
    assert_eq!(imu.dispatch_stats().submissions, 17);
    assert_eq!(imu.dispatch_stats().history_copies, 17);
    for stats in [
        frames.readback_stats(),
        imu.readback_stats(),
        contact_sensor.readback_stats(),
    ] {
        assert_eq!(stats.submissions, 0);
        assert_eq!(stats.blocking_waits, 0);
        assert_eq!(stats.staging_allocations, 0);
    }
    let before = imu.dispatch_stats();
    imu.reset();
    assert!(
        imu.dispatch(&engine, &model, &states[0], 0.01)
            .unwrap()
            .is_none()
    );
    assert!(
        imu.dispatch(&engine, &model, &states[0], 0.01)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        imu.dispatch_stats().binding_preparations,
        before.binding_preparations
    );
    let before = contact_sensor.dispatch_stats();
    contacts[0].clear();
    contact_sensor.dispatch(&engine, &contacts[0]).unwrap();
    assert_eq!(
        contact_sensor.dispatch_stats().binding_preparations,
        before.binding_preparations
    );
    assert_eq!(contact_sensor.dispatch_stats().output_clears, 1);
    let before = imu.dispatch_stats();
    assert!(imu.dispatch(&engine, &model, &states[0], 0.0).is_err());
    assert_eq!(imu.dispatch_stats(), before);
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

fn model() -> Model {
    let mut builder = ModelBuilder::new();
    for _ in 0..N {
        builder.add_body(BodyParams::new());
    }
    builder.build().unwrap()
}

fn contacts(engine: &Engine) -> Contacts {
    let mut contacts = Contacts::with_capacity(N - 1);
    contacts.count = (N - 1) as u32;
    for i in 0..N - 1 {
        contacts.a_index.host_mut().unwrap()[i] = i as u32;
        contacts.b_index.host_mut().unwrap()[i] = (i + 1) as u32;
        contacts.normal.host_mut().unwrap()[i] = Vec3::Y;
        contacts.depth.host_mut().unwrap()[i] = (i + 1) as Real * 0.001;
        contacts.point.host_mut().unwrap()[i] = Vec3::X * i as Real;
    }
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    for i in 0..N {
        if i > 0 {
            indices.push((i - 1) as u32 | 0x80000000);
        }
        if i < N - 1 {
            indices.push(i as u32);
        }
        offsets.push(indices.len() as u32);
    }
    contacts.body_contact_offsets = Buffer::from_vec(offsets, BufferUsage::StorageReadWrite);
    contacts.body_contact_indices = Buffer::from_vec(indices, BufferUsage::StorageReadWrite);
    contacts
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    contacts
}

#[test]
#[ignore = "needs a real GPU"]
fn contacts_device_snapshots_grouped_collection_and_clear() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut contacts = contacts(&engine);
    let mut sensor = ContactSensor::new(&engine, &model()).unwrap();
    let mut ring = ring(&engine, N as u64 * 40);
    let mut tickets = Vec::new();
    for sample in 1..=2 {
        let depths: Vec<f32> = (1..N).map(|i| i as f32 * 0.001 * sample as f32).collect();
        engine.queue().unwrap().write_buffer(
            contacts.depth.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&depths),
        );
        contacts.depth.mark_device_dirty();
        let output = sensor.dispatch(&engine, &contacts).unwrap();
        assert!(output.count.host().is_err());
        assert!(output.max_depth.host().is_err());
        tickets.push(
            ring.try_enqueue(
                &[
                    output.count.readback_source(0..N).unwrap(),
                    output.max_depth.readback_source(0..N).unwrap(),
                    output.normal_sum.readback_source(0..N).unwrap(),
                    output.centroid.readback_source(0..N).unwrap(),
                ],
                ReadbackStamp {
                    sample_id: sample,
                    sensor_id: 44,
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap(),
        );
    }
    assert!(contacts.depth.host().is_err());
    assert_eq!(sensor.readback_stats().submissions, 0);
    assert_eq!(sensor.readback_stats().staging_allocations, 0);
    for sample in (1..=2).rev() {
        let frame = ring
            .wait(&tickets[sample - 1], Duration::from_secs(10))
            .unwrap();
        assert_eq!(frame.stamp.sample_id, sample as u64);
        assert_eq!(frame.byte_len(), N * 40);
        let count = frame.channel::<u32>(0).unwrap();
        let depth = frame.channel::<Real>(1).unwrap();
        let normal = frame.channel::<Vec3>(2).unwrap();
        let centroid = frame.channel::<Vec3>(3).unwrap();
        for i in 0..N {
            assert_eq!(count[i], if i == 0 || i == N - 1 { 1 } else { 2 });
            assert!((depth[i] - (i + 1).min(N - 1) as Real * 0.001 * sample as Real).abs() < 1e-6);
            let expected_normal = if i == 0 {
                -Vec3::Y
            } else if i == N - 1 {
                Vec3::Y
            } else {
                Vec3::ZERO
            };
            assert!((normal[i] - expected_normal).length() < 1e-6);
            let x = if i == 0 {
                0.0
            } else if i == N - 1 {
                (N - 2) as Real
            } else {
                i as Real - 0.5
            };
            assert!((centroid[i] - Vec3::X * x).length() < 1e-6);
        }
    }
    for _ in 0..3 {
        assert_eq!(sensor.update(&engine, &contacts).unwrap()[0].count, 1);
    }
    assert_eq!(sensor.readback_stats().staging_allocations, 1);
    assert_eq!(sensor.readback_stats().submissions, 3);
    assert_eq!(sensor.readback_stats().copied_bytes, 3 * N as u64 * 40);
    contacts.clear();
    let readings = sensor.update(&engine, &contacts).unwrap();
    for reading in readings {
        assert_eq!(reading.count, 0);
        assert_eq!(reading.max_depth, 0.0);
        assert_eq!(reading.normal_sum, Vec3::ZERO);
        assert_eq!(reading.centroid, Vec3::ZERO);
    }
    let empty = Model::empty();
    let mut sensor = ContactSensor::new(&engine, &empty).unwrap();
    assert!(
        sensor
            .update(&engine, &Contacts::with_capacity(0))
            .unwrap()
            .is_empty()
    );
    assert_eq!(sensor.readback_stats().staging_allocations, 0);
}

#[test]
#[ignore = "needs a real GPU"]
fn frame_device_poses_relative_oracle_and_ordered_gpu_copy() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let model = model();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let targets: Vec<_> = (0..N)
        .map(|i| FrameTarget {
            body: BodyId::from_index(i),
            local_pose: Transform::new(
                Vec3::new(0.1 * i as Real, 0.2, -0.3),
                Quat::from_rotation_x(0.01 * i as Real),
            ),
        })
        .collect();
    let reference = FrameTarget {
        body: BodyId::from_index(0),
        local_pose: Transform::new(Vec3::Y, Quat::from_rotation_z(0.3)),
    };
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut ring = ring(&engine, N as u64 * 64);
    let mut tickets = Vec::new();
    let mut expected = Vec::new();
    let mut copied =
        DeviceBuffer::from_host(&engine, vec![Vec3::ZERO; N], BufferUsage::StorageReadWrite)
            .unwrap();
    for sample in 1..=2 {
        let poses: Vec<_> = (0..N)
            .map(|i| {
                Transform::new(
                    Vec3::new(i as Real, sample as Real, i as Real * 0.02),
                    Quat::from_rotation_y(i as Real * 0.03 + sample as Real * 0.1),
                )
            })
            .collect();
        let gpu: Vec<_> = poses.iter().map(BufferElement::to_gpu).collect();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&gpu),
        );
        state.body_q.mark_device_dirty();
        let reference_world = poses[0].compose(&reference.local_pose);
        expected.push(
            targets
                .iter()
                .enumerate()
                .map(|(i, target)| {
                    let world = poses[i].compose(&target.local_pose);
                    (world, reference_world.inverse().compose(&world))
                })
                .collect::<Vec<_>>(),
        );
        let output = sensor.dispatch(&engine, &state).unwrap();
        assert_eq!(output.target_range, 1..N + 1);
        assert!(output.world_position.host().is_err());
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            output.world_position.device_buffer().unwrap(),
            16,
            copied.device_buffer().unwrap(),
            0,
            N as u64 * 16,
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        copied.mark_device_dirty();
        tickets.push(
            ring.try_enqueue(
                &[
                    copied.readback_source(0..N).unwrap(),
                    output
                        .world_rotation
                        .readback_source(output.target_range.clone())
                        .unwrap(),
                    output
                        .relative_position
                        .readback_source(output.target_range.clone())
                        .unwrap(),
                    output
                        .relative_rotation
                        .readback_source(output.target_range.clone())
                        .unwrap(),
                ],
                ReadbackStamp {
                    sample_id: sample,
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap(),
        );
    }
    assert!(state.body_q.host().is_err());
    assert_eq!(sensor.readback_stats().submissions, 0);
    for sample in (0..2).rev() {
        let frame = ring
            .wait(&tickets[sample], Duration::from_secs(10))
            .unwrap();
        let wp = frame.channel::<Vec3>(0).unwrap();
        let wr = frame.channel::<Quat>(1).unwrap();
        let rp = frame.channel::<Vec3>(2).unwrap();
        let rr = frame.channel::<Quat>(3).unwrap();
        for i in 0..N {
            let (world, relative) = expected[sample][i];
            assert!((wp[i] - world.position).length() < 3e-5);
            assert!((rp[i] - relative.position).length() < 3e-5);
            assert!((1.0 - wr[i].dot(world.rotation).abs()).abs() < 2e-6);
            assert!((1.0 - rr[i].dot(relative.rotation).abs()).abs() < 2e-6);
        }
    }
    for _ in 0..3 {
        let result = sensor.update(&engine, &state).unwrap();
        assert_eq!(result.len(), N);
        assert!((result[N - 1].world.position - expected[1][N - 1].0.position).length() < 3e-5);
    }
    assert_eq!(sensor.readback_stats().staging_allocations, 1);
    assert_eq!(sensor.readback_stats().submissions, 3);
    assert_eq!(sensor.readback_stats().copied_bytes, 3 * N as u64 * 64);
}

#[test]
#[ignore = "needs a real GPU"]
fn aux_sensors_reject_stale_inputs_invalid_capacity_and_foreign_engine() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let model = model();
    let mut contacts = contacts(&engine);
    let mut contact_sensor = ContactSensor::new(&engine, &model).unwrap();
    contacts.normal.host_mut().unwrap()[0] = Vec3::X;
    assert!(contact_sensor.dispatch(&engine, &contacts).is_err());
    contacts.normal.upload(engine.queue().unwrap()).unwrap();
    contacts.body_contact_offsets = Buffer::from_vec(vec![0, 1], BufferUsage::StorageReadWrite);
    assert!(contact_sensor.dispatch(&engine, &contacts).is_err());
    let target = FrameTarget {
        body: BodyId::from_index(0),
        local_pose: Transform::IDENTITY,
    };
    let mut frames = FrameTransformSensor::new(&engine, &model, None, &[target]).unwrap();
    let mut state = model.state().unwrap();
    assert!(frames.dispatch(&engine, &state).is_err());
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    state.body_q.host_mut().unwrap()[0].position = Vec3::Y;
    assert!(frames.dispatch(&engine, &state).is_err());
    state.body_q.upload(engine.queue().unwrap()).unwrap();
    assert!(
        frames
            .dispatch(&engine, &Model::empty().state().unwrap())
            .is_err()
    );
    for invalid in [
        Transform::new(Vec3::splat(f64::INFINITY), Quat::IDENTITY),
        Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
    ] {
        assert!(
            FrameTransformSensor::new(
                &engine,
                &model,
                None,
                &[FrameTarget {
                    local_pose: invalid,
                    ..target
                }]
            )
            .is_err()
        );
    }
    let mut huge = Model::empty();
    huge.body_count = usize::MAX;
    assert!(ContactSensor::new(&engine, &huge).is_err());
    assert!(FrameTransformSensor::new(&engine, &huge, None, &[target]).is_err());
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    assert!(
        contact_sensor
            .dispatch(&foreign, &Contacts::with_capacity(0))
            .is_err()
    );
    assert!(frames.dispatch(&foreign, &state).is_err());
    assert_eq!(contact_sensor.readback_stats().submissions, 0);
    assert_eq!(frames.readback_stats().submissions, 0);
}
