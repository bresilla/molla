#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sensors::readback::{
    ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource, ReadbackStamp,
};
use molla_sensors::{
    ContactSensor, FrameTarget, FrameTransformSensor, ImuMount, ImuSensor, SensorCommandBatch,
    SensorFrame, SensorFramePoolConfig,
};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires GPU"]
fn direct_imu_and_contact_bindings_stay_warm_across_four_slots_and_two_inputs() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, mounts) = fixture(&engine, 70);
    let mut second_state = model.state().unwrap();
    second_state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let contact_inputs = [contacts(&engine, 70), contacts(&engine, 70)];
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut contact = ContactSensor::new(&engine, &model).unwrap();
    let mut ip = imu.output_pool(&engine, config(4)).unwrap();
    let mut cp = contact.output_pool(&engine, config(4)).unwrap();
    let mut priming = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        imu.encode(
            &mut priming,
            &engine,
            &model,
            &state,
            0.1,
            &mut ip,
            Default::default()
        )
        .unwrap()
        .unwrap()
        .priming
    );
    drop(priming.submit().unwrap());
    drain(&engine);
    assert_eq!(ip.available_slots().unwrap(), 4);
    for round in 0..6 {
        let mut frames = Vec::new();
        for slot in 0..4 {
            let input = (slot + round) % 2;
            let state = if input == 0 { &state } else { &second_state };
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            assert!(
                !imu.encode(
                    &mut batch,
                    &engine,
                    &model,
                    state,
                    0.1,
                    &mut ip,
                    Default::default()
                )
                .unwrap()
                .unwrap()
                .priming
            );
            contact
                .encode(
                    &mut batch,
                    &engine,
                    &contact_inputs[input],
                    &mut cp,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            assert_eq!(submitted.stats.kernel_dispatches, 2);
            assert_eq!(submitted.stats.history_copies, 1);
            assert_eq!(submitted.stats.submissions, 1);
            frames.extend(submitted.into_frames());
        }
        assert_eq!(ip.available_slots().unwrap(), 0);
        assert_eq!(cp.available_slots().unwrap(), 0);
        let preparations = if round == 0 { 4 } else { 8 };
        assert_eq!(imu.dispatch_stats().binding_preparations, preparations);
        assert_eq!(contact.dispatch_stats().binding_preparations, preparations);
        drain(&engine);
        drop(frames);
        assert_eq!(ip.available_slots().unwrap(), 4);
        assert_eq!(cp.available_slots().unwrap(), 4);
    }
    assert_eq!(ip.stats().buffer_allocations, 8);
    assert_eq!(cp.stats().buffer_allocations, 16);
    assert_eq!(imu.dispatch_stats().submissions, 0);
    assert_eq!(contact.dispatch_stats().submissions, 0);
    assert_eq!(imu.readback_stats().copied_bytes, 0);
    assert_eq!(imu.readback_stats().blocking_waits, 0);
    assert_eq!(contact.readback_stats().copied_bytes, 0);
    assert_eq!(contact.readback_stats().blocking_waits, 0);
}

fn fixture(engine: &Engine, count: usize) -> (Model, State, Vec<ImuMount>) {
    let mut builder = ModelBuilder::new();
    let mounts = (0..count)
        .map(|i| {
            let body = builder.add_body(BodyParams::new().with_pose(Transform::new(
                Vec3::X * i as f64,
                Quat::from_rotation_y(i as f64 * 0.02),
            )));
            ImuMount {
                body,
                local_pose: Transform::new(
                    Vec3::Z * i as f64 * 0.1,
                    Quat::from_rotation_z(i as f64 * 0.03),
                ),
            }
        })
        .collect();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state, mounts)
}

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 20,
    }
}

fn write_velocity(engine: &Engine, state: &mut State, linear: f64, angular: f64) {
    let values: Vec<_> = (0..state.body_qd.len())
        .map(|_| SpatialVector::new(Vec3::X * linear, Vec3::Y * angular).to_gpu())
        .collect();
    engine.queue().unwrap().write_buffer(
        state.body_qd.device_buffer().unwrap(),
        0,
        bytemuck::cast_slice(&values),
    );
    state.body_qd.mark_device_dirty();
    assert!(state.body_qd.host().is_err());
}

fn contacts(engine: &Engine, count: usize) -> Contacts {
    let mut contacts = Contacts::with_capacity(count - 1);
    contacts.count = (count - 1) as u32;
    for i in 0..count - 1 {
        contacts.a_index.host_mut().unwrap()[i] = i as u32;
        contacts.b_index.host_mut().unwrap()[i] = (i + 1) as u32;
        contacts.normal.host_mut().unwrap()[i] = Vec3::Y;
        contacts.depth.host_mut().unwrap()[i] = (i + 1) as f64 * 0.001;
        contacts.point.host_mut().unwrap()[i] = Vec3::X * i as f64;
    }
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    for i in 0..count {
        if i > 0 {
            indices.push((i - 1) as u32 | Contacts::B_SIDE_BIT);
        }
        if i < count - 1 {
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

fn collect(
    engine: &Engine,
    sources: &[ReadbackSource<'_>],
    stamp: ReadbackStamp,
    bytes: u64,
) -> ReadbackFrame {
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes.max(8),
            max_total_bytes: bytes.max(8),
        },
    )
    .unwrap();
    let ticket = ring.try_enqueue(sources, stamp).unwrap().unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

fn imu_values(engine: &Engine, frame: &SensorFrame) -> (Vec<Vec3>, Vec<Vec3>) {
    let n = frame.channel::<Vec3>(0).unwrap().len();
    let cpu = collect(
        engine,
        &[
            frame
                .channel::<Vec3>(0)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
            frame
                .channel::<Vec3>(1)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
        ],
        frame.stamp(),
        n as u64 * 32,
    );
    (
        cpu.channel::<Vec3>(0).unwrap(),
        cpu.channel::<Vec3>(1).unwrap(),
    )
}

fn contact_values(engine: &Engine, frame: &SensorFrame) -> ReadbackFrame {
    let n = frame.channel::<u32>(0).unwrap().len();
    collect(
        engine,
        &[
            frame
                .channel::<u32>(0)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
            frame
                .channel::<Real>(1)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
            frame
                .channel::<Vec3>(2)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
            frame
                .channel::<Vec3>(3)
                .unwrap()
                .readback_source(0..n)
                .unwrap(),
        ],
        frame.stamp(),
        n as u64 * 40,
    )
}

#[test]
#[ignore = "requires GPU"]
fn mixed_auxiliary_producers_share_one_submission_and_match_legacy_at_70_mounts() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, mounts) = fixture(&engine, 70);
    let contacts = contacts(&engine, 70);
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut second = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut oracle = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut oracle_second = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut contact = ContactSensor::new(&engine, &model).unwrap();
    let mut contact_oracle = ContactSensor::new(&engine, &model).unwrap();
    let targets: Vec<_> = mounts
        .iter()
        .map(|mount| FrameTarget {
            body: mount.body,
            local_pose: mount.local_pose,
        })
        .collect();
    let mut frames = FrameTransformSensor::new(&engine, &model, None, &targets).unwrap();
    let mut ip = imu.output_pool(&engine, config(1)).unwrap();
    let mut sp = second.output_pool(&engine, config(1)).unwrap();
    let mut cp = contact.output_pool(&engine, config(1)).unwrap();
    let mut fp = frames.output_pool(&engine, config(1)).unwrap();
    write_velocity(&engine, &mut state, 1.0, 0.25);
    assert!(
        oracle
            .update(&engine, &model, &state, 0.1)
            .unwrap()
            .is_empty()
    );
    assert!(
        oracle_second
            .update(&engine, &model, &state, 0.2)
            .unwrap()
            .is_empty()
    );
    let mut prime = SensorCommandBatch::new(&engine).unwrap();
    let first = imu
        .encode(
            &mut prime,
            &engine,
            &model,
            &state,
            0.1,
            &mut ip,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let other = second
        .encode(
            &mut prime,
            &engine,
            &model,
            &state,
            0.2,
            &mut sp,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert!(first.priming && other.priming);
    assert_eq!(prime.stats().history_copies, 2);
    assert_eq!(prime.stats().kernel_dispatches, 0);
    let primed = prime.submit().unwrap();
    assert!(
        imu_values(&engine, primed.frame(&first.frame).unwrap())
            .0
            .is_empty()
    );
    assert!(
        imu_values(&engine, primed.frame(&other.frame).unwrap())
            .1
            .is_empty()
    );
    drop(primed);
    drain(&engine);
    assert_eq!(ip.available_slots().unwrap(), 1);
    write_velocity(&engine, &mut state, 2.0, 0.75);
    let expected = oracle.update(&engine, &model, &state, 0.1).unwrap();
    let expected_second = oracle_second.update(&engine, &model, &state, 0.2).unwrap();
    let expected_contacts = contact_oracle.update(&engine, &contacts).unwrap();
    let stamp = ReadbackStamp {
        sensor_id: u64::MAX,
        sample_id: 99,
        scene_revision: 7,
        simulation_time: Some(0.2),
    };
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let a = imu
        .encode(&mut batch, &engine, &model, &state, 0.1, &mut ip, stamp)
        .unwrap()
        .unwrap();
    let b = second
        .encode(&mut batch, &engine, &model, &state, 0.2, &mut sp, stamp)
        .unwrap()
        .unwrap();
    let c = contact
        .encode(&mut batch, &engine, &contacts, &mut cp, stamp)
        .unwrap()
        .unwrap();
    let f = frames
        .encode(&mut batch, &engine, &state, &mut fp, stamp)
        .unwrap()
        .unwrap();
    assert!(!a.priming && !b.priming);
    assert_eq!(batch.stats().submissions, 0);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 4);
    assert_eq!(submitted.stats.history_copies, 2);
    for (sample, oracle) in [(&a.frame, expected), (&b.frame, expected_second)] {
        let frame = submitted.frame(sample).unwrap();
        assert_eq!(frame.stamp(), stamp);
        let (accel, gyro) = imu_values(&engine, frame);
        for i in 0..70 {
            assert!((accel[i] - oracle[i].accelerometer).length() < 2e-4);
            assert!((gyro[i] - oracle[i].gyroscope).length() < 2e-5);
        }
    }
    let (accel, gyro) = imu_values(&engine, submitted.frame(&a.frame).unwrap());
    assert!((accel[0] - Vec3::X * 10.0).length() < 1e-5);
    assert!((gyro[0] - Vec3::Y * 0.75).length() < 1e-6);
    let cpu = contact_values(&engine, submitted.frame(&c).unwrap());
    let counts = cpu.channel::<u32>(0).unwrap();
    let depths = cpu.channel::<Real>(1).unwrap();
    let normals = cpu.channel::<Vec3>(2).unwrap();
    let centroids = cpu.channel::<Vec3>(3).unwrap();
    for i in 0..70 {
        assert_eq!(counts[i], expected_contacts[i].count);
        assert!((depths[i] - expected_contacts[i].max_depth).abs() < 1e-6);
        assert!((normals[i] - expected_contacts[i].normal_sum).length() < 1e-6);
        assert!((centroids[i] - expected_contacts[i].centroid).length() < 1e-5);
    }
    assert_eq!(
        submitted
            .frame(&f)
            .unwrap()
            .channel::<Vec3>(0)
            .unwrap()
            .len(),
        70
    );
    assert_eq!(imu.dispatch_stats().submissions, 0);
    assert_eq!(imu.readback_stats().copied_bytes, 0);
    assert_eq!(contact.readback_stats().copied_bytes, 0);
}

#[test]
#[ignore = "requires GPU"]
fn imu_cancellation_and_poisoned_batches_leave_history_unchanged_across_legacy_calls() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, mounts) = fixture(&engine, 1);
    let (discarded_model, discarded_state, _) = fixture(&engine, 1);
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut pool = imu.output_pool(&engine, config(2)).unwrap();
    write_velocity(&engine, &mut state, 1.0, 0.0);
    {
        let mut dropped = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            imu.encode(
                &mut dropped,
                &engine,
                &discarded_model,
                &discarded_state,
                1.0,
                &mut pool,
                Default::default()
            )
            .unwrap()
            .unwrap()
            .priming
        );
    }
    assert_eq!(pool.available_slots().unwrap(), 2);
    write_velocity(&engine, &mut state, 2.0, 0.0);
    let mut prime = SensorCommandBatch::new(&engine).unwrap();
    let p = imu
        .encode(
            &mut prime,
            &engine,
            &model,
            &state,
            1.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert!(p.priming);
    let primed = prime.submit().unwrap();
    let historical = primed.frame(&p.frame).unwrap();
    for mode in 0..3 {
        write_velocity(&engine, &mut state, 100.0 + mode as f64, 0.0);
        let mut cancelled = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            !imu.encode(
                &mut cancelled,
                &engine,
                &model,
                &state,
                1.0,
                &mut pool,
                Default::default()
            )
            .unwrap()
            .unwrap()
            .priming
        );
        if mode == 0 {
            drop(cancelled);
        } else {
            if mode == 1 {
                assert!(
                    cancelled
                        .encode_with_frames(&[historical], |_| Err(molla_core::Error::Gpu(
                            "injected".into()
                        )))
                        .is_err()
                );
            } else {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = cancelled.encode_with_frames(&[historical], |_| {
                            panic!("injected consumer panic")
                        });
                    }))
                    .is_err()
                );
            }
            assert!(cancelled.submit().is_err());
        }
        assert_eq!(pool.available_slots().unwrap(), 1);
    }
    write_velocity(&engine, &mut state, 5.0, 0.0);
    let mut measured = SensorCommandBatch::new(&engine).unwrap();
    let receipt = imu
        .encode(
            &mut measured,
            &engine,
            &model,
            &state,
            1.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = measured.submit().unwrap();
    assert!(
        (imu_values(&engine, submitted.frame(&receipt.frame).unwrap()).0[0] - Vec3::X * 3.0)
            .length()
            < 1e-6
    );
    drop(submitted);
    drain(&engine);
    write_velocity(&engine, &mut state, 7.0, 0.0);
    assert!(
        (imu.update(&engine, &model, &state, 1.0).unwrap()[0].accelerometer - Vec3::X * 2.0)
            .length()
            < 1e-6
    );
    write_velocity(&engine, &mut state, 10.0, 0.0);
    let mut after_legacy = SensorCommandBatch::new(&engine).unwrap();
    let receipt = imu
        .encode(
            &mut after_legacy,
            &engine,
            &model,
            &state,
            2.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = after_legacy.submit().unwrap();
    assert!(
        (imu_values(&engine, submitted.frame(&receipt.frame).unwrap()).0[0] - Vec3::X * 1.5)
            .length()
            < 1e-6
    );
}

#[test]
#[ignore = "requires GPU"]
fn imu_backpressure_reset_and_model_replacement_preserve_submitted_versions() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, mounts) = fixture(&engine, 1);
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut pool = imu.output_pool(&engine, config(1)).unwrap();
    write_velocity(&engine, &mut state, 2.0, 0.0);
    let mut prime = SensorCommandBatch::new(&engine).unwrap();
    let receipt = imu
        .encode(
            &mut prime,
            &engine,
            &model,
            &state,
            1.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    pool.reset().unwrap();
    let primed = prime.submit().unwrap();
    assert!(!pool.is_current(primed.frame(&receipt.frame).unwrap()));
    write_velocity(&engine, &mut state, 100.0, 0.0);
    let mut blocked = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        imu.encode(
            &mut blocked,
            &engine,
            &model,
            &state,
            1.0,
            &mut pool,
            Default::default()
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(blocked.stats().history_copies, 0);
    assert_eq!(blocked.submit().unwrap().stats.submissions, 0);
    drop(primed);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 1);
    write_velocity(&engine, &mut state, 4.0, 0.0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = imu
        .encode(
            &mut batch,
            &engine,
            &model,
            &state,
            1.0,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let old = submitted.frame(&receipt.frame).unwrap().clone();
    assert!((imu_values(&engine, &old).0[0] - Vec3::X * 2.0).length() < 1e-6);
    let (replacement, replacement_state, _) = fixture(&engine, 1);
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            imu.encode(
                &mut batch,
                &engine,
                &replacement,
                &replacement_state,
                1.0,
                &mut pool,
                Default::default()
            )
            .is_err()
        );
        assert_eq!(batch.stats().encoded_frames, 0);
    }
    imu.reset();
    let mut replacement_pool = imu.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = imu
        .encode(
            &mut batch,
            &engine,
            &replacement,
            &replacement_state,
            1.0,
            &mut replacement_pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert!(receipt.priming);
    let reset = batch.submit().unwrap();
    assert!(
        imu_values(&engine, reset.frame(&receipt.frame).unwrap())
            .0
            .is_empty()
    );
    assert!((imu_values(&engine, &old).0[0] - Vec3::X * 2.0).length() < 1e-6);
}

#[test]
#[ignore = "requires GPU"]
fn contact_empty_frames_clear_reused_outputs_and_zero_body_frames_have_empty_channels() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, _, _) = fixture(&engine, 70);
    let mut contacts = contacts(&engine, 70);
    let mut sensor = ContactSensor::new(&engine, &model).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(
            &mut batch,
            &engine,
            &contacts,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(
        contact_values(&engine, submitted.frame(&receipt).unwrap())
            .channel::<u32>(0)
            .unwrap()[1],
        2
    );
    drop(submitted);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 1);
    contacts.clear();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(
            &mut batch,
            &engine,
            &contacts,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().kernel_dispatches, 0);
    assert_eq!(batch.stats().output_clears, 4);
    let submitted = batch.submit().unwrap();
    let cpu = contact_values(&engine, submitted.frame(&receipt).unwrap());
    assert!(cpu.channel::<u32>(0).unwrap().iter().all(|v| *v == 0));
    assert!(cpu.channel::<Real>(1).unwrap().iter().all(|v| *v == 0.0));
    assert!(
        cpu.channel::<Vec3>(2)
            .unwrap()
            .iter()
            .all(|v| *v == Vec3::ZERO)
    );
    assert!(
        cpu.channel::<Vec3>(3)
            .unwrap()
            .iter()
            .all(|v| *v == Vec3::ZERO)
    );
    let mut empty_sensor = ContactSensor::new(&engine, &Model::empty()).unwrap();
    let mut empty_pool = empty_sensor
        .output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 40,
            },
        )
        .unwrap();
    let empty = Contacts::with_capacity(0);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = empty_sensor
        .encode(
            &mut batch,
            &engine,
            &empty,
            &mut empty_pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    assert_eq!(
        contact_values(&engine, submitted.frame(&receipt).unwrap()).byte_len(),
        0
    );
}

#[test]
#[ignore = "requires GPU"]
fn direct_auxiliary_validation_rejects_foreign_stale_and_malformed_inputs_before_history_work() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (model, mut state, mounts) = fixture(&engine, 70);
    let (foreign_model, foreign_state, _) = fixture(&foreign, 70);
    let (mut unverified_model, _, _) = fixture(&engine, 70);
    unverified_model.body_com = Buffer::from_vec(vec![Vec3::ZERO; 70], BufferUsage::Storage);
    unverified_model
        .body_com
        .ensure_device(engine.device().unwrap().as_ref())
        .unwrap();
    unverified_model
        .body_com
        .upload(engine.queue().unwrap())
        .unwrap();
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    let mut ip = imu.output_pool(&engine, config(1)).unwrap();
    for (input_model, input_state, dt) in [
        (&foreign_model, &state, 0.1),
        (&unverified_model, &state, 0.1),
        (&model, &foreign_state, 0.1),
        (&model, &state, 0.0),
        (&model, &state, f64::NAN),
        (&model, &state, f64::MAX),
        (&model, &state, 1e-40),
    ] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            imu.encode(
                &mut batch,
                &engine,
                input_model,
                input_state,
                dt,
                &mut ip,
                Default::default()
            )
            .is_err()
        );
        assert_eq!(batch.stats().history_copies, 0);
        assert_eq!(batch.submit().unwrap().stats.submissions, 0);
    }
    state.body_q.host_mut().unwrap()[0].position = Vec3::X;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            imu.encode(
                &mut batch,
                &engine,
                &model,
                &state,
                0.1,
                &mut ip,
                Default::default()
            )
            .is_err()
        );
    }
    state.body_q.upload(engine.queue().unwrap()).unwrap();
    assert!(imu.output_pool(&foreign, config(1)).is_err());
    let mut contact = ContactSensor::new(&engine, &model).unwrap();
    let mut cp = contact.output_pool(&engine, config(1)).unwrap();
    let foreign_contacts = contacts(&foreign, 70);
    let mut good = contacts(&engine, 70);
    for input in [&foreign_contacts, &Contacts::with_capacity(0)] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let result = contact.encode(&mut batch, &engine, input, &mut cp, Default::default());
        if input.count > 0 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_some());
        }
    }
    good.depth.host_mut().unwrap()[0] = 0.2;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            contact
                .encode(&mut batch, &engine, &good, &mut cp, Default::default())
                .is_err()
        );
    }
    good.depth.upload(engine.queue().unwrap()).unwrap();
    good.normal = Buffer::from_vec(vec![Vec3::Y; 69], BufferUsage::StorageReadWrite);
    good.normal
        .ensure_device(engine.device().unwrap().as_ref())
        .unwrap();
    good.normal.upload(engine.queue().unwrap()).unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            contact
                .encode(&mut batch, &engine, &good, &mut cp, Default::default())
                .is_err()
        );
        assert_eq!(batch.stats().encoded_frames, 0);
    }
    good.count = u32::MAX;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            contact
                .encode(&mut batch, &engine, &good, &mut cp, Default::default())
                .is_err()
        );
    }
    {
        let mut batch = SensorCommandBatch::new(&foreign).unwrap();
        assert!(
            imu.encode(
                &mut batch,
                &foreign,
                &model,
                &state,
                0.1,
                &mut ip,
                Default::default()
            )
            .is_err()
        );
        assert_eq!(batch.stats().history_copies, 0);
    }
    assert_eq!(ip.stats().reserved_frames, 0);
    let mut prime = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        imu.encode(
            &mut prime,
            &engine,
            &model,
            &state,
            0.1,
            &mut ip,
            Default::default()
        )
        .unwrap()
        .unwrap()
        .priming
    );
    engine.destroy_device().unwrap();
    assert!(prime.submit().is_err());
    assert!(ip.available_slots().is_err());
    assert!(contact.output_pool(&engine, config(1)).is_err());
}

#[test]
#[ignore = "requires GPU"]
fn contact_backpressure_foreign_pools_and_device_loss_are_explicit() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (model, _, _) = fixture(&engine, 70);
    let (foreign_model, _, _) = fixture(&foreign, 70);
    let inputs = contacts(&engine, 70);
    let mut sensor = ContactSensor::new(&engine, &model).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();

    // A completed but still held frame keeps its only slot.
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(&mut batch, &engine, &inputs, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    let held = batch.submit().unwrap().frame(&receipt).unwrap().clone();
    drain(&engine);
    let preparations = sensor.dispatch_stats().binding_preparations;
    {
        let mut blocked = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(&mut blocked, &engine, &inputs, &mut pool, Default::default())
                .unwrap()
                .is_none()
        );
        assert_eq!(blocked.stats().encoded_frames, 0);
        assert_eq!(blocked.submit().unwrap().stats.submissions, 0);
    }
    assert_eq!(pool.stats().backpressure, 1);
    assert_eq!(sensor.dispatch_stats().binding_preparations, preparations);
    assert_eq!(contact_values(&engine, &held).channel::<u32>(0).unwrap()[1], 2);
    drop(held);
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        sensor
            .encode(&mut batch, &engine, &inputs, &mut pool, Default::default())
            .unwrap()
            .is_some()
    );
    drop(batch.submit().unwrap());
    drain(&engine);

    // Foreign pools and batches fail before any slot is reserved.
    assert!(sensor.output_pool(&foreign, config(1)).is_err());
    let foreign_sensor = ContactSensor::new(&foreign, &foreign_model).unwrap();
    let mut foreign_pool = foreign_sensor.output_pool(&foreign, config(1)).unwrap();
    let reserved = pool.stats().reserved_frames;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(&mut batch, &engine, &inputs, &mut foreign_pool, Default::default())
                .is_err()
        );
        assert_eq!(batch.stats().encoded_frames, 0);
    }
    {
        let mut batch = SensorCommandBatch::new(&foreign).unwrap();
        assert!(
            sensor
                .encode(&mut batch, &foreign, &inputs, &mut pool, Default::default())
                .is_err()
        );
    }
    assert_eq!(foreign_pool.stats().reserved_frames, 0);
    assert_eq!(pool.stats().reserved_frames, reserved);

    // Known device loss fails the pending batch and every later use.
    let mut pending = SensorCommandBatch::new(&engine).unwrap();
    sensor
        .encode(&mut pending, &engine, &inputs, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    engine.destroy_device().unwrap();
    assert!(pending.submit().is_err());
    assert!(pool.available_slots().is_err());
    assert!(sensor.output_pool(&engine, config(1)).is_err());
    if let Ok(mut batch) = SensorCommandBatch::new(&engine) {
        assert!(
            sensor
                .encode(&mut batch, &engine, &inputs, &mut pool, Default::default())
                .is_err()
        );
    }
}
