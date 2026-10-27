#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    FrameTarget, FrameTransformSensor, SensorChannelSpec, SensorCommandBatch, SensorFrame,
    SensorFramePool, SensorFramePoolConfig,
};
use molla_sim::{BodyParams, Model, ModelBuilder, State};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires GPU"]
fn deferred_gpu_consumer_keeps_direct_frame_leased_after_cpu_release() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, reference, targets) = fixture(&engine);
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    let device = engine.device().unwrap();
    let mut output = molla_sensors::readback::DeviceBuffer::from_host(
        &engine,
        vec![Vec3::ZERO],
        molla_storage::BufferUsage::StorageReadWrite,
    )
    .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("direct-output consumer"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> result: array<vec4<f32>>;
            @compute @workgroup_size(1) fn main() { result[0] = positions[1] * 2.0; }
        "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut producer = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(
            &mut producer,
            &engine,
            &state,
            &mut pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    let submitted = producer.submit().unwrap();
    let frame = submitted.frame(&receipt).unwrap();
    let channel = frame.channel::<Vec3>(0).unwrap();
    assert_eq!(channel.active_range().start, 1);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(channel.binding().unwrap()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.device_buffer().unwrap().as_entire_binding(),
            },
        ],
    });
    let mut consumer = SensorCommandBatch::new(&engine).unwrap();
    consumer
        .encode_with_frames(&[frame], |encoder| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            Ok(())
        })
        .unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let mut wrong = SensorCommandBatch::new(&foreign).unwrap();
    assert!(
        wrong
            .encode_with_frames(&[frame], |_| panic!("must reject foreign frame"))
            .is_err()
    );
    let mut failed = SensorCommandBatch::new(&engine).unwrap();
    assert!(
        failed
            .encode_with_frames(&[frame], |_| Err(molla_core::Error::Gpu(
                "injected consumer error".into()
            )))
            .is_err()
    );
    assert!(failed.submit().is_err());
    drop(submitted);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 0);
    let completion = consumer.submit().unwrap();
    assert_eq!(completion.stats.submissions, 1);
    assert!(completion.index.is_some());
    drop(completion);
    output.mark_device_dirty();
    assert!(output.host().is_err());
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 16,
            max_total_bytes: 16,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(&[output.readback_source(0..1).unwrap()], Default::default())
        .unwrap()
        .unwrap();
    let cpu = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    assert!((cpu.channel::<Vec3>(0).unwrap()[0] - Vec3::new(14.0, -4.0, 2.0)).length() < 1e-5);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 1);
    assert_eq!(sensor.readback_stats().copied_bytes, 0);
}

fn fixture(engine: &Engine) -> (Model, State, FrameTarget, Vec<FrameTarget>) {
    let mut builder = ModelBuilder::new();
    let a = builder.add_body(BodyParams::new().with_pose(Transform::new(
        Vec3::new(7.0, -2.0, 1.0),
        Quat::from_rotation_z(0.4),
    )));
    let b = builder.add_body(BodyParams::new().with_pose(Transform::new(
        Vec3::new(2.0, 3.0, -5.0),
        Quat::from_rotation_y(-0.7),
    )));
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let reference = FrameTarget {
        body: a,
        local_pose: Transform::new(Vec3::X, Quat::from_rotation_x(0.2)),
    };
    let targets = (0..65)
        .map(|i| FrameTarget {
            body: if i % 2 == 0 { a } else { b },
            local_pose: Transform::new(Vec3::Z * i as f64, Quat::from_rotation_y(0.03 * i as f64)),
        })
        .collect();
    (model, state, reference, targets)
}

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 1 << 20,
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

fn collect(engine: &Engine, frame: &SensorFrame) -> molla_sensors::readback::ReadbackFrame {
    let count = frame.channel::<Vec3>(0).unwrap().len();
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: count as u64 * 64,
            max_total_bytes: count as u64 * 64,
        },
    )
    .unwrap();
    let sources = [
        frame
            .channel::<Vec3>(0)
            .unwrap()
            .readback_source(0..count)
            .unwrap(),
        frame
            .channel::<Quat>(1)
            .unwrap()
            .readback_source(0..count)
            .unwrap(),
        frame
            .channel::<Vec3>(2)
            .unwrap()
            .readback_source(0..count)
            .unwrap(),
        frame
            .channel::<Quat>(3)
            .unwrap()
            .readback_source(0..count)
            .unwrap(),
    ];
    let ticket = ring.try_enqueue(&sources, frame.stamp()).unwrap().unwrap();
    ring.wait(&ticket, Duration::from_secs(30)).unwrap()
}

#[test]
#[ignore = "requires GPU"]
fn two_producers_one_submission_match_legacy_and_preserve_reference_ranges() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, reference, targets) = fixture(&engine);
    let mut a = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut b = FrameTransformSensor::new(&engine, &model, None, &targets).unwrap();
    let oracle_a = a.update(&engine, &state).unwrap();
    let oracle_b = b.update(&engine, &state).unwrap();
    let a_stats = a.readback_stats();
    let b_stats = b.readback_stats();
    let mut pool_a = a.output_pool(&engine, config(1)).unwrap();
    let mut pool_b = b.output_pool(&engine, config(1)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let stamp = ReadbackStamp {
        sensor_id: u64::MAX,
        sample_id: u64::MAX - 1,
        scene_revision: 73,
        simulation_time: Some(2.5),
    };
    let pa = a
        .encode(&mut batch, &engine, &state, &mut pool_a, stamp)
        .unwrap()
        .unwrap();
    let pb = b
        .encode(&mut batch, &engine, &state, &mut pool_b, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().encoded_frames, 2);
    assert_eq!(pool_a.available_slots().unwrap(), 0);
    let submitted = batch.submit().unwrap();
    assert!(submitted.index.is_some());
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.kernel_dispatches, 2);
    assert_eq!(a.readback_stats().copied_bytes, a_stats.copied_bytes);
    assert_eq!(b.readback_stats().copied_bytes, b_stats.copied_bytes);
    for (receipt, expected, start) in [(&pa, oracle_a, 1), (&pb, oracle_b, 0)] {
        let frame = submitted.frame(receipt).unwrap();
        assert_eq!(frame.channel_count(), 4);
        assert_eq!(
            frame.channel::<Vec3>(0).unwrap().active_range(),
            start..start + 65
        );
        assert_eq!(
            frame.channel::<Vec3>(0).unwrap().binding().unwrap().offset,
            0
        );
        assert!(frame.channel::<Quat>(0).is_err());
        assert!(frame.channel::<Vec3>(4).is_err());
        assert!(
            frame
                .channel::<Vec3>(0)
                .unwrap()
                .readback_source(0..66)
                .is_err()
        );
        let cpu = collect(&engine, frame);
        assert_eq!(cpu.byte_len(), 65 * 64);
        let wp = cpu.channel::<Vec3>(0).unwrap();
        let wr = cpu.channel::<Quat>(1).unwrap();
        let rp = cpu.channel::<Vec3>(2).unwrap();
        let rr = cpu.channel::<Quat>(3).unwrap();
        for i in 0..65 {
            assert!((wp[i] - expected[i].world.position).length() < 1e-5);
            assert!((wr[i] - expected[i].world.rotation).length() < 1e-5);
            assert!((rp[i] - expected[i].relative.position).length() < 1e-5);
            assert!((rr[i] - expected[i].relative.rotation).length() < 1e-5);
        }
    }
    assert_eq!(submitted.frame(&pa).unwrap().stamp(), stamp);
    let other = SensorCommandBatch::new(&engine).unwrap().submit().unwrap();
    assert!(other.index.is_none());
    assert_eq!(other.stats.submissions, 0);
    assert!(other.frame(&pa).is_err());
}

#[test]
#[ignore = "requires GPU"]
fn pending_drop_reset_backpressure_and_readback_hold_distinct_versions() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, mut state, reference, targets) = fixture(&engine);
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        sensor
            .encode(&mut batch, &engine, &state, &mut pool, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(pool.available_slots().unwrap(), 0);
    }
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(&mut batch, &engine, &state, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    pool.reset().unwrap();
    let submitted = batch.submit().unwrap();
    let old = submitted.frame(&receipt).unwrap().clone();
    assert!(!pool.is_current(&old));
    let before = collect(&engine, &old).channel::<Vec3>(0).unwrap();
    drop(submitted);
    state.body_q.host_mut().unwrap()[0].position += Vec3::new(100.0, 2.0, 3.0);
    state.body_q.upload(engine.queue().unwrap()).unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(&mut batch, &engine, &state, &mut pool, Default::default())
                .unwrap()
                .is_none()
        );
        assert_eq!(batch.stats().encoded_frames, 0);
        assert_eq!(batch.submit().unwrap().stats.submissions, 0);
    }
    assert_eq!(pool.stats().backpressure, 1);
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 16,
            max_total_bytes: 16,
        },
    )
    .unwrap();
    let source = old
        .channel::<Vec3>(0)
        .unwrap()
        .readback_source(0..1)
        .unwrap();
    let ticket = ring.try_enqueue(&[source], old.stamp()).unwrap().unwrap();
    drop(old);
    let cpu = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    assert_eq!(cpu.channel::<Vec3>(0).unwrap()[0], before[0]);
    drain(&engine);
    assert_eq!(pool.available_slots().unwrap(), 1);
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(&mut batch, &engine, &state, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let frame = submitted.frame(&receipt).unwrap();
    assert!(pool.is_current(frame));
    let after = collect(&engine, frame).channel::<Vec3>(0).unwrap();
    assert!((after[0] - before[0] - Vec3::new(100.0, 2.0, 3.0)).length() < 1e-4);
    drop(pool);
    drop(sensor);
    assert_eq!(collect(&engine, frame).channel::<Vec3>(0).unwrap(), after);
}

#[test]
#[ignore = "requires GPU"]
fn four_slots_two_pose_buffers_keep_warm_gpu_allocations_and_bindings_constant() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, reference, targets) = fixture(&engine);
    let mut second = model.state().unwrap();
    second
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut pool = sensor.output_pool(&engine, config(4)).unwrap();
    for round in 0..8 {
        let mut frames = Vec::new();
        for slot in 0..4 {
            let input = if (slot + round) % 2 == 0 {
                &state
            } else {
                &second
            };
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            sensor
                .encode(&mut batch, &engine, input, &mut pool, Default::default())
                .unwrap()
                .unwrap();
            frames.extend(batch.submit().unwrap().into_frames());
        }
        assert_eq!(pool.available_slots().unwrap(), 0);
        assert_eq!(
            sensor.dispatch_stats().binding_preparations,
            if round == 0 { 4 } else { 8 }
        );
        drain(&engine);
        drop(frames);
        assert_eq!(pool.available_slots().unwrap(), 4);
    }
    assert_eq!(pool.stats().buffer_allocations, 16);
    assert_eq!(pool.stats().reserved_frames, 32);
    assert_eq!(sensor.dispatch_stats().submissions, 0);
    assert_eq!(sensor.readback_stats().copied_bytes, 0);
}

#[test]
#[ignore = "requires GPU"]
fn foreign_inputs_pools_layouts_and_stale_state_reject_before_encoding() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let (model, mut state, reference, targets) = fixture(&engine);
    let (_, foreign_state, _, _) = fixture(&foreign);
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    assert!(sensor.output_pool(&foreign, config(1)).is_err());
    let cloned_owner = engine.device().unwrap().clone();
    state.body_q.device_buffer_for(&cloned_owner).unwrap();
    let wrapped_clone = std::sync::Arc::new(engine.device().unwrap().as_ref().clone());
    assert!(state.body_q.device_buffer_for(&wrapped_clone).is_err());
    let mut unverified = model.state().unwrap();
    unverified
        .upload_to_device(engine.device().unwrap().as_ref(), engine.queue().unwrap())
        .unwrap();
    unverified
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    assert!(
        unverified
            .body_q
            .device_buffer_for(engine.device().unwrap())
            .is_err()
    );
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(
                    &mut batch,
                    &engine,
                    &unverified,
                    &mut pool,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.stats().encoded_frames, 0);
    }
    assert!(
        state
            .body_q
            .ensure_device(foreign.device().unwrap())
            .is_err()
    );
    assert!(
        state
            .body_q
            .device_buffer_for(foreign.device().unwrap())
            .is_err()
    );
    state
        .body_q
        .ensure_device(engine.device().unwrap())
        .unwrap();
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(
                    &mut batch,
                    &engine,
                    &foreign_state,
                    &mut pool,
                    Default::default()
                )
                .is_err()
        );
        assert_eq!(batch.submit().unwrap().stats.submissions, 0);
    }
    state.body_q.host_mut().unwrap()[0].position = Vec3::ZERO;
    {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(&mut batch, &engine, &state, &mut pool, Default::default())
                .is_err()
        );
    }
    state.body_q.upload(engine.queue().unwrap()).unwrap();
    let specs = [
        SensorChannelSpec::of::<Vec3>(66),
        SensorChannelSpec::of::<Quat>(66),
        SensorChannelSpec::of::<Vec3>(66),
        SensorChannelSpec::of::<Quat>(66),
    ];
    let mut foreign_pool = SensorFramePool::new(&foreign, config(1), &specs).unwrap();
    let mut wrong_pool =
        SensorFramePool::new(&engine, config(1), &[SensorChannelSpec::of::<u32>(66); 4]).unwrap();
    for output in [&mut foreign_pool, &mut wrong_pool] {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            sensor
                .encode(&mut batch, &engine, &state, output, Default::default())
                .is_err()
        );
        assert_eq!(batch.stats().output_clears, 0);
    }
    let mut foreign_batch = SensorCommandBatch::new(&foreign).unwrap();
    assert!(
        sensor
            .encode(
                &mut foreign_batch,
                &engine,
                &state,
                &mut pool,
                Default::default()
            )
            .is_err()
    );
    drop(foreign_batch);
    assert_eq!(sensor.dispatch_stats().binding_preparations, 0);
    assert_eq!(pool.stats().reserved_frames, 0);
    assert_eq!(pool.available_slots().unwrap(), 1);
    for bad in [
        config(0),
        config(65),
        SensorFramePoolConfig {
            slots: 1,
            max_total_bytes: 1,
        },
    ] {
        assert!(SensorFramePool::new(&engine, bad, &specs).is_err());
    }
    assert!(SensorFramePool::new(&engine, config(1), &[]).is_err());
    assert!(
        SensorFramePool::new(
            &engine,
            config(1),
            &[SensorChannelSpec::of::<Vec3>(usize::MAX)]
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires GPU"]
fn device_loss_invalidates_direct_frames_and_unsubmitted_batches() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state, reference, targets) = fixture(&engine);
    let mut sensor = FrameTransformSensor::new(&engine, &model, Some(reference), &targets).unwrap();
    let mut pool = sensor.output_pool(&engine, config(2)).unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(&mut batch, &engine, &state, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    let submitted = batch.submit().unwrap();
    let frame = submitted.frame(&receipt).unwrap();
    let channel = frame.channel::<Vec3>(0).unwrap();
    let collected = collect(&engine, frame);
    let mut pending = SensorCommandBatch::new(&engine).unwrap();
    sensor
        .encode(&mut pending, &engine, &state, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    engine.destroy_device().unwrap();
    assert!(pending.submit().is_err());
    assert!(SensorCommandBatch::new(&engine).is_err());
    assert!(sensor.output_pool(&engine, config(1)).is_err());
    assert!(frame.channel::<Vec3>(0).is_err());
    assert!(channel.binding().is_err());
    assert!(channel.readback_source(0..1).is_err());
    assert!(!pool.is_current(frame));
    assert!(pool.available_slots().is_err());
    assert!(pool.reset().is_err());
    assert_eq!(collected.channel::<Vec3>(0).unwrap().len(), 65);
}
