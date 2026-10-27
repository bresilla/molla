#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, Error};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp};
use molla_sensors::{
    FrameTarget, FrameTransformSensor, Raycaster, SensorChannelSpec, SensorCommandBatch,
    SensorFramePool, SensorFramePoolConfig, SensorScene, SensorSceneOptions,
};
use molla_sim::{BodyParams, Model, ModelBuilder, State};
use molla_storage::BufferUsage;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn fixture(engine: &Engine) -> (Model, State, FrameTransformSensor) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new().with_pose(Transform::new(Vec3::new(10.0, 3.0, 4.0), Quat::IDENTITY)),
    );
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let reference = FrameTarget {
        body,
        local_pose: Transform::IDENTITY,
    };
    let targets: Vec<_> = (0..65)
        .map(|i| FrameTarget {
            body,
            local_pose: Transform::new(Vec3::X * i as f64, Quat::IDENTITY),
        })
        .collect();
    let sensor = FrameTransformSensor::new(engine, &model, Some(reference), &targets).unwrap();
    (model, state, sensor)
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

#[test]
#[ignore = "requires GPU"]
fn frame_and_ray_outputs_feed_typed_multistage_gpu_work_in_one_submission() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let (model, state, mut frames) = fixture(&engine);
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut rays = Raycaster::new(&engine, 65).unwrap();
    let mut frame_pool = frames.output_pool(&engine, config(1)).unwrap();
    let mut ray_pool = rays.output_pool(&engine, config(1)).unwrap();
    let specs = [
        SensorChannelSpec::of::<Vec3>(70),
        SensorChannelSpec::of::<u32>(70),
    ];
    let mut first_pool = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    let final_specs = [SensorChannelSpec::of::<Vec3>(8)];
    let mut final_pool = SensorFramePool::new(&engine, config(1), &final_specs).unwrap();
    let mut external =
        DeviceBuffer::from_host(&engine, vec![Vec3::ZERO; 7], BufferUsage::StorageReadWrite)
            .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pending mixed sensor consumer"),
        source: wgpu::ShaderSource::Wgsl(r#"
            @group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read> rotations: array<vec4<f32>>;
            @group(0) @binding(2) var<storage, read> distance: array<f32>;
            @group(0) @binding(3) var<storage, read> shape: array<i32>;
            @group(0) @binding(4) var<storage, read_write> combined: array<vec4<f32>>;
            @group(0) @binding(5) var<storage, read_write> counts: array<u32>;
            @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if i >= 65u { return; }
                let p = positions[i + 1u];
                combined[i + 2u] = vec4<f32>(p.x + distance[i], p.y + f32(shape[i]), p.z + rotations[i + 1u].w, 0.0);
                counts[i + 2u] = u32(distance[i]);
            }
        "#.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let stamp = ReadbackStamp {
        sensor_id: u64::MAX,
        sample_id: u64::MAX - 1,
        scene_revision: 91,
        simulation_time: Some(3.5),
    };
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let frame = frames
        .encode(&mut batch, &engine, &state, &mut frame_pool, stamp)
        .unwrap()
        .unwrap();
    let origins: Vec<_> = (0..65)
        .map(|i| Vec3::new(i as f64, 10.0 + i as f64, 0.0))
        .collect();
    let ray = rays
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &origins,
            &[Vec3::NEG_Y; 65],
            100.0,
            &mut ray_pool,
            stamp,
        )
        .unwrap()
        .unwrap();
    let first = batch
        .encode_from_pending(
            &[&frame, &ray.frame],
            &mut first_pool,
            &specs,
            &[2..67, 2..67],
            stamp,
            |encoder, inputs, outputs| {
                assert_eq!(inputs.len(), 2);
                assert!(!inputs.is_empty());
                assert_eq!(inputs.stamp(0).unwrap().sensor_id, u64::MAX);
                let p = inputs.channel::<Vec3>(0, 0)?;
                let q = inputs.channel::<Quat>(0, 1)?;
                let d = inputs.channel::<Real>(1, 2)?;
                let s = inputs.channel::<i32>(1, 3)?;
                assert_eq!(p.active_range(), 1..66);
                assert_eq!(q.active_range(), 1..66);
                assert_eq!(d.active_range(), 0..65);
                assert_eq!(p.binding()?.offset, 0);
                assert_eq!(p.slice(65..65)?.active_range(), 66..66);
                assert!(p.slice(65..65)?.is_empty());
                assert!(p.slice(0..66).is_err());
                assert!(p.slice(std::ops::Range { start: 2, end: 1 }).is_err());
                assert!(inputs.channel::<u32>(0, 0).is_err());
                assert!(inputs.channel::<Vec3>(2, 0).is_err());
                assert!(inputs.channel::<Vec3>(0, 4).is_err());
                assert!(inputs.stamp(2).is_err());
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(p.binding()?),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(q.binding()?),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(d.binding()?),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(s.binding()?),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: outputs[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: outputs[1].as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(2, 1, 1);
                Ok(1)
            },
        )
        .unwrap()
        .unwrap();
    let second = batch
        .encode_from_pending(
            &[&first],
            &mut final_pool,
            &final_specs,
            std::slice::from_ref(&(1..8)),
            stamp,
            |encoder, inputs, outputs| {
                let channel = inputs.channel::<Vec3>(0, 0)?;
                let view = channel.slice(3..10)?;
                assert_eq!(view.active_range(), 5..12);
                encoder.copy_buffer_to_buffer(
                    view.binding()?.buffer,
                    view.active_range().start as u64 * 16,
                    &outputs[0],
                    16,
                    view.len() as u64 * 16,
                );
                Ok(0)
            },
        )
        .unwrap()
        .unwrap();
    batch
        .encode_with_pending(&[&second], |encoder, inputs| {
            let view = inputs.channel::<Vec3>(0, 0)?;
            encoder.copy_buffer_to_buffer(
                view.binding()?.buffer,
                16,
                external.device_buffer()?,
                0,
                7 * 16,
            );
            Ok(0)
        })
        .unwrap();
    first_pool.reset().unwrap();
    final_pool.reset().unwrap();
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().downstream_encodes, 3);
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.submissions, 1);
    assert_eq!(submitted.stats.encoded_frames, 4);
    assert_eq!(submitted.stats.kernel_dispatches, 3);
    assert!(!final_pool.is_current(submitted.frame(&second).unwrap()));
    let old = submitted.frame(&second).unwrap().clone();
    let counts = submitted.frame(&first).unwrap().channel::<u32>(1).unwrap();
    external.mark_device_dirty();
    let mut ring = ReadbackRing::new(
        &engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: 1024,
            max_total_bytes: 1024,
        },
    )
    .unwrap();
    let ticket = ring
        .try_enqueue(
            &[
                external.readback_source(0..7).unwrap(),
                old.channel::<Vec3>(0)
                    .unwrap()
                    .readback_source(0..7)
                    .unwrap(),
                counts.readback_source(0..65).unwrap(),
            ],
            stamp,
        )
        .unwrap()
        .unwrap();
    drop(submitted);
    drop(first_pool);
    drop(final_pool);
    drop(frames);
    drop(rays);
    let cpu = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
    let expected: Vec<_> = (3..10)
        .map(|i| Vec3::new(20.0 + 2.0 * i as f64, 3.0, 5.0))
        .collect();
    assert_eq!(cpu.channel::<Vec3>(0).unwrap(), expected);
    assert_eq!(cpu.channel::<Vec3>(1).unwrap(), expected);
    assert_eq!(
        cpu.channel::<u32>(2).unwrap(),
        (10..75).collect::<Vec<u32>>()
    );
    drain(&engine);
    assert_eq!(frame_pool.available_slots().unwrap(), 1);
    assert_eq!(ray_pool.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn validation_backpressure_and_callback_failures_do_not_publish_partial_chains() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_, state, mut sensor) = fixture(&engine);
    let mut pool = sensor.output_pool(&engine, config(1)).unwrap();
    let specs = [SensorChannelSpec::of::<Vec3>(1)];
    let mut output = SensorFramePool::new(&engine, config(1), &specs).unwrap();
    for failure in 0..4 {
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let receipt = sensor
            .encode(&mut batch, &engine, &state, &mut pool, Default::default())
            .unwrap()
            .unwrap();
        let mut wrong = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            wrong
                .encode_with_pending(&[&receipt], |_, _| panic!("foreign receipt"))
                .is_err()
        );
        assert!(
            wrong
                .encode_from_pending(
                    &[&receipt],
                    &mut output,
                    &specs,
                    std::slice::from_ref(&(0..1)),
                    Default::default(),
                    |_, _, _| panic!("foreign receipt")
                )
                .is_err()
        );
        assert!(wrong.submit().unwrap().index.is_none());
        assert!(
            batch
                .encode_with_pending(&[], |_, _| panic!("empty inputs"))
                .is_err()
        );
        assert!(
            batch
                .encode_with_pending(&vec![&receipt; 65], |_, _| panic!("too many inputs"))
                .is_err()
        );
        assert!(
            batch
                .encode_from_pending(
                    &[&receipt],
                    &mut output,
                    &specs,
                    std::slice::from_ref(&(0..2)),
                    Default::default(),
                    |_, _, _| panic!("invalid output")
                )
                .is_err()
        );
        assert_eq!(output.stats().reserved_frames, failure as u64);
        let stage = batch
            .encode_from_pending(
                &[&receipt],
                &mut output,
                &specs,
                std::slice::from_ref(&(0..1)),
                Default::default(),
                |encoder, inputs, outputs| {
                    let p = inputs.channel::<Vec3>(0, 0)?;
                    encoder.copy_buffer_to_buffer(p.binding()?.buffer, 16, &outputs[0], 0, 16);
                    Ok(0)
                },
            )
            .unwrap()
            .unwrap();
        let before = batch.stats();
        assert!(
            batch
                .encode_from_pending(
                    &[&stage],
                    &mut output,
                    &specs,
                    std::slice::from_ref(&(0..1)),
                    Default::default(),
                    |_, _, _| panic!("backpressure")
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(batch.stats().downstream_encodes, before.downstream_encodes);
        assert_eq!(batch.stats().output_clears, before.output_clears);
        match failure {
            0 => {
                drop(batch);
            }
            1 => {
                assert!(
                    batch
                        .encode_with_pending(&[&stage], |_, inputs| {
                            inputs.channel::<u32>(0, 0)?;
                            Ok(0)
                        })
                        .is_err()
                );
                assert!(batch.submit().is_err());
            }
            2 => {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = batch.encode_with_pending(&[&stage], |_, _| {
                            panic!("injected callback panic")
                        });
                    }))
                    .is_err()
                );
                assert!(batch.submit().is_err());
            }
            _ => {
                assert!(
                    batch
                        .encode_with_pending(&[&stage], |encoder, _| {
                            let buffer =
                                engine
                                    .device()
                                    .unwrap()
                                    .create_buffer(&wgpu::BufferDescriptor {
                                        label: None,
                                        size: 4,
                                        usage: wgpu::BufferUsages::COPY_DST,
                                        mapped_at_creation: false,
                                    });
                            encoder.clear_buffer(&buffer, 0, None);
                            Err(Error::Gpu("injected failure after commands".into()))
                        })
                        .is_err()
                );
                assert!(batch.submit().is_err());
            }
        }
        assert_eq!(pool.available_slots().unwrap(), 1);
        assert_eq!(output.available_slots().unwrap(), 1);
    }
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let receipt = sensor
        .encode(&mut batch, &engine, &state, &mut pool, Default::default())
        .unwrap()
        .unwrap();
    batch
        .encode_with_pending(&vec![&receipt; 64], |_, inputs| {
            assert_eq!(inputs.len(), 64);
            assert_eq!(inputs.channel::<Vec3>(63, 0)?.len(), 65);
            Ok(0)
        })
        .unwrap();
    engine.destroy_device().unwrap();
    assert!(
        batch
            .encode_with_pending(&[&receipt], |_, _| panic!("lost device"))
            .is_err()
    );
    assert!(
        batch
            .encode_from_pending(
                &[&receipt],
                &mut output,
                &specs,
                std::slice::from_ref(&(0..1)),
                Default::default(),
                |_, _, _| panic!("lost device")
            )
            .is_err()
    );
    assert!(batch.submit().is_err());
}
