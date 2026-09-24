#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, Error};
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{
    DeviceCommandBatch, DeviceSnapshotConfig, DeviceSnapshotPool, Raycaster,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::ModelBuilder;
use molla_storage::BufferUsage;
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

static GPU: Mutex<()> = Mutex::new(());

fn pool(engine: &Engine, slots: usize, bytes: u64) -> DeviceSnapshotPool {
    DeviceSnapshotPool::new(
        engine,
        DeviceSnapshotConfig {
            slots,
            bytes_per_slot: bytes,
            max_total_bytes: slots as u64 * bytes,
        },
    )
    .unwrap()
}

fn ring(engine: &Engine) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 2,
            bytes_per_slot: 128,
            max_total_bytes: 256,
        },
    )
    .unwrap()
}

fn reclaim(pool: &mut DeviceSnapshotPool, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.available_slots().unwrap() != expected {
        assert!(Instant::now() < deadline, "snapshot slots did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn mixed_snapshot_channels_clones_reset_and_pool_reuse() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut positions = DeviceBuffer::from_host(
        &engine,
        vec![Vec3::X, Vec3::Y, Vec3::Z],
        BufferUsage::StorageReadWrite,
    )
    .unwrap();
    let counts =
        DeviceBuffer::from_host(&engine, vec![1u32, 2, 3], BufferUsage::StorageReadWrite).unwrap();
    let mut pool = pool(&engine, 2, 1024);
    let mut ring = ring(&engine);
    let stamp = ReadbackStamp {
        sensor_id: 5,
        sample_id: 7,
        scene_revision: 1,
        simulation_time: Some(0.5),
    };
    let first = pool
        .try_capture(
            &[
                positions.readback_source(1..3).unwrap(),
                counts.readback_source(0..3).unwrap(),
            ],
            stamp,
        )
        .unwrap()
        .unwrap();
    positions.host_mut().unwrap().fill(Vec3::ONE);
    positions.upload().unwrap();
    let second = pool
        .try_capture(
            &[positions.readback_source(1..3).unwrap()],
            ReadbackStamp {
                sample_id: 8,
                ..stamp
            },
        )
        .unwrap()
        .unwrap();
    assert!(pool.try_capture(&[], stamp).unwrap().is_none());
    assert_eq!(first.channel_count(), 2);
    assert_eq!(first.channel::<u32>(1).unwrap().len(), 3);
    assert!(first.channel::<Real>(1).is_err());
    assert!(first.channel::<u32>(2).is_err());
    assert!(
        first
            .channel::<Vec3>(0)
            .unwrap()
            .readback_source(0..3)
            .is_err()
    );
    let alignment = engine
        .device()
        .unwrap()
        .limits()
        .min_storage_buffer_offset_alignment as u64;
    assert_eq!(
        first.channel::<u32>(1).unwrap().binding().unwrap().offset % alignment,
        0
    );
    let first_ticket = ring
        .try_enqueue(
            &[
                first
                    .channel::<Vec3>(0)
                    .unwrap()
                    .readback_source(0..2)
                    .unwrap(),
                first
                    .channel::<u32>(1)
                    .unwrap()
                    .readback_source(1..3)
                    .unwrap(),
            ],
            first.stamp(),
        )
        .unwrap()
        .unwrap();
    let second_ticket = ring
        .try_enqueue(
            &[second
                .channel::<Vec3>(0)
                .unwrap()
                .readback_source(0..2)
                .unwrap()],
            second.stamp(),
        )
        .unwrap()
        .unwrap();
    let second_frame = ring.wait(&second_ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(second_frame.channel::<Vec3>(0).unwrap(), [Vec3::ONE; 2]);
    let first_frame = ring.wait(&first_ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(first_frame.stamp, stamp);
    assert_eq!(first_frame.channel::<Vec3>(0).unwrap(), [Vec3::Y, Vec3::Z]);
    assert_eq!(first_frame.channel::<u32>(1).unwrap(), [2, 3]);
    assert_eq!(first_frame.byte_len(), 40);
    let clone = first.clone();
    pool.reset().unwrap();
    assert!(!pool.is_current(&first));
    assert_eq!(pool.available_slots().unwrap(), 0);
    drop(first);
    drop(second);
    reclaim(&mut pool, 1);
    drop(clone);
    reclaim(&mut pool, 2);
    for sample in 0..20 {
        let snapshot = pool
            .try_capture(
                &[counts.readback_source(0..3).unwrap()],
                ReadbackStamp {
                    sample_id: sample,
                    ..stamp
                },
            )
            .unwrap()
            .unwrap();
        assert!(pool.is_current(&snapshot));
        assert!(snapshot.capture_id() > 2);
        drop(snapshot);
        reclaim(&mut pool, 2);
    }
    assert_eq!(pool.stats().buffer_allocations, 2);
    assert_eq!(pool.stats().captures, 22);
    assert_eq!(pool.stats().copied_bytes, 44 + 32 + 20 * 12);
    assert_eq!(pool.stats().submissions, 22);
    assert_eq!(pool.stats().backpressure, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn unsubmitted_consumers_pin_sensor_versions_until_gpu_completion() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mut sensor = Raycaster::new(&engine, 1).unwrap();
    let mut pool = pool(&engine, 1, 1024);
    let output = sensor
        .dispatch(
            &engine,
            &model,
            &state,
            &[Vec3::Y * 10.0],
            &[Vec3::NEG_Y],
            20.0,
        )
        .unwrap();
    let snapshot = pool
        .try_capture(
            &[
                output.distance.readback_source(0..1).unwrap(),
                output.shape.readback_source(0..1).unwrap(),
            ],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("leased sensor consumer"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
        @group(0) @binding(0) var<storage, read> distances: array<f32>;
        @group(0) @binding(1) var<storage, read> shapes: array<i32>;
        @group(0) @binding(2) var<storage, read_write> result: array<f32>;
        @compute @workgroup_size(1) fn main() {
            result[0] = 2.0 * distances[0] + f32(shapes[0]);
        }
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
    let mut result =
        DeviceBuffer::from_host(&engine, vec![0.0 as Real], BufferUsage::StorageReadWrite).unwrap();
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    snapshot.channel::<Real>(0).unwrap().binding().unwrap(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(
                    snapshot.channel::<i32>(1).unwrap().binding().unwrap(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result.device_buffer().unwrap().as_entire_binding(),
            },
        ],
    });
    let mut batch = DeviceCommandBatch::new(&engine).unwrap();
    batch
        .encode_with_snapshots(&[&snapshot], |encoder| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            Ok(())
        })
        .unwrap();
    drop(snapshot);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    assert_eq!(pool.available_slots().unwrap(), 0);
    let newer = sensor
        .dispatch(
            &engine,
            &model,
            &state,
            &[Vec3::Y * 3.0],
            &[Vec3::NEG_Y],
            20.0,
        )
        .unwrap();
    assert!(
        pool.try_capture(
            &[newer.distance.readback_source(0..1).unwrap()],
            ReadbackStamp::default()
        )
        .unwrap()
        .is_none()
    );
    batch.submit().unwrap();
    result.mark_device_dirty();
    let mut ring = ring(&engine);
    let ticket = ring
        .try_enqueue(
            &[result.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<Real>(0)
            .unwrap(),
        [20.0]
    );
    reclaim(&mut pool, 1);
    assert_eq!(sensor.readback_stats().submissions, 0);
    assert_eq!(pool.stats().buffer_allocations, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn dropped_failed_batches_and_cancelled_readback_release_leases() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let source =
        DeviceBuffer::from_host(&engine, vec![7u32], BufferUsage::StorageReadWrite).unwrap();
    let mut pool = pool(&engine, 1, 256);
    let snapshot = pool
        .try_capture(
            &[source.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let mut abandoned = DeviceCommandBatch::new(&engine).unwrap();
    abandoned
        .encode_with_snapshots(&[&snapshot], |_| Ok(()))
        .unwrap();
    let mut failed = DeviceCommandBatch::new(&engine).unwrap();
    assert!(
        failed
            .encode_with_snapshots(&[&snapshot], |encoder| {
                encoder.clear_buffer(source.device_buffer().unwrap(), 0, None);
                Err(Error::Solver("injected consumer failure".into()))
            })
            .is_err()
    );
    assert!(failed.submit().is_err());
    let mut panicked = DeviceCommandBatch::new(&engine).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = panicked.encode_with_snapshots(&[&snapshot], |encoder| {
            encoder.clear_buffer(source.device_buffer().unwrap(), 0, None);
            panic!("injected consumer panic");
        });
    }));
    assert!(panic.is_err());
    assert!(panicked.submit().is_err());
    let mut ring = ring(&engine);
    let ticket = ring
        .try_enqueue(
            &[snapshot
                .channel::<u32>(0)
                .unwrap()
                .readback_source(0..1)
                .unwrap()],
            snapshot.stamp(),
        )
        .unwrap()
        .unwrap();
    ring.cancel(&ticket).unwrap();
    drop(snapshot);
    assert_eq!(pool.available_slots().unwrap(), 0);
    drop(abandoned);
    reclaim(&mut pool, 1);
    ring.poll_releases().unwrap();
    assert!(ring.poll(&ticket).is_err());
    let ticket = ring
        .try_enqueue(
            &[source.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<u32>(0)
            .unwrap(),
        [7]
    );
    let snapshot = pool
        .try_capture(
            &[source.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    drop(pool);
    let ticket = ring
        .try_enqueue(
            &[snapshot
                .channel::<u32>(0)
                .unwrap()
                .readback_source(0..1)
                .unwrap()],
            snapshot.stamp(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<u32>(0)
            .unwrap(),
        [7]
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn empty_channels_invalid_budgets_and_foreign_devices_are_explicit() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let source =
        DeviceBuffer::from_host(&engine, vec![1u32], BufferUsage::StorageReadWrite).unwrap();
    let empty = DeviceBuffer::from_host(&engine, Vec::<Vec3>::new(), BufferUsage::StorageReadWrite)
        .unwrap();
    let mut pool = pool(&engine, 1, 4);
    let excess: Vec<_> = (0..65)
        .map(|_| source.readback_source(0..1).unwrap())
        .collect();
    assert!(pool.try_capture(&excess, ReadbackStamp::default()).is_err());
    assert!(
        pool.try_capture(
            &[
                source.readback_source(0..1).unwrap(),
                source.readback_source(0..1).unwrap()
            ],
            ReadbackStamp::default()
        )
        .is_err()
    );
    assert!(
        pool.try_capture(
            &[],
            ReadbackStamp {
                simulation_time: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err()
    );
    for config in [
        DeviceSnapshotConfig {
            slots: 65,
            bytes_per_slot: 4,
            max_total_bytes: 260,
        },
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: u64::MAX,
            max_total_bytes: u64::MAX,
        },
        DeviceSnapshotConfig {
            slots: 2,
            bytes_per_slot: 4,
            max_total_bytes: 7,
        },
    ] {
        assert!(DeviceSnapshotPool::new(&engine, config).is_err());
    }
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    let mut other_pool = DeviceSnapshotPool::new(
        &foreign,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 4,
            max_total_bytes: 4,
        },
    )
    .unwrap();
    assert!(
        other_pool
            .try_capture(
                &[source.readback_source(0..1).unwrap()],
                ReadbackStamp::default()
            )
            .is_err()
    );
    let snapshot = pool
        .try_capture(
            &[empty.readback_source(0..0).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(pool.stats().submissions, 0);
    assert!(snapshot.channel::<Vec3>(0).unwrap().is_empty());
    assert!(snapshot.channel::<Vec3>(0).unwrap().binding().is_err());
    assert!(!other_pool.is_current(&snapshot));
    let mut foreign_batch = DeviceCommandBatch::new(&foreign).unwrap();
    assert!(
        foreign_batch
            .encode_with_snapshots(&[&snapshot], |_| panic!("foreign consumer must not encode"))
            .is_err()
    );
    let mut ring = ring(&engine);
    let ticket = ring
        .try_enqueue(
            &[snapshot
                .channel::<Vec3>(0)
                .unwrap()
                .readback_source(0..0)
                .unwrap()],
            snapshot.stamp(),
        )
        .unwrap()
        .unwrap();
    let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert!(frame.channel::<Vec3>(0).unwrap().is_empty());
    assert_eq!(ring.stats().copied_bytes, 0);
    assert_eq!(pool.stats().captures, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn forwarded_snapshots_and_multi_pool_consumers_survive_owner_drop() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let mut source =
        DeviceBuffer::from_host(&engine, vec![7u32], BufferUsage::StorageReadWrite).unwrap();
    let mut first_pool = pool(&engine, 1, 4);
    let mut second_pool = pool(&engine, 1, 4);
    let old = first_pool
        .try_capture(
            &[source.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let forwarded = second_pool
        .try_capture(
            &[old
                .channel::<u32>(0)
                .unwrap()
                .readback_source(0..1)
                .unwrap()],
            old.stamp(),
        )
        .unwrap()
        .unwrap();
    assert!(!first_pool.is_current(&forwarded));
    assert!(second_pool.is_current(&forwarded));
    drop(old);
    reclaim(&mut first_pool, 1);
    source.host_mut().unwrap()[0] = 77;
    source.upload().unwrap();
    let newer = first_pool
        .try_capture(
            &[source.readback_source(0..1).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let mut output =
        DeviceBuffer::from_host(&engine, vec![0u32; 2], BufferUsage::StorageReadWrite).unwrap();
    let mut batch = DeviceCommandBatch::new(&engine).unwrap();
    batch
        .encode_with_snapshots(&[&newer, &forwarded], |encoder| {
            for (i, snapshot) in [&newer, &forwarded].into_iter().enumerate() {
                let channel = snapshot.channel::<u32>(0)?.binding()?;
                encoder.copy_buffer_to_buffer(
                    channel.buffer,
                    channel.offset,
                    output.device_buffer()?,
                    i as u64 * 4,
                    4,
                );
            }
            Ok(())
        })
        .unwrap();
    drop(newer);
    drop(forwarded);
    drop(first_pool);
    drop(second_pool);
    batch.submit().unwrap();
    output.mark_device_dirty();
    let mut ring = ring(&engine);
    let ticket = ring
        .try_enqueue(
            &[output.readback_source(0..2).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<u32>(0)
            .unwrap(),
        [77, 7]
    );
}
