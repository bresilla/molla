#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::PreparedTireMpmCompactionReadback as Reader;

#[test]
fn soil_mpm_observation_owner_is_send_sync() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<Reader>();
}

fn source(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MPM async observation fixture"),
        size: u64::from(count.max(1)) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn fill(queue: &wgpu::Queue, buffer: &wgpu::Buffer, count: usize, value: f32) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::cast_slice(&vec![[value, 0., 0., 0.]; count]),
    );
}

fn complete(engine: &Engine) {
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_observation_is_bounded_coalesced_and_immutable() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let data = source(device, 129);
    let mut reader = Reader::new(&engine, device, &data, 129, 1e-5, 2).unwrap();
    assert_eq!(reader.capacity(), 2);
    assert!(reader.poll_latest().unwrap().is_none());
    for (revision, value) in [(1, 0.1), (2, 0.2)] {
        fill(queue, &data, 129, value);
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        assert!(reader.try_request(revision).unwrap());
        assert_eq!(resources.finish().observations, 0);
        let transfers = transfers.finish();
        assert!(!transfers.overflowed);
        assert_eq!(transfers.readback_bytes, 0);
        assert_eq!(transfers.upload_bytes, 0);
        assert_eq!(dispatches.finish().encoded_dispatches, 1);
    }
    let dispatches = DispatchScope::start().unwrap();
    assert!(!reader.try_request(3).unwrap());
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    assert_eq!(reader.pending(), 2);
    assert_eq!(reader.stats().backpressure, 1);
    fill(queue, &data, 129, 0.9);
    queue.submit([]);
    complete(&engine);
    let transfers = TransferScope::start().unwrap();
    let latest = reader.poll_latest().unwrap().unwrap();
    assert_eq!(transfers.finish().readback_bytes, 16);
    assert_eq!(latest.revision(), 2);
    assert_eq!(latest.summary().maximum, 0.2);
    assert_eq!(latest.summary().active_particles, 129);
    assert!(reader.is_current(&latest));
    assert_eq!(reader.pending(), 0);
    assert_eq!(reader.stats().completed, 2);
    assert_eq!(reader.stats().discarded, 1);
    assert!(reader.try_request(2).is_err());
    drop(data);
    assert!(reader.try_request(3).unwrap());
    complete(&engine);
    assert_eq!(
        reader.poll_latest().unwrap().unwrap().summary().maximum,
        0.9
    );
    assert!(reader.poll_latest().unwrap().is_none());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_observation_reset_keeps_inflight_slots_and_rejects_old_owners() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let data = source(device, 1);
    fill(queue, &data, 1, 0.4);
    let mut reader = Reader::new(&engine, device, &data, 1, 1e-5, 2).unwrap();
    let other = Reader::new(&engine, device, &data, 1, 1e-5, 1).unwrap();
    reader.try_request(10).unwrap();
    complete(&engine);
    let previous = reader.poll_latest().unwrap().unwrap();
    assert!(!other.is_current(&previous));
    reader.try_request(11).unwrap();
    reader.try_request(12).unwrap();
    reader.reset().unwrap();
    assert_eq!(reader.generation(), 1);
    assert!(!reader.is_current(&previous));
    assert_eq!(reader.pending(), 2);
    assert!(!reader.try_request(0).unwrap());
    complete(&engine);
    let transfers = TransferScope::start().unwrap();
    assert!(reader.poll_latest().unwrap().is_none());
    assert_eq!(transfers.finish().readback_bytes, 0);
    assert_eq!(reader.pending(), 0);
    fill(queue, &data, 1, 0.05);
    reader.try_request(0).unwrap();
    complete(&engine);
    let next = reader.poll_latest().unwrap().unwrap();
    assert_eq!(next.generation(), 1);
    assert_eq!(next.revision(), 0);
    assert_eq!(next.summary().maximum, 0.05);
    reader.try_request(1).unwrap();
    drop(reader);
    complete(&engine);
    engine.check_device().unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_observation_reuses_resources_and_handles_empty_and_decreasing_values() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let data = source(device, 129);
    let mut reader = Reader::new(&engine, device, &data, 129, 0.05, 3).unwrap();
    for revision in 0..6 {
        let value = if revision == 5 {
            0.
        } else {
            0.2 / (revision + 1) as f32
        };
        fill(queue, &data, 129, value);
        let resources = DeviceScope::start().unwrap();
        reader.try_request(revision).unwrap();
        complete(&engine);
        let observation = reader.poll_latest().unwrap().unwrap();
        let resources = resources.finish();
        assert_eq!(resources.observations, 0);
        assert!(!resources.invalid);
        assert_eq!(observation.summary().maximum, value);
        assert_eq!(
            observation.summary().active_particles,
            if value > 0.05 { 129 } else { 0 }
        );
    }
    let mut empty = Reader::new(&engine, device, &data, 0, 0.05, 1).unwrap();
    let dispatches = DispatchScope::start().unwrap();
    empty.try_request(u64::MAX).unwrap();
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    complete(&engine);
    let observation = empty.poll_latest().unwrap().unwrap();
    assert_eq!(observation.summary().maximum, 0.);
    assert_eq!(observation.summary().active_particles, 0);
    assert!(empty.try_request(u64::MAX).is_err());
    empty.reset().unwrap();
    assert!(empty.try_request(0).unwrap());
    complete(&engine);
    empty.poll_latest().unwrap().unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_observation_rejects_invalid_foreign_and_lost_devices() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let data = source(device, 1);
    for (count, threshold, slots) in [
        (1, 0., 0),
        (1, 0., 65),
        (2, 0., 1),
        (1, f32::NAN, 1),
        (1, -1., 1),
    ] {
        let resources = DeviceScope::start().unwrap();
        assert!(Reader::new(&engine, device, &data, count, threshold, slots).is_err());
        assert_eq!(resources.finish().observations, 0);
    }
    let resources = DeviceScope::start().unwrap();
    assert!(Reader::new(&foreign, device, &data, 1, 0., 1).is_err());
    assert_eq!(resources.finish().observations, 0);
    let mut reader = Reader::new(&engine, device, &data, 1, 0., 1).unwrap();
    reader.try_request(0).unwrap();
    engine.destroy_device().unwrap();
    assert!(reader.poll_latest().is_err());
    assert!(reader.try_request(1).is_err());
    assert!(reader.reset().is_err());
    assert!(Reader::new(&engine, device, &data, 1, 0., 1).is_err());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_observation_latches_invalid_summary_without_publishing_it() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let data = source(device, 1);
    let mut reader = Reader::new(&engine, device, &data, 1, 0., 1).unwrap();
    fill(queue, &data, 1, f32::INFINITY);
    reader.try_request(0).unwrap();
    complete(&engine);
    assert!(reader.poll_latest().is_err());
    assert!(reader.try_request(1).is_err());
    assert!(reader.reset().is_err());
}
