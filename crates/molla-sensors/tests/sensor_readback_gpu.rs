#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Real, Vec3};
use molla_sensors::readback::{
    DeviceBuffer, ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp, ReadbackTicket,
};
use molla_storage::{BufferElement, BufferUsage};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

static GPU: Mutex<()> = Mutex::new(());

fn upload<T: BufferElement>(engine: &Engine, values: Vec<T>) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(engine, values, BufferUsage::StorageReadWrite).unwrap()
}

fn ring(engine: &Engine, slots: usize, bytes: u64) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots,
            bytes_per_slot: bytes,
            max_total_bytes: slots as u64 * bytes,
        },
    )
    .unwrap()
}

fn collect(ring: &mut ReadbackRing, ticket: &ReadbackTicket) -> ReadbackFrame {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(frame) = ring.poll(ticket).unwrap() {
            return frame;
        }
        assert!(Instant::now() < deadline, "readback did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn reclaim(ring: &mut ReadbackRing) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while ring.available_slots() == 0 {
        ring.poll_releases().unwrap();
        assert!(Instant::now() < deadline, "cancelled slot did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "needs a real GPU; make test-gpu P=molla-sensors T=sensor_readback_gpu"]
fn mixed_active_ranges_are_grouped_and_staging_is_reused() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let positions = upload(&engine, vec![Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE]);
    let indices = upload(&engine, vec![10u32, 20, 30, 40, 50]);
    let distances = upload(&engine, vec![1.25 as Real, 2.5, 3.75]);
    let sources = [
        positions.readback_source(1..3).unwrap(),
        indices.readback_source(1..4).unwrap(),
        distances.readback_source(0..2).unwrap(),
    ];
    let mut ring = ring(&engine, 2, 128);
    let stamp = ReadbackStamp {
        sensor_id: 7,
        sample_id: 41,
        scene_revision: 9,
        simulation_time: Some(1.5),
    };
    let a = ring.try_enqueue(&sources, stamp).unwrap().unwrap();
    let b = ring.try_enqueue(&sources, stamp).unwrap().unwrap();
    assert!(ring.try_enqueue(&sources, stamp).unwrap().is_none());
    let frame = collect(&mut ring, &b);
    assert_eq!(frame.stamp, stamp);
    assert_eq!(frame.byte_len(), 52);
    assert_eq!(frame.channel::<Vec3>(0).unwrap(), [Vec3::Y, Vec3::Z]);
    assert_eq!(frame.channel::<u32>(1).unwrap(), [20, 30, 40]);
    assert_eq!(frame.channel::<Real>(2).unwrap(), [1.25, 2.5]);
    assert!(frame.channel::<u32>(2).is_err());
    assert!(frame.channel::<Real>(3).is_err());
    assert!(ring.poll(&b).is_err());
    collect(&mut ring, &a);
    for sample_id in 0..20 {
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp { sample_id, ..stamp })
            .unwrap()
            .unwrap();
        assert_eq!(collect(&mut ring, &ticket).stamp.sample_id, sample_id);
    }
    assert_eq!(frame.channel::<u32>(1).unwrap(), [20, 30, 40]);
    let stats = ring.stats();
    assert_eq!(stats.staging_allocations, 2);
    assert_eq!(stats.submissions, 22);
    assert_eq!(stats.copied_bytes, 22 * 52);
    assert_eq!(stats.blocking_waits, 0);
}

#[test]
#[ignore = "needs a real GPU"]
fn reset_cancel_and_stale_tickets_cannot_alias_reused_slots() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut source = upload(&engine, vec![1u32; 16]);
    let mut ring = ring(&engine, 1, 64);
    let stamp = ReadbackStamp::default();
    let old = ring
        .try_enqueue(&[source.readback_source(0..16).unwrap()], stamp)
        .unwrap()
        .unwrap();
    ring.cancel(&old).unwrap();
    assert!(ring.poll(&old).is_err());
    assert!(ring.cancel(&old).is_err());
    source.host_mut().unwrap().fill(42);
    source.upload().unwrap();
    reclaim(&mut ring);
    let new = ring
        .try_enqueue(&[source.readback_source(0..16).unwrap()], stamp)
        .unwrap()
        .unwrap();
    assert_eq!(
        collect(&mut ring, &new).channel::<u32>(0).unwrap(),
        [42; 16]
    );
    let old = ring
        .try_enqueue(&[source.readback_source(0..16).unwrap()], stamp)
        .unwrap()
        .unwrap();
    ring.reset();
    assert!(ring.poll(&old).is_err());
    reclaim(&mut ring);
    assert_eq!(ring.available_slots(), 1);
    let new = ring
        .try_enqueue(&[source.readback_source(0..16).unwrap()], stamp)
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&new, Duration::from_secs(10))
            .unwrap()
            .channel::<u32>(0)
            .unwrap(),
        [42; 16]
    );
    assert_eq!(ring.stats().cancellations, 2);
    assert_eq!(ring.stats().staging_allocations, 1);
}

#[test]
#[ignore = "needs a real GPU"]
fn empty_batches_and_invalid_ranges_do_not_submit_or_allocate() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut source = upload(&engine, vec![9u32; 4]);
    let empty = upload::<u32>(&engine, vec![]);
    let mut ring = ring(&engine, 1, 8);
    assert!(source.readback_source(3..5).is_err());
    assert!(
        source
            .readback_source(std::ops::Range { start: 3, end: 2 })
            .is_err()
    );
    assert!(
        ring.try_enqueue(
            &[source.readback_source(0..4).unwrap()],
            ReadbackStamp::default()
        )
        .is_err()
    );
    let invalid_time = ReadbackStamp {
        simulation_time: Some(f64::NAN),
        ..Default::default()
    };
    assert!(ring.try_enqueue(&[], invalid_time).is_err());
    source.host_mut().unwrap()[0] = 17;
    assert!(source.readback_source(0..1).is_err());
    let ticket = ring
        .try_enqueue(
            &[empty.readback_source(0..0).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    let frame = ring.poll(&ticket).unwrap().unwrap();
    assert_eq!(frame.channel_count(), 1);
    assert!(frame.channel::<u32>(0).unwrap().is_empty());
    assert_eq!(ring.stats().submissions, 0);
    assert_eq!(ring.stats().staging_allocations, 1);
    let empty_ticket = ring
        .try_enqueue(&[], ReadbackStamp::default())
        .unwrap()
        .unwrap();
    ring.cancel(&empty_ticket).unwrap();
    ring.poll_releases().unwrap();
    assert_eq!(ring.available_slots(), 1);
    assert!(
        ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 2,
                bytes_per_slot: 9,
                max_total_bytes: 18
            }
        )
        .is_err()
    );
    assert!(
        ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: u64::MAX,
                max_total_bytes: u64::MAX
            }
        )
        .is_err()
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn cross_device_sources_and_foreign_tickets_are_rejected() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let first = Engine::new_gpu().unwrap();
    let second = Engine::new_gpu().unwrap();
    let values = upload(&first, vec![5u32]);
    let source = values.readback_source(0..1).unwrap();
    let mut a = ring(&first, 1, 8);
    let mut b = ring(&second, 1, 8);
    assert!(b.try_enqueue(&[source], ReadbackStamp::default()).is_err());
    let ticket = a
        .try_enqueue(&[], ReadbackStamp::default())
        .unwrap()
        .unwrap();
    assert!(b.poll(&ticket).is_err());
    assert!(b.cancel(&ticket).is_err());
    assert_eq!(a.poll(&ticket).unwrap().unwrap().byte_len(), 0);
}

#[test]
#[ignore = "needs a real GPU"]
fn dropping_pending_collection_does_not_invalidate_the_source_or_device() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let source = upload(&engine, vec![23u32; 16]);
    {
        let mut discarded = ring(&engine, 1, 64);
        discarded
            .try_enqueue(
                &[source.readback_source(0..16).unwrap()],
                ReadbackStamp::default(),
            )
            .unwrap()
            .unwrap();
    }
    let mut next = ring(&engine, 1, 64);
    let ticket = next
        .try_enqueue(
            &[source.readback_source(0..16).unwrap()],
            ReadbackStamp::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        collect(&mut next, &ticket).channel::<u32>(0).unwrap(),
        [23; 16]
    );
}
