use super::super::tests::fixture;
use super::*;
use std::time::{Duration, Instant};

fn budget(count: u32, slots: usize) -> u64 {
    (u64::from(count) * 160 + 16) * slots as u64
}

fn next(ring: &mut MpmSoilCheckpointReadback) -> MpmSoilCheckpointCompletion {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = ring.poll_next().unwrap() {
            return result;
        }
        assert!(Instant::now() < deadline, "checkpoint did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn asynchronous_checkpoint_types_are_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MpmSoilCheckpointReadback>();
    check::<MpmSoilCheckpointTicket>();
    check::<MpmSoilCheckpointCompletion>();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn solver_frames_archive_writeback_and_reload_keep_last_capture_exact() {
    use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let state = fixture(129);
    let ids: Vec<_> = (0..129).map(SoilParticleId).collect();
    let mut archive = state
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 0.125).unwrap(),
            &ids,
            &[SoilMaterialId(10), SoilMaterialId(20), SoilMaterialId(30)],
        )
        .unwrap();
    let mut selection = archive
        .select(&archive.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    sim.restore_soil_with(queue, selection.checkpoint())
        .unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 129, 3, budget(129, 3))
        .unwrap();
    for revision in 0..3 {
        sim.step_n_with(device, queue, 0.0001, 2).unwrap();
        ring.try_capture(&sim, revision).unwrap().unwrap();
    }
    let mut previous = state.to_bytes();
    for revision in 0..3 {
        let completion = next(&mut ring);
        assert_eq!(completion.ticket().revision(), revision);
        let snapshot = completion.into_result().unwrap();
        assert_ne!(snapshot.to_bytes(), previous);
        previous = snapshot.to_bytes();
        archive.writeback(&selection, &snapshot).unwrap();
        selection = archive
            .select(&archive.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        assert_eq!(selection.particle_ids(), ids);
        assert_eq!(selection.checkpoint().to_bytes(), previous);
    }
    assert_eq!(
        sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
        previous
    );
    drop(sim);
    drop(ring);
    let decoded = MpmSoilTileArchive::from_bytes(&archive.to_bytes(), 129).unwrap();
    let final_selection = decoded
        .select(&decoded.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    let mut restored = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    restored
        .restore_soil_with(queue, final_selection.checkpoint())
        .unwrap();
    assert_eq!(
        restored.checkpoint_soil_with(queue).unwrap().to_bytes(),
        previous
    );
    println!(
        "three FIFO archive captures across six actual GPU substeps, no waiting inside capture loop; global-ID writeback and fresh-solver reload exact"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn fifo_capture_retains_request_state_and_reuses_bounded_slots_without_hot_gpu_allocations() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let initial = fixture(129);
    let mut sim = MpmSimulation::new_sparse(&engine, initial.cfg, 128).unwrap();
    sim.restore_soil_with(queue, &initial).unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 129, 2, budget(129, 2))
        .unwrap();
    assert_eq!(
        (ring.particle_capacity(), ring.slots(), ring.staging_bytes()),
        (129, 2, budget(129, 2))
    );
    let buffers: Vec<_> = ring.slots.iter().map(|s| s.buffer.clone()).collect();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let first = ring.try_capture(&sim, 10).unwrap().unwrap();
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish().upload_bytes, 0);
    assert!(ring.is_compatible(&sim, &first));
    let mut edited = fixture(1);
    sim.restore_soil_with(queue, &edited).unwrap();
    edited.materials.reverse();
    edited.words[35] = 2;
    sim.set_materials(&edited.materials, vec![2]).unwrap();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let second = ring.try_capture(&sim, 20).unwrap().unwrap();
    assert!(!ring.is_compatible(&sim, &first));
    assert!(ring.is_compatible(&sim, &second));
    assert!(ring.try_capture(&sim, 30).unwrap().is_none());
    assert!(ring.try_capture(&sim, 20).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let transfer = transfers.finish();
    assert_eq!((transfer.upload_bytes, transfer.readback_bytes), (0, 0));
    assert_eq!(sim.particle_material_dirty, Some(vec![2]));
    let empty = fixture(0);
    sim.restore_soil_with(queue, &empty).unwrap();
    assert!(ring.is_current(&first) && !ring.is_compatible(&sim, &second));
    for (revision, expected) in [(10, &initial), (20, &edited)] {
        let result = next(&mut ring);
        assert_eq!(result.ticket().revision(), revision);
        assert_eq!(
            result.into_result().unwrap().to_bytes(),
            expected.to_bytes()
        );
    }
    assert_eq!(ring.stats().readback_bytes, budget(129, 1) + budget(1, 1));
    assert_eq!(ring.stats().backpressure, 1);
    let allocations = DeviceScope::start().unwrap();
    assert!(ring.try_capture(&sim, 30).unwrap().is_some());
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(
        next(&mut ring).into_result().unwrap().to_bytes(),
        empty.to_bytes()
    );
    for count in [65, 1, 129, 0] {
        let state = fixture(count);
        sim.restore_soil_with(queue, &state).unwrap();
        let revision = ring.last_revision.unwrap() + 1;
        ring.try_capture(&sim, revision).unwrap().unwrap();
        assert_eq!(
            next(&mut ring).into_result().unwrap().to_bytes(),
            state.to_bytes()
        );
    }
    for (a, b) in buffers.iter().zip(&ring.slots) {
        assert_eq!(a, &b.buffer);
    }
    assert_eq!(ring.pending(), 0);
    assert_eq!(ring.stats().requests, 7);
    assert_eq!(ring.stats().completed, 7);
    assert_eq!(ring.stats().rejected, 0);
    println!(
        "7 FIFO captures with counts 129/1/0/65/1/129/0; immutable request metadata, fixed two-slot staging {}, zero capture GPU allocations/uploads",
        ring.staging_bytes()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn cancellation_owner_drop_and_new_generation_do_not_reuse_inflight_buffers() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let state = fixture(1);
    let mut sim = MpmSimulation::new(&engine, state.cfg).unwrap();
    sim.restore_soil_with(queue, &state).unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 1, 2, budget(1, 2))
        .unwrap();
    let old = ring.try_capture(&sim, 0).unwrap().unwrap();
    ring.cancel_all().unwrap();
    assert!(!ring.is_current(&old));
    assert_eq!(ring.pending(), 1);
    let current = ring.try_capture(&sim, 0).unwrap().unwrap();
    assert_eq!(ring.pending(), 2);
    assert!(ring.try_capture(&sim, 1).unwrap().is_none());
    drop(sim);
    let result = next(&mut ring);
    assert!(ring.is_current(result.ticket()));
    assert_eq!(result.into_result().unwrap().to_bytes(), state.to_bytes());
    let deadline = Instant::now() + Duration::from_secs(10);
    while ring.pending() != 0 {
        assert!(ring.poll_next().unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        (
            ring.stats().cancelled,
            ring.stats().completed,
            ring.stats().readback_bytes
        ),
        (1, 1, budget(1, 1))
    );
    let mut another = MpmSimulation::new(&engine, state.cfg).unwrap();
    another.restore_soil_with(queue, &state).unwrap();
    let mut dropping = another
        .prepare_soil_checkpoint_readback(&engine, 1, 1, budget(1, 1))
        .unwrap();
    assert!(!dropping.is_current(&current));
    dropping.try_capture(&another, 0).unwrap().unwrap();
    drop(dropping);
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    engine.check_device().unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn validation_rejects_budget_capacity_foreign_device_and_foreign_source_without_capture() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let other = Engine::new_gpu().unwrap();
    let state = fixture(129);
    let mut sim = MpmSimulation::new(&engine, state.cfg).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &state)
        .unwrap();
    let second = MpmSimulation::new(&engine, state.cfg).unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(
        sim.prepare_soil_checkpoint_readback(&engine, 129, 0, u64::MAX)
            .is_err()
    );
    assert!(
        sim.prepare_soil_checkpoint_readback(&engine, 129, 65, u64::MAX)
            .is_err()
    );
    assert!(
        sim.prepare_soil_checkpoint_readback(&engine, 130, 1, u64::MAX)
            .is_err()
    );
    assert!(
        sim.prepare_soil_checkpoint_readback(&engine, 129, 2, budget(129, 2) - 1)
            .is_err()
    );
    assert!(
        sim.prepare_soil_checkpoint_readback(&other, 129, 1, u64::MAX)
            .is_err()
    );
    assert_eq!(resources.finish().observations, 0);
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 1, 1, budget(1, 1))
        .unwrap();
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(ring.try_capture(&sim, 0).is_err());
    assert!(ring.try_capture(&second, 0).is_err());
    assert_eq!(resources.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert_eq!(ring.stats(), MpmSoilReadbackStats::default());
    sim.restore_soil_with(engine.queue().unwrap(), &fixture(1))
        .unwrap();
    ring.try_capture(&sim, 0).unwrap().unwrap();
    next(&mut ring).into_result().unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn faults_are_reported_per_capture_and_grid_growth_uses_current_buffers() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let state = fixture(1);
    let mut sim = MpmSimulation::new_sparse(&engine, state.cfg, 1).unwrap();
    sim.restore_soil_with(queue, &state).unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 1, 3, budget(1, 3))
        .unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    ring.try_capture(&sim, 0).unwrap().unwrap();
    sim.reset_active_grid();
    ring.try_capture(&sim, 1).unwrap().unwrap();
    sim.restore_soil_with(queue, &state).unwrap();
    ring.try_capture(&sim, 2).unwrap().unwrap();
    for revision in [0, 1] {
        let result = next(&mut ring);
        assert_eq!(result.ticket().revision(), revision);
        assert!(result.into_result().is_err());
    }
    assert_eq!(
        next(&mut ring).into_result().unwrap().to_bytes(),
        state.to_bytes()
    );
    assert_eq!(ring.stats().rejected, 2);
    sim.reserve_sparse_tiles(64).unwrap();
    sim.step_n_with(device, queue, 0.0001, 2).unwrap();
    ring.try_capture(&sim, 3).unwrap().unwrap();
    let expected = sim.checkpoint_soil_with(queue).unwrap();
    assert_eq!(
        next(&mut ring).into_result().unwrap().to_bytes(),
        expected.to_bytes()
    );
    queue.write_buffer(&sim.p_pos, 0, bytemuck::bytes_of(&f32::NAN));
    ring.try_capture(&sim, 4).unwrap().unwrap();
    sim.restore_soil_with(queue, &state).unwrap();
    ring.try_capture(&sim, 5).unwrap().unwrap();
    assert!(next(&mut ring).into_result().is_err());
    assert_eq!(
        next(&mut ring).into_result().unwrap().to_bytes(),
        state.to_bytes()
    );
    assert_eq!(
        (
            ring.stats().completed,
            ring.stats().rejected,
            ring.stats().aborted
        ),
        (6, 3, 0)
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn map_failure_is_sticky_and_owned_device_destruction_aborts_pending_captures() {
    let engine = Engine::new_gpu().unwrap();
    let state = fixture(1);
    let mut sim = MpmSimulation::new(&engine, state.cfg).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &state)
        .unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 1, 1, budget(1, 1))
        .unwrap();
    let ticket = ring.try_capture(&sim, 0).unwrap().unwrap();
    let failure = Arc::new(OnceLock::new());
    failure.set(Err(wgpu::BufferAsyncError)).unwrap();
    ring.slots[0].request.as_mut().unwrap().completion = failure;
    assert!(ring.poll_next().is_err());
    assert_eq!((ring.pending(), ring.stats().aborted), (0, 1));
    assert!(!ring.is_current(&ticket));
    assert!(ring.try_capture(&sim, 1).is_err());
    assert!(ring.cancel_all().is_err());
    assert!(ring.poll_next().is_err());
    assert_eq!(ring.stats().aborted, 1);
    engine.check_device().unwrap();
    let mut lost = sim
        .prepare_soil_checkpoint_readback(&engine, 1, 1, budget(1, 1))
        .unwrap();
    let ticket = lost.try_capture(&sim, 0).unwrap().unwrap();
    engine.destroy_device().unwrap();
    assert!(lost.poll_next().is_err());
    assert_eq!((lost.pending(), lost.stats().aborted), (0, 1));
    assert!(!lost.is_current(&ticket));
    assert!(lost.try_capture(&sim, 1).is_err());
    assert!(lost.cancel_all().is_err());
}
