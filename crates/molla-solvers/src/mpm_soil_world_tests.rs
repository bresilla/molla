use super::super::tests::fixture;
use super::*;
use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};
use std::time::{Duration, Instant};

pub(super) fn make_archive(state: &MpmSoilCheckpoint) -> MpmSoilTileArchive {
    state
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 0.125).unwrap(),
            &(0..state.count)
                .map(|i| SoilParticleId((u64::from(i) << 40) + 7))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(u64::MAX),
                SoilMaterialId(19),
                SoilMaterialId(1 << 50),
            ],
        )
        .unwrap()
}

pub(super) fn limits() -> SoilResidencyLimits {
    SoilResidencyLimits {
        regions: 3,
        tiles: 32,
        claims: 96,
    }
}

fn poll(world: &mut MpmSoilWorld, engine: &Engine) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if world.poll_region_update(engine)? {
            return Ok(());
        }
        assert!(Instant::now() < deadline, "world capture deadline");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn request(world: &mut MpmSoilWorld, engine: &Engine, keys: &[SoilTileCoord]) {
    world
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(5),
            tiles: keys,
        }])
        .unwrap();
    poll(world, engine).unwrap();
}

#[test]
fn prepared_archive_update_is_atomic_and_its_selection_uses_the_committed_epoch() {
    let state = fixture(129);
    let mut archive = make_archive(&state);
    let original = archive.to_bytes();
    let keys: Vec<_> = archive.tiles().collect();
    let selected = archive.select(&keys, 129).unwrap();
    let mut moved = state.clone();
    for p in moved.words[..129 * 4].chunks_exact_mut(4) {
        p[0] = (f32::from_bits(p[0]) + 0.5).to_bits();
    }
    let destination = make_archive(&moved);
    let mut new_keys: Vec<_> = destination.tiles().collect();
    new_keys.push(SoilTileCoord([1000; 3]));
    {
        let transaction = archive.prepare_writeback(&selected, &moved).unwrap();
        assert!(transaction.select_existing(&new_keys, 128).is_err());
        assert_eq!(
            transaction
                .select_existing(&keys, 129)
                .unwrap()
                .checkpoint()
                .count,
            0
        );
        let preview = transaction.select_existing(&new_keys, 129).unwrap();
        assert_eq!(preview.checkpoint().words, moved.words);
    }
    assert_eq!(archive.to_bytes(), original);
    let transaction = archive.prepare_writeback(&selected, &moved).unwrap();
    let preview = transaction.select_existing(&new_keys, 129).unwrap();
    transaction.commit();
    assert!(archive.writeback(&selected, &moved).is_err());
    archive.writeback(&preview, &moved).unwrap();
    assert_eq!(archive.to_bytes(), destination.to_bytes());
}

#[test]
fn world_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MpmSoilWorld>();
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn live_overlap_freeze_steps_save_reload_and_retained_capture_preserve_all_history() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let state = fixture(129);
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, archive, limits(), 129 * 160 + 16).unwrap();
    assert_eq!(world.active_particle_count(), 0);
    assert_eq!(world.total_particle_count(), 129);
    assert_eq!(world.committed_revision(), None);
    assert_eq!(world.staging_bytes(), 129 * 160 + 16);
    world
        .begin_region_update(&[
            SoilRegionRequest {
                id: SoilRegionId(5),
                tiles: &keys,
            },
            SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys,
            },
        ])
        .unwrap();
    assert_eq!(world.status(), MpmSoilWorldStatus::Capturing);
    assert!(world.step_n(&engine, 0.0001, 2).is_err());
    assert!(world.begin_region_update(&[]).is_err());
    assert!(world.synchronized_archive_bytes().is_err());
    poll(&mut world, &engine).unwrap();
    assert_eq!(world.active_particle_count(), 129);
    for key in &keys {
        let claims = world.claims(*key).unwrap();
        assert_eq!(claims.regions().count(), 2);
        assert_eq!(claims.coordinator(), Some(SoilRegionId(1)));
    }
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        state.words
    );
    let ids = world.identity().particle_ids().to_vec();
    let buffer = world
        .simulation
        .soil_identity_buffers()
        .unwrap()
        .particles
        .clone();
    for _ in 0..4 {
        world.step_n(&engine, 0.0001, 2).unwrap();
        assert!(world.synchronized_archive_bytes().is_err());
        let evolved = world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap();
        let alloc = DeviceScope::start().unwrap();
        let transfer = TransferScope::start().unwrap();
        world.begin_region_update(&[]).unwrap();
        assert_eq!(alloc.finish().observations, 0);
        assert_eq!(transfer.finish().upload_bytes, 0);
        let alloc = DeviceScope::start().unwrap();
        let transfer = TransferScope::start().unwrap();
        poll(&mut world, &engine).unwrap();
        assert_eq!(alloc.finish().observations, 0);
        let transfer = transfer.finish();
        assert_eq!(transfer.upload_bytes, 0);
        assert_eq!(world.transfer_scratch_bytes(), 0);
        assert_eq!(world.last_transfer_stats().retained_particles, 129);
        assert_eq!(world.last_transfer_stats().imported_particles, 0);
        assert_eq!(world.active_particle_count(), 129);
        assert_eq!(world.identity().particle_ids(), ids);
        assert_eq!(
            world.simulation.soil_identity_buffers().unwrap().particles,
            &buffer
        );
        assert!(
            world
                .resident_tiles()
                .all(|key| world.claims(key).unwrap().pinned())
        );
        let actual = world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap();
        assert_eq!(actual.words, evolved.words);
        assert_eq!(world.staging_bytes(), 20656);
    }
    assert_eq!(world.readback_stats().requests, 5);
    assert_eq!(world.committed_revision(), Some(4));
    let saved = world.synchronized_archive_bytes().unwrap();
    let evolved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    drop(world);
    let archived = MpmSoilTileArchive::from_bytes(&saved, 129).unwrap();
    let keys: Vec<_> = archived.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut restored = MpmSoilWorld::new(&engine, sim, archived, limits(), 20656).unwrap();
    request(&mut restored, &engine, &keys);
    assert_eq!(restored.identity().particle_ids(), ids);
    assert_eq!(
        restored
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        evolved.words
    );
    restored.step_n(&engine, 0.0001, 1).unwrap();
    println!(
        "live world: 129 unique particles, 8 evolved steps, 5 captures; full-history reload exact"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn capacity_rejection_preserves_both_authorities_and_allows_retry() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let state = fixture(129);
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    let first = archive.select(&keys[..1], 129).unwrap();
    let mut cfg = state.cfg;
    cfg.max_particles = first.checkpoint().count;
    assert!(cfg.max_particles > 0 && cfg.max_particles < 129);
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, archive, limits(), 20656).unwrap();
    request(&mut world, &engine, &keys[..1]);
    world.step_n(&engine, 0.0001, 2).unwrap();
    let before = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let archived_before = world.archive.to_bytes();
    let identity = world.identity().clone();
    world
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(2),
            tiles: &keys,
        }])
        .unwrap();
    assert!(world.poll_region_update(&foreign).is_err());
    assert_eq!(world.status(), MpmSoilWorldStatus::Capturing);
    assert!(poll(&mut world, &engine).is_err());
    assert_eq!(world.status(), MpmSoilWorldStatus::Ready);
    assert_eq!(world.archive.to_bytes(), archived_before);
    assert!(world.simulation.is_soil_identity_current(&identity));
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        before.words
    );
    assert!(world.synchronized_archive_bytes().is_err());
    request(&mut world, &engine, &[]);
    assert_eq!(world.active_particle_count(), cfg.max_particles);
    assert_eq!(world.total_particle_count(), 129);
    assert!(world.synchronized_archive_bytes().is_ok());
    assert!(world.step_n(&foreign, 0.0001, 1).is_err());
    assert!(world.step_n(&engine, f32::NAN, 1).is_err());
    world.step_n(&engine, 0.0001, 1).unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn moving_particle_crosses_tile_without_request_or_identity_loss() {
    let engine = Engine::new_gpu().unwrap();
    let mut state = fixture(1);
    state.words[0] = (-0.00001_f32).to_bits();
    state.words[4] = 1_f32.to_bits();
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    assert_eq!(keys[0].0[0], -1);
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut restricted = limits();
    restricted.tiles = 1;
    let mut world = MpmSoilWorld::new(&engine, sim, archive, restricted, 20656).unwrap();
    world.set_activity_monitoring(true).unwrap();
    request(&mut world, &engine, &keys);
    world.step_n(&engine, 0.0001, 2).unwrap();
    let evolved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert!(f32::from_bits(evolved.words[0]) > 0.0);
    let archived = world.archive.to_bytes();
    world
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(5),
            tiles: &keys,
        }])
        .unwrap();
    assert!(poll(&mut world, &engine).is_err());
    assert_eq!(world.status(), MpmSoilWorldStatus::Ready);
    assert_eq!(world.archive.to_bytes(), archived);
    assert!(world.last_activity_report().is_none());
    request(&mut world, &engine, &[]);
    let activity = world.last_activity_report().unwrap();
    assert_eq!(activity.revision, 2);
    assert_eq!(activity.activity.substeps, 2);
    assert_eq!(activity.activity.seconds, 2.0 * f64::from(0.0001_f32));
    assert!(activity.activity.max_displacement_rate > 0.0);
    let tiles: Vec<_> = world.resident_tiles().collect();
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].0[0], 0);
    assert!(world.claims(tiles[0]).unwrap().pinned());
    assert!(world.claims(keys[0]).is_none());
    assert_eq!(world.active_particle_count(), 1);
    assert_eq!(world.identity().particle_ids(), &[SoilParticleId(7)]);
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        evolved.words
    );
    world.step_n(&engine, 0.0001, 1).unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn constructor_empty_space_owner_drop_and_device_loss_are_explicit() {
    let engine = Engine::new_gpu().unwrap();
    let state = fixture(1);
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    assert!(MpmSoilWorld::new(&engine, sim, make_archive(&state), limits(), 20655).is_err());
    let mut sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &state)
        .unwrap();
    assert!(MpmSoilWorld::new(&engine, sim, make_archive(&state), limits(), 20656).is_err());
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, make_archive(&state), limits(), 20656).unwrap();
    let saved = world.synchronized_archive_bytes().unwrap();
    let empty = [SoilTileCoord([999; 3])];
    request(&mut world, &engine, &empty);
    assert_eq!(world.active_particle_count(), 0);
    assert_eq!(world.resident_tiles().collect::<Vec<_>>(), empty);
    world.step_n(&engine, 0.0001, 1).unwrap();
    assert_eq!(world.synchronized_archive_bytes().unwrap(), saved);
    world.begin_region_update(&[]).unwrap();
    drop(world);
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, make_archive(&state), limits(), 20656).unwrap();
    world.begin_region_update(&[]).unwrap();
    engine.destroy_device().unwrap();
    assert_eq!(world.status(), MpmSoilWorldStatus::Faulted);
    assert!(world.poll_region_update(&engine).is_err());
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    assert!(world.begin_region_update(&[]).is_err());
    assert!(world.synchronized_archive_bytes().is_err());
    assert_eq!(world.committed_revision(), None);
    assert_eq!(
        world.last_committed_archive_bytes(),
        make_archive(&state).to_bytes()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn invalid_capture_faults_owner_without_overwriting_archive() {
    let engine = Engine::new_gpu().unwrap();
    let state = fixture(1);
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 1).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, archive, limits(), 20656).unwrap();
    request(&mut world, &engine, &keys);
    let archived = world.archive.to_bytes();
    assert_eq!(world.committed_revision(), Some(0));
    world.step_n(&engine, 0.0001, 1).unwrap();
    world.begin_region_update(&[]).unwrap();
    assert!(poll(&mut world, &engine).is_err());
    assert_eq!(world.status(), MpmSoilWorldStatus::Faulted);
    assert_eq!(world.archive.to_bytes(), archived);
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    assert!(world.begin_region_update(&[]).is_err());
    assert!(world.synchronized_archive_bytes().is_err());
    assert_eq!(world.committed_revision(), Some(0));
    let recovery = world.last_committed_archive_bytes();
    assert_eq!(recovery, archived);
    let archive = MpmSoilTileArchive::from_bytes(&recovery, 129).unwrap();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut recovered = MpmSoilWorld::new(&engine, sim, archive, limits(), 20656).unwrap();
    request(&mut recovered, &engine, &keys);
    assert_eq!(
        recovered
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        state.words
    );
    recovered.step_n(&engine, 0.0001, 1).unwrap();
}
