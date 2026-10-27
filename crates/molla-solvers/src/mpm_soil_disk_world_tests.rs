use super::super::tests::{limits, make_archive};
use super::*;
use crate::mpm::checkpoint::tests::fixture;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("MOLLA_STORE_TEST_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = root.join(format!(
            "molla-disk-world-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[path = "mpm_soil_framed_disk_world_tests.rs"]
mod framed;

#[path = "mpm_soil_canonical_disk_world_tests.rs"]
mod canonical;

#[path = "mpm_soil_quiescence_world_tests.rs"]
mod quiescence;

fn io_limits() -> MpmSoilStoreReadLimits {
    MpmSoilStoreReadLimits {
        particles: 129,
        tiles: 32,
        encoded_bytes: 129 * 168 + 32 * 1024,
    }
}
fn store_limits() -> MpmSoilStoreLimits {
    MpmSoilStoreLimits {
        particles: 129,
        tiles: 32,
    }
}
fn seed(directory: &Directory) -> (MpmSoilStore, MpmSoilTileArchive, MpmConfig) {
    let mut state = fixture(129);
    for (index, position) in state.words[..129 * 4].chunks_exact_mut(4).enumerate() {
        position[0] = ((index % 4) as f32 * 0.125 + 0.02).to_bits();
    }
    let archive = make_archive(&state);
    let mut store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    (store, archive, state.cfg)
}
fn poll(world: &mut MpmSoilDiskWorld, engine: &Engine) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if world.poll_region_update(engine)? {
            return Ok(());
        }
        assert!(Instant::now() < deadline, "disk world transition deadline");
        assert!(world.step_n(engine, 0.0001, 1).is_err());
        assert!(world.store_storage_stats().is_err());
        assert!(world.begin_region_update(&[]).is_err());
        assert!(world.prepare_device_transfers(engine, u64::MAX).is_err());
        assert!(world.disable_device_transfers().is_err());
        assert!(world.disable_gpu_pins().is_err());
        assert!(
            world
                .prepare_gpu_pins(engine, pin_config(64, 0), u64::MAX)
                .is_err()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn request(world: &mut MpmSoilDiskWorld, engine: &Engine, tiles: &[SoilTileCoord]) -> Result<()> {
    world.begin_region_update(&[SoilRegionRequest {
        id: SoilRegionId(5),
        tiles,
    }])?;
    poll(world, engine)
}

fn pin_config(capacity: u32, halo_tiles: u32) -> MpmSoilPinConfig {
    MpmSoilPinConfig {
        minimum: SoilTileCoord([-8; 3]),
        dimensions: [32; 3],
        capacity,
        halo_tiles,
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn disk_gpu_pins_join_capture_expand_tiles_and_retry_without_durable_mutation() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, cfg) = seed(&directory);
    let keys: Vec<_> = archive.tiles().collect();
    let mut world = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        store,
        SoilResidencyLimits {
            regions: 2,
            tiles: 256,
            claims: 512,
        },
        io_limits(),
        129 * 160 + 16,
    )
    .unwrap();
    assert_eq!(world.gpu_pin_buffer_bytes(), 0);
    let allocations = DeviceScope::start().unwrap();
    assert!(
        world
            .prepare_gpu_pins(&engine, pin_config(64, 1), 1103)
            .is_err()
    );
    assert!(
        world
            .prepare_gpu_pins(&foreign, pin_config(64, 1), 1104)
            .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    world
        .prepare_gpu_pins(&engine, pin_config(64, 1), 1104)
        .unwrap();
    assert_eq!(world.gpu_pin_buffer_bytes(), 1104);
    assert!(
        world
            .prepare_gpu_pins(&engine, pin_config(64, 5), u64::MAX)
            .is_err()
    );
    assert_eq!(world.gpu_pin_buffer_bytes(), 1104);
    world.prepare_device_transfers(&engine, u64::MAX).unwrap();
    request(&mut world, &engine, &keys[..1]).unwrap();
    let active_before = world.active_particle_count();
    let allocations = DeviceScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    world.begin_region_update(&[]).unwrap();
    assert!(world.disable_gpu_pins().is_err());
    assert!(
        world
            .prepare_gpu_pins(&engine, pin_config(64, 0), 1104)
            .is_err()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let completion = loop {
        if let Some(completion) = world.readback.poll_next().unwrap() {
            break completion;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    };
    let Some(Pending::Capture {
        checkpoint, ticket, ..
    }) = world.pending.as_mut()
    else {
        panic!("expected pending capture");
    };
    assert_eq!(ticket.revision(), completion.ticket().revision());
    *checkpoint = Some(completion.into_result().unwrap());
    poll(&mut world, &engine).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    let stats = transfers.finish();
    let transfer = world.last_transfer_stats();
    assert_eq!(
        dispatches.finish().encoded_dispatches,
        2 + u64::from(transfer.imported_particles > 0)
    );
    assert_eq!(stats.upload_bytes, transfer.upload_bytes + 48);
    assert_eq!(
        stats.readback_bytes,
        u64::from(active_before) * 160 + 16 + 16 + 64 * 4
    );
    assert!(
        world
            .resident_tiles()
            .all(|tile| world.claims(tile).unwrap().pinned())
    );
    assert_eq!(world.resident_tiles().len(), 27);
    let selected: Vec<_> = archive
        .tiles()
        .filter(|tile| world.claims(*tile).is_some())
        .collect();
    let expected = archive.select(&selected, 129).unwrap();
    assert_eq!(world.identity().particle_ids(), expected.particle_ids());
    let words = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .words;
    assert_eq!(words, expected.checkpoint().words);
    let revision = world.committed_revision();
    let generation = world.store.as_ref().unwrap().generation();
    let tiles: Vec<_> = world.resident_tiles().collect();
    let identity = world.identity().clone();
    for bad in [
        pin_config(1, 1),
        MpmSoilPinConfig {
            minimum: SoilTileCoord([1000; 3]),
            dimensions: [1; 3],
            ..pin_config(64, 0)
        },
    ] {
        world.prepare_gpu_pins(&engine, bad, u64::MAX).unwrap();
        world.begin_region_update(&[]).unwrap();
        let error = poll(&mut world, &engine).unwrap_err();
        assert!(error.to_string().contains("incomplete GPU pins"));
        assert_eq!(world.status(), MpmSoilDiskWorldStatus::Ready);
        assert_eq!(world.committed_revision(), revision);
        assert_eq!(world.store.as_ref().unwrap().generation(), generation);
        assert_eq!(world.resident_tiles().collect::<Vec<_>>(), tiles);
        assert!(world.simulation.is_soil_identity_current(&identity));
        assert_eq!(
            world
                .simulation
                .checkpoint_soil_with(engine.queue().unwrap())
                .unwrap()
                .words,
            words
        );
    }
    world.disable_gpu_pins().unwrap();
    assert_eq!(world.gpu_pin_buffer_bytes(), 0);
    world.begin_region_update(&[]).unwrap();
    poll(&mut world, &engine).unwrap();
    assert_eq!(world.identity().particle_ids(), expected.particle_ids());
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        words
    );
    assert!(world.committed_revision() > revision);
}

#[test]
fn disk_world_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MpmSoilDiskWorld>();
}

fn poll_memory(world: &mut MpmSoilWorld, engine: &Engine) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !world.poll_region_update(engine)? {
        assert!(
            Instant::now() < deadline,
            "memory world transition deadline"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

fn active_swept_case(gpu_halo: Option<u32>, fail_capacity: bool) {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let mut state = fixture(2);
    for (index, position) in state.words[..8].chunks_exact_mut(4).enumerate() {
        position[0] = (0.124_f32 + index as f32 * 0.002).to_bits();
        position[1] = 0.33_f32.to_bits();
        position[2] = 0.02_f32.to_bits();
    }
    let archive = make_archive(&state);
    let layout = archive.layout();
    let envelope = MpmSoilRegionEnvelope {
        id: SoilRegionId(5),
        minimum: [1.0, 0.33, 0.02],
        maximum: [1.0, 0.33, 0.02],
        particle_travel_per_substep: [0.001; 3],
        substeps: 1,
    };
    let tool_tiles = envelope.tiles(layout, &state.cfg, 256).unwrap();
    let neighbor = SoilTileCoord([1, 2, 0]);
    assert!(!tool_tiles.contains(&neighbor));
    let residency = SoilResidencyLimits {
        regions: 2,
        tiles: if fail_capacity { tool_tiles.len() } else { 256 },
        claims: 512,
    };
    let mut store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    let mut disk = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap(),
        store,
        residency,
        io_limits(),
        20656,
    )
    .unwrap();
    let mut memory = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap(),
        archive,
        residency,
        20656,
    )
    .unwrap();
    if let Some(halo) = gpu_halo {
        disk.prepare_gpu_pins(&engine, pin_config(256, halo), u64::MAX)
            .unwrap();
        memory
            .prepare_gpu_pins(
                &engine,
                MpmSoilPinConfig {
                    minimum: SoilTileCoord([-8; 3]),
                    dimensions: [32; 3],
                    capacity: 256,
                    halo_tiles: halo,
                },
                u64::MAX,
            )
            .unwrap();
    }
    let first = [SoilTileCoord([0, 2, 0])];
    request(&mut disk, &engine, &first).unwrap();
    memory
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(5),
            tiles: &first,
        }])
        .unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    assert_eq!(disk.active_particle_count(), 1);
    assert_eq!(memory.active_particle_count(), 1);
    let generation = disk.store.as_ref().unwrap().generation();
    let archive_before = memory.synchronized_archive_bytes().unwrap();
    let words_before = memory
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .words;
    disk.begin_swept_region_update(&[envelope]).unwrap();
    memory.begin_swept_region_update(&[envelope]).unwrap();
    let disk_result = poll(&mut disk, &engine);
    let memory_result = poll_memory(&mut memory, &engine);
    if fail_capacity {
        assert!(disk_result.is_err() && memory_result.is_err());
        assert_eq!(disk.status(), MpmSoilDiskWorldStatus::Ready);
        assert_eq!(memory.status(), MpmSoilWorldStatus::Ready);
        assert_eq!(disk.store.as_ref().unwrap().generation(), generation);
        assert_eq!(memory.synchronized_archive_bytes().unwrap(), archive_before);
        assert_eq!(disk.committed_revision(), Some(0));
        assert_eq!(memory.committed_revision(), Some(0));
        assert_eq!(disk.remaining_region_substeps(), None);
        assert_eq!(memory.remaining_region_substeps(), None);
        for sim in [&mut disk.simulation, &mut memory.simulation] {
            assert_eq!(
                sim.checkpoint_soil_with(engine.queue().unwrap())
                    .unwrap()
                    .words,
                words_before
            );
        }
        disk.begin_region_update(&[]).unwrap();
        memory.begin_region_update(&[]).unwrap();
        poll(&mut disk, &engine).unwrap();
        poll_memory(&mut memory, &engine).unwrap();
        assert_eq!(disk.active_particle_count(), 1);
        assert_eq!(memory.active_particle_count(), 1);
        return;
    }
    disk_result.unwrap();
    memory_result.unwrap();
    assert_eq!(disk.active_particle_count(), 2);
    assert_eq!(memory.active_particle_count(), 2);
    assert!(disk.claims(neighbor).unwrap().pinned());
    assert!(memory.claims(neighbor).unwrap().pinned());
    assert_eq!(disk.claims(neighbor).unwrap().regions().len(), 0);
    assert_eq!(memory.claims(neighbor).unwrap().regions().len(), 0);
    if gpu_halo == Some(2) {
        let extra = SoilTileCoord([-2, 0, -2]);
        assert!(memory.claims(extra).unwrap().pinned());
        assert!(disk.claims(extra).unwrap().pinned());
    }
    assert_eq!(
        disk.identity().particle_ids(),
        memory.identity().particle_ids()
    );
    for sim in [&mut disk.simulation, &mut memory.simulation] {
        assert_eq!(
            sim.checkpoint_soil_with(engine.queue().unwrap())
                .unwrap()
                .words,
            state.words
        );
    }
    let mut reference = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    reference
        .restore_soil_with(engine.queue().unwrap(), &state)
        .unwrap();
    reference.step_n(&engine, 0.0001, 1).unwrap();
    disk.step_n(&engine, 0.0001, 1).unwrap();
    memory.step_n(&engine, 0.0001, 1).unwrap();
    let expected = reference
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    for sim in [&mut disk.simulation, &mut memory.simulation] {
        let actual = sim.checkpoint_soil_with(engine.queue().unwrap()).unwrap();
        assert_eq!(actual.words.len(), expected.words.len());
        assert_eq!(&actual.words[70..72], &expected.words[70..72]);
        for (&a, &b) in actual.words.iter().zip(&expected.words) {
            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
            assert!(a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 * b.abs().max(1.0));
        }
    }
    assert!(disk.step_n(&engine, 0.0001, 1).is_err());
    assert!(memory.step_n(&engine, 0.0001, 1).is_err());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn active_swept_pins_load_detached_neighbor_with_host_and_gpu_pins() {
    active_swept_case(None, false);
    active_swept_case(Some(0), false);
    active_swept_case(Some(2), false);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn active_swept_pin_capacity_failure_preserves_both_worlds_and_allows_raw_retry() {
    active_swept_case(None, true);
    active_swept_case(Some(0), true);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn swept_envelopes_load_neighbor_support_in_both_owners_before_stepping() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let mut state = fixture(2);
    for (index, position) in state.words[..8].chunks_exact_mut(4).enumerate() {
        position[0] = (0.124_f32 + index as f32 * 0.002).to_bits();
        position[1] = 0.33_f32.to_bits();
        position[2] = 0.02_f32.to_bits();
    }
    let archive = make_archive(&state);
    let layout = archive.layout();
    let mut store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    let residency = SoilResidencyLimits {
        regions: 2,
        tiles: 256,
        claims: 512,
    };
    let mut disk = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap(),
        store,
        residency,
        io_limits(),
        20656,
    )
    .unwrap();
    let mut memory = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap(),
        archive,
        residency,
        20656,
    )
    .unwrap();
    let envelope = MpmSoilRegionEnvelope {
        id: SoilRegionId(5),
        minimum: [0.124, 0.33, 0.02],
        maximum: [0.124, 0.33, 0.02],
        particle_travel_per_substep: [0.001; 3],
        substeps: 1,
    };
    let expected_tiles = envelope.tiles(layout, &state.cfg, 256).unwrap();
    let oversized = MpmSoilRegionEnvelope {
        substeps: 100,
        ..envelope
    };
    assert!(disk.begin_swept_region_update(&[oversized]).is_err());
    assert!(memory.begin_swept_region_update(&[oversized]).is_err());
    assert_eq!(disk.readback_stats().requests, 0);
    assert_eq!(memory.readback_stats().requests, 0);
    disk.begin_swept_region_update(&[envelope]).unwrap();
    memory.begin_swept_region_update(&[envelope]).unwrap();
    assert!(disk.begin_swept_region_update(&[envelope]).is_err());
    assert!(memory.begin_swept_region_update(&[envelope]).is_err());
    poll(&mut disk, &engine).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !memory.poll_region_update(&engine).unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(disk.resident_tiles().collect::<Vec<_>>(), expected_tiles);
    assert_eq!(memory.resident_tiles().collect::<Vec<_>>(), expected_tiles);
    assert_eq!(disk.active_particle_count(), 2);
    assert_eq!(memory.active_particle_count(), 2);
    assert_eq!(disk.remaining_region_substeps(), Some(1));
    assert_eq!(memory.remaining_region_substeps(), Some(1));
    assert!(disk.step_n(&engine, 0.0001, 2).is_err());
    assert!(memory.step_n(&engine, 0.0001, 2).is_err());
    assert_eq!(
        disk.identity().particle_ids(),
        memory.identity().particle_ids()
    );
    let mut reference = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    reference
        .restore_soil_with(engine.queue().unwrap(), &state)
        .unwrap();
    disk.step_n(&engine, 0.0001, 1).unwrap();
    memory.step_n(&engine, 0.0001, 1).unwrap();
    reference.step_n(&engine, 0.0001, 1).unwrap();
    assert_eq!(disk.remaining_region_substeps(), Some(0));
    assert_eq!(memory.remaining_region_substeps(), Some(0));
    assert!(disk.step_n(&engine, 0.0001, 1).is_err());
    assert!(memory.step_n(&engine, 0.0001, 1).is_err());
    assert!(disk.begin_swept_region_update(&[oversized]).is_err());
    assert!(memory.begin_swept_region_update(&[oversized]).is_err());
    assert_eq!(disk.remaining_region_substeps(), Some(0));
    assert_eq!(memory.remaining_region_substeps(), Some(0));
    let expected = reference
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    for simulation in [&mut disk.simulation, &mut memory.simulation] {
        let actual = simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap();
        assert_eq!(actual.words.len(), expected.words.len());
        assert_eq!(
            &actual.words[35 * 2..36 * 2],
            &expected.words[35 * 2..36 * 2]
        );
        for (index, (&a, &b)) in actual.words.iter().zip(&expected.words).enumerate() {
            if a != b {
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "stream word {index}: {a} != {b}"
                );
            }
        }
    }
    for swept in [true, false] {
        if swept {
            disk.begin_swept_region_update(&[envelope]).unwrap();
            memory.begin_swept_region_update(&[envelope]).unwrap();
        } else {
            disk.begin_region_update(&[]).unwrap();
            memory.begin_region_update(&[]).unwrap();
        }
        let old = if swept { Some(0) } else { Some(1) };
        assert_eq!(disk.remaining_region_substeps(), old);
        assert_eq!(memory.remaining_region_substeps(), old);
        poll(&mut disk, &engine).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !memory.poll_region_update(&engine).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let next = if swept { Some(1) } else { None };
        assert_eq!(disk.remaining_region_substeps(), next);
        assert_eq!(memory.remaining_region_substeps(), next);
        assert_eq!(disk.active_particle_count(), 2);
        assert_eq!(memory.active_particle_count(), 2);
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn live_disk_capture_write_load_and_reopen_preserve_history_and_identity() {
    live_disk_capture_case(false);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn incremental_disk_capture_write_load_and_reopen_preserve_history_and_identity() {
    live_disk_capture_case(true);
}

fn live_disk_capture_case(incremental: bool) {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (mut store, mut expected, cfg) = seed(&directory);
    if incremental {
        store.upgrade_index().unwrap();
    }
    let keys: Vec<_> = expected.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 129 * 160 + 16).unwrap();
    assert_eq!(world.active_particle_count(), 0);
    assert_eq!(world.total_particle_count(), 129);
    assert_eq!(world.store_storage_stats().unwrap().archived_particles, 129);
    assert_eq!(world.read.as_ref().unwrap().stats().read_bytes, 0);
    assert_eq!(world.staging_bytes(), 129 * 160 + 16);
    world
        .begin_region_update(&[
            SoilRegionRequest {
                id: SoilRegionId(5),
                tiles: &keys[..1],
            },
            SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys[..1],
            },
        ])
        .unwrap();
    poll(&mut world, &engine).unwrap();
    assert_eq!(world.claims(keys[0]).unwrap().regions().count(), 2);
    assert_eq!(
        world.active_particle_count(),
        expected.tile_particle_count(keys[0]).unwrap() as u32
    );
    let source = expected.select(&keys[..1], 129).unwrap();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    world.step_n(&engine, 0.0001, 1).unwrap();
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    let updated = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    expected.writeback(&source, &updated).unwrap();
    request(&mut world, &engine, &keys).unwrap();
    assert_eq!(world.active_particle_count(), 129);
    let all_keys: Vec<_> = expected.tiles().collect();
    let all = expected.select(&all_keys, 129).unwrap();
    assert_eq!(world.identity().particle_ids(), all.particle_ids());
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        all.checkpoint().words
    );
    assert_eq!(
        world
            .store
            .as_mut()
            .unwrap()
            .raw_mut()
            .load()
            .unwrap()
            .unwrap()
            .to_bytes(),
        expected.to_bytes()
    );
    assert_eq!(world.committed_revision(), Some(1));
    request(&mut world, &engine, &[]).unwrap();
    assert_eq!(world.active_particle_count(), 129);
    assert!(
        world
            .resident_tiles()
            .all(|tile| world.claims(tile).unwrap().pinned())
    );
    drop(world);
    let store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 129 * 160 + 16).unwrap();
    request(&mut world, &engine, &all_keys).unwrap();
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        all.checkpoint().words
    );
    assert_eq!(world.identity().particle_ids(), all.particle_ids());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn disk_capacity_rejects_before_write_and_preserves_retryable_active_state() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, mut cfg) = seed(&directory);
    let keys: Vec<_> = archive.tiles().collect();
    cfg.max_particles = 64;
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 64 * 160 + 16).unwrap();
    world.set_activity_monitoring(true).unwrap();
    request(&mut world, &engine, &keys[..1]).unwrap();
    world.step_n(&engine, 0.0001, 1).unwrap();
    let before = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    assert!(request(&mut world, &engine, &keys).is_err());
    assert_eq!(world.status(), MpmSoilDiskWorldStatus::Ready);
    assert!(world.last_activity_report().is_none());
    assert_eq!(world.committed_revision(), Some(0));
    assert_eq!(world.last_store_write().unwrap().0, 0);
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        before.words
    );
    request(&mut world, &engine, &[]).unwrap();
    assert_eq!(world.active_particle_count(), before.count);
    assert_eq!(world.committed_revision(), Some(2));
    let activity = world.last_activity_report().unwrap();
    assert_eq!(activity.revision, 2);
    assert_eq!(activity.activity.substeps, 1);
    assert_eq!(activity.activity.seconds, f64::from(0.0001_f32));
    assert_eq!(activity.activity.particles, before.count);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn activity_reports_capture_time_identity_and_pre_import_population_in_both_owners() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, cfg) = seed(&directory);
    let keys: Vec<_> = archive.tiles().collect();
    let mut disk = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        store,
        limits(),
        io_limits(),
        20656,
    )
    .unwrap();
    let mut memory = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        archive,
        limits(),
        20656,
    )
    .unwrap();
    assert!(disk.last_activity_report().is_none());
    assert!(memory.last_activity_report().is_none());
    disk.set_activity_monitoring(true).unwrap();
    memory.set_activity_monitoring(true).unwrap();
    request(&mut disk, &engine, &keys[..1]).unwrap();
    memory
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(5),
            tiles: &keys[..1],
        }])
        .unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    assert!(disk.last_activity_report().is_none());
    assert!(memory.last_activity_report().is_none());
    let initial_count = disk.active_particle_count();
    assert_eq!(initial_count, memory.active_particle_count());
    let ids = memory.identity().particle_ids().to_vec();
    let before_disk = disk.read.as_ref().unwrap().selection().checkpoint().clone();
    let before_memory = memory.selection.checkpoint().clone();
    let tile_layout = memory.archive.layout();
    for (dt, steps) in [(0.0001, 2), (0.0002, 1)] {
        disk.step_n(&engine, dt, steps).unwrap();
        memory.step_n(&engine, dt, steps).unwrap();
    }
    let elapsed = 2.0 * f64::from(0.0001_f32) + f64::from(0.0002_f32);
    let after_disk = disk
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let after_memory = memory
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let expected_disk = after_disk
        .activity_since(&before_disk, &ids, &ids, elapsed, 3)
        .unwrap();
    let expected_memory = after_memory
        .activity_since(&before_memory, &ids, &ids, elapsed, 3)
        .unwrap();
    let (_, expected_disk_tiles) = after_disk
        .activity_by_tile_since(&before_disk, &ids, &ids, elapsed, 3, tile_layout)
        .unwrap();
    let (_, expected_memory_tiles) = after_memory
        .activity_by_tile_since(&before_memory, &ids, &ids, elapsed, 3, tile_layout)
        .unwrap();
    let requested = [SoilRegionRequest {
        id: SoilRegionId(5),
        tiles: &keys[..2],
    }];
    disk.begin_region_update(&requested).unwrap();
    memory.begin_region_update(&requested).unwrap();
    assert!(disk.set_activity_monitoring(false).is_err());
    assert!(memory.set_activity_monitoring(false).is_err());
    assert!(disk.last_activity_report().is_none());
    assert!(memory.last_activity_report().is_none());
    poll(&mut disk, &engine).unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    assert_eq!(disk.last_activity_report().unwrap().activity, expected_disk);
    assert_eq!(
        memory.last_activity_report().unwrap().activity,
        expected_memory
    );
    assert_eq!(disk.last_activity_report().unwrap().revision, 1);
    assert_eq!(memory.last_activity_report().unwrap().revision, 1);
    assert_eq!(
        disk.last_activity_report().unwrap().tile_layout,
        tile_layout
    );
    assert_eq!(
        memory.last_activity_report().unwrap().tile_layout,
        tile_layout
    );
    assert_eq!(
        disk.last_activity_report().unwrap().tiles,
        expected_disk_tiles
    );
    assert_eq!(
        memory.last_activity_report().unwrap().tiles,
        expected_memory_tiles
    );
    assert!(disk.active_particle_count() > initial_count);
    assert!(memory.active_particle_count() > initial_count);
    assert!(expected_disk.max_acceleration > 0.0 && expected_memory.max_acceleration > 0.0);
    assert_eq!(disk.readback_stats().requests, 2);
    assert_eq!(memory.readback_stats().requests, 2);
    disk.step_n(&engine, 0.0001, 1).unwrap();
    memory.step_n(&engine, 0.0001, 1).unwrap();
    assert_eq!(disk.last_activity_report().unwrap().revision, 1);
    assert_eq!(memory.last_activity_report().unwrap().revision, 1);
    disk.begin_region_update(&[]).unwrap();
    memory.begin_region_update(&[]).unwrap();
    poll(&mut disk, &engine).unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    for report in [
        disk.last_activity_report().unwrap(),
        memory.last_activity_report().unwrap(),
    ] {
        assert_eq!(report.revision, 2);
        assert_eq!(report.activity.substeps, 1);
        assert_eq!(report.activity.seconds, f64::from(0.0001_f32));
        assert!(report.activity.particles > initial_count);
        assert_eq!(
            report
                .tiles
                .iter()
                .map(|t| t.current_particles)
                .sum::<u32>(),
            report.activity.particles
        );
        assert_eq!(
            report
                .tiles
                .iter()
                .map(|t| t.previous_particles)
                .sum::<u32>(),
            report.activity.particles
        );
    }
    disk.set_activity_monitoring(false).unwrap();
    memory.set_activity_monitoring(false).unwrap();
    assert!(disk.last_activity_report().is_none());
    assert!(memory.last_activity_report().is_none());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn swept_motion_guard_fault_cannot_commit_either_world_or_change_durable_generation() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, cfg) = seed(&directory);
    let residency = SoilResidencyLimits {
        regions: 3,
        tiles: 4096,
        claims: 12288,
    };
    let mut disk = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        store,
        residency,
        MpmSoilStoreReadLimits {
            tiles: 4096,
            encoded_bytes: 1 << 20,
            ..io_limits()
        },
        20656,
    )
    .unwrap();
    let mut memory = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        archive,
        residency,
        20656,
    )
    .unwrap();
    let envelope = MpmSoilRegionEnvelope {
        id: SoilRegionId(5),
        minimum: [0.02, 0.3, -0.07],
        maximum: [0.02, 0.3, -0.07],
        particle_travel_per_substep: [0.0; 3],
        substeps: 3,
    };
    let bytes = 16 + u64::from(cfg.max_particles) * 128;
    assert!(disk.prepare_swept_motion_guard(bytes - 1).is_err());
    assert!(memory.prepare_swept_motion_guard(bytes - 1).is_err());
    assert_eq!(disk.swept_motion_guard_bytes(), 0);
    assert_eq!(memory.swept_motion_guard_bytes(), 0);
    disk.prepare_swept_motion_guard(bytes).unwrap();
    memory.prepare_swept_motion_guard(bytes).unwrap();
    assert_eq!(disk.swept_motion_guard_bytes(), bytes);
    assert_eq!(memory.swept_motion_guard_bytes(), bytes);
    disk.begin_swept_region_update(&[envelope]).unwrap();
    memory.begin_swept_region_update(&[envelope]).unwrap();
    assert!(disk.prepare_swept_motion_guard(bytes).is_err());
    assert!(memory.prepare_swept_motion_guard(bytes).is_err());
    assert!(
        disk.simulation
            .motion_guard
            .as_ref()
            .unwrap()
            .bounds
            .is_none()
    );
    assert!(
        memory
            .simulation
            .motion_guard
            .as_ref()
            .unwrap()
            .bounds
            .is_none()
    );
    poll(&mut disk, &engine).unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    assert!(disk.active_particle_count() > 0);
    assert_eq!(
        disk.simulation.motion_guard.as_ref().unwrap().bounds,
        Some([0.0; 4])
    );
    assert_eq!(
        memory.simulation.motion_guard.as_ref().unwrap().bounds,
        Some([0.0; 4])
    );
    let generation = disk.store.as_ref().unwrap().generation();
    let archived = memory.synchronized_archive_bytes().unwrap();
    let raw = |sim: &MpmSimulation| {
        sim.checkpoint_buffers()
            .iter()
            .map(|buffer| {
                crate::mpm::replay_tests::read_words(
                    engine.device().unwrap(),
                    engine.queue().unwrap(),
                    buffer,
                )
            })
            .collect::<Vec<_>>()
    };
    let disk_before = raw(&disk.simulation);
    let memory_before = raw(&memory.simulation);
    disk.step_n(&engine, 0.0001, 3).unwrap();
    memory.step_n(&engine, 0.0001, 3).unwrap();
    assert_eq!(raw(&disk.simulation), disk_before);
    assert_eq!(raw(&memory.simulation), memory_before);
    disk.begin_region_update(&[]).unwrap();
    memory.begin_region_update(&[]).unwrap();
    assert!(poll(&mut disk, &engine).is_err());
    assert!(poll_memory(&mut memory, &engine).is_err());
    assert_eq!(disk.status(), MpmSoilDiskWorldStatus::Faulted);
    assert_eq!(memory.status(), MpmSoilWorldStatus::Faulted);
    assert_eq!(disk.store.as_ref().unwrap().generation(), generation);
    assert_eq!(memory.last_committed_archive_bytes(), archived);
    assert!(disk.step_n(&engine, 0.0001, 1).is_err());
    assert!(memory.step_n(&engine, 0.0001, 1).is_err());
    assert_eq!(disk.committed_revision, Some(0));
    assert_eq!(memory.committed_revision, Some(0));
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn disk_write_failure_retains_capture_and_faults_without_losing_gpu_state() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, cfg) = seed(&directory);
    let keys: Vec<_> = archive.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 129 * 160 + 16).unwrap();
    request(&mut world, &engine, &keys[..1]).unwrap();
    world.step_n(&engine, 0.0001, 1).unwrap();
    let before = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let ids = world.identity().particle_ids().to_vec();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    fs::create_dir(directory.0.join("pending")).unwrap();
    assert!(request(&mut world, &engine, &[]).is_err());
    assert_eq!(world.status(), MpmSoilDiskWorldStatus::Faulted);
    let (captured, captured_ids) = world.captured_state().unwrap();
    assert_eq!(captured.words, before.words);
    assert_eq!(captured_ids, ids);
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    assert!(world.begin_region_update(&[]).is_err());
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        before.words
    );
    drop(world);
    fs::remove_dir(directory.0.join("pending")).unwrap();
    let store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    assert_eq!(store.generation(), Some(1));
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn load_failure_after_durable_write_faults_and_retains_the_persisted_capture() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, archive, cfg) = seed(&directory);
    let keys: Vec<_> = archive.tiles().collect();
    let bad = archive.select(&keys[1..2], 129).unwrap();
    let bytes = bad
        .checkpoint()
        .partition_tiles(archive.layout(), bad.particle_ids(), bad.material_ids())
        .unwrap()
        .to_bytes();
    let path = directory
        .0
        .join(format!("{}.tile", blake3::hash(&bytes).to_hex()));
    assert!(path.exists());
    fs::write(path, b"damaged inactive tile").unwrap();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 129 * 160 + 16).unwrap();
    request(&mut world, &engine, &keys[..1]).unwrap();
    world.step_n(&engine, 0.0001, 1).unwrap();
    let updated = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert!(request(&mut world, &engine, &keys[..2]).is_err());
    assert_eq!(world.status(), MpmSoilDiskWorldStatus::Faulted);
    assert_eq!(world.committed_revision(), Some(0));
    assert_eq!(world.active_particle_count(), updated.count);
    assert_eq!(world.store.as_ref().unwrap().generation(), Some(2));
    let (written_revision, written) = world.last_store_write().unwrap();
    assert_eq!((written_revision, written.generation), (1, 2));
    assert_eq!(world.committed_read_stats().generation, 1);
    assert_eq!(world.captured_state().unwrap().0.words, updated.words);
    let persisted = world
        .store
        .as_ref()
        .unwrap()
        .read_tiles(&keys[..1], io_limits())
        .unwrap()
        .unwrap();
    assert_eq!(persisted.selection().checkpoint().words, updated.words);
    assert!(
        world
            .store
            .as_ref()
            .unwrap()
            .validate_read(world.read.as_ref().unwrap())
            .is_err()
    );
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        updated.words
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn moving_particle_crosses_disk_tile_without_discard_or_unrequested_eviction() {
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let mut state = fixture(1);
    state.words[0] = (-0.00001_f32).to_bits();
    state.words[4] = 1_f32.to_bits();
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    let mut store = MpmSoilStore::open(&directory.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    let sim = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    let mut restricted = limits();
    restricted.tiles = 1;
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, restricted, io_limits(), 129 * 160 + 16)
            .unwrap();
    world
        .prepare_gpu_pins(&engine, pin_config(64, 0), 1104)
        .unwrap();
    request(&mut world, &engine, &keys).unwrap();
    let ids = world.identity().particle_ids().to_vec();
    world.step_n(&engine, 0.0001, 2).unwrap();
    let moved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert!(f32::from_bits(moved.words[0]) > 0.0);
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    assert!(request(&mut world, &engine, &keys).is_err());
    assert_eq!(world.status(), MpmSoilDiskWorldStatus::Ready);
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    request(&mut world, &engine, &[]).unwrap();
    let now: Vec<_> = world.resident_tiles().collect();
    assert_eq!(now.len(), 1);
    assert_eq!(now[0].0[0], 0);
    assert!(world.claims(now[0]).unwrap().pinned());
    assert_eq!(world.identity().particle_ids(), ids);
    assert_eq!(world.active_particle_count(), 1);
    assert_eq!(
        world
            .store
            .as_ref()
            .unwrap()
            .tile_particle_count(keys[0])
            .unwrap(),
        None
    );
    let read = world
        .store
        .as_ref()
        .unwrap()
        .read_tiles(&now, io_limits())
        .unwrap()
        .unwrap();
    assert_eq!(read.selection().checkpoint().words, moved.words);
    assert_eq!(read.selection().particle_ids(), ids);
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn disk_device_gather_reuses_evolved_history_and_obeys_resource_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let (store, mut expected, cfg) = seed(&directory);
    let keys: Vec<_> = expected.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world =
        MpmSoilDiskWorld::new(&engine, sim, store, limits(), io_limits(), 129 * 160 + 16).unwrap();
    let bytes = 129 * 508 + 16;
    assert_eq!(world.transfer_scratch_bytes(), 0);
    let allocations = DeviceScope::start().unwrap();
    assert!(world.prepare_device_transfers(&engine, bytes - 1).is_err());
    assert!(world.prepare_device_transfers(&foreign, bytes).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let allocations = DeviceScope::start().unwrap();
    world.prepare_device_transfers(&engine, bytes).unwrap();
    assert_eq!(allocations.finish().observations, 3);
    assert_eq!(world.transfer_scratch_bytes(), bytes);
    let allocations = DeviceScope::start().unwrap();
    world.prepare_device_transfers(&engine, bytes).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    request(&mut world, &engine, &keys[..1]).unwrap();
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::DeviceGather
    );
    assert_eq!(
        world.committed_read_stats().particles,
        world.active_particle_count()
    );
    let initial = expected.select(&keys[..1], 129).unwrap();
    let retained = world.active_particle_count();
    world.step_n(&engine, 0.0001, 1).unwrap();
    let evolved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    expected.writeback(&initial, &evolved).unwrap();
    let captures_before = world.readback_stats();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    request(&mut world, &engine, &keys[..2]).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(dispatches.finish().encoded_dispatches, 1);
    let observed = transfers.finish();
    let stats = world.last_transfer_stats();
    let total = world.active_particle_count();
    let imported = total - retained;
    assert!(retained > 0 && imported > 0);
    assert_eq!(stats.path, MpmSoilTransferPath::DeviceGather);
    assert_eq!(
        (
            stats.retained_particles,
            stats.imported_particles,
            stats.removed_particles
        ),
        (retained, imported, 0)
    );
    assert_eq!(
        stats.upload_bytes,
        16 + u64::from(total) * 4 + u64::from(imported) * 168
    );
    assert_eq!(observed.upload_bytes, stats.upload_bytes);
    assert_eq!(observed.readback_bytes, u64::from(retained) * 160 + 16);
    let full_upload_bytes = u64::from(total) * 168 + expected.material_ids().len() as u64 * 8;
    assert!(stats.upload_bytes < full_upload_bytes);
    let selected = expected.select(&keys[..2], 129).unwrap();
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        selected.checkpoint().words
    );
    assert_eq!(world.identity().particle_ids(), selected.particle_ids());
    assert_eq!(world.committed_read_stats().particles, total);
    assert!(world.committed_read_stats().read_bytes > 0);
    assert_eq!(
        world.last_store_write().unwrap().0,
        world.committed_revision().unwrap()
    );
    assert_eq!(
        world.readback_stats().completed,
        captures_before.completed + 1
    );
    println!(
        "disk gather: retained={retained} imported={imported} upload={} full_upload={full_upload_bytes} capture_readback={} scratch={bytes}",
        stats.upload_bytes, observed.readback_bytes
    );
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    request(&mut world, &engine, &[]).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    let observed = transfers.finish();
    assert_eq!(observed.upload_bytes, 0);
    assert_eq!(observed.readback_bytes, u64::from(total) * 160 + 16);
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::Unchanged
    );
    world.disable_device_transfers().unwrap();
    assert_eq!(world.transfer_scratch_bytes(), 0);
    request(&mut world, &engine, &keys).unwrap();
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::FullUpload
    );
    assert_eq!(world.active_particle_count(), 129);
    let all_keys: Vec<_> = expected.tiles().collect();
    let all = expected.select(&all_keys, 129).unwrap();
    assert_eq!(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .words,
        all.checkpoint().words
    );
    assert_eq!(world.identity().particle_ids(), all.particle_ids());
}
