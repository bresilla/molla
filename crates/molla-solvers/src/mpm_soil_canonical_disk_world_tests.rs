use super::*;
use crate::mpm::checkpoint::tests::new_sim;
use molla_vehicle::SoilMaterialId;

#[path = "mpm_soil_disk_recenter_tests.rs"]
mod recenter_tests;

fn seed_canonical(
    dir: &Directory,
) -> (
    MpmSoilCanonicalStore,
    MpmSoilCanonicalTileArchive,
    MpmConfig,
) {
    seed_canonical_step(dir, MpmConfig::sand().time_step)
}

fn seed_canonical_step(
    dir: &Directory,
    time_step: MpmTimeStep,
) -> (
    MpmSoilCanonicalStore,
    MpmSoilCanonicalTileArchive,
    MpmConfig,
) {
    let mut soil = fixture(129);
    soil.cfg.time_step = time_step;
    soil.cfg.grid_size = 64;
    soil.cfg.cell_size = 0.125;
    soil.cfg.grid_origin = [-4.0; 3];
    for (i, p) in soil.words[..129 * 4].chunks_exact_mut(4).enumerate() {
        p[0] = ((i % 4) as f32 + 0.125).to_bits();
        p[1] = 2.0f32.to_bits();
        p[2] = 2.0f32.to_bits();
    }
    let cfg = soil.cfg;
    let archive = MpmSoilFrameCheckpoint::new(soil, [1e12; 3])
        .unwrap()
        .partition_canonical_tiles(
            SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
            &(0..129)
                .map(|i| SoilParticleId((i << 40) + 7))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(0),
                SoilMaterialId(1 << 50),
                SoilMaterialId(u64::MAX),
            ],
        )
        .unwrap();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    (store, archive, cfg)
}
fn canonical_bounds() -> MpmSoilStoreReadLimits {
    MpmSoilStoreReadLimits {
        tiles: 32,
        particles: 129,
        encoded_bytes: 32 * 1052 + 129 * 181,
    }
}
fn world(
    engine: &Engine,
    store: MpmSoilCanonicalStore,
    cfg: MpmConfig,
    mode: usize,
    origin: [f64; 3],
) -> MpmSoilDiskWorld {
    MpmSoilDiskWorld::new_canonical(
        engine,
        new_sim(engine, cfg, mode),
        store,
        origin,
        SoilResidencyLimits {
            regions: 3,
            tiles: 256,
            claims: 768,
        },
        canonical_bounds(),
        129 * 160 + 16,
    )
    .unwrap()
}
fn capture(world: &mut MpmSoilDiskWorld, engine: &Engine) -> MpmSoilFrameCheckpoint {
    MpmSoilFrameCheckpoint::new(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap(),
        world.world_origin().unwrap(),
    )
    .unwrap()
}
fn corrupt_tile(dir: &Directory, key: SoilTileCoord) {
    let path = fs::read_dir(&dir.0)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension().is_some_and(|ext| ext == "tile")
                && MpmSoilCanonicalTileArchive::from_bytes(&fs::read(path).unwrap(), 129)
                    .is_ok_and(|archive| archive.tiles().next() == Some(key))
        })
        .unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[100] ^= 1;
    fs::write(path, bytes).unwrap();
}

#[test]
fn canonical_backend_dispatch_preserves_family_failure_inputs() {
    let raw_dir = Directory::new();
    let (raw, _, _) = seed(&raw_dir);
    let key = raw.tiles().unwrap().next().unwrap();
    let raw_read = raw.read_tiles(&[key], io_limits()).unwrap().unwrap();
    let updated = raw_read.selection().checkpoint().clone();
    let dir = Directory::new();
    let (canonical, _, cfg) = seed_canonical(&dir);
    let store = Store::Canonical {
        store: canonical,
        origin: [1e12; 3],
        grid: cfg.grid_origin,
        cells: [0; 3],
    };
    let admission = SelectionAdmission {
        entries: vec![],
        max_particles: 129,
        limits: io_limits(),
    };
    let mut task = store.begin_writeback(Read::Raw(raw_read), updated, io_limits(), admission);
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert!(completion.store.validate_read(&completion.read).is_err());
    assert_eq!(completion.store.generation(), Some(1));
    assert!(task.poll().is_err());
}

#[test]
#[ignore = "requires GPU; canonical live owner preserves evolving soil across storage boundaries"]
fn canonical_live_overlap_transfer_activity_step_and_reopen_all_grid_modes() {
    canonical_live_overlap_case(false);
}

#[test]
#[ignore = "requires GPU; incremental canonical storage lifecycle"]
fn incremental_canonical_live_overlap_transfer_and_reopen_all_grid_modes() {
    canonical_live_overlap_case(true);
}

fn canonical_live_overlap_case(incremental: bool) {
    let engine = Engine::new_gpu().unwrap();
    for mode in 0..4 {
        let dir = Directory::new();
        let (mut store, initial, cfg) = seed_canonical(&dir);
        if incremental {
            store.upgrade_index().unwrap();
        }
        let keys: Vec<_> = initial.tiles().collect();
        let mut live = world(&engine, store, cfg, mode, [1e12; 3]);
        assert_eq!(live.active_particle_count(), 0);
        assert_eq!(live.total_particle_count(), 129);
        assert_eq!(live.store_storage_stats().unwrap().archived_particles, 129);
        assert_eq!(live.committed_read_stats().read_bytes, 0);
        assert_eq!(live.readback_stats().requests, 0);
        live.begin_region_update(&[
            SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys[..2],
            },
            SoilRegionRequest {
                id: SoilRegionId(2),
                tiles: &keys[1..2],
            },
        ])
        .unwrap();
        poll(&mut live, &engine).unwrap();
        let selected = initial
            .select(&keys[..2], 129, [1e12; 3], cfg.grid_origin)
            .unwrap();
        assert_eq!(
            capture(&mut live, &engine).to_bytes(),
            selected
                .framed_selection()
                .to_frame_checkpoint()
                .unwrap()
                .to_bytes()
        );
        assert_eq!(
            live.identity().particle_ids(),
            selected.framed_selection().particle_ids()
        );
        live.prepare_device_transfers(&engine, u64::MAX).unwrap();
        request(&mut live, &engine, &keys).unwrap();
        assert_eq!(live.active_particle_count(), 129);
        live.set_activity_monitoring(true).unwrap();
        live.step_n(&engine, 0.0001, 2).unwrap();
        let after = capture(&mut live, &engine);
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(capture(&mut live, &engine).to_bytes(), after.to_bytes());
        assert_eq!(live.active_particle_count(), 129);
        let mut expected_counts = std::collections::BTreeMap::new();
        for tile in after.particle_tiles(initial.layout()).unwrap() {
            *expected_counts.entry(tile).or_insert(0u32) += 1;
        }
        let actual_counts: std::collections::BTreeMap<_, _> = live
            .last_activity_report()
            .unwrap()
            .tiles
            .iter()
            .filter(|tile| tile.current_particles != 0)
            .map(|tile| (tile.tile, tile.current_particles))
            .collect();
        assert_eq!(actual_counts, expected_counts);
        let ids = live.identity().particle_ids().to_vec();
        let view = live.simulation.soil_identity_buffers().unwrap();
        let words = crate::mpm::replay_tests::read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            view.particles,
        );
        let expected: Vec<_> = ids
            .iter()
            .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
            .collect();
        assert_eq!(&words[..expected.len()], expected);
        drop(live);
        let mut store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
        let saved = store.load().unwrap().unwrap();
        let all: Vec<_> = saved.tiles().collect();
        let selected = saved.select(&all, 129, [1e12; 3], cfg.grid_origin).unwrap();
        assert_eq!(
            selected
                .framed_selection()
                .to_frame_checkpoint()
                .unwrap()
                .to_bytes(),
            after.to_bytes()
        );
        let mut reloaded = world(&engine, store, cfg, mode, [1e12; 3]);
        request(&mut reloaded, &engine, &all).unwrap();
        assert_eq!(capture(&mut reloaded, &engine).to_bytes(), after.to_bytes());
        reloaded.step_n(&engine, 0.0001, 1).unwrap();
    }
}

#[test]
#[ignore = "requires GPU; canonical byte bounds precede publication and pins preserve world keys"]
fn canonical_live_budget_retry_gpu_pins_and_swept_horizon() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = seed_canonical(&dir);
    let keys: Vec<_> = archive.tiles().collect();
    let mut live = world(&engine, store, cfg, 2, [1e12; 3]);
    live.io_limits.encoded_bytes = 4 * 1024 + 129 * 168;
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    live.begin_region_update(&[SoilRegionRequest {
        id: SoilRegionId(1),
        tiles: &keys,
    }])
    .unwrap();
    assert!(live.poll_region_update(&foreign).is_err());
    assert!(poll(&mut live, &engine).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
    assert_eq!(live.committed_revision(), None);
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    assert_eq!(live.active_particle_count(), 0);
    live.io_limits.encoded_bytes = 4 * 1052 + 129 * 181;
    live.prepare_gpu_pins(
        &engine,
        MpmSoilPinConfig {
            minimum: SoilTileCoord([999_999_999_996; 3]),
            dimensions: [12; 3],
            capacity: 128,
            halo_tiles: 0,
        },
        u64::MAX,
    )
    .unwrap();
    request(&mut live, &engine, &keys).unwrap();
    request(&mut live, &engine, &[]).unwrap();
    assert_eq!(live.resident_tiles().collect::<Vec<_>>(), keys);
    live.begin_swept_region_update(&[MpmSoilRegionEnvelope {
        id: SoilRegionId(2),
        minimum: [0.0, 1.9, 1.9],
        maximum: [0.25, 2.1, 2.1],
        particle_travel_per_substep: [0.001; 3],
        substeps: 1,
    }])
    .unwrap();
    poll(&mut live, &engine).unwrap();
    assert!(live.resident_tiles().count() > keys.len());
    live.step_n(&engine, 0.0001, 1).unwrap();
    assert!(live.step_n(&engine, 0.0001, 1).is_err());
    let observed = capture(&mut live, &engine);
    let affected: BTreeSet<_> = keys
        .iter()
        .copied()
        .chain(observed.particle_tiles(archive.layout()).unwrap())
        .collect();
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    assert!(request(&mut live, &engine, &[]).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    live.io_limits.encoded_bytes = affected.len() as u64 * 1052 + 129 * 181;
    request(&mut live, &engine, &[]).unwrap();
    assert_eq!(live.remaining_region_substeps(), None);
}

#[test]
#[ignore = "requires GPU; canonical publication/load failure retains recovery state"]
fn canonical_live_unloaded_corruption_and_postwrite_load_fault_preserve_capture() {
    let engine = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = seed_canonical(&dir);
    let keys: Vec<_> = archive.tiles().collect();
    corrupt_tile(&dir, keys[1]);
    let mut live = world(&engine, store, cfg, 2, [1e12; 3]);
    request(&mut live, &engine, &keys[..1]).unwrap();
    live.step_n(&engine, 0.0001, 2).unwrap();
    let updated = capture(&mut live, &engine);
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    assert!(request(&mut live, &engine, &keys[..2]).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Faulted);
    assert_eq!(live.committed_revision(), Some(0));
    assert_eq!(live.last_store_write().unwrap().1.generation, 2);
    assert_eq!(live.committed_read_stats().generation, 1);
    assert_ne!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    assert_eq!(live.captured_state().unwrap().0.words, updated.soil().words);
    assert_eq!(capture(&mut live, &engine).to_bytes(), updated.to_bytes());
    assert!(live.step_n(&engine, 0.0001, 1).is_err());
    drop(live);
    let mut store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
    let moved_keys: Vec<_> = updated
        .particle_tiles(archive.layout())
        .unwrap()
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert_eq!(store.tile_particle_count(keys[0]).unwrap(), None);
    let persisted = store
        .read_tiles(&moved_keys, canonical_bounds())
        .unwrap()
        .unwrap()
        .select([1e12; 3], cfg.grid_origin)
        .unwrap();
    assert_eq!(
        persisted
            .framed_selection()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes(),
        updated.to_bytes()
    );
    assert!(store.load().is_err());
}

#[test]
#[ignore = "requires GPU; canonical owners choose exact frames or fault on lossy imports"]
fn canonical_live_explicit_frame_and_lossy_import_have_distinct_acceptance() {
    let engine = Engine::new_gpu().unwrap();
    for lossy in [false, true] {
        let dir = Directory::new();
        let (store, archive, mut cfg) = seed_canonical(&dir);
        let keys: Vec<_> = archive.tiles().collect();
        let origin = if lossy { [0.0; 3] } else { [1e12 + 0.25; 3] };
        if !lossy {
            cfg.grid_origin = [-4.25; 3];
        }
        let mut live = world(&engine, store, cfg, 1, origin);
        let outcome = request(&mut live, &engine, &keys);
        if lossy {
            assert!(outcome.is_err());
            assert_eq!(live.status(), MpmSoilDiskWorldStatus::Faulted);
            assert_eq!(live.active_particle_count(), 0);
            assert_eq!(live.committed_revision(), None);
        } else {
            outcome.unwrap();
            assert_eq!(live.world_origin(), Some(origin));
            let expected = archive.select(&keys, 129, origin, cfg.grid_origin).unwrap();
            assert_eq!(
                capture(&mut live, &engine).to_bytes(),
                expected
                    .framed_selection()
                    .to_frame_checkpoint()
                    .unwrap()
                    .to_bytes()
            );
            assert!(
                live.prepare_gpu_pins(&engine, pin_config(64, 0), u64::MAX)
                    .is_err()
            );
            live.step_n(&engine, 0.0001, 1).unwrap();
            request(&mut live, &engine, &[]).unwrap();
        }
    }
}
