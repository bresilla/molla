use super::tests::{Directory, archive, limits, read_limits};

#[test]
fn incremental_batched_dirty_tiles_reduce_writes_with_exact_serial_parity() {
    const COUNT: u32 = 4096;
    let mut state = crate::mpm::checkpoint::tests::fixture(COUNT);
    state.cfg.max_particles = COUNT;
    for (i, position) in state.words[..COUNT as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        position[0] = (i as f32 + 0.125).to_bits();
    }
    let initial = state
        .partition_tiles(
            SoilTileLayout::new([0.; 3], 1.).unwrap(),
            &(0..COUNT)
                .map(|id| SoilParticleId(u64::from(id)))
                .collect::<Vec<_>>(),
            &[SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)],
        )
        .unwrap();
    let limits = MpmSoilStoreLimits {
        particles: COUNT,
        tiles: COUNT,
    };
    let keys: Vec<_> = initial.tiles().collect();
    let mut reference = None;
    let mut serial_bytes = 0;
    for serial in [true, false] {
        let directory = Directory::new();
        let mut store = MpmSoilStore::open(&directory.0, limits).unwrap();
        store.save(&initial).unwrap();
        let migration = store.upgrade_index().unwrap().written_bytes;
        store.serial_index_updates = serial;
        let mut written = 0;
        for step in 0..64 {
            let start = step * 13 * 4 % COUNT as usize;
            let read = store
                .read_tiles(&keys[start..start + 4], read_limits(4, 4))
                .unwrap()
                .unwrap();
            let mut state = read.selection().checkpoint().clone();
            for particle in 0..4 {
                state.words[16 + 4 * particle] =
                    (0.375 + step as f32 * 0.001 + particle as f32 * 0.01).to_bits();
            }
            written += store
                .writeback(&read, &state, read_limits(4, 4))
                .unwrap()
                .written_bytes;
            assert_eq!(store.storage_stats().unwrap().cached_tiles, 4);
            if step % 8 == 7 {
                drop(store);
                store = MpmSoilStore::open(&directory.0, limits).unwrap();
                store.serial_index_updates = serial;
            }
        }
        let result = store.load().unwrap().unwrap().to_bytes();
        if serial {
            serial_bytes = written;
            reference = Some(result.clone());
        } else {
            assert_eq!(Some(&result), reference.as_ref());
            assert!(written * 2 < serial_bytes);
        }
        store.collect_unreferenced().unwrap();
        assert_eq!(store.load().unwrap().unwrap().to_bytes(), result);
        eprintln!(
            "BATCH_STORE {}",
            serde_json::json!({"serial":serial,"tiles":COUNT,"updates":64,"dirty_tiles_per_update":4,"migration_bytes":migration,"written_bytes":written,"root_bytes":store.storage_stats().unwrap().manifest_encoded_bytes})
        );
    }
}

#[test]
fn incremental_page_install_failure_preserves_v1_and_orphan_collection_allows_retry() {
    struct FirstPage(Option<[u8; 32]>);
    impl index::Pages for FirstPage {
        fn read(&mut self, _: &[u8; 32], _: &mut [u8; index::PAGE_BYTES]) -> Result<usize> {
            panic!("empty index read")
        }
        fn write(&mut self, hash: &[u8; 32], _: &[u8]) -> Result<()> {
            self.0.get_or_insert(*hash);
            Ok(())
        }
    }
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let expected = archive();
    store.save(&expected).unwrap();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let mut first = FirstPage(None);
    index::build_sorted(&mut first, store.manifest.as_ref().unwrap().iter()).unwrap();
    let bad = tree_manifest::page_path(&directory.0, &first.0.unwrap());
    fs::write(&bad, b"corrupt uncommitted page").unwrap();
    assert!(store.upgrade_index().is_err());
    assert!(store.storage_stats().is_err());
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert!(store.collect_unreferenced().unwrap() > 0);
    assert!(!bad.exists());
    assert_eq!(store.upgrade_index().unwrap().generation, 2);
}

#[test]
fn incremental_selected_updates_reduce_real_index_bytes_with_exact_v1_parity() {
    const COUNT: u32 = 4096;
    let mut state = crate::mpm::checkpoint::tests::fixture(COUNT);
    state.cfg.max_particles = COUNT;
    for (i, position) in state.words[..COUNT as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        position[0] = (i as f32 + 0.125).to_bits();
    }
    let initial = state
        .partition_tiles(
            SoilTileLayout::new([0.; 3], 1.).unwrap(),
            &(0..COUNT)
                .map(|id| SoilParticleId(u64::from(id)))
                .collect::<Vec<_>>(),
            &[SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)],
        )
        .unwrap();
    let limits = MpmSoilStoreLimits {
        particles: COUNT,
        tiles: COUNT,
    };
    let keys: Vec<_> = initial.tiles().collect();
    let mut legacy_bytes = 0;
    let mut reference = None;
    for incremental in [false, true] {
        let directory = Directory::new();
        let mut store = MpmSoilStore::open(&directory.0, limits).unwrap();
        store.save(&initial).unwrap();
        let migration_bytes = if incremental {
            store.upgrade_index().unwrap().written_bytes
        } else {
            0
        };
        let baseline = store.storage_stats().unwrap();
        let mut written = 0;
        for step in 0..64 {
            let start = step * 13 * 4 % COUNT as usize;
            let read = store
                .read_tiles(&keys[start..start + 4], read_limits(4, 4))
                .unwrap()
                .unwrap();
            let mut state = read.selection().checkpoint().clone();
            state.words[16] = (0.375 + step as f32 * 0.001).to_bits();
            written += store
                .writeback(&read, &state, read_limits(4, 4))
                .unwrap()
                .written_bytes;
            assert_eq!(
                store.storage_stats().unwrap().manifest_buffer_bytes,
                baseline.manifest_buffer_bytes
            );
            assert_eq!(store.storage_stats().unwrap().cached_tiles, 4);
            if step % 8 == 7 {
                drop(store);
                store = MpmSoilStore::open(&directory.0, limits).unwrap();
            }
        }
        let result = store.load().unwrap().unwrap().to_bytes();
        if incremental {
            assert_eq!(Some(&result), reference.as_ref());
            assert!(written * 16 < legacy_bytes);
            let page = store
                .manifest
                .as_ref()
                .unwrap()
                .index_files()
                .unwrap()
                .pop()
                .unwrap();
            drop(store);
            let good = fs::read(&page).unwrap();
            let mut bad = good.clone();
            *bad.last_mut().unwrap() ^= 1;
            fs::write(&page, bad).unwrap();
            assert!(MpmSoilStore::open(&directory.0, limits).is_err());
            fs::write(page, good).unwrap();
            store = MpmSoilStore::open(&directory.0, limits).unwrap();
            store.collect_unreferenced().unwrap();
            assert_eq!(store.load().unwrap().unwrap().to_bytes(), result);
        } else {
            legacy_bytes = written;
            reference = Some(result);
        }
        println!(
            "INCREMENTAL_IO {{\"incremental\":{incremental},\"tiles\":{COUNT},\"updates\":64,\"selected_particles\":4,\"migration_bytes\":{migration_bytes},\"written_bytes\":{written},\"root_bytes\":{}}}",
            baseline.manifest_encoded_bytes
        );
    }
}
use super::*;

fn changed(store: &MpmSoilStore) -> (MpmSoilStoreRead, MpmSoilCheckpoint) {
    let key = store.tiles().unwrap().next().unwrap();
    let read = store
        .read_tiles(&[key], read_limits(129, 4))
        .unwrap()
        .unwrap();
    let mut state = read.selection().checkpoint().clone();
    state.words[state.count as usize * 4] = 0.5f32.to_bits();
    (read, state)
}

#[test]
fn incremental_upgrade_roundtrip_noop_partial_update_collection_and_empty() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(store.upgrade_index().is_err());
    store.healthy().unwrap();
    let mut expected = archive();
    store.save(&expected).unwrap();
    let (old, _) = changed(&store);
    let stats = store.upgrade_index().unwrap();
    assert_eq!(stats.generation, 2);
    assert_eq!(stats.written_tiles, 0);
    assert_eq!(stats.reused_tiles as usize, expected.tiles.len());
    assert!(store.validate_read(&old).is_err());
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    assert_eq!(&head[8..12], &2u32.to_le_bytes());
    assert_eq!(store.upgrade_index().unwrap().written_bytes, 0);
    assert_eq!(store.save(&expected).unwrap().written_bytes, 0);
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    let (read, updated) = changed(&store);
    let selection = expected
        .select(&read.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    expected.writeback(&selection, &updated).unwrap();
    assert_eq!(
        store
            .writeback(&read, &updated, read_limits(129, 4))
            .unwrap()
            .generation,
        3
    );
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    let before = fs::read_dir(&directory.0).unwrap().count();
    assert!(store.collect_unreferenced().unwrap() > 0);
    assert!(fs::read_dir(&directory.0).unwrap().count() < before);
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(store.manifest.as_ref().unwrap().tree.is_some());
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    let empty = MpmSoilTileArchive::from_bytes(&metadata(&expected), 0).unwrap();
    store.save(&empty).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(0));
    assert!(store.tiles().unwrap().next().is_none());
    store.collect_unreferenced().unwrap();
    assert!(!fs::read_dir(&directory.0).unwrap().any(|entry| {
        entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|e| e == "index")
    }));
    drop(store);
    assert_eq!(
        MpmSoilStore::open(&directory.0, limits())
            .unwrap()
            .load()
            .unwrap()
            .unwrap()
            .count,
        0
    );
}

#[test]
fn incremental_root_and_referenced_page_corruption_reject_on_open() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    store.upgrade_index().unwrap();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let root = store
        .manifest
        .as_ref()
        .unwrap()
        .tree
        .as_ref()
        .unwrap()
        .root
        .clone()
        .unwrap();
    let path = tree_manifest::page_path(&directory.0, &root.hash);
    let page = fs::read(&path).unwrap();
    drop(store);
    for mutation in 0..6 {
        let mut bad = head.clone();
        match mutation {
            0 => bad[12] ^= 1,
            1 => bad[44..52].fill(0),
            2 => bad[52..56].copy_from_slice(&u32::MAX.to_le_bytes()),
            3 => bad[64..68].copy_from_slice(&2u32.to_le_bytes()),
            4 => bad[60..64].fill(0),
            _ => bad.push(0),
        }
        if mutation != 0 {
            let digest = blake3::hash(&bad[HEADER_BYTES..]);
            bad[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        }
        fs::write(directory.0.join("HEAD"), bad).unwrap();
        assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    }
    fs::write(directory.0.join("HEAD"), &head).unwrap();
    for bytes in [
        vec![0; index::PAGE_BYTES + 1],
        page[..page.len() - 1].to_vec(),
    ] {
        fs::write(&path, bytes).unwrap();
        assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    }
    fs::write(&path, &page).unwrap();
    assert!(MpmSoilFrameStore::open(&directory.0, limits()).is_err());
    assert!(MpmSoilCanonicalStore::open(&directory.0, limits()).is_err());
    assert!(
        MpmSoilStore::open(
            &directory.0,
            MpmSoilStoreLimits {
                tiles: 0,
                ..limits()
            }
        )
        .is_err()
    );
    let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        archive().to_bytes()
    );
}

#[test]
fn incremental_publication_failures_poison_and_reopen_at_each_barrier() {
    for upgrade in [true, false] {
        for stage in [
            Stage::ChunksSynced,
            Stage::ManifestSynced,
            Stage::Published,
            Stage::DirectorySynced,
        ] {
            let directory = Directory::new();
            let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
            let mut expected = archive();
            store.save(&expected).unwrap();
            if !upgrade {
                store.upgrade_index().unwrap();
            }
            let old = expected.to_bytes();
            let generation = store.generation().unwrap();
            let (read, state) = changed(&store);
            if !upgrade {
                let selection = expected
                    .select(&read.tiles().collect::<Vec<_>>(), 129)
                    .unwrap();
                expected.writeback(&selection, &state).unwrap();
            }
            let barrier = |point| {
                if point == stage {
                    Err(invalid("injected incremental barrier failure"))
                } else {
                    Ok(())
                }
            };
            let result = if upgrade {
                store.upgrade_index_with(barrier)
            } else {
                store.writeback_with(&read, &state, read_limits(129, 4), barrier)
            };
            assert!(result.is_err());
            assert!(store.storage_stats().is_err());
            drop(store);
            let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
            let published = matches!(stage, Stage::Published | Stage::DirectorySynced);
            assert_eq!(
                reopened.generation(),
                Some(generation + u64::from(published))
            );
            assert_eq!(
                reopened.load().unwrap().unwrap().to_bytes(),
                if published { expected.to_bytes() } else { old }
            );
            reopened.collect_unreferenced().unwrap();
        }
    }
}

#[test]
#[ignore = "subprocess incremental publication crash fixture"]
fn incremental_crash_writer() {
    let root = std::env::var_os("MOLLA_INDEX_CRASH_ROOT").unwrap();
    let stage: usize = std::env::var("MOLLA_INDEX_CRASH_STAGE")
        .unwrap()
        .parse()
        .unwrap();
    let upgrade = std::env::var("MOLLA_INDEX_CRASH_UPGRADE").unwrap() == "1";
    let target = [
        Stage::ChunksSynced,
        Stage::ManifestSynced,
        Stage::Published,
        Stage::DirectorySynced,
    ][stage];
    let mut store = MpmSoilStore::open(root, limits()).unwrap();
    let barrier = |point| {
        if point == target {
            std::process::exit(23);
        }
        Ok(())
    };
    if upgrade {
        store.upgrade_index_with(barrier).unwrap();
    } else {
        let (read, state) = changed(&store);
        store
            .writeback_with(&read, &state, read_limits(129, 4), barrier)
            .unwrap();
    }
    panic!("incremental crash barrier not reached");
}

#[test]
fn incremental_upgrade_and_edit_survive_process_exit_at_each_barrier() {
    for upgrade in [true, false] {
        for stage in 0..4 {
            let directory = Directory::new();
            let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
            let mut expected = archive();
            store.save(&expected).unwrap();
            if !upgrade {
                store.upgrade_index().unwrap();
            }
            let old = expected.to_bytes();
            let generation = store.generation().unwrap();
            if !upgrade {
                let (read, state) = changed(&store);
                let selection = expected
                    .select(&read.tiles().collect::<Vec<_>>(), 129)
                    .unwrap();
                expected.writeback(&selection, &state).unwrap();
            }
            drop(store);
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "mpm::checkpoint::tiles::store::incremental_tests::incremental_crash_writer",
                    "--ignored",
                    "--nocapture",
                ])
                .env("MOLLA_INDEX_CRASH_ROOT", &directory.0)
                .env("MOLLA_INDEX_CRASH_STAGE", stage.to_string())
                .env("MOLLA_INDEX_CRASH_UPGRADE", if upgrade { "1" } else { "0" })
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(23),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
            assert_eq!(store.generation(), Some(generation + u64::from(stage >= 2)));
            assert_eq!(
                store.load().unwrap().unwrap().to_bytes(),
                if stage >= 2 { expected.to_bytes() } else { old }
            );
            store.collect_unreferenced().unwrap();
        }
    }
}
