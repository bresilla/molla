use super::*;

pub(super) fn rewrite_manifest(
    store: &mut MpmSoilStore,
    edit: impl FnOnce(&mut Manifest),
) -> Vec<u8> {
    let path = store.root.join("HEAD");
    let mut manifest = Manifest::decode_family(
        &fs::read(&path).unwrap(),
        store.limits,
        store.framed,
        store.canonical,
    )
    .unwrap();
    edit(&mut manifest);
    manifest.particles = manifest.entries.values().map(|entry| entry.count).sum();
    let bytes = manifest.encode();
    fs::write(&path, &bytes).unwrap();
    store.manifest = Some(
        manifest::ManifestFile::open(&path, store.limits, store.framed, store.canonical).unwrap(),
    );
    bytes
}
use crate::mpm::checkpoint::tests::fixture;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub(super) struct Directory(pub(super) PathBuf);

impl Directory {
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("MOLLA_STORE_TEST_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let root = base.join(format!(
            "molla-soil-store-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

pub(super) fn limits() -> MpmSoilStoreLimits {
    MpmSoilStoreLimits {
        particles: 129,
        tiles: 129,
    }
}

pub(super) fn archive() -> MpmSoilTileArchive {
    let mut state = fixture(129);
    for (i, p) in state.words[..129 * 4].chunks_exact_mut(4).enumerate() {
        p[0] = ((i % 4) as f32 * 0.125 + 0.02).to_bits();
    }
    state
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 0.125).unwrap(),
            &(0..129)
                .map(|i| SoilParticleId(u64::MAX - i))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(0),
                SoilMaterialId(1 << 50),
                SoilMaterialId(u64::MAX),
            ],
        )
        .unwrap()
}

fn edit_one(archive: &mut MpmSoilTileArchive, move_tile: bool) {
    let key = archive.tiles().next().unwrap();
    let selection = archive.select(&[key], 129).unwrap();
    let mut checkpoint = selection.checkpoint().clone();
    if move_tile {
        for position in checkpoint.words[..checkpoint.count as usize * 4].chunks_exact_mut(4) {
            position[0] = (f32::from_bits(position[0]) + 2.0).to_bits();
        }
    } else {
        checkpoint.words[checkpoint.count as usize * 4] = 0.125_f32.to_bits();
    }
    archive.writeback(&selection, &checkpoint).unwrap();
}

#[test]
fn file_index_cache_stays_bounded_across_full_save_load_and_selected_migration() {
    let directory = Directory::new();
    let count = EPOCH_CACHE_TILES as u32 + 1;
    let mut state = fixture(count);
    state.cfg.max_particles = count;
    for (i, position) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        position[0] = (i as f32 + 0.125).to_bits();
    }
    let mut archive = state
        .partition_tiles(
            SoilTileLayout::new([0.; 3], 1.).unwrap(),
            &(0..count)
                .map(|id| SoilParticleId(u64::from(id)))
                .collect::<Vec<_>>(),
            &[SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)],
        )
        .unwrap();
    let bounds = MpmSoilStoreLimits {
        particles: count,
        tiles: count + 1,
    };
    let mut store = MpmSoilStore::open(&directory.0, bounds).unwrap();
    let initial = store.save(&archive).unwrap();
    assert_eq!(initial.written_tiles, count);
    assert_eq!(store.epochs.len(), EPOCH_CACHE_TILES);
    let repeated = store.save(&archive).unwrap();
    assert_eq!(repeated.generation, initial.generation);
    assert_eq!(repeated.written_bytes, 0);
    assert_eq!(repeated.reused_tiles, count);
    archive = store.load().unwrap().unwrap();
    assert_eq!(store.epochs.len(), EPOCH_CACHE_TILES);
    let key = archive.tiles().last().unwrap();
    let read = store
        .read_tiles(&[key], read_limits(1, 2))
        .unwrap()
        .unwrap();
    let mut updated = read.selection().checkpoint().clone();
    updated.words[0] = (count as f32 + 1.125).to_bits();
    let expected = archive.select(&[key], 1).unwrap();
    archive.writeback(&expected, &updated).unwrap();
    let stats = store.writeback(&read, &updated, read_limits(1, 2)).unwrap();
    assert_eq!(stats.generation, 2);
    assert_eq!(stats.reused_tiles, count - 1);
    assert_eq!(store.epochs.len(), 1);
    assert_eq!(store.total_particle_count().unwrap(), Some(count));
    drop(store);
    let mut reopened = MpmSoilStore::open(&directory.0, bounds).unwrap();
    assert!(reopened.epochs.is_empty());
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        archive.to_bytes()
    );
    assert_eq!(reopened.epochs.len(), EPOCH_CACHE_TILES);
}

pub(super) fn read_limits(particles: u32, tiles: u32) -> MpmSoilStoreReadLimits {
    MpmSoilStoreReadLimits {
        particles,
        tiles,
        encoded_bytes: u64::from(particles) * 168 + u64::from(tiles) * MAX_METADATA,
    }
}

#[test]
fn committed_particle_total_is_cached_and_generation_bound() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), None);
    let mut initial = archive();
    store.save(&initial).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    edit_one(&mut initial, true);
    store.save(&initial).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    let empty = MpmSoilTileArchive::from_bytes(&metadata(&initial), 0).unwrap();
    assert!(
        store
            .save_with(&empty, |stage| {
                if stage == Stage::ManifestSynced {
                    Err(invalid("injected prepublication failure"))
                } else {
                    Ok(())
                }
            })
            .is_err()
    );
    assert!(store.total_particle_count().is_err());
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    store.save(&empty).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(0));
    drop(store);
    assert_eq!(
        MpmSoilStore::open(&directory.0, limits())
            .unwrap()
            .total_particle_count()
            .unwrap(),
        Some(0)
    );
}

#[test]
fn read_caller_preflight_and_total_have_no_manifest_traversal() {
    let source = include_str!("mpm_soil_store.rs");
    let setup = source
        .split_once("    fn bounded_read_keys(")
        .unwrap()
        .1
        .split_once("    fn existing_read_keys(")
        .unwrap()
        .0;
    for forbidden in [".entries", ".read(", "tile_particle_count", "selected_keys"] {
        assert!(!setup.contains(forbidden));
    }
    let total = source
        .split_once("    pub fn total_particle_count(")
        .unwrap()
        .1
        .split_once("    fn healthy(")
        .unwrap()
        .0;
    assert!(!total.contains(".entries") && !total.contains(".tiles("));
    let world: String = include_str!("mpm_soil_disk_world.rs")
        .split_whitespace()
        .collect();
    assert!(world.contains("store.total_particle_count()"));
    assert!(!world.contains("store.tiles()?.try_fold"));
    assert!(!world.contains("tile_particle_count"));
}

#[test]
fn selected_read_preserves_history_and_ignores_unselected_blobs() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(store.read_tiles(&[], read_limits(0, 0)).unwrap().is_none());
    let archive = archive();
    store.save(&archive).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    assert!(keys.len() > 1);
    assert_eq!(keys, archive.tiles().collect::<Vec<_>>());
    let key = keys[0];
    let count = archive.tile_particle_count(key).unwrap() as u32;
    let expected = archive.select(&[key], count).unwrap();
    let epochs = store.epochs.clone();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let unselected = store.blob_path(
        &store
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[1])
            .unwrap()
            .unwrap()
            .hash,
    );
    fs::write(&unselected, b"damaged unselected tile").unwrap();
    let read = store
        .read_tiles(&[key, key], read_limits(count, 1))
        .unwrap()
        .unwrap();
    assert_eq!(
        read.selection().checkpoint().to_bytes(),
        expected.checkpoint().to_bytes()
    );
    assert_eq!(read.selection().particle_ids(), expected.particle_ids());
    assert_eq!(read.selection().material_ids(), expected.material_ids());
    let blob = store.blob_path(
        &store
            .manifest
            .as_ref()
            .unwrap()
            .get(key)
            .unwrap()
            .unwrap()
            .hash,
    );
    assert_eq!(
        read.stats(),
        MpmSoilStoreReadStats {
            generation: 1,
            tiles: 1,
            particles: count,
            read_bytes: fs::metadata(blob).unwrap().len()
        }
    );
    assert!(read.stats().read_bytes <= read_limits(count, 1).encoded_bytes);
    store.validate_read(&read).unwrap();
    assert!(
        epochs
            .iter()
            .all(|(key, epoch)| Arc::ptr_eq(epoch, &store.epochs[key]))
    );
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert!(store.load().is_err());
    assert!(
        store
            .read_tiles(&[key, keys[1]], read_limits(129, 129))
            .is_err()
    );
    assert!(
        epochs
            .iter()
            .all(|(key, epoch)| Arc::ptr_eq(epoch, &store.epochs[key]))
    );
}

#[test]
fn selected_read_rejects_bounds_and_unknown_tiles_before_payload_io() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let archive = archive();
    store.save(&archive).unwrap();
    let key = archive.tiles().next().unwrap();
    let count = archive.tile_particle_count(key).unwrap() as u32;
    let entry = &store.manifest.as_ref().unwrap().get(key).unwrap().unwrap();
    fs::remove_file(store.blob_path(&entry.hash)).unwrap();
    for limits in [
        read_limits(count, 0),
        read_limits(count - 1, 1),
        MpmSoilStoreReadLimits {
            encoded_bytes: read_limits(count, 1).encoded_bytes - 1,
            ..read_limits(count, 1)
        },
    ] {
        let error = match store.read_tiles(&[key], limits) {
            Err(e) => e,
            Ok(_) => panic!("bound accepted"),
        };
        assert!(error.to_string().contains("exceed"), "{error}");
    }
    let error = match store.read_tiles(&[key, SoilTileCoord([i64::MAX; 3])], read_limits(129, 129))
    {
        Err(e) => e,
        Ok(_) => panic!("unknown tile accepted"),
    };
    assert!(error.to_string().contains("unknown stored tile"), "{error}");
    let empty = store.read_tiles(&[], read_limits(0, 0)).unwrap().unwrap();
    assert_eq!(
        empty.stats(),
        MpmSoilStoreReadStats {
            generation: 1,
            tiles: 0,
            particles: 0,
            read_bytes: 0
        }
    );
    assert!(empty.selection().particle_ids().is_empty());
    assert_eq!(empty.selection().material_ids(), archive.material_ids());
    assert!(store.read_tiles(&[key], read_limits(count, 1)).is_err());
    assert!(!store.failed);
}

#[test]
fn selected_read_generation_owner_and_failed_handle_are_checked() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let mut archive = archive();
    store.save(&archive).unwrap();
    let key = archive.tiles().next().unwrap();
    let read = store
        .read_tiles(&[key], read_limits(129, 129))
        .unwrap()
        .unwrap();
    store.validate_read(&read).unwrap();
    assert_eq!(store.save(&archive).unwrap().written_bytes, 0);
    store.validate_read(&read).unwrap();
    edit_one(&mut archive, false);
    store.save(&archive).unwrap();
    assert!(store.validate_read(&read).is_err());
    let current = store
        .read_tiles(&[key], read_limits(129, 129))
        .unwrap()
        .unwrap();
    store.validate_read(&current).unwrap();
    store.failed = true;
    assert!(store.validate_read(&current).is_err());
    assert!(store.tiles().is_err());
    assert!(store.read_tiles(&[], read_limits(0, 0)).is_err());
    drop(store);
    let reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(reopened.generation(), Some(current.stats().generation));
    assert!(reopened.validate_read(&current).is_err());
    assert_eq!(
        current.selection().particle_ids(),
        read.selection().particle_ids()
    );
}

#[test]
fn selected_read_checks_duplicate_ids_across_requested_blobs() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let mut archive = archive();
    store.save(&archive).unwrap();
    let keys: Vec<_> = archive.tiles().take(2).collect();
    let duplicate = archive.tiles[&keys[0]].ids[0];
    archive.tiles.get_mut(&keys[1]).unwrap().ids[0] = duplicate;
    let bytes = archive.encode_selection(archive.select(&[keys[1]], 129).unwrap());
    let hash = *blake3::hash(&bytes).as_bytes();
    fs::write(store.blob_path(&hash), bytes).unwrap();
    rewrite_manifest(&mut store, |manifest| {
        manifest.entries.get_mut(&keys[1]).unwrap().hash = hash
    });
    assert!(store.read_tiles(&[keys[1]], read_limits(129, 129)).is_ok());
    let error = match store.read_tiles(&keys, read_limits(129, 129)) {
        Err(e) => e,
        Ok(_) => panic!("duplicate ID accepted"),
    };
    assert!(
        error.to_string().contains("duplicate global particle ID"),
        "{error}"
    );
    assert!(store.load().is_err());
}

#[test]
fn incremental_roundtrip_reuses_untouched_tiles_and_preserves_all_history() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(store.load().unwrap().is_none());
    let mut archive = archive();
    let tiles = archive.tiles.len() as u32;
    let first = store.save(&archive).unwrap();
    assert_eq!(
        (first.generation, first.written_tiles, first.reused_tiles),
        (1, tiles, 0)
    );
    let unchanged = store.save(&archive).unwrap();
    assert_eq!(
        unchanged,
        MpmSoilStoreStats {
            generation: 1,
            reused_tiles: tiles,
            ..Default::default()
        }
    );
    let old_entries = store.manifest.as_ref().unwrap().snapshot().unwrap();
    edit_one(&mut archive, false);
    let changed = store.save(&archive).unwrap();
    assert_eq!(
        (
            changed.generation,
            changed.written_tiles,
            changed.reused_tiles
        ),
        (2, 1, tiles - 1)
    );
    let new_entries = store.manifest.as_ref().unwrap().snapshot().unwrap();
    assert_eq!(
        old_entries
            .iter()
            .filter(|(key, value)| new_entries.get(key) == Some(*value))
            .count(),
        tiles as usize - 1
    );
    assert!(changed.written_bytes < first.written_bytes);
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        archive.to_bytes()
    );
    edit_one(&mut archive, true);
    assert_eq!(store.save(&archive).unwrap().written_tiles, 1);
    let expected = archive.to_bytes();
    drop(store);
    let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let loaded = reopened.load().unwrap().unwrap();
    assert_eq!(loaded.to_bytes(), expected);
    assert_eq!(reopened.save(&loaded).unwrap().written_bytes, 0);
    fs::write(directory.0.join("notes.tile"), b"unmanaged").unwrap();
    assert!(reopened.collect_unreferenced().unwrap() > 0);
    assert_eq!(reopened.collect_unreferenced().unwrap(), 0);
    assert_eq!(
        fs::read(directory.0.join("notes.tile")).unwrap(),
        b"unmanaged"
    );
    assert_eq!(reopened.load().unwrap().unwrap().to_bytes(), expected);
    println!(
        "{tiles} tiles: first save {} bytes; one changed tile {} bytes; unchanged save 0 bytes",
        first.written_bytes, changed.written_bytes
    );
}

#[test]
fn exclusive_lock_limits_empty_archives_and_failed_writes_preserve_head() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    let mut archive = archive();
    store.save(&archive).unwrap();
    let before = fs::read(directory.0.join("HEAD")).unwrap();
    store.limits.particles = 1;
    assert!(store.save(&archive).is_err());
    assert!(!store.failed);
    store.limits = limits();
    edit_one(&mut archive, false);
    fs::create_dir(directory.0.join("pending")).unwrap();
    assert!(store.save(&archive).is_err());
    assert!(store.failed);
    assert!(store.load().is_err());
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), before);
    fs::remove_dir(directory.0.join("pending")).unwrap();
    drop(store);
    assert!(
        MpmSoilStore::open(
            &directory.0,
            MpmSoilStoreLimits {
                particles: 128,
                ..limits()
            }
        )
        .is_err()
    );
    let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let empty = MpmSoilTileArchive::from_bytes(&metadata(&archive), 0).unwrap();
    let saved = reopened.save(&empty).unwrap();
    assert_eq!(saved.written_tiles, 0);
    assert_eq!(saved.generation, 2);
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        empty.to_bytes()
    );
}

#[test]
#[cfg(target_os = "linux")]
fn exclusive_lock_releases_owner_before_inherited_descriptor_closes() {
    let directory = Directory::new();
    let store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let inherited = store._lock.0.try_clone().unwrap();
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    drop(store);
    let reopened = MpmSoilStore::open(&directory.0, limits())
        .expect("owner drop must release the lock while a duplicate descriptor survives");
    drop(inherited);
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    drop(reopened);
    assert!(MpmSoilStore::open(&directory.0, limits()).is_ok());
}

#[test]
#[cfg(target_os = "linux")]
fn exclusive_lock_guard_releases_on_error_and_unwind() {
    for unwind in [false, true] {
        let directory = Directory::new();
        let mut inherited = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(directory.0.join("LOCK"))
                .unwrap();
            let lock = StoreLock::acquire(file)?;
            inherited = Some(lock.0.try_clone().unwrap());
            assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
            assert!(!unwind, "injected construction unwind");
            Err(invalid("injected construction error"))
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        let reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
        drop(inherited);
        assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
        drop(reopened);
        assert!(MpmSoilStore::open(&directory.0, limits()).is_ok());
    }
}

#[test]
fn manifest_and_blob_damage_fail_without_partial_load_or_unbounded_allocation() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let manifest = store.manifest.as_ref().unwrap();
    let good = fs::read(directory.0.join("HEAD")).unwrap();
    let mut damaged = good.clone();
    *damaged.last_mut().unwrap() ^= 1;
    assert!(Manifest::decode(&damaged, limits()).is_err());
    assert!(
        Manifest::decode(
            &good,
            MpmSoilStoreLimits {
                tiles: 0,
                ..limits()
            }
        )
        .is_err()
    );
    for cut in [0, 8, 43, 59, good.len() - 1] {
        assert!(Manifest::decode(&good[..cut], limits()).is_err());
    }
    let hash = manifest.iter().next().unwrap().unwrap().1.hash;
    let path = store.blob_path(&hash);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, &bytes).unwrap();
    assert!(store.load().is_err());
    assert!(store.collect_unreferenced().is_err());
    fs::remove_file(&path).unwrap();
    assert!(store.load().is_err());
    File::create(&path).unwrap().set_len(1 << 30).unwrap();
    assert!(store.load().is_err());
    drop(store);
    fs::write(directory.0.join("HEAD"), damaged).unwrap();
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
}

#[test]
fn inconsistent_chunk_coordinates_and_duplicate_global_ids_reject() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let first = fixture(1);
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let mids = [SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)];
    let a = first
        .partition_tiles(layout, &[SoilParticleId(9)], &mids)
        .unwrap();
    store.save(&a).unwrap();
    let mut second = first.clone();
    second.words[0] = (f32::from_bits(first.words[0]) + 1.0).to_bits();
    let b = second
        .partition_tiles(layout, &[SoilParticleId(9)], &mids)
        .unwrap();
    let bytes = b.to_bytes();
    let hash = *blake3::hash(&bytes).as_bytes();
    fs::write(store.blob_path(&hash), &bytes).unwrap();
    let key = b.tiles().next().unwrap();
    rewrite_manifest(&mut store, |manifest| {
        manifest.entries.insert(key, Entry { count: 1, hash });
    });
    assert!(
        store
            .load()
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate global")
    );
    rewrite_manifest(&mut store, |manifest| {
        manifest.entries.remove(&key);
        manifest
            .entries
            .insert(SoilTileCoord([100; 3]), Entry { count: 1, hash });
    });
    assert!(store.load().is_err());
}

#[test]
#[ignore = "subprocess crash fixture; invoked by durable_publication_survives_process_exit"]
fn crash_writer() {
    let root = std::env::var_os("MOLLA_STORE_CRASH_ROOT").expect("crash fixture root");
    let stage: usize = std::env::var("MOLLA_STORE_CRASH_STAGE")
        .unwrap()
        .parse()
        .unwrap();
    let target = [
        Stage::ChunksSynced,
        Stage::ManifestSynced,
        Stage::Published,
        Stage::DirectorySynced,
    ][stage];
    let mut store = MpmSoilStore::open(root, limits()).unwrap();
    let mut archive = store.load().unwrap().unwrap();
    edit_one(&mut archive, false);
    store
        .save_with(&archive, |point| {
            if point == target {
                std::process::exit(23);
            }
            Ok(())
        })
        .unwrap();
    panic!("crash barrier not reached");
}

#[test]
fn durable_publication_survives_process_exit() {
    for stage in 0..4 {
        let directory = Directory::new();
        let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
        let mut archive = archive();
        store.save(&archive).unwrap();
        let old = archive.to_bytes();
        edit_one(&mut archive, false);
        let new = archive.to_bytes();
        drop(store);
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "mpm::checkpoint::tiles::store::tests::crash_writer",
                "--ignored",
                "--nocapture",
            ])
            .env("MOLLA_STORE_CRASH_ROOT", &directory.0)
            .env("MOLLA_STORE_CRASH_STAGE", stage.to_string())
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(23),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
        let bytes = reopened.load().unwrap().unwrap().to_bytes();
        assert_eq!(bytes, if stage < 2 { old } else { new });
        assert_eq!(reopened.generation(), Some(if stage < 2 { 1 } else { 2 }));
    }
}

fn synchronize(world: &mut MpmSoilWorld, engine: &Engine) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !world.poll_region_update(engine).unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn live_gpu_boundary_save_reload_and_resume_preserve_history_and_freshness() {
    use molla_vehicle::{SoilRegionId, SoilRegionRequest, SoilResidencyLimits};
    let engine = Engine::new_gpu().unwrap();
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let archive = archive();
    let keys: Vec<_> = archive.tiles().collect();
    let cfg = archive.cfg;
    let residency = SoilResidencyLimits {
        regions: 2,
        tiles: 64,
        claims: 128,
    };
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, archive, residency, 20656).unwrap();
    let region = [SoilRegionRequest {
        id: SoilRegionId(3),
        tiles: &keys,
    }];
    world.begin_region_update(&region).unwrap();
    assert!(world.save_synchronized_archive(&mut store).is_err());
    assert_eq!(store.generation(), None);
    synchronize(&mut world, &engine);
    world.save_synchronized_archive(&mut store).unwrap();
    world.step_n(&engine, 0.0001, 2).unwrap();
    assert!(world.save_synchronized_archive(&mut store).is_err());
    assert_eq!(store.generation(), Some(1));
    world.begin_region_update(&region).unwrap();
    synchronize(&mut world, &engine);
    world.save_synchronized_archive(&mut store).unwrap();
    let expected = world.synchronized_archive_bytes().unwrap();
    drop(store);
    drop(world);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let loaded = store.load().unwrap().unwrap();
    assert_eq!(loaded.to_bytes(), expected);
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(&engine, sim, loaded, residency, 20656).unwrap();
    world.begin_region_update(&region).unwrap();
    synchronize(&mut world, &engine);
    assert_eq!(world.synchronized_archive_bytes().unwrap(), expected);
    assert_eq!(
        world
            .save_synchronized_archive(&mut store)
            .unwrap()
            .written_bytes,
        0
    );
    world.step_n(&engine, 0.0001, 2).unwrap();
    world.begin_region_update(&region).unwrap();
    synchronize(&mut world, &engine);
    assert_eq!(
        world
            .save_synchronized_archive(&mut store)
            .unwrap()
            .generation,
        3
    );
    assert_eq!(world.active_particle_count(), 129);
}
