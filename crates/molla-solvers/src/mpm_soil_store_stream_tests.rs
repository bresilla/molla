use super::tests::{Directory, archive, limits};
use super::*;
use crate::mpm::checkpoint::tests::fixture;
use std::time::{Duration, Instant};

fn wait<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = poll().unwrap() {
            return value;
        }
        assert!(Instant::now() < deadline, "storage worker deadline");
        std::thread::sleep(Duration::from_micros(100));
    }
}

#[test]
fn storage_accounting_covers_uncommitted_committed_reopen_and_failed_handles() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let empty = store.storage_stats().unwrap();
    assert_eq!(empty.generation, None);
    assert_eq!(empty.archived_tiles, 0);
    assert_eq!(empty.archived_particles, 0);
    assert_eq!(empty.manifest_encoded_bytes, 0);
    assert_eq!(empty.manifest_buffer_bytes, 0);
    assert_eq!(empty.cached_tiles, 0);
    assert_eq!(empty.cache_tile_limit, EPOCH_CACHE_TILES as u32);
    let archive = archive();
    store.save(&archive).unwrap();
    let committed = store.storage_stats().unwrap();
    assert_eq!(committed.generation, Some(1));
    assert_eq!(committed.archived_tiles as usize, archive.tiles.len());
    assert_eq!(committed.archived_particles, 129);
    assert_eq!(
        committed.manifest_encoded_bytes,
        fs::metadata(directory.0.join("HEAD")).unwrap().len()
    );
    assert!(committed.manifest_buffer_bytes <= 8192 + MAX_METADATA);
    assert_eq!(committed.cached_tiles, committed.archived_tiles);
    store.failed = true;
    assert!(store.storage_stats().is_err());
    drop(store);
    let reopened = MpmSoilStore::open(&directory.0, limits())
        .unwrap()
        .storage_stats()
        .unwrap();
    assert_eq!(reopened.cached_tiles, 0);
    assert_eq!(
        reopened.manifest_buffer_bytes,
        committed.manifest_buffer_bytes
    );
    assert_eq!(reopened.archived_particles, committed.archived_particles);
    let source = include_str!("mpm_soil_store.rs")
        .split_once("    pub fn storage_stats(")
        .unwrap()
        .1
        .split_once("    fn healthy(")
        .unwrap()
        .0;
    for forbidden in [".lock(", ".get(", "fs::", ".iter(", ".read("] {
        assert!(!source.contains(forbidden));
    }
}

#[test]
fn storage_fixed_working_set_traverse_preserves_history_and_bounds_index_cache() {
    const ACTIVE: u32 = 4;
    let mut retained = None;
    for count in [64, 1024, 4096] {
        let directory = Directory::new();
        let mut state = fixture(count);
        state.cfg.max_particles = count;
        for (i, position) in state.words[..count as usize * 4]
            .chunks_exact_mut(4)
            .enumerate()
        {
            position[0] = (i as f32 + 0.125).to_bits();
        }
        let mut expected = state
            .partition_tiles(
                SoilTileLayout::new([0.; 3], 1.).unwrap(),
                &(0..count)
                    .map(|id| SoilParticleId(u64::from(id)))
                    .collect::<Vec<_>>(),
                &[SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)],
            )
            .unwrap();
        let limits = MpmSoilStoreLimits {
            tiles: count,
            particles: count,
        };
        let bounds = MpmSoilStoreReadLimits {
            tiles: ACTIVE,
            particles: ACTIVE,
            encoded_bytes: u64::from(ACTIVE) * (MAX_METADATA + 168),
        };
        let mut store = MpmSoilStore::open(&directory.0, limits).unwrap();
        store.save(&expected).unwrap();
        drop(store);
        store = MpmSoilStore::open(&directory.0, limits).unwrap();
        let baseline = store.storage_stats().unwrap();
        assert_eq!(
            *retained.get_or_insert(baseline.manifest_buffer_bytes),
            baseline.manifest_buffer_bytes
        );
        assert_eq!(baseline.cached_tiles, 0);
        let keys: Vec<_> = expected.tiles().collect();
        let mut written = 0u64;
        let mut peak_read = 0u64;
        let mut peak_cache = 0;
        let started = Instant::now();
        for step in 0..count / ACTIVE {
            let start = ((step * 13) % (count / ACTIVE) * ACTIVE) as usize;
            let selected = &keys[start..start + ACTIVE as usize];
            let mut task = store.begin_read(selected, bounds);
            let completion = wait(|| task.poll());
            store = completion.store;
            let read = completion.result.unwrap().unwrap();
            assert_eq!(read.stats().particles, ACTIVE);
            assert_eq!(read.stats().tiles, ACTIVE);
            assert!(read.stats().read_bytes <= bounds.encoded_bytes);
            peak_read = peak_read.max(read.stats().read_bytes);
            let reference = expected.select(selected, ACTIVE).unwrap();
            assert_eq!(
                read.selection().checkpoint().words,
                reference.checkpoint().words
            );
            assert_eq!(read.selection().particle_ids(), reference.particle_ids());
            let mut updated = read.selection().checkpoint().clone();
            updated.words[ACTIVE as usize * 4] = (0.25 + step as f32 * 0.0001).to_bits();
            expected.writeback(&reference, &updated).unwrap();
            let mut task = store.begin_writeback(read, updated, bounds);
            let completion = wait(|| task.poll());
            store = completion.store;
            let stats = completion.result.unwrap();
            written += stats.written_bytes;
            assert_eq!(stats.written_tiles + stats.reused_tiles, count);
            assert_eq!(stats.written_tiles, 1);
            let memory = store.storage_stats().unwrap();
            assert_eq!(memory.manifest_buffer_bytes, baseline.manifest_buffer_bytes);
            assert_eq!(memory.archived_particles, count);
            assert_eq!(memory.archived_tiles, count);
            assert_eq!(
                memory.manifest_encoded_bytes,
                baseline.manifest_encoded_bytes
            );
            assert!(memory.cached_tiles <= ACTIVE);
            peak_cache = peak_cache.max(memory.cached_tiles);
            if step % 127 == 126 {
                drop(store);
                store = MpmSoilStore::open(&directory.0, limits).unwrap();
                assert_eq!(store.storage_stats().unwrap().cached_tiles, 0);
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        drop(store);
        let mut reopened = MpmSoilStore::open(&directory.0, limits).unwrap();
        assert_eq!(
            reopened.load().unwrap().unwrap().to_bytes(),
            expected.to_bytes()
        );
        assert!(reopened.storage_stats().unwrap().cached_tiles <= EPOCH_CACHE_TILES as u32);
        println!(
            "STORAGE_TRAVERSE {{\"field_tiles\":{count},\"steps\":{},\"active_particles\":{ACTIVE},\"manifest_buffer_bytes\":{},\"manifest_encoded_bytes\":{},\"peak_cached_tiles\":{peak_cache},\"peak_selected_read_bytes\":{peak_read},\"written_bytes\":{written},\"elapsed_seconds\":{elapsed}}}",
            count / ACTIVE,
            baseline.manifest_buffer_bytes,
            baseline.manifest_encoded_bytes
        );
    }
}
