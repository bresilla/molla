use super::super::tests::{Directory, limits};
use super::*;
use crate::mpm::checkpoint::tests::fixture;

pub(super) fn archive(count: u32) -> MpmSoilFrameTileArchive {
    let mut state = fixture(count);
    state.cfg.grid_size = 64;
    state.cfg.cell_size = 0.125;
    state.cfg.grid_origin = [-4.0; 3];
    for (i, p) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        p[0] = ((i % 4) as f32 - 0.875).to_bits();
        p[1] = 2.0f32.to_bits();
        p[2] = 2.0f32.to_bits();
    }
    MpmSoilFrameCheckpoint::new(state, [1e12; 3])
        .unwrap()
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
            &(0..count)
                .map(|i| SoilParticleId(u64::MAX - u64::from(i)))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(0),
                SoilMaterialId(1 << 50),
                SoilMaterialId(u64::MAX),
            ],
        )
        .unwrap()
}

pub(super) fn read_limits(tiles: u32) -> MpmSoilStoreReadLimits {
    MpmSoilStoreReadLimits {
        tiles,
        particles: 129,
        encoded_bytes: u64::MAX,
    }
}

pub(super) fn moved(read: &MpmSoilFrameStoreRead, delta: f32) -> MpmSoilFrameCheckpoint {
    let mut soil = read.soil().clone();
    for p in soil.words[..soil.count as usize * 4].chunks_exact_mut(4) {
        p[0] = (f32::from_bits(p[0]) + delta).to_bits();
    }
    MpmSoilFrameCheckpoint::new(soil, read.origin).unwrap()
}

#[test]
fn framed_incremental_upgrade_writeback_and_reopen_preserve_exact_archive() {
    let dir = Directory::new();
    let mut expected = archive(129);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&expected).unwrap();
    assert_eq!(store.upgrade_index().unwrap().generation, 2);
    let keys: Vec<_> = expected.tiles().collect();
    let read = store.read_tiles(&keys, read_limits(129)).unwrap().unwrap();
    let updated = moved(&read, 1.0);
    let reference = expected.select(&keys, 129).unwrap();
    expected.writeback(&reference, &updated).unwrap();
    assert_eq!(
        store
            .writeback(&read, &updated, read_limits(129))
            .unwrap()
            .generation,
        3
    );
    store.collect_unreferenced().unwrap();
    drop(store);
    let mut reopened = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert_eq!(reopened.upgrade_index().unwrap().written_bytes, 0);
}

#[test]
fn framed_store_roundtrip_reuse_bounded_reads_and_frame_generations() {
    let dir = Directory::new();
    let initial = archive(129);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), None);
    assert_eq!(store.storage_stats().unwrap().generation, None);
    assert_eq!(store.world_origin().unwrap(), None);
    assert!(store.load().unwrap().is_none());
    assert!(store.read_tiles(&[], read_limits(0)).unwrap().is_none());
    assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_err());
    let first = store.save(&initial).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    assert_eq!(store.storage_stats().unwrap().archived_particles, 129);
    assert_eq!((first.generation, first.written_tiles), (1, 4));
    let same = store.save(&initial).unwrap();
    assert_eq!(
        (same.generation, same.written_tiles, same.written_bytes),
        (1, 0, 0)
    );
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let read = store
        .read_tiles(&[keys[0], keys[0]], read_limits(1))
        .unwrap()
        .unwrap();
    assert_eq!(read.stats().tiles, 1);
    assert_eq!(read.world_origin(), initial.world_origin());
    assert_eq!(store.layout().unwrap(), Some(initial.layout()));
    assert_eq!(
        store.tile_particle_count(keys[0]).unwrap(),
        Some(read.soil().count)
    );
    assert_eq!(read.tiles().collect::<Vec<_>>(), vec![keys[0]]);
    let selected = initial.select(&[keys[0]], 129).unwrap();
    assert_eq!(
        read.selection_snapshot()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes(),
        selected.to_frame_checkpoint().unwrap().to_bytes()
    );
    assert_eq!(read.particle_ids(), selected.particle_ids());
    assert_eq!(read.material_ids(), selected.material_ids());
    assert!(store.read_tiles(&keys, read_limits(1)).is_err());
    assert!(
        store
            .read_tiles(
                &keys[..1],
                MpmSoilStoreReadLimits {
                    particles: 0,
                    ..read_limits(1)
                }
            )
            .is_err()
    );
    assert!(
        store
            .read_tiles(
                &keys[..1],
                MpmSoilStoreReadLimits {
                    encoded_bytes: 0,
                    ..read_limits(1)
                }
            )
            .is_err()
    );
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        initial.to_bytes()
    );
    let rebased = initial.recentered([-8.0; 3]).unwrap();
    assert_eq!(store.save(&rebased).unwrap().generation, 2);
    assert!(store.validate_read(&read).is_err());
    assert_eq!(store.world_origin().unwrap(), Some([1e12 + 4.0; 3]));
    drop(store);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        rebased.to_bytes()
    );
    assert!(store.validate_read(&read).is_err());
    assert_eq!(store.save(&rebased).unwrap().written_bytes, 0);
    assert!(store.collect_unreferenced().unwrap() > 0);
    assert_eq!(store.collect_unreferenced().unwrap(), 0);
}

#[test]
fn framed_store_selected_moves_preserve_unloaded_tiles_and_identity() {
    let dir = Directory::new();
    let initial = archive(129);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let keys: Vec<_> = initial.tiles().collect();
    let read = store
        .read_tiles(&keys[..1], read_limits(1))
        .unwrap()
        .unwrap();
    let update = moved(&read, 1.0);
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    assert!(store.writeback(&read, &update, read_limits(1)).is_err());
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    store.validate_read(&read).unwrap();
    let mut expected = MpmSoilFrameTileArchive::from_bytes(&initial.to_bytes(), 129).unwrap();
    let selection = expected.select(&keys[..1], 129).unwrap();
    expected.writeback(&selection, &update).unwrap();
    let unaffected = store
        .inner
        .manifest
        .as_ref()
        .unwrap()
        .get(keys[3])
        .unwrap()
        .unwrap()
        .hash;
    let stats = store.writeback(&read, &update, read_limits(2)).unwrap();
    assert_eq!(stats.generation, 2);
    assert_eq!(
        store
            .inner
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[3])
            .unwrap()
            .unwrap()
            .hash,
        unaffected
    );
    assert!(store.validate_read(&read).is_err());
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    drop(store);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}

#[test]
fn framed_store_rejects_wrong_family_frame_mass_and_foreign_reads_without_publication() {
    let dir = Directory::new();
    let initial = archive(129);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let keys: Vec<_> = initial.tiles().collect();
    let read = store
        .read_tiles(&keys[..1], read_limits(1))
        .unwrap()
        .unwrap();
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    let changed = MpmSoilFrameCheckpoint::new(read.soil().clone(), [0.0; 3]).unwrap();
    assert!(store.writeback(&read, &changed, read_limits(4)).is_err());
    let mut soil = read.soil().clone();
    let mass = soil.count as usize * STRIDES[..4].iter().sum::<usize>();
    soil.words[mass] = (f32::from_bits(soil.words[mass]) * 2.0).to_bits();
    let changed = MpmSoilFrameCheckpoint::new(soil, read.origin).unwrap();
    assert!(store.writeback(&read, &changed, read_limits(4)).is_err());
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    store.validate_read(&read).unwrap();
    let other_dir = Directory::new();
    let mut other = MpmSoilFrameStore::open(&other_dir.0, limits()).unwrap();
    other.save(&initial).unwrap();
    assert!(other.validate_read(&read).is_err());
    drop(store);
    assert!(MpmSoilStore::open(&dir.0, limits()).is_err());
    let raw_dir = Directory::new();
    let mut raw = MpmSoilStore::open(&raw_dir.0, limits()).unwrap();
    let selected = initial.select(&keys, 129).unwrap();
    let unframed = selected
        .soil()
        .partition_tiles(
            initial.layout(),
            selected.particle_ids(),
            selected.material_ids(),
        )
        .unwrap();
    raw.save(&unframed).unwrap();
    drop(raw);
    assert!(MpmSoilFrameStore::open(&raw_dir.0, limits()).is_err());
    assert!(
        MpmSoilFrameStore::open(
            &dir.0,
            MpmSoilStoreLimits {
                particles: 128,
                tiles: 129
            }
        )
        .is_err()
    );
    assert!(
        MpmSoilFrameStore::open(
            &dir.0,
            MpmSoilStoreLimits {
                particles: 129,
                tiles: 3
            }
        )
        .is_err()
    );
    assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_ok());
}

#[test]
fn framed_store_corrupt_selected_data_rejects_without_reading_unselected_blobs() {
    let dir = Directory::new();
    let initial = archive(129);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let keys: Vec<_> = initial.tiles().collect();
    let entry = &store
        .inner
        .manifest
        .as_ref()
        .unwrap()
        .get(keys[3])
        .unwrap()
        .unwrap();
    let path = store.inner.blob_path(&entry.hash);
    fs::write(path, b"broken framed tile").unwrap();
    assert!(
        store
            .read_tiles(&keys[..1], read_limits(1))
            .unwrap()
            .is_some()
    );
    assert!(store.read_tiles(&keys[3..], read_limits(1)).is_err());
    assert!(store.load().is_err());
    assert!(store.collect_unreferenced().is_err());
}

#[test]
fn framed_store_publication_failures_reopen_complete_old_or_new_generations() {
    for partial in [false, true] {
        for stage in [
            Stage::ChunksSynced,
            Stage::ManifestSynced,
            Stage::Published,
            Stage::DirectorySynced,
        ] {
            let dir = Directory::new();
            let initial = archive(129);
            let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
            store.save(&initial).unwrap();
            let keys: Vec<_> = initial.tiles().collect();
            let read = store
                .read_tiles(&keys[..1], read_limits(1))
                .unwrap()
                .unwrap();
            let update = moved(&read, 1.0);
            let mut expected =
                MpmSoilFrameTileArchive::from_bytes(&initial.to_bytes(), 129).unwrap();
            if partial {
                let selection = expected.select(&keys[..1], 129).unwrap();
                expected.writeback(&selection, &update).unwrap();
            } else {
                expected = initial.recentered([-8.0; 3]).unwrap();
            }
            let barrier = |at| {
                if at == stage {
                    Err(invalid("injected framed publication failure"))
                } else {
                    Ok(())
                }
            };
            let result = if partial {
                store
                    .inner
                    .writeback_with(&read.read, update.soil(), read_limits(2), barrier)
            } else {
                store
                    .inner
                    .save_with_frame(&expected.archive, Some(expected.origin), barrier)
            };
            assert!(result.is_err());
            assert!(store.load().is_err());
            assert!(store.world_origin().is_err());
            drop(store);
            let mut reopened = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
            let published = matches!(stage, Stage::Published | Stage::DirectorySynced);
            assert_eq!(reopened.generation(), Some(if published { 2 } else { 1 }));
            assert_eq!(
                reopened.load().unwrap().unwrap().to_bytes(),
                if published {
                    expected.to_bytes()
                } else {
                    initial.to_bytes()
                }
            );
        }
    }
}

#[test]
fn framed_store_empty_archive_and_read_types_are_supported() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSoilFrameStore>();
    send_sync::<MpmSoilFrameStoreRead>();
    let dir = Directory::new();
    let initial = archive(0);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let read = store.read_tiles(&[], read_limits(0)).unwrap().unwrap();
    let updated = read.selection_snapshot().to_frame_checkpoint().unwrap();
    assert_eq!(
        store
            .writeback(&read, &updated, read_limits(0))
            .unwrap()
            .written_bytes,
        0
    );
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        initial.to_bytes()
    );
}

#[test]
#[ignore = "requires GPU; framed durable reload restores physical state and device IDs"]
fn framed_store_gpu_capture_save_reopen_restore_and_writeback() {
    gpu_roundtrip(false);
}

#[test]
#[ignore = "requires GPU; background framed reload and resumed writeback"]
fn framed_store_gpu_background_capture_reload_resume_writeback() {
    gpu_roundtrip(true);
}

fn gpu_roundtrip(background: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let initial = archive(129);
    let selected = initial
        .select(&initial.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    let sim = MpmSimulation::new(&engine, selected.soil().config()).unwrap();
    let mut live = MpmSoilFrame::new(sim, [0.0; 3]).unwrap();
    live.restore_selection(&engine, &selected).unwrap();
    live.step_n_with(device, queue, 0.0001, 2).unwrap();
    let evolved = live.checkpoint_tiles_with(queue, initial.layout()).unwrap();
    let dir = Directory::new();
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&evolved).unwrap();
    drop(store);
    drop(live);
    let store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let (mut store, read) = if background {
        let mut task = store.begin_read(&keys, read_limits(129));
        let completion = tasks::tests::finish_read(&mut task);
        (completion.store, completion.result.unwrap().unwrap())
    } else {
        let read = store.read_tiles(&keys, read_limits(129)).unwrap().unwrap();
        (store, read)
    };
    let snapshot = read.selection_snapshot();
    let sim = MpmSimulation::new(&engine, snapshot.soil().config()).unwrap();
    let mut restored = MpmSoilFrame::new(sim, [0.0; 3]).unwrap();
    restored.restore_selection(&engine, &snapshot).unwrap();
    assert_eq!(
        restored
            .checkpoint_tiles_with(queue, initial.layout())
            .unwrap()
            .to_bytes(),
        evolved.to_bytes()
    );
    let view = restored.simulation().soil_identity_buffers().unwrap();
    let words = crate::mpm::replay_tests::read_words(device, queue, view.particles);
    let expected: Vec<u32> = read
        .particle_ids()
        .iter()
        .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
        .collect();
    assert_eq!(&words[..expected.len()], expected);
    restored.step_n_with(device, queue, 0.0001, 2).unwrap();
    let updated = restored.checkpoint_with(queue).unwrap();
    let expected = restored
        .checkpoint_tiles_with(queue, initial.layout())
        .unwrap();
    if background {
        let mut task = store.begin_writeback(read, updated, read_limits(129));
        let completion = tasks::tests::finish_write(&mut task);
        completion.result.unwrap();
        store = completion.store;
    } else {
        store.writeback(&read, &updated, read_limits(129)).unwrap();
    }
    drop(store);
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}
