use super::super::tests::{Directory, limits};
use super::*;
use crate::mpm::checkpoint::tests::{fixture, new_sim};

pub(super) fn state(count: u32) -> MpmSoilFrameCheckpoint {
    let mut soil = fixture(count);
    soil.cfg.grid_size = 64;
    soil.cfg.cell_size = 0.125;
    soil.cfg.grid_origin = [-4.0; 3];
    for (i, p) in soil.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        p[0] = ((i % 4) as f32 + 0.125).to_bits();
        p[1] = 2.0f32.to_bits();
        p[2] = 2.0f32.to_bits();
    }
    MpmSoilFrameCheckpoint::new(soil, [1e12; 3]).unwrap()
}
pub(super) fn archive(state: &MpmSoilFrameCheckpoint) -> MpmSoilCanonicalTileArchive {
    state
        .partition_canonical_tiles(
            SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
            &(0..state.soil().count)
                .map(|i| SoilParticleId((u64::from(i) << 40) + 7))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(0),
                SoilMaterialId(1 << 50),
                SoilMaterialId(u64::MAX),
            ],
        )
        .unwrap()
}
pub(super) fn bounds(tiles: u32) -> MpmSoilStoreReadLimits {
    MpmSoilStoreReadLimits {
        tiles,
        particles: 129,
        encoded_bytes: u64::MAX,
    }
}
pub(super) fn shifted(read: &MpmSoilCanonicalStoreRead) -> MpmSoilCanonicalSelection {
    read.select([1e12 + 1.0, 1e12, 1e12], [-5.0, -4.0, -4.0])
        .unwrap()
}
pub(super) fn moved(selection: &MpmSoilCanonicalSelection) -> MpmSoilFrameCheckpoint {
    let mut updated = selection.framed_selection().to_frame_checkpoint().unwrap();
    for p in updated.soil.words[..updated.soil.count as usize * 4].chunks_exact_mut(4) {
        p[0] = 0.125f32.to_bits();
    }
    updated
}

#[test]
fn canonical_incremental_upgrade_writeback_and_reopen_preserve_exact_archive() {
    let dir = Directory::new();
    let mut expected = archive(&state(129));
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&expected).unwrap();
    assert_eq!(store.upgrade_index().unwrap().generation, 2);
    let keys: Vec<_> = expected.tiles().collect();
    let read = store.read_tiles(&keys, bounds(129)).unwrap().unwrap();
    let selection = shifted(&read);
    let updated = moved(&selection);
    let reference = expected
        .select(&keys, 129, [1e12 + 1., 1e12, 1e12], [-5., -4., -4.])
        .unwrap();
    expected.writeback(&reference, &updated).unwrap();
    assert_eq!(
        store
            .writeback(&read, &selection, &updated, bounds(129))
            .unwrap()
            .generation,
        3
    );
    store.collect_unreferenced().unwrap();
    drop(store);
    let mut reopened = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert_eq!(reopened.upgrade_index().unwrap().written_bytes, 0);
}

#[test]
fn canonical_store_roundtrip_zero_write_reuse_bounds_and_cross_family_rejection() {
    let dir = Directory::new();
    let initial = archive(&state(129));
    let keys: Vec<_> = initial.tiles().collect();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    assert_eq!(store.total_particle_count().unwrap(), None);
    assert_eq!(store.storage_stats().unwrap().generation, None);
    assert!(store.load().unwrap().is_none());
    assert!(store.layout().unwrap().is_none());
    assert_eq!(store.save(&initial).unwrap().written_tiles, 4);
    assert_eq!(store.total_particle_count().unwrap(), Some(129));
    assert_eq!(store.storage_stats().unwrap().archived_particles, 129);
    let reused = store.save(&initial).unwrap();
    assert_eq!(
        (
            reused.generation,
            reused.written_tiles,
            reused.written_bytes
        ),
        (1, 0, 0)
    );
    assert_eq!(store.layout().unwrap(), Some(initial.layout()));
    let read = store
        .read_tiles(&[keys[0], keys[0]], bounds(1))
        .unwrap()
        .unwrap();
    assert_eq!((read.stats().tiles, read.stats().particles), (1, 33));
    assert_eq!(store.tile_particle_count(keys[0]).unwrap(), Some(33));
    assert_eq!(read.tiles().collect::<Vec<_>>(), vec![keys[0]]);
    assert_eq!(
        read.archive.to_bytes(),
        initial.encoded_tiles(&keys[..1]).unwrap()
    );
    assert!(store.read_tiles(&keys, bounds(1)).is_err());
    assert!(
        store
            .read_tiles(
                &keys[..1],
                MpmSoilStoreReadLimits {
                    encoded_bytes: TILE_BYTES + 33 * PARTICLE_BYTES - 1,
                    ..bounds(1)
                }
            )
            .is_err()
    );
    let exact = store
        .read_tiles(
            &keys[..1],
            MpmSoilStoreReadLimits {
                encoded_bytes: TILE_BYTES + 33 * PARTICLE_BYTES,
                ..bounds(1)
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(exact.stats().particles, 33);
    assert!(MpmSoilCanonicalStore::open(&dir.0, limits()).is_err());
    drop(store);
    assert!(MpmSoilStore::open(&dir.0, limits()).is_err());
    assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_err());
    let mut reopened = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    assert_eq!(reopened.total_particle_count().unwrap(), Some(129));
    assert!(reopened.validate_read(&read).is_err());
    let loaded = reopened.load().unwrap().unwrap();
    assert_eq!(loaded.to_bytes(), initial.to_bytes());
    assert_eq!(reopened.save(&loaded).unwrap().written_bytes, 0);
    for framed in [false, true] {
        let other = Directory::new();
        if framed {
            let mut raw = MpmSoilFrameStore::open(&other.0, limits()).unwrap();
            raw.save(
                &state(129)
                    .partition_tiles(
                        initial.layout(),
                        &(0..129).map(SoilParticleId).collect::<Vec<_>>(),
                        initial.material_ids(),
                    )
                    .unwrap(),
            )
            .unwrap();
        } else {
            let mut raw = MpmSoilStore::open(&other.0, limits()).unwrap();
            raw.save(&super::super::tests::archive()).unwrap();
        }
        assert!(MpmSoilCanonicalStore::open(&other.0, limits()).is_err());
    }
}

#[test]
fn canonical_store_partial_reframed_move_preserves_unloaded_blobs_and_reopens() {
    let dir = Directory::new();
    let initial = archive(&state(129));
    let keys: Vec<_> = initial.tiles().collect();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let untouched = store
        .inner
        .manifest
        .as_ref()
        .unwrap()
        .get(keys[3])
        .unwrap()
        .unwrap()
        .clone();
    let read = store.read_tiles(&keys[..1], bounds(1)).unwrap().unwrap();
    let selected = shifted(&read);
    let update = moved(&selected);
    let mut expected = MpmSoilCanonicalTileArchive::from_bytes(&initial.to_bytes(), 129).unwrap();
    let expected_read = expected
        .select(
            &keys[..1],
            129,
            selected.framed.world_origin(),
            selected.framed.soil().cfg.grid_origin,
        )
        .unwrap();
    expected.writeback(&expected_read, &update).unwrap();
    let stats = store
        .writeback(&read, &selected, &update, bounds(2))
        .unwrap();
    assert_eq!(
        (stats.generation, stats.written_tiles, stats.reused_tiles),
        (2, 1, 2)
    );
    assert!(store.validate_read(&read).is_err());
    assert!(
        store
            .inner
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[3])
            .unwrap()
            .unwrap()
            == untouched
    );
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert!(store.collect_unreferenced().unwrap() > 0);
    drop(store);
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}

#[test]
fn canonical_store_write_preflight_failures_are_nonmutating_and_retryable() {
    let dir = Directory::new();
    let initial = archive(&state(129));
    let keys: Vec<_> = initial.tiles().collect();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let read = store.read_tiles(&keys[..1], bounds(1)).unwrap().unwrap();
    let selected = shifted(&read);
    let update = moved(&selected);
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    assert!(
        store
            .writeback(&read, &selected, &update, bounds(1))
            .is_err()
    );
    let mut wrong = update.clone();
    wrong.origin[0] += 1.0;
    assert!(
        store
            .writeback(&read, &selected, &wrong, bounds(2))
            .is_err()
    );
    wrong = update.clone();
    wrong.soil.words[STRIDES[..5].iter().sum::<usize>() * wrong.soil.count as usize] =
        2.0f32.to_bits();
    assert!(
        store
            .writeback(&read, &selected, &wrong, bounds(2))
            .is_err()
    );
    let other_read = store.read_tiles(&keys[..1], bounds(1)).unwrap().unwrap();
    assert!(
        store
            .writeback(&other_read, &selected, &update, bounds(2))
            .is_err()
    );
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    store.validate_read(&read).unwrap();
    assert_eq!(
        store
            .writeback(&read, &selected, &update, bounds(2))
            .unwrap()
            .generation,
        2
    );
}

#[test]
fn canonical_store_unselected_corruption_isolated_from_bounded_reads_and_writes() {
    let dir = Directory::new();
    let initial = archive(&state(129));
    let keys: Vec<_> = initial.tiles().collect();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&initial).unwrap();
    let hash = store
        .inner
        .manifest
        .as_ref()
        .unwrap()
        .get(keys[3])
        .unwrap()
        .unwrap()
        .hash;
    let path = store.inner.blob_path(&hash);
    let mut corrupted = fs::read(&path).unwrap();
    corrupted[100] ^= 1;
    fs::write(&path, corrupted).unwrap();
    let read = store.read_tiles(&keys[..1], bounds(1)).unwrap().unwrap();
    let selected = shifted(&read);
    store
        .writeback(&read, &selected, &moved(&selected), bounds(2))
        .unwrap();
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
        hash
    );
    assert!(store.read_tiles(&keys[3..], bounds(1)).is_err());
    assert!(store.load().is_err());
    assert!(store.collect_unreferenced().is_err());
}

#[test]
fn canonical_store_publication_failures_reopen_complete_old_or_new_generations() {
    for partial in [false, true] {
        for stage in [
            Stage::ChunksSynced,
            Stage::ManifestSynced,
            Stage::Published,
            Stage::DirectorySynced,
        ] {
            let dir = Directory::new();
            let initial = archive(&state(129));
            let keys: Vec<_> = initial.tiles().collect();
            let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
            store.save(&initial).unwrap();
            let read = store.read_tiles(&keys[..1], bounds(1)).unwrap().unwrap();
            let selected = shifted(&read);
            let update = moved(&selected);
            let mut expected =
                MpmSoilCanonicalTileArchive::from_bytes(&initial.to_bytes(), 129).unwrap();
            let selection = expected
                .select(
                    &keys[..1],
                    129,
                    selected.framed.world_origin(),
                    selected.framed.soil().cfg.grid_origin,
                )
                .unwrap();
            expected.writeback(&selection, &update).unwrap();
            let barrier = |at| {
                if at == stage {
                    Err(invalid("injected canonical publication failure"))
                } else {
                    Ok(())
                }
            };
            let result = if partial {
                store.writeback_with(&read, &selected, &update, bounds(2), barrier)
            } else {
                store.save_with(&expected, barrier)
            };
            assert!(result.is_err());
            assert!(store.load().is_err());
            drop(store);
            let mut recovered = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
            let published = matches!(stage, Stage::Published | Stage::DirectorySynced);
            assert_eq!(recovered.generation(), Some(if published { 2 } else { 1 }));
            assert_eq!(
                recovered.load().unwrap().unwrap().to_bytes(),
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
fn canonical_store_empty_state_and_capacity_failure_preserve_health() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSoilCanonicalStore>();
    send_sync::<MpmSoilCanonicalStoreRead>();
    let dir = Directory::new();
    let mut store = MpmSoilCanonicalStore::open(
        &dir.0,
        MpmSoilStoreLimits {
            particles: 0,
            tiles: 0,
        },
    )
    .unwrap();
    assert!(store.save(&archive(&state(129))).is_err());
    store.save(&archive(&state(0))).unwrap();
    let read = store.read_tiles(&[], bounds(0)).unwrap().unwrap();
    let selected = shifted(&read);
    let update = selected.framed_selection().to_frame_checkpoint().unwrap();
    assert_eq!(
        store
            .writeback(&read, &selected, &update, bounds(0))
            .unwrap()
            .written_bytes,
        0
    );
    assert_eq!(store.load().unwrap().unwrap().particle_count(), 0);
}

#[test]
#[ignore = "requires GPU; canonical durable selection restores exact state and device IDs"]
fn canonical_store_gpu_capture_reopen_selected_frame_restore_resume_writeback() {
    gpu_roundtrip(false);
}

#[test]
#[ignore = "requires GPU; background canonical I/O assembles and returns selected frames"]
fn canonical_store_gpu_background_reopen_select_restore_resume_writeback() {
    gpu_roundtrip(true);
}

fn gpu_roundtrip(background: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = state(129);
    let initial = archive(&source);
    let selection = initial
        .select(
            &initial.tiles().collect::<Vec<_>>(),
            129,
            source.origin,
            source.soil.cfg.grid_origin,
        )
        .unwrap();
    let mut live = MpmSoilFrame::new(new_sim(&engine, source.soil.cfg, 2), [0.0; 3]).unwrap();
    live.restore_selection(&engine, selection.framed_selection())
        .unwrap();
    live.step_n_with(device, queue, 0.0001, 2).unwrap();
    let evolved = live.checkpoint_with(queue).unwrap();
    let evolved_archive = archive(&evolved);
    let dir = Directory::new();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&evolved_archive).unwrap();
    drop(store);
    drop(live);
    let store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let origin = source.origin.map(|v| v + 0.000244140625);
    let grid = source.soil.cfg.grid_origin.map(|v| v - 0.000244140625);
    let (mut store, read, selected) = if background {
        let mut task = store.begin_read(&keys, bounds(129), origin, grid);
        let completion = tasks::tests::finish_read(&mut task);
        let (read, selected) = completion.result.unwrap().unwrap();
        (completion.store, read, selected)
    } else {
        let read = store.read_tiles(&keys, bounds(129)).unwrap().unwrap();
        let selected = read.select(origin, grid).unwrap();
        (store, read, selected)
    };
    let expected = evolved_archive.select(&keys, 129, origin, grid).unwrap();
    assert_eq!(
        selected
            .framed_selection()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes(),
        expected
            .framed_selection()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes()
    );
    let mut restored = MpmSoilFrame::new(new_sim(&engine, source.soil.cfg, 2), [0.0; 3]).unwrap();
    restored
        .restore_selection(&engine, selected.framed_selection())
        .unwrap();
    assert_eq!(
        restored.checkpoint_with(queue).unwrap().to_bytes(),
        selected
            .framed_selection()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes()
    );
    let view = restored.simulation().soil_identity_buffers().unwrap();
    let words = crate::mpm::replay_tests::read_words(device, queue, view.particles);
    let ids: Vec<_> = selected
        .framed_selection()
        .particle_ids()
        .iter()
        .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
        .collect();
    assert_eq!(&words[..ids.len()], ids);
    restored.step_n_with(device, queue, 0.0001, 1).unwrap();
    let updated = restored.checkpoint_with(queue).unwrap();
    let expected = archive(&updated).to_bytes();
    if background {
        let mut task = store.begin_writeback(read, selected, updated, bounds(129));
        let completion = tasks::tests::finish_write(&mut task);
        completion.result.unwrap();
        store = completion.store;
    } else {
        store
            .writeback(&read, &selected, &updated, bounds(129))
            .unwrap();
    }
    drop(store);
    let mut reopened = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    assert_eq!(reopened.load().unwrap().unwrap().to_bytes(), expected);
}
