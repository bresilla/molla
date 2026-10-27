use super::super::tests::{Directory, archive, limits, read_limits};
use super::*;

fn moved(read: &MpmSoilStoreRead, targets: &[SoilTileCoord]) -> MpmSoilCheckpoint {
    let mut state = read.selection().checkpoint().clone();
    for (i, position) in state.words[..state.count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        for axis in 0..3 {
            position[axis] = ((targets[i % targets.len()].0[axis] as f32 + 0.25) * 0.125).to_bits();
        }
    }
    state
}

#[test]
fn selected_writeback_merges_destinations_and_preserves_unloaded_tiles() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let mut expected = archive();
    store.save(&expected).unwrap();
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(store.epochs.is_empty());
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let read = store
        .read_tiles(&keys[..1], read_limits(129, 1))
        .unwrap()
        .unwrap();
    let old_selection = expected.select(&keys[..1], 129).unwrap();
    let new_tile = SoilTileCoord([32, -4, 2]);
    let updated = moved(&read, &[keys[1], new_tile]);
    expected.writeback(&old_selection, &updated).unwrap();
    let untouched = store
        .manifest
        .as_ref()
        .unwrap()
        .get(keys[2])
        .unwrap()
        .unwrap()
        .clone();
    let path = store.blob_path(&untouched.hash);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, b"unloaded corruption").unwrap();
    let stats = store
        .writeback(&read, &updated, read_limits(129, 3))
        .unwrap();
    assert_eq!(stats.generation, 2);
    assert_eq!(stats.written_tiles, 2);
    assert_eq!(stats.reused_tiles as usize, keys.len() - 2);
    assert!(store.validate_read(&read).is_err());
    assert!(
        store
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[2])
            .unwrap()
            .unwrap()
            == untouched
    );
    assert!(
        store
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[0])
            .unwrap()
            .is_none()
    );
    assert!(store.load().is_err());
    fs::write(path, bytes).unwrap();
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    drop(store);
    let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(
        reopened.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}

#[test]
fn selected_writeback_unchanged_and_empty_keep_generation() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let archive = archive();
    store.save(&archive).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    for keys in [vec![key], vec![]] {
        let read = store
            .read_tiles(&keys, read_limits(129, 1))
            .unwrap()
            .unwrap();
        let stats = store
            .writeback(&read, read.selection().checkpoint(), read_limits(129, 1))
            .unwrap();
        assert_eq!(stats.generation, 1);
        assert_eq!(stats.written_bytes, 0);
        assert_eq!(stats.written_tiles, 0);
        assert_eq!(stats.reused_tiles as usize, archive.tiles.len());
        store.validate_read(&read).unwrap();
    }
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        archive.to_bytes()
    );
}

#[test]
fn selected_writeback_rejects_bounds_identity_and_physical_changes() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let read = store
        .read_tiles(&keys[..1], read_limits(129, 1))
        .unwrap()
        .unwrap();
    let updated = moved(&read, &[keys[1]]);
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let path = store.blob_path(
        &store
            .manifest
            .as_ref()
            .unwrap()
            .get(keys[1])
            .unwrap()
            .unwrap()
            .hash,
    );
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    for bounds in [
        read_limits(129, 1),
        read_limits(read.stats().particles, 2),
        MpmSoilStoreReadLimits {
            encoded_bytes: 0,
            ..read_limits(129, 2)
        },
    ] {
        let error = store.writeback(&read, &updated, bounds).err().unwrap();
        assert!(error.to_string().contains("bound"));
        store.healthy().unwrap();
    }
    assert!(
        store
            .writeback(&read, &updated, read_limits(129, 2))
            .is_err()
    );
    store.healthy().unwrap();
    fs::write(path, bytes).unwrap();
    for stream in [4, 5, 7] {
        let mut invalid = updated.clone();
        let offset = STRIDES[..stream].iter().sum::<usize>() * invalid.count as usize;
        invalid.words[offset] = if stream == 7 {
            (invalid.words[offset] + 1) % 3
        } else {
            (f32::from_bits(invalid.words[offset]) * 2.0).to_bits()
        };
        invalid.validate().unwrap();
        assert!(
            store
                .writeback(&read, &invalid, read_limits(129, 2))
                .is_err()
        );
        store.healthy().unwrap();
    }
    let mut invalid = updated.clone();
    invalid.count -= 1;
    assert!(
        store
            .writeback(&read, &invalid, read_limits(129, 2))
            .is_err()
    );
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    drop(store);
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(
        store
            .writeback(&read, &updated, read_limits(129, 2))
            .is_err()
    );
    store.healthy().unwrap();
}

#[test]
fn selected_writeback_store_tile_limit_rejects_before_publication() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    store.limits.tiles = keys.len() as u32;
    let read = store
        .read_tiles(&keys[..1], read_limits(129, 1))
        .unwrap()
        .unwrap();
    let updated = moved(
        &read,
        &[SoilTileCoord([20, 0, 0]), SoilTileCoord([21, 0, 0])],
    );
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let files = fs::read_dir(&directory.0).unwrap().count();
    assert!(
        store
            .writeback(&read, &updated, read_limits(129, 3))
            .is_err()
    );
    store.healthy().unwrap();
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), files);
}

#[test]
fn selected_writeback_publication_failure_reopens_a_complete_generation() {
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
        let before = expected.to_bytes();
        let keys: Vec<_> = store.tiles().unwrap().collect();
        let read = store
            .read_tiles(&keys[..1], read_limits(129, 1))
            .unwrap()
            .unwrap();
        let updated = moved(&read, &[keys[1]]);
        let selection = expected.select(&keys[..1], 129).unwrap();
        expected.writeback(&selection, &updated).unwrap();
        assert!(
            store
                .writeback_with(&read, &updated, read_limits(129, 2), |at| {
                    if at == stage {
                        Err(invalid("injected publication failure"))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert!(store.healthy().is_err());
        drop(store);
        let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
        let published = matches!(stage, Stage::Published | Stage::DirectorySynced);
        assert_eq!(reopened.generation(), Some(if published { 2 } else { 1 }));
        assert_eq!(
            reopened.load().unwrap().unwrap().to_bytes(),
            if published {
                expected.to_bytes()
            } else {
                before
            }
        );
    }
}

#[test]
fn selected_writeback_rejects_duplicate_destination_ids_without_writes() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let mut archive = archive();
    store.save(&archive).unwrap();
    let keys: Vec<_> = archive.tiles().take(2).collect();
    let duplicate = archive.tiles[&keys[0]].ids[0];
    archive.tiles.get_mut(&keys[1]).unwrap().ids[0] = duplicate;
    let bytes = archive.encode_selection(archive.select(&keys[1..], 129).unwrap());
    let hash = *blake3::hash(&bytes).as_bytes();
    fs::write(store.blob_path(&hash), bytes).unwrap();
    let head = super::super::tests::rewrite_manifest(&mut store, |manifest| {
        manifest.entries.get_mut(&keys[1]).unwrap().hash = hash;
    });
    let read = store
        .read_tiles(&keys[..1], read_limits(129, 1))
        .unwrap()
        .unwrap();
    let updated = moved(&read, &keys[1..]);
    let files = fs::read_dir(&directory.0).unwrap().count();
    let error = store
        .writeback(&read, &updated, read_limits(129, 2))
        .err()
        .unwrap();
    assert!(error.to_string().contains("duplicate global particle ID"));
    store.healthy().unwrap();
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), files);
}

#[test]
fn repeated_selected_moves_survive_reopen_without_full_archive_loading() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let mut expected = archive();
    store.save(&expected).unwrap();
    let mut key = store.tiles().unwrap().next().unwrap();
    for step in 0..12 {
        drop(store);
        store = MpmSoilStore::open(&directory.0, limits()).unwrap();
        assert!(store.epochs.is_empty());
        let read = store
            .read_tiles(&[key], read_limits(129, 1))
            .unwrap()
            .unwrap();
        let next = SoilTileCoord([100 + step * 64, -12, 9]);
        let updated = moved(&read, &[next]);
        let selection = expected.select(&[key], 129).unwrap();
        expected.writeback(&selection, &updated).unwrap();
        let stats = store
            .writeback(&read, &updated, read_limits(read.stats().particles, 2))
            .unwrap();
        assert_eq!(stats.written_tiles, 1);
        assert!(store.epochs.len() <= 1);
        let keys: Vec<_> = store.tiles().unwrap().collect();
        assert_eq!(keys, expected.tiles().collect::<Vec<_>>());
        key = next;
    }
    assert_eq!(
        store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}
