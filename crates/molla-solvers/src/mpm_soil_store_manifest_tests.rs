use super::super::tests::{Directory, archive};
use super::*;

fn synthetic(tiles: u32) -> Manifest {
    let metadata = metadata(&archive());
    let entries: BTreeMap<_, _> = (0..tiles)
        .map(|i| {
            let coordinate = if i == 0 {
                i64::MIN
            } else if i + 1 == tiles {
                i64::MAX
            } else {
                2 * i64::from(i) - i64::from(tiles)
            };
            (
                SoilTileCoord([coordinate, i64::from(i % 7), -5]),
                Entry {
                    count: 1 + i % 3,
                    hash: *blake3::hash(&i.to_le_bytes()).as_bytes(),
                },
            )
        })
        .collect();
    Manifest {
        generation: 7,
        particles: entries.values().map(|e| e.count).sum(),
        metadata,
        entries,
    }
}

fn bounds() -> MpmSoilStoreLimits {
    MpmSoilStoreLimits {
        particles: u32::MAX,
        tiles: u32::MAX,
    }
}

fn write(path: &Path, manifest: &Manifest) -> Vec<u8> {
    let expected = manifest.encode();
    let mut file = File::create(path).unwrap();
    assert_eq!(
        manifest.write_to(&mut file, bounds()).unwrap(),
        expected.len() as u64
    );
    assert_eq!(fs::read(path).unwrap(), expected);
    let mut file = File::create(path).unwrap();
    assert_eq!(
        write_entries(
            &mut file,
            manifest.generation,
            &manifest.metadata,
            manifest.entries.len() as u32,
            manifest.particles,
            manifest
                .entries
                .iter()
                .map(|(&key, entry)| Ok((key, entry.clone()))),
            bounds()
        )
        .unwrap(),
        expected.len() as u64
    );
    assert_eq!(fs::read(path).unwrap(), expected);
    expected
}

#[test]
fn file_index_open_and_partial_publication_keep_bounded_retained_storage() {
    let mut retained = None;
    for count in [1, 129, 16384] {
        let directory = Directory::new();
        let path = directory.0.join("HEAD");
        let mut expected = synthetic(count);
        write(&path, &expected);
        let old = ManifestFile::open(&path, bounds(), false, false).unwrap();
        let mut store = MpmSoilStore::open(&directory.0, bounds()).unwrap();
        let index = store.manifest.as_ref().unwrap();
        let bytes = index.metadata.capacity() + index.cursor.lock().unwrap().reader.capacity();
        assert_eq!(*retained.get_or_insert(bytes), bytes);
        assert!(bytes <= MAX_METADATA as usize + BUFFER_BYTES);
        assert!(store.epochs.is_empty());
        let first = *expected.entries.first_key_value().unwrap().0;
        let inserted = SoilTileCoord([i64::MIN, -100, 3]);
        let replace = BTreeSet::from([first, inserted]);
        let entry = Entry {
            count: 2,
            hash: [42; 32],
        };
        let changed = BTreeMap::from([(inserted, entry.clone())]);
        let before = old.get(first).unwrap().unwrap();
        expected.entries.remove(&first);
        expected.entries.insert(inserted, entry);
        expected.particles = expected.entries.values().map(|entry| entry.count).sum();
        expected.generation += 1;
        let result = store
            .publish(
                expected.metadata.clone(),
                changed.clone(),
                Some(&replace),
                BTreeMap::new(),
                MpmSoilStoreStats::default(),
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(result.generation, expected.generation);
        assert_eq!(fs::read(&path).unwrap(), expected.encode());
        assert_eq!(old.get(first).unwrap(), Some(before));
        assert!(old.get(inserted).unwrap().is_none());
        let index = store.manifest.as_ref().unwrap();
        assert_eq!(index.snapshot().unwrap(), expected.entries);
        assert_eq!(
            index.metadata.capacity() + index.cursor.lock().unwrap().reader.capacity(),
            bytes
        );
        let unchanged = store
            .publish(
                expected.metadata.clone(),
                changed,
                Some(&replace),
                BTreeMap::new(),
                MpmSoilStoreStats::default(),
                &mut |_| panic!("unchanged publication wrote"),
            )
            .unwrap();
        assert_eq!(unchanged.generation, expected.generation);
        assert_eq!(unchanged.written_bytes, 0);
        drop(store);
        let reopened = MpmSoilStore::open(&directory.0, bounds()).unwrap();
        assert_eq!(
            reopened.total_particle_count().unwrap(),
            Some(expected.particles)
        );
        assert_eq!(
            reopened.tiles().unwrap().collect::<Vec<_>>(),
            expected.entries.keys().copied().collect::<Vec<_>>()
        );
    }
}

#[test]
fn file_index_stream_merge_propagates_errors_and_rejects_bad_totals_or_order() {
    let directory = Directory::new();
    let path = directory.0.join("HEAD");
    let original = synthetic(129);
    write(&path, &original);
    let index = ManifestFile::open(&path, bounds(), false, false).unwrap();
    let first = *original.entries.first_key_value().unwrap().0;
    let overlap = BTreeMap::from([(first, original.entries[&first].clone())]);
    let empty = BTreeSet::new();
    assert!(
        merged(Some(&index), Some(&empty), &overlap)
            .collect::<Result<Vec<_>>>()
            .is_err()
    );
    let output = directory.0.join("output");
    for mode in 0..6 {
        let mut entries: Vec<_> = original
            .entries
            .iter()
            .take(2)
            .map(|(&key, entry)| (key, entry.clone()))
            .collect();
        let mut particles = entries.iter().map(|(_, entry)| entry.count).sum();
        let mut tiles = 2;
        match mode {
            0 => entries.reverse(),
            1 => entries[1] = entries[0].clone(),
            2 => entries[0].1.count = 0,
            3 => particles += 1,
            4 => tiles += 1,
            _ => tiles -= 1,
        }
        assert!(
            write_entries(
                &mut File::create(&output).unwrap(),
                8,
                &original.metadata,
                tiles,
                particles,
                entries.into_iter().map(Ok),
                bounds()
            )
            .is_err()
        );
    }
    assert!(
        write_entries(
            &mut File::create(&output).unwrap(),
            8,
            &original.metadata,
            1,
            1,
            std::iter::once(Err(invalid("injected entry failure"))),
            bounds()
        )
        .is_err()
    );
    let mut store = MpmSoilStore::open(&directory.0, bounds()).unwrap();
    File::create(&path).unwrap().set_len(0).unwrap();
    assert!(store.tile_particle_count(first).is_err());
    assert!(store.tiles().is_err());
    assert!(store.load().is_err());
    assert!(index.iter().collect::<Result<Vec<_>>>().is_err());
}

#[test]
fn store_manifest_stream_matches_legacy_and_bounded_lookup() {
    let directory = Directory::new();
    let path = directory.0.join("index");
    let mut retained = None;
    for count in [0, 1, 129, 16384] {
        let original = synthetic(count);
        write(&path, &original);
        let index = ManifestFile::open(&path, bounds(), false, false).unwrap();
        assert_eq!(index.generation, original.generation);
        assert_eq!(index.tiles, count);
        assert_eq!(
            index.particles,
            original.entries.values().map(|e| e.count).sum::<u32>()
        );
        let bytes = index.metadata.capacity() + index.cursor.lock().unwrap().reader.capacity();
        assert_eq!(*retained.get_or_insert(bytes), bytes);
        assert!(bytes <= MAX_METADATA as usize + BUFFER_BYTES);
        for (key, value) in original
            .entries
            .iter()
            .step_by((count as usize / 113).max(1))
        {
            assert_eq!(index.get(*key).unwrap().as_ref(), Some(value));
        }
        assert!(
            index
                .get(SoilTileCoord([i64::MAX, 9, 9]))
                .unwrap()
                .is_none()
        );
        assert!(
            index
                .get(SoilTileCoord([i64::MIN, -9, -9]))
                .unwrap()
                .is_none()
        );
        let decoded = index.materialize().unwrap();
        assert_eq!(decoded.entries, original.entries);
        assert_eq!(decoded.metadata, original.metadata);
    }
}

#[test]
fn store_manifest_stream_rejects_bounds_order_digest_and_mutation() {
    let directory = Directory::new();
    let path = directory.0.join("index");
    let original = synthetic(129);
    let bytes = write(&path, &original);
    let index = ManifestFile::open(&path, bounds(), false, false).unwrap();
    let offset = index.offset as usize;
    for cut in [0, 8, 43, 59, bytes.len() - 1] {
        fs::write(&path, &bytes[..cut]).unwrap();
        assert!(ManifestFile::open(&path, bounds(), false, false).is_err());
    }
    for mutation in 0..7 {
        let mut bad = bytes.clone();
        match mutation {
            0 => bad[0] ^= 1,
            1 => bad[12] ^= 1,
            2 => bad[52..56].copy_from_slice(&u32::MAX.to_le_bytes()),
            3 => bad[offset + 24..offset + 28].fill(0),
            4 => {
                let entry = bad[offset..offset + 60].to_vec();
                bad[offset + 60..offset + 120].copy_from_slice(&entry);
            }
            5 => bad[44..52].fill(0),
            _ => bad.push(0),
        }
        if mutation >= 2 {
            let digest = blake3::hash(&bad[HEADER_BYTES..]);
            bad[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        }
        fs::write(&path, bad).unwrap();
        assert!(ManifestFile::open(&path, bounds(), false, false).is_err());
    }
    fs::write(&path, &bytes).unwrap();
    assert!(
        ManifestFile::open(
            &path,
            MpmSoilStoreLimits {
                tiles: 128,
                ..bounds()
            },
            false,
            false
        )
        .is_err()
    );
    assert!(
        ManifestFile::open(
            &path,
            MpmSoilStoreLimits {
                particles: 128,
                ..bounds()
            },
            false,
            false
        )
        .is_err()
    );
    let mut changed = bytes.clone();
    changed[offset + 28] ^= 1;
    fs::write(&path, changed).unwrap();
    assert!(index.materialize().is_err());
    fs::write(&path, &bytes).unwrap();
    let index = ManifestFile::open(&path, bounds(), false, false).unwrap();
    File::create(&path).unwrap().set_len(offset as u64).unwrap();
    assert!(
        index
            .get(*original.entries.last_key_value().unwrap().0)
            .is_err()
    );
    assert!(index.materialize().is_err());
}

#[test]
fn store_manifest_stream_lookup_serializes_shared_cursor() {
    let directory = Directory::new();
    let path = directory.0.join("index");
    let original = synthetic(1024);
    write(&path, &original);
    let index = Arc::new(ManifestFile::open(&path, bounds(), false, false).unwrap());
    std::thread::scope(|scope| {
        for parity in 0..4 {
            let index = index.clone();
            let entries = &original.entries;
            scope.spawn(move || {
                for (&key, expected) in entries.iter().skip(parity).step_by(4) {
                    assert_eq!(index.get(key).unwrap().as_ref(), Some(expected));
                }
            });
        }
    });
    let index = Arc::try_unwrap(index).ok().unwrap();
    assert_eq!(index.materialize().unwrap().entries, original.entries);
}

#[test]
fn store_manifest_stream_writer_and_family_rejections() {
    let directory = Directory::new();
    let path = directory.0.join("index");
    let original = synthetic(129);
    let bytes = write(&path, &original);
    assert!(ManifestFile::open(&path, bounds(), true, false).is_err());
    assert!(ManifestFile::open(&path, bounds(), false, true).is_err());
    for limits in [
        MpmSoilStoreLimits {
            tiles: 128,
            ..bounds()
        },
        MpmSoilStoreLimits {
            particles: 128,
            ..bounds()
        },
    ] {
        assert!(
            original
                .write_to(&mut File::create(&path).unwrap(), limits)
                .is_err()
        );
    }
    let mut zero = synthetic(1);
    zero.particles += 1;
    assert!(
        zero.write_to(&mut File::create(&path).unwrap(), bounds())
            .is_err()
    );
    zero.particles -= 1;
    zero.entries.first_entry().unwrap().get_mut().count = 0;
    assert!(
        zero.write_to(&mut File::create(&path).unwrap(), bounds())
            .is_err()
    );
    fs::write(&path, bytes).unwrap();
    assert!(ManifestFile::open(&directory.0, bounds(), false, false).is_err());
    #[cfg(unix)]
    {
        let link = directory.0.join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(ManifestFile::open(&link, bounds(), false, false).is_err());
    }
}
