use super::super::tests::Directory;
use super::*;

struct UniquePages<'a> {
    pages: &'a mut Memory,
    reads: std::collections::BTreeSet<[u8; 32]>,
    writes: std::collections::BTreeSet<[u8; 32]>,
}

impl Pages for UniquePages<'_> {
    fn read(&mut self, hash: &[u8; 32], buffer: &mut [u8; PAGE_BYTES]) -> Result<usize> {
        assert!(self.reads.insert(*hash), "batch reread a page");
        self.pages.read(hash, buffer)
    }
    fn write(&mut self, hash: &[u8; 32], bytes: &[u8]) -> Result<()> {
        assert!(self.writes.insert(*hash), "batch rewrote a page");
        self.pages.write(hash, bytes)
    }
}

#[test]
fn batch_edits_match_reference_without_duplicate_or_orphan_pages() {
    let (mut pages, initial) = seeded(128);
    let mut root = Some(initial.clone());
    let mut reference = snapshot(&mut pages, root.as_ref());
    let original = reference.clone();
    let mut seed = 731u64;
    for iteration in 0..128 {
        let mut edits = BTreeMap::new();
        for i in 0..16 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let key = SoilTileCoord([((seed >> 16) % 512) as i64 - 256, (seed % 7) as i64 - 3, 0]);
            edits.insert(key, (seed % 3 != 0).then(|| entry(iteration * 32 + i)));
        }
        let edits: Vec<_> = edits.into_iter().collect();
        let mut serial = pages.clone();
        let mut expected = root.clone();
        for (key, value) in &edits {
            expected = update(&mut serial, expected.as_ref(), *key, value.clone()).unwrap();
            if let Some(value) = value {
                reference.insert(*key, value.clone());
            } else {
                reference.remove(key);
            }
        }
        let mut unique = UniquePages {
            pages: &mut pages,
            reads: Default::default(),
            writes: Default::default(),
        };
        root = update_many(&mut unique, root.as_ref(), &edits).unwrap();
        let writes = unique.writes;
        let mut reachable = std::collections::BTreeSet::new();
        visit_pages(&mut pages, root.as_ref(), |link| {
            reachable.insert(link.hash);
            Ok(())
        })
        .unwrap();
        assert!(writes.is_subset(&reachable));
        assert_eq!(snapshot(&mut pages, root.as_ref()), reference);
        assert_eq!(snapshot(&mut serial, expected.as_ref()), reference);
    }
    assert_eq!(snapshot(&mut pages, Some(&initial)), original);
    let delete: Vec<_> = reference.keys().map(|&key| (key, None)).collect();
    assert!(
        update_many(&mut pages, root.as_ref(), &delete)
            .unwrap()
            .is_none()
    );
}

#[test]
fn batch_noops_bad_inputs_and_page_failures_preserve_roots() {
    let (original, root) = seeded(128);
    let old = snapshot(&mut original.clone(), Some(&root));
    let edits: Vec<_> = old
        .iter()
        .map(|(&key, entry)| (key, Some(entry.clone())))
        .collect();
    let mut pages = original.clone();
    let before = pages.writes;
    assert_eq!(
        update_many(&mut pages, Some(&root), &edits).unwrap(),
        Some(root.clone())
    );
    assert_eq!(pages.writes, before);
    let missing = [
        (SoilTileCoord([i64::MIN; 3]), None),
        (SoilTileCoord([i64::MAX; 3]), None),
    ];
    assert_eq!(
        update_many(&mut pages, Some(&root), &missing).unwrap(),
        Some(root.clone())
    );
    assert_eq!(pages.writes, before);
    for bad in [
        vec![edits[0].clone(), edits[0].clone()],
        vec![edits[1].clone(), edits[0].clone()],
        vec![(
            edits[0].0,
            Some(Entry {
                count: 0,
                hash: [0; 32],
            }),
        )],
    ] {
        let reads = pages.reads;
        assert!(update_many(&mut pages, Some(&root), &bad).is_err());
        assert_eq!(pages.reads, reads);
        assert_eq!(pages.writes, before);
    }
    let edits: Vec<_> = old
        .keys()
        .enumerate()
        .map(|(i, &key)| {
            (
                key,
                if i % 3 == 0 {
                    None
                } else {
                    Some(entry(900 + i as u32))
                },
            )
        })
        .collect();
    let mut complete = original.clone();
    update_many(&mut complete, Some(&root), &edits).unwrap();
    for failed in 0..complete.writes - original.writes {
        let mut pages = original.clone();
        pages.fail = Some(pages.writes + failed);
        assert!(update_many(&mut pages, Some(&root), &edits).is_err());
        assert_eq!(snapshot(&mut pages, Some(&root)), old);
    }
    let mut corrupt = original.clone();
    corrupt.data.get_mut(&root.hash).unwrap()[0] ^= 1;
    assert!(update_many(&mut corrupt, Some(&root), &edits).is_err());
}

#[test]
fn batch_shared_paths_reduce_io_with_exact_contents() {
    for count in [64, 1024, 4096] {
        let (initial, root) = seeded(count);
        let keys: Vec<_> = snapshot(&mut initial.clone(), Some(&root))
            .keys()
            .copied()
            .collect();
        let edits: Vec<_> = keys
            .iter()
            .take(32)
            .enumerate()
            .map(|(i, &key)| (key, Some(entry(100000 + i as u32))))
            .collect();
        let mut serial = initial.clone();
        let mut serial_root = Some(root.clone());
        for (key, value) in &edits {
            serial_root = update(&mut serial, serial_root.as_ref(), *key, value.clone()).unwrap();
        }
        let mut pages = initial.clone();
        let mut unique = UniquePages {
            pages: &mut pages,
            reads: Default::default(),
            writes: Default::default(),
        };
        let result = update_many(&mut unique, Some(&root), &edits).unwrap();
        let (read_pages, written_pages) = (unique.reads.len(), unique.writes.len());
        let bytes = pages.bytes - initial.bytes;
        let reference_bytes = serial.bytes - initial.bytes;
        assert_eq!(result, serial_root);
        assert!(bytes * 2 < reference_bytes);
        assert!(read_pages < serial.reads - initial.reads);
        eprintln!(
            "BATCH_INDEX {}",
            serde_json::json!({"tiles":count,"edits":32,"written_bytes":bytes,"reference_bytes":reference_bytes,"read_pages":read_pages,"reference_reads":serial.reads-initial.reads,"written_pages":written_pages,"exact_root":true})
        );
    }
}

#[test]
fn batch_signed_extremes_empty_and_prefix_joins() {
    let mut entries = BTreeMap::new();
    for axis in 0..3 {
        for shift in 0..64 {
            let mut key = [i64::MIN; 3];
            key[axis] = ((1u64 << shift) ^ (1u64 << 63)) as i64;
            entries.insert(SoilTileCoord(key), Some(entry(shift)));
        }
    }
    entries.insert(SoilTileCoord([i64::MAX; 3]), Some(entry(1000)));
    entries.insert(SoilTileCoord([i64::MIN; 3]), Some(entry(1001)));
    let edits: Vec<_> = entries.into_iter().collect();
    let mut pages = Memory::default();
    assert_eq!(update_many(&mut pages, None, &[]).unwrap(), None);
    let mut root = None;
    for chunk in edits.rchunks(17) {
        root = update_many(&mut pages, root.as_ref(), chunk).unwrap();
    }
    let expected: BTreeMap<_, _> = edits
        .iter()
        .map(|(key, value)| (*key, value.clone().unwrap()))
        .collect();
    assert_eq!(snapshot(&mut pages, root.as_ref()), expected);
    let remove: Vec<_> = edits.iter().map(|(key, _)| (*key, None)).collect();
    assert_eq!(
        update_many(&mut pages, root.as_ref(), &remove).unwrap(),
        None
    );
    let mut pages = Memory::default();
    let overflow = [
        (
            SoilTileCoord([0; 3]),
            Some(Entry {
                count: u32::MAX,
                hash: [0; 32],
            }),
        ),
        (SoilTileCoord([1; 3]), Some(entry(1))),
    ];
    assert!(update_many(&mut pages, None, &overflow).is_err());
}

#[test]
fn sorted_builder_matches_updates_and_writes_only_reachable_pages() {
    for count in [0, 1, 31, 32, 33, 64, 257, 4096] {
        for random in [false, true] {
            let mut seed = 0x1234_5678_9abc_def0u64;
            let entries: BTreeMap<_, _> = (0..count)
                .map(|i| {
                    let key = if random {
                        SoilTileCoord(std::array::from_fn(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            seed as i64
                        }))
                    } else {
                        SoilTileCoord([i as i64 - count as i64 / 2, i64::MIN, i64::MAX])
                    };
                    (key, entry(i))
                })
                .collect();
            let mut reference = Memory::default();
            let mut expected = None;
            for (&key, entry) in &entries {
                expected =
                    update(&mut reference, expected.as_ref(), key, Some(entry.clone())).unwrap();
            }
            let mut pages = Memory::default();
            let root = build_sorted(
                &mut pages,
                entries.iter().map(|(&key, entry)| Ok((key, entry.clone()))),
            )
            .unwrap();
            assert_eq!(root, expected);
            assert_eq!(pages.reads, 0);
            let mut reachable = std::collections::BTreeSet::new();
            visit_pages(&mut pages, root.as_ref(), |link| {
                assert!(reachable.insert(link.hash));
                Ok(())
            })
            .unwrap();
            assert_eq!(pages.writes, reachable.len());
            assert_eq!(
                pages
                    .data
                    .keys()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                reachable
            );
            assert_eq!(snapshot(&mut pages, root.as_ref()), entries);
            if count == 4096 {
                assert!(pages.bytes * 8 < reference.bytes);
                eprintln!(
                    "SORTED_INDEX {}",
                    serde_json::json!({"tiles":count,"random":random,"reference_bytes":reference.bytes,"sorted_bytes":pages.bytes,"sorted_pages":pages.writes,"reachable_pages":reachable.len(),"build_reads":0,"exact_root":true})
                );
            }
        }
    }
}

#[test]
fn sorted_builder_rejects_bad_streams_and_preserves_old_roots_on_write_failure() {
    let key = |i| SoilTileCoord([i, 0, 0]);
    for entries in [
        vec![Ok((key(0), entry(0))), Ok((key(0), entry(1)))],
        vec![Ok((key(1), entry(0))), Ok((key(0), entry(1)))],
        vec![Ok((
            key(0),
            Entry {
                count: 0,
                hash: [0; 32],
            },
        ))],
        vec![
            Ok((
                key(0),
                Entry {
                    count: u32::MAX,
                    hash: [0; 32],
                },
            )),
            Ok((key(1), entry(0))),
        ],
        vec![Err(invalid("stream failure"))],
    ] {
        assert!(build_sorted(&mut Memory::default(), entries).is_err());
    }
    let entries: Vec<_> = (0..257).map(|i| (key(i), entry(i as u32))).collect();
    let mut complete = Memory::default();
    build_sorted(&mut complete, entries.iter().cloned().map(Ok)).unwrap();
    let (original, root) = seeded(64);
    for failure in 0..complete.writes {
        let mut pages = original.clone();
        pages.fail = Some(pages.writes + failure);
        assert!(build_sorted(&mut pages, entries.iter().cloned().map(Ok)).is_err());
        assert_eq!(
            snapshot(&mut pages, Some(&root)),
            snapshot(&mut original.clone(), Some(&root))
        );
    }
    let mut partial = Memory::default();
    let stream = entries
        .into_iter()
        .map(Ok)
        .chain(std::iter::once(Err(invalid("late stream failure"))));
    assert!(build_sorted(&mut partial, stream).is_err());
    assert!(partial.writes > 0);
}

#[test]
fn sorted_builder_accepts_all_signed_key_extremes_and_deep_prefixes() {
    let mut entries = BTreeMap::new();
    for x in [i64::MIN, -1, 0, i64::MAX] {
        for y in [i64::MIN, -1, 0, i64::MAX] {
            for z in [i64::MIN, -1, 0, i64::MAX] {
                entries.insert(SoilTileCoord([x, y, z]), entry(1));
            }
        }
    }
    for axis in 0..3 {
        for shift in 0..64 {
            let mut key = [i64::MIN; 3];
            key[axis] = ((1u64 << shift) ^ (1u64 << 63)) as i64;
            entries.insert(SoilTileCoord(key), entry(2));
        }
    }
    let mut expected = None;
    let mut reference = Memory::default();
    for (&key, entry) in &entries {
        expected = update(&mut reference, expected.as_ref(), key, Some(entry.clone())).unwrap();
    }
    let mut pages = Memory::default();
    let root = build_sorted(
        &mut pages,
        entries.iter().map(|(&key, entry)| Ok((key, entry.clone()))),
    )
    .unwrap();
    assert_eq!(root, expected);
    assert_eq!(pages.reads, 0);
    assert_eq!(snapshot(&mut pages, root.as_ref()), entries);
}

#[derive(Clone, Default)]
struct Memory {
    data: BTreeMap<[u8; 32], Vec<u8>>,
    reads: usize,
    writes: usize,
    bytes: usize,
    fail: Option<usize>,
}

impl Pages for Memory {
    fn read(&mut self, hash: &[u8; 32], buffer: &mut [u8; PAGE_BYTES]) -> Result<usize> {
        self.reads += 1;
        let bytes = self
            .data
            .get(hash)
            .ok_or_else(|| invalid("missing test index page"))?;
        if bytes.len() > buffer.len() {
            return Err(invalid("oversized test page"));
        }
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
    fn write(&mut self, hash: &[u8; 32], bytes: &[u8]) -> Result<()> {
        if self.fail == Some(self.writes) {
            return Err(invalid("injected page write failure"));
        }
        assert!(bytes.len() <= PAGE_BYTES);
        assert_eq!(blake3::hash(bytes).as_bytes(), hash);
        self.writes += 1;
        self.bytes += bytes.len();
        if let Some(old) = self.data.insert(*hash, bytes.to_vec()) {
            assert_eq!(old, bytes);
        }
        Ok(())
    }
}

fn entry(value: u32) -> Entry {
    Entry {
        count: 1 + value % 3,
        hash: *blake3::hash(&value.to_le_bytes()).as_bytes(),
    }
}

fn snapshot(pages: &mut impl Pages, root: Option<&Link>) -> BTreeMap<SoilTileCoord, Entry> {
    let mut entries = BTreeMap::new();
    let mut previous = None;
    scan(pages, root, |key, entry| {
        assert!(previous.is_none_or(|last| last < key));
        previous = Some(key);
        assert!(entries.insert(key, entry.clone()).is_none());
        Ok(())
    })
    .unwrap();
    if let Some(root) = root {
        assert_eq!(root.tiles as usize, entries.len());
        assert_eq!(
            root.particles,
            entries.values().map(|entry| entry.count).sum::<u32>()
        );
    } else {
        assert!(entries.is_empty());
    }
    entries
}

fn seeded(count: u32) -> (Memory, Link) {
    let mut pages = Memory::default();
    let mut root = None;
    for i in 0..count {
        root = update(
            &mut pages,
            root.as_ref(),
            SoilTileCoord([i64::from(i), -5, 7]),
            Some(entry(i)),
        )
        .unwrap();
    }
    (pages, root.unwrap())
}

#[test]
fn cow_index_signed_order_random_updates_deletion_and_old_roots_match_reference() {
    let mut pages = Memory::default();
    let mut root = None;
    let mut expected = BTreeMap::new();
    let mut random = 0x90d4_7f13_51c3_77a9u64;
    for i in 0..4096u32 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let key = if i % 101 == 0 {
            SoilTileCoord([i64::MIN, i64::from(i), i64::MAX])
        } else if i % 103 == 0 {
            SoilTileCoord([i64::MAX, i64::MIN, i64::from(i)])
        } else {
            SoilTileCoord([
                (random % 193) as i64 - 96,
                ((random >> 9) % 7) as i64 - 3,
                -5,
            ])
        };
        let value = (random & 7 != 0).then(|| entry(i));
        let old = root.clone();
        let old_value = expected.get(&key).cloned();
        let writes = pages.writes;
        root = update(&mut pages, root.as_ref(), key, value.clone()).unwrap();
        if value == old_value {
            assert_eq!(pages.writes, writes);
            assert_eq!(root, old);
        }
        assert!(pages.writes - writes <= KEY_BITS as usize + 3);
        assert_eq!(get(&mut pages, old.as_ref(), key).unwrap(), old_value);
        if let Some(entry) = value {
            expected.insert(key, entry);
        } else {
            expected.remove(&key);
        }
        let reads = pages.reads;
        assert_eq!(
            get(&mut pages, root.as_ref(), key).unwrap().as_ref(),
            expected.get(&key)
        );
        assert!(pages.reads - reads <= KEY_BITS as usize + 1);
        if i % 127 == 0 {
            assert_eq!(snapshot(&mut pages, root.as_ref()), expected);
        }
    }
    assert_eq!(snapshot(&mut pages, root.as_ref()), expected);
    let keys: Vec<_> = expected.keys().copied().collect();
    for (i, key) in keys.into_iter().enumerate() {
        root = update(&mut pages, root.as_ref(), key, None).unwrap();
        expected.remove(&key);
        if i % 127 == 0 {
            assert_eq!(snapshot(&mut pages, root.as_ref()), expected);
        }
    }
    assert!(root.is_none());
    assert!(
        get(&mut pages, None, SoilTileCoord([0; 3]))
            .unwrap()
            .is_none()
    );
}

#[test]
fn cow_index_single_edit_writes_only_its_path_at_increasing_field_sizes() {
    for count in [64, 1024, 4096] {
        let (mut pages, root) = seeded(count);
        let key = SoilTileCoord([i64::from(count / 2), -5, 7]);
        pages.reads = 0;
        let before = get(&mut pages, Some(&root), key).unwrap();
        let path = pages.reads;
        pages.writes = 0;
        pages.bytes = 0;
        pages.reads = 0;
        let changed = update(&mut pages, Some(&root), key, Some(entry(99999)))
            .unwrap()
            .unwrap();
        assert_eq!(pages.reads, path);
        assert_eq!(pages.writes, path);
        assert!(pages.bytes <= PAGE_BYTES * path);
        if count == 4096 {
            assert!(pages.bytes < count as usize * 60 / 16);
        }
        println!(
            "COW_INDEX {{\"field_tiles\":{count},\"read_pages\":{},\"written_pages\":{},\"written_bytes\":{},\"page_bound\":{PAGE_BYTES},\"path_depth\":{path}}}",
            pages.reads, pages.writes, pages.bytes
        );
        assert_eq!(get(&mut pages, Some(&root), key).unwrap(), before);
        assert_eq!(
            get(&mut pages, Some(&changed), key).unwrap(),
            Some(entry(99999))
        );
        assert_eq!(snapshot(&mut pages, Some(&changed)).len(), count as usize);
        let writes = pages.writes;
        assert_eq!(
            update(&mut pages, Some(&changed), key, Some(entry(99999))).unwrap(),
            Some(changed)
        );
        assert_eq!(pages.writes, writes);
    }
}

#[test]
fn cow_index_page_failures_leave_old_root_readable_and_reject_bad_descriptors() {
    let (pages, root) = seeded(256);
    let key = SoilTileCoord([128, -5, 7]);
    let mut control = pages.clone();
    let expected = snapshot(&mut control, Some(&root));
    let initial = control.writes;
    update(&mut control, Some(&root), key, Some(entry(99999))).unwrap();
    let changes = control.writes - initial;
    assert!(changes > 1);
    for fail in 0..changes {
        let mut failed = pages.clone();
        failed.fail = Some(failed.writes + fail);
        assert!(update(&mut failed, Some(&root), key, Some(entry(99999))).is_err());
        failed.fail = None;
        assert_eq!(snapshot(&mut failed, Some(&root)), expected);
        let retry = update(&mut failed, Some(&root), key, Some(entry(99999)))
            .unwrap()
            .unwrap();
        assert_eq!(
            get(&mut failed, Some(&retry), key).unwrap(),
            Some(entry(99999))
        );
    }
    let mut bad = root.clone();
    bad.particles += 1;
    assert!(get(&mut control, Some(&bad), key).is_err());
    let mut bad = root.clone();
    bad.first = SoilTileCoord([i64::MIN; 3]);
    assert!(get(&mut control, Some(&bad), key).is_err());
    assert!(
        update(
            &mut control,
            Some(&root),
            key,
            Some(Entry {
                count: 0,
                hash: [0; 32]
            })
        )
        .is_err()
    );
    let Node::Branch { bit, .. } = read(&mut control, &root, None).unwrap() else {
        panic!("expected branch")
    };
    assert!(read(&mut control, &root, Some(bit)).is_err());
    control.data.remove(&root.hash);
    assert!(get(&mut control, Some(&root), key).is_err());
}

#[test]
fn cow_index_codec_rejects_malformed_pages_before_unbounded_allocation() {
    let leaf = Node::Leaf(vec![(SoilTileCoord([i64::MIN, 0, i64::MAX]), entry(1))]);
    let bytes = leaf.encode().unwrap();
    let link = leaf.summary(*blake3::hash(&bytes).as_bytes()).unwrap();
    for cut in 0..bytes.len() {
        assert!(Node::decode(&bytes[..cut], &link, None).is_err());
    }
    for mutation in 0..8 {
        let mut damaged = bytes.clone();
        match mutation {
            0 => damaged[0] ^= 1,
            1 => damaged[8] = 2,
            2 => damaged[12] = 2,
            3 => damaged[13] = 1,
            4 => damaged[14..16].copy_from_slice(&u16::MAX.to_le_bytes()),
            5 => damaged[HEADER + 24..HEADER + 28].fill(0),
            6 => damaged.push(0),
            _ => damaged.resize(PAGE_BYTES + 1, 0),
        }
        let mut expected = link.clone();
        expected.hash = *blake3::hash(&damaged).as_bytes();
        assert!(Node::decode(&damaged, &expected, None).is_err());
    }
    let mut damaged = bytes.clone();
    damaged[HEADER + 28] ^= 1;
    assert!(Node::decode(&damaged, &link, None).is_err());
    assert!(Node::Leaf(vec![]).encode().is_err());
    assert!(
        Node::Leaf(vec![(link.first, entry(1)); 2])
            .encode()
            .is_err()
    );
    assert!(
        Node::Leaf(vec![(link.first, entry(1)); LEAF_ENTRIES + 1])
            .encode()
            .is_err()
    );
    let other = Link {
        first: SoilTileCoord([0; 3]),
        last: SoilTileCoord([0; 3]),
        ..link.clone()
    };
    assert!(
        Node::Branch {
            bit: KEY_BITS,
            left: link.clone(),
            right: other.clone()
        }
        .encode()
        .is_err()
    );
    assert!(
        Node::Branch {
            bit: 0,
            left: other,
            right: link
        }
        .encode()
        .is_err()
    );
}

#[test]
fn cow_index_maximum_key_depth_and_child_summary_validation_are_bounded() {
    let mut pages = Memory::default();
    let zero = SoilTileCoord([i64::MIN; 3]);
    let mut root = install(&mut pages, Node::Leaf(vec![(zero, entry(0))])).unwrap();
    for split in (0..KEY_BITS).rev() {
        let mut bytes = [0u8; 24];
        bytes[usize::from(split / 8)] = 0x80 >> (split % 8);
        let key = SoilTileCoord(std::array::from_fn(|axis| {
            (u64::from_be_bytes(bytes[axis * 8..axis * 8 + 8].try_into().unwrap()) ^ (1 << 63))
                as i64
        }));
        let right = install(&mut pages, Node::Leaf(vec![(key, entry(0))])).unwrap();
        root = install(
            &mut pages,
            Node::Branch {
                bit: split,
                left: root,
                right,
            },
        )
        .unwrap();
    }
    pages.reads = 0;
    assert_eq!(get(&mut pages, Some(&root), zero).unwrap(), Some(entry(0)));
    assert_eq!(pages.reads, KEY_BITS as usize + 1);
    pages.reads = 0;
    pages.writes = 0;
    let changed = update(&mut pages, Some(&root), zero, Some(entry(1)))
        .unwrap()
        .unwrap();
    assert_eq!(pages.reads, KEY_BITS as usize + 1);
    assert_eq!(pages.writes, KEY_BITS as usize + 1);
    assert_eq!(
        snapshot(&mut pages, Some(&changed)).len(),
        KEY_BITS as usize + 1
    );
    assert_eq!(get(&mut pages, Some(&root), zero).unwrap(), Some(entry(0)));
    let Node::Branch {
        bit,
        mut left,
        right,
    } = read(&mut pages, &root, None).unwrap()
    else {
        panic!("branch")
    };
    left.particles += 1;
    let forged = install(&mut pages, Node::Branch { bit, left, right }).unwrap();
    assert!(get(&mut pages, Some(&forged), zero).is_err());
    assert!(scan(&mut pages, Some(&forged), |_, _| Ok(())).is_err());
    assert!(
        Node::Leaf(vec![
            (
                SoilTileCoord([0; 3]),
                Entry {
                    count: u32::MAX,
                    hash: [0; 32]
                }
            ),
            (SoilTileCoord([1; 3]), entry(0))
        ])
        .encode()
        .is_err()
    );
}

struct Files(PathBuf);

impl Pages for Files {
    fn read(&mut self, hash: &[u8; 32], buffer: &mut [u8; PAGE_BYTES]) -> Result<usize> {
        let bytes = read_bounded(
            &self.0.join(format!(
                "{}.index",
                blake3::Hash::from_bytes(*hash).to_hex()
            )),
            PAGE_BYTES as u64,
        )?;
        buffer[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len())
    }
    fn write(&mut self, hash: &[u8; 32], bytes: &[u8]) -> Result<()> {
        let path = self.0.join(format!(
            "{}.index",
            blake3::Hash::from_bytes(*hash).to_hex()
        ));
        if path.exists() {
            if read_bounded(&path, PAGE_BYTES as u64)? != bytes {
                return Err(invalid("existing test page corrupted"));
            }
        } else {
            let mut file = File::create_new(path).map_err(io_error)?;
            file.write_all(bytes).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
        }
        Ok(())
    }
}

#[test]
fn cow_index_file_pages_reopen_old_roots_and_detect_corruption() {
    let directory = Directory::new();
    let mut pages = Files(directory.0.clone());
    let mut root = None;
    for i in 0..129 {
        root = update(
            &mut pages,
            root.as_ref(),
            SoilTileCoord([i, -i, i]),
            Some(entry(i as u32)),
        )
        .unwrap();
    }
    let original = root.unwrap();
    let expected = snapshot(&mut pages, Some(&original));
    let key = SoilTileCoord([64, -64, 64]);
    let changed = update(&mut pages, Some(&original), key, None)
        .unwrap()
        .unwrap();
    let mut descriptor = [0; LINK_BYTES];
    encode_link(&mut descriptor, &changed);
    drop(pages);
    let mut reopened = Files(directory.0.clone());
    let changed = decode_link(&descriptor);
    assert_eq!(snapshot(&mut reopened, Some(&original)), expected);
    assert!(get(&mut reopened, Some(&changed), key).unwrap().is_none());
    assert_eq!(snapshot(&mut reopened, Some(&changed)).len(), 128);
    let path = directory.0.join(format!(
        "{}.index",
        blake3::Hash::from_bytes(changed.hash).to_hex()
    ));
    let bytes = fs::read(&path).unwrap();
    for damaged in [bytes[..bytes.len() - 1].to_vec(), vec![0; PAGE_BYTES + 1]] {
        fs::write(&path, damaged).unwrap();
        assert!(get(&mut reopened, Some(&changed), key).is_err());
    }
    fs::write(&path, bytes).unwrap();
    assert_eq!(snapshot(&mut reopened, Some(&original)), expected);
}
