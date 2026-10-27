use super::*;
use crate::mpm::checkpoint::tests::{fixture, new_sim};

fn state(count: u32) -> MpmSoilFrameCheckpoint {
    let mut checkpoint = fixture(count);
    checkpoint.cfg.grid_size = 64;
    checkpoint.cfg.cell_size = 0.125;
    checkpoint.cfg.grid_origin = [-4.0; 3];
    for (i, particle) in checkpoint.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        particle[0] = [-0.000001f32, 0.125, 1.125, 2.125][i % 4].to_bits();
        particle[1] = 2.0f32.to_bits();
        particle[2] = 2.0f32.to_bits();
    }
    MpmSoilFrameCheckpoint::new(checkpoint, [1e12, -1e12, 1e12]).unwrap()
}

fn layout() -> SoilTileLayout {
    SoilTileLayout::new([0.0; 3], 1.0).unwrap()
}
fn ids(count: u32) -> Vec<SoilParticleId> {
    (0..count)
        .map(|i| SoilParticleId((u64::from(i) << 40) + 7))
        .collect()
}
fn materials() -> [SoilMaterialId; 3] {
    [
        SoilMaterialId(0),
        SoilMaterialId(1 << 50),
        SoilMaterialId(u64::MAX),
    ]
}
fn archive(state: &MpmSoilFrameCheckpoint) -> MpmSoilCanonicalTileArchive {
    state
        .partition_canonical_tiles(layout(), &ids(state.soil().count), &materials())
        .unwrap()
}
fn select(
    archive: &MpmSoilCanonicalTileArchive,
    state: &MpmSoilFrameCheckpoint,
) -> MpmSoilCanonicalSelection {
    archive
        .select(
            &archive.tiles().collect::<Vec<_>>(),
            129,
            state.world_origin(),
            state.soil().cfg.grid_origin,
        )
        .unwrap()
}
fn bytes(selection: &MpmSoilCanonicalSelection) -> Vec<u8> {
    selection
        .framed_selection()
        .to_frame_checkpoint()
        .unwrap()
        .to_bytes()
}
fn rehash(bytes: &mut [u8]) {
    let hash = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(hash.as_bytes());
}

#[test]
fn canonical_tiles_roundtrip_tiny_offsets_zero_signs_and_all_history() {
    for count in [0, 1, 129] {
        let mut source = state(count);
        if count > 0 {
            source.soil.words[1] = (-0.0f32).to_bits();
            source.soil.words[2] = 0.0f32.to_bits();
            source.origin[1] += 0.25;
        }
        let archive = archive(&source);
        let encoded = archive.to_bytes();
        assert_eq!(archive.particle_count(), count);
        assert_eq!(archive.material_ids(), &materials());
        assert_eq!(bytes(&select(&archive, &source)), source.to_bytes());
        let loaded = MpmSoilCanonicalTileArchive::from_bytes(&encoded, count).unwrap();
        assert_eq!(loaded.to_bytes(), encoded);
        assert_eq!(bytes(&select(&loaded, &source)), source.to_bytes());
        assert!(MpmSoilFrameTileArchive::from_bytes(&encoded, count).is_err());
        if count > 0 {
            assert!(MpmSoilCanonicalTileArchive::from_bytes(&encoded, count - 1).is_err());
            assert!(archive.metadata.words.capacity() == 0);
            let key = archive.tiles().next().unwrap();
            assert!(archive.tile_particle_count(key).is_some());
        }
    }
}

#[test]
fn canonical_selection_reframes_exactly_without_mutation_and_rejects_rounding() {
    let source = state(129);
    let archive = archive(&source);
    let original = archive.to_bytes();
    let keys: Vec<_> = archive.tiles().collect();
    let origin = source.world_origin();
    let grid = source.soil().cfg.grid_origin;
    let a = archive
        .select(&[keys[1], keys[1]], 129, origin, grid)
        .unwrap();
    assert_eq!(a.tiles().collect::<Vec<_>>(), vec![keys[1]]);
    let shifted_origin = [origin[0] + 0.25, origin[1], origin[2]];
    let shifted = archive
        .select(&[keys[1]], 129, shifted_origin, [-4.25, -4.0, -4.0])
        .unwrap();
    let before = a.framed_selection().soil();
    let after = shifted.framed_selection().soil();
    for (a, b) in before
        .stream(0)
        .chunks_exact(4)
        .zip(after.stream(0).chunks_exact(4))
    {
        assert_eq!(f32::from_bits(a[0]) - 0.25, f32::from_bits(b[0]));
        assert_eq!(&a[1..], &b[1..]);
    }
    for stream in 1..9 {
        assert_eq!(before.stream(stream), after.stream(stream));
    }
    assert!(
        archive
            .select(&keys[..1], 129, shifted_origin, grid)
            .is_err()
    );
    assert!(archive.select(&keys, 0, origin, grid).is_err());
    assert!(
        archive
            .select(&[SoilTileCoord([0; 3])], 129, origin, grid)
            .is_err()
    );
    assert!(archive.select(&[], 0, [f64::NAN; 3], grid).is_err());
    assert!(archive.select(&[], 0, origin, [f32::INFINITY; 3]).is_err());
    assert_eq!(archive.to_bytes(), original);
}

#[test]
fn canonical_writeback_merges_destinations_preserves_unloaded_and_stales_tokens() {
    let source = state(129);
    let mut archive = archive(&source);
    let keys: Vec<_> = archive.tiles().collect();
    let untouched = archive.tiles[&keys[3]].tile.words.clone();
    let selected = archive
        .select(&keys[1..2], 129, source.origin, source.soil.cfg.grid_origin)
        .unwrap();
    let mut updated = selected.framed.to_frame_checkpoint().unwrap();
    for p in updated.soil.words[..updated.soil.count as usize * 4].chunks_exact_mut(4) {
        p[0] = 1.125f32.to_bits();
    }
    let mut expected = source
        .partition_tiles(layout(), &ids(129), &materials())
        .unwrap();
    let old = expected.select(&keys[1..2], 129).unwrap();
    expected.writeback(&old, &updated).unwrap();
    archive.writeback(&selected, &updated).unwrap();
    assert_eq!(archive.particle_count(), 129);
    assert_eq!(archive.tile_particle_count(keys[1]), None);
    assert_eq!(archive.tiles[&keys[3]].tile.words, untouched);
    assert!(archive.writeback(&selected, &updated).is_err());
    let actual = select(&archive, &source)
        .framed
        .to_frame_checkpoint()
        .unwrap();
    let wanted = expected
        .select(&expected.tiles().collect::<Vec<_>>(), 129)
        .unwrap()
        .to_frame_checkpoint()
        .unwrap();
    assert_eq!(actual.to_bytes(), wanted.to_bytes());
    let current = select(&archive, &source);
    let data = archive.to_bytes();
    let mut reloaded = MpmSoilCanonicalTileArchive::from_bytes(&data, 129).unwrap();
    assert!(reloaded.writeback(&current, &actual).is_err());
    assert_eq!(reloaded.to_bytes(), data);
}

#[test]
fn canonical_writeback_rejects_changed_frame_mass_volume_material_and_config() {
    let source = state(129);
    let mut archive = archive(&source);
    let selected = select(&archive, &source);
    let initial = archive.to_bytes();
    for mutation in 0..6 {
        let mut updated = selected.framed.to_frame_checkpoint().unwrap();
        match mutation {
            0 => updated.origin[0] += 1.0,
            1 => updated.soil.cfg.grid_origin[0] += 1.0,
            2 => updated.soil.words[STRIDES[..4].iter().sum::<usize>() * 129] = 0.125f32.to_bits(),
            3 => updated.soil.words[STRIDES[..5].iter().sum::<usize>() * 129] = 0.125f32.to_bits(),
            4 => updated.soil.words[STRIDES[..7].iter().sum::<usize>() * 129] = 2,
            _ => updated.soil.words[0] = f32::NAN.to_bits(),
        }
        assert!(archive.writeback(&selected, &updated).is_err());
        assert_eq!(archive.to_bytes(), initial);
    }
}

#[test]
fn canonical_codec_rejects_corruption_truncation_and_valid_digest_invalid_payloads() {
    let initial = archive(&state(1)).to_bytes();
    for len in 0..initial.len() {
        assert!(MpmSoilCanonicalTileArchive::from_bytes(&initial[..len], 1).is_err());
    }
    for offset in 0..initial.len() {
        let mut altered = initial.clone();
        altered[offset] ^= 1;
        assert!(MpmSoilCanonicalTileArchive::from_bytes(&altered, 1).is_err());
    }
    let metadata_len = u32::from_le_bytes(
        initial[HEADER_BYTES + 44..HEADER_BYTES + 48]
            .try_into()
            .unwrap(),
    ) as usize;
    let tile = HEADER_BYTES + 48 + metadata_len + materials().len() * 8;
    let record = tile + 28;
    for (offset, replacement) in [
        (HEADER_BYTES + 32, 2u32.to_le_bytes().to_vec()),
        (tile, i64::MAX.to_le_bytes().to_vec()),
        (tile + 24, 0u32.to_le_bytes().to_vec()),
        (record + 8, f64::NAN.to_le_bytes().to_vec()),
        (record + 8, 1.0f64.to_le_bytes().to_vec()),
        (record + 32, vec![8]),
    ] {
        let mut altered = initial.clone();
        altered[offset..offset + replacement.len()].copy_from_slice(&replacement);
        rehash(&mut altered);
        assert!(MpmSoilCanonicalTileArchive::from_bytes(&altered, 1).is_err());
    }
    let mut trailing = initial;
    trailing.push(0);
    rehash(&mut trailing);
    assert!(MpmSoilCanonicalTileArchive::from_bytes(&trailing, 1).is_err());
}

#[test]
fn canonical_unsupported_anchor_or_residual_rejects_without_changing_source() {
    let mut source = state(1);
    let original = source.to_bytes();
    let unsupported = SoilTileLayout::new([0.0; 3], 0.1).unwrap();
    assert!(
        source
            .partition_canonical_tiles(unsupported, &ids(1), &materials())
            .is_err()
    );
    assert_eq!(source.to_bytes(), original);
    source.soil.words[0] = (-f32::MIN_POSITIVE).to_bits();
    assert!(
        source
            .partition_canonical_tiles(layout(), &ids(1), &materials())
            .is_err()
    );
}

#[test]
#[ignore = "requires GPU; canonical storage restores exact state/IDs in selected frames"]
fn canonical_gpu_capture_codec_reframe_restore_resume_and_writeback() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let device = engine.device().unwrap();
    for mode in 0..4 {
        let mut source = state(129);
        for p in source.soil.words[..129 * 4].chunks_exact_mut(4) {
            if f32::from_bits(p[0]) < 0.0 {
                p[0] = 0.375f32.to_bits();
            }
        }
        let initial = archive(&source);
        let selected = select(&initial, &source);
        let mut live =
            MpmSoilFrame::new(new_sim(&engine, source.soil.cfg, mode), [0.0; 3]).unwrap();
        live.restore_selection(&engine, selected.framed_selection())
            .unwrap();
        live.step_n_with(device, queue, 0.0001, 2).unwrap();
        let evolved = live.checkpoint_with(queue).unwrap();
        let canonical = archive(&evolved);
        let mut loaded =
            MpmSoilCanonicalTileArchive::from_bytes(&canonical.to_bytes(), 129).unwrap();
        assert_eq!(bytes(&select(&loaded, &evolved)), evolved.to_bytes());
        let origin = evolved.origin.map(|v| v + 0.000244140625);
        let grid = evolved.soil.cfg.grid_origin.map(|v| v - 0.000244140625);
        let keys: Vec<_> = loaded.tiles().collect();
        let selected = loaded.select(&keys, 129, origin, grid).unwrap();
        let mut restored =
            MpmSoilFrame::new(new_sim(&engine, source.soil.cfg, mode), [0.0; 3]).unwrap();
        restored
            .restore_selection(&engine, selected.framed_selection())
            .unwrap();
        assert_eq!(
            restored.checkpoint_with(queue).unwrap().to_bytes(),
            bytes(&selected)
        );
        let view = restored.simulation().soil_identity_buffers().unwrap();
        let words = crate::mpm::replay_tests::read_words(device, queue, view.particles);
        let expected: Vec<_> = ids(129)
            .iter()
            .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
            .collect();
        assert_eq!(&words[..expected.len()], expected);
        restored.step_n_with(device, queue, 0.0001, 1).unwrap();
        let updated = restored.checkpoint_with(queue).unwrap();
        loaded.writeback(&selected, &updated).unwrap();
        let reloaded = MpmSoilCanonicalTileArchive::from_bytes(&loaded.to_bytes(), 129).unwrap();
        let final_selection = reloaded
            .select(&reloaded.tiles().collect::<Vec<_>>(), 129, origin, grid)
            .unwrap();
        assert_eq!(bytes(&final_selection), updated.to_bytes());
    }
}
