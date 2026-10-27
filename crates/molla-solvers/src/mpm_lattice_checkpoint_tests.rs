use super::*;
use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};

fn offsets(state: &MpmSoilCheckpoint, cells: [i32; 3]) -> MpmSoilCheckpoint {
    let mut value = state.clone();
    value.grid_window_cells = cells;
    value
}

#[test]
fn lattice_checkpoint_signed_offsets_roundtrip_and_frame_envelope() {
    for count in [0, 1, 129] {
        let original = tests::fixture(count);
        let old = original.to_bytes();
        assert_eq!(&old[8..12], &1u32.to_le_bytes());
        assert_eq!(
            MpmSoilCheckpoint::from_bytes(&old, count)
                .unwrap()
                .to_bytes(),
            old
        );
        for cells in [[1, 0, 0], [-1, 16, -32], [i32::MIN, i32::MAX, 7]] {
            let state = offsets(&original, cells);
            let bytes = state.to_bytes();
            assert_eq!(&bytes[8..12], &2u32.to_le_bytes());
            assert_eq!(bytes.len(), old.len() + 12);
            let decoded = MpmSoilCheckpoint::from_bytes(&bytes, count).unwrap();
            assert_eq!(decoded.grid_window_cells(), cells);
            assert_eq!(decoded.config(), original.config());
            assert_eq!(decoded.words, original.words);
            assert_eq!(decoded.to_bytes(), bytes);
            let frame = MpmSoilFrameCheckpoint::new(decoded, [1e12, -1e12, 0.0]).unwrap();
            let restored = MpmSoilFrameCheckpoint::from_bytes(&frame.to_bytes(), count).unwrap();
            assert_eq!(restored.to_bytes(), frame.to_bytes());
            assert_eq!(restored.soil().grid_window_cells(), cells);
        }
        assert_eq!(original.to_bytes(), old);
    }
}

#[test]
fn lattice_checkpoint_integrity_version_and_bounds() {
    let state = offsets(&tests::fixture(2), [-1, 2, 3]);
    let bytes = state.to_bytes();
    assert!(MpmSoilCheckpoint::from_bytes(&bytes, 1).is_err());
    for length in 0..bytes.len() {
        assert!(MpmSoilCheckpoint::from_bytes(&bytes[..length], 2).is_err());
    }
    for index in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 1;
        assert!(MpmSoilCheckpoint::from_bytes(&corrupt, 2).is_err());
    }
    let mut downgrade = bytes.clone();
    downgrade[8..12].copy_from_slice(&1u32.to_le_bytes());
    assert!(MpmSoilCheckpoint::from_bytes(&downgrade, 2).is_err());
    let mut zero = bytes.clone();
    zero[44..56].fill(0);
    let digest = checkpoint_digest(2, &zero[44..]);
    zero[12..44].copy_from_slice(digest.as_bytes());
    assert!(MpmSoilCheckpoint::from_bytes(&zero, 2).is_err());
    let mut trailing = bytes.clone();
    trailing.extend_from_slice(&0u32.to_le_bytes());
    let digest = checkpoint_digest(2, &trailing[44..]);
    trailing[12..44].copy_from_slice(digest.as_bytes());
    assert!(MpmSoilCheckpoint::from_bytes(&trailing, 2).is_err());
}

#[test]
fn lattice_checkpoint_legacy_consumers_reject_without_dropping_offsets() {
    let original = tests::fixture(2);
    let changed = offsets(&original, [1, 0, 0]);
    assert!(changed.require_origin_only_grid().is_err());
    assert!(changed.recentered(original.cfg.grid_origin).is_err());
    assert!(changed.with_grid_window(original.cfg.grid_origin).is_err());
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let ids = [SoilParticleId(1), SoilParticleId(2)];
    let materials = [SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)];
    assert!(changed.partition_tiles(layout, &ids, &materials).is_err());
    let framed = MpmSoilFrameCheckpoint::new(changed.clone(), [0.0; 3]).unwrap();
    let canonical = framed
        .partition_canonical_tiles(layout, &ids, &materials)
        .unwrap();
    let selected = canonical
        .select_window(
            &canonical.tiles().collect::<Vec<_>>(),
            2,
            [0.0; 3],
            original.cfg.grid_origin,
            [1, 0, 0],
        )
        .unwrap();
    assert_eq!(
        selected.framed_selection().soil().to_bytes(),
        changed.to_bytes()
    );
    assert!(
        changed
            .activity_since(&original, &ids, &ids, 0.001, 1)
            .is_err()
    );
    let mut archive = original.partition_tiles(layout, &ids, &materials).unwrap();
    let keys = archive.tiles().collect::<Vec<_>>();
    let selection = archive.select(&keys, 2).unwrap();
    let before = archive.to_bytes();
    assert!(archive.writeback(&selection, &changed).is_err());
    assert_eq!(archive.to_bytes(), before);
    let framed = MpmSoilFrameCheckpoint::new(original, [0.0; 3]).unwrap();
    let mut archive = framed
        .partition_canonical_tiles(layout, &ids, &materials)
        .unwrap();
    let keys = archive.tiles().collect::<Vec<_>>();
    let selection = archive
        .select(&keys, 2, [0.0; 3], framed.soil().cfg.grid_origin)
        .unwrap();
    let before = archive.to_bytes();
    let mut updated = selection.framed_selection().to_frame_checkpoint().unwrap();
    updated.soil.grid_window_cells = [1, 0, 0];
    assert!(archive.writeback(&selection, &updated).is_err());
    assert_eq!(archive.to_bytes(), before);
}

#[test]
#[ignore = "requires the frozen pre-migration 50k checkpoint path"]
fn lattice_checkpoint_frozen_legacy_byte_identity() {
    let path = std::env::var_os("SOIL_LATTICE_LEGACY").unwrap();
    let bytes = std::fs::read(path).unwrap();
    let original = MpmSoilCheckpoint::from_bytes(&bytes, 50_000).unwrap();
    assert_eq!(original.grid_window_cells(), [0; 3]);
    assert_eq!(original.particle_count(), 50_000);
    assert_eq!(original.to_bytes(), bytes);
    let shifted = offsets(&original, [1, 0, -1]);
    let decoded = MpmSoilCheckpoint::from_bytes(&shifted.to_bytes(), 50_000).unwrap();
    assert_eq!(decoded.words, original.words);
    assert_eq!(decoded.grid_window_cells(), [1, 0, -1]);
    println!(
        "legacy_50k_bytes={} lattice_bytes={} particle_words_exact=true",
        bytes.len(),
        shifted.to_bytes().len()
    );
}
