use super::super::super::tests::{fixture, new_sim};
use super::*;

fn frame(count: u32, tiny: bool) -> MpmSoilFrameCheckpoint {
    let mut state = fixture(count);
    state.cfg.grid_size = 64;
    state.cfg.grid_origin = [-4.0; 3];
    state.cfg.cell_size = 0.125;
    for (i, p) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        for value in &mut p[..3] {
            *value = if tiny {
                [-1e-6f32, 0.0, 1e-6][i % 3].to_bits()
            } else {
                (2.0 + (i % 17) as f32 / 64.0).to_bits()
            };
        }
    }
    MpmSoilFrameCheckpoint::new(state, [1e12; 3]).unwrap()
}

fn ids(n: u32) -> Vec<SoilParticleId> {
    (0..n)
        .map(|i| SoilParticleId(u64::MAX - u64::from(n - 1 - i)))
        .collect()
}

fn materials() -> [SoilMaterialId; 3] {
    [
        SoilMaterialId(0),
        SoilMaterialId(u64::MAX),
        SoilMaterialId(77),
    ]
}

fn layout() -> SoilTileLayout {
    SoilTileLayout::new([0.0; 3], 1.0).unwrap()
}

fn archive(frame: &MpmSoilFrameCheckpoint) -> MpmSoilFrameTileArchive {
    frame
        .partition_tiles(layout(), &ids(frame.soil().count), &materials())
        .unwrap()
}

fn rehash(bytes: &mut [u8]) {
    let hash = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(hash.as_bytes());
}

#[test]
fn framed_tiles_preserve_offsets_ids_history_and_bounded_codec() {
    for count in [0, 1, 129] {
        let initial = frame(count, true);
        let a = archive(&initial);
        assert_eq!(a.particle_count(), count);
        assert_eq!(a.world_origin(), [1e12; 3]);
        assert_eq!(a.layout(), layout());
        assert_eq!(a.material_ids(), materials());
        let mut keys: Vec<_> = a.tiles().collect();
        assert_eq!(
            keys.iter()
                .map(|&k| a.tile_particle_count(k).unwrap())
                .sum::<usize>(),
            count as usize
        );
        keys.extend(keys.clone());
        keys.reverse();
        let selected = a.select(&keys, count).unwrap();
        assert_eq!(selected.particle_ids(), ids(count));
        assert_eq!(selected.material_ids(), materials());
        assert_eq!(selected.world_origin(), initial.world_origin());
        assert_eq!(selected.soil().words, initial.soil().words);
        assert_eq!(
            selected.to_frame_checkpoint().unwrap().to_bytes(),
            initial.to_bytes()
        );
        let bytes = a.to_bytes();
        let decoded = MpmSoilFrameTileArchive::from_bytes(&bytes, count).unwrap();
        assert_eq!(decoded.to_bytes(), bytes);
        assert_eq!(
            decoded.tiles().collect::<Vec<_>>(),
            a.tiles().collect::<Vec<_>>()
        );
        assert_eq!(
            decoded.select(&keys, count).unwrap().soil().words,
            initial.soil().words
        );
        assert!(MpmSoilTileArchive::from_bytes(&bytes, count).is_err());
        if count != 0 {
            assert!(a.select(&keys, count - 1).is_err());
            assert!(MpmSoilFrameTileArchive::from_bytes(&bytes, count - 1).is_err());
            assert!(a.tiles().any(|t| t == SoilTileCoord([999_999_999_999; 3])));
        }
        assert!(a.select(&[SoilTileCoord([7; 3])], count).is_err());
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSoilFrameSelection>();
    send_sync::<MpmSoilFrameTileArchive>();
}

#[test]
fn framed_tiles_sort_complete_records_and_reject_incomplete_identity() {
    let initial = frame(129, true);
    let mut reversed = ids(129);
    reversed.reverse();
    let a = initial
        .partition_tiles(layout(), &reversed, &materials())
        .unwrap();
    let selected = a.select(&a.tiles().collect::<Vec<_>>(), 129).unwrap();
    assert_eq!(selected.particle_ids(), ids(129));
    for (stream, stride) in STRIDES.into_iter().enumerate() {
        for out in 0..129 {
            let input = 128 - out;
            assert_eq!(
                &selected.soil().stream(stream)[out * stride..(out + 1) * stride],
                &initial.soil().stream(stream)[input * stride..(input + 1) * stride]
            );
        }
    }
    assert!(
        initial
            .partition_tiles(layout(), &ids(128), &materials())
            .is_err()
    );
    assert!(
        initial
            .partition_tiles(layout(), &[SoilParticleId(0); 129], &materials())
            .is_err()
    );
    assert!(
        initial
            .partition_tiles(layout(), &ids(129), &[SoilMaterialId(0); 3])
            .is_err()
    );
}

#[test]
fn framed_tiles_writeback_crosses_world_boundary_and_preserves_untouched_records() {
    let initial = frame(3, true);
    let mut a = archive(&initial);
    let left = SoilTileCoord([999_999_999_999; 3]);
    let right = SoilTileCoord([1_000_000_000_000; 3]);
    let selected = a.select(&[left, left], 1).unwrap();
    let mut soil = selected.soil().clone();
    for value in &mut soil.words[..3] {
        *value = 0.125f32.to_bits();
    }
    let updated = MpmSoilFrameCheckpoint::new(soil, selected.world_origin()).unwrap();
    a.writeback(&selected, &updated).unwrap();
    assert_eq!(a.tiles().collect::<Vec<_>>(), vec![right]);
    assert_eq!(a.tile_particle_count(right), Some(3));
    let joined = a.select(&[right], 3).unwrap();
    assert_eq!(joined.particle_ids(), ids(3));
    for (stream, stride) in STRIDES.into_iter().enumerate() {
        assert_eq!(
            &joined.soil().stream(stream)[..stride],
            updated.soil().stream(stream)
        );
        assert_eq!(
            &joined.soil().stream(stream)[stride..],
            &initial.soil().stream(stream)[stride..]
        );
    }
    let bytes = a.to_bytes();
    assert!(a.writeback(&selected, &updated).is_err());
    assert_eq!(a.to_bytes(), bytes);
    let loaded = MpmSoilFrameTileArchive::from_bytes(&bytes, 3).unwrap();
    assert_eq!(
        loaded.select(&[right], 3).unwrap().soil().words,
        joined.soil().words
    );
}

#[test]
fn framed_tile_writeback_failures_do_not_mutate_archive() {
    let initial = frame(3, true);
    let mut a = archive(&initial);
    let keys: Vec<_> = a.tiles().collect();
    let selected = a.select(&keys, 3).unwrap();
    let bytes = a.to_bytes();
    let changed_origin =
        MpmSoilFrameCheckpoint::new(initial.soil().clone(), [1e12 + 1.0; 3]).unwrap();
    assert!(a.writeback(&selected, &changed_origin).is_err());
    let foreign = archive(&initial).select(&keys, 3).unwrap();
    assert!(a.writeback(&foreign, &initial).is_err());
    for stream in [4, 5, 7] {
        let mut soil = initial.soil().clone();
        let start = STRIDES[..stream].iter().sum::<usize>() * 3;
        soil.words[start] = if stream == 7 {
            (soil.words[start] + 1) % 3
        } else {
            (f32::from_bits(soil.words[start]) * 2.0).to_bits()
        };
        let changed = MpmSoilFrameCheckpoint::new(soil, initial.world_origin()).unwrap();
        assert!(a.writeback(&selected, &changed).is_err());
        assert_eq!(a.to_bytes(), bytes);
    }
    let mut soil = initial.soil().clone();
    soil.words[0] = f32::MAX.to_bits();
    let outside = MpmSoilFrameCheckpoint::new(soil, initial.world_origin()).unwrap();
    assert!(a.writeback(&selected, &outside).is_err());
    assert_eq!(a.to_bytes(), bytes);
}

#[test]
fn framed_tile_recenter_preserves_membership_history_and_rejects_stale_selections() {
    let initial = frame(129, false);
    let a = archive(&initial);
    let bytes = a.to_bytes();
    let keys: Vec<_> = a.tiles().collect();
    let selected = a.select(&keys, 129).unwrap();
    let mut shifted = a.recentered([-8.0; 3]).unwrap();
    assert_eq!(shifted.tiles().collect::<Vec<_>>(), keys);
    assert_eq!(shifted.world_origin(), [1e12 + 4.0; 3]);
    let replacement = shifted.select(&keys, 129).unwrap();
    assert_eq!(replacement.particle_ids(), selected.particle_ids());
    let expected = initial.recentered([-8.0; 3]).unwrap();
    assert_eq!(
        replacement.to_frame_checkpoint().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert!(shifted.writeback(&selected, &expected).is_err());
    shifted.writeback(&replacement, &expected).unwrap();
    assert_eq!(shifted.recentered([-4.0; 3]).unwrap().to_bytes(), bytes);
    assert!(a.recentered([16_777_216.0; 3]).is_err());
    assert_eq!(a.to_bytes(), bytes);
    let same = a.recentered([-4.0; 3]).unwrap();
    let mut same = same;
    assert!(same.writeback(&selected, &initial).is_err());
    assert_eq!(same.to_bytes(), bytes);
}

#[test]
fn framed_tile_codec_rejects_corruption_versions_bounds_and_nonfinite_offsets() {
    let a = archive(&frame(3, true));
    let bytes = a.to_bytes();
    for length in [0, 8, HEADER_BYTES, FRAMED_HEADER, bytes.len() - 1] {
        assert!(MpmSoilFrameTileArchive::from_bytes(&bytes[..length], 3).is_err());
    }
    for index in [0, 8, 12, HEADER_BYTES, FRAMED_HEADER + 8, bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert!(MpmSoilFrameTileArchive::from_bytes(&bad, 3).is_err());
        rehash(&mut bad);
        if index == 8 || index == FRAMED_HEADER + 8 {
            assert!(MpmSoilFrameTileArchive::from_bytes(&bad, 3).is_err());
        }
    }
    for origin in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e30] {
        let mut bad = bytes.clone();
        bad[HEADER_BYTES..HEADER_BYTES + 8].copy_from_slice(&origin.to_le_bytes());
        rehash(&mut bad);
        assert!(MpmSoilFrameTileArchive::from_bytes(&bad, 3).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    rehash(&mut trailing);
    assert!(MpmSoilFrameTileArchive::from_bytes(&trailing, 3).is_err());
    assert!(MpmSoilFrameTileArchive::from_bytes(&a.archive.to_bytes(), 3).is_err());
    let empty = MpmSoilFrameCheckpoint::new(fixture(0), [-0.0; 3]).unwrap();
    let zero = archive(&empty);
    assert_eq!(
        MpmSoilFrameTileArchive::from_bytes(&zero.to_bytes(), 0)
            .unwrap()
            .world_origin()
            .map(f64::to_bits),
        [-0.0f64; 3].map(f64::to_bits)
    );
}

#[test]
#[ignore = "requires GPU; framed archive reload of evolved local soil"]
fn framed_tile_archive_gpu_evolved_capture_reload_and_resume() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let initial = frame(129, false);
        let sim = new_sim(&engine, initial.soil().cfg, mode);
        let mut live = MpmSoilFrame::new(sim, initial.world_origin()).unwrap();
        live.restore_with(queue, &initial).unwrap();
        live.step_n_with(device, queue, 0.0001, 2).unwrap();
        let evolved = live.checkpoint_with(queue).unwrap();
        assert_ne!(evolved.soil().words, initial.soil().words);
        let stored = archive(&evolved);
        let loaded = MpmSoilFrameTileArchive::from_bytes(&stored.to_bytes(), 129).unwrap();
        let selected = loaded
            .select(&loaded.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        assert_eq!(selected.particle_ids(), ids(129));
        assert_eq!(selected.soil().words, evolved.soil().words);
        let sim = new_sim(&engine, selected.soil().cfg, mode);
        let mut restored = MpmSoilFrame::new(sim, selected.world_origin()).unwrap();
        restored
            .restore_with(queue, &selected.to_frame_checkpoint().unwrap())
            .unwrap();
        assert_eq!(
            restored.checkpoint_with(queue).unwrap().to_bytes(),
            evolved.to_bytes()
        );
        restored.step_n_with(device, queue, 0.0001, 2).unwrap();
        let resumed = restored.checkpoint_with(queue).unwrap();
        resumed.soil().validate().unwrap();
        assert_eq!(resumed.soil().count, 129);
        assert_eq!(resumed.world_origin(), evolved.world_origin());
    }
}
