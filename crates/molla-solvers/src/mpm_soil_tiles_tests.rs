use super::super::tests::fixture;
use super::*;
use molla_vehicle::{SoilRegionId, SoilRegionRequest, SoilResidencyLimits, SoilTileResidency};

fn ids(n: u32) -> Vec<SoilParticleId> {
    (0..n)
        .map(|i| SoilParticleId(1000 + u64::from(i)))
        .collect()
}
fn material_ids() -> [SoilMaterialId; 3] {
    [SoilMaterialId(100), SoilMaterialId(50), SoilMaterialId(77)]
}
fn layout() -> SoilTileLayout {
    SoilTileLayout::new([0.0; 3], 0.125).unwrap()
}

#[test]
fn all_streams_global_ids_and_layout_survive_partition_and_archive_roundtrip() {
    for count in [0, 1, 129] {
        let state = fixture(count);
        let archive = state
            .partition_tiles(layout(), &ids(count), &material_ids())
            .unwrap();
        let mut keys: Vec<_> = archive.tiles().collect();
        assert_eq!(archive.particle_count(), count);
        assert_eq!(
            keys.iter()
                .map(|&k| archive.tile_particle_count(k).unwrap())
                .sum::<usize>(),
            count as usize
        );
        let selection = archive.select(&keys, count).unwrap();
        assert_eq!(selection.checkpoint.words, state.words);
        assert_eq!(selection.ids, ids(count));
        assert_eq!(selection.material_ids(), material_ids());
        keys.extend(keys.clone());
        keys.reverse();
        assert_eq!(
            archive.select(&keys, count).unwrap().checkpoint.to_bytes(),
            state.to_bytes()
        );
        let bytes = archive.to_bytes();
        let loaded = MpmSoilTileArchive::from_bytes(&bytes, count).unwrap();
        assert_eq!(loaded.to_bytes(), bytes);
        assert_eq!(loaded.layout(), layout());
        assert_eq!(loaded.material_ids(), material_ids());
        if count > 0 {
            assert!(archive.select(&keys, count - 1).is_err());
            assert!(MpmSoilTileArchive::from_bytes(&bytes, count - 1).is_err());
        }
        assert!(archive.select(&[SoilTileCoord([999; 3])], count).is_err());
    }
}

#[test]
fn arbitrary_id_order_keeps_each_particle_record_intact() {
    let state = fixture(129);
    let mut particle_ids = ids(129);
    particle_ids.reverse();
    let archive = state
        .partition_tiles(layout(), &particle_ids, &material_ids())
        .unwrap();
    let keys: Vec<_> = archive.tiles().collect();
    let selected = archive.select(&keys, 129).unwrap();
    assert_eq!(selected.ids, ids(129));
    for (stream, stride) in STRIDES.into_iter().enumerate() {
        for output in 0..129 {
            let input = 128 - output;
            assert_eq!(
                &selected.checkpoint.stream(stream)[output * stride..(output + 1) * stride],
                &state.stream(stream)[input * stride..(input + 1) * stride]
            );
        }
    }
    assert!(
        state
            .partition_tiles(layout(), &[SoilParticleId(1); 129], &material_ids())
            .is_err()
    );
    assert!(
        state
            .partition_tiles(layout(), &ids(129), &[SoilMaterialId(1); 3])
            .is_err()
    );
    assert!(
        state
            .partition_tiles(layout(), &ids(128), &material_ids())
            .is_err()
    );
    assert!(
        state
            .partition_tiles(layout(), &ids(129), &material_ids()[..2])
            .is_err()
    );
    let tiny = SoilTileLayout::new([0.0; 3], f64::MIN_POSITIVE).unwrap();
    assert!(
        state
            .partition_tiles(tiny, &ids(129), &material_ids())
            .is_err()
    );
}

fn rehash(bytes: &mut [u8]) {
    let digest = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
}

#[test]
fn malformed_tile_archives_reject_before_partial_state_exists() {
    let archive = fixture(2)
        .partition_tiles(layout(), &ids(2), &material_ids())
        .unwrap();
    let bytes = archive.to_bytes();
    for n in 0..bytes.len() {
        assert!(MpmSoilTileArchive::from_bytes(&bytes[..n], 129).is_err());
    }
    for n in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[n] ^= 1;
        assert!(MpmSoilTileArchive::from_bytes(&changed, 129).is_err());
    }
    for (offset, data) in [
        (HEADER_BYTES, f64::NAN.to_le_bytes().to_vec()),
        (HEADER_BYTES + 24, 0_f64.to_le_bytes().to_vec()),
        (HEADER_BYTES + 32, u32::MAX.to_le_bytes().to_vec()),
        (HEADER_BYTES + 36, u32::MAX.to_le_bytes().to_vec()),
        (HEADER_BYTES + 40, u64::MAX.to_le_bytes().to_vec()),
        (HEADER_BYTES + 48 + 8, 100_u64.to_le_bytes().to_vec()),
        (HEADER_BYTES + 48 + 24 + 8, 1000_u64.to_le_bytes().to_vec()),
    ] {
        let mut changed = bytes.clone();
        changed[offset..offset + data.len()].copy_from_slice(&data);
        rehash(&mut changed);
        assert!(
            MpmSoilTileArchive::from_bytes(&changed, 129).is_err(),
            "offset {offset}"
        );
    }
    let mut trailing = bytes;
    trailing.push(0);
    rehash(&mut trailing);
    assert!(MpmSoilTileArchive::from_bytes(&trailing, 129).is_err());
}

#[test]
fn writeback_merges_moved_particles_and_rejects_stale_or_nonconservative_updates() {
    let state = fixture(2);
    let mut archive = state
        .partition_tiles(layout(), &ids(2), &material_ids())
        .unwrap();
    let keys: Vec<_> = archive.tiles().collect();
    assert_eq!(keys.len(), 2);
    let selected = archive.select(&keys[..1], 2).unwrap();
    assert_eq!(selected.ids, vec![SoilParticleId(1000)]);
    let snapshot = archive.to_bytes();
    let mut foreign = MpmSoilTileArchive::from_bytes(&snapshot, 2).unwrap();
    assert!(foreign.writeback(&selected, selected.checkpoint()).is_err());
    for stream in [4, 5, 7] {
        let mut changed = selected.checkpoint.clone();
        let offset = STRIDES[..stream].iter().sum::<usize>();
        changed.words[offset] = if stream == 7 { 2 } else { 2_f32.to_bits() };
        assert!(archive.writeback(&selected, &changed).is_err());
        assert_eq!(archive.to_bytes(), snapshot);
    }
    let mut moved = selected.checkpoint.clone();
    moved.words[..4].copy_from_slice(&state.stream(0)[4..8]);
    moved.words[4] = 1_f32.to_bits();
    archive.writeback(&selected, &moved).unwrap();
    assert!(archive.tile_particle_count(keys[0]).is_none());
    assert_eq!(archive.tile_particle_count(keys[1]), Some(2));
    let after = archive.select(&keys[1..], 2).unwrap();
    assert_eq!(after.ids, ids(2));
    for (stream, stride) in STRIDES.into_iter().enumerate() {
        assert_eq!(
            &after.checkpoint.stream(stream)[..stride],
            moved.stream(stream)
        );
        assert_eq!(
            &after.checkpoint.stream(stream)[stride..],
            &state.stream(stream)[stride..]
        );
    }
    let bytes = archive.to_bytes();
    assert!(archive.writeback(&selected, selected.checkpoint()).is_err());
    assert_eq!(archive.to_bytes(), bytes);
    assert_eq!(
        MpmSoilTileArchive::from_bytes(&bytes, 2)
            .unwrap()
            .to_bytes(),
        bytes
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn overlapping_regions_restore_unique_gpu_particles_and_revisit_exact_history() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let state = fixture(129);
    let mut source = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    source.restore_soil_with(queue, &state).unwrap();
    source.step_n_with(device, queue, 0.0001, 3).unwrap();
    let evolved = source.checkpoint_soil_with(queue).unwrap();
    let archive = evolved
        .partition_tiles(layout(), &ids(129), &material_ids())
        .unwrap();
    let bytes = archive.to_bytes();
    drop(source);
    drop(archive);
    let mut archive = MpmSoilTileArchive::from_bytes(&bytes, 129).unwrap();
    let keys: Vec<_> = archive.tiles().collect();
    assert!(keys.len() >= 3);
    let mut residency = SoilTileResidency::new(SoilResidencyLimits {
        regions: 2,
        tiles: keys.len(),
        claims: keys.len() * 2,
    })
    .unwrap();
    let regions = [
        SoilRegionRequest {
            id: SoilRegionId(4),
            tiles: &keys[..2],
        },
        SoilRegionRequest {
            id: SoilRegionId(1),
            tiles: &keys[1..3],
        },
    ];
    let first = residency.plan(&regions, &[]).unwrap();
    let mut requests = regions[0].tiles.to_vec();
    requests.extend_from_slice(regions[1].tiles);
    let selected = archive.select(&requests, 129).unwrap();
    let unique_count: usize = keys[..3]
        .iter()
        .map(|&key| archive.tile_particle_count(key).unwrap())
        .sum();
    assert_eq!(selected.ids.len(), unique_count);
    let mut active = MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap();
    active
        .restore_soil_with(queue, selected.checkpoint())
        .unwrap();
    residency.commit(first).unwrap();
    let captured = active.checkpoint_soil_with(queue).unwrap();
    assert_eq!(captured.to_bytes(), selected.checkpoint.to_bytes());
    active.step_n_with(device, queue, 0.0001, 2).unwrap();
    let updated = active.checkpoint_soil_with(queue).unwrap();
    assert_ne!(updated.stream(0), captured.stream(0));
    archive.writeback(&selected, &updated).unwrap();
    let selected = archive.select(&requests, 129).unwrap();
    assert_eq!(selected.checkpoint.to_bytes(), updated.to_bytes());
    let second = residency
        .plan(
            &[SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys[1..3],
            }],
            &[],
        )
        .unwrap();
    assert!(second.activate().is_empty());
    assert_eq!(second.deactivate(), &keys[..1]);
    let other = archive
        .select(&second.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    active.restore_soil_with(queue, other.checkpoint()).unwrap();
    residency.commit(second).unwrap();
    let returning = residency.plan(&regions, &[]).unwrap();
    active
        .restore_soil_with(queue, selected.checkpoint())
        .unwrap();
    residency.commit(returning).unwrap();
    assert_eq!(
        active.checkpoint_soil_with(queue).unwrap().to_bytes(),
        updated.to_bytes()
    );
    println!(
        "{} archive tiles, {unique_count} unique particles for overlapping region requests; exact GPU unload/reload history",
        keys.len()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn moved_particle_changes_archive_tile_without_changing_identity_or_mass() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut state = fixture(1);
    state.words[0] = (-0.00001_f32).to_bits();
    state.words[4] = 1_f32.to_bits();
    state.cfg.gravity_y = 0.0;
    let id = [SoilParticleId(u64::MAX)];
    let mut before = state
        .partition_tiles(layout(), &id, &material_ids())
        .unwrap();
    let old_key = before.tiles().next().unwrap();
    let selected_before = before.select(&[old_key], 1).unwrap();
    assert_eq!(old_key.0[0], -1);
    let mut sim = MpmSimulation::new(&engine, state.cfg).unwrap();
    sim.restore_soil_with(queue, &state).unwrap();
    sim.step_n_with(device, queue, 0.0001, 2).unwrap();
    let moved = sim.checkpoint_soil_with(queue).unwrap();
    let after = moved
        .partition_tiles(layout(), &id, &material_ids())
        .unwrap();
    let new_key = after.tiles().next().unwrap();
    before.writeback(&selected_before, &moved).unwrap();
    assert_eq!(before.to_bytes(), after.to_bytes());
    assert_eq!(new_key.0[0], 0);
    assert!(after.tile_particle_count(old_key).is_none());
    let selected = after.select(&[new_key, new_key], 1).unwrap();
    assert_eq!(selected.ids, id);
    assert_eq!(selected.checkpoint.words, moved.words);
    assert_eq!(moved.stream(4), state.stream(4));
    assert_eq!(moved.stream(5), state.stream(5));
    assert_eq!(moved.stream(7), state.stream(7));
    assert_eq!(moved.stream(8), state.stream(8));
    println!(
        "particle u64::MAX crossed negative-to-positive tile boundary after two GPU substeps; identity, mass, volume, material and compaction retained"
    );
}
