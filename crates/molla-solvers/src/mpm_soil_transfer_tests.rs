use super::*;
use crate::mpm::{checkpoint::tests::fixture, replay_tests::read_words};
use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};
use std::time::{Duration, Instant};

fn archive(count: u32) -> MpmSoilTileArchive {
    let mut state = fixture(count);
    state.cfg.max_particles = count.max(1);
    for (index, position) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        position[0] = if index % 2 == 0 {
            (-0.13_f32).to_bits()
        } else {
            0.02_f32.to_bits()
        };
        position[1] = 0.3_f32.to_bits();
        position[2] = (-0.03_f32).to_bits();
    }
    let ids: Vec<_> = (0..count)
        .map(|i| {
            SoilParticleId(if i + 1 == count {
                u64::MAX
            } else {
                u64::from(i) << 40
            })
        })
        .collect();
    state
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 0.125).unwrap(),
            &ids,
            &[
                SoilMaterialId(u64::MAX),
                SoilMaterialId(0),
                SoilMaterialId(1 << 62),
            ],
        )
        .unwrap()
}

fn all(archive: &MpmSoilTileArchive) -> MpmSoilSelection {
    archive
        .select(
            &archive.tiles().collect::<Vec<_>>(),
            archive.particle_count(),
        )
        .unwrap()
}

fn check(engine: &Engine, simulation: &mut MpmSimulation, expected: &MpmSoilSelection) {
    let actual = simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert_eq!(actual.words, expected.checkpoint().words);
    let snapshot = simulation.soil_identity_snapshot().unwrap();
    assert_eq!(snapshot.particle_ids(), expected.particle_ids());
    assert_eq!(snapshot.material_ids(), expected.material_ids());
    let view = simulation.soil_identity_buffers().unwrap();
    let raw = read_words(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        view.particles,
    );
    let packed: Vec<u32> = expected
        .particle_ids()
        .iter()
        .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
        .collect();
    assert_eq!(&raw[..packed.len()], packed);
}

#[test]
fn mapping_is_linear_exact_and_import_payload_contains_only_new_records() {
    for count in [0, 1, 63, 64, 65, 129, 257] {
        let archive = archive(count);
        let full = all(&archive);
        let empty = archive.select(&[], count).unwrap();
        let empty_ids = MpmSoilIdentitySnapshot::from_selection(&empty);
        let initial = transfer_upload(&empty_ids, &full).unwrap();
        assert_eq!(initial.stats.imported_particles, count);
        assert_eq!(initial.stats.upload_bytes, 16 + u64::from(count) * 172);
        let full_ids = MpmSoilIdentitySnapshot::from_selection(&full);
        let retained = transfer_upload(&full_ids, &full).unwrap();
        assert_eq!(retained.stats.retained_particles, count);
        assert_eq!(retained.stats.upload_bytes, 16 + u64::from(count) * 4);
        assert_eq!(&retained.words[4..], &(0..count).collect::<Vec<_>>());
        let cleared = transfer_upload(&full_ids, &empty).unwrap();
        assert_eq!(cleared.stats.removed_particles, count);
        assert_eq!(cleared.stats.dispatches, 0);
        if count > 1 {
            let keys: Vec<_> = archive.tiles().collect();
            let even = archive.select(&keys[..1], count).unwrap();
            let ids = MpmSoilIdentitySnapshot::from_selection(&even);
            let update = transfer_upload(&ids, &full).unwrap();
            assert_eq!(update.stats.imported_particles, count / 2);
            for index in 0..count {
                assert_eq!(
                    update.words[4 + index as usize],
                    index / 2 | if index % 2 == 0 { 0 } else { IMPORT }
                );
            }
            for (record, index) in (1..count as usize).step_by(2).enumerate() {
                let start = 4 + count as usize + record * 42;
                let mut expected = Vec::new();
                for (stream, stride) in STRIDES.into_iter().enumerate() {
                    expected.extend_from_slice(
                        &full.checkpoint().stream(stream)[index * stride..(index + 1) * stride],
                    );
                }
                let id = full.particle_ids()[index].0;
                expected.extend_from_slice(&[id as u32, (id >> 32) as u32]);
                assert_eq!(&update.words[start..start + 42], expected);
            }
        }
    }
}

#[test]
fn capacity_limits_bound_scratch_indexing_and_dispatches() {
    let mut limits = wgpu::Limits::default();
    assert_eq!(
        sizes(50_000, &limits).unwrap().into_iter().sum::<u64>(),
        25_400_016
    );
    assert!(sizes(u32::MAX, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = 1;
    assert!(sizes(64, &limits).is_ok());
    assert!(sizes(65, &limits).is_err());
    limits.max_buffer_size = 100;
    assert!(sizes(1, &limits).is_err());
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn evolved_history_interleaved_import_compaction_empty_and_refill_are_exact() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let mut archive = archive(129);
    let keys: Vec<_> = archive.tiles().collect();
    let old = archive.select(&keys[..1], 129).unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, old.checkpoint().config(), 128).unwrap();
    sim.restore_soil_selection(&engine, &old).unwrap();
    let transfer = SoilTransfer::new(&engine, &sim).unwrap();
    assert_eq!(transfer.bytes(), 129 * 508 + 16);
    sim.step_n(&engine, 0.00001, 2).unwrap();
    let evolved = sim.checkpoint_soil_with(engine.queue().unwrap()).unwrap();
    archive.writeback(&old, &evolved).unwrap();
    let full = all(&archive);
    let old_epoch = sim.soil_identity_snapshot().unwrap().clone();
    let old_buffer = sim.soil_identity_buffers().unwrap().particles.clone();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let stats = transfer.apply(&engine, &mut sim, &full).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    let measured = transfers.finish();
    assert_eq!(measured.upload_bytes, stats.upload_bytes);
    assert_eq!(measured.readback_bytes, 0);
    assert_eq!(
        (stats.retained_particles, stats.imported_particles),
        (65, 64)
    );
    assert_eq!(stats.upload_bytes, 16 + 4 * 129 + 168 * 64);
    assert!(!sim.is_soil_identity_current(&old_epoch));
    assert_eq!(sim.soil_identity_buffers().unwrap().particles, &old_buffer);
    check(&engine, &mut sim, &full);
    let subset = archive.select(&keys[1..], 129).unwrap();
    let empty = archive.select(&[], 129).unwrap();
    let mut readback = sim
        .prepare_soil_checkpoint_readback(&engine, 129, 3, 3 * 20656)
        .unwrap();
    for (revision, selection) in [&subset, &empty, &full].into_iter().enumerate() {
        let allocations = DeviceScope::start().unwrap();
        let stats = transfer.apply(&engine, &mut sim, selection).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        if revision == 0 {
            assert_eq!(stats.removed_particles, 65);
        }
        if revision == 1 {
            assert_eq!(stats.removed_particles, 64);
            assert_eq!(stats.dispatches, 0);
        }
        assert!(
            readback
                .try_capture(&sim, revision as u64)
                .unwrap()
                .is_some()
        );
    }
    drop(transfer);
    let deadline = Instant::now() + Duration::from_secs(5);
    for expected in [&subset, &empty, &full] {
        let completion = loop {
            if let Some(completion) = readback.poll_next().unwrap() {
                break completion;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(
            completion.identity().unwrap().particle_ids(),
            expected.particle_ids()
        );
        assert_eq!(
            completion.into_result().unwrap().words,
            expected.checkpoint().words
        );
    }
    check(&engine, &mut sim, &full);
    sim.step_n(&engine, 0.00001, 1).unwrap();
    println!(
        "129 rows: 65 retained + 64 imports; 65 removals, empty, refill; FIFO histories exact"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn fifty_thousand_unchanged_particles_need_no_gpu_transfer() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let full = all(&archive(50_000));
    let mut sim = MpmSimulation::new_sparse(&engine, full.checkpoint().config(), 128).unwrap();
    sim.restore_soil_selection(&engine, &full).unwrap();
    let transfer = SoilTransfer::new(&engine, &sim).unwrap();
    let buffers = [
        transfer.previous.clone(),
        transfer.upload.clone(),
        transfer.output.clone(),
    ];
    assert_eq!(transfer.bytes(), 25_400_016);
    let epoch = sim.soil_identity_snapshot().unwrap().clone();
    for _ in 0..3 {
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let stats = transfer.apply(&engine, &mut sim, &full).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let measured = transfers.finish();
        assert_eq!((measured.upload_bytes, measured.readback_bytes), (0, 0));
        assert_eq!(
            (
                stats.retained_particles,
                stats.imported_particles,
                stats.removed_particles
            ),
            (50_000, 0, 0)
        );
        assert_eq!(stats.dispatches, 0);
        assert!(sim.is_soil_identity_current(&epoch));
    }
    assert_eq!(
        [&transfer.previous, &transfer.upload, &transfer.output],
        [&buffers[0], &buffers[1], &buffers[2]]
    );
    check(&engine, &mut sim, &full);
    println!(
        "50k unchanged selection: 0 uploads/readbacks/dispatches/hot application buffers; identity epoch preserved; all history/ID words exact"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn foreign_owner_and_pending_material_changes_reject_before_transfers() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let selected = all(&archive(1));
    let mut sim = MpmSimulation::new_sparse(&engine, selected.checkpoint().config(), 128).unwrap();
    sim.restore_soil_selection(&engine, &selected).unwrap();
    let transfer = SoilTransfer::new(&engine, &sim).unwrap();
    let mut other =
        MpmSimulation::new_sparse(&engine, selected.checkpoint().config(), 128).unwrap();
    other.restore_soil_selection(&engine, &selected).unwrap();
    let epoch = sim.soil_identity_snapshot().unwrap().clone();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(transfer.apply(&foreign, &mut sim, &selected).is_err());
    assert!(transfer.apply(&engine, &mut other, &selected).is_err());
    sim.particle_material_dirty = Some(vec![0]);
    assert!(transfer.apply(&engine, &mut sim, &selected).is_err());
    sim.particle_material_dirty = None;
    assert!(
        transfer
            .apply(&engine, &mut sim, &all(&archive(2)))
            .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let measured = transfers.finish();
    assert_eq!((measured.upload_bytes, measured.readback_bytes), (0, 0));
    assert!(sim.is_soil_identity_current(&epoch));
    check(&engine, &mut sim, &selected);
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn live_world_opt_in_budget_expansion_and_disable_preserve_state() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let archive = archive(129);
    let initial_bytes = archive.to_bytes();
    let keys: Vec<_> = archive.tiles().collect();
    let cfg = all(&archive).checkpoint().config();
    let sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut world = MpmSoilWorld::new(
        &engine,
        sim,
        archive,
        SoilResidencyLimits {
            regions: 2,
            tiles: 32,
            claims: 64,
        },
        20656,
    )
    .unwrap();
    assert_eq!(world.transfer_scratch_bytes(), 0);
    let bytes = 129 * 508 + 16;
    let allocations = DeviceScope::start().unwrap();
    assert!(world.prepare_device_transfers(&engine, bytes - 1).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let allocations = DeviceScope::start().unwrap();
    world.prepare_device_transfers(&engine, bytes).unwrap();
    assert_eq!(allocations.finish().observations, 3);
    let allocations = DeviceScope::start().unwrap();
    world.prepare_device_transfers(&engine, bytes).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    let update = |world: &mut MpmSoilWorld, tiles: &[SoilTileCoord]| {
        world
            .begin_region_update(&[SoilRegionRequest {
                id: SoilRegionId(0),
                tiles,
            }])
            .unwrap();
        assert!(world.disable_device_transfers().is_err());
        assert!(world.prepare_device_transfers(&engine, bytes).is_err());
        let deadline = Instant::now() + Duration::from_secs(5);
        while !world.poll_region_update(&engine).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    update(&mut world, &keys[..1]);
    assert_eq!(world.active_particle_count(), 65);
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::DeviceGather
    );
    world.step_n(&engine, 0.00001, 2).unwrap();
    let evolved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    let mut expected = MpmSoilTileArchive::from_bytes(&initial_bytes, 129).unwrap();
    let old = expected.select(&keys[..1], 129).unwrap();
    expected.writeback(&old, &evolved).unwrap();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    update(&mut world, &keys);
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish().upload_bytes, 11_284);
    assert_eq!(
        (
            world.last_transfer_stats().retained_particles,
            world.last_transfer_stats().imported_particles
        ),
        (65, 64)
    );
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::DeviceGather
    );
    world.disable_device_transfers().unwrap();
    assert_eq!(world.transfer_scratch_bytes(), 0);
    check(&engine, &mut world.simulation, &all(&expected));
    assert_eq!(
        world.synchronized_archive_bytes().unwrap(),
        expected.to_bytes()
    );
    update(&mut world, &[]);
    assert_eq!(
        world.last_transfer_stats().path,
        MpmSoilTransferPath::Unchanged
    );
    world.step_n(&engine, 0.00001, 1).unwrap();
}

#[test]
#[ignore = "explicit transfer-only diagnostic; no simulated-time advancement"]
fn fifty_thousand_transfer_benchmark() {
    let engine = Engine::new_gpu().unwrap();
    let archive = archive(50_000);
    let full = all(&archive);
    let subset = archive
        .select(&archive.tiles().take(1).collect::<Vec<_>>(), 50_000)
        .unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, full.checkpoint().config(), 128).unwrap();
    sim.restore_soil_selection(&engine, &full).unwrap();
    let transfer = SoilTransfer::new(&engine, &sim).unwrap();
    for (scenario, source) in [("unchanged", &full), ("half-import", &subset)] {
        let mut samples = [Vec::new(), Vec::new()];
        for pair in 0..8 {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            for mode in order {
                transfer.apply(&engine, &mut sim, source).unwrap();
                engine
                    .device()
                    .unwrap()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                let start = Instant::now();
                if mode == 0 {
                    sim.restore_soil_selection(&engine, &full).unwrap();
                } else {
                    transfer.apply(&engine, &mut sim, &full).unwrap();
                }
                engine
                    .device()
                    .unwrap()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                check(&engine, &mut sim, &full);
                if pair >= 2 {
                    samples[mode].push(wall_ms);
                }
                let name = if mode == 0 {
                    "full-upload"
                } else {
                    "device-route"
                };
                println!(
                    "soil_transfer_sample scenario={scenario} mode={name} pair={pair} warmup={} particles=50000 simulated_seconds=0 wall_ms={wall_ms:.6}",
                    pair < 2
                );
            }
        }
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
        }
        let medians: Vec<_> = samples.iter().map(|v| (v[2] + v[3]) * 0.5).collect();
        println!(
            "soil_transfer_summary scenario={scenario} full_upload_median_ms={:.6} device_route_median_ms={:.6} measured_pairs=6 simulated_seconds=0",
            medians[0], medians[1]
        );
    }
}
