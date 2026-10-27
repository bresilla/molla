use super::*;
use crate::mpm::checkpoint::tests::fixture;
use molla_vehicle::{SoilMaterialId, SoilParticleId};
use std::time::{Duration, Instant};

#[test]
fn framed_pin_domain_translation_is_exact_bounded_and_rejects_fractional_frames() {
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let offset = [8_000_000_000_000, -8_000_000_000_000, 0];
    let mut world = config(64, 1);
    world.minimum.0 = std::array::from_fn(|a| offset[a] - 4);
    assert_eq!(
        local_config(layout, world, Some([1e12, -1e12, 0.0])).unwrap(),
        config(64, 1)
    );
    assert!(local_config(layout, world, Some([1e12 + 0.0625, -1e12, 0.0])).is_err());
    assert!(local_config(layout, world, Some([f64::NAN, 0.0, 0.0])).is_err());
    world.minimum.0[0] = i64::MAX;
    assert!(local_config(layout, world, Some([0.0; 3])).is_err());
    world.minimum.0[0] = (1i64 << 52) - 1;
    assert!(local_config(layout, world, Some([0.0; 3])).is_err());
}

fn config(capacity: u32, halo_tiles: u32) -> MpmSoilPinConfig {
    MpmSoilPinConfig {
        minimum: SoilTileCoord([-4; 3]),
        dimensions: [8; 3],
        capacity,
        halo_tiles,
    }
}

fn simulation(engine: &Engine, count: u32) -> MpmSimulation {
    let mut cfg = fixture(0).config();
    cfg.max_particles = count.max(1);
    MpmSimulation::new_sparse(engine, cfg, 128).unwrap()
}

fn poll(query: &mut SoilPins) -> Result<PinSnapshot> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(result) = query.poll()? {
            return Ok(result);
        }
        assert!(Instant::now() < deadline, "pin capture deadline");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn reference(layout: SoilTileLayout, positions: &[[f32; 3]], radius: u32) -> Vec<SoilTileCoord> {
    let mut keys = BTreeSet::new();
    let radius = i64::from(radius);
    for position in positions {
        let base = layout.tile_at(position.map(f64::from)).unwrap();
        for z in -radius..=radius {
            for y in -radius..=radius {
                for x in -radius..=radius {
                    keys.insert(SoilTileCoord([base.0[0] + x, base.0[1] + y, base.0[2] + z]));
                }
            }
        }
    }
    keys.into_iter().collect()
}

#[test]
fn layout_domain_work_and_memory_limits_are_explicit() {
    let limits = wgpu::Limits::default();
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let (_, sizes) = parameters(layout, config(64, 1), 50_000, &limits, 1104).unwrap();
    assert_eq!(sizes.iter().sum::<u64>(), 1104);
    assert!(parameters(layout, config(64, 1), 50_000, &limits, 1103).is_err());
    assert!(
        parameters(
            SoilTileLayout::new([1.0, 0.0, 0.0], 0.125).unwrap(),
            config(64, 0),
            1,
            &limits,
            u64::MAX
        )
        .is_err()
    );
    for edge in [0.1, 2_f64.powi(-127), 2_f64.powi(128)] {
        assert!(
            parameters(
                SoilTileLayout::new([0.0; 3], edge).unwrap(),
                config(64, 0),
                1,
                &limits,
                u64::MAX
            )
            .is_err()
        );
    }
    for bad in [
        config(0, 0),
        config(u32::MAX, 0),
        config(64, 5),
        MpmSoilPinConfig {
            minimum: SoilTileCoord([i64::MAX; 3]),
            ..config(64, 0)
        },
        MpmSoilPinConfig {
            dimensions: [u32::MAX; 3],
            ..config(64, 0)
        },
        MpmSoilPinConfig {
            dimensions: [8, 0, 8],
            ..config(64, 0)
        },
    ] {
        assert!(parameters(layout, bad, 1, &limits, u64::MAX).is_err());
    }
    assert!(parameters(layout, config(64, 4), u32::MAX, &limits, u64::MAX).is_err());
}

#[test]
fn pin_query_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<SoilPins>();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn exact_membership_includes_signed_zero_subnormals_and_both_sides_of_boundaries() {
    let engine = Engine::new_gpu().unwrap();
    let mut sim = simulation(&engine, 4096);
    for exponent in [-126, -20, -3, 0, 20, 127] {
        let edge = 2_f64.powi(exponent);
        let layout = SoilTileLayout::new([0.0; 3], edge).unwrap();
        let base = (edge as f32).to_bits();
        let raw = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007fffff,
            0x807fffff,
            base - 1,
            base,
            base + 1,
        ];
        let mut positions = Vec::new();
        for bits in raw {
            let value = f32::from_bits(bits);
            positions.push([value, -value, 0.0]);
            positions.push([-value, 0.0, value]);
        }
        sim.set_particles(engine.queue().unwrap(), &positions)
            .unwrap();
        let mut pins = SoilPins::new(&engine, &sim, layout, config(128, 0), 4096).unwrap();
        pins.submit(&sim, 19).unwrap();
        let snapshot = poll(&mut pins).unwrap();
        assert_eq!(snapshot.revision, 19);
        assert_eq!(
            snapshot.tiles,
            reference(layout, &positions, 0),
            "edge exponent {exponent}"
        );
    }
    let mut seed = 0x12345678_u32;
    let positions: Vec<[f32; 3]> = (0..2048)
        .map(|_| {
            std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits((seed & 0x807fffff) | (((seed >> 20) % 126 + 1) << 23))
            })
        })
        .collect();
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    sim.set_particles(engine.queue().unwrap(), &positions)
        .unwrap();
    let config = MpmSoilPinConfig {
        minimum: SoilTileCoord([-8; 3]),
        dimensions: [16; 3],
        capacity: 4096,
        halo_tiles: 0,
    };
    let mut pins = SoilPins::new(&engine, &sim, layout, config, 131072).unwrap();
    pins.submit(&sim, 20).unwrap();
    assert_eq!(
        poll(&mut pins).unwrap().tiles,
        reference(layout, &positions, 0)
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn crowded_fifty_thousand_halo_capture_is_bounded_and_invalid_queries_never_publish_partial_sets() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let mut sim = simulation(&engine, 50_000);
    let positions = vec![[0.02, 0.3, 0.02]; 50_000];
    sim.set_particles(engine.queue().unwrap(), &positions)
        .unwrap();
    let mut query = SoilPins::new(&engine, &sim, layout, config(64, 1), 1104).unwrap();
    assert_eq!(query.bytes(), 1104);
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    query.submit(&sim, 0).unwrap();
    assert!(query.submit(&sim, 1).is_err());
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish().upload_bytes, 48);
    sim.set_particles(engine.queue().unwrap(), &[]).unwrap();
    let transfers = TransferScope::start().unwrap();
    let snapshot = poll(&mut query).unwrap();
    assert_eq!(transfers.finish().readback_bytes, 272);
    assert_eq!(snapshot.tiles, reference(layout, &positions[..1], 1));
    assert_eq!(snapshot.tiles.len(), 27);
    query.submit(&sim, 1).unwrap();
    assert!(poll(&mut query).unwrap().tiles.is_empty());
    sim.set_particles(engine.queue().unwrap(), &positions[..1])
        .unwrap();
    let mut small = SoilPins::new(&engine, &sim, layout, config(1, 1), 128).unwrap();
    small.submit(&sim, 2).unwrap();
    assert!(poll(&mut small).is_err());
    sim.set_particles(engine.queue().unwrap(), &[]).unwrap();
    small.submit(&sim, 3).unwrap();
    assert!(poll(&mut small).unwrap().tiles.is_empty());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, 100.0] {
        sim.set_particles(engine.queue().unwrap(), &[[value, 0.0, 0.0]])
            .unwrap();
        query.submit(&sim, 4).unwrap();
        assert!(poll(&mut query).is_err());
    }
    sim.set_particles(engine.queue().unwrap(), &positions[..1])
        .unwrap();
    query.submit(&sim, 5).unwrap();
    assert_eq!(poll(&mut query).unwrap().tiles.len(), 27);
    println!(
        "50k crowded particles -> 27 halo tiles; 1104 retained bytes, 48-byte upload, 272-byte readback, zero hot application buffers"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn source_drop_foreign_context_map_failure_and_device_loss_are_explicit() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let mut sim = simulation(&engine, 1);
    sim.set_particles(engine.queue().unwrap(), &[[0.0; 3]])
        .unwrap();
    assert!(SoilPins::new(&foreign, &sim, layout, config(8, 0), 4096).is_err());
    drop(SoilPins::new(&engine, &sim, layout, config(8, 0), 4096).unwrap());
    let mut abandoned = SoilPins::new(&engine, &sim, layout, config(8, 0), 4096).unwrap();
    abandoned.submit(&sim, 0).unwrap();
    drop(abandoned);
    let mut query = SoilPins::new(&engine, &sim, layout, config(8, 0), 4096).unwrap();
    let other = simulation(&engine, 1);
    assert!(query.submit(&other, 0).is_err());
    query.submit(&sim, 1).unwrap();
    drop(sim);
    assert_eq!(poll(&mut query).unwrap().tiles, vec![SoilTileCoord([0; 3])]);
    drop(query);
    let mut failed = SoilPins::new(&engine, &other, layout, config(8, 0), 4096).unwrap();
    failed.submit(&other, 2).unwrap();
    let signal = Arc::new(OnceLock::new());
    signal.set(Err(wgpu::BufferAsyncError)).unwrap();
    failed.request.as_mut().unwrap().signal = signal;
    assert!(matches!(failed.poll(), Err(Error::Gpu(_))));
    assert!(failed.submit(&other, 3).is_err());
    drop(failed);
    let mut pending = SoilPins::new(&engine, &other, layout, config(8, 0), 4096).unwrap();
    pending.submit(&other, 4).unwrap();
    engine.destroy_device().unwrap();
    assert!(matches!(pending.poll(), Err(Error::Gpu(_))));
    assert!(pending.submit(&other, 5).is_err());
}

fn world_poll(world: &mut MpmSoilWorld, engine: &Engine) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !world.poll_region_update(engine)? {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn live_owner_joins_capture_and_pins_then_retries_overflow_without_losing_moved_state() {
    let engine = Engine::new_gpu().unwrap();
    let mut state = fixture(2);
    state.words[0] = (-0.00001_f32).to_bits();
    state.words[4] = 0.1_f32.to_bits();
    state.words[8] = 1_f32.to_bits();
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let archive = state
        .partition_tiles(
            layout,
            &[SoilParticleId(7), SoilParticleId(u64::MAX)],
            &[SoilMaterialId(0), SoilMaterialId(1), SoilMaterialId(2)],
        )
        .unwrap();
    let bytes = archive.to_bytes();
    let keys: Vec<_> = archive.tiles().collect();
    let sim = MpmSimulation::new_sparse(&engine, state.config(), 128).unwrap();
    let mut world = MpmSoilWorld::new(
        &engine,
        sim,
        archive,
        SoilResidencyLimits {
            regions: 2,
            tiles: 64,
            claims: 128,
        },
        20656,
    )
    .unwrap();
    world.prepare_gpu_pins(&engine, config(1, 1), 128).unwrap();
    world
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(0),
            tiles: &keys[..1],
        }])
        .unwrap();
    assert!(world.disable_gpu_pins().is_err());
    assert!(
        world
            .prepare_gpu_pins(&engine, config(64, 0), 1104)
            .is_err()
    );
    let signal = world
        .gpu_pins
        .as_ref()
        .unwrap()
        .request
        .as_ref()
        .unwrap()
        .signal
        .clone();
    world
        .gpu_pins
        .as_mut()
        .unwrap()
        .request
        .as_mut()
        .unwrap()
        .signal = Arc::new(OnceLock::new());
    let deadline = Instant::now() + Duration::from_secs(5);
    while world.pending.as_ref().unwrap().checkpoint.is_none() {
        assert!(!world.poll_region_update(&engine).unwrap());
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(world.status(), MpmSoilWorldStatus::Capturing);
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    world
        .gpu_pins
        .as_mut()
        .unwrap()
        .request
        .as_mut()
        .unwrap()
        .signal = signal;
    world_poll(&mut world, &engine).unwrap();
    assert_eq!(world.active_particle_count(), 1);
    world.step_n(&engine, 0.0001, 2).unwrap();
    let evolved = world
        .simulation
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert!(f32::from_bits(evolved.words[0]) > 0.0);
    let before = world.last_committed_archive_bytes();
    let identity = world.identity().clone();
    world.begin_region_update(&[]).unwrap();
    assert!(world_poll(&mut world, &engine).is_err());
    assert_eq!(world.status(), MpmSoilWorldStatus::Ready);
    assert_eq!(world.last_committed_archive_bytes(), before);
    assert!(world.simulation.is_soil_identity_current(&identity));
    world
        .prepare_gpu_pins(&engine, config(64, 0), 1104)
        .unwrap();
    world.begin_region_update(&[]).unwrap();
    world_poll(&mut world, &engine).unwrap();
    assert_eq!(world.active_particle_count(), 2);
    let mut expected = MpmSoilTileArchive::from_bytes(&bytes, 129).unwrap();
    let original = expected.select(&keys[..1], 129).unwrap();
    expected.writeback(&original, &evolved).unwrap();
    assert_eq!(
        world.synchronized_archive_bytes().unwrap(),
        expected.to_bytes()
    );
    assert_eq!(world.resident_tiles().collect::<Vec<_>>(), vec![keys[1]]);
    world.disable_gpu_pins().unwrap();
    assert_eq!(world.gpu_pin_buffer_bytes(), 0);
    world.step_n(&engine, 0.0001, 1).unwrap();
}
