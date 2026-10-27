use super::*;
use crate::mpm::checkpoint::tests::new_sim;
use molla_vehicle::SoilMaterialId;

fn framed_seed(
    directory: &Directory,
    origin: [f64; 3],
) -> (MpmSoilFrameStore, MpmSoilFrameTileArchive, MpmConfig) {
    let mut state = fixture(129);
    state.cfg.grid_size = 64;
    state.cfg.cell_size = 0.125;
    state.cfg.grid_origin = [-4.0; 3];
    for (i, p) in state.words[..129 * 4].chunks_exact_mut(4).enumerate() {
        p[0] = [-0.000001f32, 0.125, 1.125, 2.125][i % 4].to_bits();
        p[1] = 2.0f32.to_bits();
        p[2] = 2.0f32.to_bits();
    }
    let cfg = state.cfg;
    let archive = MpmSoilFrameCheckpoint::new(state, origin)
        .unwrap()
        .partition_tiles(
            SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
            &(0..129)
                .map(|i| SoilParticleId((i << 40) + 7))
                .collect::<Vec<_>>(),
            &[
                SoilMaterialId(u64::MAX),
                SoilMaterialId(19),
                SoilMaterialId(1 << 50),
            ],
        )
        .unwrap();
    let mut store = MpmSoilFrameStore::open(&directory.0, store_limits()).unwrap();
    store.save(&archive).unwrap();
    (store, archive, cfg)
}

fn world(
    engine: &Engine,
    store: MpmSoilFrameStore,
    cfg: MpmConfig,
    mode: usize,
) -> MpmSoilDiskWorld {
    MpmSoilDiskWorld::new_framed(
        engine,
        new_sim(engine, cfg, mode),
        store,
        SoilResidencyLimits {
            regions: 3,
            tiles: 256,
            claims: 768,
        },
        io_limits(),
        129 * 160 + 16,
    )
    .unwrap()
}

fn captured(world: &mut MpmSoilDiskWorld, engine: &Engine) -> MpmSoilFrameCheckpoint {
    MpmSoilFrameCheckpoint::new(
        world
            .simulation
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap(),
        world.world_origin().unwrap(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires GPU; live framed disk boundaries preserve state and global IDs"]
fn framed_disk_live_overlap_steps_transfer_activity_and_reopen() {
    framed_disk_live_case(false);
}

#[test]
#[ignore = "requires GPU; incremental framed storage lifecycle"]
fn incremental_framed_disk_live_overlap_steps_and_reopen() {
    framed_disk_live_case(true);
}

fn framed_disk_live_case(incremental: bool) {
    let engine = Engine::new_gpu().unwrap();
    let origin = [1e12, -1e12, 1e12];
    for mode in 0..4 {
        let dir = Directory::new();
        let (mut store, archive, cfg) = framed_seed(&dir, origin);
        if incremental {
            store.upgrade_index().unwrap();
        }
        let keys: Vec<_> = archive.tiles().collect();
        assert_eq!(keys[0].0[0], 999_999_999_999);
        let mut live = world(&engine, store, cfg, mode);
        assert_eq!(live.world_origin(), Some(origin));
        assert_eq!(live.committed_read_stats().particles, 0);
        live.begin_region_update(&[
            SoilRegionRequest {
                id: SoilRegionId(2),
                tiles: &keys[..2],
            },
            SoilRegionRequest {
                id: SoilRegionId(3),
                tiles: &keys[1..2],
            },
        ])
        .unwrap();
        poll(&mut live, &engine).unwrap();
        let selected = archive.select(&keys[..2], 129).unwrap();
        assert_eq!(
            captured(&mut live, &engine).to_bytes(),
            selected.to_frame_checkpoint().unwrap().to_bytes()
        );
        assert_eq!(live.identity().particle_ids(), selected.particle_ids());
        live.prepare_device_transfers(&engine, u64::MAX).unwrap();
        request(&mut live, &engine, &keys).unwrap();
        assert_eq!(live.active_particle_count(), 129);
        live.set_activity_monitoring(true).unwrap();
        let before = captured(&mut live, &engine);
        let ids = live.identity().particle_ids().to_vec();
        live.step_n(&engine, 0.0001, 2).unwrap();
        let after = captured(&mut live, &engine);
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(live.active_particle_count(), 129);
        assert_eq!(captured(&mut live, &engine).to_bytes(), after.to_bytes());
        let (_, mut expected_tiles) = after
            .soil()
            .activity_by_tile_since(
                before.soil(),
                &ids,
                &ids,
                f64::from(0.0001f32) * 2.0,
                2,
                archive.layout(),
            )
            .unwrap();
        for tile in &mut expected_tiles {
            tile.tile.0 = std::array::from_fn(|a| tile.tile.0[a] + origin[a] as i64);
        }
        assert_eq!(live.last_activity_report().unwrap().tiles, expected_tiles);
        let view = live.simulation.soil_identity_buffers().unwrap();
        let words = crate::mpm::replay_tests::read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            view.particles,
        );
        let expected: Vec<_> = ids
            .iter()
            .flat_map(|id| [id.0 as u32, (id.0 >> 32) as u32])
            .collect();
        assert_eq!(&words[..expected.len()], expected);
        let frame = captured(&mut live, &engine);
        drop(live);
        let mut store = MpmSoilFrameStore::open(&dir.0, store_limits()).unwrap();
        let saved = store.load().unwrap().unwrap();
        let expected = saved
            .select(&saved.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        assert_eq!(
            expected.to_frame_checkpoint().unwrap().to_bytes(),
            frame.to_bytes()
        );
        let mut reloaded = world(&engine, store, cfg, mode);
        request(&mut reloaded, &engine, &saved.tiles().collect::<Vec<_>>()).unwrap();
        assert_eq!(
            captured(&mut reloaded, &engine).to_bytes(),
            frame.to_bytes()
        );
        reloaded.step_n(&engine, 0.0001, 1).unwrap();
    }
}

#[test]
#[ignore = "requires GPU; framed GPU pins and local swept horizons share world keys"]
fn framed_disk_gpu_pins_swept_horizon_and_fractional_host_frames() {
    let engine = Engine::new_gpu().unwrap();
    for fractional in [false, true] {
        let dir = Directory::new();
        let origin = [1e12 + if fractional { 0.25 } else { 0.0 }; 3];
        let (store, archive, cfg) = framed_seed(&dir, origin);
        let keys: Vec<_> = archive.tiles().collect();
        let mut live = world(&engine, store, cfg, 1);
        let config = MpmSoilPinConfig {
            minimum: SoilTileCoord([999_999_999_996; 3]),
            dimensions: [12; 3],
            capacity: 128,
            halo_tiles: 0,
        };
        let prepared = live.prepare_gpu_pins(&engine, config, u64::MAX);
        assert_eq!(prepared.is_err(), fractional);
        request(&mut live, &engine, &keys).unwrap();
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(live.resident_tiles().collect::<Vec<_>>(), keys);
        assert_eq!(live.active_particle_count(), 129);
        let e = MpmSoilRegionEnvelope {
            id: SoilRegionId(8),
            minimum: [-0.001, 1.9, 1.9],
            maximum: [0.126, 2.1, 2.1],
            particle_travel_per_substep: [0.001; 3],
            substeps: 1,
        };
        live.begin_swept_region_update(&[e]).unwrap();
        poll(&mut live, &engine).unwrap();
        assert_eq!(live.remaining_region_substeps(), Some(1));
        assert!(live.resident_tiles().count() > keys.len());
        let expected = e
            .tiles_in_frame(archive.layout(), &cfg, 256, Some(origin))
            .unwrap();
        assert!(expected.iter().all(|tile| live.claims(*tile).is_some()));
        live.step_n(&engine, 0.0001, 1).unwrap();
        assert!(live.step_n(&engine, 0.0001, 1).is_err());
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(live.remaining_region_substeps(), None);
    }
}

#[test]
#[ignore = "requires GPU; framed capacity failure leaves durable state retryable"]
fn framed_disk_capacity_retry_and_wrong_engine_preserve_authority() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = framed_seed(&dir, [1e12; 3]);
    let keys: Vec<_> = archive.tiles().collect();
    let mut live = world(&engine, store, cfg, 1);
    live.io_limits.particles = 33;
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    live.begin_region_update(&[SoilRegionRequest {
        id: SoilRegionId(1),
        tiles: &keys,
    }])
    .unwrap();
    assert!(live.poll_region_update(&foreign).is_err());
    assert!(poll(&mut live, &engine).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
    assert_eq!(live.active_particle_count(), 0);
    assert_eq!(live.committed_revision(), None);
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    request(&mut live, &engine, &keys[..1]).unwrap();
    assert_eq!(live.active_particle_count(), 33);
    assert_eq!(live.world_origin(), Some([1e12; 3]));
    assert_eq!(
        captured(&mut live, &engine).to_bytes(),
        archive
            .select(&keys[..1], 129)
            .unwrap()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes()
    );
}
