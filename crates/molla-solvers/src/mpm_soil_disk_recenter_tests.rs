use super::*;

#[path = "mpm_soil_recenter_trajectory_tests.rs"]
mod trajectory;

#[path = "mpm_soil_grid_window_tests.rs"]
mod grid_window;

#[path = "mpm_soil_disk_integer_window_tests.rs"]
mod integer_window;

fn assert_ids(live: &MpmSoilDiskWorld, engine: &Engine) {
    let buffers = live.simulation.soil_identity_buffers().unwrap();
    for (buffer, ids) in [
        (
            buffers.particles,
            live.identity()
                .particle_ids()
                .iter()
                .map(|id| id.0)
                .collect::<Vec<_>>(),
        ),
        (
            buffers.materials,
            live.identity()
                .material_ids()
                .iter()
                .map(|id| id.0)
                .collect(),
        ),
    ] {
        let expected: Vec<u32> = ids
            .into_iter()
            .flat_map(|id| [id as u32, (id >> 32) as u32])
            .collect();
        let actual = crate::mpm::replay_tests::read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            buffer,
        );
        assert_eq!(&actual[..expected.len()], expected);
    }
}

fn enable_pins(live: &mut MpmSoilDiskWorld, engine: &Engine) {
    live.prepare_gpu_pins(
        engine,
        MpmSoilPinConfig {
            minimum: SoilTileCoord([999_999_999_996; 3]),
            dimensions: [12; 3],
            capacity: 128,
            halo_tiles: 0,
        },
        u64::MAX,
    )
    .unwrap();
}

#[test]
#[ignore = "requires GPU; exact live frame transaction preserves IDs, pins and continued storage"]
fn canonical_recenter_identity_pins_transfer_and_reopen_all_grid_modes() {
    let engine = Engine::new_gpu().unwrap();
    for mode in 0..4 {
        let dir = Directory::new();
        let (store, archive, cfg) = seed_canonical(&dir);
        let keys: Vec<_> = archive.tiles().collect();
        let mut live = world(&engine, store, cfg, mode, [1e12; 3]);
        request(&mut live, &engine, &keys).unwrap();
        enable_pins(&mut live, &engine);
        live.prepare_device_transfers(&engine, u64::MAX).unwrap();
        let scratch = (
            live.gpu_pin_buffer_bytes(),
            live.transfer_scratch_bytes(),
            live.staging_bytes(),
        );
        let identity = live.identity().clone();
        let ids = live.simulation.soil_identity_buffers().unwrap();
        let buffers = [ids.particles.clone(), ids.materials.clone()];
        let before = capture(&mut live, &engine);
        let expected = before.recentered([-5.0, -4.0, -4.0]).unwrap();
        let head = fs::read(dir.0.join("HEAD")).unwrap();
        let write = live.last_store_write();
        live.begin_recenter([-5.0, -4.0, -4.0]).unwrap();
        assert_eq!(live.status(), MpmSoilDiskWorldStatus::Capturing);
        assert_eq!(live.world_origin(), Some([1e12; 3]));
        assert!(live.step_n(&engine, 0.0001, 1).is_err());
        assert!(live.begin_region_update(&[]).is_err());
        assert!(live.begin_recenter(cfg.grid_origin).is_err());
        poll(&mut live, &engine).unwrap();
        assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
        assert_eq!(live.resident_tiles().collect::<Vec<_>>(), keys);
        assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
        assert_eq!(live.last_store_write(), write);
        assert!(live.simulation.is_soil_identity_current(&identity));
        let ids = live.simulation.soil_identity_buffers().unwrap();
        assert_eq!((&buffers[0], &buffers[1]), (ids.particles, ids.materials));
        assert_ids(&live, &engine);
        assert_eq!(
            live.last_transfer_stats().path,
            MpmSoilTransferPath::FullUpload
        );
        assert_eq!(live.last_transfer_stats().upload_bytes, 129 * 160);
        assert_eq!(
            (
                live.gpu_pin_buffer_bytes(),
                live.transfer_scratch_bytes(),
                live.staging_bytes()
            ),
            scratch
        );
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(live.resident_tiles().collect::<Vec<_>>(), keys);
        assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
        assert_eq!(
            live.last_transfer_stats().path,
            MpmSoilTransferPath::Unchanged
        );
        live.disable_gpu_pins().unwrap();
        live.step_n(&engine, 0.0001, 1).unwrap();
        let evolved = capture(&mut live, &engine);
        let grid = [-5.0, -4.0 - 1.0 / 4096.0, -4.0];
        let expected = evolved.recentered(grid).unwrap();
        live.begin_recenter(grid).unwrap();
        poll(&mut live, &engine).unwrap();
        assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
        assert_eq!(
            expected.particle_tiles(archive.layout()).unwrap(),
            evolved.particle_tiles(archive.layout()).unwrap()
        );
        request(&mut live, &engine, &[]).unwrap();
        let origin = live.world_origin().unwrap();
        let keys = live.resident_tiles().collect::<Vec<_>>();
        let mut cfg = cfg;
        cfg.grid_origin = grid;
        drop(live);
        let store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
        let mut restored = world(&engine, store, cfg, mode, origin);
        request(&mut restored, &engine, &keys).unwrap();
        assert_eq!(
            capture(&mut restored, &engine).to_bytes(),
            expected.to_bytes()
        );
        assert_ids(&restored, &engine);
        restored.step_n(&engine, 0.0001, 1).unwrap();
    }
}

#[test]
#[ignore = "requires GPU; recenter preflight and captured rounding failures preserve the old owner"]
fn canonical_recenter_rejections_retry_foreign_engine_and_capture_fault() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = seed_canonical(&dir);
    let keys: Vec<_> = archive.tiles().collect();
    let mut live = world(&engine, store, cfg, 2, [1e12; 3]);
    request(&mut live, &engine, &keys).unwrap();
    enable_pins(&mut live, &engine);
    let requests = live.readback_stats().requests;
    assert!(live.begin_recenter([f32::NAN; 3]).is_err());
    assert!(live.begin_recenter([-4.25, -4.0, -4.0]).is_err());
    assert_eq!(live.readback_stats().requests, requests);
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
    let x = 0.125f32.to_bits() + 1;
    engine
        .queue()
        .unwrap()
        .write_buffer(&live.simulation.p_pos, 0, &x.to_le_bytes());
    let before = capture(&mut live, &engine);
    assert!(before.recentered([-3.0, -4.0, -4.0]).is_err());
    let identity = live.identity().clone();
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    let revision = live.committed_revision();
    live.begin_recenter([-3.0, -4.0, -4.0]).unwrap();
    assert!(live.poll_region_update(&foreign).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Capturing);
    assert!(poll(&mut live, &engine).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
    assert_eq!(live.committed_revision(), revision);
    assert_eq!(capture(&mut live, &engine).to_bytes(), before.to_bytes());
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    assert!(live.simulation.is_soil_identity_current(&identity));
    live.begin_recenter(cfg.grid_origin).unwrap();
    poll(&mut live, &engine).unwrap();
    assert_eq!(
        live.last_transfer_stats().path,
        MpmSoilTransferPath::Unchanged
    );
    assert_eq!(capture(&mut live, &engine).to_bytes(), before.to_bytes());
    request(&mut live, &engine, &[]).unwrap();
    assert_eq!(live.resident_tiles().collect::<Vec<_>>(), keys);
    engine.queue().unwrap().write_buffer(
        &live.simulation.p_pos,
        0,
        &f32::NAN.to_bits().to_le_bytes(),
    );
    live.begin_recenter(cfg.grid_origin).unwrap();
    assert!(poll(&mut live, &engine).is_err());
    assert_eq!(live.status(), MpmSoilDiskWorldStatus::Faulted);
    assert!(live.step_n(&engine, 0.0001, 1).is_err());
    let raw_dir = Directory::new();
    let (store, _, cfg) = seed(&raw_dir);
    let mut raw = MpmSoilDiskWorld::new(
        &engine,
        new_sim(&engine, cfg, 0),
        store,
        limits(),
        io_limits(),
        129 * 160 + 16,
    )
    .unwrap();
    assert!(raw.begin_recenter(cfg.grid_origin).is_err());
    assert_eq!(raw.readback_stats().requests, 0);
}

#[test]
#[ignore = "requires GPU; coordinate changes retain consumed swept horizon and activity duration"]
fn canonical_recenter_preserves_swept_horizon_and_activity_clock() {
    let engine = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = seed_canonical(&dir);
    let keys: Vec<_> = archive.tiles().collect();
    let mut live = world(&engine, store, cfg, 1, [1e12; 3]);
    request(&mut live, &engine, &keys).unwrap();
    live.set_activity_monitoring(true).unwrap();
    live.begin_swept_region_update(&[MpmSoilRegionEnvelope {
        id: SoilRegionId(2),
        minimum: [0.0, 1.9, 1.9],
        maximum: [0.25, 2.1, 2.1],
        particle_travel_per_substep: [0.001; 3],
        substeps: 3,
    }])
    .unwrap();
    poll(&mut live, &engine).unwrap();
    live.step_n(&engine, 0.0001, 1).unwrap();
    let tiles = live.resident_tiles().collect::<Vec<_>>();
    let before = capture(&mut live, &engine);
    let grid = [-4.0, -4.0 - 1.0 / 4096.0, -4.0];
    let expected = before.recentered(grid).unwrap();
    live.begin_recenter(grid).unwrap();
    poll(&mut live, &engine).unwrap();
    assert_eq!(live.remaining_region_substeps(), Some(2));
    assert_eq!(live.resident_tiles().collect::<Vec<_>>(), tiles);
    assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
    live.step_n(&engine, 0.0001, 2).unwrap();
    assert_eq!(live.remaining_region_substeps(), Some(0));
    assert!(live.step_n(&engine, 0.0001, 1).is_err());
    request(&mut live, &engine, &[]).unwrap();
    assert_eq!(live.last_activity_report().unwrap().activity.substeps, 3);
}
