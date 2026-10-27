use super::*;

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; decimal-lattice disk window publication, writeback and reopen"]
fn canonical_integer_window_gpu_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let temporary = Directory::new();
    let output = std::env::var_os("MOLLA_DISK_INTEGER_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| temporary.0.join("output"));
    fs::create_dir(&output).unwrap();
    for (mode, implicit) in [(1, false), (2, false), (1, true), (2, true)] {
        let dir = Directory::new();
        let mut soil = fixture(129);
        soil.cfg.grid_size = 64;
        soil.cfg.cell_size = 0.1;
        soil.cfg.grid_origin = [-3.0, -0.5, -3.0];
        for (index, position) in soil.words[..129 * 4].chunks_exact_mut(4).enumerate() {
            if index % 2 == 1 {
                position[0] = (f32::from_bits(position[0]) + 0.2).to_bits();
            }
        }
        soil.cfg.time_step = if implicit {
            MpmTimeStep::Implicit { jacobi_iters: 2 }
        } else {
            MpmTimeStep::Explicit
        };
        let cfg = soil.cfg;
        let initial = MpmSoilFrameCheckpoint::new(soil, [0.0; 3]).unwrap();
        let archive = initial
            .partition_canonical_tiles(
                SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
                &(0..129)
                    .map(|i| SoilParticleId((i << 40) + 7))
                    .collect::<Vec<_>>(),
                &[
                    SoilMaterialId(0),
                    SoilMaterialId(1 << 50),
                    SoilMaterialId(u64::MAX),
                ],
            )
            .unwrap();
        let keys: Vec<_> = archive.tiles().collect();
        let mut store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
        store.save(&archive).unwrap();
        let mut live = world(&engine, store, cfg, mode, [0.0; 3]);
        assert!(keys.len() > 1);
        request(&mut live, &engine, &keys[..1]).unwrap();
        live.prepare_device_transfers(&engine, u64::MAX).unwrap();
        live.prepare_gpu_pins(&engine, pin_config(128, 0), u64::MAX)
            .unwrap();
        let head = fs::read(dir.0.join("HEAD")).unwrap();
        let identity = live.identity().clone();
        let before = capture(&mut live, &engine);
        let cells = [-1, 0, 1];
        let expected = MpmSoilFrameCheckpoint::new(
            before.soil().with_grid_window_cells(cells).unwrap(),
            [0.0; 3],
        )
        .unwrap();
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        live.begin_grid_window_cells(cells).unwrap();
        assert_eq!(live.status(), MpmSoilDiskWorldStatus::Capturing);
        assert!(live.step_n(&engine, 0.0001, 1).is_err());
        assert!(live.begin_grid_window_cells(cells).is_err());
        poll(&mut live, &engine).unwrap();
        assert_eq!(resources.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.upload_bytes, 0);
        assert_eq!(live.last_transfer_stats().upload_bytes, 0);
        assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
        assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
        assert!(live.simulation.is_soil_identity_current(&identity));
        assert_ids(&live, &engine);
        let wrong = archive
            .select(&keys, 129, [0.0; 3], cfg.grid_origin)
            .unwrap();
        assert!(
            transfer::unchanged(&live.simulation, wrong.framed_selection().local_selection())
                .is_err()
        );
        let revision = live.committed_revision();
        assert!(live.begin_recenter(cfg.grid_origin).is_err());
        assert!(live.begin_grid_window_move(cfg.grid_origin).is_err());
        assert!(live.begin_grid_window_cells([i32::MAX, 0, 0]).is_err());
        live.begin_grid_window_cells([32, 0, 1]).unwrap();
        assert!(poll(&mut live, &engine).is_err());
        assert_eq!(live.status(), MpmSoilDiskWorldStatus::Ready);
        assert_eq!(live.committed_revision(), revision);
        assert_eq!(capture(&mut live, &engine).to_bytes(), expected.to_bytes());
        assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
        live.begin_grid_window_cells(cells).unwrap();
        poll(&mut live, &engine).unwrap();
        request(&mut live, &engine, &keys).unwrap();
        assert_eq!(
            live.last_transfer_stats().path,
            MpmSoilTransferPath::DeviceGather
        );
        assert_eq!(
            live.last_transfer_stats().retained_particles,
            before.soil().particle_count()
        );
        assert!(live.last_transfer_stats().imported_particles > 0);
        let imported = MpmSoilFrameCheckpoint::new(
            initial.soil().with_grid_window_cells(cells).unwrap(),
            [0.0; 3],
        )
        .unwrap();
        assert_eq!(capture(&mut live, &engine).to_bytes(), imported.to_bytes());
        assert_ids(&live, &engine);
        live.simulation
            .prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 129)
            .unwrap();
        live.simulation.set_particle_tile_ordering(true).unwrap();
        live.simulation
            .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
            .unwrap();
        live.simulation.set_deterministic_p2g(true).unwrap();
        live.step_n(&engine, 0.0001, 2).unwrap();
        let evolved = capture(&mut live, &engine);
        request(&mut live, &engine, &[]).unwrap();
        assert_eq!(
            live.last_transfer_stats().path,
            MpmSoilTransferPath::Unchanged
        );
        assert_eq!(capture(&mut live, &engine).to_bytes(), evolved.to_bytes());
        assert_eq!(live.simulation.grid_window_cells(), cells);
        let keys: Vec<_> = live.resident_tiles().collect();
        drop(live);
        let mut store = MpmSoilCanonicalStore::open(&dir.0, store_limits()).unwrap();
        let global = store.load().unwrap().unwrap();
        let selected = global
            .select_window(&keys, 129, [0.0; 3], cfg.grid_origin, cells)
            .unwrap();
        assert_eq!(
            selected
                .framed_selection()
                .to_frame_checkpoint()
                .unwrap()
                .to_bytes(),
            evolved.to_bytes()
        );
        let mut simulation = new_sim(&engine, cfg, mode);
        simulation
            .move_soil_grid_window_cells_with(engine.queue().unwrap(), cells, 129)
            .unwrap();
        let mut restored = MpmSoilDiskWorld::new_canonical(
            &engine,
            simulation,
            store,
            [0.0; 3],
            SoilResidencyLimits {
                regions: 3,
                tiles: 256,
                claims: 768,
            },
            canonical_bounds(),
            129 * 160 + 16,
        )
        .unwrap();
        request(&mut restored, &engine, &keys).unwrap();
        assert_eq!(
            capture(&mut restored, &engine).to_bytes(),
            evolved.to_bytes()
        );
        assert_ids(&restored, &engine);
        restored.step_n(&engine, 0.0001, 1).unwrap();
        restored.begin_grid_window_cells([0; 3]).unwrap();
        poll(&mut restored, &engine).unwrap();
        assert_eq!(restored.simulation.grid_window_cells(), [0; 3]);
        for (name, state) in [
            ("before", before),
            ("moved", expected),
            ("evolved", evolved),
            ("continued", capture(&mut restored, &engine)),
        ] {
            fs::write(
                output.join(format!("mode-{mode}-implicit-{implicit}-{name}.frame")),
                state.to_bytes(),
            )
            .unwrap();
        }
        println!(
            "canonical_integer_window mode={mode} implicit={implicit} particles=129 upload_bytes=0 readback_bytes={} exact_move=true exact_reopen=true",
            transfers.readback_bytes
        );
    }
}
