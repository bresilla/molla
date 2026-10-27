use super::*;

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; physical grid window publication and continued trajectory"]
fn canonical_grid_window_move_identity_trajectory_and_reopen() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let temporary = Directory::new();
    let output = std::env::var_os("MOLLA_GRID_WINDOW_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| temporary.0.join("outputs"));
    fs::create_dir(&output).unwrap();
    let mut records = Vec::new();
    let mut accepted = true;
    for (implicit, mode) in [false, true]
        .into_iter()
        .flat_map(|implicit| (1..4).map(move |mode| (implicit, mode)))
    {
        let dirs = [Directory::new(), Directory::new(), Directory::new()];
        let mut worlds: Vec<_> = dirs
            .iter()
            .map(|dir| {
                let time_step = if implicit {
                    MpmTimeStep::Implicit { jacobi_iters: 2 }
                } else {
                    MpmTimeStep::Explicit
                };
                let (store, archive, cfg) = seed_canonical_step(dir, time_step);
                let keys: Vec<_> = archive.tiles().collect();
                let mut live = world(&engine, store, cfg, mode, [1e12; 3]);
                request(&mut live, &engine, &keys).unwrap();
                live.simulation
                    .prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 129)
                    .unwrap();
                live.simulation.set_particle_tile_ordering(true).unwrap();
                live.simulation
                    .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
                    .unwrap();
                live.simulation.set_deterministic_p2g(true).unwrap();
                enable_pins(&mut live, &engine);
                live
            })
            .collect();
        let before = capture(&mut worlds[2], &engine);
        let identity = worlds[2].identity().clone();
        let head = fs::read(dirs[2].0.join("HEAD")).unwrap();
        let grid = [-3.875, -4.0, -4.0];
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        worlds[2].begin_grid_window_move(grid).unwrap();
        assert!(worlds[2].step_n(&engine, 0.0001, 1).is_err());
        assert!(worlds[2].begin_recenter(grid).is_err());
        poll(&mut worlds[2], &engine).unwrap();
        let buffers = resources.finish().observations;
        let transfers = transfers.finish();
        assert_eq!(buffers, 0);
        assert_eq!(transfers.upload_bytes, 0);
        assert_eq!(
            worlds[2].last_transfer_stats().path,
            MpmSoilTransferPath::Unchanged
        );
        assert_eq!(worlds[2].last_transfer_stats().upload_bytes, 0);
        println!(
            "window_publication implicit={implicit} mode={mode} buffers={buffers} uploads={} readbacks={}",
            transfers.upload_bytes, transfers.readback_bytes
        );
        let moved = capture(&mut worlds[2], &engine);
        assert_eq!(moved.world_origin(), before.world_origin());
        assert_eq!(moved.soil().config().grid_origin, grid);
        for stream in 0..9 {
            assert_eq!(moved.soil().stream(stream), before.soil().stream(stream));
        }
        assert_eq!(fs::read(dirs[2].0.join("HEAD")).unwrap(), head);
        assert!(worlds[2].simulation.is_soil_identity_current(&identity));
        assert_ids(&worlds[2], &engine);
        assert!(
            worlds[2]
                .simulation
                .deterministic_p2g_compensated_stiffness_enabled()
        );
        for step in 1..=8 {
            for index in if step % 2 == 0 { [2, 1, 0] } else { [0, 1, 2] } {
                worlds[index].step_n(&engine, 0.0001, 1).unwrap();
            }
            let states: Vec<_> = worlds
                .iter_mut()
                .map(|live| capture(live, &engine))
                .collect();
            let exact = states[0].to_bytes() == states[1].to_bytes();
            let errors = trajectory::differences(&states[0], &states[2]);
            let pass = errors.iter().all(|&x| x <= 1e-6);
            accepted &= exact && pass;
            records.push(serde_json::json!({"implicit":implicit,"mode":mode,"step":step,"exact_repeat":exact,"errors":errors,"physical_parity":pass}));
            for (run, state) in states.iter().enumerate() {
                fs::write(
                    output.join(format!(
                        "implicit-{implicit}-mode-{mode}-step-{step}-run-{run}.frame"
                    )),
                    state.to_bytes(),
                )
                .unwrap();
            }
        }
        let evolved = capture(&mut worlds[2], &engine);
        let revision = worlds[2].committed_revision();
        worlds[2]
            .begin_grid_window_move([-1.0, -4.0, -4.0])
            .unwrap();
        assert!(poll(&mut worlds[2], &engine).is_err());
        assert_eq!(worlds[2].status(), MpmSoilDiskWorldStatus::Ready);
        assert_eq!(worlds[2].committed_revision(), revision);
        assert_eq!(
            capture(&mut worlds[2], &engine).to_bytes(),
            evolved.to_bytes()
        );
        request(&mut worlds[2], &engine, &[]).unwrap();
        let keys = worlds[2].resident_tiles().collect::<Vec<_>>();
        let cfg = evolved.soil().config();
        drop(worlds);
        let store = MpmSoilCanonicalStore::open(&dirs[2].0, store_limits()).unwrap();
        let mut reopened = world(&engine, store, cfg, mode, [1e12; 3]);
        request(&mut reopened, &engine, &keys).unwrap();
        assert_eq!(
            capture(&mut reopened, &engine).to_bytes(),
            evolved.to_bytes()
        );
        assert_ids(&reopened, &engine);
    }
    let report = serde_json::json!({"accepted":accepted,"particles":129,"steps":8,"dt":0.0001,
        "tolerance":1e-6,"self_repeat":"exact","records":records,
        "scope":"detached physical grid-window shift; not machine rebasing, sleep, or long traversal"});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("grid_window_report={report}");
    assert!(accepted);
}
