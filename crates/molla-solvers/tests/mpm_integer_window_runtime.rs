#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::TransferScope;
use molla_solvers::mpm::*;

fn simulation(engine: &Engine, cfg: MpmConfig, native: bool, layout: usize) -> MpmSimulation {
    let mut sim = match (native, layout) {
        (true, 2) => MpmSimulation::new_native_sparse(engine, cfg, 128),
        (true, _) => MpmSimulation::new_native_explicit(engine, cfg),
        (false, 2) => MpmSimulation::new_sparse(engine, cfg, 128),
        (false, _) => MpmSimulation::new(engine, cfg),
    }
    .unwrap();
    if layout == 1 {
        sim.set_active_grid(true).unwrap();
    }
    sim.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), cfg.max_particles)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    if native {
        sim.prepare_deterministic_p2g_with_budget(65 * 64).unwrap();
    } else {
        sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64)
            .unwrap();
    }
    sim.set_deterministic_p2g(true).unwrap();
    sim
}

fn payload(state: &MpmSoilCheckpoint) -> Vec<u8> {
    let bytes = state.to_bytes();
    bytes[bytes.len() - state.particle_bytes()..].to_vec()
}

fn errors(a: &MpmSoilCheckpoint, b: &MpmSoilCheckpoint) -> [f64; 9] {
    assert_eq!(a.config(), b.config());
    assert_eq!(a.particle_count(), b.particle_count());
    let a = payload(a);
    let b = payload(b);
    let mut offset = 0;
    let mut result = [0.0_f64; 9];
    for (stream, stride) in [4, 4, 12, 12, 1, 1, 1, 1, 4].into_iter().enumerate() {
        for word in 0..stride * 65 {
            let x = u32::from_le_bytes(a[offset..offset + 4].try_into().unwrap());
            let y = u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap());
            if stream == 7 {
                assert_eq!(x, y);
            } else {
                let x = f64::from(f32::from_bits(x));
                let y = f64::from(f32::from_bits(y));
                assert!(x.is_finite() && y.is_finite());
                let scale = if stream == 0 && word % 4 < 3 {
                    1.0
                } else {
                    x.abs().max(y.abs()).max(1.0)
                };
                result[stream] = result[stream].max((x - y).abs() / scale);
            }
            offset += 4;
        }
    }
    assert_eq!(offset, a.len());
    result
}

#[test]
#[ignore = "requires GPU; public integer windows, native/non-native save/reload and repeated trajectories"]
fn integer_window_public_restore_and_continue() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_INTEGER_WINDOW_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut records = Vec::new();
    let mut accepted = true;
    let cases = [
        (false, 1, false),
        (false, 2, false),
        (false, 1, true),
        (false, 2, true),
        (true, 0, false),
        (true, 1, false),
        (true, 2, false),
    ];
    for (case, (native, layout, implicit)) in cases.into_iter().enumerate() {
        let cfg = MpmConfig {
            grid_origin: [-3.0, -0.5, -3.0],
            cell_size: 0.1,
            grid_size: 64,
            max_particles: 65,
            max_rigid: 1,
            youngs: 1000.0,
            time_step: if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 2 }
            } else {
                MpmTimeStep::Explicit
            },
            ..MpmConfig::sand()
        };
        let points: Vec<_> = (0..65)
            .map(|i| {
                [
                    -0.13 + (i % 7) as f32 * 0.021,
                    0.3 + (i / 7 % 7) as f32 * 0.021,
                    -0.07 + (i / 49) as f32 * 0.021,
                ]
            })
            .collect();
        let mut states = Vec::new();
        for arm in 0..6 {
            let cells = [[0; 3], [-1, 0, 1], [1, 0, -1]][arm / 2];
            let mut sim = simulation(&engine, cfg, native, layout);
            sim.set_particles_with_velocity(queue, &points, &vec![[0.013, -0.007, 0.002]; 65])
                .unwrap();
            let initial = sim.checkpoint_soil_with(queue).unwrap();
            let old_record = sim.prepare_step(0.0001, 1).unwrap();
            let traffic = TransferScope::start().unwrap();
            assert!(
                sim.move_soil_grid_window_cells_with(queue, cells, 64)
                    .is_err()
            );
            assert_eq!(traffic.finish().readback_bytes, 0);
            assert!(
                sim.move_soil_grid_window_cells_with(queue, [i32::MAX, 0, 0], 65)
                    .is_err()
            );
            assert!(
                sim.move_soil_grid_window_cells_with(queue, [30, 0, 0], 65)
                    .is_err()
            );
            assert_eq!(
                sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
                initial.to_bytes()
            );
            let traffic = TransferScope::start().unwrap();
            sim.move_soil_grid_window_cells_with(queue, cells, 65)
                .unwrap();
            let traffic = traffic.finish();
            assert_eq!(traffic.upload_bytes, 0);
            assert_eq!(sim.grid_window_cells(), cells);
            let moved = sim.checkpoint_soil_with(queue).unwrap();
            assert_eq!(
                moved.to_bytes(),
                initial.with_grid_window_cells(cells).unwrap().to_bytes()
            );
            let encoder = device.create_command_encoder(&Default::default());
            let mut batch = sim.begin_command_batch(encoder);
            assert_eq!(batch.encode_step(&old_record).is_err(), cells != [0; 3]);
            drop(batch);
            let saved = output.join(format!("case-{case}-arm-{arm}-saved.bin"));
            std::fs::write(&saved, moved.to_bytes()).unwrap();
            let restored =
                MpmSoilCheckpoint::from_bytes(&std::fs::read(saved).unwrap(), 65).unwrap();
            let mut reloaded = simulation(&engine, cfg, native, layout);
            reloaded.restore_soil_with(queue, &restored).unwrap();
            assert_eq!(reloaded.grid_window_cells(), cells);
            assert_eq!(
                reloaded.checkpoint_soil_with(queue).unwrap().to_bytes(),
                moved.to_bytes()
            );
            let record = reloaded.prepare_step(0.0001, 1).unwrap();
            for step in 0..4 {
                if step % 2 == 0 {
                    reloaded.step_n_with(device, queue, 0.0001, 1).unwrap();
                } else {
                    let encoder = device.create_command_encoder(&Default::default());
                    let mut batch = reloaded.begin_command_batch(encoder);
                    batch.encode_step(&record).unwrap();
                    batch.submit(queue);
                }
            }
            let evolved = reloaded.checkpoint_soil_with(queue).unwrap();
            std::fs::write(
                output.join(format!("case-{case}-arm-{arm}-evolved.bin")),
                evolved.to_bytes(),
            )
            .unwrap();
            reloaded
                .move_soil_grid_window_cells_with(queue, [0; 3], 65)
                .unwrap();
            let returned = reloaded.checkpoint_soil_with(queue).unwrap();
            assert_eq!(payload(&returned), payload(&evolved));
            assert_eq!(&returned.to_bytes()[8..12], &1_u32.to_le_bytes());
            states.push(evolved);
        }
        let exact =
            (0..3).all(|pair| states[2 * pair].to_bytes() == states[2 * pair + 1].to_bytes());
        let differences = [
            errors(&states[0], &states[2]),
            errors(&states[0], &states[4]),
        ];
        let physical = differences.iter().flatten().all(|&error| error <= 1e-6);
        accepted &= exact && physical;
        records.push(serde_json::json!({"native":native,"layout":layout,"implicit":implicit,"exact_self_repeats":exact,"errors":differences,"physical":physical}));
    }
    let report = serde_json::json!({"accepted":accepted,"particles":65,"dt":0.0001,"steps":4,"tolerance":1e-6,"records":records,"scope":"public detached window movement/save-reload; not canonical streaming or machine trajectory"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("integer_window_public_report={report}");
    assert!(accepted);
}
