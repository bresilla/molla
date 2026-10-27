#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::{MpmConfig, MpmSimulation, MpmSoilCheckpoint, MpmTimeStep};

#[test]
#[ignore = "requires GPU; non-test library compensated gather contract"]
fn compensated_gather_public_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut cfg = MpmConfig::sand();
        cfg.grid_size = 35;
        cfg.max_particles = 65;
        cfg.max_rigid = 1;
        cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
        let mut sim = if sparse {
            MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        sim.set_active_grid(true).unwrap();
        sim.prepare_parallel_particle_tile_ordering(queue, 65)
            .unwrap();
        sim.set_particle_tile_ordering(true).unwrap();
        assert!(!sim.deterministic_p2g_compensated_stiffness_enabled());
        sim.prepare_deterministic_p2g_with_budget(65 * 64).unwrap();
        sim.set_deterministic_p2g(true).unwrap();
        let stale = sim.prepare_step(0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        assert!(
            sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64 - 1)
                .is_err()
        );
        assert_eq!(resources.finish().observations, 0);
        assert!(!sim.deterministic_p2g_compensated_stiffness_enabled());
        sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64)
            .unwrap();
        assert!(sim.deterministic_p2g_compensated_stiffness_enabled());
        assert_eq!(sim.deterministic_p2g_storage_bytes(), 65 * 64);
        {
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        let resources = DeviceScope::start().unwrap();
        sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64)
            .unwrap();
        assert_eq!(resources.finish().observations, 0);
        for count in [65, 1, 0, 65] {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    [
                        -0.12 + (i % 5) as f32 * 0.03,
                        0.3 + (i / 5 % 5) as f32 * 0.03,
                        -0.1 + (i / 25) as f32 * 0.03,
                    ]
                })
                .collect();
            sim.set_particles_with_velocity(queue, &points, &vec![[0.01, -0.01, 0.0]; count])
                .unwrap();
            let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
            let decoded = MpmSoilCheckpoint::from_bytes(&checkpoint.to_bytes(), 65).unwrap();
            sim.restore_soil_with(queue, &decoded).unwrap();
            assert!(sim.deterministic_p2g_compensated_stiffness_enabled());
            let record = sim.prepare_step(0.0001, 2).unwrap();
            let encoder = device.create_command_encoder(&Default::default());
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let mut batch = sim.begin_command_batch(encoder);
            batch.encode_step(&record).unwrap();
            batch.submit(queue);
            assert_eq!(resources.finish().observations, 0);
            let counts = transfers.finish();
            assert_eq!(counts.upload_bytes, 0);
            assert_eq!(counts.readback_bytes, 0);
            let positions = sim.read_positions_with(device, queue).unwrap();
            assert_eq!(positions.len(), count);
            assert!(positions.iter().flatten().all(|x| x.is_finite()));
            assert_eq!(
                sim.checkpoint_soil_with(queue).unwrap().particle_count(),
                count as u32
            );
            println!(
                "public_compensated_gather sparse={sparse} count={count} hot_buffers=0 uploads=0 readbacks=0"
            );
        }
        let stale = sim.prepare_step(0.0001, 1).unwrap();
        sim.prepare_deterministic_p2g_with_budget(65 * 64).unwrap();
        assert!(!sim.deterministic_p2g_compensated_stiffness_enabled());
        {
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        assert!(sim.release_deterministic_p2g().is_err());
        sim.set_deterministic_p2g(false).unwrap();
        sim.release_deterministic_p2g().unwrap();
        assert_eq!(sim.deterministic_p2g_storage_bytes(), 0);
    }
}
