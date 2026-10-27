#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::{MpmConfig, MpmSimulation, MpmSoilCheckpoint, MpmTimeStep};

#[test]
#[ignore = "requires GPU; non-test library compensated stiffness contract"]
fn compensated_stiffness_public_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let mut cfg = MpmConfig::sand();
        cfg.grid_size = 35;
        cfg.max_particles = 65;
        cfg.max_rigid = 1;
        cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
        let mut sim = if mode >= 2 {
            MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        if mode == 3 {
            sim.set_deferred_p2g_lookup(true).unwrap();
        }
        let bytes = if mode >= 2 {
            128 * 64 * 4
        } else {
            u64::from(cfg.grid_size).pow(3) * 4
        };
        assert!(!sim.compensated_stiffness_enabled());
        assert!(sim.set_compensated_stiffness(true).is_err());
        let scope = DeviceScope::start().unwrap();
        assert!(sim.prepare_compensated_stiffness(bytes - 1).is_err());
        assert_eq!(scope.finish().observations, 0);
        sim.prepare_compensated_stiffness(bytes).unwrap();
        sim.set_compensated_stiffness(true).unwrap();
        assert_eq!(sim.compensated_stiffness_storage_bytes(), bytes);
        let scope = DeviceScope::start().unwrap();
        sim.prepare_compensated_stiffness(bytes).unwrap();
        assert_eq!(scope.finish().observations, 0);
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
            assert!(sim.compensated_stiffness_enabled());
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
                "public_compensated_stiffness mode={mode} count={count} tail_bytes={bytes} hot_buffers=0 uploads=0 readbacks=0"
            );
        }
        assert!(sim.release_compensated_stiffness().is_err());
        let stale = sim.prepare_step(0.0001, 1).unwrap();
        sim.set_compensated_stiffness(false).unwrap();
        sim.release_compensated_stiffness().unwrap();
        assert_eq!(sim.compensated_stiffness_storage_bytes(), 0);
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err());
    }
}
