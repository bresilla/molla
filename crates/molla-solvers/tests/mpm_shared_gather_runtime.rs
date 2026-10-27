#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::{MpmConfig, MpmSimulation, MpmTimeStep};

fn prepare(sim: &mut MpmSimulation, shared: bool, compensated: bool) {
    if shared {
        sim.prepare_deterministic_p2g_shared_chunks(129 * 64, compensated)
            .unwrap();
    } else if compensated {
        sim.prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
            .unwrap();
    } else {
        sim.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
    }
    sim.set_deterministic_p2g(true).unwrap();
    assert_eq!(sim.deterministic_p2g_shared_chunks_enabled(), shared);
    assert_eq!(sim.deterministic_p2g_storage_bytes(), 129 * 64);
}

#[test]
#[ignore = "requires native GPU; public shared gather lifecycle and exact history"]
fn shared_gather_public_lifecycle() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = MpmConfig {
        grid_size: 35,
        grid_origin: [-1.0; 3],
        cell_size: 0.125,
        max_particles: 129,
        youngs: 1000.0,
        particle_mass: 0.02,
        particle_volume: 0.0000125,
        time_step: MpmTimeStep::Explicit,
        ..MpmConfig::sand()
    };
    let points: Vec<_> = (0..129)
        .map(|i| {
            [
                0.9 + (i % 5) as f32 * 0.031,
                1.1 + (i / 5 % 5) as f32 * 0.031,
                0.9 + (i / 25) as f32 * 0.031,
            ]
        })
        .collect();
    let mut cases = 0;
    for (native, compensated, mode) in [
        (true, false, 0),
        (true, false, 1),
        (true, false, 2),
        (false, false, 1),
        (false, false, 2),
        (false, true, 1),
        (false, true, 2),
    ] {
        let mut sims = [false, true].map(|shared| {
            let mut sim = match (native, mode) {
                (true, 2) => MpmSimulation::new_native_sparse(&engine, cfg, 128),
                (true, _) => MpmSimulation::new_native_explicit(&engine, cfg),
                (false, 2) => MpmSimulation::new_sparse(&engine, cfg, 128),
                (false, _) => MpmSimulation::new(&engine, cfg),
            }
            .unwrap();
            sim.set_active_grid(mode != 0).unwrap();
            let scope = DeviceScope::start().unwrap();
            assert!(
                sim.prepare_deterministic_p2g_shared_chunks(129 * 64 - 1, compensated)
                    .is_err()
            );
            assert!(
                sim.prepare_deterministic_p2g_shared_chunks(129 * 64, compensated)
                    .is_err()
            );
            assert_eq!(scope.finish().observations, 0);
            sim.prepare_parallel_particle_tile_ordering(queue, 129)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
            prepare(&mut sim, shared, compensated);
            sim
        });
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        sims[1].set_deterministic_p2g(false).unwrap();
        if compensated {
            sims[1]
                .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
                .unwrap();
        } else {
            sims[1]
                .prepare_deterministic_p2g_with_budget(129 * 64)
                .unwrap();
        }
        assert!(!sims[1].deterministic_p2g_shared_chunks_enabled());
        prepare(&mut sims[1], true, compensated);
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        let scope = DeviceScope::start().unwrap();
        sims[1]
            .prepare_deterministic_p2g_shared_chunks(129 * 64, compensated)
            .unwrap();
        assert!(
            sims[1]
                .prepare_deterministic_p2g_shared_chunks(129 * 64 - 1, compensated)
                .is_err()
        );
        if native {
            assert!(
                sims[1]
                    .prepare_deterministic_p2g_shared_chunks(129 * 64, true)
                    .is_err()
            );
        }
        assert_eq!(scope.finish().observations, 0);
        assert!(sims[1].deterministic_p2g_shared_chunks_enabled());
        assert!(sims[1].release_deterministic_p2g().is_err());
        for round in 0..3 {
            if round > 0 {
                for (index, sim) in sims.iter_mut().enumerate() {
                    sim.set_deterministic_p2g(false).unwrap();
                    if mode == 2 {
                        sim.reserve_sparse_tiles(if round == 1 { 256 } else { 512 })
                            .unwrap();
                    } else if native {
                        sim.set_active_grid((mode == 1) == (round == 2)).unwrap();
                    }
                    prepare(sim, index == 1, compensated);
                }
            }
            for count in [129, 63, 64, 65, 1, 0, 129] {
                let mut histories = Vec::new();
                for sim in &mut sims {
                    sim.set_particles_with_velocity(
                        queue,
                        &points[..count],
                        &vec![[0.01, -0.02, 0.003]; count],
                    )
                    .unwrap();
                    let saved = sim.checkpoint_soil_with(queue).unwrap();
                    sim.restore_soil_with(queue, &saved).unwrap();
                    let record = sim.prepare_step(0.0001, 2).unwrap();
                    {
                        let mut cancelled = sim.begin_command_batch(
                            device.create_command_encoder(&Default::default()),
                        );
                        cancelled.encode_step(&record).unwrap();
                    }
                    let encoder = device.create_command_encoder(&Default::default());
                    let before = device.get_internal_counters().hal;
                    let scope = DeviceScope::start().unwrap();
                    let traffic = TransferScope::start().unwrap();
                    let mut batch = sim.begin_command_batch(encoder);
                    batch.encode_step(&record).unwrap();
                    batch.submit(queue);
                    assert_eq!(scope.finish().observations, 0);
                    let traffic = traffic.finish();
                    assert_eq!(traffic.upload_bytes, 0);
                    assert_eq!(traffic.readback_bytes, 0);
                    let after = device.get_internal_counters().hal;
                    assert_eq!(before.bind_groups.read(), after.bind_groups.read());
                    assert_eq!(
                        before.compute_pipelines.read(),
                        after.compute_pipelines.read()
                    );
                    histories.push(sim.checkpoint_soil_with(queue).unwrap().to_bytes());
                    if sim.active_grid_enabled() {
                        let status = sim.read_active_grid_status_with(device, queue);
                        if count == 0 {
                            assert!(
                                status
                                    .unwrap_err()
                                    .to_string()
                                    .contains("MPM grid has not stepped since reset")
                            );
                        } else {
                            assert!(!status.unwrap().faulted());
                        }
                    }
                }
                assert_eq!(
                    histories[0], histories[1],
                    "native={native} compensated={compensated} mode={mode} round={round} count={count}"
                );
                cases += 1;
            }
        }
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        sims[1].set_deterministic_p2g(false).unwrap();
        sims[1].release_deterministic_p2g().unwrap();
        assert_eq!(sims[1].deterministic_p2g_storage_bytes(), 0);
        assert!(!sims[1].deterministic_p2g_shared_chunks_enabled());
        assert!(sims[1].set_deterministic_p2g(true).is_err());
        let mut batch =
            sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err());
    }
    assert_eq!(cases, 147);
    println!(
        "shared gather public lifecycle: {cases} exact paired histories; zero hot allocations/transfers"
    );
}
