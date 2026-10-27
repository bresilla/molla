#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::{
    MpmConfig, MpmPrescribedSolid, MpmSimulation, MpmSolidShape, MpmTimeStep,
};

fn prepare_transfer(sim: &mut MpmSimulation, active: bool) {
    if active {
        sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64)
            .unwrap();
        sim.set_deterministic_p2g(true).unwrap();
    } else {
        sim.prepare_compensated_grid_transfer(u64::from(sim.config().grid_size).pow(3) * 20)
            .unwrap();
        sim.set_compensated_grid_transfer(true).unwrap();
    }
}

#[test]
#[ignore = "requires GPU; public compensated reactions across storage and prepared-step changes"]
fn compensated_wrench_public_lifecycle() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = MpmConfig {
        grid_size: 35,
        max_particles: 65,
        max_rigid: 2,
        ..MpmConfig::sand()
    };
    let points: Vec<_> = (0..65)
        .map(|i| {
            [
                -0.12 + (i % 5) as f32 * 0.03,
                0.3 + (i / 5 % 5) as f32 * 0.03,
                -0.1 + (i / 25) as f32 * 0.03,
            ]
        })
        .collect();
    for mode in 0..3 {
        let mut sims = [0, 1].map(|_| {
            let mut sim = if mode == 2 {
                MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            sim.set_active_grid(mode != 0).unwrap();
            sim.set_particles(queue, &points).unwrap();
            sim.prepare_parallel_particle_tile_ordering(queue, 65)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
            prepare_transfer(&mut sim, mode != 0);
            sim
        });
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        assert!(sims[1].prepare_compensated_rigid_wrench(63).is_err());
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 0);
        assert!(sims[1].set_compensated_rigid_wrench(true).is_err());
        sims[1].prepare_compensated_rigid_wrench(64).unwrap();
        assert!(!sims[1].compensated_rigid_wrench_enabled());
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 64);
        sims[1].set_compensated_rigid_wrench(true).unwrap();
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        let resources = DeviceScope::start().unwrap();
        sims[1].prepare_compensated_rigid_wrench(64).unwrap();
        assert!(sims[1].prepare_compensated_rigid_wrench(63).is_err());
        assert_eq!(resources.finish().observations, 0);
        assert!(sims[1].compensated_rigid_wrench_enabled());
        assert!(sims[1].release_compensated_rigid_wrench().is_err());
        for round in 0..3 {
            if round > 0 {
                let stale = sims[1].prepare_step(0.0001, 1).unwrap();
                for sim in &mut sims {
                    sim.set_rigid_spheres(queue, &[]).unwrap();
                    sim.set_deterministic_p2g(false).unwrap();
                    sim.set_compensated_grid_transfer(false).unwrap();
                    if mode == 2 {
                        sim.reserve_sparse_tiles(if round == 1 { 256 } else { 512 })
                            .unwrap();
                    } else {
                        sim.set_active_grid((mode == 1) == (round == 2)).unwrap();
                    }
                    prepare_transfer(sim, mode == 2 || (mode == 1) == (round == 2));
                }
                assert!(sims[1].compensated_rigid_wrench_enabled());
                assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 64);
                let mut batch =
                    sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
                assert!(batch.encode_step(&stale).is_err());
            }
            for (prescribed, implicit, count) in [
                (false, false, 2),
                (true, false, 2),
                (false, true, 1),
                (false, false, 0),
            ] {
                let mut checkpoints = Vec::new();
                let mut wrenches = Vec::new();
                for sim in &mut sims {
                    sim.set_rigid_spheres(queue, &[]).unwrap();
                    sim.set_time_step(if implicit {
                        MpmTimeStep::Implicit { jacobi_iters: 2 }
                    } else {
                        MpmTimeStep::Explicit
                    });
                    sim.set_particles(queue, &points).unwrap();
                    let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
                    sim.restore_soil_with(queue, &checkpoint).unwrap();
                    if prescribed {
                        let solids: Vec<_> = (0..count)
                            .map(|i| MpmPrescribedSolid {
                                shape: if i == 0 {
                                    MpmSolidShape::Cylinder {
                                        radius: 0.06,
                                        half_width: 0.05,
                                    }
                                } else {
                                    MpmSolidShape::Box {
                                        half_extents: [0.05; 3],
                                    }
                                },
                                center: [-0.08 + i as f32 * 0.1, 0.35, -0.07],
                                orientation: [0.0, 0.0, 0.0, 1.0],
                                linear_velocity: [0.01, -0.005, 0.0],
                                angular_velocity: [0.0; 3],
                            })
                            .collect();
                        sim.set_prescribed_solids(queue, &solids).unwrap();
                    } else {
                        let spheres: Vec<_> = (0..count)
                            .map(|i| {
                                (
                                    [-0.08 + i as f32 * 0.1, 0.35, -0.07],
                                    0.06,
                                    [0.01, -0.005, 0.0],
                                )
                            })
                            .collect();
                        sim.set_rigid_spheres(queue, &spheres).unwrap();
                    }
                    let record = sim.prepare_step(0.0001, 2).unwrap();
                    {
                        let mut cancelled = sim.begin_command_batch(
                            device.create_command_encoder(&Default::default()),
                        );
                        cancelled.encode_step(&record).unwrap();
                    }
                    let encoder = device.create_command_encoder(&Default::default());
                    let before = device.get_internal_counters().hal;
                    let resources = DeviceScope::start().unwrap();
                    let transfers = TransferScope::start().unwrap();
                    let mut batch = sim.begin_command_batch(encoder);
                    batch.encode_step(&record).unwrap();
                    batch.submit(queue);
                    assert_eq!(resources.finish().observations, 0);
                    let transfers = transfers.finish();
                    assert_eq!(transfers.upload_bytes, 0);
                    assert_eq!(transfers.readback_bytes, 0);
                    let after = device.get_internal_counters().hal;
                    assert_eq!(before.bind_groups.read(), after.bind_groups.read());
                    assert_eq!(
                        before.compute_pipelines.read(),
                        after.compute_pipelines.read()
                    );
                    checkpoints.push(sim.checkpoint_soil_with(queue).unwrap().to_bytes());
                    let wrench = sim.read_rigid_wrenches_with(device, queue).unwrap();
                    assert_eq!(wrench.len(), count);
                    wrenches.push(wrench);
                }
                assert_eq!(checkpoints[0], checkpoints[1]);
                let mut worst = 0_f32;
                for ((fa, ta), (fb, tb)) in wrenches[0].iter().zip(&wrenches[1]) {
                    for (&a, &b) in fa.iter().chain(ta).zip(fb.iter().chain(tb)) {
                        assert!(a.is_finite() && b.is_finite());
                        worst = worst.max((a - b).abs());
                    }
                }
                assert!(worst <= 2e-5);
                if count > 0 {
                    assert!(
                        wrenches[1]
                            .iter()
                            .any(|(f, t)| f.iter().chain(t).any(|v| *v != 0.0))
                    );
                }
                println!(
                    "public_wrench mode={mode} round={round} prescribed={prescribed} implicit={implicit} bodies={count} checkpoint_exact=true worst_force={worst} hot_buffers=0 hot_bindings=0 hot_pipelines=0 transfers=0"
                );
            }
        }
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        sims[1].set_compensated_rigid_wrench(false).unwrap();
        sims[1].release_compensated_rigid_wrench().unwrap();
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 0);
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
    }
    let zero_cfg = MpmConfig {
        grid_size: 4,
        max_rigid: 0,
        max_particles: 1,
        ..cfg
    };
    let mut zero = MpmSimulation::new(&engine, zero_cfg).unwrap();
    assert!(zero.prepare_compensated_rigid_wrench(31).is_err());
    zero.prepare_compensated_rigid_wrench(32).unwrap();
    zero.set_compensated_rigid_wrench(true).unwrap();
    assert_eq!(zero.compensated_rigid_wrench_storage_bytes(), 32);
    zero.step_n_with(device, queue, 0.0001, 1).unwrap();
    assert!(
        zero.read_rigid_wrenches_with(device, queue)
            .unwrap()
            .is_empty()
    );
    let mut native = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(native.prepare_compensated_rigid_wrench(u64::MAX).is_err());
    assert!(native.set_compensated_rigid_wrench(true).is_err());
    assert_eq!(resources.finish().observations, 0);
}
