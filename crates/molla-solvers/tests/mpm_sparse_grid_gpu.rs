#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::{MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};

fn config(size: u32, implicit: bool) -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = size;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 512;
    cfg.max_rigid = 4;
    cfg.youngs = 3000.0;
    cfg.particle_volume = 0.021_f32.powi(3);
    cfg.particle_mass = cfg.particle_volume * 1600.0;
    cfg.time_step = if implicit {
        MpmTimeStep::Implicit { jacobi_iters: 3 }
    } else {
        MpmTimeStep::Explicit
    };
    cfg
}

fn particles(n: usize, offset: [f32; 3]) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            [
                offset[0] + (i % 7) as f32 * 0.021 + 0.00013,
                offset[1] + (i / 7 % 7) as f32 * 0.021 + 0.00031,
                offset[2] + (i / 49) as f32 * 0.021 + 0.00047,
            ]
        })
        .collect()
}

fn compare(engine: &Engine, dense: &mut MpmSimulation, sparse: &mut MpmSimulation) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (name, a, b) in [
        (
            "positions",
            dense.read_positions_with(device, queue).unwrap(),
            sparse.read_positions_with(device, queue).unwrap(),
        ),
        (
            "velocities",
            dense.read_velocities_with(device, queue).unwrap(),
            sparse.read_velocities_with(device, queue).unwrap(),
        ),
    ] {
        assert_eq!(a.len(), b.len());
        for (index, (a, b)) in a.as_flattened().iter().zip(b.as_flattened()).enumerate() {
            assert!((a - b).abs() <= 2e-5, "{name}[{index}] {a} != {b}");
        }
    }
    let a = dense.read_rigid_wrenches(engine).unwrap();
    let b = sparse.read_rigid_wrenches(engine).unwrap();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(&b) {
        for (a, b) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
            assert!(
                (a - b).abs() <= 2e-5 * a.abs().max(1.0),
                "wrench {a} != {b}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_sparse_pool_matches_dense_materials_rigids_edges_and_reseeding() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (size, implicit, two_pass) in [(35, true, true), (32, false, true), (64, true, false)] {
        let cfg = config(size, implicit);
        let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
        let mut sparse = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
        assert!(sparse.active_grid_enabled());
        assert!(sparse.set_active_grid(false).is_err());
        assert!(sparse.read_active_grid_status_with(device, queue).is_err());
        assert!(sparse.read_grid_mass_momentum(&engine).is_err());
        for (window, (count, offset)) in [
            (129, [0.24, 0.1, -0.2]),
            (97, [-0.75, -0.7, -0.65]),
            (1, [1.17, 1.17, 1.17]),
            (33, [-1.01, -0.95, -0.97]),
            (129, [0.24, 0.1, -0.2]),
        ]
        .into_iter()
        .enumerate()
        {
            let points = particles(count, offset);
            for sim in [&mut dense, &mut sparse] {
                sim.set_p2g_mode(two_pass);
                sim.set_particles(queue, &[]).unwrap();
                sim.set_particles_with_velocity(queue, &points, &vec![[0.3, -0.02, 0.1]; count])
                    .unwrap();
                sim.set_materials(
                    &[
                        MpmRheology::Sand {
                            friction_angle: 0.5,
                        },
                        MpmRheology::Viscous {
                            velocity_damping: 0.1,
                            affine_damping: 0.2,
                        },
                    ],
                    (0..count).map(|i| (i % 2) as u32).collect(),
                )
                .unwrap();
                sim.set_rigid_spheres(
                    queue,
                    &[
                        (
                            [0.30 + 0.01 * window as f32, 0.18, -0.16],
                            0.065,
                            [0.1, -0.01, 0.0],
                        ),
                        ([-0.85, -0.8, -0.85], 0.11, [0.0; 3]),
                    ],
                )
                .unwrap();
                sim.step_n_with(device, queue, 0.0001, 2).unwrap();
            }
            compare(&engine, &mut dense, &mut sparse);
            let stats = sparse.read_active_grid_stats_with(device, queue).unwrap();
            assert_eq!(stats.completed_substeps, (window as u32 + 1) * 2);
            assert!(stats.active_tiles <= 64);
        }
        println!(
            "size={size} implicit={implicit} two_pass={two_pass} dense_bytes={} compact_bytes={}",
            dense.grid_storage_bytes(),
            sparse.grid_storage_bytes()
        );
        assert!(sparse.grid_storage_bytes() < dense.grid_storage_bytes());
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_sparse_pool_recycles_slots_during_continuous_transport() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for implicit in [false, true] {
        let mut cfg = config(64, implicit);
        cfg.youngs = 0.0;
        cfg.gravity_y = -0.5;
        let points = particles(129, [-0.6, 0.1, 0.1]);
        let mass = points.len() as f64 * f64::from(cfg.particle_mass);
        let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
        let mut sparse = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
        for sim in [&mut dense, &mut sparse] {
            sim.set_particles_with_velocity(queue, &points, &vec![[4.0, 0.0, 0.0]; points.len()])
                .unwrap();
        }
        for window in 1..=8 {
            for sim in [&mut dense, &mut sparse] {
                sim.step_n_with(device, queue, 1.0 / 512.0, 16).unwrap();
            }
            compare(&engine, &mut dense, &mut sparse);
            let (grid_mass, _) = sparse.read_grid_storage_mass_momentum(&engine).unwrap();
            assert_eq!(grid_mass.len(), 64 * 64);
            let grid_mass: f64 = grid_mass.into_iter().map(f64::from).sum();
            assert!((grid_mass - mass).abs() <= 2e-5 * mass);
            let mut momentum = [0.0_f64; 3];
            for v in sparse.read_velocities_with(device, queue).unwrap() {
                for axis in 0..3 {
                    momentum[axis] += f64::from(v[axis]) * f64::from(cfg.particle_mass);
                }
            }
            let expected = [
                mass * 4.0,
                mass * f64::from(cfg.gravity_y) * f64::from(window * 16) / 512.0,
                0.0,
            ];
            for axis in 0..3 {
                assert!(
                    (momentum[axis] - expected[axis]).abs() <= 2e-5 * expected[axis].abs().max(1.0)
                );
            }
            assert_eq!(
                sparse
                    .read_active_grid_stats_with(device, queue)
                    .unwrap()
                    .completed_substeps,
                window * 16
            );
        }
        assert!(
            sparse
                .read_positions_with(device, queue)
                .unwrap()
                .iter()
                .zip(&points)
                .all(|(a, b)| a[0] - b[0] > 0.99)
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_sparse_pool_overflow_freezes_particles_until_explicit_reset() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = config(64, true);
    assert!(MpmSimulation::new_sparse(&engine, cfg, 0).is_err());
    assert!(MpmSimulation::new_sparse(&engine, cfg, 4097).is_err());
    let points = [[0.1; 3], [1.1; 3]];
    let velocities = [[0.3, 0.0, 0.0]; 2];
    for two_pass in [false, true] {
        let mut sim = MpmSimulation::new_sparse(&engine, cfg, 8).unwrap();
        sim.set_p2g_mode(two_pass);
        sim.set_particles_with_velocity(queue, &points, &velocities)
            .unwrap();
        for _ in 0..2 {
            sim.step_n_with(device, queue, 0.0001, 3).unwrap();
            let stats = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(stats.overflow);
            assert_eq!(stats.completed_substeps, 0);
            assert!(sim.read_active_grid_stats_with(device, queue).is_err());
            assert!(sim.read_active_tile_ids_with(device, queue).is_err());
            assert_eq!(sim.read_positions_with(device, queue).unwrap(), points);
            assert_eq!(sim.read_velocities_with(device, queue).unwrap(), velocities);
        }
        sim.set_particles(queue, &points[..1]).unwrap();
        sim.reset_active_grid();
        sim.step_n_with(device, queue, 0.0001, 2).unwrap();
        let stats = sim.read_active_grid_stats_with(device, queue).unwrap();
        assert_eq!(stats.completed_substeps, 2);
        assert_eq!(stats.active_tiles, 8);
        assert!(!stats.overflow);
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_sparse_pool_bounds_physical_storage_in_large_domains() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for size in [32, 96, 512, 1536] {
        let cfg = config(size, true);
        let mut sim = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
        let points = particles(129, [0.1; 3]);
        sim.set_particles(queue, &points).unwrap();
        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.step_n_with(device, queue, 0.0001, 4).unwrap();
        let resources = resources.finish();
        let transfers = transfers.finish();
        assert!(!resources.invalid && !transfers.overflowed);
        assert_eq!(resources.observations, 0);
        assert_eq!(transfers.readback_bytes, 0);
        let stats = sim.read_active_grid_stats_with(device, queue).unwrap();
        assert_eq!(stats.active_tiles, 8);
        assert_eq!(stats.completed_substeps, 5);
        let expected = 64 * 64 * 56 + (128 + 64) * 8 + 16;
        assert_eq!(sim.grid_storage_bytes(), expected);
        println!("grid={size}^3 physical_cell_capacity=4096 active_tiles=8 grid_bytes={expected}");
        if size == 1536 {
            let remote = particles(129, [93.0; 3]);
            sim.set_particles(queue, &remote).unwrap();
            sim.step_n_with(device, queue, 0.0001, 2).unwrap();
            let stats = sim.read_active_grid_stats_with(device, queue).unwrap();
            assert!(!stats.overflow && stats.active_tiles > 0 && stats.active_tiles <= 64);
            assert!(
                sim.read_positions_with(device, queue)
                    .unwrap()
                    .iter()
                    .all(|p| p.iter().all(|v| v.is_finite() && *v > 92.0))
            );
            let positions = sim.read_positions_with(device, queue).unwrap();
            for velocity in sim.read_velocities_with(device, queue).unwrap() {
                assert!(velocity[0].abs() <= 2e-5 && velocity[2].abs() <= 2e-5);
                assert!((velocity[1] - 2.0 * cfg.gravity_y * 0.0001).abs() <= 2e-5);
            }
            let (mass, _) = sim.read_grid_storage_mass_momentum(&engine).unwrap();
            let mass: f64 = mass.into_iter().map(f64::from).sum();
            let expected_mass = remote.len() as f64 * f64::from(cfg.particle_mass);
            assert!((mass - expected_mass).abs() <= 2e-5 * expected_mass);
            sim.set_rigid_spheres(queue, &[([0.0; 3], 1000.0, [0.0; 3])])
                .unwrap();
            sim.step_n_with(device, queue, 0.0001, 1).unwrap();
            let fault = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(fault.overflow);
            assert_eq!(fault.completed_substeps, stats.completed_substeps);
            assert_eq!(sim.read_positions_with(device, queue).unwrap(), positions);
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_sparse_pool_growth_preserves_history_and_partial_batch_progress() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = config(64, false);
    cfg.youngs = 0.0;
    cfg.gravity_y = 0.0;
    let points = [[0.1, 0.1, 0.1], [0.21, 0.1, 0.1]];
    let velocity = [[4.0, 0.0, 0.0]; 2];
    let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
    let mut sparse = MpmSimulation::new_sparse(&engine, cfg, 8).unwrap();
    for sim in [&mut dense, &mut sparse] {
        sim.set_particles_with_velocity(queue, &points, &velocity)
            .unwrap();
        sim.step_n_with(device, queue, 0.02, 3).unwrap();
    }
    let status = sparse.read_active_grid_status_with(device, queue).unwrap();
    assert!(status.overflow);
    assert_eq!(status.completed_substeps, 2);
    sparse.reserve_sparse_tiles(32).unwrap();
    assert!(sparse.read_active_grid_status_with(device, queue).is_err());
    sparse.step_n_with(device, queue, 0.02, 1).unwrap();
    compare(&engine, &mut dense, &mut sparse);
    assert_eq!(
        sparse
            .read_active_grid_stats_with(device, queue)
            .unwrap()
            .completed_substeps,
        1
    );

    let cfg = config(35, true);
    let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
    let mut sparse = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
    let points = particles(129, [0.24, 0.1, -0.2]);
    for sim in [&mut dense, &mut sparse] {
        sim.set_particles_with_velocity(queue, &points, &vec![[0.3, -0.02, 0.1]; points.len()])
            .unwrap();
        sim.set_materials(
            &[
                MpmRheology::Snow {
                    theta_c: 0.025,
                    theta_s: 0.0075,
                    hardening: 10.0,
                },
                MpmRheology::Sand {
                    friction_angle: 0.5,
                },
            ],
            (0..points.len()).map(|i| (i % 2) as u32).collect(),
        )
        .unwrap();
        sim.set_rigid_spheres(queue, &[([0.30, 0.18, -0.16], 0.065, [0.1, -0.01, 0.0])])
            .unwrap();
        sim.step_n_with(device, queue, 0.0001, 3).unwrap();
    }
    sparse.reserve_sparse_tiles(128).unwrap();
    for _ in 0..4 {
        for sim in [&mut dense, &mut sparse] {
            sim.step_n_with(device, queue, 0.0001, 2).unwrap();
        }
        compare(&engine, &mut dense, &mut sparse);
    }
    let bytes = sparse.grid_storage_bytes();
    sparse.reserve_sparse_tiles(64).unwrap();
    assert_eq!(sparse.grid_storage_bytes(), bytes);
    assert!(sparse.reserve_sparse_tiles(730).is_err());
    assert_eq!(sparse.grid_storage_bytes(), bytes);
    assert!(dense.reserve_sparse_tiles(64).is_err());
}
