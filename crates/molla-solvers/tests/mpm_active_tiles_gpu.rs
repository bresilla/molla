#![cfg(feature = "gpu")]

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_transfer_deferred_lookup_matches_materials_edges_and_scattered_particles() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (size, implicit) in [(32, false), (35, true)] {
        let cfg = config(size, implicit);
        let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
        let mut compact = MpmSimulation::new_sparse(&engine, cfg, 256).unwrap();
        assert!(dense.set_deferred_p2g_lookup(true).is_err());
        assert!(!compact.uses_deferred_p2g_lookup());
        compact.set_deferred_p2g_lookup(true).unwrap();
        compact.set_p2g_mode(false);
        assert!(!compact.uses_deferred_p2g_lookup());
        assert!(compact.set_deferred_p2g_lookup(true).is_err());
        compact.set_p2g_mode(true);
        compact.set_deferred_p2g_lookup(true).unwrap();
        for scattered in [false, true] {
            let mut points = particles(129, [-0.3, -0.1, -0.2]);
            if scattered {
                for (i, p) in points.iter_mut().enumerate() {
                    if i % 2 == 0 {
                        p[0] += 0.65;
                    }
                    if i % 3 == 0 {
                        p[2] += 0.65;
                    }
                    if i % 31 == 0 {
                        *p = [-1.03, -0.4, -0.2];
                    }
                }
                points.reverse();
            }
            for sim in [&mut dense, &mut compact] {
                sim.set_particles_with_velocity(
                    queue,
                    &points,
                    &vec![[0.1, -0.02, 0.03]; points.len()],
                )
                .unwrap();
                sim.set_materials(
                    &[
                        MpmRheology::Sand {
                            friction_angle: 0.5,
                        },
                        MpmRheology::Snow {
                            theta_c: 0.025,
                            theta_s: 0.0075,
                            hardening: 10.,
                        },
                        MpmRheology::Viscous {
                            velocity_damping: 0.1,
                            affine_damping: 0.2,
                        },
                    ],
                    (0..points.len()).map(|i| (i % 3) as u32).collect(),
                )
                .unwrap();
                sim.set_rigid_spheres(queue, &[([-0.2, -0.05, -0.2], 0.08, [0.01, 0., 0.])])
                    .unwrap();
            }
            for _ in 0..4 {
                dense.step_n_with(device, queue, 0.0001, 1).unwrap();
                compact.step_n_with(device, queue, 0.0001, 1).unwrap();
                compare_result_mode(&engine, &mut dense, &mut compact, true).unwrap();
            }
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_transfer_deferred_lookup_preserves_remote_cells_and_overflow_freeze() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = config(1536, false);
    let mut reference = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
    let mut candidate = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
    candidate.set_deferred_p2g_lookup(true).unwrap();
    let points = particles(129, [93., 93., 93.]);
    for sim in [&mut reference, &mut candidate] {
        sim.set_particles(queue, &points).unwrap();
        sim.step_n_with(device, queue, 0.0001, 4).unwrap();
    }
    close(
        "remote positions",
        reference
            .read_positions_with(device, queue)
            .unwrap()
            .as_flattened(),
        candidate
            .read_positions_with(device, queue)
            .unwrap()
            .as_flattened(),
        false,
    )
    .unwrap();
    close(
        "remote velocity",
        reference
            .read_velocities_with(device, queue)
            .unwrap()
            .as_flattened(),
        candidate
            .read_velocities_with(device, queue)
            .unwrap()
            .as_flattened(),
        false,
    )
    .unwrap();
    let snapshot = |sim: &mut MpmSimulation| {
        let ids = sim.read_active_tile_ids_with(device, queue).unwrap();
        let (mass, momentum) = sim.read_grid_storage_mass_momentum(&engine).unwrap();
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| {
                (
                    id,
                    (
                        mass[i * 64..(i + 1) * 64].to_vec(),
                        momentum[i * 192..(i + 1) * 192].to_vec(),
                    ),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let a = snapshot(&mut reference);
    let b = snapshot(&mut candidate);
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (id, (mass, momentum)) in &a {
        close("remote grid mass", mass, &b[id].0, true).unwrap();
        close("remote grid momentum", momentum, &b[id].1, true).unwrap();
    }
    let mass: f32 = b.values().flat_map(|v| v.0.iter()).sum();
    assert!((mass - points.len() as f32 * cfg.particle_mass).abs() <= 2e-5 * mass.max(1.));
    let before = candidate.read_positions_with(device, queue).unwrap();
    candidate
        .set_rigid_spheres(queue, &[([0., 0., 0.], 1000., [0.; 3])])
        .unwrap();
    candidate.step_n_with(device, queue, 0.0001, 1).unwrap();
    assert!(
        candidate
            .read_active_grid_status_with(device, queue)
            .unwrap()
            .overflow
    );
    assert_eq!(
        candidate.read_positions_with(device, queue).unwrap(),
        before
    );
}

fn profiling_engine() -> Engine {
    Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .expect("real GPU with in-pass timestamps required")
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_profile_rejects_missing_device_features() {
    let engine = Engine::new_gpu().unwrap();
    let mut sim = MpmSimulation::new(&engine, config(8, false)).unwrap();
    assert!(sim.prepare_step_profiling(1).is_err());
    assert!(sim.read_step_profile().unwrap().is_none());
    sim.set_particles(engine.queue().unwrap(), &[[0., 0., 0.]])
        .unwrap();
    assert!(
        sim.step_n_profiled_with(engine.device().unwrap(), engine.queue().unwrap(), 0.0001, 1)
            .is_err()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn mpm_profile_preserves_stages_physics_and_capture_lifecycle() {
    use molla_solvers::MpmProfileStage as Stage;
    use molla_solvers::core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = profiling_engine();
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (mode, implicit) in [
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        (2, false),
        (2, true),
    ] {
        let cfg = config(32, implicit);
        let mut reference = MpmSimulation::new(&engine, cfg).unwrap();
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        let points = particles(129, [-0.2, -0.2, -0.2]);
        for s in [&mut reference, &mut sim] {
            s.set_particles_with_velocity(queue, &points, &vec![[0.02, -0.01, 0.03]; points.len()])
                .unwrap();
            s.set_rigid_spheres(queue, &[([-0.16, -0.13, -0.2], 0.06, [0.01, 0., 0.])])
                .unwrap();
        }
        sim.prepare_step_profiling(2).unwrap();
        assert!(sim.prepare_step_profiling(0).is_err());
        assert!(sim.prepare_step_profiling(205).is_err());
        assert!(sim.step_n_profiled_with(device, queue, 0.0001, 3).is_err());
        assert!(
            sim.step_n_profiled_with(device, queue, f32::NAN, 1)
                .is_err()
        );
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.step_n_profiled_with(device, queue, 0.0001, 2).unwrap();
        let transfers = transfers.finish();
        let resources = resources.finish();
        assert!(!resources.invalid && !transfers.overflowed);
        assert_eq!(resources.observations, 0);
        assert_eq!(transfers.readback_bytes, 0);
        assert!(sim.step_n_profiled_with(device, queue, 0.0001, 1).is_err());
        assert!(sim.prepare_step_profiling(2).is_err());
        reference.step_n_with(device, queue, 0.0001, 2).unwrap();
        compare_result_mode(&engine, &mut reference, &mut sim, mode == 2).unwrap();
        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        reference.step_n_with(device, queue, 0.0001, 1).unwrap();
        let transfers = TransferScope::start().unwrap();
        let profile = sim.read_step_profile().unwrap().unwrap();
        let transfers = transfers.finish();
        assert_eq!(transfers.readback_bytes, profile.readback_bytes);
        assert_eq!(profile.requested_substeps, 2);
        assert_eq!(
            profile.requested_simulated_seconds,
            f64::from(0.0001_f32) * 2.
        );
        assert_eq!(profile.particles, 129);
        assert_eq!(profile.active_grid, mode != 0);
        assert_eq!(profile.compact_grid, mode == 2);
        assert_eq!(
            profile.stages.len(),
            2 * (8 + usize::from(implicit) + usize::from(mode != 0)) + usize::from(mode != 0)
        );
        assert_eq!(profile.readback_bytes, profile.stages.len() as u64 * 16);
        assert!(
            profile
                .stages
                .iter()
                .all(|t| t.elapsed_ms.is_finite() && t.elapsed_ms >= 0.)
        );
        assert!(profile.stages.iter().map(|t| t.elapsed_ms).sum::<f64>() > 0.);
        for step in 0..2 {
            let actual: Vec<_> = profile
                .stages
                .iter()
                .filter(|t| t.substep == Some(step))
                .map(|t| t.stage)
                .collect();
            let mut expected = vec![Stage::GridReset];
            if mode != 0 {
                expected.push(Stage::ActiveTiles);
            }
            expected.extend([
                Stage::RigidReset,
                Stage::ParticleToGrid,
                Stage::RigidImprint,
                Stage::GridUpdate,
            ]);
            if implicit {
                expected.push(Stage::Implicit);
            }
            expected.extend([Stage::RigidForce, Stage::GridToParticle, Stage::Plasticity]);
            assert_eq!(actual, expected);
        }
        assert!(sim.read_step_profile().unwrap().is_none());
        compare_result_mode(&engine, &mut reference, &mut sim, mode == 2).unwrap();
        sim.step_n_profiled_with(device, queue, 0.0001, 1).unwrap();
        let next = sim.read_step_profile().unwrap().unwrap();
        assert!(
            next.stages
                .iter()
                .all(|t| t.stage != Stage::Initialization && t.substep == Some(0))
        );
        sim.set_particles(queue, &[]).unwrap();
        sim.step_n_profiled_with(device, queue, 0.0001, 1).unwrap();
        assert!(sim.read_step_profile().unwrap().is_none());
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_profile_overflow_reports_requested_not_completed_duration() {
    let engine = profiling_engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, config(32, false), 1).unwrap();
    let points = particles(129, [-0.2, -0.2, -0.2]);
    sim.set_particles(queue, &points).unwrap();
    sim.prepare_step_profiling(2).unwrap();
    sim.step_n_profiled_with(device, queue, 0.0001, 2).unwrap();
    let profile = sim.read_step_profile().unwrap().unwrap();
    let status = sim.read_active_grid_status_with(device, queue).unwrap();
    assert!(status.overflow);
    assert_eq!(status.completed_substeps, 0);
    assert_eq!(profile.requested_substeps, 2);
    assert_eq!(sim.read_positions_with(device, queue).unwrap(), points);
}

#[test]
#[ignore = "needs a real GPU adapter; diagnostic stage timing"]
fn mpm_profile_stage_benchmark() {
    use std::collections::BTreeMap;
    use std::time::Instant;
    let engine = profiling_engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.65 + (i % 40) as f32 * 0.025,
                -0.65 + (i / 40 % 40) as f32 * 0.025,
                -0.65 + (i / 1600) as f32 * 0.025,
            ]
        })
        .collect();
    let velocities = vec![[0.03, -0.02, 0.01]; points.len()];
    for implicit in [false, true] {
        let mut cfg = config(64, implicit);
        cfg.max_particles = 50_000;
        cfg.particle_volume = 0.025_f32.powi(3);
        cfg.particle_mass = 1600. * cfg.particle_volume;
        for mode in 0..8 {
            let name = [
                "dense",
                "active-tiles",
                "compact-tiles",
                "compact-deferred",
                "dense-ordered",
                "compact-ordered",
                "dense-parallel-ordered",
                "compact-parallel-ordered",
            ][mode];
            let mut sim = if matches!(mode, 2 | 3 | 5 | 7) {
                MpmSimulation::new_sparse(&engine, cfg, 256).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            if mode == 1 {
                sim.set_active_grid(true).unwrap();
            }
            if mode == 3 {
                sim.set_deferred_p2g_lookup(true).unwrap();
            }
            if mode >= 4 {
                if mode >= 6 {
                    sim.prepare_parallel_particle_tile_ordering(queue, 50_000)
                        .unwrap();
                } else {
                    sim.prepare_particle_tile_ordering(queue, 50_000).unwrap();
                }
                sim.set_particle_tile_ordering(true).unwrap();
            }
            sim.prepare_step_profiling(4).unwrap();
            let mut completed_substeps = 0u32;
            sim.set_rigid_spheres(
                queue,
                &[
                    ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.]),
                    ([-0.87, 0.8, -0.87], 0.08, [0.; 3]),
                ],
            )
            .unwrap();
            println!(
                "{}",
                serde_json::json!({"schema":1,"record":"profile-manifest","case":"mpm-profile",
                "mode":name,"config":format!("{cfg:?}"),"scope":"solver-only diagnostic",
                "dt":0.0001_f32,"substeps":4,"particles":50000,"rounds":3,"warmup_rounds":1,
                "two_pass_p2g":true,"deferred_p2g_lookup":sim.uses_deferred_p2g_lookup(),
                "particle_tile_ordering":sim.particle_tile_ordering_enabled(),"particle_ordering_bytes":sim.particle_tile_ordering_storage_bytes(),
                "parallel_particle_sort":sim.particle_tile_ordering_uses_parallel_sort(),
                "parity_checked":false,"performance_accepted":false})
            );
            for round in 0..4 {
                for profiled in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    sim.set_particles_with_velocity(queue, &points, &velocities)
                        .unwrap();
                    queue.submit([]);
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    let start = Instant::now();
                    if profiled {
                        sim.step_n_profiled_with(device, queue, 0.0001, 4).unwrap();
                    } else {
                        sim.step_n_with(device, queue, 0.0001, 4).unwrap();
                    }
                    let encode_ms = start.elapsed().as_secs_f64() * 1000.;
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
                    if sim.active_grid_enabled() {
                        let status = sim.read_active_grid_stats_with(device, queue).unwrap();
                        assert_eq!(
                            status.completed_substeps.wrapping_sub(completed_substeps),
                            4
                        );
                        completed_substeps = status.completed_substeps;
                    }
                    let profile = if profiled {
                        sim.read_step_profile().unwrap()
                    } else {
                        None
                    };
                    if round == 0 {
                        continue;
                    }
                    let mut stage_ms: BTreeMap<&str, f64> = BTreeMap::new();
                    let mut bytes = 0;
                    if let Some(profile) = profile {
                        bytes = profile.readback_bytes;
                        for stage in profile.stages {
                            *stage_ms.entry(stage.stage.name()).or_default() += stage.elapsed_ms;
                        }
                    }
                    println!(
                        "{}",
                        serde_json::json!({"schema":1,"record":"profile-sample","case":"mpm-profile",
                        "mode":name,"implicit":implicit,"round":round-1,"profiled":profiled,
                        "wall_ms":wall_ms,"encode_submit_ms":encode_ms,"stage_ms":stage_ms,
                        "timestamp_readback_bytes":bytes,"completed_substeps":4,"requested_simulated_seconds":f64::from(0.0001_f32)*4.,
                        "readback_in_wall_scope":false,"performance_accepted":false})
                    );
                }
            }
        }
    }
}

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

fn close(label: &str, a: &[f32], b: &[f32], relative: bool) -> Result<(), String> {
    assert_eq!(a.len(), b.len());
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        let tolerance = 2e-5 * if relative { b.abs().max(1.0) } else { 1.0 };
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
            return Err(format!(
                "{label}[{index}]: {a} != {b}, tolerance={tolerance}"
            ));
        }
    }
    Ok(())
}

fn compare(engine: &Engine, dense: &mut MpmSimulation, tiled: &mut MpmSimulation) {
    compare_result(engine, dense, tiled).unwrap();
}

fn compare_result(
    engine: &Engine,
    dense: &mut MpmSimulation,
    tiled: &mut MpmSimulation,
) -> Result<(), String> {
    compare_result_mode(engine, dense, tiled, false)
}

fn compare_result_mode(
    engine: &Engine,
    dense: &mut MpmSimulation,
    tiled: &mut MpmSimulation,
    compact: bool,
) -> Result<(), String> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    close(
        "position",
        dense
            .read_positions_with(device, queue)
            .unwrap()
            .as_flattened(),
        tiled
            .read_positions_with(device, queue)
            .unwrap()
            .as_flattened(),
        false,
    )?;
    close(
        "velocity",
        dense
            .read_velocities_with(device, queue)
            .unwrap()
            .as_flattened(),
        tiled
            .read_velocities_with(device, queue)
            .unwrap()
            .as_flattened(),
        false,
    )?;
    let (mass_a, momentum_a) = dense.read_grid_mass_momentum(engine).unwrap();
    let (mass_b, momentum_b) = if compact {
        let (slots, momenta) = tiled.read_grid_storage_mass_momentum(engine).unwrap();
        let ids = tiled.read_active_tile_ids_with(device, queue).unwrap();
        let n = tiled.config().grid_size as usize;
        let width = n.div_ceil(4);
        let mut mass = vec![0.0; n.pow(3)];
        let mut momentum = vec![0.0; n.pow(3) * 3];
        let mut seen = std::collections::BTreeSet::new();
        for (slot, tile) in ids.into_iter().enumerate() {
            if !seen.insert(tile) || tile as usize >= width.pow(3) {
                return Err(format!("duplicate or invalid world tile {tile}"));
            }
            let tile = tile as usize;
            for local in 0..64 {
                let x = (tile % width) * 4 + local % 4;
                let y = (tile / width % width) * 4 + local / 4 % 4;
                let z = tile / (width * width) * 4 + local / 16;
                if x >= n || y >= n || z >= n {
                    continue;
                }
                let world = x + n * (y + n * z);
                let physical = slot * 64 + local;
                mass[world] = slots[physical];
                momentum[world * 3..world * 3 + 3]
                    .copy_from_slice(&momenta[physical * 3..physical * 3 + 3]);
            }
        }
        (mass, momentum)
    } else {
        tiled.read_grid_mass_momentum(engine).unwrap()
    };
    close("grid mass", &mass_a, &mass_b, true)?;
    close("grid momentum", &momentum_a, &momentum_b, true)?;
    for (a, b) in dense
        .read_rigid_wrenches(engine)
        .unwrap()
        .iter()
        .zip(tiled.read_rigid_wrenches(engine).unwrap())
    {
        close("rigid force", &a.0, &b.0, true)?;
        close("rigid torque", &a.1, &b.1, true)?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_active_tiles_match_dense_explicit_implicit_materials_and_rigids() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (size, implicit, two_pass) in [(32, false, true), (35, true, true), (32, true, false)] {
        let cfg = config(size, implicit);
        let mut dense = MpmSimulation::new(&engine, cfg.clone()).unwrap();
        let mut tiled = MpmSimulation::new(&engine, cfg).unwrap();
        tiled.set_active_grid(true).unwrap();
        let points = particles(129, [0.24, 0.1, -0.2]);
        let velocities = vec![[0.3, -0.02, 0.1]; points.len()];
        for sim in [&mut dense, &mut tiled] {
            sim.set_p2g_mode(two_pass);
            sim.set_particles_with_velocity(queue, &points, &velocities)
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
                (0..points.len()).map(|i| (i % 2) as u32).collect(),
            )
            .unwrap();
        }
        for step in 0..4 {
            let spheres = if step == 3 {
                vec![]
            } else {
                vec![
                    (
                        [0.30 + 0.01 * step as f32, 0.18, -0.16],
                        0.065,
                        [0.1, -0.01, 0.0],
                    ),
                    ([-0.85, -0.8, -0.85], 0.11, [0.0; 3]),
                ]
            };
            for sim in [&mut dense, &mut tiled] {
                sim.set_rigid_spheres(queue, &spheres).unwrap();
                sim.step_n_with(device, queue, 0.0001, 2).unwrap();
            }
            compare(&engine, &mut dense, &mut tiled);
            let stats = tiled.read_active_grid_stats_with(device, queue).unwrap();
            assert!(stats.active_tiles > 0 && stats.active_tiles < stats.tile_capacity);
            println!(
                "size={size} implicit={implicit} two_pass={two_pass} step={step} tiles={}/{}",
                stats.active_tiles, stats.tile_capacity
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_active_tiles_retire_old_cells_and_survive_reseed_mode_switch_and_boundaries() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = config(35, true);
    let mut dense = MpmSimulation::new(&engine, cfg.clone()).unwrap();
    let mut tiled = MpmSimulation::new(&engine, cfg).unwrap();
    tiled.set_active_grid(true).unwrap();
    for (step, (count, location)) in [
        (65, [-0.8, -0.8, -0.8]),
        (129, [0.7, 0.6, 0.5]),
        (1, [1.17, 1.17, 1.17]),
        (97, [-1.01, -0.95, -0.97]),
        (33, [0.1, 0.1, 0.1]),
    ]
    .into_iter()
    .enumerate()
    {
        let points = particles(count, location);
        if step == 2 {
            tiled.set_active_grid(false).unwrap();
        }
        if step == 3 {
            tiled.set_active_grid(true).unwrap();
        }
        for sim in [&mut dense, &mut tiled] {
            sim.set_particles(queue, &[]).unwrap();
            sim.set_particles(queue, &points).unwrap();
            sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        }
        compare(&engine, &mut dense, &mut tiled);
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_active_tiles_bound_active_work_and_reuse_resources() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let points = particles(129, [0.1, 0.1, 0.1]);
    let mut counts = Vec::new();
    for size in [32, 64, 96] {
        let mut sim = MpmSimulation::new(&engine, config(size, true)).unwrap();
        sim.set_particles(queue, &points).unwrap();
        sim.set_active_grid(true).unwrap();
        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.step_n_with(device, queue, 0.0001, 4).unwrap();
        let resources = resources.finish();
        let transfers = transfers.finish();
        assert_eq!(resources.observations, 0);
        assert!(!resources.invalid && !transfers.overflowed);
        assert_eq!(transfers.readback_bytes, 0);
        let stats = sim.read_active_grid_stats_with(device, queue).unwrap();
        counts.push(stats.active_tiles);
        println!(
            "grid={size}^3 active_tiles={} scheduled_cell_slots={} tile_capacity={}",
            stats.active_tiles,
            stats.active_tiles * 64,
            stats.tile_capacity
        );
    }
    assert!(counts.iter().all(|&count| count == counts[0]));
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_active_tiles_transport_mass_and_momentum_across_tiles() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for implicit in [false, true] {
        let mut cfg = config(64, implicit);
        cfg.youngs = 0.0;
        cfg.gravity_y = -0.5;
        cfg.particle_volume = 0.021_f32.powi(3);
        cfg.particle_mass = 1600.0 * cfg.particle_volume;
        let points = particles(129, [-0.6, 0.1, 0.1]);
        let velocity = [4.0, 0.0, 0.0];
        let velocities = vec![velocity; points.len()];
        let mass = points.len() as f64 * f64::from(cfg.particle_mass);
        let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
        let mut tiled = MpmSimulation::new(&engine, cfg).unwrap();
        tiled.set_active_grid(true).unwrap();
        for sim in [&mut dense, &mut tiled] {
            sim.set_particles_with_velocity(queue, &points, &velocities)
                .unwrap();
        }
        for window in 1..=8 {
            for sim in [&mut dense, &mut tiled] {
                sim.step_n_with(device, queue, 1.0 / 512.0, 16).unwrap();
            }
            compare(&engine, &mut dense, &mut tiled);
            let (grid_mass, _) = tiled.read_grid_mass_momentum(&engine).unwrap();
            let grid_mass: f64 = grid_mass.into_iter().map(f64::from).sum();
            assert!((grid_mass - mass).abs() <= 2e-5 * mass);
            let velocities = tiled.read_velocities_with(device, queue).unwrap();
            let mut momentum = [0.0_f64; 3];
            for v in velocities {
                for axis in 0..3 {
                    momentum[axis] += f64::from(v[axis]) * f64::from(cfg.particle_mass);
                }
            }
            let t = f64::from(window * 16) / 512.0;
            let expected = [mass * 4.0, mass * f64::from(cfg.gravity_y) * t, 0.0];
            for axis in 0..3 {
                assert!(
                    (momentum[axis] - expected[axis]).abs() <= 2e-5 * expected[axis].abs().max(1.0),
                    "implicit={implicit} window={window} axis={axis}: {} != {}",
                    momentum[axis],
                    expected[axis]
                );
            }
        }
        let final_positions = tiled.read_positions_with(device, queue).unwrap();
        assert!(
            final_positions
                .iter()
                .zip(&points)
                .all(|(end, start)| end[0] - start[0] > 0.99)
        );
        println!(
            "implicit={implicit} transported_particles={} travel_m=1.0 windows=8 mass_kg={mass}",
            points.len()
        );
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter; run with make bench-soil CASE=active-domain MODE=compare"]
fn mpm_active_domain_benchmark() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    use std::time::Instant;

    let engine = Engine::new_gpu().expect("GPU required for active-domain benchmark");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let dt = 0.0001_f32;
    let steps = 4usize;
    let simulated_seconds = f64::from(dt) * steps as f64;
    let warmup = 2;
    let samples = 12;
    let mut failed_cases = Vec::new();
    for (size, filled, implicit) in [
        (32u32, false, false),
        (64, false, false),
        (96, false, false),
        (32, false, true),
        (64, false, true),
        (96, false, true),
        (32, true, false),
        (32, true, true),
    ] {
        let n = if filled { size.pow(3) as usize } else { 50_000 };
        let mut cfg = config(size, implicit);
        cfg.max_particles = n as u32;
        let spacing = if filled { cfg.cell_size } else { 0.025 };
        cfg.particle_volume = spacing.powi(3);
        cfg.particle_mass = 1600.0 * cfg.particle_volume;
        let points: Vec<_> = (0..n)
            .map(|i| {
                let width = if filled { size as usize } else { 40 };
                let origin = if filled { -1.0 + spacing * 0.5 } else { -0.65 };
                [
                    origin + (i % width) as f32 * spacing,
                    origin + (i / width % width) as f32 * spacing,
                    origin + (i / (width * width)) as f32 * spacing,
                ]
            })
            .collect();
        let velocities = vec![[0.03, -0.02, 0.01]; n];
        let spheres = if filled {
            vec![]
        } else {
            vec![
                ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.0]),
                ([-0.87, 0.8, -0.87], 0.08, [0.0; 3]),
            ]
        };
        let layout = if filled { "filled" } else { "patch" };
        let iterations = if implicit { 3 } else { 0 };
        let mut dense = MpmSimulation::new(&engine, cfg).unwrap();
        let mut tiled = MpmSimulation::new(&engine, cfg).unwrap();
        tiled.set_active_grid(true).unwrap();
        let compact_capacity = if filled { size.div_ceil(4).pow(3) } else { 256 };
        let mut compact = MpmSimulation::new_sparse(&engine, cfg, compact_capacity).unwrap();
        let grid_bytes = [
            dense.grid_storage_bytes(),
            tiled.grid_storage_bytes(),
            compact.grid_storage_bytes(),
        ];
        for sim in [&mut dense, &mut tiled, &mut compact] {
            sim.set_rigid_spheres(queue, &spheres).unwrap();
        }
        println!(
            "{{\"schema\":1,\"record\":\"manifest\",\"case\":\"active-domain\",\"scope\":\"solver-only diagnostic; reset, readback, raster, renderer and machine excluded\",\"grid\":{size},\"layout\":\"{layout}\",\"particles\":{n},\"jacobi_iters\":{iterations},\"dt\":{dt:?},\"substeps\":{steps},\"simulated_seconds\":{simulated_seconds},\"cell_size\":{},\"origin\":[-1,-1,-1],\"particle_spacing\":{spacing},\"particle_mass\":{},\"particle_volume\":{},\"youngs\":{},\"poisson\":{},\"gravity_y\":{},\"friction_angle\":{},\"cohesion\":{},\"boundary_friction\":{},\"initial_velocity\":[0.03,-0.02,0.01],\"rigids\":{},\"two_pass_p2g\":true,\"warmup_rounds\":{warmup},\"sample_rounds\":{samples},\"reset_each_sample\":true,\"mode_count\":3,\"order\":\"cyclic-rotation\",\"compact_tile_capacity\":{compact_capacity},\"absolute_particle_tolerance\":0.00002,\"instrumented\":true}}",
            cfg.cell_size,
            cfg.particle_mass,
            cfg.particle_volume,
            cfg.youngs,
            cfg.poisson,
            cfg.gravity_y,
            cfg.friction_angle,
            cfg.cohesion,
            cfg.boundary_friction,
            spheres.len()
        );
        println!("config={cfg:?} spheres={spheres:?}");
        let mut times: [Vec<f64>; 3] = Default::default();
        let mut passed = [true; 3];
        for round in 0..warmup + samples {
            for mode in [round % 3, (round + 1) % 3, (round + 2) % 3] {
                let sim = match mode {
                    0 => &mut dense,
                    1 => &mut tiled,
                    _ => &mut compact,
                };
                sim.set_particles_with_velocity(queue, &points, &velocities)
                    .unwrap();
                queue.submit([]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let start = Instant::now();
                sim.step_n_with(device, queue, dt, steps).unwrap();
                let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resources = resources.finish();
                let transfers = transfers.finish();
                assert!(!resources.invalid && !transfers.overflowed);
                assert_eq!(resources.observations, 0);
                assert_eq!(transfers.readback_bytes, 0);
                assert!(wall_ms.is_finite() && wall_ms > 0.0);
                if round >= warmup {
                    times[mode].push(wall_ms);
                    println!(
                        "{{\"record\":\"sample\",\"case\":\"active-domain\",\"grid\":{size},\"layout\":\"{layout}\",\"jacobi_iters\":{iterations},\"mode\":{mode},\"sample\":{},\"wall_ms\":{wall_ms},\"encode_submit_ms\":{encode_ms},\"simulated_seconds\":{simulated_seconds},\"allocation_observations\":{},\"readback_bytes\":{},\"upload_bytes\":{}}}",
                        round - warmup,
                        resources.observations,
                        transfers.readback_bytes,
                        transfers.upload_bytes
                    );
                }
            }
            if round == warmup - 1 || round == warmup + samples - 1 {
                for (mode, sim) in [(1, &mut tiled), (2, &mut compact)] {
                    if let Err(error) = compare_result_mode(&engine, &mut dense, sim, mode == 2) {
                        passed[0] = false;
                        passed[mode] = false;
                        println!(
                            "parity_failure: grid={size} layout={layout} jacobi_iters={iterations} mode={mode} round={round}: {error}"
                        );
                    }
                }
            }
        }
        let tile_stats = [
            tiled.read_active_grid_stats_with(device, queue).unwrap(),
            compact.read_active_grid_stats_with(device, queue).unwrap(),
        ];
        for stats in tile_stats {
            assert!(!stats.overflow);
            if filled {
                assert_eq!(stats.active_tiles, stats.tile_capacity);
            }
        }
        if !passed[0] {
            failed_cases.push((size, layout, iterations));
        }
        for (mode, mut samples) in times.into_iter().enumerate() {
            samples.sort_by(f64::total_cmp);
            let quantile = |q: f64| samples[(q * samples.len() as f64).ceil() as usize - 1];
            let name = ["dense", "active-tiles", "compact-tiles"][mode];
            let stats = tile_stats[usize::from(mode == 2)];
            let parity_passed = passed[mode];
            let dense_backing = mode != 2;
            let grid_storage_bytes = grid_bytes[mode];
            let p50 = quantile(0.5);
            let p95 = quantile(0.95);
            let p99 = quantile(0.99);
            let rtf = p50 / 1000.0 / simulated_seconds;
            println!(
                "{{\"schema\":1,\"record\":\"summary\",\"case\":\"active-domain\",\"grid\":{size},\"layout\":\"{layout}\",\"particles\":{n},\"jacobi_iters\":{iterations},\"mode\":\"{name}\",\"samples\":{},\"p50_ms\":{p50},\"p95_ms\":{p95},\"p99_ms\":{p99},\"max_ms\":{p99},\"simulated_seconds\":{simulated_seconds},\"p50_wall_over_simulated\":{rtf},\"active_tiles\":{},\"tile_capacity\":{},\"scheduled_cell_slots\":{},\"parity_passed\":{parity_passed},\"dense_backing_storage\":{dense_backing},\"grid_storage_bytes\":{grid_storage_bytes}}}",
                samples.len(),
                stats.active_tiles,
                stats.tile_capacity,
                if mode == 0 {
                    size.pow(3)
                } else {
                    stats.active_tiles * 64
                }
            );
        }
    }
    assert!(
        failed_cases.is_empty(),
        "active-domain parity failed: {failed_cases:?}"
    );
}

#[test]
#[ignore = "needs a real GPU adapter; diagnostic for the 50k implicit rigid-wrench gate"]
fn mpm_active_wrench_repeatability_benchmark() {
    let engine = Engine::new_gpu().expect("GPU required for wrench diagnostic");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = config(32, true);
    cfg.max_particles = 50_000;
    cfg.particle_volume = 0.025_f32.powi(3);
    cfg.particle_mass = 1600.0 * cfg.particle_volume;
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.65 + (i % 40) as f32 * 0.025,
                -0.65 + (i / 40 % 40) as f32 * 0.025,
                -0.65 + (i / 1600) as f32 * 0.025,
            ]
        })
        .collect();
    let velocity = vec![[0.03, -0.02, 0.01]; points.len()];
    let spheres = [
        ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.0]),
        ([-0.87, 0.8, -0.87], 0.08, [0.0; 3]),
    ];
    let mut simulations: Vec<_> = (0..3)
        .map(|_| MpmSimulation::new(&engine, cfg).unwrap())
        .collect();
    simulations[2].set_active_grid(true).unwrap();
    let mut failures = [0usize; 2];
    for steps in [1, 4] {
        for repeat in 0..4 {
            let mut outputs = Vec::new();
            for sim in &mut simulations {
                sim.set_particles_with_velocity(queue, &points, &velocity)
                    .unwrap();
                sim.set_rigid_spheres(queue, &spheres).unwrap();
                sim.step_n_with(device, queue, 0.0001, steps).unwrap();
                outputs.push((
                    sim.read_positions_with(device, queue).unwrap(),
                    sim.read_velocities_with(device, queue).unwrap(),
                    sim.read_grid_mass_momentum(&engine).unwrap(),
                    sim.read_rigid_wrenches(&engine).unwrap(),
                ));
            }
            for candidate in [1, 2] {
                let a = &outputs[0];
                let b = &outputs[candidate];
                let error = |a: &[f32], b: &[f32], relative: bool| {
                    assert_eq!(a.len(), b.len());
                    a.iter()
                        .zip(b)
                        .map(|(&a, &b)| {
                            assert!(a.is_finite() && b.is_finite());
                            f64::from(
                                (a - b).abs()
                                    / (2e-5 * if relative { b.abs().max(1.0) } else { 1.0 }),
                            )
                        })
                        .fold(0.0_f64, f64::max)
                };
                let position = error(a.0.as_flattened(), b.0.as_flattened(), false);
                let velocity = error(a.1.as_flattened(), b.1.as_flattened(), false);
                let mass = error(&a.2.0, &b.2.0, true);
                let momentum = error(&a.2.1, &b.2.1, true);
                let mut force = 0.0_f64;
                let mut torque = 0.0_f64;
                for (a, b) in a.3.iter().zip(&b.3) {
                    force = force.max(error(&a.0, &b.0, true));
                    torque = torque.max(error(&a.1, &b.1, true));
                }
                let passed = [position, velocity, mass, momentum, force, torque]
                    .into_iter()
                    .all(|e| e <= 1.0);
                failures[candidate - 1] += usize::from(!passed);
                let mode = ["dense-repeat", "active-tiles"][candidate - 1];
                println!(
                    "{{\"case\":\"wrench-repeatability\",\"steps\":{steps},\"repeat\":{repeat},\"mode\":\"{mode}\",\"position_error_over_tolerance\":{position},\"velocity_error_over_tolerance\":{velocity},\"mass_error_over_tolerance\":{mass},\"momentum_error_over_tolerance\":{momentum},\"force_error_over_tolerance\":{force},\"torque_error_over_tolerance\":{torque},\"parity_passed\":{passed}}}"
                );
            }
        }
    }
    println!(
        "dense_repeat_failures={} active_failures={}",
        failures[0], failures[1]
    );
    assert_eq!(failures, [0, 0], "unchanged 2e-5 gate; diagnostic only");
}
