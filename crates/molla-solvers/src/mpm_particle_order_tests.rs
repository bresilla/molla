use super::super::replay_tests::read_words;
use super::*;

#[path = "mpm_particle_order_repeat_tests.rs"]
mod repeatability;

fn config(size: u32) -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = size;
    cfg.grid_origin = [-1.; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 513;
    cfg.max_rigid = 2;
    cfg.youngs = 3000.;
    cfg.particle_volume = 0.021_f32.powi(3);
    cfg.particle_mass = cfg.particle_volume * 1600.;
    cfg
}

fn points(count: usize) -> Vec<[f32; 3]> {
    (0..count)
        .map(|id| {
            let i = (id * 131) % count;
            [
                -0.4 + (i % 7) as f32 * 0.021 + (i % 2) as f32 * 0.55,
                -0.2 + (i / 7 % 7) as f32 * 0.021,
                -0.3 + (i / 49) as f32 * 0.021,
            ]
        })
        .collect()
}

fn key(cfg: &MpmConfig, point: [f32; 3]) -> u32 {
    let tile = std::array::from_fn::<_, 3, _>(|i| {
        (((point[i] - cfg.grid_origin[i]) / cfg.cell_size)
            .floor()
            .clamp(0., (cfg.grid_size - 1) as f32) as u32)
            / 4
    });
    let width = cfg.grid_size.div_ceil(4);
    tile[0] + width * (tile[1] + width * tile[2])
}

fn close(name: &str, a: &[u32], b: &[u32]) {
    assert_eq!(a.len(), b.len());
    let mut max = 0_f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        let scale = if matches!(
            name,
            "position" | "velocity" | "travel positions" | "travel velocities"
        ) {
            1.
        } else {
            a.abs().max(1.)
        };
        let error = (a - b).abs() / scale;
        max = max.max(error);
        assert!(
            a.is_finite() && b.is_finite() && error <= 2e-5,
            "{name}[{i}]: {a} != {b}"
        );
    }
    println!("{name} max_normalized_error={max}");
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn exact_stable_indices_padding_history_and_capacity() {
    let engine = Engine::new_gpu().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (size, parallel) in [(35, false), (35, true), (1536, true)] {
        let cfg = config(size);
        let mut sim = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
        assert!(!sim.particle_tile_ordering_enabled());
        assert_eq!(sim.particle_tile_ordering_storage_bytes(), 0);
        assert!(sim.set_particle_tile_ordering(true).is_err());
        assert!(sim.prepare_particle_tile_ordering(queue, 0).is_err());
        assert!(sim.prepare_particle_tile_ordering(queue, 514).is_err());
        sim.set_particles(queue, &points(513)).unwrap();
        let ids: Vec<u32> = (0..513).map(|i| i % 3).collect();
        queue.write_buffer(&sim.p_material, 0, bytemuck::cast_slice(&ids));
        let history: Vec<_> = [
            &sim.p_pos,
            &sim.p_vel,
            &sim.p_c,
            &sim.p_f,
            &sim.p_jp,
            &sim.tire_compaction_buf,
        ]
        .map(|b| read_words(device, queue, b))
        .into();
        sim.prepare_particle_tile_ordering(queue, 513).unwrap();
        if parallel {
            sim.prepare_parallel_particle_tile_ordering(queue, 513)
                .unwrap();
        }
        assert_eq!(sim.particle_tile_ordering_uses_parallel_sort(), parallel);
        assert_eq!(
            sim.particle_tile_ordering_storage_bytes(),
            513 * 20 + 3 * 1024 + 96 + u64::from(parallel) * 1024
        );
        assert_eq!(&read_words(device, queue, &sim.p_material)[..513], &ids);
        for (before, buffer) in history.iter().zip([
            &sim.p_pos,
            &sim.p_vel,
            &sim.p_c,
            &sim.p_f,
            &sim.p_jp,
            &sim.tire_compaction_buf,
        ]) {
            assert_eq!(*before, read_words(device, queue, buffer));
        }
        for count in [513, 257, 1, 0, 63, 513] {
            let mut positions = points(count);
            if count > 1 {
                positions[0] = [-1.0625; 3];
                positions[1] = [cfg.grid_size as f32 * cfg.cell_size; 3];
            }
            sim.set_particles(queue, &positions).unwrap();
            let order = sim.particle_order.as_ref().unwrap();
            order.upload(queue, &cfg, count as u32, [0; 3]);
            let mut encoder = device.create_command_encoder(&Default::default());
            order.encode(&mut encoder, &sim.p_material, 513 * 4);
            queue.submit([encoder.finish()]);
            let actual_keys = read_words(device, queue, &order.keys);
            let actual_ids = read_words(device, queue, &order.identities);
            let mut expected: Vec<_> = (0..513_u32)
                .map(|i| {
                    (
                        positions
                            .get(i as usize)
                            .map_or(u32::MAX, |p| key(&cfg, *p)),
                        i,
                    )
                })
                .collect();
            expected.sort_by_key(|entry| entry.0);
            assert_eq!(
                actual_keys,
                expected.iter().map(|entry| entry.0).collect::<Vec<_>>()
            );
            assert_eq!(
                actual_ids,
                expected.iter().map(|entry| entry.1).collect::<Vec<_>>()
            );
            let materials = read_words(device, queue, &sim.p_material);
            assert_eq!(&materials[..513], &ids);
            assert_eq!(&materials[513..], &actual_ids);
        }
        sim.set_particles(queue, &points(31)).unwrap();
        sim.prepare_particle_tile_ordering(queue, 31).unwrap();
        sim.set_particle_tile_ordering(true).unwrap();
        sim.set_particles(queue, &points(32)).unwrap();
        let before = read_words(device, queue, &sim.p_pos);
        assert!(sim.step_n_with(device, queue, 0.0001, 1).is_err());
        assert_eq!(before, read_words(device, queue, &sim.p_pos));
        sim.prepare_particle_tile_ordering(queue, 64).unwrap();
        sim.set_particle_tile_ordering(false).unwrap();
        assert!(!sim.particle_tile_ordering_enabled());
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn ordered_transfer_preserves_material_state_and_coupling() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (mode, two_pass, implicit) in [
        (0, true, false),
        (0, false, true),
        (1, true, true),
        (1, false, false),
        (2, true, true),
        (2, false, false),
    ] {
        let mut cfg = config(35);
        cfg.time_step = if implicit {
            MpmTimeStep::Implicit { jacobi_iters: 3 }
        } else {
            MpmTimeStep::Explicit
        };
        let create = || {
            if mode == 2 {
                MpmSimulation::new_sparse(&engine, cfg, 256).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            }
        };
        let mut reference = create();
        let mut candidate = create();
        candidate
            .prepare_parallel_particle_tile_ordering(queue, 513)
            .unwrap();
        candidate.set_particle_tile_ordering(true).unwrap();
        for count in [257, 63] {
            let positions = points(count);
            for sim in [&mut reference, &mut candidate] {
                if mode == 1 {
                    sim.set_active_grid(true).unwrap();
                }
                sim.set_p2g_mode(two_pass);
                if mode == 2 && two_pass {
                    sim.set_deferred_p2g_lookup(true).unwrap();
                }
                sim.set_particles_with_velocity(
                    queue,
                    &positions,
                    &vec![[0.1, -0.02, 0.03]; count],
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
                    (0..count).map(|i| (i % 3) as u32).collect(),
                )
                .unwrap();
                sim.set_rigid_spheres(queue, &[([-0.35, -0.15, -0.27], 0.06, [0.01, 0., 0.])])
                    .unwrap();
            }
            for window in 0..3 {
                reference.step_n_with(device, queue, 0.0001, 2).unwrap();
                candidate.step_n_with(device, queue, 0.0001, 2).unwrap();
                for (name, a, b) in [
                    ("position", &reference.p_pos, &candidate.p_pos),
                    ("velocity", &reference.p_vel, &candidate.p_vel),
                    ("affine", &reference.p_c, &candidate.p_c),
                    ("deformation", &reference.p_f, &candidate.p_f),
                    ("plastic", &reference.p_jp, &candidate.p_jp),
                    (
                        "rigid wrench",
                        &reference.rigid_wrench,
                        &candidate.rigid_wrench,
                    ),
                ] {
                    close(
                        name,
                        &read_words(device, queue, a),
                        &read_words(device, queue, b),
                    );
                }
                assert_eq!(
                    read_words(device, queue, &reference.p_material),
                    read_words(device, queue, &candidate.p_material)[..513]
                );
                assert_eq!(
                    read_words(device, queue, &reference.p_mass),
                    read_words(device, queue, &candidate.p_mass)
                );
                if mode > 0 {
                    assert_eq!(
                        reference
                            .read_active_grid_stats_with(device, queue)
                            .unwrap()
                            .completed_substeps,
                        candidate
                            .read_active_grid_stats_with(device, queue)
                            .unwrap()
                            .completed_substeps
                    );
                }
                println!(
                    "mode={mode} two_pass={two_pass} implicit={implicit} count={count} window={window}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn ordering_hot_resources_transport_growth_and_overflow() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = config(64);
    cfg.youngs = 0.;
    cfg.gravity_y = -0.5;
    let mut reference = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, cfg, 1).unwrap();
    let positions = points(129);
    for s in [&mut reference, &mut sim] {
        s.set_particles_with_velocity(queue, &positions, &vec![[4., 0., 0.]; 129])
            .unwrap();
    }
    sim.prepare_parallel_particle_tile_ordering(queue, 129)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.step_n_with(device, queue, 0.0001, 2).unwrap();
    let status = sim.read_active_grid_status_with(device, queue).unwrap();
    assert!(status.overflow);
    assert_eq!(status.completed_substeps, 0);
    assert_eq!(sim.read_positions_with(device, queue).unwrap(), positions);
    sim.reserve_sparse_tiles(128).unwrap();
    let ids = sim.particle_order.as_ref().unwrap().identities.clone();
    let material = sim.p_material.clone();
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    sim.step_n_with(device, queue, 0.001953125, 128).unwrap();
    let transfers = transfers.finish();
    let resources = resources.finish();
    assert_eq!(resources.observations, 0);
    assert!(!resources.invalid && !transfers.overflowed);
    assert_eq!(transfers.readback_bytes, 0);
    assert_eq!(
        transfers.upload_bytes,
        std::mem::size_of::<Uniforms>() as u64 + 256 + 64 + 48
    );
    assert_eq!(sim.particle_order.as_ref().unwrap().identities, ids);
    assert_eq!(sim.p_material, material);
    reference
        .step_n_with(device, queue, 0.001953125, 128)
        .unwrap();
    for (label, s) in [("reference", &mut reference), ("ordered", &mut sim)] {
        let travelled = s.read_positions_with(device, queue).unwrap();
        let velocities = s.read_velocities_with(device, queue).unwrap();
        let displacement_error = positions
            .iter()
            .zip(&travelled)
            .map(|(before, after)| (after[0] - before[0] - 1.).abs())
            .fold(0_f32, f32::max);
        let vertical_error = velocities
            .iter()
            .map(|v| (v[1] + 0.125).abs())
            .fold(0_f32, f32::max);
        let momentum: [f64; 3] = std::array::from_fn(|axis| {
            velocities
                .iter()
                .map(|v| f64::from(v[axis]) * f64::from(cfg.particle_mass))
                .sum()
        });
        let expected_mass = 129. * f64::from(cfg.particle_mass);
        println!(
            "{label} analytic_particle_displacement_error={displacement_error} analytic_particle_vertical_error={vertical_error} momentum={momentum:?}"
        );
        for (actual, expected) in
            momentum
                .into_iter()
                .zip([4. * expected_mass, -0.125 * expected_mass, 0.])
        {
            assert!((actual - expected).abs() <= 2e-5 * expected.abs().max(1.));
        }
        let (mass, _) = s.read_grid_storage_mass_momentum(&engine).unwrap();
        assert!(
            (mass.iter().map(|&m| f64::from(m)).sum::<f64>() - expected_mass).abs()
                <= 2e-5 * expected_mass
        );
    }
    close(
        "travel positions",
        &read_words(device, queue, &reference.p_pos),
        &read_words(device, queue, &sim.p_pos),
    );
    close(
        "travel velocities",
        &read_words(device, queue, &reference.p_vel),
        &read_words(device, queue, &sim.p_vel),
    );
    assert_eq!(
        sim.read_active_grid_stats_with(device, queue)
            .unwrap()
            .completed_substeps,
        128
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn ordering_profile_records_sort_cost_and_rejects_query_overflow() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, config(32), 128).unwrap();
    sim.set_particles(queue, &points(129)).unwrap();
    sim.prepare_parallel_particle_tile_ordering(queue, 129)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_step_profiling(204).unwrap();
    assert!(
        sim.step_n_profiled_with(device, queue, 0.0001, 187)
            .is_err()
    );
    sim.step_n_profiled_with(device, queue, 0.0001, 2).unwrap();
    let profile = sim.read_step_profile().unwrap().unwrap();
    assert!(profile.particle_tile_ordering);
    assert!(profile.parallel_particle_sort);
    let stages: Vec<_> = profile
        .stages
        .iter()
        .filter(|s| s.stage == MpmProfileStage::ParticleOrdering)
        .collect();
    assert_eq!(stages.len(), 2);
    for (i, stage) in stages.iter().enumerate() {
        assert_eq!(stage.substep, Some(i));
        assert!(stage.elapsed_ms.is_finite() && stage.elapsed_ms > 0.);
    }
    assert_eq!(
        sim.read_active_grid_stats_with(device, queue)
            .unwrap()
            .completed_substeps,
        2
    );
}
