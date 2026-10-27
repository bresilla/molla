use super::super::replay_tests::read_words;
use super::*;

#[path = "mpm_native_force_replay_tests.rs"]
mod force_replay;

fn engine() -> Engine {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    engine
}

fn config(size: u32) -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = size;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 129;
    cfg.max_rigid = 2;
    cfg.youngs = 1000.0;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    cfg
}

fn points(offset: f32) -> Vec<[f32; 3]> {
    (0..129)
        .map(|i| {
            [
                offset + (i % 7) as f32 * 0.021,
                offset + (i / 7 % 7) as f32 * 0.021,
                offset + (i / 49) as f32 * 0.021,
            ]
        })
        .collect()
}

fn near(a: &[u32], b: &[u32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5,
            "{label}[{i}]: {a} != {b}"
        );
    }
}

fn compare(engine: &Engine, a: &MpmSimulation, b: &MpmSimulation) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    if let Some(path) = std::env::var_os("MOLLA_NATIVE_FORCE_CAPTURE") {
        let aa = a.checkpoint_buffers().into_iter().chain([&a.rigid_wrench]);
        let bb = b.checkpoint_buffers().into_iter().chain([&b.rigid_wrench]);
        for (field, (a_buffer, b_buffer)) in aa.zip(bb).enumerate() {
            let aa = read_words(device, queue, a_buffer);
            let bb = read_words(device, queue, b_buffer);
            let mismatch = aa.iter().zip(&bb).enumerate().find(|(_, (a, b))| {
                if field == 7 {
                    a != b
                } else {
                    let (a, b) = (f32::from_bits(**a), f32::from_bits(**b));
                    !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5
                }
            });
            if let Some((word, (&av, &bv))) = mismatch {
                let path = std::path::PathBuf::from(path);
                std::fs::create_dir(&path).unwrap();
                force_replay::capture_inputs(engine, a, &path.join("control"));
                force_replay::capture_inputs(engine, b, &path.join("candidate"));
                let metadata = serde_json::json!({"field":field,"word":word,"control_bits":av,"candidate_bits":bv,"control_value":f32::from_bits(av),"candidate_value":f32::from_bits(bv),"bound":2e-5,"scope":"first observed post-step mismatch; failed gate preserved"});
                std::fs::write(
                    path.join("failure.json"),
                    serde_json::to_vec_pretty(&metadata).unwrap(),
                )
                .unwrap();
                panic!(
                    "native parity capture field={field} word={word} control={} candidate={}; saved {}",
                    f32::from_bits(av),
                    f32::from_bits(bv),
                    path.display()
                );
            }
        }
    }
    for (i, (a, b)) in a
        .checkpoint_buffers()
        .into_iter()
        .zip(b.checkpoint_buffers())
        .enumerate()
    {
        let a = read_words(device, queue, a);
        let b = read_words(device, queue, b);
        if i == 7 {
            assert_eq!(a, b);
        } else {
            near(&a, &b, &format!("history {i}"));
        }
    }
    near(
        &read_words(device, queue, &a.rigid_wrench),
        &read_words(device, queue, &b.rigid_wrench),
        "wrench",
    );
}

#[test]
#[ignore = "requires native float-atomic GPU; dense/active/compact physical parity"]
fn native_active_materials_rigid_and_mode_switch() {
    materials_rigid_matrix(false, false);
}

#[test]
#[ignore = "requires native float-atomic GPU; ordered shell force parity"]
fn native_active_ordered_shell_parity() {
    materials_rigid_matrix(true, false);
}

#[test]
#[ignore = "requires native float-atomic/f64 GPU; pre-contact compensated single-pass parity"]
fn native_precontact_compensated_active_parity() {
    materials_rigid_matrix(false, true);
}

fn materials_rigid_matrix(ordered: bool, compensated: bool) {
    let engine = if compensated {
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::SHADER_F64,
        )
        .unwrap()
    } else {
        engine()
    };
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for combined in [false, true] {
        if compensated && combined {
            continue;
        }
        let cfg = config(35);
        let mut dense = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
        let mut active = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
        let mut compact = MpmSimulation::new_native_sparse(&engine, cfg, 128).unwrap();
        for sim in [&mut dense, &mut active, &mut compact] {
            sim.set_ordered_native_shell_forces(ordered).unwrap();
        }
        active.set_active_grid(true).unwrap();
        assert!(compact.set_active_grid(false).is_err());
        for offset in [-0.2, 0.75] {
            let points = points(offset);
            for sim in [&mut dense, &mut active, &mut compact] {
                sim.set_p2g_mode(combined);
                sim.set_particles_with_velocity(
                    queue,
                    &points,
                    &vec![[0.02, -0.01, 0.005]; points.len()],
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
                            hardening: 10.0,
                        },
                        MpmRheology::Viscous {
                            velocity_damping: 0.1,
                            affine_damping: 0.2,
                        },
                    ],
                    (0..129).map(|i| i % 3).collect(),
                )
                .unwrap();
                sim.set_rigid_spheres(queue, &[([offset + 0.05; 3], 0.035, [0.01, 0.0, 0.0])])
                    .unwrap();
                if compensated {
                    sim.install_compensated_active_transfer();
                }
            }
            let before = [&active, &compact].map(|sim| {
                if sim.active_grid_needs_reset {
                    0
                } else {
                    sim.read_active_grid_status_with(device, queue)
                        .unwrap()
                        .completed_substeps
                }
            });
            for step in 0..4 {
                for sim in [&mut dense, &mut active, &mut compact] {
                    sim.step(&engine, 0.0001).unwrap();
                }
                compare(&engine, &dense, &active);
                compare(&engine, &dense, &compact);
                for (sim, before) in [&active, &compact].into_iter().zip(before) {
                    let stats = sim.read_active_grid_status_with(device, queue).unwrap();
                    assert!(!stats.faulted());
                    assert_eq!(stats.completed_substeps, before.wrapping_add(step + 1));
                    assert!(
                        read_words(device, queue, &sim.g_kdiag)
                            [..(sim.active_tiles.metadata_offset / 4) as usize]
                            .iter()
                            .all(|&x| x == 0)
                    );
                }
            }
        }
        active.set_active_grid(false).unwrap();
        if compensated {
            active.install_compensated_active_transfer();
        }
        dense.step(&engine, 0.0001).unwrap();
        active.step(&engine, 0.0001).unwrap();
        compare(&engine, &dense, &active);
        active.set_active_grid(true).unwrap();
        if compensated {
            active.install_compensated_active_transfer();
        }
        dense.step(&engine, 0.0001).unwrap();
        active.step(&engine, 0.0001).unwrap();
        compare(&engine, &dense, &active);
    }
}

#[test]
#[ignore = "requires native float-atomic GPU; compact overflow/recovery and remote storage"]
fn native_active_overflow_recovery_and_domain_scaling() {
    let engine = engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut bytes = None;
    for size in [32, 96, 512, 1536] {
        let cfg = config(size);
        let offset = if size == 1536 { 93.0 } else { -0.2 };
        let points = points(offset);
        let mut sim = MpmSimulation::new_native_sparse(&engine, cfg, 1).unwrap();
        sim.set_particles(queue, &points).unwrap();
        let before = sim
            .checkpoint_buffers()
            .map(|b| read_words(device, queue, b));
        sim.step_n_with(device, queue, 0.0001, 3).unwrap();
        let stats = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(stats.overflow);
        assert_eq!(stats.completed_substeps, 0);
        for (a, b) in before.iter().zip(sim.checkpoint_buffers()) {
            assert_eq!(a, &read_words(device, queue, b));
        }
        sim.reserve_sparse_tiles(64).unwrap();
        let storage = sim.grid_storage_bytes();
        if let Some(bytes) = bytes {
            assert_eq!(storage, bytes);
        } else {
            bytes = Some(storage);
        }
        let mut reference = MpmSimulation::new_sparse(&engine, cfg, 64).unwrap();
        reference.set_p2g_mode(false);
        reference.set_particles(queue, &points).unwrap();
        sim.step(&engine, 0.0001).unwrap();
        reference.step(&engine, 0.0001).unwrap();
        compare(&engine, &reference, &sim);
        let stats = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(!stats.faulted());
        assert_eq!(stats.completed_substeps, 1);
        let (mass, _) = sim.read_grid_storage_mass_momentum(&engine).unwrap();
        let total = mass.iter().map(|&v| v as f64).sum::<f64>();
        assert!((total - 129.0 * cfg.particle_mass as f64).abs() < 2e-5);
        sim.set_particles(queue, &[]).unwrap();
        sim.step(&engine, 0.0001).unwrap();
        sim.set_particles(queue, &points).unwrap();
        sim.step(&engine, 0.0001).unwrap();
        assert!(
            !sim.read_active_grid_status_with(device, queue)
                .unwrap()
                .faulted()
        );
        sim.set_time_step(MpmTimeStep::Implicit { jacobi_iters: 1 });
        assert!(sim.step(&engine, 0.0001).is_err());
        assert!(sim.prepare_step(0.0001, 1).is_err());
        eprintln!(
            "native sparse grid={size} bytes={storage} overflow freeze, growth, mass and reseed pass"
        );
    }
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires native float-atomic GPU; continuous traversal and prepared-step lifecycle"]
fn native_active_traversal_prepared_steps_and_retirement() {
    let engine = engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = config(35);
    cfg.gravity_y = 0.0;
    cfg.youngs = 1.0;
    for compact in [false, true] {
        let mut sim = if compact {
            MpmSimulation::new_native_sparse(&engine, cfg, 64).unwrap()
        } else {
            MpmSimulation::new_native_explicit(&engine, cfg).unwrap()
        };
        sim.set_active_grid(true).unwrap();
        sim.set_particles_with_velocity(queue, &[[-0.5, 0.0, 0.0]], &[[1.0, 0.0, 0.0]])
            .unwrap();
        let record = sim.prepare_step(1.0 / 256.0, 16).unwrap();
        let initial = read_words(device, queue, &sim.p_pos);
        {
            let mut canceled =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            canceled.encode_step(&record).unwrap();
        }
        assert!(sim.active_grid_needs_reset);
        assert_eq!(initial, read_words(device, queue, &sim.p_pos));
        let mut first_tiles = Vec::new();
        for batch_index in 0..8 {
            let encoder = device.create_command_encoder(&Default::default());
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let mut batch = sim.begin_command_batch(encoder);
            batch.encode_step(&record).unwrap();
            batch.submit(queue);
            assert_eq!(allocations.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!(transfers.readback_bytes, 0);
            let stats = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(!stats.faulted());
            assert_eq!(stats.completed_substeps, (batch_index + 1) * 16);
            assert!(stats.active_tiles <= 8);
            let ids = sim.read_active_tile_ids_with(device, queue).unwrap();
            if batch_index == 0 {
                first_tiles = ids;
            } else if batch_index == 7 {
                assert!(first_tiles.iter().all(|id| !ids.contains(id)));
            }
        }
        let position = sim.read_positions_with(device, queue).unwrap()[0];
        let velocity = sim.read_velocities_with(device, queue).unwrap()[0];
        for (a, b) in position.into_iter().zip([0.0, 0.0, 0.0]) {
            assert!((a - b).abs() <= 2e-5, "travel position {a} != {b}");
        }
        for (a, b) in velocity.into_iter().zip([1.0, 0.0, 0.0]) {
            assert!((a - b).abs() <= 2e-5, "travel velocity {a} != {b}");
        }
        if compact {
            sim.reserve_sparse_tiles(128).unwrap();
        } else {
            sim.set_active_grid(false).unwrap();
        }
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&record).is_err());
        drop(batch);
        eprintln!(
            "native active compact={compact}: 128 substeps, 0.5 m travel, retired tiles, zero hot allocations/readbacks and stale-record rejection pass"
        );
    }
}

#[test]
#[ignore = "requires native float-atomic GPU; diagnostic reference repeatability, not parity acceptance"]
fn native_active_dense_reference_repeatability_diagnostic() {
    let engine = engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cases = 0;
    let mut failures = 0;
    let mut largest = 0.0f32;
    for combined in [false, true] {
        let mut a = MpmSimulation::new_native_explicit(&engine, config(35)).unwrap();
        let mut b = MpmSimulation::new_native_explicit(&engine, config(35)).unwrap();
        for offset in [-0.2, 0.75] {
            let points = points(offset);
            for sim in [&mut a, &mut b] {
                sim.set_p2g_mode(combined);
                sim.set_particles_with_velocity(
                    queue,
                    &points,
                    &vec![[0.02, -0.01, 0.005]; points.len()],
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
                            hardening: 10.0,
                        },
                        MpmRheology::Viscous {
                            velocity_damping: 0.1,
                            affine_damping: 0.2,
                        },
                    ],
                    (0..129).map(|i| i % 3).collect(),
                )
                .unwrap();
                sim.set_rigid_spheres(queue, &[([offset + 0.05; 3], 0.035, [0.01, 0.0, 0.0])])
                    .unwrap();
            }
            for step in 0..4 {
                for sim in [&mut a, &mut b] {
                    sim.step(&engine, 0.0001).unwrap();
                }
                let mut maximum = 0.0f32;
                let mut field_index = 0;
                let mut word_index = 0;
                let aa = a.checkpoint_buffers().into_iter().chain([&a.rigid_wrench]);
                let bb = b.checkpoint_buffers().into_iter().chain([&b.rigid_wrench]);
                for (field, (a, b)) in aa.zip(bb).enumerate() {
                    let aa = read_words(device, queue, a);
                    let bb = read_words(device, queue, b);
                    assert_eq!(aa.len(), bb.len());
                    if field == 7 {
                        assert_eq!(aa, bb);
                        continue;
                    }
                    for (word, (&a, &b)) in aa.iter().zip(&bb).enumerate() {
                        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                        assert!(a.is_finite() && b.is_finite());
                        if (a - b).abs() > maximum {
                            maximum = (a - b).abs();
                            field_index = field;
                            word_index = word;
                        }
                    }
                }
                cases += 1;
                if maximum > 2e-5 {
                    failures += 1;
                }
                largest = largest.max(maximum);
                eprintln!(
                    "native-dense-repeat combined={combined} offset={offset} step={step} max_absolute={maximum:e} field={field_index} word={word_index} bound=2e-5 passed={}",
                    maximum <= 2e-5
                );
            }
        }
    }
    assert_eq!(cases, 16);
    eprintln!(
        "native-dense-repeat diagnostic cases={cases} failed_comparisons={failures} max_absolute={largest:e}; test completion is not parity acceptance"
    );
}
