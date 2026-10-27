use super::*;

pub(in crate::mpm) fn fixture(count: u32) -> MpmSoilCheckpoint {
    let mut cfg = MpmConfig::sand();
    cfg.max_particles = 129;
    cfg.max_rigid = 2;
    cfg.youngs = 1000.0;
    let materials = vec![
        MpmRheology::Sand {
            friction_angle: 0.5,
        },
        MpmRheology::Snow {
            theta_c: 0.025,
            theta_s: 0.0075,
            hardening: 10.0,
        },
        MpmRheology::Viscous {
            velocity_damping: 3.0,
            affine_damping: 5.0,
        },
    ];
    let mut words = Vec::new();
    for stream in 0..9 {
        for i in 0..count {
            let f = i as f32;
            let values: Vec<f32> = match stream {
                0 => vec![
                    -0.13 + (i % 7) as f32 * 0.021,
                    0.3 + (i / 7 % 7) as f32 * 0.021,
                    -0.07 + (i / 49) as f32 * 0.021,
                    -0.0,
                ],
                1 => vec![0.013 + f * 0.0001, -0.007, 0.002, 0.0],
                2 => vec![
                    0.01, 0.002, 0.0, 0.0, 0.0, -0.02, 0.003, 0.0, 0.004, 0.0, 0.008, 0.0,
                ],
                3 => vec![
                    0.97, 0.001, 0.0, 0.0, 0.0, 1.01, 0.002, 0.0, 0.001, 0.0, 0.98, 0.0,
                ],
                4 => vec![cfg.particle_volume * (1.0 + f * 0.002)],
                5 => vec![cfg.particle_mass * (1.0 + f * 0.001)],
                6 => vec![-0.1 - f * 0.001],
                7 => {
                    words.push(i % 3);
                    continue;
                }
                8 => vec![0.005 + f * 0.00001, 0.01 + f * 0.001, 0.02, 3.0],
                _ => unreachable!(),
            };
            words.extend(values.into_iter().map(f32::to_bits));
        }
    }
    MpmSoilCheckpoint {
        cfg,
        grid_window_cells: [0; 3],
        materials,
        count,
        words,
    }
}

#[test]
fn format_is_lossless_bounded_and_rejects_corruption() {
    for count in [0, 1, 129] {
        let mut checkpoint = fixture(count);
        for rheology in checkpoint.materials.clone() {
            checkpoint.cfg.rheology = rheology;
            for step in [
                MpmTimeStep::Explicit,
                MpmTimeStep::Implicit { jacobi_iters: 3 },
            ] {
                checkpoint.cfg.time_step = step;
                let bytes = checkpoint.to_bytes();
                let restored = MpmSoilCheckpoint::from_bytes(&bytes, count).unwrap();
                assert_eq!(restored.to_bytes(), bytes);
                assert_eq!(restored.config(), checkpoint.cfg);
                assert_eq!(restored.materials(), checkpoint.materials);
                assert_eq!(restored.particle_bytes(), count as usize * 160);
                assert_eq!(restored.words, checkpoint.words);
                if count > 0 {
                    assert!(MpmSoilCheckpoint::from_bytes(&bytes, count - 1).is_err());
                }
            }
        }
    }
    let checkpoint = fixture(1);
    let bytes = checkpoint.to_bytes();
    for len in 0..bytes.len() {
        assert!(MpmSoilCheckpoint::from_bytes(&bytes[..len], 129).is_err());
    }
    for byte in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[byte] ^= 1;
        assert!(MpmSoilCheckpoint::from_bytes(&bad, 129).is_err());
    }
    let mut trailing = bytes;
    trailing.extend_from_slice(&[0; 4]);
    let digest = blake3::hash(&trailing[HEADER_BYTES..]);
    trailing[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
    assert!(MpmSoilCheckpoint::from_bytes(&trailing, 129).is_err());
}

#[test]
fn valid_digest_does_not_bypass_semantic_validation() {
    let original = fixture(1);
    let mutations: Vec<Box<dyn Fn(&mut MpmSoilCheckpoint)>> = vec![
        Box::new(|c| c.cfg.cell_size = f32::NAN),
        Box::new(|c| c.cfg.max_particles = 0),
        Box::new(|c| c.cfg.grid_size = u32::MAX),
        Box::new(|c| c.cfg.poisson = 0.5),
        Box::new(|c| {
            c.cfg.rheology = MpmRheology::Sand {
                friction_angle: f32::INFINITY,
            }
        }),
        Box::new(|c| {
            c.materials[1] = MpmRheology::Snow {
                theta_c: 1.0,
                theta_s: 0.0,
                hardening: 1.0,
            }
        }),
        Box::new(|c| c.words[0] = f32::NAN.to_bits()),
        Box::new(|c| c.words[4] = f32::INFINITY.to_bits()),
        Box::new(|c| c.words[8] = f32::NAN.to_bits()),
        Box::new(|c| c.words[20] = f32::INFINITY.to_bits()),
        Box::new(|c| c.words[32] = 0),
        Box::new(|c| c.words[33] = (-1.0_f32).to_bits()),
        Box::new(|c| c.words[34] = f32::NAN.to_bits()),
        Box::new(|c| c.words[35] = 3),
        Box::new(|c| c.words[36] = f32::NAN.to_bits()),
        Box::new(|c| c.count = u32::MAX),
        Box::new(|c| {
            c.words.pop();
        }),
        Box::new(|c| c.materials = vec![c.cfg.rheology; MAX_MPM_MATERIALS + 1]),
    ];
    for (index, change) in mutations.iter().enumerate() {
        let mut bad = original.clone();
        change(&mut bad);
        assert!(
            MpmSoilCheckpoint::from_bytes(&bad.to_bytes(), 129).is_err(),
            "mutation {index}"
        );
    }
    for index in [15, 19] {
        let mut bytes = original.to_bytes();
        bytes[HEADER_BYTES + index * 4..HEADER_BYTES + (index + 1) * 4]
            .copy_from_slice(&99_u32.to_le_bytes());
        let digest = blake3::hash(&bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        assert!(MpmSoilCheckpoint::from_bytes(&bytes, 129).is_err());
    }
}

pub(super) fn new_sim(engine: &Engine, cfg: MpmConfig, mode: usize) -> MpmSimulation {
    let mut sim = if mode >= 2 {
        MpmSimulation::new_sparse(engine, cfg, 128).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    if mode == 1 {
        sim.set_active_grid(true).unwrap();
    }
    if mode == 3 {
        sim.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), cfg.max_particles)
            .unwrap();
        sim.set_particle_tile_ordering(true).unwrap();
    }
    sim
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn archive_drop_reload_preserves_all_history_and_resource_identity() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    use std::io::Write;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let original = fixture(129);
        let mut source = new_sim(&engine, original.cfg, mode);
        source.restore_soil_with(queue, &original).unwrap();
        source.step_n_with(device, queue, 0.0001, 3).unwrap();
        let evolved = source.checkpoint_soil_with(queue).unwrap();
        assert_ne!(evolved.stream(0), original.stream(0));
        assert_ne!(evolved.stream(3), original.stream(3));
        assert_eq!(evolved.stream(4), original.stream(4));
        assert_eq!(evolved.stream(5), original.stream(5));
        assert_eq!(evolved.stream(7), original.stream(7));
        assert_eq!(evolved.stream(8), original.stream(8));
        let path = std::env::temp_dir().join(format!(
            "molla-soil-checkpoint-{}-{mode}.bin",
            std::process::id()
        ));
        let bytes = evolved.to_bytes();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);
        drop(source);
        let loaded = MpmSoilCheckpoint::from_bytes(&std::fs::read(&path).unwrap(), 129).unwrap();
        std::fs::remove_file(path).unwrap();
        let mut destination = new_sim(&engine, loaded.config(), mode);
        let buffers: Vec<_> = destination
            .checkpoint_buffers()
            .into_iter()
            .cloned()
            .collect();
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        destination.restore_soil_with(queue, &loaded).unwrap();
        let transfer = transfers.finish();
        assert_eq!(resources.finish().observations, 1);
        assert_eq!(transfer.upload_bytes, 129 * 160);
        assert_eq!(transfer.readback_bytes, 0);
        for (saved, live) in buffers.iter().zip(destination.checkpoint_buffers()) {
            assert_eq!(saved, live);
        }
        for _ in 0..5 {
            let saved = destination.checkpoint_soil_with(queue).unwrap();
            assert_eq!(saved.to_bytes(), bytes);
            destination.restore_soil_with(queue, &saved).unwrap();
        }
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let saved = destination.checkpoint_soil_with(queue).unwrap();
        let transfer = transfers.finish();
        assert_eq!(resources.finish().observations, 1);
        assert_eq!(transfer.readback_bytes, 129 * 160 + 16);
        assert_eq!(transfer.upload_bytes, 0);
        assert_eq!(saved.to_bytes(), bytes);
        destination.step_n_with(device, queue, 0.0001, 2).unwrap();
        let continued = destination.checkpoint_soil_with(queue).unwrap();
        assert_ne!(continued.stream(0), saved.stream(0));
        if mode != 0 {
            assert_eq!(
                destination
                    .read_active_grid_stats_with(device, queue)
                    .unwrap()
                    .completed_substeps,
                2
            );
        }
        println!(
            "mode={mode}: 129 mixed particles; save/drop/load and five restores exact, one archive buffer/transfer per operation, stable solver buffers"
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn continued_trajectory_and_pending_material_edits_survive_restore() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for implicit in [false, true] {
        let mut state = fixture(1);
        if implicit {
            state.cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
        }
        for mode in 0..4 {
            let mut reference = new_sim(&engine, state.cfg, mode);
            reference.restore_soil_with(queue, &state).unwrap();
            reference.step_n_with(device, queue, 0.0001, 3).unwrap();
            reference.set_materials(&state.materials, vec![1]).unwrap();
            let saved = reference.checkpoint_soil_with(queue).unwrap();
            assert_eq!(saved.stream(7), &[1]);
            assert_eq!(reference.particle_material_dirty, Some(vec![1]));
            let mut restored = new_sim(&engine, state.cfg, mode);
            restored.restore_soil_with(queue, &saved).unwrap();
            reference.step_n_with(device, queue, 0.0001, 8).unwrap();
            restored.step_n_with(device, queue, 0.0001, 8).unwrap();
            let a = reference.checkpoint_soil_with(queue).unwrap();
            let b = restored.checkpoint_soil_with(queue).unwrap();
            assert_eq!(a.words, b.words, "mode={mode} implicit={implicit}");
            println!(
                "mode={mode} implicit={implicit}: one-particle continued 8-substep trajectory bit-exact including pending material edit"
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn rejected_restore_does_not_mutate_and_empty_restore_invalidates_records() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let state = fixture(1);
    let mut small_cfg = state.cfg;
    small_cfg.max_particles = 1;
    let mut small = new_sim(&engine, small_cfg, 0);
    small.restore_soil_with(queue, &state).unwrap();
    assert!(small.restore_soil_with(queue, &fixture(129)).is_err());
    assert_eq!(
        small.checkpoint_soil_with(queue).unwrap().words,
        state.words
    );
    let mut sim = new_sim(&engine, state.cfg, 0);
    sim.restore_soil_with(queue, &state).unwrap();
    let epoch = sim.step_epoch.clone();
    let mut incompatible = state.clone();
    incompatible.cfg.grid_origin[0] += 0.1;
    let mut corrupt = state.clone();
    corrupt.words[0] = f32::NAN.to_bits();
    let mut oversized = fixture(129);
    oversized.cfg.max_particles = 130;
    oversized.count = 130;
    oversized.words = vec![0; 130 * WORDS_PER_PARTICLE];
    for bad in [&incompatible, &corrupt, &oversized] {
        assert!(sim.restore_soil_with(queue, bad).is_err());
        assert!(Arc::ptr_eq(&epoch, &sim.step_epoch));
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            state.to_bytes()
        );
    }
    sim.set_rigid_spheres(queue, &[([0.0, 0.5, 0.0], 0.1, [0.0; 3])])
        .unwrap();
    assert!(sim.restore_soil_with(queue, &state).is_err());
    sim.set_rigid_spheres(queue, &[]).unwrap();
    sim.prepare_particle_tile_ordering(queue, 1).unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    assert!(sim.restore_soil_with(queue, &fixture(129)).is_err());
    let record = sim.prepare_step(0.0001, 1).unwrap();
    let empty = fixture(0);
    sim.restore_soil_with(queue, &empty).unwrap();
    assert_eq!(sim.particle_count, 0);
    let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    assert!(batch.encode_step(&record).is_err());
    drop(batch);
    assert_eq!(
        sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
        empty.to_bytes()
    );
    sim.restore_soil_with(queue, &state).unwrap();
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().words, state.words);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn overflow_cannot_be_archived_or_hidden_by_reset_request() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = fixture(1).cfg;
    cfg.gravity_y = 0.0;
    let mut sim = MpmSimulation::new_sparse(&engine, cfg, 8).unwrap();
    sim.set_particles(queue, &[[-0.69, 0.11, -0.69]]).unwrap();
    let good = sim.checkpoint_soil_with(queue).unwrap();
    sim.set_particles(queue, &[[-0.69, 0.11, -0.69], [0.0, 0.5, 0.0]])
        .unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    assert!(
        sim.read_active_grid_status_with(device, queue)
            .unwrap()
            .overflow
    );
    assert!(sim.checkpoint_soil_with(queue).is_err());
    sim.reset_active_grid();
    assert!(sim.checkpoint_soil_with(queue).is_err());
    sim.restore_soil_with(queue, &good).unwrap();
    assert_eq!(
        sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
        good.to_bytes()
    );
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    let status = sim.read_active_grid_status_with(device, queue).unwrap();
    assert!(!status.overflow);
    assert_eq!(status.active_tiles, 8);
    sim.checkpoint_soil_with(queue).unwrap();
}
