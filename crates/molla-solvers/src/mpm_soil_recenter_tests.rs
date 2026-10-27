use super::*;

#[test]
fn grid_window_preserves_all_streams_and_rejects_support_loss() {
    let mut before = state(129);
    before.cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
    let next = [1024.5, -2048.0, 4096.0];
    let after = before.with_grid_window(next).unwrap();
    for stream in 0..STRIDES.len() {
        assert_eq!(before.stream(stream), after.stream(stream));
    }
    assert_eq!(after.cfg.grid_origin, next);
    assert_eq!(
        after
            .with_grid_window(before.cfg.grid_origin)
            .unwrap()
            .to_bytes(),
        before.to_bytes()
    );
    let bytes = before.to_bytes();
    for next in [
        [1024.0625, -2048.0, 4096.0],
        [1026.0, -2048.0, 4096.0],
        [1033.0, -2048.0, 4096.0],
        [f32::NAN; 3],
    ] {
        assert!(before.with_grid_window(next).is_err());
        assert_eq!(before.to_bytes(), bytes);
    }
    before.cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 64 };
    assert_eq!(
        before
            .with_grid_window(before.cfg.grid_origin)
            .unwrap()
            .to_bytes(),
        before.to_bytes()
    );
    assert!(before.with_grid_window(next).is_err());
}

#[test]
fn grid_window_keeps_unchanged_floor_support_and_rejects_moved_floor() {
    let mut before = state(1);
    before.cfg.time_step = MpmTimeStep::Explicit;
    before.words[1] = (before.cfg.grid_origin[1] + 0.25).to_bits();
    let shifted = before.with_grid_window([1024.5, -2048.0, 4096.0]).unwrap();
    assert_eq!(shifted.words, before.words);
    assert!(
        before
            .with_grid_window([1024.0, -2047.875, 4096.0])
            .is_err()
    );
}

#[test]
fn grid_window_decimal_origin_admission_is_not_a_cell_offset() {
    let mut before = state(1);
    before.cfg.grid_origin = [-3.0, -0.5, -3.0];
    before.cfg.cell_size = 0.1;
    before.cfg.time_step = MpmTimeStep::Explicit;
    before.words[0] = 0.0_f32.to_bits();
    before.words[1] = 0.0_f32.to_bits();
    before.words[2] = 0.0_f32.to_bits();
    let bytes = before.to_bytes();
    for cells in [-1.0_f32, 1.0] {
        let mut next = before.cfg.grid_origin;
        next[2] += cells * before.cfg.cell_size;
        let ratio = (f64::from(next[2]) - f64::from(before.cfg.grid_origin[2]))
            / f64::from(before.cfg.cell_size);
        assert_ne!(ratio.fract(), 0.0);
        let error = before.with_grid_window(next).unwrap_err();
        println!(
            "decimal_window cells={cells} origin={} cell_ratio={ratio:.17} rejection={error}",
            next[2]
        );
        assert_eq!(before.to_bytes(), bytes);
    }
    assert_eq!(
        before
            .with_grid_window(before.cfg.grid_origin)
            .unwrap()
            .to_bytes(),
        bytes
    );
}

fn state(count: u32) -> MpmSoilCheckpoint {
    let mut state = super::super::tests::fixture(count);
    state.cfg.grid_size = 64;
    state.cfg.cell_size = 0.125;
    state.cfg.grid_origin = [1024.0, -2048.0, 4096.0];
    for (i, p) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        for axis in 0..3 {
            p[axis] = (state.cfg.grid_origin[axis] + 2.0 + (i % 17) as f32 / 64.0).to_bits();
        }
    }
    state
}

#[test]
fn lossless_recenter_preserves_history_momenta_energy_and_roundtrip() {
    for count in [0, 1, 129] {
        let before = state(count);
        let bytes = before.to_bytes();
        let origin = [-4.0; 3];
        let after = before.recentered(origin).unwrap();
        let a = before
            .momenta_about(before.cfg.grid_origin.map(f64::from))
            .unwrap();
        let b = after.momenta_about(origin.map(f64::from)).unwrap();
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.linear_momentum, b.linear_momentum);
        assert_eq!(a.angular_momentum, b.angular_momentum);
        assert_eq!(a.affine_angular_momentum, b.affine_angular_momentum);
        assert_eq!(
            before
                .energy_at_height(-2048.0)
                .unwrap()
                .mechanical_energy(),
            after.energy_at_height(-4.0).unwrap().mechanical_energy()
        );
        for stream in 1..STRIDES.len() {
            assert_eq!(before.stream(stream), after.stream(stream));
        }
        for (a, b) in before
            .stream(0)
            .chunks_exact(4)
            .zip(after.stream(0).chunks_exact(4))
        {
            assert_eq!(a[3], b[3]);
            for axis in 0..3 {
                assert_eq!(
                    f64::from(f32::from_bits(a[axis])) - f64::from(before.cfg.grid_origin[axis]),
                    f64::from(f32::from_bits(b[axis])) - f64::from(origin[axis])
                );
            }
        }
        assert_eq!(
            after.recentered(before.cfg.grid_origin).unwrap().to_bytes(),
            bytes
        );
        assert_eq!(before.to_bytes(), bytes);
        assert_eq!(
            MpmSoilCheckpoint::from_bytes(&after.to_bytes(), 129)
                .unwrap()
                .to_bytes(),
            after.to_bytes()
        );
    }
}

#[test]
fn exact_translation_rejects_hidden_f64_loss_f32_loss_and_clamp_changes() {
    assert!(exact_sum(1.0, f64::from(f32::from_bits(1))).is_err());
    assert!(shifted(1.0, 16_777_216.0).is_err());
    assert!(shifted(f32::MAX, f64::from(f32::MAX)).is_err());
    assert!(shifted(f32::MIN_POSITIVE, -f64::from(f32::MIN_POSITIVE.next_down())).is_err());
    assert_eq!(shifted(-0.0, 0.0).unwrap().to_bits(), (-0.0f32).to_bits());
    let before = state(1);
    let bytes = before.to_bytes();
    for origin in [
        [f32::NAN; 3],
        [f32::INFINITY; 3],
        [f32::MAX; 3],
        [16_777_216.0; 3],
    ] {
        assert!(before.recentered(origin).is_err());
        assert_eq!(before.to_bytes(), bytes);
    }
    let mut clamp = state(0);
    clamp.cfg.cell_size = 0.1;
    assert!(clamp.recentered([0.0; 3]).is_err());
    let mut invalid_state = before.clone();
    invalid_state.words[0] = f32::NAN.to_bits();
    assert!(invalid_state.recentered([0.0; 3]).is_err());
    let same = before.recentered(before.cfg.grid_origin).unwrap();
    assert_eq!(same.to_bytes(), bytes);
    let mut rounding = state(1).recentered([0.0; 3]).unwrap();
    rounding.words[0] = (2.0f32.next_up()).to_bits();
    assert!(translation(rounding.cfg, [4.0, 0.0, 0.0]).is_ok());
    assert!(rounding.recentered([4.0, 0.0, 0.0]).is_err());
}

#[test]
fn supported_translation_sweep_matches_integer_coordinate_oracle() {
    let mut count = 0;
    for exponent in -8..=16 {
        let h = 2.0f32.powi(exponent);
        for source_cell in [-8192, -128, 0, 128, 8192] {
            for destination_cell in [-128, 0, 128] {
                let mut before = state(129);
                before.cfg.cell_size = h;
                before.cfg.grid_origin = [source_cell as f32 * h; 3];
                let origin = [destination_cell as f32 * h; 3];
                for (i, p) in before.words[..129 * 4].chunks_exact_mut(4).enumerate() {
                    for word in &mut p[..3] {
                        *word = ((source_cell * 64 + 128 + i as i32) as f32 * h / 64.0).to_bits();
                    }
                }
                let after = before.recentered(origin).unwrap();
                for (i, p) in after.stream(0).chunks_exact(4).enumerate() {
                    let expected =
                        (destination_cell * 64 + 128 + i as i32) as f64 * f64::from(h) / 64.0;
                    for &word in &p[..3] {
                        assert_eq!(f64::from(f32::from_bits(word)), expected);
                        count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(count, 145125);
}

#[test]
fn admitted_boundaries_preserve_separate_and_fused_arithmetic() {
    let mut accepted = 0;
    let mut rejected = 0;
    for bits in [0x3dcccccd, 0x3e000000, 0x3e124925, 0x3f7fffff, 0x3f800001] {
        for source in [-1024.0, -1.0, 0.0, 1.0, 1024.0] {
            for target in [-4.0, -0.125, 0.0, 0.125, 4.0] {
                let mut before = state(0);
                before.cfg.cell_size = f32::from_bits(bits);
                before.cfg.grid_origin = [source; 3];
                let Ok(after) = before.recentered([target; 3]) else {
                    rejected += 1;
                    continue;
                };
                let delta = f64::from(target) - f64::from(source);
                for factor in [2.0f32, 61.0] {
                    for (a, b) in [
                        (
                            source + factor * before.cfg.cell_size,
                            target + factor * after.cfg.cell_size,
                        ),
                        (
                            factor.mul_add(before.cfg.cell_size, source),
                            factor.mul_add(after.cfg.cell_size, target),
                        ),
                    ] {
                        assert_eq!(f64::from(b) - f64::from(a), delta);
                    }
                }
                accepted += 1;
            }
        }
    }
    assert!(accepted > 25);
    assert!(rejected > 25);
    println!(
        "boundary matrix: {accepted} admitted, {rejected} rejected; fused and separate endpoints preserved"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn detached_recenter_reuses_buffers_invalidates_work_and_continues() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let before = state(129);
        let mut simulation = super::super::tests::new_sim(&engine, before.cfg, mode);
        simulation.restore_soil_with(queue, &before).unwrap();
        let buffers: Vec<_> = simulation
            .checkpoint_buffers()
            .into_iter()
            .cloned()
            .collect();
        let old = simulation.prepare_step(0.0001, 1).unwrap();
        assert_eq!(
            simulation.recenter_soil_with(queue, [-4.0; 3]).unwrap(),
            [-1028.0, 2044.0, -4100.0]
        );
        let after = before.recentered([-4.0; 3]).unwrap();
        assert_eq!(
            simulation.checkpoint_soil_with(queue).unwrap().to_bytes(),
            after.to_bytes()
        );
        assert_eq!(simulation.cfg, after.cfg);
        assert!(simulation.restore_soil_with(queue, &before).is_err());
        for (saved, live) in buffers.iter().zip(simulation.checkpoint_buffers()) {
            assert_eq!(saved, live);
        }
        let mut batch =
            simulation.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&old).is_err());
        drop(batch);
        let epoch = simulation.step_epoch.clone();
        assert!(
            simulation
                .recenter_soil_with(queue, [16_777_216.0; 3])
                .is_err()
        );
        assert!(Arc::ptr_eq(&epoch, &simulation.step_epoch));
        assert_eq!(
            simulation.checkpoint_soil_with(queue).unwrap().to_bytes(),
            after.to_bytes()
        );
        simulation
            .recenter_soil_with(queue, before.cfg.grid_origin)
            .unwrap();
        assert_eq!(
            simulation.checkpoint_soil_with(queue).unwrap().to_bytes(),
            before.to_bytes()
        );
        simulation.recenter_soil_with(queue, [-4.0; 3]).unwrap();
        simulation.step_n_with(device, queue, 0.0001, 2).unwrap();
        let evolved = simulation.checkpoint_soil_with(queue).unwrap();
        assert_ne!(evolved.stream(0), after.stream(0));
        assert!(evolved.stream(0).chunks_exact(4).all(|p| {
            p[..3]
                .iter()
                .all(|&v| (-4.0..4.0).contains(&f32::from_bits(v)))
        }));
        println!(
            "mode={mode}: 129 mixed particles recentered, lossless return, stable buffers, stale work rejected, resumed local stepping"
        );
    }
    let before = state(1);
    let local = before.recentered([-4.0; 3]).unwrap();
    let mut reference = super::super::tests::new_sim(&engine, local.cfg, 2);
    reference.restore_soil_with(queue, &local).unwrap();
    let mut candidate = super::super::tests::new_sim(&engine, before.cfg, 2);
    candidate.restore_soil_with(queue, &before).unwrap();
    candidate.recenter_soil_with(queue, [-4.0; 3]).unwrap();
    for _ in 0..4 {
        reference.step_n_with(device, queue, 0.0001, 2).unwrap();
        candidate.step_n_with(device, queue, 0.0001, 2).unwrap();
        assert_eq!(
            candidate.checkpoint_soil_with(queue).unwrap().to_bytes(),
            reference.checkpoint_soil_with(queue).unwrap().to_bytes()
        );
    }
    let mut rounding = local.recentered([0.0; 3]).unwrap();
    rounding.words[0] = (2.0f32.next_up()).to_bits();
    let mut simulation = super::super::tests::new_sim(&engine, rounding.cfg, 2);
    simulation.restore_soil_with(queue, &rounding).unwrap();
    let epoch = simulation.step_epoch.clone();
    assert!(
        simulation
            .recenter_soil_with(queue, [4.0, 0.0, 0.0])
            .is_err()
    );
    assert!(Arc::ptr_eq(&epoch, &simulation.step_epoch));
    assert_eq!(
        simulation.checkpoint_soil_with(queue).unwrap().to_bytes(),
        rounding.to_bytes()
    );
    simulation
        .recenter_soil_with(queue, rounding.cfg.grid_origin)
        .unwrap();
    assert!(Arc::ptr_eq(&epoch, &simulation.step_epoch));
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn recenter_rejects_owned_rigid_and_guarded_soil_without_mutation() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let before = state(1);
    let mut simulation = super::super::tests::new_sim(&engine, before.cfg, 2);
    simulation.restore_soil_with(queue, &before).unwrap();
    simulation
        .set_rigid_spheres(queue, &[([1026.0, -2046.0, 4098.0], 0.1, [0.0; 3])])
        .unwrap();
    let epoch = simulation.step_epoch.clone();
    assert!(simulation.recenter_soil_with(queue, [-4.0; 3]).is_err());
    assert!(Arc::ptr_eq(&epoch, &simulation.step_epoch));
    simulation.set_rigid_spheres(queue, &[]).unwrap();
    let archive = before
        .partition_tiles(
            molla_vehicle::SoilTileLayout::new([0.0; 3], 8.0).unwrap(),
            &[molla_vehicle::SoilParticleId(9)],
            &[
                molla_vehicle::SoilMaterialId(1),
                molla_vehicle::SoilMaterialId(2),
                molla_vehicle::SoilMaterialId(3),
            ],
        )
        .unwrap();
    let selection = archive
        .select(&archive.tiles().collect::<Vec<_>>(), 1)
        .unwrap();
    simulation
        .restore_soil_selection(&engine, &selection)
        .unwrap();
    let identity = simulation.soil_identity_snapshot().unwrap().clone();
    assert!(simulation.recenter_soil_with(queue, [-4.0; 3]).is_err());
    assert!(simulation.is_soil_identity_current(&identity));
    simulation.restore_soil_with(queue, &before).unwrap();
    simulation.prepare_soil_motion_guard().unwrap();
    assert!(simulation.recenter_soil_with(queue, [-4.0; 3]).is_err());
    assert_eq!(simulation.cfg, before.cfg);
    assert_eq!(
        simulation.checkpoint_soil_with(queue).unwrap().to_bytes(),
        before.to_bytes()
    );
}
