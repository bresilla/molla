use super::*;

#[test]
fn quiescence_support_budget_decode_and_shader_contract() {
    let limits = wgpu::Limits::default();
    assert_eq!(
        sizes(129, 64, &limits, 1049688).unwrap(),
        [64, 1048576, 516, 532]
    );
    assert!(sizes(129, 64, &limits, 1049687).is_err());
    assert!(sizes(1, u32::MAX, &limits, u64::MAX).is_err());
    assert!(sizes(1, 2, &limits, u64::MAX).is_err());
    assert!(sizes(u32::MAX, 64, &limits, u64::MAX).is_err());
    assert_eq!(sizes(0, 4, &limits, 344).unwrap(), [64, 256, 4, 20]);
    let decoded = decode(&[9, 18, 25, 26]).unwrap();
    assert_eq!(
        decoded[2],
        MpmSoilStencilSupport {
            quiet: true,
            touches_quiet: true,
            touches_nonquiet: true
        }
    );
    for bad in [0, 1, 2, 4, 8, 16, 17, 24, 27, u32::MAX] {
        assert!(decode(&[bad]).is_err());
    }
    assert_eq!(
        decode_closure(&[1, 65, 66]).unwrap(),
        [
            MpmSoilSupportClosureParticle {
                quiet: true,
                connected_to_nonquiet: false
            },
            MpmSoilSupportClosureParticle {
                quiet: true,
                connected_to_nonquiet: true
            },
            MpmSoilSupportClosureParticle {
                quiet: false,
                connected_to_nonquiet: true
            },
        ]
    );
    for bad in [0, 2, 4, 9, 18, 64, 67, 129, 193, u32::MAX] {
        assert!(decode_closure(&[1, bad]).is_err());
    }
    assert!(
        decode_closure(&[129])
            .unwrap_err()
            .to_string()
            .contains("did not converge")
    );
    let source = include_str!("../shaders/mpm_soil_support.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let base = "let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));";
    assert!(source.contains(base));
    assert!(include_str!("../shaders/mpm.wgsl").contains(base));
}

#[cfg(feature = "resource-counters")]
fn closure_reference(
    positions: &[[f32; 3]],
    origin: [f32; 3],
    spacing: f32,
    quiet: &[bool],
) -> Vec<MpmSoilSupportClosureParticle> {
    let bases: Vec<[i32; 3]> = positions
        .iter()
        .map(|p| {
            std::array::from_fn(|a| (((p[a] - origin[a]) * (1.0 / spacing) - 0.5).floor()) as i32)
        })
        .collect();
    let mut reached: Vec<_> = quiet.iter().map(|q| !q).collect();
    let mut queue: std::collections::VecDeque<_> = reached
        .iter()
        .enumerate()
        .filter_map(|(i, &r)| r.then_some(i))
        .collect();
    while let Some(i) = queue.pop_front() {
        for j in 0..bases.len() {
            if !reached[j] && (0..3).all(|a| bases[i][a].abs_diff(bases[j][a]) <= 2) {
                reached[j] = true;
                queue.push_back(j);
            }
        }
    }
    quiet
        .iter()
        .zip(reached)
        .map(
            |(&quiet, connected_to_nonquiet)| MpmSoilSupportClosureParticle {
                quiet,
                connected_to_nonquiet,
            },
        )
        .collect()
}

#[cfg(feature = "resource-counters")]
fn reference(
    positions: &[[f32; 3]],
    origin: [f32; 3],
    spacing: f32,
    quiet: &[bool],
) -> Vec<MpmSoilStencilSupport> {
    let bases: Vec<[i32; 3]> = positions
        .iter()
        .map(|p| {
            std::array::from_fn(|a| (((p[a] - origin[a]) * (1.0 / spacing) - 0.5).floor()) as i32)
        })
        .collect();
    bases
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut record = MpmSoilStencilSupport {
                quiet: quiet[i],
                touches_quiet: false,
                touches_nonquiet: false,
            };
            for (j, b) in bases.iter().enumerate() {
                if (0..3).all(|k| a[k].abs_diff(b[k]) <= 2) {
                    if quiet[j] {
                        record.touches_quiet = true;
                    } else {
                        record.touches_nonquiet = true;
                    }
                }
            }
            record
        })
        .collect()
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; conservative support occupancy, provenance and unchanged physics"]
fn quiescence_gpu_support_stencils_and_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let device = engine.device().unwrap();
    let positions = [
        [-1.0, 0.5, 0.0],
        [-0.8, 0.5, 0.0],
        [-0.6, 0.5, 0.0],
        [1.0, 0.5, 0.0],
    ];
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 129,
            max_rigid: 1,
            grid_size: 64,
            grid_origin: [-3.0, -0.5, -3.0],
            cell_size: 0.1,
            gravity_y: 0.0,
            time_step: MpmTimeStep::Explicit,
            ..MpmConfig::sand()
        };
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        sim.set_particles(queue, &positions).unwrap();
        if mode == 1 {
            sim.move_soil_grid_window_cells_with(queue, [1, -1, 1], 129)
                .unwrap();
        }
        assert!(sim.prepare_soil_support(u64::MAX).is_err());
        sim.prepare_soil_motion_guard().unwrap();
        sim.set_soil_motion_bound(Some([0.01; 3]));
        sim.prepare_soil_quiescence(2080).unwrap();
        let allocations = DeviceScope::start().unwrap();
        assert!(sim.prepare_soil_support(1049687).is_err());
        assert_eq!(allocations.finish().observations, 0);
        sim.prepare_soil_support(1049688).unwrap();
        assert_eq!(sim.soil_support_bytes(), 1049688);
        let step = sim.prepare_step(0.0001, 2).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&step).unwrap();
        batch.submit(queue);
        let snapshot = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert!(
            snapshot
                .particles()
                .iter()
                .all(|p| p.consecutive_steps == 2)
        );
        let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        let result = sim.read_soil_support(&engine, &snapshot, 2, 129).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.upload_bytes, 64);
        assert_eq!(transfers.readback_bytes, 32);
        assert_eq!(dispatches.finish().encoded_dispatches, 2);
        assert_eq!(
            result.particles(),
            reference(&positions, cfg.grid_origin, cfg.cell_size, &[true; 4])
        );
        assert_eq!(result.minimum_quiet_steps(), 2);
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), before);
        assert!(sim.read_soil_support(&engine, &snapshot, 0, 129).is_err());
        assert!(sim.read_soil_support(&engine, &snapshot, 2, 3).is_err());
        let all_nonquiet = sim.read_soil_support(&engine, &snapshot, 3, 129).unwrap();
        assert!(
            all_nonquiet
                .particles()
                .iter()
                .all(|p| !p.quiet && p.touches_nonquiet && !p.touches_quiet)
        );

        let records = [
            0u32,
            0.0001f32.to_bits(),
            2,
            0.0001f32.to_bits(),
            2,
            0.0001f32.to_bits(),
            2,
            0.0001f32.to_bits(),
        ];
        queue.write_buffer(
            &sim.motion_guard
                .as_ref()
                .unwrap()
                .quiescence
                .as_ref()
                .unwrap()
                .records,
            0,
            bytemuck::cast_slice(&records),
        );
        let controlled = sim.read_soil_quiescence(&engine, 129).unwrap();
        let mixed = sim.read_soil_support(&engine, &controlled, 2, 129).unwrap();
        let expected = reference(
            &positions,
            cfg.grid_origin,
            cfg.cell_size,
            &[false, true, true, true],
        );
        assert_eq!(mixed.particles(), expected);
        assert!(mixed.particles()[1].quiet && mixed.particles()[1].touches_nonquiet);
        assert!(!mixed.particles()[2].touches_nonquiet);
        assert!(!mixed.particles()[3].touches_nonquiet);
        assert!(
            sim.read_soil_support_closure(&engine, &controlled, 2, 129, 1)
                .unwrap_err()
                .to_string()
                .contains("did not converge")
        );
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        let closure = sim
            .read_soil_support_closure(&engine, &controlled, 2, 129, 2)
            .unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (64, 32));
        assert_eq!(dispatches.finish().encoded_dispatches, 6);
        assert_eq!(
            closure.particles(),
            closure_reference(
                &positions,
                cfg.grid_origin,
                cfg.cell_size,
                &[false, true, true, true]
            )
        );
        assert!(closure.particles()[2].connected_to_nonquiet);
        assert!(!closure.particles()[3].connected_to_nonquiet);
        assert_eq!((closure.minimum_quiet_steps(), closure.rounds()), (2, 2));
        for rounds in [2, 3, 64] {
            assert_eq!(
                sim.read_soil_support_closure(&engine, &controlled, 2, 129, rounds)
                    .unwrap()
                    .particles(),
                closure.particles()
            );
        }
        for (threshold, bound, rounds) in [(2, 129, 0), (2, 129, 65), (0, 129, 2), (2, 3, 2)] {
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            assert!(
                sim.read_soil_support_closure(&engine, &controlled, threshold, bound, rounds)
                    .is_err()
            );
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
            assert_eq!(dispatches.finish().encoded_dispatches, 0);
        }
        assert_eq!(
            sim.read_soil_support(&engine, &controlled, 2, 129)
                .unwrap()
                .particles(),
            mixed.particles()
        );
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), before);

        for bad in [
            cfg.grid_origin,
            [f32::NAN, 0.5, 0.0],
            [f32::INFINITY, 0.5, 0.0],
            [1.0e20, 0.5, 0.0],
        ] {
            queue.write_buffer(
                &sim.p_pos,
                0,
                bytemuck::cast_slice(&[bad[0], bad[1], bad[2], 0.0]),
            );
            assert!(sim.read_soil_support(&engine, &controlled, 2, 129).is_err());
            assert!(
                sim.read_soil_support_closure(&engine, &controlled, 2, 129, 2)
                    .is_err()
            );
        }
        queue.write_buffer(
            &sim.p_pos,
            0,
            bytemuck::cast_slice(&[positions[0][0], positions[0][1], positions[0][2], 0.0]),
        );
        assert_eq!(
            sim.read_soil_support(&engine, &controlled, 2, 129)
                .unwrap()
                .particles(),
            mixed.particles()
        );
        sim.disable_soil_support();
        assert_eq!(sim.soil_support_bytes(), 0);
        assert!(sim.read_soil_support(&engine, &controlled, 2, 129).is_err());
        assert!(
            sim.read_soil_support_closure(&engine, &controlled, 2, 129, 2)
                .is_err()
        );
        sim.prepare_soil_support(1049688).unwrap();
        sim.validate_soil_quiescence(mixed.quiescence()).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&step).unwrap();
        drop(batch);
        assert!(sim.validate_soil_quiescence(mixed.quiescence()).is_err());
        assert!(sim.validate_soil_quiescence(closure.quiescence()).is_err());
        assert!(
            sim.read_soil_support_closure(&engine, &controlled, 2, 129, 2)
                .is_err()
        );
        assert!(sim.read_soil_support(&engine, &controlled, 2, 129).is_err());
        sim.set_particles(queue, &vec![positions[3]; 129]).unwrap();
        let full = sim.prepare_step(0.0001, 2).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&full).unwrap();
        batch.submit(queue);
        let full = sim.read_soil_quiescence(&engine, 129).unwrap();
        let full_support = sim.read_soil_support(&engine, &full, 2, 129).unwrap();
        assert_eq!(full_support.particles().len(), 129);
        assert!(
            full_support
                .particles()
                .iter()
                .all(|p| p.quiet && p.touches_quiet && !p.touches_nonquiet)
        );
        for threshold in [2, 3] {
            let closure = sim
                .read_soil_support_closure(&engine, &full, threshold, 129, 1)
                .unwrap();
            assert_eq!(
                closure.particles(),
                closure_reference(
                    &vec![positions[3]; 129],
                    cfg.grid_origin,
                    cfg.cell_size,
                    &vec![threshold == 2; 129]
                )
            );
        }
        sim.set_particles(queue, &[]).unwrap();
        assert!(
            sim.validate_soil_quiescence(full_support.quiescence())
                .is_err()
        );
        let dispatches = DispatchScope::start().unwrap();
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        assert!(
            sim.read_soil_support_closure(&engine, &full, 2, 129, 1)
                .is_err()
        );
        assert_eq!(dispatches.finish().encoded_dispatches, 0);
        sim.disable_soil_quiescence();
        assert_eq!(sim.soil_support_bytes(), 0);
        println!(
            "support mode={mode} direct_and_transitive=true short_round_rejected=true physical_bytes_unchanged=true direct_dispatches=2 closure_round2_dispatches=6 retained_bytes=1049688"
        );
    }
}
