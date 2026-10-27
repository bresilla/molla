use super::super::replay_tests::read_words;
use super::*;

pub(super) fn seeded(engine: &Engine, sparse: bool) -> MpmSimulation {
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 35;
    cfg.max_particles = 129;
    cfg.youngs = 1000.;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
    let mut sim = if sparse {
        MpmSimulation::new_sparse(engine, cfg, 128).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    sim.set_active_grid(true).unwrap();
    let mut points: Vec<_> = (0..129)
        .map(|i| {
            [
                -0.15 + (i % 7) as f32 * 0.021,
                0.2 + (i / 7 % 7) as f32 * 0.021,
                -0.1 + (i / 49) as f32 * 0.021,
            ]
        })
        .collect();
    points[0] = [-0.82, 0.025, -0.775];
    points[1] = [0.94, 1.74, 0.94];
    points[2] = [0.96, 1.76, 0.96];
    sim.set_particles_with_velocity(queue, &points, &vec![[0.02, -0.01, 0.005]; 129])
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
                velocity_damping: 1.,
                affine_damping: 2.,
            },
        ],
        (0..129).map(|i| i % 3).collect(),
    )
    .unwrap();
    sim.set_rigid_spheres(queue, &[([-0.09, 0.23, -0.07], 0.045, [0.015, -0.005, 0.])])
        .unwrap();
    sim.prepare_parallel_particle_tile_ordering(queue, 129)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_deterministic_p2g().unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim
}

pub(super) fn world_grid(engine: &Engine, sim: &MpmSimulation) -> [Vec<u32>; 3] {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = [&sim.g_mass, &sim.g_momentum, &sim.g_kdiag].map(|b| read_words(device, queue, b));
    let cells = sim.cfg.grid_size.pow(3) as usize;
    if !sim.sparse_grid {
        return [
            source[0][..cells].to_vec(),
            source[1][..cells * 3].to_vec(),
            source[2][..cells].to_vec(),
        ];
    }
    let mut output = [vec![0; cells], vec![0; cells * 3], vec![0; cells]];
    let ids = sim.read_active_tile_ids_with(device, queue).unwrap();
    let width = sim.cfg.grid_size.div_ceil(4);
    for (slot, tile) in ids.into_iter().enumerate() {
        for local in 0..64 {
            let xyz = [
                tile % width * 4 + local % 4,
                tile / width % width * 4 + local / 4 % 4,
                tile / (width * width) * 4 + local / 16,
            ];
            if xyz.iter().any(|&v| v >= sim.cfg.grid_size) {
                continue;
            }
            let world =
                (xyz[0] + sim.cfg.grid_size * (xyz[1] + sim.cfg.grid_size * xyz[2])) as usize;
            let physical = slot * 64 + local as usize;
            for (field, stride) in [1, 3, 1].into_iter().enumerate() {
                output[field][world * stride..(world + 1) * stride]
                    .copy_from_slice(&source[field][physical * stride..(physical + 1) * stride]);
            }
        }
    }
    output
}

pub(super) fn encode_transfer(
    sim: &MpmSimulation,
    encoder: &mut wgpu::CommandEncoder,
    gather: bool,
) {
    sim.particle_order.as_ref().unwrap().encode(
        encoder,
        &sim.p_material,
        u64::from(sim.cfg.max_particles) * 4,
    );
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.active_pipelines[0]);
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    }
    sim.active_tiles
        .encode_build(encoder, sim.particle_count, sim.rigid_count);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    if gather {
        sim.deterministic_p2g
            .as_ref()
            .unwrap()
            .encode(&mut pass, sim);
    } else {
        pass.set_pipeline(&sim.pipe_p2g);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn gather_fixed_input_is_bit_repeatable_and_matches_scatter() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut sim = seeded(&engine, sparse);
        sim.step_n_with(device, queue, 0.0001, 2).unwrap();
        let mut first = None;
        for replay in 0..6 {
            let mut encoder = device.create_command_encoder(&Default::default());
            encode_transfer(&sim, &mut encoder, true);
            queue.submit([encoder.finish()]);
            let output = world_grid(&engine, &sim);
            if let Some(first) = &first {
                assert_eq!(&output, first, "sparse={sparse} replay={replay}");
            } else {
                first = Some(output);
            }
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        encode_transfer(&sim, &mut encoder, false);
        queue.submit([encoder.finish()]);
        let reference = world_grid(&engine, &sim);
        let gathered = first.unwrap();
        let mut errors = [0_f32; 3];
        for (field, (a, b)) in gathered.iter().zip(&reference).enumerate() {
            for (&a, &b) in a.iter().zip(b) {
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                assert!(a.is_finite() && b.is_finite());
                errors[field] = errors[field].max((a - b).abs());
            }
        }
        println!(
            "sparse={sparse} 6 exact fixed-input gathers; scatter absolute errors mass/momentum/stiffness={errors:?}"
        );
        assert!(errors.iter().all(|&v| v <= 2e-5));
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn gather_solver_repeatability_command_records_and_hot_resources() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut a = seeded(&engine, sparse);
        let mut b = seeded(&engine, sparse);
        let steps = [(0.0001, 2), (0.0002, 2), (0.00008, 1)];
        let records: Vec<_> = steps
            .iter()
            .map(|&(dt, n)| b.prepare_step(dt, n).unwrap())
            .collect();
        for &(dt, n) in &steps {
            a.step_n_with(device, queue, dt, n).unwrap();
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let before = device.get_internal_counters().hal;
        let encoder = device.create_command_encoder(&Default::default());
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let mut batch = b.begin_command_batch(encoder);
        for record in &records {
            batch.encode_step(record).unwrap();
        }
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
        for (a, b) in [
            (&a.p_pos, &b.p_pos),
            (&a.p_vel, &b.p_vel),
            (&a.p_c, &b.p_c),
            (&a.p_f, &b.p_f),
            (&a.p_jp, &b.p_jp),
        ] {
            assert_eq!(read_words(device, queue, a), read_words(device, queue, b));
        }
        let mut wrench_error = 0_f32;
        for (&a, &b) in read_words(device, queue, &a.rigid_wrench)
            .iter()
            .zip(&read_words(device, queue, &b.rigid_wrench))
        {
            wrench_error = wrench_error.max((f32::from_bits(a) - f32::from_bits(b)).abs());
        }
        assert!(wrench_error <= 2e-5, "wrench error {wrench_error}");
        assert_eq!(
            b.read_active_grid_status_with(device, queue)
                .unwrap()
                .completed_substeps,
            5
        );
        assert_eq!(b.deterministic_p2g_storage_bytes(), 129 * 64);
        println!(
            "sparse={sparse} 5 substeps: particle/history bits identical across schedules, wrench_error={wrench_error}, zero hot buffers/transfers"
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn gather_modes_growth_invalidation_and_profile_are_explicit() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = seeded(&engine, true);
    assert!(sim.set_active_grid(false).is_err());
    assert!(sim.set_particle_tile_ordering(false).is_err());
    assert!(sim.set_deferred_p2g_lookup(true).is_err());
    assert!(sim.prepare_particle_tile_ordering(queue, 129).is_err());
    let stale = sim.prepare_step(0.0001, 1).unwrap();
    sim.set_deterministic_p2g(false).unwrap();
    {
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err());
    }
    sim.prepare_particle_tile_ordering(queue, 129).unwrap();
    assert_eq!(sim.deterministic_p2g_storage_bytes(), 0);
    assert!(sim.set_deterministic_p2g(true).is_err());
    sim.prepare_deterministic_p2g().unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim.reserve_sparse_tiles(256).unwrap();
    sim.prepare_step_profiling(2).unwrap();
    sim.step_n_profiled_with(device, queue, 0.0001, 2).unwrap();
    let profile = sim.read_step_profile().unwrap().unwrap();
    assert!(profile.particle_tile_ordering);
    assert!(
        profile
            .stages
            .iter()
            .any(|stage| stage.stage == MpmProfileStage::ParticleToGrid)
    );
    sim.set_p2g_mode(true);
    assert!(!sim.deterministic_p2g_enabled());
}

pub(super) fn seeded_benchmark(engine: &Engine) -> MpmSimulation {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 64;
    cfg.max_particles = 50_000;
    let mut sim = MpmSimulation::new_sparse(engine, cfg, 512).unwrap();
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.5 + (i % 40) as f32 * 0.021,
                0.3 + (i / 40 % 40) as f32 * 0.021,
                -0.5 + (i / 1600) as f32 * 0.021,
            ]
        })
        .collect();
    sim.set_particles_with_velocity(queue, &points, &vec![[0.03, -0.02, 0.01]; 50_000])
        .unwrap();
    sim.prepare_parallel_particle_tile_ordering(queue, 50_000)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_deterministic_p2g().unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    let mut uniform = sim.step_uniforms(0.0001);
    uniform.grid_layout[2] = 0;
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniform));
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    sim
}

#[test]
#[ignore = "needs a real GPU adapter; component timing diagnostic"]
fn gather_transfer_benchmark() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let sim = seeded_benchmark(&engine);
    let mut samples: [Vec<f64>; 2] = Default::default();
    let mut last: [Option<[Vec<u32>; 3]>; 2] = [None, None];
    for round in 0..5 {
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let started = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            if mode == 1 {
                sim.particle_order.as_ref().unwrap().encode(
                    &mut encoder,
                    &sim.p_material,
                    50_000 * 4,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.active_pipelines[0]);
                pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            }
            sim.active_tiles
                .encode_build(&mut encoder, sim.particle_count, 0);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                if mode == 1 {
                    sim.deterministic_p2g
                        .as_ref()
                        .unwrap()
                        .encode(&mut pass, &sim);
                } else {
                    pass.set_pipeline(&sim.pipe_p2g_two_pass);
                    pass.dispatch_workgroups(50_000_u32.div_ceil(64), 1, 1);
                }
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let ms = started.elapsed().as_secs_f64() * 1000.;
            if round > 0 {
                samples[mode].push(ms);
                println!(
                    "gather_transfer_sample round={round} mode={mode} particles=50000 completed_ms={ms} simulated_seconds=0 accepted=false"
                );
            }
            if round == 4 {
                last[mode] = Some(world_grid(&engine, &sim));
            }
        }
    }
    let mut errors = [0_f32; 3];
    for (field, (a, b)) in last[0]
        .as_ref()
        .unwrap()
        .iter()
        .zip(last[1].as_ref().unwrap())
        .enumerate()
    {
        for (&a, &b) in a.iter().zip(b) {
            let a = f32::from_bits(a);
            let b = f32::from_bits(b);
            assert!(a.is_finite() && b.is_finite());
            errors[field] = errors[field].max((a - b).abs());
        }
    }
    for (mode, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        println!(
            "gather_transfer_summary mode={mode} median_completed_ms={} range_ms=[{},{}] accepted=false",
            (values[1] + values[2]) * 0.5,
            values[0],
            values[3]
        );
    }
    println!(
        "gather_transfer_final_grid_absolute_errors={errors:?} cache_bytes={} sort_bytes={} accepted=false",
        sim.deterministic_p2g_storage_bytes(),
        sim.particle_tile_ordering_storage_bytes()
    );
    assert!(
        errors.iter().all(|&v| v <= 2e-5),
        "transfer comparison exceeds frozen field tolerance"
    );
}
