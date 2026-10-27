use super::*;
use crate::mpm::replay_tests::read_words;

#[cfg(feature = "resource-counters")]
#[path = "mpm_compensated_grid_evolution_tests.rs"]
mod grid_evolution;

#[cfg(feature = "resource-counters")]
#[path = "mpm_rigid_wrench_compensation_tests.rs"]
mod wrench_compensation;

fn encode(sim: &mut MpmSimulation, encoder: &mut wgpu::CommandEncoder) {
    sim.initialize_active_grid(encoder, false);
    sim.encode_particle_order(encoder, false, 0);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.active_pipelines[0]);
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    }
    sim.active_tiles
        .encode_build(encoder, sim.particle_count, 0);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    sim.encode_p2g(&mut pass);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU and frozen50k inputs; corrected pair and matched-work timings"]
fn corrected_transfer_frozen_50k_pairs() {
    frozen_50k_pairs(false);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU and frozen50k inputs; all-field transfer accuracy and paired cost"]
fn compensated_grid_transfer_frozen_50k_pairs() {
    frozen_50k_pairs(true);
}

#[cfg(feature = "resource-counters")]
fn frozen_50k_pairs(all_fields: bool) {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_CORRECTED_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_CORRECTED_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let frozen = std::fs::read(input.join("soil.bin")).unwrap();
    let checkpoint = MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap();
    let expected = std::fs::read(input.join("gather-2.bin")).unwrap();
    let reference: Vec<_> = std::fs::read(input.join("sum.bin"))
        .unwrap()
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(reference.len(), 64_usize.pow(3));
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let modes = if all_fields { 4 } else { 3 };
    let mut sims: Vec<_> = (0..modes)
        .map(|mode| {
            let mut sim = tests::seeded_benchmark(&engine);
            sim.restore_soil_with(queue, &checkpoint).unwrap();
            if mode != 2 {
                sim.set_p2g_mode(true);
                sim.set_particle_tile_ordering(false).unwrap();
                if mode == 1 {
                    sim.prepare_compensated_stiffness(131072).unwrap();
                    sim.set_compensated_stiffness(true).unwrap();
                } else if mode == 3 {
                    let bytes = sim.g_mass.size() * 5;
                    sim.prepare_compensated_grid_transfer(bytes).unwrap();
                    sim.set_compensated_grid_transfer(true).unwrap();
                    assert_eq!(sim.compensated_grid_transfer_storage_bytes(), bytes);
                }
            } else {
                sim.prepare_deterministic_p2g_with_compensated_stiffness(3_200_000)
                    .unwrap();
                sim.set_deterministic_p2g(true).unwrap();
                assert!(sim.deterministic_p2g_compensated_stiffness_enabled());
            }
            queue.write_buffer(
                &sim.uniform_buf,
                0,
                bytemuck::bytes_of(&sim.step_uniforms(0.0001)),
            );
            sim
        })
        .collect();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let times = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut samples = Vec::new();
    let mut first_gather = None;
    let mut first_candidate: Option<[Vec<u32>; 3]> = None;
    let mut repeat_errors = [0_f64; 3];
    let mut repeat_changed_words = [0_usize; 3];
    let mut max_errors = vec![[0_f64; 4]; modes];
    for round in 0..8 {
        let mut grids: Vec<Option<[Vec<u32>; 3]>> = vec![None; modes];
        let mut order: Vec<_> = (0..modes).collect();
        if round % 2 != 0 {
            order.reverse();
        }
        for mode in order {
            let sim = &mut sims[mode];
            let mut encoder = device.create_command_encoder(&Default::default());
            sim.initialize_active_grid(&mut encoder, false);
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let start = std::time::Instant::now();
            encoder.write_timestamp(&queries, 0);
            encode(sim, &mut encoder);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &times, 0);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(resources.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!(transfers.upload_bytes, 0);
            assert_eq!(transfers.readback_bytes, 0);
            let ticks = read_words(device, queue, &times);
            let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
            let gpu_ms = (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            let grid = tests::world_grid(&engine, sim);
            assert!(
                grid.iter()
                    .flatten()
                    .all(|&v| f32::from_bits(v).is_finite())
            );
            for (&word, &value) in grid[2].iter().zip(&reference) {
                max_errors[mode][3] =
                    max_errors[mode][3].max((f64::from(f32::from_bits(word)) - value).abs());
            }
            for (field, words) in grid.iter().enumerate() {
                std::fs::write(
                    output.join(format!("round-{round}-mode-{mode}-field-{field}.bin")),
                    bytemuck::cast_slice(words),
                )
                .unwrap();
            }
            if mode > 0 {
                assert_eq!(
                    bytemuck::cast_slice::<u32, u8>(&grid[2]),
                    expected,
                    "corrected stiffness matches archived rounded reference"
                );
            }
            if mode == 2 {
                if let Some(first) = &first_gather {
                    assert_eq!(&grid, first);
                } else {
                    first_gather = Some(grid.clone());
                }
            }
            if mode == 3 {
                let first = first_candidate.get_or_insert_with(|| grid.clone());
                for field in 0..3 {
                    for (&a, &b) in grid[field].iter().zip(&first[field]) {
                        repeat_changed_words[field] += usize::from(a != b);
                        repeat_errors[field] = repeat_errors[field].max(
                            (f64::from(f32::from_bits(a)) - f64::from(f32::from_bits(b))).abs(),
                        );
                    }
                }
            }
            grids[mode] = Some(grid);
            samples.push(serde_json::json!({"round":round,"mode":mode,"warmup":round<2,"gpu_ms":gpu_ms,"completed_wall_ms":wall_ms}));
        }
        for mode in 0..modes {
            for field in 0..3 {
                for (&a, &b) in grids[mode].as_ref().unwrap()[field]
                    .iter()
                    .zip(&grids[2].as_ref().unwrap()[field])
                {
                    max_errors[mode][field] = max_errors[mode][field]
                        .max(f64::from((f32::from_bits(a) - f32::from_bits(b)).abs()));
                }
            }
        }
    }
    for sim in &mut sims {
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    }
    let passed = max_errors[1..].iter().flatten().all(|&v| v <= 2e-5)
        && repeat_errors.iter().all(|&v| v <= 2e-5);
    let tail_bytes: Vec<_> = sims
        .iter()
        .map(MpmSimulation::compensated_stiffness_storage_bytes)
        .collect();
    let report = serde_json::json!({"samples":samples,"max_errors":max_errors,"bound":2e-5,"corrected_pair_passed":passed,"all_fields":all_fields,"candidate_repeat_errors":repeat_errors,"candidate_repeat_changed_words":repeat_changed_words,"tail_bytes":tail_bytes,"gather_exact_repeats":8,"particle_checkpoints_exact":true,"hot_buffers":0,"uploads":0,"readbacks_in_encode":0,"simulated_seconds":0,"scope":"ordering when selected, active reset/build and all transfer dispatches; excludes setup and observation; no whole-simulator performance claim"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("corrected_transfer_max_errors={max_errors:?} passed={passed}");
    assert!(passed);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; corrected gather lifecycle and implicit evolution"]
fn corrected_transfer_gather_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut sims = [0, 1].map(|_| tests::seeded(&engine, sparse));
        sims[0].set_p2g_mode(true);
        sims[0].prepare_compensated_stiffness(u64::MAX).unwrap();
        sims[0].set_compensated_stiffness(true).unwrap();
        let old = sims[1].prepare_step(0.0001, 1).unwrap();
        let scope = DeviceScope::start().unwrap();
        assert!(
            sims[1]
                .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64 - 1)
                .is_err()
        );
        assert_eq!(scope.finish().observations, 0);
        sims[1]
            .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
            .unwrap();
        assert!(sims[1].deterministic_p2g_compensated_stiffness_enabled());
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&old).is_err());
        }
        let scope = DeviceScope::start().unwrap();
        sims[1]
            .prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
            .unwrap();
        assert_eq!(scope.finish().observations, 0);
        let points = sims[0].read_positions_with(device, queue).unwrap();
        for count in [129, 1, 0, 129] {
            for sim in &mut sims {
                sim.set_particles_with_velocity(
                    queue,
                    &points[..count],
                    &vec![[0.02, -0.01, 0.005]; count],
                )
                .unwrap();
                let record = sim.prepare_step(0.0001, 2).unwrap();
                let encoder = device.create_command_encoder(&Default::default());
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let mut batch = sim.begin_command_batch(encoder);
                batch.encode_step(&record).unwrap();
                batch.submit(queue);
                assert_eq!(resources.finish().observations, 0);
                let t = transfers.finish();
                assert_eq!(t.upload_bytes, 0);
                assert_eq!(t.readback_bytes, 0);
            }
            let mut worst = 0_f32;
            for (field, (a, b)) in sims[0]
                .checkpoint_buffers()
                .into_iter()
                .zip(sims[1].checkpoint_buffers())
                .enumerate()
            {
                let a = read_words(device, queue, a);
                let b = read_words(device, queue, b);
                if field == 7 {
                    assert_eq!(a, b);
                    continue;
                }
                for (&a, &b) in a.iter().zip(&b) {
                    let a = f32::from_bits(a);
                    let b = f32::from_bits(b);
                    assert!(a.is_finite() && b.is_finite());
                    worst = worst.max((a - b).abs());
                }
            }
            println!(
                "corrected_pair_evolution sparse={sparse} count={count} error={worst} hot_buffers=0 uploads=0 readbacks=0"
            );
            assert!(worst <= 2e-5);
        }
        let old = sims[1].prepare_step(0.0001, 1).unwrap();
        sims[1]
            .prepare_deterministic_p2g_with_budget(129 * 64)
            .unwrap();
        assert!(!sims[1].deterministic_p2g_compensated_stiffness_enabled());
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&old).is_err());
        }
    }
    let mut native = MpmSimulation::new_native_explicit(&engine, MpmConfig::sand()).unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(
        native
            .prepare_deterministic_p2g_with_compensated_stiffness(u64::MAX)
            .is_err()
    );
    assert_eq!(resources.finish().observations, 0);
}
