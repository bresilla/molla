use super::*;
use crate::mpm::checkpoint::tests::fixture;
use crate::mpm::replay_tests::read_words;

fn particle_buffers(sim: &MpmSimulation) -> [&wgpu::Buffer; 9] {
    [
        &sim.p_pos,
        &sim.p_vel,
        &sim.p_c,
        &sim.p_f,
        &sim.p_volume,
        &sim.p_mass,
        &sim.p_jp,
        &sim.p_material,
        &sim.tire_compaction_buf,
    ]
}

fn grid_buffers(sim: &MpmSimulation) -> [&wgpu::Buffer; 6] {
    [
        &sim.g_mass,
        &sim.g_momentum,
        &sim.g_velocity,
        &sim.g_rigid_mask,
        &sim.g_kdiag,
        &sim.g_velocity_prev,
    ]
}

fn read(engine: &Engine, buffers: &[&wgpu::Buffer]) -> Vec<Vec<u32>> {
    buffers
        .iter()
        .map(|buffer| read_words(engine.device().unwrap(), engine.queue().unwrap(), buffer))
        .collect()
}

fn restore(queue: &wgpu::Queue, buffers: &[&wgpu::Buffer], words: &[Vec<u32>]) {
    for (buffer, words) in buffers.iter().zip(words) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(words));
    }
}

fn simulation(engine: &Engine, mode: usize, cfg: MpmConfig) -> MpmSimulation {
    let mut sim = if mode == 2 {
        MpmSimulation::new_sparse(engine, cfg, 512).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    sim.set_active_grid(mode != 0).unwrap();
    sim
}

fn errors(a: &[Vec<u32>], b: &[Vec<u32>]) -> [f32; 9] {
    std::array::from_fn(|field| {
        assert_eq!(a[field].len(), b[field].len());
        if field == 7 {
            assert_eq!(a[field], b[field]);
            return 0.0;
        }
        a[field]
            .iter()
            .zip(&b[field])
            .map(|(&x, &y)| {
                let (x, y) = (f32::from_bits(x), f32::from_bits(y));
                assert!(x.is_finite() && y.is_finite());
                (x - y).abs()
            })
            .fold(0.0, f32::max)
    })
}

#[test]
fn fused_source_changes_only_entry_declarations_and_appends_the_ordered_calls() {
    let original = include_str!("../shaders/mpm.wgsl");
    let generated = fused_source(original).unwrap();
    assert_eq!(generated.matches("fn gather_projection(").count(), 1);
    assert!(generated.ends_with("    fused_g2p(gid);\n    fused_plasticity(gid);\n}\n"));
    let mut reconstructed = generated
        .split("\n@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn gather_projection")
        .next()
        .unwrap()
        .to_owned();
    for entry in ["g2p", "plasticity"] {
        reconstructed = reconstructed.replace(&format!("fn fused_{entry}(gid: vec3<u32>)"),
            &format!("@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>)"));
    }
    assert_eq!(reconstructed, original);
    assert!(fused_source("").is_err());
    assert!(fused_source(&format!("{original}\n{original}")).is_err());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn fixed_grid_replay_is_bit_exact_for_all_history_and_partial_workgroups() {
    let engine = Engine::new_gpu().unwrap();
    for mode in 0..3 {
        let mut sim = simulation(&engine, mode, fixture(129).config());
        sim.prepare_fused_gather_projection().unwrap();
        for count in [1, 63, 64, 65, 129] {
            sim.set_fused_gather_projection(false).unwrap();
            sim.restore_soil_with(engine.queue().unwrap(), &fixture(count))
                .unwrap();
            sim.step_n(&engine, 0.0001, 2).unwrap();
            let particles = read(&engine, &particle_buffers(&sim));
            let grid = read(&engine, &grid_buffers(&sim));
            let mut outputs = Vec::new();
            for fused in [false, true] {
                restore(engine.queue().unwrap(), &particle_buffers(&sim), &particles);
                restore(engine.queue().unwrap(), &grid_buffers(&sim), &grid);
                sim.set_fused_gather_projection(fused).unwrap();
                let mut encoder = engine
                    .device()
                    .unwrap()
                    .create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    sim.bind_step(&mut pass);
                    sim.encode_gather_projection(&mut pass, count.div_ceil(64), false, 0);
                }
                engine.queue().unwrap().submit([encoder.finish()]);
                let mut result = read(&engine, &particle_buffers(&sim));
                result.extend(read(&engine, &grid_buffers(&sim)));
                outputs.push(result);
            }
            for (field, (a, b)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
                for (word, (&a, &b)) in a.iter().zip(b).enumerate() {
                    assert_eq!(
                        a, b,
                        "mode {mode}, count {count}, field {field}, word {word}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn full_trajectory_compares_fusion_and_reference_repeat_without_relaxed_gates() {
    let engine = Engine::new_gpu().unwrap();
    let mut passed = true;
    for mode in 0..3 {
        for implicit in [false, true] {
            let mut cfg = fixture(129).config();
            cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 2 }
            } else {
                MpmTimeStep::Explicit
            };
            let mut sim = simulation(&engine, mode, cfg);
            sim.prepare_fused_gather_projection().unwrap();
            let points: Vec<_> = (0..129)
                .map(|i| {
                    [
                        -0.13 + (i % 7) as f32 * 0.021,
                        0.3 + (i / 7 % 7) as f32 * 0.021,
                        -0.07 + (i / 49) as f32 * 0.021,
                    ]
                })
                .collect();
            let mut outputs = Vec::new();
            for fused in [false, false, true] {
                sim.set_particles_with_velocity(
                    engine.queue().unwrap(),
                    &points,
                    &vec![[0.02, -0.01, 0.005]; 129],
                )
                .unwrap();
                sim.set_materials(fixture(129).materials(), (0..129).map(|i| i % 3).collect())
                    .unwrap();
                sim.set_fused_gather_projection(fused).unwrap();
                sim.step_n(&engine, 0.0001, 32).unwrap();
                outputs.push(read(&engine, &particle_buffers(&sim)));
            }
            let repeat = errors(&outputs[0], &outputs[1]);
            let fused = errors(&outputs[0], &outputs[2]);
            println!(
                "trajectory mode={mode} implicit={implicit} steps=32 dt=0.0001 reference_repeat={repeat:?} fused={fused:?}"
            );
            passed &= repeat.iter().chain(&fused).all(|&e| e <= 2e-5);
        }
    }
    assert!(
        passed,
        "full-history absolute 2e-5 gate failed; fusion remains experimental"
    );
}

#[test]
#[ignore = "needs a real GPU adapter with timestamp queries"]
#[cfg(feature = "resource-counters")]
fn prepared_epoch_profiling_and_retained_resources_follow_the_selected_path() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let mut sim = simulation(&engine, 2, fixture(129).config());
    assert!(!sim.fused_gather_projection_enabled());
    assert!(sim.set_fused_gather_projection(true).is_err());
    sim.restore_soil_with(engine.queue().unwrap(), &fixture(129))
        .unwrap();
    let record = sim.prepare_step(0.0001, 2).unwrap();
    sim.prepare_fused_gather_projection().unwrap();
    sim.prepare_fused_gather_projection().unwrap();
    sim.set_fused_gather_projection(true).unwrap();
    {
        let encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        let mut batch = sim.begin_command_batch(encoder);
        assert!(batch.encode_step(&record).is_err());
    }
    sim.prepare_step_profiling(4).unwrap();
    for fused in [true, false] {
        sim.set_fused_gather_projection(fused).unwrap();
        let buffers = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.step_n_profiled_with(engine.device().unwrap(), engine.queue().unwrap(), 0.0001, 4)
            .unwrap();
        assert_eq!(buffers.finish().observations, 0);
        assert_eq!(transfers.finish().readback_bytes, 0);
        let profile = sim.read_step_profile().unwrap().unwrap();
        assert_eq!(profile.fused_gather_projection, fused);
        let count = |stage| {
            profile
                .stages
                .iter()
                .filter(|entry| entry.stage == stage)
                .count()
        };
        assert_eq!(
            count(MpmProfileStage::GatherProjection),
            if fused { 4 } else { 0 }
        );
        assert_eq!(
            count(MpmProfileStage::GridToParticle),
            if fused { 0 } else { 4 }
        );
        assert_eq!(
            count(MpmProfileStage::Plasticity),
            if fused { 0 } else { 4 }
        );
    }
    sim.set_particles(engine.queue().unwrap(), &[]).unwrap();
    sim.set_fused_gather_projection(true).unwrap();
    sim.step_n(&engine, 0.0001, 4).unwrap();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn prescribed_rigid_force_and_torque_match_the_separate_pass_control() {
    let engine = Engine::new_gpu().unwrap();
    let mut passed = true;
    for mode in [0, 2] {
        for implicit in [false, true] {
            let mut cfg = MpmConfig::sand();
            cfg.max_particles = 129;
            cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 2 }
            } else {
                MpmTimeStep::Explicit
            };
            let mut sim = simulation(&engine, mode, cfg);
            sim.prepare_fused_gather_projection().unwrap();
            let points: Vec<_> = (0..129)
                .map(|i| {
                    [
                        -0.13 + (i % 7) as f32 * 0.021,
                        0.3 + (i / 7 % 7) as f32 * 0.021,
                        -0.07 + (i / 49) as f32 * 0.021,
                    ]
                })
                .collect();
            let mut particles = Vec::new();
            let mut wrenches = Vec::new();
            for fused in [false, false, true] {
                sim.set_particles_with_velocity(
                    engine.queue().unwrap(),
                    &points,
                    &vec![[0.02, -0.01, 0.005]; 129],
                )
                .unwrap();
                sim.set_rigid_spheres(
                    engine.queue().unwrap(),
                    &[([-0.08, 0.33, -0.05], 0.06, [0.015, -0.005, 0.003])],
                )
                .unwrap();
                sim.set_fused_gather_projection(fused).unwrap();
                sim.step_n(&engine, 0.0001, 32).unwrap();
                particles.push(read(&engine, &particle_buffers(&sim)));
                wrenches.push(read_words(
                    engine.device().unwrap(),
                    engine.queue().unwrap(),
                    &sim.rigid_wrench,
                ));
            }
            let mut reaction = [0.0_f32; 2];
            for arm in 1..3 {
                for index in [0, 1, 2, 4, 5, 6] {
                    let a = f32::from_bits(wrenches[0][index]);
                    let b = f32::from_bits(wrenches[arm][index]);
                    assert!(a.is_finite() && b.is_finite());
                    reaction[arm - 1] = reaction[arm - 1].max((a - b).abs());
                }
            }
            assert!(
                wrenches[0][..3]
                    .iter()
                    .any(|&word| f32::from_bits(word).abs() > 1e-8)
            );
            let repeat = errors(&particles[0], &particles[1]);
            let fused = errors(&particles[0], &particles[2]);
            println!(
                "rigid mode={mode} implicit={implicit} steps=32 reference_repeat={repeat:?} fused={fused:?} reaction_repeat_fused={reaction:?}"
            );
            passed &= repeat
                .iter()
                .chain(&fused)
                .chain(&reaction)
                .all(|&value| value <= 2e-5);
        }
    }
    assert!(
        passed,
        "coupled particle/force/torque 2e-5 gate failed; no adoption"
    );
}

#[test]
#[ignore = "explicit paired 50k diagnostic; not frame acceptance"]
#[cfg(feature = "resource-counters")]
fn paired_fifty_thousand_benchmark() {
    use std::time::Instant;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 64;
    cfg.max_particles = 50_000;
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.5 + (i % 40) as f32 * 0.021,
                0.3 + (i / 40 % 40) as f32 * 0.021,
                -0.5 + (i / 1600) as f32 * 0.021,
            ]
        })
        .collect();
    let velocity = vec![[0.03, -0.02, 0.01]; 50_000];
    let mut passed = true;
    let modes: &[usize] = match std::env::var("MOLLA_FUSION_BENCH_MODE").as_deref() {
        Ok("dense") => &[0],
        Ok("compact") => &[2],
        Ok("both") | Err(_) => &[0, 2],
        Ok(value) => panic!("invalid fusion benchmark mode {value}"),
    };
    for &mode in modes {
        let mut sim = simulation(&engine, mode, cfg);
        sim.prepare_fused_gather_projection().unwrap();
        let mut times: [Vec<f64>; 2] = Default::default();
        for round in 0..8 {
            let mut outputs: [Option<Vec<Vec<u32>>>; 2] = [None, None];
            for candidate in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                sim.set_particles_with_velocity(queue, &points, &velocity)
                    .unwrap();
                sim.set_fused_gather_projection(candidate == 1).unwrap();
                queue.submit([]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let before = sim
                    .active_tiles
                    .stats(device, queue)
                    .unwrap()
                    .completed_substeps;
                let start = Instant::now();
                sim.step_n(&engine, 0.0001, 4).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let status = sim.active_tiles.stats(device, queue).unwrap();
                assert!(!status.overflow);
                assert_eq!(status.completed_substeps.checked_sub(before), Some(4));
                if round >= 2 {
                    times[candidate].push(elapsed);
                }
                outputs[candidate] = Some(read(&engine, &particle_buffers(&sim)));
                println!(
                    "sample mode={mode} fused={} round={round} warmup={} wall_ms={elapsed:.6} simulated_seconds={:.12}",
                    candidate == 1,
                    round < 2,
                    4.0 * f64::from(0.0001_f32)
                );
            }
            let drift = errors(outputs[0].as_ref().unwrap(), outputs[1].as_ref().unwrap());
            println!("parity mode={mode} round={round} max_abs={drift:?}");
            passed &= drift.iter().all(|&e| e <= 2e-5);
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "median mode={mode} separate_ms={:.6} fused_ms={:.6}",
            (times[0][2] + times[0][3]) / 2.0,
            (times[1][2] + times[1][3]) / 2.0
        );
    }
    assert!(passed, "paired full-history 2e-5 gate failed; no adoption");
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn compact_overflow_freezes_fused_history_and_growth_resumes_the_same_mode() {
    let engine = Engine::new_gpu().unwrap();
    let mut sim = MpmSimulation::new_sparse(&engine, fixture(129).config(), 1).unwrap();
    sim.prepare_fused_gather_projection().unwrap();
    sim.set_fused_gather_projection(true).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &fixture(129))
        .unwrap();
    let before = read(&engine, &particle_buffers(&sim));
    sim.step_n(&engine, 0.0001, 2).unwrap();
    let status = sim
        .read_active_grid_status_with(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    assert!(status.overflow);
    assert_eq!(status.completed_substeps, 0);
    assert_eq!(read(&engine, &particle_buffers(&sim)), before);
    sim.reserve_sparse_tiles(128).unwrap();
    assert!(sim.fused_gather_projection_enabled());
    sim.step_n(&engine, 0.0001, 2).unwrap();
    let status = sim
        .read_active_grid_stats_with(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    assert_eq!(status.completed_substeps, 2);
    assert_ne!(read(&engine, &particle_buffers(&sim)), before);
}
