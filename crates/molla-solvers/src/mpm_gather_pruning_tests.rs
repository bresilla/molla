use super::*;
use crate::mpm::replay_tests::read_words;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LegacyReplayUniforms {
    particle_count: u32,
    rigid_count: u32,
    grid_size: u32,
    grid_total: u32,
    grid_origin: [f32; 3],
    cell_size: f32,
    dt: f32,
    gravity_y: f32,
    youngs: f32,
    poisson: f32,
    dp_alpha: f32,
    cohesion: f32,
    boundary_friction: f32,
    inv_cell_size: f32,
    rheology: u32,
    snow_theta_c: f32,
    snow_theta_s: f32,
    snow_hardening: f32,
    grid_layout: [u32; 4],
}

fn replay_uniforms(bytes: &[u8]) -> Result<Uniforms, &'static str> {
    if bytes.len() == std::mem::size_of::<Uniforms>() {
        return Ok(bytemuck::pod_read_unaligned(bytes));
    }
    if bytes.len() != std::mem::size_of::<LegacyReplayUniforms>() {
        return Err("unsupported replay uniform size");
    }
    let old: LegacyReplayUniforms = bytemuck::pod_read_unaligned(bytes);
    Ok(Uniforms {
        particle_count: old.particle_count,
        rigid_count: old.rigid_count,
        grid_size: old.grid_size,
        grid_total: old.grid_total,
        grid_origin: old.grid_origin,
        cell_size: old.cell_size,
        dt: old.dt,
        gravity_y: old.gravity_y,
        youngs: old.youngs,
        poisson: old.poisson,
        dp_alpha: old.dp_alpha,
        cohesion: old.cohesion,
        boundary_friction: old.boundary_friction,
        inv_cell_size: old.inv_cell_size,
        rheology: old.rheology,
        snow_theta_c: old.snow_theta_c,
        snow_theta_s: old.snow_theta_s,
        snow_hardening: old.snow_hardening,
        grid_layout: old.grid_layout,
        grid_window_cells: [0; 4],
    })
}

#[test]
fn gather_replay_uniform_versions_preserve_all_fields() {
    assert_eq!(std::mem::size_of::<LegacyReplayUniforms>(), 96);
    assert_eq!(std::mem::size_of::<Uniforms>(), 112);
    let words: Vec<u32> = (0..24).map(|i| 0x3e80_0000 + i * 7919).collect();
    let old = bytemuck::cast_slice(&words);
    let mut decoded = replay_uniforms(old).unwrap();
    assert_eq!(&bytemuck::bytes_of(&decoded)[..96], old);
    assert_eq!(decoded.grid_window_cells, [0; 4]);
    decoded.grid_window_cells = [-64, 32, -96, 0];
    let current = bytemuck::bytes_of(&decoded);
    assert_eq!(
        bytemuck::bytes_of(&replay_uniforms(current).unwrap()),
        current
    );
    for size in [0, 95, 97, 111, 113] {
        assert!(replay_uniforms(&vec![0; size]).is_err());
    }
}

#[test]
fn gather_cached_base_preserves_scratch_layout() {
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/mpm.wgsl"),
        include_str!("../shaders/mpm_gather_base_cache.wgsl")
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let ty = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("GatherParticle"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("gather record is not a struct")
    };
    assert_eq!(*span, 64);
    assert_eq!(
        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
        vec![0, 48, 52, 56, 60]
    );
    assert_eq!(
        members
            .iter()
            .map(|m| m.name.as_deref().unwrap())
            .collect::<Vec<_>>(),
        vec!["stress", "stiffness", "base_x", "base_y", "base_z"]
    );
}

fn paired_gather(sim: &MpmSimulation, candidate: bool, base_cache: bool) -> Gather {
    if base_cache {
        Gather::new_cached(sim, true, candidate).unwrap()
    } else {
        Gather::new_pruned(sim, candidate).unwrap()
    }
}

#[test]
#[ignore = "requires GPU; cached stencil base preserves exact grid/evolution"]
fn gather_cached_base_exact_grid_and_evolution() {
    exact_grid_and_evolution(true);
}

#[test]
#[ignore = "requires GPU and saved 50k transfer fixture; paired cached-base diagnostic"]
fn gather_cached_base_saved_50k_transfer_pairs() {
    saved_50k_transfer_pairs(true);
}

#[test]
fn gather_pruning_preserves_stencil_support_and_order() {
    let mut checked = 0u64;
    for grid in [5, 7, 35, 64, 128] {
        for cell in 0..grid {
            let lo = (cell - 2).max(0) / 4;
            let hi = (cell + 1).min(grid - 1) / 4;
            assert!(hi - lo <= 1);
            for sample in -64..=(grid + 4) * 16 {
                let x = sample as f32 / 16.0;
                for x in [x.next_down(), x, x.next_up()] {
                    let base = (x - 0.5).floor() as i32;
                    if (0..=2).contains(&(cell - base)) {
                        let key = x.floor().clamp(0.0, (grid - 1) as f32) as i32 / 4;
                        assert!((lo..=hi).contains(&key), "grid={grid} cell={cell} x={x}");
                    }
                    checked += 1;
                }
            }
        }
    }
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let cell = [4 + x, 4 + y, 4 + z];
                let lo = cell.map(|c| (c - 2) / 4);
                let hi = cell.map(|c| (c + 1) / 4);
                let indices: Vec<_> = (0..27)
                    .filter(|&i| {
                        let tile = [i % 3, i / 3 % 3, i / 9];
                        (0..3).all(|a| tile[a] >= lo[a] && tile[a] <= hi[a])
                    })
                    .collect();
                assert!((1..=8).contains(&indices.len()));
                assert!(indices.windows(2).all(|p| p[0] < p[1]));
            }
        }
    }
    println!("pruning support: {checked} sampled positions including adjacent f32 boundary values");
}

#[test]
#[ignore = "requires GPU; pruning preserves exact transfers and later state across all supported layouts"]
fn gather_pruning_exact_grid_and_evolution() {
    exact_grid_and_evolution(false);
}

fn exact_grid_and_evolution(base_cache: bool) {
    exact_grid_variant(base_cache, false);
}

fn paired_variant(sim: &MpmSimulation, candidate: bool, base_cache: bool, masked: bool) -> Gather {
    if masked {
        if candidate {
            Gather::new_masked(sim).unwrap()
        } else {
            Gather::new_pruned(sim, true).unwrap()
        }
    } else {
        paired_gather(sim, candidate, base_cache)
    }
}

pub(super) fn exact_grid_variant(base_cache: bool, masked: bool) {
    exact_grid_policy(base_cache, masked, false);
}

pub(super) fn exact_grid_shared_chunk() {
    exact_grid_policy(false, true, true);
}

fn exact_grid_policy(base_cache: bool, masked: bool, chunk: bool) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let mut sims = [false, true].map(|prune| {
            let mut sim = if mode < 2 {
                tests::seeded(&engine, mode == 1)
            } else {
                dense_tests::seeded(&engine, if mode == 2 { 35 } else { 64 })
            };
            sim.deterministic_p2g = Some(if chunk {
                chunk_tests::gather(&sim, prune)
            } else {
                paired_variant(&sim, prune, base_cache, masked)
            });
            sim
        });
        for step in 0..8 {
            for index in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
                sims[index].step_n_with(device, queue, 0.0001, 1).unwrap();
            }
            assert_eq!(
                sims[0].checkpoint_soil_with(queue).unwrap().to_bytes(),
                sims[1].checkpoint_soil_with(queue).unwrap().to_bytes(),
                "mode={mode} step={step}"
            );
            assert_eq!(
                tests::world_grid(&engine, &sims[0]),
                tests::world_grid(&engine, &sims[1]),
                "grid mode={mode} step={step}"
            );
        }
        println!(
            "gather base_cache={base_cache} masked={masked} chunk={chunk} mode={mode}: 8 exact evolved checkpoints and world-grid fields"
        );
    }
}

#[test]
#[ignore = "requires GPU and saved 50k transfer fixture; paired diagnostic includes sorting"]
fn gather_pruning_saved_50k_transfer_pairs() {
    saved_50k_transfer_pairs(false);
}

fn saved_50k_transfer_pairs(base_cache: bool) {
    saved_50k_variant(base_cache, false);
}

pub(super) fn saved_50k_variant(base_cache: bool, masked: bool) {
    saved_50k_policy(base_cache, masked, false);
}

pub(super) fn saved_50k_shared_chunk() {
    saved_50k_policy(false, true, true);
}

fn saved_50k_policy(base_cache: bool, masked: bool, chunk: bool) {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 50_000)
            .unwrap();
    assert_eq!(checkpoint.particle_count(), 50_000);
    let original = replay_uniforms(&std::fs::read(input.join("uniforms.bin")).unwrap())
        .expect("supported frozen replay uniform layout");
    assert_eq!(original.particle_count, checkpoint.particle_count());
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sims = [false, true].map(|prune| {
        let mut sim = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
        sim.restore_soil_with(queue, &checkpoint).unwrap();
        sim.prepare_parallel_particle_tile_ordering(queue, sim.cfg.max_particles)
            .unwrap();
        sim.set_particle_tile_ordering(true).unwrap();
        sim.deterministic_p2g = Some(if chunk {
            chunk_tests::gather(&sim, prune)
        } else {
            paired_variant(&sim, prune, base_cache, masked)
        });
        sim.set_deterministic_p2g(true).unwrap();
        let mut uniforms = original;
        uniforms.grid_layout = sim.step_uniforms(original.dt).grid_layout;
        queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &sim.material_table,
            0,
            bytemuck::cast_slice(&sim.step_material_table()),
        );
        sim.particle_order.as_ref().unwrap().upload(
            queue,
            &sim.cfg,
            sim.particle_count,
            sim.grid_window_cells,
        );
        sim
    });
    let before: Vec<_> = sims
        .iter_mut()
        .map(|sim| sim.checkpoint_soil_with(queue).unwrap().to_bytes())
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
    let mut baseline = None;
    let mut samples = Vec::new();
    for round in 0..8 {
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let sim = &sims[mode];
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, 0);
            dense_tests::encode_transfer(sim, &mut encoder, true);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &times, 0);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let ticks = read_words(device, queue, &times);
            let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
            let gpu_ms = (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            let grid = [&sim.g_mass, &sim.g_momentum, &sim.g_velocity]
                .map(|b| read_words(device, queue, b));
            if let Some(first) = &baseline {
                assert_eq!(
                    &grid, first,
                    "exact pair/repeat round={round} candidate={mode}"
                );
            } else {
                baseline = Some(grid);
            }
            samples.push(serde_json::json!({"round":round,"candidate":mode==1,"warmup":round<2,"completed_wall_ms":wall_ms,"transfer_gpu_ms":gpu_ms}));
        }
    }
    for (sim, bytes) in sims.iter_mut().zip(before) {
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), bytes);
    }
    let control = if chunk {
        "workgroup-mask gather with original particle loads"
    } else if base_cache || masked {
        "stencil-pruned uncached-base gather"
    } else {
        "27-range fixed-order gather"
    };
    let candidate = if chunk {
        "workgroup-mask gather with shared immutable particle chunks"
    } else if masked {
        "workgroup integer support masks with stable contributing-particle traversal"
    } else if base_cache {
        "stencil-pruned cached-base gather"
    } else {
        "stencil-pruned fixed-order gather"
    };
    let report = serde_json::json!({"schema":1,"particles":50_000,"scope":"fixed-state ordering/reset/P2G/grid update; zero simulated time, no machine solve","control":control,"candidate":candidate,"exact_pairs_and_repeats":true,"particle_history_unchanged":true,"samples":samples,"full_acceptance":false});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("paired pruning transfer: {report}");
}
