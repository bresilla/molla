use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
fn dense_gather_dispatch_and_shader_contract() {
    for grid in [1u32, 3, 4, 5, 7, 35, 64] {
        let [x, y] = dense_groups(grid, 256).unwrap();
        let width = grid.div_ceil(4);
        let mut cells = vec![0u8; grid.pow(3) as usize];
        for tile in 0..x * y {
            if tile >= width.pow(3) {
                continue;
            }
            for lane in 0..64 {
                let xyz = [
                    tile % width * 4 + lane % 4,
                    tile / width % width * 4 + lane / 4 % 4,
                    tile / (width * width) * 4 + lane / 16,
                ];
                if xyz.iter().any(|&c| c >= grid) {
                    continue;
                }
                cells[(xyz[0] + grid * (xyz[1] + grid * xyz[2])) as usize] += 1;
            }
        }
        assert!(cells.iter().all(|&n| n == 1));
    }
    for (grid, limit) in [(0, 256), (64, 0), (64, 16), (u32::MAX, 256)] {
        assert!(dense_groups(grid, limit).is_err());
    }
    for native in [false, true] {
        let base = if native {
            native_atomics::shader_source()
        } else {
            include_str!("../shaders/mpm.wgsl").to_owned()
        };
        let source = format!(
            "{base}\n{}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl")
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
#[ignore = "requires native GPU; exact dense/active/compact gather and mode lifecycle"]
fn native_gather_layouts_and_reconfiguration() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 129;
    cfg.youngs = 1000.0;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    cfg.time_step = MpmTimeStep::Explicit;
    let mut sims: Vec<_> = (0..6)
        .map(|mode| {
            let mut sim = if mode / 2 == 2 {
                MpmSimulation::new_native_sparse(&engine, cfg.clone(), 128).unwrap()
            } else {
                MpmSimulation::new_native_explicit(&engine, cfg.clone()).unwrap()
            };
            if mode / 2 == 1 {
                sim.set_active_grid(true).unwrap();
            }
            sim.prepare_parallel_particle_tile_ordering(queue, 129)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
            sim.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
            sim.set_deterministic_p2g(true).unwrap();
            sim
        })
        .collect();
    let points: Vec<_> = (0..129)
        .map(|i| {
            [
                0.9 + (i % 5) as f32 * 0.031,
                1.1 + (i / 5 % 5) as f32 * 0.031,
                0.9 + (i / 25) as f32 * 0.031,
            ]
        })
        .collect();
    let mut comparisons = 0;
    for count in [129, 1, 0, 129] {
        for sim in &mut sims {
            sim.set_particles_with_velocity(
                queue,
                &points[..count],
                &vec![[0.01, -0.02, 0.003]; count],
            )
            .unwrap();
            sim.step(&engine, 0.0001).unwrap();
        }
        for _ in 0..8 {
            for sim in &mut sims {
                let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                sim.step(&engine, 0.0001).unwrap();
                assert_eq!(scope.finish().observations, 0);
                assert!(
                    read_words(device, queue, &sim.g_kdiag)
                        [..(sim.active_tiles.metadata_offset / 4) as usize]
                        .iter()
                        .all(|&word| word == 0)
                );
                if sim.active_grid_enabled() {
                    assert!(
                        !sim.read_active_grid_status_with(device, queue)
                            .unwrap()
                            .faulted()
                    );
                }
            }
            let expected = sims[0].checkpoint_soil_with(queue).unwrap().to_bytes();
            for sim in &mut sims[1..] {
                assert_eq!(
                    sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
                    expected
                );
                comparisons += 1;
            }
        }
    }
    let sim = &mut sims[0];
    assert!(sim.set_active_grid(true).is_err());
    sim.set_deterministic_p2g(false).unwrap();
    sim.set_active_grid(true).unwrap();
    assert!(sim.set_deterministic_p2g(true).is_err());
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        sim.prepare_deterministic_p2g_with_budget(129 * 64 - 1)
            .is_err()
    );
    assert_eq!(scope.finish().observations, 0);
    sim.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim.set_active_grid(true).unwrap();
    assert!(sim.set_active_grid(false).is_err());
    sim.step(&engine, 0.0001).unwrap();
    sim.set_deterministic_p2g(false).unwrap();
    sim.set_active_grid(false).unwrap();
    assert!(sim.set_deterministic_p2g(true).is_err());
    sim.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim.step(&engine, 0.0001).unwrap();
    println!(
        "native gather: {comparisons} exact full-history comparisons; zero explicit stiffness/hot buffers; dense-active-dense cache guards pass"
    );
}

fn prepare(sim: &mut MpmSimulation, queue: &wgpu::Queue) {
    sim.prepare_parallel_particle_tile_ordering(queue, sim.cfg.max_particles)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_deterministic_p2g().unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    assert!(!sim.active_grid_enabled());
}

pub(super) fn seeded(engine: &Engine, grid: u32) -> MpmSimulation {
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = grid;
    cfg.max_particles = 129;
    cfg.youngs = 1000.;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    cfg.time_step = MpmTimeStep::Explicit;
    let points: Vec<_> = (0..129)
        .map(|i| {
            [
                cfg.grid_origin[0] + cfg.cell_size * (3.1 + (i % 7) as f32 * 0.42),
                cfg.grid_origin[1] + cfg.cell_size * (3.2 + (i / 7 % 7) as f32 * 0.42),
                cfg.grid_origin[2] + cfg.cell_size * (3.3 + (i / 49) as f32 * 0.42),
            ]
        })
        .collect();
    let mut sim = MpmSimulation::new_native_explicit(engine, cfg).unwrap();
    sim.set_particles_with_velocity(queue, &points, &vec![[0.02, -0.01, 0.005]; 129])
        .unwrap();
    prepare(&mut sim, queue);
    sim
}

#[test]
#[ignore = "requires native GPU and resource counters"]
fn dense_gather_native_lifecycle_and_schedules() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for grid in [35, 64] {
        let mut a = seeded(&engine, grid);
        let mut b = seeded(&engine, grid);
        assert!(a.set_active_grid(true).is_err());
        a.set_active_grid(false).unwrap();
        assert!(a.set_particle_tile_ordering(false).is_err());
        assert!(a.prepare_particle_tile_ordering(queue, 129).is_err());
        let stale = a.prepare_step(0.0001, 1).unwrap();
        a.set_deterministic_p2g(false).unwrap();
        {
            let mut batch =
                a.begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        a.set_deterministic_p2g(true).unwrap();
        let steps = [(0.0001, 2), (0.0002, 2), (0.00008, 1)];
        let records: Vec<_> = steps
            .iter()
            .map(|&(dt, n)| b.prepare_step(dt, n).unwrap())
            .collect();
        for &(dt, n) in &steps {
            a.step_n_with(device, queue, dt, n).unwrap();
        }
        let encoder = device.create_command_encoder(&Default::default());
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut batch = b.begin_command_batch(encoder);
        for record in &records {
            batch.encode_step(record).unwrap();
        }
        batch.submit(queue);
        let allocation = allocations.finish();
        let transfer = traffic.finish();
        assert!(!allocation.invalid && !transfer.overflowed);
        assert_eq!(
            (
                allocation.observations,
                transfer.upload_bytes,
                transfer.readback_bytes
            ),
            (0, 0, 0)
        );
        for (left, right) in [
            (&a.p_pos, &b.p_pos),
            (&a.p_vel, &b.p_vel),
            (&a.p_c, &b.p_c),
            (&a.p_f, &b.p_f),
            (&a.p_jp, &b.p_jp),
        ] {
            assert_eq!(
                read_words(device, queue, left),
                read_words(device, queue, right)
            );
        }
        assert!(
            read_words(device, queue, &b.g_kdiag)[..grid.pow(3) as usize]
                .iter()
                .all(|&w| w == 0)
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        b.encode_particle_order(&mut encoder, false, 0);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            b.bind_step(&mut pass);
            let scope = molla_core::gpu_metrics::DispatchScope::start().unwrap();
            b.deterministic_p2g.as_ref().unwrap().encode(&mut pass, &b);
            let counts = scope.finish();
            assert!(!counts.overflowed);
            assert_eq!(counts.encoded_dispatches, 2);
        }
        queue.submit([encoder.finish()]);
        let before = b.checkpoint_soil_with(queue).unwrap().to_bytes();
        b.set_particles(queue, &[]).unwrap();
        b.step_n_with(device, queue, 0.0001, 1).unwrap();
        assert_eq!(b.particle_count(), 0);
        let restore = MpmSoilCheckpoint::from_bytes(&before, 129).unwrap();
        b.restore_soil_with(queue, &restore).unwrap();
        assert_eq!(b.particle_count(), 129);
        assert!(b.deterministic_p2g_enabled() && b.particle_tile_ordering_enabled());
        b.step_n_with(device, queue, 0.0001, 1).unwrap();
        println!(
            "dense-gather grid={grid} five substeps exact across direct/prepared schedules; guards, empty, restore and zero hot buffers/transfers pass"
        );
    }
}

pub(super) fn encode_transfer(
    sim: &MpmSimulation,
    encoder: &mut wgpu::CommandEncoder,
    gather: bool,
) {
    if gather {
        sim.particle_order.as_ref().unwrap().encode(
            encoder,
            &sim.p_material,
            u64::from(sim.cfg.max_particles.max(1)) * 4,
        );
    }
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    pass.set_pipeline(&sim.pipe_reset);
    pass.dispatch_workgroups(sim.cfg.grid_size.pow(3).div_ceil(64), 1, 1);
    if gather {
        sim.deterministic_p2g
            .as_ref()
            .unwrap()
            .encode(&mut pass, sim);
    } else {
        pass.set_pipeline(&sim.pipe_p2g);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
    }
    pass.set_pipeline(&sim.pipe_grid);
    pass.dispatch_workgroups(sim.cfg.grid_size.pow(3).div_ceil(64), 1, 1);
}

#[test]
#[ignore = "requires native GPU"]
fn dense_gather_native_two_dimensional_dispatch() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for grid in [35, 64] {
        let mut sims = [seeded(&engine, grid), seeded(&engine, grid)];
        let split = dense_groups(grid, 128).unwrap();
        assert!(split[1] > 1);
        sims[1].deterministic_p2g.as_mut().unwrap().dense_groups = Some(split);
        for sim in &mut sims {
            let point = |offset| {
                sim.cfg
                    .grid_origin
                    .map(|origin| origin + sim.cfg.cell_size * offset)
            };
            let mut points = vec![point(4.2); 129];
            points[0] = point(0.2);
            points[1] = point(grid as f32 - 1.1);
            sim.set_particles(queue, &points).unwrap();
            sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        }
        for (left, right) in [
            (&sims[0].g_mass, &sims[1].g_mass),
            (&sims[0].g_momentum, &sims[1].g_momentum),
            (&sims[0].g_velocity, &sims[1].g_velocity),
            (&sims[0].g_kdiag, &sims[1].g_kdiag),
            (&sims[0].p_pos, &sims[1].p_pos),
            (&sims[0].p_vel, &sims[1].p_vel),
            (&sims[0].p_c, &sims[1].p_c),
            (&sims[0].p_f, &sims[1].p_f),
            (&sims[0].p_jp, &sims[1].p_jp),
        ] {
            assert_eq!(
                read_words(device, queue, left),
                read_words(device, queue, right)
            );
        }
        let mass = read_words(device, queue, &sims[1].g_mass);
        assert!(f32::from_bits(mass[0]) > 0.0);
        assert!(f32::from_bits(mass[grid.pow(3) as usize - 1]) > 0.0);
        println!(
            "dense-gather grid={grid} groups={split:?} exact 1D/2D grid/history with loaded low/high edge cells"
        );
    }
}

#[test]
#[ignore = "requires native GPU and saved SOIL_NATIVE_REPLAY_INPUT/OUTPUT"]
fn dense_gather_native_evolved_transfer_pair() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 1_000_000)
            .unwrap();
    let original: Uniforms =
        bytemuck::pod_read_unaligned(&std::fs::read(input.join("uniforms.bin")).unwrap());
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sims = [false, true].map(|gather| {
        let mut sim = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
        sim.restore_soil_with(queue, &checkpoint).unwrap();
        if gather {
            prepare(&mut sim, queue);
        }
        let mut uniforms = original;
        uniforms.grid_layout = sim.step_uniforms(original.dt).grid_layout;
        assert_eq!(uniforms.particle_count, sim.particle_count);
        assert_eq!(uniforms.grid_total, sim.cfg.grid_size.pow(3));
        queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &sim.material_table,
            0,
            bytemuck::cast_slice(&sim.step_material_table()),
        );
        if let Some(order) = &sim.particle_order {
            order.upload(queue, &sim.cfg, sim.particle_count, sim.grid_window_cells);
        }
        sim
    });
    let before: Vec<_> = sims
        .iter_mut()
        .map(|s| s.checkpoint_soil_with(queue).unwrap().to_bytes())
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
    let mut baselines: [Option<Vec<Vec<u32>>>; 2] = [None, None];
    let mut variation = [[0f64; 3]; 2];
    let mut changed = [[0usize; 3]; 2];
    let mut errors = [0f64; 3];
    let mut support = 0;
    let mut samples = Vec::new();
    for round in 0..8 {
        let mut pair: [Vec<Vec<u32>>; 2] = Default::default();
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let sim = &sims[mode];
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, 0);
            encode_transfer(sim, &mut encoder, mode == 1);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &times, 0);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let ticks = read_words(device, queue, &times);
            let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
            let gpu_ms = (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            pair[mode] = [&sim.g_mass, &sim.g_momentum, &sim.g_velocity]
                .map(|b| read_words(device, queue, b))
                .to_vec();
            for (field, words) in pair[mode].iter().enumerate() {
                for (i, &word) in words.iter().enumerate() {
                    let value = f64::from(f32::from_bits(word));
                    assert!(value.is_finite());
                    if field == 0 {
                        assert!(value >= 0.0);
                    }
                    if let Some(first) = &baselines[mode] {
                        if field != 2 || i % 4 != 3 {
                            variation[mode][field] = variation[mode][field]
                                .max((value - f64::from(f32::from_bits(first[field][i]))).abs());
                            changed[mode][field] += usize::from(word != first[field][i]);
                        }
                    }
                }
            }
            if baselines[mode].is_none() {
                baselines[mode] = Some(pair[mode].clone());
            }
            samples.push(serde_json::json!({"round":round,"candidate":mode==1,"warmup":round<2,"completed_wall_ms":wall_ms,"free_grid_gpu_ms":gpu_ms}));
        }
        for field in 0..3 {
            for (i, (&a, &b)) in pair[0][field].iter().zip(&pair[1][field]).enumerate() {
                let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                if field != 2 || i % 4 != 3 {
                    errors[field] = errors[field].max((f64::from(a) - f64::from(b)).abs());
                }
                if field == 0 {
                    support += usize::from((a > 0.0) != (b > 0.0));
                }
            }
        }
    }
    for (sim, bytes) in sims.iter_mut().zip(before) {
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), bytes);
    }
    let passed = support == 0 && errors.iter().all(|&e| e <= 2e-5) && changed[1] == [0; 3];
    let report = serde_json::json!({"schema":1,"scope":"reset/P2G/grid update on saved 50k state; candidate includes key generation and stable sort; no simulated time advanced","control":"native unordered scatter","candidate":"dense fixed-order gather","particles":sims[0].particle_count,"samples":samples,"max_reference_candidate_errors_mass_momentum_velocity_xyz":errors,"max_repeat_variation_mass_momentum_velocity_xyz":variation,"changed_repeat_words_mass_momentum_velocity_xyz":changed,"positive_mass_support_mismatches":support,"particle_history_unchanged":true,"component_parity_passed":passed,"extra_gather_bytes":sims[1].deterministic_p2g_storage_bytes(),"extra_ordering_bytes":sims[1].particle_tile_ordering_storage_bytes(),"production_accepted":false});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("dense native gather: {report}");
    assert!(passed, "unchanged 2e-5/support and exact repeat gates");
}
