use super::*;

#[test]
#[ignore = "requires GPU; identical-particle transfer/contact term isolation against archived failure"]
fn transfer_contact_terms_50k_diagnostic() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_TERMS_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_TERMS_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let frozen = std::fs::read(input.join("soil.bin")).unwrap();
    let checkpoint = MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut captures = Vec::new();
    let mut grids = Vec::new();
    let mut wrenches = Vec::new();
    let mut sums = [[[0_f64; 8]; 2]; 2];
    for arm in 0..2 {
        let mut sim = tests::seeded_benchmark(&engine);
        sim.restore_soil_with(queue, &checkpoint).unwrap();
        if arm == 0 {
            sim.prepare_deterministic_p2g_with_compensated_stiffness(3_200_000)
                .unwrap();
            sim.set_deterministic_p2g(true).unwrap();
        } else {
            sim.set_p2g_mode(true);
            sim.set_particle_tile_ordering(false).unwrap();
            sim.prepare_compensated_grid_transfer(sim.g_mass.size() * 5)
                .unwrap();
            sim.set_compensated_grid_transfer(true).unwrap();
        }
        sim.set_rigid_spheres(queue, &[([-0.15, 0.52, -0.15], 0.12, [0.01, -0.005, 0.0])])
            .unwrap();
        queue.write_buffer(
            &sim.uniform_buf,
            0,
            bytemuck::bytes_of(&sim.step_uniforms(0.0001)),
        );
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tail = buffer("captured reaction residual", sim.rigid_wrench.size());
        let terms = buffer(
            "per-world-node imprint/shell terms",
            u64::from(sim.cfg.grid_size).pow(3) * 16 * 4,
        );
        let rigid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &mixed_storage_entries(&[true, true, false, false, false]),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rigid,
            entries: &[
                buffer_entry(0, &sim.rigid_center),
                buffer_entry(1, &sim.rigid_vel),
                buffer_entry(2, &sim.rigid_wrench),
                buffer_entry(3, &tail),
                buffer_entry(4, &terms),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&sim.layout_uniforms),
                Some(&sim.layout_particles),
                Some(&sim.layout_grid),
                Some(&rigid),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source(true).into()),
        });
        let pipelines = [
            "mpm_rigid_wrench_reset",
            "rigid_imprint",
            "mpm_rigid_force_integrate",
            "wrench_resolve",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SPARSE_GRID", 1.0), ("ACTIVE_GRID", 1.0)],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        sim.initialize_active_grid(&mut encoder, false);
        sim.encode_particle_order(&mut encoder, false, 0);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.active_pipelines[0]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        }
        sim.active_tiles
            .encode_build(&mut encoder, sim.particle_count, sim.rigid_count);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            sim.encode_p2g(&mut pass);
        }
        queue.submit([encoder.finish()]);
        let before = tests::world_grid(&engine, &sim);
        for (field, words) in before.iter().enumerate() {
            std::fs::write(
                output.join(format!("pre-grid-{arm}-{field}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
        grids.push(before);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&terms, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_bind_group(3, &bindings, &[]);
            pass.set_pipeline(&pipelines[0]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines[1]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            pass.set_bind_group(3, &sim.bg_rigid, &[]);
            pass.set_pipeline(&sim.active_pipelines[2]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            pass.set_bind_group(3, &bindings, &[]);
            pass.set_pipeline(&pipelines[2]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            pass.set_pipeline(&pipelines[3]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let values = read_words(device, queue, &terms);
        for node in values.chunks_exact(16) {
            for stage in 0..2 {
                for k in 0..8 {
                    let value = f64::from(f32::from_bits(node[stage * 8 + k]));
                    assert!(value.is_finite());
                    sums[arm][stage][k] += value;
                }
            }
        }
        let wrench = read_words(device, queue, &sim.rigid_wrench);
        assert!(wrench[..8].iter().any(|v| f32::from_bits(*v) != 0.0));
        std::fs::write(
            output.join(format!("terms-{arm}.bin")),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        std::fs::write(
            output.join(format!("wrench-{arm}.bin")),
            bytemuck::cast_slice(&wrench),
        )
        .unwrap();
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
        captures.push(values);
        wrenches.push(wrench);
    }
    let mut grid_max = [0_f64; 3];
    let mut grid_changed = [0_usize; 3];
    for field in 0..3 {
        for (&a, &b) in grids[0][field].iter().zip(&grids[1][field]) {
            grid_changed[field] += usize::from(a != b);
            let a = f64::from(f32::from_bits(a));
            let b = f64::from(f32::from_bits(b));
            assert!(a.is_finite() && b.is_finite());
            grid_max[field] = grid_max[field].max((a - b).abs());
        }
    }
    let mut term_changed = [[0_usize; 8]; 2];
    let mut term_max = [[0_f64; 8]; 2];
    let mut term_l1 = [[0_f64; 8]; 2];
    for (a, b) in captures[0]
        .chunks_exact(16)
        .zip(captures[1].chunks_exact(16))
    {
        for stage in 0..2 {
            for k in 0..8 {
                let i = stage * 8 + k;
                term_changed[stage][k] += usize::from(a[i] != b[i]);
                let difference =
                    (f64::from(f32::from_bits(a[i])) - f64::from(f32::from_bits(b[i]))).abs();
                term_max[stage][k] = term_max[stage][k].max(difference);
                term_l1[stage][k] += difference;
            }
        }
    }
    let total: [[f64; 8]; 2] =
        std::array::from_fn(|arm| std::array::from_fn(|k| sums[arm][0][k] + sums[arm][1][k]));
    let sum_difference: [f64; 8] = std::array::from_fn(|k| (total[0][k] - total[1][k]).abs());
    let mut reduction_error = [0_f64; 2];
    let mut cross_error = 0_f64;
    for k in 0..8 {
        for arm in 0..2 {
            reduction_error[arm] = reduction_error[arm]
                .max((f64::from(f32::from_bits(wrenches[arm][k])) - total[arm][k]).abs());
        }
        cross_error = cross_error.max(
            (f64::from(f32::from_bits(wrenches[0][k])) - f64::from(f32::from_bits(wrenches[1][k])))
                .abs(),
        );
    }
    let archive_exact = [0, 1].map(|arm| {
        std::fs::read(input.join(format!("wrench-0-{arm}.bin"))).unwrap()
            == bytemuck::cast_slice::<u32, u8>(&wrenches[arm])
    });
    let report = serde_json::json!({"particles":50_000,"simulated_seconds":0,"bound":2e-5,"same_particle_input":true,"particle_checkpoints_exact":true,"grid_changed_words":grid_changed,"grid_max_errors":grid_max,"term_changed_words":term_changed,"term_max_errors":term_max,"term_l1_difference":term_l1,"imprint_shell_sums":sums,"total_sums":total,"sum_difference":sum_difference,"reduction_errors":reduction_error,"cross_force_error":cross_error,"cross_force_passed":cross_error<=2e-5,"archived_first_step_wrenches_exact":archive_exact,"scope":"two transfer methods from identical frozen50k particles; captured first-step contact contributions and f64 sums; diagnostic, not trajectory or performance acceptance"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "transfer contact grid_errors={grid_max:?} reduction_errors={reduction_error:?} cross_force_error={cross_error} f64_sum_difference={sum_difference:?} archive_exact={archive_exact:?}"
    );
    assert!(archive_exact.into_iter().all(|v| v));
    assert!(reduction_error.into_iter().all(|v| v <= 2e-5));
}
