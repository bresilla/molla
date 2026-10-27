use super::*;

#[path = "mpm_transfer_contact_terms_tests.rs"]
mod transfer_terms;

fn source(compensated: bool) -> String {
    let mut source = if compensated {
        compensated_wrench::source().unwrap()
    } else {
        format!(
            "{}{}",
            include_str!("../shaders/mpm.wgsl"),
            include_str!("../shaders/mpm_compensated_wrench.wgsl")
        )
    };
    for (entry, stage) in [("rigid_imprint", 0), ("mpm_rigid_force_integrate", 1)] {
        let signature = format!("fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{");
        assert_eq!(source.matches(&signature).count(), 1);
        source = source.replace(&signature, &format!("{signature}\n    wrench_term_base=(2u*grid_cell_id(gid)+{stage}u)*u.rigid_count*8u;"));
    }
    let signature = "fn atomic_add_rigid_wrench(idx: u32, val: f32) {";
    assert_eq!(source.matches(signature).count(), 1);
    source = source.replace(
        signature,
        &format!("{signature}\n    wrench_terms[wrench_term_base+idx]=val;"),
    );
    source.push_str("\n@group(3) @binding(4) var<storage,read_write> wrench_terms:array<f32>;\nvar<private> wrench_term_base:u32;\n");
    source
}
#[test]
fn wrench_compensation_probe_source_contract() {
    for compensated in [false, true] {
        let module = naga::front::wgsl::parse_str(&source(compensated)).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

fn grid(sim: &MpmSimulation) -> [&wgpu::Buffer; 6] {
    [
        &sim.g_mass,
        &sim.g_momentum,
        &sim.g_velocity,
        &sim.g_rigid_mask,
        &sim.g_kdiag,
        &sim.g_velocity_prev,
    ]
}

#[test]
#[ignore = "requires GPU; frozen50k contact terms and compensated rigid reduction"]
fn wrench_compensation_fixed_50k_terms() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_WRENCH_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_WRENCH_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let frozen = std::fs::read(input).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = tests::seeded_benchmark(&engine);
    sim.restore_soil_with(
        queue,
        &MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap(),
    )
    .unwrap();
    sim.set_p2g_mode(true);
    sim.set_particle_tile_ordering(false).unwrap();
    sim.prepare_compensated_grid_transfer(sim.g_mass.size() * 5)
        .unwrap();
    sim.set_compensated_grid_transfer(true).unwrap();
    sim.set_rigid_spheres(queue, &[([-0.15, 0.52, -0.15], 0.12, [0.01, -0.005, 0.0])])
        .unwrap();
    queue.write_buffer(
        &sim.uniform_buf,
        0,
        bytemuck::bytes_of(&sim.step_uniforms(0.0001)),
    );
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let saved = grid(&sim).map(|b| buffer("fixed pre-imprint grid", b.size(), copy));
    let prefix = buffer("imprint wrench", sim.rigid_wrench.size(), copy);
    let tail = buffer(
        "diagnostic wrench residual",
        sim.rigid_wrench.size(),
        copy | wgpu::BufferUsages::STORAGE,
    );
    let terms = buffer(
        "per-node imprint/shell terms",
        u64::from(sim.cfg.grid_size).pow(3) * 16 * 4,
        copy | wgpu::BufferUsages::STORAGE,
    );
    let rigid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &mixed_storage_entries(&[true, true, false, false, false]),
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &rigid_layout,
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
            Some(&rigid_layout),
        ],
        immediate_size: 0,
    });
    let pipelines = [false, true].map(|compensated| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source(compensated).into()),
        });
        [
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
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    sim.initialize_active_grid(&mut encoder, false);
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
    for (from, to) in grid(&sim).into_iter().zip(&saved) {
        encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
    }
    queue.submit([encoder.finish()]);
    let mut first_terms = None;
    let mut first_grid = None;
    let mut first_wrench: [Option<Vec<u32>>; 2] = Default::default();
    let mut first_prefix: [Option<Vec<u32>>; 2] = Default::default();
    let mut sums = [[0_f64; 8]; 2];
    let mut maxima = [[0_f64; 2]; 2];
    let mut repeat = [[0_f64; 2]; 2];
    let mut records = Vec::new();
    for round in 0..6 {
        for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let mut encoder = device.create_command_encoder(&Default::default());
            for (from, to) in saved.iter().zip(grid(&sim)) {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
            }
            encoder.clear_buffer(&tail, 0, None);
            encoder.clear_buffer(&terms, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_bind_group(3, &bindings, &[]);
                pass.set_pipeline(&pipelines[arm][0]);
                pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            }
            encoder.copy_buffer_to_buffer(&sim.rigid_wrench, 0, &prefix, 0, prefix.size());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.active_pipelines[2]);
                pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
                pass.set_bind_group(3, &bindings, &[]);
                pass.set_pipeline(&pipelines[arm][1]);
                pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
                if arm == 1 {
                    pass.set_pipeline(&pipelines[arm][2]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            queue.submit([encoder.finish()]);
            let values = read_words(device, queue, &terms);
            if let Some(first) = &first_terms {
                assert_eq!(&values, first, "per-node terms changed");
            } else {
                for node in values.chunks_exact(16) {
                    for stage in 0..2 {
                        for k in 0..8 {
                            let value = f64::from(f32::from_bits(node[stage * 8 + k]));
                            assert!(value.is_finite());
                            sums[stage][k] += value;
                        }
                    }
                }
                std::fs::write(output.join("terms.bin"), bytemuck::cast_slice(&values)).unwrap();
                first_terms = Some(values);
            }
            let state = grid(&sim).map(|b| read_words(device, queue, b));
            if let Some(first) = &first_grid {
                assert_eq!(&state, first);
            } else {
                first_grid = Some(state);
            }
            let wrench = read_words(device, queue, &sim.rigid_wrench);
            let imprint = read_words(device, queue, &prefix);
            let initial = first_wrench[arm].get_or_insert_with(|| wrench.clone());
            let initial_prefix = first_prefix[arm].get_or_insert_with(|| imprint.clone());
            for k in 0..8 {
                let total = f64::from(f32::from_bits(wrench[k]));
                let part = f64::from(f32::from_bits(imprint[k]));
                assert!(total.is_finite() && part.is_finite());
                maxima[arm][0] = maxima[arm][0].max((part - sums[0][k]).abs());
                maxima[arm][1] = maxima[arm][1].max((total - sums[0][k] - sums[1][k]).abs());
                repeat[arm][0] =
                    repeat[arm][0].max((part - f64::from(f32::from_bits(initial_prefix[k]))).abs());
                repeat[arm][1] =
                    repeat[arm][1].max((total - f64::from(f32::from_bits(initial[k]))).abs());
            }
            std::fs::write(
                output.join(format!("wrench-{round}-{arm}.bin")),
                bytemuck::cast_slice(&wrench),
            )
            .unwrap();
            records.push(serde_json::json!({"round":round,"arm":arm,"imprint":imprint[..8],"wrench":wrench[..8]}));
        }
    }
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    assert!(sums.iter().flatten().any(|v| v.abs() > 0.0));
    let passed = maxima[1][1] <= 2e-5 && repeat[1][1] == 0.0;
    let report = serde_json::json!({"records":records,"sum_imprint":sums[0],"sum_shell":sums[1],"max_f64_errors_imprint_total":maxima,"max_repeat_errors_imprint_total":repeat,"candidate_passed":passed,"bound":2e-5,"terms_identical":true,"grid_identical":true,"particle_checkpoint_exact":true,"extra_tail_bytes":tail.size(),"diagnostic_term_bytes":terms.size(),"scope":"fixed50k grid, captured per-node force terms and test-only compensated reduction; no evolution, performance or runtime acceptance"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("wrench f64_errors={maxima:?} repeat_errors={repeat:?} passed={passed}");
    assert!(passed);
}
