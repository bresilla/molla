use super::*;

const RESIDUAL: &str = include_str!("../shaders/mpm_atomic_residual.wgsl");

fn fast_residual_source() -> String {
    format!(
        "{}\n{}",
        RESIDUAL.replace("fn native_sum_residual(", "fn native_sum_residual_f64("),
        include_str!("../shaders/mpm_atomic_residual_fast.wgsl")
    )
}

pub(super) fn shader_source() -> String {
    shader_source_with(RESIDUAL)
}

fn shader_source_with(residual: &str) -> String {
    let mut source = native_atomics::shader_source();
    for (old, new) in [
        (
            "atomicAdd(&g_mass[idx], val);",
            "let old=atomicAdd(&g_mass[idx],val); atomicAdd(&transfer_error[4u*idx],native_sum_residual(old,val));",
        ),
        (
            "atomicAdd(&g_momentum[idx], val);",
            "let old=atomicAdd(&g_momentum[idx],val); atomicAdd(&transfer_error[4u*(idx/3u)+1u+idx%3u],native_sum_residual(old,val));",
        ),
        (
            "    g_rigid_mask[idx] = 0u;",
            "    for(var component=0u;component<4u;component++) { atomicStore(&transfer_error[4u*idx+component],0.0); }\n    g_rigid_mask[idx] = 0u;",
        ),
    ] {
        assert_eq!(source.matches(old).count(), 1);
        source = source.replace(old, new);
    }
    let start = source.find("fn grid_update(").unwrap();
    let suffix = &source[start..];
    let anchor = "    let m_bits = atomicLoad(&g_mass[cidx]);";
    assert_eq!(suffix.matches(anchor).count(), 1);
    let position = start + suffix.find(anchor).unwrap();
    source.insert_str(position, "    atomicStore(&g_mass[cidx],fma(atomicLoad(&transfer_error[4u*cidx]),5.960464477539063e-8,atomicLoad(&g_mass[cidx])));\n    for(var component=0u;component<3u;component++) {\n        let index=3u*cidx+component;\n        atomicStore(&g_momentum[index],fma(atomicLoad(&transfer_error[4u*cidx+1u+component]),5.960464477539063e-8,atomicLoad(&g_momentum[index])));\n    }\n");
    format!(
        "{source}\n{residual}\n@group(2) @binding(6) var<storage,read_write> transfer_error:array<atomic<f32>>;\n"
    )
}

struct Compensated {
    error: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
}

impl Compensated {
    fn new(sim: &MpmSimulation) -> Self {
        Self::new_with_residual(sim, RESIDUAL)
    }
    fn new_with_residual(sim: &MpmSimulation, residual: &str) -> Self {
        assert!(sim.uses_native_float_atomics());
        assert!(!sim.active_grid_enabled());
        assert!(!sim.use_particle_order);
        let device = &sim.device;
        let bytes = u64::from(sim.cfg.grid_size).pow(3) * 16;
        assert!(bytes <= device.limits().max_storage_buffer_binding_size);
        assert!(bytes <= device.limits().max_buffer_size);
        let error = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test native P2G rounding residuals"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &storage_entries(7, false),
        });
        let uniforms = sim.pipe_reset.get_bind_group_layout(0);
        let particles = sim.pipe_reset.get_bind_group_layout(1);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&uniforms),
                Some(&particles),
                Some(&layout),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test compensated native P2G"),
            source: wgpu::ShaderSource::Wgsl(shader_source_with(residual).into()),
        });
        let pipelines = ["reset_grid", "p2g", "grid_update"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                buffer_entry(0, &sim.g_mass),
                buffer_entry(1, &sim.g_momentum),
                buffer_entry(2, &sim.g_velocity),
                buffer_entry(3, &sim.g_rigid_mask),
                buffer_entry(4, &sim.g_kdiag),
                buffer_entry(5, &sim.g_velocity_prev),
                buffer_entry(6, &error),
            ],
        });
        Self {
            error,
            bindings,
            pipelines,
        }
    }
}

#[test]
fn compensated_source_keeps_native_scatter_and_explicit_scope() {
    let source = shader_source();
    assert_eq!(source.matches("native_sum_residual(old,val)").count(), 2);
    assert_eq!(source.matches("@group(2) @binding(6)").count(), 1);
    let start = source.find("fn p2g(@builtin").unwrap();
    let end = start + source[start..].find("\n}\n").unwrap();
    let original = native_atomics::shader_source();
    let old_start = original.find("fn p2g(@builtin").unwrap();
    let old_end = old_start + original[old_start..].find("\n}\n").unwrap();
    assert_eq!(&source[start..end], &original[old_start..old_end]);
    let fast = fast_residual_source();
    assert_eq!(shader_source_with(&fast).replace(&fast, RESIDUAL), source);
    assert!(fast.contains("return native_sum_residual_f64(a,b);"));
}

#[test]
#[ignore = "requires GPU"]
fn native_residual_matches_independent_f64_reference() {
    residual_cases(RESIDUAL);
}

#[test]
#[ignore = "requires GPU f64; bounded f32 common range and unchanged f64 fallback"]
fn native_fast_residual_matches_independent_f64_reference() {
    residual_cases(&fast_residual_source());
}

fn residual_cases(residual: &str) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut pairs = vec![
        [1e20f32, 1.0],
        [-1e20, 1.0],
        [1.0, -1e20],
        [1.0, 1e-20],
        [16777216.0, 1.0],
        [1.0, -1.0],
        [0.0, 1e-30],
    ];
    let mut seed = 0x5138ac17u32;
    for _ in 0..4097 {
        let pair = std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            f32::from_bits((seed & 0x807fffff) | ((1 + ((seed >> 23) % 250)) << 23))
        });
        if (pair[0] + pair[1]).is_finite() {
            pairs.push(pair);
        }
    }
    for _ in 0..8192 {
        pairs.push(std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            f32::from_bits((seed & 0x807fffff) | ((67 + ((seed >> 23) % 140)) << 23))
        }));
    }
    for boundary in [2f32.powi(-60), 2f32.powi(80)] {
        for a in [
            f32::from_bits(boundary.to_bits() - 1),
            boundary,
            f32::from_bits(boundary.to_bits() + 1),
        ] {
            for b in [0.0, -0.0, a, -a, a * 2f32.powi(-24), -a * 2f32.powi(-24)] {
                pairs.push([a, b]);
                pairs.push([b, a]);
            }
        }
    }
    pairs.extend([
        [f32::MAX, -f32::MAX],
        [-0.0, -0.0],
        [0.0, -0.0],
        [1.0, f32::MIN_POSITIVE],
        [-1.0, f32::MIN_POSITIVE],
    ]);
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: pairs.len() as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: pairs.len() as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&pairs));
    let reference = RESIDUAL.replace("fn native_sum_residual(", "fn native_sum_reference(");
    let source = format!(
        "{residual}\n{reference}\n@group(0) @binding(0) var<storage,read> input:array<vec2<f32>>;\n@group(0) @binding(1) var<storage,read_write> output:array<vec2<f32>>;\n@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) id:vec3<u32>) {{ if id.x<arrayLength(&input) {{ let pair=input[id.x]; output[id.x]=vec2<f32>(native_sum_residual(pair.x,pair.y),native_sum_reference(pair.x,pair.y)); }} }}"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("evaluate"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[buffer_entry(0, &input), buffer_entry(1, &output)],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((pairs.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let words = replay_tests::read_words(device, queue, &output);
    for (index, ([a, b], words)) in pairs.iter().zip(words.chunks_exact(2)).enumerate() {
        let high = a + b;
        let (large, small) = if a.abs() >= b.abs() {
            (*a, *b)
        } else {
            (*b, *a)
        };
        let expected =
            (((f64::from(large) - f64::from(high)) + f64::from(small)) * 16777216.0) as f32;
        assert_eq!(
            f32::from_bits(words[0]),
            expected,
            "residual case {index}: {a}+{b}"
        );
        assert_eq!(words[0], words[1], "GPU f64 residual bits case {index}");
    }
    eprintln!(
        "native residual: {} independent f64 cases pass",
        pairs.len()
    );
}

#[test]
#[ignore = "requires native GPU and SOIL_NATIVE_REPLAY_INPUT/OUTPUT"]
fn compensated_native_evolved_transfer_pair() {
    evolved_pair(false, false);
}

#[test]
#[ignore = "requires native/f64 GPU and saved SOIL_NATIVE_REPLAY_INPUT/OUTPUT"]
fn fast_compensated_native_evolved_transfer_pair() {
    evolved_pair(true, true);
}

#[test]
#[ignore = "requires native/f64 GPU and saved SOIL_NATIVE_REPLAY_INPUT/OUTPUT"]
fn fast_compensated_native_baseline_pair() {
    evolved_pair(true, false);
}

fn evolved_pair(fast: bool, compensated_control: bool) {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 1_000_000)
            .unwrap();
    let uniform = bytemuck::pod_read_unaligned::<Uniforms>(
        &std::fs::read(input.join("uniforms.bin")).unwrap(),
    );
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::SHADER_F64,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
    sim.restore_soil_with(queue, &checkpoint).unwrap();
    assert_eq!(uniform.particle_count, sim.particle_count);
    assert_eq!(uniform.grid_total, sim.cfg.grid_size.pow(3));
    let residual = if fast {
        fast_residual_source()
    } else {
        RESIDUAL.to_owned()
    };
    let candidate = Compensated::new_with_residual(&sim, &residual);
    let control = compensated_control.then(|| Compensated::new(&sim));
    let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniform));
    queue.write_buffer(
        &sim.material_table,
        0,
        bytemuck::cast_slice(&sim.step_material_table()),
    );
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
    let mut cross_errors = [0f64; 3];
    let mut support_mismatches = 0;
    let mut samples = Vec::new();
    for round in 0..8 {
        let mut pair: [Vec<Vec<u32>>; 2] = Default::default();
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let started = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                sim.bind_step(&mut pass);
                let owner = if mode == 1 {
                    Some(&candidate)
                } else {
                    control.as_ref()
                };
                if let Some(owner) = owner {
                    pass.set_bind_group(2, &owner.bindings, &[]);
                }
                let pipelines = if let Some(owner) = owner {
                    [
                        &owner.pipelines[0],
                        &owner.pipelines[1],
                        &owner.pipelines[2],
                    ]
                } else {
                    [&sim.pipe_reset, &sim.pipe_p2g, &sim.pipe_grid]
                };
                for (pipeline, threads) in pipelines.into_iter().zip([
                    uniform.grid_total,
                    sim.particle_count,
                    uniform.grid_total,
                ]) {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
                }
            }
            encoder.resolve_query_set(&queries, 0..2, &times, 0);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let ticks = replay_tests::read_words(device, queue, &times);
            let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
            let gpu_ms = (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            pair[mode] = [&sim.g_mass, &sim.g_momentum, &sim.g_velocity]
                .map(|b| replay_tests::read_words(device, queue, b))
                .to_vec();
            for (field, words) in pair[mode].iter().enumerate() {
                for (i, &word) in words.iter().enumerate() {
                    let value = f64::from(f32::from_bits(word));
                    assert!(value.is_finite());
                    if field == 0 {
                        assert!(value >= 0.0);
                    }
                    if let Some(base) = &baselines[mode] {
                        if field != 2 || i % 4 != 3 {
                            let error = (value - f64::from(f32::from_bits(base[field][i]))).abs();
                            variation[mode][field] = variation[mode][field].max(error);
                            changed[mode][field] += usize::from(word != base[field][i]);
                        }
                    }
                }
            }
            if baselines[mode].is_none() {
                baselines[mode] = Some(pair[mode].clone());
            }
            samples.push(serde_json::json!({"round":round,"candidate":mode==1,"warmup":round<2,"free_grid_gpu_ms":gpu_ms,"completed_wall_ms":wall_ms}));
        }
        for field in 0..3 {
            for (i, (&a, &b)) in pair[0][field].iter().zip(&pair[1][field]).enumerate() {
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                if field != 2 || i % 4 != 3 {
                    cross_errors[field] =
                        cross_errors[field].max((f64::from(a) - f64::from(b)).abs());
                }
                if field == 0 {
                    support_mismatches += usize::from((a > 0.0) != (b > 0.0));
                }
            }
        }
    }
    assert_eq!(before, sim.checkpoint_soil_with(queue).unwrap().to_bytes());
    let passed = support_mismatches == 0 && cross_errors.iter().all(|e| *e <= 2e-5);
    let report = serde_json::json!({"schema":1,"control":if compensated_control {"f64-compensated"} else {"native-uncorrected"},"candidate":if fast {"bounded-f32-with-f64-fallback"}else{"f64-compensated"},"particles":sim.particle_count,"extra_retained_bytes":candidate.error.size(),"control_extra_retained_bytes":control.as_ref().map_or(0,|o|o.error.size()),"samples":samples,"max_reference_candidate_errors_mass_momentum_velocity_xyz":cross_errors,"max_repeat_variation_mass_momentum_velocity_xyz":variation,"changed_repeat_words_mass_momentum_velocity_xyz":changed,"positive_mass_support_mismatches":support_mismatches,"particle_history_unchanged":true,"component_parity_passed":passed,"production_accepted":false,"determinism_guaranteed":false,"simulated_seconds":0});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("compensated transfer: {report}");
    assert!(passed, "unchanged 2e-5 component/support gate");
}
