use super::*;

const LANES_PER_PARTICLE: u32 = 4;

fn shader_source() -> String {
    let source = native_atomics::shader_source();
    let body = source.split_once("fn p2g(@builtin").unwrap().1;
    let body = body.split_once("    let pid =").unwrap().1;
    let body = format!("    let pid ={}", body.split_once("\n}\n").unwrap().0);
    let (prepare, stencil) = body.split_once("    for (var i:").unwrap();
    let setup = prepare.split_once("    let raw_svd =").unwrap().0;
    let deposit = stencil.split_once("                let w =").unwrap().1;
    let deposit = format!(
        "                let w ={}",
        deposit
            .split_once("                // Stage 24d.1")
            .unwrap()
            .0
    );
    format!(
        "{source}\n@group(1) @binding(9) var<storage,read_write> stencil_stress:array<mat3x3<f32>>;\n\
        @compute @workgroup_size(64) fn stencil_prepare(@builtin(global_invocation_id) gid:vec3<u32>) {{\n\
        {prepare}\n stencil_stress[pid]=stress;\n}}\n\
        @compute @workgroup_size(64) fn stencil_scatter(@builtin(global_invocation_id) thread:vec3<u32>) {{\n\
        let gid=vec3<u32>(thread.x/{LANES_PER_PARTICLE}u,0u,0u);\n\
        let first_lane=thread.x%{LANES_PER_PARTICLE}u;\n\
        {setup}\n let stress=stencil_stress[pid];\n\
        let dx=u.cell_size; let inv_dx2=u.inv_cell_size*u.inv_cell_size;\n\
        for(var lane=first_lane; lane<27u; lane+={LANES_PER_PARTICLE}u) {{\n\
        let i=lane/9u; let j=(lane/3u)%3u; let k=lane%3u;\n\
        {deposit}\n}}\n}}"
    )
}

struct Stencil {
    cache: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 2],
}

impl Stencil {
    fn new(sim: &MpmSimulation) -> Self {
        assert!(sim.uses_native_float_atomics());
        assert!(!sim.active_grid_enabled());
        assert!(!sim.use_particle_order);
        let device = &sim.device;
        let bytes = u64::from(sim.cfg.max_particles.max(1)) * 48;
        assert!(bytes <= device.limits().max_storage_buffer_binding_size);
        assert!(bytes <= device.limits().max_buffer_size);
        let cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test native stencil stress"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &storage_entries(10, false),
        });
        let uniforms = sim.pipe_reset.get_bind_group_layout(0);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&uniforms),
                Some(&layout),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test native parallel stencil"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipelines = ["stencil_prepare", "stencil_scatter"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                buffer_entry(0, &sim.p_pos),
                buffer_entry(1, &sim.p_vel),
                buffer_entry(2, &sim.p_c),
                buffer_entry(3, &sim.p_f),
                buffer_entry(4, &sim.p_volume),
                buffer_entry(5, &sim.p_mass),
                buffer_entry(6, &sim.p_jp),
                buffer_entry(7, &sim.p_material),
                buffer_entry(8, &sim.material_table),
                buffer_entry(9, &cache),
            ],
        });
        Self {
            cache,
            bindings,
            pipelines,
        }
    }

    fn encode(&self, sim: &MpmSimulation, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(1, &self.bindings, &[]);
        for (pipeline, threads) in self.pipelines.iter().zip([
            sim.particle_count,
            sim.particle_count.checked_mul(LANES_PER_PARTICLE).unwrap(),
        ]) {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
        }
        pass.set_bind_group(1, &sim.bg_particles, &[]);
    }
}

fn engine() -> Engine {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::TIMESTAMP_QUERY,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    engine
}

fn compare(engine: &Engine, sim: &mut MpmSimulation, uniform: Uniforms) -> serde_json::Value {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let candidate = Stencil::new(sim);
    let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniform));
    queue.write_buffer(
        &sim.material_table,
        0,
        bytemuck::cast_slice(&sim.step_material_table()),
    );
    let query = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let timestamps = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut samples = Vec::new();
    let mut errors = [0.0f64; 3];
    let mut positive_mass_mismatches = 0usize;
    for round in 0..8 {
        let mut pair: [Vec<Vec<u32>>; 2] = Default::default();
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let started = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_reset);
                pass.dispatch_workgroups(uniform.grid_total.div_ceil(64), 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &query,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                sim.bind_step(&mut pass);
                if mode == 0 {
                    pass.set_pipeline(&sim.pipe_p2g);
                    pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
                } else {
                    candidate.encode(sim, &mut pass);
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_grid);
                pass.dispatch_workgroups(uniform.grid_total.div_ceil(64), 1, 1);
            }
            encoder.resolve_query_set(&query, 0..2, &timestamps, 0);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let words = replay_tests::read_words(device, queue, &timestamps);
            let time = |i| u64::from(words[i]) | (u64::from(words[i + 1]) << 32);
            let gpu_ms = (time(2) - time(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            pair[mode] = [&sim.g_mass, &sim.g_momentum, &sim.g_velocity]
                .map(|b| replay_tests::read_words(device, queue, b))
                .to_vec();
            samples.push(serde_json::json!({"round":round,"mode":mode,"warmup":round<2,"transfer_gpu_ms":gpu_ms,"free_grid_completed_wall_ms":wall_ms}));
            eprintln!(
                "stencil round={round} mode={mode} transfer_gpu_ms={gpu_ms} free_grid_wall_ms={wall_ms}"
            );
        }
        for field in 0..3 {
            assert_eq!(pair[0][field].len(), pair[1][field].len());
            for (&a, &b) in pair[0][field].iter().zip(&pair[1][field]) {
                let a = f64::from(f32::from_bits(a));
                let b = f64::from(f32::from_bits(b));
                assert!(a.is_finite() && b.is_finite());
                errors[field] = errors[field].max((a - b).abs());
                if field == 0 {
                    assert!(a >= 0.0 && b >= 0.0);
                    positive_mass_mismatches += usize::from((a > 0.0) != (b > 0.0));
                }
            }
        }
    }
    let after = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    assert_eq!(after.len(), before.len());
    assert_eq!(after.iter().zip(&before).position(|(a, b)| a != b), None);
    let passed = errors.iter().all(|e| *e <= 2e-5) && positive_mass_mismatches == 0;
    serde_json::json!({"schema":1,"particles":sim.particle_count,"cache_bytes":candidate.cache.size(),"max_absolute_mass_momentum_free_grid_errors":errors,"positive_mass_mismatches":positive_mass_mismatches,"particle_history_unchanged":true,"component_parity_passed":passed,"production_accepted":false,"simulated_seconds":0,"samples":samples})
}

#[test]
#[ignore = "requires native atomics and timestamps"]
fn native_stencil_boundary_and_partial_workgroup() {
    let engine = engine();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 65;
    cfg.youngs = 3000.0;
    let mut sim = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    for count in [0, 1, 65] {
        let points: Vec<_> = (0..count)
            .map(|i| {
                [
                    -0.51 + (i % 5) as f32 * 0.25,
                    -0.49 + (i / 5 % 5) as f32 * 0.245,
                    (i / 25) as f32 * 0.19,
                ]
            })
            .collect();
        sim.set_particles_with_velocity(
            engine.queue().unwrap(),
            &points,
            &vec![[0.03, -0.02, 0.01]; count],
        )
        .unwrap();
        let uniform = sim.step_uniforms(0.0001);
        let report = compare(&engine, &mut sim, uniform);
        eprintln!("boundary stencil report={report}");
        assert_eq!(report["component_parity_passed"], true);
    }
}

#[test]
#[ignore = "requires native GPU and SOIL_NATIVE_REPLAY_INPUT/OUTPUT"]
fn native_stencil_evolved_transfer_pair() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 1_000_000)
            .unwrap();
    let uniform = bytemuck::pod_read_unaligned::<Uniforms>(
        &std::fs::read(input.join("uniforms.bin")).unwrap(),
    );
    let engine = engine();
    let mut sim = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &checkpoint)
        .unwrap();
    assert_eq!(uniform.particle_count, sim.particle_count);
    assert_eq!(uniform.grid_total, sim.cfg.grid_size.pow(3));
    let report = compare(&engine, &mut sim, uniform);
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("evolved stencil report={report}");
    assert_eq!(report["component_parity_passed"], true);
}
