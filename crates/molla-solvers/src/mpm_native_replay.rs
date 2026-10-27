use super::*;
use std::path::Path;

fn measure(sim: &mut MpmSimulation, engine: &Engine, uniforms: Uniforms, output: &Path) {
    assert!(sim.uses_native_float_atomics());
    assert!(!sim.uses_two_pass_p2g());
    assert!(!sim.active_grid_enabled());
    assert_eq!(uniforms.particle_count, sim.particle_count);
    assert_eq!(uniforms.grid_size, sim.cfg.grid_size);
    assert_eq!(uniforms.grid_total, sim.cfg.grid_size.pow(3));
    assert_eq!(
        uniforms.grid_layout,
        sim.step_uniforms(uniforms.dt).grid_layout
    );
    assert!(uniforms.dt.is_finite() && uniforms.dt > 0.0);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    queue.write_buffer(
        &sim.material_table,
        0,
        bytemuck::cast_slice(&sim.step_material_table()),
    );
    let mut baseline: Option<[Vec<u32>; 4]> = None;
    let mut samples = Vec::new();
    for repeat in 0..6 {
        let mut encoder = device.create_command_encoder(&Default::default());
        for (pipeline, groups) in [
            (
                &sim.pipe_reset,
                uniforms.grid_total.div_ceil(WORKGROUP_SIZE),
            ),
            (
                sim.p2g_pipeline(),
                sim.particle_count.div_ceil(WORKGROUP_SIZE),
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let mut fields = [&sim.g_mass, &sim.g_momentum, &sim.g_kdiag]
            .map(|buffer| super::replay_tests::read_words(device, queue, buffer))
            .to_vec();
        for (field, count) in fields.iter_mut().zip([1, 3, 1]) {
            let physical_words = uniforms.grid_total as usize * count;
            assert!(field.len() >= physical_words);
            field.truncate(physical_words);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.pipe_grid);
            pass.dispatch_workgroups(uniforms.grid_total.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit([encoder.finish()]);
        fields.push(super::replay_tests::read_words(
            device,
            queue,
            &sim.g_velocity,
        ));
        let fields: [Vec<u32>; 4] = fields.try_into().unwrap();
        let mut comparison = Vec::new();
        for (index, name) in ["mass", "momentum", "stiffness", "free-grid"]
            .iter()
            .enumerate()
        {
            let current = &fields[index];
            assert!(
                current.iter().all(|&v| f32::from_bits(v).is_finite()),
                "nonfinite {name}"
            );
            let mut changed = 0usize;
            let mut numeric_changes = 0usize;
            let mut max_absolute = 0.0f64;
            let mut max_scaled = 0.0f64;
            let mut max_velocity_xyz = 0.0f64;
            if let Some(reference) = &baseline {
                assert_eq!(current.len(), reference[index].len());
                for (word, (&a, &b)) in current.iter().zip(&reference[index]).enumerate() {
                    changed += usize::from(a != b);
                    let a = f64::from(f32::from_bits(a));
                    let b = f64::from(f32::from_bits(b));
                    numeric_changes += usize::from(a != b);
                    let error = (a - b).abs();
                    max_absolute = max_absolute.max(error);
                    max_scaled = max_scaled.max(error / b.abs().max(1.0));
                    if index == 3 && word % 4 != 3 {
                        max_velocity_xyz = max_velocity_xyz.max(error);
                    }
                }
            } else {
                let bytes: Vec<_> = current.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(output.join(format!("{name}.bin")), bytes).unwrap();
            }
            comparison.push(serde_json::json!({"field":name,"words":current.len(),"bit_changes":changed,"numeric_changes":numeric_changes,"max_absolute_difference":max_absolute,"max_scaled_difference":max_scaled,"max_velocity_xyz_difference":(index==3).then_some(max_velocity_xyz)}));
        }
        let total_mass: f64 = fields[0]
            .iter()
            .map(|&v| f64::from(f32::from_bits(v)))
            .sum();
        eprintln!("native P2G repeat={repeat} mass={total_mass} fields={comparison:?}");
        samples.push(
            serde_json::json!({"repeat":repeat,"total_grid_mass":total_mass,"fields":comparison}),
        );
        if baseline.is_none() {
            baseline = Some(fields);
        }
    }
    let after = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    assert_eq!(after.len(), before.len());
    assert_eq!(
        after.iter().zip(&before).position(|(a, b)| a != b),
        None,
        "free-grid replay changed a particle/history byte"
    );
    let report = serde_json::json!({"schema":1,"native_float_atomics":true,"particles":sim.particle_count,"grid_size":sim.cfg.grid_size,"dt":uniforms.dt,"particle_history_unchanged":true,"free_grid_layout":"vec4 velocity_xyz,mass; velocity-only maximum reported separately","stiffness_scope":"physical grid prefix only; native explicit stiffness is unused","scope":"same particle/history/material inputs; reset and native P2G, then grid update; no contact, G2P or plasticity; not physical or performance acceptance","samples":samples});
    std::fs::write(
        output.join("repeatability.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

pub(crate) fn capture(sim: &mut MpmSimulation, engine: &Engine, output: &Path) {
    std::fs::create_dir(output).unwrap();
    let saved = sim
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .to_bytes();
    let words = uniform_words(sim, engine);
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    let uniforms = bytemuck::pod_read_unaligned::<Uniforms>(&bytes);
    std::fs::write(output.join("soil.bin"), saved).unwrap();
    std::fs::write(output.join("uniforms.bin"), bytes).unwrap();
    measure(sim, engine, uniforms, output);
}

fn uniform_words(sim: &MpmSimulation, engine: &Engine) -> Vec<u32> {
    let device = engine.device().unwrap();
    let bytes = sim.uniform_buf.size();
    assert_eq!(bytes, std::mem::size_of::<Uniforms>() as u64);
    assert_eq!(bytes % 16, 0);
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-only uniform snapshot"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let source = format!(
        "struct Words {{values:array<vec4<u32>,{}>}}\n@group(0) @binding(0) var<uniform> input:Words;\n@group(0) @binding(1) var<storage,read_write> output:array<u32>;\n@compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3<u32>) {{if id.x<arrayLength(&output) {{output[id.x]=input.values[id.x/4u][id.x%4u];}}}}",
        bytes / 16
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("copy"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&sim.uniform_buf, &output]
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((bytes / 4).div_ceil(64) as u32, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    super::replay_tests::read_words(device, engine.queue().unwrap(), &output)
}

#[test]
#[ignore = "requires native GPU and a new SOIL_NATIVE_REPLAY_OUTPUT directory"]
fn native_p2g_capture_roundtrip_small_fixture() {
    let output = std::path::PathBuf::from(
        std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").expect("new output required"),
    );
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 4;
    let mut sim = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    sim.set_particles_with_velocity(
        queue,
        &[[0.0; 3], [0.01; 3], [0.02; 3], [0.03; 3]],
        &[[0.03, -0.02, 0.01]; 4],
    )
    .unwrap();
    for _ in 0..3 {
        sim.step(&engine, 0.0001).unwrap();
    }
    capture(&mut sim, &engine, &output);
    let bytes = std::fs::read(output.join("soil.bin")).unwrap();
    let checkpoint = MpmSoilCheckpoint::from_bytes(&bytes, 4).unwrap();
    let uniforms = bytemuck::pod_read_unaligned::<Uniforms>(
        &std::fs::read(output.join("uniforms.bin")).unwrap(),
    );
    let mut restored = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
    restored.restore_soil_with(queue, &checkpoint).unwrap();
    assert_eq!(
        restored.checkpoint_soil_with(queue).unwrap().to_bytes(),
        bytes
    );
    let replay = output.join("roundtrip");
    std::fs::create_dir(&replay).unwrap();
    measure(&mut restored, &engine, uniforms, &replay);
    for root in [&output, &replay] {
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("repeatability.json")).unwrap())
                .unwrap();
        assert_eq!(report["particles"], 4);
        assert_eq!(report["particle_history_unchanged"], true);
        assert_eq!(report["samples"].as_array().unwrap().len(), 6);
        assert!(
            report["samples"]
                .as_array()
                .unwrap()
                .iter()
                .all(|s| s["fields"].as_array().unwrap().len() == 4)
        );
    }
}

#[test]
#[ignore = "requires native GPU and SOIL_NATIVE_REPLAY_INPUT/OUTPUT directories"]
fn replay_captured_native_p2g() {
    let input = std::path::PathBuf::from(
        std::env::var_os("SOIL_NATIVE_REPLAY_INPUT").expect("input required"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").expect("new output required"),
    );
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 1_000_000)
            .unwrap();
    let uniforms = bytemuck::pod_read_unaligned::<Uniforms>(
        &std::fs::read(input.join("uniforms.bin")).unwrap(),
    );
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut sim = MpmSimulation::new_native_explicit(&engine, checkpoint.config()).unwrap();
    sim.restore_soil_with(engine.queue().unwrap(), &checkpoint)
        .unwrap();
    measure(&mut sim, &engine, uniforms, &output);
}
