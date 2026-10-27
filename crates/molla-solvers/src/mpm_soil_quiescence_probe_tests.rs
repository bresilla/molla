use super::*;

#[test]
#[ignore = "requires GPU; exact-rest observer detects controlled between-checkpoint changes"]
fn quiescence_gpu_publication_probes_keep_transients_visible() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = MpmConfig {
        max_particles: 129,
        max_rigid: 1,
        grid_size: 64,
        grid_origin: [-3.0, -0.5, -3.0],
        cell_size: 0.1,
        gravity_y: 0.0,
        time_step: MpmTimeStep::Explicit,
        ..MpmConfig::sand()
    };
    let mut sim = MpmSimulation::new_sparse(&engine, cfg, 128).unwrap();
    sim.set_particles(queue, &vec![[0.0, 0.3, 0.0]; 129])
        .unwrap();
    let initial = sim.checkpoint_soil_with(queue).unwrap();
    sim.prepare_soil_motion_guard().unwrap();
    sim.set_soil_motion_bound(Some([0.01; 3]));
    sim.prepare_soil_quiescence(2080).unwrap();
    let output = std::env::var_os("MOLLA_QUIESCENCE_OUTPUT").map(std::path::PathBuf::from);
    if let Some(path) = &output {
        std::fs::create_dir_all(path).unwrap();
    }
    let mut reports = Vec::new();
    for (name, component, changed, restored) in [
        ("position", 0, 0.001_f32, 0.0_f32),
        ("velocity", 4, 0.1, 0.0),
        ("affine", 8, 0.01, 0.0),
        ("deformation", 20, 0.9, 1.0),
        ("plastic", 3, 0.2, 0.0),
    ] {
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.step_n(&engine, 0.0001, 3).unwrap();
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles
                .iter()
                .all(|p| p.consecutive_steps == 3)
        );
        let mut counts = Vec::new();
        for value in [changed, restored, restored] {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quiescence publication probe"),
                source: wgpu::ShaderSource::Wgsl(format!(
                    "@group(0) @binding(0) var<storage, read_write> proposals: array<u32>;\n@compute @workgroup_size(1) fn main() {{ proposals[{}u] = {}u; }}",
                    128 * 32 + component, value.to_bits(),
                ).into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let guard = sim.motion_guard.as_ref().unwrap();
            let observer = guard.quiescence.as_ref().unwrap();
            let injection = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[buffer_entry(0, &guard.proposals)],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &injection, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                sim.bind_step(&mut pass);
                pass.set_bind_group(0, &observer.bindings, &[]);
                pass.set_pipeline(&observer.pipeline);
                pass.dispatch_workgroups(3, 1, 1);
            }
            queue.submit([encoder.finish()]);
            let snapshot = sim.read_soil_quiescence(&engine, 129).unwrap();
            counts.push(snapshot.particles[128].consecutive_steps);
            assert!(
                snapshot.particles[..128]
                    .iter()
                    .all(|p| p.consecutive_steps == 3 + counts.len() as u32)
            );
        }
        assert_eq!(counts, [0, 0, 1]);
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            initial.to_bytes()
        );
        reports.push(serde_json::json!({"field":name,"controlled_publications":counts,"other_particles_streak":6,"restored_physical_state_exact":true}));
    }
    if let Some(path) = output {
        std::fs::write(
            path.join("publication-probes.json"),
            serde_json::to_vec_pretty(&reports).unwrap(),
        )
        .unwrap();
    }
    println!(
        "quiescence controlled_publication_probes=5 transient_streaks=0,0,1 endpoint_state_exact=true"
    );
}
