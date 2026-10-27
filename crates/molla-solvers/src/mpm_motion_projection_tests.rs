use super::*;

#[test]
#[ignore = "requires GPU; staged plasticity faults cannot partially publish physical state"]
fn motion_guard_gpu_plasticity_transaction_and_rheology_parity() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::env::var_os("MOLLA_MOTION_PROJECTION_OUTPUT").map(std::path::PathBuf::from);
    if let Some(path) = &output {
        std::fs::create_dir_all(path).unwrap();
    }
    let words = |sim: &MpmSimulation| {
        sim.checkpoint_buffers()
            .iter()
            .map(|buffer| read_words(device, queue, buffer))
            .collect::<Vec<_>>()
    };
    for mode in 0..3 {
        for (material, rheology) in [
            MpmRheology::Sand {
                friction_angle: 0.5,
            },
            MpmRheology::Snow {
                theta_c: 0.025,
                theta_s: 0.0075,
                hardening: 10.0,
            },
            MpmRheology::Viscous {
                velocity_damping: 3.0,
                affine_damping: 5.0,
            },
        ]
        .into_iter()
        .enumerate()
        {
            let cfg = MpmConfig {
                max_particles: 129,
                max_rigid: 1,
                grid_size: 64,
                cell_size: 0.1,
                grid_origin: [-3.0, -0.5, -3.0],
                youngs: 1000.0,
                rheology,
                time_step: MpmTimeStep::Explicit,
                ..MpmConfig::sand()
            };
            let mut sim = if mode == 2 {
                MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            if mode == 1 {
                sim.set_active_grid(true).unwrap();
            }
            sim.set_particles(queue, &vec![[0.0, 0.3, 0.0]; 129])
                .unwrap();
            let f = [
                0.97_f32, 0.001, 0.0, 0.0, 0.0, 1.01, 0.002, 0.0, 0.001, 0.0, 0.98, 0.0,
            ];
            queue.write_buffer(&sim.p_f, 0, bytemuck::cast_slice(&vec![f; 129]));
            queue.write_buffer(&sim.p_jp, 0, bytemuck::cast_slice(&vec![0.2_f32; 129]));
            let initial = sim.checkpoint_soil_with(queue).unwrap();
            sim.step_n(&engine, 0.0001, 1).unwrap();
            let control = sim.checkpoint_soil_with(queue).unwrap();
            sim.restore_soil_with(queue, &initial).unwrap();
            sim.prepare_soil_motion_guard().unwrap();
            sim.set_soil_motion_bound(Some([0.1; 3]));
            sim.step_n(&engine, 0.0001, 1).unwrap();
            assert_eq!(
                sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
                control.to_bytes()
            );
            let before = words(&sim);
            for (fault, component, bits) in [
                ("nan-f", 20, f32::NAN.to_bits()),
                ("infinite-jp", 3, f32::INFINITY.to_bits()),
            ] {
                sim.restore_soil_with(queue, &initial).unwrap();
                sim.step_n(&engine, 0.0001, 1).unwrap();
                assert_eq!(words(&sim), before);
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fault staged soil projection"),
                    source: wgpu::ShaderSource::Wgsl(format!(
                        "@group(0) @binding(0) var<storage, read_write> proposals: array<u32>;\n@compute @workgroup_size(1) fn main() {{ proposals[{}u] = {}u; }}",
                        128 * 32 + component, bits,
                    ).into()),
                });
                let inject = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fault staged soil projection"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &inject.get_bind_group_layout(0),
                    entries: &[buffer_entry(
                        0,
                        &sim.motion_guard.as_ref().unwrap().proposals,
                    )],
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    sim.bind_step(&mut pass);
                    sim.encode_gather(&mut pass, 3);
                    pass.set_pipeline(&inject);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    sim.bind_step(&mut pass);
                    sim.encode_plasticity(&mut pass, 3);
                }
                queue.submit([encoder.finish()]);
                let status = sim.read_active_grid_status_with(device, queue).unwrap();
                assert!(
                    status.numerical_fault && !status.motion_bound_exceeded && !status.overflow
                );
                let rejected = words(&sim);
                assert_eq!(rejected, before);
                assert!(sim.checkpoint_soil_with(queue).is_err());
                sim.step_n(&engine, 0.0001, 2).unwrap();
                assert_eq!(words(&sim), before);
                if let Some(path) = &output {
                    let raw: Vec<u32> = rejected.into_iter().flatten().collect();
                    std::fs::write(
                        path.join(format!("mode-{mode}-material-{material}-{fault}.words")),
                        bytemuck::cast_slice(&raw),
                    )
                    .unwrap();
                }
            }
            sim.restore_soil_with(queue, &initial).unwrap();
            let step = sim.prepare_step(0.0001, 1).unwrap();
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            batch.encode_step(&step).unwrap();
            batch.submit(queue);
            assert_eq!(
                sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
                control.to_bytes()
            );
            if let Some(path) = &output {
                std::fs::write(
                    path.join(format!("mode-{mode}-material-{material}-initial.soil")),
                    initial.to_bytes(),
                )
                .unwrap();
                std::fs::write(
                    path.join(format!("mode-{mode}-material-{material}-control.soil")),
                    control.to_bytes(),
                )
                .unwrap();
            }
            println!(
                "motion_projection mode={mode} material={material} particles=129 success_exact=true faults=2 rejection_exact=true prepared_retry_exact=true"
            );
        }
    }
}
