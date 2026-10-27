use super::super::tests as transfer_tests;
use super::*;
use crate::mpm::replay_tests::read_words;

fn program(native: bool) -> String {
    let base = if native {
        native_atomics::shader_source()
    } else {
        include_str!("../shaders/mpm.wgsl").to_owned()
    };
    source(format!(
        "{base}\n{}",
        include_str!("../shaders/mpm_deterministic_p2g.wgsl")
    ))
    .unwrap()
}

#[test]
fn gather_stress_cache_shader_inputs_and_layout() {
    for native in [false, true] {
        let text = program(native);
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let ty = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("GatherStressKey"))
            .unwrap()
            .1;
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("key struct");
        };
        assert_eq!(u64::from(*span), KEY_BYTES);
        assert_eq!(
            members.iter().map(|m| m.offset).collect::<Vec<_>>(),
            [0, 48, 64, 80]
        );
        assert_eq!(text.matches("fn gather_prepare_uncached(").count(), 1);
        assert_eq!(text.matches("fn gather_prepare(@").count(), 1);
        for input in [
            "p_volume[pid]",
            "p_jp[pid]",
            "u.dt",
            "u.youngs",
            "u.poisson",
            "u.inv_cell_size",
            "material.p0.x",
            "material.p1.x",
            "p_F[pid]",
        ] {
            assert!(include_str!("../shaders/mpm_gather_stress_cache.wgsl").contains(input));
        }
    }
    assert!(source(String::new()).is_err());
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; exact constitutive-cache evolution and mode lifecycle"]
fn gather_stress_cache_gpu_evolution_and_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        for shared in [false, true] {
            let mut sims = [false, true].map(|cached| {
                let mut sim = if mode < 2 {
                    transfer_tests::seeded(&engine, mode == 1)
                } else {
                    dense_tests::seeded(&engine, if mode == 2 { 35 } else { 64 })
                };
                let compensated = !sim.native_float_atomics;
                let required = u64::from(sim.cfg.max_particles.max(1)) * 160;
                if cached {
                    let allocations = DeviceScope::start().unwrap();
                    assert!(
                        sim.prepare_deterministic_p2g_stress_cache(
                            required - 1,
                            compensated,
                            shared
                        )
                        .is_err()
                    );
                    assert_eq!(allocations.finish().observations, 0);
                    sim.prepare_deterministic_p2g_stress_cache(required, compensated, shared)
                        .unwrap();
                    assert_eq!(sim.deterministic_p2g_storage_bytes(), required);
                    let allocations = DeviceScope::start().unwrap();
                    sim.prepare_deterministic_p2g_stress_cache(required, compensated, shared)
                        .unwrap();
                    assert_eq!(allocations.finish().observations, 0);
                } else if shared {
                    sim.prepare_deterministic_p2g_shared_chunks(required, compensated)
                        .unwrap();
                } else if compensated {
                    sim.prepare_deterministic_p2g_with_compensated_stiffness(required)
                        .unwrap();
                } else {
                    sim.prepare_deterministic_p2g_with_budget(required).unwrap();
                }
                sim.set_deterministic_p2g(true).unwrap();
                assert_eq!(sim.deterministic_p2g_stress_cache_enabled(), cached);
                sim
            });
            for step in 0..8 {
                let dt = if step < 4 { 0.0001 } else { 0.0002 };
                for index in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let allocations = DeviceScope::start().unwrap();
                    let traffic = TransferScope::start().unwrap();
                    let dispatches = DispatchScope::start().unwrap();
                    sims[index].step_n_with(device, queue, dt, 1).unwrap();
                    assert_eq!(allocations.finish().observations, 0);
                    assert_eq!(traffic.finish().readback_bytes, 0);
                    assert!(dispatches.finish().encoded_dispatches > 0);
                }
                assert_eq!(
                    sims[0].checkpoint_soil_with(queue).unwrap().to_bytes(),
                    sims[1].checkpoint_soil_with(queue).unwrap().to_bytes(),
                    "mode={mode} shared={shared} step={step}"
                );
                assert_eq!(
                    transfer_tests::world_grid(&engine, &sims[0]),
                    transfer_tests::world_grid(&engine, &sims[1]),
                    "grid mode={mode} shared={shared} step={step}"
                );
            }
            let before = sims[1].prepare_step(0.0001, 1).unwrap();
            sims[1]
                .prepare_deterministic_p2g_with_budget(u64::MAX)
                .unwrap();
            assert!(!sims[1].deterministic_p2g_stress_cache_enabled());
            assert_eq!(
                sims[1].deterministic_p2g_storage_bytes(),
                u64::from(sims[1].cfg.max_particles.max(1)) * 64
            );
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&before).is_err());
            drop(batch);
            sims[1].set_deterministic_p2g(false).unwrap();
            sims[1].release_deterministic_p2g().unwrap();
            assert_eq!(sims[1].deterministic_p2g_storage_bytes(), 0);
            println!(
                "stress-cache mode={mode} shared={shared} exact_evolved_boundaries=8 hot_allocations=0 hot_readbacks=0"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; actual hit bypass and all seventeen key input misses"]
fn gather_stress_cache_gpu_hit_and_input_misses() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = transfer_tests::seeded(&engine, false);
    sim.set_rigid_spheres(queue, &[]).unwrap();
    sim.prepare_deterministic_p2g_stress_cache(u64::MAX, false, false)
        .unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    let initial = sim.checkpoint_soil_with(queue).unwrap();
    let input = program(false);
    let marker = "    gather_particles[pid].stiffness = kdiag_p;";
    assert_eq!(input.matches(marker).count(), 1);
    let instrumented = input.replace(marker, &format!("{marker}\n    numerical_fault();"));
    let gather = sim.deterministic_p2g.as_ref().unwrap();
    let layouts =
        std::array::from_fn::<_, 4, _>(|i| gather.pipelines[0].get_bind_group_layout(i as u32));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stress cache miss probe"),
        bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stress cache miss probe"),
        source: wgpu::ShaderSource::Wgsl(instrumented.into()),
    });
    let probe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stress cache miss probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("gather_prepare"),
        compilation_options: Default::default(),
        cache: None,
    });
    for changed in 0u32..=17 {
        sim.restore_soil_with(queue, &initial).unwrap();
        let uniforms = sim.step_uniforms(0.0001);
        queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &sim.material_table,
            0,
            bytemuck::cast_slice(&sim.step_material_table()),
        );
        queue.write_buffer(
            &sim.active_tiles.list,
            sim.active_tiles.metadata_offset + 8,
            bytemuck::cast_slice(&[0u32]),
        );
        let execute = |pipeline: &wgpu::ComputePipeline| {
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                let gather = sim.deterministic_p2g.as_ref().unwrap();
                pass.set_bind_group(1, &gather.bindings, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
            }
            queue.submit([encoder.finish()]);
        };
        execute(&sim.deterministic_p2g.as_ref().unwrap().pipelines[0]);
        let value = 0.137f32.to_bits();
        match changed {
            0 => {}
            1..=9 => {
                let i = changed - 1;
                queue.write_buffer(
                    &sim.p_f,
                    u64::from((i / 3) * 16 + (i % 3) * 4),
                    bytemuck::cast_slice(&[value]),
                );
            }
            10 => queue.write_buffer(&sim.p_volume, 0, bytemuck::cast_slice(&[value])),
            11 => queue.write_buffer(&sim.p_jp, 0, bytemuck::cast_slice(&[value])),
            12..=15 => {
                let offset = [32, 40, 44, 60][(changed - 12) as usize];
                queue.write_buffer(&sim.uniform_buf, offset, bytemuck::cast_slice(&[value]));
            }
            16 => queue.write_buffer(&sim.material_table, 0, bytemuck::cast_slice(&[2.0f32])),
            17 => queue.write_buffer(&sim.material_table, 16, bytemuck::cast_slice(&[value])),
            _ => unreachable!(),
        }
        execute(&probe);
        let metadata = read_words(device, queue, &sim.active_tiles.list);
        let fault = metadata[sim.active_tiles.metadata_offset as usize / 4 + 2];
        assert_eq!(fault != 0, changed != 0, "key input {changed}");
    }
    println!("stress-cache exact_hit_bypasses_miss=true changed_input_misses=17");
}
