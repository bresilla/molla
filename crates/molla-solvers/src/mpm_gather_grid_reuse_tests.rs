use super::super::tests as transfer_tests;
use super::*;
use crate::mpm::replay_tests::read_words;

fn program(native: bool, compensated: bool, shared: bool) -> String {
    let base = if native {
        native_atomics::shader_source()
    } else {
        include_str!("../shaders/mpm.wgsl").to_owned()
    };
    let mask = if shared {
        shared_chunks::source(compensated).unwrap()
    } else if compensated {
        compensated_gather::source().unwrap()
    } else {
        include_str!("../shaders/mpm_gather_masks.wgsl").to_owned()
    };
    format!(
        "{base}\n{}\n{}",
        include_str!("../shaders/mpm_deterministic_p2g.wgsl"),
        mask_source(mask).unwrap()
    )
}

#[test]
fn gather_grid_reuse_shader_layout_budget_and_support() {
    let limits = wgpu::Limits::default();
    assert_eq!(
        sizes(129, 64, &limits).unwrap(),
        [14448, 5242880, 32768, 64]
    );
    for (n, grid) in [(1, 0), (1, u32::MAX), (u32::MAX, 64), (1, 512)] {
        assert!(sizes(n, grid, &limits).is_err());
    }
    for (native, compensated) in [(false, false), (false, true), (true, false)] {
        for shared in [false, true] {
            let text = program(native, compensated, shared);
            let module = naga::front::wgsl::parse_str(&text)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            for (name, bytes) in [("GatherReuseKey", 112), ("GatherReuseTile", 8)] {
                let ty = module
                    .types
                    .iter()
                    .find(|(_, t)| t.name.as_deref() == Some(name))
                    .unwrap()
                    .1;
                let naga::TypeInner::Struct { span, .. } = &ty.inner else {
                    panic!("struct");
                };
                assert_eq!(*span, bytes);
            }
        }
    }
    for base in -8i32..80 {
        for grid in [5i32, 35, 64] {
            let lo = base.max(0);
            let hi = (base + 2).min(grid - 1);
            let marked: Vec<_> = if hi < lo {
                vec![]
            } else {
                (lo / 4..=hi / 4).collect()
            };
            let expected: std::collections::BTreeSet<_> = (base..=base + 2)
                .filter(|&n| n >= 0 && n < grid)
                .map(|n| n / 4)
                .collect();
            assert_eq!(
                marked
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>(),
                expected
            );
        }
    }
}

fn seed(engine: &Engine, mode: usize, reuse: bool, shared: bool) -> MpmSimulation {
    let mut sim = if mode < 2 {
        transfer_tests::seeded(engine, mode == 1)
    } else {
        dense_tests::seeded(engine, if mode == 2 { 35 } else { 64 })
    };
    let compensated = !sim.native_float_atomics;
    if reuse {
        sim.prepare_deterministic_p2g_grid_reuse(u64::MAX, compensated, shared)
            .unwrap();
    } else if shared {
        sim.prepare_deterministic_p2g_shared_chunks(u64::MAX, compensated)
            .unwrap();
    } else if compensated {
        sim.prepare_deterministic_p2g_with_compensated_stiffness(u64::MAX)
            .unwrap();
    }
    sim
}

fn transfer(engine: &Engine, sim: &mut MpmSimulation) {
    let queue = engine.queue().unwrap();
    queue.write_buffer(
        &sim.uniform_buf,
        0,
        bytemuck::bytes_of(&sim.step_uniforms(0.0001)),
    );
    queue.write_buffer(
        &sim.material_table,
        0,
        bytemuck::cast_slice(&sim.step_material_table()),
    );
    if let Some(ids) = sim.particle_material_dirty.take() {
        assert_eq!(ids.len(), sim.particle_count as usize);
        queue.write_buffer(&sim.p_material, 0, bytemuck::cast_slice(&ids));
    }
    sim.particle_order.as_ref().unwrap().upload(
        queue,
        &sim.cfg,
        sim.particle_count,
        sim.grid_window_cells,
    );
    if sim.use_active_grid {
        sim.active_tiles
            .upload(
                queue,
                &sim.cfg,
                sim.particle_count,
                sim.rigid_count,
                sim.grid_window_cells,
            )
            .unwrap();
    }
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if sim.use_active_grid {
        sim.initialize_active_grid(&mut encoder, false);
    } else if sim.active_grid_needs_reset {
        encoder.clear_buffer(
            &sim.active_tiles.list,
            sim.active_tiles.metadata_offset,
            Some(16),
        );
        sim.active_grid_needs_reset = false;
    }
    if sim.native_float_atomics && !sim.use_active_grid {
        dense_tests::encode_transfer(sim, &mut encoder, true);
    } else {
        transfer_tests::encode_transfer(sim, &mut encoder, true);
    }
    queue.submit([encoder.finish()]);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; exact retained transfer and evolving state across layouts"]
fn gather_grid_reuse_gpu_exact_evolution_and_held_inputs() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        for shared in [false, true] {
            let mut sims = [false, true].map(|reuse| seed(&engine, mode, reuse, shared));
            let required = u64::from(sims[1].cfg.max_particles.max(1)) * 64
                + sizes(
                    sims[1].cfg.max_particles,
                    sims[1].cfg.grid_size,
                    &device.limits(),
                )
                .unwrap()
                .iter()
                .sum::<u64>();
            assert_eq!(sims[1].deterministic_p2g_storage_bytes(), required);
            let scope = DeviceScope::start().unwrap();
            assert!(
                sims[1]
                    .prepare_deterministic_p2g_grid_reuse(required - 1, mode < 2, shared)
                    .is_err()
            );
            sims[1]
                .prepare_deterministic_p2g_grid_reuse(required, mode < 2, shared)
                .unwrap();
            assert_eq!(scope.finish().observations, 0);
            for step in 0..8 {
                let dt = if step < 4 { 0.0001 } else { 0.0002 };
                for arm in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
                    sims[arm].step_n_with(device, queue, dt, 1).unwrap();
                }
                assert_eq!(
                    sims[0].checkpoint_soil_with(queue).unwrap().to_bytes(),
                    sims[1].checkpoint_soil_with(queue).unwrap().to_bytes(),
                    "mode={mode} shared={shared} step={step}"
                );
                assert_eq!(
                    transfer_tests::world_grid(&engine, &sims[0]),
                    transfer_tests::world_grid(&engine, &sims[1])
                );
            }
            for held in 0..3 {
                let mut dispatch_counts = [0; 2];
                for arm in 0..2 {
                    let allocations = DeviceScope::start().unwrap();
                    let traffic = TransferScope::start().unwrap();
                    let dispatches = DispatchScope::start().unwrap();
                    transfer(&engine, &mut sims[arm]);
                    assert_eq!(allocations.finish().observations, 0);
                    assert_eq!(traffic.finish().readback_bytes, 0);
                    dispatch_counts[arm] = dispatches.finish().encoded_dispatches;
                }
                assert_eq!(dispatch_counts[1], dispatch_counts[0] + 2);
                assert_eq!(
                    transfer_tests::world_grid(&engine, &sims[0]),
                    transfer_tests::world_grid(&engine, &sims[1]),
                    "held mode={mode} shared={shared} repeat={held}"
                );
            }
            let stale = sims[1].prepare_step(0.0001, 1).unwrap();
            sims[1]
                .prepare_deterministic_p2g_with_budget(u64::MAX)
                .unwrap();
            assert!(!sims[1].deterministic_p2g_grid_reuse_enabled());
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
            drop(batch);
            println!(
                "grid-reuse mode={mode} shared={shared} evolved_exact=8 held_exact=3 additional_dispatches=2 bytes={required}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; departed particles, empty tile reactivation and changed inputs"]
fn gather_grid_reuse_gpu_removal_reactivation_and_inputs() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut sims = [false, true].map(|reuse| seed(&engine, usize::from(sparse), reuse, false));
        let cfg = sims[0].cfg;
        let point = |c: f32| std::array::from_fn(|a| cfg.grid_origin[a] + c * cfg.cell_size);
        let a = point(5.25);
        let b = point(22.25);
        for (case, (points, rigid)) in [
            (vec![a, b], false),
            (vec![a], false),
            (vec![a], true),
            (vec![b, a], true),
            (vec![], true),
            (vec![a, b], false),
        ]
        .into_iter()
        .enumerate()
        {
            for sim in &mut sims {
                sim.set_particles(queue, &points).unwrap();
                sim.set_materials(&[cfg.rheology], vec![0; points.len()])
                    .unwrap();
                sim.set_rigid_spheres(
                    queue,
                    &if rigid {
                        vec![(b, cfg.cell_size, [0.0; 3])]
                    } else {
                        vec![]
                    },
                )
                .unwrap();
                transfer(&engine, sim);
            }
            assert_eq!(
                transfer_tests::world_grid(&engine, &sims[0]),
                transfer_tests::world_grid(&engine, &sims[1]),
                "removed sparse={sparse} case={case}"
            );
        }
        for changed in 0u32..25 {
            for sim in &mut sims {
                let value = (0.13 + changed as f32 * 0.01).to_bits();
                match changed {
                    0..=2 => {
                        let pos = b[changed as usize].to_bits();
                        queue.write_buffer(
                            &sim.p_pos,
                            u64::from(changed * 4),
                            bytemuck::cast_slice(&[pos]),
                        );
                    }
                    3..=5 => queue.write_buffer(
                        &sim.p_vel,
                        u64::from((changed - 3) * 4),
                        bytemuck::cast_slice(&[value]),
                    ),
                    6 => queue.write_buffer(&sim.p_mass, 0, bytemuck::cast_slice(&[value])),
                    7..=15 => {
                        let i = changed - 7;
                        queue.write_buffer(
                            &sim.p_c,
                            u64::from(i / 3 * 16 + i % 3 * 4),
                            bytemuck::cast_slice(&[value]),
                        );
                    }
                    _ => {
                        let i = changed - 16;
                        queue.write_buffer(
                            &sim.p_f,
                            u64::from(i / 3 * 16 + i % 3 * 4),
                            bytemuck::cast_slice(&[value]),
                        );
                    }
                }
                transfer(&engine, sim);
            }
            assert_eq!(
                transfer_tests::world_grid(&engine, &sims[0]),
                transfer_tests::world_grid(&engine, &sims[1]),
                "input sparse={sparse} changed={changed}"
            );
        }
        for sim in &mut sims {
            sim.move_soil_grid_window_cells_with(queue, [1, 0, -1], 129)
                .unwrap();
            transfer(&engine, sim);
        }
        assert_eq!(
            transfer_tests::world_grid(&engine, &sims[0]),
            transfer_tests::world_grid(&engine, &sims[1]),
            "window sparse={sparse}"
        );
        println!(
            "grid-reuse sparse={sparse} removal_reactivation_cases=6 changed_inputs=25 integer_window_changed=true exact=true"
        );
    }
}

#[test]
#[ignore = "requires GPU; unchanged transfer bypasses deliberately faulting gather"]
fn gather_grid_reuse_gpu_hit_bypasses_gather() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = seed(&engine, 0, true, false);
    transfer(&engine, &mut sim);
    transfer(&engine, &mut sim);
    let text = program(false, true, false);
    let marker = "    if reuse_hit != 0u { reuse_restore_node(tile_id, lane, gid); return; }";
    assert_eq!(text.matches(marker).count(), 1);
    let text = text.replace(marker, &format!("{marker}\n    numerical_fault();"));
    let gather = sim.deterministic_p2g.as_ref().unwrap();
    let layouts =
        std::array::from_fn::<_, 4, _>(|i| gather.pipelines[1].get_bind_group_layout(i as u32));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reuse miss fault"),
        source: wgpu::ShaderSource::Wgsl(text.into()),
    });
    sim.deterministic_p2g.as_mut().unwrap().pipelines[1] =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reuse miss fault"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("gather_grid_masked"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("ACTIVE_GRID", 1.0)],
                ..Default::default()
            },
            cache: None,
        });
    transfer(&engine, &mut sim);
    assert_eq!(
        read_words(device, queue, &sim.active_tiles.list)
            [sim.active_tiles.metadata_offset as usize / 4 + 2],
        0
    );
    assert!(
        read_words(device, queue, &sim.active_tiles.list)
            [sim.active_tiles.metadata_offset as usize / 4]
            > 0
    );
    queue.write_buffer(&sim.p_vel, 48, bytemuck::cast_slice(&[0.5f32]));
    transfer(&engine, &mut sim);
    assert_ne!(
        read_words(device, queue, &sim.active_tiles.list)
            [sim.active_tiles.metadata_offset as usize / 4 + 2],
        0
    );
    println!("grid-reuse actual_unchanged_hit=true changed_velocity_miss=true");
}

fn storage_words(engine: &Engine, source: &wgpu::Buffer) -> Vec<u32> {
    let device = engine.device().unwrap();
    let copied = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> a:array<u32>; @group(0) @binding(1) var<storage,read_write> b:array<u32>; @compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3<u32>) { if id.x<arrayLength(&a) { b[id.x]=a[id.x]; } }".into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("copy"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[buffer_entry(0, source), buffer_entry(1, &copied)],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((source.size() / 4).div_ceil(64) as u32, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    read_words(device, engine.queue().unwrap(), &copied)
}

#[test]
#[ignore = "requires GPU; faulted preparation cannot advance reuse keys"]
fn gather_grid_reuse_gpu_fault_does_not_commit_keys() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sims = [false, true].map(|reuse| seed(&engine, 1, reuse, false));
    for sim in &mut sims {
        transfer(&engine, sim);
        transfer(&engine, sim);
    }
    let before = storage_words(
        &engine,
        &sims[1]
            .deterministic_p2g
            .as_ref()
            .unwrap()
            .grid_reuse
            .as_ref()
            .unwrap()
            .keys,
    );
    for sim in &sims {
        queue.write_buffer(&sim.p_vel, 48, bytemuck::cast_slice(&[0.5f32]));
    }
    let sim = &sims[1];
    queue.write_buffer(
        &sim.active_tiles.list,
        sim.active_tiles.metadata_offset + 8,
        bytemuck::cast_slice(&[2u32]),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        sim.deterministic_p2g
            .as_ref()
            .unwrap()
            .encode(&mut pass, sim);
    }
    queue.submit([encoder.finish()]);
    assert_eq!(
        storage_words(
            &engine,
            &sim.deterministic_p2g
                .as_ref()
                .unwrap()
                .grid_reuse
                .as_ref()
                .unwrap()
                .keys
        ),
        before
    );
    queue.write_buffer(
        &sim.active_tiles.list,
        sim.active_tiles.metadata_offset + 8,
        bytemuck::cast_slice(&[0u32]),
    );
    for sim in &mut sims {
        transfer(&engine, sim);
    }
    assert_eq!(
        transfer_tests::world_grid(&engine, &sims[0]),
        transfer_tests::world_grid(&engine, &sims[1])
    );
    assert_ne!(
        storage_words(
            &engine,
            &sims[1]
                .deterministic_p2g
                .as_ref()
                .unwrap()
                .grid_reuse
                .as_ref()
                .unwrap()
                .keys
        ),
        before
    );
    println!(
        "grid-reuse injected_preparation_fault_preserves_keys=true later_valid_transfer_exact=true"
    );
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires timestamp GPU and frozen 50k fixture; not tractor acceptance"]
fn gather_grid_reuse_profile_50k() {
    stress_cache::profile_tests::profile_50k(
        |sim| {
            sim.prepare_deterministic_p2g_grid_reuse(64_000_000, false, false)
                .unwrap();
        },
        "grid-reuse",
    );
}
