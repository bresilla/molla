use super::*;
use crate::mpm::replay_tests::grid_buffers;

pub(super) fn capture_inputs(engine: &Engine, sim: &MpmSimulation, path: &std::path::Path) {
    std::fs::create_dir(path).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (i, b) in grid_buffers(sim)
        .into_iter()
        .chain([
            &sim.uniform_buf,
            &sim.rigid_center,
            &sim.rigid_vel,
            &sim.active_tiles.indirect,
        ])
        .chain(sim.checkpoint_buffers())
        .enumerate()
    {
        std::fs::write(
            path.join(format!("input-{i}.bin")),
            bytemuck::cast_slice(&read_words(device, queue, b)),
        )
        .unwrap();
    }
    std::fs::write(
        path.join("wrench.bin"),
        bytemuck::cast_slice(&read_words(device, queue, &sim.rigid_wrench)),
    )
    .unwrap();
    let stats = sim.read_active_grid_status_with(device, queue).unwrap();
    let metadata = serde_json::json!({"grid_size":sim.cfg.grid_size,"cell_size":sim.cfg.cell_size,"grid_origin":sim.cfg.grid_origin,"particles":sim.particle_count,"rigids":sim.rigid_count,"sparse":sim.sparse_grid,"active":sim.use_active_grid,"two_pass":sim.use_two_pass_p2g,"precontact_compensated":sim.compensated_active_transfer.is_some(),"ordered_shell":sim.ordered_native_shell_forces,"completed_substeps":stats.completed_substeps,"active_tiles":stats.active_tiles,"tile_capacity":stats.tile_capacity});
    std::fs::write(
        path.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
}

fn term_source() -> String {
    let mut source = super::super::shader_source();
    let start = source.find("fn mpm_rigid_force_integrate(").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let mut body =
        source[start..end].replace("fn mpm_rigid_force_integrate(", "fn native_force_terms(");
    for (axis, value) in [
        (0, "force.x"),
        (1, "force.y"),
        (2, "force.z"),
        (4, "torque.x"),
        (5, "torque.y"),
        (6, "torque.z"),
    ] {
        let from = format!("atomic_add_rigid_wrench(base + {axis}u, {value});");
        assert_eq!(body.matches(&from).count(), 1);
        body = body.replace(&from, &format!("probe_terms[world*8u+{axis}u]={value};"));
    }
    source.replace_range(start..end, &body);
    source.push_str("\n@group(3) @binding(3) var<storage,read_write> probe_terms:array<f32>;\n");
    source
}

#[test]
fn native_force_terms_preserve_scalar_expressions() {
    let original = super::super::shader_source();
    let traced = term_source();
    for text in [
        "let force = -(m * w / max(u.dt, 1.0e-6)) * dv;",
        "let torque = cross(arm, force);",
        "if (g_rigid_mask[cidx] != 0u) { return; }",
    ] {
        assert!(original.contains(text) && traced.contains(text));
    }
    assert_eq!(traced.matches("probe_terms[world*8u+").count(), 6);
}

#[test]
#[ignore = "requires native float-atomic GPU and fresh capture path; not full parity acceptance"]
fn native_rigid_identical_grid_diagnostic() {
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_NATIVE_FORCE_REPLAY").expect("fresh output directory required"),
    );
    run(output, None);
}

#[test]
#[ignore = "requires saved first-divergence inputs and native float-atomic GPU; not parity acceptance"]
fn native_rigid_grid_substitution_diagnostic() {
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_NATIVE_FORCE_REPLAY").expect("fresh output directory required"),
    );
    let input = std::path::PathBuf::from(
        std::env::var_os("MOLLA_NATIVE_FORCE_INPUT").expect("first-divergence capture required"),
    );
    std::fs::create_dir(&output).unwrap();
    for mode in ["control", "mass", "velocity", "momentum", "candidate"] {
        run(output.join(mode), Some((&input, mode)));
    }
}

fn run(output: std::path::PathBuf, input: Option<(&std::path::Path, &str)>) {
    std::fs::create_dir(&output).unwrap();
    let engine = engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new_native_explicit(&engine, config(35)).unwrap();
    sim.set_active_grid(true).unwrap();
    let points = points(-0.2);
    sim.set_particles_with_velocity(queue, &points, &vec![[0.02, -0.01, 0.005]; points.len()])
        .unwrap();
    sim.set_materials(
        &[
            MpmRheology::Sand {
                friction_angle: 0.5,
            },
            MpmRheology::Snow {
                theta_c: 0.025,
                theta_s: 0.0075,
                hardening: 10.0,
            },
            MpmRheology::Viscous {
                velocity_damping: 0.1,
                affine_damping: 0.2,
            },
        ],
        (0..129).map(|i| i % 3).collect(),
    )
    .unwrap();
    sim.set_rigid_spheres(queue, &[([-0.15; 3], 0.035, [0.01, 0.0, 0.0])])
        .unwrap();
    if let Some((path, mode)) = input {
        assert!(matches!(
            mode,
            "control" | "mass" | "velocity" | "momentum" | "candidate"
        ));
        for variant in ["control", "candidate"] {
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path.join(variant).join("metadata.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(meta["grid_size"], 35);
            assert_eq!(meta["particles"], 129);
            assert_eq!(meta["rigids"], 1);
            assert_eq!(meta["sparse"], false);
            assert_eq!(meta["cell_size"], serde_json::json!(sim.cfg.cell_size));
            assert_eq!(meta["grid_origin"], serde_json::json!(sim.cfg.grid_origin));
            assert_eq!(meta["two_pass"], false);
        }
        for i in [6, 7, 8] {
            assert_eq!(
                std::fs::read(path.join("control").join(format!("input-{i}.bin"))).unwrap(),
                std::fs::read(path.join("candidate").join(format!("input-{i}.bin"))).unwrap(),
                "shared uniform/rigid input {i}"
            );
        }
        let buffers: Vec<_> = grid_buffers(&sim)
            .into_iter()
            .chain([
                &sim.uniform_buf,
                &sim.rigid_center,
                &sim.rigid_vel,
                &sim.active_tiles.indirect,
            ])
            .chain(sim.checkpoint_buffers())
            .collect();
        let loaded: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let candidate = matches!(i, 4 | 9)
                    || mode == "candidate"
                    || (mode == "mass" && i == 0)
                    || (mode == "velocity" && i == 2)
                    || (mode == "momentum" && i == 1);
                let variant = if candidate { "candidate" } else { "control" };
                let bytes =
                    std::fs::read(path.join(variant).join(format!("input-{i}.bin"))).unwrap();
                assert_eq!(bytes.len() as u64, b.size(), "input {i} size");
                bytes
            })
            .collect();
        for (b, bytes) in buffers.iter().zip(&loaded) {
            queue.write_buffer(b, 0, bytes);
        }
        sim.active_grid_needs_reset = false;
    } else {
        sim.step_n_with(device, queue, 0.0001, 4).unwrap();
    }
    assert!(
        !sim.read_active_grid_status_with(device, queue)
            .unwrap()
            .faulted()
    );
    let inputs: Vec<_> = grid_buffers(&sim)
        .into_iter()
        .chain([
            &sim.uniform_buf,
            &sim.rigid_center,
            &sim.rigid_vel,
            &sim.active_tiles.indirect,
        ])
        .chain(sim.checkpoint_buffers())
        .collect();
    let saved: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let words = read_words(device, queue, b);
            std::fs::write(
                output.join(format!("input-{i}.bin")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
            words
        })
        .collect();
    let terms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per-world-cell force terms"),
        size: u64::from(sim.cfg.grid_size.pow(3)) * 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let rigid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &mixed_storage_entries(&[true, true, false, false]),
    });
    let rigid = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &rigid_layout,
        entries: &[
            buffer_entry(0, &sim.rigid_center),
            buffer_entry(1, &sim.rigid_vel),
            buffer_entry(2, &sim.rigid_wrench),
            buffer_entry(3, &terms),
        ],
    });
    let uniforms = sim.pipe_reset.get_bind_group_layout(0);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            Some(&uniforms),
            Some(&sim.layout_particles),
            Some(&sim.layout_grid),
            Some(&rigid_layout),
        ],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("native force contributions"),
        source: wgpu::ShaderSource::Wgsl(term_source().into()),
    });
    let pipelines = [false, true].map(|active| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("native_force_terms"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("ACTIVE_GRID", f64::from(active)), ("SPARSE_GRID", 0.0)],
                ..Default::default()
            },
            cache: None,
        })
    });
    let dispatch = |pass: &mut wgpu::ComputePass<'_>, active: bool| {
        if active {
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        } else {
            pass.dispatch_workgroups(sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    };
    let mut term_results = Vec::new();
    for active in [false, true] {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&terms, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_bind_group(3, &rigid, &[]);
            pass.set_pipeline(&pipelines[usize::from(active)]);
            dispatch(&mut pass, active);
        }
        queue.submit([encoder.finish()]);
        let words = read_words(device, queue, &terms);
        std::fs::write(
            output.join(format!("terms-{}.bin", usize::from(active))),
            bytemuck::cast_slice(&words),
        )
        .unwrap();
        term_results.push(words);
    }
    assert_eq!(
        term_results[0], term_results[1],
        "per-cell force/torque differs across dispatch modes"
    );
    let mut exact = [0.0f64; 8];
    let mut contributors = 0;
    for cell in term_results[0].chunks_exact(8) {
        if cell.iter().any(|&x| f32::from_bits(x) != 0.0) {
            contributors += 1;
        }
        for i in 0..8 {
            let value = f32::from_bits(cell[i]);
            assert!(value.is_finite());
            exact[i] += f64::from(value);
        }
    }
    assert!(contributors > 0);
    let mut samples = Vec::new();
    let mut largest_pair = 0.0f64;
    for pair in 0..8 {
        let mut results = [Vec::new(), Vec::new()];
        for active in if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(if active {
                    &sim.active_pipelines[5]
                } else {
                    &sim.pipe_rigid_force
                });
                dispatch(&mut pass, active);
            }
            queue.submit([encoder.finish()]);
            let words = read_words(device, queue, &sim.rigid_wrench);
            std::fs::write(
                output.join(format!("wrench-{pair}-{}.bin", usize::from(active))),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
            results[usize::from(active)] = words;
        }
        let mut errors = [0.0f64; 2];
        let mut delta = 0.0f64;
        for i in 0..8 {
            let a = f64::from(f32::from_bits(results[0][i]));
            let b = f64::from(f32::from_bits(results[1][i]));
            assert!(a.is_finite() && b.is_finite());
            errors[0] = errors[0].max((a - exact[i]).abs());
            errors[1] = errors[1].max((b - exact[i]).abs());
            delta = delta.max((a - b).abs());
        }
        largest_pair = largest_pair.max(delta);
        samples.push(serde_json::json!({"pair":pair,"max_absolute_delta":delta,"bound":2e-5,"passed":delta<=2e-5,"max_error_from_f64_term_sum":errors}));
    }
    for (before, b) in saved.iter().zip(inputs) {
        assert_eq!(
            before,
            &read_words(device, queue, b),
            "replay changed grid, metadata or particle input"
        );
    }
    let report = serde_json::json!({"scope":"identical post-step dense-backed grid, shell force stage only; not original failed trajectory or full coupling acceptance","substitution":input.map(|(_,mode)|mode),"layout_source":if input.is_some(){"captured candidate tile map and indirect dimensions for all variants"}else{"generated active map"},"contributors":contributors,"exact_term_bytes":term_results[0].len()*4,"term_sum_f64":exact,"max_pair_delta":largest_pair,"pairs":samples});
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!(
        "identical native force grid: {contributors} contributors, exact per-cell terms, max paired difference {largest_pair:e}, 8 alternating pairs; all inputs unchanged"
    );
}
