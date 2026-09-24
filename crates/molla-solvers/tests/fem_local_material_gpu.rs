#![cfg(feature = "gpu")]

use molla_compute::Engine;
use std::path::Path;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

const PROBE: &str = r#"
@group(2) @binding(5) var<storage, read_write> probe_results: array<vec4<f32>>;
@group(2) @binding(6) var<storage, read_write> probe_tets: array<vec4<f32>>;
@compute @workgroup_size(1)
fn local_probe() {
    let v = u.color_index;
    for (var i = 0u; i < 129u; i++) {
        solve_vertex(v, true);
        probe_results[i] = vec4<f32>(particle_delta[v].xyz, inertia_y[v].w);
        if i < 128u { solve_vertex(v, false); }
    }
}
@compute @workgroup_size(1)
fn contribution_probe() {
    let v = u.color_index;
    let start = vertex_adj_tets_offsets[v];
    let end = vertex_adj_tets_offsets[v + 1u];
    for (var t = start; t < end; t += 2u) {
        var force: vec3<f32>;
        var hessian: mat3x3<f32>;
        tet_neo_hookean_block(vertex_adj_tets[t], vertex_adj_tets[t + 1u], true, &force, &hessian);
        probe_results[(t - start) / 2u] = vec4<f32>(force, 0.0);
    }
}
"#;

#[test]
#[ignore = "requires GPU and MOLLA_LOCAL_MATERIAL_FIXTURE through make test-gpu"]
fn captured_node_updates_with_fixed_neighbors() {
    let root = std::env::var("MOLLA_LOCAL_MATERIAL_FIXTURE").expect("native fixture directory");
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut source = format!(
        "{}\n{}\n{PROBE}",
        include_str!("../../molla-kernels/shaders/tet_viscosity.wgsl"),
        include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl"),
    );
    let detail = std::env::var_os("MOLLA_LOCAL_TET_DETAIL").is_some();
    if detail {
        source = source.replace("let cof = cofactor3(f_mat);", "let record = (tet_id * 4u + v_order) * 12u; probe_tets[record] = vec4<f32>(f_mat[0], 0.0); probe_tets[record + 1u] = vec4<f32>(f_mat[1], 0.0); probe_tets[record + 2u] = vec4<f32>(f_mat[2], 0.0); let cof = cofactor3(f_mat);");
        source = source.replace("*out_force = f_clean;", "probe_tets[record + 3u] = vec4<f32>(f_clean, 0.0); probe_tets[record + 11u] = vec4<f32>(m, 0.0); *out_force = f_clean;");
        source = source.replace("*out_force += viscous.force;", "probe_tets[record + 4u] = vec4<f32>(viscous.force, 0.0); for (var column = 0u; column < 3u; column++) { probe_tets[record + 5u + column] = vec4<f32>(previous[column], 0.0); probe_tets[record + 8u + column] = vec4<f32>(increment[column], 0.0); } *out_force += viscous.force;");
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fixed-neighbor native material probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let counts = [7, 9, 7, 1, 5, 17];
    let writable = |g, b| g == 2 || (g == 5 && [10, 11, 14].contains(&b));
    let layouts: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(g, &count)| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &(0..count)
                    .map(|b| wgpu::BindGroupLayoutEntry {
                        binding: b,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if g == 3 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !writable(g, b),
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("local_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut nodes: Vec<i32> = std::fs::read_dir(&root).unwrap()
        .map(|entry| entry.unwrap().file_name().to_str().unwrap().parse().unwrap()).collect();
    nodes.sort();
    for node in nodes {
        let directory = Path::new(&root).join(node.to_string());
        if node == -1 && !directory.exists() {
            continue;
        }
        let buffers: Vec<Vec<_>> = counts
            .iter()
            .enumerate()
            .map(|(g, &count)| {
                (0..count)
                    .map(|b| {
                        let path = directory.join(format!("{g}-{b}.bin"));
                        let mut data = if path.exists() {
                            std::fs::read(path).unwrap()
                        } else {
                            vec![0; 64]
                        };
                        if data.is_empty() {
                            data.resize(64, 0);
                        }
                        if g == 2 && b == 5 {
                            data.resize(129 * 16, 0);
                        }
                        if g == 2 && b == 6 && detail {
                            let indices = std::fs::metadata(directory.join("5-0.bin")).unwrap().len();
                            data.resize(indices as usize * 48, 0);
                        }
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: &data,
                            usage: wgpu::BufferUsages::COPY_SRC
                                | if g == 3 {
                                    wgpu::BufferUsages::UNIFORM
                                } else {
                                    wgpu::BufferUsages::STORAGE
                                },
                        })
                    })
                    .collect()
            })
            .collect();
        let groups: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(g, buffers)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[g],
                    entries: &buffers
                        .iter()
                        .enumerate()
                        .map(|(b, buffer)| wgpu::BindGroupEntry {
                            binding: b as u32,
                            resource: buffer.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        if node == -1 {
            global_probe(
                &engine, &directory, &shader, &layout, &layouts, &buffers, &groups,
            );
            continue;
        }
        let contributions = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&layout), module: &shader,
            entry_point: Some("contribution_probe"), compilation_options: Default::default(), cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&contributions);
            for (i, group) in groups.iter().enumerate() { pass.set_bind_group(i as u32, group, &[]); }
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let values = readback::read::<[f32; 4]>(&engine, &buffers[2][5], 129);
        std::fs::write(directory.join("contributions.bin"), bytemuck::cast_slice(&values)).unwrap();
        if detail {
            let values = readback::read::<[f32; 4]>(&engine, &buffers[2][6], buffers[2][6].size() as usize / 16);
            std::fs::write(directory.join("tet-detail.bin"), bytemuck::cast_slice(&values)).unwrap();
        }
        let before =
            readback::read::<[f32; 4]>(&engine, &buffers[2][2], buffers[2][2].size() as usize / 16);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let results = readback::read::<[f32; 4]>(&engine, &buffers[2][5], 129);
        let after = readback::read::<[f32; 4]>(&engine, &buffers[2][2], before.len());
        let uniform = std::fs::read(directory.join("3-0.bin")).unwrap();
        let local = u32::from_le_bytes(uniform[12..16].try_into().unwrap()) as usize;
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            if i != local {
                assert_eq!(a, b, "neighbor changed: node={node} local={i}");
            }
        }
        assert!(results.iter().flatten().all(|v| v.is_finite()));
        let original = std::fs::read(directory.join("2-3.bin")).unwrap();
        let residual = f32::from_le_bytes(
            original[16 * local + 12..16 * local + 16]
                .try_into()
                .unwrap(),
        );
        if std::env::var_os("MOLLA_LOCAL_MATERIAL_REQUIRE_BASELINE").is_some() {
            assert_eq!(
                results[0][3], residual,
                "native residual mismatch at node {node}"
            );
        }
        std::fs::write(directory.join("local-updates.bin"), bytemuck::cast_slice(&results)).unwrap();
        for i in [0, 1, 2, 4, 8, 16, 32, 64, 128] {
            println!(
                "LOCAL node={node} iteration={i} delta_residual={:?}",
                results[i]
            );
        }
    }
}

fn global_probe(
    engine: &Engine,
    directory: &Path,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    layouts: &[wgpu::BindGroupLayout],
    buffers: &[Vec<wgpu::Buffer>],
    groups: &[wgpu::BindGroup],
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let solve = pipeline("solve_color");
    let residual = pipeline("force_residual");
    let offsets = std::fs::read(directory.join("1-6.bin")).unwrap();
    let offsets: Vec<u32> = offsets
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let base = std::fs::read(directory.join("3-0.bin")).unwrap();
    let count = u32::from_le_bytes(base[0..4].try_into().unwrap());
    let uniforms: Vec<_> = offsets
        .windows(2)
        .enumerate()
        .map(|(color, range)| {
            let mut data = base.clone();
            data[12..16].copy_from_slice(&(color as u32).to_le_bytes());
            data[16..20].copy_from_slice(&(range[1] - range[0]).to_le_bytes());
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &data,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts[3],
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
        .collect();
    let initial = std::fs::read(directory.join("2-3.bin")).unwrap();
    let mut previous = 0;
    for iterations in [0, 1, 8, 32, 128, 512] {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.set_pipeline(&solve);
            for _ in previous..iterations {
                for (color, group) in uniforms.iter().enumerate() {
                    pass.set_bind_group(3, group, &[]);
                    pass.dispatch_workgroups(
                        (offsets[color + 1] - offsets[color]).div_ceil(64),
                        1,
                        1,
                    );
                }
            }
            pass.set_bind_group(3, &groups[3], &[]);
            pass.set_pipeline(&residual);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let samples = readback::read::<[f32; 4]>(engine, &buffers[2][3], count as usize);
        let delta = readback::read::<[f32; 4]>(engine, &buffers[2][2], count as usize);
        assert!(
            samples
                .iter()
                .flatten()
                .chain(delta.iter().flatten())
                .all(|v| v.is_finite())
        );
        if iterations == 0 {
            let mut max_difference = 0.0_f32;
            let mut changed = 0;
            for (i, (sample, bytes)) in samples.iter().zip(initial.chunks_exact(16)).enumerate() {
                let expected: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(&sample[..3], &expected[..3], "initial target node={i}");
                max_difference = max_difference.max((sample[3] - expected[3]).abs());
                changed += usize::from(sample[3] != expected[3]);
            }
            println!("GLOBAL baseline changed={changed} max_scalar_difference={max_difference}");
        }
        let (worst, value) = samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[3].total_cmp(&b.1[3]))
            .unwrap();
        println!(
            "GLOBAL iterations={iterations} worst={worst} residual={} node5332={}",
            value[3], samples[5332][3]
        );
        std::fs::write(
            directory.join(format!("delta-{iterations}.bin")),
            bytemuck::cast_slice(&delta),
        )
        .unwrap();
        std::fs::write(
            directory.join(format!("force-{iterations}.bin")),
            bytemuck::cast_slice(&samples),
        )
        .unwrap();
        previous = iterations;
    }
}
