#![cfg(feature = "gpu")]

use molla_compute::Engine;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

#[test]
#[ignore = "requires GPU and MOLLA_LINEARIZATION_SOURCE/OUTPUT captured buffers"]
fn captured_ceol_material_operator_preserves_state() {
    let root = PathBuf::from(std::env::var_os("MOLLA_LINEARIZATION_SOURCE").unwrap());
    let output = PathBuf::from(std::env::var_os("MOLLA_LINEARIZATION_OUTPUT").unwrap());
    let particles = std::fs::metadata(root.join("0-0.bin")).unwrap().len() / 16;
    let tets = std::fs::metadata(root.join("5-0.bin")).unwrap().len() / 16;
    assert!(particles > 10000 && tets > 10000);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = concat!(
        include_str!("../../molla-kernels/shaders/tet_viscosity.wgsl"),
        "\n",
        include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl"),
        "\n",
        include_str!("../../molla-kernels/shaders/fem_material_linearization.wgsl")
    )
    .replace(
        "const CACHE_PREVIOUS_TETS: bool = false;",
        "const CACHE_PREVIOUS_TETS: bool = true;",
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("captured CEOL material linearization"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let counts = [7, 9, 5, 1, 5, 17, 4];
    let layouts: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(group, &count)| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &(0..count)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if group == 3 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !(group == 2
                                        || (group == 5 && [10, 11, 14].contains(&binding))
                                        || (group == 6 && binding > 0)),
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
    let pipelines = [
        "cache_previous_tets",
        "material_linearization",
        "cache_material_tangents",
        "material_cached_action",
    ]
    .map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let bytes: Vec<Vec<_>> = counts
        .iter()
        .enumerate()
        .map(|(group, &count)| {
            (0..count)
                .map(|binding| {
                    let path = root.join(format!("{group}-{binding}.bin"));
                    let mut data = if group == 6 {
                        vec![
                            0;
                            if binding == 3 {
                                tets as usize * 432
                            } else {
                                particles as usize * 16
                            }
                        ]
                    } else if path.exists() {
                        std::fs::read(path).unwrap()
                    } else {
                        vec![0; 64]
                    };
                    if group == 2 && binding == 2 {
                        data.resize((particles + 6 * tets) as usize * 16, 0);
                    }
                    if group == 3 {
                        data[16..20].copy_from_slice(&(tets as u32).to_le_bytes());
                    }
                    if data.is_empty() {
                        data.resize(64, 0);
                    }
                    data
                })
                .collect()
        })
        .collect();
    let buffers: Vec<Vec<_>> = bytes
        .iter()
        .enumerate()
        .map(|(group, entries)| {
            entries
                .iter()
                .map(|data| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: data,
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                            | if group == 3 {
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
        .map(|(group, entries)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts[group],
                entries: &entries
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    for case in 0..2 {
        let direction = std::fs::read(output.join(format!("direction-{case}.bin"))).unwrap();
        assert_eq!(direction.len(), particles as usize * 16);
        queue.write_buffer(&buffers[6][0], 0, &direction);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (group, bindings) in groups.iter().enumerate() {
                pass.set_bind_group(group as u32, bindings, &[]);
            }
            pass.set_pipeline(&pipelines[0]);
            pass.dispatch_workgroups((tets as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&pipelines[1]);
            for start in (0..particles as u32).step_by(512) {
                let mut params = bytes[3][0].clone();
                params[12..16].copy_from_slice(&start.to_le_bytes());
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("captured material batch"),
                    contents: &params,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let batch = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[3],
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                pass.set_bind_group(3, &batch, &[]);
                pass.dispatch_workgroups((particles as u32 - start).min(512).div_ceil(64), 1, 1);
            }
        }
        queue.submit([encoder.finish()]);
        for (name, binding) in [("residual", 1), ("action", 2)] {
            let result =
                readback::read::<[f32; 4]>(&engine, &buffers[6][binding], particles as usize);
            assert!(result.iter().flatten().all(|value| value.is_finite()));
            std::fs::write(
                output.join(format!("{name}-{case}.bin")),
                bytemuck::cast_slice(&result),
            )
            .unwrap();
        }
        let direct = readback::read::<[f32; 4]>(&engine, &buffers[6][2], particles as usize);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (group, bindings) in groups.iter().enumerate() {
                pass.set_bind_group(group as u32, bindings, &[]);
            }
            pass.set_pipeline(&pipelines[2]);
            for start in (0..tets as u32).step_by(512) {
                let count = (tets as u32 - start).min(512);
                let mut params = bytes[3][0].clone();
                params[12..16].copy_from_slice(&start.to_le_bytes());
                params[16..20].copy_from_slice(&count.to_le_bytes());
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("captured tangent batch"),
                    contents: &params,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let batch = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[3],
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                pass.set_bind_group(3, &batch, &[]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            }
            pass.set_bind_group(3, &groups[3], &[]);
            pass.set_pipeline(&pipelines[3]);
            pass.dispatch_workgroups((particles as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let cached = readback::read::<[f32; 4]>(&engine, &buffers[6][2], particles as usize);
        let mut maximum_difference = 0.0_f32;
        for (a, b) in cached.iter().flatten().zip(direct.iter().flatten()) {
            assert!(
                a.is_finite() && (*a - *b).abs() <= 2e-5 * (1.0 + b.abs()),
                "cached={a} direct={b}"
            );
            maximum_difference = maximum_difference.max((*a - *b).abs());
        }
        std::fs::write(
            output.join(format!("action-{case}.bin")),
            bytemuck::cast_slice(&cached),
        )
        .unwrap();
        println!("cached tangent case={case} maximum_action_difference={maximum_difference}");
        for (group, entries) in buffers.iter().enumerate() {
            if group == 3 || group == 6 {
                continue;
            }
            for (binding, buffer) in entries.iter().enumerate() {
                let count = if group == 2 && binding == 2 {
                    particles as usize * 16
                } else {
                    bytes[group][binding].len()
                };
                assert_eq!(
                    readback::read::<u8>(&engine, buffer, count),
                    bytes[group][binding][..count],
                    "group={group} binding={binding}"
                );
            }
        }
        println!(
            "CEOL material operator case={case} particles={particles} tets={tets}; state preserved, independent oracle pending"
        );
    }
}
