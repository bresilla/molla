#![cfg(feature = "gpu")]

use molla_compute::Engine;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

#[test]
#[ignore = "requires GPU and captured patch buffers through make test-gpu"]
fn parallel_patch_matches_serial_and_preserves_material_state() {
    let root = PathBuf::from(std::env::var_os("MOLLA_PATCH_MATERIAL").unwrap());
    let input = PathBuf::from(std::env::var_os("MOLLA_PATCH_INPUT").unwrap());
    let golden = std::fs::read(input.with_file_name("gpu-solution.bin")).unwrap();
    assert_eq!(golden.len(), 34 * 4);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = format!(
        "{}\n{}\n{}\n@compute @workgroup_size(1) fn serial_patch() {{ if patch_input[191] != 0.0 {{ patch_solve_impl(); }} }}",
        include_str!("../../molla-kernels/shaders/tet_viscosity.wgsl"),
        include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl"),
        include_str!("../../molla-kernels/shaders/fem_contact_patch.wgsl"),
    ).replace("const LOCAL_MATERIAL_TRIAL: bool = false;", "const LOCAL_MATERIAL_TRIAL: bool = true;");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("parallel material/contact patch regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let counts = [7, 9, 5, 1, 5, 17, 8];
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
                                        || (group == 6 && [0, 3, 4, 6, 7].contains(&binding))),
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
    let pipelines = ["serial_patch", "patch_live_solve"].map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let mut metadata = std::fs::read(input).unwrap();
    assert!(metadata.len() >= 207 * 4);
    metadata.truncate(207 * 4);
    metadata[191 * 4..192 * 4].copy_from_slice(&1.0_f32.to_le_bytes());
    let bytes: Vec<Vec<Vec<u8>>> = counts
        .iter()
        .enumerate()
        .map(|(group, &count)| {
            (0..count)
                .map(|binding| {
                    let mut bytes = if group == 6 {
                        match binding {
                            6 => metadata.clone(),
                            7 => vec![0; 37 * 4],
                            _ => vec![0; 80],
                        }
                    } else {
                        let path = root.join(format!("{group}-{binding}.bin"));
                        if path.exists() {
                            std::fs::read(path).unwrap()
                        } else {
                            vec![0; 64]
                        }
                    };
                    if bytes.is_empty() {
                        bytes.resize(64, 0);
                    }
                    bytes
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
                .map(|bytes| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytes,
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
    let mut baseline = None;
    for (index, pipeline) in pipelines.iter().enumerate() {
        for (repeat, enabled) in [true, true, true, false].into_iter().enumerate() {
            queue.write_buffer(&buffers[6][7], 0, &[0; 37 * 4]);
            queue.write_buffer(
                &buffers[6][6],
                191 * 4,
                bytemuck::bytes_of(&f32::from(enabled)),
            );
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                for (group, binding) in groups.iter().enumerate() {
                    pass.set_bind_group(group as u32, binding, &[]);
                }
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit([encoder.finish()]);
            let result = readback::read::<f32>(&engine, &buffers[6][7], 37);
            println!(
                "PATCH parallel={} repeat={repeat} enabled={enabled} wall={:?} residual={}",
                index == 1,
                start.elapsed(),
                result[33]
            );
            for (group, entries) in buffers.iter().enumerate() {
                for (binding, buffer) in entries.iter().enumerate() {
                    if group == 6 || group == 3 {
                        continue;
                    }
                    let actual = readback::read::<u8>(&engine, buffer, bytes[group][binding].len());
                    assert_eq!(
                        actual, bytes[group][binding],
                        "state changed group={group} binding={binding}"
                    );
                }
            }
            if !enabled {
                assert_eq!(result, vec![0.0; 37]);
            } else {
                assert!(result.iter().all(|value| value.is_finite()));
                assert_eq!(result[32], 1.0);
                assert!(result[33] <= 1e-5);
                assert_eq!(bytemuck::cast_slice::<f32, u8>(&result[..34]), golden);
                let bits: Vec<_> = result.iter().map(|value| value.to_bits()).collect();
                if let Some(expected) = &baseline {
                    assert_eq!(&bits, expected);
                } else {
                    baseline = Some(bits);
                }
                if index == 1
                    && let Some(path) = std::env::var_os("MOLLA_PATCH_OUTPUT")
                {
                    std::fs::write(path, bytemuck::cast_slice(&result[..34])).unwrap();
                }
            }
        }
    }
}
