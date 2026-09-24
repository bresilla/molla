#![cfg(feature = "gpu")]

use molla_compute::Engine;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

#[test]
#[ignore = "requires GPU and captured local coupled fixtures through make test-gpu"]
fn captured_patch_residuals_match_independent_cpu() {
    let root = PathBuf::from(std::env::var_os("MOLLA_LOCAL_MATERIAL_FIXTURE").unwrap()).join("-1");
    let patch = PathBuf::from(std::env::var_os("MOLLA_LOCAL_COUPLED_INPUT").unwrap());
    let expected = std::fs::read(patch.join("expected.bin")).unwrap();
    let expected: Vec<f64> = expected
        .chunks_exact(8)
        .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert!(!expected.is_empty() && expected.len() % 15 == 0);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../../molla-kernels/shaders/tet_viscosity.wgsl"),
        include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl"),
        include_str!("local_coupled_probe.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("captured local FEM/contact residual"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let counts = [7, 9, 7, 1, 5, 17];
    let layouts: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(g, &count)| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &(0..count)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if g == 3 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !(g == 2
                                        || (g == 5 && [10, 11, 14].contains(&binding))),
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
        label: Some("local coupled residuals"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("patch_residuals"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers: Vec<Vec<_>> = counts
        .iter()
        .enumerate()
        .map(|(g, &count)| {
            (0..count)
                .map(|b| {
                    let path = root.join(format!("{g}-{b}.bin"));
                    let mut bytes = if g == 2 && b == 5 {
                        std::fs::read(patch.join("input.bin")).unwrap()
                    } else if g == 2 && b == 6 {
                        vec![0; expected.len() * 4]
                    } else if path.exists() {
                        std::fs::read(path).unwrap()
                    } else {
                        vec![0; 64]
                    };
                    if bytes.is_empty() {
                        bytes.resize(64, 0);
                    }
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
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
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
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
    let actual = readback::read::<f32>(&engine, &buffers[2][6], expected.len());
    std::fs::write(
        patch.join("gpu-residuals.bin"),
        bytemuck::cast_slice(&actual),
    )
    .unwrap();
    let maximum_error = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0., f64::max);
    println!(
        "PATCH residual samples={} maximum_cpu_gpu_error={maximum_error}",
        actual.len() / 15
    );
    for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (f64::from(a) - b).abs() <= 2e-5,
            "residual {index}: gpu={a} cpu={b}"
        );
    }
    let root = &actual[15..30];
    for v in [
        &root[..3],
        &root[3..6],
        &root[7..9],
        &root[10..12],
        &root[13..15],
    ] {
        assert!(v.iter().map(|x| x * x).sum::<f32>().sqrt() <= 1e-4);
    }
    assert!([root[6], root[9], root[12]].iter().all(|v| v.abs() <= 1e-4));
    println!("PATCH CPU-selected root residual on GPU: {root:?}; not a GPU nonlinear solve");
    let solver = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GPU local coupled Newton probe"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("patch_solve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let poison = vec![f32::NAN; expected.len() - 15];
    queue.write_buffer(
        &buffers[2][5],
        (192 + 15) * 4,
        bytemuck::cast_slice(&poison),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&solver);
        for (i, group) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let solution = readback::read::<f32>(&engine, &buffers[2][6], 34);
    std::fs::write(
        patch.join("gpu-solution.bin"),
        bytemuck::cast_slice(&solution),
    )
    .unwrap();
    println!(
        "PATCH GPU-selected mask={} iterations={} converged={} residual={} x={:?}",
        solution[30],
        solution[31],
        solution[32],
        solution[33],
        &solution[..15]
    );
    assert!(solution.iter().all(|value| value.is_finite()));
    assert_eq!(solution[32], 1.0);
    assert!(solution[33] <= 1e-5);
}
