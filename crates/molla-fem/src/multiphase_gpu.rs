//! GPU level-set curvature + CSF surface-tension force. Mirrors the CPU
//! [`level_set_curvature_2d`](crate::multiphase::level_set_curvature_2d)
//! and [`csf_surface_tension_force_2d`](crate::multiphase::csf_surface_tension_force_2d):
//! three per-cell passes (normals → curvature → force) on a periodic
//! 2D grid.

use molla_core::{Error, Result};
use molla_math::Real;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dims: [u32; 4],
    params: [f32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU curvature `κ = ∇·(∇φ/|∇φ|)` on a periodic 2D grid.
pub fn level_set_curvature_2d_gpu(
    engine: &molla_compute::Engine,
    phi: &[Real],
    nx: usize,
    ny: usize,
    dx: Real,
) -> Result<Vec<Real>> {
    let (kappa, _) = run(engine, phi, nx, ny, dx, 0.0, false)?;
    Ok(kappa)
}

/// GPU CSF body force `f = σ κ δ_ε(φ) ∇φ` on a periodic 2D grid.
pub fn csf_surface_tension_force_2d_gpu(
    engine: &molla_compute::Engine,
    phi: &[Real],
    nx: usize,
    ny: usize,
    dx: Real,
    sigma: Real,
) -> Result<Vec<[Real; 2]>> {
    let (_, force) = run(engine, phi, nx, ny, dx, sigma, true)?;
    Ok(force.unwrap_or_default())
}

fn run(
    engine: &molla_compute::Engine,
    phi: &[Real],
    nx: usize,
    ny: usize,
    dx: Real,
    sigma: Real,
    want_force: bool,
) -> Result<(Vec<Real>, Option<Vec<[Real; 2]>>)> {
    let count = nx * ny;
    if nx == 0 || ny == 0 || phi.len() != count || dx <= 0.0 {
        return Err(Error::Fem(format!(
            "multiphase_gpu: bad grid (len {}, nx {nx}, ny {ny}, dx {dx})",
            phi.len()
        )));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("multiphase_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("multiphase_gpu: engine has no queue".into()))?
        .clone();

    let phi_f32: Vec<f32> = phi.iter().map(|&x| x as f32).collect();
    let phi_buf = make_storage(&device, "molla::mp phi", bytemuck::cast_slice(&phi_f32));
    queue.write_buffer(&phi_buf, 0, bytemuck::cast_slice(&phi_f32));
    // Device-only normal field (vec2 per cell).
    let nfield_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::mp nfield"),
        size: (count * 8).max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut kappa = Buffer::from_vec(vec![0.0 as Real; count], BufferUsage::StorageReadWrite);
    kappa.ensure_device(&device)?;
    kappa.upload(&queue)?;
    let mut force = Buffer::from_vec(vec![0.0 as Real; count * 2], BufferUsage::StorageReadWrite);
    force.ensure_device(&device)?;
    force.upload(&queue)?;

    let eps = 1.5 * dx;
    let uniforms = Uniforms {
        dims: [nx as u32, ny as u32, count as u32, 0],
        params: [dx as f32, sigma as f32, eps as f32, 0.0],
    };

    let kernel = MultiphaseKernel::new(&device);
    let bg = kernel.bind_group(
        &device,
        &phi_buf,
        &nfield_buf,
        kappa.device_buffer().unwrap(),
        force.device_buffer().unwrap(),
    );
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::mp uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::mp uniform bg"),
        layout: &kernel.layout_uniform,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut passes = vec![&kernel.normals, &kernel.curvature];
    if want_force {
        passes.push(&kernel.csf);
    }
    for pipeline in passes {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mp encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::mp pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &bg_uniform, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    kappa.download(&device, &queue)?;
    let kappa_out = kappa.host()?.to_vec();
    let force_out = if want_force {
        force.download(&device, &queue)?;
        let flat = force.host()?;
        Some((0..count).map(|k| [flat[2 * k], flat[2 * k + 1]]).collect())
    } else {
        None
    };
    Ok((kappa_out, force_out))
}

struct MultiphaseKernel {
    normals: wgpu::ComputePipeline,
    curvature: wgpu::ComputePipeline,
    csf: wgpu::ComputePipeline,
    layout_data: wgpu::BindGroupLayout,
    layout_uniform: wgpu::BindGroupLayout,
}

impl MultiphaseKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::multiphase"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/multiphase.wgsl").into()),
        });
        let mut data_entries = vec![entry(0, true)];
        data_entries.push(entry(1, false));
        data_entries.push(entry(2, false));
        data_entries.push(entry(3, false));
        let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mp data"),
            entries: &data_entries,
        });
        let layout_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mp uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::mp layout"),
            bind_group_layouts: &[Some(&layout_data), Some(&layout_uniform)],
            immediate_size: 0,
        });
        let mk = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::mp pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            normals: mk("normals"),
            curvature: mk("curvature"),
            csf: mk("csf"),
            layout_data,
            layout_uniform,
        }
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        phi: &wgpu::Buffer,
        nfield: &wgpu::Buffer,
        kappa: &wgpu::Buffer,
        force: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mp data bg"),
            layout: &self.layout_data,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: phi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nfield.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kappa.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: force.as_entire_binding(),
                },
            ],
        })
    }
}

fn entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
