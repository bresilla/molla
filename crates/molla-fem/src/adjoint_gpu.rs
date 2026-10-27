//! GPU adjoint gradient for linear-elasticity Young's modulus. Mirrors
//! the CPU [`compute_grad_youngs`](crate::adjoint::compute_grad_youngs):
//! one thread per tet accumulates `λᵀ K u`; the host sums and divides by
//! `−E`. Tet meshes only (dim = 3).

use crate::mesh::{FemCell, FemMesh};
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    counts: [u32; 4],
    mat: [f32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU counterpart of [`compute_grad_youngs`](crate::adjoint::compute_grad_youngs)
/// for an all-tet mesh. Returns `∂J/∂E`, matching the CPU value within
/// f32 tolerance.
pub fn compute_grad_youngs_gpu(
    engine: &molla_compute::Engine,
    mesh: &FemMesh,
    vertices: &[Vec3],
    u: &[Real],
    lambda: &[Real],
    poisson: Real,
    youngs: Real,
) -> Result<Real> {
    debug_assert_eq!(u.len(), lambda.len(), "u/λ length mismatch");
    debug_assert!(youngs > 0.0, "E must be > 0");

    let mut tets: Vec<u32> = Vec::with_capacity(mesh.cells.len() * 4);
    for cell in &mesh.cells {
        match cell {
            FemCell::Tet(idx) => tets.extend_from_slice(idx),
            FemCell::Tri(_) => {
                return Err(Error::Gpu(
                    "compute_grad_youngs_gpu: tet meshes only (v1)".into(),
                ));
            }
        }
    }
    let cell_count = mesh.cells.len();
    if cell_count == 0 {
        return Ok(0.0);
    }

    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("adjoint_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("adjoint_gpu: engine has no queue".into()))?
        .clone();

    let mut vert_pad: Vec<[f32; 4]> = Vec::with_capacity(vertices.len());
    for v in vertices {
        let g = v.as_vec3();
        vert_pad.push([g.x, g.y, g.z, 0.0]);
    }
    let vert_buf = make_storage(
        &device,
        "molla::adjoint verts",
        bytemuck::cast_slice(&vert_pad),
    );
    queue.write_buffer(&vert_buf, 0, bytemuck::cast_slice(&vert_pad));
    let tet_buf = make_storage(&device, "molla::adjoint tets", bytemuck::cast_slice(&tets));
    queue.write_buffer(&tet_buf, 0, bytemuck::cast_slice(&tets));

    let mut u_buf = Buffer::from_vec(u.to_vec(), BufferUsage::Storage);
    u_buf.ensure_device(&device)?;
    u_buf.upload(&queue)?;
    let mut lam_buf = Buffer::from_vec(lambda.to_vec(), BufferUsage::Storage);
    lam_buf.ensure_device(&device)?;
    lam_buf.upload(&queue)?;

    let mut out = Buffer::from_vec(vec![0.0 as Real; cell_count], BufferUsage::StorageReadWrite);
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let lam_l = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu_l = youngs / (2.0 * (1.0 + poisson));
    let uniforms = Uniforms {
        counts: [cell_count as u32, 0, 0, 0],
        mat: [lam_l as f32, mu_l as f32, 0.0, 0.0],
    };
    dispatch(
        &device, &queue, &vert_buf, &tet_buf, &u_buf, &lam_buf, &mut out, uniforms, cell_count,
    )?;
    out.download(&device, &queue)?;

    let accum: Real = out.host()?.iter().sum();
    Ok(-accum / youngs)
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    verts: &wgpu::Buffer,
    tets: &wgpu::Buffer,
    u: &Buffer<Real>,
    lam: &Buffer<Real>,
    out: &mut Buffer<Real>,
    uniforms: Uniforms,
    cell_count: usize,
) -> Result<()> {
    let kernel = AdjointGradKernel::new(device);
    let u_dev = u
        .device_buffer()
        .ok_or_else(|| Error::Gpu("adjoint u: no device mirror".into()))?;
    let lam_dev = lam
        .device_buffer()
        .ok_or_else(|| Error::Gpu("adjoint λ: no device mirror".into()))?;
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::adjoint in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: verts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: u_dev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lam_dev.as_entire_binding(),
            },
        ],
    });
    let out_dev = out
        .device_buffer()
        .ok_or_else(|| Error::Gpu("adjoint out: no device mirror".into()))?;
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::adjoint out bg"),
        layout: &kernel.layout_out,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_dev.as_entire_binding(),
        }],
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::adjoint uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::adjoint uniforms bg"),
        layout: &kernel.layout_uniforms,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (cell_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::adjoint encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::adjoint pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniforms, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out.mark_device_dirty();
    Ok(())
}

struct AdjointGradKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl AdjointGradKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem_adjoint_grad"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fem_adjoint_grad.wgsl").into(),
            ),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::adjoint in"),
            entries: &storage_entries(4, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::adjoint out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::adjoint uniforms"),
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
            label: Some("molla::adjoint layout"),
            bind_group_layouts: &[Some(&layout_in), Some(&layout_out), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::adjoint pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("adjoint_grad"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_in,
            layout_out,
            layout_uniforms,
        }
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

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
