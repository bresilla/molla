//! Stage 41 v3 — Rust dispatcher for the Taylor-Hood Navier-Stokes
//! **convective RHS** GPU assembly kernel (`fem_ns_advect.wgsl`).
//!
//! Per-cell scatter of `∫ N_i · (u_old·∇)u_old dx` into a global RHS
//! buffer, with optional Hughes SUPG stabilisation on the test
//! function. The higher-level IMEX time stepper lives in
//! `crates/molla-fem/src/stokes_gpu.rs`.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NsAdvectUniforms {
    cell_count: u32,
    n_v: u32,
    dim: u32,
    _pad0: u32,
    viscosity: f32,
    supg_alpha: f32,
    _pad1: f32,
    _pad2: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU dispatcher for the NS convection kernel.
pub struct FemNsAdvectKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline: wgpu::ComputePipeline,

    layout_mesh: wgpu::BindGroupLayout,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FemNsAdvectKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::ns_advect"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_ns_advect.wgsl").into(),
            ),
        });

        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_advect mesh"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2), ro_storage(3)],
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_advect io"),
            entries: &[ro_storage(0), rw_storage(1)],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_advect uniforms"),
            entries: &[uniform_entry(0)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::ns_advect layout"),
            bind_group_layouts: &[&layout_mesh, &layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::ns_advect pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_convection"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            layout_mesh,
            layout_io,
            layout_uniforms,
        }
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Dispatch the convection assembly. `f_conv` is the **target** RHS
    /// buffer (length `n_v`, `f32`). The caller MUST pre-zero it before
    /// each dispatch — the kernel scatters with atomic-add.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        mesh_vertices: &wgpu::Buffer,
        mesh_cells: &wgpu::Buffer,
        mesh_cell_offsets: &wgpu::Buffer,
        p2_dof_table: &wgpu::Buffer,
        u_old: &wgpu::Buffer,
        f_conv: &wgpu::Buffer,
        cell_count: u32,
        n_v: u32,
        dim: u32,
        viscosity: f32,
        supg_alpha: f32,
    ) -> Result<()> {
        if cell_count == 0 {
            return Ok(());
        }
        let uniforms = NsAdvectUniforms {
            cell_count,
            n_v,
            dim,
            _pad0: 0,
            viscosity,
            supg_alpha,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::ns_advect uniforms"),
            size: core::mem::size_of::<NsAdvectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_mesh = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_advect mesh bg"),
            layout: &self.layout_mesh,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_cell_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p2_dof_table.as_entire_binding(),
                },
            ],
        });
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_advect io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_old.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: f_conv.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_advect uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = cell_count.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::ns_advect encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem::ns_advect pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_mesh, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
