//! Stage 41.1 — Rust dispatcher for the linear-elasticity GPU
//! assembly kernel.
//!
//! Compiles `shaders/fem_elasticity_assemble.wgsl` and dispatches it
//! against an already-uploaded [`crate::fem::FemGpuData`]. The CSR
//! pattern must be vector-sized (`nrows = vertex_count * dim`); the
//! caller is responsible for pre-populating BC rows with the
//! `(row, row) = 1.0` identity before dispatch and providing the
//! `is_bc_row` mask buffer.
//!
//! Companion of [`crate::fem::elasticity_gpu`] which wires the full
//! end-to-end (pattern build → BC marking → upload → assemble →
//! resident CG) flow.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::Result;
use crate::fem::FemGpuData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ElasticityUniforms {
    cell_count: u32,
    nrows: u32,
    lambda: f32,
    mu: f32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU dispatcher for `assemble_elasticity_tet`. Stateless w.r.t. the
/// `FemGpuData` it operates on — pass the same kernel across many
/// (mesh, BCs) configurations.
pub struct FemElasticityGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_assemble_tet: wgpu::ComputePipeline,

    layout_mesh: wgpu::BindGroupLayout,
    layout_csr: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FemElasticityGpuKernel {
    /// Build the kernel — compiles the WGSL module and creates the
    /// compute pipeline. Cheap; reuse across many dispatches.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::elasticity_assembly"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_elasticity_assemble.wgsl").into(),
            ),
        });

        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::elasticity asm mesh"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2)],
        });
        let layout_csr = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::elasticity asm csr"),
            entries: &[
                ro_storage(0), // csr_row_offsets
                ro_storage(1), // csr_col_indices
                rw_storage(2), // csr_values (atomic<u32>)
                ro_storage(3), // is_bc_row
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::elasticity asm uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::elasticity asm layout"),
            bind_group_layouts: &[
                Some(&layout_mesh),
                Some(&layout_csr),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });
        let pipe_assemble_tet = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::assemble_elasticity_tet"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_elasticity_tet"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipe_assemble_tet,
            layout_mesh,
            layout_csr,
            layout_uniforms,
        }
    }

    /// Borrow the device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Borrow the queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Dispatch `assemble_elasticity_tet` against `gpu_data`. Caller
    /// guarantees:
    ///   * `gpu_data.csr_nrows == gpu_data.vertex_count * dim`,
    ///   * `csr_values` has been pre-filled with the BC identity
    ///     (`1.0` on BC diagonal, `0.0` elsewhere) — the caller's
    ///     responsibility because we don't see the BC list here,
    ///   * `bc_mask_buf` is a `u32` storage buffer of length
    ///     `csr_nrows` where each entry is `1` if the row is BC and
    ///     `0` otherwise.
    ///
    /// `(lambda, mu)` are the Lamé parameters used for **all** cells —
    /// a per-element material array is a v2 follow-up.
    pub fn assemble(
        &self,
        gpu_data: &FemGpuData,
        bc_mask_buf: &wgpu::Buffer,
        lambda: f32,
        mu: f32,
        dim: u32,
    ) -> Result<()> {
        if gpu_data.cell_count == 0 {
            return Ok(());
        }

        // Uniforms.
        let uniforms = ElasticityUniforms {
            cell_count: gpu_data.cell_count,
            nrows: gpu_data.csr_nrows,
            lambda,
            mu,
            dim,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::elasticity asm uniforms"),
            size: core::mem::size_of::<ElasticityUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_mesh = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::elasticity asm mesh bg"),
            layout: &self.layout_mesh,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_data.mesh_vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_data.mesh_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_data.mesh_cell_offsets.as_entire_binding(),
                },
            ],
        });
        let bg_csr = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::elasticity asm csr bg"),
            layout: &self.layout_csr,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_data.csr_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_data.csr_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_data.csr_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bc_mask_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::elasticity asm uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = gpu_data.cell_count.div_ceil(WORKGROUP_SIZE).max(1);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::elasticity asm encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem::assemble_elasticity_tet"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_assemble_tet);
            pass.set_bind_group(0, &bg_mesh, &[]);
            pass.set_bind_group(1, &bg_csr, &[]);
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
