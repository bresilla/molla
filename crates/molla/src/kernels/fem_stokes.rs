//! Stage 41 v2 — Rust dispatcher for the Taylor-Hood Stokes GPU
//! assembly kernel (`fem_stokes_assemble.wgsl`).
//!
//! Two entry points share the same bind-group layout (mesh + p2 DoF
//! table + target CSR + uniforms); the dispatcher carries one pipeline
//! per entry point and lets the caller pick which CSR to scatter into.
//!
//! Lives next to `FemElasticityGpuKernel` (Stage 41 v1) because they
//! share the elementary atomic-CAS scatter pattern; the higher-level
//! Stokes wiring lives in `crates/molla/src/fem/stokes_gpu.rs`.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::Result;

/// Per-cell P2 node table: `p2_dof_table[cell * 6 + local_node]` →
/// global P2 node index. Built CPU-side from `MixedTaylorHood::map`.
///
/// The shader uses this to map local node `i ∈ [0, 6)` → global node
/// → interleaved DoF `node * dim + c`. We pass this as a flat
/// `u32` storage buffer rather than re-deriving the edge lookup on
/// the GPU (the edge-id hash map is host-only).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StokesUniforms {
    cell_count: u32,
    n_v: u32,
    n_p: u32,
    dim: u32,
    viscosity: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU dispatcher for the Stokes assembly kernels. Single pipeline pair
/// (velocity + divergence) — caller picks via the public methods.
pub struct FemStokesGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_velocity: wgpu::ComputePipeline,
    pipe_div: wgpu::ComputePipeline,

    layout_mesh: wgpu::BindGroupLayout,
    layout_csr: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FemStokesGpuKernel {
    /// Build the kernel — compiles the WGSL module and creates both
    /// pipelines. Cheap; reuse across many dispatches.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::stokes_assembly"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_stokes_assemble.wgsl").into(),
            ),
        });

        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::stokes asm mesh"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2), ro_storage(3)],
        });
        let layout_csr = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::stokes asm csr"),
            entries: &[ro_storage(0), ro_storage(1), rw_storage(2)],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::stokes asm uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::stokes asm layout"),
            bind_group_layouts: &[&layout_mesh, &layout_csr, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipe_velocity = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::stokes assemble_velocity_block"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_velocity_block"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_div = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::stokes assemble_div_block"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_div_block"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipe_velocity,
            pipe_div,
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

    /// Dispatch one of the two assembly entry points. The caller
    /// must pre-zero `csr_values` (the kernel uses atomic-add scatter).
    ///
    /// `mesh_vertices`, `mesh_cells`, `mesh_cell_offsets` are the
    /// standard `FemGpuData` buffers.
    /// `p2_dof_table` is a `u32` storage buffer of length
    /// `cell_count * 6` mapping each cell's local P2 node index to its
    /// global P2 node index.
    /// `csr_row_offsets`, `csr_col_indices`, `csr_values` are the
    /// target CSR (size depends on which entry point is dispatched —
    /// `n_v × n_v` for velocity, `n_p × n_v` for divergence).
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        entry: StokesEntry,
        mesh_vertices: &wgpu::Buffer,
        mesh_cells: &wgpu::Buffer,
        mesh_cell_offsets: &wgpu::Buffer,
        p2_dof_table: &wgpu::Buffer,
        csr_row_offsets: &wgpu::Buffer,
        csr_col_indices: &wgpu::Buffer,
        csr_values: &wgpu::Buffer,
        cell_count: u32,
        n_v: u32,
        n_p: u32,
        dim: u32,
        viscosity: f32,
    ) -> Result<()> {
        if cell_count == 0 {
            return Ok(());
        }
        let uniforms = StokesUniforms {
            cell_count,
            n_v,
            n_p,
            dim,
            viscosity,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::stokes asm uniforms"),
            size: core::mem::size_of::<StokesUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_mesh = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::stokes asm mesh bg"),
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
        let bg_csr = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::stokes asm csr bg"),
            layout: &self.layout_csr,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: csr_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: csr_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: csr_values.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::stokes asm uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = cell_count.div_ceil(WORKGROUP_SIZE).max(1);
        let pipeline = match entry {
            StokesEntry::Velocity => &self.pipe_velocity,
            StokesEntry::Divergence => &self.pipe_div,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::stokes asm encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem::stokes asm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_mesh, &[]);
            pass.set_bind_group(1, &bg_csr, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

/// Which assembly entry point to dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StokesEntry {
    /// `assemble_velocity_block` — scatter into the `n_v × n_v` velocity CSR.
    Velocity,
    /// `assemble_div_block` — scatter into the `n_p × n_v` divergence CSR.
    Divergence,
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
