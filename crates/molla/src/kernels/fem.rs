//! Stage 36.5 — Rust dispatchers for the FEM GPU kernels.
//!
//! Two compute pipelines built from `shaders/fem_assembly.wgsl` and
//! `shaders/fem_matvec.wgsl`:
//!   - `assemble_p1_stiffness_tet` — populates the CSR `values` buffer
//!     by atomically scattering per-cell stiffness contributions.
//!   - `csr_matvec` — `y = A · x` with one workgroup per row + tree
//!     reduction within the workgroup.
//!
//! The dispatchers are deliberately stateless re: bind groups — they
//! re-create them per call against a [`crate::fem::FemGpuData`].
//! Buffer creation is amortised in `FemGpuData::upload`; only the
//! per-call `x` / `y` matvec buffers come from the caller.
//!
//! See also `crates/molla/src/fem/solver.rs::solve_steady_state_gpu`
//! which wraps the matvec dispatch in a closure for `cg_solve`.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::fem::FemGpuData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AssemblyUniforms {
    cell_count: u32,
    nrows: u32,
    coeff: f32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecUniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Kernel handle for the FEM GPU dispatchers. Owns the shader modules
/// + pipelines + bind-group layouts; reuse across many dispatches.
pub struct FemGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_assemble_tet: wgpu::ComputePipeline,
    pipe_assemble_mass_tet: wgpu::ComputePipeline,
    pipe_assemble_tri: wgpu::ComputePipeline,
    pipe_assemble_mass_tri: wgpu::ComputePipeline,
    pipe_matvec: wgpu::ComputePipeline,

    layout_asm_mesh: wgpu::BindGroupLayout,
    layout_asm_csr: wgpu::BindGroupLayout,
    layout_asm_uniforms: wgpu::BindGroupLayout,

    layout_mv_io: wgpu::BindGroupLayout,
    layout_mv_uniforms: wgpu::BindGroupLayout,
}

impl FemGpuKernel {
    /// Build the kernel against `device` + `queue`. Compiles the two
    /// shader modules and creates the pipelines.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        // ── Assembly pipeline ──────────────────────────────────────
        let asm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::assembly"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_assembly.wgsl").into(),
            ),
        });

        let layout_asm_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::asm mesh"),
            entries: &[
                ro_storage(0), // mesh_vertices (vec4<f32> array)
                ro_storage(1), // mesh_cells
                ro_storage(2), // mesh_cell_offsets
            ],
        });
        let layout_asm_csr = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::asm csr"),
            entries: &[
                ro_storage(0), // csr_row_offsets
                ro_storage(1), // csr_col_indices
                rw_storage(2), // csr_values (atomic<u32>)
            ],
        });
        let layout_asm_uniforms = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem::asm uniforms"),
                entries: &[uniform_entry(0)],
            },
        );
        let asm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::asm layout"),
            bind_group_layouts: &[&layout_asm_mesh, &layout_asm_csr, &layout_asm_uniforms],
            push_constant_ranges: &[],
        });
        let pipe_assemble_tet = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem::assemble_p1_stiffness_tet"),
                layout: Some(&asm_pipeline_layout),
                module: &asm_shader,
                entry_point: Some("assemble_p1_stiffness_tet"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );
        let pipe_assemble_mass_tet = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem::assemble_p1_mass_tet"),
                layout: Some(&asm_pipeline_layout),
                module: &asm_shader,
                entry_point: Some("assemble_p1_mass_tet"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );
        let pipe_assemble_tri = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem::assemble_p1_stiffness_tri"),
                layout: Some(&asm_pipeline_layout),
                module: &asm_shader,
                entry_point: Some("assemble_p1_stiffness_tri"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );
        let pipe_assemble_mass_tri = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem::assemble_p1_mass_tri"),
                layout: Some(&asm_pipeline_layout),
                module: &asm_shader,
                entry_point: Some("assemble_p1_mass_tri"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── Matvec pipeline ────────────────────────────────────────
        let mv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::matvec"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_matvec.wgsl").into(),
            ),
        });
        let layout_mv_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::mv io"),
            entries: &[
                ro_storage(0), // csr_row_offsets
                ro_storage(1), // csr_col_indices
                ro_storage(2), // csr_values (f32 — read-only for matvec)
                ro_storage(3), // x
                rw_storage(4), // y
            ],
        });
        let layout_mv_uniforms = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem::mv uniforms"),
                entries: &[uniform_entry(0)],
            },
        );
        let mv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::mv layout"),
            bind_group_layouts: &[&layout_mv_io, &layout_mv_uniforms],
            push_constant_ranges: &[],
        });
        let pipe_matvec = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem::csr_matvec"),
                layout: Some(&mv_pipeline_layout),
                module: &mv_shader,
                entry_point: Some("csr_matvec"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        Self {
            device,
            queue,
            pipe_assemble_tet,
            pipe_assemble_mass_tet,
            pipe_assemble_tri,
            pipe_assemble_mass_tri,
            pipe_matvec,
            layout_asm_mesh,
            layout_asm_csr,
            layout_asm_uniforms,
            layout_mv_io,
            layout_mv_uniforms,
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

    /// Run `assemble_p1_stiffness_tet` against `gpu_data`, with the
    /// scalar `coeff` (e.g. thermal diffusivity α). Idempotent — the
    /// CSR values buffer is zeroed first so repeated calls don't
    /// accumulate.
    ///
    /// v1 only handles tet meshes; tri-mesh assembly remains CPU-only
    /// (the heat-equation demo uses tets). A `assemble_p1_stiffness_tri`
    /// entry can be added later — the dispatcher infrastructure is
    /// shared.
    pub fn assemble(&self, gpu_data: &FemGpuData, coeff: f32) -> Result<()> {
        self.dispatch_assembly(gpu_data, coeff, &self.pipe_assemble_tet, "assemble")
    }

    /// Run `assemble_p1_mass_tet` — assembles the consistent P1 nodal
    /// mass matrix `M[i, j] = coeff · ∫ φᵢ φⱼ dx` into `gpu_data`'s
    /// CSR buffer. `coeff` is the per-element scale (use `1.0` for
    /// the standard mass matrix, `ρ` for inertial scaling, `c²` for
    /// the wave-equation modal mass). Idempotent — clears the CSR
    /// before scattering so repeated calls don't accumulate.
    pub fn assemble_mass(&self, gpu_data: &FemGpuData, coeff: f32) -> Result<()> {
        self.dispatch_assembly(
            gpu_data,
            coeff,
            &self.pipe_assemble_mass_tet,
            "assemble_mass",
        )
    }

    /// Triangle-mesh equivalent of [`Self::assemble`] — P1 stiffness
    /// for 2D meshes (the (x, y) projection of vertex positions; `z`
    /// is ignored). Used for the wave-equation height-field, the
    /// fractional-step NS predictor on a triangulated 2D domain, and
    /// any other 2D PDE built on `tri_grid_2d`.
    pub fn assemble_tri(&self, gpu_data: &FemGpuData, coeff: f32) -> Result<()> {
        self.dispatch_assembly(gpu_data, coeff, &self.pipe_assemble_tri, "assemble_tri")
    }

    /// Triangle-mesh equivalent of [`Self::assemble_mass`] — P1
    /// consistent mass `M[i, j] = coeff · ∫ φᵢ φⱼ dA` for 2D meshes.
    pub fn assemble_mass_tri(&self, gpu_data: &FemGpuData, coeff: f32) -> Result<()> {
        self.dispatch_assembly(
            gpu_data,
            coeff,
            &self.pipe_assemble_mass_tri,
            "assemble_mass_tri",
        )
    }

    fn dispatch_assembly(
        &self,
        gpu_data: &FemGpuData,
        coeff: f32,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
    ) -> Result<()> {
        if gpu_data.cell_count == 0 {
            return Ok(());
        }

        // Zero the CSR values BEFORE submitting so the atomic-add
        // scatter starts from a clean baseline.
        gpu_data.clear_csr_values(&self.queue);

        // Uniforms.
        let uniforms = AssemblyUniforms {
            cell_count: gpu_data.cell_count,
            nrows: gpu_data.csr_nrows,
            coeff,
            _pad0: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::asm uniforms"),
            size: core::mem::size_of::<AssemblyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Bind groups.
        let bg_mesh = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::asm mesh bg"),
            layout: &self.layout_asm_mesh,
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
            label: Some("molla::fem::asm csr bg"),
            layout: &self.layout_asm_csr,
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
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::asm uniforms bg"),
            layout: &self.layout_asm_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = gpu_data.cell_count.div_ceil(WORKGROUP_SIZE).max(1);

        let enc_label = format!("molla::fem::{label} encoder");
        let pass_label = format!("molla::fem::{label}");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&enc_label),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass_label),
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

    /// Run `csr_matvec` against `gpu_data` with input `x` (length
    /// `csr_nrows`, f32) and write the result into `y` (same length,
    /// f32). Both buffers must be the caller's storage buffers; this
    /// dispatcher does NOT allocate them.
    ///
    /// One workgroup per row, tree reduction within. Submits + does
    /// NOT readback — the caller is responsible for the readback (see
    /// [`Self::matvec_blocking`] for the convenience variant).
    pub fn matvec(
        &self,
        gpu_data: &FemGpuData,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) -> Result<()> {
        if gpu_data.csr_nrows == 0 {
            return Ok(());
        }
        let uniforms = MatvecUniforms {
            nrows: gpu_data.csr_nrows,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::mv uniforms"),
            size: core::mem::size_of::<MatvecUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::mv io bg"),
            layout: &self.layout_mv_io,
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
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::mv uniforms bg"),
            layout: &self.layout_mv_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::mv encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem::matvec"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_matvec);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            // One workgroup per row, packed into a 2D grid because
            // WebGPU caps any single dispatch dim at 65535. The
            // shader unpacks `row = wid.y * 65535 + wid.x`.
            const MAX_DIM: u32 = 65535;
            let nrows = gpu_data.csr_nrows;
            let dim_x = nrows.min(MAX_DIM).max(1);
            let dim_y = nrows.div_ceil(MAX_DIM).max(1);
            pass.dispatch_workgroups(dim_x, dim_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Convenience: run [`Self::matvec`] with a freshly-allocated `y`,
    /// upload `x` from a host slice, read `y` back. Submits twice
    /// (matvec + readback) — one host bounce per call.
    ///
    /// **Per-iter readback cost:** at 16³ tet resolution (~5k DoFs)
    /// the `y` readback is a 20 KiB transfer; on RTX 4080 measured
    /// at ~0.5 ms wall-clock per iteration including dispatch. CG
    /// converges in ~50-200 iters on the heat-equation problem so
    /// the host bounces dominate the solve. v2 ("fully on-device CG"
    /// — Stage 36.5.1) eliminates them by keeping `r`, `p`, `x`
    /// resident on the GPU.
    pub fn matvec_blocking(
        &self,
        gpu_data: &FemGpuData,
        x_host: &[f32],
    ) -> Result<Vec<f32>> {
        let n = gpu_data.csr_nrows as usize;
        if x_host.len() != n {
            return Err(Error::Build(format!(
                "matvec_blocking: x length {} != nrows {}",
                x_host.len(),
                n
            )));
        }
        let bytes = (n.max(4) * 4) as u64;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let x_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::mv x"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(x_host));
        let y_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::mv y"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        // y must be zeroed (matvec only writes y[row] = ...; rows
        // beyond nrows would carry stale data, and the row-zero check
        // in the shader returns early so y[0..nrows] is fully written).
        self.queue.write_buffer(&y_buf, 0, &vec![0u8; bytes as usize]);

        self.matvec(gpu_data, &x_buf, &y_buf)?;

        // Readback into a fresh Vec<f32>.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::mv y staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::mv readback enc"),
            });
        encoder.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("matvec readback poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("matvec readback recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("matvec readback map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let mut out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out.truncate(n);
        Ok(out)
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
