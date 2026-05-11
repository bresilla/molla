//! Stage 38a — GPU sparse BSR algebra (Stage 39F: full-GPU SpGEMM).
//!
//! On-device mirrors of [`super::sparse::BsrMatrix`] (scalar `B = 1`
//! only in v1, matching the rest of the GPU FEM stack) plus a kernel
//! dispatcher that provides:
//!
//!   - [`BsrAlgebraKernel::matvec`]      — `y = α·A·x + β·y`
//!   - [`BsrAlgebraKernel::axpy`]        — `B = α·A + β·B`
//!   - [`BsrAlgebraKernel::transpose`]   — `B = Aᵀ`
//!   - [`BsrAlgebraKernel::spgemm`]      — `C = A · B`
//!
//! ## Precision
//!
//! GPU-side values are `f32`; host-side `BsrMatrix` is `f64`. Upload
//! converts `f64 → f32`, readback the reverse. The same tradeoff that
//! [`crate::fem::gpu_data`] documents — WGSL has no `f64`, and a
//! Vulkan-only path would fork the codebase.
//!
//! ## SpGEMM design — fully on-device (Stage 39F)
//!
//! Stage 38a shipped "symbolic on CPU, numeric on GPU". Stage 39F moves
//! the symbolic phase on-device using a per-row dense bitmap:
//!
//!   1. **Symbolic count** — one workgroup per A-row, atomicOrs the B-col
//!      indices into a per-row bitmap, then popcounts to get unique
//!      counts.
//!   2. **Scan** — exclusive scan of counts → `c_row_offsets[]`, via
//!      [`crate::kernels::ScanKernel`] (Stage 38e).
//!   3. **Symbolic write** — walks each row's saved bitmap and emits
//!      set bit positions in ascending order to `c_col_indices[]`. The
//!      output is sorted-by-column per row, matching the
//!      `BsrMatrix` invariant.
//!   4. **Numeric** — per A-row workgroup, for each `(a_slot, b_slot)`
//!      pair binary-searches `c_col_indices` to find the output slot
//!      and CAS-adds `a · b` into `c_values`.
//!
//! For tiny problems (`n < 64`) the host-symbolic v1 path is faster
//! (kernel-launch overhead dominates); the dispatcher falls back to it
//! below the threshold. See [`SPGEMM_GPU_SYMBOLIC_CUTOVER`].
//!
//! ### Why dense bitmaps and not a parallel hashtable?
//!
//! Warp's `bsr_mm` uses a workgroup-local hashtable with global
//! overflow. For the FEM/Schur problems that motivate Stage 39F (≈ 7-
//! banded × 7-banded → ~49 unique cols per C-row) the hashtable
//! overflow path always fires and the 16-bucket WG cache buys nothing.
//! Dense bitmaps cost `nrows · ⌈ncols/32⌉ · 4` bytes (≈ 2 MB at
//! 4096²), use O(1) atomicOr per insert, naturally dedup, and produce
//! sorted output for free. See the shader for the full rationale.
//!
//! ## Transpose design
//!
//! Same shape as old SpGEMM — host computes the transposed pattern
//! (counts per column, prefix-sum, scatter map) and uploads. GPU runs
//! a single `bsr_transpose_scatter` dispatch. Could be moved fully
//! on-device with the same bitmap-count approach if it ever shows up
//! in a profile.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::kernels::ScanKernel;
use crate::math::sparse::BsrMatrix;
use crate::math::Real;

/// Below this matrix dimension (in scalar `B = 1` units) `spgemm`
/// falls back to the host-symbolic v1 path. The GPU symbolic costs
/// four shader dispatches + bitmap allocation; for small problems
/// the host walk + single numeric dispatch wins. Empirically the
/// cross-over sits between 32 and 64 on every GPU we've tested.
pub const SPGEMM_GPU_SYMBOLIC_CUTOVER: u32 = 64;

/// GPU mirror of a [`BsrMatrix`] with block size `B = 1`. Stored as
/// three storage buffers (CSR layout). Shape metadata is kept on the
/// host so dispatch can size workgroup grids without round-tripping.
///
/// `values` is `f32`; `row_offsets` / `col_indices` are `u32`. See the
/// module docstring for the precision rationale.
pub struct BsrGpu {
    pub row_offsets: Arc<wgpu::Buffer>,
    pub col_indices: Arc<wgpu::Buffer>,
    pub values: Arc<wgpu::Buffer>,
    pub nrows: u32,
    pub ncols: u32,
    pub nnz: u32,
}

impl BsrGpu {
    /// Upload `m` (a scalar `B = 1` BSR / CSR matrix) to three storage
    /// buffers. Values convert `f64 → f32` on the wire.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        m: &BsrMatrix,
    ) -> Result<Self> {
        if m.b != 1 {
            return Err(Error::Build(format!(
                "BsrGpu::upload: only B = 1 supported in v1, got B = {}",
                m.b
            )));
        }
        // CSR shape consistency — the public `BsrMatrix` API permits
        // construction with malformed buffers (row_offsets/values/
        // col_idx out of sync). Without this check we'd upload
        // inconsistent buffers and produce silent OOB reads inside
        // the WGSL kernel.
        let expected_nnz = *m.row_offsets.last().unwrap_or(&0);
        if m.values.len() != expected_nnz || m.col_idx.len() != expected_nnz {
            return Err(Error::Build(format!(
                "BsrGpu::upload: BsrMatrix shape inconsistency — \
                 row_offsets[last]={expected_nnz} but values.len()={} and col_idx.len()={}",
                m.values.len(),
                m.col_idx.len()
            )));
        }
        if m.row_offsets.len() != m.block_rows + 1 {
            return Err(Error::Build(format!(
                "BsrGpu::upload: row_offsets.len() = {} but block_rows + 1 = {}",
                m.row_offsets.len(),
                m.block_rows + 1
            )));
        }
        let row_offsets_u32: Vec<u32> = m.row_offsets.iter().map(|&v| v as u32).collect();
        let col_idx_u32: Vec<u32> = m.col_idx.iter().map(|&v| v as u32).collect();
        let values_f32: Vec<f32> = m.values.iter().map(|&v| v as f32).collect();

        let row_offsets = create_storage_buffer(
            device,
            queue,
            "molla::sparse_gpu row_offsets",
            bytemuck::cast_slice(&row_offsets_u32),
        );
        let col_indices = create_storage_buffer(
            device,
            queue,
            "molla::sparse_gpu col_indices",
            bytemuck::cast_slice(&col_idx_u32),
        );
        let values = create_storage_buffer(
            device,
            queue,
            "molla::sparse_gpu values",
            bytemuck::cast_slice(&values_f32),
        );
        Ok(Self {
            row_offsets: Arc::new(row_offsets),
            col_indices: Arc::new(col_indices),
            values: Arc::new(values),
            nrows: m.block_rows as u32,
            ncols: m.block_cols as u32,
            nnz: values_f32.len() as u32,
        })
    }

    /// Read the three buffers back into a host [`BsrMatrix`]. Values
    /// promote `f32 → f64`. Used by parity tests + any pipeline that
    /// needs to hand the matrix to a CPU consumer (e.g. the host
    /// iterative solver wrappers).
    pub fn read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<BsrMatrix> {
        let row_off_bytes = ((self.nrows as u64) + 1) * 4;
        let col_idx_bytes = (self.nnz as u64) * 4;
        let values_bytes = (self.nnz as u64) * 4;

        let row_off_u32 =
            read_u32_buffer(device, queue, &self.row_offsets, row_off_bytes, "row_offsets")?;
        let col_idx_u32 =
            read_u32_buffer(device, queue, &self.col_indices, col_idx_bytes, "col_indices")?;
        let values_f32 =
            read_f32_buffer(device, queue, &self.values, values_bytes, "values")?;

        // The buffers are padded up to 16 bytes — slice to the logical
        // length so the host arrays match what was uploaded.
        let row_offsets: Vec<usize> = row_off_u32
            .iter()
            .take((self.nrows as usize) + 1)
            .map(|&v| v as usize)
            .collect();
        let col_idx: Vec<usize> = col_idx_u32
            .iter()
            .take(self.nnz as usize)
            .map(|&v| v as usize)
            .collect();
        let values: Vec<Real> = values_f32
            .iter()
            .take(self.nnz as usize)
            .map(|&v| v as Real)
            .collect();

        Ok(BsrMatrix {
            b: 1,
            block_rows: self.nrows as usize,
            block_cols: self.ncols as usize,
            row_offsets,
            col_idx,
            values,
        })
    }
}

// ─── Uniform structs ────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecUniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyUniforms {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarU32Uniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SymCountUniforms {
    a_nrows: u32,
    b_nrows: u32,
    b_ncols: u32,
    bitmap_words: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SymWriteUniforms {
    a_nrows: u32,
    bitmap_words: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NumericAtomicUniforms {
    a_nrows: u32,
    b_nrows: u32,
    _pad0: u32,
    _pad1: u32,
}

const WG_SIZE: u32 = 256;
const MAX_DIM: u32 = 65535;

/// Owns the compute pipelines + bind-group layouts for the four BSR
/// kernels. Build once per device, reuse across many dispatches.
/// Cached matvec dispatch — created by
/// [`BsrAlgebraKernel::prepare_matvec`] and consumed repeatedly by
/// [`BsrAlgebraKernel::matvec_into_pass`]. Holds the uniform buffer +
/// the two cached bind groups so the inner-loop dispatch is a single
/// `pass.dispatch_workgroups` with no per-call alloc.
pub struct PreparedMatvec {
    #[allow(dead_code)]
    uniform_buf: wgpu::Buffer,
    bg_io: wgpu::BindGroup,
    bg_u: wgpu::BindGroup,
    dim_x: u32,
    dim_y: u32,
}

pub struct BsrAlgebraKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_matvec: wgpu::ComputePipeline,
    pipe_axpy: wgpu::ComputePipeline,
    pipe_transpose: wgpu::ComputePipeline,
    pipe_spgemm: wgpu::ComputePipeline,
    // Stage 39F — GPU symbolic pipelines.
    pipe_sym_count: wgpu::ComputePipeline,
    pipe_sym_write: wgpu::ComputePipeline,
    pipe_num_atomic: wgpu::ComputePipeline,

    layout_mv_io: wgpu::BindGroupLayout,
    layout_mv_u: wgpu::BindGroupLayout,
    layout_axpy_io: wgpu::BindGroupLayout,
    layout_axpy_u: wgpu::BindGroupLayout,
    layout_tr_io: wgpu::BindGroupLayout,
    layout_tr_u: wgpu::BindGroupLayout,
    layout_spgemm_io: wgpu::BindGroupLayout,
    layout_spgemm_u: wgpu::BindGroupLayout,
    layout_sym_count_io: wgpu::BindGroupLayout,
    layout_sym_count_u: wgpu::BindGroupLayout,
    layout_sym_write_io: wgpu::BindGroupLayout,
    layout_sym_write_u: wgpu::BindGroupLayout,
    layout_num_atomic_io: wgpu::BindGroupLayout,
    layout_num_atomic_u: wgpu::BindGroupLayout,

    // Stage 38e scan dispatcher — used by the symbolic-count→offsets
    // step. Owned here so callers don't need to construct one.
    scan: ScanKernel,
}

impl BsrAlgebraKernel {
    /// Build the kernel — compiles four WGSL modules and stamps out
    /// four pipelines + their bind-group layouts.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let mv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_gpu::bsr_mv"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bsr_mv_gpu.wgsl").into(),
            ),
        });
        let axpy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_gpu::bsr_axpy"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bsr_axpy.wgsl").into(),
            ),
        });
        let tr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_gpu::bsr_transpose"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bsr_transpose.wgsl").into(),
            ),
        });
        let spgemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_gpu::bsr_spgemm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bsr_spgemm.wgsl").into(),
            ),
        });
        let spgemm_sym_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_gpu::bsr_spgemm_symbolic"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/bsr_spgemm_symbolic.wgsl").into(),
            ),
        });

        // ── matvec ─────────────────────────────────────────────────
        let layout_mv_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::mv io"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                ro_storage(2),
                ro_storage(3),
                rw_storage(4),
            ],
        });
        let layout_mv_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::mv uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_matvec = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::bsr_mv_gpu"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::mv layout"),
                bind_group_layouts: &[&layout_mv_io, &layout_mv_u],
                push_constant_ranges: &[],
            })),
            module: &mv_shader,
            entry_point: Some("bsr_mv_gpu"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── axpy ───────────────────────────────────────────────────
        let layout_axpy_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::axpy io"),
            entries: &[ro_storage(0), rw_storage(1)],
        });
        let layout_axpy_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::axpy uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_axpy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::bsr_axpy"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::axpy layout"),
                bind_group_layouts: &[&layout_axpy_io, &layout_axpy_u],
                push_constant_ranges: &[],
            })),
            module: &axpy_shader,
            entry_point: Some("bsr_axpy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── transpose ──────────────────────────────────────────────
        let layout_tr_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::transpose io"),
            entries: &[ro_storage(0), ro_storage(1), rw_storage(2)],
        });
        let layout_tr_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::transpose uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_transpose =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::sparse_gpu::bsr_transpose_scatter"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::sparse_gpu::transpose layout"),
                    bind_group_layouts: &[&layout_tr_io, &layout_tr_u],
                    push_constant_ranges: &[],
                })),
                module: &tr_shader,
                entry_point: Some("bsr_transpose_scatter"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ── spgemm ─────────────────────────────────────────────────
        let layout_spgemm_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::spgemm io"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                ro_storage(2),
                ro_storage(3),
                rw_storage(4),
            ],
        });
        let layout_spgemm_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::spgemm uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_spgemm = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::bsr_spgemm_numeric"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::spgemm layout"),
                bind_group_layouts: &[&layout_spgemm_io, &layout_spgemm_u],
                push_constant_ranges: &[],
            })),
            module: &spgemm_shader,
            entry_point: Some("bsr_spgemm_numeric"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── spgemm symbolic count ──────────────────────────────────
        // Bindings: 0..3 ro (a_row_off, a_col_idx, b_row_off, b_col_idx),
        //           4..5 rw (row_bitmaps as atomic<u32>, count[]).
        let layout_sym_count_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::sym_count io"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                ro_storage(2),
                ro_storage(3),
                rw_storage(4),
                rw_storage(5),
            ],
        });
        let layout_sym_count_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::sym_count uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_sym_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::symbolic_count"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::sym_count layout"),
                bind_group_layouts: &[&layout_sym_count_io, &layout_sym_count_u],
                push_constant_ranges: &[],
            })),
            module: &spgemm_sym_shader,
            entry_point: Some("symbolic_count"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── spgemm symbolic write ──────────────────────────────────
        // Bindings: 0 ro c_row_offsets, 1 ro row_bitmaps, 2 rw c_col_indices.
        let layout_sym_write_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::sym_write io"),
            entries: &[ro_storage(0), ro_storage(1), rw_storage(2)],
        });
        let layout_sym_write_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_gpu::sym_write uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_sym_write = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::symbolic_write"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::sym_write layout"),
                bind_group_layouts: &[&layout_sym_write_io, &layout_sym_write_u],
                push_constant_ranges: &[],
            })),
            module: &spgemm_sym_shader,
            entry_point: Some("symbolic_write"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── spgemm numeric (atomic add) ────────────────────────────
        // Bindings: 0..2 A (row_off, col_idx, values),
        //           3..5 B (row_off, col_idx, values),
        //           6..7 C (row_off, col_idx),
        //           8 rw c_values as atomic<u32>.
        let layout_num_atomic_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::sparse_gpu::num_atomic io"),
                entries: &[
                    ro_storage(0),
                    ro_storage(1),
                    ro_storage(2),
                    ro_storage(3),
                    ro_storage(4),
                    ro_storage(5),
                    ro_storage(6),
                    ro_storage(7),
                    rw_storage(8),
                ],
            });
        let layout_num_atomic_u =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::sparse_gpu::num_atomic uniforms"),
                entries: &[uniform_entry(0)],
            });
        let pipe_num_atomic = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_gpu::numeric_atomic_add"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::sparse_gpu::num_atomic layout"),
                bind_group_layouts: &[&layout_num_atomic_io, &layout_num_atomic_u],
                push_constant_ranges: &[],
            })),
            module: &spgemm_sym_shader,
            entry_point: Some("numeric_atomic_add"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scan = ScanKernel::new(device.clone(), queue.clone());

        Self {
            device,
            queue,
            pipe_matvec,
            pipe_axpy,
            pipe_transpose,
            pipe_spgemm,
            pipe_sym_count,
            pipe_sym_write,
            pipe_num_atomic,
            layout_mv_io,
            layout_mv_u,
            layout_axpy_io,
            layout_axpy_u,
            layout_tr_io,
            layout_tr_u,
            layout_spgemm_io,
            layout_spgemm_u,
            layout_sym_count_io,
            layout_sym_count_u,
            layout_sym_write_io,
            layout_sym_write_u,
            layout_num_atomic_io,
            layout_num_atomic_u,
            scan,
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

    /// Generalised matvec: `y = α · A · x + β · y`.
    ///
    /// `x_buf` length must match `A.ncols`, `y_buf` must match
    /// `A.nrows`. Buffers must already be allocated as storage buffers
    /// (`STORAGE | COPY_SRC | COPY_DST`); the dispatcher does NOT
    /// allocate them. With `β = 0` the previous contents of `y_buf`
    /// are discarded — the kernel still reads `y_buf` so it must be
    /// zero-initialised (or the read is masked by `β = 0`). The
    /// implementation multiplies by `β` so leaving `y_buf` with stale
    /// data is fine as long as the caller passes `β = 0`.
    pub fn matvec(
        &self,
        a: &BsrGpu,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if a.nrows == 0 {
            return Ok(());
        }
        let uniforms = MatvecUniforms {
            nrows: a.nrows,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha,
            beta,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::mv uniforms"),
            size: core::mem::size_of::<MatvecUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::mv io bg"),
            layout: &self.layout_mv_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: y_buf.as_entire_binding() },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::mv u bg"),
            layout: &self.layout_mv_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let nrows = a.nrows;
        let dim_x = nrows.min(MAX_DIM).max(1);
        let dim_y = nrows.div_ceil(MAX_DIM).max(1);

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::sparse_gpu::mv encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_gpu::matvec"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_matvec);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(dim_x, dim_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Stage 41 v5 — "prepared matvec" path used by the lazy CG inner
    /// loop. Caches the uniform buffer + IO bind group for a fixed
    /// `(A, x, y, alpha, beta)` so the inner loop can replay the
    /// matvec across thousands of iters with zero per-call alloc.
    ///
    /// Returns the dispatch handle that [`Self::matvec_into_pass`]
    /// consumes. Hold the handle alive for the lifetime of the
    /// repeated dispatches.
    pub fn prepare_matvec(
        &self,
        a: &BsrGpu,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        alpha: f32,
        beta: f32,
    ) -> Result<PreparedMatvec> {
        let uniforms = MatvecUniforms {
            nrows: a.nrows,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha,
            beta,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::prep-mv uniforms"),
            size: core::mem::size_of::<MatvecUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::prep-mv io bg"),
            layout: &self.layout_mv_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: y_buf.as_entire_binding() },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::prep-mv u bg"),
            layout: &self.layout_mv_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let nrows = a.nrows;
        let dim_x = nrows.min(MAX_DIM).max(1);
        let dim_y = nrows.div_ceil(MAX_DIM).max(1);

        Ok(PreparedMatvec {
            uniform_buf,
            bg_io,
            bg_u,
            dim_x,
            dim_y,
        })
    }

    /// Record a prepared matvec into an existing compute pass. The
    /// matvec runs as one extra dispatch — no extra encoder/submit.
    ///
    /// Note: `wgpu::ComputePass`'s `set_bind_group` and `set_pipeline`
    /// reborrow the supplied references for the *remainder* of the
    /// pass (until the next `set_*` call on the same slot), not
    /// statically — so the prepared handle only needs to outlive the
    /// dispatch we record here. wgpu's runtime tracks the actual
    /// resource lifetime independently.
    pub fn matvec_into_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        prepared: &PreparedMatvec,
    ) {
        pass.set_pipeline(&self.pipe_matvec);
        pass.set_bind_group(0, &prepared.bg_io, &[]);
        pass.set_bind_group(1, &prepared.bg_u, &[]);
        pass.dispatch_workgroups(prepared.dim_x, prepared.dim_y, 1);
    }

    /// Sparse axpy: `B = α · A + β · B` for the SAME-PATTERN case.
    ///
    /// **Safety contract:** caller asserts that A and B share
    /// **identical `row_offsets` AND `col_indices`** — only `values`
    /// differ. We verify only the cheap scalar shape `(nnz, nrows,
    /// ncols)` here; a true pattern check would require reading
    /// `row_offsets` + `col_indices` back to the host (~2 mappable
    /// readbacks per call) which makes the function unusable inside
    /// any per-iter solver loop. The pattern-union case is not
    /// implemented in v1 — see module docstring.
    ///
    /// If you pass mismatched patterns this routine produces
    /// **numerical garbage silently** (each slot in B's `values` is
    /// updated with α·A's same-slot value, but those slots address
    /// different `(row, col)` pairs). Build the pattern once via
    /// [`BsrGpu::upload`] and keep the same pattern across all axpy
    /// calls — or use `bsr_mm` / `bsr_axpy_union` (v2) for the
    /// pattern-changing case.
    pub fn axpy_same_pattern_unchecked(
        &self,
        a: &BsrGpu,
        b: &mut BsrGpu,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if a.nnz != b.nnz || a.nrows != b.nrows || a.ncols != b.ncols {
            return Err(Error::Build(format!(
                "BsrAlgebraKernel::axpy_same_pattern_unchecked: shape mismatch — \
                 A is {}x{} nnz={}, B is {}x{} nnz={}; same-pattern axpy requires \
                 identical (nrows, ncols, nnz). The pattern-union case is not \
                 implemented in v1.",
                a.nrows, a.ncols, a.nnz, b.nrows, b.ncols, b.nnz
            )));
        }
        if a.nnz == 0 {
            return Ok(());
        }
        let uniforms = AxpyUniforms {
            nnz: a.nnz,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha,
            beta,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::axpy uniforms"),
            size: core::mem::size_of::<AxpyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::axpy io bg"),
            layout: &self.layout_axpy_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.values.as_entire_binding() },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::axpy u bg"),
            layout: &self.layout_axpy_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = a.nnz.div_ceil(WG_SIZE).max(1);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::sparse_gpu::axpy encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_gpu::axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_axpy);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Transpose: `B = Aᵀ`. Returns a fresh [`BsrGpu`] of shape
    /// `(A.ncols, A.nrows)`.
    ///
    /// v1 splits the symbolic + numeric phases — see module docstring.
    /// Symbolic walks A's pattern host-side to build B's `row_offsets`
    /// + `col_indices` and a per-source-slot destination map; numeric
    /// dispatches `bsr_transpose_scatter` on the GPU. The symbolic step
    /// reads A's row/col buffers back ONCE.
    pub fn transpose(&self, a: &BsrGpu) -> Result<BsrGpu> {
        let nrows = a.nrows;
        let ncols = a.ncols;
        let nnz = a.nnz;

        // ── 1. Read A's pattern back to the host ───────────────────
        let row_off_bytes = ((nrows as u64) + 1) * 4;
        let col_idx_bytes = (nnz as u64) * 4;
        let a_row_off = read_u32_buffer(
            &self.device,
            &self.queue,
            &a.row_offsets,
            row_off_bytes,
            "transpose row_offsets",
        )?;
        let a_col_idx = read_u32_buffer(
            &self.device,
            &self.queue,
            &a.col_indices,
            col_idx_bytes,
            "transpose col_indices",
        )?;

        // ── 2. Symbolic transpose on host ──────────────────────────
        // bt_row_offsets[j+1] counts entries with column j in A.
        let mut bt_row_offsets = vec![0u32; (ncols as usize) + 1];
        for slot in 0..(nnz as usize) {
            let j = a_col_idx[slot] as usize;
            bt_row_offsets[j + 1] += 1;
        }
        // Prefix sum.
        for i in 1..=(ncols as usize) {
            bt_row_offsets[i] += bt_row_offsets[i - 1];
        }
        // Fill bt_col_indices + dst_map by walking A in row-major order.
        let mut bt_col_indices = vec![0u32; nnz as usize];
        let mut dst_map = vec![0u32; nnz as usize];
        let mut cursor = bt_row_offsets.clone(); // reused as per-column write head
        for i in 0..(nrows as usize) {
            let lo = a_row_off[i] as usize;
            let hi = a_row_off[i + 1] as usize;
            for slot in lo..hi {
                let j = a_col_idx[slot] as usize;
                let dst = cursor[j] as usize;
                bt_col_indices[dst] = i as u32;
                dst_map[slot] = dst as u32;
                cursor[j] += 1;
            }
        }

        // ── 3. Allocate B's three buffers ──────────────────────────
        let row_offsets_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::transpose B row_offsets",
            bytemuck::cast_slice(&bt_row_offsets),
        );
        let col_indices_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::transpose B col_indices",
            bytemuck::cast_slice(&bt_col_indices),
        );
        // Values buffer: allocate empty (zeroed) so the scatter has
        // somewhere to land. f32 storage.
        let values_bytes = (nnz as u64).max(4) * 4;
        let values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::transpose B values"),
            size: values_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero so any rows missing nonzeros stay 0 (defensive — the
        // scatter writes every slot, so this is just hygiene).
        self.queue
            .write_buffer(&values_buf, 0, &vec![0u8; values_bytes as usize]);

        // ── 4. Upload dst_map + dispatch the scatter shader ────────
        let dst_map_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::transpose dst_map",
            bytemuck::cast_slice(&dst_map),
        );

        if nnz > 0 {
            let uniforms = ScalarU32Uniforms {
                n: nnz,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::sparse_gpu::transpose uniforms"),
                size: core::mem::size_of::<ScalarU32Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

            let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::sparse_gpu::transpose io bg"),
                layout: &self.layout_tr_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.values.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst_map_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: values_buf.as_entire_binding() },
                ],
            });
            let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::sparse_gpu::transpose u bg"),
                layout: &self.layout_tr_u,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
            });
            let groups = nnz.div_ceil(WG_SIZE).max(1);
            let mut encoder =
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::transpose encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::sparse_gpu::transpose"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_transpose);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        Ok(BsrGpu {
            row_offsets: Arc::new(row_offsets_buf),
            col_indices: Arc::new(col_indices_buf),
            values: Arc::new(values_buf),
            nrows: ncols,
            ncols: nrows,
            nnz,
        })
    }

    /// Sparse matrix-matrix product `C = A · B`.
    ///
    /// **Stage 39F** runs the full pipeline on the GPU for problems
    /// with `max(A.nrows, B.ncols) ≥ SPGEMM_GPU_SYMBOLIC_CUTOVER`
    /// (currently 64). Below the threshold falls back to the
    /// host-symbolic v1 path (see [`Self::spgemm_host_symbolic`]) —
    /// kernel-launch + buffer-allocation overhead dominates at tiny
    /// sizes.
    pub fn spgemm(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu> {
        if a.ncols != b.nrows {
            return Err(Error::Build(format!(
                "BsrAlgebraKernel::spgemm: shape mismatch — A is {}x{}, B is {}x{}",
                a.nrows, a.ncols, b.nrows, b.ncols
            )));
        }
        let max_dim = a.nrows.max(b.ncols);
        if max_dim < SPGEMM_GPU_SYMBOLIC_CUTOVER {
            return self.spgemm_host_symbolic(a, b);
        }
        self.spgemm_gpu_symbolic(a, b)
    }

    /// Host-symbolic + GPU-numeric path (Stage 38a). Public-but-unstable
    /// — exposed so the parity tests + bench can compare both paths
    /// directly. Identical semantics to [`Self::spgemm`].
    #[doc(hidden)]
    pub fn spgemm_host_symbolic(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu> {
        if a.ncols != b.nrows {
            return Err(Error::Build(format!(
                "BsrAlgebraKernel::spgemm_host_symbolic: shape mismatch — A is {}x{}, B is {}x{}",
                a.nrows, a.ncols, b.nrows, b.ncols
            )));
        }
        let nrows = a.nrows;
        let ncols = b.ncols;

        // ── 1. Read both patterns back ─────────────────────────────
        let a_row_off = read_u32_buffer(
            &self.device,
            &self.queue,
            &a.row_offsets,
            ((a.nrows as u64) + 1) * 4,
            "spgemm A row_offsets",
        )?;
        let a_col_idx = read_u32_buffer(
            &self.device,
            &self.queue,
            &a.col_indices,
            (a.nnz as u64) * 4,
            "spgemm A col_indices",
        )?;
        let b_row_off = read_u32_buffer(
            &self.device,
            &self.queue,
            &b.row_offsets,
            ((b.nrows as u64) + 1) * 4,
            "spgemm B row_offsets",
        )?;
        let b_col_idx = read_u32_buffer(
            &self.device,
            &self.queue,
            &b.col_indices,
            (b.nnz as u64) * 4,
            "spgemm B col_indices",
        )?;

        // ── 2. Symbolic pass — build C's pattern + per-slot
        //       contribution lists.
        // For each row i of A, walk A[i, :]. For every nonzero (i, k)
        // walk B[k, :] and gather all (j, slot_a, slot_b) triples.
        // Sort + coalesce by j → that produces C.col_idx[lo..hi] and
        // groups the contribution pairs per output slot.
        let mut c_row_offsets = vec![0u32; (nrows as usize) + 1];
        // Per output slot: contribution pairs flat-packed (slot_a, slot_b).
        let mut c_col_idx: Vec<u32> = Vec::new();
        let mut contrib_offsets: Vec<u32> = vec![0];
        let mut contrib_pairs: Vec<u32> = Vec::new();

        // Scratch — temporary list of (j, slot_a, slot_b) for row i.
        let mut row_buf: Vec<(u32, u32, u32)> = Vec::new();

        for i in 0..(nrows as usize) {
            row_buf.clear();
            let a_lo = a_row_off[i] as usize;
            let a_hi = a_row_off[i + 1] as usize;
            for sa in a_lo..a_hi {
                let k = a_col_idx[sa] as usize;
                if k >= b.nrows as usize {
                    return Err(Error::Build(format!(
                        "BsrAlgebraKernel::spgemm: A col {k} out of B row range {}",
                        b.nrows
                    )));
                }
                let b_lo = b_row_off[k] as usize;
                let b_hi = b_row_off[k + 1] as usize;
                for sb in b_lo..b_hi {
                    let j = b_col_idx[sb];
                    row_buf.push((j, sa as u32, sb as u32));
                }
            }
            // Sort by j; coalesce equal-j triples into a single output
            // slot whose contribution list aggregates the underlying
            // (slot_a, slot_b) pairs.
            row_buf.sort_unstable_by_key(|t| t.0);

            let mut cursor = 0;
            while cursor < row_buf.len() {
                let j = row_buf[cursor].0;
                let mut end = cursor;
                while end < row_buf.len() && row_buf[end].0 == j {
                    end += 1;
                }
                // Append output column.
                c_col_idx.push(j);
                // Append contribution pairs for this slot.
                for &(_, sa, sb) in &row_buf[cursor..end] {
                    contrib_pairs.push(sa);
                    contrib_pairs.push(sb);
                }
                contrib_offsets.push((contrib_pairs.len() / 2) as u32);
                cursor = end;
            }
            c_row_offsets[i + 1] = c_col_idx.len() as u32;
        }
        let c_nnz = c_col_idx.len() as u32;

        // ── 3. Allocate C's three buffers + the contribution side data ──
        let c_row_offsets_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::spgemm C row_offsets",
            bytemuck::cast_slice(&c_row_offsets),
        );
        let c_col_idx_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::spgemm C col_indices",
            bytemuck::cast_slice(&c_col_idx),
        );
        let values_bytes = (c_nnz as u64).max(4) * 4;
        let c_values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm C values"),
            size: values_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&c_values_buf, 0, &vec![0u8; values_bytes as usize]);

        if c_nnz == 0 {
            return Ok(BsrGpu {
                row_offsets: Arc::new(c_row_offsets_buf),
                col_indices: Arc::new(c_col_idx_buf),
                values: Arc::new(c_values_buf),
                nrows,
                ncols,
                nnz: 0,
            });
        }

        let contrib_offsets_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::spgemm contrib_offsets",
            bytemuck::cast_slice(&contrib_offsets),
        );
        let contrib_pairs_buf = create_storage_buffer(
            &self.device,
            &self.queue,
            "molla::sparse_gpu::spgemm contrib_pairs",
            bytemuck::cast_slice(&contrib_pairs),
        );

        // ── 4. Numeric pass ────────────────────────────────────────
        let uniforms = ScalarU32Uniforms {
            n: c_nnz,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm uniforms"),
            size: core::mem::size_of::<ScalarU32Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::spgemm io bg"),
            layout: &self.layout_spgemm_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.values.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: contrib_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: contrib_pairs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 4, resource: c_values_buf.as_entire_binding() },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::spgemm u bg"),
            layout: &self.layout_spgemm_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = c_nnz.div_ceil(WG_SIZE).max(1);
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::sparse_gpu::spgemm encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_gpu::spgemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_spgemm);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(BsrGpu {
            row_offsets: Arc::new(c_row_offsets_buf),
            col_indices: Arc::new(c_col_idx_buf),
            values: Arc::new(c_values_buf),
            nrows,
            ncols,
            nnz: c_nnz,
        })
    }

    /// Full-GPU SpGEMM via the Stage 39F dense-bitmap symbolic. See
    /// the module docstring for the four-pass pipeline.
    #[doc(hidden)]
    pub fn spgemm_gpu_symbolic(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu> {
        if a.ncols != b.nrows {
            return Err(Error::Build(format!(
                "BsrAlgebraKernel::spgemm_gpu_symbolic: shape mismatch — A is {}x{}, B is {}x{}",
                a.nrows, a.ncols, b.nrows, b.ncols
            )));
        }
        let nrows = a.nrows;
        let ncols = b.ncols;
        if nrows == 0 || ncols == 0 {
            // Empty output. Allocate minimal buffers.
            let c_row_offsets_buf = create_storage_buffer(
                &self.device,
                &self.queue,
                "molla::sparse_gpu::spgemm C row_offsets (empty)",
                bytemuck::cast_slice(&vec![0u32; (nrows as usize) + 1]),
            );
            let c_col_idx_buf = create_storage_buffer(
                &self.device,
                &self.queue,
                "molla::sparse_gpu::spgemm C col_indices (empty)",
                &[],
            );
            let c_values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::sparse_gpu::spgemm C values (empty)"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            return Ok(BsrGpu {
                row_offsets: Arc::new(c_row_offsets_buf),
                col_indices: Arc::new(c_col_idx_buf),
                values: Arc::new(c_values_buf),
                nrows,
                ncols,
                nnz: 0,
            });
        }

        // ── Step 0: allocate per-row bitmap buffer ────────────────────
        // BITMAP_WORDS per row = ⌈ncols / 32⌉. Each row owns
        // BITMAP_WORDS u32 entries. The buffer is zeroed per-row inside
        // the count shader (so we don't pay a separate clear pass).
        let bitmap_words: u32 = ncols.div_ceil(32);
        let total_bitmap_u32 = (nrows as u64) * (bitmap_words as u64);
        let bitmap_bytes = (total_bitmap_u32 * 4).max(16);
        let bitmap_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm row_bitmaps"),
            size: bitmap_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // count[row] — u32 unique-col tally per row. One extra slot at
        // the end so the exclusive scan can produce `nrows+1` offsets
        // (we'll scan into a separate buffer of length `nrows+1`).
        let count_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm count"),
            size: ((nrows as u64) * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Step 1: symbolic_count ────────────────────────────────────
        let sym_count_u = SymCountUniforms {
            a_nrows: nrows,
            b_nrows: b.nrows,
            b_ncols: ncols,
            bitmap_words,
        };
        let sym_count_u_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::sym_count uniforms"),
            size: core::mem::size_of::<SymCountUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&sym_count_u_buf, 0, bytemuck::bytes_of(&sym_count_u));

        let bg_sym_count_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::sym_count io bg"),
            layout: &self.layout_sym_count_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bitmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: count_buf.as_entire_binding() },
            ],
        });
        let bg_sym_count_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::sym_count u bg"),
            layout: &self.layout_sym_count_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sym_count_u_buf.as_entire_binding(),
            }],
        });

        {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::spgemm sym_count enc"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::sparse_gpu::sym_count"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_sym_count);
                pass.set_bind_group(0, &bg_sym_count_io, &[]);
                pass.set_bind_group(1, &bg_sym_count_u, &[]);
                // One workgroup per A-row.
                let dim_x = nrows.min(MAX_DIM).max(1);
                let dim_y = nrows.div_ceil(MAX_DIM).max(1);
                pass.dispatch_workgroups(dim_x, dim_y, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // ── Step 2: scan count[] → c_row_offsets[] ────────────────────
        // The exclusive scan outputs `nrows` entries; we need `nrows + 1`
        // for the CSR convention. We do: scan into a length-`nrows+1`
        // buffer (with the last slot zero-padded) — the scan kernel
        // ignores the padded zero so `out[nrows] = sum(count[0..nrows])`.
        //
        // Concretely: build an input of length `nrows + 1` whose last
        // slot is 0, scan inclusively, and the inclusive[nrows] = total.
        // OR: scan exclusively over `count` (length nrows) into a length
        // `nrows + 1` buffer where slot `nrows` is filled separately.
        //
        // Simplest path: stage count into a length-(nrows+1) buffer,
        // then run exclusive_scan over `nrows + 1`. The "value" at slot
        // nrows is irrelevant since exclusive scan never reads slot `i`
        // for its own output — out[i] = sum(in[0..i]).
        let n_plus_1 = nrows + 1;
        let count_padded_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm count padded"),
            size: ((n_plus_1 as u64) * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-init the padded buffer (so slot `nrows` is 0).
        // Then copy `count[0..nrows]` into the first `nrows` slots.
        self.queue.write_buffer(
            &count_padded_buf,
            0,
            &vec![0u8; ((n_plus_1 as u64) * 4).max(16) as usize],
        );
        {
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::spgemm copy count"),
                },
            );
            enc.copy_buffer_to_buffer(
                &count_buf,
                0,
                &count_padded_buf,
                0,
                (nrows as u64) * 4,
            );
            self.queue.submit(std::iter::once(enc.finish()));
        }

        let c_row_offsets_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm C row_offsets"),
            size: ((n_plus_1 as u64) * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.scan
            .exclusive_scan_u32(&count_padded_buf, &c_row_offsets_buf, n_plus_1)?;

        // Read back the total nnz = c_row_offsets[nrows]. (One u32
        // readback — host-side blocking. The Stage 39F profile shows
        // this readback dominates the symbolic wall-time at small N
        // and is a wash at large N because we need the nnz to allocate
        // the col_indices + values buffers anyway.)
        let total_nnz_u32 = {
            let bytes = 4u64;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::sparse_gpu::spgemm total_nnz staging"),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::spgemm total_nnz enc"),
                },
            );
            enc.copy_buffer_to_buffer(
                &c_row_offsets_buf,
                (nrows as u64) * 4,
                &staging,
                0,
                bytes,
            );
            self.queue.submit(std::iter::once(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("sparse_gpu spgemm total_nnz poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("sparse_gpu spgemm total_nnz recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("sparse_gpu spgemm total_nnz map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let v = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
            drop(mapped);
            staging.unmap();
            v
        };

        let c_nnz = total_nnz_u32;

        // ── Step 3: allocate c_col_indices + c_values ─────────────────
        let c_col_idx_bytes = ((c_nnz as u64) * 4).max(16);
        let c_col_idx_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm C col_indices"),
            size: c_col_idx_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let c_values_bytes = ((c_nnz as u64) * 4).max(16);
        let c_values_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::spgemm C values"),
            size: c_values_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero c_values — the atomic-add numeric kernel accumulates
        // into it.
        self.queue
            .write_buffer(&c_values_buf, 0, &vec![0u8; c_values_bytes as usize]);

        if c_nnz == 0 {
            return Ok(BsrGpu {
                row_offsets: Arc::new(c_row_offsets_buf),
                col_indices: Arc::new(c_col_idx_buf),
                values: Arc::new(c_values_buf),
                nrows,
                ncols,
                nnz: 0,
            });
        }

        // ── Step 4: symbolic_write — enumerate set bits → c_col_indices.
        let sym_write_u = SymWriteUniforms {
            a_nrows: nrows,
            bitmap_words,
            _pad0: 0,
            _pad1: 0,
        };
        let sym_write_u_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::sym_write uniforms"),
            size: core::mem::size_of::<SymWriteUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&sym_write_u_buf, 0, bytemuck::bytes_of(&sym_write_u));

        let bg_sym_write_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::sym_write io bg"),
            layout: &self.layout_sym_write_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: c_row_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bitmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_col_idx_buf.as_entire_binding() },
            ],
        });
        let bg_sym_write_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::sym_write u bg"),
            layout: &self.layout_sym_write_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sym_write_u_buf.as_entire_binding(),
            }],
        });

        {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::spgemm sym_write enc"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::sparse_gpu::sym_write"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_sym_write);
                pass.set_bind_group(0, &bg_sym_write_io, &[]);
                pass.set_bind_group(1, &bg_sym_write_u, &[]);
                let dim_x = nrows.min(MAX_DIM).max(1);
                let dim_y = nrows.div_ceil(MAX_DIM).max(1);
                pass.dispatch_workgroups(dim_x, dim_y, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // ── Step 5: numeric_atomic_add ────────────────────────────────
        let num_u = NumericAtomicUniforms {
            a_nrows: nrows,
            b_nrows: b.nrows,
            _pad0: 0,
            _pad1: 0,
        };
        let num_u_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_gpu::num_atomic uniforms"),
            size: core::mem::size_of::<NumericAtomicUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&num_u_buf, 0, bytemuck::bytes_of(&num_u));

        let bg_num_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::num_atomic io bg"),
            layout: &self.layout_num_atomic_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b.col_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: c_row_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: c_col_idx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: c_values_buf.as_entire_binding() },
            ],
        });
        let bg_num_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_gpu::num_atomic u bg"),
            layout: &self.layout_num_atomic_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: num_u_buf.as_entire_binding(),
            }],
        });

        {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sparse_gpu::spgemm numeric enc"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::sparse_gpu::numeric_atomic_add"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_num_atomic);
                pass.set_bind_group(0, &bg_num_io, &[]);
                pass.set_bind_group(1, &bg_num_u, &[]);
                let dim_x = nrows.min(MAX_DIM).max(1);
                let dim_y = nrows.div_ceil(MAX_DIM).max(1);
                pass.dispatch_workgroups(dim_x, dim_y, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        Ok(BsrGpu {
            row_offsets: Arc::new(c_row_offsets_buf),
            col_indices: Arc::new(c_col_idx_buf),
            values: Arc::new(c_values_buf),
            nrows,
            ncols,
            nnz: c_nnz,
        })
    }
}

// ─── Internal helpers ───────────────────────────────────────────────

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

fn create_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
) -> wgpu::Buffer {
    let unpadded = data.len() as u64;
    let size = unpadded.max(16).next_multiple_of(4);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buf, 0, data);
    }
    buf
}

fn read_u32_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    bytes: u64,
    label: &str,
) -> Result<Vec<u32>> {
    let size = bytes.max(16);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::sparse_gpu::readback enc"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("sparse_gpu read poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("sparse_gpu read recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("sparse_gpu read map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let v: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(v)
}

fn read_f32_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    bytes: u64,
    label: &str,
) -> Result<Vec<f32>> {
    let size = bytes.max(16);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::sparse_gpu::readback enc"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("sparse_gpu read poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("sparse_gpu read recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("sparse_gpu read map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let v: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(v)
}
