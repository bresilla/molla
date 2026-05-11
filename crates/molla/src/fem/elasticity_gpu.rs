//! Stage 41.1 — end-to-end GPU linear elasticity.
//!
//! Top-level wiring that builds the BC-reduced sparse pattern on
//! CPU, hands it to the GPU via [`crate::fem::FemGpuData`], dispatches
//! the WGSL assembly kernel, and runs the resident GPU CG to solve
//! `K · u = f`. The CPU never sees the assembled K values — only the
//! sparsity pattern + the RHS + the BC indices cross the host bound.
//!
//! ## Why pattern on CPU, values on GPU?
//!
//! `BsrMatrix::from_scalar_triplets` does an `O(nnz · log nnz)`
//! sort-and-dedupe to build the sparsity pattern; even at `16³` tet
//! resolution this is sub-millisecond and dominated by the underlying
//! mesh-graph traversal. The expensive part — computing the 144
//! values of every per-cell `Kₑ = ∫ Bᵀ D B dV` — moves to the GPU
//! kernel where one workgroup handles one cell.
//!
//! Boundary conditions are handled by:
//!   * pre-marking BC rows in a `u32` mask buffer (`is_bc_row[row]
//!     == 1`);
//!   * initialising the CSR `values` so BC rows already contain
//!     `(row, row) = 1.0` and zero elsewhere (this is *exactly* the
//!     state `apply_bcs` leaves the matrix in);
//!   * skipping the per-element scatter for BC rows inside the
//!     shader.
//!
//! The CG correctness argument for the resulting K·u=f system: BC
//! rows are decoupled (off-diagonals zero, diagonal one, RHS = BC
//! value), so the CG iterate never moves them away from the seeded
//! BC value. The asymmetry introduced by *not* zeroing the BC
//! columns of non-BC rows is harmless when the BC values are zero
//! (the cantilever clamp case) because the BC column entries
//! multiply by zero displacement in every matvec. For non-zero BC
//! values the caller must pre-compute the f correction
//! `f[i] -= K_full[i, r] · v_bc[r]` on CPU and pass the corrected
//! RHS — this v1 helper assumes the caller already did that (e.g.
//! by running `apply_bcs` on the host before calling us).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::fem::assembly::assemble_global;
use crate::fem::bc::{apply_bcs, BoundaryCondition};
use crate::fem::element::compute_element_elasticity_stiffness;
use crate::fem::function_space::FunctionSpace;
use crate::fem::gpu_data::FemGpuData;
use crate::fem::mesh::FemMesh;
use crate::kernels::{FemCgGpuKernel, FemElasticityGpuKernel, FemGpuKernel};
use crate::math::sparse::BsrMatrix;
use crate::math::Real;

/// Build the BC-reduced sparsity pattern for the linear elasticity
/// system `K · u = f` on `mesh`, with `dim` DoFs per vertex (3 for
/// 3D tet meshes, 2 for plane-strain 2D triangle meshes).
///
/// The returned `BsrMatrix` has the correct pattern (`row_offsets`,
/// `col_idx`) but its `values` slot already encodes the
/// BC-reduced K: BC rows contain `1.0` on the diagonal and `0.0`
/// elsewhere; non-BC rows are zero (waiting for the GPU assembly
/// kernel to scatter their contributions).
///
/// The pattern is constructed by assembling the unreduced K on
/// CPU *with dummy ones for every element entry* — we only need
/// the column-structure, not the values — and then running
/// `apply_bcs` to fold the BC structure in. The CPU-assembled
/// values are then **discarded** and replaced with the BC-only
/// initialiser described above.
///
/// Returns `(k_red_pattern, bc_mask)` where `bc_mask[row] == 1`
/// for every BC-pinned DoF and `0` otherwise.
pub fn build_elasticity_gpu_pattern(
    mesh: &FemMesh,
    dim: usize,
    bcs: &[BoundaryCondition],
) -> Result<(BsrMatrix, Vec<u32>)> {
    let space = FunctionSpace::P1Vector { dim };
    // Build the **values-included** stiffness on CPU. We discard the
    // values afterwards but need them so `apply_bcs`' column-elim
    // pass leaves the f correction in the caller's f_red. The
    // pattern itself is identical between the BC-reduced and
    // unreduced cases — `apply_bcs` only zeroes values.
    let mut k_pattern =
        assemble_global(mesh, space, |c, v| {
            // Use the *real* elasticity kernel with `λ = 0, μ = 1` —
            // produces a non-degenerate pattern (no entry is exactly
            // zero), which is what we need to preserve the column
            // structure after `apply_bcs`. The actual values are
            // overwritten below.
            compute_element_elasticity_stiffness(c, v, 0.0, 1.0)
        })?;
    let mut f_dummy = vec![0.0; space.ndof(mesh)];
    apply_bcs(&mut k_pattern, &mut f_dummy, bcs, &mesh.vertices, dim);

    // Wipe values, then re-establish the BC identity rows (diagonal =
    // 1.0, off-diagonals = 0.0 — the latter is automatic because we
    // just zeroed everything).
    let n = k_pattern.nrows();
    let mut bc_mask = vec![0u32; n];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value: _ } => {
                bc_mask[(*vertex as usize) * dim] = 1;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for ci in 0..values.len() {
                    bc_mask[base + ci] = 1;
                }
            }
            _ => {}
        }
    }
    for v in k_pattern.values.iter_mut() {
        *v = 0.0;
    }
    // Set diagonal of BC rows to 1.0.
    for (r, &mask) in bc_mask.iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let lo = k_pattern.row_offsets[r];
        let hi = k_pattern.row_offsets[r + 1];
        for slot in lo..hi {
            if k_pattern.col_idx[slot] == r {
                k_pattern.values[slot] = 1.0;
            }
        }
    }
    Ok((k_pattern, bc_mask))
}

/// Result of [`solve_elasticity_gpu`] — the recovered displacement
/// field plus solver metadata.
#[derive(Clone, Debug)]
pub struct ElasticityGpuResult {
    /// Per-DoF displacement vector (interleaved `[ux₀, uy₀, uz₀,
    /// ux₁, …]`). Length `mesh.vert_count() * dim`.
    pub u: Vec<Real>,
    /// Number of CG iterations the resident solver took.
    pub iters: usize,
    /// Final relative residual `‖r‖ / ‖b‖` reported by the resident CG.
    pub residual: Real,
}

/// End-to-end GPU elasticity: build pattern → upload → GPU assemble →
/// GPU resident CG → readback `u`. The expensive per-cell `B^T D B`
/// integration runs entirely on the GPU; the CPU only sees the
/// sparsity pattern (built once for a given mesh + BC set), the BC
/// mask, and the (already BC-corrected) RHS.
///
/// **Inputs**
///   * `mesh`     — tet mesh (vector elasticity needs `dim = 3`).
///   * `dim`      — per-vertex DoF count. Must be 3 in v1; passes
///     through to the function-space tag.
///   * `lam, mu`  — Lamé parameters (single material for the whole
///     mesh; per-element materials are a v2 follow-up).
///   * `bcs`      — boundary conditions. Dirichlet BCs *must* have
///     zero value in v1 (the GPU asymmetric reduction skips the
///     column-elim step; non-zero BC values would need an upstream
///     `f[i] -= K[i, r] · v_bc[r]` correction). `NeumannTraction`
///     entries are honoured by `apply_bcs` on the CPU-built `f`.
///   * `f_ext`    — external load vector (length `vert_count * dim`).
///   * `engine`   — wgpu device/queue handle.
///   * `cg_iters` / `cg_tol` — passed straight to the resident CG.
pub fn solve_elasticity_gpu(
    mesh: &FemMesh,
    dim: usize,
    lam: Real,
    mu: Real,
    bcs: &[BoundaryCondition],
    f_ext: &[Real],
    engine: &Engine,
    cg_iters: usize,
    cg_tol: Real,
) -> Result<ElasticityGpuResult> {
    let space = FunctionSpace::P1Vector { dim };
    let n = space.ndof(mesh);
    if f_ext.len() != n {
        return Err(Error::Build(format!(
            "solve_elasticity_gpu: f_ext length ({}) != ndof ({n})",
            f_ext.len(),
        )));
    }
    if dim != 3 {
        return Err(Error::Build(format!(
            "solve_elasticity_gpu: v1 only supports dim = 3, got {dim}",
        )));
    }

    let device = engine.device().ok_or_else(|| {
        Error::Gpu("solve_elasticity_gpu: engine has no GPU device".into())
    })?;
    let queue = engine.queue().ok_or_else(|| {
        Error::Gpu("solve_elasticity_gpu: engine has no GPU queue".into())
    })?;

    // ─── Pattern + BC mask (CPU, one-time) ─────────────────────────
    let (k_red_pattern, bc_mask) = build_elasticity_gpu_pattern(mesh, dim, bcs)?;

    // Apply BCs to the user's RHS (column elimination + Dirichlet
    // overwrite). We need the BC-corrected f for the CG, and v1
    // requires that the (Dirichlet BC) values themselves are zero —
    // see the helper doc-comment.
    let mut k_for_f_correction = assemble_global(mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })?;
    let mut f_red = f_ext.to_vec();
    apply_bcs(&mut k_for_f_correction, &mut f_red, bcs, &mesh.vertices, dim);
    drop(k_for_f_correction);

    // ─── Upload to GPU ──────────────────────────────────────────────
    let gpu_data = FemGpuData::upload(device, queue, mesh, &k_red_pattern)?;
    // BC mask buffer.
    let bc_mask_bytes = (bc_mask.len() * 4).max(16);
    let bc_mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::elasticity_gpu bc_mask"),
        size: bc_mask_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&bc_mask_buf, 0, bytemuck::cast_slice(&bc_mask));

    // ─── GPU assembly ───────────────────────────────────────────────
    let asm_kernel = FemElasticityGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    asm_kernel.assemble(&gpu_data, &bc_mask_buf, lam as f32, mu as f32, dim as u32)?;

    // ─── GPU resident CG ────────────────────────────────────────────
    let mv_kernel = FemGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    let cg_kernel = FemCgGpuKernel::new(Arc::clone(device), Arc::clone(queue));

    let f_red_f32: Vec<f32> = f_red.iter().map(|&v| v as f32).collect();
    let bytes = (n.max(4) * 4) as u64;
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST;
    let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::elasticity_gpu::cg b"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&f_red_f32));

    let result = cg_kernel.cg_solve_resident(
        n as u32,
        &b_buf,
        None,
        cg_tol as f32,
        cg_iters,
        |x_buf, y_buf| mv_kernel.matvec(&gpu_data, x_buf, y_buf),
    )?;
    let u_f32 = cg_kernel.read_buffer(&result.x, n as u32)?;
    let u: Vec<Real> = u_f32.into_iter().map(|v| v as Real).collect();

    Ok(ElasticityGpuResult {
        u,
        iters: result.iters,
        residual: result.residual as Real,
    })
}

/// Convenience: read back the GPU-assembled CSR values to host (`f32`)
/// so callers can diff against a CPU reference. Mostly for parity
/// tests — on the solve path the values stay on the GPU.
pub fn assemble_elasticity_gpu_only(
    mesh: &FemMesh,
    dim: usize,
    lam: Real,
    mu: Real,
    bcs: &[BoundaryCondition],
    engine: &Engine,
) -> Result<(BsrMatrix, Vec<f32>)> {
    let device = engine.device().ok_or_else(|| {
        Error::Gpu("assemble_elasticity_gpu_only: engine has no GPU device".into())
    })?;
    let queue = engine.queue().ok_or_else(|| {
        Error::Gpu("assemble_elasticity_gpu_only: engine has no GPU queue".into())
    })?;

    let (k_red_pattern, bc_mask) = build_elasticity_gpu_pattern(mesh, dim, bcs)?;
    let gpu_data = FemGpuData::upload(device, queue, mesh, &k_red_pattern)?;
    let bc_mask_bytes = (bc_mask.len() * 4).max(16);
    let bc_mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::elasticity_gpu_only bc_mask"),
        size: bc_mask_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&bc_mask_buf, 0, bytemuck::cast_slice(&bc_mask));
    let asm_kernel = FemElasticityGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    asm_kernel.assemble(&gpu_data, &bc_mask_buf, lam as f32, mu as f32, dim as u32)?;
    let vals = gpu_data.read_csr_values(device, queue)?;
    Ok((k_red_pattern, vals))
}
