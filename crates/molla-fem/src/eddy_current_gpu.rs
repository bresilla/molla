//! GPU eddy-current steady solve. Mirrors the CPU
//! [`solve_eddy_current_steady`](crate::eddy_current::solve_eddy_current_steady):
//! the block-real system is assembled host-side (as in the CPU path),
//! then the GMRES solve runs on the device via
//! [`bsr_gmres_gpu`](molla_kernels::bsr_gmres_gpu) — the sparse mat-vecs
//! that dominate the iteration cost now execute on the GPU. Jacobi
//! (diagonal) preconditioning matches the CPU solver.

use crate::eddy_current::{EddyCurrentConfig, EddyCurrentResult, assemble_eddy_system};
use crate::math::Real;
use crate::math::sparse::BsrMatrix;
use crate::mesh::FemMesh;
use molla_core::{Error, Result};
use molla_kernels::{BsrAlgebraKernel, BsrGpu, bsr_gmres_gpu};
use molla_math::Vec3;

/// Assemble the `[[K_main, K_off], [-K_off, K_main]]` block-real system
/// into a single `2n × 2n` scalar BSR matrix.
fn assemble_block(k_main: &BsrMatrix, k_off: &BsrMatrix, n: usize) -> Result<BsrMatrix> {
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for i in 0..n {
        for slot in k_main.row_offsets[i]..k_main.row_offsets[i + 1] {
            let j = k_main.col_idx[slot];
            let v = k_main.values[slot];
            triplets.push((i, j, v)); // (0,0)
            triplets.push((n + i, n + j, v)); // (1,1)
        }
        for slot in k_off.row_offsets[i]..k_off.row_offsets[i + 1] {
            let j = k_off.col_idx[slot];
            let v = k_off.values[slot];
            triplets.push((i, n + j, v)); // (0,1) = +K_off
            triplets.push((n + i, j, -v)); // (1,0) = -K_off
        }
    }
    BsrMatrix::from_scalar_triplets(2 * n, 2 * n, &triplets)
        .map_err(|e| Error::Fem(format!("eddy_current_gpu: block assembly failed: {e}")))
}

/// GPU counterpart of [`solve_eddy_current_steady`](crate::eddy_current::solve_eddy_current_steady).
/// Produces the same `(A_re, A_im)` within f32 tolerance.
#[allow(clippy::too_many_arguments)]
pub fn solve_eddy_current_steady_gpu(
    engine: &molla_compute::Engine,
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[crate::bc::BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    let (map, k_main, k_off, f_re, f_im, _pinned) = assemble_eddy_system(
        mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
    )?;
    let n = map.ndof();

    // Block Jacobi inverse diagonal — diag is K_main[r,r] on both halves.
    let diag_main = k_main.diagonal();
    let mut inv_diag = vec![0.0; 2 * n];
    for i in 0..n {
        let d = diag_main[i];
        let inv = if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 };
        inv_diag[i] = inv;
        inv_diag[n + i] = inv;
    }

    let block = assemble_block(&k_main, &k_off, n)?;
    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);

    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("eddy_current_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("eddy_current_gpu: engine has no queue".into()))?
        .clone();
    let block_gpu = BsrGpu::upload(&device, &queue, &block)?;
    let kernel = BsrAlgebraKernel::new(device, queue);

    let res = bsr_gmres_gpu(
        &kernel,
        &block_gpu,
        &b,
        Some(&inv_diag),
        cfg.tol,
        cfg.max_iter,
        cfg.restart.max(1),
    )?;

    Ok(EddyCurrentResult {
        a_re: res.x[..n].to_vec(),
        a_im: res.x[n..].to_vec(),
        residual: res.residual,
        iters: res.iters,
    })
}
