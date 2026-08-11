//! GPU conjugate-gradient solve over a sparse BSR operator. The
//! expensive sparse mat-vec runs on the device via
//! [`BsrAlgebraKernel::matvec`](crate::sparse_gpu::BsrAlgebraKernel::matvec);
//! the Krylov recurrence (dots / axpy) is driven host-side. This is the
//! device-side symmetric-solver core the CPU-only FEM modules
//! (magnetostatics, Poisson-type systems) can route through instead of
//! the host `cg_solve`.

use crate::sparse_gpu::{BsrAlgebraKernel, BsrGpu};
use molla_core::Result;
use molla_math::Real;
use molla_storage::{Buffer, BufferUsage};

/// Summary of a [`bsr_cg_gpu`] solve.
#[derive(Clone, Debug)]
pub struct BsrCgResult {
    pub x: Vec<Real>,
    pub iters: usize,
    pub residual: Real,
}

/// Solve `A x = b` for symmetric-positive-definite `A` (scalar BSR,
/// `b = 1`) with conjugate gradients, `x0 = 0`. The per-iteration
/// sparse mat-vec `A·p` is dispatched on the GPU; the residual/search
/// recurrence runs on the host. Matches the host
/// [`cg_solve`](molla_math::iterative::cg_solve) within f32 tolerance.
pub fn bsr_cg_gpu(
    kernel: &BsrAlgebraKernel,
    a_gpu: &BsrGpu,
    b: &[Real],
    tol: Real,
    max_iters: usize,
) -> Result<BsrCgResult> {
    let n = b.len();
    let mut x = vec![0.0 as Real; n];
    if n == 0 {
        return Ok(BsrCgResult {
            x,
            iters: 0,
            residual: 0.0,
        });
    }

    let device = kernel.device().clone();
    let queue = kernel.queue().clone();

    // Device scratch for the mat-vec: p (input) and ap (output).
    let mut p_buf = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    p_buf.ensure_device(&device)?;
    let mut ap_buf = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    ap_buf.ensure_device(&device)?;
    ap_buf.upload(&queue)?;

    // x0 = 0 ⇒ r = b, p = b.
    let mut r = b.to_vec();
    let mut p = b.to_vec();
    let mut rs_old = dot(&r, &r);
    let mut residual = rs_old.sqrt();
    let mut iters = 0;

    if residual <= tol {
        return Ok(BsrCgResult { x, iters, residual });
    }

    for it in 0..max_iters {
        iters = it + 1;
        // Ap = A · p on the device.
        {
            let host = p_buf.host_mut()?;
            host.copy_from_slice(&p);
        }
        p_buf.upload(&queue)?;
        kernel.matvec(
            a_gpu,
            p_buf.device_buffer().unwrap(),
            ap_buf.device_buffer().unwrap(),
            1.0,
            0.0,
        )?;
        ap_buf.mark_device_dirty();
        ap_buf.download(&device, &queue)?;
        let ap = ap_buf.host()?;

        let p_ap = dot(&p, ap);
        if p_ap <= 0.0 || p_ap.is_nan() {
            break; // not SPD / numerically stalled
        }
        let alpha = rs_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        residual = rs_new.sqrt();
        if residual <= tol {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(BsrCgResult { x, iters, residual })
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
