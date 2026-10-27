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

fn norm(v: &[Real]) -> Real {
    dot(v, v).sqrt()
}

fn axpy(y: &mut [Real], a: Real, x: &[Real]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// Summary of a [`bsr_gmres_gpu`] solve.
#[derive(Clone, Debug)]
pub struct BsrGmresResult {
    pub x: Vec<Real>,
    pub iters: usize,
    pub residual: Real,
    pub converged: bool,
}

/// Solve `A x = b` for a general (non-symmetric) `A` with restarted
/// GMRES(`restart`), `x0 = 0`, optional diagonal (Jacobi) left
/// preconditioner `m_inv_diag`. The per-Arnoldi sparse mat-vec runs on
/// the GPU; the Arnoldi/Givens recurrence is host-side — a faithful
/// mirror of the CPU [`gmres`](molla_math::sparse::gmres), so it matches
/// within f32 tolerance. This is the device-side non-symmetric solver
/// core (e.g. for eddy-current).
pub fn bsr_gmres_gpu(
    kernel: &BsrAlgebraKernel,
    a_gpu: &BsrGpu,
    b: &[Real],
    m_inv_diag: Option<&[Real]>,
    tol: Real,
    max_iter: usize,
    restart: usize,
) -> Result<BsrGmresResult> {
    let n = b.len();
    let mut x = vec![0.0 as Real; n];
    if n == 0 {
        return Ok(BsrGmresResult {
            x,
            iters: 0,
            residual: 0.0,
            converged: true,
        });
    }

    let device = kernel.device().clone();
    let queue = kernel.queue().clone();
    let mut p_buf = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    p_buf.ensure_device(&device)?;
    let mut ap_buf = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    ap_buf.ensure_device(&device)?;
    ap_buf.upload(&queue)?;

    // A·v on the device.
    let mut mv = |v: &[Real]| -> Result<Vec<Real>> {
        {
            let h = p_buf.host_mut()?;
            h.copy_from_slice(v);
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
        Ok(ap_buf.host()?.to_vec())
    };

    let apply_m_inv = |r: &[Real]| -> Vec<Real> {
        match m_inv_diag {
            Some(d) => r.iter().zip(d.iter()).map(|(ri, di)| ri * di).collect(),
            None => r.to_vec(),
        }
    };

    let m = restart.max(1).min(n);
    let bnorm = norm(b).max(1.0e-30);
    let mut total_iters = 0usize;
    let mut residual = 1.0;

    while total_iters < max_iter {
        let ax = mv(&x)?;
        let mut r = vec![0.0; n];
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        let z = apply_m_inv(&r);
        let beta = norm(&z);
        residual = beta / bnorm;
        if residual < tol {
            return Ok(BsrGmresResult {
                x,
                iters: total_iters,
                residual,
                converged: true,
            });
        }

        let mut v_basis: Vec<Vec<Real>> = vec![Vec::new(); m + 1];
        v_basis[0] = z.iter().map(|zi| zi / beta).collect();
        let mut h = vec![0.0; (m + 1) * m];
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut converged = false;
        let mut k_done = m;

        for k in 0..m {
            let w_raw = mv(&v_basis[k])?;
            let mut w = apply_m_inv(&w_raw);
            for j in 0..=k {
                h[j * m + k] = dot(&v_basis[j], &w);
                axpy(&mut w, -h[j * m + k], &v_basis[j]);
            }
            h[(k + 1) * m + k] = norm(&w);
            if h[(k + 1) * m + k].abs() > 1.0e-30 {
                v_basis[k + 1] = w.iter().map(|z| z / h[(k + 1) * m + k]).collect();
            } else {
                v_basis[k + 1] = vec![0.0; n];
            }

            for i in 0..k {
                let temp = cs[i] * h[i * m + k] + sn[i] * h[(i + 1) * m + k];
                h[(i + 1) * m + k] = -sn[i] * h[i * m + k] + cs[i] * h[(i + 1) * m + k];
                h[i * m + k] = temp;
            }
            let h_kk = h[k * m + k];
            let h_k1k = h[(k + 1) * m + k];
            let r_kk = (h_kk * h_kk + h_k1k * h_k1k).sqrt();
            cs[k] = h_kk / r_kk;
            sn[k] = h_k1k / r_kk;
            h[k * m + k] = r_kk;
            h[(k + 1) * m + k] = 0.0;
            let g_temp = cs[k] * g[k] + sn[k] * g[k + 1];
            g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1];
            g[k] = g_temp;

            total_iters += 1;
            if total_iters >= max_iter {
                k_done = k + 1;
                break;
            }
            if g[k + 1].abs() / bnorm < tol {
                converged = true;
                k_done = k + 1;
                break;
            }
        }

        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let mut s = g[i];
            for j in (i + 1)..k_done {
                s -= h[i * m + j] * y[j];
            }
            y[i] = s / h[i * m + i];
        }
        for i in 0..k_done {
            axpy(&mut x, y[i], &v_basis[i]);
        }
        residual = g[k_done].abs() / bnorm;
        if converged {
            return Ok(BsrGmresResult {
                x,
                iters: total_iters,
                residual,
                converged: true,
            });
        }
    }

    Ok(BsrGmresResult {
        x,
        iters: total_iters,
        residual,
        converged: false,
    })
}
