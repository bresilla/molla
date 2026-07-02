//! Iterative linear solvers — CG, Preconditioned CG, BiCGSTAB.
//!
//! These complement the direct dense solvers in
//! [`crate::dense_solve`] (Cholesky / pseudoinverse) with
//! Krylov-subspace methods that scale to large systems and only need
//! the matrix as a *callable* — they never materialise it.
//!
//! Unlike the lower-level [`crate::sparse`] solvers (which take
//! `&dyn LinearOperator` and operate in-place on a mutable `x`), these
//! present a cleaner *closure-driven* API that returns a fresh result
//! struct: ideal for ad-hoc consumers who already have an MV product
//! lying around (implicit MPM at Stage 24d.4, Style3D, FEM, …).
//!
//! Conventions:
//! - Newton / molla are **f64 on the host**; everything here is `f64`.
//! - `op` represents `A · x`; it must be a self-adjoint mapping for
//!   `cg_solve` / `pcg_solve`, no symmetry constraint for `bicgstab`.
//! - `tol` is a relative tolerance against `‖b‖`: convergence when
//!   `‖b - A·x‖ ≤ tol · max(‖b‖, ε)`.
//! - All three solvers return after `max_iters` if they fail to
//!   converge — the caller inspects [`CgResult::residual`] and decides
//!   what to do next (typically: relax `tol`, swap preconditioner, or
//!   fall back to a direct solve).
//!
//! GPU port lives at Stage 27.1 (deferred — wgpu has no native sparse
//! BLAS, the WGSL kernels need to be hand-written per consumer).

use crate::Real;

/// Outcome of a single iterative solve.
#[derive(Clone, Debug)]
pub struct CgResult {
    /// The approximate solution `x` such that `A · x ≈ b`.
    pub x: Vec<Real>,
    /// Number of iterations consumed (0 ≤ iters ≤ max_iters).
    pub iters: usize,
    /// Relative residual `‖b - A·x‖ / max(‖b‖, ε)` at exit. Less than
    /// `tol` ⇒ converged; otherwise the solver bailed at `max_iters`.
    pub residual: Real,
}

impl CgResult {
    /// Convenience: did the solve hit the requested tolerance?
    #[must_use]
    pub fn converged(&self, tol: Real) -> bool {
        self.residual <= tol
    }
}

/// Diagonal (Jacobi) preconditioner: `M⁻¹ = diag(1 / aᵢᵢ)`.
///
/// Cheapest possible preconditioner — one division per row per matvec.
/// Effective when the diagonal already dominates (mass matrices,
/// implicit-MPM grid systems, finite-difference Laplacians with
/// non-uniform coefficients, …) and harmless on well-conditioned
/// systems. For SPD systems, `JacobiPreconditioner` is itself SPD,
/// preserving the symmetry guarantee that `pcg_solve` requires.
///
/// Plug into [`pcg_solve`] via the `apply` closure — see the example
/// in the [`pcg_solve`] doc comment.
#[derive(Clone, Debug)]
pub struct JacobiPreconditioner {
    /// `M_diag_inv[i] = 1 / diag_i` (zero diagonals are mapped to 0
    /// — i.e. those rows are passed through untouched, which is the
    /// safe fallback for null-space rows).
    inv_diag: Vec<Real>,
}

impl JacobiPreconditioner {
    /// Build from the matrix diagonal. Zero entries are treated as
    /// pass-through (`1` is *not* substituted because that would scale
    /// non-zero RHS in a null-space row).
    #[must_use]
    pub fn new(diag: &[Real]) -> Self {
        let inv_diag: Vec<Real> = diag
            .iter()
            .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
            .collect();
        Self { inv_diag }
    }

    /// Apply `M⁻¹ · r` and return a fresh vector. Length-checked at
    /// the call site (the closure passed into `pcg_solve` already
    /// trusts the dimension).
    #[must_use]
    pub fn apply(&self, r: &[Real]) -> Vec<Real> {
        debug_assert_eq!(
            r.len(),
            self.inv_diag.len(),
            "JacobiPreconditioner::apply: length mismatch (r={}, diag={})",
            r.len(),
            self.inv_diag.len(),
        );
        let n = r.len().min(self.inv_diag.len());
        let mut z = vec![0.0; n];
        for i in 0..n {
            z[i] = r[i] * self.inv_diag[i];
        }
        z
    }

    /// Borrow the inverse-diagonal vector (length matches the original
    /// diagonal).
    #[must_use]
    pub fn inv_diag(&self) -> &[Real] {
        &self.inv_diag
    }
}

// ─── Internal vector helpers (kept private; the `sparse` module has
// equivalent in-place primitives we don't reuse here because that
// module's solvers operate on a mutable `&mut [Real]` while we need
// fresh `Vec<Real>` for the closure-style API). ──────────────────────

#[inline]
fn dot(a: &[Real], b: &[Real]) -> Real {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn norm(a: &[Real]) -> Real {
    dot(a, a).sqrt()
}

#[inline]
fn axpy(y: &mut [Real], alpha: Real, x: &[Real]) {
    debug_assert_eq!(x.len(), y.len());
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

/// Conjugate Gradient for symmetric positive-definite `A · x = b`.
///
/// Matrix-free interface: `op(v)` returns `A · v` as a fresh vector.
/// `op` must implement a self-adjoint, positive-definite linear
/// operator — convergence guarantees fail otherwise (use
/// [`bicgstab_solve`] for general systems).
///
/// `x0` is the initial guess (zero vector if you have nothing better);
/// `tol` is the relative residual tolerance against `‖b‖`; `max_iters`
/// caps the work. Returns the solution + iteration count + final
/// relative residual regardless of whether convergence was reached —
/// inspect [`CgResult::residual`].
///
/// Shipping the full algorithm here (rather than delegating to
/// [`crate::sparse::cg`]) keeps the public matrix-free API free
/// of the `LinearOperator` trait — consumers can call this with a
/// closure or method reference directly.
pub fn cg_solve<F>(op: F, b: &[Real], x0: &[Real], tol: Real, max_iters: usize) -> CgResult
where
    F: Fn(&[Real]) -> Vec<Real>,
{
    let n = b.len();
    assert_eq!(
        x0.len(),
        n,
        "cg_solve: x0 length {} != b length {}",
        x0.len(),
        n
    );

    let bnorm = norm(b);
    if bnorm <= 1.0e-30 {
        // Zero RHS ⇒ zero solution (within tolerance trivially).
        return CgResult {
            x: vec![0.0; n],
            iters: 0,
            residual: 0.0,
        };
    }

    let mut x = x0.to_vec();
    // r₀ = b − A·x₀
    let ax0 = op(&x);
    debug_assert_eq!(ax0.len(), n);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax0[i];
    }
    let mut rr = dot(&r, &r);
    let mut residual = rr.sqrt() / bnorm;
    if residual <= tol {
        return CgResult {
            x,
            iters: 0,
            residual,
        };
    }

    let mut p = r.clone();
    for it in 0..max_iters {
        let ap = op(&p);
        debug_assert_eq!(ap.len(), n);
        let pap = dot(&p, &ap);
        if pap.abs() < 1.0e-30 {
            // Degenerate search direction — give up gracefully.
            residual = rr.sqrt() / bnorm;
            return CgResult {
                x,
                iters: it,
                residual,
            };
        }
        let alpha = rr / pap;
        axpy(&mut x, alpha, &p);
        axpy(&mut r, -alpha, &ap);
        let rr_new = dot(&r, &r);
        residual = rr_new.sqrt() / bnorm;
        if residual <= tol {
            return CgResult {
                x,
                iters: it + 1,
                residual,
            };
        }
        let beta = rr_new / rr;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    CgResult {
        x,
        iters: max_iters,
        residual,
    }
}

/// Preconditioned Conjugate Gradient — same as [`cg_solve`] except a
/// user-supplied preconditioner `precond(r) → M⁻¹·r` is applied each
/// iteration. `M` must be SPD for the convergence guarantee to hold;
/// the canonical default is the [`JacobiPreconditioner`].
///
/// Example with the built-in Jacobi preconditioner:
///
/// ```ignore
/// use molla::math::iterative::{pcg_solve, JacobiPreconditioner};
/// let m = JacobiPreconditioner::new(&diag);
/// let result = pcg_solve(|x| matvec(x), |r| m.apply(r), &b, &x0, 1e-8, 200);
/// ```
pub fn pcg_solve<F, P>(
    op: F,
    precond: P,
    b: &[Real],
    x0: &[Real],
    tol: Real,
    max_iters: usize,
) -> CgResult
where
    F: Fn(&[Real]) -> Vec<Real>,
    P: Fn(&[Real]) -> Vec<Real>,
{
    let n = b.len();
    assert_eq!(
        x0.len(),
        n,
        "pcg_solve: x0 length {} != b length {}",
        x0.len(),
        n
    );

    let bnorm = norm(b);
    if bnorm <= 1.0e-30 {
        return CgResult {
            x: vec![0.0; n],
            iters: 0,
            residual: 0.0,
        };
    }

    let mut x = x0.to_vec();
    let ax0 = op(&x);
    debug_assert_eq!(ax0.len(), n);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax0[i];
    }
    let mut residual = norm(&r) / bnorm;
    if residual <= tol {
        return CgResult {
            x,
            iters: 0,
            residual,
        };
    }

    let mut z = precond(&r);
    debug_assert_eq!(z.len(), n);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for it in 0..max_iters {
        let ap = op(&p);
        debug_assert_eq!(ap.len(), n);
        let pap = dot(&p, &ap);
        if pap.abs() < 1.0e-30 {
            residual = norm(&r) / bnorm;
            return CgResult {
                x,
                iters: it,
                residual,
            };
        }
        let alpha = rz / pap;
        axpy(&mut x, alpha, &p);
        axpy(&mut r, -alpha, &ap);
        residual = norm(&r) / bnorm;
        if residual <= tol {
            return CgResult {
                x,
                iters: it + 1,
                residual,
            };
        }
        z = precond(&r);
        debug_assert_eq!(z.len(), n);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    CgResult {
        x,
        iters: max_iters,
        residual,
    }
}

/// Biconjugate Gradient Stabilised (BiCGSTAB) for general
/// **non-symmetric** `A · x = b`. Same matrix-free interface as
/// [`cg_solve`].
///
/// **Stability caveat.** BiCGSTAB can stagnate when one of the
/// orthogonality scalars (`ρ`, `ω`) collapses near zero — this is
/// inherent to the algorithm, not a bug. We guard with two heuristics
/// drawn from Saad's "Iterative Methods for Sparse Linear Systems"
/// §7.4:
///
/// 1. **Restart on `|ρ|` or `|ω|` collapse.** When either scalar
///    falls below `1e-30`, we re-seed the shadow residual from the
///    current true residual and restart the recurrence (`r̂ = r`,
///    `ρ = α = ω = 1`, `p = v = 0`). This costs a fresh matvec but
///    almost always recovers convergence on systems where plain
///    BiCGSTAB would stall.
/// 2. **Bail on `tt → 0`.** When `‖t‖² < 1e-30` (i.e. the
///    preconditioned `s` lies in the kernel of `A`), we accept the
///    half-step `x ← x + α·p` and exit; this is the textbook safe-exit.
///
/// Even with these guards, BiCGSTAB remains less robust than GMRES for
/// pathological non-symmetric systems — fall back to
/// [`crate::sparse::gmres`] if you hit a system where BiCGSTAB
/// won't converge.
pub fn bicgstab_solve<F>(op: F, b: &[Real], x0: &[Real], tol: Real, max_iters: usize) -> CgResult
where
    F: Fn(&[Real]) -> Vec<Real>,
{
    let n = b.len();
    assert_eq!(
        x0.len(),
        n,
        "bicgstab_solve: x0 length {} != b length {}",
        x0.len(),
        n
    );

    let bnorm = norm(b);
    if bnorm <= 1.0e-30 {
        return CgResult {
            x: vec![0.0; n],
            iters: 0,
            residual: 0.0,
        };
    }

    let mut x = x0.to_vec();
    let ax0 = op(&x);
    debug_assert_eq!(ax0.len(), n);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax0[i];
    }
    let mut residual = norm(&r) / bnorm;
    if residual <= tol {
        return CgResult {
            x,
            iters: 0,
            residual,
        };
    }
    let mut r_hat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut rho: Real = 1.0;
    let mut alpha: Real = 1.0;
    let mut omega: Real = 1.0;

    let mut iters_done: usize = 0;
    for it in 0..max_iters {
        iters_done = it + 1;
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1.0e-30 {
            // ρ collapse — restart with fresh shadow residual.
            r_hat = r.clone();
            rho = 1.0;
            alpha = 1.0;
            omega = 1.0;
            for slot in p.iter_mut() {
                *slot = 0.0;
            }
            for slot in v.iter_mut() {
                *slot = 0.0;
            }
            continue;
        }
        if it == 0 {
            p.copy_from_slice(&r);
        } else {
            // β = (ρ_new / ρ) · (α / ω)
            let beta = (rho_new / rho) * (alpha / omega);
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }
        let v_new = op(&p);
        debug_assert_eq!(v_new.len(), n);
        v.copy_from_slice(&v_new);
        let rhat_v = dot(&r_hat, &v);
        if rhat_v.abs() < 1.0e-30 {
            // Search direction orthogonal to shadow — restart.
            r_hat = r.clone();
            rho = 1.0;
            alpha = 1.0;
            omega = 1.0;
            for slot in p.iter_mut() {
                *slot = 0.0;
            }
            for slot in v.iter_mut() {
                *slot = 0.0;
            }
            continue;
        }
        alpha = rho_new / rhat_v;
        let mut s = vec![0.0; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let s_norm = norm(&s);
        if s_norm / bnorm <= tol {
            // Accept half-step and exit.
            axpy(&mut x, alpha, &p);
            return CgResult {
                x,
                iters: iters_done,
                residual: s_norm / bnorm,
            };
        }
        let t = op(&s);
        debug_assert_eq!(t.len(), n);
        let tt = dot(&t, &t);
        if tt < 1.0e-30 {
            // t ≈ 0 — accept half-step and exit (textbook safe exit).
            axpy(&mut x, alpha, &p);
            residual = norm(&s) / bnorm;
            return CgResult {
                x,
                iters: iters_done,
                residual,
            };
        }
        omega = dot(&t, &s) / tt;
        if omega.abs() < 1.0e-30 {
            // ω collapse — accept half-step + restart on next iter.
            axpy(&mut x, alpha, &p);
            r[..n].copy_from_slice(&s[..n]);
            r_hat = r.clone();
            rho = 1.0;
            alpha = 1.0;
            omega = 1.0;
            for slot in p.iter_mut() {
                *slot = 0.0;
            }
            for slot in v.iter_mut() {
                *slot = 0.0;
            }
            residual = norm(&r) / bnorm;
            if residual <= tol {
                return CgResult {
                    x,
                    iters: iters_done,
                    residual,
                };
            }
            continue;
        }
        axpy(&mut x, alpha, &p);
        axpy(&mut x, omega, &s);
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        residual = norm(&r) / bnorm;
        if residual <= tol {
            return CgResult {
                x,
                iters: iters_done,
                residual,
            };
        }
        rho = rho_new;
    }
    CgResult {
        x,
        iters: iters_done,
        residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D Laplacian (tridiag `[-1, 2, -1]`) of size `n`. SPD.
    fn laplacian_1d_mv(x: &[Real]) -> Vec<Real> {
        let n = x.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = 2.0 * x[i];
            if i > 0 {
                s -= x[i - 1];
            }
            if i + 1 < n {
                s -= x[i + 1];
            }
            y[i] = s;
        }
        y
    }

    fn laplacian_1d_diag(n: usize) -> Vec<Real> {
        vec![2.0; n]
    }

    #[test]
    fn cg_zero_rhs_gives_zero_solution() {
        let n = 16;
        let b = vec![0.0; n];
        let x0 = vec![0.0; n];
        let res = cg_solve(laplacian_1d_mv, &b, &x0, 1.0e-10, 100);
        assert_eq!(res.x, vec![0.0; n]);
        assert_eq!(res.iters, 0);
        assert!(res.residual <= 1.0e-10);
    }

    #[test]
    fn bicgstab_zero_rhs_gives_zero_solution() {
        let n = 16;
        let b = vec![0.0; n];
        let x0 = vec![0.0; n];
        let res = bicgstab_solve(laplacian_1d_mv, &b, &x0, 1.0e-10, 100);
        assert_eq!(res.x, vec![0.0; n]);
        assert_eq!(res.iters, 0);
    }

    #[test]
    fn pcg_zero_rhs_gives_zero_solution() {
        let n = 16;
        let b = vec![0.0; n];
        let x0 = vec![0.0; n];
        let m = JacobiPreconditioner::new(&laplacian_1d_diag(n));
        let res = pcg_solve(laplacian_1d_mv, |r| m.apply(r), &b, &x0, 1.0e-10, 100);
        assert_eq!(res.x, vec![0.0; n]);
        assert_eq!(res.iters, 0);
    }

    #[test]
    fn cg_identity_returns_rhs() {
        // A = I → x = b in one iteration.
        let n = 8;
        let b: Vec<Real> = (0..n).map(|i| (i + 1) as Real).collect();
        let x0 = vec![0.0; n];
        let res = cg_solve(|x| x.to_vec(), &b, &x0, 1.0e-12, 10);
        for i in 0..n {
            assert!(
                (res.x[i] - b[i]).abs() < 1.0e-10,
                "i={i}: {} vs {}",
                res.x[i],
                b[i]
            );
        }
        assert!(
            res.iters <= 2,
            "CG on I should take ≤ 2 iters, took {}",
            res.iters
        );
    }

    #[test]
    fn pcg_identity_returns_rhs() {
        let n = 8;
        let b: Vec<Real> = (0..n).map(|i| (i + 1) as Real).collect();
        let x0 = vec![0.0; n];
        let m = JacobiPreconditioner::new(&vec![1.0; n]);
        let res = pcg_solve(|x| x.to_vec(), |r| m.apply(r), &b, &x0, 1.0e-12, 10);
        for i in 0..n {
            assert!((res.x[i] - b[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn bicgstab_identity_returns_rhs() {
        let n = 8;
        let b: Vec<Real> = (0..n).map(|i| (i + 1) as Real).collect();
        let x0 = vec![0.0; n];
        let res = bicgstab_solve(|x| x.to_vec(), &b, &x0, 1.0e-12, 10);
        for i in 0..n {
            assert!((res.x[i] - b[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cg_solves_small_laplacian() {
        let n = 8;
        let b: Vec<Real> = (0..n).map(|i| (i as Real - 3.5).sin()).collect();
        let x0 = vec![0.0; n];
        let res = cg_solve(laplacian_1d_mv, &b, &x0, 1.0e-12, 100);
        // Verify A·x ≈ b directly.
        let ax = laplacian_1d_mv(&res.x);
        for i in 0..n {
            assert!((ax[i] - b[i]).abs() < 1.0e-9);
        }
    }

    #[test]
    fn jacobi_preconditioner_zero_diag_passes_through() {
        let m = JacobiPreconditioner::new(&[1.0, 0.0, 4.0]);
        let r = vec![5.0, 7.0, 8.0];
        let z = m.apply(&r);
        assert!((z[0] - 5.0).abs() < 1.0e-15);
        assert!(z[1].abs() < 1.0e-15);
        assert!((z[2] - 2.0).abs() < 1.0e-15);
    }
}
