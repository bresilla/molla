//! Integration tests for `crate::math::iterative` — Stage 27.
//!
//! All CPU-only, no `#[ignore]` (the iterative solvers don't touch
//! the GPU; they're matrix-free Krylov methods). Mirrors the test
//! plan in the Stage-27 brief:
//!
//! - CG converges on a 100×100 SPD synthetic Laplacian.
//! - BiCGSTAB converges on a 100×100 non-symmetric system.
//! - PCG with Jacobi beats plain CG on an ill-conditioned (cond ~ 1e4)
//!   SPD matrix.
//! - Edge cases: zero RHS → zero solution; identity → RHS unchanged.

use molla_math::Real;
use molla_math::iterative::{JacobiPreconditioner, bicgstab_solve, cg_solve, pcg_solve};

/// Dense `n × n` row-major matrix-vector product. Sufficient for the
/// 100×100 test sizes in this file (the iterative solvers themselves
/// are matrix-free; the *tests* materialise A purely so they can check
/// the answer with a regular matvec).
fn dense_mv(a: &[Real], x: &[Real], n: usize) -> Vec<Real> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// 1D Laplacian `[-1, 2, -1]` of size `n`. SPD; condition number
/// scales like `O(n²)` (mild — well-conditioned for n=100).
fn laplacian_1d(n: usize) -> Vec<Real> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 2.0;
        if i > 0 {
            a[i * n + (i - 1)] = -1.0;
        }
        if i + 1 < n {
            a[i * n + (i + 1)] = -1.0;
        }
    }
    a
}

/// Diagonal vector of `a`.
fn diag_of(a: &[Real], n: usize) -> Vec<Real> {
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Non-symmetric `n × n` matrix: tridiag `[-2, 5, -1]`. Diagonally
/// dominant ⇒ BiCGSTAB converges robustly without preconditioning.
fn nonsym_tridiag(n: usize) -> Vec<Real> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 5.0;
        if i > 0 {
            a[i * n + (i - 1)] = -2.0;
        }
        if i + 1 < n {
            a[i * n + (i + 1)] = -1.0;
        }
    }
    a
}

/// SPD with controlled condition number. Diagonal eigenvalues geometric
/// from `1` to `kappa`; off-diagonal mass `0.05` couples it slightly so
/// Jacobi has actual work to do (pure diagonal matrices are 1-iter
/// trivial for both CG and PCG).
fn ill_conditioned_spd(n: usize, kappa: Real) -> Vec<Real> {
    let mut a = vec![0.0; n * n];
    let r = kappa.powf(1.0 / (n - 1) as Real);
    for i in 0..n {
        let lambda = r.powi(i as i32);
        a[i * n + i] = lambda;
    }
    // Symmetric off-diagonal coupling.
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.05 * ((i + j) as Real / n as Real).sin();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

// ─── DoD: CG on 100×100 SPD ──────────────────────────────────────────

#[test]
fn cg_converges_on_100x100_laplacian() {
    let n = 100;
    let a = laplacian_1d(n);
    let b: Vec<Real> = (0..n).map(|i| ((i + 1) as Real).sin()).collect();
    let x0 = vec![0.0; n];

    let res = cg_solve(|x| dense_mv(&a, x, n), &b, &x0, 1.0e-8, 200);

    eprintln!(
        "cg_converges_on_100x100_laplacian: {} iters, residual {:.3e}",
        res.iters, res.residual
    );
    assert!(
        res.converged(1.0e-8),
        "CG failed to reach 1e-8: {} iters, residual {:.3e}",
        res.iters,
        res.residual
    );
    assert!(res.iters <= 200, "CG took {} iters (cap 200)", res.iters);

    // Sanity: actually solves the system.
    let ax = dense_mv(&a, &res.x, n);
    let mut max_err: Real = 0.0;
    for i in 0..n {
        let e = (ax[i] - b[i]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1.0e-6,
        "CG residual leaked: max_err={max_err:.3e}"
    );
}

// ─── DoD: BiCGSTAB on 100×100 non-symmetric ──────────────────────────

#[test]
fn bicgstab_converges_on_100x100_nonsymmetric() {
    let n = 100;
    let a = nonsym_tridiag(n);
    let b: Vec<Real> = (0..n).map(|i| 1.0 + 0.1 * (i as Real)).collect();
    let x0 = vec![0.0; n];

    let res = bicgstab_solve(|x| dense_mv(&a, x, n), &b, &x0, 1.0e-6, 300);

    eprintln!(
        "bicgstab_converges_on_100x100_nonsymmetric: {} iters, residual {:.3e}",
        res.iters, res.residual
    );
    assert!(
        res.converged(1.0e-6),
        "BiCGSTAB failed to reach 1e-6: {} iters, residual {:.3e}",
        res.iters,
        res.residual
    );
    assert!(
        res.iters <= 300,
        "BiCGSTAB took {} iters (cap 300)",
        res.iters
    );

    let ax = dense_mv(&a, &res.x, n);
    let mut max_err: Real = 0.0;
    for i in 0..n {
        let e = (ax[i] - b[i]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1.0e-4,
        "BiCGSTAB residual leaked: max_err={max_err:.3e}"
    );
}

// ─── DoD: PCG with Jacobi beats plain CG on ill-conditioned SPD ──────

#[test]
fn pcg_jacobi_beats_cg_on_ill_conditioned_spd() {
    let n = 100;
    let kappa = 1.0e4;
    let a = ill_conditioned_spd(n, kappa);
    let b: Vec<Real> = (0..n).map(|i| ((i + 1) as Real).cos()).collect();
    let x0 = vec![0.0; n];
    let tol = 1.0e-6;
    let max_iters = 5000;

    let cg = cg_solve(|x| dense_mv(&a, x, n), &b, &x0, tol, max_iters);
    let m = JacobiPreconditioner::new(&diag_of(&a, n));
    let pcg = pcg_solve(
        |x| dense_mv(&a, x, n),
        |r| m.apply(r),
        &b,
        &x0,
        tol,
        max_iters,
    );

    eprintln!(
        "pcg_jacobi_beats_cg_on_ill_conditioned_spd: cg {} iters (res {:.3e}), pcg {} iters (res {:.3e})",
        cg.iters, cg.residual, pcg.iters, pcg.residual
    );

    assert!(
        cg.converged(tol),
        "plain CG didn't converge for the comparison ({} iters, res {:.3e})",
        cg.iters,
        cg.residual
    );
    assert!(
        pcg.converged(tol),
        "PCG didn't converge ({} iters, res {:.3e})",
        pcg.iters,
        pcg.residual
    );
    assert!(
        pcg.iters < cg.iters,
        "PCG ({} iters) should beat plain CG ({} iters) on ill-conditioned SPD",
        pcg.iters,
        cg.iters,
    );
}

// ─── Edge cases: zero RHS, identity ─────────────────────────────────

#[test]
fn cg_zero_rhs_zero_solution() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = vec![0.0; n];
    let x0 = vec![0.0; n];
    let res = cg_solve(|x| dense_mv(&a, x, n), &b, &x0, 1.0e-10, 100);
    for v in &res.x {
        assert!(v.abs() < 1.0e-15);
    }
    assert_eq!(res.iters, 0);
    assert_eq!(res.residual, 0.0);
}

#[test]
fn bicgstab_zero_rhs_zero_solution() {
    let n = 50;
    let a = nonsym_tridiag(n);
    let b = vec![0.0; n];
    let x0 = vec![0.0; n];
    let res = bicgstab_solve(|x| dense_mv(&a, x, n), &b, &x0, 1.0e-10, 100);
    for v in &res.x {
        assert!(v.abs() < 1.0e-15);
    }
    assert_eq!(res.iters, 0);
    assert_eq!(res.residual, 0.0);
}

#[test]
fn pcg_zero_rhs_zero_solution() {
    let n = 50;
    let a = laplacian_1d(n);
    let m = JacobiPreconditioner::new(&diag_of(&a, n));
    let b = vec![0.0; n];
    let x0 = vec![0.0; n];
    let res = pcg_solve(
        |x| dense_mv(&a, x, n),
        |r| m.apply(r),
        &b,
        &x0,
        1.0e-10,
        100,
    );
    for v in &res.x {
        assert!(v.abs() < 1.0e-15);
    }
    assert_eq!(res.iters, 0);
    assert_eq!(res.residual, 0.0);
}

#[test]
fn cg_identity_returns_rhs_unchanged() {
    let n = 32;
    let b: Vec<Real> = (0..n).map(|i| (i as Real + 1.0) * 0.5).collect();
    let x0 = vec![0.0; n];
    let res = cg_solve(|x| x.to_vec(), &b, &x0, 1.0e-12, 10);
    for i in 0..n {
        assert!(
            (res.x[i] - b[i]).abs() < 1.0e-10,
            "identity·CG failed at i={i}"
        );
    }
}

#[test]
fn bicgstab_identity_returns_rhs_unchanged() {
    let n = 32;
    let b: Vec<Real> = (0..n).map(|i| (i as Real + 1.0) * 0.5).collect();
    let x0 = vec![0.0; n];
    let res = bicgstab_solve(|x| x.to_vec(), &b, &x0, 1.0e-12, 10);
    for i in 0..n {
        assert!(
            (res.x[i] - b[i]).abs() < 1.0e-10,
            "identity·BiCGSTAB failed at i={i}"
        );
    }
}

#[test]
fn pcg_identity_returns_rhs_unchanged() {
    let n = 32;
    let b: Vec<Real> = (0..n).map(|i| (i as Real + 1.0) * 0.5).collect();
    let x0 = vec![0.0; n];
    let m = JacobiPreconditioner::new(&vec![1.0; n]);
    let res = pcg_solve(|x| x.to_vec(), |r| m.apply(r), &b, &x0, 1.0e-12, 10);
    for i in 0..n {
        assert!(
            (res.x[i] - b[i]).abs() < 1.0e-10,
            "identity·PCG failed at i={i}"
        );
    }
}
