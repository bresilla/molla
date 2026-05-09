//! Dense linear-algebra primitives — Cholesky factorisation + forward
//! and back substitution.
//!
//! Used for solving `M·x = b` when `M` is small dense SPD: CRBA mass
//! matrices on small articulations, IK normal equations
//! `(J^T·J + λ·I)·dq = J^T·e`, etc. Newton/Warp uses an equivalent
//! `dense_chol` / `dense_subs` pair via cuBLAS for the same purpose.
//!
//! Convention: row-major flat storage. `n` is the matrix dimension; an
//! `n × n` matrix lives in `Vec<Real>` of length `n * n`.

use crate::core::{Error, Result};
use crate::math::Real;

/// In-place Cholesky factorisation `A = L · Lᵀ` where `L` is lower
/// triangular. On success the lower triangle of `a` (including the
/// diagonal) holds `L`; the upper triangle is left untouched. Errors
/// when `a` isn't SPD (a non-positive pivot turns up).
pub fn dense_chol(a: &mut [Real], n: usize) -> Result<()> {
    if a.len() != n * n {
        return Err(Error::Build(format!(
            "dense_chol: a.len()={} but n²={}",
            a.len(),
            n * n
        )));
    }
    for j in 0..n {
        // Diagonal: L[j,j]² = A[j,j] - Σ L[j,k]² for k < j.
        let mut diag = a[j * n + j];
        for k in 0..j {
            let l_jk = a[j * n + k];
            diag -= l_jk * l_jk;
        }
        if diag <= 0.0 {
            return Err(Error::Build(format!(
                "dense_chol: matrix not SPD (pivot {diag} ≤ 0 at row {j})"
            )));
        }
        let l_jj = diag.sqrt();
        a[j * n + j] = l_jj;

        // Below-diagonal column: L[i,j] = (A[i,j] - Σ L[i,k]·L[j,k]) / L[j,j].
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / l_jj;
        }
    }
    Ok(())
}

/// Solve `L · y = b` (forward substitution) with `L` lower triangular.
/// Reads only the lower triangle (including diagonal) of `l`.
pub fn forward_sub(l: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>> {
    if l.len() != n * n || b.len() != n {
        return Err(Error::Build(format!(
            "forward_sub: l.len()={}, b.len()={}, expected n²={}, n={}",
            l.len(),
            b.len(),
            n * n,
            n
        )));
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-30 {
            return Err(Error::Build(format!(
                "forward_sub: zero diagonal at row {i}"
            )));
        }
        y[i] = s / l_ii;
    }
    Ok(y)
}

/// Solve `Lᵀ · x = y` (back substitution) with `L` lower triangular.
/// Reads only the lower triangle of `l` (treats it as `L`, computes
/// using `Lᵀ`).
pub fn back_sub(l: &[Real], y: &[Real], n: usize) -> Result<Vec<Real>> {
    if l.len() != n * n || y.len() != n {
        return Err(Error::Build(format!(
            "back_sub: l.len()={}, y.len()={}, expected n²={}, n={}",
            l.len(),
            y.len(),
            n * n,
            n
        )));
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            // Lᵀ[i,k] = L[k,i].
            s -= l[k * n + i] * x[k];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-30 {
            return Err(Error::Build(format!(
                "back_sub: zero diagonal at row {i}"
            )));
        }
        x[i] = s / l_ii;
    }
    Ok(x)
}

/// Solve `A · x = b` for SPD `A` via Cholesky: factors `A` in-place
/// (overwriting its lower triangle with `L`) and runs forward + back
/// substitution.
pub fn dense_chol_solve(a: &mut [Real], b: &[Real], n: usize) -> Result<Vec<Real>> {
    dense_chol(a, n)?;
    let y = forward_sub(a, b, n)?;
    back_sub(a, &y, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_mv(a: &[Real], x: &[Real], n: usize) -> Vec<Real> {
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i * n + j] * x[j];
            }
        }
        y
    }

    #[test]
    fn cholesky_solves_spd_2x2() {
        // A = [[4, 2], [2, 5]] is SPD. b = [6, 7]. Solution x = [1, 1].
        let mut a = vec![4.0, 2.0, 2.0, 5.0];
        let b = vec![6.0, 7.0];
        let x = dense_chol_solve(&mut a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solves_spd_4x4() {
        // Diagonally-dominant SPD: A = D + 0.1 · randn-style perturbation.
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as Real * 2.0;
        }
        a[0 * n + 1] = 0.5;
        a[1 * n + 0] = 0.5;
        a[1 * n + 2] = 0.3;
        a[2 * n + 1] = 0.3;
        a[2 * n + 3] = 0.7;
        a[3 * n + 2] = 0.7;
        let a_orig = a.clone();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = dense_chol_solve(&mut a, &b, n).unwrap();
        let bx = dense_mv(&a_orig, &x, n);
        for i in 0..n {
            assert!((bx[i] - b[i]).abs() < 1e-10, "row {i}: {} vs {}", bx[i], b[i]);
        }
    }

    #[test]
    fn cholesky_rejects_non_spd() {
        // Indefinite: eigenvalues 1 and -1.
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        assert!(dense_chol(&mut a, 2).is_err());
    }
}
