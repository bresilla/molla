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

use crate::Real;
use molla_core::{Error, Result};

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
        let (leading, trailing) = a.split_at_mut((j + 1) * n);
        let pivot_row = &mut leading[j * n..];
        // Diagonal: L[j,j]² = A[j,j] - Σ L[j,k]² for k < j.
        let mut diag = pivot_row[j];
        for &l_jk in &pivot_row[..j] {
            diag -= l_jk * l_jk;
        }
        if diag <= 0.0 {
            return Err(Error::Build(format!(
                "dense_chol: matrix not SPD (pivot {diag} ≤ 0 at row {j})"
            )));
        }
        let l_jj = diag.sqrt();
        pivot_row[j] = l_jj;

        // Below-diagonal column: L[i,j] = (A[i,j] - Σ L[i,k]·L[j,k]) / L[j,j].
        for row in trailing.chunks_exact_mut(n) {
            let mut s = row[j];
            for (&l_ik, &l_jk) in row[..j].iter().zip(&pivot_row[..j]) {
                s -= l_ik * l_jk;
            }
            row[j] = s / l_jj;
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
            return Err(Error::Build(format!("back_sub: zero diagonal at row {i}")));
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

/// Cyclic-Jacobi symmetric eigendecomposition for an `n × n` symmetric
/// matrix. Returns `(λ, V)` such that `A = V · diag(λ) · Vᵀ` with `V`
/// orthogonal. Reads only the upper triangle of `a` (the lower is
/// assumed to mirror it, the caller is responsible for symmetrising).
///
/// This is the n-dimensional generalisation of
/// [`crate::dense3::eig3_sym`]. We need it because the IK
/// solver's `JᵀJ + λI` matrix is `dof × dof`, with `dof` up to ~30
/// for full humanoids — beyond what the 3×3 primitive can handle.
///
/// Algorithm: standard cyclic Jacobi (Press et al., NR §11.1 — same
/// rotations as the 3×3 case, just applied to all `n*(n-1)/2` off-
/// diagonal pairs per sweep).
pub fn dense_eig_sym(a: &[Real], n: usize) -> Result<(Vec<Real>, Vec<Real>)> {
    if a.len() != n * n {
        return Err(Error::Build(format!(
            "dense_eig_sym: a.len()={} but n²={}",
            a.len(),
            n * n
        )));
    }
    // Working copy (symmetrised: enforce symmetry from the upper tri).
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let jdx = j * n + i;
            let v = 0.5 * (a[idx] + a[jdx]);
            m[idx] = v;
        }
    }
    // V starts as identity.
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 50; // ample for n ≤ 30; cyclic Jacobi is O(n²) per sweep.
    let eps: Real = 1.0e-14;
    for _ in 0..max_sweeps {
        let mut off2: Real = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off2 += m[p * n + q] * m[p * n + q];
            }
        }
        if off2 < eps * eps {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < eps {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = m[r * n + p];
                    let arq = m[r * n + q];
                    m[r * n + p] = c * arp - s * arq;
                    m[p * n + r] = m[r * n + p];
                    m[r * n + q] = s * arp + c * arq;
                    m[q * n + r] = m[r * n + q];
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut lambda = vec![0.0; n];
    for i in 0..n {
        lambda[i] = m[i * n + i];
    }
    Ok((lambda, v))
}

/// Damped-pseudoinverse solve for `A · x = b` where `A = JᵀJ + λI` is
/// SPD by construction but may be **ill-conditioned** (a near-singular
/// `J`). Builds the eigendecomposition of `A`, then forms
///   `x = Σ_i (vᵢ · vᵢᵀ · b) / max(λᵢ, λ_floor)`
/// where `λ_floor = max(eigval) · 1e-12` clamps tiny eigenvalues to a
/// numerical floor. Equivalent to the SVD-based pseudoinverse for SPD
/// matrices since `A = V · diag(λ) · Vᵀ` is its own SVD.
///
/// Used as the IK fallback when `dense_chol` reports a non-SPD pivot
/// (numerical Cholesky breakdown on near-singular `JᵀJ + λI`).
pub fn dense_psd_pinv_solve(a: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>> {
    if a.len() != n * n || b.len() != n {
        return Err(Error::Build(format!(
            "dense_psd_pinv_solve: a.len()={}, b.len()={}, expected n²={}, n={}",
            a.len(),
            b.len(),
            n * n,
            n
        )));
    }
    let (lambda, v) = dense_eig_sym(a, n)?;
    // Compute `Vᵀ · b` (length n).
    let mut vt_b = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for k in 0..n {
            s += v[k * n + i] * b[k];
        }
        vt_b[i] = s;
    }
    // Clamp eigenvalues at a floor relative to the largest.
    let mut max_l: Real = 0.0;
    for &l in &lambda {
        if l.abs() > max_l {
            max_l = l.abs();
        }
    }
    let floor = (max_l * 1.0e-12).max(1.0e-30);
    for i in 0..n {
        let li = lambda[i].abs().max(floor);
        vt_b[i] /= li;
    }
    // x = V · (filtered Vᵀ b).
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for k in 0..n {
            s += v[i * n + k] * vt_b[k];
        }
        x[i] = s;
    }
    Ok(x)
}

#[cfg(test)]
// Test matrices are written in the explicit `row * n + col` row-major
// idiom even when row or col is 0, so every entry reads as a coordinate.
#[allow(clippy::erasing_op, clippy::identity_op)]
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
            assert!(
                (bx[i] - b[i]).abs() < 1e-10,
                "row {i}: {} vs {}",
                bx[i],
                b[i]
            );
        }
    }

    #[test]
    fn cholesky_rejects_non_spd() {
        // Indefinite: eigenvalues 1 and -1.
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        assert!(dense_chol(&mut a, 2).is_err());
    }

    #[test]
    fn psd_pinv_matches_chol_on_well_conditioned() {
        // Diagonally-dominant 4×4 SPD — both solvers should agree.
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
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut a_chol = a.clone();
        let x_chol = dense_chol_solve(&mut a_chol, &b, n).unwrap();
        let x_pinv = dense_psd_pinv_solve(&a, &b, n).unwrap();
        for i in 0..n {
            assert!(
                (x_chol[i] - x_pinv[i]).abs() < 1.0e-8,
                "row {i}: chol {} vs pinv {}",
                x_chol[i],
                x_pinv[i],
            );
        }
    }

    #[test]
    fn psd_pinv_handles_near_singular() {
        // Rank-1 + tiny diagonal: cholesky breakdown, pinv tolerates.
        let n = 3;
        let u = [1.0, 2.0, -1.0];
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = u[i] * u[j];
            }
            a[i * n + i] += 1.0e-10;
        }
        let b = vec![1.0, 0.0, 0.0];
        // Pinv should not panic and should produce something finite.
        let x = dense_psd_pinv_solve(&a, &b, n).unwrap();
        for v in &x {
            assert!(v.is_finite(), "pinv produced non-finite: {x:?}");
        }
    }

    #[test]
    fn eig_sym_diag_recovers_eigenvalues() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        a[0 * n + 0] = 2.0;
        a[1 * n + 1] = 5.0;
        a[2 * n + 2] = 3.0;
        a[3 * n + 3] = 7.0;
        let (lam, _) = dense_eig_sym(&a, n).unwrap();
        let mut sorted = lam.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = [2.0, 3.0, 5.0, 7.0];
        for i in 0..n {
            assert!((sorted[i] - expected[i]).abs() < 1.0e-10);
        }
    }
}
