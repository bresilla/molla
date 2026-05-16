//! Small-matrix decompositions for **3×3** dense matrices — SVD, QR,
//! symmetric eigendecomposition, and the polar decomposition wrapper.
//!
//! These primitives are the small-matrix counterpart of
//! [`super::dense_solve`]. They show up in:
//!
//! * **IK regularisation** (Stage 16) — when `(JᵀJ + λI)` Cholesky
//!   fails or condition is huge, we fall back to an SVD-based damped
//!   pseudoinverse.
//! * **Per-vertex polar decomposition** for VBD soft-body co-rotational
//!   energy (Stage 20 / 20.1).
//! * **Per-particle inertia eigendecomposition** for MPM plasticity
//!   (Stage 24d).
//!
//! Algorithms are chosen for *WGSL portability* — Jacobi-style sweeps
//! and Givens rotations only, no Householder reflections (the latter
//! need conditional compaction that's painful in WGSL). The matching
//! WGSL kernel lives in `shaders/dense3_svd.wgsl` and is dispatched by
//! [`crate::kernels::Dense3Kernel`].
//!
//! Convention: matrices are [`crate::Mat3`] (= `glam::DMat3`),
//! which stores columns left-to-right. We treat `m.col(i)` as the
//! `i`-th column and `m.row(i)` as the `i`-th row, matching standard
//! linear-algebra conventions.

use crate::{Mat3, Real, Vec3};

/// Off-diagonal Frobenius-norm threshold below which a Jacobi sweep is
/// considered converged.
const JACOBI_EPS: Real = 1.0e-14;

/// Hard cap on Jacobi sweeps (each sweep visits 3 off-diagonal pairs).
/// Most well-conditioned 3×3 matrices converge in 4-6 sweeps; near-
/// singular matrices may need more. The cap protects against
/// pathological inputs (e.g. matrices with duplicate eigenvalues that
/// cause Jacobi to spin on rotation choice).
const JACOBI_MAX_SWEEPS: usize = 32;

/// 3×3 SVD via cyclic one-sided Jacobi rotations.
///
/// Returns `(U, σ, V)` such that `M = U · diag(σ) · Vᵀ`. Singular
/// values are **non-negative** but not necessarily sorted — callers
/// that want descending order should sort externally.
///
/// Algorithm: one-sided Jacobi on `M`. We rotate column pairs of `M`
/// by Givens rotations chosen to zero the off-diagonal entries of
/// `MᵀM`. After convergence the columns of `M` are orthogonal; their
/// norms are the singular values, normalised columns form `U`, and
/// the accumulated rotations form `V`.
///
/// Reference: Demmel & Veselić "Jacobi's Method is More Accurate than
/// QR" (1992), and the textbook one-sided variant in Golub & Van Loan
/// (3rd ed., §8.6.3).
pub fn svd3(m: Mat3) -> (Mat3, Vec3, Mat3) {
    // Work on copies — `b` accumulates `M · V`; `v` accumulates `V`.
    let mut b = m;
    let mut v = Mat3::IDENTITY;

    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0;
        // Cyclic order: (0,1), (0,2), (1,2).
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            // a_pp = ‖b_p‖², a_qq = ‖b_q‖², a_pq = b_p · b_q.
            let bp = b.col(p);
            let bq = b.col(q);
            let app = bp.dot(bp);
            let aqq = bq.dot(bq);
            let apq = bp.dot(bq);
            off += apq * apq;
            if apq.abs() < JACOBI_EPS * (app.sqrt() * aqq.sqrt() + JACOBI_EPS) {
                continue;
            }
            // Rotate to zero apq via Jacobi angle θ:
            //   tan(2θ) = 2·a_pq / (a_pp - a_qq)
            // numerically-stable variant from Golub-Van Loan.
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                1.0 / (tau - (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            // Apply rotation to columns p, q of b and v.
            let bp_new = c * bp - s * bq;
            let bq_new = s * bp + c * bq;
            set_col(&mut b, p, bp_new);
            set_col(&mut b, q, bq_new);
            let vp = v.col(p);
            let vq = v.col(q);
            set_col(&mut v, p, c * vp - s * vq);
            set_col(&mut v, q, s * vp + c * vq);
        }
        if off < JACOBI_EPS * JACOBI_EPS {
            break;
        }
    }

    // After convergence: columns of `b` are orthogonal.
    // Singular values = ‖b_i‖, U columns = b_i / σ_i.
    let mut sigma = Vec3::ZERO;
    let mut u = Mat3::IDENTITY;
    for i in 0..3 {
        let bi = b.col(i);
        let n = bi.length();
        sigma[i] = n;
        if n > 0.0 {
            set_col(&mut u, i, bi / n);
        } else {
            // Zero column — leave U's column as identity-i so the
            // factorisation stays well-defined; reconstruction still
            // works because σ_i = 0 multiplies it out.
            set_col(&mut u, i, basis_vec(i));
        }
    }

    // Force U to be a proper rotation if it ended up reflective, by
    // flipping the smallest-σ column's sign (with the matching V col).
    if u.determinant() < 0.0 {
        // Find min sigma index to absorb the sign flip with least
        // numerical impact (a flip on a near-zero σ is essentially
        // free; flipping a large one rotates the basis).
        let mut k = 0usize;
        for i in 1..3 {
            if sigma[i] < sigma[k] {
                k = i;
            }
        }
        let uk = -u.col(k);
        set_col(&mut u, k, uk);
        // Sign flip on σ[k] would make it negative; instead flip V's
        // column so M = U · diag(σ) · Vᵀ still holds.
        let vk = -v.col(k);
        set_col(&mut v, k, vk);
    }

    (u, sigma, v)
}

/// 3×3 QR via Givens rotations.
///
/// Returns `(Q, R)` such that `M = Q · R` with `Q` orthogonal (proper
/// or improper rotation) and `R` upper-triangular.
///
/// Algorithm: apply 3 Givens rotations to zero the sub-diagonal
/// entries of `M` in the order (1,0), (2,0), (2,1). Each Givens
/// rotation `G(i,j,θ)` is its own transpose, so `R = G_3 · G_2 · G_1 · M`
/// and `Q = G_1ᵀ · G_2ᵀ · G_3ᵀ`.
pub fn qr3(m: Mat3) -> (Mat3, Mat3) {
    let mut r = m;
    let mut q = Mat3::IDENTITY;

    // Zero the (1,0), (2,0), then (2,1) entries.
    for &(i, j) in &[(1usize, 0usize), (2, 0), (2, 1)] {
        let a = mget(&r, j, j);
        let b = mget(&r, i, j);
        if b.abs() < 1.0e-30 {
            continue;
        }
        let h = (a * a + b * b).sqrt();
        let c = a / h;
        let s = b / h;

        // R ← G · R  — left-multiply rows j and i of R.
        for k in 0..3 {
            let rjk = mget(&r, j, k);
            let rik = mget(&r, i, k);
            mset(&mut r, j, k, c * rjk + s * rik);
            mset(&mut r, i, k, -s * rjk + c * rik);
        }
        // Q ← Q · Gᵀ — right-multiply columns j and i of Q.
        for k in 0..3 {
            let qkj = mget(&q, k, j);
            let qki = mget(&q, k, i);
            mset(&mut q, k, j, c * qkj + s * qki);
            mset(&mut q, k, i, -s * qkj + c * qki);
        }
    }

    // Tidy up sub-diagonal numerical dust so callers can trust
    // `R[i,j]` is exactly 0 for i > j.
    for i in 0..3 {
        for j in 0..i {
            mset(&mut r, i, j, 0.0);
        }
    }
    (q, r)
}

/// 3×3 symmetric eigendecomposition via cyclic Jacobi sweeps.
///
/// Input must be symmetric (we read the upper triangle and assume the
/// lower mirrors it; an asymmetric input gives garbage). Returns
/// `(λ, V)` where `λ` are the eigenvalues and the columns of `V` are
/// the matching unit eigenvectors. `V` is orthogonal.
///
/// Eigenvalues are *not* sorted.
///
/// Reference: Press et al. "Numerical Recipes" §11.1, Jacobi rotations.
pub fn eig3_sym(m: Mat3) -> (Vec3, Mat3) {
    // Work on a 3×3 array copy of the upper triangle (symmetrised).
    let mut a = [
        [
            m.col(0).x,
            0.5 * (m.col(0).y + m.col(1).x),
            0.5 * (m.col(0).z + m.col(2).x),
        ],
        [0.0, m.col(1).y, 0.5 * (m.col(1).z + m.col(2).y)],
        [0.0, 0.0, m.col(2).z],
    ];
    a[1][0] = a[0][1];
    a[2][0] = a[0][2];
    a[2][1] = a[1][2];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..JACOBI_MAX_SWEEPS {
        let off2 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off2 < JACOBI_EPS * JACOBI_EPS {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < JACOBI_EPS {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let theta = (aqq - app) / (2.0 * apq);
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                1.0 / (theta - (1.0 + theta * theta).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            // Update a (rotation J(p,q,θ) applied as Jᵀ A J).
            let app_new = app - t * apq;
            let aqq_new = aqq + t * apq;
            a[p][p] = app_new;
            a[q][q] = aqq_new;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            for r in 0..3 {
                if r == p || r == q {
                    continue;
                }
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
            // Accumulate rotation into V (V ← V · J).
            for r in 0..3 {
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = c * vrp - s * vrq;
                v[r][q] = s * vrp + c * vrq;
            }
        }
    }

    let lambda = Vec3::new(a[0][0], a[1][1], a[2][2]);
    let v_mat = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );
    (lambda, v_mat)
}

/// Polar decomposition `M = R · S` with `R` orthogonal and `S`
/// symmetric positive semi-definite.
///
/// Implemented as a thin wrapper over [`svd3`]:
/// `R = U · Vᵀ`, `S = V · diag(σ) · Vᵀ`.
///
/// If `det(M) < 0`, `R` ends up improper (`det(R) = -1`). For VBD-style
/// co-rotational energy you usually want a proper rotation; flip the
/// sign of the smallest-singular-value column of `U` and the matching
/// `σ` entry to recover one (callers that need this can post-process).
pub fn polar_decompose_3(m: Mat3) -> (Mat3, Mat3) {
    let (u, sigma, v) = svd3(m);
    let r = u * v.transpose();
    let s = v * Mat3::from_diagonal(sigma) * v.transpose();
    (r, s)
}

/// Regularisation epsilon for the SVD adjoint's `F_ij = 1/(σⱼ² − σᵢ²)`
/// divisor. When two singular values are within this distance (in σ²
/// units) the divisor blows up; we Tikhonov-regularise as
/// `F_ij = (σⱼ² − σᵢ²) / max((σⱼ² − σᵢ²)², ε²)` which stays bounded by
/// `1/ε` and is exactly `1/(σⱼ² − σᵢ²)` away from the degeneracy.
const ADJ_SVD_EPS: Real = 1.0e-12;

/// Reverse-mode adjoint of [`svd3`].
///
/// Given the forward decomposition `A = U · diag(σ) · Vᵀ` and the
/// upstream gradients `adj_U`, `adj_σ`, `adj_V` of some downstream
/// scalar loss `J`, returns `adj_A = ∂J/∂A`.
///
/// Reference: McAdams et al. "Computing the Singular Value Decomposition
/// of 3×3 Matrices" (2011) for the forward; Townsend "Differentiating
/// the Singular Value Decomposition" (2016) and Klingner & Shewchuk's
/// note for the adjoint. The literal arithmetic mirrors the structure
/// of `xtra/warp/warp/native/svd.h::adj_svd3` (Apache-2.0).
///
/// **Degenerate σ handling.** When two singular values are equal, the
/// off-diagonal entries of `F` (with `F_ij = 1/(σⱼ² − σᵢ²)`) become
/// undefined. We use a Tikhonov regularisation
/// `F_ij = (σⱼ² − σᵢ²) / max((σⱼ² − σᵢ²)², ε²)` (with `ε = ADJ_SVD_EPS`)
/// which stays bounded by `1/ε` and is exactly `1/(σⱼ² − σᵢ²)` away
/// from the degeneracy. At exact degeneracy `F_ij → 0` — the rotational
/// gauge between equal-σ columns is genuinely ambiguous, so projecting
/// the gradient onto the symmetric part is the right limit.
///
/// The first `a` argument is accepted for signature symmetry with the
/// Warp / autograd convention but currently unused — the adjoint is
/// expressed purely in terms of the forward outputs.
pub fn adj_svd3(
    _a: Mat3,
    u: Mat3,
    sigma: Vec3,
    v: Mat3,
    adj_u: Mat3,
    adj_sigma: Vec3,
    adj_v: Mat3,
) -> Mat3 {
    // F_ij is anti-symmetric in (i,j); we only need the strict-upper
    // entries and negate for the lower.
    let s0 = sigma.x;
    let s1 = sigma.y;
    let s2 = sigma.z;
    let f01 = safe_inv_sq_diff(s0, s1, ADJ_SVD_EPS);
    let f02 = safe_inv_sq_diff(s0, s2, ADJ_SVD_EPS);
    let f12 = safe_inv_sq_diff(s1, s2, ADJ_SVD_EPS);
    // F = [[ 0,    F01,  F02 ],
    //      [-F01,  0,    F12 ],
    //      [-F02, -F12,  0   ]]
    let f = Mat3::from_cols(
        Vec3::new(0.0, -f01, -f02),
        Vec3::new(f01, 0.0, -f12),
        Vec3::new(f02, f12, 0.0),
    );

    let sigma_diag = Mat3::from_diagonal(sigma);
    let adj_sigma_diag = Mat3::from_diagonal(adj_sigma);
    let ut = u.transpose();
    let vt = v.transpose();

    // 1) Direct σ contribution: U · diag(adj_σ) · Vᵀ.
    let sigma_term = u * adj_sigma_diag * vt;

    // 2) U-rotation contribution.
    //    Ω_U = Uᵀ·adj_U − adj_Uᵀ·U   (twice the skew part of Uᵀ·adj_U).
    //    block_u = (F ⊙ Ω_U) · Σ
    //    u_term = U · block_u · Vᵀ
    let omega_u = ut * adj_u - adj_u.transpose() * u;
    let skew_u = cw_mul(f, omega_u);
    let u_term = u * (skew_u * sigma_diag) * vt;

    // 3) V-rotation contribution.
    //    Ω_V = Vᵀ·adj_V − adj_Vᵀ·V   (twice the skew part).
    //    block_v = Σ · (F ⊙ Ω_V)
    //    v_term = U · block_v · Vᵀ
    let omega_v = vt * adj_v - adj_v.transpose() * v;
    let skew_v = cw_mul(f, omega_v);
    let v_term = u * (sigma_diag * skew_v) * vt;

    sigma_term + u_term + v_term
}

/// Reverse-mode adjoint of [`polar_decompose_3`].
///
/// Given `F = R · S` (forward) with `R = U·Vᵀ` and `S = V·Σ·Vᵀ`,
/// returns `adj_F = ∂J/∂F` from upstream gradients `adj_R`, `adj_S`.
///
/// Implementation: chain through SVD. Differentiation of `R = U·Vᵀ`
/// gives
///   `adj_U ← adj_R · V`
///   `adj_V ← adj_Rᵀ · U`
/// and of `S = V·Σ·Vᵀ` gives
///   `adj_σᵢ ← (Vᵀ · adj_S · V)_{ii}`
///   `adj_V += (adj_S + adj_Sᵀ) · V · Σ`
/// then we hand off to [`adj_svd3`].
///
/// The `r` and `s` arguments are unused in the math (the polar adjoint
/// is expressed purely through the SVD path) but kept in the signature
/// for parity with the autograd convention and to allow future
/// algebraic shortcuts.
pub fn adj_polar(f: Mat3, r: Mat3, s: Mat3, adj_r: Mat3, adj_s: Mat3) -> Mat3 {
    let _ = (r, s);
    // Recover (U, σ, V) from F. Cheap (single SVD) and matches what the
    // forward did, so all approximation errors cancel under FD.
    let (u, sigma, v) = svd3(f);

    // From R = U·Vᵀ.
    let adj_u_r = adj_r * v;
    let adj_v_r = adj_r.transpose() * u;

    // From S = V·Σ·Vᵀ.
    let vt = v.transpose();
    let vt_adjs_v = vt * adj_s * v;
    let adj_sigma_s = Vec3::new(vt_adjs_v.col(0).x, vt_adjs_v.col(1).y, vt_adjs_v.col(2).z);
    let adj_v_s = (adj_s + adj_s.transpose()) * v * Mat3::from_diagonal(sigma);

    adj_svd3(f, u, sigma, v, adj_u_r, adj_sigma_s, adj_v_r + adj_v_s)
}

/// Estimate the **2-norm condition number** of a 3×3 matrix as
/// `σ_max / σ_min`. Returns `Real::INFINITY` if `σ_min` is exactly
/// zero (numerically singular).
///
/// Used by [`crate::solvers::ik`] to decide whether to fall back from
/// Cholesky to SVD-based pseudoinverse.
pub fn cond2_3(m: Mat3) -> Real {
    let (_, sigma, _) = svd3(m);
    let mut smax = 0.0_f64;
    let mut smin = Real::INFINITY;
    for i in 0..3 {
        let s = sigma[i];
        if s > smax {
            smax = s;
        }
        if s < smin {
            smin = s;
        }
    }
    if smin <= 0.0 {
        Real::INFINITY
    } else {
        smax / smin
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

#[inline]
fn set_col(m: &mut Mat3, i: usize, v: Vec3) {
    match i {
        0 => *m = Mat3::from_cols(v, m.col(1), m.col(2)),
        1 => *m = Mat3::from_cols(m.col(0), v, m.col(2)),
        _ => *m = Mat3::from_cols(m.col(0), m.col(1), v),
    }
}

#[inline]
fn basis_vec(i: usize) -> Vec3 {
    match i {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    }
}

#[inline]
fn mget(m: &Mat3, i: usize, j: usize) -> Real {
    // glam stores columns; m.col(j)[i] is row i, col j.
    m.col(j)[i]
}

#[inline]
fn mset(m: &mut Mat3, i: usize, j: usize, v: Real) {
    let mut c = m.col(j);
    c[i] = v;
    set_col(m, j, c);
}

/// Component-wise (Hadamard) product of two 3×3 matrices.
#[inline]
fn cw_mul(a: Mat3, b: Mat3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(
            a.col(0).x * b.col(0).x,
            a.col(0).y * b.col(0).y,
            a.col(0).z * b.col(0).z,
        ),
        Vec3::new(
            a.col(1).x * b.col(1).x,
            a.col(1).y * b.col(1).y,
            a.col(1).z * b.col(1).z,
        ),
        Vec3::new(
            a.col(2).x * b.col(2).x,
            a.col(2).y * b.col(2).y,
            a.col(2).z * b.col(2).z,
        ),
    )
}

/// Returns `1 / (b² − a²)` Tikhonov-regularised so that the result is
/// bounded by `1/ε` even when `b² ≈ a²`. Used by [`adj_svd3`] to handle
/// degenerate singular values.
#[inline]
fn safe_inv_sq_diff(a: Real, b: Real, eps: Real) -> Real {
    let d = b * b - a * a;
    d / (d * d + eps * eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frob(a: Mat3) -> Real {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let v = mget(&a, i, j);
                s += v * v;
            }
        }
        s.sqrt()
    }

    #[test]
    fn svd_identity() {
        let (u, sigma, v) = svd3(Mat3::IDENTITY);
        let recon = u * Mat3::from_diagonal(sigma) * v.transpose();
        assert!(frob(recon - Mat3::IDENTITY) < 1.0e-12);
        for i in 0..3 {
            assert!((sigma[i] - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn svd_zero() {
        let (u, sigma, v) = svd3(Mat3::ZERO);
        let recon = u * Mat3::from_diagonal(sigma) * v.transpose();
        assert!(frob(recon - Mat3::ZERO) < 1.0e-12);
        for i in 0..3 {
            assert!(sigma[i].abs() < 1.0e-12);
        }
    }

    #[test]
    fn svd_diagonal() {
        let m = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 5.0));
        let (u, sigma, v) = svd3(m);
        let recon = u * Mat3::from_diagonal(sigma) * v.transpose();
        assert!(frob(recon - m) < 1.0e-10);
    }

    #[test]
    fn qr_identity_is_identity() {
        let (q, r) = qr3(Mat3::IDENTITY);
        assert!(frob(q - Mat3::IDENTITY) < 1.0e-12);
        assert!(frob(r - Mat3::IDENTITY) < 1.0e-12);
    }

    #[test]
    fn eig_diagonal_recovers_eigenvalues() {
        let m = Mat3::from_diagonal(Vec3::new(1.0, 4.0, 9.0));
        let (lambda, v) = eig3_sym(m);
        // Eigenvalues should be a permutation of {1, 4, 9}.
        let mut got: [Real; 3] = [lambda.x, lambda.y, lambda.z];
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((got[0] - 1.0).abs() < 1.0e-10);
        assert!((got[1] - 4.0).abs() < 1.0e-10);
        assert!((got[2] - 9.0).abs() < 1.0e-10);
        // V should be a signed permutation matrix → orthogonal.
        let vt_v = v.transpose() * v;
        assert!(frob(vt_v - Mat3::IDENTITY) < 1.0e-10);
    }

    #[test]
    fn polar_recovers_rotation_and_stretch() {
        // M = R · S with a known rotation and SPD stretch.
        let r_true = Mat3::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        let s_true = Mat3::from_diagonal(Vec3::new(1.5, 2.0, 0.8));
        let m = r_true * s_true;
        let (r, s) = polar_decompose_3(m);
        let recon = r * s;
        assert!(frob(recon - m) < 1.0e-10);
        // R is orthogonal.
        assert!(frob(r.transpose() * r - Mat3::IDENTITY) < 1.0e-10);
        // S is symmetric.
        assert!(frob(s - s.transpose()) < 1.0e-10);
    }

    #[test]
    fn cond_identity_is_one() {
        assert!((cond2_3(Mat3::IDENTITY) - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn adj_svd_identity_zero_upstream_is_zero() {
        // With all upstream gradients = 0, adj_A must be 0.
        let a = Mat3::IDENTITY;
        let (u, sigma, v) = svd3(a);
        let adj_a = adj_svd3(a, u, sigma, v, Mat3::ZERO, Vec3::ZERO, Mat3::ZERO);
        assert!(frob(adj_a) < 1.0e-12);
    }

    #[test]
    fn adj_polar_identity_zero_upstream_is_zero() {
        let f = Mat3::IDENTITY;
        let (r, s) = polar_decompose_3(f);
        let adj_f = adj_polar(f, r, s, Mat3::ZERO, Mat3::ZERO);
        assert!(frob(adj_f) < 1.0e-12);
    }

    #[test]
    fn cond_singular_is_inf() {
        // Rank-1 matrix.
        let m = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(2.0, 4.0, 6.0),
            Vec3::new(3.0, 6.0, 9.0),
        );
        assert!(cond2_3(m).is_infinite());
    }
}
