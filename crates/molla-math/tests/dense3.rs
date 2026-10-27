//! Stage 24a — small-matrix decomposition correctness tests.
//!
//! 10k random-matrix samples per primitive plus targeted edge cases
//! (identity, zero, rank-deficient, near-singular). All checks run on
//! the CPU; the GPU parity tests live in `dense3_gpu.rs` (gated behind
//! the `gpu` feature + `#[ignore]`).

use molla_math::dense3::{cond2_3, eig3_sym, polar_decompose_3, qr3, svd3};
use molla_math::{Mat3, Real, Vec3};

const N_RANDOM: usize = 10_000;
const RECON_TOL: Real = 1.0e-6;
const ORTHO_TOL: Real = 1.0e-6;
const EIG_TOL: Real = 1.0e-6;

/// Tiny LCG so tests are dependency-free and reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF)
    }
    fn next(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> Real {
        // (0, 1) excluding 0 to avoid the all-zero matrix sneaking in
        // (we test that explicitly elsewhere).
        let x = self.next() >> 11;
        (x as Real + 1.0) / ((1u64 << 53) as Real + 1.0)
    }
    fn signed(&mut self) -> Real {
        2.0 * self.uniform() - 1.0
    }
    fn rand_mat(&mut self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
        )
    }
    fn rand_sym(&mut self) -> Mat3 {
        let m = self.rand_mat();
        // Symmetrise (M + Mᵀ) / 2.
        0.5 * (m + m.transpose())
    }
}

fn mget(m: &Mat3, i: usize, j: usize) -> Real {
    m.col(j)[i]
}

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

// ─── SVD ───────────────────────────────────────────────────────────

#[test]
fn svd_random_reconstruction() {
    let mut rng = Lcg::new(0xA1A1_A1A1);
    let mut max_err = 0.0_f64;
    for _ in 0..N_RANDOM {
        let m = rng.rand_mat();
        let (u, sigma, v) = svd3(m);
        let recon = u * Mat3::from_diagonal(sigma) * v.transpose();
        let err = frob(recon - m);
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < RECON_TOL,
            "SVD reconstruction failed: ‖M - U·Σ·Vᵀ‖ = {err}"
        );
        // U orthogonal.
        assert!(frob(u.transpose() * u - Mat3::IDENTITY) < ORTHO_TOL);
        // V orthogonal.
        assert!(frob(v.transpose() * v - Mat3::IDENTITY) < ORTHO_TOL);
        // σ ≥ 0.
        for i in 0..3 {
            assert!(
                sigma[i] >= -1.0e-12,
                "negative singular value: {}",
                sigma[i]
            );
        }
    }
    eprintln!("SVD max reconstruction error over {N_RANDOM} samples: {max_err:e}");
}

#[test]
fn svd_identity_zero_diag_edge_cases() {
    // Identity.
    let (u, s, v) = svd3(Mat3::IDENTITY);
    assert!(frob(u * Mat3::from_diagonal(s) * v.transpose() - Mat3::IDENTITY) < 1.0e-12);

    // Zero matrix — singular values are all zero, U and V can be any
    // orthogonal completion.
    let (u, s, v) = svd3(Mat3::ZERO);
    let recon = u * Mat3::from_diagonal(s) * v.transpose();
    assert!(frob(recon) < 1.0e-12);
    for i in 0..3 {
        assert!(s[i].abs() < 1.0e-12);
    }

    // Diagonal.
    let m = Mat3::from_diagonal(Vec3::new(0.5, 1.0, 100.0));
    let (u, s, v) = svd3(m);
    assert!(frob(u * Mat3::from_diagonal(s) * v.transpose() - m) < 1.0e-9);
}

#[test]
fn svd_rank_deficient_and_near_singular() {
    // Rank-1: outer product u·vᵀ.
    let u = Vec3::new(1.0, 2.0, -1.5).normalize();
    let v = Vec3::new(0.7, -0.3, 0.4).normalize();
    let m = Mat3::from_cols(u * v.x, u * v.y, u * v.z);
    let (uu, ss, vv) = svd3(m);
    let recon = uu * Mat3::from_diagonal(ss) * vv.transpose();
    assert!(
        frob(recon - m) < 1.0e-9,
        "rank-1 recon err {}",
        frob(recon - m)
    );

    // Two of the singular values should be (numerically) zero.
    let mut sorted: [Real; 3] = [ss.x, ss.y, ss.z];
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(sorted[0].abs() < 1.0e-12, "rank-1: σ_min = {}", sorted[0]);
    assert!(sorted[1].abs() < 1.0e-12, "rank-1: σ_mid = {}", sorted[1]);
    assert!(sorted[2] > 0.5, "rank-1: σ_max = {}", sorted[2]);

    // Near-singular: condition ~ 1e6.
    let m_ns = Mat3::from_diagonal(Vec3::new(1.0, 1.0e-6, 1.0));
    let c = cond2_3(m_ns);
    assert!(c > 9.9e5 && c < 1.1e6, "expected cond ~ 1e6, got {c}");
    let (u, s, v) = svd3(m_ns);
    let recon = u * Mat3::from_diagonal(s) * v.transpose();
    assert!(frob(recon - m_ns) < 1.0e-12);
}

// ─── QR ────────────────────────────────────────────────────────────

#[test]
fn qr_random_reconstruction() {
    let mut rng = Lcg::new(0xB2B2_B2B2);
    let mut max_recon = 0.0_f64;
    let mut max_orth = 0.0_f64;
    for _ in 0..N_RANDOM {
        let m = rng.rand_mat();
        let (q, r) = qr3(m);
        let recon = q * r;
        let recon_err = frob(recon - m);
        let orth_err = frob(q.transpose() * q - Mat3::IDENTITY);
        if recon_err > max_recon {
            max_recon = recon_err;
        }
        if orth_err > max_orth {
            max_orth = orth_err;
        }
        assert!(recon_err < RECON_TOL, "QR recon err = {recon_err}");
        assert!(orth_err < ORTHO_TOL, "Q not orthogonal, err = {orth_err}");
        // R upper triangular.
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    mget(&r, i, j).abs() < 1.0e-14,
                    "R not upper-tri at ({i},{j}): {}",
                    mget(&r, i, j),
                );
            }
        }
    }
    eprintln!("QR max recon: {max_recon:e}, max ortho: {max_orth:e}");
}

#[test]
fn qr_identity_zero_diag_edge_cases() {
    let (q, r) = qr3(Mat3::IDENTITY);
    assert!(frob(q - Mat3::IDENTITY) < 1.0e-14);
    assert!(frob(r - Mat3::IDENTITY) < 1.0e-14);

    let (q, r) = qr3(Mat3::ZERO);
    // Zero R, Q stays identity (no rotations applied).
    assert!(frob(q * r - Mat3::ZERO) < 1.0e-14);
    assert!(frob(r) < 1.0e-14);

    let m = Mat3::from_diagonal(Vec3::new(2.0, -3.0, 5.0));
    let (q, r) = qr3(m);
    assert!(frob(q * r - m) < 1.0e-12);
}

// ─── Symmetric eigendecomposition ─────────────────────────────────

#[test]
fn eig_random_symmetric() {
    let mut rng = Lcg::new(0xC3C3_C3C3);
    let mut max_pair_err = 0.0_f64;
    for _ in 0..N_RANDOM {
        let m = rng.rand_sym();
        let (lambda, v) = eig3_sym(m);
        // Each pair must satisfy M·v_i = λ_i · v_i.
        for i in 0..3 {
            let vi = v.col(i);
            let mvi = m * vi;
            let expected = vi * lambda[i];
            let err = (mvi - expected).length();
            if err > max_pair_err {
                max_pair_err = err;
            }
            assert!(
                err < EIG_TOL,
                "eigenpair {i} fails: ‖M·v - λ·v‖ = {err}, λ = {}",
                lambda[i]
            );
        }
        // V orthogonal.
        assert!(
            frob(v.transpose() * v - Mat3::IDENTITY) < ORTHO_TOL,
            "V not orthogonal"
        );
    }
    eprintln!("Eig max pair-residual over {N_RANDOM} samples: {max_pair_err:e}");
}

#[test]
fn eig_diagonal_and_repeated_eigenvalues() {
    // Diagonal with repeated eigenvalue.
    let m = Mat3::from_diagonal(Vec3::new(2.0, 2.0, 5.0));
    let (lambda, v) = eig3_sym(m);
    let mut got: [Real; 3] = [lambda.x, lambda.y, lambda.z];
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((got[0] - 2.0).abs() < 1.0e-10);
    assert!((got[1] - 2.0).abs() < 1.0e-10);
    assert!((got[2] - 5.0).abs() < 1.0e-10);
    // Eigenvectors orthogonal.
    assert!(frob(v.transpose() * v - Mat3::IDENTITY) < 1.0e-10);
    for i in 0..3 {
        let vi = v.col(i);
        let err = (m * vi - vi * lambda[i]).length();
        assert!(err < 1.0e-10, "diag-repeated: pair {i} err {err}");
    }

    // All identical eigenvalues — every direction is an eigenvector.
    let m = Mat3::from_diagonal(Vec3::splat(3.0));
    let (lambda, v) = eig3_sym(m);
    for i in 0..3 {
        assert!((lambda[i] - 3.0).abs() < 1.0e-12);
    }
    assert!(frob(v.transpose() * v - Mat3::IDENTITY) < 1.0e-10);
}

// ─── Polar decomposition ──────────────────────────────────────────

#[test]
fn polar_random_reconstruction() {
    let mut rng = Lcg::new(0xD4D4_D4D4);
    let mut max_err = 0.0_f64;
    for _ in 0..N_RANDOM {
        let m = rng.rand_mat();
        let (r, s) = polar_decompose_3(m);
        let recon = r * s;
        let err = frob(recon - m);
        if err > max_err {
            max_err = err;
        }
        assert!(err < RECON_TOL, "polar recon err = {err}");
        // R orthogonal.
        assert!(
            frob(r.transpose() * r - Mat3::IDENTITY) < ORTHO_TOL,
            "R not orthogonal"
        );
        // S symmetric.
        assert!(frob(s - s.transpose()) < 1.0e-9, "S not symmetric");
        // S PSD: all eigenvalues ≥ -tol.
        let (lam, _) = eig3_sym(s);
        for i in 0..3 {
            assert!(lam[i] > -1.0e-9, "S not PSD: λ = {}", lam[i]);
        }
    }
    eprintln!("Polar max recon error over {N_RANDOM} samples: {max_err:e}");
}

#[test]
fn polar_identity_zero() {
    let (r, s) = polar_decompose_3(Mat3::IDENTITY);
    assert!(frob(r - Mat3::IDENTITY) < 1.0e-12);
    assert!(frob(s - Mat3::IDENTITY) < 1.0e-12);

    let (r, s) = polar_decompose_3(Mat3::ZERO);
    assert!(frob(r * s) < 1.0e-12);
    assert!(frob(s) < 1.0e-12);
    // R is some orthogonal completion.
    assert!(frob(r.transpose() * r - Mat3::IDENTITY) < 1.0e-12);
}

// ─── Conditioning ─────────────────────────────────────────────────

#[test]
fn cond_well_conditioned_close_to_one() {
    // Slight perturbation of identity → cond ~ 1.
    let m = Mat3::from_diagonal(Vec3::new(1.0, 1.05, 0.95));
    let c = cond2_3(m);
    assert!(c > 1.0 && c < 1.2, "cond {c}");
}

#[test]
fn cond_ill_conditioned_above_threshold() {
    // Cond = 1e8, well above the 1e6 IK threshold.
    let m = Mat3::from_diagonal(Vec3::new(1.0e4, 1.0, 1.0e-4));
    let c = cond2_3(m);
    assert!(c > 1.0e6, "expected cond > 1e6, got {c}");
}
