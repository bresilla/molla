//! Stage 38b — adjoint SVD3 + polar finite-difference oracle tests.
//!
//! Three tests on the CPU:
//!   * `adj_svd3_fd_oracle_random_matrices` — central-difference check
//!     against a quadratic loss `J(A) = ‖U−Q‖² + ‖σ−s‖² + ‖V−R‖²` on
//!     16 random well-conditioned 3×3 matrices.
//!   * `adj_polar_fd_oracle` — same idea against `J(F) = ‖R−Q‖² + ‖S−T‖²`
//!     for random target matrices.
//!   * `adj_svd3_degenerate_sigma_handles_cleanly` — degenerate (two
//!     equal singular values) input; assert `adj_A` is finite + bounded.
//!     No FD comparison because the gauge between equal-σ columns is
//!     ambiguous and FD is unreliable in that regime — we only check
//!     that the regularised Tikhonov form stays bounded.

use molla_math::dense3::{adj_polar, adj_svd3, polar_decompose_3, svd3};
use molla_math::{Mat3, Real, Vec3};

// ─── RNG + helpers ─────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFEF00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> Real {
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
    /// Random well-conditioned matrix: random rotation × diag(σ) × random
    /// rotation, with σ_i ∈ [0.5, 2.0] so singular values are well
    /// separated (away from the degenerate-σ regime).
    fn rand_wellcond(&mut self) -> Mat3 {
        let q1 = rand_rotation(self);
        let q2 = rand_rotation(self);
        // Spread σ over [0.5, 2.5] and shuffle so degeneracy is unlikely.
        let s = Vec3::new(
            0.5 + 1.0 * self.uniform(),
            1.0 + 1.0 * self.uniform(),
            2.0 + 0.5 * self.uniform(),
        );
        q1 * Mat3::from_diagonal(s) * q2
    }
}

/// Build a random rotation via three axis-angle composes.
fn rand_rotation(rng: &mut Lcg) -> Mat3 {
    let ax = Vec3::new(rng.signed(), rng.signed(), rng.signed()).normalize();
    let theta = std::f64::consts::PI * rng.uniform();
    Mat3::from_axis_angle(ax, theta)
}

fn mget(m: &Mat3, i: usize, j: usize) -> Real {
    m.col(j)[i]
}

fn mset(m: &mut Mat3, i: usize, j: usize, v: Real) {
    let mut col = m.col(j);
    col[i] = v;
    let cols = [m.col(0), m.col(1), m.col(2)];
    *m = Mat3::from_cols(
        if j == 0 { col } else { cols[0] },
        if j == 1 { col } else { cols[1] },
        if j == 2 { col } else { cols[2] },
    );
    let _ = cols;
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

fn frob_sq(a: Mat3) -> Real {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let v = mget(&a, i, j);
            s += v * v;
        }
    }
    s
}

// ─── SVD oracle ────────────────────────────────────────────────────

/// Loss `J(A) = ‖U − Q‖² + ‖σ − s‖² + ‖V − R‖²` evaluated by running a
/// fresh SVD on A.
fn loss_svd(a: Mat3, q: Mat3, s_tgt: Vec3, r: Mat3) -> Real {
    let (u, sigma, v) = svd3(a);
    let du = u - q;
    let dv = v - r;
    let ds = sigma - s_tgt;
    frob_sq(du) + ds.dot(ds) + frob_sq(dv)
}

#[test]
fn adj_svd3_fd_oracle_random_matrices() {
    let mut rng = Lcg::new(0xA1A1_FEED);
    let n = 16usize;
    let h = 1.0e-6;
    let mut max_rel_err = 0.0_f64;
    for trial in 0..n {
        let a = rng.rand_wellcond();
        // Pick fixed but random targets per trial so the gradient is
        // not at a degenerate stationary point.
        let q_tgt = rand_rotation(&mut rng);
        let r_tgt = rand_rotation(&mut rng);
        let s_tgt = Vec3::new(rng.signed(), rng.signed(), rng.signed());

        // Analytic adjoint.
        let (u, sigma, v) = svd3(a);
        let adj_u = 2.0 * (u - q_tgt);
        let adj_sigma = 2.0 * (sigma - s_tgt);
        let adj_v = 2.0 * (v - r_tgt);
        let adj_a = adj_svd3(a, u, sigma, v, adj_u, adj_sigma, adj_v);

        // Finite-difference each of the 9 entries.
        let mut fd = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut ap = a;
                let mut am = a;
                mset(&mut ap, i, j, mget(&a, i, j) + h);
                mset(&mut am, i, j, mget(&a, i, j) - h);
                let jp = loss_svd(ap, q_tgt, s_tgt, r_tgt);
                let jm = loss_svd(am, q_tgt, s_tgt, r_tgt);
                let d = (jp - jm) / (2.0 * h);
                mset(&mut fd, i, j, d);
            }
        }

        let diff = adj_a - fd;
        let denom = frob(fd).max(frob(adj_a)).max(1.0);
        let rel = frob(diff) / denom;
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        assert!(
            rel < 1.0e-3,
            "trial {trial}: rel err {rel} (analytic {:?}, fd {:?})",
            adj_a,
            fd,
        );
    }
    eprintln!("adj_svd3 FD oracle max relative error over {n} trials: {max_rel_err:e}");
}

// ─── Polar oracle ──────────────────────────────────────────────────

fn loss_polar(f: Mat3, q: Mat3, t: Mat3) -> Real {
    let (r, s) = polar_decompose_3(f);
    frob_sq(r - q) + frob_sq(s - t)
}

#[test]
fn adj_polar_fd_oracle() {
    let mut rng = Lcg::new(0xB2B2_C0DE);
    let n = 16usize;
    let h = 1.0e-6;
    let mut max_rel_err = 0.0_f64;
    for trial in 0..n {
        let f = rng.rand_wellcond();
        let q_tgt = rand_rotation(&mut rng);
        let t_tgt = {
            // symmetric SPD-ish target so the polar S has something to
            // chase; doesn't have to be exactly SPD for the gradient
            // check.
            let m = rng.rand_mat();
            0.5 * (m + m.transpose())
        };

        let (r, s) = polar_decompose_3(f);
        let adj_r = 2.0 * (r - q_tgt);
        let adj_s = 2.0 * (s - t_tgt);
        let adj_f = adj_polar(f, r, s, adj_r, adj_s);

        let mut fd = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut fp = f;
                let mut fm = f;
                mset(&mut fp, i, j, mget(&f, i, j) + h);
                mset(&mut fm, i, j, mget(&f, i, j) - h);
                let jp = loss_polar(fp, q_tgt, t_tgt);
                let jm = loss_polar(fm, q_tgt, t_tgt);
                mset(&mut fd, i, j, (jp - jm) / (2.0 * h));
            }
        }

        let denom = frob(fd).max(frob(adj_f)).max(1.0);
        let rel = frob(adj_f - fd) / denom;
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        assert!(
            rel < 1.0e-3,
            "polar trial {trial}: rel err {rel}, adj_f={:?}, fd={:?}",
            adj_f,
            fd,
        );
    }
    eprintln!("adj_polar FD oracle max relative error over {n} trials: {max_rel_err:e}");
}

// ─── Degenerate σ handling ─────────────────────────────────────────

#[test]
fn adj_svd3_degenerate_sigma_handles_cleanly() {
    // A near-uniform scale with a tiny perturbation. Two singular
    // values are extremely close; without regularisation the F_ij
    // divisor blows up and we'd see NaN / Inf.
    let cases: [Mat3; 3] = [
        // Exact double singular value (uniform scale).
        Mat3::from_diagonal(Vec3::new(2.0, 2.0, 5.0)),
        // All three equal (gauge totally ambiguous).
        Mat3::from_diagonal(Vec3::new(3.0, 3.0, 3.0)),
        // Tiny perturbation around a double σ.
        {
            let r = Mat3::from_axis_angle(Vec3::new(0.3, 0.7, 0.5).normalize(), 0.4);
            let s = Mat3::from_diagonal(Vec3::new(2.0, 2.0 + 1.0e-10, 5.0));
            r * s * r.transpose()
        },
    ];

    let mut max_norm = 0.0_f64;
    for (i, a) in cases.iter().enumerate() {
        let (u, sigma, v) = svd3(*a);
        // Pick a non-trivial upstream so we exercise the F-skew terms.
        let adj_u = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let adj_sigma = Vec3::new(0.1, 0.2, 0.3);
        let adj_v = Mat3::from_cols(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let adj_a = adj_svd3(*a, u, sigma, v, adj_u, adj_sigma, adj_v);

        // Every entry must be finite.
        for ii in 0..3 {
            for jj in 0..3 {
                let v = mget(&adj_a, ii, jj);
                assert!(v.is_finite(), "case {i}: adj_A[{ii},{jj}] = {v} not finite",);
            }
        }
        let n = frob(adj_a);
        if n > max_norm {
            max_norm = n;
        }
        // Bound: 1/ε with ε = 1e-12 gives upper bound ~1e12. We allow
        // the result to be that large but not unbounded — anything
        // beyond 1e14 is a sign the regularisation has misfired.
        assert!(n < 1.0e14, "case {i}: ‖adj_A‖ = {n} exceeds bound 1e14",);
    }
    eprintln!("adj_svd3 degenerate-σ max ‖adj_A‖ across cases: {max_norm:e}");
}
