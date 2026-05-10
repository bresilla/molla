//! Stage 37.8 — Monte Carlo gradient propagation.
//!
//! Stochastic gradient estimator for losses that are non-differentiable
//! (or where building a tape is too expensive). Given a black-box
//! `f: &[Real] -> Real` and a parameter vector, runs `n_samples`
//! perturbations of the form `params ± ε · noise[k]` and accumulates
//! a per-component gradient estimate. This is the "SPSA" recipe of
//! Spall (1992) — one of the cheapest stochastic gradient
//! approximations and a useful complement to molla's tape autograd
//! ([`crate::autograd::tape`]) when the loss isn't tape-recordable.
//!
//! The Monte Carlo path is **noisy** by design: an `n_samples = 64`
//! estimator on a smooth quadratic typically lands within ~5% of
//! the true gradient component. Tighten the tolerance by raising
//! `n_samples` (variance ∝ 1 / N) or by using a Richardson-style
//! averaged perturbation pattern (a v2 nice-to-have).
//!
//! ## API surface
//!
//! - [`mcgp_estimate`] — the standalone SPSA-style estimator. Takes a
//!   loss closure plus parameters and returns an estimated gradient
//!   the same length as `params`.
//! - [`McgpConfig`] — knobs (sample count, perturbation magnitude,
//!   RNG seed).
//! - [`finite_diff_central`] — central-difference reference for tests.

use crate::math::Real;

/// Knobs for [`mcgp_estimate`].
#[derive(Copy, Clone, Debug)]
pub struct McgpConfig {
    /// Number of MC samples. Higher = lower variance, more compute.
    pub n_samples: usize,
    /// Perturbation magnitude (`ε` in the SPSA formulation).
    pub epsilon: Real,
    /// Deterministic RNG seed. Two runs with the same seed produce
    /// the same gradient estimate.
    pub seed: u64,
}

impl Default for McgpConfig {
    fn default() -> Self {
        Self {
            n_samples: 64,
            epsilon: 1.0e-3,
            seed: 0x5EED_F00D,
        }
    }
}

/// Tiny xorshift64 — RNG-only (uniform u64 stream); the only
/// random primitive this module needs.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state which xorshift can't escape.
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// ±1 uniformly.
    fn rademacher(&mut self) -> Real {
        if (self.next() & 1) == 1 {
            1.0
        } else {
            -1.0
        }
    }
}

/// Stochastic SPSA gradient estimate for a black-box loss.
///
/// Returns a `Vec<Real>` of the same length as `params` containing
/// the estimated `∂f/∂params[i]`. The estimate is the per-sample
/// average of `(f(p + ε·d) - f(p - ε·d)) / (2 ε d_i)` where `d` is a
/// fresh ±1 Rademacher vector each sample.
pub fn mcgp_estimate<F>(f: &F, params: &[Real], config: &McgpConfig) -> Vec<Real>
where
    F: Fn(&[Real]) -> Real,
{
    let n = params.len();
    let mut grad: Vec<Real> = vec![0.0; n];
    if n == 0 || config.n_samples == 0 {
        return grad;
    }
    let mut rng = XorShift64::new(config.seed);
    let mut p_plus: Vec<Real> = vec![0.0; n];
    let mut p_minus: Vec<Real> = vec![0.0; n];
    let mut noise: Vec<Real> = vec![0.0; n];
    let inv_two_eps = 1.0 / (2.0 * config.epsilon);

    for _ in 0..config.n_samples {
        for i in 0..n {
            let d = rng.rademacher();
            noise[i] = d;
            p_plus[i] = params[i] + config.epsilon * d;
            p_minus[i] = params[i] - config.epsilon * d;
        }
        let f_plus = f(&p_plus);
        let f_minus = f(&p_minus);
        let scale = (f_plus - f_minus) * inv_two_eps;
        for i in 0..n {
            // SPSA: g_i ≈ (Δf) / (2 ε d_i). For Rademacher d_i ∈ {-1, 1},
            // 1 / d_i = d_i, so we multiply by d_i.
            grad[i] += scale * noise[i];
        }
    }

    let inv_samples = 1.0 / config.n_samples as Real;
    for g in grad.iter_mut() {
        *g *= inv_samples;
    }
    grad
}

/// Central finite-difference gradient. Used as the reference oracle in
/// tests and exposed as a public utility for callers that want a
/// deterministic check on the SPSA estimate.
pub fn finite_diff_central<F>(f: &F, params: &[Real], h: Real) -> Vec<Real>
where
    F: Fn(&[Real]) -> Real,
{
    let n = params.len();
    let mut grad: Vec<Real> = vec![0.0; n];
    let mut p = params.to_vec();
    for i in 0..n {
        let saved = p[i];
        p[i] = saved + h;
        let f_plus = f(&p);
        p[i] = saved - h;
        let f_minus = f(&p);
        p[i] = saved;
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}
