//! Robust-loss math primitives used by IK regularisation. All
//! functions are scalar; vectorise at the call site.
//!
//! Mirrors Newton's `_src/math/robust_loss.py` API.
//!
//! # Loss families
//!
//! Every loss `ρ(r)` is paired with its derivative `ρ'(r)`. The
//! IRLS-equivalent weight is `w(r) = ρ'(r) / r` (with a small
//! tolerance at `r ≈ 0`); the [`RobustLoss::weight`] helper bundles
//! that formula so call-sites stay tidy.
//!
//! - **L2** — `½ r²`, `r`. The plain quadratic; included so callers
//!   can tag back-compat behaviour explicitly via [`RobustLoss::L2`].
//! - **L1** — `|r|`, `sign(r)`. Sub-gradient at `r = 0` returns 0.
//! - **Smooth-L2** — `√(r² + ε²) − ε`, `r / √(r² + ε²)`. Smooth
//!   approximation of L1/L2 transition; reduces to L2 as `ε → 0`.
//! - **Huber** — quadratic inside `|r| ≤ δ`, linear outside.
//! - **Pseudo-Huber** — C∞ Huber surrogate, `δ²·(√(1 + (r/δ)²) − 1)`.
//! - **Tukey biweight** — `(c²/6)·(1 − (1 − (r/c)²)³)` for `|r| ≤ c`,
//!   constant past that. Hard outlier rejection (zero gradient).
//! - **Cauchy** — `(c²/2)·ln(1 + (r/c)²)`. Soft outlier down-weighting
//!   that never quite hits zero gradient.

use crate::Real;

/// Selector for which robust loss to apply during an IRLS-weighted IK
/// solve (Stage 24b). Default is [`RobustLoss::L2`] so existing call
/// sites keep their plain Cholesky-+-L2 behaviour.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RobustLoss {
    /// Plain L2 — `½ r²`, gradient `r`, weight `1`.
    L2,
    /// Absolute-value loss — IRLS weight `1/|r|` (clamped near zero).
    L1,
    /// Smoothed L2/L1 with floor `eps`.
    SmoothL2 { eps: Real },
    /// Huber: quadratic inside `|r| ≤ delta`, linear past that.
    Huber { delta: Real },
    /// C∞ pseudo-Huber surrogate.
    PseudoHuber { delta: Real },
    /// Tukey biweight: rejects `|r| > c` outright.
    Tukey { c: Real },
    /// Cauchy / Lorentzian: soft down-weighting.
    Cauchy { c: Real },
}

impl Default for RobustLoss {
    fn default() -> Self {
        RobustLoss::L2
    }
}

impl RobustLoss {
    /// Evaluate the loss `ρ(r)`.
    #[must_use]
    pub fn rho(self, r: Real) -> Real {
        match self {
            RobustLoss::L2 => 0.5 * r * r,
            RobustLoss::L1 => norm_l1(r),
            RobustLoss::SmoothL2 { eps } => norm_l2_smooth(r, eps),
            RobustLoss::Huber { delta } => norm_huber(r, delta),
            RobustLoss::PseudoHuber { delta } => norm_pseudo_huber(r, delta),
            RobustLoss::Tukey { c } => norm_tukey(r, c),
            RobustLoss::Cauchy { c } => norm_cauchy(r, c),
        }
    }

    /// Evaluate the loss derivative `ρ'(r)`.
    #[must_use]
    pub fn grad(self, r: Real) -> Real {
        match self {
            RobustLoss::L2 => r,
            RobustLoss::L1 => norm_l1_grad(r),
            RobustLoss::SmoothL2 { eps } => norm_l2_smooth_grad(r, eps),
            RobustLoss::Huber { delta } => norm_huber_grad(r, delta),
            RobustLoss::PseudoHuber { delta } => norm_pseudo_huber_grad(r, delta),
            RobustLoss::Tukey { c } => norm_tukey_grad(r, c),
            RobustLoss::Cauchy { c } => norm_cauchy_grad(r, c),
        }
    }

    /// IRLS weight `w(r) = ρ'(r) / r`. The `r ≈ 0` branch returns the
    /// loss-specific limit (L2 → 1, L1 → `1/eps`, Tukey → 1, …) so
    /// callers don't have to special-case it.
    ///
    /// Used by the robust IK solver (Stage 24b) to convert any robust
    /// loss into a re-weighted least-squares system: replace each
    /// residual row by `√w(r)·row` and solve the standard normal
    /// equations.
    #[must_use]
    pub fn weight(self, r: Real) -> Real {
        // Tolerance at the singularity. 1e-6 is comfortably above f32
        // round-off and small enough that any real residual will use
        // the analytic branch. L1's sub-gradient at 0 means we need
        // *some* tolerance even when working in f64.
        const ZERO_TOL: Real = 1.0e-6;
        match self {
            RobustLoss::L2 => 1.0,
            RobustLoss::L1 => {
                if r.abs() < ZERO_TOL {
                    1.0 / ZERO_TOL
                } else {
                    1.0 / r.abs()
                }
            }
            RobustLoss::SmoothL2 { eps } => 1.0 / (r * r + eps * eps).sqrt(),
            RobustLoss::Huber { delta } => {
                let abs = r.abs();
                if abs <= delta { 1.0 } else { delta / abs }
            }
            RobustLoss::PseudoHuber { delta } => 1.0 / (1.0 + (r / delta).powi(2)).sqrt(),
            RobustLoss::Tukey { c } => {
                let abs = r.abs();
                if abs >= c {
                    0.0
                } else {
                    let s = r / c;
                    let u = 1.0 - s * s;
                    u * u
                }
            }
            RobustLoss::Cauchy { c } => 1.0 / (1.0 + (r / c).powi(2)),
        }
    }
}

// ─── L1 ────────────────────────────────────────────────────────────

/// Absolute-value loss `ρ(r) = |r|`.
#[must_use]
pub fn norm_l1(r: Real) -> Real {
    r.abs()
}

/// L1 sub-gradient. Convention: `0` at the kink.
#[must_use]
pub fn norm_l1_grad(r: Real) -> Real {
    if r > 0.0 {
        1.0
    } else if r < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ─── Smooth L2 ─────────────────────────────────────────────────────

/// Smooth approximation of L2:
///
/// ```text
/// ρ(r) = √(r² + ε²) − ε
/// ```
///
/// As `ε → 0` this collapses to `|r|` (L1); for `|r| ≪ ε` it behaves
/// like `r²/(2ε)` (L2 scaled). Always C∞ (no kink).
#[must_use]
pub fn norm_l2_smooth(r: Real, eps: Real) -> Real {
    (r * r + eps * eps).sqrt() - eps
}

/// Gradient of [`norm_l2_smooth`].
#[must_use]
pub fn norm_l2_smooth_grad(r: Real, eps: Real) -> Real {
    r / (r * r + eps * eps).sqrt()
}

// ─── Huber ─────────────────────────────────────────────────────────

/// Huber loss `ρ(r)`: quadratic when `|r| ≤ δ`, linear past that.
/// Smooth at the transition (C¹).
///
/// ```text
/// ρ(r) = ½ r²              if |r| ≤ δ
///      = δ·(|r| - δ/2)     if |r| > δ
/// ```
#[must_use]
pub fn norm_huber(r: Real, delta: Real) -> Real {
    let abs = r.abs();
    if abs <= delta {
        0.5 * r * r
    } else {
        delta * (abs - 0.5 * delta)
    }
}

/// Huber-loss derivative `ρ'(r)` w.r.t. `r`.
#[must_use]
pub fn norm_huber_grad(r: Real, delta: Real) -> Real {
    let abs = r.abs();
    if abs <= delta { r } else { delta * r.signum() }
}

// ─── Pseudo-Huber ──────────────────────────────────────────────────

/// Pseudo-Huber loss `ρ(r) = δ²·(√(1 + (r/δ)²) − 1)`. C∞ everywhere
/// (no kink at `r = ±δ`); approaches Huber for small `r/δ`.
#[must_use]
pub fn norm_pseudo_huber(r: Real, delta: Real) -> Real {
    let s = r / delta;
    delta * delta * ((1.0 + s * s).sqrt() - 1.0)
}

/// Pseudo-Huber gradient `ρ'(r)` w.r.t. `r`.
#[must_use]
pub fn norm_pseudo_huber_grad(r: Real, delta: Real) -> Real {
    let s = r / delta;
    r / (1.0 + s * s).sqrt()
}

// ─── Tukey biweight ────────────────────────────────────────────────

/// Tukey biweight loss. Inside the band `|r| ≤ c`:
///
/// ```text
/// ρ(r) = (c²/6)·(1 - (1 - (r/c)²)³)
/// ```
///
/// Outside (`|r| > c`) it saturates at `c²/6` and `ρ'(r) = 0`,
/// rejecting outliers entirely.
#[must_use]
pub fn norm_tukey(r: Real, c: Real) -> Real {
    let cap = c * c / 6.0;
    let abs = r.abs();
    if abs >= c {
        cap
    } else {
        let s = r / c;
        let u = 1.0 - s * s;
        cap * (1.0 - u * u * u)
    }
}

/// Tukey biweight gradient. `0` outside the rejection band.
#[must_use]
pub fn norm_tukey_grad(r: Real, c: Real) -> Real {
    let abs = r.abs();
    if abs >= c {
        0.0
    } else {
        let s = r / c;
        let u = 1.0 - s * s;
        // d/dr[(c²/6)·(1 - u³)] = (c²/6)·(-3u²)·du/dr; du/dr = -2r/c²
        // → r · u²
        r * u * u
    }
}

// ─── Cauchy / Lorentzian ───────────────────────────────────────────

/// Cauchy / Lorentzian loss `ρ(r) = (c²/2)·ln(1 + (r/c)²)`. Soft
/// down-weighting that never reaches a hard zero gradient.
#[must_use]
pub fn norm_cauchy(r: Real, c: Real) -> Real {
    let s = r / c;
    0.5 * c * c * (1.0 + s * s).ln()
}

/// Cauchy gradient `r / (1 + (r/c)²)`.
#[must_use]
pub fn norm_cauchy_grad(r: Real, c: Real) -> Real {
    let s = r / c;
    r / (1.0 + s * s)
}

// ─── Smooth normalisation (Stage 16 leftover) ──────────────────────

/// Smooth-norm normalisation: returns `v / max(|v|, eps)` so the
/// gradient is well-defined at `v = 0`. Useful for direction
/// vectors in IK Jacobian columns when the residual hits the target.
#[must_use]
pub fn smooth_normalize(v: glam::DVec3, eps: Real) -> glam::DVec3 {
    let len = v.length();
    if len > eps { v / len } else { v / eps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_quadratic_inside_band() {
        let r = 0.5;
        let d = 1.0;
        // Inside the band: ½·r².
        assert!((norm_huber(r, d) - 0.125).abs() < 1e-12);
        // Derivative = r.
        assert!((norm_huber_grad(r, d) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn huber_linear_outside_band() {
        let r = 3.0;
        let d = 1.0;
        // Outside: δ·(|r| - δ/2) = 1·(3 - 0.5) = 2.5.
        assert!((norm_huber(r, d) - 2.5).abs() < 1e-12);
        // Derivative = ±δ.
        assert!((norm_huber_grad(r, d) - 1.0).abs() < 1e-12);
        assert!((norm_huber_grad(-r, d) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pseudo_huber_smooth_at_zero() {
        // ρ(0) = 0, ρ'(0) = 0.
        assert!(norm_pseudo_huber(0.0, 1.0).abs() < 1e-12);
        assert!(norm_pseudo_huber_grad(0.0, 1.0).abs() < 1e-12);
    }

    #[test]
    fn pseudo_huber_approaches_huber_for_small_r() {
        // For r/δ → 0, ρ_pH ≈ ½·r² = ρ_H.
        let r = 0.01;
        let d = 1.0;
        let pseudo = norm_pseudo_huber(r, d);
        let huber = norm_huber(r, d);
        assert!((pseudo - huber).abs() < 1e-6);
    }

    #[test]
    fn smooth_normalize_handles_near_zero() {
        let n = smooth_normalize(glam::DVec3::new(0.0, 0.0, 1e-12), 1e-9);
        // Should NOT divide by 1e-12; returns v / eps which is small.
        assert!(n.length() < 0.01);
    }

    #[test]
    fn l1_basic_values() {
        assert!((norm_l1(-2.0) - 2.0).abs() < 1e-12);
        assert!((norm_l1_grad(2.0) - 1.0).abs() < 1e-12);
        assert!((norm_l1_grad(-2.0) + 1.0).abs() < 1e-12);
        assert!(norm_l1_grad(0.0).abs() < 1e-12);
    }

    #[test]
    fn smooth_l2_reduces_to_l2_as_eps_shrinks() {
        // For |r| ≫ eps the loss tracks |r|−eps which → |r| as eps→0.
        let r = 0.5;
        for &eps in &[1e-2, 1e-4, 1e-8] {
            let v = norm_l2_smooth(r, eps);
            // Gradient also approaches sign(r).
            let g = norm_l2_smooth_grad(r, eps);
            assert!(v >= 0.0);
            assert!(g > 0.0 && g <= 1.0);
        }
        // With eps → 0, value → |r|.
        assert!((norm_l2_smooth(0.5, 1e-12) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tukey_rejects_outliers() {
        let c = 1.0;
        // Outside the band → constant + zero gradient.
        assert!((norm_tukey(2.0, c) - c * c / 6.0).abs() < 1e-12);
        assert!(norm_tukey_grad(2.0, c).abs() < 1e-12);
        assert!(norm_tukey_grad(-3.0, c).abs() < 1e-12);
        // At r = c, gradient is zero.
        assert!(norm_tukey_grad(c, c).abs() < 1e-12);
    }

    #[test]
    fn cauchy_zero_at_origin() {
        assert!(norm_cauchy(0.0, 1.0).abs() < 1e-12);
        assert!(norm_cauchy_grad(0.0, 1.0).abs() < 1e-12);
    }

    #[test]
    fn weight_l2_is_unit() {
        assert!((RobustLoss::L2.weight(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weight_tukey_drops_to_zero_outside_band() {
        let loss = RobustLoss::Tukey { c: 1.0 };
        assert!(loss.weight(2.0).abs() < 1e-12);
        // Inside the band, weight ≈ (1 - (r/c)²)².
        let r = 0.5;
        let expected = (1.0 - 0.25_f64).powi(2);
        assert!((loss.weight(r) - expected).abs() < 1e-12);
    }
}
