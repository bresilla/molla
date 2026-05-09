//! Robust-loss math primitives used by IK regularisation. All
//! functions are scalar; vectorise at the call site.
//!
//! Mirrors Newton's `_src/math/robust_loss.py` API.

use crate::math::Real;

/// Huber loss `ρ(r)`: quadratic when `|r| ≤ δ`, linear past that.
/// Smooth at the transition (C¹).
///
///     ρ(r) = ½ r²              if |r| ≤ δ
///          = δ·(|r| - δ/2)     if |r| > δ
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
    if abs <= delta {
        r
    } else {
        delta * r.signum()
    }
}

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

/// Smooth-norm normalisation: returns `v / max(|v|, eps)` so the
/// gradient is well-defined at `v = 0`. Useful for direction
/// vectors in IK Jacobian columns when the residual hits the target.
#[must_use]
pub fn smooth_normalize(v: glam::DVec3, eps: Real) -> glam::DVec3 {
    let len = v.length();
    if len > eps {
        v / len
    } else {
        v / eps
    }
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
}
