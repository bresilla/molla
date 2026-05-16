//! Adam — adaptive-moment first-order optimizer (CPU host path).
//!
//! Mirrors `xtra/warp/warp/_src/optim/adam.py::adam_step_kernel_float`
//! but in pure Rust on f64. Optional decoupled weight decay
//! (AdamW-style) is applied AFTER the standard Adam update — this
//! matches Loshchilov & Hutter (2019).

use molla_math::Real;

/// Hyper-parameters for [`AdamState::step`].
#[derive(Copy, Clone, Debug)]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: Real,
    /// Exponential decay rate for the first moment estimate.
    pub beta1: Real,
    /// Exponential decay rate for the second moment estimate.
    pub beta2: Real,
    /// Numerical stability denominator. Added to `√v̂` before the
    /// division.
    pub eps: Real,
    /// Decoupled weight-decay coefficient (AdamW). `0.0` disables it.
    pub weight_decay: Real,
}

impl Default for AdamConfig {
    /// Standard Kingma & Ba defaults (lr 1e-3, β₁ 0.9, β₂ 0.999,
    /// ε 1e-8, no weight decay).
    fn default() -> Self {
        Self {
            lr: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        }
    }
}

/// Persistent per-parameter state for Adam: first moment `m`, second
/// moment `v`, and the global step counter `step` (1-based for the
/// bias-correction factors; `step == 0` means "no updates yet").
#[derive(Clone, Debug)]
pub struct AdamState {
    /// First-moment EMA — `‖m‖ = ‖params‖`.
    pub m: Vec<Real>,
    /// Second-moment EMA — `‖v‖ = ‖params‖`.
    pub v: Vec<Real>,
    /// Number of updates applied so far (1-based for bias correction).
    pub step: u32,
}

impl AdamState {
    /// Allocate zero-initialised moment buffers for `n_params`
    /// parameters and set the step counter to zero.
    pub fn new(n_params: usize) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            step: 0,
        }
    }

    /// Re-initialise the optimizer state — zero moments + step.
    /// Used by warm-starts and parameter-vector re-shapes.
    pub fn reset(&mut self) {
        for m in &mut self.m {
            *m = 0.0;
        }
        for v in &mut self.v {
            *v = 0.0;
        }
        self.step = 0;
    }

    /// Apply one Adam update to `params` using `grads`. All three
    /// slices must have the same length, and that length must match
    /// the buffers allocated by [`Self::new`]; otherwise the function
    /// panics.
    ///
    /// The update is the standard Kingma & Ba formula with optional
    /// decoupled weight decay applied after the moment-correction
    /// step (AdamW). When `weight_decay == 0.0` the path is
    /// bit-identical to plain Adam.
    pub fn step(&mut self, params: &mut [Real], grads: &[Real], cfg: &AdamConfig) {
        debug_assert_eq!(params.len(), grads.len(), "params/grads length mismatch");
        debug_assert_eq!(params.len(), self.m.len(), "params/state length mismatch");
        debug_assert_eq!(params.len(), self.v.len(), "params/state length mismatch");

        self.step = self.step.saturating_add(1);
        let bc1 = 1.0 - cfg.beta1.powi(self.step as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.step as i32);
        // Guard against numerical edge cases (β = 0 or β = 1) — the
        // bias-correction divisor goes to 0 only if β₁ or β₂ are 1,
        // which is degenerate; we still want a finite update so we
        // clamp to ε.
        let bc1 = if bc1.abs() < Real::EPSILON {
            Real::EPSILON
        } else {
            bc1
        };
        let bc2 = if bc2.abs() < Real::EPSILON {
            Real::EPSILON
        } else {
            bc2
        };

        for i in 0..params.len() {
            let g = grads[i];
            let m_new = cfg.beta1 * self.m[i] + (1.0 - cfg.beta1) * g;
            let v_new = cfg.beta2 * self.v[i] + (1.0 - cfg.beta2) * g * g;
            self.m[i] = m_new;
            self.v[i] = v_new;
            let m_hat = m_new / bc1;
            let v_hat = v_new / bc2;
            let mut theta = params[i] - cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
            if cfg.weight_decay != 0.0 {
                // Decoupled weight decay (AdamW): apply on θ_{t-1}.
                // Using the pre-update θ is what makes "decoupled" meaningful — the wd term doesn't
                // interact with the moment estimates.
                theta -= cfg.lr * cfg.weight_decay * params[i];
            }
            params[i] = theta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_zero_grad_zero_update() {
        // No gradient => params unchanged (when wd is 0).
        let mut p = vec![1.0_f64, -2.0, 3.0];
        let g = vec![0.0_f64; 3];
        let mut s = AdamState::new(3);
        let cfg = AdamConfig::default();
        s.step(&mut p, &g, &cfg);
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!((p[1] - (-2.0)).abs() < 1e-12);
        assert!((p[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn adam_sign_of_update_opposes_grad() {
        // Positive gradient should decrease the parameter.
        let mut p = vec![1.0_f64];
        let g = vec![1.0_f64];
        let mut s = AdamState::new(1);
        let cfg = AdamConfig {
            lr: 0.1,
            ..AdamConfig::default()
        };
        s.step(&mut p, &g, &cfg);
        assert!(p[0] < 1.0);
        // First step magnitude is ~lr (m_hat / sqrt(v_hat) ≈ 1).
        assert!((p[0] - 0.9).abs() < 1e-6);
    }
}
