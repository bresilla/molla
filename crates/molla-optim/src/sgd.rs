//! SGD — stochastic gradient descent with momentum, Nesterov, and
//! weight decay (CPU host path).
//!
//! Mirrors `xtra/warp/warp/_src/optim/sgd.py::sgd_step_kernel`.

use molla_math::Real;

/// Hyper-parameters for [`SgdState::step`].
#[derive(Copy, Clone, Debug)]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: Real,
    /// Momentum factor (`μ`). `0.0` disables momentum (plain SGD).
    pub momentum: Real,
    /// Dampening factor applied to the gradient term when updating
    /// the momentum buffer. `0.0` is the standard heavy-ball form.
    pub dampening: Real,
    /// L2 weight-decay coefficient. `0.0` disables it.
    pub weight_decay: Real,
    /// Whether to apply Nesterov accelerated gradient. Requires
    /// `momentum > 0` to have any effect.
    pub nesterov: bool,
}

impl Default for SgdConfig {
    /// Plain SGD, no momentum, lr 1e-3.
    fn default() -> Self {
        Self {
            lr: 1.0e-3,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

/// Persistent per-parameter state for SGD: the momentum buffer `buf`
/// (allocated even when `momentum == 0` for type symmetry) and a step
/// counter (used to distinguish the first step from subsequent steps
/// — on step 0 the buffer is just the gradient, not `μ·b + (1−damp)·g`).
#[derive(Clone, Debug)]
pub struct SgdState {
    /// Momentum buffer — same length as the parameter vector.
    pub buf: Vec<Real>,
    /// Step counter. Used to detect the first call so the momentum
    /// initialisation rule (`b_0 = g₀` rather than `μ·0 + (1−d)·g₀`)
    /// can fire.
    pub step: u32,
}

impl SgdState {
    /// Allocate a zero-initialised momentum buffer for `n_params`.
    pub fn new(n_params: usize) -> Self {
        Self {
            buf: vec![0.0; n_params],
            step: 0,
        }
    }

    /// Reset the momentum buffer + step counter.
    pub fn reset(&mut self) {
        for b in &mut self.buf {
            *b = 0.0;
        }
        self.step = 0;
    }

    /// Apply one SGD update to `params` using `grads`. Slice lengths
    /// must match the buffers allocated by [`Self::new`].
    pub fn step(&mut self, params: &mut [Real], grads: &[Real], cfg: &SgdConfig) {
        debug_assert_eq!(params.len(), grads.len(), "params/grads length mismatch");
        debug_assert_eq!(params.len(), self.buf.len(), "params/state length mismatch");

        let first_step = self.step == 0;
        for i in 0..params.len() {
            let mut g = grads[i];
            if cfg.weight_decay != 0.0 {
                g += cfg.weight_decay * params[i];
            }
            let g_eff = if cfg.momentum != 0.0 {
                let b_new = if first_step {
                    g
                } else {
                    cfg.momentum * self.buf[i] + (1.0 - cfg.dampening) * g
                };
                self.buf[i] = b_new;
                if cfg.nesterov {
                    g + cfg.momentum * b_new
                } else {
                    b_new
                }
            } else {
                g
            };
            params[i] -= cfg.lr * g_eff;
        }
        self.step = self.step.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_plain_descent() {
        // No momentum, no wd: θ -= lr · g.
        let mut p = vec![1.0_f64];
        let g = vec![2.0_f64];
        let mut s = SgdState::new(1);
        let cfg = SgdConfig {
            lr: 0.1,
            ..SgdConfig::default()
        };
        s.step(&mut p, &g, &cfg);
        assert!((p[0] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn sgd_momentum_accumulates() {
        // With μ = 0.9, constant gradient → buffer grows toward
        // g / (1 - μ) = 10·g.
        let mut p = vec![0.0_f64];
        let g = vec![1.0_f64];
        let mut s = SgdState::new(1);
        let cfg = SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..SgdConfig::default()
        };
        for _ in 0..50 {
            s.step(&mut p, &g, &cfg);
        }
        // Should overshoot a single-step update many times over.
        assert!(p[0] < -1.0);
    }
}
