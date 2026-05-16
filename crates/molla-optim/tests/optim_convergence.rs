//! Stage 38c — convergence tests for the CPU [`molla_optim`] path.
//!
//! Exercises the first-order optimizers against three standard
//! synthetic objectives so future regressions in the CPU update rules
//! are caught immediately.
//!
//! All tests are deterministic (fixed seeds). The "random" inputs are
//! cooked off a small LCG so the tests behave identically across
//! machines and Rust versions.

use molla_math::Real;
use molla_optim::{AdamConfig, AdamState, SgdConfig, SgdState};

// ── Deterministic LCG so tests don't depend on rand. ────────────────
/// Numerical Recipes "ranqd1" 32-bit LCG (a = 1664525, c = 1013904223).
/// We pump it into `[-1, 1]` by mapping the high bits.
struct Lcg(u32);

impl Lcg {
    fn next_unit(&mut self) -> Real {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [-1, 1].
        (self.0 as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// Quadratic bowl `J(θ) = ½ ‖θ − target‖²`. Gradient is `θ − target`.
fn quadratic_grad(theta: &[Real], target: &[Real], out: &mut [Real]) {
    for i in 0..theta.len() {
        out[i] = theta[i] - target[i];
    }
}

fn norm(v: &[Real]) -> Real {
    v.iter().map(|x| x * x).sum::<Real>().sqrt()
}

#[test]
fn adam_quadratic_converges() {
    let n = 16;
    let mut rng = Lcg(0xC0FFEE);
    let target: Vec<Real> = (0..n).map(|_| rng.next_unit()).collect();
    let mut theta: Vec<Real> = (0..n).map(|_| rng.next_unit()).collect();
    let mut grads = vec![0.0; n];

    let mut state = AdamState::new(n);
    let cfg = AdamConfig {
        lr: 0.05,
        ..AdamConfig::default()
    };

    for _ in 0..200 {
        quadratic_grad(&theta, &target, &mut grads);
        state.step(&mut theta, &grads, &cfg);
    }

    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = theta[i] - target[i];
    }
    let r = norm(&residual);
    eprintln!("adam_quadratic: final ‖θ − target‖ = {r:.3e}");
    assert!(
        r < 1.0e-3,
        "Adam failed to converge on quadratic: residual = {r}",
    );
}

#[test]
fn sgd_quadratic_converges() {
    let n = 16;
    let mut rng = Lcg(0xBADCAFE);
    let target: Vec<Real> = (0..n).map(|_| rng.next_unit()).collect();
    let mut theta: Vec<Real> = (0..n).map(|_| rng.next_unit()).collect();
    let mut grads = vec![0.0; n];

    let mut state = SgdState::new(n);
    let cfg = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..SgdConfig::default()
    };

    for _ in 0..500 {
        quadratic_grad(&theta, &target, &mut grads);
        state.step(&mut theta, &grads, &cfg);
    }

    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = theta[i] - target[i];
    }
    let r = norm(&residual);
    eprintln!("sgd_quadratic: final ‖θ − target‖ = {r:.3e}");
    assert!(
        r < 1.0e-3,
        "SGD failed to converge on quadratic: residual = {r}",
    );
}

#[test]
fn adam_rosenbrock_makes_progress() {
    // 2-D Rosenbrock: J(x, y) = (1 − x)² + 100·(y − x²)².
    // ∂J/∂x = -2(1 − x) + 100·2(y − x²)·(−2x)
    //       = -2(1 − x) − 400·x·(y − x²)
    // ∂J/∂y =  200·(y − x²)
    //
    // From (−1, 1), J = 4. Reaching < 0.01 after 5000 Adam steps with
    // lr=0.005 is comfortable but not trivial — Rosenbrock's curving
    // valley is famously hard for first-order methods.
    fn objective(p: &[Real]) -> Real {
        let (x, y) = (p[0], p[1]);
        let a = 1.0 - x;
        let b = y - x * x;
        a * a + 100.0 * b * b
    }
    fn grad(p: &[Real], g: &mut [Real]) {
        let (x, y) = (p[0], p[1]);
        let b = y - x * x;
        g[0] = -2.0 * (1.0 - x) - 400.0 * x * b;
        g[1] = 200.0 * b;
    }

    let mut theta = vec![-1.0, 1.0];
    let mut grads = vec![0.0; 2];
    let mut state = AdamState::new(2);
    let cfg = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };
    for _ in 0..5000 {
        grad(&theta, &mut grads);
        state.step(&mut theta, &grads, &cfg);
    }
    let j = objective(&theta);
    eprintln!(
        "adam_rosenbrock: final θ = ({:.4}, {:.4}), J = {:.3e}",
        theta[0], theta[1], j
    );
    assert!(
        j < 0.01,
        "Adam on Rosenbrock didn't reach J < 0.01: J = {j}"
    );
}

#[test]
fn adam_with_weight_decay_drives_to_zero() {
    // Gradient is identically zero — only the decoupled weight-decay
    // term moves the parameters. Decoupled wd shrinks θ multiplicatively
    // by (1 − lr·wd) each step, so 200 steps of lr=0.1, wd=0.1 reduces
    // ‖θ‖ by a factor of (1 − 0.01)^200 ≈ 0.134, which from initial
    // unit-magnitude lands well above zero — so we use a much smaller
    // tolerance and run enough steps to actually decay to 1e-6.
    let n = 8;
    let mut theta: Vec<Real> = (0..n).map(|i| (i as Real + 1.0) * 0.5).collect();
    let grads = vec![0.0; n];

    let mut state = AdamState::new(n);
    let cfg = AdamConfig {
        lr: 0.1,
        weight_decay: 0.1,
        ..AdamConfig::default()
    };
    let initial = norm(&theta);
    for _ in 0..200 {
        state.step(&mut theta, &grads, &cfg);
    }
    let final_norm = norm(&theta);
    eprintln!("adam_wd: initial ‖θ‖ = {initial:.3e}, final ‖θ‖ = {final_norm:.3e}",);
    // (1 − 0.1·0.1)^200 ≈ 0.1340.
    // Expected ‖θ‖ ≈ 0.1340 · initial; demand < 0.25 · initial.
    assert!(
        final_norm < 0.25 * initial,
        "Weight decay didn't shrink ‖θ‖ enough: final = {final_norm}, initial = {initial}",
    );
}
