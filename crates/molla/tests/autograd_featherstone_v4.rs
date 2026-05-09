//! Stage 23.1.4 — pure-analytic ∂h/∂q (CPU) verification + perf
//! comparison vs the central-FD path.
//!
//! Three live CPU tests (no `#[ignore]`):
//! 1. `analytic_full_matches_central_fd_pendulum` — gradient parity
//!    on the single-pendulum chain.
//! 2. `analytic_full_matches_central_fd_franka6` — gradient parity
//!    on the 6-DOF Franka chain.
//! 3. `analytic_full_beats_central_fd_in_wallclock` — wall-clock
//!    `chain.reverse(AnalyticFull)` < `chain.reverse(CentralFd)`
//!    on the 6-DOF Franka. Logs the actual bwd/fwd ratio for the
//!    AnalyticFull path even if it lands above the < 5× DoD.

use molla::autograd::featherstone_adjoint::JacobianStrategy;
use molla::autograd::{FeatherstoneRevoluteChain, JacobianScratch};
use molla::math::{Real, Vec3};

const GRAD_TOL: Real = 1e-3;

fn max_grad_diff(a: &[Vec<Real>], b: &[Vec<Real>]) -> Real {
    let mut m = 0.0_f64;
    for t in 0..a.len() {
        for k in 0..a[t].len() {
            let d = (a[t][k] - b[t][k]).abs();
            if d > m {
                m = d;
            }
        }
    }
    m
}

#[test]
fn analytic_full_matches_central_fd_pendulum() {
    let chain = FeatherstoneRevoluteChain::pendulum(0.5, 1.0, 0.01, 8);
    let q0 = vec![0.2_f64];
    let qd0 = vec![-0.1_f64];
    let tau0 = vec![vec![0.05_f64]; chain.horizon];
    let trace = chain.rollout(&q0, &qd0, &tau0);
    let n = chain.links.len();

    let mut s_fd = JacobianScratch::new(n);
    let mut s_af = JacobianScratch::new(n);
    let g_fd = chain.reverse_with_strategy(&trace, Vec3::Y, &mut s_fd, JacobianStrategy::CentralDifference);
    let g_af = chain.reverse_with_strategy(&trace, Vec3::Y, &mut s_af, JacobianStrategy::AnalyticFull);
    let d = max_grad_diff(&g_fd, &g_af);
    println!("pendulum AnalyticFull vs CentralFd: max diff = {d:.3e}");
    assert!(
        d < GRAD_TOL,
        "AnalyticFull diverged from CentralFd: max diff = {d:.3e}",
    );
}

#[test]
fn analytic_full_matches_central_fd_franka6() {
    let chain = FeatherstoneRevoluteChain::franka6(0.02, 5);
    let n = chain.links.len();
    let q0 = vec![0.1_f64, -0.2, 0.15, -0.25, 0.05, 0.1];
    let qd0 = vec![0.0_f64; n];
    let tau0 = vec![vec![0.02_f64; n]; chain.horizon];
    let trace = chain.rollout(&q0, &qd0, &tau0);

    let mut s_fd = JacobianScratch::new(n);
    let mut s_af = JacobianScratch::new(n);
    let g_fd = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_fd, JacobianStrategy::CentralDifference);
    let g_af = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_af, JacobianStrategy::AnalyticFull);
    let d_z = max_grad_diff(&g_fd, &g_af);
    let g_fd_y = chain.reverse_with_strategy(&trace, Vec3::Y, &mut s_fd, JacobianStrategy::CentralDifference);
    let g_af_y = chain.reverse_with_strategy(&trace, Vec3::Y, &mut s_af, JacobianStrategy::AnalyticFull);
    let d_y = max_grad_diff(&g_fd_y, &g_af_y);
    println!(
        "franka6 AnalyticFull vs CentralFd: max diff z = {d_z:.3e}, y = {d_y:.3e}"
    );
    assert!(
        d_z < GRAD_TOL && d_y < GRAD_TOL,
        "AnalyticFull diverged from CentralFd: d_z={d_z:.3e}, d_y={d_y:.3e}",
    );
}

#[test]
fn analytic_full_beats_central_fd_in_wallclock() {
    // Franka6 with a long enough horizon so per-step Jacobian build
    // dominates the trajectory adjoint loop.
    let chain = FeatherstoneRevoluteChain::franka6(0.005, 50);
    let n = chain.links.len();
    let q0 = vec![0.1_f64, -0.2, 0.15, -0.25, 0.05, 0.1];
    let qd0 = vec![0.0_f64; n];
    let tau0 = vec![vec![0.02_f64; n]; chain.horizon];

    // Time the forward pass for the bwd/fwd ratio readout.
    let warm = chain.rollout(&q0, &qd0, &tau0);
    let _ = warm;

    // Median of N samples — single-shot timing is too noisy.
    let n_samples = 9_usize;
    let mut fwd_us = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let t = std::time::Instant::now();
        let _ = chain.rollout(&q0, &qd0, &tau0);
        fwd_us.push(t.elapsed().as_micros() as f64);
    }
    fwd_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fwd_med = fwd_us[n_samples / 2];

    let trace = chain.rollout(&q0, &qd0, &tau0);

    // Warm-up reverse passes (allocator cache, etc).
    let mut s_warm = JacobianScratch::new(n);
    let _ = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_warm, JacobianStrategy::CentralDifference);
    let _ = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_warm, JacobianStrategy::AnalyticFull);

    let mut fd_us = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut s = JacobianScratch::new(n);
        let t = std::time::Instant::now();
        let _ = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s, JacobianStrategy::CentralDifference);
        fd_us.push(t.elapsed().as_micros() as f64);
    }
    fd_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fd_med = fd_us[n_samples / 2];

    let mut af_us = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut s = JacobianScratch::new(n);
        let t = std::time::Instant::now();
        let _ = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s, JacobianStrategy::AnalyticFull);
        af_us.push(t.elapsed().as_micros() as f64);
    }
    af_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let af_med = af_us[n_samples / 2];

    let bwd_fwd = af_med / fwd_med;
    println!(
        "franka6 reverse: forward = {fwd_med:.1} us, CentralFd = {fd_med:.1} us, \
         AnalyticFull = {af_med:.1} us → AnalyticFull bwd/fwd = {bwd_fwd:.2}×"
    );
    assert!(
        af_med < fd_med,
        "AnalyticFull ({af_med:.1} us) should beat CentralFd ({fd_med:.1} us) on franka6"
    );
}
