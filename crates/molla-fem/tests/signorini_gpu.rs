//! GPU batched rate-and-state friction vs the CPU `advance_rate_state_friction`.
//!
//! Run with: `make test-gpu P=molla-fem T=signorini_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::signorini::{
    RateStateFrictionParams, RateStateFrictionState, advance_rate_state_friction,
};
use molla_fem::signorini_gpu::advance_rate_state_friction_batch_gpu;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rate_state_friction_matches_cpu() {
    let params = RateStateFrictionParams::default();
    let dt = 1.0e-4;

    // A spread of slip rates and states across many contacts.
    let n = 500usize;
    let slip: Vec<f64> = (0..n).map(|i| 1.0e-6 * (1.0 + i as f64 * 0.03)).collect();
    let states: Vec<RateStateFrictionState> = (0..n)
        .map(|i| RateStateFrictionState {
            theta: 1.0e-3 * (1.0 + (i as f64 * 0.02).sin().abs()),
        })
        .collect();

    let cpu: Vec<(RateStateFrictionState, f64)> = states
        .iter()
        .zip(slip.iter())
        .map(|(s, v)| advance_rate_state_friction(params, *s, *v, dt))
        .collect();

    let engine = Engine::new_gpu().expect("gpu");
    let (gpu_states, gpu_mu) =
        advance_rate_state_friction_batch_gpu(&engine, params, &states, &slip, dt).expect("gpu");

    assert_eq!(gpu_states.len(), n);
    for (i, ((cs, cmu), (gs, gmu))) in cpu
        .iter()
        .zip(gpu_states.iter().zip(gpu_mu.iter()))
        .enumerate()
    {
        let td = (cs.theta - gs.theta).abs() / cs.theta.abs().max(1.0e-9);
        assert!(td < 1.0e-4, "contact {i}: theta rel-err {td}");
        assert!(
            (cmu - gmu).abs() < 1.0e-4,
            "contact {i}: mu err {}",
            (cmu - gmu).abs()
        );
    }
    // Non-degenerate: coefficients vary across the batch.
    let (lo, hi) = gpu_mu
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &m| (lo.min(m), hi.max(m)));
    assert!(
        hi - lo > 1.0e-3,
        "degenerate test: μ constant across contacts"
    );
}
