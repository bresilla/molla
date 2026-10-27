//! Stage 24b — WGSL parity tests for the robust-loss kernel.
//!
//! All `#[ignore]` because they need a real wgpu adapter (CI doesn't
//! have one); run with
//! `cargo test -p molla --features gpu --test robust_loss_gpu -- --ignored`.
//!
//! The shader runs in `f32`; the parity tolerance budget is 1e-5
//! (matches Stage 24a's `dense3_gpu` parity tests). For each loss we
//! run 1024 random samples and compare the per-sample `(ρ, ρ')`
//! pair against the CPU reference.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::RobustLossKernel;
use molla_math::Real;
use molla_math::robust_loss::{
    RobustLoss, norm_cauchy, norm_cauchy_grad, norm_huber, norm_huber_grad, norm_l1, norm_l1_grad,
    norm_l2_smooth, norm_l2_smooth_grad, norm_pseudo_huber, norm_pseudo_huber_grad, norm_tukey,
    norm_tukey_grad,
};

const BATCH: usize = 1024;
const PARITY_TOL: Real = 1.0e-5;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn signed(&mut self) -> Real {
        let x = self.next() >> 11;
        let u = (x as Real + 1.0) / ((1u64 << 53) as Real + 1.0);
        2.0 * u - 1.0
    }
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_residuals(seed: u64, range: Real) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..BATCH).map(|_| (range * rng.signed()) as f32).collect()
}

fn assert_parity<F, G>(name: &str, loss: RobustLoss, cpu_rho: F, cpu_grad: G, residuals: &[f32])
where
    F: Fn(Real) -> Real,
    G: Fn(Real) -> Real,
{
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = RobustLossKernel::new(&device);
    let pairs = kernel
        .dispatch_blocking(&device, &queue, loss, residuals)
        .expect("dispatch_blocking");
    assert_eq!(pairs.len(), residuals.len());

    let mut max_rho_err: Real = 0.0;
    let mut max_grad_err: Real = 0.0;
    for (i, p) in pairs.iter().enumerate() {
        let r = residuals[i] as Real;
        let cpu_v = cpu_rho(r);
        let cpu_g = cpu_grad(r);
        let rho_err = ((p.value as Real) - cpu_v).abs();
        let grad_err = ((p.grad as Real) - cpu_g).abs();
        if rho_err > max_rho_err {
            max_rho_err = rho_err;
        }
        if grad_err > max_grad_err {
            max_grad_err = grad_err;
        }
    }
    eprintln!("{name}: max ρ err = {max_rho_err:e}, max ρ' err = {max_grad_err:e}",);
    // Tolerance budget = 1e-5 (relative to the range used). Add a
    // slop term scaled by the input magnitude to absorb f32
    // round-off on losses with `r²` terms.
    let max_input = residuals.iter().fold(0.0_f32, |a, b| a.max(b.abs())) as Real;
    let abs_tol = PARITY_TOL * (1.0 + max_input * max_input);
    assert!(
        max_rho_err < abs_tol,
        "{name}: ρ err {max_rho_err} > {abs_tol}"
    );
    assert!(
        max_grad_err < abs_tol,
        "{name}: ρ' err {max_grad_err} > {abs_tol}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn l1_gpu_matches_cpu() {
    let residuals = sample_residuals(0x1111_2222, 1.0);
    assert_parity("l1", RobustLoss::L1, norm_l1, norm_l1_grad, &residuals);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn smooth_l2_gpu_matches_cpu() {
    let eps = 0.05;
    let residuals = sample_residuals(0x3333_4444, 1.0);
    assert_parity(
        "smooth_l2",
        RobustLoss::SmoothL2 { eps },
        |r| norm_l2_smooth(r, eps),
        |r| norm_l2_smooth_grad(r, eps),
        &residuals,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn huber_gpu_matches_cpu() {
    let delta = 0.5;
    let residuals = sample_residuals(0x5555_6666, 1.5);
    assert_parity(
        "huber",
        RobustLoss::Huber { delta },
        |r| norm_huber(r, delta),
        |r| norm_huber_grad(r, delta),
        &residuals,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn pseudo_huber_gpu_matches_cpu() {
    let delta = 0.5;
    let residuals = sample_residuals(0x7777_8888, 2.0);
    assert_parity(
        "pseudo_huber",
        RobustLoss::PseudoHuber { delta },
        |r| norm_pseudo_huber(r, delta),
        |r| norm_pseudo_huber_grad(r, delta),
        &residuals,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tukey_gpu_matches_cpu() {
    let c = 1.0;
    let residuals = sample_residuals(0x9999_AAAA, 2.0);
    assert_parity(
        "tukey",
        RobustLoss::Tukey { c },
        |r| norm_tukey(r, c),
        |r| norm_tukey_grad(r, c),
        &residuals,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn cauchy_gpu_matches_cpu() {
    let c = 1.0;
    let residuals = sample_residuals(0xBBBB_CCCC, 2.0);
    assert_parity(
        "cauchy",
        RobustLoss::Cauchy { c },
        |r| norm_cauchy(r, c),
        |r| norm_cauchy_grad(r, c),
        &residuals,
    );
}
