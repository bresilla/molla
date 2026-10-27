//! Reduced elastic shape-optimization checks.
//!
//! This is a CPU oracle for PLAN Tier 5.3. It optimizes a cantilever
//! beam's cross-section height from a target tip deflection using the
//! Euler-Bernoulli closed form. The full FEM adjoint tests cover
//! stiffness/load fitting; this gate adds a geometry/shape parameter
//! with a finite-difference gradient check and an Adam recovery loop.

use molla_math::Real;
use molla_optim::{AdamConfig, AdamState};

#[derive(Copy, Clone, Debug)]
struct RectCantilever {
    length: Real,
    width: Real,
    youngs: Real,
    tip_force: Real,
}

impl RectCantilever {
    fn second_moment(&self, log_height: Real) -> Real {
        let h = log_height.exp();
        self.width * h.powi(3) / 12.0
    }

    fn tip_deflection(&self, log_height: Real) -> Real {
        let i = self.second_moment(log_height);
        self.tip_force * self.length.powi(3) / (3.0 * self.youngs * i)
    }

    fn max_bending_stress(&self, log_height: Real) -> Real {
        let h = log_height.exp();
        // σ_max = M c / I, M = F L, c = h/2, I = b h³ / 12.
        6.0 * self.tip_force * self.length / (self.width * h * h)
    }

    fn loss_and_grad(&self, log_height: Real, target_deflection: Real) -> (Real, Real) {
        let deflection = self.tip_deflection(log_height);
        let residual = deflection - target_deflection;
        // δ ∝ h⁻³, h = exp(ξ) ⇒ dδ/dξ = -3δ.
        let d_deflection_d_log_height = -3.0 * deflection;
        (
            0.5 * residual * residual,
            residual * d_deflection_d_log_height,
        )
    }
}

fn beam() -> RectCantilever {
    RectCantilever {
        length: 2.0,
        width: 0.18,
        youngs: 8.0e6,
        tip_force: 35.0,
    }
}

#[test]
fn cantilever_shape_gradient_matches_fd() {
    let beam = beam();
    let true_log_h = 0.30_f64.ln();
    let log_h = 0.18_f64.ln();
    let target = beam.tip_deflection(true_log_h);
    let (loss, grad) = beam.loss_and_grad(log_h, target);
    let eps = 1.0e-6;
    let (loss_p, _) = beam.loss_and_grad(log_h + eps, target);
    let (loss_m, _) = beam.loss_and_grad(log_h - eps, target);
    let fd = (loss_p - loss_m) / (2.0 * eps);
    let diff = (grad - fd).abs();
    let rel = diff / fd.abs().max(1.0e-12);

    assert!(loss.is_finite() && loss > 0.0);
    assert!(beam.tip_deflection(log_h).is_finite());
    assert!(beam.max_bending_stress(log_h).is_finite());
    assert!(
        rel < 1.0e-8,
        "shape gradient mismatch: analytic={grad:.8e}, fd={fd:.8e}, diff={diff:.3e}, rel={rel:.3e}"
    );
}

#[test]
fn adam_recovers_cantilever_height_from_tip_deflection() {
    let beam = beam();
    let true_log_h = 0.34_f64.ln();
    let target = beam.tip_deflection(true_log_h);
    let mut log_h = 0.12_f64.ln();
    let log_min = 0.05_f64.ln();
    let log_max = 0.80_f64.ln();
    let (initial_loss, _) = beam.loss_and_grad(log_h, target);
    let mut best_loss = initial_loss;
    let mut best_log_h = log_h;
    let mut adam = AdamState::new(1);
    let cfg = AdamConfig {
        lr: 0.06,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    for _ in 0..700 {
        let (loss, grad) = beam.loss_and_grad(log_h, target);
        if loss < best_loss {
            best_loss = loss;
            best_log_h = log_h;
        }
        let mut params = [log_h];
        adam.step(&mut params, &[grad], &cfg);
        log_h = params[0].clamp(log_min, log_max);
    }

    let recovered_h = best_log_h.exp();
    let true_h = true_log_h.exp();
    let rel_err = (recovered_h - true_h).abs() / true_h;
    let reduction = best_loss / initial_loss.max(1.0e-30);
    let recovered_stress = beam.max_bending_stress(best_log_h);
    eprintln!(
        "cantilever shape fit: true_h={true_h:.5}, recovered_h={recovered_h:.5}, \
         rel_err={:.3}%, target_deflection={target:.5e}, stress={recovered_stress:.5e}, \
         initial_loss={initial_loss:.3e}, best_loss={best_loss:.3e}, reduction={reduction:.3e}",
        rel_err * 100.0
    );

    assert!(recovered_h.is_finite() && recovered_h > 0.0);
    assert!(recovered_stress.is_finite() && recovered_stress > 0.0);
    assert!(
        rel_err < 0.02,
        "Adam did not recover cantilever height: rel_err={rel_err:.4}"
    );
    assert!(
        reduction < 1.0e-5,
        "shape objective did not decrease enough: ratio={reduction:.3e}"
    );
}
