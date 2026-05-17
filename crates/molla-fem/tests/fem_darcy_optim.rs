//! Reduced Darcy/level-set optimization checks.
//!
//! This is a deliberately small CPU oracle for the Tier 5.3
//! "level-set/Darcy-like optimization" item. A 1-D layered Darcy
//! column uses a signed `phi` layout to split fixed high-permeability
//! channel cells from a tunable matrix material. We optimize the
//! matrix log-permeability from a target steady flux, checking both
//! the analytic sensitivity and the optimizer loop.

use molla_math::Real;
use molla_optim::{AdamConfig, AdamState};

#[derive(Clone, Debug)]
struct LayeredDarcyColumn {
    dx: Vec<Real>,
    phi: Vec<Real>,
    area: Real,
    delta_pressure: Real,
    channel_permeability: Real,
}

#[derive(Clone, Debug)]
struct DarcyEval {
    flux: Real,
    pressure_drops: Vec<Real>,
    resistance: Real,
    matrix_resistance: Real,
}

impl LayeredDarcyColumn {
    fn permeability(&self, i: usize, log_matrix_permeability: Real) -> Real {
        if self.phi[i] <= 0.0 {
            self.channel_permeability
        } else {
            log_matrix_permeability.exp()
        }
    }

    fn eval(&self, log_matrix_permeability: Real) -> DarcyEval {
        assert_eq!(self.dx.len(), self.phi.len());
        let mut resistance = 0.0;
        let mut matrix_resistance = 0.0;
        for i in 0..self.dx.len() {
            let k = self.permeability(i, log_matrix_permeability);
            let r_i = self.dx[i] / (self.area * k);
            resistance += r_i;
            if self.phi[i] > 0.0 {
                matrix_resistance += r_i;
            }
        }
        let flux = self.delta_pressure / resistance;
        let pressure_drops = (0..self.dx.len())
            .map(|i| {
                let k = self.permeability(i, log_matrix_permeability);
                flux * self.dx[i] / (self.area * k)
            })
            .collect();
        DarcyEval {
            flux,
            pressure_drops,
            resistance,
            matrix_resistance,
        }
    }

    fn loss_and_grad(&self, log_matrix_permeability: Real, target_flux: Real) -> (Real, Real) {
        let e = self.eval(log_matrix_permeability);
        let residual = e.flux - target_flux;
        // R = Σ dx/(A k). For matrix cells, k = exp(ξ), so
        // dR/dξ = -R_matrix and dq/dξ = Δp * R_matrix / R².
        let dflux_dlog_matrix =
            self.delta_pressure * e.matrix_resistance / (e.resistance * e.resistance);
        (0.5 * residual * residual, residual * dflux_dlog_matrix)
    }
}

fn test_column() -> LayeredDarcyColumn {
    LayeredDarcyColumn {
        dx: vec![0.10, 0.15, 0.10, 0.20, 0.15, 0.10],
        // Negative phi = channel inclusion; positive phi = matrix.
        phi: vec![1.0, -0.5, -0.25, 1.0, 1.0, -0.1],
        area: 0.75,
        delta_pressure: 4.0,
        channel_permeability: 3.0,
    }
}

#[test]
fn darcy_level_set_flux_conserves_pressure_drop_and_matches_fd_grad() {
    let column = test_column();
    let log_k = 0.18_f64.ln();
    let target_flux = column.eval(0.42_f64.ln()).flux;
    let (loss, grad) = column.loss_and_grad(log_k, target_flux);
    let eps = 1.0e-6;
    let (loss_p, _) = column.loss_and_grad(log_k + eps, target_flux);
    let (loss_m, _) = column.loss_and_grad(log_k - eps, target_flux);
    let fd = (loss_p - loss_m) / (2.0 * eps);
    let diff = (grad - fd).abs();

    let e = column.eval(log_k);
    let pressure_sum: Real = e.pressure_drops.iter().sum();
    assert!(e.flux.is_finite() && e.flux > 0.0);
    assert!(loss.is_finite() && loss > 0.0);
    assert!(
        (pressure_sum - column.delta_pressure).abs() < 1.0e-12,
        "Darcy pressure drops must telescope to Δp: sum={pressure_sum}, Δp={}",
        column.delta_pressure
    );
    assert!(
        diff < 1.0e-8,
        "Darcy log-permeability gradient mismatch: analytic={grad:.8e}, fd={fd:.8e}, diff={diff:.3e}"
    );
}

#[test]
fn adam_recovers_matrix_permeability_from_darcy_flux_target() {
    let column = test_column();
    let true_log_k = 0.36_f64.ln();
    let target_flux = column.eval(true_log_k).flux;
    let mut log_k = 0.04_f64.ln();
    let (initial_loss, _) = column.loss_and_grad(log_k, target_flux);
    let mut best_loss = initial_loss;
    let mut best_log_k = log_k;
    let mut adam = AdamState::new(1);
    let cfg = AdamConfig {
        lr: 0.08,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    for _ in 0..260 {
        let (loss, grad) = column.loss_and_grad(log_k, target_flux);
        if loss < best_loss {
            best_loss = loss;
            best_log_k = log_k;
        }
        let mut params = [log_k];
        adam.step(&mut params, &[grad], &cfg);
        log_k = params[0];
    }

    let recovered_k = best_log_k.exp();
    let true_k = true_log_k.exp();
    let rel_err = (recovered_k - true_k).abs() / true_k;
    let reduction = best_loss / initial_loss.max(1.0e-30);
    eprintln!(
        "Darcy permeability fit: true_k={true_k:.5}, recovered_k={recovered_k:.5}, \
         rel_err={:.3}%, initial_loss={initial_loss:.3e}, best_loss={best_loss:.3e}, \
         reduction={reduction:.3e}",
        rel_err * 100.0
    );

    assert!(
        rel_err < 0.02,
        "Adam did not recover matrix permeability: rel_err={rel_err:.4}"
    );
    assert!(
        reduction < 1.0e-5,
        "Darcy objective did not decrease enough: ratio={reduction:.3e}"
    );
}
