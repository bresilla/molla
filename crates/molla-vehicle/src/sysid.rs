//! V6 — differentiable vehicle demo: parameter identification.
//!
//! Recover an unknown physical parameter — here the tyre-ground
//! friction coefficient — from an observed vehicle trajectory. The
//! rollout `μ → distance driven` is a deterministic scalar map; its
//! sensitivity is taken by finite differences through the full vehicle
//! rollout, and a Gauss-Newton optimiser drives the trajectory loss to
//! zero, recovering the planted friction.
//!
//! This is the differentiable-simulation showcase: system
//! identification from a trajectory — estimating what the ground was
//! like from how the vehicle moved — which a non-differentiable engine
//! like Chrono cannot do. Viewer scene 124 (`Scene::VehicleSysid`)
//! unrolls [`identify_friction`] one Gauss-Newton iteration per frame
//! with live candidate-vs-ghost cars and a μ-convergence strip chart.

use crate::powertrain::{DriverInput, Powertrain};
use crate::terrain::FlatTerrain;
use crate::vehicle::Vehicle;

/// Drive a standard car at full throttle for `steps` of `dt` over flat
/// ground of friction `mu`; return the distance travelled along `x`.
pub fn drive_distance(mu: f64, steps: usize, dt: f64) -> f64 {
    let terrain = FlatTerrain::new(0.0, mu);
    let mut powertrain = Powertrain::default();
    let mut car = Vehicle::car();
    let go = DriverInput {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    };
    for _ in 0..steps {
        car.apply_driver(&mut powertrain, &go);
        car.step(&terrain, dt);
    }
    car.pos[0]
}

/// Squared-distance trajectory loss for a candidate friction.
pub fn distance_loss(mu: f64, target: f64, steps: usize, dt: f64) -> f64 {
    let d = drive_distance(mu, steps, dt) - target;
    d * d
}

/// Central finite-difference gradient of [`distance_loss`] w.r.t. `mu`.
pub fn distance_loss_grad(mu: f64, target: f64, steps: usize, dt: f64) -> f64 {
    let eps = 1.0e-3;
    (distance_loss(mu + eps, target, steps, dt) - distance_loss(mu - eps, target, steps, dt))
        / (2.0 * eps)
}

/// Identify the tyre-ground friction that reproduces the observed
/// `target` distance, by Gauss-Newton iteration from the guess `mu0`.
pub fn identify_friction(target: f64, mu0: f64, steps: usize, dt: f64, iters: usize) -> f64 {
    let mut mu = mu0;
    let eps = 1.0e-3;
    for _ in 0..iters {
        let d = drive_distance(mu, steps, dt);
        let sens = (drive_distance(mu + eps, steps, dt) - drive_distance(mu - eps, steps, dt))
            / (2.0 * eps);
        if sens.abs() < 1.0e-9 {
            break;
        }
        // Gauss-Newton step on the residual (d − target).
        mu -= (d - target) / sens;
        mu = mu.clamp(0.1, 1.5);
    }
    mu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_ground_friction_from_a_trajectory() {
        let (steps, dt) = (4000, 1.0e-3);
        // Ground truth: the car was driven on this friction.
        let true_mu = 0.75;
        let target = drive_distance(true_mu, steps, dt);

        // Start from a wrong guess and recover the friction.
        let mu0 = 0.35;
        let loss0 = distance_loss(mu0, target, steps, dt);
        assert!(loss0 > 0.0, "degenerate initial loss");

        let recovered = identify_friction(target, mu0, steps, dt, 20);
        let loss1 = distance_loss(recovered, target, steps, dt);

        assert!(
            loss1 < loss0 * 0.05,
            "optimization did not reduce the trajectory loss: {loss0} → {loss1}"
        );
        assert!(
            (recovered - true_mu).abs() < 0.05,
            "friction not recovered: got {recovered}, planted {true_mu}"
        );
    }

    #[test]
    fn distance_increases_with_friction() {
        // Sanity: more grip → the full-throttle car covers more ground,
        // so the loss landscape the optimiser climbs is monotone.
        let (steps, dt) = (3000, 1.0e-3);
        let low = drive_distance(0.4, steps, dt);
        let high = drive_distance(0.9, steps, dt);
        assert!(
            high > low,
            "distance should grow with friction: mu=0.4 → {low}, mu=0.9 → {high}"
        );
    }
}
