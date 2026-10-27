//! Stage F5 — differentiable SPH (CPU).
//!
//! A differentiable SPH rollout: the fluid is launched with a uniform
//! initial velocity `v0`, stepped with the [`crate::sph_oracle`] WCSPH
//! solver, and a scalar loss is taken on the final fluid centroid.
//!
//! The gradient `dLoss/dv0` is exact. The SPH inter-particle forces are
//! pairwise antisymmetric, so they conserve linear momentum and the
//! fluid centre of mass follows the closed-form ballistic path
//! `c(T) = c(0) + v0·T + ½·g·T²`. The adjoint of that map is therefore
//! `dLoss/dv0 = 2(c(T) − target)·T`, with no tape needed. The tests
//! verify this against finite differences taken through the *full* SPH
//! rollout — which only agree if the solver really does conserve
//! momentum — and run a gradient-descent loop that drives the loss to
//! zero ("shoot the fluid blob at a target", the WCSPH analogue of the
//! Warp/Newton `diffsim_ball` example).
//!
//! A full reverse-mode adjoint through the internal density/pressure
//! kernels (for losses that depend on the internal dynamics, e.g. the
//! fluid spread) and a GPU adjoint are deferred — see `PLAN_ENGINE.md`.

use crate::sph_oracle::SphOracle;

/// Centroid of a set of points.
pub fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Run an SPH rollout from `template`, launching every fluid particle
/// at uniform velocity `v0`; return the final fluid centroid.
pub fn rollout_centroid(template: &SphOracle, v0: [f64; 3], steps: usize, dt: f64) -> [f64; 3] {
    let mut o = template.clone();
    for v in o.vel.iter_mut().take(o.n_fluid) {
        *v = v0;
    }
    for _ in 0..steps {
        o.step(dt);
    }
    centroid(&o.pos[..o.n_fluid])
}

/// Squared-distance loss of the fluid centroid to `target`.
pub fn centroid_loss(c: [f64; 3], target: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = c[k] - target[k];
        s += d * d;
    }
    s
}

/// Reverse-mode gradient `dLoss/dv0` for [`centroid_loss`] on a
/// [`rollout_centroid`] of duration `total_time`. Exact (see the module
/// docs): `2(c − target)·T`.
pub fn centroid_loss_grad_v0(
    centroid_final: [f64; 3],
    target: [f64; 3],
    total_time: f64,
) -> [f64; 3] {
    let mut g = [0.0; 3];
    for k in 0..3 {
        g[k] = 2.0 * (centroid_final[k] - target[k]) * total_time;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small water block in a domain large enough that it never
    /// touches a wall during a short rollout (so momentum is conserved).
    fn block() -> SphOracle {
        let h = 0.05;
        let rho0 = 1000.0;
        let mut pos = Vec::new();
        for ix in 0..3 {
            for iy in 0..3 {
                for iz in 0..3 {
                    pos.push([ix as f64 * h, iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        let n = pos.len();
        SphOracle {
            pos,
            vel: vec![[0.0; 3]; n],
            n_fluid: n,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.25,
            gravity: [0.0, -9.81, 0.0],
            domain_min: [-50.0; 3],
            domain_max: [50.0; 3],
            wall_restitution: 0.0,
            friction: 0.0,
        }
    }

    #[test]
    fn centroid_gradient_matches_finite_difference() {
        let template = block();
        let target = [0.30, 0.40, -0.10];
        let v0 = [0.6, 1.2, -0.3];
        let steps = 40;
        let dt = 5.0e-4;
        let total_time = steps as f64 * dt;

        let c = rollout_centroid(&template, v0, steps, dt);
        let grad = centroid_loss_grad_v0(c, target, total_time);

        // Finite differences through the full SPH rollout.
        let eps = 1.0e-4;
        for k in 0..3 {
            let mut vp = v0;
            let mut vm = v0;
            vp[k] += eps;
            vm[k] -= eps;
            let lp = centroid_loss(rollout_centroid(&template, vp, steps, dt), target);
            let lm = centroid_loss(rollout_centroid(&template, vm, steps, dt), target);
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 1.0e-3 * (1.0 + fd.abs()),
                "adjoint/FD mismatch on axis {k}: adjoint={} fd={fd}",
                grad[k]
            );
        }
    }

    #[test]
    fn centroid_optimization_decreases_loss() {
        let template = block();
        let target = [0.25, 0.55, 0.05];
        let steps = 40;
        let dt = 5.0e-4;
        let total_time = steps as f64 * dt;

        let mut v0 = [0.0, 0.0, 0.0];
        let loss0 = centroid_loss(rollout_centroid(&template, v0, steps, dt), target);
        // The loss is exactly quadratic in v0 with per-axis curvature
        // 2·T², so a step of 1/(2·T²) is the Newton-optimal rate.
        let lr = 1.0 / (2.0 * total_time * total_time);
        for _ in 0..20 {
            let c = rollout_centroid(&template, v0, steps, dt);
            let g = centroid_loss_grad_v0(c, target, total_time);
            for k in 0..3 {
                v0[k] -= lr * g[k];
            }
        }
        let loss1 = centroid_loss(rollout_centroid(&template, v0, steps, dt), target);
        assert!(loss0 > 0.0, "degenerate initial loss");
        assert!(
            loss1 < loss0 * 1.0e-3,
            "optimization did not drive the loss down: loss0={loss0} loss1={loss1}"
        );
    }
}
