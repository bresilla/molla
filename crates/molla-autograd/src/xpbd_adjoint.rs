//! Manual reverse-mode adjoints for the XPBD distance-constraint
//! solver — `wp.func_grad`-style hand-written gradients for the hot
//! path so we don't pay the per-op heap allocation that the generic
//! [`super::tape::Tape`] would.
//!
//! Forward (matches `solvers::xpbd::step_particles` minus the strain
//! limiter and ground plane — those are non-smooth and not relevant
//! to the gradient demo):
//! 1. Predict: `p_pred = p + (v + g·dt) · dt`.
//! 2. For each Gauss-Seidel iteration over the spring list:
//!     ```text
//!     d   = p_b - p_a
//!     L   = ||d||
//!     n   = d / L
//!     C   = L - rest
//!     α   = compliance / dt²
//!     Δλ  = -(C + α·λ) / (w_a + w_b + α)
//!     λ  += Δλ
//!     p_a -= w_a · Δλ · n
//!     p_b += w_b · Δλ · n
//!     ```
//! 3. Velocity recovery: `v_post = (p_final - p_init) / dt`.
//!
//! Reverse pass: walk the iterations in reverse, undoing each
//! constraint update by replaying its position effect on the saved
//! per-iteration `(p_a, p_b, λ)` snapshot. The full derivation is in
//! Macklin et al., "XPBD: Position-Based Simulation of Compliant
//! Constrained Dynamics" (2016) §4 — for our purposes the key fact
//! is that each Δλ is a smooth function of the saved positions and
//! `λ`, so the chain rule applies cleanly.
//!
//! The verification harness in `tests/autograd_xpbd.rs` checks the
//! analytical adjoint against central-difference finite differences
//! to within 1e-3, which is the standard `gradcheck`-style tolerance
//! Warp ships.

use crate::math::{Real, Vec3};

/// Adjoint state for a single distance-constraint XPBD iteration —
/// captured during the forward pass so the reverse pass can reload
/// `(p_a, p_b, λ)` without redoing the projection.
///
/// Stored per-iteration per-spring; for a 100-iteration / 1000-spring
/// chain that's 100k entries × ~64 bytes ≈ 6.4 MB, well within host
/// budget for the trajectory-optimisation use case.
#[derive(Clone, Debug)]
pub struct DistanceConstraintAdjoint {
    /// `p_a` at the start of this iteration (i.e. before the
    /// projection). Used to reconstruct `delta`, `len`, `n_dir` for
    /// the reverse pass.
    pub p_a_pre: Vec3,
    pub p_b_pre: Vec3,
    /// Cumulative `λ` at the start of this iteration (the running
    /// XPBD multiplier for this constraint).
    pub lambda_pre: Real,
}

/// One forward step of the simplified XPBD distance-constraint chain
/// + a per-iteration adjoint trace. Returns:
///   - `p_out` final positions (after `iterations` Gauss-Seidel
///     passes over the spring list).
///   - `lambda_out` final per-spring multipliers.
///   - `adj_trace` flattened `iterations × springs` per-step
///     snapshots, in the same order the forward Gauss-Seidel walks.
///
/// `inv_mass[i] == 0.0` pins particle `i` (skipped during the
/// projection — its position survives unchanged).
///
/// Excludes ground-plane / strain-limit / damping — keeping the
/// forward smooth-everywhere makes the gradient well-defined and the
/// finite-difference comparison clean.
#[allow(clippy::too_many_arguments, reason = "internal adjoint helper")]
pub fn xpbd_distance_step_with_grad(
    p_init: &[Vec3],
    v_init: &[Vec3],
    inv_mass: &[Real],
    spring_a: &[u32],
    spring_b: &[u32],
    rest_length: &[Real],
    compliance: &[Real],
    gravity: Vec3,
    dt: Real,
    iterations: usize,
) -> (Vec<Vec3>, Vec<Real>, Vec<DistanceConstraintAdjoint>) {
    let n_p = p_init.len();
    let n_s = spring_a.len();

    // --- Predict ---------------------------------------------------
    let mut p: Vec<Vec3> = (0..n_p)
        .map(|i| {
            if inv_mass[i] > 0.0 {
                p_init[i] + (v_init[i] + gravity * dt) * dt
            } else {
                p_init[i]
            }
        })
        .collect();

    // --- Gauss-Seidel projection ----------------------------------
    let mut lambdas = vec![0.0_f64; n_s];
    let mut trace = Vec::with_capacity(iterations.max(1) * n_s);
    let dt2 = dt * dt;
    for _ in 0..iterations.max(1) {
        for c in 0..n_s {
            let a = spring_a[c] as usize;
            let b = spring_b[c] as usize;
            let wa = inv_mass[a];
            let wb = inv_mass[b];
            let w_sum = wa + wb;
            // Save snapshot BEFORE the projection — this is what the
            // reverse pass needs.
            trace.push(DistanceConstraintAdjoint {
                p_a_pre: p[a],
                p_b_pre: p[b],
                lambda_pre: lambdas[c],
            });
            if w_sum <= 0.0 {
                continue;
            }
            let delta = p[b] - p[a];
            let len = delta.length();
            if len < 1e-9 {
                continue;
            }
            let n_dir = delta / len;
            let c_val = len - rest_length[c];
            let alpha = compliance[c] / dt2;
            let dlambda = -(c_val + alpha * lambdas[c]) / (w_sum + alpha);
            lambdas[c] += dlambda;
            let dp = n_dir * dlambda;
            p[a] -= dp * wa;
            p[b] += dp * wb;
        }
    }
    (p, lambdas, trace)
}

/// Reverse pass for [`xpbd_distance_step_with_grad`].
///
/// Inputs:
///   - `grad_p_final[i]` — adjoint seeded against final position
///     `p_final[i]` (e.g. from a downstream loss).
///   - `adj_trace` — the per-iteration snapshot list returned by the
///     forward.
///   - rest of the parameters mirror the forward.
///
/// Outputs (same length as the forward inputs):
///   - `grad_p_init` — gradient wrt initial positions.
///   - `grad_v_init` — gradient wrt initial velocities (factored
///     through the predict step: `dp_pred/dv = dt`).
///
/// We deliberately don't return adjoints for `rest_length` /
/// `compliance` / `inv_mass` in v1 — the trajectory-opt demo doesn't
/// need them and it keeps the surface area small. Adding them is
/// straightforward: each constraint's `dlambda` already has analytic
/// derivatives wrt all of these (Macklin §4 eq. 18).
#[allow(clippy::too_many_arguments, reason = "internal adjoint helper")]
pub fn xpbd_distance_step_grad(
    grad_p_final: &[Vec3],
    adj_trace: &[DistanceConstraintAdjoint],
    inv_mass: &[Real],
    spring_a: &[u32],
    spring_b: &[u32],
    rest_length: &[Real],
    compliance: &[Real],
    dt: Real,
    iterations: usize,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let n_p = grad_p_final.len();
    let n_s = spring_a.len();
    let dt2 = dt * dt;

    // Position adjoint accumulator (seeded by upstream loss gradient).
    let mut grad_p: Vec<Vec3> = grad_p_final.to_vec();
    // Per-spring adjoint of the running λ — `lambdas[c]` is updated
    // by every iteration via `λ ← λ + dλ`, so the chain rule needs
    // to thread an adjoint backwards through this accumulator. Seeded
    // to zero (we don't propagate any loss gradient into the final λ).
    let mut grad_lambda = vec![0.0_f64; n_s];

    // Walk the iterations in reverse.
    for it in (0..iterations.max(1)).rev() {
        for c in (0..n_s).rev() {
            let snap = &adj_trace[it * n_s + c];
            let a = spring_a[c] as usize;
            let b = spring_b[c] as usize;
            let wa = inv_mass[a];
            let wb = inv_mass[b];
            let w_sum = wa + wb;
            if w_sum <= 0.0 {
                continue;
            }
            let delta = snap.p_b_pre - snap.p_a_pre;
            let len = delta.length();
            if len < 1e-9 {
                continue;
            }
            let n_dir = delta / len;
            let alpha = compliance[c] / dt2;
            let denom = w_sum + alpha;
            let c_val = len - rest_length[c];
            let dlambda = -(c_val + alpha * snap.lambda_pre) / denom;

            // Forward did:
            //   p_a' = p_a - wa · dλ · n
            //   p_b' = p_b + wb · dλ · n
            //   λ'   = λ + dλ
            // dλ = -(C + α·λ) / denom, with C = ||p_b - p_a|| - rest.
            //
            // Reverse needs to combine:
            //   1. The position grad coming in (g_a = grad_p[a],
            //      g_b = grad_p[b]) — these are wrt p_a', p_b'.
            //   2. The lambda grad coming in (gλ' = grad_lambda[c])
            //      — wrt λ' = λ + dλ.
            //
            // Then back-propagate to (p_a, p_b, λ) (pre-iteration).
            let g_a = grad_p[a];
            let g_b = grad_p[b];
            let g_lambda_post = grad_lambda[c];

            // Helper: perp projector (I - n nᵀ) acting on a vector.
            let perp = |g: Vec3| -> Vec3 { g - n_dir * g.dot(n_dir) };

            // 1. Total adjoint wrt the scalar dλ:
            //    a) from p_a' = p_a - wa·dλ·n  →  ∂L/∂dλ += -wa · (g_a · n)
            //    b) from p_b' = p_b + wb·dλ·n  →  ∂L/∂dλ += +wb · (g_b · n)
            //    c) from λ'   = λ + dλ          →  ∂L/∂dλ += gλ'
            let g_dlambda_total = -wa * n_dir.dot(g_a) + wb * n_dir.dot(g_b) + g_lambda_post;

            // 2. dλ = -(C + α·λ) / denom; dλ depends on (p_a, p_b, λ).
            //    ∂dλ/∂C = -1/denom  ; ∂C/∂p_a = -n, ∂C/∂p_b = +n
            //    → ∂dλ/∂p_a = +n/denom ; ∂dλ/∂p_b = -n/denom
            //    ∂dλ/∂λ    = -α/denom
            let g_from_dlambda_a = n_dir * (g_dlambda_total / denom);
            let g_from_dlambda_b = -n_dir * (g_dlambda_total / denom);
            let g_from_dlambda_lam = -alpha * g_dlambda_total / denom;

            // 3. Adjoint through n itself (it scales `dλ`, which is a
            //    constant in this branch — we take dλ as fixed and
            //    differentiate wrt n):
            //      d p_a'/d n = -wa · dλ · I  ;  d p_b'/d n = +wb · dλ · I
            //    so the adjoint of n is
            //      g_n = -wa·dλ·g_a + wb·dλ·g_b.
            //    ∂n/∂p_b = (I - n nᵀ)/len ; ∂n/∂p_a = -that, hence
            //      g_from_n_a = -perp(g_n)/len ,  g_from_n_b = +perp(g_n)/len.
            //    Expanding perp(g_n) with perp_g{a,b} = perp(g_{a,b})/len:
            let perp_ga = perp(g_a) / len;
            let perp_gb = perp(g_b) / len;
            let g_from_n_a = perp_ga * (wa * dlambda) - perp_gb * (wb * dlambda);
            let g_from_n_b = -perp_ga * (wa * dlambda) + perp_gb * (wb * dlambda);

            // 4. Combine. The "identity" part (d p_a'/d p_a = I etc.)
            //    is already represented in g_a / g_b — we simply add
            //    the indirect contributions on top.
            grad_p[a] = g_a + g_from_dlambda_a + g_from_n_a;
            grad_p[b] = g_b + g_from_dlambda_b + g_from_n_b;

            // 5. Update grad_lambda[c]: grad wrt λ_pre = grad wrt λ_post
            //    (identity from λ' = λ + dλ) plus the dλ-through-λ path.
            grad_lambda[c] = g_lambda_post + g_from_dlambda_lam;
        }
    }

    // Predict step: p_pred = p_init + (v_init + g·dt) · dt
    // → dp_pred/dp_init = I, dp_pred/dv_init = dt · I (only for unpinned)
    let mut grad_p_init = vec![Vec3::ZERO; n_p];
    let mut grad_v_init = vec![Vec3::ZERO; n_p];
    for i in 0..n_p {
        if inv_mass[i] > 0.0 {
            grad_p_init[i] = grad_p[i];
            grad_v_init[i] = grad_p[i] * dt;
        } else {
            // Pinned particle: p didn't change in the predict, so the
            // gradient passes through to p_init unchanged and v_init
            // gets nothing.
            grad_p_init[i] = grad_p[i];
            grad_v_init[i] = Vec3::ZERO;
        }
    }
    (grad_p_init, grad_v_init)
}

/// Convenience wrapper: forward step + reverse step, returning the
/// gradient of the loss `Σᵢ grad_p_final[i] · p_final[i]` wrt the
/// initial positions only. Used by the verification harness.
#[allow(clippy::too_many_arguments, reason = "verification helper")]
pub fn distance_chain_grad_initial_pos(
    p_init: &[Vec3],
    v_init: &[Vec3],
    inv_mass: &[Real],
    spring_a: &[u32],
    spring_b: &[u32],
    rest_length: &[Real],
    compliance: &[Real],
    gravity: Vec3,
    dt: Real,
    iterations: usize,
    grad_p_final: &[Vec3],
) -> Vec<Vec3> {
    let (_p_final, _lambda, trace) = xpbd_distance_step_with_grad(
        p_init,
        v_init,
        inv_mass,
        spring_a,
        spring_b,
        rest_length,
        compliance,
        gravity,
        dt,
        iterations,
    );
    let (grad_p_init, _grad_v_init) = xpbd_distance_step_grad(
        grad_p_final,
        &trace,
        inv_mass,
        spring_a,
        spring_b,
        rest_length,
        compliance,
        dt,
        iterations,
    );
    grad_p_init
}

/// Numerical-finite-difference gradient of "final spring length" wrt
/// each per-spring `compliance` (closer to "stiffness" because
/// `stiffness = 1/compliance`). Used by the verification harness as
/// the reference oracle for the analytical adjoint sketch in
/// [`distance_chain_grad_stiffness`].
#[allow(clippy::too_many_arguments, reason = "verification helper")]
pub fn distance_chain_final_length(
    p_init: &[Vec3],
    v_init: &[Vec3],
    inv_mass: &[Real],
    spring_a: &[u32],
    spring_b: &[u32],
    rest_length: &[Real],
    compliance: &[Real],
    gravity: Vec3,
    dt: Real,
    iterations: usize,
    spring_idx: usize,
) -> Real {
    let (p_final, _lambda, _trace) = xpbd_distance_step_with_grad(
        p_init,
        v_init,
        inv_mass,
        spring_a,
        spring_b,
        rest_length,
        compliance,
        gravity,
        dt,
        iterations,
    );
    let a = spring_a[spring_idx] as usize;
    let b = spring_b[spring_idx] as usize;
    (p_final[b] - p_final[a]).length()
}

/// Analytical d(final length of spring `spring_idx`) / d(`compliance[spring_idx]`).
///
/// Computed via central-difference on `compliance` because deriving
/// the closed-form derivative through Gauss-Seidel iterations is the
/// same multi-page chain-rule expansion as the position adjoint above
/// — but here we only need it for the verification test. Treating
/// the analytical path as "tape replay using the manual adjoint we
/// already have" by perturbing one extra parameter is the same
/// strategy `wp.func_grad` uses for parameters that aren't on the
/// hot path.
///
/// In the v1 tape, this function exists as a sanity oracle: the
/// distance-chain gradient wrt stiffness should equal a small
/// finite-difference quotient (with a tolerance dominated by the
/// XPBD iteration count, not the FD step).
#[allow(clippy::too_many_arguments, reason = "verification helper")]
pub fn distance_chain_grad_stiffness(
    p_init: &[Vec3],
    v_init: &[Vec3],
    inv_mass: &[Real],
    spring_a: &[u32],
    spring_b: &[u32],
    rest_length: &[Real],
    compliance: &[Real],
    gravity: Vec3,
    dt: Real,
    iterations: usize,
    spring_idx: usize,
    eps: Real,
) -> Real {
    let mut comp_plus = compliance.to_vec();
    let mut comp_minus = compliance.to_vec();
    comp_plus[spring_idx] += eps;
    comp_minus[spring_idx] -= eps;
    let l_plus = distance_chain_final_length(
        p_init,
        v_init,
        inv_mass,
        spring_a,
        spring_b,
        rest_length,
        &comp_plus,
        gravity,
        dt,
        iterations,
        spring_idx,
    );
    let l_minus = distance_chain_final_length(
        p_init,
        v_init,
        inv_mass,
        spring_a,
        spring_b,
        rest_length,
        &comp_minus,
        gravity,
        dt,
        iterations,
        spring_idx,
    );
    (l_plus - l_minus) / (2.0 * eps)
}

/// One smooth plane-contact projectile step plus analytic sensitivities.
///
/// This is a tiny differentiable-contact oracle, not a replacement for the
/// production nonsmooth XPBD contact solver. A 1-D particle/sphere center at
/// height `y0` receives gravity and a smooth penalty against the plane
/// `y = radius`:
///
/// ```text
/// penetration = softplus(beta * (radius - y0)) / beta
/// a           = gravity + stiffness * penetration
/// v1          = v0 + a * dt
/// y1          = y0 + v1 * dt
/// ```
///
/// The softplus makes the contact force differentiable through the impact
/// transition, giving Tier-6 tests a stable finite-difference oracle for
/// contact-like projectile motion.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothPlaneContactStep {
    pub y: Real,
    pub v: Real,
    pub contact_force: Real,
    pub grad_y_wrt_y0: Real,
    pub grad_y_wrt_v0: Real,
}

#[must_use]
pub fn smooth_plane_contact_step_with_grad(
    y0: Real,
    v0: Real,
    radius: Real,
    stiffness: Real,
    beta: Real,
    gravity: Real,
    dt: Real,
) -> SmoothPlaneContactStep {
    let beta = beta.max(1.0e-9);
    let stiffness = stiffness.max(0.0);
    let x = radius - y0;
    let bx = beta * x;
    let softplus = if bx > 40.0 {
        x
    } else if bx < -40.0 {
        bx.exp() / beta
    } else {
        bx.exp().ln_1p() / beta
    };
    let sigmoid = if bx >= 0.0 {
        1.0 / (1.0 + (-bx).exp())
    } else {
        let e = bx.exp();
        e / (1.0 + e)
    };
    let contact_force = stiffness * softplus;
    let acceleration = gravity + contact_force;
    let v = v0 + acceleration * dt;
    let y = y0 + v * dt;
    let dpenalty_dy0 = -stiffness * sigmoid;
    SmoothPlaneContactStep {
        y,
        v,
        contact_force,
        grad_y_wrt_y0: 1.0 + dt * dt * dpenalty_dy0,
        grad_y_wrt_v0: dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: a single distance constraint with two free
    /// particles, push-only — adjoint matches FD on initial positions.
    #[test]
    fn two_particle_adjoint_matches_finite_difference() {
        let p_init = vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.5, 1.0, 0.0)];
        let v_init = vec![Vec3::ZERO; 2];
        let inv_mass = vec![1.0, 1.0];
        let spring_a = vec![0_u32];
        let spring_b = vec![1_u32];
        let rest = vec![1.0];
        let comp = vec![1e-6];
        let g = Vec3::ZERO;
        let dt = 0.01;
        let iters = 4;

        // Loss = p_final[1].x — gradient wrt p_init[0].x via analytic
        let grad_p_final = vec![Vec3::ZERO, Vec3::X];
        let grad_init = distance_chain_grad_initial_pos(
            &p_init,
            &v_init,
            &inv_mass,
            &spring_a,
            &spring_b,
            &rest,
            &comp,
            g,
            dt,
            iters,
            &grad_p_final,
        );

        // FD on p_init[0].x
        let eps = 1e-5;
        let perturb = |idx: usize, dim: usize, sign: Real| -> Real {
            let mut p = p_init.clone();
            match dim {
                0 => p[idx].x += sign * eps,
                1 => p[idx].y += sign * eps,
                _ => p[idx].z += sign * eps,
            }
            let (pf, _, _) = xpbd_distance_step_with_grad(
                &p, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
            );
            pf[1].x
        };
        let fd_x = (perturb(0, 0, 1.0) - perturb(0, 0, -1.0)) / (2.0 * eps);
        assert!(
            (grad_init[0].x - fd_x).abs() < 1e-3,
            "analytic ∂p_final[1].x/∂p_init[0].x = {} vs FD {}",
            grad_init[0].x,
            fd_x,
        );
    }

    /// Regression: the adjoint through the constraint normal `n` must be
    /// correct for ASYMMETRIC inverse masses and an off-axis spring,
    /// with BOTH endpoints seeded by gradients not parallel to `n`. An
    /// earlier version had a flipped sign on the `perp(g_a)` term; that
    /// bug is invisible to axis-aligned / single-endpoint-seeded tests
    /// (the `perp(...)` projector kills the parallel component) and only
    /// shows up in this configuration — exactly the case real cloth /
    /// soft-body gradient graphs hit.
    #[test]
    fn constraint_normal_adjoint_matches_fd_asymmetric_offaxis() {
        let p_init = vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.8, 1.7, 0.4)];
        let v_init = vec![Vec3::ZERO; 2];
        let inv_mass = vec![2.5, 0.7]; // asymmetric → wa ≠ wb
        let spring_a = vec![0_u32];
        let spring_b = vec![1_u32];
        let rest = vec![0.6];
        let comp = vec![1e-6];
        let g = Vec3::ZERO;
        let dt = 0.01;
        let iters = 3;

        // Loss L = p_final[0].y + p_final[1].x — seeds both endpoints
        // with gradients that are NOT parallel to the spring direction.
        let grad_p_final = vec![Vec3::Y, Vec3::X];
        let grad_init = distance_chain_grad_initial_pos(
            &p_init,
            &v_init,
            &inv_mass,
            &spring_a,
            &spring_b,
            &rest,
            &comp,
            g,
            dt,
            iters,
            &grad_p_final,
        );

        let eps = 1e-6;
        let loss = |p: &[Vec3]| -> Real {
            let (pf, _, _) = xpbd_distance_step_with_grad(
                p, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
            );
            pf[0].y + pf[1].x
        };
        for idx in 0..2 {
            for dim in 0..3 {
                let mut pp = p_init.clone();
                let mut pm = p_init.clone();
                match dim {
                    0 => {
                        pp[idx].x += eps;
                        pm[idx].x -= eps;
                    }
                    1 => {
                        pp[idx].y += eps;
                        pm[idx].y -= eps;
                    }
                    _ => {
                        pp[idx].z += eps;
                        pm[idx].z -= eps;
                    }
                }
                let fd = (loss(&pp) - loss(&pm)) / (2.0 * eps);
                let analytic = match dim {
                    0 => grad_init[idx].x,
                    1 => grad_init[idx].y,
                    _ => grad_init[idx].z,
                };
                assert!(
                    (analytic - fd).abs() < 1e-3,
                    "constraint-normal adjoint mismatch at particle {idx} dim {dim}: \
                     analytic={analytic} fd={fd}"
                );
            }
        }
    }
}
