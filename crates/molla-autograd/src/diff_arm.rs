//! 2-link planar arm trajectory-optimisation primitive used by
//! `examples/diff_arm_reach.rs` and the viewer's `Scene::DiffArm`
//! pane.
//!
//! Hand-rolled hand-differentiated dynamics so the demo is small,
//! interactive, and doesn't carry the full molla `ModelBuilder`
//! pipeline. The closed-form adjoint mirrors what the
//! [`crate::autograd::xpbd_adjoint`] family does for XPBD: walk the
//! trajectory in reverse, propagate ∂L/∂q + ∂L/∂qd through each
//! integration step, accumulate ∂L/∂tau at each timestep.

use crate::math::Real;

/// Default arm parameters for the demo. The viewer can override
/// these per-instance via [`DiffArm::new`] but the default reads as
/// "the arm in the example" so the two stay in sync.
#[derive(Copy, Clone, Debug)]
pub struct DiffArm {
    pub l1: Real,
    pub l2: Real,
    pub inertia: Real,
    pub damping: Real,
    pub dt: Real,
    pub horizon: usize,
}

impl Default for DiffArm {
    fn default() -> Self {
        Self {
            l1: 0.5,
            l2: 0.4,
            inertia: 0.05,
            damping: 0.2,
            dt: 0.02,
            horizon: 40,
        }
    }
}

impl DiffArm {
    pub const fn new(l1: Real, l2: Real, horizon: usize) -> Self {
        Self {
            l1,
            l2,
            inertia: 0.05,
            damping: 0.2,
            dt: 0.02,
            horizon,
        }
    }

    /// Forward kinematics: planar end-effector position `(x, y)`.
    pub fn forward_kinematics(&self, q: [Real; 2]) -> [Real; 2] {
        let elbow_x = self.l1 * q[0].sin();
        let elbow_y = -self.l1 * q[0].cos();
        let s12 = (q[0] + q[1]).sin();
        let c12 = (q[0] + q[1]).cos();
        [elbow_x + self.l2 * s12, elbow_y - self.l2 * c12]
    }

    /// Forward-kinematics Jacobian: `[[∂ee_x/∂q1, ∂ee_x/∂q2], [∂ee_y/∂q1, ∂ee_y/∂q2]]`.
    pub fn fk_jacobian(&self, q: [Real; 2]) -> [[Real; 2]; 2] {
        let s1 = q[0].sin();
        let c1 = q[0].cos();
        let s12 = (q[0] + q[1]).sin();
        let c12 = (q[0] + q[1]).cos();
        [
            [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
            [self.l1 * s1 + self.l2 * s12, self.l2 * s12],
        ]
    }

    /// Roll out the dynamics under torque sequence `tau` (length =
    /// [`Self::horizon`]). Returns the per-step `(q, qd)` trace
    /// (length `horizon + 1`).
    pub fn rollout(
        &self,
        q0: [Real; 2],
        qd0: [Real; 2],
        tau: &[[Real; 2]],
    ) -> Vec<([Real; 2], [Real; 2])> {
        let mut q = q0;
        let mut qd = qd0;
        let mut trace = Vec::with_capacity(tau.len() + 1);
        trace.push((q, qd));
        for &t in tau {
            let qdd0 = (t[0] - self.damping * qd[0]) / self.inertia;
            let qdd1 = (t[1] - self.damping * qd[1]) / self.inertia;
            qd[0] += qdd0 * self.dt;
            qd[1] += qdd1 * self.dt;
            q[0] += qd[0] * self.dt;
            q[1] += qd[1] * self.dt;
            trace.push((q, qd));
        }
        trace
    }

    /// Reverse pass: given the trajectory trace and an end-effector
    /// gradient seed `(g_ee_x, g_ee_y)`, return the per-timestep
    /// torque gradient.
    pub fn reverse(&self, trace: &[([Real; 2], [Real; 2])], grad_ee: [Real; 2]) -> Vec<[Real; 2]> {
        let final_q = trace[trace.len() - 1].0;
        let dee_dq = self.fk_jacobian(final_q);
        let g_q = [
            grad_ee[0] * dee_dq[0][0] + grad_ee[1] * dee_dq[1][0],
            grad_ee[0] * dee_dq[0][1] + grad_ee[1] * dee_dq[1][1],
        ];
        let mut g_qd = [0.0; 2];
        let mut g_tau = vec![[0.0; 2]; trace.len() - 1];
        let damp_factor = 1.0 - self.damping * self.dt / self.inertia;
        let tau_factor = self.dt / self.inertia;
        for t in (0..trace.len() - 1).rev() {
            let g_qd_post0 = g_qd[0] + g_q[0] * self.dt;
            let g_qd_post1 = g_qd[1] + g_q[1] * self.dt;
            g_tau[t][0] = g_qd_post0 * tau_factor;
            g_tau[t][1] = g_qd_post1 * tau_factor;
            g_qd[0] = g_qd_post0 * damp_factor;
            g_qd[1] = g_qd_post1 * damp_factor;
        }
        g_tau
    }

    /// Run gradient-descent optimisation on `tau` to drive the EE
    /// to `target`. Mutates `tau` in place; returns
    /// `(final_residual, iters_to_5cm)`. `iters_to_5cm` is
    /// `Some(it)` if the residual fell below 0.05 m at iter `it`.
    ///
    /// `lr` should be ~5.0 for the default arm parameters; smaller
    /// LRs converge but take longer.
    pub fn gradient_descent(
        &self,
        q0: [Real; 2],
        qd0: [Real; 2],
        tau: &mut [[Real; 2]],
        target: [Real; 2],
        lr: Real,
        iters: usize,
    ) -> (Real, Option<usize>) {
        let mut last_residual = Real::INFINITY;
        let mut converged: Option<usize> = None;
        for it in 0..iters {
            let trace = self.rollout(q0, qd0, tau);
            let ee = self.forward_kinematics(trace[trace.len() - 1].0);
            let dx = ee[0] - target[0];
            let dy = ee[1] - target[1];
            last_residual = (dx * dx + dy * dy).sqrt();
            if converged.is_none() && last_residual < 0.05 {
                converged = Some(it);
            }
            let g_tau = self.reverse(&trace, [2.0 * dx, 2.0 * dy]);
            for i in 0..tau.len() {
                tau[i][0] -= lr * g_tau[i][0];
                tau[i][1] -= lr * g_tau[i][1];
            }
        }
        (last_residual, converged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fk_jacobian_matches_finite_difference() {
        let arm = DiffArm::default();
        let q = [0.7, -0.4];
        let j = arm.fk_jacobian(q);
        let eps = 1e-6;
        for i in 0..2 {
            let mut qp = q;
            let mut qm = q;
            qp[i] += eps;
            qm[i] -= eps;
            let ep = arm.forward_kinematics(qp);
            let em = arm.forward_kinematics(qm);
            let fd_x = (ep[0] - em[0]) / (2.0 * eps);
            let fd_y = (ep[1] - em[1]) / (2.0 * eps);
            assert!((j[0][i] - fd_x).abs() < 1e-5);
            assert!((j[1][i] - fd_y).abs() < 1e-5);
        }
    }

    #[test]
    fn reach_target_in_bounded_iters() {
        let arm = DiffArm::default();
        let mut tau = vec![[0.0; 2]; arm.horizon];
        let (residual, conv) =
            arm.gradient_descent([0.0; 2], [0.0; 2], &mut tau, [0.45, -0.25], 5.0, 100);
        assert!(residual < 0.01, "residual {residual} > 1 cm");
        let conv = conv.expect("did not converge to 5 cm in 100 iters");
        assert!(conv < 50, "took {conv} iters to reach 5 cm");
    }
}
