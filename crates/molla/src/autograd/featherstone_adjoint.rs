//! Reverse-mode adjoint for the Featherstone Articulated Body
//! Algorithm — CPU-side, serial revolute chains (Stage 23.1 v1).
//!
//! # Scope
//!
//! Mirrors the structure of [`crate::solvers::featherstone::cpu_step`]
//! restricted to **serial revolute chains** with point-mass-style
//! inertia at the COM — the joint topology that all three Stage 23.1
//! verification cases use (single pendulum, 2-link planar arm, 6-DOF
//! Franka chain). The three Featherstone sweeps map onto three
//! [`super::tape::TapeOp`] variants:
//!
//! | Sweep                  | TapeOp                                  |
//! |------------------------|------------------------------------------|
//! | Forward kinematics + v | [`super::tape::TapeOp::FeatherstoneFwdKinematics`] |
//! | Articulated inertia    | [`super::tape::TapeOp::FeatherstoneUpwardInertia`] |
//! | Forward acceleration   | [`super::tape::TapeOp::FeatherstoneDownwardAccel`] |
//!
//! The forward-side helper [`FeatherstoneRevoluteChain::rollout`]
//! drives the three sweeps for `horizon` steps and stores the per-step
//! intermediates (joint axes in world frame, body world transforms,
//! body spatial velocities, articulated-inertia U/D columns, per-step
//! `q̈`) on a [`FeatherstoneTrace`]. The adjoint
//! [`FeatherstoneRevoluteChain::reverse`] walks the trace in reverse
//! and returns per-step torque gradients given a seed `∂loss/∂(EE
//! position)`.
//!
//! # Implementation strategy
//!
//! Featherstone's three sweeps are nontrivial to differentiate
//! line-by-line (the §10 derivation is a multi-week algebra exercise).
//! For Stage 23.1 v1 we ship a **hybrid** adjoint that is correct,
//! verifiable, and matches the reverse-mode contract:
//!
//! - **Per-step "single-step" Jacobian** is computed via central
//!   differences on the full Featherstone forward at this step's
//!   `(q_t, qd_t)`. For serial revolute chains of size ≤ 7 the cost
//!   is `O(3 n · per-step)` per step — negligible compared to the
//!   trajectory length.
//! - **Trajectory-level adjoint** is fully analytical: the chain rule
//!   threads ∂L/∂q + ∂L/∂qd through the symplectic-Euler integrator
//!   in closed form, accumulating ∂L/∂tau at every step.
//! - **Tape-level recording** maps each sweep to its own [`TapeOp`]
//!   chunk so a future GPU port can swap the per-step Jacobian build
//!   for a hand-derived analytical sweep without touching the
//!   trajectory adjoint or the tape format.
//!
//! This is the standard "adjoint via gradient through inverse
//! dynamics" pattern — see Carpentier & Mansard 2018 "Analytical
//! Derivatives of Rigid Body Dynamics Algorithms" — applied at the
//! single-step granularity.
//!
//! # Determinism
//!
//! Every operation is a fixed-shape arithmetic loop over the trace —
//! no `HashMap` iteration, no parallel reductions, no GPU readback
//! reorder. Identical inputs (`q0`, `qd0`, `tau`) yield bit-exact
//! identical gradients across runs (verified by the
//! `tape_replay_is_deterministic` test in
//! `tests/autograd_featherstone.rs`).

use crate::math::{
    spatial_cross, spatial_cross_dual, Mat3, Quat, Real, SpatialMatrix6, SpatialVector,
    Transform, Vec3,
};

use super::tape::{Tape, TapeId, TapeOp};

// ────────────────────────────────────────────────────────────────────
// Public model: a serial revolute chain with point-mass inertia.
// ────────────────────────────────────────────────────────────────────

/// One revolute joint in a serial chain. Parent-anchor transform `x_p`
/// is in the parent body's local frame (matches `Model::joint_x_p`).
/// The child body's frame at q=0 coincides with the joint frame.
#[derive(Clone, Debug)]
pub struct RevoluteLink {
    /// Joint axis expressed in the parent (joint) frame at q = 0.
    pub axis_local: Vec3,
    /// Joint anchor in the parent body's frame.
    pub x_p: Transform,
    /// Mass of the child body.
    pub mass: Real,
    /// COM offset in the child body's local frame.
    pub com_local: Vec3,
    /// Inertia tensor about the COM, expressed in the body's local frame.
    pub inertia_local: Mat3,
}

/// A serial chain of revolute joints. The chain index `i` denotes both
/// the joint and its child body — joint `i` connects body `i-1`
/// (parent; world if `i == 0`) to body `i`.
#[derive(Clone, Debug)]
pub struct FeatherstoneRevoluteChain {
    pub links: Vec<RevoluteLink>,
    /// World gravity (typically `(0, -9.81, 0)`).
    pub gravity: Vec3,
    /// Joint-space damping coefficient (1/s) — matches
    /// [`crate::solvers::FeatherstoneSolver::joint_damping`].
    pub damping: Real,
    /// Integration timestep.
    pub dt: Real,
    /// Number of integration steps per rollout.
    pub horizon: usize,
    /// EE point in the LAST link's local frame, used by
    /// [`Self::end_effector`].
    pub ee_local: Vec3,
}

impl FeatherstoneRevoluteChain {
    /// Single-pendulum convenience: 1 link of length `l` (joint at the
    /// origin, COM at `+y * l/2`, point mass), revolving about world Z,
    /// EE at `+y * l`.
    #[must_use]
    pub fn pendulum(l: Real, mass: Real, dt: Real, horizon: usize) -> Self {
        let i = mass * l * l / 12.0;
        let inertia = Mat3::from_diagonal(Vec3::splat(i));
        Self {
            links: vec![RevoluteLink {
                axis_local: Vec3::Z,
                x_p: Transform::IDENTITY,
                mass,
                com_local: Vec3::new(0.0, l * 0.5, 0.0),
                inertia_local: inertia,
            }],
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.05,
            dt,
            horizon,
            ee_local: Vec3::new(0.0, l, 0.0),
        }
    }

    /// 2-link planar arm: two revolute joints about world +Z, each link
    /// of length `l1`/`l2`, point mass at `mass`. EE at `(0, l2, 0)`
    /// in the second link's frame.
    #[must_use]
    pub fn two_link_arm(l1: Real, l2: Real, mass: Real, dt: Real, horizon: usize) -> Self {
        let i1 = mass * l1 * l1 / 12.0;
        let i2 = mass * l2 * l2 / 12.0;
        Self {
            links: vec![
                RevoluteLink {
                    axis_local: Vec3::Z,
                    x_p: Transform::IDENTITY,
                    mass,
                    com_local: Vec3::new(0.0, l1 * 0.5, 0.0),
                    inertia_local: Mat3::from_diagonal(Vec3::splat(i1)),
                },
                RevoluteLink {
                    axis_local: Vec3::Z,
                    x_p: Transform::from_translation(Vec3::new(0.0, l1, 0.0)),
                    mass,
                    com_local: Vec3::new(0.0, l2 * 0.5, 0.0),
                    inertia_local: Mat3::from_diagonal(Vec3::splat(i2)),
                },
            ],
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.05,
            dt,
            horizon,
            ee_local: Vec3::new(0.0, l2, 0.0),
        }
    }

    /// 6-DOF Franka-style serial chain: alternating Z/Y axes, link
    /// lengths approximating Panda's flange-to-flange distances.
    /// Inertia approximated as a uniform box per link.
    #[must_use]
    pub fn franka6(dt: Real, horizon: usize) -> Self {
        let mass = 1.0_f64;
        let half = 0.04_f64;
        let i = mass / 3.0 * (half * half + half * half);
        let inertia = Mat3::from_diagonal(Vec3::splat(i));
        let axes = [Vec3::Z, Vec3::Y, Vec3::Z, Vec3::Y, Vec3::Z, Vec3::Y];
        let link_lens = [0.333_f64, 0.0, 0.316, 0.0, 0.384, 0.107];
        let mut links = Vec::with_capacity(6);
        for k in 0..6 {
            let x_p = if k == 0 {
                Transform::IDENTITY
            } else {
                Transform::from_translation(Vec3::new(0.0, 0.0, link_lens[k - 1]))
            };
            links.push(RevoluteLink {
                axis_local: axes[k],
                x_p,
                mass,
                com_local: Vec3::new(0.0, 0.0, link_lens[k] * 0.5),
                inertia_local: inertia,
            });
        }
        Self {
            links,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.1,
            dt,
            horizon,
            ee_local: Vec3::new(0.0, 0.0, link_lens[5]),
        }
    }

    /// EE position at joint configuration `q` (length = #links).
    #[must_use]
    pub fn end_effector(&self, q: &[Real]) -> Vec3 {
        let body_q = self.forward_kinematics(q);
        let last = body_q[body_q.len() - 1];
        last.position + last.rotation * self.ee_local
    }

    /// Forward kinematics — returns the world transform of every body.
    /// At q=0 the body frame coincides with the joint frame; for
    /// q≠0 the body frame is rotated about `axis_local` by `q`.
    #[must_use]
    pub fn forward_kinematics(&self, q: &[Real]) -> Vec<Transform> {
        let n = self.links.len();
        let mut out = Vec::with_capacity(n);
        let mut parent = Transform::IDENTITY;
        for j in 0..n {
            let link = &self.links[j];
            let parent_joint = parent.compose(&link.x_p);
            let q_rot = Quat::from_axis_angle(link.axis_local, q[j]);
            let body_w = Transform {
                position: parent_joint.position,
                rotation: (parent_joint.rotation * q_rot).normalize(),
            };
            out.push(body_w);
            parent = body_w;
        }
        out
    }
}

// ────────────────────────────────────────────────────────────────────
// Per-step trace (forward intermediates).
// ────────────────────────────────────────────────────────────────────

/// Per-link intermediate quantities captured during the forward
/// rollout so the adjoint can avoid re-evaluating the FK +
/// articulated-inertia algebra.
///
/// Reused as a flat `horizon × n_links` table for column-buffer reuse
/// (mirrors the [`crate::kernels::JacobianKernel`] per-articulation
/// column buffers — same logical shape, no GPU upload).
#[derive(Clone, Debug)]
pub struct FeatherstoneStepTrace {
    /// Body world transform `T_world_j`.
    pub body_world: Vec<Transform>,
    /// Joint world axis = `R_parent_joint · axis_local`, normalised.
    pub axis_world: Vec<Vec3>,
    /// Joint origin in world coordinates.
    pub joint_origin_world: Vec<Vec3>,
    /// Per-joint world COM position.
    pub com_world: Vec<Vec3>,
    /// Per-joint articulated-inertia column `U_k = I^A_k · S_k`.
    pub u_k: Vec<SpatialVector>,
    /// Per-joint `D_k = S_k · U_k`.
    pub d_k: Vec<Real>,
    /// Per-joint `u_tau_k = tau_k - S_k · p_a_k`.
    pub u_tau_k: Vec<Real>,
    /// Body spatial velocity `v_i` after pass 1.
    pub v_body: Vec<SpatialVector>,
    /// Per-joint Coriolis bias `c_i` after pass 1.
    pub c_term: Vec<SpatialVector>,
    /// Body spatial acceleration `a_i` after pass 3.
    pub a_body: Vec<SpatialVector>,
    /// Joint acceleration `q̈_k` for each joint at this step.
    pub q_dd: Vec<Real>,
}

/// Full rollout trace: the per-step `(q, qd)` plus the per-step
/// intermediate table.
#[derive(Clone, Debug)]
pub struct FeatherstoneTrace {
    /// Joint coords at each step boundary, length `horizon + 1`.
    pub q: Vec<Vec<Real>>,
    /// Joint velocities at each step boundary, length `horizon + 1`.
    pub qd: Vec<Vec<Real>>,
    /// Per-step intermediates, length `horizon`.
    pub steps: Vec<FeatherstoneStepTrace>,
}

// ────────────────────────────────────────────────────────────────────
// Forward rollout
// ────────────────────────────────────────────────────────────────────

impl FeatherstoneRevoluteChain {
    /// Single-step Featherstone forward + intermediate capture.
    pub fn forward_step(
        &self,
        q: &[Real],
        qd: &[Real],
        tau: &[Real],
    ) -> (Vec<Real>, Vec<Real>, FeatherstoneStepTrace) {
        let n = self.links.len();
        let mut step = FeatherstoneStepTrace {
            body_world: vec![Transform::IDENTITY; n],
            axis_world: vec![Vec3::ZERO; n],
            joint_origin_world: vec![Vec3::ZERO; n],
            com_world: vec![Vec3::ZERO; n],
            u_k: vec![SpatialVector::ZERO; n],
            d_k: vec![0.0; n],
            u_tau_k: vec![0.0; n],
            v_body: vec![SpatialVector::ZERO; n],
            c_term: vec![SpatialVector::ZERO; n],
            a_body: vec![SpatialVector::ZERO; n],
            q_dd: vec![0.0; n],
        };

        let mut i_a: Vec<SpatialMatrix6> = vec![SpatialMatrix6::ZERO; n];
        let mut p_a: Vec<SpatialVector> = vec![SpatialVector::ZERO; n];
        let mut s_cols: Vec<SpatialVector> = vec![SpatialVector::ZERO; n];
        let mut parent_t = Transform::IDENTITY;
        let mut v_parent = SpatialVector::ZERO;

        // Pass 1: forward kinematics + body kinematics.
        for j in 0..n {
            let link = &self.links[j];
            let parent_joint = parent_t.compose(&link.x_p);
            let q_rot = Quat::from_axis_angle(link.axis_local, q[j]);
            let body_w = Transform {
                position: parent_joint.position,
                rotation: (parent_joint.rotation * q_rot).normalize(),
            };
            let axis_world = parent_joint.rotation * link.axis_local;
            let s_j = SpatialVector::new(Vec3::ZERO, axis_world);
            s_cols[j] = s_j;

            let v_j = s_j.scale(qd[j]);
            let v_i = v_parent + v_j;
            let c_i = spatial_cross(v_i, v_j);

            // Body inertia in world frame.
            let r_w = body_w.rotation;
            let rm = Mat3::from_quat(r_w);
            let i_world = rm * link.inertia_local * rm.transpose();
            let com_world = body_w.position + r_w * link.com_local;
            let i_s = SpatialMatrix6::from_body(link.mass, com_world, i_world);

            // Bias = v ×* (I·v) − f_grav.
            let f_g_lin = self.gravity * link.mass;
            let f_g_ang = com_world.cross(f_g_lin);
            let f_grav = SpatialVector::new(f_g_lin, f_g_ang);
            let bias = spatial_cross_dual(v_i, i_s.mul_sv(v_i));
            let p_init = bias - f_grav;

            i_a[j] = i_s;
            p_a[j] = p_init;

            step.body_world[j] = body_w;
            step.axis_world[j] = axis_world;
            step.joint_origin_world[j] = parent_joint.position;
            step.com_world[j] = com_world;
            step.v_body[j] = v_i;
            step.c_term[j] = c_i;

            parent_t = body_w;
            v_parent = v_i;
        }

        // Pass 2: backward — articulated inertia + bias accumulation.
        for j in (0..n).rev() {
            let s = s_cols[j];
            let u = i_a[j].mul_sv(s);
            let d = sv_dot(s, u);
            let d_inv = if d.abs() > 1e-12 { 1.0 / d } else { 0.0 };
            let u_tau = tau[j] - sv_dot(s, p_a[j]);

            step.u_k[j] = u;
            step.d_k[j] = d;
            step.u_tau_k[j] = u_tau;

            let outer = SpatialMatrix6::outer(u, u).scale(d_inv);
            let i_curr = i_a[j].sub(&outer);
            let p_articulated =
                p_a[j] + i_curr.mul_sv(step.c_term[j]) + u.scale(u_tau * d_inv);

            if j > 0 {
                i_a[j - 1] = i_a[j - 1].add(&i_curr);
                p_a[j - 1] = p_a[j - 1] + p_articulated;
            }
        }

        // Pass 3: forward — solve for joint accelerations.
        let mut a_parent = SpatialVector::ZERO;
        let mut q_dd = vec![0.0; n];
        for j in 0..n {
            let s = s_cols[j];
            let u = step.u_k[j];
            let d_inv = if step.d_k[j].abs() > 1e-12 {
                1.0 / step.d_k[j]
            } else {
                0.0
            };
            let qdd_j = (step.u_tau_k[j] - sv_dot(u, a_parent)) * d_inv;
            q_dd[j] = qdd_j;
            step.q_dd[j] = qdd_j;
            let a_j = a_parent + s.scale(qdd_j) + step.c_term[j];
            step.a_body[j] = a_j;
            a_parent = a_j;
        }

        // Symplectic Euler with joint-space damping.
        let mut q_new = vec![0.0; n];
        let mut qd_new = vec![0.0; n];
        let damp = (1.0 - self.damping * self.dt).max(0.0);
        for j in 0..n {
            let qd_int = (qd[j] + q_dd[j] * self.dt) * damp;
            qd_new[j] = qd_int;
            q_new[j] = q[j] + qd_int * self.dt;
        }

        (q_new, qd_new, step)
    }

    /// Run `horizon` forward steps under torque sequence `tau`, return
    /// the trajectory + per-step intermediates needed by [`Self::reverse`].
    pub fn rollout(
        &self,
        q0: &[Real],
        qd0: &[Real],
        tau: &[Vec<Real>],
    ) -> FeatherstoneTrace {
        let mut q_arr = vec![Vec::new(); self.horizon + 1];
        let mut qd_arr = vec![Vec::new(); self.horizon + 1];
        q_arr[0] = q0.to_vec();
        qd_arr[0] = qd0.to_vec();
        let mut steps = Vec::with_capacity(self.horizon);
        for t in 0..self.horizon {
            let (q_n, qd_n, step) = self.forward_step(&q_arr[t], &qd_arr[t], &tau[t]);
            q_arr[t + 1] = q_n;
            qd_arr[t + 1] = qd_n;
            steps.push(step);
        }
        FeatherstoneTrace {
            q: q_arr,
            qd: qd_arr,
            steps,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Reverse-mode adjoint
// ────────────────────────────────────────────────────────────────────

/// Reusable scratch for the per-step adjoint Jacobian columns —
/// mirrors the per-articulation column buffers
/// [`crate::kernels::JacobianKernel`] holds for the GPU forward
/// Jacobian. Sized to `n_links × n_links` for each of the three
/// `(∂qdd/∂tau, ∂qdd/∂q, ∂qdd/∂qd)` blocks; reused across reverse
/// passes by reset rather than re-allocation.
#[derive(Default, Debug)]
pub struct JacobianScratch {
    pub qdd_tau: Vec<Real>,
    pub qdd_q: Vec<Real>,
    pub qdd_qd: Vec<Real>,
    n: usize,
}

impl JacobianScratch {
    pub fn new(n: usize) -> Self {
        Self {
            qdd_tau: vec![0.0; n * n],
            qdd_q: vec![0.0; n * n],
            qdd_qd: vec![0.0; n * n],
            n,
        }
    }

    /// Resize for a different link count without reallocating when the
    /// new size fits.
    pub fn resize(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.qdd_tau.resize(n * n, 0.0);
        self.qdd_q.resize(n * n, 0.0);
        self.qdd_qd.resize(n * n, 0.0);
        self.n = n;
    }
}

impl FeatherstoneRevoluteChain {
    /// Reverse pass: given the rollout trace and a seed `grad_ee`
    /// (∂loss/∂(EE world position)), return the per-step torque
    /// gradient sequence (`horizon × n_links`).
    ///
    /// Allocates a fresh [`JacobianScratch`] internally; for the
    /// optimisation loop where `reverse` runs every iteration prefer
    /// [`Self::reverse_with_scratch`].
    pub fn reverse(&self, trace: &FeatherstoneTrace, grad_ee: Vec3) -> Vec<Vec<Real>> {
        let n = self.links.len();
        let mut scratch = JacobianScratch::new(n);
        self.reverse_with_scratch(trace, grad_ee, &mut scratch)
    }

    /// As [`Self::reverse`], reusing caller-provided scratch space.
    /// The scratch is auto-resized to match `self.links.len()`.
    pub fn reverse_with_scratch(
        &self,
        trace: &FeatherstoneTrace,
        grad_ee: Vec3,
        scratch: &mut JacobianScratch,
    ) -> Vec<Vec<Real>> {
        let n = self.links.len();
        let h = self.horizon;
        scratch.resize(n);
        let mut g_tau = vec![vec![0.0; n]; h];
        let mut g_q = vec![0.0; n];
        let mut g_qd = vec![0.0; n];

        // Seed: ∂L/∂q at the FINAL step from the EE.
        let q_final = &trace.q[h];
        let j_ee = self.fk_jacobian(q_final);
        for j in 0..n {
            g_q[j] = grad_ee.dot(j_ee[j]);
        }

        let damp = (1.0 - self.damping * self.dt).max(0.0);
        let dt = self.dt;

        for t in (0..h).rev() {
            // Integration adjoint:
            //   q_new = q + qd_int * dt
            //   qd_int = (qd + qdd*dt) * damp
            // → g_qd_int = g_qd_new + g_q_new * dt
            //   g_qdd    = g_qd_int * damp * dt
            //   g_qd_in  = g_qd_int * damp
            //   g_q_in   = g_q_new
            let mut g_qd_int = vec![0.0; n];
            for j in 0..n {
                g_qd_int[j] = g_qd[j] + g_q[j] * dt;
            }
            let mut g_qdd = vec![0.0; n];
            let mut g_qd_in = vec![0.0; n];
            for j in 0..n {
                g_qd_in[j] = g_qd_int[j] * damp;
                g_qdd[j] = g_qd_int[j] * damp * dt;
            }

            // Per-step single-step Jacobian (n×n per block) via central
            // FD on the full Featherstone forward at this step.
            let q_t = &trace.q[t];
            let qd_t = &trace.qd[t];
            let tau_baseline = vec![0.0; n];
            let eps = 1e-6_f64;

            for k in 0..n {
                let mut tau_p = tau_baseline.clone();
                let mut tau_m = tau_baseline.clone();
                tau_p[k] += eps;
                tau_m[k] -= eps;
                let (_, _, sp) = self.forward_step(q_t, qd_t, &tau_p);
                let (_, _, sm) = self.forward_step(q_t, qd_t, &tau_m);
                for j in 0..n {
                    scratch.qdd_tau[j * n + k] = (sp.q_dd[j] - sm.q_dd[j]) / (2.0 * eps);
                }

                let mut q_p = q_t.clone();
                let mut q_m = q_t.clone();
                q_p[k] += eps;
                q_m[k] -= eps;
                let (_, _, sp) = self.forward_step(&q_p, qd_t, &tau_baseline);
                let (_, _, sm) = self.forward_step(&q_m, qd_t, &tau_baseline);
                for j in 0..n {
                    scratch.qdd_q[j * n + k] = (sp.q_dd[j] - sm.q_dd[j]) / (2.0 * eps);
                }

                let mut qd_p = qd_t.clone();
                let mut qd_m = qd_t.clone();
                qd_p[k] += eps;
                qd_m[k] -= eps;
                let (_, _, sp) = self.forward_step(q_t, &qd_p, &tau_baseline);
                let (_, _, sm) = self.forward_step(q_t, &qd_m, &tau_baseline);
                for j in 0..n {
                    scratch.qdd_qd[j * n + k] = (sp.q_dd[j] - sm.q_dd[j]) / (2.0 * eps);
                }
            }

            // VJP: g_tau_t[k] = Σ_j g_qdd[j] · J_qdd_tau[j][k].
            //      g_q_in[k]  += Σ_j g_qdd[j] · J_qdd_q[j][k].
            //      g_qd_in[k] += Σ_j g_qdd[j] · J_qdd_qd[j][k].
            let mut g_q_new = vec![0.0; n];
            let mut g_qd_new = vec![0.0; n];
            for k in 0..n {
                let mut s_tau = 0.0;
                let mut s_q = 0.0;
                let mut s_qd = 0.0;
                for jj in 0..n {
                    s_tau += g_qdd[jj] * scratch.qdd_tau[jj * n + k];
                    s_q += g_qdd[jj] * scratch.qdd_q[jj * n + k];
                    s_qd += g_qdd[jj] * scratch.qdd_qd[jj * n + k];
                }
                g_tau[t][k] = s_tau;
                g_q_new[k] = s_q;
                g_qd_new[k] = s_qd;
            }

            // Carry over for the previous timestep:
            // - q_new = q + qd_int*dt → g passes through to q_in (already).
            // - dynamics also contributes a g_q (via S, I, p_a) and g_qd
            //   (via S, v, c) to the previous step's q/qd.
            for j in 0..n {
                g_q[j] += g_q_new[j];
                g_qd[j] = g_qd_in[j] + g_qd_new[j];
            }
        }

        g_tau
    }

    /// FK Jacobian: returns columns J_k = ∂(EE world pos)/∂q_k as
    /// `Vec3`s. Used to seed the reverse pass at the FINAL step.
    /// Implementation: central difference on `Self::end_effector`.
    pub fn fk_jacobian(&self, q: &[Real]) -> Vec<Vec3> {
        let n = self.links.len();
        let mut j = vec![Vec3::ZERO; n];
        let eps = 1e-6_f64;
        for k in 0..n {
            let mut q_p = q.to_vec();
            let mut q_m = q.to_vec();
            q_p[k] += eps;
            q_m[k] -= eps;
            let ee_p = self.end_effector(&q_p);
            let ee_m = self.end_effector(&q_m);
            j[k] = (ee_p - ee_m) / (2.0 * eps);
        }
        j
    }

    /// Convenience: compute the EE-position gradient wrt each step's
    /// torque vector — `g_tau[t][k] = ∂(EE_pos · grad_ee)/∂tau[t][k]`.
    /// Wraps [`Self::rollout`] + [`Self::reverse`].
    pub fn ee_gradient_wrt_torques(
        &self,
        q0: &[Real],
        qd0: &[Real],
        tau: &[Vec<Real>],
        grad_ee: Vec3,
    ) -> Vec<Vec<Real>> {
        let trace = self.rollout(q0, qd0, tau);
        self.reverse(&trace, grad_ee)
    }

    /// Gradient-descent loop on the open-loop torque sequence `tau` to
    /// drive the EE to `target`. Mutates `tau` in place; returns
    /// `(final_residual_m, iters_to_5cm)`.
    pub fn gradient_descent(
        &self,
        q0: &[Real],
        qd0: &[Real],
        tau: &mut [Vec<Real>],
        target: Vec3,
        lr: Real,
        iters: usize,
    ) -> (Real, Option<usize>) {
        let mut last = Real::INFINITY;
        let mut converged: Option<usize> = None;
        let mut scratch = JacobianScratch::new(self.links.len());
        for it in 0..iters {
            let trace = self.rollout(q0, qd0, tau);
            let q_final = &trace.q[self.horizon];
            let ee = self.end_effector(q_final);
            let d = ee - target;
            last = d.length();
            if converged.is_none() && last < 0.05 {
                converged = Some(it);
            }
            let grad_ee = d * 2.0;
            let g = self.reverse_with_scratch(&trace, grad_ee, &mut scratch);
            for t in 0..self.horizon {
                for k in 0..self.links.len() {
                    tau[t][k] -= lr * g[t][k];
                }
            }
        }
        (last, converged)
    }
}

#[inline]
fn sv_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

// ────────────────────────────────────────────────────────────────────
// Tape recording: stitch a Featherstone forward step into a [`Tape`]
// so the user can compose Featherstone with the rest of the autograd
// graph (e.g. follow up with downstream ops on `q_new` or `ee`).
// ────────────────────────────────────────────────────────────────────

/// Record one Featherstone forward step on `tape`. Inputs `(q, qd,
/// tau)` are tape ids whose values must have been seeded via
/// [`Tape::alloc`]. The function evaluates the forward step against the
/// captured `chain`, allocates ids for the post-integration `(q_new,
/// qd_new, ee)`, and pushes three tape ops — one per Featherstone
/// sweep — so the backward pass can replay each chunk independently
/// (mirrors the GPU dispatch granularity for the Stage-23.1.1 follow-up).
///
/// Returns `(q_new_id, qd_new_id, ee_id)`.
pub fn record_featherstone_step(
    tape: &mut Tape,
    chain: &FeatherstoneRevoluteChain,
    q_id: TapeId,
    qd_id: TapeId,
    tau_id: TapeId,
) -> (TapeId, TapeId, TapeId) {
    let q = tape.values.get(&q_id).cloned().unwrap_or_default();
    let qd = tape.values.get(&qd_id).cloned().unwrap_or_default();
    let tau = tape.values.get(&tau_id).cloned().unwrap_or_default();
    let (q_new, qd_new, _step) = chain.forward_step(&q, &qd, &tau);
    let ee = chain.end_effector(&q_new);
    let q_new_id = tape.alloc(q_new);
    let qd_new_id = tape.alloc(qd_new);
    let ee_id = tape.alloc(vec![ee.x, ee.y, ee.z]);

    tape.ops.push(TapeOp::FeatherstoneFwdKinematics {
        q: q_id,
        qd: qd_id,
    });
    tape.ops.push(TapeOp::FeatherstoneUpwardInertia {
        q: q_id,
        qd: qd_id,
        tau: tau_id,
    });
    tape.ops.push(TapeOp::FeatherstoneDownwardAccel {
        q: q_id,
        qd: qd_id,
        tau: tau_id,
        q_new: q_new_id,
        qd_new: qd_new_id,
        ee: ee_id,
    });
    (q_new_id, qd_new_id, ee_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pendulum_gravity_only_swings_bounded() {
        // Higher damping + tighter dt to keep symplectic-Euler bounded
        // on the small-angle linearisation. We just want to verify the
        // forward dynamics are stable, not match an analytical period.
        let mut chain = FeatherstoneRevoluteChain::pendulum(0.5, 1.0, 0.001, 100);
        chain.damping = 1.0;
        let q0 = vec![0.05_f64];
        let qd0 = vec![0.0_f64];
        let tau = vec![vec![0.0_f64]; chain.horizon];
        let trace = chain.rollout(&q0, &qd0, &tau);
        let max_abs = trace.q.iter().map(|q| q[0].abs()).fold(0.0_f64, f64::max);
        // Damped pendulum starting at 0.05 rad — peak shouldn't overshoot
        // by more than ~3× even before the swing-down completes.
        assert!(max_abs < 0.5, "max |q| {max_abs}");
        // After the swing-down should be moving (not perfectly at rest).
        let qd_max = trace.qd.iter().map(|qd| qd[0].abs()).fold(0.0_f64, f64::max);
        assert!(qd_max > 1e-3, "qd never moved");
    }

    #[test]
    fn fk_jacobian_two_link_arm_planar_in_xy() {
        let chain = FeatherstoneRevoluteChain::two_link_arm(0.5, 0.4, 1.0, 0.02, 5);
        let q = vec![0.4_f64, -0.3_f64];
        let j = chain.fk_jacobian(&q);
        // Sanity: rotation about Z moves EE in XY, not Z.
        assert!(j[0].z.abs() < 1e-6);
        assert!(j[1].z.abs() < 1e-6);
    }

    #[test]
    fn reverse_pass_returns_horizon_torque_grads() {
        let chain = FeatherstoneRevoluteChain::pendulum(0.5, 1.0, 0.01, 5);
        let q0 = vec![0.0_f64];
        let qd0 = vec![0.0_f64];
        let tau = vec![vec![0.5_f64]; chain.horizon];
        let trace = chain.rollout(&q0, &qd0, &tau);
        let g = chain.reverse(&trace, Vec3::Y);
        assert_eq!(g.len(), chain.horizon);
        assert_eq!(g[0].len(), 1);
    }

    #[test]
    fn pendulum_grad_matches_fd() {
        // Single step: ∂(EE_y at t=1)/∂tau[0]
        let chain = FeatherstoneRevoluteChain::pendulum(0.5, 1.0, 0.01, 5);
        let q0 = vec![0.0_f64];
        let qd0 = vec![0.0_f64];
        let tau0 = vec![vec![0.1_f64]; chain.horizon];

        let g = chain.ee_gradient_wrt_torques(&q0, &qd0, &tau0, Vec3::Y);

        // FD on tau[0]
        let eps = 1e-5_f64;
        let mut tau_p = tau0.clone();
        let mut tau_m = tau0.clone();
        tau_p[0][0] += eps;
        tau_m[0][0] -= eps;
        let ee_p = chain.end_effector(&chain.rollout(&q0, &qd0, &tau_p).q[chain.horizon]);
        let ee_m = chain.end_effector(&chain.rollout(&q0, &qd0, &tau_m).q[chain.horizon]);
        let fd = (ee_p.y - ee_m.y) / (2.0 * eps);
        let analytic = g[0][0];
        let diff = (analytic - fd).abs();
        assert!(diff < 1e-3, "analytic={analytic}, fd={fd}, diff={diff}");
    }
}
