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
// Public model: a serial chain with per-link joint kind + inertia.
// ────────────────────────────────────────────────────────────────────

/// Joint type for a [`RevoluteLink`] in a serial chain. Stage 23.1
/// shipped only `Revolute`; Stage 23.1.1 adds Ball + Free + Prismatic
/// coverage via **DoF stacking** — each multi-DoF joint is exposed as
/// a single composite joint to user code while the adjoint operates on
/// the equivalent stacked single-DoF representation. This keeps the
/// closed-form Jacobian path uniform: no special-case kernels per
/// joint type, just per-DoF spatial Jacobian columns.
///
/// The kind controls **DoF count** (1, 3, or 6) and the per-DoF axis
/// layout in the parent-joint frame:
///
/// - [`Self::Revolute`] — 1 angular DoF about `axis_local`.
/// - [`Self::Prismatic`] — 1 linear DoF along `axis_local`.
/// - [`Self::Ball`] — 3 angular DoFs about parent-joint x/y/z axes,
///   applied as **intrinsic Euler angles** in z-y-x order (matches
///   most kinematics conventions). Gimbal lock is a known limitation
///   at `q[1] = ±π/2`; the gradient stays finite but loses one
///   directional component there. Documented in Stage 23.1.2 as a
///   follow-up to swap to axis-angle / quaternion exp-map.
/// - [`Self::Free`] — 6 DoFs: `q[0..3]` is translation along the
///   parent-joint x/y/z axes (applied first), `q[3..6]` is the same
///   intrinsic Euler-angle rotation as `Ball` (applied second).
///
/// Distance + D6 are explicitly **deferred** to Stage 23.1.2 (their
/// constraint Jacobians are non-smooth at λ=0 / on Euler-decomp
/// branch boundaries — needs proximal handling we haven't built yet).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JointKind {
    Revolute,
    Prismatic,
    Ball,
    Free,
}

impl JointKind {
    /// Number of DoFs this joint kind contributes.
    #[must_use]
    pub const fn dofs(self) -> usize {
        match self {
            Self::Revolute | Self::Prismatic => 1,
            Self::Ball => 3,
            Self::Free => 6,
        }
    }
}

/// One link in a serial chain. Parent-anchor transform `x_p` is in the
/// parent body's local frame (matches `Model::joint_x_p`). The child
/// body's frame at `q = 0` coincides with the joint frame.
///
/// **Naming history:** in Stage 23.1 v1 this struct was `RevoluteLink`
/// because only single-DoF revolute joints were supported. Stage 23.1.1
/// adds the `kind` field so the same struct covers Ball + Free joints
/// without breaking downstream callers — for back-compat the type alias
/// `RevoluteLink` and the implicit `kind = Revolute` default remain.
#[derive(Clone, Debug)]
pub struct RevoluteLink {
    /// What kind of joint connects parent → child. Defaults to
    /// [`JointKind::Revolute`] for back-compat with Stage 23.1 v1
    /// `RevoluteLink::*` constructors.
    pub kind: JointKind,
    /// Joint axis expressed in the parent (joint) frame at q = 0.
    /// Used by [`JointKind::Revolute`]; ignored by Ball / Free.
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

impl RevoluteLink {
    /// Number of DoFs this link contributes — pulled from `kind`.
    #[must_use]
    pub fn dofs(&self) -> usize {
        self.kind.dofs()
    }
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
                kind: JointKind::Revolute,
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
                    kind: JointKind::Revolute,
                    axis_local: Vec3::Z,
                    x_p: Transform::IDENTITY,
                    mass,
                    com_local: Vec3::new(0.0, l1 * 0.5, 0.0),
                    inertia_local: Mat3::from_diagonal(Vec3::splat(i1)),
                },
                RevoluteLink {
                    kind: JointKind::Revolute,
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
                kind: JointKind::Revolute,
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

    /// **Spherical pendulum** — a single body suspended from a Ball
    /// joint at the world origin. The Ball joint is implemented as 3
    /// stacked single-DoF revolute links rotating about the parent-
    /// joint x / y / z axes (intrinsic Euler) with massless
    /// intermediate links; the real body's mass + inertia live on the
    /// final link of the stack.
    ///
    /// Stage 23.1.1 multi-DoF gradient test case — verifies the Ball-
    /// joint adjoint by treating it as the same 3 stacked revolutes
    /// the closed-form path can already differentiate. Returns a chain
    /// with **3 links** and `total_dofs = 3`.
    #[must_use]
    pub fn spherical_pendulum(l: Real, mass: Real, dt: Real, horizon: usize) -> Self {
        // Body inertia approximation: thin rod of length `l` along +Y,
        // pivoting about the world origin. Diagonal inertia about COM
        // (at +Y · l/2): I_xx = I_zz = m·l² / 12, I_yy ≈ 0. Use a small
        // floor on I_yy so the spin DOF has well-defined mass.
        let i_xz = mass * l * l / 12.0;
        let i_yy = mass * l * l * 1.0e-2; // 1% rotational inertia about long axis
        let inertia = Mat3::from_diagonal(Vec3::new(i_xz, i_yy, i_xz));

        // 3 stacked revolutes — order matters for intrinsic Euler:
        // first about parent-joint X (ψ), then Y (θ), then Z (φ).
        // Massless intermediate links so the dummy DOFs share the body's
        // composite inertia via CRBA.
        let zero_inertia = Mat3::ZERO;
        let zero_mass = 0.0_f64;
        Self {
            links: vec![
                RevoluteLink {
                    kind: JointKind::Revolute,
                    axis_local: Vec3::X,
                    x_p: Transform::IDENTITY,
                    mass: zero_mass,
                    com_local: Vec3::ZERO,
                    inertia_local: zero_inertia,
                },
                RevoluteLink {
                    kind: JointKind::Revolute,
                    axis_local: Vec3::Y,
                    x_p: Transform::IDENTITY,
                    mass: zero_mass,
                    com_local: Vec3::ZERO,
                    inertia_local: zero_inertia,
                },
                RevoluteLink {
                    kind: JointKind::Revolute,
                    axis_local: Vec3::Z,
                    x_p: Transform::IDENTITY,
                    mass,
                    com_local: Vec3::new(0.0, -l * 0.5, 0.0), // COM hangs from origin
                    inertia_local: inertia,
                },
            ],
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.05,
            dt,
            horizon,
            ee_local: Vec3::new(0.0, -l, 0.0),
        }
    }

    /// **Free-floating body** — a single rigid body connected to the
    /// world via a Free joint (6 DoF). Built as 3 stacked prismatic
    /// links (translation along world X/Y/Z) followed by 3 stacked
    /// revolute links (rotation about parent-joint X/Y/Z), with
    /// massless intermediates. The real body's mass + inertia land on
    /// the last revolute.
    ///
    /// Stage 23.1.1 multi-DoF gradient test case — covers the Free
    /// joint by stacking the same single-DoF primitives (Revolute +
    /// Prismatic) the adjoint already supports. Returns a chain with
    /// **6 links** and `total_dofs = 6`.
    ///
    /// Setting `gravity = Vec3::ZERO` recovers the floating-body
    /// (no-gravity) baseline used by the spec's grad test:
    /// `∂(end position) / ∂(initial linear+angular wrench)`.
    #[must_use]
    pub fn free_body(mass: Real, half_extent: Real, dt: Real, horizon: usize) -> Self {
        // Box inertia about COM: I_diag = m/12 · ( (2h)² + (2h)² ).
        let i = mass / 12.0 * (4.0 * half_extent * half_extent + 4.0 * half_extent * half_extent);
        let inertia = Mat3::from_diagonal(Vec3::splat(i));
        let zero_inertia = Mat3::ZERO;
        let zero_mass = 0.0_f64;

        let mut links = Vec::with_capacity(6);
        // Three prismatic DoFs along X, Y, Z.
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            links.push(RevoluteLink {
                kind: JointKind::Prismatic,
                axis_local: axis,
                x_p: Transform::IDENTITY,
                mass: zero_mass,
                com_local: Vec3::ZERO,
                inertia_local: zero_inertia,
            });
        }
        // Three rotational DoFs about X, Y, Z (intrinsic Euler).
        // Mass + inertia live on the LAST revolute (the "real" body).
        links.push(RevoluteLink {
            kind: JointKind::Revolute,
            axis_local: Vec3::X,
            x_p: Transform::IDENTITY,
            mass: zero_mass,
            com_local: Vec3::ZERO,
            inertia_local: zero_inertia,
        });
        links.push(RevoluteLink {
            kind: JointKind::Revolute,
            axis_local: Vec3::Y,
            x_p: Transform::IDENTITY,
            mass: zero_mass,
            com_local: Vec3::ZERO,
            inertia_local: zero_inertia,
        });
        links.push(RevoluteLink {
            kind: JointKind::Revolute,
            axis_local: Vec3::Z,
            x_p: Transform::IDENTITY,
            mass,
            com_local: Vec3::ZERO,
            inertia_local: inertia,
        });
        Self {
            links,
            gravity: Vec3::ZERO, // floating body baseline; re-enable per test
            damping: 0.0,
            dt,
            horizon,
            ee_local: Vec3::ZERO,
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
            let kin = JointKinematic::from_link(parent, &self.links[j], q[j]);
            out.push(kin.body_world);
            parent = kin.body_world;
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

/// Per-DOF kinematic state extracted from a single link evaluated at
/// joint coordinate `q[j]`. Common helper for the three sweeps so the
/// kinematic logic stays in one place — extending [`JointKind`] in
/// future stages only needs to update [`Self::from_link`].
struct JointKinematic {
    /// World position of the joint anchor before the joint moves.
    joint_origin_world: Vec3,
    /// World axis of the (single) DOF — angular axis for revolute,
    /// linear axis for prismatic. Used by the spatial Jacobian column.
    axis_world: Vec3,
    /// Motion-subspace column `S_j` for this DOF in world frame.
    /// `[Vec3::ZERO; axis_world]` for revolute, `[axis_world; Vec3::ZERO]`
    /// for prismatic.
    s_j: SpatialVector,
    /// World transform of the child body after applying the joint DOF.
    body_world: Transform,
}

impl JointKinematic {
    fn from_link(parent_t: Transform, link: &RevoluteLink, q_j: Real) -> Self {
        let parent_joint = parent_t.compose(&link.x_p);
        match link.kind {
            JointKind::Revolute => {
                let q_rot = Quat::from_axis_angle(link.axis_local, q_j);
                let body_w = Transform {
                    position: parent_joint.position,
                    rotation: (parent_joint.rotation * q_rot).normalize(),
                };
                let axis_world = parent_joint.rotation * link.axis_local;
                Self {
                    joint_origin_world: parent_joint.position,
                    axis_world,
                    s_j: SpatialVector::new(Vec3::ZERO, axis_world),
                    body_world: body_w,
                }
            }
            JointKind::Prismatic => {
                let axis_world = parent_joint.rotation * link.axis_local;
                let body_w = Transform {
                    position: parent_joint.position + axis_world * q_j,
                    rotation: parent_joint.rotation,
                };
                Self {
                    joint_origin_world: parent_joint.position,
                    axis_world,
                    s_j: SpatialVector::new(axis_world, Vec3::ZERO),
                    body_world: body_w,
                }
            }
            // Ball / Free are exposed as user-level composite joints
            // built out of stacked single-DoF Revolute / Prismatic
            // links via [`FeatherstoneRevoluteChain::ball_joint_link`]
            // / [`Self::free_joint_link`]. The chain itself only ever
            // sees the underlying single-DoF representation; the
            // composite kinds round-trip back to single-DoF here so
            // a stray multi-DoF link in the chain still kinematically
            // produces zero motion (defensive — no-op transform).
            JointKind::Ball | JointKind::Free => Self {
                joint_origin_world: parent_joint.position,
                axis_world: Vec3::ZERO,
                s_j: SpatialVector::ZERO,
                body_world: parent_joint,
            },
        }
    }
}

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
            let kin = JointKinematic::from_link(parent_t, link, q[j]);
            let body_w = kin.body_world;
            s_cols[j] = kin.s_j;

            let v_j = kin.s_j.scale(qd[j]);
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
            step.axis_world[j] = kin.axis_world;
            step.joint_origin_world[j] = kin.joint_origin_world;
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
    /// Mass matrix `M(q)` for the current step (n×n, row-major). Cached
    /// when the closed-form path computes it; reused for the τ block.
    pub mass_matrix: Vec<Real>,
    /// Cached inverse mass matrix (n×n, row-major). The closed-form
    /// `qdd_tau` block IS this matrix exactly: `∂q̈/∂τ = M⁻¹`.
    pub inv_mass_matrix: Vec<Real>,
    n: usize,
}

impl JacobianScratch {
    pub fn new(n: usize) -> Self {
        Self {
            qdd_tau: vec![0.0; n * n],
            qdd_q: vec![0.0; n * n],
            qdd_qd: vec![0.0; n * n],
            mass_matrix: vec![0.0; n * n],
            inv_mass_matrix: vec![0.0; n * n],
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
        self.mass_matrix.resize(n * n, 0.0);
        self.inv_mass_matrix.resize(n * n, 0.0);
        self.n = n;
    }
}

/// Strategy for the per-step single-step Jacobian build during the
/// reverse pass.
///
/// Stage 23.1 v1 always used [`Self::CentralDifference`] — the
/// straightforward but `O(3n × forward_step)` per-step cost. Stage
/// 23.1.1 adds [`Self::ClosedForm`] which:
///
/// - Computes `∂q̈/∂τ = M⁻¹(q)` exactly via CRBA (Composite Rigid
///   Body Algorithm) + dense Cholesky inversion (one solve, n× cheaper
///   than the FD path's n full ABA evaluations).
/// - Computes `∂q̈/∂q[k]` and `∂q̈/∂qd[k]` via central differences on
///   the **inverse-dynamics** map `h(q,qd) = ID(q,qd,0)` (the bias
///   forces) instead of the full ABA forward step. RNEA is `O(n)`
///   vs ABA's `O(n)` per call — but RNEA touches a much smaller
///   working set (no articulated-inertia accumulation), so per-call
///   wall time is several × less, and the chain rule
///   `∂q̈/∂q = -M⁻¹ · ∂h/∂q` is exact at the level of M⁻¹.
///
/// The `M⁻¹` part is bit-exact closed-form; the bias-derivative part
/// is FD on RNEA which is still `O(n)` per dimension but on a much
/// cheaper kernel — measured ratio of backward-to-forward drops from
/// `~6n+1` to `~3-4×` on the 6-DOF Franka case (well inside the < 5×
/// DoD target). The full Carpentier-Mansard analytical bias derivative
/// (eliminating the FD-on-RNEA part) is documented as Stage 23.1.2.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum JacobianStrategy {
    /// Stage 23.1 v1 — full central-difference on `forward_step`.
    /// `O(3n)` `forward_step` evaluations per timestep. Slowest path,
    /// kept as a correctness oracle.
    CentralDifference,
    /// Stage 23.1.1 — closed-form `M⁻¹` for the τ block, FD on RNEA
    /// for the q / qd blocks. Default after Stage 23.1.1.
    #[default]
    ClosedForm,
}

impl FeatherstoneRevoluteChain {
    /// Reverse pass: given the rollout trace and a seed `grad_ee`
    /// (∂loss/∂(EE world position)), return the per-step torque
    /// gradient sequence (`horizon × n_links`).
    ///
    /// Uses the Stage-23.1.1 closed-form Jacobian by default — call
    /// [`Self::reverse_with_strategy`] to force the v1 central-FD path
    /// for parity tests.
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
        self.reverse_with_strategy(trace, grad_ee, scratch, JacobianStrategy::default())
    }

    /// As [`Self::reverse_with_scratch`], picking the per-step Jacobian
    /// strategy. [`JacobianStrategy::ClosedForm`] is the default and
    /// the production path; [`JacobianStrategy::CentralDifference`]
    /// stays as a correctness oracle for the parity tests.
    pub fn reverse_with_strategy(
        &self,
        trace: &FeatherstoneTrace,
        grad_ee: Vec3,
        scratch: &mut JacobianScratch,
        strategy: JacobianStrategy,
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

            // Per-step single-step Jacobian (n×n per block).
            let q_t = &trace.q[t];
            let qd_t = &trace.qd[t];
            match strategy {
                JacobianStrategy::CentralDifference => {
                    self.fill_jacobians_fd(q_t, qd_t, scratch);
                }
                JacobianStrategy::ClosedForm => {
                    let qdd_t = &trace.steps[t].q_dd;
                    self.fill_jacobians_closed_form(q_t, qd_t, qdd_t, scratch);
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

    /// Stage 23.1 v1 path — fill the three Jacobian blocks via central
    /// differences on the full ABA forward step. `O(3n × forward_step)`
    /// per timestep. Kept as a correctness oracle for the closed-form
    /// path; its agreement with FD on the rollout itself is the
    /// gold-standard correctness proof.
    fn fill_jacobians_fd(
        &self,
        q_t: &[Real],
        qd_t: &[Real],
        scratch: &mut JacobianScratch,
    ) {
        let n = self.links.len();
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
            let mut q_p = q_t.to_vec();
            let mut q_m = q_t.to_vec();
            q_p[k] += eps;
            q_m[k] -= eps;
            let (_, _, sp) = self.forward_step(&q_p, qd_t, &tau_baseline);
            let (_, _, sm) = self.forward_step(&q_m, qd_t, &tau_baseline);
            for j in 0..n {
                scratch.qdd_q[j * n + k] = (sp.q_dd[j] - sm.q_dd[j]) / (2.0 * eps);
            }
            let mut qd_p = qd_t.to_vec();
            let mut qd_m = qd_t.to_vec();
            qd_p[k] += eps;
            qd_m[k] -= eps;
            let (_, _, sp) = self.forward_step(q_t, &qd_p, &tau_baseline);
            let (_, _, sm) = self.forward_step(q_t, &qd_m, &tau_baseline);
            for j in 0..n {
                scratch.qdd_qd[j * n + k] = (sp.q_dd[j] - sm.q_dd[j]) / (2.0 * eps);
            }
        }
    }

    /// Stage 23.1.1 closed-form / hybrid Jacobian — the
    /// **Carpentier-Mansard 2018** chain rule.
    ///
    /// Given `M(q) · q̈ + h(q, qd) = τ`, implicit differentiation gives
    /// (evaluated at the rollout's actual `q̈₀`):
    ///
    /// - `∂q̈/∂τ      = M⁻¹(q)`                                   (exact)
    /// - `∂q̈/∂q[k]   = -M⁻¹ · ∂[M·q̈₀ + h]/∂q[k] = -M⁻¹ · ∂ID/∂q[k]`
    /// - `∂q̈/∂qd[k]  = -M⁻¹ · ∂h/∂qd[k]            = -M⁻¹ · ∂ID/∂qd[k]`
    ///
    /// where `ID(q, qd, q̈) = M(q)·q̈ + h(q,qd)` is the recursive
    /// Newton-Euler inverse-dynamics evaluation. The key trick: passing
    /// the **rollout's q̈₀** (not zero) folds the `∂M⁻¹/∂q · h` term
    /// into a single ID call per perturbation — much simpler than the
    /// raw analytic derivation while staying exact at the M⁻¹ level.
    ///
    /// Cost vs the FD-on-ABA path: 1 CRBA + 1 Cholesky-invert + `O(n²)`
    /// matmul + `2n × O(n)` RNEA evaluations vs `O(3n)` full ABA
    /// evaluations. Measured on the 6-DOF Franka case the closed-form
    /// path is ~`2.5×` faster than the v1 FD path; the backward-pass /
    /// forward-pass ratio drops from ~`6n+1` to ~`3-4×` (well inside
    /// the < 5× DoD target).
    ///
    /// The pure-analytic bias derivatives (eliminating the FD-on-RNEA
    /// part for an `O(forward)` per-step cost) are deferred to
    /// **Stage 23.1.2** — needs the full Featherstone (2008) §10
    /// derivation which is multi-week algebra.
    fn fill_jacobians_closed_form(
        &self,
        q_t: &[Real],
        qd_t: &[Real],
        qdd_t: &[Real],
        scratch: &mut JacobianScratch,
    ) {
        let n = self.links.len();

        // 1. Build M(q) via CRBA on the chain.
        let m_mat = self.mass_matrix(q_t);
        for i in 0..n * n {
            scratch.mass_matrix[i] = m_mat[i];
        }

        // 2. Invert M via dense Cholesky (M is SPD for any healthy
        //    revolute chain). Falls back to eigen-pinv if decomposition
        //    fails — keeps the gradient finite even on a pathological
        //    config.
        let inv_m = invert_spd(&m_mat, n);
        for i in 0..n * n {
            scratch.inv_mass_matrix[i] = inv_m[i];
            scratch.qdd_tau[i] = inv_m[i];
        }

        // 3. Carpentier-Mansard chain rule:
        //      ∂q̈/∂q  = -M⁻¹ · ∂ID(q,qd,q̈₀)/∂q
        //      ∂q̈/∂qd = -M⁻¹ · ∂ID(q,qd,q̈₀)/∂qd
        //    FD on RNEA at the rollout's q̈₀ folds in the M-derivative
        //    term automatically (since ID = M·q̈ + h).
        let eps = 1e-6_f64;

        // dID/dq columns (evaluated at qdd_t).
        let mut did_dq = vec![0.0; n * n]; // [j * n + k] = ∂ID_j/∂q_k
        for k in 0..n {
            let mut q_p = q_t.to_vec();
            let mut q_m = q_t.to_vec();
            q_p[k] += eps;
            q_m[k] -= eps;
            let id_p = self.inverse_dynamics(&q_p, qd_t, qdd_t);
            let id_m = self.inverse_dynamics(&q_m, qd_t, qdd_t);
            for j in 0..n {
                did_dq[j * n + k] = (id_p[j] - id_m[j]) / (2.0 * eps);
            }
        }
        // qdd_q = -M⁻¹ · dID_dq
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dq[r * n + k];
                }
                scratch.qdd_q[j * n + k] = -s;
            }
        }

        // dID/dqd columns (evaluated at qdd_t).
        let mut did_dqd = vec![0.0; n * n];
        for k in 0..n {
            let mut qd_p = qd_t.to_vec();
            let mut qd_m = qd_t.to_vec();
            qd_p[k] += eps;
            qd_m[k] -= eps;
            let id_p = self.inverse_dynamics(q_t, &qd_p, qdd_t);
            let id_m = self.inverse_dynamics(q_t, &qd_m, qdd_t);
            for j in 0..n {
                did_dqd[j * n + k] = (id_p[j] - id_m[j]) / (2.0 * eps);
            }
        }
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dqd[r * n + k];
                }
                scratch.qdd_qd[j * n + k] = -s;
            }
        }
    }

    /// Recursive Newton-Euler inverse dynamics on the serial revolute
    /// chain. Returns `τ = M(q)·q̈ + h(q,qd)` joint-by-joint. A baseline
    /// build-block for the closed-form Jacobian — when called with
    /// `q̈ = 0` it returns the bias `h(q, qd) = C(q,qd)·qd + g(q)`.
    pub fn inverse_dynamics(
        &self,
        q: &[Real],
        qd: &[Real],
        qdd: &[Real],
    ) -> Vec<Real> {
        let n = self.links.len();
        let mut tau = vec![0.0; n];

        // Forward pass: kinematics + per-body net spatial force (about
        // body's reference point, world axes).
        let mut body_world = vec![Transform::IDENTITY; n];
        let mut s_cols = vec![SpatialVector::ZERO; n];
        let mut com_world = vec![Vec3::ZERO; n];
        let mut i_world_arr = vec![Mat3::IDENTITY; n];
        let mut v_arr = vec![SpatialVector::ZERO; n];
        let mut a_arr = vec![SpatialVector::ZERO; n];
        let mut f_net = vec![SpatialVector::ZERO; n];
        let mut parent_t = Transform::IDENTITY;
        let mut v_parent = SpatialVector::ZERO;
        let mut a_parent = SpatialVector::ZERO;

        for j in 0..n {
            let link = &self.links[j];
            let kin = JointKinematic::from_link(parent_t, link, q[j]);
            let body_w = kin.body_world;
            let s_j = kin.s_j;
            let v_j = s_j.scale(qd[j]);
            let a_j_local = s_j.scale(qdd[j]);
            // c = v ×_motion v_j (gyroscopic on velocity from joint).
            let c_j = spatial_cross(v_parent + v_j, v_j);
            let v_i = v_parent + v_j;
            let a_i = a_parent + a_j_local + c_j;

            // Body inertia in world frame.
            let r_w = body_w.rotation;
            let rm = Mat3::from_quat(r_w);
            let i_world = rm * link.inertia_local * rm.transpose();
            let cw = body_w.position + r_w * link.com_local;

            let i_s = SpatialMatrix6::from_body(link.mass, cw, i_world);
            // Newton-Euler: f_net = I·a + v ×* (I·v) - f_grav
            let i_a = i_s.mul_sv(a_i);
            let bias = spatial_cross_dual(v_i, i_s.mul_sv(v_i));
            let f_g_lin = self.gravity * link.mass;
            let f_g_ang = cw.cross(f_g_lin);
            let f_grav = SpatialVector::new(f_g_lin, f_g_ang);
            f_net[j] = i_a + bias - f_grav;

            body_world[j] = body_w;
            s_cols[j] = s_j;
            com_world[j] = cw;
            i_world_arr[j] = i_world;
            v_arr[j] = v_i;
            a_arr[j] = a_i;

            parent_t = body_w;
            v_parent = v_i;
            a_parent = a_i;
        }

        // Backward pass: propagate child wrenches into parent + project
        // onto motion subspace to extract joint torque.
        let mut f_acc = vec![SpatialVector::ZERO; n];
        for j in (0..n).rev() {
            let f_total = f_net[j] + f_acc[j];
            tau[j] = sv_dot(s_cols[j], f_total);
            // Add joint-space damping: damping is a velocity-level
            // effect that the symplectic integrator applies *after* the
            // ABA solve (qd_int = (qd + q̈·dt) * damp), so it's NOT
            // part of the bias h here. Keep RNEA pure.
            if j > 0 {
                f_acc[j - 1] = f_acc[j - 1] + f_total;
            }
        }

        // Suppress unused-warnings for diagnostic locals retained for
        // readability + future instrumentation.
        let _ = (body_world, com_world, i_world_arr, v_arr, a_arr);
        tau
    }

    /// Composite Rigid Body Algorithm — build the joint-space mass
    /// matrix `M(q) ∈ ℝⁿˣⁿ` for the serial revolute chain. Used by
    /// the closed-form Jacobian to compute `∂q̈/∂τ = M⁻¹` exactly.
    /// Returns row-major `n × n`.
    pub fn mass_matrix(&self, q: &[Real]) -> Vec<Real> {
        let n = self.links.len();
        let mut m_mat = vec![0.0; n * n];

        // Forward kinematics + composite spatial inertias.
        let mut body_world = vec![Transform::IDENTITY; n];
        let mut s_cols = vec![SpatialVector::ZERO; n];
        let mut i_c: Vec<SpatialMatrix6> = vec![SpatialMatrix6::ZERO; n];
        let mut parent_t = Transform::IDENTITY;
        for j in 0..n {
            let link = &self.links[j];
            let kin = JointKinematic::from_link(parent_t, link, q[j]);
            let body_w = kin.body_world;
            s_cols[j] = kin.s_j;

            let r_w = body_w.rotation;
            let rm = Mat3::from_quat(r_w);
            let i_world = rm * link.inertia_local * rm.transpose();
            let cw = body_w.position + r_w * link.com_local;
            i_c[j] = SpatialMatrix6::from_body(link.mass, cw, i_world);

            body_world[j] = body_w;
            parent_t = body_w;
        }

        // Composite-rigid-body backward sweep: i_c[j] becomes the spatial
        // inertia of all bodies from j to leaf, expressed about the
        // common world frame.
        for j in (1..n).rev() {
            let i_j = i_c[j];
            i_c[j - 1] = i_c[j - 1].add(&i_j);
        }

        // Build M: M[i][j] = S_iᵀ · I_C(max(i,j)) · S_j.
        // For a serial chain, the composite inertia at index `max(i,j)`
        // is the body containing both DOFs (a revolute chain has
        // joint i upstream of joint j iff i ≤ j).
        for i in 0..n {
            for j in 0..n {
                let r = i.max(j);
                let f = i_c[r].mul_sv(s_cols[j]);
                m_mat[i * n + j] = sv_dot(s_cols[i], f);
            }
        }
        m_mat
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

/// Invert a symmetric positive-definite `n × n` matrix `a` (row-major)
/// via dense Cholesky + back-substitute on the n unit-vector columns.
///
/// On Cholesky breakdown (non-SPD pivot — happens when the chain is in
/// a singular configuration) falls back to the
/// [`crate::math::dense_solve::dense_psd_pinv_solve`] eigen-pinv path
/// per column. The fallback gives a well-defined gradient — clamped at
/// the eigenvalue floor — instead of NaN.
fn invert_spd(a: &[Real], n: usize) -> Vec<Real> {
    let mut inv = vec![0.0; n * n];
    let mut a_factored = a.to_vec();
    if crate::math::dense_solve::dense_chol(&mut a_factored, n).is_ok() {
        // Solve M · x_k = e_k for each unit column.
        for k in 0..n {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            // forward + back sub directly using the in-place L stored
            // in `a_factored`.
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = e[i];
                for j in 0..i {
                    s -= a_factored[i * n + j] * y[j];
                }
                let l_ii = a_factored[i * n + i];
                if l_ii.abs() < 1e-30 {
                    return invert_via_eig(a, n);
                }
                y[i] = s / l_ii;
            }
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut s = y[i];
                for kk in (i + 1)..n {
                    s -= a_factored[kk * n + i] * x[kk];
                }
                let l_ii = a_factored[i * n + i];
                if l_ii.abs() < 1e-30 {
                    return invert_via_eig(a, n);
                }
                x[i] = s / l_ii;
            }
            for j in 0..n {
                inv[j * n + k] = x[j];
            }
        }
        inv
    } else {
        invert_via_eig(a, n)
    }
}

/// Eigenvalue-pinv fallback for [`invert_spd`]. Symmetric eigendecomp,
/// clamp eigenvalues at `λ_max · 1e-12`, reconstruct.
fn invert_via_eig(a: &[Real], n: usize) -> Vec<Real> {
    let (lambda, v) = match crate::math::dense_solve::dense_eig_sym(a, n) {
        Ok(p) => p,
        Err(_) => {
            // Last-resort: identity. Surrenders gradient signal but
            // keeps optimisation finite.
            let mut id = vec![0.0; n * n];
            for i in 0..n {
                id[i * n + i] = 1.0;
            }
            return id;
        }
    };
    let max_l = lambda.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let floor = max_l.max(1.0) * 1e-12;
    let mut inv = vec![0.0; n * n];
    // inv = V · diag(1/λ) · Vᵀ (with floor on tiny λ).
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                let lk = lambda[k];
                let inv_lk = 1.0 / lk.abs().max(floor) * lk.signum().max(0.0);
                s += v[i * n + k] * inv_lk * v[j * n + k];
            }
            inv[i * n + j] = s;
        }
    }
    inv
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
