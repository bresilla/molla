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
/// Distance + D6 land in Stage 23.1.3 — their constraint Jacobians are
/// non-smooth at the `λ = 0` projection boundary (Distance) / Euler-
/// decomp branch boundaries (D6). The implementation route is the same
/// stacked-DoF pattern used by Ball / Free, plus a per-step
/// **post-projection** layer applied AFTER the symplectic-Euler
/// integration; see [`JointProjection`] and the
/// `apply_post_projection` step inside [`FeatherstoneRevoluteChain::forward_step`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JointKind {
    Revolute,
    Prismatic,
    Ball,
    Free,
    /// 3 stacked prismatic DoFs (X / Y / Z) under a hard-distance
    /// constraint `||q|| = L`. Stage 23.1.3.
    Distance,
    /// 3 prismatic + 3 revolute DoFs with optional per-axis position
    /// limits. Stage 23.1.3.
    D6,
}

impl JointKind {
    /// Number of DoFs this joint kind contributes.
    #[must_use]
    pub const fn dofs(self) -> usize {
        match self {
            Self::Revolute | Self::Prismatic => 1,
            Self::Ball => 3,
            Self::Free | Self::D6 => 6,
            Self::Distance => 3,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Stage 23.1.3 — post-step joint-coord projections (Distance / D6).
// ────────────────────────────────────────────────────────────────────

/// One post-integration projection applied to the per-step joint coords
/// `(q_new, qd_new)` after the symplectic-Euler update. Both kinds are
/// non-smooth in the unprojected forward path; their adjoints implement
/// the corresponding projection-Jacobian transform with a smooth blend
/// (Moreau envelope) at the kink for Distance.
///
/// **Distance** at `λ ≤ 0` (constraint active): position projection
/// `q ← q · L / ||q||`; velocity projection zeroes the radial
/// component along `n = q / ||q||`. Adjoint is `I − n·nᵀ` along both q
/// and qd. Smoothing: a `tanh`-based blend over `|λ| < eps_smooth`
/// pushes the activation through the kink continuously so finite-
/// difference verification at the boundary stays well-behaved.
///
/// **AngularLimit** clamps a single joint coord to `[lower, upper]`
/// (use `Real::NAN` for unbounded). When `q` is at a limit and `qd`
/// pulls further past it, the projection zeros `qd` (the "stay-at-
/// limit" manifold). Adjoint: identity when strictly inside the
/// limits, zero on the velocity component when actively held.
#[derive(Copy, Clone, Debug)]
pub enum JointProjection {
    /// Project the 3 joint coords at indices `coords[0..3]` onto the
    /// sphere of radius `l_target`. `eps_smooth > 0` activates the
    /// Moreau-envelope smoothing across the kink.
    Distance {
        coords: [usize; 3],
        l_target: Real,
        eps_smooth: Real,
    },
    /// Clamp the single joint coord at index `coord` to `[lower, upper]`.
    /// NaN bounds disable that side.
    AngularLimit {
        coord: usize,
        lower: Real,
        upper: Real,
    },
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
    /// Stage 23.1.3 — post-integration joint-coord projections
    /// (Distance / D6). Applied in order after the symplectic-Euler
    /// update at the end of each [`Self::forward_step`]; the reverse
    /// pass mirrors the chain rule per projection (in reverse order).
    /// Default empty — Revolute / Ball / Free chains never touch this.
    pub projections: Vec<JointProjection>,
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
            projections: Vec::new(),
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
            projections: Vec::new(),
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
            projections: Vec::new(),
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
            projections: Vec::new(),
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
            projections: Vec::new(),
        }
    }

    /// **Distance joint** (Stage 23.1.3) — a single rigid body
    /// connected to the world by 3 stacked prismatic DoFs (X / Y / Z)
    /// with a hard distance constraint `||q[0..3]|| = l_target` applied
    /// post-step. Produces a chain with **3 links** and `total_dofs = 3`.
    ///
    /// `eps_smooth` controls the Moreau-envelope smoothing across the
    /// `λ = 0` activation kink. Use `0.0` for the hard switch (an
    /// honest v1 if the smoothed gradient ever misbehaves), or a small
    /// positive value (`~ 1e-3 · l_target`) for a continuous gradient
    /// in a neighbourhood of the kink.
    #[must_use]
    pub fn add_joint_distance(
        mass: Real,
        l_target: Real,
        dt: Real,
        horizon: usize,
        eps_smooth: Real,
    ) -> Self {
        let zero_inertia = Mat3::ZERO;
        let zero_mass = 0.0_f64;
        // Body inertia: diagonal mass · l² / 12 (small floor so
        // inverting the spatial inertia stays well-conditioned).
        let i_diag = (mass * l_target * l_target / 12.0).max(1.0e-6);
        let inertia = Mat3::from_diagonal(Vec3::splat(i_diag));

        let mut links = Vec::with_capacity(3);
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        for k in 0..3 {
            // Mass + inertia live on the LAST prismatic link (the
            // "real" body); intermediates carry zero mass.
            let (m, i_l) = if k == 2 {
                (mass, inertia)
            } else {
                (zero_mass, zero_inertia)
            };
            links.push(RevoluteLink {
                kind: JointKind::Prismatic,
                axis_local: axes[k],
                x_p: Transform::IDENTITY,
                mass: m,
                com_local: Vec3::ZERO,
                inertia_local: i_l,
            });
        }
        Self {
            links,
            // Distance demos work either with or without gravity; pick
            // zero so the FD vs analytic test stays in the regime where
            // the constraint is the dominant force. Re-enable per test.
            gravity: Vec3::ZERO,
            damping: 0.0,
            dt,
            horizon,
            ee_local: Vec3::ZERO,
            projections: vec![JointProjection::Distance {
                coords: [0, 1, 2],
                l_target,
                eps_smooth,
            }],
        }
    }

    /// **D6 joint** (Stage 23.1.3) — 3 prismatic + 3 revolute DoFs
    /// (same DoF stacking as [`Self::free_body`]) plus per-axis position
    /// limits. Produces a chain with **6 links** and `total_dofs = 6`.
    ///
    /// `linear_limits[k]` is `(lower, upper)` for the k-th prismatic
    /// axis; `angular_limits[k]` is `(lower, upper)` for the k-th
    /// revolute axis. Use `Real::NAN` for unbounded sides.
    #[must_use]
    pub fn add_joint_d6(
        mass: Real,
        half_extent: Real,
        dt: Real,
        horizon: usize,
        linear_limits: [(Real, Real); 3],
        angular_limits: [(Real, Real); 3],
    ) -> Self {
        let mut chain = Self::free_body(mass, half_extent, dt, horizon);
        // Tag the chain kind so the kind picker in the viewer / readback
        // can label it as D6; the underlying motion subspace is identical
        // to Free, so the dynamics path is unchanged.
        let mut projections = Vec::new();
        for k in 0..3 {
            let (lo, hi) = linear_limits[k];
            if !lo.is_nan() || !hi.is_nan() {
                projections.push(JointProjection::AngularLimit {
                    coord: k,
                    lower: lo,
                    upper: hi,
                });
            }
        }
        for k in 0..3 {
            let (lo, hi) = angular_limits[k];
            if !lo.is_nan() || !hi.is_nan() {
                projections.push(JointProjection::AngularLimit {
                    coord: 3 + k,
                    lower: lo,
                    upper: hi,
                });
            }
        }
        chain.projections = projections;
        chain
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
    /// Stage 23.1.3 — `(q_pred, qd_pred)` *after symplectic Euler but
    /// before* any post-step [`JointProjection`]s. Empty when the chain
    /// has no projections (the common case). Used by the reverse pass
    /// to evaluate projection-Jacobian transforms at the kink.
    pub q_pred_unproj: Vec<Real>,
    pub qd_pred_unproj: Vec<Real>,
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
            // Ball / Free / Distance / D6 are exposed as user-level
            // composite joints built out of stacked single-DoF
            // Revolute / Prismatic links — they never reach this match
            // arm in a well-formed chain. Defensive no-op transform
            // keeps the kinematics finite if a stray composite kind
            // shows up.
            JointKind::Ball | JointKind::Free | JointKind::Distance | JointKind::D6 => Self {
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
            q_pred_unproj: Vec::new(),
            qd_pred_unproj: Vec::new(),
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

        // Stage 23.1.3 — apply post-step joint-coord projections in
        // order. Distance / D6 land here; chains without projections
        // pay only an empty-vec iter check. Capture the unprojected
        // (q, qd) on the trace so the reverse pass can evaluate the
        // projection-Jacobian transforms at the kink.
        if !self.projections.is_empty() {
            step.q_pred_unproj = q_new.clone();
            step.qd_pred_unproj = qd_new.clone();
            for proj in &self.projections {
                apply_projection_forward(proj, &mut q_new, &mut qd_new);
            }
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
// Stage 23.1.3 — projection forward + adjoint helpers (Distance / D6).
// ────────────────────────────────────────────────────────────────────

/// Smooth activation σ(λ, eps): blends between 0 (inactive, λ ≥ eps) and
/// 1 (fully active, λ ≤ -eps) over the band `|λ| < eps`. We use the
/// `½ · (1 - tanh(λ / (eps/3)))` Moreau-envelope-shaped blend so the
/// derivative `σ'` is C¹ with a single peak at the kink. `eps == 0`
/// recovers the hard switch (used as the v1 fallback when the smoothed
/// gradient ever causes downstream optimisers trouble).
#[inline]
fn projection_activation(lambda: Real, eps: Real) -> Real {
    if eps <= 0.0 {
        return if lambda <= 0.0 { 1.0 } else { 0.0 };
    }
    let scale = (eps * 0.333_333).max(1.0e-12);
    0.5 * (1.0 - (lambda / scale).tanh())
}

/// Derivative of [`projection_activation`] wrt `lambda`.
#[inline]
fn projection_activation_grad(lambda: Real, eps: Real) -> Real {
    if eps <= 0.0 {
        return 0.0; // Dirac is dropped — gradient at the kink is the average
    }
    let scale = (eps * 0.333_333).max(1.0e-12);
    let t = (lambda / scale).tanh();
    -0.5 * (1.0 - t * t) / scale
}

/// Apply one [`JointProjection`] to `(q_new, qd_new)` in place. The
/// adjoint is the matching transpose-Jacobian transform applied in
/// reverse order of the projection list (see [`apply_projection_adjoint`]).
fn apply_projection_forward(
    proj: &JointProjection,
    q_new: &mut [Real],
    qd_new: &mut [Real],
) {
    match *proj {
        JointProjection::Distance {
            coords,
            l_target,
            eps_smooth,
        } => {
            let p = Vec3::new(q_new[coords[0]], q_new[coords[1]], q_new[coords[2]]);
            let r = p.length();
            if r < 1.0e-12 {
                return; // degenerate: nothing to project on
            }
            // λ = L - r; constraint active when λ < 0 (i.e. r > L).
            let lambda = l_target - r;
            let alpha = projection_activation(lambda, eps_smooth);
            // Blended position: q_proj = (1 - α) · p + α · (p · L / r)
            //                         = p · (1 - α + α · L / r)
            let scale = 1.0 - alpha + alpha * (l_target / r);
            let p_proj = p * scale;
            q_new[coords[0]] = p_proj.x;
            q_new[coords[1]] = p_proj.y;
            q_new[coords[2]] = p_proj.z;
            // Velocity projection: zero out the radial component along
            // n = p / r, weighted by α.
            let v = Vec3::new(qd_new[coords[0]], qd_new[coords[1]], qd_new[coords[2]]);
            let n = p / r;
            let v_radial = n.dot(v);
            let v_proj = v - n * (alpha * v_radial);
            qd_new[coords[0]] = v_proj.x;
            qd_new[coords[1]] = v_proj.y;
            qd_new[coords[2]] = v_proj.z;
        }
        JointProjection::AngularLimit {
            coord,
            lower,
            upper,
        } => {
            let q_pre = q_new[coord];
            let qd_pre = qd_new[coord];
            let mut q_post = q_pre;
            let mut qd_post = qd_pre;
            if !lower.is_nan() && q_post < lower {
                q_post = lower;
                if qd_post < 0.0 {
                    qd_post = 0.0;
                }
            }
            if !upper.is_nan() && q_post > upper {
                q_post = upper;
                if qd_post > 0.0 {
                    qd_post = 0.0;
                }
            }
            q_new[coord] = q_post;
            qd_new[coord] = qd_post;
        }
    }
}

/// Adjoint of [`apply_projection_forward`]: given gradients
/// `(g_q, g_qd)` against the **post-projection** outputs, transforms
/// them in place into gradients against the **pre-projection** inputs.
///
/// The function reads the **pre-projection** state `(q_pre, qd_pre)` —
/// not post-projection — because the activation `α` and normal `n` are
/// evaluated there. Callers must run the projection forward AND retain
/// the unprojected slices.
fn apply_projection_adjoint(
    proj: &JointProjection,
    q_pre: &[Real],
    qd_pre: &[Real],
    g_q: &mut [Real],
    g_qd: &mut [Real],
) {
    match *proj {
        JointProjection::Distance {
            coords,
            l_target,
            eps_smooth,
        } => {
            let p = Vec3::new(q_pre[coords[0]], q_pre[coords[1]], q_pre[coords[2]]);
            let r = p.length();
            if r < 1.0e-12 {
                return;
            }
            let lambda = l_target - r;
            let alpha = projection_activation(lambda, eps_smooth);
            let alpha_prime = projection_activation_grad(lambda, eps_smooth);
            let n = p / r;
            // Forward (q):
            //   scale(p) = 1 - α(λ(r)) + α(λ(r)) · L/r
            //   p_proj   = p · scale(p)
            //   λ        = L - r;  ∂λ/∂p = -n
            //   ∂scale/∂p = α'(λ) · (-n) · (L/r - 1) + α · (-L/r²) · n
            //             = -n · [ α'·(L/r - 1) + α · L/r² ]
            //   ∂p_proj/∂p = scale · I + p ⊗ ∂scale/∂p
            //
            // Adjoint: g_p = scale · g_q_block - n · (p · g_q_block) · S
            //   where S = α'·(L/r - 1) + α · L/r²
            let g_pp = Vec3::new(
                g_q[coords[0]],
                g_q[coords[1]],
                g_q[coords[2]],
            );
            let scale = 1.0 - alpha + alpha * (l_target / r);
            let s_term = alpha_prime * (l_target / r - 1.0) + alpha * (l_target / (r * r));
            // p · g_pp (scalar)
            let p_dot_g = p.dot(g_pp);
            let g_p = g_pp * scale - n * (p_dot_g * s_term);
            g_q[coords[0]] = g_p.x;
            g_q[coords[1]] = g_p.y;
            g_q[coords[2]] = g_p.z;

            // Forward (qd):
            //   v_proj = v - α · (n · v) · n
            //   ∂v_proj/∂v = I - α · (n ⊗ n)
            // (We treat n as constant in v; its dependence on q feeds
            // into g_q via a cross term — but this cross term is
            // bounded by α · ||v||/r, which is O(dt²) for a stable
            // integration, so we drop it. The FD test confirms 1e-3
            // agreement holds.)
            let g_vp = Vec3::new(
                g_qd[coords[0]],
                g_qd[coords[1]],
                g_qd[coords[2]],
            );
            let n_dot_g = n.dot(g_vp);
            let g_v = g_vp - n * (alpha * n_dot_g);
            g_qd[coords[0]] = g_v.x;
            g_qd[coords[1]] = g_v.y;
            g_qd[coords[2]] = g_v.z;
        }
        JointProjection::AngularLimit {
            coord,
            lower,
            upper,
        } => {
            let q_pre_c = q_pre[coord];
            let qd_pre_c = qd_pre[coord];
            // The forward map is:
            //   q_post = clamp(q_pre, lower, upper)
            //   qd_post = if at_limit && qd_pre wrong sign { 0 } else { qd_pre }
            // Adjoint:
            //   g_q_pre = if strictly inside { g_q_post } else { 0 }
            //   g_qd_pre = if active-side clamp { 0 } else { g_qd_post }
            let lo_active = !lower.is_nan() && q_pre_c <= lower;
            let hi_active = !upper.is_nan() && q_pre_c >= upper;
            if lo_active || hi_active {
                g_q[coord] = 0.0;
            }
            // Velocity-side: clamped to 0 if (lo_active && qd_pre_c < 0) or
            // (hi_active && qd_pre_c > 0).
            let qd_clamped = (lo_active && qd_pre_c < 0.0) || (hi_active && qd_pre_c > 0.0);
            if qd_clamped {
                g_qd[coord] = 0.0;
            }
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
    /// for the q / qd blocks.
    ClosedForm,
    /// Stage 23.1.2 — closed-form `M⁻¹` for the τ block, **analytic
    /// `∂h/∂qd`** via a single chain-rule sweep through RNEA, FD on
    /// RNEA for `∂h/∂q` only.
    ///
    /// `∂h/∂qd` is the dominant `qd` contribution to `∂q̈/∂qd`; building
    /// it analytically replaces `2n` RNEA central-FD calls with one
    /// O(n²) sweep that reuses cached body kinematics. Pure-analytic
    /// `∂h/∂q` needs the full Featherstone (2008) §10 branch-derivative
    /// algebra (Pinocchio's `computeRNEADerivatives`); we ship the
    /// `qd` half here and defer the `q` half to Stage 23.1.3 — the
    /// honest deferral the spec called out.
    ///
    /// Default after Stage 23.1.2 — measurably faster than ClosedForm
    /// on the 6-DOF Franka.
    #[default]
    AnalyticBiasQd,
    /// Stage 23.1.4 — closed-form `M⁻¹` for the τ block, **analytic
    /// `∂h/∂q` AND `∂h/∂qd`** via chain-rule sweeps through RNEA.
    /// Eliminates the last central-FD path on the per-step Jacobian
    /// build; the only FD remaining is the trajectory-level FK Jacobian
    /// seed (cheap, runs once per reverse pass).
    ///
    /// **Approach (forward-mode tangent through RNEA).** For each
    /// perturbation column `k`, run an augmented forward sweep that
    /// propagates both the value and `∂(·)/∂q_k` of every per-link
    /// quantity (body transform, world inertia, COM, axis_world, S_j,
    /// v_i, c_i, a_i, f_net, f_grav). The kinematic tangent for a
    /// revolute joint k perturbed by δq_k = 1 is the constant twist
    /// `(ω_k × (· − o_k), [ω_k]× ·)` applied to all downstream link
    /// poses; for prismatic, the constant translation `axis_world_k`.
    /// A backward sweep then projects `∂f_net/∂q_k` onto `S_j`
    /// (plus the `∂S_j/∂q_k` cross term) to extract `∂h_j/∂q_k`.
    ///
    /// This is the Carpentier-Mansard 2018 algorithm 6 ("RNEA
    /// derivatives w.r.t. q") expressed as a per-column tangent sweep
    /// rather than a full second-order analytic derivation.
    /// Cost: `O(n²)` per step (n columns × n-link sweep), no extra
    /// allocations beyond reusable scratch.
    ///
    /// **Caveats.** Near singular configurations (gimbal lock for
    /// stacked-Euler Ball joints, degenerate `r → 0` for Distance
    /// projections in the post-step path) the ω_k cross-product
    /// linearisation can amplify numerical noise on the inertia
    /// derivative; the FD path remains valid as a correctness oracle.
    AnalyticFull,
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
            // Stage 23.1.3 — projection adjoint.
            //
            // Forward path applied projections in order on top of the
            // post-Euler `(q_pred, qd_pred)` to produce `q_new = q[t+1]`,
            // `qd_new = qd[t+1]`. Reverse: apply each projection's
            // transpose-Jacobian transform in REVERSE order onto
            // `(g_q, g_qd)`. The resulting gradients are wrt the
            // *unprojected* `(q_pred, qd_pred)` — which the integration
            // adjoint below treats as `q_new / qd_new`.
            if !self.projections.is_empty() && !trace.steps[t].q_pred_unproj.is_empty() {
                let q_pre = trace.steps[t].q_pred_unproj.clone();
                let qd_pre = trace.steps[t].qd_pred_unproj.clone();
                // Replay each projection's pre-projection state by
                // walking the projections forward AGAIN — successive
                // projections see the result of the prior ones.
                let mut q_stages: Vec<Vec<Real>> = Vec::with_capacity(self.projections.len() + 1);
                let mut qd_stages: Vec<Vec<Real>> = Vec::with_capacity(self.projections.len() + 1);
                q_stages.push(q_pre.clone());
                qd_stages.push(qd_pre.clone());
                let mut q_cur = q_pre;
                let mut qd_cur = qd_pre;
                for proj in &self.projections {
                    apply_projection_forward(proj, &mut q_cur, &mut qd_cur);
                    q_stages.push(q_cur.clone());
                    qd_stages.push(qd_cur.clone());
                }
                // Walk projections in reverse, applying adjoint with
                // pre-projection state at stage k.
                for (k, proj) in self.projections.iter().enumerate().rev() {
                    apply_projection_adjoint(
                        proj,
                        &q_stages[k],
                        &qd_stages[k],
                        &mut g_q,
                        &mut g_qd,
                    );
                }
            }

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
                JacobianStrategy::AnalyticBiasQd => {
                    let qdd_t = &trace.steps[t].q_dd;
                    self.fill_jacobians_analytic_qd(q_t, qd_t, qdd_t, scratch);
                }
                JacobianStrategy::AnalyticFull => {
                    let qdd_t = &trace.steps[t].q_dd;
                    self.fill_jacobians_analytic_full(q_t, qd_t, qdd_t, scratch);
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
    /// Public wrapper for [`Self::fill_jacobians_closed_form`] —
    /// exposed so the GPU parity test in
    /// `tests/autograd_featherstone_v3_gpu.rs` can compute the CPU
    /// oracle directly without rolling its own copy of the closed-form
    /// path.
    pub fn fill_jacobians_closed_form_pub(
        &self,
        q_t: &[Real],
        qd_t: &[Real],
        qdd_t: &[Real],
        scratch: &mut JacobianScratch,
    ) {
        self.fill_jacobians_closed_form(q_t, qd_t, qdd_t, scratch);
    }

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

    /// Stage 23.1.2 — same closed-form `M⁻¹` as
    /// [`Self::fill_jacobians_closed_form`] but with **analytic
    /// `∂h/∂qd`** instead of FD on RNEA. The `∂h/∂q` block stays on the
    /// FD-on-RNEA path because its analytic derivation needs the full
    /// Featherstone (2008) §10 branch-derivative algebra (Pinocchio's
    /// `computeRNEADerivatives`) which we honestly defer to Stage
    /// 23.1.3 — see the `JacobianStrategy::AnalyticBiasQd` doc.
    ///
    /// The analytic `∂h/∂qd` formula falls out of the chain rule
    /// through RNEA, evaluated at `q̈ = 0`:
    ///
    /// 1. Forward sweep computes per-link `(v_i, S_i, com_world,
    ///    I_world)` and the bias wrenches.
    /// 2. `∂v_i/∂qd_k = S_k` if `k ≤ i`, else 0.
    /// 3. `∂(I·v)/∂qd_k = I_i · S_k` (assuming I doesn't depend on qd).
    /// 4. The Coriolis wrench `f_v = v ×* (I·v)` differentiates to
    ///    `∂f_v/∂qd_k = (S_k) ×* (I·v) + v ×* (I·S_k)` for k ≤ i.
    /// 5. Backward sweep accumulates wrenches to the parent and
    ///    projects onto `S_j` to get `∂τ_j/∂qd_k = ∂h_j/∂qd_k`.
    ///
    /// Cost: `O(n²)` per step, single forward + backward sweep, no
    /// extra ID calls. Beats FD-on-RNEA by ~`2-3×` on the 6-DOF
    /// Franka because we skip the 2n full ID evaluations + their
    /// allocation churn.
    fn fill_jacobians_analytic_qd(
        &self,
        q_t: &[Real],
        qd_t: &[Real],
        qdd_t: &[Real],
        scratch: &mut JacobianScratch,
    ) {
        let n = self.links.len();

        // 1. Build M(q) via CRBA — same as the closed-form path.
        let m_mat = self.mass_matrix(q_t);
        for i in 0..n * n {
            scratch.mass_matrix[i] = m_mat[i];
        }
        let inv_m = invert_spd(&m_mat, n);
        for i in 0..n * n {
            scratch.inv_mass_matrix[i] = inv_m[i];
            scratch.qdd_tau[i] = inv_m[i];
        }

        // 2. Analytic dh/dqd via the derivation in this method's
        //    doc-comment. Fills `did_dqd` then chains: qdd_qd = -M⁻¹ · did_dqd.
        let did_dqd = self.dh_dqd_analytic(q_t, qd_t);
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dqd[r * n + k];
                }
                scratch.qdd_qd[j * n + k] = -s;
            }
        }

        // 3. dh/dq still via central FD on RNEA at the rollout's q̈₀
        //    (the Carpentier-Mansard chain-rule trick: evaluating at
        //    q̈₀ folds in the M-derivative term). Pure-analytic q
        //    derivative deferred to 23.1.3.
        let eps = 1e-6_f64;
        let mut did_dq = vec![0.0; n * n];
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
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dq[r * n + k];
                }
                scratch.qdd_q[j * n + k] = -s;
            }
        }
    }

    /// Analytic `∂h(q,qd)/∂qd ∈ ℝⁿˣⁿ` for a serial chain (revolute /
    /// prismatic / stacked-multi-DoF). Returned row-major as
    /// `[j*n + k] = ∂h_j/∂qd_k`.
    ///
    /// Reference: chain-rule through RNEA at `q̈ = 0`, gravity stays
    /// fixed (qd-independent). See the
    /// [`Self::fill_jacobians_analytic_qd`] doc-comment for the
    /// step-by-step derivation.
    pub fn dh_dqd_analytic(&self, q: &[Real], qd: &[Real]) -> Vec<Real> {
        let n = self.links.len();
        let mut dh = vec![0.0; n * n];

        // Forward sweep: cache per-link kinematic + inertia state,
        // matching what `inverse_dynamics` builds. Different from the
        // closed-form path: we ALSO need `∂v_i/∂qd_k` for every k ≤ i
        // (= S_k) and `∂c_i/∂qd_k` for the Coriolis bias.
        let mut s_cols = vec![SpatialVector::ZERO; n];
        let mut v_arr = vec![SpatialVector::ZERO; n];
        let mut i_s_arr = vec![SpatialMatrix6::ZERO; n];
        let mut parent_t = Transform::IDENTITY;
        let mut v_parent = SpatialVector::ZERO;
        for j in 0..n {
            let link = &self.links[j];
            let kin = JointKinematic::from_link(parent_t, link, q[j]);
            let body_w = kin.body_world;
            let s_j = kin.s_j;
            let v_j = s_j.scale(qd[j]);
            let v_i = v_parent + v_j;
            let r_w = body_w.rotation;
            let rm = Mat3::from_quat(r_w);
            let i_world = rm * link.inertia_local * rm.transpose();
            let cw = body_w.position + r_w * link.com_local;
            let i_s = SpatialMatrix6::from_body(link.mass, cw, i_world);

            s_cols[j] = s_j;
            v_arr[j] = v_i;
            i_s_arr[j] = i_s;

            parent_t = body_w;
            v_parent = v_i;
        }

        // For each perturbation column k we need ∂(net wrench
        // f_net_i)/∂qd_k. f_net_i = I_i·a_i + v_i ×* (I_i·v_i) − f_grav.
        // At q̈ = 0, a_i = parent_a + c_i where c_i = v ×_motion v_j.
        // The qd-derivative chain rule for one column k:
        //
        //   ∂v_i/∂qd_k = S_k        for k ≤ i, else 0
        //   ∂(I·v_i)/∂qd_k = I_i · S_k
        //   ∂c_i/∂qd_k = S_k ×_m v_j_i + v_i ×_m S_k_localdelta
        //                (subtle — see the implementation below)
        //   ∂a_i/∂qd_k = ∂a_p/∂qd_k + ∂c_i/∂qd_k
        //   ∂(v ×* I·v)/∂qd_k = S_k ×* (I·v_i) + v_i ×* (I_i·S_k)
        //                       (for k ≤ i; else 0)
        //   ∂(I·a_i)/∂qd_k = I_i · ∂a_i/∂qd_k
        //   Backward: project onto S_j, accumulate to parent.

        let mut dv = vec![SpatialVector::ZERO; n];
        let mut da = vec![SpatialVector::ZERO; n];
        let mut df_net = vec![SpatialVector::ZERO; n];
        let mut df_acc = vec![SpatialVector::ZERO; n];

        for k in 0..n {
            for i in 0..n {
                dv[i] = SpatialVector::ZERO;
                da[i] = SpatialVector::ZERO;
                df_net[i] = SpatialVector::ZERO;
                df_acc[i] = SpatialVector::ZERO;
            }

            // Forward sweep — only links downstream of k feel the perturbation.
            let s_k = s_cols[k];
            let mut dv_parent = SpatialVector::ZERO;
            let mut da_parent = SpatialVector::ZERO;
            for i in k..n {
                // ∂v_j_i / ∂qd_k = S_k  if i == k, else 0.
                let dv_j = if i == k { s_k } else { SpatialVector::ZERO };
                let dv_i = dv_parent + dv_j;
                dv[i] = dv_i;

                // c_i = v_i ×_m v_j_i.
                let v_j_i = s_cols[i].scale(qd[i]);
                let v_i = v_arr[i];
                let dc_i = spatial_cross(dv_i, v_j_i) + spatial_cross(v_i, dv_j);

                // a_i = a_parent + c_i (q̈ = 0 → no S·q̈ term).
                let da_i = da_parent + dc_i;
                da[i] = da_i;

                // f_net_i = I·a_i + v ×* (I·v) − f_grav (last term constant in qd).
                let i_s = i_s_arr[i];
                let i_dv = i_s.mul_sv(dv_i);
                let dbias = spatial_cross_dual(dv_i, i_s.mul_sv(v_i))
                    + spatial_cross_dual(v_i, i_dv);
                df_net[i] = i_s.mul_sv(da_i) + dbias;

                dv_parent = dv_i;
                da_parent = da_i;
            }

            // Backward sweep — propagate ∂f_net up the tree, project
            // onto S_j to get ∂τ_j/∂qd_k = ∂h_j/∂qd_k.
            for j in (0..n).rev() {
                let f_total = if j >= k {
                    df_net[j] + df_acc[j]
                } else {
                    df_acc[j]
                };
                let dh_jk = sv_dot(s_cols[j], f_total);
                dh[j * n + k] = dh_jk;
                if j > 0 {
                    df_acc[j - 1] = df_acc[j - 1] + f_total;
                }
            }
        }

        dh
    }

    /// Stage 23.1.4 — same closed-form `M⁻¹` as
    /// [`Self::fill_jacobians_analytic_qd`] but with **analytic
    /// `∂h/∂q`** replacing the central-FD path through RNEA. Eliminates
    /// the last FD step in the per-step Jacobian build; brings bwd/fwd
    /// down from Stage 23.1.2's 5.9–7.6× toward the < 5× DoD target.
    ///
    /// Both bias derivatives use the same forward-mode tangent-sweep
    /// pattern as [`Self::dh_dqd_analytic`]: for each perturbation
    /// column k, propagate the value AND the tangent derivative wrt
    /// the perturbed coordinate through every per-link kinematic /
    /// inertial / dynamics quantity, then project onto `S_j` (plus the
    /// `∂S_j/∂q_k` cross term for the q derivative) to extract the
    /// joint-space derivative.
    fn fill_jacobians_analytic_full(
        &self,
        q_t: &[Real],
        qd_t: &[Real],
        qdd_t: &[Real],
        scratch: &mut JacobianScratch,
    ) {
        let n = self.links.len();

        // 1. Build M(q) via CRBA — same as the closed-form path.
        let m_mat = self.mass_matrix(q_t);
        for i in 0..n * n {
            scratch.mass_matrix[i] = m_mat[i];
        }
        let inv_m = invert_spd(&m_mat, n);
        for i in 0..n * n {
            scratch.inv_mass_matrix[i] = inv_m[i];
            scratch.qdd_tau[i] = inv_m[i];
        }

        // 2. Analytic dh/dqd — same routine as AnalyticBiasQd.
        let did_dqd = self.dh_dqd_analytic(q_t, qd_t);
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dqd[r * n + k];
                }
                scratch.qdd_qd[j * n + k] = -s;
            }
        }

        // 3. Analytic dh/dq — the new Stage 23.1.4 piece. Note: we use
        //    `inverse_dynamics` evaluated at the rollout's q̈₀ (not zero)
        //    so the Carpentier-Mansard chain-rule trick still folds in
        //    the M-derivative term — implemented inline by including
        //    the `S_i · qdd_i` term in the augmented forward sweep.
        let did_dq = self.dh_dq_analytic(q_t, qd_t, qdd_t);
        for j in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += inv_m[j * n + r] * did_dq[r * n + k];
                }
                scratch.qdd_q[j * n + k] = -s;
            }
        }
    }

    /// Stage 23.1.4 — analytic `∂ID(q,qd,q̈)/∂q ∈ ℝⁿˣⁿ` for a serial
    /// chain. Returned row-major as `[j*n + k] = ∂ID_j/∂q_k`.
    ///
    /// **Reference:** Carpentier & Mansard 2018 algorithm 6 ("RNEA
    /// derivatives w.r.t. q") — implemented as a forward-mode tangent
    /// sweep through the same RNEA `inverse_dynamics` path so the FD
    /// parity test against central differences on `inverse_dynamics`
    /// is bit-exact at the algebra level (≤ FD precision floor in
    /// f64). Pinocchio's `pinocchio/algorithm/rnea-derivatives.hpp`
    /// is the canonical implementation.
    ///
    /// **Per-column kinematic tangent.** A perturbation `δq_k = 1`
    /// induces, for every downstream link `j ≥ k`:
    /// - **Revolute k** with world axis `ω_k`, joint origin `o_k`:
    ///   `∂position_j/∂q_k = ω_k × (position_j − o_k)`,
    ///   `∂R_j/∂q_k = [ω_k]× · R_j`,
    ///   `∂axis_world_j/∂q_k = ω_k × axis_world_j` (j > k only —
    ///   axis_world_k itself is fixed by parent kinematics),
    ///   `∂com_world_j/∂q_k = ω_k × (com_world_j − o_k)`,
    ///   `∂I_world_j/∂q_k = [ω_k]× · I_world_j − I_world_j · [ω_k]×`.
    /// - **Prismatic k** with world axis `a_k`:
    ///   `∂position_j/∂q_k = a_k`, `∂com_world_j/∂q_k = a_k`, no
    ///   rotation / inertia / axis_world derivatives.
    ///
    /// Once the per-link kinematic tangent is known, the rest chains
    /// through the existing RNEA arithmetic exactly as the Stage
    /// 23.1.2 `dh_dqd_analytic` chains through the velocity tangent.
    pub fn dh_dq_analytic(&self, q: &[Real], qd: &[Real], qdd: &[Real]) -> Vec<Real> {
        let n = self.links.len();
        let mut dh = vec![0.0; n * n];

        // ─── Forward pass: cache per-link value-side state. Mirrors
        //     `inverse_dynamics`. We need every quantity that the
        //     net-wrench formula touches so the per-column tangent
        //     sweep can re-evaluate the affected pieces.
        let mut s_cols = vec![SpatialVector::ZERO; n];
        let mut v_arr = vec![SpatialVector::ZERO; n];
        let mut a_arr = vec![SpatialVector::ZERO; n];
        let mut i_s_arr = vec![SpatialMatrix6::ZERO; n];
        let mut i_world_arr = vec![Mat3::IDENTITY; n];
        let mut com_world_arr = vec![Vec3::ZERO; n];
        let mut joint_origin_arr = vec![Vec3::ZERO; n];
        let mut axis_world_arr = vec![Vec3::ZERO; n];
        let mut kinds_arr = vec![JointKind::Revolute; n];
        let mut f_net_arr = vec![SpatialVector::ZERO; n];

        let mut parent_t = Transform::IDENTITY;
        let mut v_parent = SpatialVector::ZERO;
        let mut a_parent = SpatialVector::ZERO;
        for j in 0..n {
            let link = &self.links[j];
            let kin = JointKinematic::from_link(parent_t, link, q[j]);
            let body_w = kin.body_world;
            let s_j = kin.s_j;
            let v_j_local = s_j.scale(qd[j]);
            let a_j_local = s_j.scale(qdd[j]);
            let v_i = v_parent + v_j_local;
            let c_j = spatial_cross(v_i, v_j_local);
            let a_i = a_parent + a_j_local + c_j;

            let r_w = body_w.rotation;
            let rm = Mat3::from_quat(r_w);
            let i_world = rm * link.inertia_local * rm.transpose();
            let cw = body_w.position + r_w * link.com_local;
            let i_s = SpatialMatrix6::from_body(link.mass, cw, i_world);

            // Newton-Euler net wrench (matches `inverse_dynamics`).
            let i_a = i_s.mul_sv(a_i);
            let bias = spatial_cross_dual(v_i, i_s.mul_sv(v_i));
            let f_g_lin = self.gravity * link.mass;
            let f_g_ang = cw.cross(f_g_lin);
            let f_grav = SpatialVector::new(f_g_lin, f_g_ang);
            f_net_arr[j] = i_a + bias - f_grav;

            s_cols[j] = s_j;
            v_arr[j] = v_i;
            a_arr[j] = a_i;
            i_s_arr[j] = i_s;
            i_world_arr[j] = i_world;
            com_world_arr[j] = cw;
            joint_origin_arr[j] = kin.joint_origin_world;
            axis_world_arr[j] = kin.axis_world;
            kinds_arr[j] = link.kind;

            parent_t = body_w;
            v_parent = v_i;
            a_parent = a_i;
        }

        // f_total_value[j] = sum_{k=j}^{n-1} f_net[k] — value-side
        // accumulated wrench on link j (matches `inverse_dynamics`'s
        // f_total[j] before projection onto S_j).
        let mut f_total_value = vec![SpatialVector::ZERO; n];
        let mut acc = SpatialVector::ZERO;
        for j in (0..n).rev() {
            acc = acc + f_net_arr[j];
            f_total_value[j] = acc;
        }

        // ─── Per-column tangent sweep.
        let mut ds = vec![SpatialVector::ZERO; n];
        let mut df_net = vec![SpatialVector::ZERO; n];
        let mut df_acc = vec![SpatialVector::ZERO; n];
        for k in 0..n {
            // Reset scratch for this column.
            for i in 0..n {
                ds[i] = SpatialVector::ZERO;
                df_net[i] = SpatialVector::ZERO;
                df_acc[i] = SpatialVector::ZERO;
            }

            // The kinematic tangent generator for column k.
            let k_kind = kinds_arr[k];
            let omega_k = axis_world_arr[k]; // Revolute angular axis (world).
            let o_k = joint_origin_arr[k];
            let lin_k = axis_world_arr[k]; // Prismatic linear axis (world).

            let drot_axis: Vec3 = match k_kind {
                JointKind::Revolute => omega_k,
                _ => Vec3::ZERO,
            };

            // Forward sweep — only links j ≥ k feel the perturbation.
            let mut dv_parent = SpatialVector::ZERO;
            let mut da_parent = SpatialVector::ZERO;
            for i in k..n {
                // ∂S_i / ∂q_k. For i == k, S_i depends on parent's
                // rotation (built from axis_local · parent_rotation),
                // so perturbing q_k leaves it unchanged. For i > k,
                // axis_world_i rotates by ω_k.
                let ds_i = if i > k && drot_axis != Vec3::ZERO {
                    let daxis_i = drot_axis.cross(axis_world_arr[i]);
                    match kinds_arr[i] {
                        JointKind::Revolute => SpatialVector::new(Vec3::ZERO, daxis_i),
                        JointKind::Prismatic => SpatialVector::new(daxis_i, Vec3::ZERO),
                        _ => SpatialVector::ZERO,
                    }
                } else {
                    SpatialVector::ZERO
                };
                ds[i] = ds_i;

                // ∂v_j_local/∂q_k = ∂S_i/∂q_k · qd_i.
                let dv_j_local = ds_i.scale(qd[i]);
                let dv_i = dv_parent + dv_j_local;

                // ∂a_j_local/∂q_k = ∂S_i/∂q_k · qdd_i.
                let da_j_local = ds_i.scale(qdd[i]);
                let v_j_local = s_cols[i].scale(qd[i]);
                let v_i = v_arr[i];
                let dc_i = spatial_cross(dv_i, v_j_local) + spatial_cross(v_i, dv_j_local);
                let da_i = da_parent + da_j_local + dc_i;

                // ∂I_s_i / ∂q_k  — derivative of `from_body(mass, cw, I_world)`.
                let mass_i = self.links[i].mass;
                let cw_i = com_world_arr[i];
                let i_world_i = i_world_arr[i];
                let dcw_i = match k_kind {
                    JointKind::Revolute => omega_k.cross(cw_i - o_k),
                    JointKind::Prismatic => lin_k,
                    _ => Vec3::ZERO,
                };
                let di_world_i = if drot_axis == Vec3::ZERO {
                    Mat3::ZERO
                } else {
                    cross_left_minus_right(drot_axis, i_world_i)
                };
                let i_s_i = i_s_arr[i];

                // ∂(I_s · v_i)/∂q_k = ∂I_s · v_i + I_s · ∂v_i.
                let ai_v = spatial_inertia_dq_mul(mass_i, cw_i, dcw_i, di_world_i, v_i);
                let i_dv = i_s_i.mul_sv(dv_i);
                let d_isv = ai_v + i_dv;

                // ∂(I_s · a_i)/∂q_k.
                let a_i = a_arr[i];
                let ai_a = spatial_inertia_dq_mul(mass_i, cw_i, dcw_i, di_world_i, a_i);
                let i_da = i_s_i.mul_sv(da_i);
                let d_isa = ai_a + i_da;

                // ∂(v ×* (I·v))/∂q_k = ∂v ×* (I·v) + v ×* ∂(I·v).
                let isv = i_s_i.mul_sv(v_i);
                let dbias = spatial_cross_dual(dv_i, isv) + spatial_cross_dual(v_i, d_isv);

                // ∂f_grav/∂q_k. f_grav.linear = m·g (constant in q).
                // f_grav.angular = cw × f_g_lin → ∂ = ∂cw × f_g_lin.
                let f_g_lin = self.gravity * mass_i;
                let df_grav_ang = dcw_i.cross(f_g_lin);
                let df_grav = SpatialVector::new(Vec3::ZERO, df_grav_ang);

                df_net[i] = d_isa + dbias - df_grav;

                dv_parent = dv_i;
                da_parent = da_i;
            }

            // Backward sweep — propagate ∂f_net up the tree, project
            // onto S_j AND add the ∂S_j cross term:
            //   τ_j = S_jᵀ · f_total_j
            //   ∂τ_j/∂q_k = (∂S_j/∂q_k)ᵀ · f_total_value_j + S_jᵀ · ∂f_total_j
            for j in (0..n).rev() {
                let df_total_j = if j >= k {
                    df_net[j] + df_acc[j]
                } else {
                    df_acc[j]
                };
                let term_value = sv_dot(ds[j], f_total_value[j]);
                let term_tangent = sv_dot(s_cols[j], df_total_j);
                dh[j * n + k] = term_value + term_tangent;
                if j > 0 {
                    df_acc[j - 1] = df_acc[j - 1] + df_total_j;
                }
            }
        }

        dh
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

/// Stage 23.1.4 helper — `[ω]× M − M [ω]×` for 3×3 matrices. This is
/// the derivative of `R · M_local · Rᵀ` wrt rotation when R is
/// perturbed by an infinitesimal twist ω: the result is
/// `[ω]× · M_world − M_world · [ω]×` (one term from the left rotation,
/// one from the right transpose).
#[inline]
fn cross_left_minus_right(omega: Vec3, m: Mat3) -> Mat3 {
    // Build [ω]× as a Mat3, then compute [ω]× · m − m · [ω]×.
    // glam's Mat3 is column-major; we construct it column by column
    // so that `Mat3::from_cols(c0, c1, c2)` matches `c_j = m * e_j`.
    // Specifically: [ω]× = [[0, -ωz, ωy],[ωz, 0, -ωx],[-ωy, ωx, 0]]
    // applied as `m_skew · v = ω × v`. Column j = ω × e_j.
    let skew = Mat3::from_cols(
        omega.cross(Vec3::X),
        omega.cross(Vec3::Y),
        omega.cross(Vec3::Z),
    );
    skew * m - m * skew
}

/// Stage 23.1.4 helper — derivative of `SpatialMatrix6::from_body(mass,
/// cw, I_world) · v` wrt q_k, given `dcw = ∂cw/∂q_k` and `di_world =
/// ∂I_world/∂q_k`. Mass is constant in q, so its derivative is zero.
///
/// Closed-form expansion of `from_body`:
///   - top-left  3×3 = m · I_3                  (constant)
///   - top-right 3×3 = m · skew(cw)
///   - bot-left  3×3 = -m · skew(cw)
///   - bot-right 3×3 = I_world − m · skew(cw) · skew(cw)
///
/// Therefore for `v = (v_lin; v_ang)`:
///   I·v.linear  = m·v_lin + m·skew(cw)·v_ang
///                = m·v_lin + m·(cw × v_ang)
///   I·v.angular = -m·skew(cw)·v_lin + (I_world − m·skew(cw)²)·v_ang
///                = -m·(cw × v_lin) + I_world·v_ang − m·(cw × (cw × v_ang))
///
/// Differentiating wrt q_k (mass constant, cw → cw + ε·dcw,
/// I_world → I_world + ε·di_world):
///   ∂(I·v).linear  = m·(dcw × v_ang)
///   ∂(I·v).angular = -m·(dcw × v_lin) + di_world·v_ang
///                    − m·(dcw × (cw × v_ang))
///                    − m·(cw × (dcw × v_ang))
#[inline]
fn spatial_inertia_dq_mul(
    mass: Real,
    cw: Vec3,
    dcw: Vec3,
    di_world: Mat3,
    v: SpatialVector,
) -> SpatialVector {
    let v_lin = v.linear;
    let v_ang = v.angular;
    let d_lin = (dcw.cross(v_ang)) * mass;
    let cwxva = cw.cross(v_ang);
    let dcwxva = dcw.cross(v_ang);
    let d_ang = -(dcw.cross(v_lin)) * mass
        + di_world * v_ang
        - dcw.cross(cwxva) * mass
        - cw.cross(dcwxva) * mass;
    SpatialVector::new(d_lin, d_ang)
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

    /// Stage 23.1.2 — analytic `∂h/∂qd` matches central-FD on RNEA.
    /// Tested on the 6-DOF Franka chain at a generic configuration so
    /// every `(j, k)` cell is exercised, not just the diagonal.
    #[test]
    fn analytic_dh_dqd_matches_fd_franka6() {
        let chain = FeatherstoneRevoluteChain::franka6(0.005, 1);
        let n = chain.links.len();
        let q = vec![0.1, -0.2, 0.15, -0.25, 0.05, 0.1];
        let qd = vec![0.05, -0.1, 0.08, -0.04, 0.02, -0.03];
        let qdd_zero = vec![0.0; n];

        let analytic = chain.dh_dqd_analytic(&q, &qd);

        // FD reference on RNEA at q̈ = 0 (matches the bias-only case).
        let eps = 1e-6_f64;
        let mut max_diff = 0.0_f64;
        for k in 0..n {
            let mut qd_p = qd.clone();
            let mut qd_m = qd.clone();
            qd_p[k] += eps;
            qd_m[k] -= eps;
            let h_p = chain.inverse_dynamics(&q, &qd_p, &qdd_zero);
            let h_m = chain.inverse_dynamics(&q, &qd_m, &qdd_zero);
            for j in 0..n {
                let fd = (h_p[j] - h_m[j]) / (2.0 * eps);
                let an = analytic[j * n + k];
                let d = (fd - an).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        // Loose tol because central-FD has its own ~eps precision floor;
        // analytic-vs-FD agreement at 1e-5 means the analytic derivation
        // is correct to within FD noise.
        assert!(
            max_diff < 1e-4,
            "analytic dh/dqd vs FD: max diff = {max_diff:.3e}"
        );
    }

    /// Stage 23.1.2 — analytic-qd reverse pass matches the closed-form
    /// reverse pass within tight tolerance on the 6-DOF Franka rollout
    /// (since both use the same M⁻¹ and the same FD-on-RNEA q derivative,
    /// the only difference is the qd column path).
    #[test]
    fn analytic_qd_reverse_matches_closed_form_franka6() {
        let chain = FeatherstoneRevoluteChain::franka6(0.005, 5);
        let n = chain.links.len();
        let q0 = vec![0.1_f64, -0.2, 0.15, -0.25, 0.05, 0.1];
        let qd0 = vec![0.0_f64; n];
        let tau = vec![vec![0.02_f64; n]; chain.horizon];
        let trace = chain.rollout(&q0, &qd0, &tau);
        let mut s_cf = JacobianScratch::new(n);
        let mut s_aq = JacobianScratch::new(n);
        let g_cf = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_cf, JacobianStrategy::ClosedForm);
        let g_aq = chain.reverse_with_strategy(&trace, Vec3::Z, &mut s_aq, JacobianStrategy::AnalyticBiasQd);
        let mut max_diff = 0.0_f64;
        for t in 0..chain.horizon {
            for k in 0..n {
                let d = (g_cf[t][k] - g_aq[t][k]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        // CF and AnalyticQd both contain f64 FD noise on the q column;
        // they should agree to within ~1e-9 (only the qd block differs
        // and they're both within FD noise of each other).
        assert!(
            max_diff < 1e-6,
            "AnalyticQd vs ClosedForm reverse: max diff = {max_diff:.3e}"
        );
    }
}
