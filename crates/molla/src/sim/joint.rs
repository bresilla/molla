//! Joint enums + DOF accounting.
//!
//! Mirrors Newton's `JointType` shape (see
//! `xtra/newton/newton/_src/sim/enums.py`). Supports `Revolute`,
//! `Prismatic`, `Fixed`, `Free`, `Ball`, `Distance`, and `D6`.
//! [`EqualityType`] covers ball-weld, weld, and joint-mimic
//! equality constraints (Stage 32).

use crate::math::Real;

/// Equality-constraint kind. Stored as a `u32` per constraint on
/// [`super::Model`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum EqualityType {
    /// Ball weld — pin two anchor points together (3 positional
    /// constraints, 0 rotational). Bodies remain free to rotate.
    Connect = 0,
    /// Fixed weld — pin position AND orientation. 6 constraints.
    Weld = 1,
    /// Joint-mimic (Newton's `JOINT` equality kind) — constrains one
    /// joint coordinate as a polynomial of another:
    ///
    /// ```text
    /// q_driven = poly[0] + poly[1]·q_driver + poly[2]·q_driver²
    ///          + poly[3]·q_driver³ + poly[4]·q_driver⁴
    /// ```
    ///
    /// Used for closed kinematic loops (4-bar linkage), gear
    /// couplings, and mimic-axis grippers (e.g. Robotiq 2-finger).
    /// The driver and driven joint indices + the polynomial live in
    /// the parallel `Model.mimic_*` arrays (one slot per equality
    /// constraint; only consulted when `equality_type == JointMimic`).
    /// Stage 32.
    JointMimic = 2,
}

impl EqualityType {
    /// Decode a `u32` to the enum, defaulting to `Connect`.
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Weld,
            2 => Self::JointMimic,
            _ => Self::Connect,
        }
    }
}

/// Per-DOF actuator target mode. Mirrors Newton's
/// `sim/enums.py:JointTargetMode`. Stored as a `u32` per DOF on
/// [`super::Control`] so each DOF can be driven independently.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum JointTargetMode {
    /// No drive; the DOF responds only to gravity / friction / external
    /// forces / explicit `joint_force`.
    None = 0,
    /// PD around a position target: `τ = kp·(q_target − q) + kd·(0 − qd)`.
    Position = 1,
    /// PD around a velocity target: `τ = kd·(qd_target − qd)`.
    Velocity = 2,
    /// PD around both: `τ = kp·(q_target − q) + kd·(qd_target − qd)`.
    PositionVelocity = 3,
    /// Direct effort — `τ = q_target` (interpreted as torque/force).
    Effort = 4,
}

impl JointTargetMode {
    /// Decode a `u32` to the enum, falling back to `None` for unknown values.
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Position,
            2 => Self::Velocity,
            3 => Self::PositionVelocity,
            4 => Self::Effort,
            _ => Self::None,
        }
    }
}

/// Joint kind. Lays out which transform map a joint imposes between
/// its parent and child.
///
/// Numeric values are kept stable for serialisation and for
/// matching Newton's `JointType` enum:
///
/// | enum         | u32 | coord | dof |
/// |--------------|-----|-------|-----|
/// | `Prismatic`  | 0   | 1     | 1   |
/// | `Revolute`   | 1   | 1     | 1   |
/// | `Ball`       | 2   | 4     | 3   |
/// | `Fixed`      | 3   | 0     | 0   |
/// | `Free`       | 4   | 7     | 6   |
/// | `Distance`   | 5   | 7     | 6   |
/// | `D6`         | 6   | 7     | 6   |
/// | `Cable`      | 7   | 2     | 2   |
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum JointType {
    /// 1 DoF translation along a fixed axis (1 coord, 1 dof).
    Prismatic = 0,
    /// 1 DoF rotation about a fixed axis (1 coord, 1 dof).
    Revolute = 1,
    /// 3-DoF spherical / ball joint — rotation only, parameterised
    /// as a quaternion (4 coord) with body-frame angular velocity
    /// (3 dof). Full Featherstone forward dynamics — gravity-driven
    /// pendulum tests pass on both CPU and GPU backends.
    Ball = 2,
    /// Locks all relative motion (0 coord, 0 dof).
    Fixed = 3,
    /// 6 DoF unconstrained motion (7 coord — quaternion + translation, 6 dof).
    Free = 4,
    /// 6-DoF "soft" distance joint: like `Free` but with `min_distance`
    /// / `max_distance` between the joint anchors enforced by XPBD.
    /// Storage matches `Free` (7 coord, 6 dof); the limits live in
    /// the joint's per-DOF limit arrays.
    Distance = 5,
    /// 6-DoF joint with per-axis lock/unlock + per-axis limits. The
    /// per-DOF axis lock pattern is encoded via the existing
    /// `joint_limit_lower / joint_limit_upper` arrays (a locked
    /// axis sets `lower == upper == 0`).
    D6 = 6,
    /// Newton-style CABLE joint (Stage 39E) — 1 linear stretch DoF
    /// along the joint-local **+z axis** + 1 isotropic bend DoF about
    /// the joint-local **+x axis** (the perpendicular "bend" axis is
    /// chosen deterministically as the joint's local x). 2 coords:
    /// `(q_linear_m, q_bend_rad)`; 2 dofs.
    ///
    /// Per-joint stiffness + damping live in `Model.joint_cable_*`
    /// parallel buffers (see [`super::ModelBuilder::add_joint_cable`]):
    ///
    /// ```text
    ///   τ_l = −k_l · q_l − c_l · q̇_l
    ///   τ_b = −k_b · q_b − c_b · q̇_b
    /// ```
    ///
    /// They model an axial-stretch spring (tendon stretch) and an
    /// isotropic bend spring (rope curvature) and are added to the
    /// generalised force in Featherstone's backward pass.
    Cable = 7,
}

impl JointType {
    /// `(coord_count, dof_count)` — coord is the size in `joint_q`,
    /// dof is the size in `joint_qd`. They differ for joints whose
    /// orientation is parameterised by a quaternion.
    #[must_use]
    pub fn coord_dof(self) -> (usize, usize) {
        match self {
            JointType::Prismatic => (1, 1),
            JointType::Revolute => (1, 1),
            JointType::Ball => (4, 3),
            JointType::Fixed => (0, 0),
            JointType::Free => (7, 6),
            JointType::Distance => (7, 6),
            JointType::D6 => (7, 6),
            JointType::Cable => (2, 2),
        }
    }

    /// Number of joint coordinates (entries in `joint_q`).
    #[must_use]
    pub fn coord_count(self) -> usize {
        self.coord_dof().0
    }

    /// Number of joint velocities (entries in `joint_qd`).
    #[must_use]
    pub fn dof_count(self) -> usize {
        self.coord_dof().1
    }
}

impl JointType {
    /// Initial coordinates for a fresh joint at "zero". Each joint
    /// type writes `coord_count()` values.
    pub(crate) fn write_zero_coords(self, out: &mut Vec<Real>) {
        match self {
            JointType::Revolute | JointType::Prismatic => out.push(0.0),
            JointType::Fixed => {}
            JointType::Ball => {
                // 4 coords = identity quaternion (qx, qy, qz, qw).
                out.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            }
            JointType::Free | JointType::Distance | JointType::D6 => {
                // 7 coords: [tx, ty, tz, qx, qy, qz, qw] — identity.
                out.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            }
            JointType::Cable => {
                // 2 coords: linear stretch (m), bend angle (rad).
                out.extend_from_slice(&[0.0, 0.0]);
            }
        }
    }
}
