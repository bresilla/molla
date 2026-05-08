//! Joint enums + DOF accounting.
//!
//! Mirrors Newton's `JointType` shape (see
//! `xtra/newton/newton/_src/sim/enums.py`). Stage 3.0 ships
//! `Revolute`, `Prismatic`, `Fixed`, `Free`; ball / distance / D6 /
//! cable arrive in stage 3.x as needed.

use crate::math::Real;

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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum JointType {
    /// 1 DoF translation along a fixed axis (1 coord, 1 dof).
    Prismatic = 0,
    /// 1 DoF rotation about a fixed axis (1 coord, 1 dof).
    Revolute = 1,
    /// Locks all relative motion (0 coord, 0 dof).
    Fixed = 3,
    /// 6 DoF unconstrained motion (7 coord — quaternion + translation, 6 dof).
    Free = 4,
}

impl JointType {
    /// `(coord_count, dof_count)` — coord is the size in `joint_q`,
    /// dof is the size in `joint_qd`. They differ for `Free` because
    /// the rotation is parameterised as a quaternion.
    #[must_use]
    pub fn coord_dof(self) -> (usize, usize) {
        match self {
            JointType::Prismatic => (1, 1),
            JointType::Revolute => (1, 1),
            JointType::Fixed => (0, 0),
            JointType::Free => (7, 6),
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
            JointType::Free => {
                // [tx, ty, tz, qx, qy, qz, qw] — identity = (0,0,0, 0,0,0,1)
                out.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            }
        }
    }
}
