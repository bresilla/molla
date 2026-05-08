//! Joint enums + DOF accounting.
//!
//! Mirrors Newton's `JointType` shape (see
//! `xtra/newton/newton/_src/sim/enums.py`). Stage 3.0 ships
//! `Revolute`, `Prismatic`, `Fixed`, `Free`; ball / distance / D6 /
//! cable arrive in stage 3.x as needed.

use crate::math::Real;

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
