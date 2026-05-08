//! Shared types used across the crate: errors, IDs, axis enums.

mod error;
mod ids;

pub use error::{Error, Result};
pub use ids::{BodyId, JointId, ParticleId, ShapeId, WorldId};

/// Sentinel for "no maximum" / "unbounded" in joint limits and similar
/// places. Mirrors Newton's `MAXVAL`.
pub const MAXVAL: f64 = 1.0e30;

/// Cardinal axes. Used by joint axis specs, raycasts, and conventions.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Axis {
    /// Positive X.
    X,
    /// Positive Y.
    Y,
    /// Positive Z.
    Z,
}

impl Axis {
    /// Unit vector for this axis.
    #[must_use]
    pub fn unit(self) -> crate::math::Vec3 {
        match self {
            Axis::X => crate::math::Vec3::X,
            Axis::Y => crate::math::Vec3::Y,
            Axis::Z => crate::math::Vec3::Z,
        }
    }
}
