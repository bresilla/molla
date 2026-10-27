//! Shared types used across the crate: errors, IDs, axis enums.

mod error;
mod ids;

#[cfg(feature = "gpu")]
pub mod gpu_metrics;

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
    pub fn unit(self) -> glam::DVec3 {
        match self {
            Axis::X => glam::DVec3::X,
            Axis::Y => glam::DVec3::Y,
            Axis::Z => glam::DVec3::Z,
        }
    }
}
