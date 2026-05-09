//! Host-side math types.
//!
//! Everything here is `f64`. The GPU-side mirrors ([`GpuTransform`],
//! [`GpuSpatialVector`], [`GpuMat3`]) are `f32`. Conversion happens
//! exactly once, at the buffer boundary.

pub mod dense3;
pub mod dense_solve;
pub mod gpu;
pub mod iterative;
pub mod robust_loss;
mod spatial;
pub mod sparse;
mod transform;

pub use gpu::{GpuMat3, GpuSpatialVector, GpuTransform};
pub use spatial::{
    spatial_cross, spatial_cross_dual, spatial_jacobian_column, JacobianKind,
    SpatialInertia, SpatialMatrix6, SpatialVector,
};
pub use transform::Transform;

/// Host-side scalar. The crate is `f64` everywhere on the host — see the
/// precision policy in `PLAN.md` §7.
pub type Real = f64;

/// Host-side 3D vector.
pub type Vec3 = glam::DVec3;

/// Host-side unit quaternion (w, x, y, z layout per `glam`).
pub type Quat = glam::DQuat;

/// Host-side 3×3 matrix.
pub type Mat3 = glam::DMat3;

/// Standard gravity, m/s². Provided as a convenience constant.
pub const GRAVITY: Real = 9.806_65;
