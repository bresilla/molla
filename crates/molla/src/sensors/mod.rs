//! Sensors — primitives for closing the perception loop in RL /
//! teleop scenes.
//!
//! **GPU-first** by construction: each sensor's heavy compute lives
//! in WGSL (`crates/molla/shaders/<sensor>.wgsl`) with a Rust
//! dispatch wrapper and a thin orchestrator. CPU fallback paths
//! exist only as test oracles, not as the primary execution path.
//!
//! Modules:
//! - [`raycast`] — closest-hit ray vs collision shapes. Lives in
//!   `shaders/raycast.wgsl` with the [`Raycaster`] orchestrator on
//!   top. Sphere / Plane / Box / Capsule / Cylinder supported on GPU;
//!   Triangle / TriMesh skipped (mesh BVH integration TODO — see the
//!   [`raycast`] module docs).
//! - [`imu`] — body-frame linear acceleration + angular velocity.
//! - [`contact`] — per-body contact summary.
//! - [`frame_transform`] — pose tracking with optional smoothing.
//! - [`tiled_camera`] — tiled depth/colour/normal/albedo camera for
//!   RL pixel observations (Stage 14).

#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub mod raycast;
#[cfg(feature = "gpu")]
pub mod tiled_camera;

#[cfg(feature = "gpu")]
pub use contact::{BodyContactReading, ContactSensor};
#[cfg(feature = "gpu")]
pub use frame_transform::{FrameReading, FrameTarget, FrameTransformSensor};
#[cfg(feature = "gpu")]
pub use imu::{ImuMount, ImuReading, ImuSensor};
#[cfg(feature = "gpu")]
pub use raycast::{RayHit, Raycaster};
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraChannels, TiledCamera, TiledCameraConfig};
