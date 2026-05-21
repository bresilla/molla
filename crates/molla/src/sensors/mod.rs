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
//!   top. Sphere / Plane / Box / Capsule / Cylinder / Cone /
//!   Ellipsoid supported analytically; static TriMesh shapes are
//!   handled via a second GPU pass over the mesh BVH (H1).
//! - [`lidar`] — terrestrial-LiDAR scan model: a (θ, φ) angular grid
//!   on top of `Raycaster`, with metadata-aware [`Scan`] output (H4).
//! - [`imu`] — body-frame linear acceleration + angular velocity.
//! - [`contact`] — per-body contact summary.
//! - [`frame_transform`] — pose tracking with optional smoothing.
//! - [`tiled_camera`] — tiled depth/colour/normal/albedo camera for
//!   RL pixel observations (Stage 14).
//! - [`raycast_adjoint`] — CPU finite-difference adjoint for the
//!   raycast pipeline (Stage 37.7). Differentiable raycasting for
//!   inverse-rendering optimisations; CPU-only by design (FD probes
//!   the forward pass `O(P · N_rays)` times, no GPU buffers needed).

#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub mod camera_post {
    pub use molla_sensors::camera_post::*;
}
#[cfg(feature = "gpu")]
pub mod dataset {
    pub use molla_sensors::dataset::*;
}
#[cfg(feature = "gpu")]
pub mod lidar {
    pub use molla_sensors::lidar::*;
}
#[cfg(feature = "gpu")]
pub mod pointcloud {
    pub use molla_sensors::pointcloud::*;
}
#[cfg(feature = "gpu")]
pub mod raycast;
pub mod raycast_adjoint {
    pub use molla_sensors::raycast_adjoint::*;
}
#[cfg(feature = "gpu")]
pub mod tiled_camera;

#[cfg(feature = "gpu")]
pub use contact::{BodyContactReading, ContactSensor};
#[cfg(feature = "gpu")]
pub use frame_transform::{FrameReading, FrameTarget, FrameTransformSensor};
#[cfg(feature = "gpu")]
pub use imu::{ImuMount, ImuReading, ImuSensor};
#[cfg(feature = "gpu")]
pub use molla_sensors::{
    AerialLidar, AerialScan, AerialScanMetadata, AnnotationFrame, BoundingBox2D, Compare, Dataset,
    GridCell, HitPoint, Lidar, MISS_SHAPE_ID, Scan, ScanMetadata, bin_into_grid, distance_filter,
    read_xyz_rgb, reflectance_filter, rotate_cloud, scalar_filter, transform_cloud,
    translate_cloud, write_ply_ascii, write_ptx, write_xyz_rgb, xyz_filter,
};
pub use molla_sensors::{
    DiffRay, DiffSdfShape, RayScene, compute_grad_raycast, compute_loss_l2, forward_with_loss,
};
#[cfg(feature = "gpu")]
pub use raycast::{RayHit, Raycaster};
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraChannels, PhysicalOptics, TiledCamera, TiledCameraConfig};
