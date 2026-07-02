//! Sensor models and sensor adjoints.

#[cfg(feature = "gpu")]
pub mod camera_post;
#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod dataset;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub mod lidar;
#[cfg(feature = "gpu")]
pub mod pointcloud;
#[cfg(feature = "gpu")]
pub mod raycast;
pub mod raycast_adjoint;
#[cfg(feature = "gpu")]
pub mod tiled_camera;

#[cfg(feature = "gpu")]
pub use camera_post::{
    apply_brightness_contrast, apply_exposure, apply_gamma, apply_gamma_srgb, apply_gray_world_wb,
    apply_reinhard_tone_map, apply_reinhard_white_point, apply_saturation, apply_white_patch_wb,
    linear_to_srgb, pack_rgba8, srgb_to_linear, unpack_rgba8,
};
#[cfg(feature = "gpu")]
pub use contact::{BodyContactReading, ContactSensor};
#[cfg(feature = "gpu")]
pub use dataset::{AnnotationFrame, BoundingBox2D, Dataset, MISS_SHAPE_ID};
#[cfg(feature = "gpu")]
pub use frame_transform::{FrameReading, FrameTarget, FrameTransformSensor};
#[cfg(feature = "gpu")]
pub use imu::{ImuMount, ImuReading, ImuSensor};
#[cfg(feature = "gpu")]
pub use lidar::{AerialLidar, AerialScan, AerialScanMetadata, Lidar, Scan, ScanMetadata};
#[cfg(feature = "gpu")]
pub use pointcloud::{
    Compare, GridCell, HitPoint, bin_into_grid, distance_filter, read_xyz_rgb, reflectance_filter,
    rotate_cloud, scalar_filter, transform_cloud, translate_cloud, write_ply_ascii, write_ptx,
    write_xyz_rgb, xyz_filter,
};
#[cfg(feature = "gpu")]
pub use raycast::{RayHit, Raycaster};
pub use raycast_adjoint::{
    Ray as DiffRay, RayScene, SdfShape as DiffSdfShape, compute_grad_raycast, compute_loss_l2,
    forward_with_loss,
};
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraChannels, PhysicalOptics, TiledCamera, TiledCameraConfig};
