//! Sensor models and sensor adjoints.

#[cfg(feature = "gpu")]
pub mod contact;
#[cfg(feature = "gpu")]
pub mod frame_transform;
#[cfg(feature = "gpu")]
pub mod imu;
#[cfg(feature = "gpu")]
pub mod raycast;
pub mod raycast_adjoint;
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
pub use raycast_adjoint::{
    Ray as DiffRay, RayScene, SdfShape as DiffSdfShape, compute_grad_raycast, compute_loss_l2,
    forward_with_loss,
};
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraChannels, TiledCamera, TiledCameraConfig};
