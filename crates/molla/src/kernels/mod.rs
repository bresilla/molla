//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
#[cfg(feature = "gpu")]
mod contact_sensor;
#[cfg(feature = "gpu")]
mod crba;
#[cfg(feature = "gpu")]
mod epa;
#[cfg(feature = "gpu")]
mod featherstone;
#[cfg(feature = "gpu")]
mod gjk;
#[cfg(feature = "gpu")]
mod mesh_bvh_raycast;
#[cfg(feature = "gpu")]
mod frame_transform;
#[cfg(feature = "gpu")]
mod imu;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod jacobian;
#[cfg(feature = "gpu")]
mod raycast;
#[cfg(feature = "gpu")]
mod rnea;
#[cfg(feature = "gpu")]
mod tiled_camera;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;
#[cfg(feature = "gpu")]
mod vbd_cloth;

#[cfg(feature = "gpu")]
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
#[cfg(feature = "gpu")]
pub use crba::MassMatrixKernel;
#[cfg(feature = "gpu")]
pub use epa::{
    decode_epa_out, EpaKernel, EpaPairResult, EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE,
    EPA_STATUS_OK,
};
#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
#[cfg(feature = "gpu")]
pub use gjk::{GjkKernel, GJK_SIMPLEX_STRIDE};
#[cfg(feature = "gpu")]
pub use mesh_bvh_raycast::{MeshBvhGpu, MeshBvhRaycastKernel};
#[cfg(feature = "gpu")]
pub use frame_transform::FrameTransformKernel;
#[cfg(feature = "gpu")]
pub use imu::ImuKernel;
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use jacobian::{jacobian_gpu, JacobianKernel, JacobianRequest};
#[cfg(feature = "gpu")]
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use rnea::{inverse_dynamics_gpu, InverseDynamicsKernel};
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraRaysKernel, CameraTraceKernel};
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
