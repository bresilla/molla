//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
#[cfg(feature = "gpu")]
mod contact_sensor;
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
mod raycast;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;

#[cfg(feature = "gpu")]
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
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
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
