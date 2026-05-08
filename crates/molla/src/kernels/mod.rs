//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
#[cfg(feature = "gpu")]
mod contact_sensor;
#[cfg(feature = "gpu")]
mod featherstone;
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
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
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
