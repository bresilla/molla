//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod featherstone;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod raycast;
#[cfg(feature = "gpu")]
mod xpbd;

#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
