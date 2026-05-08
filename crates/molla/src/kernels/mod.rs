//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod integrate_bodies;

#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
