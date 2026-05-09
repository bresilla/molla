//! GPU runtime: `wgpu` instance, adapter, device, queue.
//!
//! Stage 0 just stands up the runtime and exposes adapter info. Kernel
//! dispatch wrappers land in stage 2.

#[cfg(feature = "gpu")]
mod engine;
#[cfg(feature = "gpu")]
mod mpm;

#[cfg(feature = "gpu")]
pub use engine::{AdapterInfo, Backend, Engine};
#[cfg(feature = "gpu")]
pub use mpm::{MpmConfig, MpmSimulation};
