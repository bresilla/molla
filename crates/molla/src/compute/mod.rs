//! GPU runtime: `wgpu` instance, adapter, device, queue.
//!
//! [`Engine`] owns the device + queue used by every GPU kernel in the
//! crate; pick a [`Backend`] to select the adapter (auto / native GPU /
//! browser WebGPU / CPU fallback). Kernel dispatch wrappers live in
//! [`crate::kernels`] — this module only stands up the runtime.
//!
//! [`MpmSimulation`] is the persistent-buffer MPM driver (Stage 24d.x);
//! see [`MpmConfig`] / [`MpmRheology`] / [`MpmTimeStep`] for configuration.

#[cfg(feature = "gpu")]
mod engine;
#[cfg(feature = "gpu")]
mod mpm;

#[cfg(feature = "gpu")]
pub use engine::{AdapterInfo, Backend, Engine};
#[cfg(feature = "gpu")]
pub use mpm::{MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};
