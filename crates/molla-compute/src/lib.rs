//! wgpu runtime selection and device/queue ownership.

#[cfg(feature = "gpu")]
mod engine;

#[cfg(feature = "gpu")]
mod program_cache;

#[cfg(feature = "gpu")]
pub use engine::{AdapterInfo, Backend, DeviceHealth, DeviceLoss, Engine};
