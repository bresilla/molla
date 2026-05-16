//! wgpu runtime selection and device/queue ownership.

#[cfg(feature = "gpu")]
mod engine;

#[cfg(feature = "gpu")]
pub use engine::{AdapterInfo, Backend, Engine};
