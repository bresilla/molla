//! Facade re-export for host-side math plus legacy GPU sparse algebra.

pub use molla_math::*;

#[cfg(feature = "gpu")]
pub mod sparse_gpu;

#[cfg(feature = "gpu")]
pub use sparse_gpu::{BsrAlgebraKernel, BsrGpu, SPGEMM_GPU_SYMBOLIC_CUTOVER};
