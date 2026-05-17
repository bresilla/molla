//! Facade compatibility for GPU BSR sparse algebra.
//!
//! The implementation lives in `molla-kernels`; this module preserves the
//! historical `molla::math::{BsrGpu, BsrAlgebraKernel}` path.

pub use molla_kernels::{BsrAlgebraKernel, BsrGpu, SPGEMM_GPU_SYMBOLIC_CUTOVER};
