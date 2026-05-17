//! Facade compatibility wrapper for Featherstone adjoint GPU dispatch.

pub use molla_kernels::{
    ADJ_MAX_LINKS as MAX_LINKS, ADJ_OUT_STRIDE as OUT_STRIDE, FeatherstoneAdjointGpuKernel,
    GpuChainInput, GpuChainJacobian,
};
