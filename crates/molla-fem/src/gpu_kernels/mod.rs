//! FEM-specific GPU dispatch wrappers.
//!
//! These kernels stay with the FEM crate because their API is tied to
//! FEM-owned data structures such as [`crate::FemGpuData`].

pub mod dg_swe_flux;
pub mod dg_swe_limiter;
pub mod fem;
pub mod fem_cg;
pub mod fem_elasticity;
pub mod fem_ns_advect;
pub mod fem_ns_rhs_assemble;
pub mod fem_step_ops;
pub mod fem_stokes;

pub use dg_swe_flux::{
    DG_SWE_COMPONENTS, DG_SWE_FACE_STRIDE, DG_SWE_NODES_PER_CELL, DgSweFace, DgSweFluxKernel,
    DgSweFluxVariant,
};
pub use dg_swe_limiter::{DgSweLimiterCell, DgSweLimiterKernel, pack_dg_swe_limiter_cells};
pub use fem::FemGpuKernel;
pub use fem_cg::{CgGpuResult, FemCgGpuKernel};
pub use fem_elasticity::FemElasticityGpuKernel;
pub use fem_ns_advect::FemNsAdvectKernel;
pub use fem_ns_rhs_assemble::{FemNsRhsAssembleKernel, NsRhsMode};
pub use fem_step_ops::{TransientStepOpsKernel, WaveStepOpsKernel};
pub use fem_stokes::{FemStokesGpuKernel, StokesEntry};
