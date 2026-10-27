//! Facade compatibility wrapper for the implicit MPM GPU kernel.

pub use molla_kernels::{
    ImplicitMpmGpuStepState as GpuStepState, ImplicitMpmKernel,
    ImplicitMpmStepUniforms as StepUniforms, implicit_mpm_readback_flat_f32 as readback_flat_f32,
    implicit_mpm_readback_g_mass_f32 as readback_g_mass_f32,
    implicit_mpm_readback_g_momentum_f32 as readback_g_momentum_f32,
    implicit_mpm_readback_v_grid_f32 as readback_v_grid_f32,
    implicit_mpm_readback_v_pred_f32 as readback_v_pred_f32,
    implicit_mpm_readback_vec3_f32 as readback_vec3_f32,
    implicit_mpm_readback_vec3_real as readback_vec3_real,
    implicit_mpm_upload_storage_f32 as upload_storage_f32,
    implicit_mpm_upload_storage_vec3_f32 as upload_storage_vec3_f32,
};
