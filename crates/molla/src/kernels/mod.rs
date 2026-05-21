//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod crba;
#[cfg(feature = "gpu")]
mod dg_swe_flux;
#[cfg(feature = "gpu")]
mod featherstone;
#[cfg(feature = "gpu")]
mod fem;
#[cfg(feature = "gpu")]
mod fem_cg;
#[cfg(feature = "gpu")]
mod fem_elasticity;
#[cfg(feature = "gpu")]
mod fem_ns_advect;
#[cfg(feature = "gpu")]
mod fem_ns_rhs_assemble;
#[cfg(feature = "gpu")]
mod fem_step_ops;
#[cfg(feature = "gpu")]
mod fem_stokes;
// Adjoint dispatcher is feature-gated separately — it does not (yet)
// require GPU and is needed by the autograd CPU path + viewer
// regardless of the `gpu` feature.
mod featherstone_adjoint;
// Stage 23.1.2 GPU multi-articulation Jacobian kernel — only compiled
// when the gpu feature is on (needs wgpu for compile + dispatch).
#[cfg(feature = "gpu")]
mod featherstone_adjoint_gpu;
#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
mod implicit_mpm;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod jacobian;
#[cfg(feature = "gpu")]
mod lbvh;
#[cfg(feature = "gpu")]
mod rnea;
#[cfg(feature = "gpu")]
mod soft_bvh_gpu;
#[cfg(feature = "gpu")]
mod soft_vt_contact;
#[cfg(feature = "gpu")]
mod sparse_grid;
#[cfg(feature = "gpu")]
mod sparse_grid_delete;
#[cfg(feature = "gpu")]
pub(crate) mod sparse_grid_insert;
#[cfg(feature = "gpu")]
mod splat_raytrace;
#[cfg(feature = "gpu")]
mod vbd_cloth;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;

#[cfg(feature = "gpu")]
pub use crba::MassMatrixKernel;
#[cfg(feature = "gpu")]
pub use dg_swe_flux::{
    DG_SWE_COMPONENTS, DG_SWE_FACE_STRIDE, DG_SWE_NODES_PER_CELL, DgSweFace, DgSweFluxKernel,
    DgSweFluxVariant,
};
#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
pub use featherstone_adjoint::{
    AdjointBackend, AdjointResult, FeatherstoneAdjointDispatcher, MultiAdjointBatch,
    MultiAdjointInput, MultiAdjointOutput, time_forward_us,
};
#[cfg(feature = "gpu")]
pub use featherstone_adjoint_gpu::{
    FeatherstoneAdjointGpuKernel, GpuChainInput, GpuChainJacobian, MAX_LINKS as ADJ_MAX_LINKS,
    OUT_STRIDE as ADJ_OUT_STRIDE,
};
#[cfg(feature = "gpu")]
pub use fem::FemGpuKernel;
#[cfg(feature = "gpu")]
pub use fem_cg::{CgGpuResult, FemCgGpuKernel};
#[cfg(feature = "gpu")]
pub use fem_elasticity::FemElasticityGpuKernel;
#[cfg(feature = "gpu")]
pub use fem_ns_advect::FemNsAdvectKernel;
#[cfg(feature = "gpu")]
pub use fem_ns_rhs_assemble::{FemNsRhsAssembleKernel, NsRhsMode};
#[cfg(feature = "gpu")]
pub use fem_step_ops::{TransientStepOpsKernel, WaveStepOpsKernel};
#[cfg(feature = "gpu")]
pub use fem_stokes::{FemStokesGpuKernel, StokesEntry};
#[cfg(feature = "gpu")]
pub use fft::{Complex32, FftKernel};
#[cfg(feature = "gpu")]
pub use implicit_mpm::{
    GpuStepState as ImplicitMpmGpuStepState, ImplicitMpmKernel,
    StepUniforms as ImplicitMpmStepUniforms, readback_flat_f32 as implicit_mpm_readback_flat_f32,
    readback_g_mass_f32 as implicit_mpm_readback_g_mass_f32,
    readback_g_momentum_f32 as implicit_mpm_readback_g_momentum_f32,
    readback_v_grid_f32 as implicit_mpm_readback_v_grid_f32,
    readback_v_pred_f32 as implicit_mpm_readback_v_pred_f32,
    readback_vec3_f32 as implicit_mpm_readback_vec3_f32,
    readback_vec3_real as implicit_mpm_readback_vec3_real,
    upload_storage_f32 as implicit_mpm_upload_storage_f32,
    upload_storage_vec3_f32 as implicit_mpm_upload_storage_vec3_f32,
};
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use jacobian::{JacobianKernel, JacobianRequest, jacobian_gpu};
#[cfg(feature = "gpu")]
pub use lbvh::{LbvhBroadPhase, LbvhPairs, collect_shape_aabbs};
#[cfg(feature = "gpu")]
pub use molla_kernels::{
    ActuatorDelayKernel, ActuatorNeuralLstmKernel, ActuatorNeuralMlpKernel, ActuatorPdKernel,
    ActuatorPidKernel, ActuatorPipelinePdKernel, ActuatorPipelinePidKernel, AdamGpuKernel,
    BsrAlgebraKernel, BsrGpu, CONTACT_STRIDE, CameraRaysKernel, CameraTraceKernel,
    ClampingDcMotorKernel, ClampingMaxEffortKernel, ClampingPositionBasedKernel,
    ContactAggregateKernel, ContactReductionKernel, DahlConfig, DahlFrictionKernel, Dense3Kernel,
    Dense3Op, Dense3Result, EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE, EPA_STATUS_OK,
    EpaKernel, EpaPairResult, FrameTransformKernel, GJK_SIMPLEX_STRIDE, GjkKernel,
    HydroelasticKernel, ImuKernel, LossPair, MAX_OUT_PER_GROUP, MAX_PER_GROUP,
    MPR_DEFAULT_TOLERANCE, MPR_OUT_STRIDE, MPR_STATUS_FALLBACK, MPR_STATUS_NONE, MPR_STATUS_OK,
    MeshBvhGpu, MeshBvhRaycastKernel, MprKernel, MprPairResult, MuscleKernel, RadixSortKernel,
    RadixSortKernel64, RaycastKernel, RobustLossKernel, SPGEMM_GPU_SYMBOLIC_CUTOVER, ScanKernel,
    SgdGpuKernel, TireFemGpuInput, TireFemGpuKernel, TireFemGpuOutput, TireFemGpuParams,
    dahl_oracle_step, decode_epa_out, decode_mpr_out, pack_groups_host, radix_sort_pairs,
    radix_sort_pairs_u64, unpack_output_host,
};
#[cfg(feature = "gpu")]
pub use rnea::{InverseDynamicsKernel, inverse_dynamics_gpu};
#[cfg(feature = "gpu")]
pub use soft_bvh_gpu::SoftBvhGpu;
#[cfg(feature = "gpu")]
pub use soft_vt_contact::{SoftVtContactKernel, SoftVtContactStats, VtContactGpu};
#[cfg(feature = "gpu")]
pub use sparse_grid::SparseGridKernel;
#[cfg(feature = "gpu")]
pub use sparse_grid_delete::SparseGridDeleteKernel;
#[cfg(feature = "gpu")]
pub use sparse_grid_insert::{
    SparseGridInsertKernel, SparseGridStorage, SparseGridStorageStats, Value, ValueF16, ValueVec3,
    read_storage_to_pairs,
};
#[cfg(feature = "gpu")]
pub use splat_raytrace::{
    SplatGradients, SplatRaytraceKernel, SplatShBand3Gradients, SplatShGradients,
};
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
