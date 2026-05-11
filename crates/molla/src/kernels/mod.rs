//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod actuator_delay;
#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
#[cfg(feature = "gpu")]
mod actuator_neural_lstm;
#[cfg(feature = "gpu")]
mod actuator_neural_mlp;
#[cfg(feature = "gpu")]
mod actuator_pipeline_pd;
#[cfg(feature = "gpu")]
mod actuator_pipeline_pid;
#[cfg(feature = "gpu")]
mod clamping_dc_motor;
#[cfg(feature = "gpu")]
mod clamping_max_effort;
#[cfg(feature = "gpu")]
mod clamping_position_based;
#[cfg(feature = "gpu")]
mod contact_sensor;
#[cfg(feature = "gpu")]
mod crba;
#[cfg(feature = "gpu")]
mod dense3;
#[cfg(feature = "gpu")]
mod epa;
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
mod fem_stokes;
#[cfg(feature = "gpu")]
mod fem_step_ops;
#[cfg(feature = "gpu")]
mod dg_swe_flux;
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
mod gjk;
#[cfg(feature = "gpu")]
mod hydroelastic;
#[cfg(feature = "gpu")]
mod mpr;
#[cfg(feature = "gpu")]
mod muscle;
#[cfg(feature = "gpu")]
mod contact_reduction;
#[cfg(feature = "gpu")]
mod mesh_bvh_raycast;
#[cfg(feature = "gpu")]
mod frame_transform;
#[cfg(feature = "gpu")]
mod imu;
#[cfg(feature = "gpu")]
mod implicit_mpm;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod jacobian;
#[cfg(feature = "gpu")]
mod lbvh;
#[cfg(feature = "gpu")]
mod soft_bvh_gpu;
#[cfg(feature = "gpu")]
mod soft_vt_contact;
#[cfg(feature = "gpu")]
mod radix_sort;
#[cfg(feature = "gpu")]
mod raycast;
#[cfg(feature = "gpu")]
mod scan;
#[cfg(feature = "gpu")]
mod splat_raytrace;
#[cfg(feature = "gpu")]
pub(crate) mod optim_adam;
#[cfg(feature = "gpu")]
mod optim_sgd;
#[cfg(feature = "gpu")]
mod robust_loss;
#[cfg(feature = "gpu")]
mod rnea;
#[cfg(feature = "gpu")]
mod sparse_grid;
#[cfg(feature = "gpu")]
mod sparse_grid_delete;
#[cfg(feature = "gpu")]
pub(crate) mod sparse_grid_insert;
#[cfg(feature = "gpu")]
mod tiled_camera;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;
#[cfg(feature = "gpu")]
mod vbd_cloth;
#[cfg(feature = "gpu")]
mod dahl_friction;

#[cfg(feature = "gpu")]
pub use actuator_delay::ActuatorDelayKernel;
#[cfg(feature = "gpu")]
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
#[cfg(feature = "gpu")]
pub use actuator_neural_lstm::ActuatorNeuralLstmKernel;
#[cfg(feature = "gpu")]
pub use actuator_neural_mlp::ActuatorNeuralMlpKernel;
#[cfg(feature = "gpu")]
pub use actuator_pipeline_pd::ActuatorPipelinePdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pipeline_pid::ActuatorPipelinePidKernel;
#[cfg(feature = "gpu")]
pub use clamping_dc_motor::ClampingDcMotorKernel;
#[cfg(feature = "gpu")]
pub use clamping_max_effort::ClampingMaxEffortKernel;
#[cfg(feature = "gpu")]
pub use clamping_position_based::ClampingPositionBasedKernel;
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
#[cfg(feature = "gpu")]
pub use crba::MassMatrixKernel;
#[cfg(feature = "gpu")]
pub use dense3::{Dense3Kernel, Dense3Op, Dense3Result};
#[cfg(feature = "gpu")]
pub use epa::{
    decode_epa_out, EpaKernel, EpaPairResult, EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE,
    EPA_STATUS_OK,
};
#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
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
pub use fem_stokes::{FemStokesGpuKernel, StokesEntry};
#[cfg(feature = "gpu")]
pub use fem_step_ops::{TransientStepOpsKernel, WaveStepOpsKernel};
#[cfg(feature = "gpu")]
pub use dg_swe_flux::{
    DgSweFace, DgSweFluxKernel, DgSweFluxVariant, DG_SWE_COMPONENTS, DG_SWE_FACE_STRIDE,
    DG_SWE_NODES_PER_CELL,
};
pub use featherstone_adjoint::{
    time_forward_us, AdjointBackend, AdjointResult, FeatherstoneAdjointDispatcher,
    MultiAdjointBatch, MultiAdjointInput, MultiAdjointOutput,
};
#[cfg(feature = "gpu")]
pub use featherstone_adjoint_gpu::{
    FeatherstoneAdjointGpuKernel, GpuChainInput, GpuChainJacobian, MAX_LINKS as ADJ_MAX_LINKS,
    OUT_STRIDE as ADJ_OUT_STRIDE,
};
#[cfg(feature = "gpu")]
pub use fft::{Complex32, FftKernel};
#[cfg(feature = "gpu")]
pub use gjk::{GjkKernel, GJK_SIMPLEX_STRIDE};
#[cfg(feature = "gpu")]
pub use hydroelastic::HydroelasticKernel;
#[cfg(feature = "gpu")]
pub use mpr::{
    decode_mpr_out, MprKernel, MprPairResult, MPR_DEFAULT_TOLERANCE, MPR_OUT_STRIDE,
    MPR_STATUS_FALLBACK, MPR_STATUS_NONE, MPR_STATUS_OK,
};
#[cfg(feature = "gpu")]
pub use muscle::MuscleKernel;
#[cfg(feature = "gpu")]
pub use contact_reduction::{
    pack_groups_host, unpack_output_host, ContactReductionKernel, CONTACT_STRIDE,
    MAX_OUT_PER_GROUP, MAX_PER_GROUP,
};
#[cfg(feature = "gpu")]
pub use mesh_bvh_raycast::{MeshBvhGpu, MeshBvhRaycastKernel};
#[cfg(feature = "gpu")]
pub use frame_transform::FrameTransformKernel;
#[cfg(feature = "gpu")]
pub use imu::ImuKernel;
#[cfg(feature = "gpu")]
pub use implicit_mpm::{
    readback_flat_f32 as implicit_mpm_readback_flat_f32,
    readback_g_mass_f32 as implicit_mpm_readback_g_mass_f32,
    readback_g_momentum_f32 as implicit_mpm_readback_g_momentum_f32,
    readback_v_grid_f32 as implicit_mpm_readback_v_grid_f32,
    readback_v_pred_f32 as implicit_mpm_readback_v_pred_f32,
    readback_vec3_f32 as implicit_mpm_readback_vec3_f32,
    readback_vec3_real as implicit_mpm_readback_vec3_real,
    upload_storage_f32 as implicit_mpm_upload_storage_f32,
    upload_storage_vec3_f32 as implicit_mpm_upload_storage_vec3_f32, GpuStepState as ImplicitMpmGpuStepState,
    ImplicitMpmKernel, StepUniforms as ImplicitMpmStepUniforms,
};
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use jacobian::{jacobian_gpu, JacobianKernel, JacobianRequest};
#[cfg(feature = "gpu")]
pub use lbvh::{collect_shape_aabbs, LbvhBroadPhase, LbvhPairs};
#[cfg(feature = "gpu")]
pub use soft_bvh_gpu::SoftBvhGpu;
#[cfg(feature = "gpu")]
pub use soft_vt_contact::{SoftVtContactKernel, SoftVtContactStats, VtContactGpu};
#[cfg(feature = "gpu")]
pub use radix_sort::{
    radix_sort_pairs, radix_sort_pairs_u64, RadixSortKernel, RadixSortKernel64,
};
#[cfg(feature = "gpu")]
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use scan::ScanKernel;
#[cfg(feature = "gpu")]
pub use splat_raytrace::{
    SplatGradients, SplatRaytraceKernel, SplatShBand3Gradients, SplatShGradients,
};
#[cfg(feature = "gpu")]
pub use optim_adam::AdamGpuKernel;
#[cfg(feature = "gpu")]
pub use optim_sgd::SgdGpuKernel;
#[cfg(feature = "gpu")]
pub use robust_loss::{LossPair, RobustLossKernel};
#[cfg(feature = "gpu")]
pub use rnea::{inverse_dynamics_gpu, InverseDynamicsKernel};
#[cfg(feature = "gpu")]
pub use sparse_grid::SparseGridKernel;
#[cfg(feature = "gpu")]
pub use sparse_grid_insert::{
    read_storage_to_pairs, SparseGridInsertKernel, SparseGridStorage, SparseGridStorageStats,
    Value, ValueF16, ValueVec3,
};
#[cfg(feature = "gpu")]
pub use sparse_grid_delete::SparseGridDeleteKernel;
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraRaysKernel, CameraTraceKernel};
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
#[cfg(feature = "gpu")]
pub use dahl_friction::{dahl_oracle_step, DahlConfig, DahlFrictionKernel};
