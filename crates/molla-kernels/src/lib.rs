//! WGSL compute kernel dispatch wrappers.

pub use molla_autograd as autograd;
#[cfg(feature = "gpu")]
pub use molla_compute as compute;
pub use molla_core as core;
pub use molla_math as math;

#[cfg(feature = "gpu")]
mod actuator_delay;
#[cfg(feature = "gpu")]
mod actuator_neural_lstm;
#[cfg(feature = "gpu")]
mod actuator_neural_mlp;
#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
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
mod contact_reduction;
#[cfg(feature = "gpu")]
mod contact_sensor;
#[cfg(feature = "gpu")]
mod crba;
#[cfg(feature = "gpu")]
mod dahl_friction;
mod dahl_oracle;
#[cfg(feature = "gpu")]
mod dense3;
#[cfg(feature = "gpu")]
mod featherstone;
mod featherstone_adjoint;
#[cfg(feature = "gpu")]
mod featherstone_adjoint_gpu;
#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
mod frame_transform;
#[cfg(feature = "gpu")]
mod hydroelastic;
#[cfg(feature = "gpu")]
mod implicit_mpm;
#[cfg(feature = "gpu")]
mod imu;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod jacobian;
#[cfg(feature = "gpu")]
mod muscle;
#[cfg(feature = "gpu")]
pub(crate) mod optim_adam;
#[cfg(feature = "gpu")]
mod optim_sgd;
#[cfg(feature = "gpu")]
mod raycast;
#[cfg(feature = "gpu")]
mod rnea;
#[cfg(feature = "gpu")]
mod robust_loss;
#[cfg(feature = "gpu")]
mod scan;
#[cfg(feature = "gpu")]
mod soft_bvh_gpu;
#[cfg(feature = "gpu")]
mod soft_vt_contact;
#[cfg(feature = "gpu")]
mod sparse_gpu;
#[cfg(feature = "gpu")]
mod sparse_grid;
#[cfg(feature = "gpu")]
mod sparse_grid_delete;
#[cfg(feature = "gpu")]
pub(crate) mod sparse_grid_insert;
#[cfg(feature = "gpu")]
mod splat_raytrace;
#[cfg(feature = "gpu")]
mod tiled_camera;
#[cfg(feature = "gpu")]
mod tire_fem;
#[cfg(feature = "gpu")]
mod vbd_cloth;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;

#[cfg(feature = "gpu")]
pub use actuator_delay::ActuatorDelayKernel;
#[cfg(feature = "gpu")]
pub use actuator_neural_lstm::ActuatorNeuralLstmKernel;
#[cfg(feature = "gpu")]
pub use actuator_neural_mlp::ActuatorNeuralMlpKernel;
#[cfg(feature = "gpu")]
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
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
pub use contact_reduction::{
    CONTACT_STRIDE, ContactReductionKernel, MAX_OUT_PER_GROUP, MAX_PER_GROUP, pack_groups_host,
    unpack_output_host,
};
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
#[cfg(feature = "gpu")]
pub use crba::MassMatrixKernel;
#[cfg(feature = "gpu")]
pub use dahl_friction::DahlFrictionKernel;
pub use dahl_oracle::{DahlConfig, dahl_oracle_step};
#[cfg(feature = "gpu")]
pub use dense3::{Dense3Kernel, Dense3Op, Dense3Result};
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
pub use fft::{Complex32, FftKernel};
#[cfg(feature = "gpu")]
pub use frame_transform::FrameTransformKernel;
#[cfg(feature = "gpu")]
pub use hydroelastic::HydroelasticKernel;
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
pub use imu::ImuKernel;
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use jacobian::{JacobianKernel, JacobianRequest, jacobian_gpu};
#[cfg(feature = "gpu")]
pub use molla_compute_primitives::{
    RadixSortKernel, RadixSortKernel64, radix_sort_pairs, radix_sort_pairs_u64,
};
#[cfg(feature = "gpu")]
pub use molla_geometry::{
    EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE, EPA_STATUS_OK, EpaKernel, EpaPairResult,
    GJK_SIMPLEX_STRIDE, GjkKernel, MPR_DEFAULT_TOLERANCE, MPR_OUT_STRIDE, MPR_STATUS_FALLBACK,
    MPR_STATUS_NONE, MPR_STATUS_OK, MprKernel, MprPairResult, decode_epa_out, decode_mpr_out,
};
#[cfg(feature = "gpu")]
pub use molla_geometry::{LbvhBroadPhase, LbvhPairs, collect_shape_aabbs};
#[cfg(feature = "gpu")]
pub use molla_geometry::{MeshBvhGpu, MeshBvhRaycastKernel};
#[cfg(feature = "gpu")]
pub use muscle::MuscleKernel;
#[cfg(feature = "gpu")]
pub use optim_adam::AdamGpuKernel;
#[cfg(feature = "gpu")]
pub use optim_sgd::SgdGpuKernel;
#[cfg(feature = "gpu")]
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use rnea::{InverseDynamicsKernel, inverse_dynamics_gpu};
#[cfg(feature = "gpu")]
pub use robust_loss::{LossPair, RobustLossKernel};
#[cfg(feature = "gpu")]
pub use scan::ScanKernel;
#[cfg(feature = "gpu")]
pub use soft_bvh_gpu::SoftBvhGpu;
#[cfg(feature = "gpu")]
pub use soft_vt_contact::{SoftVtContactKernel, SoftVtContactStats, VtContactGpu};
#[cfg(feature = "gpu")]
pub use sparse_gpu::{BsrAlgebraKernel, BsrGpu, SPGEMM_GPU_SYMBOLIC_CUTOVER};
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
pub use tiled_camera::{CameraRaysKernel, CameraTraceKernel};
#[cfg(feature = "gpu")]
pub use tire_fem::{TireFemGpuInput, TireFemGpuKernel, TireFemGpuOutput, TireFemGpuParams};
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
