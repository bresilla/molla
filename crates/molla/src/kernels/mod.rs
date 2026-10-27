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
pub use featherstone::{FeatherstoneExecution, FeatherstoneKernel, FeatherstoneRounding};
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
    BrushTerrain, BrushTerrainGpu, BrushWheelInput, BrushWheelOutput, BsrAlgebraKernel, BsrGpu,
    CONTACT_STRIDE, CameraRaysKernel, CameraTraceKernel, ClampingDcMotorKernel,
    ClampingMaxEffortKernel, ClampingPositionBasedKernel, ContactAggregateKernel,
    ContactReductionKernel, DahlConfig, DahlFrictionKernel, Dense3Kernel, Dense3Op, Dense3Result,
    EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE, EPA_STATUS_OK, EpaKernel, EpaPairResult,
    FrameTransformKernel, GJK_SIMPLEX_STRIDE, GjkKernel, HydroelasticKernel, ImuKernel, LossPair,
    MAX_OUT_PER_GROUP, MAX_PER_GROUP, MPR_DEFAULT_TOLERANCE, MPR_OUT_STRIDE, MPR_STATUS_FALLBACK,
    MPR_STATUS_NONE, MPR_STATUS_OK, MeshBvhGpu, MeshBvhRaycastKernel, MprKernel, MprPairResult,
    MuscleKernel, PreparedSoilSurface, PreparedSoilTerrainQueries, PreparedTireBrushTerrain,
    PreparedTireDemRaster, PreparedTireFemCavity, PreparedTireFemTerrainFeedback,
    PreparedTireMpmCompactionReadback, PreparedTireMpmFeedback, PreparedTireMpmRaster,
    PreparedTireMpmTerrainReadback, PreparedTireVehicleChassisStep, PreparedTireVehicleFemReadback,
    PreparedTireVehicleFemStep, RadixSortKernel, RadixSortKernel64, RaycastKernel,
    RobustLossKernel, SPGEMM_GPU_SYMBOLIC_CUTOVER, ScanKernel, SgdGpuKernel, SoilSurfaceConfig,
    SoilSurfaceObservation, SoilSurfaceSnapshot, SoilTerrainSample, TireBekkerContactCell,
    TireBekkerFemTerrainOwner, TireBekkerGpuInput, TireBekkerGpuKernel, TireBekkerGpuOutput,
    TireBekkerGpuResidentTerrain, TireBekkerResidentOutput, TireBrushGpuBatch, TireBrushGpuKernel,
    TireBrushResidentOutput, TireDemContactAudit, TireDemContactCell, TireDemFemTerrainOwner,
    TireDemFemTerrainStepReport, TireDemGpuBuiltGridStepInput, TireDemGpuInput, TireDemGpuKernel,
    TireDemGpuLbvhLeafRows, TireDemGpuLbvhTreeRows, TireDemGpuMortonOrder, TireDemGpuOutput,
    TireDemGpuRadixGridStepInput, TireDemGridStepGpuInput, TireDemLbvhDispatchPlan,
    TireDemLbvhNode, TireDemLbvhStepGpuInput, TireDemResidentFemTerrainOwner, TireDemResidentLbvh,
    TireDemResidentOutput, TireDemResidentState, TireDemResidentTerrainOutput,
    TireDemResidentTerrainParams, TireDemResidentTerrainReadback, TireDemSparseLbvh,
    TireDemStepGpuInput, TireDemStepParams, TireDemSurfaceVelocityMap, TireDemTopologyStats,
    TireDemUnboundedSortedGrid, TireFemContactFeedbackGpuTerrain, TireFemContactFeedbackInput,
    TireFemContactFeedbackKernel, TireFemContactFeedbackOutput, TireFemContactFeedbackParams,
    TireFemContactFeedbackPatchAccounting, TireFemContactFeedbackResidentOutput,
    TireFemContactReductionGpuTerrain, TireFemContactReductionInput, TireFemContactReductionKernel,
    TireFemContactReductionOutput, TireFemContactReductionParams, TireFemContactSdfPrimitive,
    TireFemContactTerrain, TireFemContactTriangle, TireFemGpuAtlasBatch,
    TireFemGpuAtlasHubParamRow, TireFemGpuBatchAtlasBuffers, TireFemGpuBatchDispatchPlan,
    TireFemGpuInput, TireFemGpuKernel, TireFemGpuOutput, TireFemGpuParams, TireFemHubParams,
    TireFemPersistentMesh, TireFemTransientFeedbackCommand, TireMpmCompactionObservation,
    TireMpmCompactionReadbackStats, TireMpmCompactionSummary, TireMpmContactCell,
    TireMpmFemTerrainOwner, TireMpmGpuInPlaceParams, TireMpmGpuInput, TireMpmGpuKernel,
    TireMpmGpuOutput, TireMpmResidentFeedback, TireMpmTerrainRasterGpuOutput,
    TireMpmTerrainRasterParams, TireMpmTerrainSnapshot, TireResidentVehicleFemAtlasOwner,
    TireThermalGpuBatch, TireThermalGpuInput, TireThermalGpuKernel, TireThermalGpuOutput,
    TireThermalGpuState, TireThermalRigReplayBatch, TireTransientGpuBatch, TireTransientGpuInput,
    TireTransientGpuKernel, TireTransientGpuOutput, TireTransientGpuState,
    TireVehicleBrushRowsFromSpinGpuInput, TireVehicleBrushRowsGpuInput,
    TireVehicleBrushRowsGpuOutput, TireVehicleBrushRowsResidentOutput, TireVehicleFemFrame,
    TireVehiclePlanarFromBrushGpuInput, TireVehiclePlanarGpuInput, TireVehiclePlanarGpuOutput,
    TireVehiclePlanarResidentState, TireVehicleResidentBrushOwner,
    TireVehicleResidentChassisBrushOwner, TireVehicleSuspensionGpuInput,
    TireVehicleSuspensionGpuOutput, TireVehicleSuspensionResidentBatch,
    TireVehicleVerticalGpuInput, TireVehicleVerticalGpuOutput, TireVehicleVerticalResidentState,
    TireVehicleWheelKinematicsFromSpinGpuInput, TireVehicleWheelKinematicsGpuInput,
    TireVehicleWheelKinematicsGpuOutput, TireVehicleWheelSpinGpuInput,
    TireVehicleWheelSpinGpuOutput, TireVehicleWheelSpinResidentBatch,
    TireVehicleWheelSpinStateGpuOutput, TireVehicleYawGpuInput, TireVehicleYawGpuKernel,
    TireVehicleYawGpuOutput, dahl_oracle_step, decode_epa_out, decode_mpr_out, pack_groups_host,
    radix_sort_pairs, radix_sort_pairs_u64, tire_bekker_reference_writeback,
    tire_dem_build_grid_step_input, tire_dem_build_sparse_lbvh,
    tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix,
    tire_dem_build_sparse_lbvh_morton_order_gpu_radix, tire_dem_build_sparse_lbvh_step_input,
    tire_dem_build_sparse_lbvh_tree_rows_gpu_radix,
    tire_dem_build_unbounded_sorted_grid_step_input,
    tire_dem_build_unbounded_sorted_grid_step_input_gpu_radix, tire_dem_raster_surface_height_map,
    tire_dem_raster_surface_velocity_map, tire_dem_reference_grid_step,
    tire_dem_reference_impulses, tire_dem_reference_sparse_lbvh_step, tire_dem_reference_step,
    tire_dem_sparse_lbvh_dispatch_plan, tire_dem_sparse_lbvh_gpu_rows,
    tire_fem_contact_terrain_with_dem_surface_velocity,
    tire_fem_contact_terrain_with_transient_surface_velocity,
    tire_fem_feedback_params_with_transient, tire_fem_gpu_batch_dispatch_plan,
    tire_fem_hub_params_with_brush_outputs, tire_fem_hub_params_with_contact_reduction,
    tire_fem_hub_params_with_feedback_output, tire_fem_transient_feedback_command,
    tire_mpm_reference_feedback, tire_thermal_build_rig_replay_batch,
    tire_thermal_input_from_rig_snapshot, tire_thermal_reference_rig_replay_batch,
    tire_thermal_reference_step, tire_thermal_state_from_rig_snapshot,
    tire_transient_reference_step, tire_vehicle_brush_rows_from_spin_reference,
    tire_vehicle_brush_rows_from_spin_suspension_reference, tire_vehicle_brush_rows_reference,
    tire_vehicle_planar_from_brush_reference,
    tire_vehicle_planar_from_contact_reduction_rows_reference,
    tire_vehicle_planar_from_feedback_output_reference, tire_vehicle_planar_reference,
    tire_vehicle_suspension_reference, tire_vehicle_vertical_from_suspension_reference,
    tire_vehicle_wheel_kinematics_from_spin_reference, tire_vehicle_wheel_kinematics_reference,
    tire_vehicle_wheel_spin_reference, tire_vehicle_wheel_spin_state_reference,
    tire_vehicle_yaw_reference, unpack_output_host,
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
