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
mod tire_bekker;
#[cfg(feature = "gpu")]
mod tire_brush;
#[cfg(feature = "gpu")]
mod tire_dem;
#[cfg(feature = "gpu")]
mod tire_fem;
#[cfg(feature = "gpu")]
mod tire_mpm;
#[cfg(feature = "gpu")]
mod tire_thermal;
#[cfg(feature = "gpu")]
mod tire_transient;
#[cfg(feature = "gpu")]
mod tire_vehicle;
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
pub use tire_bekker::{
    TireBekkerContactCell, TireBekkerFemTerrainOwner, TireBekkerGpuInput, TireBekkerGpuKernel,
    TireBekkerGpuOutput, TireBekkerGpuResidentTerrain, TireBekkerResidentOutput,
    tire_bekker_reference_writeback,
};
#[cfg(feature = "gpu")]
pub use tire_brush::{
    BrushTerrain, BrushWheelInput, BrushWheelOutput, TireBrushGpuBatch, TireBrushGpuKernel,
    TireBrushResidentOutput,
};
#[cfg(feature = "gpu")]
pub use tire_dem::{
    TireDemContactCell, TireDemFemTerrainOwner, TireDemFemTerrainStepReport,
    TireDemGpuBuiltGridStepInput, TireDemGpuInput, TireDemGpuKernel, TireDemGpuLbvhLeafRows,
    TireDemGpuLbvhTreeRows, TireDemGpuMortonOrder, TireDemGpuOutput, TireDemGpuRadixGridStepInput,
    TireDemGridStepGpuInput, TireDemLbvhDispatchPlan, TireDemLbvhNode, TireDemLbvhStepGpuInput,
    TireDemResidentFemTerrainOwner, TireDemResidentLbvh, TireDemResidentOutput,
    TireDemResidentState, TireDemResidentTerrainOutput, TireDemResidentTerrainParams,
    TireDemResidentTerrainReadback, TireDemSparseLbvh, TireDemStepGpuInput, TireDemStepParams,
    TireDemSurfaceVelocityMap, TireDemUnboundedSortedGrid, tire_dem_build_grid_step_input,
    tire_dem_build_sparse_lbvh, tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix,
    tire_dem_build_sparse_lbvh_morton_order_gpu_radix, tire_dem_build_sparse_lbvh_step_input,
    tire_dem_build_sparse_lbvh_tree_rows_gpu_radix,
    tire_dem_build_unbounded_sorted_grid_step_input,
    tire_dem_build_unbounded_sorted_grid_step_input_gpu_radix, tire_dem_raster_surface_height_map,
    tire_dem_raster_surface_velocity_map, tire_dem_reference_grid_step,
    tire_dem_reference_impulses, tire_dem_reference_sparse_lbvh_step, tire_dem_reference_step,
    tire_dem_sparse_lbvh_dispatch_plan, tire_dem_sparse_lbvh_gpu_rows,
};
#[cfg(feature = "gpu")]
pub use tire_fem::{
    TIRE_FEM_ATLAS_SHADER, TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET,
    TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS, TIRE_FEM_MATRIX_FREE_NEWTON_SHADER,
    TIRE_FEM_TORQUE_BRAKE_SOFT_SHOULDER_RAD_S, TIRE_FEM_TORQUE_HUB_MASS_MIN_KG,
    TireFemContactFeedbackGpuTerrain, TireFemContactFeedbackInput, TireFemContactFeedbackKernel,
    TireFemContactFeedbackOutput, TireFemContactFeedbackParams,
    TireFemContactFeedbackPatchAccounting, TireFemContactFeedbackResidentOutput,
    TireFemContactReductionGpuTerrain, TireFemContactReductionInput, TireFemContactReductionKernel,
    TireFemContactReductionOutput, TireFemContactReductionParams, TireFemContactSdfPrimitive,
    TireFemContactTerrain, TireFemContactTriangle, TireFemGpuAtlasBatch,
    TireFemGpuAtlasHubParamRow, TireFemGpuBatch, TireFemGpuBatchAtlasBindingPlan,
    TireFemGpuBatchAtlasBuffers, TireFemGpuBatchAtlasDispatchContract, TireFemGpuBatchAtlasLayout,
    TireFemGpuBatchAtlasOffsetRow, TireFemGpuBatchAtlasOutputRows, TireFemGpuBatchAtlasParamRow,
    TireFemGpuBatchAtlasRow, TireFemGpuBatchDispatchPlan, TireFemGpuInput, TireFemGpuKernel,
    TireFemGpuOutput, TireFemGpuParams, TireFemHubParams, TireFemHubSpinParams,
    TireFemMatrixFreeNewtonGpuContract, TireFemMatrixFreeNewtonGpuInput,
    TireFemMatrixFreeNewtonGpuKernel, TireFemMatrixFreeNewtonGpuOutcome,
    TireFemMatrixFreeNewtonGpuOutput, TireFemMatrixFreeNewtonGpuStatus, TireFemPersistentMesh,
    TireFemResidentContact, TireFemTorqueGpuContract, TireFemTorqueGpuStatus,
    TireFemTransientFeedbackCommand, TireResidentVehicleFemAtlasOwner,
    tire_fem_contact_terrain_with_dem_surface_velocity,
    tire_fem_contact_terrain_with_transient_surface_velocity, tire_fem_contact_triangle_bvh_rows,
    tire_fem_feedback_params_with_transient, tire_fem_gpu_batch_atlas_binding_plan,
    tire_fem_gpu_batch_atlas_buffers, tire_fem_gpu_batch_atlas_dispatch_contract,
    tire_fem_gpu_batch_atlas_layout, tire_fem_gpu_batch_atlas_offset_row,
    tire_fem_gpu_batch_atlas_param_row, tire_fem_gpu_batch_dispatch_plan,
    tire_fem_gpu_input_from_tetra_fem, tire_fem_hub_params_with_brush_outputs,
    tire_fem_hub_params_with_contact_reduction, tire_fem_hub_params_with_feedback_output,
    tire_fem_matrix_free_newton_gpu_contract, tire_fem_torque_gpu_contract,
    tire_fem_transient_feedback_command,
};
#[cfg(feature = "gpu")]
pub use tire_mpm::{
    TireMpmContactCell, TireMpmFemTerrainOwner, TireMpmGpuInPlaceParams, TireMpmGpuInput,
    TireMpmGpuKernel, TireMpmGpuOutput, TireMpmTerrainRasterGpuOutput, TireMpmTerrainRasterParams,
    tire_mpm_reference_feedback,
};
#[cfg(feature = "gpu")]
pub use tire_thermal::{
    TireThermalGpuBatch, TireThermalGpuInput, TireThermalGpuKernel, TireThermalGpuOutput,
    TireThermalGpuState, TireThermalRigReplayBatch, tire_thermal_build_rig_replay_batch,
    tire_thermal_input_from_rig_snapshot, tire_thermal_reference_rig_replay_batch,
    tire_thermal_reference_step, tire_thermal_state_from_rig_snapshot,
};
#[cfg(feature = "gpu")]
pub use tire_transient::{
    TireTransientGpuBatch, TireTransientGpuInput, TireTransientGpuKernel, TireTransientGpuOutput,
    TireTransientGpuState, tire_transient_reference_step,
};
#[cfg(feature = "gpu")]
pub use tire_vehicle::{
    TireVehicleBrushRowsFromSpinGpuInput, TireVehicleBrushRowsGpuInput,
    TireVehicleBrushRowsGpuOutput, TireVehicleBrushRowsResidentOutput,
    TireVehiclePlanarFromBrushGpuInput, TireVehiclePlanarGpuInput, TireVehiclePlanarGpuOutput,
    TireVehiclePlanarResidentState, TireVehicleResidentBrushOwner,
    TireVehicleResidentChassisBrushOwner, TireVehicleSuspensionGpuInput,
    TireVehicleSuspensionGpuOutput, TireVehicleSuspensionResidentBatch,
    TireVehicleVerticalGpuInput, TireVehicleVerticalGpuOutput, TireVehicleVerticalResidentState,
    TireVehicleWheelKinematicsFromSpinGpuInput, TireVehicleWheelKinematicsGpuInput,
    TireVehicleWheelKinematicsGpuOutput, TireVehicleWheelSpinGpuInput,
    TireVehicleWheelSpinGpuOutput, TireVehicleWheelSpinResidentBatch,
    TireVehicleWheelSpinStateGpuOutput, TireVehicleYawGpuInput, TireVehicleYawGpuKernel,
    TireVehicleYawGpuOutput, tire_vehicle_brush_rows_from_spin_reference,
    tire_vehicle_brush_rows_from_spin_suspension_reference, tire_vehicle_brush_rows_reference,
    tire_vehicle_planar_from_brush_reference,
    tire_vehicle_planar_from_contact_reduction_rows_reference,
    tire_vehicle_planar_from_feedback_output_reference, tire_vehicle_planar_reference,
    tire_vehicle_suspension_reference, tire_vehicle_vertical_from_suspension_reference,
    tire_vehicle_wheel_kinematics_from_spin_reference, tire_vehicle_wheel_kinematics_reference,
    tire_vehicle_wheel_spin_reference, tire_vehicle_wheel_spin_state_reference,
    tire_vehicle_yaw_reference,
};
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
