//! Real tire contact primitives.
//!
//! This module is the first implementation step from `PLAN_REALTIRE.md`.
//! It models the tire/ground interface as a pressure-dependent contact patch
//! plus a longitudinal brush/bristle law. It is still compact enough for the
//! viewer, but unlike the old viewer patch multiplier, forces come from:
//!
//! - tire pressure and carcass compliance determining patch length;
//! - per-contact-cell normal load;
//! - tread shear stiffness;
//! - combined longitudinal/lateral brush forces;
//! - material static/dynamic friction caps.

#[cfg(test)]
use crate::tire::TireModel;

mod internal;
use internal::{finite_nonnegative, finite_or};
mod surface_material;
pub use surface_material::SurfaceMaterial;
mod surface_film;
pub use surface_film::SurfaceFilmState;
mod carcass;
pub use carcass::{DynamicTireCarcass, DynamicTireCarcassParams};
mod governance;
pub use governance::{
    TirePresetGovernanceCheck, TirePresetGovernanceReport, tire_preset_governance_report,
};
mod parity_ledger;
pub use parity_ledger::{
    TireApproximationStatus, TireGpuParityItem, TireGpuParityStatus, TireKnownApproximation,
    tire_gpu_parity_checklist, tire_gpu_parity_text_report, tire_known_approximation_ledger,
    tire_known_approximation_text_report,
};
mod realism;
pub use realism::{
    TireModelProvenance, TireRealismSourceClassificationReport, TireRealismSourceItem,
    tire_realism_source_classification_report,
};
mod params;
pub use params::RealTireParams;
mod envelope;
pub use envelope::LoadedTireEnvelope;
mod wheel;
pub use wheel::{RealWheelOrientation, RealWheelState};

mod bristle;
mod brush;
pub use bristle::{TireBrushState, TreadBristle, TreadBristleOutput, TreadBristleParams};
pub use brush::RealBrushTireModel;
pub(crate) use brush::{
    brush_bristle_profile, brush_load_sensitivity, brush_pressure_patch_gain,
    brush_pressure_stiffness_factor, sanitize_brush_params, smooth_combined_brush_force,
};

mod thermal_wear;
pub use thermal_wear::{
    TireRigThermalWearPhase, TireRigThermalWearPoint, TireThermalConditionSanityReport,
    TireThermalLayers, TireThermalState, TireWearState, tire_rig_thermal_wear_cycle,
    tire_rig_thermal_wear_run, tire_thermal_condition_sanity_report,
};

mod contact;
pub use contact::{RealTireContact, TireContactCell};

mod viewer_target;
pub use viewer_target::{TireViewerTargetCoverageCheck, TireViewerTargetCoverageReport};

mod fem;
pub(crate) use fem::builders::tet_is_sidewall_laminate;
pub use fem::builders::{
    TetraConstructionSanityReport, TetraFemContactSlipPatchReport, TetraFemDeformationForceReport,
    TetraFemHardeningEnvelopeReport, tetra_bead_seat_load_case_report,
    tetra_cavity_bead_sanity_report, tetra_cavity_coupling_effect_report,
    tetra_construction_sanity_report, tetra_fem_contact_slip_patch_report,
    tetra_fem_deformation_force_report, tetra_sidewall_dihedral_xpbd_effect_report,
    tetra_sidewall_dynamic_load_report, tetra_sidewall_laminate_coupling_report,
    tetra_sidewall_ply_strain_xpbd_report, tire_fem_resolution_sweep,
};
pub use fem::diagnostics::{
    TetraBeadSeatDiagnostics, TetraBeadSeatLoadCaseReport, TetraCavityBeadSanityReport,
    TetraCavityCouplingEffectReport, TetraTireFemDiagnostics, TetraTireFemResolutionPoint,
};
pub(crate) use fem::math::{
    add_holzapfel_fiber_stress, det3, inv3, lame_from_young_poisson, make_tetra_element, mat_vec,
    project_edge_length_constraints, project_sidewall_dihedral_constraints,
    project_sidewall_ply_strain_constraints, project_to_ground, quad_dihedral_angle,
    tetra_signed_volume, transpose3,
};
pub(crate) use fem::newton_builders::{
    matrix_free_newton_correction, patch_flatten_nodes, patch_newton_residual_with_masses,
    patch_total_energy_with_masses, patch_unflatten_nodes, vec_max_abs, vec_norm,
};
pub use fem::newton_builders::{
    tetra_fem_diagonal_newton_report, tetra_fem_element_newton_report,
    tetra_fem_hardening_envelope_report, tetra_fem_matrix_free_newton_preconditioner_report,
    tetra_fem_matrix_free_newton_scale_report, tetra_fem_matrix_free_newton_solve_report,
    tetra_fem_matrix_free_newton_step_report, tetra_fem_matrix_free_newton_timestep_report,
    tetra_fem_patch_newton_report, tetra_fem_solver_stiffness_report,
    tetra_fem_tiny_mesh_newton_report,
};
pub use fem::newton_reports::{
    TetraFemDiagonalNewtonReport, TetraFemElementNewtonReport,
    TetraFemMatrixFreeNewtonPreconditionerReport, TetraFemMatrixFreeNewtonScaleReport,
    TetraFemMatrixFreeNewtonSolveReport, TetraFemMatrixFreeNewtonStepReport,
    TetraFemPatchNewtonReport, TetraFemSolverStiffnessReport, TetraFemTinyMeshNewtonReport,
    TetraSidewallDihedralXpbdEffectReport, TetraSidewallLaminateCouplingReport,
    TetraSidewallPlyStrainXpbdReport,
};
pub use fem::params::{TetraTireFemParams, TetraTireFemSolverMode};
pub use fem::sidewall::{
    TetraSidewallDihedralDiagnostics, TetraSidewallDynamicLoadReport, TetraSidewallLaminateAudit,
    TetraSidewallLaminateSanityReport, TetraSidewallLaminateStretchDiagnostics,
    TetraTreadBeltStretchDiagnostics,
};
pub(crate) use fem::state::{SidewallDihedral, StructuralEdge, TetraElement, tire_nodes_finite};
pub use fem::state::{TetraTireFem, TetraTireFemGpuSnapshot};

mod observability;
pub use observability::{
    TireParameterObservabilityReport, TireRuntimeProductionGateReport,
    tire_parameter_observability_report, tire_runtime_production_gate_report,
};

mod transient;
pub use transient::{
    RealTireTransientState, TireLayeredTransientState, TireRelaxationContext,
    TireTransientLayerState, evaluate_brush_tire_layered_transient,
    evaluate_brush_tire_layered_transient_with_context, evaluate_brush_tire_transient,
    evaluate_brush_tire_transient_with_context,
};

mod telemetry;
pub use telemetry::{
    RealTireTelemetrySample, TireContactCellTelemetry, TireContactPatchTelemetry,
    TirePressureDistributionBin, TirePressureDistributionTelemetry,
};

mod condition;
mod rig;
pub use condition::{
    TireConditionFailureModeReport, TireConditionLevel, TireConditionMaterialResponse,
    TireConditionReport, tire_condition_failure_mode_report,
};

pub(crate) use rig::runners::surface_material_with_terrain_film;
pub use rig::runners::{
    evaluate_brush_tire, evaluate_brush_tire_oriented, evaluate_planar_brush_tire_persistent,
    solve_loaded_tire_hub_height, tire_rig_camber_sweep, tire_rig_combined_slip_sweep,
    tire_rig_step_brake_response, tire_rig_step_steer_response, tire_rig_step_torque_response,
};
pub(crate) use rig::sensitivity::tire_rig_step_response_snapshot;
pub use rig::sensitivity::{
    TireRigSensitivityAxis, TireRigSensitivityPoint, TireRigSweepPoint,
    tire_rig_standard_sensitivity_map, tire_rig_standard_sensitivity_text_report,
};
pub use rig::snapshot::{
    TireEnergySignAudit, TireReplayArtifactCoverageReport, TireRigAcceptanceCheck,
    TireRigAcceptanceReport, TireRigDimensionlessSanityMetrics, TireRigRunConfig, TireRigRunOutput,
    TireRigSnapshot, TireSolverBudgetTelemetry,
};
pub use rig::standard_run::{
    tire_rig_default_non_calibration_run, tire_rig_standard_non_calibration_run,
};
pub(crate) use rig::standard_run::{tire_rig_point, tire_rig_point_at};
pub(crate) use rig::sweeps::tire_rig_step_response;
pub use rig::sweeps::{
    TireRigEnvelopePoint, TireRigRollingLossPoint, TireRigRollingObstaclePoint,
    TireRigStepResponseConfig, TireRigStepResponsePoint, TireRigSurfaceKind,
    TireRigSurfaceSlipPoint, TireRigSurfaceTransitionBand, TireRigSurfaceTransitionPoint,
    TireRigSurfaceTransitionTerrain, tire_rig_heightfield_envelope_sweep,
    tire_rig_load_deflection_sweep, tire_rig_longitudinal_slip_sweep, tire_rig_rolling_loss_sweep,
    tire_rig_rolling_obstacle_sequence, tire_rig_surface_slip_matrix,
    tire_rig_surface_transition_sequence,
};

mod reports;
pub use reports::convergence::{
    TetraFemSubstepConvergenceReport, TireHeightfieldResolutionConvergenceReport,
    TireLayerAblationDebugReport, TireThermalWearConvergenceReport,
    tetra_fem_substep_convergence_report, tire_contact_patch_convergence_report,
    tire_heightfield_resolution_convergence_report, tire_layer_ablation_debug_report,
    tire_thermal_wear_convergence_report,
};
pub use reports::physical_shape::{
    TireAdhesionSlidingContinuityReport, TireContactPatchConvergenceReport,
    TireFrameConventionReport, TireIceLargePatchReport, TireLayeredTransientEnvelopeReport,
    TireLoadCombinedSlipLawReport, TireLowSpeedRobustnessReport, TireReferenceShapeDiagnostics,
    TireRoadConditionStateReport, tire_adhesion_sliding_continuity_report,
    tire_frame_convention_report, tire_ice_large_patch_report,
    tire_layered_transient_envelope_report, tire_load_combined_slip_law_report,
    tire_low_speed_robustness_report, tire_rig_combined_reference_shape_diagnostics,
    tire_rig_longitudinal_reference_shape_diagnostics, tire_road_condition_state_report,
};
pub use reports::soil::{
    TireDemSoilFeedbackReport, TireHeightfieldSurfaceVelocityMapReport, TireMpmSoilFeedbackReport,
    TireSoilNormalShearCouplingReport, tire_dem_soil_feedback_report,
    tire_heightfield_surface_velocity_map_report, tire_mpm_soil_feedback_report,
    tire_soil_normal_shear_coupling_report,
};
pub use reports::thermal::{
    TireThermalAbuseEnvelopeReport, TireThermalBrakeRimReport, TireThermalRollingCarcassReport,
    TireWearPowerTemperatureReport, TireWornSurfaceSensitivityReport,
    tire_thermal_abuse_envelope_report, tire_thermal_brake_rim_report,
    tire_thermal_rolling_carcass_report, tire_wear_power_temperature_report,
    tire_worn_surface_sensitivity_report,
};
pub use reports::triangle::{
    TireTriangleDensityConvergenceReport, TireTriangleLargeManifoldReport,
    TireTriangleSeamManifoldReport, TireTriangleSurfaceForceReport,
    TireTriangleSurfaceStressReport, tire_triangle_density_convergence_report,
    tire_triangle_large_manifold_report, tire_triangle_seam_manifold_report,
    tire_triangle_surface_force_report, tire_triangle_surface_stress_report,
};
pub use reports::vehicle::{
    TireTorqueSlipOutcomeReport, TireVehicleDynamicLoadTransferReport,
    TireVehicleLongitudinalMotionReport, TireVehicleSanityCase, TireWheelSpinTorqueBalanceReport,
    tire_torque_induced_slip_outcome_report, tire_vehicle_dynamic_load_transfer_report,
    tire_vehicle_longitudinal_motion_report, tire_vehicle_sanity_scenarios,
    tire_wheel_spin_torque_balance_report,
};

use internal::{add, cross, dot, scale};

use internal::normalize;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod width_tests;

#[cfg(test)]
mod dispatch_tests;
