//! molla-vehicle — wheeled-vehicle simulation for molla.
//!
//! Built per `PLAN_ENGINE.md` Track V. The vehicle is assembled from
//! parts molla already provides (rigid bodies, joints) plus the
//! vehicle-specific physics in this crate. Modules are added stage by
//! stage:
//!
//! - **V0** — engine prerequisites: a [`Terrain`] query abstraction and
//!   a 1-DOF rotational [`Shaft`] (the driveline building block).
//! - **V1/V2** — `tire` force-element models.
//! - **V3** — `suspension` templates.
//! - **V4** — `powertrain` (engine map, transmission, driveline).

pub mod deformable;
pub mod granular;
pub mod powertrain;
pub mod real_tire;
pub mod shaft;
pub mod suspension;
pub mod sysid;
pub mod terrain;
pub mod tire;
pub mod track;
pub mod vehicle;

pub use deformable::{
    DeformableTerrain, TireSoilFeedback, TireSoilRutRigPoint, tire_soil_rut_matrix,
};
pub use granular::{GrainObstacle, GranularBed, TireDemFeedback};
pub use powertrain::{Driveline, DriverInput, Engine, Powertrain, Transmission, ackermann_steer};
pub use real_tire::{
    DynamicTireCarcass, DynamicTireCarcassParams, RealBrushTireModel, RealTireContact,
    RealTireParams, RealTireTelemetrySample, RealTireTransientState, RealWheelOrientation,
    RealWheelState, SurfaceFilmState, SurfaceMaterial, TetraBeadSeatDiagnostics,
    TetraBeadSeatLoadCaseReport, TetraCavityBeadSanityReport, TetraCavityCouplingEffectReport,
    TetraConstructionSanityReport, TetraFemContactSlipPatchReport, TetraFemDeformationForceReport,
    TetraFemDiagonalNewtonReport, TetraFemElementNewtonReport, TetraFemHardeningEnvelopeReport,
    TetraFemMatrixFreeNewtonPreconditionerReport, TetraFemMatrixFreeNewtonScaleReport,
    TetraFemMatrixFreeNewtonSolveReport, TetraFemMatrixFreeNewtonStepReport,
    TetraFemPatchNewtonReport, TetraFemSolverStiffnessReport, TetraFemSubstepConvergenceReport,
    TetraFemTinyMeshNewtonReport, TetraSidewallDihedralDiagnostics,
    TetraSidewallDihedralXpbdEffectReport, TetraSidewallDynamicLoadReport,
    TetraSidewallLaminateAudit, TetraSidewallLaminateCouplingReport,
    TetraSidewallLaminateSanityReport, TetraSidewallLaminateStretchDiagnostics,
    TetraSidewallPlyStrainXpbdReport, TetraTireFem, TetraTireFemGpuSnapshot, TetraTireFemParams,
    TetraTireFemResolutionPoint, TetraTireFemSolverMode, TireAdhesionSlidingContinuityReport,
    TireApproximationStatus, TireConditionFailureModeReport, TireConditionMaterialResponse,
    TireContactCell, TireContactCellTelemetry, TireContactPatchConvergenceReport,
    TireContactPatchTelemetry, TireDemSoilFeedbackReport, TireFrameConventionReport,
    TireGpuParityItem, TireGpuParityStatus, TireHeightfieldResolutionConvergenceReport,
    TireHeightfieldSurfaceVelocityMapReport, TireIceLargePatchReport, TireKnownApproximation,
    TireLayerAblationDebugReport, TireLayeredTransientEnvelopeReport, TireLayeredTransientState,
    TireLoadCombinedSlipLawReport, TireLowSpeedRobustnessReport, TireMpmSoilFeedbackReport,
    TireParameterObservabilityReport, TirePresetGovernanceCheck, TirePresetGovernanceReport,
    TirePressureDistributionBin, TirePressureDistributionTelemetry,
    TireRealismSourceClassificationReport, TireRealismSourceItem, TireReferenceShapeDiagnostics,
    TireRelaxationContext, TireReplayArtifactCoverageReport, TireRigAcceptanceCheck,
    TireRigEnvelopePoint, TireRigRollingLossPoint, TireRigRunConfig, TireRigRunOutput,
    TireRigSensitivityAxis, TireRigSensitivityPoint, TireRigSnapshot, TireRigStepResponseConfig,
    TireRigStepResponsePoint, TireRigSurfaceKind, TireRigSurfaceSlipPoint,
    TireRigSurfaceTransitionBand, TireRigSurfaceTransitionPoint, TireRigSurfaceTransitionTerrain,
    TireRigSweepPoint, TireRigThermalWearPhase, TireRigThermalWearPoint,
    TireRoadConditionStateReport, TireRuntimeProductionGateReport,
    TireSoilNormalShearCouplingReport, TireSolverBudgetTelemetry, TireThermalAbuseEnvelopeReport,
    TireThermalBrakeRimReport, TireThermalConditionSanityReport, TireThermalLayers,
    TireThermalRollingCarcassReport, TireThermalState, TireThermalWearConvergenceReport,
    TireTorqueSlipOutcomeReport, TireTransientLayerState, TireTriangleDensityConvergenceReport,
    TireTriangleLargeManifoldReport, TireTriangleSeamManifoldReport,
    TireTriangleSurfaceForceReport, TireTriangleSurfaceStressReport,
    TireVehicleDynamicLoadTransferReport, TireVehicleSanityCase, TireWearState,
    TireWheelSpinTorqueBalanceReport, evaluate_brush_tire, evaluate_brush_tire_layered_transient,
    evaluate_brush_tire_layered_transient_with_context, evaluate_brush_tire_oriented,
    evaluate_brush_tire_transient, evaluate_brush_tire_transient_with_context,
    solve_loaded_tire_hub_height, tetra_bead_seat_load_case_report,
    tetra_cavity_bead_sanity_report, tetra_cavity_coupling_effect_report,
    tetra_construction_sanity_report, tetra_fem_contact_slip_patch_report,
    tetra_fem_deformation_force_report, tetra_fem_diagonal_newton_report,
    tetra_fem_element_newton_report, tetra_fem_hardening_envelope_report,
    tetra_fem_matrix_free_newton_preconditioner_report, tetra_fem_matrix_free_newton_scale_report,
    tetra_fem_matrix_free_newton_solve_report, tetra_fem_matrix_free_newton_step_report,
    tetra_fem_matrix_free_newton_timestep_report, tetra_fem_patch_newton_report,
    tetra_fem_solver_stiffness_report, tetra_fem_substep_convergence_report,
    tetra_fem_tiny_mesh_newton_report, tetra_sidewall_dihedral_xpbd_effect_report,
    tetra_sidewall_dynamic_load_report, tetra_sidewall_laminate_coupling_report,
    tetra_sidewall_ply_strain_xpbd_report, tire_adhesion_sliding_continuity_report,
    tire_condition_failure_mode_report, tire_contact_patch_convergence_report,
    tire_dem_soil_feedback_report, tire_fem_resolution_sweep, tire_frame_convention_report,
    tire_gpu_parity_checklist, tire_gpu_parity_text_report,
    tire_heightfield_resolution_convergence_report, tire_heightfield_surface_velocity_map_report,
    tire_ice_large_patch_report, tire_known_approximation_ledger,
    tire_known_approximation_text_report, tire_layer_ablation_debug_report,
    tire_layered_transient_envelope_report, tire_load_combined_slip_law_report,
    tire_low_speed_robustness_report, tire_mpm_soil_feedback_report,
    tire_parameter_observability_report, tire_preset_governance_report,
    tire_realism_source_classification_report, tire_rig_camber_sweep,
    tire_rig_combined_reference_shape_diagnostics, tire_rig_combined_slip_sweep,
    tire_rig_default_non_calibration_run, tire_rig_heightfield_envelope_sweep,
    tire_rig_load_deflection_sweep, tire_rig_longitudinal_reference_shape_diagnostics,
    tire_rig_longitudinal_slip_sweep, tire_rig_rolling_loss_sweep,
    tire_rig_standard_non_calibration_run, tire_rig_standard_sensitivity_map,
    tire_rig_standard_sensitivity_text_report, tire_rig_step_brake_response,
    tire_rig_step_steer_response, tire_rig_step_torque_response, tire_rig_surface_slip_matrix,
    tire_rig_surface_transition_sequence, tire_rig_thermal_wear_cycle, tire_rig_thermal_wear_run,
    tire_road_condition_state_report, tire_runtime_production_gate_report,
    tire_soil_normal_shear_coupling_report, tire_thermal_abuse_envelope_report,
    tire_thermal_brake_rim_report, tire_thermal_condition_sanity_report,
    tire_thermal_rolling_carcass_report, tire_thermal_wear_convergence_report,
    tire_torque_induced_slip_outcome_report, tire_triangle_density_convergence_report,
    tire_triangle_large_manifold_report, tire_triangle_seam_manifold_report,
    tire_triangle_surface_force_report, tire_triangle_surface_stress_report,
    tire_vehicle_dynamic_load_transfer_report, tire_vehicle_sanity_scenarios,
    tire_wheel_spin_torque_balance_report,
};
pub use shaft::Shaft;
pub use suspension::{Suspension, SuspensionKind};
pub use sysid::{drive_distance, identify_friction};
pub use terrain::{
    FlatTerrain, HeightfieldTerrain, SdfTerrain, SdfTerrainPrimitive, Terrain,
    TriangleMeshContactPatch, TriangleMeshTerrain, TriangleMeshTriangleBounds,
};
pub use tire::{
    Pac89Tire, RigidTire, TireModel, TireOutput, TmSimpleTire, WheelContact, magic_formula,
    magic_formula_deriv, wheel_ground_wrench,
};
pub use vehicle::{Corner, Vehicle};
