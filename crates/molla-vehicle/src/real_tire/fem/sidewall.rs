//! Tetra-tire sidewall + tread-belt diagnostic types.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::TireRigAcceptanceCheck;
use super::super::internal::bool_metric;

/// Reusable sidewall fold-angle diagnostics for the tetra tire.
///
/// This is a validation/inspection tool for the remaining laminated sidewall
/// work and for the first CPU dihedral-angle XPBD projection. The solver still
/// keeps the older structural edge-distance proxy for CPU/GPU parity, but
/// these diagnostics measure actual sidewall quad fold-angle drift so tests
/// and viewer/rig reports can see whether folding is bounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallDihedralDiagnostics {
    pub quad_count: usize,
    pub mean_abs_deviation_rad: f64,
    pub max_abs_deviation_rad: f64,
}

impl TetraSidewallDihedralDiagnostics {
    pub fn is_finite(self) -> bool {
        self.mean_abs_deviation_rad.is_finite() && self.max_abs_deviation_rad.is_finite()
    }
}

/// Audit for the compact explicit sidewall laminate/fibre stack.
///
/// This does not make the sidewall a full shell laminate, but it verifies that
/// sidewall tetrahedra actually carry crossed bias plies plus radial/axial
/// helper families. It gives the remaining laminated-sidewall work a concrete
/// regression signal instead of only a visual "sidewall looks okay" check.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallLaminateAudit {
    pub sidewall_element_count: usize,
    pub crossed_pair_count: usize,
    pub mean_bias_angle_deg: f64,
    pub max_bias_angle_error_deg: f64,
    pub mean_radial_helper_alignment: f64,
    pub mean_axial_helper_alignment: f64,
}

impl TetraSidewallLaminateAudit {
    pub fn is_finite(self) -> bool {
        self.mean_bias_angle_deg.is_finite()
            && self.max_bias_angle_error_deg.is_finite()
            && self.mean_radial_helper_alignment.is_finite()
            && self.mean_axial_helper_alignment.is_finite()
    }
}

/// Runtime stretch diagnostics for the compact sidewall ply families.
///
/// The current model is still a volumetric tetra approximation, not a full
/// laminated shell. These values make that approximation harder to overclaim:
/// they report whether the explicit sidewall ply directions are seeing finite
/// tension/compression under the current deformation, using the same `F·a₀`
/// stretch measure that drives the Holzapfel fibre stress.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallLaminateStretchDiagnostics {
    pub sidewall_element_count: usize,
    pub tensile_family_count: usize,
    pub mean_tensile_strain: f64,
    pub max_tensile_strain: f64,
    pub max_compressive_strain: f64,
}

impl TetraSidewallLaminateStretchDiagnostics {
    pub fn is_finite(self) -> bool {
        self.mean_tensile_strain.is_finite()
            && self.max_tensile_strain.is_finite()
            && self.max_compressive_strain.is_finite()
    }
}

/// Runtime stretch diagnostics for the tread-belt XPBD edge network.
///
/// This is a compact belt-inextensibility signal for the current volumetric
/// approximation. It does not model individual tread blocks yet, but it tells
/// rigs/viewers whether the circumferential/axial/crossed belt edges are
/// stretching far beyond their rest lengths before drive/brake forces are
/// trusted.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraTreadBeltStretchDiagnostics {
    pub edge_count: usize,
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
}

impl TetraTreadBeltStretchDiagnostics {
    pub fn is_finite(self) -> bool {
        self.mean_relative_error.is_finite() && self.max_relative_error.is_finite()
    }
}

/// Combined sanity gate for the current sidewall laminate approximation.
///
/// This still does not claim a full laminated shell solve. It packages the
/// explicit ply-stack audit with live sidewall fold diagnostics so rigs/viewers
/// can reject a tire that has the right fibre directions but is currently
/// folded far outside the validated real-time envelope.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallLaminateSanityReport {
    pub audit: TetraSidewallLaminateAudit,
    pub dihedral: TetraSidewallDihedralDiagnostics,
    pub stretch: TetraSidewallLaminateStretchDiagnostics,
    pub crossed_coverage: f64,
    pub helper_alignment_min: f64,
    pub fold_mean_limit_rad: f64,
    pub fold_max_limit_rad: f64,
    pub stretch_limit: f64,
    pub passed: bool,
}

impl TetraSidewallLaminateSanityReport {
    pub fn is_finite(self) -> bool {
        self.audit.is_finite()
            && self.dihedral.is_finite()
            && self.stretch.is_finite()
            && self.crossed_coverage.is_finite()
            && self.helper_alignment_min.is_finite()
            && self.fold_mean_limit_rad.is_finite()
            && self.fold_max_limit_rad.is_finite()
            && self.stretch_limit.is_finite()
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_crossed_ply_coverage",
                "sidewall laminate should carry crossed bias plies around the tire",
                "crossed_coverage",
                self.crossed_coverage,
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_bias_angle_error",
                "sidewall bias ply angle should stay near the declared preset",
                "max_bias_angle_error_deg",
                self.audit.max_bias_angle_error_deg,
                6.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_laminate_helper_alignment",
                "sidewall helper plies should stay radial and axial",
                "helper_alignment_min",
                self.helper_alignment_min,
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_mean_fold_drift",
                "sidewall fold drift should stay inside the realtime envelope",
                "mean_abs_deviation_rad",
                self.dihedral.mean_abs_deviation_rad,
                self.fold_mean_limit_rad,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_max_fold_drift",
                "sidewall peak fold drift should stay inside the realtime envelope",
                "max_abs_deviation_rad",
                self.dihedral.max_abs_deviation_rad,
                self.fold_max_limit_rad,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_max_tensile_strain",
                "sidewall ply stretch should stay finite and bounded until a full laminated shell solve exists",
                "max_tensile_strain",
                self.stretch.max_tensile_strain,
                self.stretch_limit,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_laminate_max_compressive_strain",
                "sidewall ply compression should stay finite and bounded even though cords only carry tensile stress",
                "max_compressive_strain",
                self.stretch.max_compressive_strain,
                self.stretch_limit,
            ),
        ]
    }
}

/// Dynamic-load sanity gate for the current sidewall laminate approximation.
///
/// This still does not claim a full laminated shell sidewall solve. It makes
/// the current explicit ply stack and dihedral XPBD guardrail harder to
/// overclaim by checking the sidewall after deterministic vertical and lateral
/// tire load cases, plus a steered slip-angle case, rather than only in the
/// rest shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraSidewallDynamicLoadReport {
    pub rest_report: TetraSidewallLaminateSanityReport,
    pub vertical_report: TetraSidewallLaminateSanityReport,
    pub lateral_report: TetraSidewallLaminateSanityReport,
    pub steer_report: TetraSidewallLaminateSanityReport,
    pub lateral_contact_force_n: f64,
    pub steer_contact_force_n: f64,
    pub steer_aligning_moment_nm: f64,
    pub vertical_max_fold_drift_rad: f64,
    pub lateral_max_fold_drift_rad: f64,
    pub steer_max_fold_drift_rad: f64,
    pub vertical_max_ply_strain: f64,
    pub lateral_max_ply_strain: f64,
    pub steer_max_ply_strain: f64,
    pub finite_outputs: bool,
}

impl TetraSidewallDynamicLoadReport {
    fn load_case_layout_ok(report: TetraSidewallLaminateSanityReport) -> bool {
        report.audit.is_finite()
            && report.dihedral.is_finite()
            && report.stretch.is_finite()
            && report.crossed_coverage >= 0.98
            && report.helper_alignment_min >= 0.98
            && report.audit.max_bias_angle_error_deg <= 8.0
            && report.dihedral.max_abs_deviation_rad <= report.fold_max_limit_rad
            && report.stretch.max_tensile_strain <= report.stretch_limit
            && report.stretch.max_compressive_strain <= report.stretch_limit
    }

    pub fn is_finite(self) -> bool {
        self.rest_report.is_finite()
            && self.vertical_report.is_finite()
            && self.lateral_report.is_finite()
            && self.steer_report.is_finite()
            && self.lateral_contact_force_n.is_finite()
            && self.steer_contact_force_n.is_finite()
            && self.steer_aligning_moment_nm.is_finite()
            && self.vertical_max_fold_drift_rad.is_finite()
            && self.lateral_max_fold_drift_rad.is_finite()
            && self.steer_max_fold_drift_rad.is_finite()
            && self.vertical_max_ply_strain.is_finite()
            && self.lateral_max_ply_strain.is_finite()
            && self.steer_max_ply_strain.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.rest_report.passed
            && Self::load_case_layout_ok(self.vertical_report)
            && Self::load_case_layout_ok(self.lateral_report)
            && Self::load_case_layout_ok(self.steer_report)
            && self.lateral_contact_force_n >= 10.0
            && self.steer_contact_force_n >= 10.0
            && self.steer_aligning_moment_nm.abs() >= 1.0e-3
            && self.vertical_max_fold_drift_rad <= self.vertical_report.fold_max_limit_rad
            && self.lateral_max_fold_drift_rad <= self.lateral_report.fold_max_limit_rad
            && self.steer_max_fold_drift_rad <= self.steer_report.fold_max_limit_rad
            && self.vertical_max_ply_strain <= self.vertical_report.stretch_limit
            && self.lateral_max_ply_strain <= self.lateral_report.stretch_limit
            && self.steer_max_ply_strain <= self.steer_report.stretch_limit
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_outputs_finite",
                "sidewall dynamic-load diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_rest_sanity_passes",
                "rest sidewall laminate sanity should pass before load cases",
                "rest_passed",
                bool_metric(self.rest_report.passed),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_vertical_sanity_passes",
                "vertical load should not fold or over-stretch the sidewall laminate approximation",
                "vertical_passed",
                bool_metric(Self::load_case_layout_ok(self.vertical_report)),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_lateral_sanity_passes",
                "lateral tire load should not fold or over-stretch the sidewall laminate approximation",
                "lateral_passed",
                bool_metric(Self::load_case_layout_ok(self.lateral_report)),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_steer_sanity_passes",
                "steered slip-angle load should not fold or over-stretch the sidewall laminate approximation",
                "steer_passed",
                bool_metric(Self::load_case_layout_ok(self.steer_report)),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_lateral_force_applied",
                "lateral sidewall load case should actually apply lateral tire force",
                "abs_lateral_contact_force_n",
                self.lateral_contact_force_n,
                10.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_steer_force_applied",
                "steered sidewall load case should produce lateral tire force from wheel-frame angle",
                "abs_steer_contact_force_n",
                self.steer_contact_force_n,
                10.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "sidewall_dynamic_steer_aligning_moment",
                "steered sidewall load case should produce a finite aligning moment, not a yaw-hack",
                "abs_steer_aligning_moment_nm",
                self.steer_aligning_moment_nm.abs(),
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_vertical_fold_bounded",
                "vertical load sidewall fold drift should stay inside the realtime envelope",
                "vertical_max_fold_drift_rad",
                self.vertical_max_fold_drift_rad,
                self.vertical_report.fold_max_limit_rad,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_lateral_fold_bounded",
                "lateral load sidewall fold drift should stay inside the realtime envelope",
                "lateral_max_fold_drift_rad",
                self.lateral_max_fold_drift_rad,
                self.lateral_report.fold_max_limit_rad,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_steer_fold_bounded",
                "steered slip-angle sidewall fold drift should stay inside the realtime envelope",
                "steer_max_fold_drift_rad",
                self.steer_max_fold_drift_rad,
                self.steer_report.fold_max_limit_rad,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_vertical_ply_strain_bounded",
                "vertical load sidewall ply strain should stay inside the realtime envelope",
                "vertical_max_ply_strain",
                self.vertical_max_ply_strain,
                self.vertical_report.stretch_limit,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_lateral_ply_strain_bounded",
                "lateral load sidewall ply strain should stay inside the realtime envelope",
                "lateral_max_ply_strain",
                self.lateral_max_ply_strain,
                self.lateral_report.stretch_limit,
            ),
            TireRigAcceptanceCheck::at_most(
                "sidewall_dynamic_steer_ply_strain_bounded",
                "steered slip-angle sidewall ply strain should stay inside the realtime envelope",
                "steer_max_ply_strain",
                self.steer_max_ply_strain,
                self.steer_report.stretch_limit,
            ),
        ]
    }
}
