//! Compact snapshot + acceptance + dimensionless-sanity + run config for the
//! deterministic non-calibration tire rig.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::Terrain;

use super::super::internal::{
    bool_metric, csv_optional_f64, csv_optional_vec3, csv_token, dot, finite_nonnegative,
    finite_or, finite_positive_or, sanitize_vec3,
};
use super::super::{
    RealTireTelemetrySample, TetraTireFemDiagnostics, TetraTireFemParams, TireModelProvenance,
    TireRigSensitivityPoint, TireRigSweepPoint, TireThermalLayers, TireVehicleSanityCase,
    TireViewerTargetCoverageReport, TireWearState, tire_gpu_parity_checklist,
    tire_known_approximation_ledger, tire_preset_governance_report,
};

/// Compact non-calibration snapshot for deterministic tire-rig runs.
///
/// A snapshot records the command inputs, preset names, terrain state,
/// optional heat/wear state, optional CPU-vs-GPU delta, and unit-labelled
/// contact telemetry. It is meant for regression/falsification reports, not
/// for fitting the model to measured tire data.
#[derive(Clone, Debug)]
pub struct TireRigSnapshot {
    pub scenario_id: String,
    pub tire_preset: String,
    pub surface_preset: String,
    pub terrain_preset: String,
    pub physics_path: String,
    pub tire_parameter_provenance: TireModelProvenance,
    pub surface_parameter_provenance: TireModelProvenance,
    pub solver_parameter_provenance: TireModelProvenance,
    pub load_n: f64,
    pub pressure_pa: f64,
    pub hub_height_m: f64,
    pub deflection_m: f64,
    pub speed_m_per_s: f64,
    pub slip_ratio_command: f64,
    pub slip_angle_command_rad: f64,
    pub camber_angle_command_rad: f64,
    pub terrain_height_m: f64,
    pub terrain_normal: [f64; 3],
    pub terrain_surface_velocity_m_per_s: [f64; 3],
    pub moving_ground_power_w: f64,
    pub surface_temperature_c: Option<f64>,
    pub tread_temperature_c: Option<f64>,
    pub carcass_temperature_c: Option<f64>,
    pub tread_depth_m: Option<f64>,
    pub wear_ratio: Option<f64>,
    pub cpu_gpu_force_delta_n: Option<[f64; 3]>,
    pub cpu_gpu_relative_force_error: Option<f64>,
    pub solver_budget: Option<TireSolverBudgetTelemetry>,
    pub telemetry: RealTireTelemetrySample,
}

/// Solver-budget telemetry for tire rigs and viewer diagnostics.
///
/// These are numerical-health fields, not physical calibration targets. They
/// make explicit how much XPBD/solver stabilization was used and whether a
/// visually stable tire is hiding large corrections or velocities.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireSolverBudgetTelemetry {
    pub substeps_per_frame: usize,
    pub xpbd_volume_iterations: usize,
    pub xpbd_structural_iterations: usize,
    pub xpbd_volume_compliance: f64,
    pub xpbd_bead_compliance: f64,
    pub xpbd_tread_belt_compliance: f64,
    pub xpbd_sidewall_bend_compliance: f64,
    pub xpbd_sidewall_dihedral_compliance: f64,
    pub semi_implicit_stiffness_scale: f64,
    pub last_xpbd_volume_correction_m: f64,
    pub last_xpbd_bead_correction_m: f64,
    pub last_xpbd_structural_correction_m: f64,
    pub max_tet_volume_error_m3: f64,
    pub mean_bead_mount_error_m: f64,
    pub max_bead_mount_error_m: f64,
    pub max_node_speed_m_per_s: f64,
    pub dissipated_power_w: f64,
    pub dissipated_energy_j: f64,
}

impl TireSolverBudgetTelemetry {
    pub fn from_tetra_fem(
        params: &TetraTireFemParams,
        diagnostics: &TetraTireFemDiagnostics,
        max_node_speed_m_per_s: f64,
        substeps_per_frame: usize,
        telemetry: &RealTireTelemetrySample,
    ) -> Self {
        Self {
            substeps_per_frame,
            xpbd_volume_iterations: params.xpbd_volume_iterations,
            xpbd_structural_iterations: params.xpbd_structural_iterations,
            xpbd_volume_compliance: finite_nonnegative(params.xpbd_volume_compliance),
            xpbd_bead_compliance: finite_nonnegative(params.xpbd_bead_compliance),
            xpbd_tread_belt_compliance: finite_nonnegative(params.xpbd_tread_belt_compliance),
            xpbd_sidewall_bend_compliance: finite_nonnegative(params.xpbd_sidewall_bend_compliance),
            xpbd_sidewall_dihedral_compliance: finite_nonnegative(
                params.xpbd_sidewall_dihedral_compliance,
            ),
            semi_implicit_stiffness_scale: finite_nonnegative(params.semi_implicit_stiffness_scale),
            last_xpbd_volume_correction_m: finite_nonnegative(
                diagnostics.last_xpbd_volume_correction_m,
            ),
            last_xpbd_bead_correction_m: finite_nonnegative(
                diagnostics.last_xpbd_bead_correction_m,
            ),
            last_xpbd_structural_correction_m: finite_nonnegative(
                diagnostics.last_xpbd_structural_correction_m,
            ),
            max_tet_volume_error_m3: finite_nonnegative(diagnostics.max_tet_volume_error_m3),
            mean_bead_mount_error_m: finite_nonnegative(diagnostics.mean_bead_mount_error_m),
            max_bead_mount_error_m: finite_nonnegative(diagnostics.max_bead_mount_error_m),
            max_node_speed_m_per_s: finite_nonnegative(max_node_speed_m_per_s),
            dissipated_power_w: finite_nonnegative(telemetry.dissipated_power_w),
            dissipated_energy_j: finite_nonnegative(telemetry.dissipated_energy_j),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.xpbd_volume_compliance.is_finite()
            && self.xpbd_bead_compliance.is_finite()
            && self.xpbd_tread_belt_compliance.is_finite()
            && self.xpbd_sidewall_bend_compliance.is_finite()
            && self.xpbd_sidewall_dihedral_compliance.is_finite()
            && self.semi_implicit_stiffness_scale.is_finite()
            && self.last_xpbd_volume_correction_m.is_finite()
            && self.last_xpbd_bead_correction_m.is_finite()
            && self.last_xpbd_structural_correction_m.is_finite()
            && self.max_tet_volume_error_m3.is_finite()
            && self.mean_bead_mount_error_m.is_finite()
            && self.max_bead_mount_error_m.is_finite()
            && self.max_node_speed_m_per_s.is_finite()
            && self.dissipated_power_w.is_finite()
            && self.dissipated_energy_j.is_finite()
    }
}

/// One data-free acceptance envelope attached to a tire-rig report.
///
/// The bounds are qualitative engineering envelopes (monotonic/sane behavior),
/// not measured tire targets. Use these to explain why a rig passed/failed
/// without turning the native model into fitted curve output.
#[derive(Clone, Debug)]
pub struct TireRigAcceptanceCheck {
    pub check_id: String,
    pub expectation: String,
    pub metric_name: String,
    pub observed_value: f64,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub passed: bool,
}

impl TireRigAcceptanceCheck {
    pub fn at_least(
        check_id: &str,
        expectation: &str,
        metric_name: &str,
        observed_value: f64,
        min_value: f64,
    ) -> Self {
        let observed_value = finite_or(observed_value, 0.0);
        let min_value = finite_or(min_value, 0.0);
        Self {
            check_id: csv_token(check_id),
            expectation: csv_token(expectation),
            metric_name: csv_token(metric_name),
            observed_value,
            min_value: Some(min_value),
            max_value: None,
            passed: observed_value >= min_value,
        }
    }

    pub fn at_most(
        check_id: &str,
        expectation: &str,
        metric_name: &str,
        observed_value: f64,
        max_value: f64,
    ) -> Self {
        let observed_value = finite_or(observed_value, 0.0);
        let max_value = finite_or(max_value, 0.0);
        Self {
            check_id: csv_token(check_id),
            expectation: csv_token(expectation),
            metric_name: csv_token(metric_name),
            observed_value,
            min_value: None,
            max_value: Some(max_value),
            passed: observed_value <= max_value,
        }
    }

    pub fn between(
        check_id: &str,
        expectation: &str,
        metric_name: &str,
        observed_value: f64,
        min_value: f64,
        max_value: f64,
    ) -> Self {
        let observed_value = finite_or(observed_value, 0.0);
        let lo = finite_or(min_value.min(max_value), 0.0);
        let hi = finite_or(max_value.max(min_value), lo);
        Self {
            check_id: csv_token(check_id),
            expectation: csv_token(expectation),
            metric_name: csv_token(metric_name),
            observed_value,
            min_value: Some(lo),
            max_value: Some(hi),
            passed: observed_value >= lo && observed_value <= hi,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.observed_value.is_finite()
            && self.min_value.is_none_or(f64::is_finite)
            && self.max_value.is_none_or(f64::is_finite)
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "{} [{}] metric={} observed={:.6} min={} max={} passed={}",
            self.check_id,
            self.expectation,
            self.metric_name,
            self.observed_value,
            csv_optional_f64(self.min_value),
            csv_optional_f64(self.max_value),
            self.passed
        )
    }
}

/// A compact non-calibration acceptance report for one tire-rig snapshot.
#[derive(Clone, Debug)]
pub struct TireRigAcceptanceReport {
    pub snapshot: TireRigSnapshot,
    pub checks: Vec<TireRigAcceptanceCheck>,
}

impl TireRigAcceptanceReport {
    pub fn new(snapshot: TireRigSnapshot, checks: Vec<TireRigAcceptanceCheck>) -> Self {
        Self { snapshot, checks }
    }

    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|c| c.passed)
    }

    pub fn is_finite(&self) -> bool {
        self.snapshot.is_finite() && self.checks.iter().all(TireRigAcceptanceCheck::is_finite)
    }

    pub fn to_text_report(&self) -> String {
        let mut out = format!(
            "scenario={} tire={} surface={} terrain={} path={} non_calibration=true passed={}",
            csv_token(&self.snapshot.scenario_id),
            csv_token(&self.snapshot.tire_preset),
            csv_token(&self.snapshot.surface_preset),
            csv_token(&self.snapshot.terrain_preset),
            csv_token(&self.snapshot.physics_path),
            self.passed()
        );
        for check in &self.checks {
            out.push('\n');
            out.push_str(&check.to_text_line());
        }
        out
    }
}

/// Dimensionless sanity metrics derived from one rig snapshot.
///
/// These ratios are not calibration targets. They are compact invariants for
/// catching impossible tire behavior across hand-tuned rigs and GPU/CPU
/// parity runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireRigDimensionlessSanityMetrics {
    pub abs_fx_over_fz: f64,
    pub abs_fy_over_fz: f64,
    pub consumed_over_available_friction: f64,
    pub patch_length_over_radius: f64,
    pub contact_pressure_area_over_load: f64,
    pub dissipated_energy_over_load_radius: f64,
    pub cpu_gpu_relative_force_error: Option<f64>,
}

/// Contact energy/sign accounting derived from unit-labelled telemetry.
///
/// This is a model-audit helper: it verifies that dissipative terms are
/// non-negative, friction use is inside the reported budget, and the reported
/// aligning moment follows the pneumatic-trail convention when that convention
/// is strong enough to be meaningful.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireEnergySignAudit {
    pub dissipated_power_w: f64,
    pub dissipated_energy_j: f64,
    pub friction_budget_residual_n: f64,
    pub moving_ground_speed_m_per_s: f64,
    pub moving_ground_power_w: f64,
    pub expected_pneumatic_aligning_moment_nm: f64,
    pub reported_aligning_moment_nm: f64,
    pub dissipated_power_nonnegative: bool,
    pub dissipated_energy_nonnegative: bool,
    pub friction_budget_not_exceeded: bool,
    pub moving_ground_work_sign_ok: bool,
    pub aligning_moment_sign_ok: bool,
}

impl TireEnergySignAudit {
    pub fn from_telemetry(telemetry: &RealTireTelemetrySample) -> Self {
        Self::from_telemetry_with_surface_velocity(telemetry, [0.0; 3])
    }

    pub fn from_snapshot(snapshot: &TireRigSnapshot) -> Self {
        Self::from_telemetry_with_surface_velocity(
            &snapshot.telemetry,
            snapshot.terrain_surface_velocity_m_per_s,
        )
    }

    pub fn from_telemetry_with_surface_velocity(
        telemetry: &RealTireTelemetrySample,
        surface_velocity_m_per_s: [f64; 3],
    ) -> Self {
        let dissipated_power_w = finite_or(telemetry.dissipated_power_w, 0.0);
        let dissipated_energy_j = finite_or(telemetry.dissipated_energy_j, 0.0);
        let friction_budget_residual_n = finite_or(
            telemetry.available_friction_budget_n - telemetry.consumed_friction_budget_n,
            0.0,
        );
        let surface_velocity_m_per_s = sanitize_vec3(surface_velocity_m_per_s, [0.0; 3]);
        let moving_ground_speed_m_per_s = dot(surface_velocity_m_per_s, surface_velocity_m_per_s)
            .sqrt()
            .max(0.0);
        let moving_ground_power_w =
            finite_or(dot(telemetry.force_n, surface_velocity_m_per_s), 0.0);
        let force_norm_n = dot(telemetry.force_n, telemetry.force_n).sqrt().max(0.0);
        let moving_power_scale_w = (force_norm_n * moving_ground_speed_m_per_s).max(1.0);
        let moving_ground_work_sign_ok = moving_ground_speed_m_per_s <= 1.0e-6
            || moving_ground_power_w >= -0.05 * moving_power_scale_w;
        let pneumatic_trail_m = finite_or(telemetry.patch.pneumatic_trail_m, 0.0);
        // y-up frame with `side = heading × normal`: the restoring aligning
        // moment about the up-axis is `Mz = Fy · trail` for positive trail
        // (resultant behind the contact center) — the ISO z-up convention
        // writes the same torque as `-Fy · t`.
        let expected_pneumatic_aligning_moment_nm =
            finite_or(telemetry.lateral_force_n, 0.0) * pneumatic_trail_m;
        let reported_aligning_moment_nm = finite_or(telemetry.aligning_moment_nm, 0.0);
        // 0.05 N·m absolute floor: below that the moment is FP noise from
        // degenerate (single-cell) patches, not a sign convention signal.
        let sign_eps = (finite_nonnegative(telemetry.normal_load_n)
            * finite_nonnegative(telemetry.patch.patch_length_m)
            * 1.0e-5)
            .max(0.05);
        let trail_geometry_in_patch = pneumatic_trail_m.abs()
            <= finite_nonnegative(telemetry.patch.patch_length_m).max(1.0e-6);
        // The reported moment is taken about the hub, so it also carries a
        // `-Fy·center` term. On obstacles/uneven ground the displaced patch
        // center can dominate the trail term; the convention is only
        // testable when the trail term actually dominates.
        let lateral_force_n = finite_or(telemetry.lateral_force_n, 0.0);
        let trail_term_nm = lateral_force_n * pneumatic_trail_m;
        let center_term_nm = lateral_force_n * finite_or(telemetry.patch.contact_center_x_m, 0.0);
        let trail_term_dominates = trail_term_nm.abs() >= 2.0 * center_term_nm.abs();
        let aligning_moment_sign_ok = !trail_geometry_in_patch
            || !trail_term_dominates
            || expected_pneumatic_aligning_moment_nm.abs() <= sign_eps
            || reported_aligning_moment_nm.abs() <= sign_eps
            || expected_pneumatic_aligning_moment_nm.signum()
                == reported_aligning_moment_nm.signum();
        Self {
            dissipated_power_w,
            dissipated_energy_j,
            friction_budget_residual_n,
            moving_ground_speed_m_per_s,
            moving_ground_power_w,
            expected_pneumatic_aligning_moment_nm,
            reported_aligning_moment_nm,
            dissipated_power_nonnegative: dissipated_power_w >= -1.0e-9,
            dissipated_energy_nonnegative: dissipated_energy_j >= -1.0e-9,
            friction_budget_not_exceeded: friction_budget_residual_n
                >= -0.05 * telemetry.available_friction_budget_n.max(1.0),
            moving_ground_work_sign_ok,
            aligning_moment_sign_ok,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.dissipated_power_w.is_finite()
            && self.dissipated_energy_j.is_finite()
            && self.friction_budget_residual_n.is_finite()
            && self.moving_ground_speed_m_per_s.is_finite()
            && self.moving_ground_power_w.is_finite()
            && self.expected_pneumatic_aligning_moment_nm.is_finite()
            && self.reported_aligning_moment_nm.is_finite()
    }

    pub fn passed(&self) -> bool {
        self.is_finite()
            && self.dissipated_power_nonnegative
            && self.dissipated_energy_nonnegative
            && self.friction_budget_not_exceeded
            && self.moving_ground_work_sign_ok
            && self.aligning_moment_sign_ok
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "dissipated_power_nonnegative",
                "sliding/rolling contact should not report negative dissipated power",
                "dissipated_power_w",
                self.dissipated_power_w,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dissipated_energy_nonnegative",
                "integrated contact energy loss should be non-negative",
                "dissipated_energy_j",
                self.dissipated_energy_j,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "friction_budget_residual_nonnegative",
                "consumed friction should stay inside the reported available budget",
                "friction_budget_residual_n",
                self.friction_budget_residual_n,
                -1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "moving_ground_work_expected_sign",
                "moving terrain should not inject a large opposite-sign tangential work term",
                "moving_ground_work_sign_ok",
                if self.moving_ground_work_sign_ok {
                    1.0
                } else {
                    0.0
                },
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "aligning_moment_sign_convention",
                "reported aligning moment should follow Fy*pneumatic_trail (y-up frame) when the trail term dominates",
                "aligning_moment_sign_ok",
                if self.aligning_moment_sign_ok {
                    1.0
                } else {
                    0.0
                },
                1.0,
            ),
        ]
    }
}

impl TireRigDimensionlessSanityMetrics {
    pub fn from_snapshot(snapshot: &TireRigSnapshot, tire_radius_m: f64) -> Self {
        let radius = finite_positive_or(tire_radius_m, 1.0, 1.0e-6);
        let load = finite_nonnegative(snapshot.telemetry.normal_load_n).max(1.0);
        let available = finite_nonnegative(snapshot.telemetry.available_friction_budget_n).max(1.0);
        let patch = snapshot.telemetry.patch;
        Self {
            abs_fx_over_fz: (finite_or(snapshot.telemetry.longitudinal_force_n, 0.0) / load).abs(),
            abs_fy_over_fz: (finite_or(snapshot.telemetry.lateral_force_n, 0.0) / load).abs(),
            consumed_over_available_friction: finite_nonnegative(
                snapshot.telemetry.consumed_friction_budget_n,
            ) / available,
            patch_length_over_radius: finite_nonnegative(patch.patch_length_m) / radius,
            contact_pressure_area_over_load: finite_nonnegative(patch.mean_normal_pressure_pa)
                * finite_nonnegative(patch.contact_area_m2)
                / load,
            dissipated_energy_over_load_radius: finite_nonnegative(
                snapshot.telemetry.dissipated_energy_j,
            ) / (load * radius).max(1.0),
            cpu_gpu_relative_force_error: snapshot.cpu_gpu_relative_force_error,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.abs_fx_over_fz.is_finite()
            && self.abs_fy_over_fz.is_finite()
            && self.consumed_over_available_friction.is_finite()
            && self.patch_length_over_radius.is_finite()
            && self.contact_pressure_area_over_load.is_finite()
            && self.dissipated_energy_over_load_radius.is_finite()
            && self.cpu_gpu_relative_force_error.is_none_or(f64::is_finite)
    }

    pub fn default_acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        let mut checks = vec![
            TireRigAcceptanceCheck::between(
                "fx_over_fz_sane",
                "longitudinal force ratio should stay inside a broad physical envelope",
                "abs_fx_over_fz",
                self.abs_fx_over_fz,
                0.0,
                2.0,
            ),
            TireRigAcceptanceCheck::between(
                "fy_over_fz_sane",
                "lateral force ratio should stay inside a broad physical envelope",
                "abs_fy_over_fz",
                self.abs_fy_over_fz,
                0.0,
                2.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "friction_budget_ratio_sane",
                "consumed friction budget should not explode far above the available budget",
                "consumed_over_available_friction",
                self.consumed_over_available_friction,
                1.25,
            ),
            TireRigAcceptanceCheck::between(
                "patch_length_radius_sane",
                "contact patch length should be non-negative and not dwarf the tire radius",
                "patch_length_over_radius",
                self.patch_length_over_radius,
                0.0,
                2.5,
            ),
            TireRigAcceptanceCheck::between(
                "pressure_load_accounting_sane",
                "mean pressure times area should account for the reported normal load",
                "contact_pressure_area_over_load",
                self.contact_pressure_area_over_load,
                0.0,
                1.25,
            ),
            TireRigAcceptanceCheck::at_most(
                "slip_energy_ratio_sane",
                "dissipated slip energy should stay finite relative to load times radius",
                "dissipated_energy_over_load_radius",
                self.dissipated_energy_over_load_radius,
                10.0,
            ),
        ];
        if let Some(err) = self.cpu_gpu_relative_force_error {
            checks.push(TireRigAcceptanceCheck::at_most(
                "cpu_gpu_force_error_sane",
                "CPU and GPU force paths should stay within the configured broad parity envelope",
                "cpu_gpu_relative_force_error",
                err,
                0.25,
            ));
        }
        checks
    }
}

impl TireRigSnapshot {
    pub const CSV_HEADER: &'static str = concat!(
        "scenario_id,tire_preset,surface_preset,terrain_preset,physics_path,",
        "tire_parameter_provenance,surface_parameter_provenance,solver_parameter_provenance,",
        "load_n,pressure_pa,hub_height_m,deflection_m,speed_m_per_s,",
        "slip_ratio_command,slip_angle_command_rad,camber_angle_command_rad,",
        "terrain_height_m,terrain_normal_x,terrain_normal_y,terrain_normal_z,",
        "terrain_surface_velocity_x_m_per_s,terrain_surface_velocity_y_m_per_s,",
        "terrain_surface_velocity_z_m_per_s,moving_ground_power_w,",
        "surface_temperature_c,tread_temperature_c,carcass_temperature_c,",
        "tread_depth_m,wear_ratio,cpu_gpu_force_delta_x_n,",
        "cpu_gpu_force_delta_y_n,cpu_gpu_force_delta_z_n,cpu_gpu_relative_force_error,",
        "solver_substeps_per_frame,solver_xpbd_volume_iterations,",
        "solver_xpbd_structural_iterations,solver_xpbd_volume_compliance,",
        "solver_xpbd_bead_compliance,solver_xpbd_tread_belt_compliance,",
        "solver_xpbd_sidewall_bend_compliance,solver_xpbd_sidewall_dihedral_compliance,",
        "solver_semi_implicit_stiffness_scale,",
        "solver_last_xpbd_volume_correction_m,",
        "solver_last_xpbd_bead_correction_m,solver_last_xpbd_structural_correction_m,",
        "solver_max_tet_volume_error_m3,solver_mean_bead_mount_error_m,",
        "solver_max_bead_mount_error_m,solver_max_node_speed_m_per_s,",
        "solver_dissipated_power_w,solver_dissipated_energy_j,",
        "in_contact,force_x_n,force_y_n,force_z_n,normal_load_n,longitudinal_force_n,",
        "lateral_force_n,aligning_moment_nm,patch_length_m,contact_area_m2,",
        "mean_normal_pressure_pa,peak_normal_pressure_pa,contact_center_x_m,",
        "force_resultant_x_m,pneumatic_trail_m,adhesion_fraction,sliding_fraction,",
        "dissipated_power_w,dissipated_energy_j"
    );

    pub fn from_sweep_point_on_terrain(
        scenario_id: &str,
        tire_preset: &str,
        surface_preset: &str,
        terrain_preset: &str,
        physics_path: &str,
        point: &TireRigSweepPoint,
        terrain: &dyn Terrain,
        x_m: f64,
        z_m: f64,
    ) -> Self {
        let x_m = finite_or(x_m, 0.0);
        let z_m = finite_or(z_m, 0.0);
        let terrain_surface_velocity_m_per_s =
            sanitize_vec3(terrain.surface_velocity(x_m, z_m), [0.0; 3]);
        Self {
            scenario_id: csv_token(scenario_id),
            tire_preset: csv_token(tire_preset),
            surface_preset: csv_token(surface_preset),
            terrain_preset: csv_token(terrain_preset),
            physics_path: csv_token(physics_path),
            tire_parameter_provenance: TireModelProvenance::HandTuned,
            surface_parameter_provenance: TireModelProvenance::HandTuned,
            solver_parameter_provenance: TireModelProvenance::NumericalStability,
            load_n: finite_nonnegative(point.load_n),
            pressure_pa: finite_nonnegative(point.pressure_pa),
            hub_height_m: finite_or(point.hub_height_m, 0.0),
            deflection_m: finite_nonnegative(point.deflection_m),
            speed_m_per_s: finite_or(point.speed_m_per_s, 0.0),
            slip_ratio_command: finite_or(point.slip_ratio_command, 0.0),
            slip_angle_command_rad: finite_or(point.slip_angle_command_rad, 0.0),
            camber_angle_command_rad: finite_or(point.camber_angle_command_rad, 0.0),
            terrain_height_m: finite_or(terrain.height(x_m, z_m), 0.0),
            terrain_normal: sanitize_vec3(terrain.normal(x_m, z_m), [0.0, 1.0, 0.0]),
            terrain_surface_velocity_m_per_s,
            moving_ground_power_w: finite_or(
                dot(point.telemetry.force_n, terrain_surface_velocity_m_per_s),
                0.0,
            ),
            surface_temperature_c: None,
            tread_temperature_c: None,
            carcass_temperature_c: None,
            tread_depth_m: None,
            wear_ratio: None,
            cpu_gpu_force_delta_n: None,
            cpu_gpu_relative_force_error: None,
            solver_budget: None,
            telemetry: point.telemetry.clone(),
        }
    }

    pub fn with_thermal_wear(mut self, thermal: &TireThermalLayers, wear: &TireWearState) -> Self {
        self.surface_temperature_c = Some(finite_or(thermal.surface_c, thermal.ambient_c));
        self.tread_temperature_c = Some(finite_or(thermal.tread_c, thermal.ambient_c));
        self.carcass_temperature_c = Some(finite_or(thermal.carcass_c, thermal.ambient_c));
        self.tread_depth_m = Some(finite_nonnegative(wear.depth_m));
        self.wear_ratio = Some(
            (finite_nonnegative(wear.depth_m) / finite_positive_or(wear.max_depth_m, 1.0, 1.0e-9))
                .clamp(0.0, 1.0),
        );
        self
    }

    pub fn with_cpu_gpu_force_delta(mut self, gpu_force_n: [f64; 3]) -> Self {
        let gpu = sanitize_vec3(gpu_force_n, [0.0; 3]);
        let cpu = self.telemetry.force_n;
        let delta = [gpu[0] - cpu[0], gpu[1] - cpu[1], gpu[2] - cpu[2]];
        let delta_norm = dot(delta, delta).sqrt();
        let ref_norm = dot(cpu, cpu).sqrt().max(1.0);
        self.cpu_gpu_force_delta_n = Some(delta);
        self.cpu_gpu_relative_force_error = Some(delta_norm / ref_norm);
        self
    }

    pub fn with_solver_budget(mut self, solver_budget: TireSolverBudgetTelemetry) -> Self {
        self.solver_budget = Some(solver_budget);
        self
    }

    pub fn is_finite(&self) -> bool {
        self.load_n.is_finite()
            && self.pressure_pa.is_finite()
            && self.hub_height_m.is_finite()
            && self.deflection_m.is_finite()
            && self.speed_m_per_s.is_finite()
            && self.slip_ratio_command.is_finite()
            && self.slip_angle_command_rad.is_finite()
            && self.camber_angle_command_rad.is_finite()
            && self.terrain_height_m.is_finite()
            && self.terrain_normal.iter().all(|v| v.is_finite())
            && self
                .terrain_surface_velocity_m_per_s
                .iter()
                .all(|v| v.is_finite())
            && self.moving_ground_power_w.is_finite()
            && self.surface_temperature_c.is_none_or(f64::is_finite)
            && self.tread_temperature_c.is_none_or(f64::is_finite)
            && self.carcass_temperature_c.is_none_or(f64::is_finite)
            && self.tread_depth_m.is_none_or(f64::is_finite)
            && self.wear_ratio.is_none_or(f64::is_finite)
            && self
                .cpu_gpu_force_delta_n
                .is_none_or(|v| v.iter().all(|x| x.is_finite()))
            && self.cpu_gpu_relative_force_error.is_none_or(f64::is_finite)
            && self.solver_budget.is_none_or(|v| v.is_finite())
            && self.telemetry.is_finite()
    }

    pub fn to_csv_row(&self) -> String {
        let solver = self.solver_budget;
        let fields = vec![
            csv_token(&self.scenario_id),
            csv_token(&self.tire_preset),
            csv_token(&self.surface_preset),
            csv_token(&self.terrain_preset),
            csv_token(&self.physics_path),
            self.tire_parameter_provenance.label().to_owned(),
            self.surface_parameter_provenance.label().to_owned(),
            self.solver_parameter_provenance.label().to_owned(),
            format!("{:.6}", self.load_n),
            format!("{:.6}", self.pressure_pa),
            format!("{:.9}", self.hub_height_m),
            format!("{:.9}", self.deflection_m),
            format!("{:.6}", self.speed_m_per_s),
            format!("{:.9}", self.slip_ratio_command),
            format!("{:.9}", self.slip_angle_command_rad),
            format!("{:.9}", self.camber_angle_command_rad),
            format!("{:.9}", self.terrain_height_m),
            format!("{:.9}", self.terrain_normal[0]),
            format!("{:.9}", self.terrain_normal[1]),
            format!("{:.9}", self.terrain_normal[2]),
            format!("{:.6}", self.terrain_surface_velocity_m_per_s[0]),
            format!("{:.6}", self.terrain_surface_velocity_m_per_s[1]),
            format!("{:.6}", self.terrain_surface_velocity_m_per_s[2]),
            format!("{:.6}", self.moving_ground_power_w),
            csv_optional_f64(self.surface_temperature_c),
            csv_optional_f64(self.tread_temperature_c),
            csv_optional_f64(self.carcass_temperature_c),
            csv_optional_f64(self.tread_depth_m),
            csv_optional_f64(self.wear_ratio),
            csv_optional_vec3(self.cpu_gpu_force_delta_n, 0),
            csv_optional_vec3(self.cpu_gpu_force_delta_n, 1),
            csv_optional_vec3(self.cpu_gpu_force_delta_n, 2),
            csv_optional_f64(self.cpu_gpu_relative_force_error),
            solver
                .map(|s| s.substeps_per_frame.to_string())
                .unwrap_or_default(),
            solver
                .map(|s| s.xpbd_volume_iterations.to_string())
                .unwrap_or_default(),
            solver
                .map(|s| s.xpbd_structural_iterations.to_string())
                .unwrap_or_default(),
            csv_optional_f64(solver.map(|s| s.xpbd_volume_compliance)),
            csv_optional_f64(solver.map(|s| s.xpbd_bead_compliance)),
            csv_optional_f64(solver.map(|s| s.xpbd_tread_belt_compliance)),
            csv_optional_f64(solver.map(|s| s.xpbd_sidewall_bend_compliance)),
            csv_optional_f64(solver.map(|s| s.xpbd_sidewall_dihedral_compliance)),
            csv_optional_f64(solver.map(|s| s.semi_implicit_stiffness_scale)),
            csv_optional_f64(solver.map(|s| s.last_xpbd_volume_correction_m)),
            csv_optional_f64(solver.map(|s| s.last_xpbd_bead_correction_m)),
            csv_optional_f64(solver.map(|s| s.last_xpbd_structural_correction_m)),
            csv_optional_f64(solver.map(|s| s.max_tet_volume_error_m3)),
            csv_optional_f64(solver.map(|s| s.mean_bead_mount_error_m)),
            csv_optional_f64(solver.map(|s| s.max_bead_mount_error_m)),
            csv_optional_f64(solver.map(|s| s.max_node_speed_m_per_s)),
            csv_optional_f64(solver.map(|s| s.dissipated_power_w)),
            csv_optional_f64(solver.map(|s| s.dissipated_energy_j)),
            self.telemetry.in_contact.to_string(),
            format!("{:.6}", self.telemetry.force_n[0]),
            format!("{:.6}", self.telemetry.force_n[1]),
            format!("{:.6}", self.telemetry.force_n[2]),
            format!("{:.6}", self.telemetry.normal_load_n),
            format!("{:.6}", self.telemetry.longitudinal_force_n),
            format!("{:.6}", self.telemetry.lateral_force_n),
            format!("{:.6}", self.telemetry.aligning_moment_nm),
            format!("{:.9}", self.telemetry.patch.patch_length_m),
            format!("{:.9}", self.telemetry.patch.contact_area_m2),
            format!("{:.6}", self.telemetry.patch.mean_normal_pressure_pa),
            format!("{:.6}", self.telemetry.patch.peak_normal_pressure_pa),
            format!("{:.9}", self.telemetry.patch.contact_center_x_m),
            format!("{:.9}", self.telemetry.patch.force_resultant_x_m),
            format!("{:.9}", self.telemetry.patch.pneumatic_trail_m),
            format!("{:.6}", self.telemetry.patch.adhesion_fraction),
            format!("{:.6}", self.telemetry.patch.sliding_fraction),
            format!("{:.6}", self.telemetry.dissipated_power_w),
            format!("{:.6}", self.telemetry.dissipated_energy_j),
        ];
        fields.join(",")
    }

    pub fn acceptance_report(self, checks: Vec<TireRigAcceptanceCheck>) -> TireRigAcceptanceReport {
        TireRigAcceptanceReport::new(self, checks)
    }

    pub fn dimensionless_sanity_metrics(
        &self,
        tire_radius_m: f64,
    ) -> TireRigDimensionlessSanityMetrics {
        TireRigDimensionlessSanityMetrics::from_snapshot(self, tire_radius_m)
    }

    pub fn dimensionless_sanity_report(self, tire_radius_m: f64) -> TireRigAcceptanceReport {
        let metrics = self.dimensionless_sanity_metrics(tire_radius_m);
        TireRigAcceptanceReport::new(self, metrics.default_acceptance_checks())
    }

    pub fn energy_sign_audit(&self) -> TireEnergySignAudit {
        TireEnergySignAudit::from_snapshot(self)
    }

    pub fn energy_sign_report(self) -> TireRigAcceptanceReport {
        let audit = self.energy_sign_audit();
        TireRigAcceptanceReport::new(self, audit.acceptance_checks())
    }
}

/// Configuration for the built-in deterministic tire-rig runner.
///
/// The runner deliberately carries a seed/provenance even though the current
/// scenarios are deterministic and do not sample randomness. That keeps the
/// artifact format stable if future non-calibration rigs need seeded jitter or
/// randomized ordering for robustness checks. It must remain a replay/regression
/// tool, not a parameter-fitting harness.
#[derive(Clone, Debug)]
pub struct TireRigRunConfig {
    pub seed: u64,
    pub tire_preset: String,
    pub high_pressure_tire_preset: String,
    pub low_pressure_tire_preset: String,
    pub dry_surface_preset: String,
    pub ice_surface_preset: String,
    pub wet_surface_preset: String,
    pub soil_surface_preset: String,
    pub fixed_dt_s: f64,
}

impl Default for TireRigRunConfig {
    fn default() -> Self {
        Self {
            seed: 0x6d6f_6c6c_615f_7469,
            tire_preset: "passenger_radial".to_owned(),
            high_pressure_tire_preset: "high_pressure_reference".to_owned(),
            low_pressure_tire_preset: "offroad_soft".to_owned(),
            dry_surface_preset: "dry_asphalt".to_owned(),
            ice_surface_preset: "ice".to_owned(),
            wet_surface_preset: "wet_asphalt_film".to_owned(),
            soil_surface_preset: "loose_soil".to_owned(),
            fixed_dt_s: 1.0 / 120.0,
        }
    }
}

/// Output from the built-in deterministic tire-rig runner.
///
/// This is intentionally a collection of internal snapshots and broad
/// acceptance envelopes. It contains no measured targets and performs no
/// fitting. Use it to replay/falsify the native brush/FEM tire model.
#[derive(Clone, Debug)]
pub struct TireRigRunOutput {
    pub runner_id: String,
    pub seed: u64,
    pub snapshots: Vec<TireRigSnapshot>,
    pub reports: Vec<TireRigAcceptanceReport>,
    pub sensitivity_points: Vec<TireRigSensitivityPoint>,
    pub vehicle_sanity_cases: Vec<TireVehicleSanityCase>,
}

/// Coverage report for compact replay/regression artifacts.
///
/// This is intentionally about artifact completeness, not about matching
/// measured data. It checks that each runner snapshot carries the command
/// inputs, preset/provenance labels, terrain state, force/moment telemetry,
/// patch/contact telemetry, optional CPU/GPU columns, solver budget support,
/// and known-approximation accounting needed to reproduce a broken viewer
/// example from a small CSV/text artifact.
#[derive(Clone, Debug, Default)]
pub struct TireReplayArtifactCoverageReport {
    pub snapshot_count: usize,
    pub csv_header_columns: usize,
    pub csv_row_column_mismatches: usize,
    pub max_csv_row_bytes: usize,
    pub command_input_rows: usize,
    pub terrain_state_rows: usize,
    pub provenance_rows: usize,
    pub force_moment_rows: usize,
    pub patch_telemetry_rows: usize,
    pub thermal_wear_rows: usize,
    pub solver_budget_rows: usize,
    pub cpu_gpu_delta_columns_present: bool,
    pub known_approximation_rows: usize,
}

impl TireReplayArtifactCoverageReport {
    pub fn passed(&self) -> bool {
        self.snapshot_count >= 20
            && self.csv_header_columns >= 40
            && self.csv_row_column_mismatches == 0
            && self.max_csv_row_bytes <= 4096
            && self.command_input_rows == self.snapshot_count
            && self.terrain_state_rows == self.snapshot_count
            && self.provenance_rows == self.snapshot_count
            && self.force_moment_rows == self.snapshot_count
            && self.patch_telemetry_rows == self.snapshot_count
            && self.thermal_wear_rows >= 3
            && self.solver_budget_rows >= 1
            && self.cpu_gpu_delta_columns_present
            && self.known_approximation_rows >= 5
    }

    pub fn is_finite(&self) -> bool {
        true
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_snapshot_count",
                "default tire rig should emit enough compact replay snapshots for regression triage",
                "snapshot_count",
                self.snapshot_count as f64,
                20.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "replay_artifact_csv_rows_match_header",
                "every replay CSV row should match the dependency-free snapshot header",
                "csv_row_column_mismatches",
                self.csv_row_column_mismatches as f64,
                0.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "replay_artifact_rows_stay_compact",
                "snapshot CSV rows should stay small enough for tests and issue reports",
                "max_csv_row_bytes",
                self.max_csv_row_bytes as f64,
                4096.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_command_inputs_present",
                "every replay snapshot should carry load, pressure, speed, slip, camber, and hub state",
                "command_input_rows",
                self.command_input_rows as f64,
                self.snapshot_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_terrain_state_present",
                "every replay snapshot should carry terrain height, normal, surface velocity, and moving-ground work",
                "terrain_state_rows",
                self.terrain_state_rows as f64,
                self.snapshot_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_provenance_present",
                "every replay snapshot should carry tire/surface/solver provenance labels",
                "provenance_rows",
                self.provenance_rows as f64,
                self.snapshot_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_force_moment_present",
                "every replay snapshot should carry force and aligning-moment telemetry",
                "force_moment_rows",
                self.force_moment_rows as f64,
                self.snapshot_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_patch_telemetry_present",
                "every replay snapshot should carry patch/contact telemetry",
                "patch_telemetry_rows",
                self.patch_telemetry_rows as f64,
                self.snapshot_count as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_thermal_wear_state_present",
                "thermal replay snapshots should carry heat/wear state",
                "thermal_wear_rows",
                self.thermal_wear_rows as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_solver_budget_present",
                "at least one replay snapshot should carry FEM solver-budget telemetry",
                "solver_budget_rows",
                self.solver_budget_rows as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_cpu_gpu_delta_columns_present",
                "snapshot artifact format should reserve CPU/GPU force-delta columns",
                "cpu_gpu_delta_columns_present",
                bool_metric(self.cpu_gpu_delta_columns_present),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "replay_artifact_known_approximations_available",
                "replay artifacts should be paired with known-approximation ledger rows",
                "known_approximation_rows",
                self.known_approximation_rows as f64,
                5.0,
            ),
        ]
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "replay_artifact_coverage snapshots={} header_cols={} row_mismatches={} max_row_bytes={} command_rows={} terrain_rows={} provenance_rows={} force_moment_rows={} patch_rows={} thermal_wear_rows={} solver_budget_rows={} cpu_gpu_delta_columns={} known_approximations={} passed={}",
            self.snapshot_count,
            self.csv_header_columns,
            self.csv_row_column_mismatches,
            self.max_csv_row_bytes,
            self.command_input_rows,
            self.terrain_state_rows,
            self.provenance_rows,
            self.force_moment_rows,
            self.patch_telemetry_rows,
            self.thermal_wear_rows,
            self.solver_budget_rows,
            self.cpu_gpu_delta_columns_present,
            self.known_approximation_rows,
            self.passed()
        )
    }
}

impl TireRigRunOutput {
    pub fn report_has_replay_context(&self, report: &TireRigAcceptanceReport) -> bool {
        let scenario_id = report.snapshot.scenario_id.as_str();
        self.snapshots
            .iter()
            .any(|snapshot| snapshot.scenario_id == scenario_id)
            || (scenario_id == "rolling_obstacle_sequence"
                && [
                    "rolling_obstacle_approach",
                    "rolling_obstacle_crest",
                    "rolling_obstacle_exit",
                ]
                .into_iter()
                .all(|frame| {
                    self.snapshots
                        .iter()
                        .any(|snapshot| snapshot.scenario_id == frame)
                }))
    }

    pub fn reports_without_replay_context(&self) -> Vec<String> {
        self.reports
            .iter()
            .filter(|report| !self.report_has_replay_context(report))
            .map(|report| report.snapshot.scenario_id.clone())
            .collect()
    }

    pub fn viewer_target_coverage_report(&self) -> TireViewerTargetCoverageReport {
        TireViewerTargetCoverageReport::from_run(self)
    }

    pub fn replay_artifact_coverage_report(&self) -> TireReplayArtifactCoverageReport {
        let csv_header_columns = TireRigSnapshot::CSV_HEADER.split(',').count();
        let mut report = TireReplayArtifactCoverageReport {
            snapshot_count: self.snapshots.len(),
            csv_header_columns,
            cpu_gpu_delta_columns_present: TireRigSnapshot::CSV_HEADER
                .contains("cpu_gpu_force_delta_x_n")
                && TireRigSnapshot::CSV_HEADER.contains("cpu_gpu_relative_force_error"),
            known_approximation_rows: tire_known_approximation_ledger().len(),
            ..Default::default()
        };
        for snapshot in &self.snapshots {
            let row = snapshot.to_csv_row();
            report.max_csv_row_bytes = report.max_csv_row_bytes.max(row.len());
            report.csv_row_column_mismatches +=
                usize::from(row.split(',').count() != csv_header_columns);
            report.command_input_rows += usize::from(
                snapshot.load_n.is_finite()
                    && snapshot.pressure_pa.is_finite()
                    && snapshot.hub_height_m.is_finite()
                    && snapshot.deflection_m.is_finite()
                    && snapshot.speed_m_per_s.is_finite()
                    && snapshot.slip_ratio_command.is_finite()
                    && snapshot.slip_angle_command_rad.is_finite()
                    && snapshot.camber_angle_command_rad.is_finite(),
            );
            report.terrain_state_rows += usize::from(
                snapshot.terrain_height_m.is_finite()
                    && snapshot.terrain_normal.iter().all(|v| v.is_finite())
                    && snapshot
                        .terrain_surface_velocity_m_per_s
                        .iter()
                        .all(|v| v.is_finite())
                    && snapshot.moving_ground_power_w.is_finite(),
            );
            report.provenance_rows += usize::from(
                !snapshot.tire_preset.is_empty()
                    && !snapshot.surface_preset.is_empty()
                    && !snapshot.terrain_preset.is_empty()
                    && !snapshot.physics_path.is_empty(),
            );
            report.force_moment_rows += usize::from(
                snapshot.telemetry.force_n.iter().all(|v| v.is_finite())
                    && snapshot.telemetry.longitudinal_force_n.is_finite()
                    && snapshot.telemetry.lateral_force_n.is_finite()
                    && snapshot.telemetry.aligning_moment_nm.is_finite(),
            );
            report.patch_telemetry_rows += usize::from(snapshot.telemetry.patch.is_finite());
            report.thermal_wear_rows += usize::from(
                snapshot.surface_temperature_c.is_some()
                    && snapshot.tread_temperature_c.is_some()
                    && snapshot.carcass_temperature_c.is_some()
                    && snapshot.tread_depth_m.is_some()
                    && snapshot.wear_ratio.is_some(),
            );
            report.solver_budget_rows += usize::from(snapshot.solver_budget.is_some());
        }
        report
    }

    pub fn is_finite(&self) -> bool {
        self.snapshots.iter().all(TireRigSnapshot::is_finite)
            && self.reports.iter().all(TireRigAcceptanceReport::is_finite)
            && self
                .sensitivity_points
                .iter()
                .all(TireRigSensitivityPoint::is_finite)
            && self.vehicle_sanity_cases.iter().all(|case| {
                case.is_finite()
                    && case
                        .acceptance_checks()
                        .iter()
                        .all(TireRigAcceptanceCheck::is_finite)
            })
    }

    pub fn passed(&self) -> bool {
        !self.snapshots.is_empty()
            && self.reports.iter().all(TireRigAcceptanceReport::passed)
            && self.sensitivity_points.iter().all(|point| point.passed)
            && self
                .vehicle_sanity_cases
                .iter()
                .all(TireVehicleSanityCase::acceptance_passed)
            && self.viewer_target_coverage_report().passed()
            && self.replay_artifact_coverage_report().passed()
    }

    pub fn to_snapshot_csv_table(&self) -> String {
        let mut out = TireRigSnapshot::CSV_HEADER.to_owned();
        for snapshot in &self.snapshots {
            out.push('\n');
            out.push_str(&snapshot.to_csv_row());
        }
        out
    }

    pub fn to_text_report(&self) -> String {
        let mut out = format!(
            "runner={} seed={} non_calibration=true snapshots={} reports={} sensitivities={} vehicle_sanity={} passed={}",
            csv_token(&self.runner_id),
            self.seed,
            self.snapshots.len(),
            self.reports.len(),
            self.sensitivity_points.len(),
            self.vehicle_sanity_cases.len(),
            self.passed()
        );
        for report in &self.reports {
            out.push('\n');
            out.push_str(&report.to_text_report());
        }
        for point in &self.sensitivity_points {
            out.push('\n');
            out.push_str("sensitivity=");
            out.push_str(&point.to_text_line());
        }
        for case in &self.vehicle_sanity_cases {
            out.push('\n');
            out.push_str("vehicle_sanity=");
            out.push_str(&case.to_text_line());
        }
        let viewer_targets = self.viewer_target_coverage_report();
        out.push('\n');
        out.push_str(&format!(
            "viewer_target_coverage non_calibration=true targets={} passed={}",
            viewer_targets.checks.len(),
            viewer_targets.passed()
        ));
        for check in &viewer_targets.checks {
            out.push('\n');
            out.push_str("viewer_target=");
            out.push_str(&check.to_text_line());
        }
        let replay_artifacts = self.replay_artifact_coverage_report();
        out.push('\n');
        out.push_str(&replay_artifacts.to_text_line());
        for check in replay_artifacts.acceptance_checks() {
            out.push('\n');
            out.push_str("replay_artifact=");
            out.push_str(&check.to_text_line());
        }
        for item in tire_gpu_parity_checklist() {
            out.push('\n');
            out.push_str("gpu_parity=");
            out.push_str(&item.to_text_line());
        }
        for item in tire_known_approximation_ledger() {
            out.push('\n');
            out.push_str("known_approximation=");
            out.push_str(&item.to_text_line());
        }
        let preset_governance = tire_preset_governance_report();
        out.push('\n');
        out.push_str(&format!(
            "preset_governance_summary non_calibration=true checks={} passed={}",
            preset_governance.checks.len(),
            preset_governance.passed()
        ));
        for check in &preset_governance.checks {
            out.push('\n');
            out.push_str("preset_governance=");
            out.push_str(&check.to_text_line());
        }
        out
    }
}
