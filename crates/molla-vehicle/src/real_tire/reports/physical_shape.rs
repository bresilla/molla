//! Physical-shape reports: reference diagnostics, combined slip law, ice
//! large-patch, adhesion/sliding continuity, layered transient envelope,
//! low-speed robustness, frame convention, contact-patch convergence,
//! road-condition state.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, Terrain};

use super::super::internal::{
    bool_metric, cross, csv_token, dot, finite_nonnegative, finite_or, normalize,
};
use super::super::{
    RealTireContact, RealTireParams, RealTireTelemetrySample, RealTireTransientState,
    RealWheelOrientation, RealWheelState, SurfaceFilmState, SurfaceMaterial,
    TireConditionMaterialResponse, TireLayeredTransientState, TireRelaxationContext,
    TireRigAcceptanceCheck, TireRigAcceptanceReport, TireRigSnapshot, TireRigSurfaceTransitionBand,
    TireRigSurfaceTransitionTerrain, TireRigSweepPoint, TireThermalLayers, TireWearState,
    evaluate_brush_tire, evaluate_brush_tire_layered_transient_with_context,
    evaluate_brush_tire_oriented, evaluate_brush_tire_transient, solve_loaded_tire_hub_height,
    tire_rig_point,
};

#[derive(Clone, Debug)]
pub struct TireReferenceShapeDiagnostics {
    pub scenario_id: String,
    pub sample_count: usize,
    pub peak_longitudinal_force_n: f64,
    pub peak_lateral_force_n: f64,
    pub max_normalized_force_jump: f64,
    pub tail_slope_ratio: f64,
    pub monotonic_buildup: bool,
    pub smooth_saturation: bool,
    pub combined_slip_tradeoff: bool,
    pub aligning_moment_sign_ok: bool,
    pub friction_budget_ok: bool,
}

impl TireReferenceShapeDiagnostics {
    pub fn is_finite(&self) -> bool {
        self.peak_longitudinal_force_n.is_finite()
            && self.peak_lateral_force_n.is_finite()
            && self.max_normalized_force_jump.is_finite()
            && self.tail_slope_ratio.is_finite()
    }

    pub fn passed(&self) -> bool {
        self.is_finite()
            && self.sample_count > 0
            && self.monotonic_buildup
            && self.smooth_saturation
            && self.combined_slip_tradeoff
            && self.aligning_moment_sign_ok
            && self.friction_budget_ok
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "reference_shape_has_samples",
                "reference-shape diagnostics should be based on a non-empty rig sweep",
                "sample_count",
                self.sample_count as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "reference_shape_monotonic_buildup",
                "force should build monotonically before the broad saturation region",
                "monotonic_buildup",
                bool_metric(self.monotonic_buildup),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "reference_shape_smooth_jump",
                "adjacent normalized force changes should not teleport",
                "max_normalized_force_jump",
                self.max_normalized_force_jump,
                0.65,
            ),
            TireRigAcceptanceCheck::at_most(
                "reference_shape_tail_slope_sane",
                "tail slope should stay bounded after force approaches saturation",
                "tail_slope_ratio",
                self.tail_slope_ratio,
                1.75,
            ),
            TireRigAcceptanceCheck::at_least(
                "reference_shape_combined_slip_tradeoff",
                "combined slip should trade longitudinal and lateral force instead of adding independent peaks",
                "combined_slip_tradeoff",
                bool_metric(self.combined_slip_tradeoff),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "reference_shape_aligning_moment_sign",
                "aligning moment should follow the pneumatic-trail sign convention when nonzero",
                "aligning_moment_sign_ok",
                bool_metric(self.aligning_moment_sign_ok),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "reference_shape_friction_budget",
                "consumed friction should stay inside the reported budget",
                "friction_budget_ok",
                bool_metric(self.friction_budget_ok),
                1.0,
            ),
        ]
    }

    pub fn report(self, snapshot: TireRigSnapshot) -> TireRigAcceptanceReport {
        let checks = self.acceptance_checks();
        TireRigAcceptanceReport::new(snapshot, checks)
    }
}

/// Data-free sanity report for the native load-sensitive combined-slip law.
///
/// This is the compact executable counterpart for PLAN_REALTIRE C2: it checks
/// that the brush model's pressure/load/temperature inputs produce a smooth,
/// shared-budget saturation response instead of independent axis clamps or
/// fitted curve coefficients.
#[derive(Clone, Copy, Debug)]
pub struct TireLoadCombinedSlipLawReport {
    pub load_force_ratio: f64,
    pub load_ratio: f64,
    pub heavy_stiffness_over_light: f64,
    pub low_mu_low_pressure_over_high_pressure_fx: f64,
    pub low_mu_low_pressure_patch_over_high_pressure: f64,
    pub combined_fx_over_pure_fx: f64,
    pub combined_fy_over_pure_fy: f64,
    pub combined_budget_ratio: f64,
    pub max_saturation_jump_n: f64,
    pub max_saturation_jump_over_load: f64,
    pub saturation_start_sliding_fraction: f64,
    pub saturation_end_sliding_fraction: f64,
    pub saturation_sliding_fraction_delta: f64,
    pub ice_over_dry_fx: f64,
    pub hot_tread_shear_over_healthy: f64,
    pub hot_longitudinal_stiffness_over_healthy: f64,
    pub hot_cornering_stiffness_over_healthy: f64,
    pub finite_outputs: bool,
}

impl TireLoadCombinedSlipLawReport {
    pub fn is_finite(self) -> bool {
        self.load_force_ratio.is_finite()
            && self.load_ratio.is_finite()
            && self.heavy_stiffness_over_light.is_finite()
            && self.low_mu_low_pressure_over_high_pressure_fx.is_finite()
            && self
                .low_mu_low_pressure_patch_over_high_pressure
                .is_finite()
            && self.combined_fx_over_pure_fx.is_finite()
            && self.combined_fy_over_pure_fy.is_finite()
            && self.combined_budget_ratio.is_finite()
            && self.max_saturation_jump_n.is_finite()
            && self.max_saturation_jump_over_load.is_finite()
            && self.saturation_start_sliding_fraction.is_finite()
            && self.saturation_end_sliding_fraction.is_finite()
            && self.saturation_sliding_fraction_delta.is_finite()
            && self.ice_over_dry_fx.is_finite()
            && self.hot_tread_shear_over_healthy.is_finite()
            && self.hot_longitudinal_stiffness_over_healthy.is_finite()
            && self.hot_cornering_stiffness_over_healthy.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.load_force_ratio > 1.0
            && self.load_force_ratio < self.load_ratio * 0.98
            && self.heavy_stiffness_over_light > 1.0
            && self.low_mu_low_pressure_over_high_pressure_fx > 1.05
            && self.low_mu_low_pressure_patch_over_high_pressure > 1.05
            && self.combined_fx_over_pure_fx <= 0.98
            && self.combined_fy_over_pure_fy <= 1.05
            && self.combined_budget_ratio <= 1.05
            && self.max_saturation_jump_over_load <= 0.12
            && self.saturation_start_sliding_fraction <= 0.65
            && self.saturation_sliding_fraction_delta > 0.10
            && self.ice_over_dry_fx <= 0.30
            && self.hot_tread_shear_over_healthy < 0.95
            && self.hot_longitudinal_stiffness_over_healthy < 0.95
            && self.hot_cornering_stiffness_over_healthy < 0.95
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "load_combined_outputs_finite",
                "load/combined-slip law samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "load_sensitivity_force_increases",
                "higher vertical load should increase peak longitudinal force",
                "load_force_ratio",
                self.load_force_ratio,
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "load_sensitivity_sublinear_peak",
                "higher vertical load should increase peak force sub-linearly",
                "force_ratio_over_load_ratio",
                self.load_force_ratio / self.load_ratio.max(1.0e-9),
                0.98,
            ),
            TireRigAcceptanceCheck::at_least(
                "load_sensitivity_stiffness_increases",
                "higher vertical load should still raise tread stiffness before saturation",
                "heavy_stiffness_over_light",
                self.heavy_stiffness_over_light,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "low_mu_pressure_tractive_stiffness_improves",
                "lower pressure should improve low-friction tractive stiffness before the friction cap dominates",
                "low_pressure_over_high_pressure_fx",
                self.low_mu_low_pressure_over_high_pressure_fx,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "low_mu_pressure_patch_grows",
                "lower pressure should grow the low-friction contact patch in the same stiffness check",
                "low_pressure_patch_over_high_pressure",
                self.low_mu_low_pressure_patch_over_high_pressure,
                1.05,
            ),
            TireRigAcceptanceCheck::at_most(
                "combined_slip_reduces_longitudinal_axis",
                "lateral slip should consume shared friction budget and reduce drive force",
                "combined_fx_over_pure_fx",
                self.combined_fx_over_pure_fx,
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "combined_slip_single_budget",
                "combined longitudinal/lateral force should stay inside one friction budget",
                "combined_consumed_over_available",
                self.combined_budget_ratio,
                1.05,
            ),
            TireRigAcceptanceCheck::at_most(
                "combined_slip_saturation_continuous",
                "smooth saturation should avoid jumpy adhesion-to-sliding force transitions",
                "max_saturation_jump_over_load",
                self.max_saturation_jump_over_load,
                0.12,
            ),
            TireRigAcceptanceCheck::at_most(
                "combined_slip_saturation_starts_with_adhesion",
                "saturation sweep should begin with a meaningful adhesion fraction",
                "start_sliding_fraction",
                self.saturation_start_sliding_fraction,
                0.65,
            ),
            TireRigAcceptanceCheck::at_least(
                "combined_slip_sliding_fraction_grows",
                "saturation sweep should expose sliding fraction growth as slip rises",
                "end_minus_start_sliding_fraction",
                self.saturation_sliding_fraction_delta,
                0.10,
            ),
            TireRigAcceptanceCheck::at_most(
                "combined_slip_ice_force_cap",
                "ice should cap drive force far below asphalt even with the same tire/load",
                "ice_over_dry_fx",
                self.ice_over_dry_fx,
                0.30,
            ),
            TireRigAcceptanceCheck::at_most(
                "temperature_state_reduces_tread_shear",
                "hot/worn condition state should reduce live tread shear stiffness",
                "hot_tread_shear_over_healthy",
                self.hot_tread_shear_over_healthy,
                0.95,
            ),
            TireRigAcceptanceCheck::at_most(
                "temperature_state_reduces_longitudinal_stiffness",
                "hot/worn condition state should reduce effective longitudinal stiffness output",
                "hot_longitudinal_stiffness_over_healthy",
                self.hot_longitudinal_stiffness_over_healthy,
                0.95,
            ),
            TireRigAcceptanceCheck::at_most(
                "temperature_state_reduces_cornering_stiffness",
                "hot/worn condition state should reduce effective cornering stiffness output",
                "hot_cornering_stiffness_over_healthy",
                self.hot_cornering_stiffness_over_healthy,
                0.95,
            ),
        ]
    }
}

/// Non-calibration gate for "ice still slips with a big patch".
///
/// This targets the viewer checklist item that low pressure/large footprint
/// should not magically create grip on ice. The report compares a low-pressure
/// tire on dry asphalt vs ice under the same load/slip, and also proves the
/// low-pressure footprint is larger than a high-pressure reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireIceLargePatchReport {
    pub high_pressure_patch_length_m: f64,
    pub low_pressure_ice_patch_length_m: f64,
    pub low_pressure_dry_abs_fx_n: f64,
    pub low_pressure_ice_abs_fx_n: f64,
    pub ice_over_dry_fx: f64,
    pub low_over_high_patch_ratio: f64,
    pub dry_sliding_fraction: f64,
    pub ice_sliding_fraction: f64,
    pub ice_budget_ratio: f64,
    pub finite_outputs: bool,
}

impl TireIceLargePatchReport {
    pub fn is_finite(self) -> bool {
        self.high_pressure_patch_length_m.is_finite()
            && self.low_pressure_ice_patch_length_m.is_finite()
            && self.low_pressure_dry_abs_fx_n.is_finite()
            && self.low_pressure_ice_abs_fx_n.is_finite()
            && self.ice_over_dry_fx.is_finite()
            && self.low_over_high_patch_ratio.is_finite()
            && self.dry_sliding_fraction.is_finite()
            && self.ice_sliding_fraction.is_finite()
            && self.ice_budget_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.low_over_high_patch_ratio > 1.15
            && self.ice_over_dry_fx <= 0.30
            && self.ice_sliding_fraction >= self.dry_sliding_fraction
            && self.ice_budget_ratio <= 1.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "ice_large_patch_outputs_finite",
                "ice large-patch report should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "ice_large_patch_low_pressure_patch_grows",
                "low pressure should create a larger patch than the high-pressure reference",
                "low_over_high_patch_ratio",
                self.low_over_high_patch_ratio,
                1.15,
            ),
            TireRigAcceptanceCheck::at_most(
                "ice_large_patch_still_caps_force",
                "ice should cap force far below dry asphalt even with a large patch",
                "ice_over_dry_fx",
                self.ice_over_dry_fx,
                0.30,
            ),
            TireRigAcceptanceCheck::at_least(
                "ice_large_patch_sliding_not_hidden",
                "ice large-patch contact should slide at least as much as dry contact",
                "ice_minus_dry_sliding_fraction",
                self.ice_sliding_fraction - self.dry_sliding_fraction,
                0.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "ice_large_patch_friction_budget_bounded",
                "ice large-patch force should stay inside the shared friction budget",
                "ice_budget_ratio",
                self.ice_budget_ratio,
                1.05,
            ),
        ]
    }
}

/// Data-free guard for force/moment continuity through adhesion-to-sliding.
///
/// This is intentionally not a fitted tire curve. It sweeps the native brush
/// contact through small, medium, and saturated longitudinal slip, then runs a
/// separate lateral-slip sweep for aligning moment continuity. The report
/// checks that force, moment, sliding fraction, and shared friction-budget
/// accounting change continuously instead of jumping between hidden
/// static/dynamic modes.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireAdhesionSlidingContinuityReport {
    pub sample_count: usize,
    pub finite_outputs: bool,
    pub min_sliding_fraction: f64,
    pub max_sliding_fraction: f64,
    pub max_force_jump_over_load: f64,
    pub max_moment_jump_over_load_radius: f64,
    pub max_budget_ratio: f64,
    pub force_sign_stable: bool,
    pub moment_sign_stable: bool,
}

impl TireAdhesionSlidingContinuityReport {
    pub fn is_finite(self) -> bool {
        self.min_sliding_fraction.is_finite()
            && self.max_sliding_fraction.is_finite()
            && self.max_force_jump_over_load.is_finite()
            && self.max_moment_jump_over_load_radius.is_finite()
            && self.max_budget_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 8
            && self.min_sliding_fraction <= 0.25
            && self.max_sliding_fraction >= 0.55
            && self.max_force_jump_over_load <= 0.30
            && self.max_moment_jump_over_load_radius <= 0.16
            && self.max_budget_ratio <= 1.05
            && self.force_sign_stable
            && self.moment_sign_stable
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "adhesion_sliding_continuity_has_samples",
                "adhesion/sliding continuity sweep should include pre-cap, transition, and saturated samples",
                "sample_count",
                self.sample_count as f64,
                8.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "adhesion_sliding_continuity_outputs_finite",
                "adhesion/sliding continuity sweep should keep all brush outputs finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "adhesion_sliding_starts_with_adhesion",
                "the sweep should begin with mostly adhesive contact cells",
                "min_sliding_fraction",
                self.min_sliding_fraction,
                0.25,
            ),
            TireRigAcceptanceCheck::at_least(
                "adhesion_sliding_reaches_sliding",
                "the sweep should reach mostly sliding contact cells near saturation",
                "max_sliding_fraction",
                self.max_sliding_fraction,
                0.55,
            ),
            TireRigAcceptanceCheck::at_most(
                "adhesion_sliding_force_continuous",
                "adjacent force samples should not teleport at the adhesion/sliding transition",
                "max_force_jump_over_load",
                self.max_force_jump_over_load,
                0.30,
            ),
            TireRigAcceptanceCheck::at_most(
                "adhesion_sliding_moment_continuous",
                "aligning moment should change continuously through the same transition",
                "max_moment_jump_over_load_radius",
                self.max_moment_jump_over_load_radius,
                0.16,
            ),
            TireRigAcceptanceCheck::at_most(
                "adhesion_sliding_budget_bounded",
                "adhesion/sliding sweep should stay inside the shared friction budget",
                "max_budget_ratio",
                self.max_budget_ratio,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "adhesion_sliding_force_sign_stable",
                "force sign should not flip while the commanded slip direction stays fixed",
                "force_sign_stable",
                bool_metric(self.force_sign_stable),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "adhesion_sliding_moment_sign_stable",
                "aligning moment sign should not flip while lateral slip direction stays fixed",
                "moment_sign_stable",
                bool_metric(self.moment_sign_stable),
                1.0,
            ),
        ]
    }
}

/// Data-free guard for the layered tread/belt/carcass transient model.
///
/// This makes the C-stage "force does not teleport" requirement executable in
/// the standard non-calibration runner: tread should react fastest, belt next,
/// carcass slowest; the aggregate effective slip should sit between the fast
/// and slow layers; hot/soft/overloaded tires should respond more slowly; and
/// contact force should build over repeated samples instead of appearing fully
/// formed on the first frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireLayeredTransientEnvelopeReport {
    pub sample_count: usize,
    pub finite_outputs: bool,
    pub tread_over_belt_kappa: f64,
    pub belt_over_carcass_kappa: f64,
    pub tread_over_belt_alpha: f64,
    pub belt_over_carcass_alpha: f64,
    pub aggregate_kappa_between_layers: bool,
    pub aggregate_alpha_between_layers: bool,
    pub hot_kappa_over_cold: f64,
    pub hot_alpha_over_cold: f64,
    pub final_over_first_force: f64,
    pub max_abs_effective_slip: f64,
}

impl TireLayeredTransientEnvelopeReport {
    pub fn is_finite(self) -> bool {
        self.tread_over_belt_kappa.is_finite()
            && self.belt_over_carcass_kappa.is_finite()
            && self.tread_over_belt_alpha.is_finite()
            && self.belt_over_carcass_alpha.is_finite()
            && self.hot_kappa_over_cold.is_finite()
            && self.hot_alpha_over_cold.is_finite()
            && self.final_over_first_force.is_finite()
            && self.max_abs_effective_slip.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 4
            && self.tread_over_belt_kappa >= 1.05
            && self.belt_over_carcass_kappa >= 1.05
            && self.tread_over_belt_alpha >= 1.05
            && self.belt_over_carcass_alpha >= 1.05
            && self.aggregate_kappa_between_layers
            && self.aggregate_alpha_between_layers
            && self.hot_kappa_over_cold <= 0.85
            && self.hot_alpha_over_cold <= 0.85
            && self.final_over_first_force >= 1.05
            && self.max_abs_effective_slip <= 1.5
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "layered_transient_outputs_finite",
                "layered transient replay should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_has_samples",
                "layered transient report should include first, repeated, cold, and hot samples",
                "sample_count",
                self.sample_count as f64,
                4.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_tread_faster_than_belt_kappa",
                "tread longitudinal slip should react faster than belt slip",
                "tread_over_belt_kappa",
                self.tread_over_belt_kappa,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_belt_faster_than_carcass_kappa",
                "belt longitudinal slip should react faster than carcass slip",
                "belt_over_carcass_kappa",
                self.belt_over_carcass_kappa,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_tread_faster_than_belt_alpha",
                "tread slip angle should react faster than belt slip angle",
                "tread_over_belt_alpha",
                self.tread_over_belt_alpha,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_belt_faster_than_carcass_alpha",
                "belt slip angle should react faster than carcass slip angle",
                "belt_over_carcass_alpha",
                self.belt_over_carcass_alpha,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_aggregate_kappa_between_layers",
                "aggregate longitudinal slip should sit between fast tread and slow carcass layers",
                "aggregate_kappa_between_layers",
                bool_metric(self.aggregate_kappa_between_layers),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_aggregate_alpha_between_layers",
                "aggregate slip angle should sit between fast tread and slow carcass layers",
                "aggregate_alpha_between_layers",
                bool_metric(self.aggregate_alpha_between_layers),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "layered_transient_hot_overloaded_slower_kappa",
                "hot overloaded soft tire should build longitudinal slip more slowly than a cold stiff tire",
                "hot_kappa_over_cold",
                self.hot_kappa_over_cold,
                0.85,
            ),
            TireRigAcceptanceCheck::at_most(
                "layered_transient_hot_overloaded_slower_alpha",
                "hot overloaded soft tire should build slip angle more slowly than a cold stiff tire",
                "hot_alpha_over_cold",
                self.hot_alpha_over_cold,
                0.85,
            ),
            TireRigAcceptanceCheck::at_least(
                "layered_transient_force_builds_over_time",
                "layered transient contact force should build over repeated samples instead of teleporting on the first frame",
                "final_over_first_force",
                self.final_over_first_force,
                1.05,
            ),
            TireRigAcceptanceCheck::at_most(
                "layered_transient_effective_slip_bounded",
                "layered transient effective slip should remain bounded by the low-speed regularized model",
                "max_abs_effective_slip",
                self.max_abs_effective_slip,
                1.5,
            ),
        ]
    }
}

/// Data-free low-speed robustness guard for launch/lockup/reversal cases.
///
/// Realistic tire models are especially easy to break near zero speed because
/// slip ratio and slip angle have singular textbook definitions there. This
/// report exercises the native brush + transient path at tiny rolling speeds
/// and checks only broad sanity: finite outputs, bounded regularized slips,
/// one shared friction budget, and no force teleport when direction reverses.
#[derive(Clone, Copy, Debug)]
pub struct TireLowSpeedRobustnessReport {
    pub sample_count: usize,
    pub finite_outputs: bool,
    pub max_force_over_normal: f64,
    pub max_abs_raw_slip_ratio: f64,
    pub max_abs_effective_slip_ratio: f64,
    pub max_abs_effective_slip_angle_rad: f64,
    pub max_friction_budget_overrun_n: f64,
    pub max_reversal_force_jump_over_normal: f64,
}

impl TireLowSpeedRobustnessReport {
    pub fn is_finite(self) -> bool {
        self.max_force_over_normal.is_finite()
            && self.max_abs_raw_slip_ratio.is_finite()
            && self.max_abs_effective_slip_ratio.is_finite()
            && self.max_abs_effective_slip_angle_rad.is_finite()
            && self.max_friction_budget_overrun_n.is_finite()
            && self.max_reversal_force_jump_over_normal.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.sample_count >= 8
            && self.finite_outputs
            && self.max_force_over_normal <= 1.25
            && self.max_abs_raw_slip_ratio <= 1.5
            && self.max_abs_effective_slip_ratio <= 1.5
            && self.max_abs_effective_slip_angle_rad <= 0.9
            && self.max_friction_budget_overrun_n <= 1.0e-3
            && self.max_reversal_force_jump_over_normal <= 0.85
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "low_speed_has_samples",
                "low-speed robustness report should exercise launch, lockup, and reversal samples",
                "sample_count",
                self.sample_count as f64,
                8.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "low_speed_outputs_finite",
                "low-speed brush/transient outputs must stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_force_bounded",
                "low-speed contact force should stay bounded by the regularized normal load envelope",
                "max_force_over_normal",
                self.max_force_over_normal,
                1.25,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_raw_slip_regularized",
                "near-zero raw slip ratio should be regularized by the tire preset relaxation speed",
                "max_abs_raw_slip_ratio",
                self.max_abs_raw_slip_ratio,
                1.5,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_effective_slip_bounded",
                "transient effective slip should stay bounded through launch and lockup",
                "max_abs_effective_slip_ratio",
                self.max_abs_effective_slip_ratio,
                1.5,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_effective_angle_bounded",
                "transient effective slip angle should stay bounded near zero speed",
                "max_abs_effective_slip_angle_rad",
                self.max_abs_effective_slip_angle_rad,
                0.9,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_friction_budget_respected",
                "low-speed combined force should not overrun the shared friction budget",
                "max_friction_budget_overrun_n",
                self.max_friction_budget_overrun_n,
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_most(
                "low_speed_reversal_continuity",
                "force should not teleport when low-speed rolling direction reverses",
                "max_reversal_force_jump_over_normal",
                self.max_reversal_force_jump_over_normal,
                0.85,
            ),
        ]
    }
}

/// Data-free audit for the native tire frame and force/moment sign conventions.
///
/// This turns the conventions used by `RealWheelState`, `RealWheelOrientation`
/// and brush contact into executable checks:
///
/// - `heading` is the positive rolling/longitudinal axis;
/// - `terrain.normal` is the positive support axis;
/// - `side = heading × normal` is the positive lateral axis;
/// - positive `omega` rolls forward along `heading`;
/// - positive drive slip gives positive longitudinal force;
/// - positive lateral hub velocity produces force opposite that side-slip;
/// - positive camber is `wheel_up` tilted toward positive `side`;
/// - pneumatic aligning moment follows `Mz = -Fy * trail`;
/// - moving-ground work sign is checked by `TireEnergySignAudit`.
#[derive(Clone, Copy, Debug)]
pub struct TireFrameConventionReport {
    pub sample_count: usize,
    pub basis_is_right_handed: bool,
    pub positive_drive_slip_fx_positive: bool,
    pub brake_slip_fx_negative: bool,
    pub lateral_velocity_force_opposes_slip: bool,
    pub positive_camber_force_sign: bool,
    pub pneumatic_trail_moment_sign: bool,
    pub moving_ground_work_sign: bool,
    pub max_friction_budget_overrun_n: f64,
}

impl TireFrameConventionReport {
    pub fn is_finite(self) -> bool {
        self.max_friction_budget_overrun_n.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.sample_count >= 5
            && self.basis_is_right_handed
            && self.positive_drive_slip_fx_positive
            && self.brake_slip_fx_negative
            && self.lateral_velocity_force_opposes_slip
            && self.positive_camber_force_sign
            && self.pneumatic_trail_moment_sign
            && self.moving_ground_work_sign
            && self.max_friction_budget_overrun_n <= 1.0e-3
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "frame_convention_has_samples",
                "frame/sign convention audit should exercise drive, brake, side-slip, camber, and moving-ground cases",
                "sample_count",
                self.sample_count as f64,
                5.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_basis_heading_cross_normal",
                "positive lateral axis should be heading cross terrain normal",
                "basis_is_right_handed",
                bool_metric(self.basis_is_right_handed),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_drive_slip_force_sign",
                "positive wheel-speed minus road-speed slip should push along heading",
                "positive_drive_slip_fx_positive",
                bool_metric(self.positive_drive_slip_fx_positive),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_brake_slip_force_sign",
                "negative wheel-speed minus road-speed slip should brake against heading",
                "brake_slip_fx_negative",
                bool_metric(self.brake_slip_fx_negative),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_lateral_slip_force_sign",
                "positive lateral hub velocity should generate lateral force opposing that side-slip",
                "lateral_velocity_force_opposes_slip",
                bool_metric(self.lateral_velocity_force_opposes_slip),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_positive_camber_force_sign",
                "positive camber should use the documented wheel_up toward side convention",
                "positive_camber_force_sign",
                bool_metric(self.positive_camber_force_sign),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_pneumatic_trail_moment_sign",
                "aligning moment should follow Mz = Fy * pneumatic_trail (y-up frame)",
                "pneumatic_trail_moment_sign",
                bool_metric(self.pneumatic_trail_moment_sign),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "frame_moving_ground_work_sign",
                "moving terrain work sign should match the tire-rig energy/sign audit",
                "moving_ground_work_sign",
                bool_metric(self.moving_ground_work_sign),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "frame_friction_budget_respected",
                "frame/sign convention samples should still respect the shared friction budget",
                "max_friction_budget_overrun_n",
                self.max_friction_budget_overrun_n,
                1.0e-3,
            ),
        ]
    }
}

/// Data-free contact-patch convergence report for the sampled brush patch.
///
/// This does not claim high-resolution tread-block accuracy. It records how
/// the native patch/pressure/force accounting changes as the circumferential
/// contact discretization is refined, so coarse viewer settings have a
/// regression signal before higher-density or GPU atlas work changes defaults.
#[derive(Clone, Copy, Debug)]
pub struct TireContactPatchConvergenceReport {
    pub sample_count: usize,
    pub min_segments: usize,
    pub max_segments: usize,
    pub finite_outputs: bool,
    pub contact_count_increases: bool,
    pub max_normal_load_error_ratio: f64,
    pub patch_length_delta_over_radius: f64,
    pub force_delta_over_load: f64,
    pub aligning_delta_over_load_radius: f64,
    pub peak_pressure_relative_delta: f64,
    pub max_pressure_bin_load_error_ratio: f64,
}

impl TireContactPatchConvergenceReport {
    pub fn is_finite(self) -> bool {
        self.max_normal_load_error_ratio.is_finite()
            && self.patch_length_delta_over_radius.is_finite()
            && self.force_delta_over_load.is_finite()
            && self.aligning_delta_over_load_radius.is_finite()
            && self.peak_pressure_relative_delta.is_finite()
            && self.max_pressure_bin_load_error_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.sample_count >= 4
            && self.finite_outputs
            && self.contact_count_increases
            && self.max_normal_load_error_ratio <= 0.08
            && self.patch_length_delta_over_radius <= 0.20
            && self.force_delta_over_load <= 0.18
            && self.aligning_delta_over_load_radius <= 0.12
            && self.peak_pressure_relative_delta <= 0.35
            && self.max_pressure_bin_load_error_ratio <= 1.0e-8
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "patch_convergence_has_samples",
                "contact-patch convergence should compare multiple circumferential discretizations",
                "sample_count",
                self.sample_count as f64,
                4.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "patch_convergence_outputs_finite",
                "patch convergence samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "patch_convergence_contact_count_refines",
                "refining tire segments should not reduce active contact-cell count",
                "contact_count_increases",
                bool_metric(self.contact_count_increases),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_load_error_bounded",
                "resolved normal load should remain near the requested rig load",
                "max_normal_load_error_ratio",
                self.max_normal_load_error_ratio,
                0.08,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_patch_length_stable",
                "patch length should not move wildly between the two finest segment counts",
                "patch_length_delta_over_radius",
                self.patch_length_delta_over_radius,
                0.20,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_force_stable",
                "world force should be broadly stable between the two finest segment counts",
                "force_delta_over_load",
                self.force_delta_over_load,
                0.18,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_aligning_moment_stable",
                "aligning moment should be broadly stable between the two finest segment counts",
                "aligning_delta_over_load_radius",
                self.aligning_delta_over_load_radius,
                0.12,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_peak_pressure_stable",
                "peak contact pressure should be broadly stable between the two finest segment counts",
                "peak_pressure_relative_delta",
                self.peak_pressure_relative_delta,
                0.35,
            ),
            TireRigAcceptanceCheck::at_most(
                "patch_convergence_pressure_bins_conserve_load",
                "pressure-distribution bins should conserve normal load in every convergence sample",
                "max_pressure_bin_load_error_ratio",
                self.max_pressure_bin_load_error_ratio,
                1.0e-8,
            ),
        ]
    }
}

/// Data-free road-condition state report.
///
/// This checks that "road condition" is not only a string label: water film,
/// ice friction, moving ground, thermal/wear state, and soft-soil material
/// coefficients all flow through native tire/contact state and telemetry.
#[derive(Clone, Copy, Debug)]
pub struct TireRoadConditionStateReport {
    pub sample_count: usize,
    pub finite_outputs: bool,
    pub wet_film_reduces_mu: bool,
    pub wet_film_reduces_force: bool,
    pub standing_water_adds_rolling_loss: bool,
    pub ice_caps_force: bool,
    pub moving_ground_velocity_recorded: bool,
    pub hot_worn_condition_reduces_mu: bool,
    pub worn_tread_worse_on_wet_film: bool,
    pub soft_soil_adds_rolling_loss: bool,
}

impl TireRoadConditionStateReport {
    pub fn is_finite(self) -> bool {
        true
    }

    pub fn passed(self) -> bool {
        self.sample_count >= 8
            && self.finite_outputs
            && self.wet_film_reduces_mu
            && self.wet_film_reduces_force
            && self.standing_water_adds_rolling_loss
            && self.ice_caps_force
            && self.moving_ground_velocity_recorded
            && self.hot_worn_condition_reduces_mu
            && self.worn_tread_worse_on_wet_film
            && self.soft_soil_adds_rolling_loss
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "road_condition_has_samples",
                "road-condition state report should exercise wet, ice, moving-ground, thermal/wear, and soil cases",
                "sample_count",
                self.sample_count as f64,
                8.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_outputs_finite",
                "road-condition state samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_wet_film_reduces_mu",
                "water-film state should reduce material shear without a viewer-only grip switch",
                "wet_film_reduces_mu",
                bool_metric(self.wet_film_reduces_mu),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_wet_film_reduces_force",
                "terrain water-film state should reduce tire force through the contact API",
                "wet_film_reduces_force",
                bool_metric(self.wet_film_reduces_force),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_standing_water_adds_rolling_loss",
                "standing water should add rolling/water-displacement loss",
                "standing_water_adds_rolling_loss",
                bool_metric(self.standing_water_adds_rolling_loss),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_ice_caps_force",
                "ice should cap usable tire force far below dry asphalt",
                "ice_caps_force",
                bool_metric(self.ice_caps_force),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_moving_ground_velocity_recorded",
                "moving-ground state should be exported in reusable telemetry",
                "moving_ground_velocity_recorded",
                bool_metric(self.moving_ground_velocity_recorded),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_hot_worn_state_reduces_mu",
                "thermal/wear condition should reduce live material μ from state",
                "hot_worn_condition_reduces_mu",
                bool_metric(self.hot_worn_condition_reduces_mu),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_worn_tread_worse_on_wet_film",
                "worn tread should lose more wet-film grip than fresh tread",
                "worn_tread_worse_on_wet_film",
                bool_metric(self.worn_tread_worse_on_wet_film),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "road_condition_soft_soil_adds_rolling_loss",
                "soft-soil material state should add rolling loss compared with dry asphalt",
                "soft_soil_adds_rolling_loss",
                bool_metric(self.soft_soil_adds_rolling_loss),
                1.0,
            ),
        ]
    }
}

pub fn tire_road_condition_state_report() -> TireRoadConditionStateReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let dry = SurfaceMaterial::dry_asphalt();
    let ice = SurfaceMaterial::ice();
    let soil = SurfaceMaterial::soft_soil();
    let load_n = 3_800.0;
    let speed_m_per_s = 14.0;
    let slip = 0.12;
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let dry_point = tire_rig_point(
        &params,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let wet_material =
        SurfaceFilmState::wet_road().apply_to(dry, speed_m_per_s, &TireWearState::default());
    let standing_material =
        SurfaceFilmState::standing_water().apply_to(dry, speed_m_per_s, &TireWearState::default());
    let wet_terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        dry,
        vec![
            TireRigSurfaceTransitionBand::new("wet", -1.0, 1.0, dry)
                .with_water_film_depth(SurfaceFilmState::wet_road().water_depth_m),
        ],
    );
    let wet_point = tire_rig_point(
        &params,
        &wet_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let standing_point = tire_rig_point(
        &params,
        &FlatTerrain::new(0.0, standing_material.mu_static),
        &standing_material,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let ice_point = tire_rig_point(
        &params,
        &FlatTerrain::new(0.0, ice.mu_static),
        &ice,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let moving_terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        dry,
        vec![
            TireRigSurfaceTransitionBand::new("moving", -1.0, 1.0, dry)
                .with_surface_velocity([3.0, 0.0, 0.0]),
        ],
    );
    let moving_point = tire_rig_point(
        &params,
        &moving_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        0.0,
        0.0,
        0.0,
    );
    let moving_snapshot = moving_point.snapshot_on_terrain(
        "road_condition_moving_ground",
        "passenger_radial",
        "dry_asphalt",
        "moving_flat_belt",
        "cpu_brush_road_condition",
        &moving_terrain,
        0.0,
        0.0,
    );
    let soft_soil_point = tire_rig_point(
        &params,
        &FlatTerrain::new(0.0, soil.mu_static),
        &soil,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.95;
    let fresh_wet =
        SurfaceFilmState::wet_road().apply_to(dry, speed_m_per_s, &TireWearState::default());
    let worn_wet = SurfaceFilmState::wet_road().apply_to(dry, speed_m_per_s, &worn);
    let hot_worn_thermal = TireThermalLayers {
        surface_c: 180.0,
        tread_c: 170.0,
        carcass_c: 150.0,
        rim_c: 105.0,
        ..TireThermalLayers::passenger_road()
    };
    let hot_worn_response =
        TireConditionMaterialResponse::from_state(&params, &dry, &hot_worn_thermal, &worn);

    let finite_outputs = dry_point.is_finite()
        && wet_point.is_finite()
        && standing_point.is_finite()
        && ice_point.is_finite()
        && moving_point.is_finite()
        && soft_soil_point.is_finite()
        && hot_worn_response.is_finite();
    TireRoadConditionStateReport {
        sample_count: 8,
        finite_outputs,
        wet_film_reduces_mu: wet_material.mu_static < dry.mu_static
            && wet_material.mu_dynamic < dry.mu_dynamic,
        wet_film_reduces_force: wet_point.telemetry.longitudinal_force_n.abs()
            < dry_point.telemetry.longitudinal_force_n.abs() * 0.92,
        standing_water_adds_rolling_loss: standing_material.rolling_resistance
            > wet_material.rolling_resistance
            && standing_point.telemetry.rolling_resistance_force_n.abs()
                > dry_point.telemetry.rolling_resistance_force_n.abs(),
        ice_caps_force: ice_point.telemetry.longitudinal_force_n.abs()
            < dry_point.telemetry.longitudinal_force_n.abs() * 0.35,
        moving_ground_velocity_recorded: moving_snapshot.terrain_surface_velocity_m_per_s[0].abs()
            > 2.5
            && moving_snapshot.moving_ground_power_w.abs() > 1.0,
        hot_worn_condition_reduces_mu: hot_worn_response.material.mu_static < dry.mu_static * 0.85
            && hot_worn_response.overheated_grip_loss
            && hot_worn_response.worn_tread_limited,
        worn_tread_worse_on_wet_film: worn_wet.mu_static < fresh_wet.mu_static * 0.90,
        soft_soil_adds_rolling_loss: soft_soil_point.telemetry.rolling_resistance_force_n.abs()
            > dry_point.telemetry.rolling_resistance_force_n.abs() * 2.0,
    }
}
/// Data-free tire force-shape diagnostics.
///
/// This is explicitly **not** a Pacejka/TMeasy fitting result. It checks only
/// qualitative reference-shape properties that racing-sim and tire-rig
/// workflows care about: smooth force buildup, saturation, combined-slip
/// tradeoff, friction-budget accounting, and aligning-moment sign.

pub fn tire_frame_convention_report() -> TireFrameConventionReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 3_800.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let heading = [1.0, 0.0, 0.0];
    let normal = terrain.normal(0.0, 0.0);
    let side = normalize(cross(heading, normal));
    let basis_is_right_handed =
        dot(side, [0.0, 0.0, 1.0]) > 0.999 && dot(cross(normal, side), heading) > 0.999;
    let base_state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [6.0, 0.0, 0.0],
        heading,
        omega: 6.0 / params.radius,
    };
    let drive = evaluate_brush_tire(
        &params,
        &RealWheelState {
            omega: 7.2 / params.radius,
            ..base_state
        },
        &terrain,
        &material,
    );
    let brake = evaluate_brush_tire(
        &params,
        &RealWheelState {
            omega: 4.8 / params.radius,
            ..base_state
        },
        &terrain,
        &material,
    );
    let lateral = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_vel: [6.0, 0.0, 0.8],
            ..base_state
        },
        &terrain,
        &material,
    );
    let camber = evaluate_brush_tire_oriented(
        &params,
        &base_state,
        &terrain,
        &material,
        &RealWheelOrientation {
            wheel_up: normalize([0.0, 1.0, 0.18]),
        },
    );
    let trail_case = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_vel: [6.0, 0.0, 1.2],
            omega: 6.2 / params.radius,
            ..base_state
        },
        &terrain,
        &material,
    );
    // In this y-up frame with `side = heading × normal`, the aligning
    // moment about the up-axis is `Mz = Fy · trail` for positive trail
    // (force resultant behind the contact center). This is the same
    // restoring torque the ISO z-up convention writes as `Mz = -Fy · t`;
    // the sign flips with the frame's side-vector orientation.
    let expected_mz = trail_case.lateral_force * trail_case.pneumatic_trail_m;
    let pneumatic_trail_moment_sign = expected_mz.abs() <= 1.0e-6
        || trail_case.aligning_moment_z.abs() <= 1.0e-6
        || expected_mz.signum() == trail_case.aligning_moment_z.signum();
    let moving_terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        material,
        vec![
            TireRigSurfaceTransitionBand::new("moving", -1.0, 1.0, material)
                .with_surface_velocity([2.0, 0.0, 0.0]),
        ],
    );
    let moving = tire_rig_point(
        &params,
        &moving_terrain,
        &material,
        load_n,
        3.0,
        0.0,
        0.0,
        0.0,
    );
    let moving_snapshot = moving.snapshot_on_terrain(
        "frame_convention_moving_ground",
        "passenger_radial",
        "dry_asphalt",
        "moving_flat_belt",
        "cpu_brush_frame_convention",
        &moving_terrain,
        0.0,
        0.0,
    );
    let moving_ground_work_sign = moving_snapshot
        .energy_sign_audit()
        .moving_ground_work_sign_ok;
    let mut max_friction_budget_overrun_n = 0.0_f64;
    for contact in [&drive, &brake, &lateral, &camber, &trail_case] {
        max_friction_budget_overrun_n = max_friction_budget_overrun_n
            .max((contact.consumed_friction_budget - contact.available_friction_budget).max(0.0));
    }
    max_friction_budget_overrun_n = max_friction_budget_overrun_n.max(
        (moving.telemetry.consumed_friction_budget_n
            - moving.telemetry.available_friction_budget_n)
            .max(0.0),
    );

    TireFrameConventionReport {
        sample_count: 6,
        basis_is_right_handed,
        positive_drive_slip_fx_positive: drive.longitudinal_force > 1.0,
        brake_slip_fx_negative: brake.longitudinal_force < -1.0,
        lateral_velocity_force_opposes_slip: lateral.lateral_force < -1.0,
        positive_camber_force_sign: camber.camber_angle_rad > 0.0 && camber.lateral_force < -1.0,
        pneumatic_trail_moment_sign,
        moving_ground_work_sign,
        max_friction_budget_overrun_n,
    }
}

pub fn tire_layered_transient_envelope_report() -> TireLayeredTransientEnvelopeReport {
    let dry = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, dry.mu_static);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 190_000.0)
        .with_segments(96);
    let load_n = 4_500.0;
    let speed = 10.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [speed, 0.0, 0.7],
        heading: [1.0, 0.0, 0.0],
        omega: speed * 1.06 / params.radius.max(1.0e-9),
    };
    let context = TireRelaxationContext::new(load_n, 70.0);
    let mut layered = TireLayeredTransientState::default();
    let first = evaluate_brush_tire_layered_transient_with_context(
        &params,
        &state,
        &terrain,
        &dry,
        &RealWheelOrientation::default(),
        &mut layered,
        1.0 / 60.0,
        context,
    );
    let first_layer = layered;
    let mut last = first.clone();
    let mut sample_count = 1usize;
    let mut finite_outputs = first.is_finite() && first_layer.is_finite();
    let mut max_abs_effective_slip = first_layer
        .kappa_eff
        .abs()
        .max(first_layer.alpha_eff_rad.abs());
    for _ in 0..59 {
        last = evaluate_brush_tire_layered_transient_with_context(
            &params,
            &state,
            &terrain,
            &dry,
            &RealWheelOrientation::default(),
            &mut layered,
            1.0 / 60.0,
            context,
        );
        sample_count += 1;
        finite_outputs &= last.is_finite() && layered.is_finite();
        max_abs_effective_slip = max_abs_effective_slip
            .max(layered.kappa_eff.abs())
            .max(layered.alpha_eff_rad.abs());
    }
    let layer_abs = |v: f64| v.abs().max(1.0e-9);
    let tread_over_belt_kappa =
        layer_abs(first_layer.tread.kappa_eff) / layer_abs(first_layer.belt.kappa_eff);
    let belt_over_carcass_kappa =
        layer_abs(first_layer.belt.kappa_eff) / layer_abs(first_layer.carcass.kappa_eff);
    let tread_over_belt_alpha =
        layer_abs(first_layer.tread.alpha_eff_rad) / layer_abs(first_layer.belt.alpha_eff_rad);
    let belt_over_carcass_alpha =
        layer_abs(first_layer.belt.alpha_eff_rad) / layer_abs(first_layer.carcass.alpha_eff_rad);
    let aggregate_kappa_between_layers = {
        let agg = layer_abs(first_layer.kappa_eff);
        agg < layer_abs(first_layer.tread.kappa_eff)
            && agg > layer_abs(first_layer.carcass.kappa_eff)
    };
    let aggregate_alpha_between_layers = {
        let agg = layer_abs(first_layer.alpha_eff_rad);
        agg < layer_abs(first_layer.tread.alpha_eff_rad)
            && agg > layer_abs(first_layer.carcass.alpha_eff_rad)
    };

    let cold_params = RealTireParams::high_pressure_reference()
        .with_geometry_and_pressure(0.32, 0.185, 280_000.0)
        .with_segments(96);
    let hot_params = RealTireParams::offroad_soft()
        .with_geometry_and_pressure(0.32, 0.205, 95_000.0)
        .with_segments(96);
    let cold_hub_y = solve_loaded_tire_hub_height(&cold_params, &terrain, 0.0, 0.0, 2_500.0);
    let hot_hub_y = solve_loaded_tire_hub_height(&hot_params, &terrain, 0.0, 0.0, 6_500.0);
    let cold_state = RealWheelState {
        hub_pos: [0.0, cold_hub_y, 0.0],
        hub_vel: [speed, 0.0, 2.0],
        heading: [1.0, 0.0, 0.0],
        omega: speed * 1.30 / cold_params.radius.max(1.0e-9),
    };
    let hot_state = RealWheelState {
        hub_pos: [0.0, hot_hub_y, 0.0],
        hub_vel: [speed, 0.0, 2.0],
        heading: [1.0, 0.0, 0.0],
        omega: speed * 1.30 / hot_params.radius.max(1.0e-9),
    };
    let mut cold = TireLayeredTransientState::default();
    let mut hot = TireLayeredTransientState::default();
    cold.step_with_context(
        &cold_params,
        &cold_state,
        [0.0, 1.0, 0.0],
        1.0 / 60.0,
        TireRelaxationContext::new(2_500.0, 20.0),
    );
    hot.step_with_context(
        &hot_params,
        &hot_state,
        [0.0, 1.0, 0.0],
        1.0 / 60.0,
        TireRelaxationContext::new(6_500.0, 175.0),
    );
    sample_count += 2;
    finite_outputs &= cold.is_finite() && hot.is_finite();

    let force_mag = |c: RealTireContact| {
        (c.longitudinal_force * c.longitudinal_force + c.lateral_force * c.lateral_force).sqrt()
    };
    TireLayeredTransientEnvelopeReport {
        sample_count,
        finite_outputs,
        tread_over_belt_kappa,
        belt_over_carcass_kappa,
        tread_over_belt_alpha,
        belt_over_carcass_alpha,
        aggregate_kappa_between_layers,
        aggregate_alpha_between_layers,
        hot_kappa_over_cold: hot.kappa_eff.abs() / cold.kappa_eff.abs().max(1.0e-9),
        hot_alpha_over_cold: hot.alpha_eff_rad.abs() / cold.alpha_eff_rad.abs().max(1.0e-9),
        final_over_first_force: force_mag(last) / force_mag(first).max(1.0),
        max_abs_effective_slip,
    }
}

pub fn tire_low_speed_robustness_report() -> TireLowSpeedRobustnessReport {
    let params = RealTireParams {
        pressure_pa: 150_000.0,
        relaxation_speed: 0.32,
        ..RealTireParams::passenger_radial().with_segments(64)
    };
    let material = SurfaceMaterial::ice();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 3_200.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let mut transient = RealTireTransientState {
        relaxation_length_longitudinal: 0.45,
        relaxation_length_lateral: 0.55,
        tread_yaw_relaxation_length: 0.65,
        ..Default::default()
    };
    let mut sample_count = 0usize;
    let mut finite_outputs = true;
    let mut max_force_over_normal = 0.0_f64;
    let mut max_abs_raw_slip_ratio = 0.0_f64;
    let mut max_abs_effective_slip_ratio = 0.0_f64;
    let mut max_abs_effective_slip_angle_rad = 0.0_f64;
    let mut max_friction_budget_overrun_n = 0.0_f64;
    let mut max_reversal_force_jump_over_normal = 0.0_f64;
    let mut previous_force: Option<[f64; 3]> = None;

    let samples = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0e-4, 0.0, 1.0e-4, 0.20),
        (1.0e-4, 0.0, -1.0e-4, -0.20),
        (0.02, 0.0, 0.01, 0.0),
        (0.02, 0.0, -0.01, -0.12),
        (-0.02, 0.0, 0.01, 0.12),
        (-1.0e-4, 0.0, -1.0e-4, -0.20),
        (0.0, 0.0, 0.0, 0.0),
    ];

    for (vx, vy, vz, omega) in samples {
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [vx, vy, vz],
            heading: [1.0, 0.0, 0.0],
            omega,
        };
        let out = evaluate_brush_tire_transient(
            &params,
            &state,
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            1.0 / 240.0,
        );
        sample_count += 1;
        let force_mag = dot(out.force, out.force).sqrt();
        let normal = out.normal_load.max(1.0);
        let force_over_normal = force_mag / normal;
        let budget_overrun = (out.consumed_friction_budget - out.available_friction_budget)
            .max(0.0)
            .max(out.consumed_friction_budget - out.available_friction_budget - 1.0e-9);
        finite_outputs &= out.force.iter().all(|v| v.is_finite())
            && out.slip_ratio.is_finite()
            && out.slip_angle_rad.is_finite()
            && out.available_friction_budget.is_finite()
            && out.consumed_friction_budget.is_finite()
            && transient.kappa_eff.is_finite()
            && transient.alpha_eff.is_finite()
            && transient.tread_yaw.is_finite();
        max_force_over_normal = max_force_over_normal.max(force_over_normal);
        max_abs_raw_slip_ratio = max_abs_raw_slip_ratio.max(out.slip_ratio.abs());
        max_abs_effective_slip_ratio = max_abs_effective_slip_ratio.max(transient.kappa_eff.abs());
        max_abs_effective_slip_angle_rad =
            max_abs_effective_slip_angle_rad.max(transient.alpha_eff.abs());
        max_friction_budget_overrun_n = max_friction_budget_overrun_n.max(budget_overrun);
        if let Some(prev) = previous_force {
            let jump = [
                out.force[0] - prev[0],
                out.force[1] - prev[1],
                out.force[2] - prev[2],
            ];
            max_reversal_force_jump_over_normal =
                max_reversal_force_jump_over_normal.max(dot(jump, jump).sqrt() / normal);
        }
        previous_force = Some(out.force);
    }

    TireLowSpeedRobustnessReport {
        sample_count,
        finite_outputs,
        max_force_over_normal,
        max_abs_raw_slip_ratio,
        max_abs_effective_slip_ratio,
        max_abs_effective_slip_angle_rad,
        max_friction_budget_overrun_n,
        max_reversal_force_jump_over_normal,
    }
}

/// Qualitative longitudinal reference-shape check for a pure-slip sweep.
pub fn tire_rig_longitudinal_reference_shape_diagnostics(
    scenario_id: &str,
    points: &[TireRigSweepPoint],
) -> TireReferenceShapeDiagnostics {
    let mut samples = points
        .iter()
        .filter(|p| p.is_finite())
        .map(|p| {
            (
                finite_or(p.slip_ratio_command.abs(), 0.0),
                finite_or(p.telemetry.longitudinal_force_n.abs(), 0.0),
                finite_nonnegative(p.telemetry.available_friction_budget_n),
                finite_nonnegative(p.telemetry.consumed_friction_budget_n),
            )
        })
        .collect::<Vec<_>>();
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sample_count = samples.len();
    let peak = samples.iter().map(|s| s.1).fold(0.0_f64, f64::max);
    let denom = peak.max(1.0);
    let mut max_jump: f64 = 0.0;
    let mut monotonic_buildup = sample_count >= 2;
    let mut prev_force = samples.first().map(|s| s.1).unwrap_or(0.0);
    for &(_, force, _, _) in samples.iter().skip(1) {
        max_jump = max_jump.max((force - prev_force).abs() / denom);
        if prev_force < 0.80 * peak && force + 0.08 * denom < prev_force {
            monotonic_buildup = false;
        }
        prev_force = force;
    }
    let first_slope = samples
        .windows(2)
        .find_map(|w| {
            let ds = w[1].0 - w[0].0;
            (ds.abs() > 1.0e-9).then_some((w[1].1 - w[0].1).abs() / ds.abs())
        })
        .unwrap_or(0.0);
    let tail_slope = samples
        .windows(2)
        .rev()
        .find_map(|w| {
            let ds = w[1].0 - w[0].0;
            (ds.abs() > 1.0e-9).then_some((w[1].1 - w[0].1).abs() / ds.abs())
        })
        .unwrap_or(0.0);
    let tail_slope_ratio = tail_slope / first_slope.max(1.0);
    let friction_budget_ok = samples
        .iter()
        .all(|&(_, _, available, consumed)| consumed <= available * 1.05 + 1.0);
    TireReferenceShapeDiagnostics {
        scenario_id: csv_token(scenario_id),
        sample_count,
        peak_longitudinal_force_n: peak,
        peak_lateral_force_n: 0.0,
        max_normalized_force_jump: max_jump,
        tail_slope_ratio,
        monotonic_buildup,
        smooth_saturation: max_jump <= 0.65 && tail_slope_ratio <= 1.75,
        combined_slip_tradeoff: true,
        aligning_moment_sign_ok: true,
        friction_budget_ok,
    }
}

/// Qualitative combined-slip reference-shape check from pure-longitudinal,
/// pure-lateral, and combined-slip points.
pub fn tire_rig_combined_reference_shape_diagnostics(
    scenario_id: &str,
    pure_longitudinal: &TireRigSweepPoint,
    pure_lateral: &TireRigSweepPoint,
    combined: &TireRigSweepPoint,
) -> TireReferenceShapeDiagnostics {
    let pure_fx = pure_longitudinal.telemetry.longitudinal_force_n.abs();
    let pure_fy = pure_lateral.telemetry.lateral_force_n.abs();
    let combined_fx = combined.telemetry.longitudinal_force_n.abs();
    let combined_fy = combined.telemetry.lateral_force_n.abs();
    let combined_slip_tradeoff = combined_fx <= pure_fx * 1.05 + 1.0
        && combined_fy <= pure_fy * 1.05 + 1.0
        && combined.telemetry.consumed_friction_budget_n
            <= combined.telemetry.available_friction_budget_n * 1.05 + 1.0;
    // y-up frame with `side = heading × normal`: the restoring aligning
    // moment is `Mz = Fy · trail` (see tire_frame_convention_report).
    let expected_mz =
        combined.telemetry.lateral_force_n * combined.telemetry.patch.pneumatic_trail_m;
    let reported_mz = combined.telemetry.aligning_moment_nm;
    let sign_eps = 1.0e-6;
    let aligning_moment_sign_ok = expected_mz.abs() <= sign_eps
        || reported_mz.abs() <= sign_eps
        || expected_mz.signum() == reported_mz.signum();
    TireReferenceShapeDiagnostics {
        scenario_id: csv_token(scenario_id),
        sample_count: usize::from(pure_longitudinal.is_finite())
            + usize::from(pure_lateral.is_finite())
            + usize::from(combined.is_finite()),
        peak_longitudinal_force_n: pure_fx.max(combined_fx),
        peak_lateral_force_n: pure_fy.max(combined_fy),
        max_normalized_force_jump: 0.0,
        tail_slope_ratio: 0.0,
        monotonic_buildup: true,
        smooth_saturation: true,
        combined_slip_tradeoff,
        aligning_moment_sign_ok,
        friction_budget_ok: combined.telemetry.consumed_friction_budget_n
            <= combined.telemetry.available_friction_budget_n * 1.05 + 1.0,
    }
}

pub fn tire_load_combined_slip_law_report() -> TireLoadCombinedSlipLawReport {
    let dry = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 220_000.0)
        .with_segments(96);
    let eval = |load: f64, speed: f64, slip_ratio: f64, lateral_speed: f64| {
        let hub_y = solve_loaded_tire_hub_height(&params, &dry_terrain, 0.0, 0.0, load);
        evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [speed, 0.0, lateral_speed],
                heading: [1.0, 0.0, 0.0],
                omega: (speed * (1.0 + slip_ratio)) / params.radius.max(1.0e-9),
            },
            &dry_terrain,
            &dry,
        )
    };

    let light = eval(2_500.0, 8.0, 1.5, 0.0);
    let heavy = eval(5_000.0, 8.0, 1.5, 0.0);
    let load_force_ratio = heavy.longitudinal_force.abs() / light.longitudinal_force.abs().max(1.0);
    let load_ratio = heavy.normal_load / light.normal_load.max(1.0);
    let heavy_stiffness_over_light =
        heavy.effective_longitudinal_stiffness / light.effective_longitudinal_stiffness.max(1.0);

    let low_mu = SurfaceMaterial::new(0.22, 0.17);
    let low_mu_terrain = FlatTerrain::new(0.0, low_mu.mu_static);
    let low_mu_eval = |pressure_pa: f64| {
        let pressure_params = RealTireParams::passenger_radial()
            .with_geometry_and_pressure(0.32, 0.185, pressure_pa)
            .with_segments(96);
        let hub_y =
            solve_loaded_tire_hub_height(&pressure_params, &low_mu_terrain, 0.0, 0.0, 4_500.0);
        evaluate_brush_tire(
            &pressure_params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [1.0, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: (1.0 + 0.02) / pressure_params.radius.max(1.0e-9),
            },
            &low_mu_terrain,
            &low_mu,
        )
    };
    let low_mu_high_pressure = low_mu_eval(420_000.0);
    let low_mu_low_pressure = low_mu_eval(90_000.0);
    let low_mu_low_pressure_over_high_pressure_fx = low_mu_low_pressure.longitudinal_force.abs()
        / low_mu_high_pressure.longitudinal_force.abs().max(1.0);
    let low_mu_low_pressure_patch_over_high_pressure =
        low_mu_low_pressure.patch_length / low_mu_high_pressure.patch_length.max(1.0e-9);

    let pure_long = eval(4_500.0, 6.0, 1.15, 0.0);
    let pure_lat = eval(4_500.0, 6.0, 0.0, 3.0);
    let combined = eval(4_500.0, 6.0, 1.15, 3.0);
    let combined_fx_over_pure_fx =
        combined.longitudinal_force.abs() / pure_long.longitudinal_force.abs().max(1.0);
    let combined_fy_over_pure_fy =
        combined.lateral_force.abs() / pure_lat.lateral_force.abs().max(1.0);
    let combined_budget_ratio =
        combined.consumed_friction_budget / combined.available_friction_budget.max(1.0);

    let saturation_material = SurfaceMaterial {
        mu_static: 0.55,
        mu_dynamic: 0.42,
        bristle_damping: 0.0,
        rolling_resistance: 0.0,
        cohesion: 0.0,
    };
    let saturation_terrain = FlatTerrain::new(0.0, saturation_material.mu_static);
    let hub_y = solve_loaded_tire_hub_height(&params, &saturation_terrain, 0.0, 0.0, 4_000.0);
    let mut prev: Option<f64> = None;
    let mut max_saturation_jump_n = 0.0_f64;
    let mut saturation_start_sliding_fraction = 0.0_f64;
    let mut saturation_end_sliding_fraction = 0.0_f64;
    let mut finite_outputs = light.in_contact
        && heavy.in_contact
        && pure_long.in_contact
        && pure_lat.in_contact
        && combined.in_contact;
    for k in 0..32 {
        let slip = k as f64 * 0.012;
        let out = evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [6.0, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: (6.0 * (1.0 + slip)) / params.radius,
            },
            &saturation_terrain,
            &saturation_material,
        );
        finite_outputs &= out.force.iter().all(|v| v.is_finite())
            && out.longitudinal_force.is_finite()
            && out.lateral_force.is_finite()
            && out.consumed_friction_budget.is_finite();
        let mag = (out.longitudinal_force * out.longitudinal_force
            + out.lateral_force * out.lateral_force)
            .sqrt();
        if let Some(p) = prev {
            max_saturation_jump_n = max_saturation_jump_n.max((mag - p).abs());
        } else {
            saturation_start_sliding_fraction = out.sliding_fraction;
        }
        saturation_end_sliding_fraction = out.sliding_fraction;
        prev = Some(mag);
    }

    let ice = SurfaceMaterial::ice();
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let dry_slip = tire_rig_point(&params, &dry_terrain, &dry, 4_500.0, 10.0, 0.22, 0.0, 0.0);
    let ice_slip = tire_rig_point(&params, &ice_terrain, &ice, 4_500.0, 10.0, 0.22, 0.0, 0.0);
    let ice_over_dry_fx = ice_slip.telemetry.longitudinal_force_n.abs()
        / dry_slip.telemetry.longitudinal_force_n.abs().max(1.0);

    let fresh = TireWearState::default();
    let healthy_thermal = TireThermalLayers {
        surface_c: 80.0,
        tread_c: 75.0,
        carcass_c: 65.0,
        ..Default::default()
    };
    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.80;
    let hot_thermal = TireThermalLayers {
        surface_c: 190.0,
        tread_c: 170.0,
        carcass_c: 145.0,
        ..Default::default()
    };
    let healthy = healthy_thermal.material_response(&fresh, &params, &dry);
    let hot = hot_thermal.material_response(&worn, &params, &dry);
    let eval_response = |response: TireConditionMaterialResponse,
                         speed: f64,
                         slip_ratio: f64,
                         lateral_speed: f64| {
        let terrain = FlatTerrain::new(0.0, response.material.mu_static);
        let hub_y = solve_loaded_tire_hub_height(&response.params, &terrain, 0.0, 0.0, 4_500.0);
        evaluate_brush_tire(
            &response.params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [speed, 0.0, lateral_speed],
                heading: [1.0, 0.0, 0.0],
                omega: (speed * (1.0 + slip_ratio)) / response.params.radius.max(1.0e-9),
            },
            &terrain,
            &response.material,
        )
    };
    let healthy_longitudinal = eval_response(healthy, 6.0, 0.04, 0.0);
    let hot_longitudinal = eval_response(hot, 6.0, 0.04, 0.0);
    let healthy_cornering = eval_response(healthy, 6.0, 0.0, 0.35);
    let hot_cornering = eval_response(hot, 6.0, 0.0, 0.35);
    finite_outputs &= dry_slip.is_finite()
        && ice_slip.is_finite()
        && healthy.is_finite()
        && hot.is_finite()
        && healthy_longitudinal.is_finite()
        && hot_longitudinal.is_finite()
        && healthy_cornering.is_finite()
        && hot_cornering.is_finite()
        && low_mu_high_pressure.is_finite()
        && low_mu_low_pressure.is_finite()
        && light.force.iter().all(|v| v.is_finite())
        && heavy.force.iter().all(|v| v.is_finite())
        && pure_long.force.iter().all(|v| v.is_finite())
        && pure_lat.force.iter().all(|v| v.is_finite())
        && combined.force.iter().all(|v| v.is_finite());

    TireLoadCombinedSlipLawReport {
        load_force_ratio,
        load_ratio,
        heavy_stiffness_over_light,
        low_mu_low_pressure_over_high_pressure_fx,
        low_mu_low_pressure_patch_over_high_pressure,
        combined_fx_over_pure_fx,
        combined_fy_over_pure_fy,
        combined_budget_ratio,
        max_saturation_jump_n,
        max_saturation_jump_over_load: max_saturation_jump_n / 4_000.0,
        saturation_start_sliding_fraction,
        saturation_end_sliding_fraction,
        saturation_sliding_fraction_delta: saturation_end_sliding_fraction
            - saturation_start_sliding_fraction,
        ice_over_dry_fx,
        hot_tread_shear_over_healthy: hot.params.tread_shear_stiffness
            / healthy.params.tread_shear_stiffness.max(1.0),
        hot_longitudinal_stiffness_over_healthy: hot_longitudinal.effective_longitudinal_stiffness
            / healthy_longitudinal
                .effective_longitudinal_stiffness
                .max(1.0),
        hot_cornering_stiffness_over_healthy: hot_cornering.effective_cornering_stiffness
            / healthy_cornering.effective_cornering_stiffness.max(1.0),
        finite_outputs,
    }
}

pub fn tire_ice_large_patch_report() -> TireIceLargePatchReport {
    let dry = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let ice = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::ice()
    };
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let high = RealTireParams::high_pressure_reference()
        .with_geometry_and_pressure(0.32, 0.185, 320_000.0)
        .with_segments(96);
    let low = RealTireParams::offroad_soft()
        .with_geometry_and_pressure(0.32, 0.205, 90_000.0)
        .with_segments(128);
    let load_n = 4_500.0;
    let speed_m_per_s = 8.0;
    let slip_ratio = 0.24;
    let high_dry = tire_rig_point(
        &high,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip_ratio,
        0.0,
        0.0,
    );
    let low_dry = tire_rig_point(
        &low,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip_ratio,
        0.0,
        0.0,
    );
    let low_ice = tire_rig_point(
        &low,
        &ice_terrain,
        &ice,
        load_n,
        speed_m_per_s,
        slip_ratio,
        0.0,
        0.0,
    );
    let low_pressure_dry_abs_fx_n = low_dry.telemetry.longitudinal_force_n.abs();
    let low_pressure_ice_abs_fx_n = low_ice.telemetry.longitudinal_force_n.abs();
    let ice_budget_ratio = low_ice.telemetry.consumed_friction_budget_n
        / low_ice.telemetry.available_friction_budget_n.max(1.0);
    let finite_outputs = high_dry.is_finite()
        && low_dry.is_finite()
        && low_ice.is_finite()
        && high_dry.telemetry.patch.patch_length_m > 0.0
        && low_ice.telemetry.patch.patch_length_m > 0.0;
    TireIceLargePatchReport {
        high_pressure_patch_length_m: high_dry.telemetry.patch.patch_length_m,
        low_pressure_ice_patch_length_m: low_ice.telemetry.patch.patch_length_m,
        low_pressure_dry_abs_fx_n,
        low_pressure_ice_abs_fx_n,
        ice_over_dry_fx: low_pressure_ice_abs_fx_n / low_pressure_dry_abs_fx_n.max(1.0),
        low_over_high_patch_ratio: low_ice.telemetry.patch.patch_length_m
            / high_dry.telemetry.patch.patch_length_m.max(1.0e-9),
        dry_sliding_fraction: low_dry.telemetry.patch.sliding_fraction,
        ice_sliding_fraction: low_ice.telemetry.patch.sliding_fraction,
        ice_budget_ratio,
        finite_outputs,
    }
}

pub fn tire_adhesion_sliding_continuity_report() -> TireAdhesionSlidingContinuityReport {
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(128);
    let load_n = 4_200.0;
    let speed_m_per_s = 11.0;
    let mut previous: Option<RealTireTelemetrySample> = None;
    let mut sample_count = 0usize;
    let mut finite_outputs = true;
    let mut min_sliding_fraction = f64::INFINITY;
    let mut max_sliding_fraction = 0.0_f64;
    let mut max_force_jump_over_load = 0.0_f64;
    let mut max_moment_jump_over_load_radius = 0.0_f64;
    let mut max_budget_ratio = 0.0_f64;
    let mut force_sign: Option<(f64, f64)> = None;
    let mut moment_sign: Option<f64> = None;
    let mut force_sign_stable = true;
    let mut moment_sign_stable = true;
    let mut sweep_moments: Vec<f64> = Vec::new();

    for slip_ratio in [
        0.000, 0.015, 0.030, 0.045, 0.060, 0.080, 0.105, 0.135, 0.170, 0.215, 0.270,
    ] {
        let point = tire_rig_point(
            &params,
            &terrain,
            &material,
            load_n,
            speed_m_per_s,
            slip_ratio,
            0.0,
            0.0,
        );
        let point_finite = point.is_finite();
        let sample = point.telemetry;
        finite_outputs &= point_finite
            && sample.force_n.iter().all(|v| v.is_finite())
            && sample.aligning_moment_nm.is_finite()
            && sample.patch.sliding_fraction.is_finite()
            && sample.available_friction_budget_n.is_finite()
            && sample.consumed_friction_budget_n.is_finite();
        min_sliding_fraction = min_sliding_fraction.min(sample.patch.sliding_fraction);
        max_sliding_fraction = max_sliding_fraction.max(sample.patch.sliding_fraction);
        max_budget_ratio = max_budget_ratio
            .max(sample.consumed_friction_budget_n / sample.available_friction_budget_n.max(1.0));
        if sample.longitudinal_force_n.abs() > 1.0 {
            let sign = sample.longitudinal_force_n.signum();
            if let Some(first) = force_sign {
                force_sign_stable &= sign == first.0;
            } else {
                force_sign = Some((sign, 0.0));
            }
        }
        if let Some(prev) = previous {
            let force_delta = [
                sample.force_n[0] - prev.force_n[0],
                sample.force_n[1] - prev.force_n[1],
                sample.force_n[2] - prev.force_n[2],
            ];
            max_force_jump_over_load =
                max_force_jump_over_load.max(dot(force_delta, force_delta).sqrt() / load_n);
            max_moment_jump_over_load_radius = max_moment_jump_over_load_radius.max(
                (sample.aligning_moment_nm - prev.aligning_moment_nm).abs()
                    / (load_n * params.radius.max(1.0e-9)),
            );
        }
        sample_count += 1;
        previous = Some(sample);
    }

    let mut previous_moment: Option<f64> = None;
    for slip_angle_rad in [0.010, 0.020, 0.035, 0.050, 0.070, 0.095, 0.125, 0.160] {
        let point = tire_rig_point(
            &params,
            &terrain,
            &material,
            load_n,
            speed_m_per_s,
            0.080,
            slip_angle_rad,
            0.0,
        );
        let point_finite = point.is_finite();
        let sample = point.telemetry;
        finite_outputs &= point_finite
            && sample.force_n.iter().all(|v| v.is_finite())
            && sample.aligning_moment_nm.is_finite()
            && sample.patch.sliding_fraction.is_finite()
            && sample.available_friction_budget_n.is_finite()
            && sample.consumed_friction_budget_n.is_finite();
        max_budget_ratio = max_budget_ratio
            .max(sample.consumed_friction_budget_n / sample.available_friction_budget_n.max(1.0));
        sweep_moments.push(sample.aligning_moment_nm);
        if let Some(prev) = previous_moment {
            max_moment_jump_over_load_radius = max_moment_jump_over_load_radius.max(
                (sample.aligning_moment_nm - prev).abs() / (load_n * params.radius.max(1.0e-9)),
            );
        }
        sample_count += 1;
        previous_moment = Some(sample.aligning_moment_nm);
    }

    // The aligning moment physically decays toward zero (and may cross it)
    // as the patch saturates at high slip angles, so sign stability is only
    // required while the moment is a meaningful fraction of the sweep peak.
    let peak_moment = sweep_moments
        .iter()
        .fold(0.0_f64, |acc, m| acc.max(m.abs()));
    let moment_floor = (0.10 * peak_moment).max(0.05);
    for moment in &sweep_moments {
        if moment.abs() > moment_floor {
            let sign = moment.signum();
            if let Some(first) = moment_sign {
                moment_sign_stable &= sign == first;
            } else {
                moment_sign = Some(sign);
            }
        }
    }

    TireAdhesionSlidingContinuityReport {
        sample_count,
        finite_outputs,
        min_sliding_fraction,
        max_sliding_fraction,
        max_force_jump_over_load,
        max_moment_jump_over_load_radius,
        max_budget_ratio,
        force_sign_stable,
        moment_sign_stable,
    }
}
