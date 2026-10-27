//! Non-calibration thermal/wear envelope reports and falsification gates.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::FlatTerrain;

use super::super::internal::{bool_metric, finite_nonnegative};
use super::super::{
    RealTireParams, SurfaceFilmState, SurfaceMaterial, TireConditionLevel,
    TireConditionMaterialResponse, TireRigAcceptanceCheck, TireThermalLayers, TireWearState,
    tire_rig_point,
};

/// Non-calibration thermal/wear abuse envelope.
///
/// This is deliberately not rubber-chemistry calibration. It hardens the
/// state-derived failure-mode path by comparing healthy, cold, overheated,
/// worn, and abused states through the reusable [`TireConditionMaterialResponse`]
/// bridge. The gate catches regressions where a viewer/scene bypasses thermal
/// state with a hidden grip multiplier, or where the named condition flags stop
/// agreeing with the material/parameter response.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireThermalAbuseEnvelopeReport {
    pub healthy_mu_factor: f64,
    pub cold_mu_factor: f64,
    pub hot_mu_factor: f64,
    pub worn_mu_factor: f64,
    pub abused_mu_factor: f64,
    pub healthy_wear_rate_factor: f64,
    pub hot_wear_rate_factor: f64,
    pub abused_wear_rate_factor: f64,
    pub healthy_rolling_resistance_factor: f64,
    pub cold_rolling_resistance_factor: f64,
    pub healthy_tread_shear_factor: f64,
    pub hot_tread_shear_factor: f64,
    pub healthy_carcass_stiffness_factor: f64,
    pub abused_carcass_stiffness_factor: f64,
    pub cold_flag: bool,
    pub overheated_flag: bool,
    pub worn_flag: bool,
    pub abused_flag: bool,
    pub abused_level: bool,
    pub finite_outputs: bool,
}

impl TireThermalAbuseEnvelopeReport {
    pub fn is_finite(self) -> bool {
        self.healthy_mu_factor.is_finite()
            && self.cold_mu_factor.is_finite()
            && self.hot_mu_factor.is_finite()
            && self.worn_mu_factor.is_finite()
            && self.abused_mu_factor.is_finite()
            && self.healthy_wear_rate_factor.is_finite()
            && self.hot_wear_rate_factor.is_finite()
            && self.abused_wear_rate_factor.is_finite()
            && self.healthy_rolling_resistance_factor.is_finite()
            && self.cold_rolling_resistance_factor.is_finite()
            && self.healthy_tread_shear_factor.is_finite()
            && self.hot_tread_shear_factor.is_finite()
            && self.healthy_carcass_stiffness_factor.is_finite()
            && self.abused_carcass_stiffness_factor.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.cold_flag
            && self.overheated_flag
            && self.worn_flag
            && self.abused_flag
            && self.abused_level
            && self.cold_mu_factor < self.healthy_mu_factor
            && self.hot_mu_factor < self.healthy_mu_factor
            && self.worn_mu_factor < self.healthy_mu_factor
            && self.abused_mu_factor < self.hot_mu_factor
            && self.hot_wear_rate_factor > self.healthy_wear_rate_factor
            && self.abused_wear_rate_factor > self.hot_wear_rate_factor
            && self.cold_rolling_resistance_factor > self.healthy_rolling_resistance_factor
            && self.hot_tread_shear_factor < self.healthy_tread_shear_factor
            && self.abused_carcass_stiffness_factor < self.healthy_carcass_stiffness_factor
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_envelope_outputs_finite",
                "thermal abuse envelope response should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_cold_flag_state_derived",
                "cold tire response should be flagged from thermal state",
                "cold_flag",
                bool_metric(self.cold_flag),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_hot_flag_state_derived",
                "overheated tire response should be flagged from thermal state",
                "overheated_flag",
                bool_metric(self.overheated_flag),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_worn_flag_state_derived",
                "worn tire response should be flagged from wear depth state",
                "worn_flag",
                bool_metric(self.worn_flag),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_abused_level_state_derived",
                "abused tire label should come from combined heat and wear state",
                "abused_level",
                bool_metric(self.abused_flag && self.abused_level),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_cold_reduces_mu",
                "cold state should reduce usable friction below healthy warm tire",
                "cold_over_healthy_mu",
                self.cold_mu_factor / self.healthy_mu_factor.max(1.0e-9),
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_hot_reduces_mu",
                "overheated state should reduce usable friction below healthy warm tire",
                "hot_over_healthy_mu",
                self.hot_mu_factor / self.healthy_mu_factor.max(1.0e-9),
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_worn_reduces_mu",
                "worn tread state should reduce usable friction below healthy warm tire",
                "worn_over_healthy_mu",
                self.worn_mu_factor / self.healthy_mu_factor.max(1.0e-9),
                0.98,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_combined_heat_wear_worst_mu",
                "combined hot+worn abuse should reduce grip more than hot-only state",
                "abused_over_hot_mu",
                self.abused_mu_factor / self.hot_mu_factor.max(1.0e-9),
                0.98,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_hot_increases_wear_rate",
                "hot tire state should increase wear rate over healthy warm tire",
                "hot_over_healthy_wear_rate",
                self.hot_wear_rate_factor / self.healthy_wear_rate_factor.max(1.0e-9),
                1.01,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_combined_increases_wear_rate",
                "combined hot+worn abuse should increase wear rate over hot-only state",
                "abused_over_hot_wear_rate",
                self.abused_wear_rate_factor / self.hot_wear_rate_factor.max(1.0e-9),
                1.01,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_abuse_cold_increases_rolling_loss",
                "cold tire state should increase rolling resistance factor",
                "cold_over_healthy_rolling_resistance",
                self.cold_rolling_resistance_factor
                    / self.healthy_rolling_resistance_factor.max(1.0e-9),
                1.01,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_hot_softens_tread_shear",
                "overheated tire state should soften tread shear stiffness",
                "hot_over_healthy_tread_shear",
                self.hot_tread_shear_factor / self.healthy_tread_shear_factor.max(1.0e-9),
                0.99,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_abuse_abused_softens_carcass",
                "abused tire state should soften carcass stiffness below healthy state",
                "abused_over_healthy_carcass_stiffness",
                self.abused_carcass_stiffness_factor
                    / self.healthy_carcass_stiffness_factor.max(1.0e-9),
                0.99,
            ),
        ]
    }
}

pub fn tire_thermal_abuse_envelope_report() -> TireThermalAbuseEnvelopeReport {
    let params = RealTireParams::passenger_radial();
    let material = SurfaceMaterial::dry_asphalt();
    let fresh = TireWearState::default();
    let mut worn_wear = TireWearState::default();
    worn_wear.depth_m = worn_wear.max_depth_m * 0.82;
    let mut abused_wear = TireWearState::default();
    abused_wear.depth_m = abused_wear.max_depth_m * 0.96;
    let healthy = TireThermalLayers {
        surface_c: 78.0,
        tread_c: 82.0,
        carcass_c: 70.0,
        rim_c: 48.0,
        ..TireThermalLayers::passenger_road()
    }
    .material_response(&fresh, &params, &material);
    let cold = TireThermalLayers {
        surface_c: -12.0,
        tread_c: -4.0,
        carcass_c: 2.0,
        rim_c: 5.0,
        ambient_c: -8.0,
        ..TireThermalLayers::passenger_road()
    }
    .material_response(&fresh, &params, &material);
    let hot = TireThermalLayers {
        surface_c: 160.0,
        tread_c: 154.0,
        carcass_c: 126.0,
        rim_c: 90.0,
        ambient_c: 32.0,
        ..TireThermalLayers::passenger_road()
    }
    .material_response(&fresh, &params, &material);
    let worn = TireThermalLayers {
        surface_c: 80.0,
        tread_c: 82.0,
        carcass_c: 70.0,
        rim_c: 48.0,
        ..TireThermalLayers::passenger_road()
    }
    .material_response(&worn_wear, &params, &material);
    let abused = TireThermalLayers {
        surface_c: 226.0,
        tread_c: 215.0,
        carcass_c: 188.0,
        rim_c: 115.0,
        ambient_c: 36.0,
        ..TireThermalLayers::passenger_road()
    }
    .material_response(&abused_wear, &params, &material);
    let finite_outputs = healthy.is_finite()
        && cold.is_finite()
        && hot.is_finite()
        && worn.is_finite()
        && abused.is_finite();
    TireThermalAbuseEnvelopeReport {
        healthy_mu_factor: healthy.mu_factor,
        cold_mu_factor: cold.mu_factor,
        hot_mu_factor: hot.mu_factor,
        worn_mu_factor: worn.mu_factor,
        abused_mu_factor: abused.mu_factor,
        healthy_wear_rate_factor: healthy.condition.wear_rate_factor,
        hot_wear_rate_factor: hot.condition.wear_rate_factor,
        abused_wear_rate_factor: abused.condition.wear_rate_factor,
        healthy_rolling_resistance_factor: healthy.rolling_resistance_factor,
        cold_rolling_resistance_factor: cold.rolling_resistance_factor,
        healthy_tread_shear_factor: healthy.tread_shear_factor,
        hot_tread_shear_factor: hot.tread_shear_factor,
        healthy_carcass_stiffness_factor: healthy.carcass_stiffness_factor,
        abused_carcass_stiffness_factor: abused.carcass_stiffness_factor,
        cold_flag: cold.cold_grip_limited && cold.condition.level == TireConditionLevel::Cold,
        overheated_flag: hot.overheated_grip_loss
            && matches!(
                hot.condition.level,
                TireConditionLevel::Hot | TireConditionLevel::Overheated
            ),
        worn_flag: worn.worn_tread_limited,
        abused_flag: abused.abused,
        abused_level: abused.condition.level == TireConditionLevel::Abused,
        finite_outputs,
    }
}

/// Non-calibration wear sensitivity report.
///
/// This is deliberately not a tire-compound wear calibration. It checks the
/// state equation itself: for the same duration, higher sliding power must
/// remove more tread, and the same sliding power must remove more tread when
/// the surface layer is overheated. The numbers are broad monotonic gates so
/// viewer scenes cannot fake wear by changing a label without energy and
/// temperature state.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireWearPowerTemperatureReport {
    pub duration_s: f64,
    pub low_power_w: f64,
    pub high_power_w: f64,
    pub warm_surface_c: f64,
    pub hot_surface_c: f64,
    pub low_power_warm_wear_m: f64,
    pub high_power_warm_wear_m: f64,
    pub high_power_hot_wear_m: f64,
    pub high_over_low_wear_ratio: f64,
    pub hot_over_warm_wear_ratio: f64,
    pub finite_outputs: bool,
}

impl TireWearPowerTemperatureReport {
    pub fn is_finite(self) -> bool {
        self.duration_s.is_finite()
            && self.low_power_w.is_finite()
            && self.high_power_w.is_finite()
            && self.warm_surface_c.is_finite()
            && self.hot_surface_c.is_finite()
            && self.low_power_warm_wear_m.is_finite()
            && self.high_power_warm_wear_m.is_finite()
            && self.high_power_hot_wear_m.is_finite()
            && self.high_over_low_wear_ratio.is_finite()
            && self.hot_over_warm_wear_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.high_power_w > self.low_power_w
            && self.hot_surface_c > self.warm_surface_c
            && self.high_power_warm_wear_m > self.low_power_warm_wear_m
            && self.high_power_hot_wear_m > self.high_power_warm_wear_m
            && self.high_over_low_wear_ratio > 1.5
            && self.hot_over_warm_wear_ratio > 1.2
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_power_temp_outputs_finite",
                "wear power/temperature sensitivity replay should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_power_increases_wear",
                "higher sliding power should remove more tread at the same surface temperature",
                "high_over_low_wear_ratio",
                self.high_over_low_wear_ratio,
                1.5,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_hot_surface_accelerates_wear",
                "hotter surface should remove more tread at the same sliding power",
                "hot_over_warm_wear_ratio",
                self.hot_over_warm_wear_ratio,
                1.2,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_temperature_ordered",
                "hot surface fixture should be hotter than the warm baseline",
                "hot_minus_warm_surface_c",
                self.hot_surface_c - self.warm_surface_c,
                1.0,
            ),
        ]
    }
}

pub fn tire_wear_power_temperature_report() -> TireWearPowerTemperatureReport {
    let duration_s = 12.0;
    let dt_s = 0.05;
    let low_power_w = 2_000.0;
    let high_power_w = 8_000.0;
    let warm_surface_c = 70.0;
    let hot_surface_c = 150.0;
    let steps = (duration_s / dt_s) as usize;
    let integrate = |power_w: f64, surface_c: f64| {
        let mut wear = TireWearState::default();
        for _ in 0..steps {
            wear.step_with_surface_temperature(power_w, surface_c, dt_s);
        }
        wear.depth_m
    };
    let low_power_warm_wear_m = integrate(low_power_w, warm_surface_c);
    let high_power_warm_wear_m = integrate(high_power_w, warm_surface_c);
    let high_power_hot_wear_m = integrate(high_power_w, hot_surface_c);
    let finite_outputs = low_power_warm_wear_m.is_finite()
        && high_power_warm_wear_m.is_finite()
        && high_power_hot_wear_m.is_finite();
    TireWearPowerTemperatureReport {
        duration_s,
        low_power_w,
        high_power_w,
        warm_surface_c,
        hot_surface_c,
        low_power_warm_wear_m,
        high_power_warm_wear_m,
        high_power_hot_wear_m,
        high_over_low_wear_ratio: high_power_warm_wear_m / low_power_warm_wear_m.max(1.0e-12),
        hot_over_warm_wear_ratio: high_power_hot_wear_m / high_power_warm_wear_m.max(1.0e-12),
        finite_outputs,
    }
}

/// Non-calibration worn-tread surface sensitivity report.
///
/// A worn tire should not be represented as one scene grip scalar. Drainage
/// loss, weak shear, and loose surface coupling should make wet, ice, and
/// loose-soil performance more vulnerable than dry asphalt through the same
/// thermal/wear/material response path used by live tire contacts.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireWornSurfaceSensitivityReport {
    pub dry_loss_fraction: f64,
    pub wet_loss_fraction: f64,
    pub ice_loss_fraction: f64,
    pub loose_soil_loss_fraction: f64,
    pub wet_extra_loss_fraction: f64,
    pub ice_extra_loss_fraction: f64,
    pub loose_soil_extra_loss_fraction: f64,
    pub finite_outputs: bool,
}

impl TireWornSurfaceSensitivityReport {
    pub fn is_finite(self) -> bool {
        self.dry_loss_fraction.is_finite()
            && self.wet_loss_fraction.is_finite()
            && self.ice_loss_fraction.is_finite()
            && self.loose_soil_loss_fraction.is_finite()
            && self.wet_extra_loss_fraction.is_finite()
            && self.ice_extra_loss_fraction.is_finite()
            && self.loose_soil_extra_loss_fraction.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.dry_loss_fraction > 0.0
            && self.wet_extra_loss_fraction > 0.02
            && self.ice_extra_loss_fraction > 0.02
            && self.loose_soil_extra_loss_fraction > 0.02
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "worn_surface_sensitivity_outputs_finite",
                "worn-surface sensitivity report should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "worn_surface_dry_loses_force",
                "worn tread should reduce dry asphalt force from tire state",
                "dry_loss_fraction",
                self.dry_loss_fraction,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "worn_surface_wet_extra_loss",
                "worn tread should hurt wet-film force more than dry asphalt",
                "wet_minus_dry_loss_fraction",
                self.wet_extra_loss_fraction,
                0.02,
            ),
            TireRigAcceptanceCheck::at_least(
                "worn_surface_ice_extra_loss",
                "worn tread should hurt ice force more than dry asphalt",
                "ice_minus_dry_loss_fraction",
                self.ice_extra_loss_fraction,
                0.02,
            ),
            TireRigAcceptanceCheck::at_least(
                "worn_surface_loose_soil_extra_loss",
                "worn tread should hurt loose-soil force more than dry asphalt",
                "loose_soil_minus_dry_loss_fraction",
                self.loose_soil_extra_loss_fraction,
                0.02,
            ),
        ]
    }
}

pub fn tire_worn_surface_sensitivity_report() -> TireWornSurfaceSensitivityReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let dry = SurfaceMaterial::dry_asphalt();
    let ice = SurfaceMaterial::ice();
    let soil = SurfaceMaterial::soft_soil();
    let thermal = TireThermalLayers::passenger_road();
    let fresh = TireWearState::default();
    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.92;
    let load_n = 4_000.0;
    let speed_m_per_s = 13.0;
    let slip_ratio = 0.14;

    let force_for = |material: SurfaceMaterial, wear: &TireWearState| {
        let response =
            TireConditionMaterialResponse::from_state(&params, &material, &thermal, wear);
        let point = tire_rig_point(
            &response.params,
            &FlatTerrain::new(0.0, response.material.mu_static),
            &response.material,
            load_n,
            speed_m_per_s,
            slip_ratio,
            0.0,
            0.0,
        );
        let material_capacity_n = (response.material.mu_dynamic * load_n
            + response.material.cohesion * point.telemetry.patch.contact_area_m2
            - response.material.rolling_resistance * load_n)
            .max(1.0);
        (
            material_capacity_n,
            point.is_finite() && response.is_finite(),
        )
    };
    let wet_for = |wear: &TireWearState| {
        let wet = SurfaceFilmState::wet_road().apply_to(dry, speed_m_per_s, wear);
        force_for(wet, wear)
    };
    let loss_fraction = |fresh_force: f64, worn_force: f64| {
        (1.0 - finite_nonnegative(worn_force) / finite_nonnegative(fresh_force).max(1.0))
            .clamp(0.0, 1.0)
    };

    let (dry_fresh, dry_finite_a) = force_for(dry, &fresh);
    let (dry_worn, dry_finite_b) = force_for(dry, &worn);
    let (wet_fresh, wet_finite_a) = wet_for(&fresh);
    let (wet_worn, wet_finite_b) = wet_for(&worn);
    let (ice_fresh, ice_finite_a) = force_for(ice, &fresh);
    let (ice_worn, ice_finite_b) = force_for(ice, &worn);
    let (soil_fresh, soil_finite_a) = force_for(soil, &fresh);
    let (soil_worn, soil_finite_b) = force_for(soil, &worn);

    let dry_loss_fraction = loss_fraction(dry_fresh, dry_worn);
    let wet_loss_fraction = loss_fraction(wet_fresh, wet_worn);
    let ice_loss_fraction = loss_fraction(ice_fresh, ice_worn);
    let loose_soil_loss_fraction = loss_fraction(soil_fresh, soil_worn);
    TireWornSurfaceSensitivityReport {
        dry_loss_fraction,
        wet_loss_fraction,
        ice_loss_fraction,
        loose_soil_loss_fraction,
        wet_extra_loss_fraction: wet_loss_fraction - dry_loss_fraction,
        ice_extra_loss_fraction: ice_loss_fraction - dry_loss_fraction,
        loose_soil_extra_loss_fraction: loose_soil_loss_fraction - dry_loss_fraction,
        finite_outputs: dry_finite_a
            && dry_finite_b
            && wet_finite_a
            && wet_finite_b
            && ice_finite_a
            && ice_finite_b
            && soil_finite_a
            && soil_finite_b,
    }
}

/// Non-calibration sustained rolling/carcass heat envelope.
///
/// This exercises the "rolling heats the tire body even without a burnout"
/// path in [`TireThermalLayers`]. It deliberately commands zero brush slip, so
/// any temperature/pressure movement comes from rolling resistance and cyclic
/// carcass flex instead of hidden viewer grip multipliers or sliding heat.
#[derive(Clone, Copy, Debug)]
pub struct TireThermalRollingCarcassReport {
    pub sample_count: usize,
    pub duration_s: f64,
    pub load_n: f64,
    pub speed_m_per_s: f64,
    pub initial_surface_c: f64,
    pub final_surface_c: f64,
    pub initial_tread_c: f64,
    pub final_tread_c: f64,
    pub initial_carcass_c: f64,
    pub final_carcass_c: f64,
    pub initial_pressure_factor: f64,
    pub final_pressure_factor: f64,
    pub min_mu_factor: f64,
    pub max_mu_factor: f64,
    pub max_sliding_power_w: f64,
    pub rolling_energy_j: f64,
    pub carcass_flex_energy_j: f64,
    pub final_thermal: TireThermalLayers,
    pub final_wear: TireWearState,
    pub finite_outputs: bool,
}

impl Default for TireThermalRollingCarcassReport {
    fn default() -> Self {
        let thermal = TireThermalLayers::passenger_road();
        Self {
            sample_count: 0,
            duration_s: 0.0,
            load_n: 0.0,
            speed_m_per_s: 0.0,
            initial_surface_c: thermal.surface_c,
            final_surface_c: thermal.surface_c,
            initial_tread_c: thermal.tread_c,
            final_tread_c: thermal.tread_c,
            initial_carcass_c: thermal.carcass_c,
            final_carcass_c: thermal.carcass_c,
            initial_pressure_factor: thermal.carcass_pressure_factor(),
            final_pressure_factor: thermal.carcass_pressure_factor(),
            min_mu_factor: thermal.mu_factor(),
            max_mu_factor: thermal.mu_factor(),
            max_sliding_power_w: 0.0,
            rolling_energy_j: 0.0,
            carcass_flex_energy_j: 0.0,
            final_thermal: thermal,
            final_wear: TireWearState::default(),
            finite_outputs: true,
        }
    }
}

impl TireThermalRollingCarcassReport {
    pub fn surface_delta_c(self) -> f64 {
        self.final_surface_c - self.initial_surface_c
    }

    pub fn tread_delta_c(self) -> f64 {
        self.final_tread_c - self.initial_tread_c
    }

    pub fn carcass_delta_c(self) -> f64 {
        self.final_carcass_c - self.initial_carcass_c
    }

    pub fn pressure_factor_delta(self) -> f64 {
        self.final_pressure_factor - self.initial_pressure_factor
    }

    pub fn is_finite(self) -> bool {
        self.duration_s.is_finite()
            && self.load_n.is_finite()
            && self.speed_m_per_s.is_finite()
            && self.initial_surface_c.is_finite()
            && self.final_surface_c.is_finite()
            && self.initial_tread_c.is_finite()
            && self.final_tread_c.is_finite()
            && self.initial_carcass_c.is_finite()
            && self.final_carcass_c.is_finite()
            && self.initial_pressure_factor.is_finite()
            && self.final_pressure_factor.is_finite()
            && self.min_mu_factor.is_finite()
            && self.max_mu_factor.is_finite()
            && self.max_sliding_power_w.is_finite()
            && self.rolling_energy_j.is_finite()
            && self.carcass_flex_energy_j.is_finite()
            && self.final_thermal.surface_c.is_finite()
            && self.final_thermal.tread_c.is_finite()
            && self.final_thermal.carcass_c.is_finite()
            && self.final_thermal.rim_c.is_finite()
            && self.final_wear.depth_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 120
            && self.max_sliding_power_w <= 1.0e-6
            && self.rolling_energy_j > 10_000.0
            && self.carcass_flex_energy_j > 2_000.0
            && self.carcass_delta_c() > 0.75
            && self.tread_delta_c() > 0.35
            && self.pressure_factor_delta() > 0.001
            && self.min_mu_factor >= 0.45
            && self.max_mu_factor <= 1.08
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_outputs_finite",
                "rolling/carcass heat envelope should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_has_samples",
                "rolling/carcass heat envelope should contain sustained samples",
                "sample_count",
                self.sample_count as f64,
                120.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_rolling_carcass_low_sliding_power",
                "rolling/carcass heat envelope should not rely on sliding heat",
                "max_sliding_power_w",
                self.max_sliding_power_w,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_rolling_energy_active",
                "rolling resistance should inject measurable rolling energy",
                "rolling_energy_j",
                self.rolling_energy_j,
                10_000.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_flex_energy_active",
                "cyclic casing flex should inject measurable carcass energy",
                "carcass_flex_energy_j",
                self.carcass_flex_energy_j,
                2_000.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_heats_carcass",
                "sustained rolling should heat the carcass even at zero slip",
                "carcass_delta_c",
                self.carcass_delta_c(),
                0.75,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_warms_tread_bulk",
                "rolling/carcass heat should conduct into the tread bulk",
                "tread_delta_c",
                self.tread_delta_c(),
                0.35,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_rolling_carcass_pressure_factor_rises",
                "carcass heating should raise the slow pressure/stiffness factor",
                "pressure_factor_delta",
                self.pressure_factor_delta(),
                0.001,
            ),
            TireRigAcceptanceCheck::between(
                "thermal_rolling_carcass_mu_factor_clamped",
                "thermal grip factor should remain finite and clamped during rolling heat",
                "mu_factor_range",
                self.max_mu_factor.max(self.min_mu_factor),
                0.45,
                1.08,
            ),
        ]
    }
}

pub fn tire_thermal_rolling_carcass_report() -> TireThermalRollingCarcassReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let mut thermal = TireThermalLayers::passenger_road();
    let mut wear = TireWearState::default();
    let initial_thermal = thermal;
    let load_n = 4_500.0;
    let speed_m_per_s: f64 = 22.0;
    let dt_s: f64 = 0.25;
    let sample_count = 240;
    let mut min_mu_factor = thermal.mu_factor();
    let mut max_mu_factor = thermal.mu_factor();
    let mut max_sliding_power_w: f64 = 0.0;
    let mut rolling_energy_j = 0.0;
    let mut carcass_flex_energy_j = 0.0;
    let mut finite_outputs = true;

    for _ in 0..sample_count {
        let response = thermal.material_response(&wear, &params, &material);
        let rig = tire_rig_point(
            &response.params,
            &terrain,
            &response.material,
            load_n,
            speed_m_per_s,
            0.0,
            0.0,
            0.0,
        );
        let sliding_power_w = rig.telemetry.dissipated_power_w.abs();
        let rolling_power_w =
            (rig.telemetry.rolling_resistance_force_n * speed_m_per_s.abs()).abs();
        let carcass_flex_power_w = load_n * speed_m_per_s.abs() * 0.0025;

        max_sliding_power_w = max_sliding_power_w.max(sliding_power_w);
        rolling_energy_j += rolling_power_w * dt_s;
        carcass_flex_energy_j += carcass_flex_power_w * dt_s;
        thermal.step(
            sliding_power_w,
            carcass_flex_power_w,
            rolling_power_w,
            0.0,
            20.0,
            dt_s,
        );
        wear.step_with_surface_temperature(sliding_power_w, thermal.surface_c, dt_s);
        let mu_factor = thermal.mu_factor();
        min_mu_factor = min_mu_factor.min(mu_factor);
        max_mu_factor = max_mu_factor.max(mu_factor);
        finite_outputs &= rig.telemetry.is_finite()
            && thermal.surface_c.is_finite()
            && thermal.tread_c.is_finite()
            && thermal.carcass_c.is_finite()
            && thermal.rim_c.is_finite()
            && wear.depth_m.is_finite()
            && response.is_finite();
    }

    TireThermalRollingCarcassReport {
        sample_count,
        duration_s: sample_count as f64 * dt_s,
        load_n,
        speed_m_per_s,
        initial_surface_c: initial_thermal.surface_c,
        final_surface_c: thermal.surface_c,
        initial_tread_c: initial_thermal.tread_c,
        final_tread_c: thermal.tread_c,
        initial_carcass_c: initial_thermal.carcass_c,
        final_carcass_c: thermal.carcass_c,
        initial_pressure_factor: initial_thermal.carcass_pressure_factor(),
        final_pressure_factor: thermal.carcass_pressure_factor(),
        min_mu_factor,
        max_mu_factor,
        max_sliding_power_w,
        rolling_energy_j,
        carcass_flex_energy_j,
        final_thermal: thermal,
        final_wear: wear,
        finite_outputs,
    }
}

/// Non-calibration brake/rim heat-source envelope.
///
/// This isolates the fourth thermal input in [`TireThermalLayers`]: brake heat
/// enters the rim first, then conducts into the carcass/tread instead of being
/// disguised as surface sliding heat. It is a deterministic regression gate,
/// not a brake package or tire aging calibration.
#[derive(Clone, Copy, Debug)]
pub struct TireThermalBrakeRimReport {
    pub sample_count: usize,
    pub duration_s: f64,
    pub brake_power_w: f64,
    pub brake_energy_j: f64,
    pub initial_surface_c: f64,
    pub final_surface_c: f64,
    pub initial_tread_c: f64,
    pub final_tread_c: f64,
    pub initial_carcass_c: f64,
    pub final_carcass_c: f64,
    pub initial_rim_c: f64,
    pub final_rim_c: f64,
    pub initial_pressure_factor: f64,
    pub final_pressure_factor: f64,
    pub min_mu_factor: f64,
    pub max_mu_factor: f64,
    pub max_sliding_power_w: f64,
    pub final_thermal: TireThermalLayers,
    pub final_wear: TireWearState,
    pub finite_outputs: bool,
}

impl Default for TireThermalBrakeRimReport {
    fn default() -> Self {
        let thermal = TireThermalLayers::passenger_road();
        Self {
            sample_count: 0,
            duration_s: 0.0,
            brake_power_w: 0.0,
            brake_energy_j: 0.0,
            initial_surface_c: thermal.surface_c,
            final_surface_c: thermal.surface_c,
            initial_tread_c: thermal.tread_c,
            final_tread_c: thermal.tread_c,
            initial_carcass_c: thermal.carcass_c,
            final_carcass_c: thermal.carcass_c,
            initial_rim_c: thermal.rim_c,
            final_rim_c: thermal.rim_c,
            initial_pressure_factor: thermal.carcass_pressure_factor(),
            final_pressure_factor: thermal.carcass_pressure_factor(),
            min_mu_factor: thermal.mu_factor(),
            max_mu_factor: thermal.mu_factor(),
            max_sliding_power_w: 0.0,
            final_thermal: thermal,
            final_wear: TireWearState::default(),
            finite_outputs: true,
        }
    }
}

impl TireThermalBrakeRimReport {
    pub fn surface_delta_c(self) -> f64 {
        self.final_surface_c - self.initial_surface_c
    }

    pub fn tread_delta_c(self) -> f64 {
        self.final_tread_c - self.initial_tread_c
    }

    pub fn carcass_delta_c(self) -> f64 {
        self.final_carcass_c - self.initial_carcass_c
    }

    pub fn rim_delta_c(self) -> f64 {
        self.final_rim_c - self.initial_rim_c
    }

    pub fn pressure_factor_delta(self) -> f64 {
        self.final_pressure_factor - self.initial_pressure_factor
    }

    pub fn is_finite(self) -> bool {
        self.duration_s.is_finite()
            && self.brake_power_w.is_finite()
            && self.brake_energy_j.is_finite()
            && self.initial_surface_c.is_finite()
            && self.final_surface_c.is_finite()
            && self.initial_tread_c.is_finite()
            && self.final_tread_c.is_finite()
            && self.initial_carcass_c.is_finite()
            && self.final_carcass_c.is_finite()
            && self.initial_rim_c.is_finite()
            && self.final_rim_c.is_finite()
            && self.initial_pressure_factor.is_finite()
            && self.final_pressure_factor.is_finite()
            && self.min_mu_factor.is_finite()
            && self.max_mu_factor.is_finite()
            && self.max_sliding_power_w.is_finite()
            && self.final_thermal.surface_c.is_finite()
            && self.final_thermal.tread_c.is_finite()
            && self.final_thermal.carcass_c.is_finite()
            && self.final_thermal.rim_c.is_finite()
            && self.final_wear.depth_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 90
            && self.brake_energy_j > 20_000.0
            && self.max_sliding_power_w <= 1.0e-6
            && self.rim_delta_c() > 0.45
            && self.carcass_delta_c() > 0.05
            && self.rim_delta_c() > self.carcass_delta_c()
            && self.carcass_delta_c() > self.surface_delta_c()
            && self.pressure_factor_delta() > 0.00005
            && self.min_mu_factor >= 0.45
            && self.max_mu_factor <= 1.08
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_outputs_finite",
                "brake/rim heat envelope should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_has_samples",
                "brake/rim heat envelope should contain sustained samples",
                "sample_count",
                self.sample_count as f64,
                90.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_energy_active",
                "brake/rim source should inject measurable energy",
                "brake_energy_j",
                self.brake_energy_j,
                20_000.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_brake_rim_low_sliding_power",
                "brake/rim heat envelope should not rely on brush sliding heat",
                "max_sliding_power_w",
                self.max_sliding_power_w,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_heats_rim_first",
                "brake heat should raise rim temperature before tire-body temperature",
                "rim_minus_carcass_delta_c",
                self.rim_delta_c() - self.carcass_delta_c(),
                0.25,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_conducts_to_carcass",
                "rim heat should conduct into the carcass node",
                "carcass_delta_c",
                self.carcass_delta_c(),
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_carcass_over_surface",
                "brake heat should enter the tire body before the road-contact skin",
                "carcass_minus_surface_delta_c",
                self.carcass_delta_c() - self.surface_delta_c(),
                0.01,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_brake_rim_pressure_factor_rises",
                "brake-conducted carcass heat should nudge the pressure/stiffness factor",
                "pressure_factor_delta",
                self.pressure_factor_delta(),
                0.00005,
            ),
            TireRigAcceptanceCheck::between(
                "thermal_brake_rim_mu_factor_clamped",
                "thermal grip factor should stay finite and clamped under rim heating",
                "mu_factor_range",
                self.max_mu_factor.max(self.min_mu_factor),
                0.45,
                1.08,
            ),
        ]
    }
}

pub fn tire_thermal_brake_rim_report() -> TireThermalBrakeRimReport {
    let mut thermal = TireThermalLayers::passenger_road();
    let mut wear = TireWearState::default();
    let initial_thermal = thermal;
    let sample_count = 160;
    let dt_s: f64 = 0.20;
    let brake_power_w: f64 = 3_200.0;
    let mut min_mu_factor = thermal.mu_factor();
    let mut max_mu_factor = thermal.mu_factor();
    let mut max_sliding_power_w: f64 = 0.0;
    let mut finite_outputs = true;

    for _ in 0..sample_count {
        let sliding_power_w: f64 = 0.0;
        max_sliding_power_w = max_sliding_power_w.max(sliding_power_w);
        thermal.step(sliding_power_w, 0.0, 0.0, brake_power_w, 20.0, dt_s);
        wear.step_with_surface_temperature(sliding_power_w, thermal.surface_c, dt_s);
        let mu_factor = thermal.mu_factor();
        min_mu_factor = min_mu_factor.min(mu_factor);
        max_mu_factor = max_mu_factor.max(mu_factor);
        finite_outputs &= thermal.surface_c.is_finite()
            && thermal.tread_c.is_finite()
            && thermal.carcass_c.is_finite()
            && thermal.rim_c.is_finite()
            && wear.depth_m.is_finite();
    }

    TireThermalBrakeRimReport {
        sample_count,
        duration_s: sample_count as f64 * dt_s,
        brake_power_w,
        brake_energy_j: brake_power_w * sample_count as f64 * dt_s,
        initial_surface_c: initial_thermal.surface_c,
        final_surface_c: thermal.surface_c,
        initial_tread_c: initial_thermal.tread_c,
        final_tread_c: thermal.tread_c,
        initial_carcass_c: initial_thermal.carcass_c,
        final_carcass_c: thermal.carcass_c,
        initial_rim_c: initial_thermal.rim_c,
        final_rim_c: thermal.rim_c,
        initial_pressure_factor: initial_thermal.carcass_pressure_factor(),
        final_pressure_factor: thermal.carcass_pressure_factor(),
        min_mu_factor,
        max_mu_factor,
        max_sliding_power_w,
        final_thermal: thermal,
        final_wear: wear,
        finite_outputs,
    }
}
