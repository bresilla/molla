//! State-derived tire condition (cold/healthy/hot/overheated/abused/worn)
//! reports plus failure-mode falsification.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::FlatTerrain;

use super::internal::{bool_metric, finite_nonnegative, finite_or, finite_positive_or};
use super::{
    RealTireParams, SurfaceMaterial, TireRigAcceptanceCheck, TireThermalLayers, TireWearState,
    tire_rig_point,
};

/// Coarse state-derived tire condition label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireConditionLevel {
    Cold,
    Healthy,
    Hot,
    Overheated,
    Abused,
}

impl TireConditionLevel {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Healthy => "healthy",
            Self::Hot => "hot",
            Self::Overheated => "overheated",
            Self::Abused => "abused",
        }
    }
}

/// A deterministic report of tire condition derived from thermal + wear state.
#[derive(Clone, Copy, Debug)]
pub struct TireConditionReport {
    pub level: TireConditionLevel,
    /// `[0,1]` score for how far the tire has moved from healthy operation.
    pub abuse_index: f64,
    /// Extra cold-grip penalty from an under-temperature surface/tread.
    pub cold_penalty: f64,
    /// Extra hot-grip penalty from overheated rubber.
    pub overheat_penalty: f64,
    /// Wear-depth ratio, `0=new` and `1=fully worn`.
    pub wear_ratio: f64,
    /// Final contact-friction multiplier from state-derived condition.
    pub grip_factor: f64,
    /// Slow carcass stiffness/pressure multiplier from the same report.
    pub carcass_stiffness_factor: f64,
    /// Wear-rate multiplier for the current surface temperature/state.
    pub wear_rate_factor: f64,
}

impl TireConditionReport {
    pub fn from_thermal_and_wear(thermal: &TireThermalLayers, wear: &TireWearState) -> Self {
        let ambient_c = finite_or(thermal.ambient_c, 20.0);
        let surface_c = finite_or(thermal.surface_c, ambient_c);
        let tread_c = finite_or(thermal.tread_c, ambient_c);
        let carcass_c = finite_or(thermal.carcass_c, ambient_c);
        let rubber_peak_c = surface_c.max(tread_c).max(carcass_c);
        let wear_ratio = (wear.depth_m / wear.max_depth_m.max(1.0e-9)).clamp(0.0, 1.0);

        // Cold and overheat penalties are intentionally smooth, bounded, and
        // monotonic. They do not fit a tire dataset; they expose plausible
        // state-driven failure modes for regression tests and viewer readouts.
        let cold_penalty = ((55.0 - 0.65 * surface_c - 0.35 * tread_c) / 55.0).clamp(0.0, 0.45);
        let overheat_penalty =
            (((surface_c - 125.0) / 95.0).max((tread_c - 135.0) / 110.0)).clamp(0.0, 0.55);
        let carcass_heat_penalty = ((carcass_c - 125.0) / 120.0).clamp(0.0, 0.28);
        let abuse_index = (0.55 * overheat_penalty / 0.55
            + 0.25 * carcass_heat_penalty / 0.28
            + 0.20 * wear_ratio)
            .clamp(0.0, 1.0);
        let grip_factor = (thermal.mu_factor()
            * wear.mu_factor()
            * (1.0 - cold_penalty)
            * (1.0 - 0.30 * abuse_index))
            .clamp(0.08, 1.12);
        let carcass_stiffness_factor =
            (thermal.carcass_pressure_factor() * (1.0 - 0.18 * abuse_index)).clamp(0.70, 1.20);
        let wear_rate_factor = (TireWearState::surface_temperature_wear_factor(surface_c)
            * (1.0 + 1.5 * abuse_index))
            .clamp(0.35, 10.0);
        let level = if abuse_index > 0.72 || wear_ratio > 0.92 || rubber_peak_c > 210.0 {
            TireConditionLevel::Abused
        } else if overheat_penalty > 0.34 || rubber_peak_c > 165.0 {
            TireConditionLevel::Overheated
        } else if overheat_penalty > 0.08 || rubber_peak_c > 115.0 {
            TireConditionLevel::Hot
        } else if cold_penalty > 0.12 {
            TireConditionLevel::Cold
        } else {
            TireConditionLevel::Healthy
        };

        Self {
            level,
            abuse_index,
            cold_penalty,
            overheat_penalty,
            wear_ratio,
            grip_factor,
            carcass_stiffness_factor,
            wear_rate_factor,
        }
    }

    pub fn is_finite(self) -> bool {
        self.abuse_index.is_finite()
            && self.cold_penalty.is_finite()
            && self.overheat_penalty.is_finite()
            && self.wear_ratio.is_finite()
            && self.grip_factor.is_finite()
            && self.carcass_stiffness_factor.is_finite()
            && self.wear_rate_factor.is_finite()
    }
}

/// State-derived material/parameter response from thermal + wear condition.
///
/// This is the reusable "failure mode" bridge: callers hand in the nominal
/// tire and road material, and the current thermal/wear state returns the live
/// tire parameters plus material coefficients. The factors are broad,
/// bounded, and deterministic; they are not fitted tire-life curves.
#[derive(Clone, Copy, Debug)]
pub struct TireConditionMaterialResponse {
    pub condition: TireConditionReport,
    pub params: RealTireParams,
    pub material: SurfaceMaterial,
    pub mu_factor: f64,
    pub pressure_factor: f64,
    pub carcass_stiffness_factor: f64,
    pub tread_shear_factor: f64,
    pub bristle_damping_factor: f64,
    pub rolling_resistance_factor: f64,
    pub cold_grip_limited: bool,
    pub overheated_grip_loss: bool,
    pub carcass_softened: bool,
    pub worn_tread_limited: bool,
    pub abused: bool,
}

impl TireConditionMaterialResponse {
    pub fn from_state(
        params: &RealTireParams,
        material: &SurfaceMaterial,
        thermal: &TireThermalLayers,
        wear: &TireWearState,
    ) -> Self {
        let condition = thermal.condition_report(wear);
        let pressure_factor = thermal.carcass_pressure_factor().clamp(0.70, 1.30);
        let carcass_stiffness_factor = (condition.carcass_stiffness_factor
            * (1.0
                - 0.18 * condition.overheat_penalty
                - 0.12 * condition.abuse_index
                - 0.08 * condition.wear_ratio))
            .clamp(0.55, 1.25);
        let tread_shear_factor = (1.0 + 0.10 * condition.cold_penalty
            - 0.30 * condition.overheat_penalty
            - 0.18 * condition.wear_ratio
            - 0.22 * condition.abuse_index)
            .clamp(0.45, 1.12);
        let bristle_damping_factor = (1.0 + 0.18 * condition.cold_penalty
            - 0.28 * condition.overheat_penalty
            - 0.22 * condition.abuse_index)
            .clamp(0.45, 1.18);
        let rolling_resistance_factor = (1.0
            + 0.35 * condition.cold_penalty
            + 0.20 * condition.overheat_penalty
            + 0.25 * condition.wear_ratio
            + 0.35 * condition.abuse_index)
            .clamp(0.75, 2.20);
        let dry_ref = SurfaceMaterial::dry_asphalt();
        let low_mu_surface_vulnerability =
            (1.0 - material.mu_static / dry_ref.mu_static.max(1.0e-9)).clamp(0.0, 1.0);
        let loose_surface_vulnerability =
            ((material.rolling_resistance - dry_ref.rolling_resistance) / 0.08).clamp(0.0, 1.0);
        let surface_wear_vulnerability =
            low_mu_surface_vulnerability.max(loose_surface_vulnerability);
        let worn_surface_penalty =
            (0.18 * condition.wear_ratio * surface_wear_vulnerability).clamp(0.0, 0.22);
        let mu_factor = (condition.grip_factor * (1.0 - worn_surface_penalty)).clamp(0.05, 1.15);

        let mut live_params = *params;
        live_params.pressure_pa = finite_nonnegative(params.pressure_pa * pressure_factor);
        live_params.carcass_normal_stiffness = finite_positive_or(
            params.carcass_normal_stiffness * carcass_stiffness_factor,
            params.carcass_normal_stiffness,
            1.0,
        );
        live_params.tread_shear_stiffness = finite_positive_or(
            params.tread_shear_stiffness * tread_shear_factor,
            params.tread_shear_stiffness,
            1.0,
        );

        let mut live_material = *material;
        live_material.mu_static = finite_nonnegative(material.mu_static * mu_factor);
        live_material.mu_dynamic =
            finite_nonnegative(material.mu_dynamic * mu_factor).min(live_material.mu_static);
        live_material.bristle_damping =
            finite_nonnegative(material.bristle_damping * bristle_damping_factor);
        live_material.rolling_resistance =
            finite_nonnegative(material.rolling_resistance * rolling_resistance_factor);
        // Cohesion is a ground property, but a badly hot/worn tire cannot make
        // full use of cohesive shear through the tread; keep the effect mild.
        let cohesive_tread_factor = ((0.75 + 0.25 * mu_factor)
            * (1.0 - 0.55 * condition.wear_ratio * surface_wear_vulnerability))
            .clamp(0.25, 1.05);
        live_material.cohesion = finite_nonnegative(material.cohesion * cohesive_tread_factor);

        Self {
            condition,
            params: live_params,
            material: live_material,
            mu_factor,
            pressure_factor,
            carcass_stiffness_factor,
            tread_shear_factor,
            bristle_damping_factor,
            rolling_resistance_factor,
            cold_grip_limited: condition.cold_penalty > 0.12,
            overheated_grip_loss: condition.overheat_penalty > 0.16,
            carcass_softened: carcass_stiffness_factor < 0.88,
            worn_tread_limited: condition.wear_ratio > 0.65,
            abused: condition.level == TireConditionLevel::Abused,
        }
    }

    pub fn is_finite(self) -> bool {
        self.condition.is_finite()
            && self.params.radius.is_finite()
            && self.params.width.is_finite()
            && self.params.pressure_pa.is_finite()
            && self.params.carcass_normal_stiffness.is_finite()
            && self.params.tread_shear_stiffness.is_finite()
            && self.params.relaxation_speed.is_finite()
            && self.material.mu_static.is_finite()
            && self.material.mu_dynamic.is_finite()
            && self.material.bristle_damping.is_finite()
            && self.material.rolling_resistance.is_finite()
            && self.material.cohesion.is_finite()
            && self.mu_factor.is_finite()
            && self.pressure_factor.is_finite()
            && self.carcass_stiffness_factor.is_finite()
            && self.tread_shear_factor.is_finite()
            && self.bristle_damping_factor.is_finite()
            && self.rolling_resistance_factor.is_finite()
    }
}

/// Data-free gate proving tire failure labels come from thermal/wear state.
///
/// This packages the condition-report and material-response checks used by the
/// viewer: cold, healthy, hot/overheated, worn, and abused tire behavior must
/// be derived from [`TireThermalLayers`] plus [`TireWearState`], then flow
/// through the reusable material-response bridge into force output. It is not
/// a rubber-chemistry model and does not fit any external tire data.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireConditionFailureModeReport {
    pub cold_level_matches: bool,
    pub healthy_level_matches: bool,
    pub hot_or_overheated_level_matches: bool,
    pub abused_level_matches: bool,
    pub cold_grip_limited: bool,
    pub overheated_grip_loss: bool,
    pub carcass_softened: bool,
    pub worn_tread_limited: bool,
    pub abused_flag: bool,
    pub healthy_grip_factor: f64,
    pub cold_grip_factor: f64,
    pub abused_grip_factor: f64,
    pub healthy_mu_factor: f64,
    pub cold_mu_factor: f64,
    pub abused_mu_factor: f64,
    pub healthy_carcass_stiffness_factor: f64,
    pub abused_carcass_stiffness_factor: f64,
    pub healthy_rolling_resistance: f64,
    pub abused_rolling_resistance: f64,
    pub healthy_force_n: f64,
    pub abused_force_n: f64,
    pub finite_outputs: bool,
}

impl TireConditionFailureModeReport {
    pub fn is_finite(self) -> bool {
        self.healthy_grip_factor.is_finite()
            && self.cold_grip_factor.is_finite()
            && self.abused_grip_factor.is_finite()
            && self.healthy_mu_factor.is_finite()
            && self.cold_mu_factor.is_finite()
            && self.abused_mu_factor.is_finite()
            && self.healthy_carcass_stiffness_factor.is_finite()
            && self.abused_carcass_stiffness_factor.is_finite()
            && self.healthy_rolling_resistance.is_finite()
            && self.abused_rolling_resistance.is_finite()
            && self.healthy_force_n.is_finite()
            && self.abused_force_n.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.cold_level_matches
            && self.healthy_level_matches
            && self.hot_or_overheated_level_matches
            && self.abused_level_matches
            && self.cold_grip_limited
            && self.overheated_grip_loss
            && self.carcass_softened
            && self.worn_tread_limited
            && self.abused_flag
            && self.cold_grip_factor < self.healthy_grip_factor
            && self.abused_grip_factor < self.healthy_grip_factor
            && self.cold_mu_factor < self.healthy_mu_factor
            && self.abused_mu_factor < self.healthy_mu_factor
            && self.abused_carcass_stiffness_factor < self.healthy_carcass_stiffness_factor
            && self.abused_rolling_resistance > self.healthy_rolling_resistance
            && self.abused_force_n < self.healthy_force_n * 0.75
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "condition_failure_levels_state_derived",
                "cold/healthy/hot/abused labels should come from thermal and wear state",
                "level_flags_passed",
                bool_metric(
                    self.cold_level_matches
                        && self.healthy_level_matches
                        && self.hot_or_overheated_level_matches
                        && self.abused_level_matches,
                ),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_flags_state_derived",
                "condition response flags should expose cold, hot, worn, carcass-softened, and abused states",
                "response_flags_passed",
                bool_metric(
                    self.cold_grip_limited
                        && self.overheated_grip_loss
                        && self.carcass_softened
                        && self.worn_tread_limited
                        && self.abused_flag,
                ),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_cold_reduces_grip",
                "cold tire state should reduce grip factor versus healthy state",
                "healthy_minus_cold_grip_factor",
                self.healthy_grip_factor - self.cold_grip_factor,
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_cold_reduces_mu",
                "cold tire state should reduce live material friction through material_response",
                "healthy_minus_cold_mu_factor",
                self.healthy_mu_factor - self.cold_mu_factor,
                1.0e-3,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_abuse_reduces_mu",
                "abused worn tire should reduce live material friction through material_response",
                "healthy_minus_abused_mu_factor",
                self.healthy_mu_factor - self.abused_mu_factor,
                0.20,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_abuse_softens_carcass",
                "abused worn tire should reduce carcass stiffness through material_response",
                "healthy_minus_abused_carcass_factor",
                self.healthy_carcass_stiffness_factor - self.abused_carcass_stiffness_factor,
                0.10,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_abuse_increases_rolling_loss",
                "abused worn tire should increase rolling resistance from state",
                "abused_minus_healthy_rolling_resistance",
                self.abused_rolling_resistance - self.healthy_rolling_resistance,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_most(
                "condition_failure_abuse_reduces_force",
                "state-driven abused/worn response should reduce tractive force",
                "abused_over_healthy_force",
                self.abused_force_n / self.healthy_force_n.max(1.0e-9),
                0.75,
            ),
            TireRigAcceptanceCheck::at_least(
                "condition_failure_outputs_finite",
                "condition failure-mode outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
        ]
    }
}

pub fn tire_condition_failure_mode_report() -> TireConditionFailureModeReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let fresh = TireWearState::default();
    let worn = TireWearState {
        depth_m: 0.0075,
        ..Default::default()
    };
    let cold_thermal = TireThermalLayers {
        surface_c: -5.0,
        tread_c: 0.0,
        carcass_c: 8.0,
        ..Default::default()
    };
    let healthy_thermal = TireThermalLayers {
        surface_c: 80.0,
        tread_c: 78.0,
        carcass_c: 62.0,
        ..Default::default()
    };
    let hot_thermal = TireThermalLayers {
        surface_c: 145.0,
        tread_c: 130.0,
        carcass_c: 115.0,
        ..Default::default()
    };
    let abused_thermal = TireThermalLayers {
        surface_c: 225.0,
        tread_c: 195.0,
        carcass_c: 185.0,
        ..Default::default()
    };

    let cold_condition = cold_thermal.condition_report(&fresh);
    let healthy_condition = healthy_thermal.condition_report(&fresh);
    let hot_condition = hot_thermal.condition_report(&fresh);
    let abused_condition = abused_thermal.condition_report(&worn);
    let cold = cold_thermal.material_response(&fresh, &params, &material);
    let healthy = healthy_thermal.material_response(&fresh, &params, &material);
    let abused = abused_thermal.material_response(&worn, &params, &material);

    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 4_200.0;
    let speed = 14.0;
    let healthy_force = tire_rig_point(
        &healthy.params,
        &terrain,
        &healthy.material,
        load_n,
        speed,
        0.18,
        0.0,
        0.0,
    )
    .telemetry
    .longitudinal_force_n
    .abs();
    let abused_force = tire_rig_point(
        &abused.params,
        &terrain,
        &abused.material,
        load_n,
        speed,
        0.18,
        0.0,
        0.0,
    )
    .telemetry
    .longitudinal_force_n
    .abs();
    let finite_outputs = cold_condition.is_finite()
        && healthy_condition.is_finite()
        && hot_condition.is_finite()
        && abused_condition.is_finite()
        && cold.is_finite()
        && healthy.is_finite()
        && abused.is_finite()
        && healthy_force.is_finite()
        && abused_force.is_finite();

    TireConditionFailureModeReport {
        cold_level_matches: cold_condition.level == TireConditionLevel::Cold,
        healthy_level_matches: healthy_condition.level == TireConditionLevel::Healthy,
        hot_or_overheated_level_matches: matches!(
            hot_condition.level,
            TireConditionLevel::Hot | TireConditionLevel::Overheated
        ),
        abused_level_matches: abused_condition.level == TireConditionLevel::Abused,
        cold_grip_limited: cold.cold_grip_limited,
        overheated_grip_loss: abused.overheated_grip_loss,
        carcass_softened: abused.carcass_softened,
        worn_tread_limited: abused.worn_tread_limited,
        abused_flag: abused.abused,
        healthy_grip_factor: healthy_condition.grip_factor,
        cold_grip_factor: cold_condition.grip_factor,
        abused_grip_factor: abused_condition.grip_factor,
        healthy_mu_factor: healthy.mu_factor,
        cold_mu_factor: cold.mu_factor,
        abused_mu_factor: abused.mu_factor,
        healthy_carcass_stiffness_factor: healthy.carcass_stiffness_factor,
        abused_carcass_stiffness_factor: abused.carcass_stiffness_factor,
        healthy_rolling_resistance: healthy.material.rolling_resistance,
        abused_rolling_resistance: abused.material.rolling_resistance,
        healthy_force_n: healthy_force,
        abused_force_n: abused_force,
        finite_outputs,
    }
}
