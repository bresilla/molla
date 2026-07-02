//! Tire thermal/wear state types and deterministic thermal/wear rig runs.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::Terrain;

use super::internal::{bool_metric, finite_nonnegative, finite_or, finite_positive_or};
use super::{
    RealTireParams, RealTireTelemetrySample, SurfaceMaterial, TireConditionMaterialResponse,
    TireConditionReport, TireRigAcceptanceCheck, tire_rig_point,
};

/// Stage (j) — first-order tire thermal model.
///
/// Integrates the dissipated power reported by `evaluate_brush_tire`
/// against a passive Newton-cooling sink. Optional: feed the resulting
/// `temperature_c` back into `SurfaceMaterial::mu_*` via your own
/// curve. The defaults are reasonable for a road tire: ~2 kJ/(kg·K)
/// rubber heat capacity × ~10 kg active tread mass = ~2e4 J/K, and a
/// ~15 W/K cooling coefficient to ambient.
#[derive(Clone, Copy, Debug)]
pub struct TireThermalState {
    pub temperature_c: f64,
    pub ambient_c: f64,
    pub heat_capacity_j_per_k: f64,
    pub cooling_w_per_k: f64,
}

impl Default for TireThermalState {
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            ambient_c: 20.0,
            heat_capacity_j_per_k: 2.0e4,
            cooling_w_per_k: 15.0,
        }
    }
}

impl TireThermalState {
    /// Advance the thermal state by `dt` seconds with `power_w` heat
    /// input (from `RealTireContact::dissipated_power`).
    pub fn step(&mut self, power_w: f64, dt: f64) {
        let cool = self.cooling_w_per_k * (self.temperature_c - self.ambient_c);
        let net_w = power_w - cool;
        self.temperature_c += net_w / self.heat_capacity_j_per_k.max(1.0) * dt;
    }

    /// Multiplicative correction on `μ` from the current tread
    /// temperature. Cold tires (≪ optimal) and over-heated tires
    /// (≫ optimal) both grip worse than the design window. Returns
    /// a value in `[mu_min_factor, 1.0]`.
    pub fn mu_factor(&self) -> f64 {
        let optimal_c = 80.0;
        let drop_per_c = 0.006;
        let mu_min = 0.55;
        let delta = (self.temperature_c - optimal_c).abs();
        (1.0 - delta * drop_per_c).clamp(mu_min, 1.0)
    }
}

/// Multi-layer tire thermal model.
///
/// This is the production-oriented replacement for the old single lumped
/// `TireThermalState`. It is still intentionally cheap: four thermal nodes
/// exchange heat through conductances, while separate heat inputs feed the
/// layers where the work physically enters the tire.
#[derive(Clone, Copy, Debug)]
pub struct TireThermalLayers {
    /// Outer tread/rubber skin temperature. Sliding heats this first and it
    /// cools fastest through road conduction and air convection.
    pub surface_c: f64,
    /// Bulk tread temperature under the surface skin.
    pub tread_c: f64,
    /// Sidewall/carcass/belt body temperature. Flexing and rolling losses
    /// mostly enter here and change pressure/stiffness slowly.
    pub carcass_c: f64,
    /// Wheel/rim thermal node. Brake heat enters here when supplied by the
    /// caller.
    pub rim_c: f64,
    pub ambient_c: f64,
    pub surface_heat_capacity_j_per_k: f64,
    pub tread_heat_capacity_j_per_k: f64,
    pub carcass_heat_capacity_j_per_k: f64,
    pub rim_heat_capacity_j_per_k: f64,
    pub road_conductance_w_per_k: f64,
    pub air_conductance_w_per_k: f64,
    pub surface_tread_conductance_w_per_k: f64,
    pub tread_carcass_conductance_w_per_k: f64,
    pub carcass_rim_conductance_w_per_k: f64,
}

impl Default for TireThermalLayers {
    fn default() -> Self {
        Self::passenger_road()
    }
}

impl TireThermalLayers {
    /// Hand-tuned road-tire thermal preset.
    ///
    /// Units are SI except temperatures in °C: heat capacities are J/K and
    /// conductances are W/K. This is the default cheap four-node model used by
    /// tire-rig benches and the viewer.
    pub const fn passenger_road() -> Self {
        Self {
            surface_c: 20.0,
            tread_c: 20.0,
            carcass_c: 20.0,
            rim_c: 20.0,
            ambient_c: 20.0,
            surface_heat_capacity_j_per_k: 800.0,
            tread_heat_capacity_j_per_k: 8.0e3,
            carcass_heat_capacity_j_per_k: 1.8e4,
            rim_heat_capacity_j_per_k: 3.0e4,
            road_conductance_w_per_k: 45.0,
            air_conductance_w_per_k: 18.0,
            surface_tread_conductance_w_per_k: 120.0,
            tread_carcass_conductance_w_per_k: 35.0,
            carcass_rim_conductance_w_per_k: 22.0,
        }
    }

    /// Lower thermal mass / higher convection preset for aggressive repeated
    /// slip tests where surface temperature should respond quickly.
    pub const fn fast_response_test() -> Self {
        Self {
            surface_heat_capacity_j_per_k: 520.0,
            tread_heat_capacity_j_per_k: 5.5e3,
            carcass_heat_capacity_j_per_k: 1.25e4,
            rim_heat_capacity_j_per_k: 2.5e4,
            road_conductance_w_per_k: 60.0,
            air_conductance_w_per_k: 28.0,
            surface_tread_conductance_w_per_k: 150.0,
            tread_carcass_conductance_w_per_k: 45.0,
            carcass_rim_conductance_w_per_k: 25.0,
            ..Self::passenger_road()
        }
    }

    /// Heavier tire/rim thermal preset with slower layer response.
    pub const fn heavy_duty_slow_response() -> Self {
        Self {
            surface_heat_capacity_j_per_k: 1_400.0,
            tread_heat_capacity_j_per_k: 1.4e4,
            carcass_heat_capacity_j_per_k: 3.2e4,
            rim_heat_capacity_j_per_k: 5.5e4,
            road_conductance_w_per_k: 38.0,
            air_conductance_w_per_k: 14.0,
            surface_tread_conductance_w_per_k: 105.0,
            tread_carcass_conductance_w_per_k: 30.0,
            carcass_rim_conductance_w_per_k: 20.0,
            ..Self::passenger_road()
        }
    }

    pub fn is_finite(&self) -> bool {
        self.surface_c.is_finite()
            && self.tread_c.is_finite()
            && self.carcass_c.is_finite()
            && self.rim_c.is_finite()
            && self.ambient_c.is_finite()
            && self.surface_heat_capacity_j_per_k.is_finite()
            && self.tread_heat_capacity_j_per_k.is_finite()
            && self.carcass_heat_capacity_j_per_k.is_finite()
            && self.rim_heat_capacity_j_per_k.is_finite()
            && self.road_conductance_w_per_k.is_finite()
            && self.air_conductance_w_per_k.is_finite()
            && self.surface_tread_conductance_w_per_k.is_finite()
            && self.tread_carcass_conductance_w_per_k.is_finite()
            && self.carcass_rim_conductance_w_per_k.is_finite()
    }

    /// Advance the layer model.
    ///
    /// Inputs are power in watts:
    ///
    /// - `sliding_power_w`: brush/contact slip work into surface/tread;
    /// - `carcass_flex_power_w`: cyclic casing/belt flex work;
    /// - `rolling_power_w`: rolling resistance work;
    /// - `brake_rim_power_w`: optional brake/wheel heat source;
    /// - `road_c`: current road surface temperature.
    pub fn step(
        &mut self,
        sliding_power_w: f64,
        carcass_flex_power_w: f64,
        rolling_power_w: f64,
        brake_rim_power_w: f64,
        road_c: f64,
        dt: f64,
    ) {
        let total_dt = finite_nonnegative(dt).min(10.0);
        if total_dt <= 0.0 {
            return;
        }

        let sliding_power_w = finite_nonnegative(sliding_power_w);
        let carcass_flex_power_w = finite_nonnegative(carcass_flex_power_w);
        let rolling_power_w = finite_nonnegative(rolling_power_w);
        let brake_rim_power_w = finite_nonnegative(brake_rim_power_w);
        let road_c = finite_or(road_c, self.ambient_c);
        let ambient_c = finite_or(self.ambient_c, 20.0);
        let c_surface = finite_positive_or(self.surface_heat_capacity_j_per_k, 800.0, 1.0);
        let c_tread = finite_positive_or(self.tread_heat_capacity_j_per_k, 8.0e3, 1.0);
        let c_carcass = finite_positive_or(self.carcass_heat_capacity_j_per_k, 1.8e4, 1.0);
        let c_rim = finite_positive_or(self.rim_heat_capacity_j_per_k, 3.0e4, 1.0);
        let g_road = finite_nonnegative(self.road_conductance_w_per_k);
        let g_air = finite_nonnegative(self.air_conductance_w_per_k);
        let g_st = finite_nonnegative(self.surface_tread_conductance_w_per_k);
        let g_tc = finite_nonnegative(self.tread_carcass_conductance_w_per_k);
        let g_cr = finite_nonnegative(self.carcass_rim_conductance_w_per_k);

        // Small explicit thermal substeps keep this stable if a caller gives
        // a large frame delta or unusually high conductance.
        let mut remaining = total_dt;
        while remaining > 1.0e-9 {
            let h = remaining.min(0.05);
            remaining -= h;

            let surface_c = finite_or(self.surface_c, ambient_c);
            let tread_c = finite_or(self.tread_c, ambient_c);
            let carcass_c = finite_or(self.carcass_c, ambient_c);
            let rim_c = finite_or(self.rim_c, ambient_c);

            // Heat-source routing. Sliding mostly heats the contact skin,
            // rolling/flex mostly heat the casing, and braking enters the rim.
            let q_surface_in = 0.70 * sliding_power_w + 0.15 * rolling_power_w;
            let q_tread_in =
                0.30 * sliding_power_w + 0.35 * rolling_power_w + 0.20 * carcass_flex_power_w;
            let q_carcass_in =
                0.80 * carcass_flex_power_w + 0.50 * rolling_power_w + 0.15 * brake_rim_power_w;
            let q_rim_in = 0.85 * brake_rim_power_w;

            // Positive q_ab means heat flows from a -> b.
            let q_st = g_st * (surface_c - tread_c);
            let q_tc = g_tc * (tread_c - carcass_c);
            let q_cr = g_cr * (carcass_c - rim_c);
            let q_surface_air = g_air * (surface_c - ambient_c);
            let q_surface_road = g_road * (surface_c - road_c);
            let q_carcass_air = 0.35 * g_air * (carcass_c - ambient_c);
            let q_rim_air = 0.25 * g_air * (rim_c - ambient_c);

            self.surface_c +=
                (q_surface_in - q_st - q_surface_air - q_surface_road) / c_surface * h;
            self.tread_c += (q_tread_in + q_st - q_tc) / c_tread * h;
            self.carcass_c += (q_carcass_in + q_tc - q_cr - q_carcass_air) / c_carcass * h;
            self.rim_c += (q_rim_in + q_cr - q_rim_air) / c_rim * h;
        }

        self.surface_c = finite_or(self.surface_c, ambient_c).clamp(-80.0, 350.0);
        self.tread_c = finite_or(self.tread_c, ambient_c).clamp(-80.0, 350.0);
        self.carcass_c = finite_or(self.carcass_c, ambient_c).clamp(-80.0, 350.0);
        self.rim_c = finite_or(self.rim_c, ambient_c).clamp(-80.0, 350.0);
    }

    /// Multiplicative correction on contact friction from thermal state.
    /// Surface temperature dominates quick grip; overheated carcass/tread add
    /// a slower penalty.
    pub fn mu_factor(&self) -> f64 {
        let surface_c = finite_or(self.surface_c, self.ambient_c);
        let tread_c = finite_or(self.tread_c, self.ambient_c);
        let carcass_c = finite_or(self.carcass_c, self.ambient_c);
        let surface_peak = 1.08 - (surface_c - 80.0).abs() * 0.006;
        let bulk_hot_penalty =
            ((0.5 * (tread_c + carcass_c) - 115.0).max(0.0) * 0.0025).clamp(0.0, 0.25);
        (surface_peak - bulk_hot_penalty).clamp(0.45, 1.08)
    }

    /// Slow pressure/stiffness feedback from carcass gas/body temperature.
    pub fn carcass_pressure_factor(&self) -> f64 {
        let carcass_c = finite_or(self.carcass_c, self.ambient_c);
        let ambient_c = finite_or(self.ambient_c, 20.0);
        (1.0 + (carcass_c - ambient_c) * 0.0015).clamp(0.85, 1.20)
    }

    /// Temperature multiplier for tread wear. Cold tires wear a little less;
    /// severe hot sliding tears rubber quickly.
    pub fn wear_temperature_factor(&self) -> f64 {
        TireWearState::surface_temperature_wear_factor(self.surface_c)
    }

    /// Compact display/readout temperature. This tracks the hottest rubber
    /// layer without letting a hot rim dominate the tire-body tint.
    pub fn rubber_peak_c(&self) -> f64 {
        finite_or(self.surface_c, self.ambient_c)
            .max(finite_or(self.tread_c, self.ambient_c))
            .max(finite_or(self.carcass_c, self.ambient_c))
    }

    /// State-derived tire condition/failure-mode report.
    ///
    /// This is deliberately not a fitted tire-life model. It turns existing
    /// physical state variables (layer temperatures + tread wear depth) into
    /// bounded multipliers and labels so viewers/tests can expose cold grip,
    /// overheating, worn/abused tires, and carcass-softening effects without
    /// inventing hidden viewer-side thresholds.
    pub fn condition_report(&self, wear: &TireWearState) -> TireConditionReport {
        TireConditionReport::from_thermal_and_wear(self, wear)
    }

    /// Apply the current thermal/wear condition to nominal tire parameters and
    /// road material. This is the preferred reusable path for thermal
    /// failure-mode feedback; it keeps grip, pressure/stiffness, tread shear,
    /// damping, and rolling loss tied to state variables.
    pub fn material_response(
        &self,
        wear: &TireWearState,
        params: &RealTireParams,
        material: &SurfaceMaterial,
    ) -> TireConditionMaterialResponse {
        TireConditionMaterialResponse::from_state(params, material, self, wear)
    }
}

/// Stage (j) — first-order tread-wear model.
///
/// Integrates a wear depth (m) from the brush dissipated power.
/// `wear_rate_m3_per_j` is the volume of rubber removed per joule of
/// sliding work. Multiply by `1.0 / patch_area` to get a depth rate;
/// for a single-state model we approximate the patch as a constant
/// `ref_patch_area` so the user just gets one number to monitor.
#[derive(Clone, Copy, Debug)]
pub struct TireWearState {
    pub depth_m: f64,
    pub wear_rate_m3_per_j: f64,
    pub ref_patch_area_m2: f64,
    pub max_depth_m: f64,
}

impl Default for TireWearState {
    fn default() -> Self {
        Self {
            depth_m: 0.0,
            // ~10⁻¹³ m³/J is a sane order of magnitude for road rubber
            // wear (Archard-style coefficient × shear stress).
            wear_rate_m3_per_j: 2.0e-13,
            ref_patch_area_m2: 0.02, // ~20 cm² typical loaded patch
            max_depth_m: 0.008,      // 8 mm tread, then bald
        }
    }
}

impl TireWearState {
    pub fn is_finite(&self) -> bool {
        self.depth_m.is_finite()
            && self.wear_rate_m3_per_j.is_finite()
            && self.ref_patch_area_m2.is_finite()
            && self.max_depth_m.is_finite()
    }

    /// Integrate wear depth over `dt` seconds from contact dissipation.
    pub fn step(&mut self, power_w: f64, dt: f64) {
        let volume_removed = self.wear_rate_m3_per_j * power_w.max(0.0) * dt;
        let depth_step = volume_removed / self.ref_patch_area_m2.max(1.0e-6);
        self.depth_m = (self.depth_m + depth_step).min(self.max_depth_m);
    }

    /// Temperature-aware wear integration. Use this with
    /// [`TireThermalLayers`] when available; it keeps the legacy `step`
    /// method as the cold/warm fallback.
    pub fn step_with_surface_temperature(
        &mut self,
        power_w: f64,
        surface_temperature_c: f64,
        dt: f64,
    ) {
        let factor = Self::surface_temperature_wear_factor(surface_temperature_c);
        self.step(finite_nonnegative(power_w) * factor, dt);
    }

    /// Integrate wear using a full tire-condition report. This keeps the
    /// failure-mode path tied to thermal/wear state instead of viewer-only
    /// flags: overheated/abused tires wear faster, while cold/healthy tires
    /// use the bounded report multiplier.
    pub fn step_with_condition_report(
        &mut self,
        power_w: f64,
        report: TireConditionReport,
        dt: f64,
    ) {
        self.step(finite_nonnegative(power_w) * report.wear_rate_factor, dt);
    }

    /// Wear-rate multiplier from surface temperature only.
    pub fn surface_temperature_wear_factor(surface_temperature_c: f64) -> f64 {
        let t = finite_or(surface_temperature_c, 20.0);
        if t < 70.0 {
            (0.65 + (t + 20.0).max(0.0) / 90.0 * 0.35).clamp(0.45, 1.0)
        } else {
            (1.0 + ((t - 70.0) / 45.0).powi(2)).clamp(1.0, 6.0)
        }
    }

    /// Multiplicative correction on `μ` from worn tread. A fully-worn
    /// tire (`depth_m == max_depth_m`) grips ~30 % worse than new.
    pub fn mu_factor(&self) -> f64 {
        let ratio = (self.depth_m / self.max_depth_m.max(1.0e-9)).clamp(0.0, 1.0);
        1.0 - 0.30 * ratio
    }
}

/// One phase in a deterministic thermal/wear tire-rig run.
#[derive(Clone, Copy, Debug)]
pub struct TireRigThermalWearPhase {
    pub duration_s: f64,
    pub load_n: f64,
    pub speed_m_per_s: f64,
    pub slip_ratio_command: f64,
    pub road_temperature_c: f64,
}

impl TireRigThermalWearPhase {
    pub const fn new(
        duration_s: f64,
        load_n: f64,
        speed_m_per_s: f64,
        slip_ratio_command: f64,
        road_temperature_c: f64,
    ) -> Self {
        Self {
            duration_s,
            load_n,
            speed_m_per_s,
            slip_ratio_command,
            road_temperature_c,
        }
    }

    pub const fn sustained_slip(
        duration_s: f64,
        load_n: f64,
        speed_m_per_s: f64,
        slip_ratio_command: f64,
    ) -> Self {
        Self::new(duration_s, load_n, speed_m_per_s, slip_ratio_command, 20.0)
    }

    pub const fn cooling_recovery(duration_s: f64, load_n: f64) -> Self {
        Self::new(duration_s, load_n, 0.0, 0.0, 20.0)
    }
}

/// One sample from a deterministic thermal/wear tire-rig run.
#[derive(Clone, Debug)]
pub struct TireRigThermalWearPoint {
    pub time_s: f64,
    pub phase_index: usize,
    pub thermal: TireThermalLayers,
    pub wear: TireWearState,
    pub thermal_mu_factor: f64,
    pub wear_mu_factor: f64,
    pub pressure_factor: f64,
    pub telemetry: RealTireTelemetrySample,
}

impl TireRigThermalWearPoint {
    pub fn is_finite(&self) -> bool {
        self.time_s.is_finite()
            && self.thermal.surface_c.is_finite()
            && self.thermal.tread_c.is_finite()
            && self.thermal.carcass_c.is_finite()
            && self.thermal.rim_c.is_finite()
            && self.wear.depth_m.is_finite()
            && self.thermal_mu_factor.is_finite()
            && self.wear_mu_factor.is_finite()
            && self.pressure_factor.is_finite()
            && self.telemetry.is_finite()
    }
}

/// Qualitative thermal/condition validation for non-calibration rig runs.
///
/// This is **not** rubber chemistry fitting. It turns a thermal/wear replay
/// into broad sanity metrics: heat must build under abuse, cooling should lower
/// the surface temperature, tread wear must not heal, and condition multipliers
/// must stay finite/bounded. Use this as the regression gate until explicit
/// calibration/chemistry data is in scope.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireThermalConditionSanityReport {
    pub sample_count: usize,
    pub peak_surface_c: f64,
    pub final_surface_c: f64,
    pub peak_tread_c: f64,
    pub final_tread_c: f64,
    pub max_wear_depth_m: f64,
    pub final_wear_depth_m: f64,
    pub min_mu_factor: f64,
    pub max_pressure_factor: f64,
    pub cooling_drop_c: f64,
    pub wear_non_decreasing: bool,
    pub finite: bool,
}

impl TireThermalConditionSanityReport {
    pub fn is_finite(self) -> bool {
        self.peak_surface_c.is_finite()
            && self.final_surface_c.is_finite()
            && self.peak_tread_c.is_finite()
            && self.final_tread_c.is_finite()
            && self.max_wear_depth_m.is_finite()
            && self.final_wear_depth_m.is_finite()
            && self.min_mu_factor.is_finite()
            && self.max_pressure_factor.is_finite()
            && self.cooling_drop_c.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite
            && self.sample_count > 0
            && self.peak_surface_c >= self.final_surface_c
            && self.cooling_drop_c >= 0.0
            && self.wear_non_decreasing
            && self.final_wear_depth_m >= 0.0
            && self.min_mu_factor > 0.0
            && self.max_pressure_factor.is_finite()
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_sanity_finite",
                "thermal/wear condition replay should stay finite",
                "finite_outputs",
                bool_metric(self.finite && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_has_samples",
                "thermal/wear condition replay should contain samples",
                "sample_count",
                self.sample_count as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_cooling_drop_nonnegative",
                "cooling phase should not heat the surface beyond the abuse peak",
                "cooling_drop_c",
                self.cooling_drop_c,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_wear_non_healing",
                "tread wear should not heal during cooling",
                "wear_non_decreasing",
                bool_metric(self.wear_non_decreasing),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_mu_factor_positive",
                "state-derived thermal/wear grip multiplier should remain positive",
                "min_mu_factor",
                self.min_mu_factor,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_condition_pressure_factor_finite",
                "state-derived carcass pressure factor should stay finite and positive",
                "max_pressure_factor",
                self.max_pressure_factor,
                1.0e-6,
            ),
        ]
    }
}

pub fn tire_thermal_condition_sanity_report(
    run: &[TireRigThermalWearPoint],
) -> TireThermalConditionSanityReport {
    if run.is_empty() {
        return TireThermalConditionSanityReport::default();
    }
    let mut peak_surface_c = f64::NEG_INFINITY;
    let mut peak_tread_c = f64::NEG_INFINITY;
    let mut max_wear_depth_m = 0.0_f64;
    let mut min_mu_factor = f64::INFINITY;
    let mut max_pressure_factor = 0.0_f64;
    let mut wear_non_decreasing = true;
    let mut finite = true;
    let mut prev_wear = 0.0_f64;
    for (i, point) in run.iter().enumerate() {
        finite &= point.is_finite();
        peak_surface_c = peak_surface_c.max(point.thermal.surface_c);
        peak_tread_c = peak_tread_c.max(point.thermal.tread_c);
        max_wear_depth_m = max_wear_depth_m.max(point.wear.depth_m);
        min_mu_factor = min_mu_factor.min(point.thermal_mu_factor.min(point.wear_mu_factor));
        max_pressure_factor = max_pressure_factor.max(point.pressure_factor);
        if i > 0 && point.wear.depth_m + 1.0e-12 < prev_wear {
            wear_non_decreasing = false;
        }
        prev_wear = point.wear.depth_m;
    }
    let final_point = run.last().expect("checked non-empty");
    TireThermalConditionSanityReport {
        sample_count: run.len(),
        peak_surface_c,
        final_surface_c: final_point.thermal.surface_c,
        peak_tread_c,
        final_tread_c: final_point.thermal.tread_c,
        max_wear_depth_m,
        final_wear_depth_m: final_point.wear.depth_m,
        min_mu_factor,
        max_pressure_factor,
        cooling_drop_c: peak_surface_c - final_point.thermal.surface_c,
        wear_non_decreasing,
        finite,
    }
}

/// Deterministic sustained-slip thermal/wear rig run.
///
/// Each step evaluates the brush contact using thermal/wear-adjusted material
/// and pressure, then feeds the resulting dissipated power into
/// [`TireThermalLayers`] and [`TireWearState`]. This is a regression bench,
/// not fitted external calibration.
pub fn tire_rig_thermal_wear_run(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    dt_s: f64,
    steps: usize,
) -> Vec<TireRigThermalWearPoint> {
    tire_rig_thermal_wear_cycle(
        params,
        terrain,
        material,
        dt_s,
        &[TireRigThermalWearPhase::new(
            steps as f64 * finite_nonnegative(dt_s),
            load_n,
            speed_m_per_s,
            slip_ratio_command,
            20.0,
        )],
    )
}

/// Deterministic multi-phase thermal/wear rig run.
///
/// Use this for heat-then-cool benches: each phase can command a different
/// load, road speed, slip ratio, and road temperature while the same thermal
/// and wear state carries through the whole run.
pub fn tire_rig_thermal_wear_cycle(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    dt_s: f64,
    phases: &[TireRigThermalWearPhase],
) -> Vec<TireRigThermalWearPoint> {
    let dt_s = finite_nonnegative(dt_s).min(1.0);
    if phases.is_empty() || dt_s <= 0.0 {
        return Vec::new();
    }
    let mut thermal = TireThermalLayers::default();
    let mut wear = TireWearState::default();
    let estimated_steps = phases
        .iter()
        .map(|phase| (finite_nonnegative(phase.duration_s) / dt_s).ceil() as usize)
        .sum();
    let mut out = Vec::with_capacity(estimated_steps);
    let mut time_s = 0.0;
    for (phase_index, phase) in phases.iter().enumerate() {
        let phase_steps = (finite_nonnegative(phase.duration_s) / dt_s).ceil() as usize;
        for _ in 0..phase_steps {
            let response = thermal.material_response(&wear, params, material);
            let live_params = response.params;
            let live_material = response.material;
            let pressure_factor = response.pressure_factor;
            let thermal_mu_factor = thermal.mu_factor();
            let wear_mu_factor = wear.mu_factor();

            let load_n = finite_nonnegative(phase.load_n);
            let speed_m_per_s = finite_or(phase.speed_m_per_s, 0.0);
            let rig = tire_rig_point(
                &live_params,
                terrain,
                &live_material,
                load_n,
                speed_m_per_s,
                finite_or(phase.slip_ratio_command, 0.0),
                0.0,
                0.0,
            );
            let contact_power = rig.telemetry.dissipated_power_w;
            let rolling_power =
                (rig.telemetry.rolling_resistance_force_n * speed_m_per_s.abs()).abs();
            let carcass_flex_power = load_n * speed_m_per_s.abs() * 0.0025;
            thermal.step(
                contact_power,
                carcass_flex_power,
                rolling_power,
                0.0,
                finite_or(phase.road_temperature_c, thermal.ambient_c),
                dt_s,
            );
            wear.step_with_surface_temperature(contact_power, thermal.surface_c, dt_s);
            time_s += dt_s;
            out.push(TireRigThermalWearPoint {
                time_s,
                phase_index,
                thermal,
                wear,
                thermal_mu_factor,
                wear_mu_factor,
                pressure_factor,
                telemetry: rig.telemetry,
            });
        }
    }
    out
}
