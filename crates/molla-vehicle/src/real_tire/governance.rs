//! Preset-governance checks for hand-tuned tire/terrain constants.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::{csv_token, finite_or};
use super::{
    RealTireParams, SurfaceFilmState, SurfaceMaterial, TetraTireFemParams, TireModelProvenance,
    TireThermalLayers,
};

/// One preset-governance check for hand-tuned tire/terrain constants.
///
/// These ranges are intentionally broad engineering guardrails, not
/// calibration targets. They make preset edits reviewable: each field carries
/// units, provenance, a sane range, and pass/fail status.
#[derive(Clone, Debug)]
pub struct TirePresetGovernanceCheck {
    pub group: String,
    pub preset: String,
    pub field: String,
    pub units: String,
    pub provenance: TireModelProvenance,
    pub value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub passed: bool,
    pub note: String,
}

impl TirePresetGovernanceCheck {
    pub fn range(
        group: &str,
        preset: &str,
        field: &str,
        units: &str,
        provenance: TireModelProvenance,
        value: f64,
        min_value: f64,
        max_value: f64,
        note: &str,
    ) -> Self {
        let value = finite_or(value, f64::NAN);
        let min_value = finite_or(min_value.min(max_value), 0.0);
        let max_value = finite_or(max_value.max(min_value), min_value);
        Self {
            group: csv_token(group),
            preset: csv_token(preset),
            field: csv_token(field),
            units: csv_token(units),
            provenance,
            value,
            min_value,
            max_value,
            passed: value.is_finite() && value >= min_value && value <= max_value,
            note: csv_token(note),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.value.is_finite() && self.min_value.is_finite() && self.max_value.is_finite()
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "{}.{}.{} units={} provenance={} value={:.6} range=[{:.6},{:.6}] passed={} note={}",
            self.group,
            self.preset,
            self.field,
            self.units,
            self.provenance.label(),
            self.value,
            self.min_value,
            self.max_value,
            self.passed,
            self.note
        )
    }
}

/// Non-calibration preset governance report for the real-tire stack.
#[derive(Clone, Debug, Default)]
pub struct TirePresetGovernanceReport {
    pub checks: Vec<TirePresetGovernanceCheck>,
}

impl TirePresetGovernanceReport {
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|check| check.passed)
    }

    pub fn is_finite(&self) -> bool {
        self.checks.iter().all(TirePresetGovernanceCheck::is_finite)
    }

    pub fn checks_for_preset<'a>(
        &'a self,
        group: &str,
        preset: &str,
    ) -> impl Iterator<Item = &'a TirePresetGovernanceCheck> {
        let group = csv_token(group);
        let preset = csv_token(preset);
        self.checks
            .iter()
            .filter(move |check| check.group == group && check.preset == preset)
    }

    pub fn to_text_report(&self) -> String {
        let mut out = format!(
            "tire_preset_governance non_calibration=true passed={}",
            self.passed()
        );
        for check in &self.checks {
            out.push('\n');
            out.push_str(&check.to_text_line());
        }
        out
    }
}

/// Build the current preset-governance checklist.
///
/// Add fields here when a new production tire/terrain preset is introduced.
/// A preset change should keep this report passing or include a conscious
/// range/provenance update beside the changed field.
pub fn tire_preset_governance_report() -> TirePresetGovernanceReport {
    let mut checks = Vec::new();
    let mut add = |group: &str,
                   preset: &str,
                   field: &str,
                   units: &str,
                   provenance: TireModelProvenance,
                   value: f64,
                   min_value: f64,
                   max_value: f64,
                   note: &str| {
        checks.push(TirePresetGovernanceCheck::range(
            group, preset, field, units, provenance, value, min_value, max_value, note,
        ));
    };

    for (name, params) in [
        ("passenger_radial", RealTireParams::passenger_radial()),
        ("offroad_soft", RealTireParams::offroad_soft()),
        (
            "high_pressure_reference",
            RealTireParams::high_pressure_reference(),
        ),
    ] {
        add(
            "RealTireParams",
            name,
            "radius",
            "m",
            TireModelProvenance::Physical,
            params.radius,
            0.15,
            0.80,
            "loaded tire geometry",
        );
        add(
            "RealTireParams",
            name,
            "width",
            "m",
            TireModelProvenance::Physical,
            params.width,
            0.08,
            0.60,
            "contact tread width",
        );
        add(
            "RealTireParams",
            name,
            "pressure_pa",
            "Pa",
            TireModelProvenance::HandTuned,
            params.pressure_pa,
            20_000.0,
            600_000.0,
            "inflation pressure preset",
        );
        add(
            "RealTireParams",
            name,
            "segments",
            "count",
            TireModelProvenance::NumericalStability,
            params.segments as f64,
            8.0,
            256.0,
            "brush contact resolution",
        );
        add(
            "RealTireParams",
            name,
            "carcass_normal_stiffness",
            "Pa_per_m",
            TireModelProvenance::HandTuned,
            params.carcass_normal_stiffness,
            5.0e4,
            5.0e6,
            "vertical carcass support",
        );
        add(
            "RealTireParams",
            name,
            "tread_shear_stiffness",
            "N_per_m3",
            TireModelProvenance::HandTuned,
            params.tread_shear_stiffness,
            1.0e5,
            5.0e7,
            "brush tread shear stiffness",
        );
    }

    for (name, material) in [
        ("dry_asphalt", SurfaceMaterial::dry_asphalt()),
        ("loose_dirt", SurfaceMaterial::loose_dirt()),
        ("soft_soil", SurfaceMaterial::soft_soil()),
        ("ice", SurfaceMaterial::ice()),
    ] {
        add(
            "SurfaceMaterial",
            name,
            "mu_static",
            "1",
            TireModelProvenance::HandTuned,
            material.mu_static,
            0.0,
            2.0,
            "static friction coefficient",
        );
        add(
            "SurfaceMaterial",
            name,
            "mu_dynamic",
            "1",
            TireModelProvenance::HandTuned,
            material.mu_dynamic,
            0.0,
            material.mu_static.max(0.0),
            "dynamic friction must not exceed static friction",
        );
        add(
            "SurfaceMaterial",
            name,
            "rolling_resistance",
            "1",
            TireModelProvenance::HandTuned,
            material.rolling_resistance,
            0.0,
            0.25,
            "dimensionless rolling loss",
        );
        add(
            "SurfaceMaterial",
            name,
            "cohesion",
            "Pa",
            TireModelProvenance::HandTuned,
            material.cohesion,
            0.0,
            25_000.0,
            "soil-like cohesive shear",
        );
    }

    for (name, film) in [
        ("dry", SurfaceFilmState::dry()),
        ("damp", SurfaceFilmState::damp()),
        ("wet_road", SurfaceFilmState::wet_road()),
        ("standing_water", SurfaceFilmState::standing_water()),
    ] {
        add(
            "SurfaceFilmState",
            name,
            "water_depth_m",
            "m",
            TireModelProvenance::Physical,
            film.water_depth_m,
            0.0,
            0.02,
            "road film depth",
        );
        add(
            "SurfaceFilmState",
            name,
            "fresh_tread_void_depth_m",
            "m",
            TireModelProvenance::HandTuned,
            film.fresh_tread_void_depth_m,
            0.001,
            0.020,
            "drainage capacity proxy",
        );
        add(
            "SurfaceFilmState",
            name,
            "drainage_speed_mps",
            "m_per_s",
            TireModelProvenance::HandTuned,
            film.drainage_speed_mps,
            1.0,
            80.0,
            "film speed response",
        );
    }

    for (name, thermal) in [
        ("passenger_road", TireThermalLayers::passenger_road()),
        (
            "fast_response_test",
            TireThermalLayers::fast_response_test(),
        ),
        (
            "heavy_duty_slow_response",
            TireThermalLayers::heavy_duty_slow_response(),
        ),
    ] {
        add(
            "TireThermalLayers",
            name,
            "surface_heat_capacity_j_per_k",
            "J_per_K",
            TireModelProvenance::HandTuned,
            thermal.surface_heat_capacity_j_per_k,
            100.0,
            10_000.0,
            "surface thermal inertia",
        );
        add(
            "TireThermalLayers",
            name,
            "carcass_heat_capacity_j_per_k",
            "J_per_K",
            TireModelProvenance::HandTuned,
            thermal.carcass_heat_capacity_j_per_k,
            1_000.0,
            100_000.0,
            "carcass thermal inertia",
        );
        add(
            "TireThermalLayers",
            name,
            "air_conductance_w_per_k",
            "W_per_K",
            TireModelProvenance::HandTuned,
            thermal.air_conductance_w_per_k,
            1.0,
            250.0,
            "air cooling rate",
        );
    }

    for (name, params) in [
        ("realtime_viewer", TetraTireFemParams::realtime_viewer()),
        ("offroad_soft", TetraTireFemParams::offroad_soft()),
        (
            "high_pressure_reference",
            TetraTireFemParams::high_pressure_reference(),
        ),
    ] {
        add(
            "TetraTireFemParams",
            name,
            "outer_radius",
            "m",
            TireModelProvenance::Physical,
            params.outer_radius,
            0.15,
            0.90,
            "FEM tire outer geometry",
        );
        add(
            "TetraTireFemParams",
            name,
            "pressure_pa",
            "Pa",
            TireModelProvenance::HandTuned,
            params.pressure_pa,
            20_000.0,
            600_000.0,
            "FEM inflation preset",
        );
        add(
            "TetraTireFemParams",
            name,
            "young_modulus",
            "Pa",
            TireModelProvenance::HandTuned,
            params.young_modulus,
            1.0e5,
            1.0e8,
            "rubber-body stiffness",
        );
        add(
            "TetraTireFemParams",
            name,
            "cavity_carcass_coupling",
            "1",
            TireModelProvenance::HandTuned,
            params.cavity_carcass_coupling,
            0.02,
            0.75,
            "audited gas-pressure coupling fraction before full bead/rim contact",
        );
        add(
            "TetraTireFemParams",
            name,
            "xpbd_volume_compliance",
            "m2_per_N",
            TireModelProvenance::NumericalStability,
            params.xpbd_volume_compliance,
            0.0,
            1.0e-5,
            "volume stabilization compliance",
        );
        add(
            "TetraTireFemParams",
            name,
            "xpbd_structural_iterations",
            "count",
            TireModelProvenance::NumericalStability,
            params.xpbd_structural_iterations as f64,
            0.0,
            16.0,
            "structural XPBD solver budget",
        );
        add(
            "TetraTireFemParams",
            name,
            "xpbd_sidewall_dihedral_compliance",
            "m2_per_N",
            TireModelProvenance::NumericalStability,
            params.xpbd_sidewall_dihedral_compliance,
            0.0,
            5.0e-2,
            "sidewall dihedral-angle stabilization compliance",
        );
        add(
            "TetraTireFemParams",
            name,
            "semi_implicit_stiffness_scale",
            "unitless",
            TireModelProvenance::NumericalStability,
            params.semi_implicit_stiffness_scale,
            0.0,
            4.0,
            "diagonal semi-implicit stiffness scale",
        );
    }

    for (name, soil) in [
        (
            "compacted_dirt_patch",
            crate::deformable::DeformableTerrain::compacted_dirt_patch(
                [-1.0, -1.0],
                0.25,
                5,
                5,
                0.0,
            ),
        ),
        (
            "sandy_loam_patch",
            crate::deformable::DeformableTerrain::sandy_loam_patch([-1.0, -1.0], 0.25, 5, 5, 0.0),
        ),
        (
            "soft_clay_patch",
            crate::deformable::DeformableTerrain::soft_clay_patch([-1.0, -1.0], 0.25, 5, 5, 0.0),
        ),
    ] {
        add(
            "DeformableTerrain",
            name,
            "stiffness",
            "Pa_per_m_n",
            TireModelProvenance::HandTuned,
            soil.stiffness,
            1.0e4,
            1.0e7,
            "Bekker pressure-sinkage stiffness",
        );
        add(
            "DeformableTerrain",
            name,
            "max_sinkage",
            "m",
            TireModelProvenance::HandTuned,
            soil.max_sinkage,
            0.005,
            0.50,
            "rut/sinkage clamp",
        );
        add(
            "DeformableTerrain",
            name,
            "friction",
            "1",
            TireModelProvenance::HandTuned,
            soil.friction,
            0.05,
            1.50,
            "soil surface shear factor",
        );
        add(
            "DeformableTerrain",
            name,
            "bekker_n",
            "1",
            TireModelProvenance::HandTuned,
            soil.bekker_n,
            0.40,
            2.00,
            "Bekker exponent",
        );
    }

    TirePresetGovernanceReport { checks }
}
