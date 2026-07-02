//! Tire-rig sensitivity sweep machinery + standard map.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, HeightfieldTerrain, Terrain};

use super::super::internal::{csv_token, finite_nonnegative, finite_or};
use super::super::{
    RealTireParams, RealTireTelemetrySample, SurfaceFilmState, SurfaceMaterial, TireRigSnapshot,
    TireRigStepResponsePoint, TireThermalLayers, TireWearState, solve_loaded_tire_hub_height,
    tire_rig_point, tire_rig_point_at,
};

pub(crate) fn tire_rig_step_response_snapshot(
    scenario_id: &str,
    tire_preset: &str,
    surface_preset: &str,
    physics_path: &str,
    params: &RealTireParams,
    terrain: &dyn Terrain,
    load_n: f64,
    speed_m_per_s: f64,
    point: &TireRigStepResponsePoint,
) -> TireRigSnapshot {
    let hub_height_m = solve_loaded_tire_hub_height(params, terrain, 0.0, 0.0, load_n);
    TireRigSweepPoint {
        load_n: finite_nonnegative(load_n),
        pressure_pa: finite_nonnegative(params.pressure_pa),
        hub_height_m,
        deflection_m: (params.radius - (hub_height_m - terrain.height(0.0, 0.0))).max(0.0),
        speed_m_per_s: finite_or(speed_m_per_s, 0.0),
        slip_ratio_command: finite_or(point.effective_slip_ratio, 0.0),
        slip_angle_command_rad: finite_or(point.effective_slip_angle_rad, 0.0),
        camber_angle_command_rad: 0.0,
        telemetry: point.telemetry.clone(),
    }
    .snapshot_on_terrain(
        scenario_id,
        tire_preset,
        surface_preset,
        "flat",
        physics_path,
        terrain,
        0.0,
        0.0,
    )
}

/// One deterministic tire-rig sweep sample.
///
/// This captures the command inputs and resulting model telemetry for
/// repeatable non-calibration validation. The values are not fitted to
/// external data; they are regression/inspection points for the physical
/// model.
#[derive(Clone, Debug)]
pub struct TireRigSweepPoint {
    pub load_n: f64,
    pub pressure_pa: f64,
    pub hub_height_m: f64,
    pub deflection_m: f64,
    pub speed_m_per_s: f64,
    pub slip_ratio_command: f64,
    pub slip_angle_command_rad: f64,
    pub camber_angle_command_rad: f64,
    pub telemetry: RealTireTelemetrySample,
}

impl TireRigSweepPoint {
    pub fn is_finite(&self) -> bool {
        self.load_n.is_finite()
            && self.pressure_pa.is_finite()
            && self.hub_height_m.is_finite()
            && self.deflection_m.is_finite()
            && self.speed_m_per_s.is_finite()
            && self.slip_ratio_command.is_finite()
            && self.slip_angle_command_rad.is_finite()
            && self.camber_angle_command_rad.is_finite()
            && self.telemetry.is_finite()
    }

    pub fn snapshot_on_terrain(
        &self,
        scenario_id: &str,
        tire_preset: &str,
        surface_preset: &str,
        terrain_preset: &str,
        physics_path: &str,
        terrain: &dyn Terrain,
        x_m: f64,
        z_m: f64,
    ) -> TireRigSnapshot {
        TireRigSnapshot::from_sweep_point_on_terrain(
            scenario_id,
            tire_preset,
            surface_preset,
            terrain_preset,
            physics_path,
            self,
            terrain,
            x_m,
            z_m,
        )
    }
}

/// Axis varied by a deterministic, data-free tire sensitivity map.
///
/// These are not calibration coefficients. Each axis changes one physical
/// state/preset input and records whether the native tire model reacts in the
/// expected qualitative direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireRigSensitivityAxis {
    Pressure,
    Load,
    DrySpeed,
    WetSpeed,
    Camber,
    SurfaceTemperature,
    WaterFilm,
    TreadWear,
    SoftSoilSurface,
    SoilStiffness,
    ObstacleHeight,
    ContactResolution,
}

impl TireRigSensitivityAxis {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pressure => "pressure",
            Self::Load => "load",
            Self::DrySpeed => "dry_speed",
            Self::WetSpeed => "wet_speed",
            Self::Camber => "camber",
            Self::SurfaceTemperature => "surface_temperature",
            Self::WaterFilm => "water_film",
            Self::TreadWear => "tread_wear",
            Self::SoftSoilSurface => "soft_soil_surface",
            Self::SoilStiffness => "soil_stiffness",
            Self::ObstacleHeight => "obstacle_height",
            Self::ContactResolution => "contact_resolution",
        }
    }
}

/// One qualitative sensitivity-map comparison.
///
/// `baseline_metric` and `comparison_metric` are unit-labelled by
/// `metric_label`. `expected_direction` is `+1` when the comparison should be
/// larger than the baseline, and `-1` when it should be smaller. The embedded
/// rig points make the comparison auditable without importing measured data or
/// fitting target curves.
#[derive(Clone, Debug)]
pub struct TireRigSensitivityPoint {
    pub axis: TireRigSensitivityAxis,
    pub baseline_input: f64,
    pub comparison_input: f64,
    pub input_unit: &'static str,
    pub metric_label: &'static str,
    pub baseline_metric: f64,
    pub comparison_metric: f64,
    pub expected_direction: f64,
    pub expectation: &'static str,
    pub baseline: TireRigSweepPoint,
    pub comparison: TireRigSweepPoint,
    pub passed: bool,
}

impl TireRigSensitivityPoint {
    pub fn is_finite(&self) -> bool {
        self.baseline_input.is_finite()
            && self.comparison_input.is_finite()
            && self.baseline_metric.is_finite()
            && self.comparison_metric.is_finite()
            && self.expected_direction.is_finite()
            && self.baseline.is_finite()
            && self.comparison.is_finite()
    }

    pub fn delta_metric(&self) -> f64 {
        self.comparison_metric - self.baseline_metric
    }

    pub fn direction_margin(&self) -> f64 {
        self.delta_metric() * self.expected_direction.signum()
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "{} input={:.6}->{:.6} unit={} metric={} {:.6}->{:.6} direction_margin={:.6} passed={} expectation={}",
            self.axis.label(),
            self.baseline_input,
            self.comparison_input,
            self.input_unit,
            self.metric_label,
            self.baseline_metric,
            self.comparison_metric,
            self.direction_margin(),
            self.passed,
            csv_token(self.expectation),
        )
    }
}

fn tire_rig_sensitivity_point(
    axis: TireRigSensitivityAxis,
    baseline_input: f64,
    comparison_input: f64,
    input_unit: &'static str,
    metric_label: &'static str,
    baseline_metric: f64,
    comparison_metric: f64,
    expected_direction: f64,
    min_direction_margin: f64,
    expectation: &'static str,
    baseline: TireRigSweepPoint,
    comparison: TireRigSweepPoint,
) -> TireRigSensitivityPoint {
    let expected_direction = if expected_direction >= 0.0 { 1.0 } else { -1.0 };
    let direction_margin = (comparison_metric - baseline_metric) * expected_direction;
    let passed = baseline_metric.is_finite()
        && comparison_metric.is_finite()
        && baseline.is_finite()
        && comparison.is_finite()
        && direction_margin >= min_direction_margin.max(0.0);
    TireRigSensitivityPoint {
        axis,
        baseline_input: finite_or(baseline_input, 0.0),
        comparison_input: finite_or(comparison_input, 0.0),
        input_unit,
        metric_label,
        baseline_metric: finite_or(baseline_metric, 0.0),
        comparison_metric: finite_or(comparison_metric, 0.0),
        expected_direction,
        expectation,
        baseline,
        comparison,
        passed,
    }
}

/// Deterministic non-calibration sensitivity map for hand-tuned tire changes.
///
/// This is the reusable "did the model react sensibly?" map requested by the
/// plan. It varies pressure, load, dry rolling speed, wet speed, camber,
/// temperature, water film, tread wear, soft-soil material/stiffness,
/// obstacle height, and contact resolution. The outputs are broad
/// monotonic/directional checks only; they are not target curves and do not
/// introduce fitted constants.
pub fn tire_rig_standard_sensitivity_map() -> Vec<TireRigSensitivityPoint> {
    let base = RealTireParams::passenger_radial().with_segments(96);
    let dry = SurfaceMaterial::dry_asphalt();
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let load_n = 4_200.0;
    let speed = 14.0;
    let slip = 0.16;
    let camber = 8.0_f64.to_radians();

    let mut out = Vec::with_capacity(12);

    let stiff = base.with_geometry_and_pressure(base.radius, base.width, 300_000.0);
    let soft = base.with_geometry_and_pressure(base.radius, base.width, 95_000.0);
    let stiff_point = tire_rig_point(&stiff, &dry_terrain, &dry, load_n, speed, slip, 0.0, 0.0);
    let soft_point = tire_rig_point(&soft, &dry_terrain, &dry, load_n, speed, slip, 0.0, 0.0);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::Pressure,
        stiff.pressure_pa,
        soft.pressure_pa,
        "Pa",
        "patch_length_m",
        stiff_point.telemetry.patch.patch_length_m,
        soft_point.telemetry.patch.patch_length_m,
        1.0,
        1.0e-4,
        "lower pressure should grow the contact patch",
        stiff_point,
        soft_point,
    ));

    let light_load = 2_500.0;
    let heavy_load = 7_500.0;
    let light = tire_rig_point(&base, &dry_terrain, &dry, light_load, speed, 0.30, 0.0, 0.0);
    let heavy = tire_rig_point(&base, &dry_terrain, &dry, heavy_load, speed, 0.30, 0.0, 0.0);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::Load,
        light_load,
        heavy_load,
        "N",
        "abs_fx_over_fz",
        light.telemetry.longitudinal_force_n.abs() / light.telemetry.normal_load_n.max(1.0),
        heavy.telemetry.longitudinal_force_n.abs() / heavy.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "higher load should increase force sub-linearly, reducing force/load",
        light,
        heavy,
    ));

    let slow_dry_speed = 4.0;
    let fast_dry_speed = 24.0;
    let slow_dry_roll = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        slow_dry_speed,
        0.0,
        0.0,
        0.0,
    );
    let fast_dry_roll = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        fast_dry_speed,
        0.0,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::DrySpeed,
        slow_dry_speed,
        fast_dry_speed,
        "m/s",
        "dry_rolling_resistance_power_w",
        slow_dry_roll.telemetry.rolling_resistance_force_n.abs() * slow_dry_speed,
        fast_dry_roll.telemetry.rolling_resistance_force_n.abs() * fast_dry_speed,
        1.0,
        1.0,
        "dry rolling-loss power should increase with speed without using fitted curves",
        slow_dry_roll,
        fast_dry_roll,
    ));

    let wet = SurfaceFilmState::wet(0.0012).apply_to(dry, speed, &TireWearState::default());
    let slow_wet_speed = 5.0;
    let fast_wet_speed = 30.0;
    let slow_wet_material =
        SurfaceFilmState::wet(0.0012).apply_to(dry, slow_wet_speed, &TireWearState::default());
    let fast_wet_material =
        SurfaceFilmState::wet(0.0012).apply_to(dry, fast_wet_speed, &TireWearState::default());
    let slow_wet = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, slow_wet_material.mu_static),
        &slow_wet_material,
        load_n,
        slow_wet_speed,
        slip,
        0.0,
        0.0,
    );
    let fast_wet = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, fast_wet_material.mu_static),
        &fast_wet_material,
        load_n,
        fast_wet_speed,
        slip,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::WetSpeed,
        slow_wet_speed,
        fast_wet_speed,
        "m/s",
        "wet_abs_fx_over_fz",
        slow_wet.telemetry.longitudinal_force_n.abs() / slow_wet.telemetry.normal_load_n.max(1.0),
        fast_wet.telemetry.longitudinal_force_n.abs() / fast_wet.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "the same water film should reduce usable force more at higher speed",
        slow_wet,
        fast_wet,
    ));

    let zero_camber = tire_rig_point(&base, &dry_terrain, &dry, load_n, speed, 0.0, 0.0, 0.0);
    let high_camber = tire_rig_point(&base, &dry_terrain, &dry, load_n, speed, 0.0, 0.0, camber);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::Camber,
        0.0,
        camber,
        "rad",
        "abs_lateral_force_n",
        zero_camber.telemetry.lateral_force_n.abs(),
        high_camber.telemetry.lateral_force_n.abs(),
        1.0,
        10.0,
        "larger camber should create a larger camber-thrust response",
        zero_camber,
        high_camber,
    ));

    let mut hot_thermal = TireThermalLayers::passenger_road();
    hot_thermal.surface_c = 185.0;
    hot_thermal.tread_c = 145.0;
    hot_thermal.carcass_c = 125.0;
    let fresh_wear = TireWearState::default();
    let hot_response = hot_thermal.material_response(&fresh_wear, &base, &dry);
    let healthy_response =
        TireThermalLayers::passenger_road().material_response(&fresh_wear, &base, &dry);
    let healthy = tire_rig_point(
        &healthy_response.params,
        &dry_terrain,
        &healthy_response.material,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    let overheated = tire_rig_point(
        &hot_response.params,
        &FlatTerrain::new(0.0, hot_response.material.mu_static),
        &hot_response.material,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::SurfaceTemperature,
        healthy_response.condition.abuse_index,
        hot_response.condition.abuse_index,
        "abuse_index",
        "abs_fx_over_fz",
        healthy.telemetry.longitudinal_force_n.abs() / healthy.telemetry.normal_load_n.max(1.0),
        overheated.telemetry.longitudinal_force_n.abs()
            / overheated.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "overheated state-derived material response should reduce usable force",
        healthy,
        overheated,
    ));

    let wet_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, wet.mu_static),
        &wet,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    let dry_point = tire_rig_point(&base, &dry_terrain, &dry, load_n, speed, slip, 0.0, 0.0);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::WaterFilm,
        0.0,
        0.0012,
        "m",
        "abs_fx_over_fz",
        dry_point.telemetry.longitudinal_force_n.abs() / dry_point.telemetry.normal_load_n.max(1.0),
        wet_point.telemetry.longitudinal_force_n.abs() / wet_point.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "wet film should reduce usable force without a viewer grip switch",
        dry_point,
        wet_point,
    ));

    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.92;
    let fresh_wet = SurfaceFilmState::wet(0.0010).apply_to(dry, speed, &TireWearState::default());
    let worn_wet = SurfaceFilmState::wet(0.0010).apply_to(dry, speed, &worn);
    let fresh_wet_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, fresh_wet.mu_static),
        &fresh_wet,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    let worn_wet_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, worn_wet.mu_static),
        &worn_wet,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::TreadWear,
        0.0,
        worn.depth_m,
        "m",
        "wet_abs_fx_over_fz",
        fresh_wet_point.telemetry.longitudinal_force_n.abs()
            / fresh_wet_point.telemetry.normal_load_n.max(1.0),
        worn_wet_point.telemetry.longitudinal_force_n.abs()
            / worn_wet_point.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "worn tread should hurt wet-film force more than fresh tread",
        fresh_wet_point,
        worn_wet_point,
    ));

    let soft_soil = SurfaceMaterial::soft_soil();
    let soil_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, soft_soil.mu_static),
        &soft_soil,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    let dry_for_soil = tire_rig_point(&base, &dry_terrain, &dry, load_n, speed, slip, 0.0, 0.0);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::SoftSoilSurface,
        dry.mu_static,
        soft_soil.mu_static,
        "mu_static",
        "abs_fx_over_fz",
        dry_for_soil.telemetry.longitudinal_force_n.abs()
            / dry_for_soil.telemetry.normal_load_n.max(1.0),
        soil_point.telemetry.longitudinal_force_n.abs()
            / soil_point.telemetry.normal_load_n.max(1.0),
        -1.0,
        1.0e-4,
        "soft-soil material should reduce usable brush force relative to dry asphalt",
        dry_for_soil,
        soil_point,
    ));

    let mut weak_soil = SurfaceMaterial::soft_soil();
    weak_soil.cohesion = 350.0;
    weak_soil.bristle_damping = 5.5e4;
    let mut firm_soil = SurfaceMaterial::soft_soil();
    firm_soil.cohesion = 3_200.0;
    firm_soil.bristle_damping = 1.25e5;
    let weak_soil_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, weak_soil.mu_static),
        &weak_soil,
        load_n,
        speed,
        0.42,
        0.0,
        0.0,
    );
    let firm_soil_point = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, firm_soil.mu_static),
        &firm_soil,
        load_n,
        speed,
        0.42,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::SoilStiffness,
        weak_soil.cohesion,
        firm_soil.cohesion,
        "Pa_cohesion",
        "soil_abs_fx_over_fz",
        weak_soil_point.telemetry.longitudinal_force_n.abs()
            / weak_soil_point.telemetry.normal_load_n.max(1.0),
        firm_soil_point.telemetry.longitudinal_force_n.abs()
            / firm_soil_point.telemetry.normal_load_n.max(1.0),
        1.0,
        1.0e-4,
        "firmer cohesive soil should support more longitudinal shear than weak soil",
        weak_soil_point,
        firm_soil_point,
    ));

    let obstacle_point = |height: f64| {
        let nx = 41;
        let nz = 5;
        let spacing = 0.04;
        let origin = [-0.8, -0.08];
        let mut heights = Vec::with_capacity(nx * nz);
        for _iz in 0..nz {
            for ix in 0..nx {
                let x = origin[0] + ix as f64 * spacing;
                heights.push(if x >= 0.0 { height } else { 0.0 });
            }
        }
        let terrain = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, dry.mu_static);
        let x = -0.02;
        let point = tire_rig_point_at(&soft, &terrain, &dry, [x, 0.0], load_n, 4.0, 0.04, 0.0, 0.0);
        let normal_metric = terrain.normal(x, 0.0)[0].abs();
        (point, normal_metric)
    };
    let (low_obstacle, low_obstacle_metric) = obstacle_point(0.025);
    let (high_obstacle, high_obstacle_metric) = obstacle_point(0.085);
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::ObstacleHeight,
        0.025,
        0.085,
        "m",
        "abs_terrain_normal_x",
        low_obstacle_metric,
        high_obstacle_metric,
        1.0,
        1.0e-3,
        "taller obstacle should produce a stronger non-flat contact normal",
        low_obstacle,
        high_obstacle,
    ));

    let coarse_segments = 24usize;
    let fine_segments = 128usize;
    let coarse_contact = tire_rig_point(
        &base.with_segments(coarse_segments),
        &dry_terrain,
        &dry,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    let fine_contact = tire_rig_point(
        &base.with_segments(fine_segments),
        &dry_terrain,
        &dry,
        load_n,
        speed,
        slip,
        0.0,
        0.0,
    );
    out.push(tire_rig_sensitivity_point(
        TireRigSensitivityAxis::ContactResolution,
        coarse_segments as f64,
        fine_segments as f64,
        "segments",
        "active_contact_cells",
        coarse_contact.telemetry.patch.contact_count as f64,
        fine_contact.telemetry.patch.contact_count as f64,
        1.0,
        1.0,
        "finer tread/contact resolution should expose more active contact cells",
        coarse_contact,
        fine_contact,
    ));

    out
}

/// Text report for the standard non-calibration sensitivity map.
pub fn tire_rig_standard_sensitivity_text_report() -> String {
    tire_rig_standard_sensitivity_map()
        .into_iter()
        .map(|point| point.to_text_line())
        .collect::<Vec<_>>()
        .join("\n")
}
