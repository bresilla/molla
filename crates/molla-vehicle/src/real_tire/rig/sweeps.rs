//! Tire-rig deterministic sweep machinery: envelope, rolling obstacle,
//! load-deflection, longitudinal-slip, surface-slip matrix,
//! surface-transition terrain, step response.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, Terrain};

use super::super::internal::{
    cross, csv_token, dot, finite_nonnegative, finite_or, finite_positive_or, normalize,
    sanitize_vec3, smoothstep01,
};
use super::super::surface_material::sanitized_surface_material;
use super::super::{
    RealTireParams, RealTireTelemetrySample, RealTireTransientState, RealWheelOrientation,
    RealWheelState, SurfaceFilmState, SurfaceMaterial, TetraTireFem, TetraTireFemDiagnostics,
    TetraTireFemParams, TireModelProvenance, TireRigSnapshot, TireRigSweepPoint, TireWearState,
    evaluate_brush_tire, evaluate_brush_tire_transient, project_to_ground, sanitize_brush_params,
    solve_loaded_tire_hub_height, tire_rig_point, tire_rig_point_at,
};

/// One point from a deterministic heightfield envelope-contact sweep.
#[derive(Clone, Debug)]
pub struct TireRigEnvelopePoint {
    pub x_m: f64,
    pub z_m: f64,
    pub terrain_height_m: f64,
    pub terrain_normal: [f64; 3],
    pub point: TireRigSweepPoint,
}

impl TireRigEnvelopePoint {
    pub fn is_finite(&self) -> bool {
        self.x_m.is_finite()
            && self.z_m.is_finite()
            && self.terrain_height_m.is_finite()
            && self.terrain_normal.iter().all(|v| v.is_finite())
            && self.point.is_finite()
    }
}

/// One ordered point from a deterministic rolling obstacle sequence.
#[derive(Clone, Debug)]
pub struct TireRigRollingObstaclePoint {
    pub step_index: usize,
    pub time_s: f64,
    pub x_m: f64,
    pub z_m: f64,
    pub wheel_angle_rad: f64,
    pub hub: [f64; 3],
    pub hub_deflection_m: f64,
    pub terrain_height_m: f64,
    pub terrain_normal: [f64; 3],
    pub contact_patch_width_m: f64,
    pub active_tread_nodes: usize,
    pub telemetry: RealTireTelemetrySample,
    pub fem: TetraTireFemDiagnostics,
}

impl TireRigRollingObstaclePoint {
    pub fn is_finite(&self) -> bool {
        self.time_s.is_finite()
            && self.x_m.is_finite()
            && self.z_m.is_finite()
            && self.wheel_angle_rad.is_finite()
            && self.hub.iter().all(|v| v.is_finite())
            && self.hub_deflection_m.is_finite()
            && self.terrain_height_m.is_finite()
            && self.terrain_normal.iter().all(|v| v.is_finite())
            && self.contact_patch_width_m.is_finite()
            && self.telemetry.is_finite()
            && self.fem.cavity_volume_m3.is_finite()
            && self.fem.max_tet_volume_error_m3.is_finite()
    }

    pub fn snapshot(
        &self,
        scenario_id: &str,
        tire_preset: &str,
        surface_preset: &str,
        terrain_preset: &str,
        physics_path: &str,
        pressure_pa: f64,
        load_n: f64,
        speed_m_per_s: f64,
        slip_ratio_command: f64,
    ) -> TireRigSnapshot {
        TireRigSnapshot {
            scenario_id: csv_token(scenario_id),
            tire_preset: csv_token(tire_preset),
            surface_preset: csv_token(surface_preset),
            terrain_preset: csv_token(terrain_preset),
            physics_path: csv_token(physics_path),
            tire_parameter_provenance: TireModelProvenance::HandTuned,
            surface_parameter_provenance: TireModelProvenance::HandTuned,
            solver_parameter_provenance: TireModelProvenance::NumericalStability,
            load_n: finite_nonnegative(load_n),
            pressure_pa: finite_nonnegative(pressure_pa),
            hub_height_m: finite_or(self.hub[1], 0.0),
            deflection_m: finite_nonnegative(self.hub_deflection_m),
            speed_m_per_s: finite_or(speed_m_per_s, 0.0),
            slip_ratio_command: finite_or(slip_ratio_command, 0.0),
            slip_angle_command_rad: finite_or(self.telemetry.slip_angle_rad, 0.0),
            camber_angle_command_rad: finite_or(self.telemetry.camber_angle_rad, 0.0),
            terrain_height_m: finite_or(self.terrain_height_m, 0.0),
            terrain_normal: sanitize_vec3(self.terrain_normal, [0.0, 1.0, 0.0]),
            terrain_surface_velocity_m_per_s: [0.0; 3],
            moving_ground_power_w: 0.0,
            surface_temperature_c: None,
            tread_temperature_c: None,
            carcass_temperature_c: None,
            tread_depth_m: None,
            wear_ratio: None,
            cpu_gpu_force_delta_n: None,
            cpu_gpu_relative_force_error: None,
            solver_budget: None,
            telemetry: self.telemetry.clone(),
        }
    }
}

/// Deterministic heightfield bump/curb envelope-contact sweep.
///
/// The caller supplies the terrain and longitudinal sample positions. Each
/// sample solves hub height from the requested load at that `(x,z)`, evaluates
/// the real brush/contact patch against the terrain, and records local terrain
/// height/normal plus wheel telemetry. Use this to regression-test that bumps,
/// ramps, curbs, and heightfield material changes are visible to the tire
/// contact model instead of collapsing to a flat single-point contact.
pub fn tire_rig_heightfield_envelope_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    x_positions_m: &[f64],
    z_m: f64,
) -> Vec<TireRigEnvelopePoint> {
    x_positions_m
        .iter()
        .map(|&x| {
            let x = finite_or(x, 0.0);
            let z = finite_or(z_m, 0.0);
            TireRigEnvelopePoint {
                x_m: x,
                z_m: z,
                terrain_height_m: terrain.height(x, z),
                terrain_normal: terrain.normal(x, z),
                point: tire_rig_point_at(
                    params,
                    terrain,
                    material,
                    [x, z],
                    load_n,
                    speed_m_per_s,
                    slip_ratio_command,
                    0.0,
                    0.0,
                ),
            }
        })
        .collect()
}

/// Deterministic rolling obstacle sequence for FEM envelope-contact checks.
///
/// Unlike the static heightfield sweep, this advances one `TetraTireFem`
/// forward in time, keeps the hub moving over the terrain, evaluates contact
/// from the deformed tread nodes, and feeds the brush wrench back into the FEM
/// nodes. It is still a compact tire rig, not a full vehicle solver, but it
/// catches regressions where the tire only works at isolated sample points and
/// fails during ordered bump/curb approach → crest → exit motion.
pub fn tire_rig_rolling_obstacle_sequence(
    mut params: TetraTireFemParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    start_x_m: f64,
    z_m: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    dt_s: f64,
    steps: usize,
) -> Vec<TireRigRollingObstaclePoint> {
    let material = sanitized_surface_material(material);
    params.surface_material = material;
    let dt_s = finite_positive_or(dt_s, 1.0 / 240.0, 1.0e-6).min(1.0 / 30.0);
    let speed_m_per_s = finite_or(speed_m_per_s, 0.0);
    let slip_ratio_command = finite_or(slip_ratio_command, 0.0).clamp(-0.9, 2.0);
    let start_x_m = finite_or(start_x_m, 0.0);
    let z_m = finite_or(z_m, 0.0);
    let steps = steps.min(512);
    let mut tire = TetraTireFem::new(start_x_m, params.pressure_pa, params);

    // Let the tire settle into its loaded shape before it encounters the
    // obstacle. This makes the sequence a rolling-envelope regression instead
    // of just a drop test.
    for _ in 0..24 {
        let hub = tire.hub();
        tire.set_hub_kinematic([start_x_m, hub[1], z_m], [0.0, tire.hub_v_y, 0.0]);
        tire.step_on_terrain(terrain, dt_s);
    }

    let mut out = Vec::with_capacity(steps);
    let mut wheel_angle = 0.0;
    for step_index in 0..steps {
        let time_s = step_index as f64 * dt_s;
        let x_m = start_x_m + speed_m_per_s * time_s;
        let hub_before = tire.hub();
        tire.set_hub_kinematic(
            [x_m, hub_before[1], z_m],
            [speed_m_per_s, tire.hub_v_y, 0.0],
        );
        tire.step_on_terrain(terrain, dt_s);
        wheel_angle += speed_m_per_s * dt_s / params.outer_radius.max(1.0e-6);

        let hub = tire.hub();
        let state = RealWheelState {
            hub_pos: hub,
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio_command) / params.outer_radius.max(1.0e-6),
        };
        let contact = tire.apply_tread_contact_wrench(&state, terrain, &material, dt_s);
        let active_tread_nodes = contact.cells.len();
        out.push(TireRigRollingObstaclePoint {
            step_index,
            time_s,
            x_m,
            z_m,
            wheel_angle_rad: wheel_angle,
            hub,
            hub_deflection_m: tire.hub_deflection_m(),
            terrain_height_m: terrain.height(x_m, z_m),
            terrain_normal: terrain.normal(x_m, z_m),
            contact_patch_width_m: tire.contact_patch_width(),
            active_tread_nodes,
            telemetry: contact.telemetry_sample(dt_s),
            fem: tire.diagnostics(),
        });
    }
    out
}

/// Deterministic vertical load/deflection sweep for tire-rig validation.
pub fn tire_rig_load_deflection_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    loads_n: &[f64],
) -> Vec<TireRigSweepPoint> {
    loads_n
        .iter()
        .map(|&load_n| tire_rig_point(params, terrain, material, load_n, 0.0, 0.0, 0.0, 0.0))
        .collect()
}

/// Deterministic longitudinal slip sweep for tire-rig validation.
pub fn tire_rig_longitudinal_slip_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratios: &[f64],
) -> Vec<TireRigSweepPoint> {
    slip_ratios
        .iter()
        .map(|&slip| {
            tire_rig_point(
                params,
                terrain,
                material,
                load_n,
                speed_m_per_s,
                slip,
                0.0,
                0.0,
            )
        })
        .collect()
}

/// Surface cases used by deterministic tire-rig material matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireRigSurfaceKind {
    DryAsphalt,
    LooseDirt,
    WetAsphalt,
    SoftSoil,
    Ice,
}

/// One tire-rig point tagged with the surface/material preset that produced it.
#[derive(Clone, Debug)]
pub struct TireRigSurfaceSlipPoint {
    pub surface: TireRigSurfaceKind,
    pub material: SurfaceMaterial,
    pub point: TireRigSweepPoint,
}

impl TireRigSurfaceSlipPoint {
    pub fn is_finite(&self) -> bool {
        self.material.mu_static.is_finite()
            && self.material.mu_dynamic.is_finite()
            && self.material.bristle_damping.is_finite()
            && self.material.rolling_resistance.is_finite()
            && self.material.cohesion.is_finite()
            && self.point.is_finite()
    }
}

/// One spatial band for deterministic surface-transition tire rigs.
#[derive(Clone, Debug)]
pub struct TireRigSurfaceTransitionBand {
    pub label: String,
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub material: SurfaceMaterial,
    pub water_film_depth_m: f64,
    pub surface_velocity_m_per_s: [f64; 3],
}

impl TireRigSurfaceTransitionBand {
    pub fn new(label: &str, x_min_m: f64, x_max_m: f64, material: SurfaceMaterial) -> Self {
        Self {
            label: csv_token(label),
            x_min_m: finite_or(x_min_m.min(x_max_m), 0.0),
            x_max_m: finite_or(x_max_m.max(x_min_m), 0.0),
            material: sanitized_surface_material(&material),
            water_film_depth_m: 0.0,
            surface_velocity_m_per_s: [0.0; 3],
        }
    }

    pub fn with_water_film_depth(mut self, water_film_depth_m: f64) -> Self {
        self.water_film_depth_m = finite_nonnegative(water_film_depth_m);
        self
    }

    pub fn with_surface_velocity(mut self, surface_velocity_m_per_s: [f64; 3]) -> Self {
        self.surface_velocity_m_per_s = sanitize_vec3(surface_velocity_m_per_s, [0.0; 3]);
        self
    }

    fn contains(&self, x_m: f64) -> bool {
        x_m >= self.x_min_m && x_m <= self.x_max_m
    }
}

#[derive(Clone, Debug)]
struct TireRigSurfaceSample {
    label: String,
    material: SurfaceMaterial,
    water_film_depth_m: f64,
    surface_velocity_m_per_s: [f64; 3],
}

/// Flat rig terrain with spatially varying tire material, water film, and
/// ground velocity.
///
/// This intentionally lives in `molla-vehicle` as non-calibration test
/// infrastructure: the same tire can cross asphalt → ice, dry → wet, dirt →
/// loose soil, and moving-belt bands without hard-coding viewer-only flags.
#[derive(Clone, Debug)]
pub struct TireRigSurfaceTransitionTerrain {
    pub height_m: f64,
    pub fallback_material: SurfaceMaterial,
    pub transition_width_m: f64,
    pub bands: Vec<TireRigSurfaceTransitionBand>,
}

impl TireRigSurfaceTransitionTerrain {
    pub fn new(
        height_m: f64,
        fallback_material: SurfaceMaterial,
        mut bands: Vec<TireRigSurfaceTransitionBand>,
    ) -> Self {
        bands.sort_by(|a, b| {
            a.x_min_m
                .partial_cmp(&b.x_min_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            height_m: finite_or(height_m, 0.0),
            fallback_material: sanitized_surface_material(&fallback_material),
            transition_width_m: 0.20,
            bands,
        }
    }

    pub fn with_transition_width(mut self, transition_width_m: f64) -> Self {
        self.transition_width_m = finite_nonnegative(transition_width_m);
        self
    }

    fn sample(&self, x_m: f64) -> TireRigSurfaceSample {
        let x_m = finite_or(x_m, 0.0);
        let transition = self.transition_width_m;
        if transition > 1.0e-9 {
            for pair in self.bands.windows(2) {
                let left = &pair[0];
                let right = &pair[1];
                let boundary = 0.5 * (left.x_max_m + right.x_min_m);
                if x_m >= boundary - transition && x_m <= boundary + transition {
                    let raw = (x_m - (boundary - transition)) / (2.0 * transition);
                    let t = smoothstep01(raw);
                    return TireRigSurfaceSample {
                        label: format!("{}_to_{}", left.label, right.label),
                        material: lerp_surface_material(left.material, right.material, t),
                        water_film_depth_m: lerp(
                            left.water_film_depth_m,
                            right.water_film_depth_m,
                            t,
                        ),
                        surface_velocity_m_per_s: lerp_vec3(
                            left.surface_velocity_m_per_s,
                            right.surface_velocity_m_per_s,
                            t,
                        ),
                    };
                }
            }
        }

        if let Some(band) = self.bands.iter().find(|band| band.contains(x_m)) {
            TireRigSurfaceSample {
                label: band.label.clone(),
                material: band.material,
                water_film_depth_m: band.water_film_depth_m,
                surface_velocity_m_per_s: band.surface_velocity_m_per_s,
            }
        } else {
            TireRigSurfaceSample {
                label: "fallback".to_owned(),
                material: self.fallback_material,
                water_film_depth_m: 0.0,
                surface_velocity_m_per_s: [0.0; 3],
            }
        }
    }

    pub(crate) fn material_at(&self, x_m: f64) -> SurfaceMaterial {
        self.sample(x_m).material
    }

    pub(crate) fn label_at(&self, x_m: f64) -> String {
        self.sample(x_m).label
    }
}

impl Terrain for TireRigSurfaceTransitionTerrain {
    fn height(&self, _x: f64, _z: f64) -> f64 {
        self.height_m
    }

    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }

    fn friction(&self) -> f64 {
        self.fallback_material.mu_static
    }

    fn friction_at(&self, x: f64, _z: f64) -> f64 {
        self.sample(x).material.mu_static
    }

    fn surface_velocity(&self, x: f64, _z: f64) -> [f64; 3] {
        self.sample(x).surface_velocity_m_per_s
    }

    fn water_film_depth(&self, x: f64, _z: f64) -> f64 {
        self.sample(x).water_film_depth_m
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    a * (1.0 - t) + b * t
}

fn lerp_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn lerp_surface_material(a: SurfaceMaterial, b: SurfaceMaterial, t: f64) -> SurfaceMaterial {
    SurfaceMaterial {
        mu_static: lerp(a.mu_static, b.mu_static, t),
        mu_dynamic: lerp(a.mu_dynamic, b.mu_dynamic, t),
        bristle_damping: lerp(a.bristle_damping, b.bristle_damping, t),
        rolling_resistance: lerp(a.rolling_resistance, b.rolling_resistance, t),
        cohesion: lerp(a.cohesion, b.cohesion, t),
    }
}

/// One point from a deterministic spatial surface-transition rig.
#[derive(Clone, Debug)]
pub struct TireRigSurfaceTransitionPoint {
    pub step_index: usize,
    pub x_m: f64,
    pub z_m: f64,
    pub surface_label: String,
    pub material: SurfaceMaterial,
    pub friction_mu: f64,
    pub water_film_depth_m: f64,
    pub surface_velocity_m_per_s: [f64; 3],
    pub telemetry: RealTireTelemetrySample,
}

impl TireRigSurfaceTransitionPoint {
    pub fn is_finite(&self) -> bool {
        self.x_m.is_finite()
            && self.z_m.is_finite()
            && self.material.mu_static.is_finite()
            && self.material.mu_dynamic.is_finite()
            && self.friction_mu.is_finite()
            && self.water_film_depth_m.is_finite()
            && self.surface_velocity_m_per_s.iter().all(|v| v.is_finite())
            && self.telemetry.is_finite()
    }
}

/// Deterministic surface-transition sequence over a banded rig terrain.
pub fn tire_rig_surface_transition_sequence(
    params: &RealTireParams,
    terrain: &TireRigSurfaceTransitionTerrain,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    x_positions_m: &[f64],
    z_m: f64,
) -> Vec<TireRigSurfaceTransitionPoint> {
    let params = sanitize_brush_params(params);
    let params = RealTireParams {
        radius: params.radius,
        width: params.width,
        pressure_pa: params.pressure_pa,
        segments: params.segments,
        carcass_normal_stiffness: params.carcass_normal_stiffness,
        tread_shear_stiffness: params.tread_shear_stiffness,
        structural_lateral_bias_per_normal: params.structural_lateral_bias_per_normal,
        relaxation_speed: params.relaxation_speed,
    };
    let load_n = finite_nonnegative(load_n).max(1.0);
    let speed_m_per_s = finite_or(speed_m_per_s, 0.0);
    let slip_ratio_command = finite_or(slip_ratio_command, 0.0).clamp(-0.95, 3.0);
    let z_m = finite_or(z_m, 0.0);
    x_positions_m
        .iter()
        .enumerate()
        .map(|(step_index, &x_m)| {
            let x_m = finite_or(x_m, 0.0);
            let material = terrain.material_at(x_m);
            let hub_y = solve_loaded_tire_hub_height(&params, terrain, x_m, z_m, load_n);
            let state = RealWheelState {
                hub_pos: [x_m, hub_y, z_m],
                hub_vel: [speed_m_per_s, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: speed_m_per_s * (1.0 + slip_ratio_command) / params.radius.max(1.0e-6),
            };
            let contact = evaluate_brush_tire(&params, &state, terrain, &material);
            TireRigSurfaceTransitionPoint {
                step_index,
                x_m,
                z_m,
                surface_label: terrain.label_at(x_m),
                material,
                friction_mu: terrain.friction_at(x_m, z_m),
                water_film_depth_m: terrain.water_film_depth(x_m, z_m),
                surface_velocity_m_per_s: terrain.surface_velocity(x_m, z_m),
                telemetry: contact.telemetry_sample(0.0),
            }
        })
        .collect()
}

/// One rolling-resistance / aerodynamic-drag loss sample.
#[derive(Clone, Debug)]
pub struct TireRigRollingLossPoint {
    pub surface_label: String,
    pub speed_m_per_s: f64,
    pub load_n: f64,
    pub rolling_resistance_force_n: f64,
    pub rolling_resistance_power_w: f64,
    pub aero_drag_force_n: f64,
    pub aero_drag_power_w: f64,
    pub total_loss_force_n: f64,
    pub total_loss_power_w: f64,
    pub coastdown_deceleration_m_per_s2: f64,
    pub telemetry: RealTireTelemetrySample,
}

impl TireRigRollingLossPoint {
    pub fn is_finite(&self) -> bool {
        self.speed_m_per_s.is_finite()
            && self.load_n.is_finite()
            && self.rolling_resistance_force_n.is_finite()
            && self.rolling_resistance_power_w.is_finite()
            && self.aero_drag_force_n.is_finite()
            && self.aero_drag_power_w.is_finite()
            && self.total_loss_force_n.is_finite()
            && self.total_loss_power_w.is_finite()
            && self.coastdown_deceleration_m_per_s2.is_finite()
            && self.telemetry.is_finite()
    }
}

/// Deterministic rolling-resistance and drag-loss sweep.
///
/// This is not a vehicle calibration rig. It checks that pure rolling still
/// reports material/film-dependent rolling losses, that aero drag scales with
/// speed, and that coastdown deceleration can be computed from explicit loss
/// terms instead of hidden viewer constants.
pub fn tire_rig_rolling_loss_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    surface_label: &str,
    material: &SurfaceMaterial,
    load_n: f64,
    speeds_m_per_s: &[f64],
    vehicle_mass_kg: f64,
    aero_cd_area_m2: f64,
    air_density_kg_per_m3: f64,
) -> Vec<TireRigRollingLossPoint> {
    let params = sanitize_brush_params(params);
    let params = RealTireParams {
        radius: params.radius,
        width: params.width,
        pressure_pa: params.pressure_pa,
        segments: params.segments,
        carcass_normal_stiffness: params.carcass_normal_stiffness,
        tread_shear_stiffness: params.tread_shear_stiffness,
        structural_lateral_bias_per_normal: params.structural_lateral_bias_per_normal,
        relaxation_speed: params.relaxation_speed,
    };
    let material = sanitized_surface_material(material);
    let load_n = finite_nonnegative(load_n).max(1.0);
    let vehicle_mass_kg = finite_positive_or(vehicle_mass_kg, 1_200.0, 1.0);
    let aero_cd_area_m2 = finite_nonnegative(aero_cd_area_m2);
    let air_density_kg_per_m3 = finite_positive_or(air_density_kg_per_m3, 1.225, 1.0e-6);
    speeds_m_per_s
        .iter()
        .map(|&speed| {
            let speed = finite_or(speed, 0.0);
            let hub_y = solve_loaded_tire_hub_height(&params, terrain, 0.0, 0.0, load_n);
            let state = RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [speed, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: speed / params.radius.max(1.0e-6),
            };
            let contact = evaluate_brush_tire(&params, &state, terrain, &material);
            let telemetry = contact.telemetry_sample(0.0);
            let rolling_resistance_force_n = telemetry.rolling_resistance_force_n.abs();
            let rolling_resistance_power_w = rolling_resistance_force_n * speed.abs();
            let aero_drag_force_n =
                0.5 * air_density_kg_per_m3 * aero_cd_area_m2 * speed.abs() * speed.abs();
            let aero_drag_power_w = aero_drag_force_n * speed.abs();
            let total_loss_force_n = rolling_resistance_force_n + aero_drag_force_n;
            let total_loss_power_w = rolling_resistance_power_w + aero_drag_power_w;
            TireRigRollingLossPoint {
                surface_label: csv_token(surface_label),
                speed_m_per_s: speed,
                load_n,
                rolling_resistance_force_n,
                rolling_resistance_power_w,
                aero_drag_force_n,
                aero_drag_power_w,
                total_loss_force_n,
                total_loss_power_w,
                coastdown_deceleration_m_per_s2: total_loss_force_n / vehicle_mass_kg,
                telemetry,
            }
        })
        .collect()
}

/// Deterministic longitudinal slip matrix over the built-in road/loose-surface
/// presets.
///
/// This is still a brush/material bench, not a full deformable-soil solve:
/// `SoftSoil` uses the soil-like material coefficients here, while Bekker rut
/// coupling is covered by `DeformableTerrain::apply_tire_contact`.
pub fn tire_rig_surface_slip_matrix(
    params: &RealTireParams,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratios: &[f64],
    wet_film_depth_m: f64,
    wear: &TireWearState,
) -> Vec<TireRigSurfaceSlipPoint> {
    let dry = SurfaceMaterial::dry_asphalt();
    let wet = SurfaceFilmState::wet(wet_film_depth_m).apply_to(dry, speed_m_per_s, wear);
    let cases = [
        (TireRigSurfaceKind::DryAsphalt, dry),
        (TireRigSurfaceKind::LooseDirt, SurfaceMaterial::loose_dirt()),
        (TireRigSurfaceKind::WetAsphalt, wet),
        (TireRigSurfaceKind::SoftSoil, SurfaceMaterial::soft_soil()),
        (TireRigSurfaceKind::Ice, SurfaceMaterial::ice()),
    ];
    let mut out = Vec::with_capacity(cases.len() * slip_ratios.len());
    for (surface, material) in cases {
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        for &slip in slip_ratios {
            out.push(TireRigSurfaceSlipPoint {
                surface,
                material,
                point: tire_rig_point(
                    params,
                    &terrain,
                    &material,
                    load_n,
                    speed_m_per_s,
                    slip,
                    0.0,
                    0.0,
                ),
            });
        }
    }
    out
}

/// One sample from a deterministic step-response tire rig.
///
/// These points are not calibration samples. They are regression/falsification
/// samples for transient behavior: raw commanded slip/steer changes should not
/// teleport directly into effective tire force, wheel spin should evolve from
/// torque balance, and all force/energy accounting should remain finite.
#[derive(Clone, Debug)]
pub struct TireRigStepResponsePoint {
    pub step_index: usize,
    pub time_s: f64,
    pub drive_torque_nm: f64,
    pub brake_torque_nm: f64,
    pub steer_angle_rad: f64,
    pub wheel_omega_rad_s: f64,
    pub raw_slip_ratio: f64,
    pub effective_slip_ratio: f64,
    pub raw_slip_angle_rad: f64,
    pub effective_slip_angle_rad: f64,
    pub longitudinal_deflection_m: f64,
    pub lateral_deflection_m: f64,
    pub tread_yaw_rad: f64,
    pub telemetry: RealTireTelemetrySample,
}

impl TireRigStepResponsePoint {
    pub fn is_finite(&self) -> bool {
        self.time_s.is_finite()
            && self.drive_torque_nm.is_finite()
            && self.brake_torque_nm.is_finite()
            && self.steer_angle_rad.is_finite()
            && self.wheel_omega_rad_s.is_finite()
            && self.raw_slip_ratio.is_finite()
            && self.effective_slip_ratio.is_finite()
            && self.raw_slip_angle_rad.is_finite()
            && self.effective_slip_angle_rad.is_finite()
            && self.longitudinal_deflection_m.is_finite()
            && self.lateral_deflection_m.is_finite()
            && self.tread_yaw_rad.is_finite()
            && self.telemetry.is_finite()
    }
}

/// Deterministic step-response rig configuration.
#[derive(Clone, Copy, Debug)]
pub struct TireRigStepResponseConfig {
    pub load_n: f64,
    pub speed_m_per_s: f64,
    pub dt_s: f64,
    pub steps: usize,
    pub step_time_s: f64,
    pub wheel_inertia_kg_m2: f64,
    pub drive_torque_after_nm: f64,
    pub brake_torque_after_nm: f64,
    pub steer_angle_after_rad: f64,
}

impl Default for TireRigStepResponseConfig {
    fn default() -> Self {
        Self {
            load_n: 3_800.0,
            speed_m_per_s: 12.0,
            dt_s: 1.0 / 120.0,
            steps: 120,
            step_time_s: 0.25,
            wheel_inertia_kg_m2: 1.2,
            drive_torque_after_nm: 0.0,
            brake_torque_after_nm: 0.0,
            steer_angle_after_rad: 0.0,
        }
    }
}

pub(crate) fn tire_rig_step_response(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    config: TireRigStepResponseConfig,
) -> Vec<TireRigStepResponsePoint> {
    let brush_params = sanitize_brush_params(params);
    let params = RealTireParams {
        radius: brush_params.radius,
        width: brush_params.width,
        pressure_pa: brush_params.pressure_pa,
        segments: brush_params.segments,
        carcass_normal_stiffness: brush_params.carcass_normal_stiffness,
        tread_shear_stiffness: brush_params.tread_shear_stiffness,
        structural_lateral_bias_per_normal: brush_params.structural_lateral_bias_per_normal,
        relaxation_speed: brush_params.relaxation_speed,
    };
    let material = sanitized_surface_material(material);
    let load_n = finite_nonnegative(config.load_n).max(1.0);
    let speed = finite_or(config.speed_m_per_s, 0.0);
    let dt = finite_positive_or(config.dt_s, 1.0 / 120.0, 1.0e-6).min(0.05);
    let steps = config.steps.min(2048);
    let step_time_s = finite_nonnegative(config.step_time_s);
    let wheel_inertia = finite_positive_or(config.wheel_inertia_kg_m2, 1.0, 1.0e-6);
    let hub_y = solve_loaded_tire_hub_height(&params, terrain, 0.0, 0.0, load_n);
    let normal = terrain.normal(0.0, 0.0);
    let mut omega = speed / params.radius.max(1.0e-6);
    let mut transient = RealTireTransientState::default();
    let mut out = Vec::with_capacity(steps);

    for step_index in 0..steps {
        let time_s = step_index as f64 * dt;
        let after_step = time_s >= step_time_s;
        let drive_torque_nm = if after_step {
            finite_or(config.drive_torque_after_nm, 0.0).max(0.0)
        } else {
            0.0
        };
        let brake_torque_nm = if after_step {
            finite_or(config.brake_torque_after_nm, 0.0).max(0.0)
        } else {
            0.0
        };
        let steer_angle_rad = if after_step {
            finite_or(config.steer_angle_after_rad, 0.0).clamp(-0.8, 0.8)
        } else {
            0.0
        };
        let normal = normalize(normal);
        let heading = normalize(project_to_ground(
            [steer_angle_rad.cos(), 0.0, steer_angle_rad.sin()],
            normal,
        ));
        let side = normalize(cross(heading, normal));
        let hub_vel = [speed, 0.0, 0.0];
        let v_long = dot(hub_vel, heading);
        let v_lat = dot(hub_vel, side);
        let denom = v_long.abs().max(params.relaxation_speed);
        let raw_slip_ratio = ((omega * params.radius - v_long) / denom).clamp(-8.0, 8.0);
        let raw_slip_angle_rad = v_lat.atan2(denom).clamp(-1.2, 1.2);
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel,
            heading,
            omega,
        };
        let contact = evaluate_brush_tire_transient(
            &params,
            &state,
            terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            dt,
        );
        let telemetry = contact.telemetry_sample(dt);
        let tire_reaction_torque_nm = telemetry.longitudinal_force_n * params.radius;
        out.push(TireRigStepResponsePoint {
            step_index,
            time_s,
            drive_torque_nm,
            brake_torque_nm,
            steer_angle_rad,
            wheel_omega_rad_s: omega,
            raw_slip_ratio,
            effective_slip_ratio: transient.kappa_eff,
            raw_slip_angle_rad,
            effective_slip_angle_rad: transient.alpha_eff,
            longitudinal_deflection_m: transient.longitudinal_deflection,
            lateral_deflection_m: transient.lateral_deflection,
            tread_yaw_rad: transient.tread_yaw,
            telemetry,
        });

        let brake_sign = if omega.abs() > 1.0e-6 {
            omega.signum()
        } else if speed >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let omega_dot = (drive_torque_nm - brake_torque_nm * brake_sign - tire_reaction_torque_nm)
            / wheel_inertia;
        omega = finite_or(omega + omega_dot * dt, 0.0).clamp(-350.0, 350.0);
    }

    out
}
