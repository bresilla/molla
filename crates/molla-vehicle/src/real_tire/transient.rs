//! Transient lag/relaxation tire state (single + layered) + brush-evaluation
//! convenience wrappers.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::Terrain;

use super::internal::{
    add, cross, dot, finite_nonnegative, finite_or, finite_positive_or, finite_vec3, normalize,
    scale,
};
use super::{
    RealTireContact, RealTireParams, RealWheelOrientation, RealWheelState, SurfaceMaterial,
    evaluate_brush_tire_oriented, project_to_ground, sanitize_brush_params,
};

/// Per-wheel lag state for transient tire dynamics.
///
/// This is deliberately small and serializable: callers keep one instance per
/// wheel, call [`RealTireTransientState::step`] before evaluating the brush
/// contact, and feed the returned effective wheel state into the steady-state
/// contact model. It models relaxation length without requiring calibrated
/// tire data.
#[derive(Clone, Copy, Debug)]
pub struct RealTireTransientState {
    pub kappa_eff: f64,
    pub alpha_eff: f64,
    pub longitudinal_deflection: f64,
    pub lateral_deflection: f64,
    pub tread_yaw: f64,
    pub relaxation_length_longitudinal: f64,
    pub relaxation_length_lateral: f64,
    pub tread_yaw_relaxation_length: f64,
}

/// Operating context for pressure/load/temperature-sensitive tire relaxation.
///
/// These values are not calibration targets; they are broad physics-derived
/// modifiers for the existing relaxation-length state. A soft, overloaded, or
/// hot carcass responds more slowly than a stiff, lightly loaded, cool tire.
#[derive(Clone, Copy, Debug)]
pub struct TireRelaxationContext {
    pub normal_load_n: f64,
    pub carcass_temperature_c: f64,
}

impl Default for TireRelaxationContext {
    fn default() -> Self {
        Self {
            normal_load_n: 3_600.0,
            carcass_temperature_c: 40.0,
        }
    }
}

impl TireRelaxationContext {
    pub const fn new(normal_load_n: f64, carcass_temperature_c: f64) -> Self {
        Self {
            normal_load_n,
            carcass_temperature_c,
        }
    }

    /// Multiplicative scale applied to the base relaxation lengths.
    ///
    /// Lower pressure, higher vertical load, and hot carcass temperature all
    /// increase the effective length. The clamps keep this a robust
    /// non-calibrated modifier instead of a hidden fitted tire curve.
    pub fn relaxation_length_scale(&self, params: &RealTireParams) -> f64 {
        relaxation_length_scale_from_values(
            finite_nonnegative(params.pressure_pa),
            self.normal_load_n,
            self.carcass_temperature_c,
        )
    }
}

fn relaxation_length_scale_from_values(
    pressure_pa: f64,
    normal_load_n: f64,
    carcass_temperature_c: f64,
) -> f64 {
    let pressure_ref = RealTireParams::passenger_radial().pressure_pa;
    let pressure_ratio = (pressure_ref / pressure_pa.max(20_000.0)).clamp(0.45, 3.0);
    let pressure_scale = pressure_ratio.sqrt().clamp(0.70, 1.75);

    let load_ratio = (finite_nonnegative(normal_load_n) / 3_600.0).clamp(0.25, 3.5);
    let load_scale = load_ratio.powf(0.18).clamp(0.80, 1.30);

    let temp_c = finite_or(carcass_temperature_c, 40.0);
    let hot_scale = (1.0 + ((temp_c - 40.0) / 130.0).clamp(-0.15, 0.75)).clamp(0.85, 1.75);

    (pressure_scale * load_scale * hot_scale).clamp(0.55, 3.0)
}

/// One physical lag layer inside a tire transient model.
///
/// The fields are intentionally the same units as [`RealTireTransientState`]
/// so rig telemetry and viewer debug panels can show where the fast tread
/// shear, the belt, and the slower carcass body disagree after a step input.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTransientLayerState {
    pub kappa_eff: f64,
    pub alpha_eff_rad: f64,
    pub yaw_rad: f64,
    pub longitudinal_deflection_m: f64,
    pub lateral_deflection_m: f64,
}

impl TireTransientLayerState {
    pub fn is_finite(&self) -> bool {
        self.kappa_eff.is_finite()
            && self.alpha_eff_rad.is_finite()
            && self.yaw_rad.is_finite()
            && self.longitudinal_deflection_m.is_finite()
            && self.lateral_deflection_m.is_finite()
    }
}

#[derive(Clone, Copy, Debug)]
struct TireTransientLayerLengths {
    longitudinal_m: f64,
    lateral_m: f64,
    yaw_m: f64,
}

impl TireTransientLayerLengths {
    fn scaled(self, scale: f64) -> Self {
        Self {
            longitudinal_m: finite_positive_or(self.longitudinal_m * scale, 0.5, 1.0e-4),
            lateral_m: finite_positive_or(self.lateral_m * scale, 0.5, 1.0e-4),
            yaw_m: finite_positive_or(self.yaw_m * scale, 0.5, 1.0e-4),
        }
    }
}

/// Multi-layer transient tire dynamics state.
///
/// This is the next step after the compact [`RealTireTransientState`]: the
/// tread band reacts quickly to slip, the belt follows at an intermediate
/// rate, and the carcass body lags slowest. The evaluated brush force uses a
/// weighted effective slip, while callers can inspect all three layers to see
/// whether a tire is mostly in quick tread shear or whole-body deformation.
#[derive(Clone, Copy, Debug)]
pub struct TireLayeredTransientState {
    pub tread: TireTransientLayerState,
    pub belt: TireTransientLayerState,
    pub carcass: TireTransientLayerState,
    pub kappa_eff: f64,
    pub alpha_eff_rad: f64,
    pub tread_yaw_rad: f64,
    pub longitudinal_deflection_m: f64,
    pub lateral_deflection_m: f64,
    pub tread_relaxation_length_longitudinal: f64,
    pub tread_relaxation_length_lateral: f64,
    pub tread_yaw_relaxation_length: f64,
    pub belt_relaxation_length_longitudinal: f64,
    pub belt_relaxation_length_lateral: f64,
    pub belt_yaw_relaxation_length: f64,
    pub carcass_relaxation_length_longitudinal: f64,
    pub carcass_relaxation_length_lateral: f64,
    pub carcass_yaw_relaxation_length: f64,
}

impl Default for TireLayeredTransientState {
    fn default() -> Self {
        Self {
            tread: TireTransientLayerState::default(),
            belt: TireTransientLayerState::default(),
            carcass: TireTransientLayerState::default(),
            kappa_eff: 0.0,
            alpha_eff_rad: 0.0,
            tread_yaw_rad: 0.0,
            longitudinal_deflection_m: 0.0,
            lateral_deflection_m: 0.0,
            // Fast local tread shear.
            tread_relaxation_length_longitudinal: 0.16,
            tread_relaxation_length_lateral: 0.22,
            tread_yaw_relaxation_length: 0.28,
            // Belt / tread-band structural lag, close to the old single-layer
            // defaults so existing transient behavior remains familiar.
            belt_relaxation_length_longitudinal: 0.45,
            belt_relaxation_length_lateral: 0.65,
            belt_yaw_relaxation_length: 0.80,
            // Slow carcass-body deformation.
            carcass_relaxation_length_longitudinal: 1.05,
            carcass_relaxation_length_lateral: 1.35,
            carcass_yaw_relaxation_length: 1.60,
        }
    }
}

impl TireLayeredTransientState {
    pub fn reset(&mut self) {
        let lengths = (
            self.tread_relaxation_length_longitudinal,
            self.tread_relaxation_length_lateral,
            self.tread_yaw_relaxation_length,
            self.belt_relaxation_length_longitudinal,
            self.belt_relaxation_length_lateral,
            self.belt_yaw_relaxation_length,
            self.carcass_relaxation_length_longitudinal,
            self.carcass_relaxation_length_lateral,
            self.carcass_yaw_relaxation_length,
        );
        *self = Self {
            tread_relaxation_length_longitudinal: lengths.0,
            tread_relaxation_length_lateral: lengths.1,
            tread_yaw_relaxation_length: lengths.2,
            belt_relaxation_length_longitudinal: lengths.3,
            belt_relaxation_length_lateral: lengths.4,
            belt_yaw_relaxation_length: lengths.5,
            carcass_relaxation_length_longitudinal: lengths.6,
            carcass_relaxation_length_lateral: lengths.7,
            carcass_yaw_relaxation_length: lengths.8,
            ..Default::default()
        };
    }

    pub fn is_finite(&self) -> bool {
        self.tread.is_finite()
            && self.belt.is_finite()
            && self.carcass.is_finite()
            && self.kappa_eff.is_finite()
            && self.alpha_eff_rad.is_finite()
            && self.tread_yaw_rad.is_finite()
            && self.longitudinal_deflection_m.is_finite()
            && self.lateral_deflection_m.is_finite()
    }

    pub fn step(
        &mut self,
        params: &RealTireParams,
        state: &RealWheelState,
        terrain_normal: [f64; 3],
        dt: f64,
    ) -> RealWheelState {
        self.step_with_context(
            params,
            state,
            terrain_normal,
            dt,
            TireRelaxationContext::default(),
        )
    }

    pub fn step_with_context(
        &mut self,
        params: &RealTireParams,
        state: &RealWheelState,
        terrain_normal: [f64; 3],
        dt: f64,
        context: TireRelaxationContext,
    ) -> RealWheelState {
        if !finite_vec3(state.hub_pos) || !finite_vec3(state.hub_vel) || !finite_vec3(state.heading)
        {
            return *state;
        }
        let params = sanitize_brush_params(params);
        let normal = normalize(terrain_normal);
        let heading = normalize(project_to_ground(state.heading, normal));
        let side = normalize(cross(heading, normal));
        let v_long = dot(state.hub_vel, heading);
        let v_lat = dot(state.hub_vel, side);
        let denom = v_long.abs().max(params.relaxation_speed);
        let omega = finite_or(state.omega, 0.0);
        let raw_kappa = ((omega * params.radius - v_long) / denom).clamp(-8.0, 8.0);
        let raw_alpha = v_lat.atan2(denom).clamp(-1.2, 1.2);
        let speed = (v_long * v_long + v_lat * v_lat)
            .sqrt()
            .max(params.relaxation_speed);
        let ds = speed * finite_nonnegative(dt);
        let length_scale = relaxation_length_scale_from_values(
            params.pressure_pa,
            context.normal_load_n,
            context.carcass_temperature_c,
        );

        advance_tire_transient_layer(
            &mut self.tread,
            raw_kappa,
            raw_alpha,
            ds,
            TireTransientLayerLengths {
                longitudinal_m: self.tread_relaxation_length_longitudinal,
                lateral_m: self.tread_relaxation_length_lateral,
                yaw_m: self.tread_yaw_relaxation_length,
            }
            .scaled(length_scale * 0.85),
            params.radius,
        );
        advance_tire_transient_layer(
            &mut self.belt,
            raw_kappa,
            raw_alpha,
            ds,
            TireTransientLayerLengths {
                longitudinal_m: self.belt_relaxation_length_longitudinal,
                lateral_m: self.belt_relaxation_length_lateral,
                yaw_m: self.belt_yaw_relaxation_length,
            }
            .scaled(length_scale),
            params.radius,
        );
        advance_tire_transient_layer(
            &mut self.carcass,
            raw_kappa,
            raw_alpha,
            ds,
            TireTransientLayerLengths {
                longitudinal_m: self.carcass_relaxation_length_longitudinal,
                lateral_m: self.carcass_relaxation_length_lateral,
                yaw_m: self.carcass_yaw_relaxation_length,
            }
            .scaled(length_scale * 1.15),
            params.radius,
        );

        // Tread shear dominates the immediate local contact, but the belt and
        // carcass still carry enough weight that force does not unrealistically
        // jump to the fast layer alone.
        self.kappa_eff = 0.58 * self.tread.kappa_eff
            + 0.30 * self.belt.kappa_eff
            + 0.12 * self.carcass.kappa_eff;
        self.alpha_eff_rad = 0.55 * self.tread.alpha_eff_rad
            + 0.32 * self.belt.alpha_eff_rad
            + 0.13 * self.carcass.alpha_eff_rad;
        self.tread_yaw_rad =
            0.45 * self.tread.yaw_rad + 0.35 * self.belt.yaw_rad + 0.20 * self.carcass.yaw_rad;
        self.longitudinal_deflection_m = 0.35 * self.tread.longitudinal_deflection_m
            + 0.35 * self.belt.longitudinal_deflection_m
            + 0.30 * self.carcass.longitudinal_deflection_m;
        self.lateral_deflection_m = 0.35 * self.tread.lateral_deflection_m
            + 0.35 * self.belt.lateral_deflection_m
            + 0.30 * self.carcass.lateral_deflection_m;

        let v_long_eff = v_long;
        let v_lat_eff = self.alpha_eff_rad.tan().clamp(-4.0, 4.0) * denom;
        let omega_eff = (v_long_eff + self.kappa_eff * denom) / params.radius;
        RealWheelState {
            hub_pos: state.hub_pos,
            hub_vel: add(scale(heading, v_long_eff), scale(side, v_lat_eff)),
            heading,
            omega: omega_eff,
        }
    }
}

fn advance_tire_transient_layer(
    layer: &mut TireTransientLayerState,
    raw_kappa: f64,
    raw_alpha: f64,
    ds: f64,
    lengths: TireTransientLayerLengths,
    radius: f64,
) {
    let blend = |length: f64| {
        1.0 - (-finite_nonnegative(ds) / finite_positive_or(length, 0.5, 1.0e-4)).exp()
    };
    let bx = blend(lengths.longitudinal_m).clamp(0.0, 1.0);
    let by = blend(lengths.lateral_m).clamp(0.0, 1.0);
    let bt = blend(lengths.yaw_m).clamp(0.0, 1.0);
    layer.kappa_eff += (raw_kappa - layer.kappa_eff) * bx;
    layer.alpha_eff_rad += (raw_alpha - layer.alpha_eff_rad) * by;
    layer.yaw_rad += (raw_alpha - layer.yaw_rad) * bt;
    layer.longitudinal_deflection_m = layer.kappa_eff * radius;
    layer.lateral_deflection_m = layer.alpha_eff_rad.tan().clamp(-4.0, 4.0) * radius;
}

impl Default for RealTireTransientState {
    fn default() -> Self {
        Self {
            kappa_eff: 0.0,
            alpha_eff: 0.0,
            longitudinal_deflection: 0.0,
            lateral_deflection: 0.0,
            tread_yaw: 0.0,
            relaxation_length_longitudinal: 0.45,
            relaxation_length_lateral: 0.65,
            tread_yaw_relaxation_length: 0.80,
        }
    }
}

impl RealTireTransientState {
    pub fn reset(&mut self) {
        let lengths = (
            self.relaxation_length_longitudinal,
            self.relaxation_length_lateral,
            self.tread_yaw_relaxation_length,
        );
        *self = Self {
            relaxation_length_longitudinal: lengths.0,
            relaxation_length_lateral: lengths.1,
            tread_yaw_relaxation_length: lengths.2,
            ..Default::default()
        };
    }

    /// Advance relaxation state and return an effective wheel state for the
    /// brush law. The update is distance-based (`ds = |v| dt`) with a low-speed
    /// regularizer, matching physical relaxation-length behavior.
    pub fn step(
        &mut self,
        params: &RealTireParams,
        state: &RealWheelState,
        terrain_normal: [f64; 3],
        dt: f64,
    ) -> RealWheelState {
        self.step_with_context(
            params,
            state,
            terrain_normal,
            dt,
            TireRelaxationContext::default(),
        )
    }

    /// Advance relaxation state with pressure/load/temperature-sensitive
    /// effective relaxation lengths.
    pub fn step_with_context(
        &mut self,
        params: &RealTireParams,
        state: &RealWheelState,
        terrain_normal: [f64; 3],
        dt: f64,
        context: TireRelaxationContext,
    ) -> RealWheelState {
        if !finite_vec3(state.hub_pos) || !finite_vec3(state.hub_vel) || !finite_vec3(state.heading)
        {
            return *state;
        }
        let params = sanitize_brush_params(params);
        let normal = normalize(terrain_normal);
        let heading = normalize(project_to_ground(state.heading, normal));
        let side = normalize(cross(heading, normal));
        let v_long = dot(state.hub_vel, heading);
        let v_lat = dot(state.hub_vel, side);
        let denom = v_long.abs().max(params.relaxation_speed);
        let omega = finite_or(state.omega, 0.0);
        let raw_kappa = ((omega * params.radius - v_long) / denom).clamp(-8.0, 8.0);
        let raw_alpha = v_lat.atan2(denom).clamp(-1.2, 1.2);
        let speed = (v_long * v_long + v_lat * v_lat)
            .sqrt()
            .max(params.relaxation_speed);
        let ds = speed * finite_nonnegative(dt);
        let length_scale = relaxation_length_scale_from_values(
            params.pressure_pa,
            context.normal_load_n,
            context.carcass_temperature_c,
        );
        let blend = |length: f64| 1.0 - (-ds / finite_positive_or(length, 0.5, 1.0e-4)).exp();
        let bx = blend(self.relaxation_length_longitudinal * length_scale).clamp(0.0, 1.0);
        let by = blend(self.relaxation_length_lateral * length_scale).clamp(0.0, 1.0);
        let bt = blend(self.tread_yaw_relaxation_length * length_scale).clamp(0.0, 1.0);
        self.kappa_eff += (raw_kappa - self.kappa_eff) * bx;
        self.alpha_eff += (raw_alpha - self.alpha_eff) * by;
        self.tread_yaw += (raw_alpha - self.tread_yaw) * bt;
        self.longitudinal_deflection = self.kappa_eff * params.radius;
        self.lateral_deflection = self.alpha_eff.tan().clamp(-4.0, 4.0) * params.radius;

        let v_long_eff = v_long;
        let v_lat_eff = self.alpha_eff.tan().clamp(-4.0, 4.0) * denom;
        let omega_eff = (v_long_eff + self.kappa_eff * denom) / params.radius;
        RealWheelState {
            hub_pos: state.hub_pos,
            hub_vel: add(scale(heading, v_long_eff), scale(side, v_lat_eff)),
            heading,
            omega: omega_eff,
        }
    }
}

/// Convenience wrapper: advance a transient relaxation state and evaluate the
/// camber-aware brush contact from the effective wheel state.
pub fn evaluate_brush_tire_transient(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
    transient: &mut RealTireTransientState,
    dt: f64,
) -> RealTireContact {
    let normal = terrain.normal(state.hub_pos[0], state.hub_pos[2]);
    let effective = transient.step(params, state, normal, dt);
    evaluate_brush_tire_oriented(params, &effective, terrain, material, orientation)
}

/// Context-aware transient tire evaluation. Use this when the caller already
/// has the current wheel load and carcass temperature from a rig, vehicle, or
/// thermal model.
pub fn evaluate_brush_tire_transient_with_context(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
    transient: &mut RealTireTransientState,
    dt: f64,
    context: TireRelaxationContext,
) -> RealTireContact {
    let normal = terrain.normal(state.hub_pos[0], state.hub_pos[2]);
    let effective = transient.step_with_context(params, state, normal, dt, context);
    evaluate_brush_tire_oriented(params, &effective, terrain, material, orientation)
}

/// Evaluate brush contact with the layered tread/belt/carcass transient model.
pub fn evaluate_brush_tire_layered_transient(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
    transient: &mut TireLayeredTransientState,
    dt: f64,
) -> RealTireContact {
    let normal = terrain.normal(state.hub_pos[0], state.hub_pos[2]);
    let effective = transient.step(params, state, normal, dt);
    evaluate_brush_tire_oriented(params, &effective, terrain, material, orientation)
}

/// Context-aware layered transient tire evaluation. This is the preferred
/// path for vehicle/rig callers that know wheel load and carcass temperature.
pub fn evaluate_brush_tire_layered_transient_with_context(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
    transient: &mut TireLayeredTransientState,
    dt: f64,
    context: TireRelaxationContext,
) -> RealTireContact {
    let normal = terrain.normal(state.hub_pos[0], state.hub_pos[2]);
    let effective = transient.step_with_context(params, state, normal, dt, context);
    evaluate_brush_tire_oriented(params, &effective, terrain, material, orientation)
}
