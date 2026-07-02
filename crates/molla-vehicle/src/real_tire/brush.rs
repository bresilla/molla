//! `RealBrushTireModel` [`TireModel`] adapter and shared brush parameter
//! sanitization / pressure-stiffness helpers.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::FlatTerrain;
use crate::tire::TireModel;

use super::internal::{finite_nonnegative, finite_or, finite_positive_or};
use super::{RealTireParams, SurfaceMaterial, tire_rig_point};

/// [`TireModel`] adapter backed by the reusable pressure/contact brush tire.
///
/// The legacy [`Vehicle`](crate::Vehicle) integration point only passes
/// `(slip_ratio, slip_angle, normal_load, friction)`, not full wheel pose or a
/// terrain object. This adapter keeps those older vehicle scenes on the native
/// real-tire force law by evaluating the brush contact on a synthetic flat
/// patch at a nominal speed and target normal load. It is not a FEM tire and
/// does not add calibration data; it is a bridge for broader runtime adoption
/// until those scenes move to resident wheel/chassis/FEM ownership.
#[derive(Clone, Copy, Debug)]
pub struct RealBrushTireModel {
    pub params: RealTireParams,
    pub material: SurfaceMaterial,
    pub nominal_speed_m_per_s: f64,
}

impl RealBrushTireModel {
    pub fn new(params: RealTireParams, material: SurfaceMaterial) -> Self {
        Self {
            params,
            material,
            nominal_speed_m_per_s: 10.0,
        }
    }

    pub fn passenger_radial() -> Self {
        Self::new(
            RealTireParams::passenger_radial(),
            SurfaceMaterial::dry_asphalt(),
        )
    }

    pub fn offroad_soft() -> Self {
        Self::new(RealTireParams::offroad_soft(), SurfaceMaterial::soft_soil())
    }

    pub fn with_nominal_speed(mut self, speed_m_per_s: f64) -> Self {
        self.nominal_speed_m_per_s = finite_positive_or(speed_m_per_s, 10.0, 0.25);
        self
    }

    fn material_for_friction(&self, friction: f64) -> SurfaceMaterial {
        let mut material = self.material;
        let mu_static = finite_nonnegative(friction).max(1.0e-4);
        let scale = mu_static / finite_positive_or(material.mu_static, 1.0, 1.0e-6);
        material.mu_static = mu_static;
        material.mu_dynamic = (material.mu_dynamic * scale).min(mu_static).max(0.0);
        material.cohesion = finite_nonnegative(material.cohesion) * scale;
        material
    }
}

impl TireModel for RealBrushTireModel {
    fn evaluate(
        &self,
        slip_ratio: f64,
        slip_angle: f64,
        fz: f64,
        friction: f64,
    ) -> (f64, f64, f64) {
        let load_n = finite_nonnegative(fz);
        if load_n <= 1.0 {
            return (0.0, 0.0, 0.0);
        }
        let material = self.material_for_friction(friction);
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        let point = tire_rig_point(
            &self.params,
            &terrain,
            &material,
            load_n,
            self.nominal_speed_m_per_s,
            slip_ratio,
            slip_angle,
            0.0,
        );
        (
            point.telemetry.longitudinal_force_n,
            point.telemetry.lateral_force_n,
            point.telemetry.aligning_moment_nm,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BrushParams {
    pub(crate) radius: f64,
    pub(crate) width: f64,
    pub(crate) pressure_pa: f64,
    pub(crate) segments: usize,
    pub(crate) carcass_normal_stiffness: f64,
    pub(crate) tread_shear_stiffness: f64,
    pub(crate) structural_lateral_bias_per_normal: f64,
    pub(crate) relaxation_speed: f64,
}

pub(crate) fn sanitize_brush_params(params: &RealTireParams) -> BrushParams {
    let d = RealTireParams::default();
    BrushParams {
        radius: finite_positive_or(params.radius, d.radius, 1.0e-4),
        width: finite_positive_or(params.width, d.width, 1.0e-4),
        pressure_pa: finite_nonnegative(params.pressure_pa),
        // Hard cap protects the CPU contact path from accidental million-
        // segment inputs; the GPU kernel has its own tighter cap.
        segments: params.segments.clamp(8, 512),
        carcass_normal_stiffness: finite_positive_or(
            params.carcass_normal_stiffness,
            d.carcass_normal_stiffness,
            1.0,
        ),
        tread_shear_stiffness: finite_nonnegative(params.tread_shear_stiffness),
        structural_lateral_bias_per_normal: finite_or(
            params.structural_lateral_bias_per_normal,
            d.structural_lateral_bias_per_normal,
        )
        .clamp(-0.08, 0.08),
        relaxation_speed: finite_positive_or(params.relaxation_speed, d.relaxation_speed, 1.0e-4),
    }
}

use super::internal::smoothstep01;

pub(crate) fn brush_load_sensitivity(normal_pressure_pa: f64) -> (f64, f64) {
    let load_ratio =
        (finite_positive_or(normal_pressure_pa, 180_000.0, 1.0) / 220_000.0).clamp(0.15, 6.0);
    // Peak friction is mildly load-sensitive: high local pressure reduces
    // usable μ, while lightly loaded cells get a small boost. The exponent is
    // deliberately modest so material still dominates ultimate grip.
    let mu_factor = load_ratio.powf(-0.07).clamp(0.72, 1.12);
    // Tread/carcass stiffness grows sub-linearly with normal pressure.
    let stiffness_factor = load_ratio.powf(0.25).clamp(0.55, 1.45);
    (mu_factor, stiffness_factor)
}

pub(crate) fn brush_pressure_stiffness_factor(pressure_pa: f64) -> f64 {
    let pressure_ratio =
        (finite_positive_or(pressure_pa, 180_000.0, 1.0) / 180_000.0).clamp(0.25, 2.50);
    // Inflation tension stiffens the tread/carcass response, but this is only
    // a broad hand-tuned effect: pressure changes should shift cornering and
    // longitudinal stiffness without becoming a hidden force multiplier.
    pressure_ratio.powf(0.14).clamp(0.78, 1.14)
}

/// Pressure-dependent patch gain for the linear brush deflection profile.
///
/// At fixed load the brush patch half-length scales as `a ∝ k_n^(-1/3)`
/// (parabolic penetration of a circular arc), where `k_n` is the normal
/// stiffness per area `carcass + 3·P/R`. Bristle force before saturation
/// scales with patch length, so the deflection gain is the patch-length
/// ratio between the 180 kPa reference inflation and the current one.
/// Lower pressure → longer patch → proportionally more tractive force
/// before the friction cap, which is the aired-down-tire traction effect.
pub(crate) fn brush_pressure_patch_gain(
    carcass_normal_stiffness: f64,
    pressure_pa: f64,
    radius: f64,
) -> f64 {
    let radius = finite_positive_or(radius, 0.32, 1.0e-3);
    let carcass = finite_nonnegative(carcass_normal_stiffness);
    let k_now = (carcass + 3.0 * finite_positive_or(pressure_pa, 180_000.0, 1.0) / radius).max(1.0);
    let k_ref = (carcass + 3.0 * 180_000.0 / radius).max(1.0);
    (k_ref / k_now).powf(1.0 / 3.0).clamp(0.55, 1.80)
}

/// Linear leading-edge bristle deflection profile (textbook brush model):
/// zero where the tread enters contact, growing toward the trailing edge.
///
/// `leading_x` is the patch edge where material enters contact (front for
/// forward travel), `patch_extent` the leading-to-trailing patch span, and
/// `cell_len` one contact cell's length along the rolling direction. The
/// half-cell offset keeps the patch average at exactly 1 for uniformly
/// spaced cells (including a single-cell grazing patch), so the calibrated
/// total force is preserved and only the in-patch distribution changes:
/// trailing cells carry more bristle deflection and saturate first, which
/// is what produces pneumatic trail and patch-length-dependent stiffness.
pub(crate) fn brush_bristle_profile(
    local_x: f64,
    leading_x: f64,
    patch_extent: f64,
    cell_len: f64,
) -> f64 {
    let cell = finite_positive_or(cell_len, 1.0e-3, 1.0e-6).max(1.0e-6);
    let extent = finite_nonnegative(patch_extent);
    let from_leading = (leading_x - local_x).abs();
    ((2.0 * (from_leading + 0.5 * cell)) / (extent + cell)).clamp(0.0, 4.0)
}

pub(crate) fn smooth_combined_brush_force(
    trial_long: f64,
    trial_lat: f64,
    static_cap: f64,
    dynamic_cap: f64,
) -> (f64, f64, bool, f64, f64) {
    let trial_mag = (trial_long * trial_long + trial_lat * trial_lat).sqrt();
    let static_cap = finite_nonnegative(static_cap);
    let dynamic_cap = finite_nonnegative(dynamic_cap).min(static_cap);
    if trial_mag <= 1.0e-12 || static_cap <= 1.0e-12 {
        return (0.0, 0.0, false, static_cap, 0.0);
    }

    let utilization = trial_mag / static_cap.max(1.0e-12);
    let slide_blend = smoothstep01((utilization - 0.75) / 0.75);
    let cap = (static_cap + (dynamic_cap - static_cap) * slide_blend).max(1.0e-12);
    // Smooth vector saturation. Low utilization is nearly linear; high
    // utilization approaches the blended static/dynamic cap without the old
    // discontinuous static->dynamic drop.
    let u = trial_mag / cap;
    let scale = (1.0 + u.powi(8)).powf(-1.0 / 8.0);
    let ft_long = trial_long * scale;
    let ft_lat = trial_lat * scale;
    let actual_mag = trial_mag * scale;
    let sliding = utilization > 0.90 || actual_mag < trial_mag * 0.995;
    (ft_long, ft_lat, sliding, static_cap, actual_mag)
}
