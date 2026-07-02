//! Thin wet/film surface state for tire/road contact.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::{finite_nonnegative, finite_positive_or};
use super::surface_material::sanitized_surface_material;
use super::{SurfaceMaterial, TireWearState};

/// Thin wet/film surface state for tire/road contact.
///
/// This is deliberately not a calibrated aquaplaning model. It is a
/// physics-shaped drainage/shear reduction that turns water depth, speed, and
/// remaining tread void depth into a material modifier:
///
/// - deeper films reduce static/dynamic shear;
/// - speed makes the same film harder to drain;
/// - worn tread loses wet performance faster than fresh tread;
/// - rolling loss rises from water displacement.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceFilmState {
    /// Water/film depth above the road surface.
    pub water_depth_m: f64,
    /// Effective drainage/void depth of a fresh tread pattern.
    pub fresh_tread_void_depth_m: f64,
    /// Characteristic speed where drainage starts to struggle.
    pub drainage_speed_mps: f64,
    /// Extra rolling resistance from pushing water film at speed.
    pub water_drag_rolling_per_mm: f64,
}

impl Default for SurfaceFilmState {
    fn default() -> Self {
        Self {
            water_depth_m: 0.0,
            fresh_tread_void_depth_m: 0.004,
            drainage_speed_mps: 18.0,
            water_drag_rolling_per_mm: 0.0025,
        }
    }
}

impl SurfaceFilmState {
    pub fn dry() -> Self {
        Self::default()
    }

    /// Barely damp road film preset. Useful for validation that wet logic is
    /// active without approaching standing-water behavior.
    pub const fn damp() -> Self {
        Self {
            water_depth_m: 0.00025,
            fresh_tread_void_depth_m: 0.004,
            drainage_speed_mps: 22.0,
            water_drag_rolling_per_mm: 0.0012,
        }
    }

    pub fn wet(water_depth_m: f64) -> Self {
        Self {
            water_depth_m,
            ..Self::default()
        }
    }

    /// Ordinary wet-road preset with a thin water layer.
    pub const fn wet_road() -> Self {
        Self {
            water_depth_m: 0.0012,
            fresh_tread_void_depth_m: 0.004,
            drainage_speed_mps: 18.0,
            water_drag_rolling_per_mm: 0.0025,
        }
    }

    /// Standing-water validation preset. This remains a deterministic film
    /// modifier, not a calibrated aquaplaning model.
    pub const fn standing_water() -> Self {
        Self {
            water_depth_m: 0.0045,
            fresh_tread_void_depth_m: 0.004,
            drainage_speed_mps: 14.0,
            water_drag_rolling_per_mm: 0.0045,
        }
    }

    /// Apply film/wet-road effects to a contact material.
    ///
    /// `speed_mps` is the local tire/road sweep speed. `wear` provides tread
    /// depth; if the tire is worn, drainage capacity shrinks and the same film
    /// produces a larger shear reduction.
    pub fn apply_to(
        &self,
        material: SurfaceMaterial,
        speed_mps: f64,
        wear: &TireWearState,
    ) -> SurfaceMaterial {
        let water_depth = finite_nonnegative(self.water_depth_m);
        if water_depth <= 1.0e-7 {
            return sanitized_surface_material(&material);
        }

        let speed = finite_nonnegative(speed_mps);
        let remaining_tread = (1.0 - wear.depth_m / wear.max_depth_m.max(1.0e-9)).clamp(0.0, 1.0);
        let void_depth = finite_positive_or(self.fresh_tread_void_depth_m, 0.004, 1.0e-5)
            * (0.18 + 0.82 * remaining_tread);
        let film_ratio = (water_depth / (void_depth + 2.0e-4)).clamp(0.0, 8.0);
        let speed_ratio = speed / finite_positive_or(self.drainage_speed_mps, 18.0, 0.5).max(0.5);
        let severity = film_ratio * (1.0 + 0.85 * speed_ratio.powf(1.35));
        let static_factor = (1.0 / (1.0 + 0.48 * severity)).clamp(0.12, 1.0);
        let dynamic_factor = (1.0 / (1.0 + 0.68 * severity)).clamp(0.08, 1.0);
        let damping_factor = (1.0 / (1.0 + 0.20 * severity)).clamp(0.35, 1.0);

        let mut out = sanitized_surface_material(&material);
        out.mu_static *= static_factor;
        out.mu_dynamic *= dynamic_factor.min(static_factor);
        out.bristle_damping *= damping_factor;
        out.cohesion *= (1.0 - 0.22 * severity).clamp(0.0, 1.0);
        out.rolling_resistance += finite_nonnegative(self.water_drag_rolling_per_mm)
            * (water_depth * 1000.0)
            * (1.0 + 0.08 * speed);
        sanitized_surface_material(&out)
    }
}
