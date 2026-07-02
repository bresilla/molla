//! Per-tire reusable parameters (geometry, inflation, stiffness).
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::finite_or;

/// Parameters for one deformable tire contact patch.
#[derive(Clone, Copy, Debug)]
pub struct RealTireParams {
    /// Unloaded tire radius, metres. Typical passenger/light off-road values
    /// are roughly `0.25..0.45`.
    pub radius: f64,
    /// Tread/contact width, metres. Keep this as the loaded tread width, not
    /// cosmetic sidewall-to-sidewall width.
    pub width: f64,
    /// Inflation pressure, pascals. Viewer sliders may override this, but
    /// reusable presets should carry the baseline value.
    pub pressure_pa: f64,
    /// Number of circumferential samples used for the contact patch.
    pub segments: usize,
    /// Carcass normal stiffness per contact area (Pa/m). This is the part
    /// that still supports load at zero pressure.
    pub carcass_normal_stiffness: f64,
    /// Tread longitudinal shear stiffness per contact area (N/m³).
    pub tread_shear_stiffness: f64,
    /// Structural lateral bias as a fraction of local normal load.
    ///
    /// This is the reusable conicity / ply-steer hook: a nonzero value
    /// produces a small lateral force bias from tire construction itself,
    /// without viewer-side steering drift. Keep presets at zero unless a tire
    /// is intentionally asymmetric.
    pub structural_lateral_bias_per_normal: f64,
    /// Slip speed used to regularize bristle displacement near standstill.
    pub relaxation_speed: f64,
}

impl Default for RealTireParams {
    fn default() -> Self {
        Self::passenger_radial()
    }
}

impl RealTireParams {
    /// Baseline hand-tuned passenger radial preset.
    ///
    /// This is the source-of-truth replacement for scattered viewer constants:
    /// examples should override this preset instead of inventing unrelated
    /// tire parameters locally. It is not calibrated to a measured tire.
    pub const fn passenger_radial() -> Self {
        Self {
            radius: 0.32,
            width: 0.185,
            pressure_pa: 180_000.0,
            segments: 48,
            carcass_normal_stiffness: 4.0e5,
            tread_shear_stiffness: 2.6e6,
            structural_lateral_bias_per_normal: 0.0,
            relaxation_speed: 0.35,
        }
    }

    /// Softer off-road / aired-down hand-tuned preset for loose terrain.
    pub const fn offroad_soft() -> Self {
        Self {
            radius: 0.38,
            width: 0.24,
            pressure_pa: 95_000.0,
            segments: 64,
            carcass_normal_stiffness: 3.2e5,
            tread_shear_stiffness: 2.1e6,
            structural_lateral_bias_per_normal: 0.0,
            relaxation_speed: 0.28,
        }
    }

    /// Stiff, high-pressure hand-tuned preset for low-deflection comparisons.
    pub const fn high_pressure_reference() -> Self {
        Self {
            radius: 0.32,
            width: 0.185,
            pressure_pa: 320_000.0,
            segments: 64,
            carcass_normal_stiffness: 5.2e5,
            tread_shear_stiffness: 3.1e6,
            structural_lateral_bias_per_normal: 0.0,
            relaxation_speed: 0.42,
        }
    }

    /// Return a copy with geometry and inflation overridden by a viewer/test
    /// scenario. This keeps all other hand-tuned physics parameters anchored
    /// to the named preset.
    pub fn with_geometry_and_pressure(
        mut self,
        radius_m: f64,
        width_m: f64,
        pressure_pa: f64,
    ) -> Self {
        self.radius = radius_m;
        self.width = width_m;
        self.pressure_pa = pressure_pa;
        self
    }

    /// Return a copy with a different circumferential contact resolution.
    pub fn with_segments(mut self, segments: usize) -> Self {
        self.segments = segments;
        self
    }

    /// Return a copy with structural conicity / ply-steer lateral bias.
    ///
    /// The value is dimensionless: `0.01` means the tire contributes roughly
    /// one percent of local normal load as lateral bias before the shared
    /// friction cap. This is a tire-construction property, not a steering hack.
    pub fn with_structural_lateral_bias(mut self, bias_per_normal: f64) -> Self {
        self.structural_lateral_bias_per_normal =
            finite_or(bias_per_normal, 0.0).clamp(-0.08, 0.08);
        self
    }
}
