//! Mass and quadratic-APIC momentum represented by a soil checkpoint.

use super::*;
use glam::DVec3;

/// Particle-only moments in world axes about a fixed world-space origin.
/// Angular momentum includes quadratic-APIC affine spin with D = h² I / 4.
/// Excludes grid state, machines and external impulses; not a conservation verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct MpmSoilMomenta {
    pub particles: u32,
    pub origin: [f64; 3],
    pub mass: f64,
    pub linear_momentum: [f64; 3],
    pub angular_momentum: [f64; 3],
    pub affine_angular_momentum: [f64; 3],
}

impl MpmSoilCheckpoint {
    /// Sum actual particle masses and momenta in f64 without GPU work or scratch.
    /// Uses the solver's quadratic-APIC representation even near truncated grid
    /// support; boundary transfer losses are not included as external impulses.
    pub fn momenta_about(&self, origin: [f64; 3]) -> Result<MpmSoilMomenta> {
        self.validate()?;
        let origin_vector = DVec3::from_array(origin);
        if !origin_vector.is_finite() {
            return Err(invalid("nonfinite momentum origin"));
        }
        let streams = [
            self.stream(0),
            self.stream(1),
            self.stream(2),
            self.stream(5),
        ];
        let mut mass = 0.0;
        let mut linear = DVec3::ZERO;
        let mut orbital = DVec3::ZERO;
        let mut affine = DVec3::ZERO;
        let d = 0.25 * f64::from(self.cfg.cell_size).powi(2);
        for i in 0..self.count as usize {
            let scalar = |s: usize, k: usize| f64::from(f32::from_bits(streams[s][k]));
            let vector =
                |s| DVec3::new(scalar(s, 4 * i), scalar(s, 4 * i + 1), scalar(s, 4 * i + 2));
            let m = scalar(3, i);
            let momentum = m * vector(1);
            let c = |k| scalar(2, 12 * i + k);
            mass += m;
            linear += momentum;
            orbital += (vector(0) - origin_vector).cross(momentum);
            affine += (m * d) * DVec3::new(c(6) - c(9), c(8) - c(2), c(1) - c(4));
        }
        let angular = orbital + affine;
        if !mass.is_finite() || !linear.is_finite() || !angular.is_finite() || !affine.is_finite() {
            return Err(invalid("nonfinite accumulated particle momentum"));
        }
        Ok(MpmSoilMomenta {
            particles: self.count,
            origin,
            mass,
            linear_momentum: linear.to_array(),
            angular_momentum: angular.to_array(),
            affine_angular_momentum: affine.to_array(),
        })
    }
}

#[cfg(test)]
#[path = "mpm_soil_momenta_tests.rs"]
mod tests;
