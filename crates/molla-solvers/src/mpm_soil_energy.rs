//! Mechanical energy represented by a quadratic-APIC soil checkpoint.

use super::*;
use glam::{DMat3, DVec3};

/// Particle energy in joules, with frozen plastic history and current gravity.
/// Elastic energy is unavailable for inverted/degenerate F or stretches outside
/// the shader's stress-clamp domain. Heat, plastic/contact work, machine energy
/// and grid state are excluded; this is not a work-balance verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct MpmSoilEnergy {
    pub particles: u32,
    pub reference_height: f64,
    pub translational_kinetic: f64,
    pub affine_kinetic: f64,
    pub gravitational_potential: f64,
    pub elastic_potential: Option<f64>,
    pub elastic_unavailable_particles: u32,
}

impl MpmSoilEnergy {
    pub fn mechanical_energy(&self) -> Option<f64> {
        self.elastic_potential.map(|elastic| {
            self.translational_kinetic
                + self.affine_kinetic
                + self.gravitational_potential
                + elastic
        })
    }
}

impl MpmSoilCheckpoint {
    /// Evaluate f64 particle energies without GPU work or per-particle scratch.
    /// APIC kinetic energy uses D=h²I/4; elastic energy uses the fixed-corotated
    /// potential and per-material frozen hardening, not a plastic-dissipation model.
    pub fn energy_at_height(&self, reference_height: f64) -> Result<MpmSoilEnergy> {
        self.validate()?;
        if !reference_height.is_finite() {
            return Err(invalid("nonfinite energy reference height"));
        }
        let streams: [&[u32]; 8] = std::array::from_fn(|s| self.stream(s));
        let mut result = MpmSoilEnergy {
            particles: self.count,
            reference_height,
            translational_kinetic: 0.0,
            affine_kinetic: 0.0,
            gravitational_potential: 0.0,
            elastic_potential: Some(0.0),
            elastic_unavailable_particles: 0,
        };
        let d = 0.25 * f64::from(self.cfg.cell_size).powi(2);
        let youngs = f64::from(self.cfg.youngs);
        let poisson = f64::from(self.cfg.poisson);
        let mu = youngs / (2.0 * (1.0 + poisson));
        let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        for i in 0..self.count as usize {
            let value =
                |s: usize, k: usize| f64::from(f32::from_bits(streams[s][i * STRIDES[s] + k]));
            let vector = |s, k| DVec3::new(value(s, k), value(s, k + 1), value(s, k + 2));
            let mass = value(5, 0);
            result.translational_kinetic += 0.5 * mass * vector(1, 0).length_squared();
            result.affine_kinetic += 0.5
                * mass
                * d
                * (0..3)
                    .map(|col| vector(2, col * 4).length_squared())
                    .sum::<f64>();
            result.gravitational_potential -=
                f64::from(self.cfg.gravity_y) * mass * (value(0, 1) - reference_height);
            let f = DMat3::from_cols(vector(3, 0), vector(3, 4), vector(3, 8));
            let (_, s, _) = molla_math::dense3::svd3(f);
            if f.determinant() <= 0.0
                || !s.is_finite()
                || s.min_element() < f64::from(0.1f32)
                || s.max_element() > 10.0
            {
                result.elastic_unavailable_particles += 1;
                result.elastic_potential = None;
                continue;
            }
            let material = self
                .materials
                .get(streams[7][i] as usize)
                .copied()
                .unwrap_or(self.cfg.rheology);
            let hardening = match material {
                MpmRheology::Sand { .. } => value(6, 0).clamp(-3.0, 3.0).exp(),
                MpmRheology::Snow { hardening, .. } => {
                    (-f64::from(hardening) * value(6, 0)).clamp(-3.0, 3.0).exp()
                }
                MpmRheology::Viscous { .. } => 1.0,
            };
            let strain = s - DVec3::ONE;
            let j_minus_one = s.x * s.y * s.z - 1.0;
            let energy = value(4, 0)
                * hardening
                * (mu * strain.length_squared() + 0.5 * lambda * j_minus_one * j_minus_one);
            if let Some(total) = &mut result.elastic_potential {
                *total += energy;
            }
        }
        if [
            result.translational_kinetic,
            result.affine_kinetic,
            result.gravitational_potential,
        ]
        .into_iter()
        .chain(result.elastic_potential)
        .chain(result.mechanical_energy())
        .any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite accumulated particle energy"));
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "mpm_soil_energy_tests.rs"]
mod tests;
