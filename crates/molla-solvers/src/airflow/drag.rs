use super::invalid;
use molla_core::Result;
use molla_math::Vec3;

/// Quadratic drag on a cylinder, integrated at weighted quadrature points.
#[derive(Clone, Copy, Debug)]
pub struct CylinderDrag {
    pub radius: f64,
    pub length: f64,
    pub density: f64,
    pub transverse_coefficient: f64,
    pub axial_coefficient: f64,
}

impl CylinderDrag {
    /// Force in newtons; inverse_mass=0 denotes a fixed obstacle, and weights sum to one.
    pub fn sample_force(
        self,
        relative_velocity: Vec3,
        axis: Vec3,
        inverse_mass: f64,
        dt: f64,
        weight: f64,
    ) -> Result<Vec3> {
        if [self.radius, self.length, self.density, dt, weight]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || [
                self.transverse_coefficient,
                self.axial_coefficient,
                inverse_mass,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || weight > 1.0
            || !relative_velocity.is_finite()
            || !axis.is_finite()
            || (axis.length_squared() - 1.0).abs() > 1e-6
        {
            return Err(invalid("invalid cylinder drag sample"));
        }
        let along = axis * relative_velocity.dot(axis);
        let across = relative_velocity - along;
        let cross_drag = self.density
            * self.transverse_coefficient
            * self.radius
            * self.length
            * weight
            * across.length();
        let axial_drag = 0.5
            * self.density
            * self.axial_coefficient
            * std::f64::consts::PI
            * self.radius.powi(2)
            * weight
            * along.length();
        let denominator = 1.0 + dt * (cross_drag + axial_drag) * inverse_mass / weight;
        let force = (across * cross_drag + along * axial_drag) / denominator;
        if !force.is_finite() {
            return Err(invalid("cylinder drag overflow"));
        }
        Ok(force)
    }
}
