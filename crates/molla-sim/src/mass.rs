//! Rigid mass properties about a local center of mass.

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, Transform, Vec3, dense3::eig3_sym};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    pub mass: Real,
    pub local_com: Vec3,
    /// Inertia about `local_com`, expressed in the containing frame.
    pub inertia: Mat3,
}

impl MassProperties {
    pub const ZERO: Self = Self {
        mass: 0.0,
        local_com: Vec3::ZERO,
        inertia: Mat3::ZERO,
    };

    pub fn validate(self) -> Result<()> {
        let invalid = || Error::Build("invalid rigid mass properties".into());
        if !self.mass.is_finite()
            || self.mass < 0.0
            || (self.mass > 0.0 && !(1.0 / self.mass).is_finite())
            || !self.local_com.is_finite()
            || !self.inertia.is_finite()
            || (self.mass == 0.0 && self.inertia != Mat3::ZERO)
        {
            return Err(invalid());
        }
        let scale = magnitude(self.inertia);
        if scale > 0.0 {
            let normalized = self.inertia / scale;
            if magnitude(normalized - normalized.transpose()) > 1e-12 {
                return Err(invalid());
            }
            let (moments, _) = eig3_sym(normalized);
            if moments.min_element() < -1e-12
                || 2.0 * moments.max_element() > moments.element_sum() + 1e-12
            {
                return Err(invalid());
            }
        }
        Ok(())
    }

    /// Principal moments and their matching orthonormal axes, without sorting.
    pub fn principal(self) -> Result<(Vec3, Mat3)> {
        self.validate()?;
        let scale = magnitude(self.inertia);
        if scale == 0.0 {
            return Ok((Vec3::ZERO, Mat3::IDENTITY));
        }
        let (moments, axes) = eig3_sym(self.inertia / scale);
        Ok((moments.max(Vec3::ZERO) * scale, axes))
    }

    /// Pseudoinverse; moments below `1e-14` of the largest have zero response.
    pub fn inverse_inertia(self) -> Result<Mat3> {
        let (moments, axes) = self.principal()?;
        let floor = moments.max_element() * 1e-14;
        let inverse = moments
            .to_array()
            .map(|v| if v > floor { 1.0 / v } else { 0.0 });
        let result = axes * Mat3::from_diagonal(Vec3::from_array(inverse)) * axes.transpose();
        if !result.is_finite() {
            return Err(Error::Build("rigid inverse inertia overflow".into()));
        }
        Ok(result)
    }

    pub fn transformed(self, pose: Transform) -> Result<Self> {
        self.validate()?;
        if !pose.position.is_finite()
            || !pose.rotation.is_finite()
            || (pose.rotation.length_squared() - 1.0).abs() > 1e-6
        {
            return Err(Error::Build("invalid mass-properties transform".into()));
        }
        let rotation = Mat3::from_quat(pose.rotation);
        let result = Self {
            local_com: pose.position + pose.rotation * self.local_com,
            inertia: rotation * self.inertia * rotation.transpose(),
            ..self
        };
        result.validate()?;
        Ok(result)
    }

    /// Combine properties in the same frame using the parallel-axis theorem.
    pub fn combined(self, other: Self) -> Result<Self> {
        self.validate()?;
        other.validate()?;
        if self.mass == 0.0 {
            return Ok(other);
        }
        if other.mass == 0.0 {
            return Ok(self);
        }
        let mass = self.mass + other.mass;
        let local_com = self.local_com * (self.mass / mass) + other.local_com * (other.mass / mass);
        let shift = |p: Self| {
            let d = p.local_com - local_com;
            (Mat3::from_diagonal(Vec3::splat(d.length_squared()))
                - Mat3::from_cols(d * d.x, d * d.y, d * d.z))
                * p.mass
        };
        let result = Self {
            mass,
            local_com,
            inertia: self.inertia + other.inertia + shift(self) + shift(other),
        };
        result.validate()?;
        Ok(result)
    }
}

fn magnitude(matrix: Mat3) -> Real {
    matrix
        .to_cols_array()
        .into_iter()
        .map(Real::abs)
        .fold(0.0, Real::max)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderMass {
    Density(Real),
    Mass(Real),
}

impl Default for ColliderMass {
    fn default() -> Self {
        Self::Density(1.0)
    }
}

impl ColliderMass {
    pub fn validate(self) -> Result<()> {
        let (Self::Density(value) | Self::Mass(value)) = self;
        if !value.is_finite() || value < 0.0 {
            return Err(Error::Build("invalid collider density or mass".into()));
        }
        Ok(())
    }
}
