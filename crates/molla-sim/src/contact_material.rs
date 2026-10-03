//! Rigid-contact material parameters and coefficient combination.

use molla_core::{Error, Result};
use molla_math::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrictionCoefficients {
    pub mu_static: Real,
    pub mu_dynamic: Real,
}

impl FrictionCoefficients {
    pub fn validate(self) -> Result<()> {
        if !self.mu_static.is_finite()
            || !self.mu_dynamic.is_finite()
            || self.mu_dynamic < 0.0
            || self.mu_static < self.mu_dynamic
        {
            return Err(Error::Build(
                "friction requires finite 0 <= mu_dynamic <= mu_static".into(),
            ));
        }
        Ok(())
    }

    pub fn uniform(coefficient: Real) -> Self {
        Self {
            mu_static: coefficient,
            mu_dynamic: coefficient,
        }
    }
}

/// Pair rule priority is Average < Min < Multiply < Max.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum CoefficientCombineRule {
    #[default]
    Average,
    Min,
    Multiply,
    Max,
}

impl CoefficientCombineRule {
    pub fn combine(self, other: Self, a: Real, b: Real) -> Real {
        match self.max(other) {
            Self::Average => 0.5 * a + 0.5 * b,
            Self::Min => a.min(b),
            Self::Multiply => a * b,
            Self::Max => a.max(b),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactMaterial {
    /// Explicit pair; None uses the legacy shape coefficient for both regimes.
    pub friction: Option<FrictionCoefficients>,
    /// Normal spring stiffness in N/m.
    pub stiffness: Real,
    /// Compression damping in N s/m.
    pub damping: Real,
    /// Tangential viscous gain in N s/m, capped by Coulomb friction.
    pub friction_stiffness: Real,
    /// Tangential speed regularization in m/s.
    pub friction_smoothing: Real,
    pub friction_combine: CoefficientCombineRule,
}

impl Default for ContactMaterial {
    fn default() -> Self {
        Self {
            friction: None,
            stiffness: 1.0e5,
            damping: 1.0e3,
            friction_stiffness: 1.0e3,
            friction_smoothing: 0.01,
            friction_combine: CoefficientCombineRule::Max,
        }
    }
}

impl ContactMaterial {
    pub fn validate(self) -> Result<()> {
        if let Some(friction) = self.friction {
            friction.validate()?;
        }
        if [
            self.stiffness,
            self.damping,
            self.friction_stiffness,
            self.friction_smoothing,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Error::Build(
                "contact material requires finite nonnegative coefficients".into(),
            ));
        }
        Ok(())
    }

    pub fn averaged(self, other: Self) -> Self {
        Self {
            friction: self
                .friction
                .zip(other.friction)
                .map(|(a, b)| FrictionCoefficients {
                    mu_static: self.friction_combine.combine(
                        other.friction_combine,
                        a.mu_static,
                        b.mu_static,
                    ),
                    mu_dynamic: self.friction_combine.combine(
                        other.friction_combine,
                        a.mu_dynamic,
                        b.mu_dynamic,
                    ),
                }),
            stiffness: 0.5 * self.stiffness + 0.5 * other.stiffness,
            damping: 0.5 * self.damping + 0.5 * other.damping,
            friction_stiffness: 0.5 * self.friction_stiffness + 0.5 * other.friction_stiffness,
            friction_smoothing: 0.5 * self.friction_smoothing + 0.5 * other.friction_smoothing,
            friction_combine: self.friction_combine.max(other.friction_combine),
        }
    }
}
