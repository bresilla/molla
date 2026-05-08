//! Spatial vectors (twists, wrenches) and spatial inertia.
//!
//! See PLAN.md §7 and the glossary for the convention. Linear component
//! comes first, angular second — matches Newton's `wp.spatial_vector`.

use super::{Mat3, Real, Vec3};

/// Six-component spatial vector — pairs a linear and an angular component.
///
/// Used for both **twists** (linear + angular velocity) and **wrenches**
/// (force + torque), distinguished by context not by type.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SpatialVector {
    /// Linear part: velocity (m/s) or force (N).
    pub linear: Vec3,
    /// Angular part: angular velocity (rad/s) or torque (N·m).
    pub angular: Vec3,
}

impl SpatialVector {
    /// All-zero spatial vector.
    pub const ZERO: Self = Self {
        linear: Vec3::ZERO,
        angular: Vec3::ZERO,
    };

    /// Construct from explicit linear and angular components.
    #[must_use]
    pub const fn new(linear: Vec3, angular: Vec3) -> Self {
        Self { linear, angular }
    }

    /// A pure-linear (translation-only) twist or pure-force wrench.
    #[must_use]
    pub const fn linear(v: Vec3) -> Self {
        Self {
            linear: v,
            angular: Vec3::ZERO,
        }
    }

    /// A pure-angular twist or pure-torque wrench.
    #[must_use]
    pub const fn angular(w: Vec3) -> Self {
        Self {
            linear: Vec3::ZERO,
            angular: w,
        }
    }

    /// Scale both components.
    #[must_use]
    pub fn scale(self, s: Real) -> Self {
        Self {
            linear: self.linear * s,
            angular: self.angular * s,
        }
    }
}

impl core::ops::Add for SpatialVector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular,
        }
    }
}

impl core::ops::Sub for SpatialVector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular,
        }
    }
}

impl core::ops::Mul<Real> for SpatialVector {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self {
        self.scale(rhs)
    }
}

impl core::ops::Neg for SpatialVector {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            linear: -self.linear,
            angular: -self.angular,
        }
    }
}

/// Mass + inertia tensor + centre of mass for a single rigid body.
///
/// The inertia is expressed at the COM, in body frame, as `kg·m²`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialInertia {
    /// Total mass (kg).
    pub mass: Real,
    /// Centre of mass in body frame (m).
    pub com: Vec3,
    /// 3×3 inertia tensor at COM, body frame (kg·m²).
    pub inertia: Mat3,
}

impl SpatialInertia {
    /// Zero-mass placeholder. Convenient for kinematic / static bodies.
    pub const ZERO: Self = Self {
        mass: 0.0,
        com: Vec3::ZERO,
        inertia: Mat3::ZERO,
    };

    /// A solid sphere of given mass and radius.
    #[must_use]
    pub fn sphere(mass: Real, radius: Real) -> Self {
        let i = 0.4 * mass * radius * radius;
        Self {
            mass,
            com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(Vec3::splat(i)),
        }
    }

    /// A solid box with full extents `(sx, sy, sz)`.
    #[must_use]
    pub fn cuboid(mass: Real, full_extents: Vec3) -> Self {
        let (sx, sy, sz) = (full_extents.x, full_extents.y, full_extents.z);
        let k = mass / 12.0;
        Self {
            mass,
            com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(Vec3::new(
                k * (sy * sy + sz * sz),
                k * (sx * sx + sz * sz),
                k * (sx * sx + sy * sy),
            )),
        }
    }

    /// `1.0 / mass`, or zero if mass is zero (sentinel for static / kinematic).
    #[must_use]
    pub fn inv_mass(&self) -> Real {
        if self.mass > 0.0 { 1.0 / self.mass } else { 0.0 }
    }

    /// Inverse inertia tensor; zero matrix if `mass` is zero.
    #[must_use]
    pub fn inv_inertia(&self) -> Mat3 {
        if self.mass > 0.0 {
            self.inertia.inverse()
        } else {
            Mat3::ZERO
        }
    }
}

impl Default for SpatialInertia {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_close(a: Vec3, b: Vec3, eps: Real) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn add_and_subtract_are_componentwise() {
        let a = SpatialVector::new(Vec3::X, Vec3::Y);
        let b = SpatialVector::new(Vec3::Y, Vec3::Z);
        let s = a + b;
        assert!(vec_close(s.linear, Vec3::new(1.0, 1.0, 0.0), 1e-12));
        assert!(vec_close(s.angular, Vec3::new(0.0, 1.0, 1.0), 1e-12));
        let d = a - b;
        assert!(vec_close(d.linear, Vec3::new(1.0, -1.0, 0.0), 1e-12));
        assert!(vec_close(d.angular, Vec3::new(0.0, 1.0, -1.0), 1e-12));
    }

    #[test]
    fn sphere_inertia_is_isotropic() {
        let si = SpatialInertia::sphere(2.0, 0.5);
        let i = si.inertia;
        assert!((i.x_axis.x - 0.2).abs() < 1e-12);
        assert!((i.y_axis.y - 0.2).abs() < 1e-12);
        assert!((i.z_axis.z - 0.2).abs() < 1e-12);
    }

    #[test]
    fn zero_mass_inv_is_zero() {
        let zero = SpatialInertia::ZERO;
        assert_eq!(zero.inv_mass(), 0.0);
        assert_eq!(zero.inv_inertia(), Mat3::ZERO);
    }
}
