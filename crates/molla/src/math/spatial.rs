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

/// 6×6 spatial inertia / articulated-inertia matrix in row-major
/// layout, matching Warp's `wp.spatial_matrix`. Featherstone ABA's
/// articulated inertia outgrows the simple `(mass, com, inertia)`
/// representation as it accumulates child contributions through the
/// backward pass, so we carry a full 6×6.
///
/// Convention: `[linear; angular]` ordering for the spatial vector
/// it operates on (matches molla's `SpatialVector` and Warp's
/// `spatial_top` = linear).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialMatrix6 {
    /// Row-major 6×6.
    pub m: [[Real; 6]; 6],
}

impl SpatialMatrix6 {
    /// Zero matrix.
    pub const ZERO: Self = Self { m: [[0.0; 6]; 6] };

    /// 6×6 identity.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = [[0.0; 6]; 6];
        for i in 0..6 {
            m[i][i] = 1.0;
        }
        Self { m }
    }

    /// Scale every entry.
    #[must_use]
    pub fn scale(self, s: Real) -> Self {
        let mut out = self;
        for i in 0..6 {
            for j in 0..6 {
                out.m[i][j] *= s;
            }
        }
        out
    }

    /// Componentwise add.
    #[must_use]
    pub fn add(&self, o: &Self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..6 {
            for j in 0..6 {
                out.m[i][j] = self.m[i][j] + o.m[i][j];
            }
        }
        out
    }

    /// Componentwise subtract.
    #[must_use]
    pub fn sub(&self, o: &Self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..6 {
            for j in 0..6 {
                out.m[i][j] = self.m[i][j] - o.m[i][j];
            }
        }
        out
    }

    /// `M · v` — multiply by a spatial vector.
    /// Layout: `v = [linear; angular]` → `[v0..v2, v3..v5]`.
    #[must_use]
    pub fn mul_sv(&self, v: SpatialVector) -> SpatialVector {
        let x = [
            v.linear.x,
            v.linear.y,
            v.linear.z,
            v.angular.x,
            v.angular.y,
            v.angular.z,
        ];
        let mut out = [0.0_f64; 6];
        for i in 0..6 {
            let mut s = 0.0_f64;
            for j in 0..6 {
                s += self.m[i][j] * x[j];
            }
            out[i] = s;
        }
        SpatialVector::new(
            Vec3::new(out[0], out[1], out[2]),
            Vec3::new(out[3], out[4], out[5]),
        )
    }

    /// Outer product `u · v^T` — for rank-1 update used in ABA's
    /// `I^A := I^A - (I^A · S) · (I^A · S)^T / (S^T · I^A · S)`.
    #[must_use]
    pub fn outer(u: SpatialVector, v: SpatialVector) -> Self {
        let ua = [u.linear.x, u.linear.y, u.linear.z, u.angular.x, u.angular.y, u.angular.z];
        let va = [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z];
        let mut m = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] = ua[i] * va[j];
            }
        }
        Self { m }
    }

    /// Build the spatial inertia of a rigid body about a frame
    /// whose origin coincides with the body's reference point and
    /// whose axes are aligned with the world. `com` is the COM
    /// offset from the reference point in world; `inertia_world`
    /// is the 3×3 inertia tensor at COM, expressed in world axes.
    ///
    /// ```text
    ///        [ m·I_3                     m·skew(c)         ]
    /// I_s =  [                                               ]
    ///        [ m·skew(c)^T   I_C - m·skew(c)·skew(c)         ]
    /// ```
    /// matching Featherstone (2008) eq. (2.63) for spatial inertia
    /// about an arbitrary point with `[v; w]` ordering.
    #[must_use]
    pub fn from_body(mass: Real, com: Vec3, inertia_world: Mat3) -> Self {
        // Build skew(c).
        let s = skew3(com);
        // Top-left 3×3: m·I_3
        // Top-right 3×3: m·skew(c)
        // Bottom-left 3×3: -m·skew(c) (skew is antisymmetric → skew(c)^T = -skew(c))
        // Bottom-right 3×3: I_C - m·skew(c)·skew(c)
        let s2 = mat3_mul(&s, &s);
        let mut m_out = [[0.0; 6]; 6];
        // Top-left: m·I3
        for i in 0..3 {
            m_out[i][i] = mass;
        }
        // Top-right: m · skew(c)
        for i in 0..3 {
            for j in 0..3 {
                m_out[i][3 + j] = mass * s[i][j];
            }
        }
        // Bottom-left: -m · skew(c)  (because skew is antisymmetric, transpose = negate)
        for i in 0..3 {
            for j in 0..3 {
                m_out[3 + i][j] = -mass * s[i][j];
            }
        }
        // Bottom-right: I_C - m·skew(c)·skew(c)
        let i_w = mat3_to_arr(&inertia_world);
        for i in 0..3 {
            for j in 0..3 {
                m_out[3 + i][3 + j] = i_w[i][j] - mass * s2[i][j];
            }
        }
        Self { m: m_out }
    }
}

/// Spatial cross-product `a × b` for motion vectors. Per Newton's
/// `spatial_cross` (warp/_src formulation):
/// ```text
///   given a = [v_a; w_a],  b = [v_b; w_b]
///   linear  = w_a × v_b + v_a × w_b
///   angular = w_a × w_b
/// ```
#[must_use]
pub fn spatial_cross(a: SpatialVector, b: SpatialVector) -> SpatialVector {
    let w_a = a.angular;
    let v_a = a.linear;
    let w_b = b.angular;
    let v_b = b.linear;
    SpatialVector::new(w_a.cross(v_b) + v_a.cross(w_b), w_a.cross(w_b))
}

/// Dual spatial cross-product `a ×* b` for force vectors (Coriolis
/// term in ABA's bias). Per Newton's `spatial_cross_dual`:
/// ```text
///   linear  = w_a × v_b
///   angular = w_a × w_b + v_a × v_b
/// ```
#[must_use]
pub fn spatial_cross_dual(a: SpatialVector, b: SpatialVector) -> SpatialVector {
    let w_a = a.angular;
    let v_a = a.linear;
    let w_b = b.angular;
    let v_b = b.linear;
    SpatialVector::new(w_a.cross(v_b), w_a.cross(w_b) + v_a.cross(v_b))
}

/// Skew-symmetric matrix from a 3-vector — `skew(v) · w = v × w`.
fn skew3(v: Vec3) -> [[Real; 3]; 3] {
    [
        [0.0, -v.z, v.y],
        [v.z, 0.0, -v.x],
        [-v.y, v.x, 0.0],
    ]
}

fn mat3_mul(a: &[[Real; 3]; 3], b: &[[Real; 3]; 3]) -> [[Real; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_to_arr(m: &Mat3) -> [[Real; 3]; 3] {
    [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z],
    ]
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
