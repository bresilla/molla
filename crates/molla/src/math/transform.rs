//! Rigid-body pose: position + unit quaternion.

use super::{Mat3, Quat, Real, Vec3};

/// A rigid transform — translation followed by rotation.
///
/// Matches Newton's `wp.transform`: `(position, rotation)` with the
/// rotation applied to the body-local frame to produce world-space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    /// Translation in metres.
    pub position: Vec3,
    /// Unit quaternion. Caller is responsible for normalisation when
    /// composing many rotations; integrators should renormalise each step.
    pub rotation: Quat,
}

impl Transform {
    /// The identity transform: zero translation, unit rotation.
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Construct from translation and rotation components.
    #[must_use]
    pub const fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// Pure translation.
    #[must_use]
    pub const fn from_translation(t: Vec3) -> Self {
        Self {
            position: t,
            rotation: Quat::IDENTITY,
        }
    }

    /// Pure rotation about the origin.
    #[must_use]
    pub const fn from_rotation(r: Quat) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: r,
        }
    }

    /// Compose two transforms: `self ∘ other` = first apply `other`, then `self`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            position: self.position + self.rotation * other.position,
            rotation: (self.rotation * other.rotation).normalize(),
        }
    }

    /// Inverse: undoes the transform.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.conjugate();
        Self {
            position: -(inv_rot * self.position),
            rotation: inv_rot,
        }
    }

    /// Transform a point from local space to world space.
    #[must_use]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.position + self.rotation * p
    }

    /// Transform a direction (no translation) from local to world.
    #[must_use]
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation * v
    }

    /// Inverse-transform a world-space point into the local frame.
    #[must_use]
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3 {
        self.rotation.conjugate() * (p - self.position)
    }

    /// Convert the rotation to a 3×3 matrix.
    #[must_use]
    pub fn rotation_matrix(&self) -> Mat3 {
        Mat3::from_quat(self.rotation)
    }

    /// Re-normalise the rotation in place; cheap finite-precision hygiene
    /// that integrators should call every step.
    pub fn renormalize(&mut self) {
        self.rotation = self.rotation.normalize();
    }

    /// Convert to the `f32` GPU mirror.
    #[must_use]
    pub fn to_gpu(&self) -> super::GpuTransform {
        super::GpuTransform::from(*self)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Linear interpolation of position and slerp of rotation.
#[must_use]
#[allow(dead_code, reason = "stage-2 entry point; kept here so the API is stable from day one")]
pub fn lerp(a: &Transform, b: &Transform, t: Real) -> Transform {
    Transform {
        position: a.position.lerp(b.position, t),
        rotation: a.rotation.slerp(b.rotation, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_close(a: Vec3, b: Vec3, eps: Real) -> bool {
        (a - b).length() < eps
    }

    fn quat_close(a: Quat, b: Quat, eps: Real) -> bool {
        a.abs_diff_eq(b, eps) || a.abs_diff_eq(-b, eps)
    }

    #[test]
    fn identity_compose_is_no_op() {
        let t = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(0.5));
        let composed = Transform::IDENTITY.compose(&t);
        assert!(vec_close(composed.position, t.position, 1e-12));
        assert!(quat_close(composed.rotation, t.rotation, 1e-12));
    }

    #[test]
    fn inverse_round_trips() {
        let t = Transform::new(Vec3::new(1.0, -2.0, 3.5), Quat::from_rotation_y(1.2));
        let id = t.compose(&t.inverse());
        assert!(vec_close(id.position, Vec3::ZERO, 1e-12));
        assert!(quat_close(id.rotation, Quat::IDENTITY, 1e-12));
    }

    #[test]
    fn transform_point_round_trips() {
        let t = Transform::new(Vec3::new(0.5, 0.0, -1.0), Quat::from_rotation_x(0.3));
        let p_local = Vec3::new(1.0, 2.0, 3.0);
        let p_world = t.transform_point(p_local);
        let back = t.inverse_transform_point(p_world);
        assert!(vec_close(back, p_local, 1e-12));
    }

    #[test]
    fn compose_associative() {
        let a = Transform::new(Vec3::X, Quat::from_rotation_z(0.1));
        let b = Transform::new(Vec3::Y, Quat::from_rotation_x(0.2));
        let c = Transform::new(Vec3::Z, Quat::from_rotation_y(0.3));
        let lhs = a.compose(&b).compose(&c);
        let rhs = a.compose(&b.compose(&c));
        assert!(vec_close(lhs.position, rhs.position, 1e-12));
        assert!(quat_close(lhs.rotation, rhs.rotation, 1e-12));
    }
}
