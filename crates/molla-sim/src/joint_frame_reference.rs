use molla_core::{Error, Result};
use molla_math::{Transform, Vec3};

/// Authored joint-frame and axis tails for compensated GPU forward kinematics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointFrameReference {
    pub parent_position: [f32; 4],
    pub parent_rotation: [f32; 4],
    pub child_position: [f32; 4],
    pub child_rotation: [f32; 4],
    pub axis: [f32; 4],
}

impl JointFrameReference {
    pub fn new(parent: Transform, child: Transform, axis: Vec3) -> Result<Self> {
        let tail = |values: [f64; 4]| -> Result<[f32; 4]> {
            let low = values.map(|v| (v - f64::from(v as f32)) as f32);
            if values.iter().any(|v| !v.is_finite()) || low.iter().any(|v| !v.is_finite()) {
                return Err(Error::Build(
                    "joint frame exceeds GPU precision range".into(),
                ));
            }
            Ok(low)
        };
        Ok(Self {
            parent_position: tail(parent.position.extend(0.).to_array())?,
            parent_rotation: tail(parent.rotation.to_array())?,
            child_position: tail(child.position.extend(0.).to_array())?,
            child_rotation: tail(child.rotation.to_array())?,
            axis: tail(axis.extend(0.).to_array())?,
        })
    }
}

impl molla_storage::BufferElement for JointFrameReference {
    type Gpu = Self;

    fn to_gpu(&self) -> Self {
        *self
    }
    fn from_gpu(value: &Self) -> Self {
        *value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Quat;

    #[test]
    fn authored_frame_and_axis_tails_reconstruct_and_reject_overflow() {
        let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), Quat::from_rotation_y(0.3));
        let child = Transform::new(Vec3::new(-0.3, 0.5, -0.9), Quat::from_rotation_x(-0.4));
        let axis = Vec3::new(0.2, 0.3, 0.4).normalize();
        let reference = JointFrameReference::new(parent, child, axis).unwrap();
        assert_eq!(std::mem::size_of::<JointFrameReference>(), 80);
        for (values, low) in [
            (
                parent.position.extend(0.).to_array(),
                reference.parent_position,
            ),
            (parent.rotation.to_array(), reference.parent_rotation),
            (
                child.position.extend(0.).to_array(),
                reference.child_position,
            ),
            (child.rotation.to_array(), reference.child_rotation),
            (axis.extend(0.).to_array(), reference.axis),
        ] {
            for (value, tail) in values.into_iter().zip(low) {
                let restored = f64::from(value as f32) + f64::from(tail);
                assert!((restored - value).abs() <= 8. * f64::EPSILON * value.abs().max(1.));
            }
        }
        assert!(
            JointFrameReference::new(Transform::from_translation(Vec3::X * f64::MAX), child, axis,)
                .is_err()
        );
    }
}
