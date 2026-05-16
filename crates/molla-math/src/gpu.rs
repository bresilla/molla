//! `f32` GPU-side mirrors of the host math types.
//!
//! These are `#[repr(C)]` POD structs with explicit padding so the layout
//! matches WGSL's `alignment-of-vec3 = 16` rule. They derive `bytemuck::Pod`
//! / `Zeroable` so they can be uploaded with `wgpu::Queue::write_buffer`.

use super::{SpatialVector, Transform};

/// `f32` mirror of [`Transform`]. 32 bytes; vec3 padded to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransform {
    /// `(x, y, z, _pad)`
    pub position: [f32; 4],
    /// `(x, y, z, w)`
    pub rotation: [f32; 4],
}

impl From<Transform> for GpuTransform {
    fn from(t: Transform) -> Self {
        let p = t.position.as_vec3();
        let r = t.rotation.as_quat();
        Self {
            position: [p.x, p.y, p.z, 0.0],
            rotation: [r.x, r.y, r.z, r.w],
        }
    }
}

impl From<GpuTransform> for Transform {
    fn from(g: GpuTransform) -> Self {
        Self {
            position: glam::Vec3::new(g.position[0], g.position[1], g.position[2]).as_dvec3(),
            rotation: glam::Quat::from_xyzw(
                g.rotation[0],
                g.rotation[1],
                g.rotation[2],
                g.rotation[3],
            )
            .as_dquat(),
        }
    }
}

/// `f32` mirror of [`SpatialVector`]. 32 bytes; both vec3s padded to 16.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpatialVector {
    /// `(x, y, z, _pad)`
    pub linear: [f32; 4],
    /// `(x, y, z, _pad)`
    pub angular: [f32; 4],
}

impl From<SpatialVector> for GpuSpatialVector {
    fn from(s: SpatialVector) -> Self {
        let l = s.linear.as_vec3();
        let a = s.angular.as_vec3();
        Self {
            linear: [l.x, l.y, l.z, 0.0],
            angular: [a.x, a.y, a.z, 0.0],
        }
    }
}

impl From<GpuSpatialVector> for SpatialVector {
    fn from(g: GpuSpatialVector) -> Self {
        Self {
            linear: glam::Vec3::new(g.linear[0], g.linear[1], g.linear[2]).as_dvec3(),
            angular: glam::Vec3::new(g.angular[0], g.angular[1], g.angular[2]).as_dvec3(),
        }
    }
}

/// `f32` mirror of a 3×3 matrix. WGSL `mat3x3<f32>` has column alignment of
/// 16, so we store it as three `vec4`-padded columns.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMat3 {
    /// Column 0: `(x, y, z, _pad)`
    pub col0: [f32; 4],
    /// Column 1: `(x, y, z, _pad)`
    pub col1: [f32; 4],
    /// Column 2: `(x, y, z, _pad)`
    pub col2: [f32; 4],
}

impl From<super::Mat3> for GpuMat3 {
    fn from(m: super::Mat3) -> Self {
        let m32 = m.as_mat3();
        Self {
            col0: [m32.x_axis.x, m32.x_axis.y, m32.x_axis.z, 0.0],
            col1: [m32.y_axis.x, m32.y_axis.y, m32.y_axis.z, 0.0],
            col2: [m32.z_axis.x, m32.z_axis.y, m32.z_axis.z, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Quat, Vec3};

    fn vec_close(a: Vec3, b: Vec3, eps: f64) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn transform_round_trip_loses_precision_only() {
        let t = Transform {
            position: Vec3::new(1.5, -2.25, 3.125),
            rotation: Quat::from_rotation_z(0.7),
        };
        let g: GpuTransform = t.into();
        let back: Transform = g.into();
        // f64 → f32 → f64 is lossy at the f32 epsilon scale.
        assert!(vec_close(back.position, t.position, 1e-6));
        assert!(back.rotation.abs_diff_eq(t.rotation, 1e-6));
    }

    #[test]
    fn spatial_vector_round_trip() {
        let s = SpatialVector::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));
        let g: GpuSpatialVector = s.into();
        let back: SpatialVector = g.into();
        assert!(vec_close(back.linear, s.linear, 1e-6));
        assert!(vec_close(back.angular, s.angular, 1e-6));
    }

    #[test]
    fn gpu_struct_sizes_match_wgsl_expectations() {
        // GpuTransform: vec4 + vec4 = 32 bytes
        assert_eq!(core::mem::size_of::<GpuTransform>(), 32);
        // GpuSpatialVector: vec4 + vec4 = 32 bytes
        assert_eq!(core::mem::size_of::<GpuSpatialVector>(), 32);
        // GpuMat3: 3 × vec4 = 48 bytes (matches WGSL mat3x3<f32>)
        assert_eq!(core::mem::size_of::<GpuMat3>(), 48);
    }
}
