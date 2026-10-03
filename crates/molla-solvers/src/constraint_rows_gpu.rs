//! Device scalar rows for shared articulated joint/contact response.

/// Scalar equation in generalized velocities, with two body endpoints for connectivity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarConstraintRow {
    /// Body A, body B, kind (1 bilateral, 2 nonnegative, 3 bounded feedback), active (0 or 1).
    /// World endpoints use `u32::MAX`.
    pub endpoints: [u32; 4],
    /// Rate minus target in x; bounded rows use feedback diagonal/lower/upper impulse bounds in yzw.
    pub velocity: [f32; 4],
}

/// Dense scalar Jacobian rows use model-global generalized-velocity order.
/// Nonzero columns must belong to an endpoint's mobile articulation.
#[derive(Clone, Copy)]
pub struct ScalarConstraintBuffers<'a> {
    pub rows: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub capacity: u32,
}

/// ScalarConstraintRow layout with displacement residual in `velocity.x`, not a rate.
/// Jacobians use model-global generalized displacement order at the input pose.
/// Active position rows accept only bilateral/nonnegative domains, not feedback.
#[derive(Clone, Copy)]
pub struct ScalarPositionBuffers<'a> {
    pub rows: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub capacity: u32,
}
