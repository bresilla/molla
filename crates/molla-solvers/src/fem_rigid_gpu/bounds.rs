use super::*;
use molla_kernels::soft_rigid_bounds::{SoftRigidBoundsKernel, SoftRigidScalarBound};
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;

pub(super) fn build(
    device: &wgpu::Device,
    model: &Model,
) -> Result<(Option<SoftRigidBoundsKernel>, Vec<SoftRigidContactGpu>)> {
    let kinds = model.joint_type.host()?;
    let dofs = model.joint_dof_offset.host()?;
    let coords = model.joint_coord_offset.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let speed = model.joint_velocity_limit.host()?;
    let mut descriptors = Vec::new();
    for (joint, &kind) in kinds.iter().enumerate() {
        if !matches!(kind, 0 | 1) {
            continue;
        }
        let dof = dofs[joint] as usize;
        if (upper[dof] - lower[dof]).abs() < 1e-12 {
            continue;
        }
        if lower[dof].is_nan() || upper[dof].is_nan() || lower[dof] > upper[dof] || speed[dof] < 0.0
        {
            return Err(Error::Build("invalid GPU FEM scalar limits".into()));
        }
        let lo = lower[dof].max(-1e30) as f32;
        let hi = upper[dof].min(1e30) as f32;
        let cap = if speed[dof].is_nan() {
            1e30
        } else {
            speed[dof].min(1e30) as f32
        };
        if lo <= -1e30 && hi >= 1e30 && cap >= 1e30 {
            continue;
        }
        descriptors.push(SoftRigidScalarBound {
            indices: [dof as u32, coords[joint], 0, 0],
            limits: [lo, hi, cap, 0.0],
        });
    }
    if descriptors.is_empty() {
        return Ok((None, Vec::new()));
    }
    let (kernel, rows) = SoftRigidBoundsKernel::new(
        device,
        &descriptors,
        model.joint_dof_total as u32,
        model.joint_coord_total as u32,
    )?;
    Ok((Some(kernel), rows))
}
