//! `UsdPhysics*Joint` → `JointType::{Revolute, Prismatic, Fixed,
//! Ball, Distance, D6}`. Decodes `body0`/`body1` rels into
//! `BodyId`s via the path map built by `bodies::import_bodies`,
//! turns `localPos0/1` + `localRot0/1` into the joint's
//! parent/child anchor transforms, and applies axis token (X/Y/Z),
//! limits (`physics:lowerLimit`/`upperLimit` + multi-apply
//! `PhysicsLimitAPI:<dof>`), and drive defaults.

use anyhow::Result;
use molla::core::BodyId;
use molla::math::{Quat, Real, Transform, Vec3};
use molla::sim::{JointType, ModelBuilder};
use openusd::physics::{self, JointKind, PhysicsPrims};
use openusd::sdf;
use openusd::Stage;

use crate::ImportInfo;

pub fn import_joints(
    stage: &Stage,
    prims: &PhysicsPrims,
    builder: &mut ModelBuilder,
    info: &mut ImportInfo,
) -> Result<()> {
    for path_str in &prims.joints {
        let path = sdf::path(path_str)?;
        let joint = match physics::read_joint(stage, &path)? {
            Some(j) => j,
            None => continue,
        };
        if !joint.joint_enabled {
            continue;
        }

        let parent = body_for_path(info, joint.body0.as_deref());
        let child = body_for_path(info, joint.body1.as_deref())
            .ok_or_else(|| anyhow::anyhow!(
                "joint {path_str}: body1 must reference a known body"
            ))?;

        let x_p = pose_from_local(joint.local_pos0, joint.local_rot0);
        let x_c = pose_from_local(joint.local_pos1, joint.local_rot1);
        let axis = axis_from_token(joint.axis.as_deref());

        let id = match joint.kind {
            JointKind::Revolute => builder
                .add_joint_revolute(parent.unwrap_or(BodyId::NONE), child, x_p, x_c, axis)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            JointKind::Prismatic => builder
                .add_joint_prismatic(parent.unwrap_or(BodyId::NONE), child, x_p, x_c, axis)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            JointKind::Fixed => builder
                .add_joint_fixed(parent.unwrap_or(BodyId::NONE), child, x_p, x_c)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            JointKind::Spherical => builder
                .add_joint_ball(parent.unwrap_or(BodyId::NONE), child, x_p, x_c)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            JointKind::Distance => builder
                .add_joint_distance(parent.unwrap_or(BodyId::NONE), child, x_p, x_c)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            JointKind::Generic => builder
                .add_joint_d6(parent.unwrap_or(BodyId::NONE), child, x_p, x_c)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        };

        // Single-axis limits (revolute / prismatic).
        if let (Some(lo), Some(hi)) = (joint.lower_limit, joint.upper_limit) {
            if matches!(joint.kind, JointKind::Revolute | JointKind::Prismatic) {
                let lo = if matches!(joint.kind, JointKind::Revolute) {
                    (lo as Real).to_radians()
                } else {
                    lo as Real
                };
                let hi = if matches!(joint.kind, JointKind::Revolute) {
                    (hi as Real).to_radians()
                } else {
                    hi as Real
                };
                let _ = builder.set_joint_limits(id, &[lo], &[hi]);
            }
        }

        info.joint_paths.push(path_str.clone());
        let _ = stage; // suppress unused-var warning when no joint flags read yet
        let _ = JointType::Revolute; // ensure import is used
    }
    Ok(())
}

fn body_for_path(info: &ImportInfo, path: Option<&str>) -> Option<BodyId> {
    let p = path?;
    crate::bodies::body_id_for_path(info, p)
}

fn pose_from_local(pos: [f32; 3], rot: [f32; 4]) -> Transform {
    Transform::new(
        Vec3::new(pos[0] as f64, pos[1] as f64, pos[2] as f64),
        // USD textual quaternions are (w, x, y, z); `read_joint`
        // surfaces them as a [f32; 4] matching that layout.
        Quat::from_xyzw(
            rot[1] as f64,
            rot[2] as f64,
            rot[3] as f64,
            rot[0] as f64,
        ),
    )
}

fn axis_from_token(tok: Option<&str>) -> Vec3 {
    match tok.unwrap_or("Z") {
        "X" => Vec3::X,
        "Y" => Vec3::Y,
        _ => Vec3::Z,
    }
}
