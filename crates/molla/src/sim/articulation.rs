//! Forward kinematics for articulated systems.
//!
//! [`eval_fk`] walks the joint list in declaration order and writes
//! each child body's world transform to `state.body_q`. Single-chain
//! topologies are required at stage 3.0 — Featherstone proper handles
//! multi-branch trees and adds dynamics in stage 3.1.

use crate::core::Result;
use crate::math::{Quat, Real, Transform, Vec3};

use super::{JointType, Model, State};

/// Compute world poses for every body from `state.joint_q`.
///
/// For each joint `i`, given parent's world transform `X_w_p` (or
/// the world identity if `parent == -1`), the child's world transform
/// is:
///
/// ```text
///   X_w_c = X_w_p · X_p · J(q) · X_c⁻¹
/// ```
///
/// where `J(q)` is the joint's local transform map (rotation about
/// `axis` for revolute, translation along `axis` for prismatic,
/// identity for fixed, full 7-DoF for free).
///
/// **Stage 3.0 limitation**: assumes joints are listed in
/// topological order (parents before children). Multi-branch
/// hierarchies still work, only the ordering matters.
pub fn eval_fk(model: &Model, state: &mut State) -> Result<()> {
    if model.joint_count == 0 {
        return Ok(());
    }

    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_x_p = model.joint_x_p.host()?;
    let joint_x_c = model.joint_x_c.host()?;
    let joint_axis = model.joint_axis.host()?;
    let coord_offset = model.joint_coord_offset.host()?;
    let q = state.joint_q.host()?.to_vec();

    let body_q = state.body_q.host_mut()?;
    for j in 0..model.joint_count {
        let parent = joint_parent[j];
        let child = joint_child[j] as usize;
        let parent_world = if parent < 0 {
            Transform::IDENTITY
        } else {
            body_q[parent as usize]
        };

        let coord_start = coord_offset[j] as usize;
        let coord_end = coord_offset[j + 1] as usize;
        let qj = &q[coord_start..coord_end];

        let jt = joint_type_from_u32(joint_type[j]);
        let local = joint_local_transform(jt, qj, joint_axis[j]);

        let x_p = joint_x_p[j];
        let x_c_inv = joint_x_c[j].inverse();
        let world = parent_world.compose(&x_p).compose(&local).compose(&x_c_inv);
        body_q[child] = world;
    }
    Ok(())
}

fn joint_type_from_u32(v: u32) -> JointType {
    match v {
        0 => JointType::Prismatic,
        1 => JointType::Revolute,
        2 => JointType::Ball,
        3 => JointType::Fixed,
        4 => JointType::Free,
        5 => JointType::Distance,
        6 => JointType::D6,
        _ => JointType::Fixed, // safe fallback
    }
}

/// Joint-local transform `J(q)`. Reads exactly `coord_count(jt)`
/// entries from `qj`.
fn joint_local_transform(jt: JointType, qj: &[Real], axis: Vec3) -> Transform {
    match jt {
        JointType::Revolute => {
            let q = qj.first().copied().unwrap_or(0.0);
            Transform::from_rotation(Quat::from_axis_angle(axis, q))
        }
        JointType::Prismatic => {
            let q = qj.first().copied().unwrap_or(0.0);
            Transform::from_translation(axis * q)
        }
        JointType::Fixed => Transform::IDENTITY,
        JointType::Ball => {
            // 4 coords = quaternion (qx, qy, qz, qw). Pure rotation.
            let r = Quat::from_xyzw(qj[0], qj[1], qj[2], qj[3]).normalize();
            Transform::from_rotation(r)
        }
        JointType::Free | JointType::Distance | JointType::D6 => {
            // 7 coords: [tx, ty, tz, qx, qy, qz, qw].
            let t = Vec3::new(qj[0], qj[1], qj[2]);
            let r = Quat::from_xyzw(qj[3], qj[4], qj[5], qj[6]).normalize();
            Transform::new(t, r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{BodyParams, JointType as Jt, ModelBuilder};
    use crate::core::BodyId;

    /// Pendulum: a single-link pendulum hanging from world via a
    /// revolute joint about z. At joint angle = 0, the link sits
    /// along -Y. At angle π/2, it should be along +X.
    #[test]
    fn revolute_fk_rotates_pendulum() {
        let mut b = ModelBuilder::new();
        // Link: 1 m long, COM at midpoint along -Y in body frame.
        let link = b.add_body(BodyParams::new());
        // Joint at world origin; child frame attaches at the link's
        // top end. To make the link hang downward at q=0, place the
        // child frame's anchor point at +Y of the link's body frame.
        b.add_joint_revolute(
            BodyId::NONE,
            link,
            Transform::IDENTITY,
            Transform::from_translation(crate::math::Vec3::new(0.0, 0.5, 0.0)),
            crate::math::Vec3::Z,
        )
        .unwrap();

        let model = b.build().unwrap();
        let mut state = model.state().unwrap();

        // q = 0 → identity rotation. body_q = X_p · I · X_c⁻¹ =
        //   I · I · trans(0,-0.5,0) = link's body origin at (0,-0.5,0).
        eval_fk(&model, &mut state).unwrap();
        let q0 = state.body_q.host().unwrap()[link.index()];
        assert!((q0.position - crate::math::Vec3::new(0.0, -0.5, 0.0)).length() < 1e-12);

        // q = π/2 → rotate pendulum 90° about +Z (right-handed).
        // Point (0,-0.5,0) rotates to (+0.5, 0, 0) — fingers curl
        // from +X toward +Y, so -Y goes toward +X.
        state.joint_q.host_mut().unwrap()[0] = std::f64::consts::FRAC_PI_2;
        eval_fk(&model, &mut state).unwrap();
        let q90 = state.body_q.host().unwrap()[link.index()];
        assert!(
            (q90.position - crate::math::Vec3::new(0.5, 0.0, 0.0)).length() < 1e-9,
            "pendulum at π/2 should be at (0.5, 0, 0); got {:?}",
            q90.position
        );
    }

    #[test]
    fn prismatic_fk_translates_along_axis() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new());
        b.add_joint_prismatic(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            crate::math::Vec3::Y,
        )
        .unwrap();
        let model = b.build().unwrap();
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap()[0] = 2.5;
        eval_fk(&model, &mut state).unwrap();
        let p = state.body_q.host().unwrap()[body.index()].position;
        assert!((p - crate::math::Vec3::new(0.0, 2.5, 0.0)).length() < 1e-12);
    }

    /// Ball joint: setting joint_q to a non-identity quaternion
    /// rotates the child accordingly via FK.
    #[test]
    fn ball_joint_fk_rotates_via_quat() {
        use crate::math::Quat;
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new());
        b.add_joint_ball(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let model = b.build().unwrap();
        // Coord layout: 4 quaternion components (x, y, z, w).
        assert_eq!(model.joint_coord_total, 4);
        assert_eq!(model.joint_dof_total, 3);
        let mut state = model.state().unwrap();
        // Set quaternion = rotation of 90° about +Z.
        let target = Quat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let q = state.joint_q.host_mut().unwrap();
        q[0] = target.x;
        q[1] = target.y;
        q[2] = target.z;
        q[3] = target.w;
        eval_fk(&model, &mut state).unwrap();
        let pose = state.body_q.host().unwrap()[body.index()];
        // The world rotation should match `target`. Compare via dot
        // (handle the q == -q ambiguity).
        let dot = pose.rotation.x * target.x
            + pose.rotation.y * target.y
            + pose.rotation.z * target.z
            + pose.rotation.w * target.w;
        assert!(dot.abs() > 1.0 - 1e-9, "ball fk rotation off, dot = {dot}");
    }

    /// D6 joint: 7-coord pose (3 translation + 4 quat) maps directly
    /// through FK. With identity transforms and only translation set,
    /// the child sits at that translation.
    #[test]
    fn d6_joint_fk_translates() {
        use crate::math::Vec3;
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new());
        b.add_joint_d6(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let model = b.build().unwrap();
        assert_eq!(model.joint_coord_total, 7);
        assert_eq!(model.joint_dof_total, 6);
        let mut state = model.state().unwrap();
        let q = state.joint_q.host_mut().unwrap();
        q[0] = 1.0;
        q[1] = 2.0;
        q[2] = 3.0;
        q[3] = 0.0; q[4] = 0.0; q[5] = 0.0; q[6] = 1.0; // identity quat
        eval_fk(&model, &mut state).unwrap();
        let p = state.body_q.host().unwrap()[body.index()].position;
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
    }

    #[test]
    fn fixed_joint_locks_pose() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new());
        b.add_joint_fixed(
            BodyId::NONE,
            body,
            Transform::from_translation(crate::math::Vec3::new(1.0, 2.0, 3.0)),
            Transform::IDENTITY,
        )
        .unwrap();
        let model = b.build().unwrap();
        let mut state = model.state().unwrap();
        eval_fk(&model, &mut state).unwrap();
        let p = state.body_q.host().unwrap()[body.index()].position;
        assert!((p - crate::math::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-12);

        // Joint coords should be empty for fixed.
        assert_eq!(state.joint_q.host().unwrap().len(), 0);
        assert_eq!(model.joint_dof_total, 0);

        let _ = Jt::Fixed; // ensure the import is used
    }

    #[test]
    fn double_pendulum_chain_fk() {
        // Two-link chain: link0 ⟂ world via z-revolute; link1 ⟂ link0
        // via z-revolute at link0's bottom. Both at q=0 hang along -Y.
        let mut b = ModelBuilder::new();
        let l0 = b.add_body(BodyParams::new());
        let l1 = b.add_body(BodyParams::new());
        // joint 0: world → l0, child anchor at +Y so l0 hangs down.
        b.add_joint_revolute(
            BodyId::NONE,
            l0,
            Transform::IDENTITY,
            Transform::from_translation(crate::math::Vec3::new(0.0, 0.5, 0.0)),
            crate::math::Vec3::Z,
        )
        .unwrap();
        // joint 1: l0 → l1. Parent anchor at l0's bottom (-Y, 0.5),
        // child anchor at l1's top (+Y, 0.5).
        b.add_joint_revolute(
            l0,
            l1,
            Transform::from_translation(crate::math::Vec3::new(0.0, -0.5, 0.0)),
            Transform::from_translation(crate::math::Vec3::new(0.0, 0.5, 0.0)),
            crate::math::Vec3::Z,
        )
        .unwrap();

        let model = b.build().unwrap();
        let mut state = model.state().unwrap();
        eval_fk(&model, &mut state).unwrap();
        let l0_p = state.body_q.host().unwrap()[l0.index()].position;
        let l1_p = state.body_q.host().unwrap()[l1.index()].position;
        // l0 origin at (0, -0.5, 0); l1 origin at (0, -1.5, 0).
        assert!((l0_p - crate::math::Vec3::new(0.0, -0.5, 0.0)).length() < 1e-12);
        assert!((l1_p - crate::math::Vec3::new(0.0, -1.5, 0.0)).length() < 1e-12);
    }
}
