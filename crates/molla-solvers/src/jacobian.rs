//! Body-Jacobian — `J(q)` mapping joint velocities to a target
//! body-point's spatial velocity.
//!
//! For a serial chain articulation:
//!
//! ```text
//! v_target = J(q) · q̇
//! ```
//!
//! where `J` is a `6 × N` matrix with one column per joint DOF, and
//! `v_target = [v_linear; ω]` is the spatial velocity at
//! `point_local` (in the target body's frame), expressed in world
//! axes. Columns for joints that are NOT ancestors of the target
//! body are zero (they don't move it).
//!
//! Used by IK (gradient of position/rotation residuals) and
//! operational-space control. Stays CPU until a high-throughput
//! consumer needs the GPU port.

use crate::core::{Error, Result};
use crate::featherstone::{MAX_JOINT_DOFS, build_motion_subspace};
use crate::math::{Real, SpatialVector, Vec3};
use crate::sim::{BodyFlags, JointType, Model, State};

/// `6 × N` Jacobian for a body point. Returns a flat row-major
/// `6 · N` Vec where row `r ∈ [0,5]` and column `k ∈ [0,N)` give
/// the spatial-velocity contribution from joint DOF `k`. Layout is
/// `[v_linear (3), ω (3)]` per column, just like `SpatialVector`.
///
/// `target_body` is a global body index; `point_local` is the point's
/// position in the body's local frame. `articulation` selects which
/// articulation owns the joints — the result is sized to that
/// articulation's DOF total.
// The 6×dof J fill below keeps the explicit `row * dof_total + col`
// row-major idiom for rows 0 and 1 so all six rows read as coordinates.
#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn jacobian(
    model: &Model,
    state: &State,
    articulation: usize,
    target_body: usize,
    point_local: Vec3,
) -> Result<Vec<Real>> {
    if model.joint_count == 0 || target_body >= model.body_count {
        return Ok(Vec::new());
    }
    let (j_begin, j_end) = articulation_range(model, articulation)?;

    let body_q = state.body_q.host()?;
    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;

    let dof_total = (joint_dof_offset[j_end] - joint_dof_offset[j_begin]) as usize;
    if dof_total == 0 {
        return Ok(Vec::new());
    }

    // Target point in WORLD frame.
    let cw = body_q[target_body];
    let target_world = cw.position + cw.rotation * point_local;

    // For each joint in the articulation, decide if it's an ancestor
    // of `target_body`. Walk up from the target via joint_parent /
    // joint_child links and mark each ancestor joint.
    //
    // Ancestor relationship: joint k is an ancestor of body B if
    // following parent chain from B (B → parent_of_B → ... → root)
    // hits joint k's child body.
    let mut is_ancestor = vec![false; j_end - j_begin];
    let mut walker: i32 = target_body as i32;
    while walker >= 0 {
        let mut found = false;
        for kk in 0..(j_end - j_begin) {
            let jj = j_begin + kk;
            if joint_child[jj] == walker {
                is_ancestor[kk] = true;
                walker = joint_parent[jj];
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
    }

    // Build columns.
    let mut j_cols = vec![0.0; 6 * dof_total];
    for k in 0..(j_end - j_begin) {
        if !is_ancestor[k] {
            continue;
        }
        let j = j_begin + k;
        if model.joint_enabled.host()?[j] == 0
            || model.body_flags.host()?[joint_child[j] as usize] & BodyFlags::IMMOVABLE != 0
        {
            continue;
        }
        let (mut xpw, direction) = model.joint_motion_frame(body_q, state.joint_q.host()?, j)?;
        let jt = decode_joint_type(joint_type[j])?;
        xpw.position -= target_world;
        let dof_local = (joint_dof_offset[j] - joint_dof_offset[j_begin]) as usize;

        // Motion subspace about the target point; locked columns remain zero.
        let mut s_cols = [SpatialVector::ZERO; MAX_JOINT_DOFS];
        let mut dof_active = [false; MAX_JOINT_DOFS];
        let width = build_motion_subspace(
            jt,
            xpw,
            joint_axis[j],
            joint_dof_offset[j] as usize,
            limit_lower,
            limit_upper,
            &mut s_cols,
            &mut dof_active,
        );
        for a in 0..width {
            let s = s_cols[a].scale(direction);
            let lin = s.linear;
            let col_idx = dof_local + a;
            j_cols[0 * dof_total + col_idx] = lin.x;
            j_cols[1 * dof_total + col_idx] = lin.y;
            j_cols[2 * dof_total + col_idx] = lin.z;
            j_cols[3 * dof_total + col_idx] = s.angular.x;
            j_cols[4 * dof_total + col_idx] = s.angular.y;
            j_cols[5 * dof_total + col_idx] = s.angular.z;
        }
    }
    Ok(j_cols)
}

fn decode_joint_type(v: u32) -> Result<JointType> {
    match v {
        0 => Ok(JointType::Prismatic),
        1 => Ok(JointType::Revolute),
        2 => Ok(JointType::Ball),
        3 => Ok(JointType::Fixed),
        4 => Ok(JointType::Free),
        5 => Ok(JointType::Distance),
        6 => Ok(JointType::D6),
        7 => Ok(JointType::Cable),
        _ => Err(Error::Build(format!("jacobian: unknown joint_type {v}"))),
    }
}

fn articulation_range(model: &Model, articulation: usize) -> Result<(usize, usize)> {
    if model.articulation_count == 0 {
        return Ok((0, model.joint_count));
    }
    let starts = model.articulation_start.host()?;
    if articulation + 1 >= starts.len() {
        return Err(Error::Build(format!(
            "jacobian: articulation {articulation} out of range"
        )));
    }
    Ok((
        starts[articulation] as usize,
        starts[articulation + 1] as usize,
    ))
}
