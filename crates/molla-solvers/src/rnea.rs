//! Recursive Newton-Euler — joint-space inverse dynamics.
//!
//! Given `(q, q̇, q̈)`, returns the joint forces `τ` that would
//! produce the requested accelerations. Two passes per articulation:
//!
//!   1. Forward (root → leaf): compute each body's spatial velocity,
//!      acceleration, and net force.
//!      v_i = v_p + S_i·q̇_i
//!      a_i = a_p + S_i·q̈_i + v_i × (S_i·q̇_i)
//!      f_i = I_i·a_i + v_i ×* (I_i·v_i)
//!   2. Backward (leaf → root): project body forces onto joint motion
//!      subspaces and accumulate.
//!      τ_i = S_i^T · f_i
//!      f_p += f_i  (transferred via parent-child cross terms;
//!      handled implicitly by world-origin spatial
//!      vectors).
//!
//! Gravity is folded into the base acceleration `a_root = -g`, so any
//! body subject to gravity sees an "effective" acceleration of `-g`
//! relative to free-fall. This matches Featherstone (2008) §5 and
//! Newton's `eval_rigid_id` kernel. World-origin spatial vector
//! convention throughout — same as `crba::mass_matrix` so the two
//! algorithms compose trivially.
//!
//! Single-articulation, single-DoF joints (Revolute / Prismatic /
//! Fixed). Multi-DoF joints (Ball / Free / Distance / D6) are not
//! yet supported by this routine — callers receive `Err(Build)` from
//! the upstream DOF guard so they can fall back to Featherstone /
//! XPBD instead.

use crate::core::{Error, Result};
use crate::featherstone::{MAX_JOINT_DOFS, build_motion_subspace};
use crate::math::{
    Mat3, Real, SpatialMatrix6, SpatialVector, Vec3, spatial_cross, spatial_cross_dual,
};
use crate::sim::{JointType, Model, State};

/// Inverse dynamics: returns `τ` (length = articulation DOFs) for the
/// requested `(q, q̇, q̈)`.
///
/// `state.joint_q` and `state.joint_qd` provide `q` and `q̇`. Pass `q̈`
/// (length = articulation DOFs) explicitly. State.body_q must be
/// up-to-date for the current `q` (call `eval_fk` first if unsure).
pub fn inverse_dynamics(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
) -> Result<Vec<Real>> {
    if model.joint_count == 0 {
        return Ok(Vec::new());
    }

    let (j_begin, j_end) = articulation_range(model, articulation)?;
    let n_joints = j_end - j_begin;
    if n_joints == 0 {
        return Ok(Vec::new());
    }

    let body_q = state.body_q.host()?;
    let joint_q = state.joint_q.host()?;
    let joint_qd = state.joint_qd.host()?;
    let _ = joint_q;
    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_x_p = model.joint_x_p.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let body_world = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;

    let dof_total = (joint_dof_offset[j_end] - joint_dof_offset[j_begin]) as usize;
    if qdd.len() != dof_total {
        return Err(Error::Build(format!(
            "RNEA: qdd len {} != articulation DOFs {}",
            qdd.len(),
            dof_total
        )));
    }

    // ── Per-joint local indexing (within articulation) ──────────
    let mut parent_joint_local: Vec<Option<usize>> = vec![None; n_joints];
    for k in 0..n_joints {
        let j = j_begin + k;
        let parent_body = joint_parent[j];
        if parent_body < 0 {
            continue;
        }
        for kk in 0..k {
            if joint_child[j_begin + kk] == parent_body {
                parent_joint_local[k] = Some(kk);
                break;
            }
        }
    }

    // ── Pass 1: forward sweep ─────────────────────────────────────
    // Per-DOF motion subspace shared with Featherstone / CRBA — all
    // joint types expand to their columns (Ball → 3, Free/Distance/D6
    // → 6 with lock filtering, Cable → 2, Revolute/Prismatic → 1).
    let mut s_cols_per_joint: Vec<[SpatialVector; MAX_JOINT_DOFS]> =
        vec![[SpatialVector::ZERO; MAX_JOINT_DOFS]; n_joints];
    let mut dof_width = vec![0usize; n_joints];
    let mut v_body = vec![SpatialVector::ZERO; n_joints];
    let mut a_body = vec![SpatialVector::ZERO; n_joints];
    let mut f_body = vec![SpatialVector::ZERO; n_joints];
    let mut local_dof_offset = vec![0usize; n_joints + 1];

    let mut local_dof_total = 0usize;
    for k in 0..n_joints {
        let j = j_begin + k;
        local_dof_offset[k] = local_dof_total;
        let jt = decode_joint_type(joint_type[j])?;

        let parent_body = joint_parent[j];
        let parent_world = if parent_body < 0 {
            crate::math::Transform::IDENTITY
        } else {
            body_q[parent_body as usize]
        };
        let xpw = parent_world.compose(&joint_x_p[j]);

        let mut dof_active = [false; MAX_JOINT_DOFS];
        let width = build_motion_subspace(
            jt,
            xpw,
            joint_axis[j],
            joint_dof_offset[j] as usize,
            limit_lower,
            limit_upper,
            &mut s_cols_per_joint[k],
            &mut dof_active,
        );
        dof_width[k] = width;
        local_dof_total += width;

        // Parent body's spatial velocity / acceleration. Root: zero
        // velocity, gravity-loaded acceleration so f = I·a yields
        // the true gravity wrench at body i.
        let (v_p, a_p) = if let Some(p_local) = parent_joint_local[k] {
            (v_body[p_local], a_body[p_local])
        } else {
            let world = body_world[joint_child[j] as usize].max(0) as usize;
            let g = gravity[world];
            // Spatial acceleration at world origin equivalent to
            // free-fall in -g: linear part is -g (so f = I·(-g)
            // gives the wrench m·g downward).
            (SpatialVector::ZERO, SpatialVector::new(-g, Vec3::ZERO))
        };

        let gbase = joint_dof_offset[j] as usize;
        let lbase = (joint_dof_offset[j] - joint_dof_offset[j_begin]) as usize;
        let mut v_j = SpatialVector::ZERO;
        let mut a_j = SpatialVector::ZERO;
        for a in 0..width {
            let qd_a = joint_qd.get(gbase + a).copied().unwrap_or(0.0);
            v_j = v_j + s_cols_per_joint[k][a].scale(qd_a);
            a_j = a_j + s_cols_per_joint[k][a].scale(qdd[lbase + a]);
        }
        let v_i = v_p + v_j;
        let a_i = a_p + a_j + spatial_cross(v_i, v_j);
        v_body[k] = v_i;
        a_body[k] = a_i;

        // Body spatial inertia at world origin.
        let child = joint_child[j] as usize;
        let cw = body_q[child];
        let com_world = cw.position + cw.rotation * body_com[child];
        let r = Mat3::from_quat(cw.rotation);
        let i_world = r * body_inertia[child] * r.transpose();
        let i_s = spatial_inertia_at_world_origin(body_mass[child], com_world, i_world);

        // Net spatial force on body i: f = I·a + v ×* (I·v).
        let i_v = i_s.mul_sv(v_i);
        let bias = spatial_cross_dual(v_i, i_v);
        f_body[k] = i_s.mul_sv(a_i) + bias;
    }
    local_dof_offset[n_joints] = local_dof_total;

    // ── Pass 2: backward sweep ────────────────────────────────────
    // For world-origin spatial vectors, parent-child force transfer
    // is just summation (no parallel-axis term needed because all
    // forces are already expressed at world origin).
    let mut tau = vec![0.0_f64; local_dof_total];
    for k in (0..n_joints).rev() {
        let j = j_begin + k;
        if dof_width[k] == 0 {
            // Fixed joints still relay force to the parent below.
            if let Some(p_local) = parent_joint_local[k] {
                f_body[p_local] = f_body[p_local] + f_body[k];
            }
            continue;
        }
        let f_k = f_body[k];
        // τ_k[a] = S_k[a]^T · f_k.
        for a in 0..dof_width[k] {
            tau[local_dof_offset[k] + a] = spatial_dot(s_cols_per_joint[k][a], f_k);
        }

        // Propagate force to parent.
        if let Some(p_local) = parent_joint_local[k] {
            f_body[p_local] = f_body[p_local] + f_k;
        }
        let _ = j;
    }
    Ok(tau)
}

fn spatial_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

fn spatial_inertia_at_world_origin(mass: Real, c: Vec3, i_world: Mat3) -> SpatialMatrix6 {
    let s = [[0.0, -c.z, c.y], [c.z, 0.0, -c.x], [-c.y, c.x, 0.0]];
    let mut s2 = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0_f64;
            for k in 0..3 {
                acc += s[i][k] * s[k][j];
            }
            s2[i][j] = acc;
        }
    }
    let mut m_out = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        m_out[i][i] = mass;
    }
    for i in 0..3 {
        for j in 0..3 {
            m_out[i][3 + j] = -mass * s[i][j];
            m_out[3 + i][j] = mass * s[i][j];
        }
    }
    let i_arr = [
        [i_world.x_axis.x, i_world.y_axis.x, i_world.z_axis.x],
        [i_world.x_axis.y, i_world.y_axis.y, i_world.z_axis.y],
        [i_world.x_axis.z, i_world.y_axis.z, i_world.z_axis.z],
    ];
    for i in 0..3 {
        for j in 0..3 {
            m_out[3 + i][3 + j] = i_arr[i][j] - mass * s2[i][j];
        }
    }
    SpatialMatrix6 { m: m_out }
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
        _ => Err(Error::Build(format!("RNEA: unknown joint_type {v}"))),
    }
}

fn articulation_range(model: &Model, articulation: usize) -> Result<(usize, usize)> {
    if model.articulation_count == 0 {
        return Ok((0, model.joint_count));
    }
    let starts = model.articulation_start.host()?;
    if articulation + 1 >= starts.len() {
        return Err(Error::Build(format!(
            "RNEA: articulation {articulation} out of range (count = {})",
            model.articulation_count
        )));
    }
    Ok((
        starts[articulation] as usize,
        starts[articulation + 1] as usize,
    ))
}
