//! Recursive Newton-Euler — joint-space inverse dynamics.
//!
//! Given `(q, q̇, q̈)`, returns the joint forces `τ` that would
//! produce the requested accelerations. Two passes per articulation:
//!
//!   1. Forward (root → leaf): compute each body's spatial velocity,
//!      acceleration, and net force.
//!         v_i = v_p + S_i·q̇_i
//!         a_i = a_p + S_i·q̈_i + v_i × (S_i·q̇_i)
//!         f_i = I_i·a_i + v_i ×* (I_i·v_i)
//!   2. Backward (leaf → root): project body forces onto joint motion
//!      subspaces and accumulate.
//!         τ_i = S_i^T · f_i
//!         f_p += f_i  (transferred via parent-child cross terms;
//!                      handled implicitly by world-origin spatial
//!                      vectors).
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
use crate::math::{
    JacobianKind, Mat3, Real, SpatialMatrix6, SpatialVector, Vec3, spatial_cross,
    spatial_cross_dual, spatial_jacobian_column,
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
    let mut s_per_joint = vec![SpatialVector::ZERO; n_joints];
    let mut v_body = vec![SpatialVector::ZERO; n_joints];
    let mut a_body = vec![SpatialVector::ZERO; n_joints];
    let mut f_body = vec![SpatialVector::ZERO; n_joints];
    let mut local_dof_offset = vec![0usize; n_joints + 1];

    let mut local_dof_total = 0usize;
    for k in 0..n_joints {
        let j = j_begin + k;
        local_dof_offset[k] = local_dof_total;
        let jt = decode_joint_type(joint_type[j])?;
        let dof = match jt {
            JointType::Revolute | JointType::Prismatic => 1,
            JointType::Fixed => 0,
            _ => {
                return Err(Error::Build(format!(
                    "RNEA: joint type {:?} not yet supported",
                    jt
                )));
            }
        };
        local_dof_total += dof;

        let parent_body = joint_parent[j];
        let parent_world = if parent_body < 0 {
            crate::math::Transform::IDENTITY
        } else {
            body_q[parent_body as usize]
        };
        let xpw = parent_world.compose(&joint_x_p[j]);
        let joint_origin_w = xpw.position;
        let axis_w = xpw.rotation * joint_axis[j];

        // Joint motion subspace at world origin (matches CRBA).
        let s_k = match jt {
            JointType::Revolute => {
                spatial_jacobian_column(JacobianKind::Revolute, axis_w, joint_origin_w, Vec3::ZERO)
            }
            JointType::Prismatic => {
                spatial_jacobian_column(JacobianKind::Prismatic, axis_w, joint_origin_w, Vec3::ZERO)
            }
            JointType::Fixed => SpatialVector::ZERO,
            // Multi-DOF joints (Ball/Free/Distance/D6) bail out at
            // the dof-count match above (returns Err); reaching this
            // arm would mean a new JointType variant was added without
            // updating the dof match. Guard against that drift.
            _ => unreachable!(
                "RNEA reached unsupported JointType {:?} after dof guard",
                jt
            ),
        };
        s_per_joint[k] = s_k;

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

        let qd_idx = (joint_dof_offset[j] - joint_dof_offset[j_begin]) as usize;
        let qd_k = if dof == 0 {
            0.0
        } else {
            joint_qd[joint_dof_offset[j] as usize]
        };
        let qdd_k = if dof == 0 { 0.0 } else { qdd[qd_idx] };

        let v_j = s_k.scale(qd_k);
        let v_i = v_p + v_j;
        let a_i = a_p + s_k.scale(qdd_k) + spatial_cross(v_i, v_j);
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
        let dof_width = local_dof_offset[k + 1] - local_dof_offset[k];
        if dof_width == 0 {
            continue;
        }
        let s_k = s_per_joint[k];
        let f_k = f_body[k];
        // τ_k = S_k^T · f_k.
        let tau_k = spatial_dot(s_k, f_k);
        tau[local_dof_offset[k]] = tau_k;

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
