//! Composite Rigid Body Algorithm — joint-space mass matrix M(q).
//!
//! Returns the symmetric `N×N` mass matrix where `N` = total joint
//! DOFs in the requested articulation. M is the matrix that satisfies
//!
//!     τ = M(q)·q̈ + C(q, q̇) + G(q)
//!
//! and is the linchpin for inverse dynamics (RNEA computes the
//! right-hand side), impedance control, IK damping, and the operational-
//! space inertia. Algorithm follows Featherstone (2008) §6.3 — three
//! passes over the joint chain:
//!
//!   1. Forward kinematics + per-body world-frame spatial inertia.
//!   2. Leaf-to-root composite-inertia accumulation: `Iᶜᵢ = Iᵢ + Σ Iᶜⱼ`
//!      where `j` ranges over child joints.
//!   3. For each joint `i`, compute `F = Iᶜᵢ · Sᵢ`, set `M[i,i] = Sᵢ·F`,
//!      then walk parent chain setting `M[i,p] = M[p,i] = Sₚ·F`.
//!
//! Single-articulation. Revolute + Prismatic only (the two 1-DOF joints
//! Featherstone forward dynamics already supports). `Fixed` is handled
//! by skipping the joint (it contributes no row/column). Multi-DOF
//! joints (Ball/Free/D6) are TODO — they need a per-DOF axis split
//! identical to RNEA's.

use crate::core::{Error, Result};
use crate::math::{
    spatial_jacobian_column, JacobianKind, Mat3, Real, SpatialMatrix6, SpatialVector, Vec3,
};
use crate::sim::{JointType, Model, State};

/// Joint-space mass matrix `M(q)` for the requested articulation.
///
/// Returns a flat `N·N` row-major array; `M[i,j] = out[i*N + j]`.
/// `N` equals the articulation's total joint DOFs (counted via
/// `joint_dof_offset`). Use `model.articulation_count == 0` to fall
/// back to the implicit "all joints in declaration order" articulation
/// — matches the convention used by `FeatherstoneSolver`.
pub fn mass_matrix(model: &Model, state: &State, articulation: usize) -> Result<Vec<Real>> {
    if model.joint_count == 0 {
        return Ok(Vec::new());
    }

    let (j_begin, j_end) = articulation_range(model, articulation)?;
    let n_joints = j_end - j_begin;
    if n_joints == 0 {
        return Ok(Vec::new());
    }

    // Caller's responsibility to ensure `state.body_q` reflects the
    // current `joint_q` — typically this means calling `mass_matrix`
    // immediately after a Featherstone step (which writes `body_q`
    // via FK in pass 4) or after explicit `eval_fk(model, state)`.
    let body_q = state.body_q.host()?;
    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_x_p = model.joint_x_p.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;

    // ── Local indexing within this articulation ──────────────────
    // For each local joint k, find its parent's local index (or None
    // if root). Parent here means the joint whose CHILD body is this
    // joint's parent body.
    let mut parent_joint_local: Vec<Option<usize>> = vec![None; n_joints];
    for k in 0..n_joints {
        let j = j_begin + k;
        let parent_body = joint_parent[j];
        if parent_body < 0 {
            parent_joint_local[k] = None;
            continue;
        }
        // Search earlier joints in the same articulation for the one
        // whose `child == parent_body`. Featherstone's CSR articulation
        // layout puts parents before children in declaration order,
        // so a forward scan is sufficient.
        for kk in 0..k {
            let jj = j_begin + kk;
            if joint_child[jj] == parent_body {
                parent_joint_local[k] = Some(kk);
                break;
            }
        }
    }

    // ── Per-joint world-frame motion subspace S (at world origin) ─
    // and joint origin in world. For revolute joints S has nonzero
    // angular component (the axis); for prismatic, S is purely
    // linear. Ball / Free / D6 are still TODO.
    let mut s_per_joint: Vec<SpatialVector> = vec![SpatialVector::ZERO; n_joints];
    // Per-joint DOF offset in the local M layout.
    let mut local_dof_offset: Vec<usize> = vec![0; n_joints + 1];

    // Per-body world-frame spatial inertia at world origin.
    let mut body_inertia_world: Vec<SpatialMatrix6> = vec![SpatialMatrix6::ZERO; n_joints];

    let mut local_dof_total = 0usize;
    for k in 0..n_joints {
        let j = j_begin + k;
        let jt = decode_joint_type(joint_type[j])?;
        local_dof_offset[k] = local_dof_total;
        let dof_count = match jt {
            JointType::Revolute | JointType::Prismatic => 1,
            JointType::Fixed => 0,
            JointType::Ball | JointType::Free | JointType::Distance | JointType::D6 => {
                return Err(Error::Build(format!(
                    "CRBA: joint type {:?} not yet supported (revolute/prismatic only)",
                    jt
                )));
            }
        };
        local_dof_total += dof_count;

        // Joint origin in world — parent's body frame composed with x_p.
        let parent_body = joint_parent[j];
        let parent_world = if parent_body < 0 {
            crate::math::Transform::IDENTITY
        } else {
            body_q[parent_body as usize]
        };
        let xpw = parent_world.compose(&joint_x_p[j]);
        let joint_origin_w = xpw.position;
        let axis_w = xpw.rotation * joint_axis[j];

        s_per_joint[k] = match jt {
            JointType::Revolute => spatial_jacobian_column(
                JacobianKind::Revolute,
                axis_w,
                joint_origin_w,
                Vec3::ZERO,
            ),
            JointType::Prismatic => spatial_jacobian_column(
                JacobianKind::Prismatic,
                axis_w,
                joint_origin_w,
                Vec3::ZERO,
            ),
            JointType::Fixed => SpatialVector::ZERO,
            _ => unreachable!(),
        };

        // Body spatial inertia at world origin (in `[linear; angular]`
        // convention). Built directly from the parallel-axis formula
        // because the existing `SpatialMatrix6::from_body` was set up
        // for Featherstone's chain-velocity convention, which differs
        // from CRBA's "spatial vectors at world origin" convention by
        // a sign flip in the off-diagonal blocks.
        let child = joint_child[j] as usize;
        let cw = body_q[child];
        let com_world = cw.position + cw.rotation * body_com[child];
        let r = Mat3::from_quat(cw.rotation);
        let i_world = r * body_inertia[child] * r.transpose();
        body_inertia_world[k] =
            spatial_inertia_at_world_origin(body_mass[child], com_world, i_world);
    }
    local_dof_offset[n_joints] = local_dof_total;

    // Sanity: this articulation's DOF total should agree with the
    // model's `joint_dof_offset` accounting (sum of DOF widths
    // across `[j_begin, j_end)`).
    let model_dof_total = (joint_dof_offset[j_end] - joint_dof_offset[j_begin]) as usize;
    debug_assert_eq!(model_dof_total, local_dof_total);

    let n = local_dof_total;
    if n == 0 {
        return Ok(Vec::new());
    }

    // ── Composite inertia (leaf-to-root accumulate) ──────────────
    let mut composite = body_inertia_world.clone();
    for k in (0..n_joints).rev() {
        if let Some(p) = parent_joint_local[k] {
            composite[p] = composite[p].add(&composite[k]);
        }
    }

    // ── Fill M ────────────────────────────────────────────────────
    let mut m_out = vec![0.0_f64; n * n];
    for k in 0..n_joints {
        let s_k = s_per_joint[k];
        let dof_k = local_dof_offset[k];
        let dof_width_k = local_dof_offset[k + 1] - local_dof_offset[k];
        if dof_width_k == 0 {
            continue;
        }
        // F = Iᶜ · S
        let f = composite[k].mul_sv(s_k);

        // Diagonal entry M[k,k] = S · F.
        let m_kk = spatial_dot(s_k, f);
        m_out[dof_k * n + dof_k] = m_kk;

        // Walk up parent chain — every ancestor joint p contributes
        // M[k,p] = M[p,k] = Sₚ · F.
        let mut cur = parent_joint_local[k];
        while let Some(p) = cur {
            let s_p = s_per_joint[p];
            let dof_p = local_dof_offset[p];
            let dof_width_p = local_dof_offset[p + 1] - local_dof_offset[p];
            if dof_width_p > 0 {
                let m_kp = spatial_dot(s_p, f);
                m_out[dof_k * n + dof_p] = m_kp;
                m_out[dof_p * n + dof_k] = m_kp;
            }
            cur = parent_joint_local[p];
        }
    }
    Ok(m_out)
}

fn spatial_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

/// Spatial inertia of a rigid body about world origin, in
/// `[linear; angular]` convention.
///
///     I_s = [   m · I_3       -m · [c]×        ]
///           [   m · [c]×       I_C - m · [c]×² ]
///
/// `c` is the COM position in world. `i_world` is the inertia tensor
/// at the COM, expressed in world axes. Mapping spatial velocity
/// `v_O = [v_at_origin, ω]` through this matrix yields
/// `[P; L_about_origin]`. Verified by `M[1,1] = m·d² + I_xx` for the
/// double-pendulum's leaf joint in `tests/crba_mass_matrix.rs`.
fn spatial_inertia_at_world_origin(mass: Real, c: Vec3, i_world: Mat3) -> SpatialMatrix6 {
    let s = [
        [0.0, -c.z, c.y],
        [c.z, 0.0, -c.x],
        [-c.y, c.x, 0.0],
    ];
    // skew(c)² (3×3 matrix mul).
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
    // Top-left: m·I_3
    for i in 0..3 {
        m_out[i][i] = mass;
    }
    // Top-right: -m · skew(c)
    for i in 0..3 {
        for j in 0..3 {
            m_out[i][3 + j] = -mass * s[i][j];
        }
    }
    // Bottom-left: +m · skew(c)
    for i in 0..3 {
        for j in 0..3 {
            m_out[3 + i][j] = mass * s[i][j];
        }
    }
    // Bottom-right: I_C - m · skew(c)²
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
        _ => Err(Error::Build(format!("CRBA: unknown joint_type {v}"))),
    }
}

fn articulation_range(model: &Model, articulation: usize) -> Result<(usize, usize)> {
    if model.articulation_count == 0 {
        return Ok((0, model.joint_count));
    }
    let starts = model.articulation_start.host()?;
    if articulation + 1 >= starts.len() {
        return Err(Error::Build(format!(
            "CRBA: articulation {articulation} out of range (count = {})",
            model.articulation_count
        )));
    }
    Ok((starts[articulation] as usize, starts[articulation + 1] as usize))
}
