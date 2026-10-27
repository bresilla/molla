//! Composite Rigid Body Algorithm — joint-space mass matrix M(q).
//!
//! Returns the symmetric `N×N` mass matrix where `N` = total joint
//! DOFs in the requested articulation. M is the matrix that satisfies
//!
//! ```text
//! τ = M(q)·q̈ + C(q, q̇) + G(q)
//! ```
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
//! Single-articulation. All joint types are supported via the shared
//! per-DOF motion subspace (`featherstone::build_motion_subspace`):
//! Revolute / Prismatic (1 DOF), Ball (3), Free / Distance / D6 (6,
//! with D6 lock filtering), Cable (2). `Fixed` is handled by skipping
//! the joint (it contributes no row/column).

use crate::core::{Error, Result};
use crate::featherstone::{MAX_JOINT_DOFS, build_motion_subspace};
use crate::math::{Mat3, Real, SpatialMatrix6, SpatialVector, Vec3};
use crate::sim::{BodyFlags, JointType, Model, State};

/// Joint-space mass matrix `M(q)` for the requested articulation.
///
/// Returns a flat `N·N` row-major array; `M[i,j] = out[i*N + j]`.
/// `N` equals the articulation's total joint DOFs (counted via
/// `joint_dof_offset`). Use `model.articulation_count == 0` to fall
/// back to the implicit "all joints in declaration order" articulation
/// — matches the convention used by `FeatherstoneSolver`.
/// Finite equality Ball/D6 axes have zero rows and columns.
pub fn mass_matrix(model: &Model, state: &State, articulation: usize) -> Result<Vec<Real>> {
    matrix::<true>(model, state, articulation)
}

#[cfg(test)]
thread_local! {
    /// Response matrices this thread has built.
    pub(crate) static RESPONSE_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Mass matrix with partial angular columns retained for coupled constraint response.
pub fn response_mass_matrix(
    model: &Model,
    state: &State,
    articulation: usize,
) -> Result<Vec<Real>> {
    #[cfg(test)]
    RESPONSE_BUILDS.with(|builds| builds.set(builds.get() + 1));
    matrix::<false>(model, state, articulation)
}

pub(crate) fn filter_locked_columns(
    kind: JointType,
    offset: usize,
    lower: &[Real],
    upper: &[Real],
    columns: &mut [SpatialVector; MAX_JOINT_DOFS],
) {
    let width = match kind {
        JointType::Ball => 3,
        JointType::D6 => 6,
        _ => return,
    };
    for (axis, column) in columns.iter_mut().take(width).enumerate() {
        let lo = lower.get(offset + axis).copied().unwrap_or(Real::NAN);
        let hi = upper.get(offset + axis).copied().unwrap_or(Real::NAN);
        if lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12 {
            *column = SpatialVector::ZERO;
        }
    }
}

fn matrix<const FILTER_LOCKED: bool>(
    model: &Model,
    state: &State,
    articulation: usize,
) -> Result<Vec<Real>> {
    let _span = tracing::debug_span!("mass_matrix").entered();
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
    let origin = body_q[model.joint_child.host()?[j_begin] as usize].position;
    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;

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
    // Shared with Featherstone forward dynamics — every joint type is
    // expanded to its per-DOF columns (Ball → 3, Free/Distance/D6 → 6
    // with D6 lock filtering, Cable → 2, Revolute/Prismatic → 1).
    let mut s_cols_per_joint: Vec<[SpatialVector; MAX_JOINT_DOFS]> =
        vec![[SpatialVector::ZERO; MAX_JOINT_DOFS]; n_joints];
    let mut dof_width: Vec<usize> = vec![0; n_joints];
    // Per-joint DOF offset in the local M layout.
    let mut local_dof_offset: Vec<usize> = vec![0; n_joints + 1];

    // Per-body world-frame spatial inertia at world origin, then composite.
    let mut composite: Vec<SpatialMatrix6> = vec![SpatialMatrix6::ZERO; n_joints];

    let mut local_dof_total = 0usize;
    for k in 0..n_joints {
        let j = j_begin + k;
        let jt = decode_joint_type(joint_type[j])?;
        local_dof_offset[k] = local_dof_total;

        let (mut xpw, direction) = model.joint_motion_frame(body_q, state.joint_q.host()?, j)?;
        xpw.position -= origin;

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
        if FILTER_LOCKED {
            filter_locked_columns(
                jt,
                joint_dof_offset[j] as usize,
                limit_lower,
                limit_upper,
                &mut s_cols_per_joint[k],
            );
        }
        for column in &mut s_cols_per_joint[k] {
            *column = column.scale(direction);
        }
        dof_width[k] = width;
        if model.joint_enabled.host()?[j] == 0
            || model.body_flags.host()?[joint_child[j] as usize] & BodyFlags::IMMOVABLE != 0
        {
            s_cols_per_joint[k].fill(SpatialVector::ZERO);
        }
        local_dof_total += width;

        // Body spatial inertia at world origin (in `[linear; angular]`
        // convention). Built directly from the parallel-axis formula
        // because the existing `SpatialMatrix6::from_body` was set up
        // for Featherstone's chain-velocity convention, which differs
        // from CRBA's "spatial vectors at world origin" convention by
        // a sign flip in the off-diagonal blocks.
        let child = joint_child[j] as usize;
        let cw = body_q[child];
        let com_world = (cw.position - origin) + cw.rotation * body_com[child];
        let r = Mat3::from_quat(cw.rotation);
        let i_world = r * body_inertia[child] * r.transpose();
        composite[k] = spatial_inertia_at_world_origin(body_mass[child], com_world, i_world);
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
    for k in (0..n_joints).rev() {
        if let Some(p) = parent_joint_local[k] {
            composite[p] = composite[p].add(&composite[k]);
        }
    }

    // ── Fill M ────────────────────────────────────────────────────
    // Per DOF column a of joint k: F = Iᶜₖ · Sₖ[a]; the joint's own
    // dof-block gets M[a,b] = Sₖ[b]·F (symmetric since Iᶜ is), and
    // every ancestor joint p contributes M[ka,pb] = M[pb,ka] = Sₚ[b]·F.
    // Locked D6 columns are ZERO vectors and produce zero rows/cols.
    let mut m_out = vec![0.0_f64; n * n];
    for k in 0..n_joints {
        let dof_k = local_dof_offset[k];
        for a in 0..dof_width[k] {
            let f = composite[k].mul_sv(s_cols_per_joint[k][a]);
            for b in 0..dof_width[k] {
                m_out[(dof_k + a) * n + (dof_k + b)] = spatial_dot(s_cols_per_joint[k][b], f);
            }
            let mut cur = parent_joint_local[k];
            while let Some(pj) = cur {
                let dof_p = local_dof_offset[pj];
                for b in 0..dof_width[pj] {
                    let m_ab = spatial_dot(s_cols_per_joint[pj][b], f);
                    m_out[(dof_k + a) * n + (dof_p + b)] = m_ab;
                    m_out[(dof_p + b) * n + (dof_k + a)] = m_ab;
                }
                cur = parent_joint_local[pj];
            }
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
/// ```text
/// I_s = [   m · I_3       -m · [c]×        ]
///       [   m · [c]×       I_C - m · [c]×² ]
/// ```
///
/// `c` is the COM position in world. `i_world` is the inertia tensor
/// at the COM, expressed in world axes. Mapping spatial velocity
/// `v_O = [v_at_origin, ω]` through this matrix yields
/// `[P; L_about_origin]`. Verified by `M[1,1] = m·d² + I_xx` for the
/// double-pendulum's leaf joint in `tests/crba_mass_matrix.rs`.
fn spatial_inertia_at_world_origin(mass: Real, c: Vec3, i_world: Mat3) -> SpatialMatrix6 {
    let s = [[0.0, -c.z, c.y], [c.z, 0.0, -c.x], [-c.y, c.x, 0.0]];
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
        7 => Ok(JointType::Cable),
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
    Ok((
        starts[articulation] as usize,
        starts[articulation + 1] as usize,
    ))
}
