//! Recursive Newton-Euler inverse dynamics and transmitted joint wrenches.
//! Spatial recursion uses a fixed, translated origin for each evaluation.

use crate::core::{Error, Result};
use crate::featherstone::{MAX_JOINT_DOFS, build_motion_subspace};
use crate::math::{
    Mat3, Real, SpatialMatrix6, SpatialVector, Vec3, spatial_cross, spatial_cross_dual,
};
use crate::sim::{BodyFlags, JointType, Model, State};

#[derive(Clone, Debug, Default)]
pub struct InverseDynamicsWrenches {
    /// Generalized forces in articulation-local DOF order.
    pub efforts: Vec<Real>,
    /// Child-side world wrenches about each compiled child joint anchor, in local joint order.
    pub joints: Vec<SpatialVector>,
}

/// Inverse dynamics: returns `τ` (length = articulation DOFs) for the
/// requested `(q, q̇, q̈)`.
///
/// `state.joint_q` and `state.joint_qd` provide `q` and `q̇`. Pass `q̈`
/// (length = articulation DOFs) explicitly. State.body_q must be
/// up-to-date for the current `q` (call `eval_fk` first if unsure).
/// Finite equality Ball/D6 axes have zero motion columns and generalized efforts.
pub fn inverse_dynamics(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
) -> Result<Vec<Real>> {
    Ok(evaluate::<true>(model, state, articulation, qdd, None, None)?.efforts)
}

pub(crate) fn inverse_dynamics_response(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
) -> Result<Vec<Real>> {
    Ok(evaluate::<false>(model, state, articulation, qdd, None, None)?.efforts)
}

/// Required child-side joint wrenches for supplied accelerations and external COM loads.
/// Gravity is included separately; `external` contains one world COM wrench per body.
/// State body forces are not read. Fixed joints retain samples despite having no DOFs.
/// Kinematic bodies use prescribed COM velocities with zero COM/angular acceleration.
/// Partial angular columns remain available for constraint support loads.
pub fn inverse_dynamics_wrenches(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
    external: &[SpatialVector],
) -> Result<InverseDynamicsWrenches> {
    WrenchInputs::new(model, external, None)?.evaluate(state, articulation, qdd)
}

/// Required joint wrenches with prescribed world COM and angular boundary accelerations.
/// `boundary_accelerations` has one entry per body; nonkinematic entries must be zero.
/// Inputs are instantaneous inertial derivatives; this function does not advance state.
pub fn inverse_dynamics_wrenches_with_boundary_accelerations(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
    external: &[SpatialVector],
    boundary_accelerations: &[SpatialVector],
) -> Result<InverseDynamicsWrenches> {
    WrenchInputs::new(model, external, Some(boundary_accelerations))?.evaluate(
        state,
        articulation,
        qdd,
    )
}

pub(crate) struct WrenchInputs<'a> {
    model: &'a Model,
    external: &'a [SpatialVector],
    boundary_accelerations: Option<&'a [SpatialVector]>,
}

impl<'a> WrenchInputs<'a> {
    pub(crate) fn new(
        model: &'a Model,
        external: &'a [SpatialVector],
        boundary_accelerations: Option<&'a [SpatialVector]>,
    ) -> Result<Self> {
        if let Some(boundary) = boundary_accelerations {
            let flags = model.body_flags.host()?;
            if boundary.len() != model.body_count
                || boundary.iter().enumerate().any(|(i, a)| {
                    !a.linear.is_finite()
                        || !a.angular.is_finite()
                        || (a != &SpatialVector::ZERO
                            && (flags[i] & BodyFlags::KINEMATIC == 0
                                || flags[i] & BodyFlags::FIXED != 0))
                })
            {
                return Err(Error::Build(
                    "RNEA: invalid prescribed boundary acceleration".into(),
                ));
            }
        }
        if external.len() != model.body_count
            || external
                .iter()
                .any(|f| !f.linear.is_finite() || !f.angular.is_finite())
        {
            return Err(Error::Build("RNEA: invalid external COM wrenches".into()));
        }
        Ok(Self {
            model,
            external,
            boundary_accelerations,
        })
    }

    pub(crate) fn evaluate(
        &self,
        state: &State,
        articulation: usize,
        qdd: &[Real],
    ) -> Result<InverseDynamicsWrenches> {
        let result = evaluate::<false>(
            self.model,
            state,
            articulation,
            qdd,
            Some(self.external),
            self.boundary_accelerations,
        )?;
        if result.efforts.iter().any(|value| !value.is_finite())
            || result
                .joints
                .iter()
                .any(|wrench| !wrench.linear.is_finite() || !wrench.angular.is_finite())
        {
            return Err(Error::Solver("RNEA: non-finite transmitted wrench".into()));
        }
        Ok(result)
    }
}

fn evaluate<const FILTER_LOCKED: bool>(
    model: &Model,
    state: &State,
    articulation: usize,
    qdd: &[Real],
    external: Option<&[SpatialVector]>,
    boundary_accelerations: Option<&[SpatialVector]>,
) -> Result<InverseDynamicsWrenches> {
    if model.joint_count == 0 {
        if articulation != 0 {
            return Err(Error::Build(
                "RNEA: articulation out of range for an empty model".into(),
            ));
        }
        if !qdd.is_empty() {
            return Err(Error::Build(
                "RNEA: nonempty acceleration for an empty model".into(),
            ));
        }
        return Ok(InverseDynamicsWrenches::default());
    }

    let (j_begin, j_end) = articulation_range(model, articulation)?;
    let n_joints = j_end - j_begin;
    if n_joints == 0 {
        if !qdd.is_empty() {
            return Err(Error::Build(
                "RNEA: nonempty acceleration for an empty articulation".into(),
            ));
        }
        return Ok(InverseDynamicsWrenches::default());
    }

    let body_q = state.body_q.host()?;
    let joint_q = state.joint_q.host()?;
    let joint_qd = state.joint_qd.host()?;
    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let body_flags = model.body_flags.host()?;
    let joint_enabled = model.joint_enabled.host()?;
    let body_world = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;
    let origin = body_q[joint_child[j_begin] as usize].position;

    let dof_total = (joint_dof_offset[j_end] - joint_dof_offset[j_begin]) as usize;
    if qdd.len() != dof_total {
        return Err(Error::Build(format!(
            "RNEA: qdd len {} != articulation DOFs {}",
            qdd.len(),
            dof_total
        )));
    }
    if qdd.iter().any(|a| !a.is_finite()) {
        return Err(Error::Build("RNEA: non-finite joint acceleration".into()));
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

        let (mut xpw, direction) = model.joint_motion_frame(body_q, joint_q, j)?;
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
            crate::crba::filter_locked_columns(
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
        let child = joint_child[j] as usize;
        if body_flags[child] & BodyFlags::IMMOVABLE != 0 || joint_enabled[j] == 0 {
            s_cols_per_joint[k].fill(SpatialVector::ZERO);
        }
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
        let mut v_i = v_p + v_j;
        let mut a_i = a_p + a_j + spatial_cross(v_i, v_j);
        if matches!(jt, JointType::Free | JointType::Distance | JointType::D6) {
            let translation =
                xpw.rotation * Vec3::new(joint_qd[gbase], joint_qd[gbase + 1], joint_qd[gbase + 2]);
            a_i.linear -= v_j.angular.cross(translation);
        }
        if body_flags[child] & BodyFlags::IMMOVABLE != 0 {
            let g = gravity[body_world[child].max(0) as usize];
            v_i = SpatialVector::ZERO;
            a_i = SpatialVector::new(-g, Vec3::ZERO);
            if body_flags[child] & BodyFlags::FIXED == 0 {
                let velocity = state.body_qd.host()?[child];
                let center =
                    (body_q[child].position - origin) + body_q[child].rotation * body_com[child];
                v_i = SpatialVector::new(
                    velocity.linear - velocity.angular.cross(center),
                    velocity.angular,
                );
                a_i.linear -= velocity.angular.cross(velocity.linear);
                if let Some(boundary) = boundary_accelerations {
                    a_i.linear += boundary[child].linear - boundary[child].angular.cross(center);
                    a_i.angular = boundary[child].angular;
                }
            }
        }
        v_body[k] = v_i;
        a_body[k] = a_i;

        // Body spatial inertia about the translated recursion origin.
        let cw = body_q[child];
        let com_world = (cw.position - origin) + cw.rotation * body_com[child];
        let r = Mat3::from_quat(cw.rotation);
        let i_world = r * body_inertia[child] * r.transpose();
        let i_s = spatial_inertia_about_origin(body_mass[child], com_world, i_world);

        // Net spatial force on body i: f = I·a + v ×* (I·v).
        let i_v = i_s.mul_sv(v_i);
        let bias = spatial_cross_dual(v_i, i_v);
        f_body[k] = i_s.mul_sv(a_i) + bias;
        if let Some(external) = external {
            let load = external[child];
            f_body[k] = f_body[k]
                - SpatialVector::new(load.linear, load.angular + com_world.cross(load.linear));
        }
    }
    local_dof_offset[n_joints] = local_dof_total;

    // Accumulate subtree wrenches about the common recursion origin.
    let mut tau = vec![0.0_f64; local_dof_total];
    let mut joints = if external.is_some() {
        vec![SpatialVector::ZERO; n_joints]
    } else {
        Vec::new()
    };
    let child_frames = model.joint_x_c.host()?;
    for k in (0..n_joints).rev() {
        let j = j_begin + k;
        if external.is_some() {
            let child = body_q[joint_child[j] as usize];
            let anchor = (child.position - origin) + child.rotation * child_frames[j].position;
            joints[k] = SpatialVector::new(
                f_body[k].linear,
                f_body[k].angular - anchor.cross(f_body[k].linear),
            );
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
    }
    Ok(InverseDynamicsWrenches {
        efforts: tau,
        joints,
    })
}

fn spatial_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

fn spatial_inertia_about_origin(mass: Real, c: Vec3, i_world: Mat3) -> SpatialMatrix6 {
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
        if articulation != 0 {
            return Err(Error::Build("RNEA: articulation out of range".into()));
        }
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

#[cfg(test)]
mod wrench_input_tests {
    use super::*;
    use molla_core::{BodyId, WorldId};
    use molla_math::Transform;
    use molla_sim::{BodyParams, ModelBuilder};

    #[test]
    fn validated_inputs_reuse_independent_articulations_and_reject_global_faults() {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), -Vec3::Y * 10.0);
        for mass in [2.0, 3.0, 4.0] {
            builder.begin_articulation();
            let body = builder.add_body(BodyParams::new().with_mass(mass));
            builder
                .add_joint_fixed(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
                .unwrap();
        }
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        molla_sim::eval_fk(&model, &mut state).unwrap();
        let loads = vec![SpatialVector::new(Vec3::new(3.0, 2.0, 1.0), Vec3::new(4.0, 5.0, 6.0)); 3];
        let boundary = vec![SpatialVector::ZERO; 3];
        let inputs = WrenchInputs::new(&model, &loads, Some(&boundary)).unwrap();
        for (art, mass) in [2.0, 3.0, 4.0].into_iter().enumerate() {
            let result = inputs.evaluate(&state, art, &[]).unwrap();
            let public = inverse_dynamics_wrenches_with_boundary_accelerations(
                &model,
                &state,
                art,
                &[],
                &loads,
                &boundary,
            )
            .unwrap();
            assert_eq!(result.joints, public.joints);
            assert!(result.efforts.is_empty());
            let expected = SpatialVector::new(
                Vec3::new(-3.0, mass * 10.0 - 2.0, -1.0),
                Vec3::new(-4.0, -5.0, -6.0),
            );
            assert!((result.joints[0].linear - expected.linear).length() < 1e-12);
            assert!((result.joints[0].angular - expected.angular).length() < 1e-12);
        }
        assert!(inputs.evaluate(&state, 0, &[0.0]).is_err());
        let mut invalid_loads = loads.clone();
        invalid_loads[2].linear.x = Real::NAN;
        assert!(WrenchInputs::new(&model, &invalid_loads, Some(&boundary)).is_err());
        assert!(inverse_dynamics_wrenches(&model, &state, 0, &[], &invalid_loads).is_err());
        let mut invalid_boundary = boundary.clone();
        invalid_boundary[2].linear.x = 1.0;
        assert!(WrenchInputs::new(&model, &loads, Some(&invalid_boundary)).is_err());
        assert!(
            inverse_dynamics_wrenches_with_boundary_accelerations(
                &model,
                &state,
                0,
                &[],
                &loads,
                &invalid_boundary,
            )
            .is_err()
        );
        assert!(WrenchInputs::new(&model, &loads[..2], None).is_err());
        assert!(WrenchInputs::new(&model, &loads, Some(&boundary[..2])).is_err());
    }
}
