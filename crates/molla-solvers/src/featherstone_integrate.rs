use super::{clamp_joint_pos, clamp_or_passthrough, joint_type_from_u32, project_angular_limits};
use crate::core::{Error, Result};
use crate::math::{Quat, Real, Vec3};
use crate::sim::{BodyFlags, JointType, Model};

#[allow(clippy::too_many_arguments)]
pub(super) fn integrate(
    model: &Model,
    joints: std::ops::Range<usize>,
    q_in: &[Real],
    qd_in: &[Real],
    q_dd: &[Real],
    dt: Real,
    joint_damping: Real,
    rigorous_distance: bool,
    external_scalar_bounds: bool,
    cap_rates: bool,
    q_new: &mut [Real],
    qd_new: &mut [Real],
) -> Result<()> {
    let joint_type = model.joint_type.host()?;
    let joint_coord_offset = model.joint_coord_offset.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let body_flags = model.body_flags.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_enabled = model.joint_enabled.host()?;
    let velocity_limit = model.joint_velocity_limit.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;
    let cap = |value, limit| {
        if cap_rates && !external_scalar_bounds {
            clamp_or_passthrough(value, limit)
        } else {
            value
        }
    };
    for j in joints {
        let jt = joint_type_from_u32(joint_type[j]);
        let coord_idx = joint_coord_offset[j] as usize;
        let dof_idx = joint_dof_offset[j] as usize;
        if body_flags[joint_child[j] as usize] & BodyFlags::IMMOVABLE != 0 || joint_enabled[j] == 0
        {
            let end = joint_dof_offset[j + 1] as usize;
            qd_new[dof_idx..end].fill(0.0);
            continue;
        }
        match jt {
            JointType::Revolute | JointType::Prismatic => {
                let lo = limit_lower.get(dof_idx).copied().unwrap_or(Real::NAN);
                let hi = limit_upper.get(dof_idx).copied().unwrap_or(Real::NAN);
                if lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12 {
                    q_new[coord_idx] = lo;
                    qd_new[dof_idx] = 0.0;
                    continue;
                }
                let qd_old = qd_in[dof_idx];
                let qd_int = qd_old + q_dd[dof_idx] * dt;
                let qd_damped = qd_int * (1.0 - joint_damping * dt).max(0.0);
                let qd_capped = cap(
                    qd_damped,
                    velocity_limit.get(dof_idx).copied().unwrap_or(Real::NAN),
                );
                let q_pred = q_in[coord_idx] + qd_capped * dt;
                if external_scalar_bounds {
                    qd_new[dof_idx] = qd_capped;
                    q_new[coord_idx] = q_pred;
                    continue;
                }
                let q_clamped = q_pred.clamp(lo, hi);
                let roundoff =
                    64.0 * Real::EPSILON * (1.0 + q_in[coord_idx].abs() + (qd_capped * dt).abs());
                let reaction_tolerance = 1e-8 * (dt + (q_clamped - q_in[coord_idx]).abs());
                if (q_pred - q_clamped).abs() > roundoff + reaction_tolerance {
                    return Err(Error::Solver(
                        "scalar stop reaction left an infeasible velocity".into(),
                    ));
                }
                qd_new[dof_idx] = qd_capped;
                q_new[coord_idx] = q_clamped;
            }
            JointType::Ball => {
                let mut omega = Vec3::new(
                    qd_in.get(dof_idx).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 1).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let alpha = Vec3::new(
                    q_dd.get(dof_idx).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 1).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                omega += alpha * dt;
                let damp = (1.0 - joint_damping * dt).max(0.0);
                omega *= damp;
                for k in 0..3 {
                    omega[k] = cap(omega[k], velocity_limit[dof_idx + k]);
                }
                let q_old = Quat::from_xyzw(
                    q_in[coord_idx],
                    q_in[coord_idx + 1],
                    q_in[coord_idx + 2],
                    q_in[coord_idx + 3],
                );
                let dq = Quat::from_xyzw(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
                let q_dot = dq * q_old;
                let q_pred = Quat::from_xyzw(
                    q_old.x + q_dot.x * dt,
                    q_old.y + q_dot.y * dt,
                    q_old.z + q_dot.z * dt,
                    q_old.w + q_dot.w * dt,
                )
                .normalize();
                let fixed = (0..3).all(|axis| {
                    let lo = limit_lower[dof_idx + axis];
                    let hi = limit_upper[dof_idx + axis];
                    lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12
                });
                let (q_pred, bounded_omega) = if external_scalar_bounds && !fixed {
                    (q_pred, omega)
                } else {
                    project_angular_limits(q_pred, omega, dof_idx, limit_lower, limit_upper, None)
                };
                omega = bounded_omega;
                q_new[coord_idx] = q_pred.x;
                q_new[coord_idx + 1] = q_pred.y;
                q_new[coord_idx + 2] = q_pred.z;
                q_new[coord_idx + 3] = q_pred.w;
                qd_new[dof_idx] = omega.x;
                qd_new[dof_idx + 1] = omega.y;
                qd_new[dof_idx + 2] = omega.z;
            }
            JointType::Free | JointType::Distance | JointType::D6 => {
                let mut v = Vec3::new(
                    qd_in.get(dof_idx).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 1).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let mut omega = Vec3::new(
                    qd_in.get(dof_idx + 3).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 4).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 5).copied().unwrap_or(0.0),
                );
                let a_lin = Vec3::new(
                    q_dd.get(dof_idx).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 1).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let alpha = Vec3::new(
                    q_dd.get(dof_idx + 3).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 4).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 5).copied().unwrap_or(0.0),
                );
                v += a_lin * dt;
                omega += alpha * dt;
                let damp = (1.0 - joint_damping * dt).max(0.0);
                v *= damp;
                omega *= damp;
                for k in 0..3 {
                    v[k] = cap(v[k], velocity_limit[dof_idx + k]);
                    omega[k] = cap(omega[k], velocity_limit[dof_idx + 3 + k]);
                }

                let p_old = Vec3::new(q_in[coord_idx], q_in[coord_idx + 1], q_in[coord_idx + 2]);
                let p_new = p_old + v * dt;
                let r_old = Quat::from_xyzw(
                    q_in[coord_idx + 3],
                    q_in[coord_idx + 4],
                    q_in[coord_idx + 5],
                    q_in[coord_idx + 6],
                );
                let dq = Quat::from_xyzw(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
                let r_dot = dq * r_old;
                let r_new = Quat::from_xyzw(
                    r_old.x + r_dot.x * dt,
                    r_old.y + r_dot.y * dt,
                    r_old.z + r_dot.z * dt,
                    r_old.w + r_dot.w * dt,
                )
                .normalize();
                q_new[coord_idx] = p_new.x;
                q_new[coord_idx + 1] = p_new.y;
                q_new[coord_idx + 2] = p_new.z;
                q_new[coord_idx + 3] = r_new.x;
                q_new[coord_idx + 4] = r_new.y;
                q_new[coord_idx + 5] = r_new.z;
                q_new[coord_idx + 6] = r_new.w;
                qd_new[dof_idx] = v.x;
                qd_new[dof_idx + 1] = v.y;
                qd_new[dof_idx + 2] = v.z;
                qd_new[dof_idx + 3] = omega.x;
                qd_new[dof_idx + 4] = omega.y;
                qd_new[dof_idx + 5] = omega.z;

                if matches!(jt, JointType::Distance)
                    && !external_scalar_bounds
                    && !rigorous_distance
                {
                    let lo = limit_lower.get(dof_idx).copied().unwrap_or(Real::NAN);
                    let hi = limit_upper.get(dof_idx).copied().unwrap_or(Real::NAN);

                    let p_curr =
                        Vec3::new(q_new[coord_idx], q_new[coord_idx + 1], q_new[coord_idx + 2]);
                    let dist = p_curr.length();
                    let mut clamped = p_curr;
                    let mut zero_radial_v = false;
                    if !lo.is_nan() && dist < lo && dist > 1e-12 {
                        clamped = p_curr * (lo / dist);
                        zero_radial_v = true;
                    }
                    if !hi.is_nan() && dist > hi && dist > 1e-12 {
                        clamped = p_curr * (hi / dist);
                        zero_radial_v = true;
                    }
                    if zero_radial_v {
                        q_new[coord_idx] = clamped.x;
                        q_new[coord_idx + 1] = clamped.y;
                        q_new[coord_idx + 2] = clamped.z;
                        let n = clamped.normalize_or_zero();
                        let radial = n * v.dot(n);
                        let v_proj = v - radial;
                        qd_new[dof_idx] = v_proj.x;
                        qd_new[dof_idx + 1] = v_proj.y;
                        qd_new[dof_idx + 2] = v_proj.z;
                    }
                }

                if matches!(jt, JointType::D6) {
                    for k in 0..3 {
                        let lo = limit_lower.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                        let hi = limit_upper.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                        if external_scalar_bounds
                            && !(lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12)
                        {
                            continue;
                        }
                        let p_k = q_new[coord_idx + k];
                        let qd_k = qd_new[dof_idx + k];
                        let (p_c, qd_c) = clamp_joint_pos(p_k, qd_k, lo, hi);
                        q_new[coord_idx + k] = p_c;
                        qd_new[dof_idx + k] = qd_c;
                    }

                    let r_now = Quat::from_xyzw(
                        q_new[coord_idx + 3],
                        q_new[coord_idx + 4],
                        q_new[coord_idx + 5],
                        q_new[coord_idx + 6],
                    );
                    let omega = Vec3::new(
                        qd_new[dof_idx + 3],
                        qd_new[dof_idx + 4],
                        qd_new[dof_idx + 5],
                    );
                    let partial = molla_sim::joint_control::partial_angular_lock(
                        &limit_lower[dof_idx + 3..dof_idx + 6],
                        &limit_upper[dof_idx + 3..dof_idx + 6],
                    );
                    let (r_proj, omega) = if external_scalar_bounds && partial {
                        (r_now, omega)
                    } else if external_scalar_bounds {
                        let locked = |k: usize| {
                            let lo = limit_lower[dof_idx + 3 + k];
                            let hi = limit_upper[dof_idx + 3 + k];
                            lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12
                        };
                        let lower: [Real; 3] = std::array::from_fn(|k| {
                            if locked(k) {
                                limit_lower[dof_idx + 3 + k]
                            } else {
                                Real::NEG_INFINITY
                            }
                        });
                        let upper: [Real; 3] = std::array::from_fn(|k| {
                            if locked(k) {
                                limit_upper[dof_idx + 3 + k]
                            } else {
                                Real::INFINITY
                            }
                        });
                        project_angular_limits(
                            r_now,
                            omega,
                            0,
                            &lower,
                            &upper,
                            Some((
                                Vec3::from_slice(&limit_lower[dof_idx + 3..dof_idx + 6]),
                                Vec3::from_slice(&limit_upper[dof_idx + 3..dof_idx + 6]),
                            )),
                        )
                    } else {
                        project_angular_limits(
                            r_now,
                            omega,
                            dof_idx + 3,
                            limit_lower,
                            limit_upper,
                            None,
                        )
                    };
                    q_new[coord_idx + 3] = r_proj.x;
                    q_new[coord_idx + 4] = r_proj.y;
                    q_new[coord_idx + 5] = r_proj.z;
                    q_new[coord_idx + 6] = r_proj.w;
                    qd_new[dof_idx + 3] = omega.x;
                    qd_new[dof_idx + 4] = omega.y;
                    qd_new[dof_idx + 5] = omega.z;
                }
            }
            JointType::Cable => {
                for k in 0..2 {
                    let qd_old = qd_in[dof_idx + k];
                    let qd_int = qd_old + q_dd[dof_idx + k] * dt;
                    let qd_damped = qd_int * (1.0 - joint_damping * dt).max(0.0);
                    let qd_capped = cap(
                        qd_damped,
                        velocity_limit
                            .get(dof_idx + k)
                            .copied()
                            .unwrap_or(Real::NAN),
                    );
                    let q_pred = q_in[coord_idx + k] + qd_capped * dt;
                    let lo = limit_lower.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                    let hi = limit_upper.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                    let (q_clamped, qd_after) = clamp_joint_pos(q_pred, qd_capped, lo, hi);
                    qd_new[dof_idx + k] = qd_after;
                    q_new[coord_idx + k] = q_clamped;
                }
            }
            JointType::Fixed => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BodyId;
    use crate::math::Transform;
    use crate::sim::{BodyParams, ModelBuilder};

    #[test]
    fn position_displacements_are_not_clipped_by_physical_velocity_limits() {
        for kind in [0, 1, 2, 4, 5, 6] {
            let mut builder = ModelBuilder::new();
            let body = builder.add_body(BodyParams::new().with_mass(1.));
            let p = BodyId::NONE;
            let frame = Transform::IDENTITY;
            let joint = match kind {
                0 => builder.add_joint_prismatic(p, body, frame, frame, Vec3::X),
                1 => builder.add_joint_revolute(p, body, frame, frame, Vec3::Y),
                2 => builder.add_joint_ball(p, body, frame, frame),
                4 => builder.add_joint_free(p, body, frame, frame),
                5 => builder.add_joint_distance(p, body, frame, frame),
                6 => builder.add_joint_d6(p, body, frame, frame),
                _ => unreachable!(),
            }
            .unwrap();
            let count = match kind {
                0 | 1 => 1,
                2 => 3,
                _ => 6,
            };
            builder
                .set_joint_velocity_limit(joint, &vec![1e-6; count])
                .unwrap();
            let model = builder.build().unwrap();
            let input = model.state().unwrap();
            for cap_rates in [true, false] {
                let mut q = input.joint_q.host().unwrap().to_vec();
                let mut rates = vec![0.; count];
                integrate(
                    &model,
                    0..model.joint_count,
                    input.joint_q.host().unwrap(),
                    &vec![0.25; count],
                    &vec![0.; count],
                    1.,
                    0.,
                    false,
                    false,
                    cap_rates,
                    &mut q,
                    &mut rates,
                )
                .unwrap();
                let amount = if cap_rates { 1e-6 } else { 0.25 };
                assert!(
                    rates.iter().all(|r| (r - amount).abs() < 1e-14),
                    "kind={kind}, cap={cap_rates}, rates={rates:?}"
                );
                if kind == 0 || kind == 1 {
                    assert!((q[0] - amount).abs() < 1e-14);
                } else {
                    let offset = if kind == 2 { 0 } else { 3 };
                    for coordinate in &q[..offset] {
                        assert!((coordinate - amount).abs() < 1e-14);
                    }
                    let rotation =
                        Quat::from_xyzw(amount * 0.5, amount * 0.5, amount * 0.5, 1.).normalize();
                    for (a, b) in q[offset..offset + 4].iter().zip(rotation.to_array()) {
                        assert!((a - b).abs() < 1e-14);
                    }
                }
            }
        }
    }
}
