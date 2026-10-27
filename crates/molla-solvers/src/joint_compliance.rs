use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::joint_control::JointCompliance;
use molla_sim::{JointType, Model, State};

fn effort(c: JointCompliance, position: Real, rate: Real, inertia: Real, dt: Real) -> Real {
    if c.enabled == 0 || c.natural_frequency == 0.0 || inertia == 0.0 {
        return 0.0;
    }
    if c.lower != c.upper && position >= c.lower && position <= c.upper {
        return 0.0;
    }
    let target = position.clamp(c.lower, c.upper);
    let omega = std::f64::consts::TAU * c.natural_frequency;
    let stiffness = omega * omega;
    let damping = 2.0 * c.damping_ratio * omega;
    let force = inertia * (-stiffness * (position - target) - (damping + stiffness * dt) * rate)
        / (1.0 + damping * dt + stiffness * dt * dt);
    if c.lower == c.upper {
        force
    } else if position < c.lower {
        force.max(0.0)
    } else {
        force.min(0.0)
    }
}

pub(crate) fn apply(
    model: &Model,
    state: &State,
    inertias: &[Real],
    dt: Real,
    forces: &mut [SpatialVector],
) -> Result<()> {
    let compliance = model.joint_compliance.host()?;
    if !compliance.iter().any(|c| c.enabled != 0) {
        return Ok(());
    }
    if compliance.len() != model.joint_dof_total || inertias.len() != model.joint_dof_total {
        return Err(Error::Solver(
            "invalid joint compliance buffer length".into(),
        ));
    }
    let q = state.joint_q.host()?;
    let qd = state.joint_qd.host()?;
    let poses = state.body_q.host()?;
    let com = model.body_com.host()?;
    let parents = model.joint_parent.host()?;
    let children = model.joint_child.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let coords = model.joint_coord_offset.host()?;
    let enabled = model.joint_enabled.host()?;
    let kinds = model.joint_type.host()?;
    for j in 0..model.joint_count {
        let range = offsets[j] as usize..offsets[j + 1] as usize;
        if enabled[j] == 0 || !compliance[range.clone()].iter().any(|c| c.enabled != 0) {
            continue;
        }
        if kinds[j] != JointType::D6 as u32 || range.len() != 6 {
            return Err(Error::Solver("joint compliance requires a D6 joint".into()));
        }
        let c = coords[j] as usize;
        let rotation = Quat::from_xyzw(q[c + 3], q[c + 4], q[c + 5], q[c + 6]);
        for record in &compliance[range.clone()] {
            record
                .validate()
                .map_err(|_| Error::Solver("invalid joint compliance parameters".into()))?;
        }
        let lower = std::array::from_fn(|axis| {
            let record = compliance[range.start + 3 + axis];
            if record.enabled != 0 {
                record.lower
            } else {
                Real::NEG_INFINITY
            }
        });
        let upper = std::array::from_fn(|axis| {
            let record = compliance[range.start + 3 + axis];
            if record.enabled != 0 {
                record.upper
            } else {
                Real::INFINITY
            }
        });
        let geometry = crate::joint_angles::geometry(
            rotation,
            Vec3::from_array(lower),
            Vec3::from_array(upper),
        );
        let angles = geometry.angles.to_array();
        let positions = [q[c], q[c + 1], q[c + 2], angles[0], angles[1], angles[2]];
        let mut efforts = [0.0; 6];
        for (axis, dof) in range.enumerate() {
            efforts[axis] = effort(compliance[dof], positions[axis], qd[dof], inertias[dof], dt);
        }
        if efforts.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver("non-finite joint compliance force".into()));
        }
        let (frame, direction) = model.joint_motion_frame(poses, q, j)?;
        let force = frame.rotation * Vec3::new(efforts[0], efforts[1], efforts[2]) * direction;
        let torque = frame.rotation * Vec3::new(efforts[3], efforts[4], efforts[5]) * direction;
        for (body, sign) in [(children[j], 1.0), (parents[j], -1.0)] {
            if body >= 0 {
                let body = body as usize;
                let center = poses[body].position + poses[body].rotation * com[body];
                forces[body].linear += sign * force;
                forces[body].angular += sign * (torque + (frame.position - center).cross(force));
                if !forces[body].linear.is_finite() || !forces[body].angular.is_finite() {
                    return Err(Error::Solver("non-finite joint compliance wrench".into()));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_core::BodyId;
    use molla_math::{Mat3, SpatialInertia, Transform};
    use molla_sim::joint_control::JointSoftness;
    use molla_sim::{BodyParams, ModelBuilder, eval_body_velocities, eval_fk};

    #[test]
    fn offset_rotated_and_reversed_wrenches_conserve_momentum_and_virtual_work() {
        for reversed in [false, true] {
            let mut builder = ModelBuilder::new();
            let params = BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 3.0,
                com: Vec3::new(0.3, -0.2, 0.1),
                inertia: Mat3::IDENTITY,
            });
            let parent = builder.add_body(params.clone());
            let child = builder.add_body(params);
            builder
                .add_joint_free(
                    BodyId::NONE,
                    parent,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                )
                .unwrap();
            let j = builder
                .add_joint_d6(
                    parent,
                    child,
                    Transform {
                        position: Vec3::new(0.8, -0.4, 0.6),
                        rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.4, 0.5),
                    },
                    Transform::from_translation(Vec3::new(-0.6, 0.7, 0.2)),
                )
                .unwrap();
            builder.set_joint_reversed(j, reversed).unwrap();
            builder
                .set_joint_compliance(
                    j,
                    JointSoftness {
                        natural_frequency: 8.0,
                        damping_ratio: 1.0,
                    },
                    &[0.0; 6],
                    &[0.0; 6],
                )
                .unwrap();
            let model = builder.build().unwrap();
            let mut state = model.state().unwrap();
            let c = model.joint_coord_offset.host().unwrap()[j.index()] as usize;
            let d = model.joint_dof_offset.host().unwrap()[j.index()] as usize;
            state.joint_q.host_mut().unwrap()[0..3].copy_from_slice(&[20.0, -10.0, 7.0]);
            state.joint_q.host_mut().unwrap()[c..c + 3].copy_from_slice(&[0.3, -0.1, 0.2]);
            let angles = [0.12, -0.08, 0.18];
            state.joint_q.host_mut().unwrap()[c + 3..c + 7].copy_from_slice(
                &Quat::from_euler(glam::EulerRot::XYZ, angles[0], angles[1], angles[2]).to_array(),
            );
            state.joint_qd.host_mut().unwrap()[..6]
                .copy_from_slice(&[0.3, -0.4, 0.2, -0.2, 0.1, 0.4]);
            state.joint_qd.host_mut().unwrap()[d..d + 6]
                .copy_from_slice(&[-0.2, 0.1, 0.4, 0.3, -0.1, 0.2]);
            eval_fk(&model, &mut state).unwrap();
            eval_body_velocities(&model, &mut state).unwrap();
            let mut forces = vec![SpatialVector::ZERO; 2];
            let inertias = vec![2.0; model.joint_dof_total];
            let dt = 1.0 / 960.0;
            apply(&model, &state, &inertias, dt, &mut forces).unwrap();
            let mut force = Vec3::ZERO;
            let mut torque = Vec3::ZERO;
            let mut work = 0.0;
            for (body, wrench) in forces.iter().enumerate() {
                let pose = state.body_q.host().unwrap()[body];
                let center = pose.position + pose.rotation * model.body_com.host().unwrap()[body];
                let velocity = state.body_qd.host().unwrap()[body];
                force += wrench.linear;
                torque += wrench.angular + center.cross(wrench.linear);
                work += wrench.linear.dot(velocity.linear) + wrench.angular.dot(velocity.angular);
            }
            assert!(force.length() < 1e-9);
            assert!(torque.length() < 1e-9, "{torque:?}");
            let positions = [0.3, -0.1, 0.2, angles[0], angles[1], angles[2]];
            let expected_work: f64 = (0..6)
                .map(|axis| {
                    let rate = state.joint_qd.host().unwrap()[d + axis];
                    effort(
                        model.joint_compliance.host().unwrap()[d + axis],
                        positions[axis],
                        rate,
                        2.0,
                        dt,
                    ) * rate
                })
                .sum();
            assert!(
                (work - expected_work).abs() < 1e-8,
                "{work} != {expected_work}"
            );
        }
    }
}
