//! Constrained acceleration response for isolated free roots with singular inertia.

use super::*;
use molla_sim::MassProperties;

pub(super) fn acceleration(
    model: &Model,
    state: &State,
    articulation: usize,
    joint: usize,
    applied_tau: &[Real; MAX_JOINT_DOFS],
    forces: &[SpatialVector],
    mass: &mut crate::joint_mass::JointMass<'_>,
) -> Result<Option<Vec<Real>>> {
    let kind = joint_type_from_u32(model.joint_type.host()?[joint]);
    let body = model.joint_child.host()?[joint] as usize;
    if !matches!(kind, JointType::Free | JointType::Distance | JointType::D6)
        || model.joint_parent.host()?[joint] >= 0
        || model.joint_enabled.host()?[joint] == 0
        || model.body_flags.host()?[body] & BodyFlags::IMMOVABLE != 0
        || model.body_mass.host()?[body] <= 0.0
    {
        return Ok(None);
    }
    let inertia = model.body_inertia.host()?[body];
    let scale = inertia
        .to_cols_array()
        .into_iter()
        .map(Real::abs)
        .fold(0.0, Real::max);
    if scale > 0.0 && (inertia / scale).determinant() > 1e-12 {
        return Ok(None);
    }
    let props = MassProperties {
        mass: model.body_mass.host()?[body],
        local_com: model.body_com.host()?[body],
        inertia,
    };
    let (moments, axes) = props.principal()?;
    let floor = moments.max_element() * 1e-14;
    if moments.min_element() > floor {
        return Ok(None);
    }
    let jacobian = crate::jacobian(model, state, articulation, body, props.local_com)?;
    let n = jacobian.len() / 6;
    let mut locked = Vec::new();
    for axis in 0..3 {
        if moments[axis] > floor {
            continue;
        }
        let direction = state.body_q.host()?[body].rotation * axes.col(axis);
        locked.push(
            (0..n)
                .map(|col| {
                    direction.x * jacobian[3 * n + col]
                        + direction.y * jacobian[4 * n + col]
                        + direction.z * jacobian[5 * n + col]
                })
                .collect(),
        );
    }
    let bias = crate::inverse_dynamics(model, state, articulation, &vec![0.0; n])?;
    let force = forces[body];
    let wrench = [
        force.linear.x,
        force.linear.y,
        force.linear.z,
        force.angular.x,
        force.angular.y,
        force.angular.z,
    ];
    let rhs: Vec<_> = (0..n)
        .map(|col| {
            applied_tau[col] - bias[col]
                + (0..6)
                    .map(|row| jacobian[row * n + col] * wrench[row])
                    .sum::<Real>()
        })
        .collect();
    crate::rigid_impulse::solve_constrained_mass(mass.matrix(articulation)?, &rhs, n, &locked)
        .map(Some)
}
