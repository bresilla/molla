//! Transactional articulated impulse response for a supplied impact active set.

use crate::{
    articulated_response::ArticulatedResponse,
    impact_constraints::{self, Contact, Domain},
};
use molla_core::{BodyId, Error, Result};
use molla_math::{Mat3, Real, SpatialVector, Vec3};
use molla_sim::{BodyFlags, Model, State, eval_body_velocities};

/// A scalar row of world COM wrenches and prescribed relative velocity.
#[derive(Clone, Debug, Default)]
pub struct ImpactRow {
    pub ends: Vec<(BodyId, SpatialVector)>,
    pub velocity_bias: Real,
    /// Optional generalized velocity coefficient, added to the body-wrench row.
    pub direct: Option<(usize, Real)>,
}

/// A contact frame with normal pointing from the first endpoint to the second.
#[derive(Clone, Copy, Debug)]
pub struct ImpactContact {
    pub bodies: [Option<BodyId>; 2],
    pub point: Vec3,
    pub normal: Vec3,
    pub static_friction: Real,
    pub dynamic_friction: Real,
    pub restitution: Real,
    /// World-space surface velocities; only their tangential components contribute.
    pub surface_velocity: [Vec3; 2],
}

#[derive(Clone, Debug)]
pub enum ImpactConstraint {
    Contact(ImpactContact),
    Bilateral(ImpactRow),
    Nonnegative(ImpactRow),
    /// Scalar feedback with impulse bounds and a nonnegative velocity/impulse diagonal.
    Bounded {
        row: ImpactRow,
        impulses: [Real; 2],
        response_diagonal: Real,
    },
}

/// World contact impulses and work are indexed by constraint block in supplied order.
#[derive(Debug)]
pub struct ImpactResponse {
    /// World-space contact impulse on the second endpoint; `None` for scalar rows.
    pub contact_impulses: Vec<Option<Vec3>>,
    pub solution: impact_constraints::Solution,
    /// Impulse work in the relative row frame, including prescribed surface motion.
    pub relative_work: Vec<Real>,
    /// Work transferred from kinematic boundaries and prescribed row motion.
    pub boundary_work: Vec<Real>,
}

fn contact_rows(
    model: &Model,
    state: &State,
    c: ImpactContact,
) -> Result<([ImpactRow; 3], [Vec3; 3])> {
    if !c.point.is_finite()
        || !c.normal.is_finite()
        || (c.normal.length_squared() - 1.).abs() > 1e-10
        || c.surface_velocity.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Solver("invalid impact contact frame".into()));
    }
    let axis = if c.normal.x.abs() < 0.6 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = c.normal.cross(axis).normalize();
    let directions = [c.normal, tangent, c.normal.cross(tangent)];
    let mut rows: [ImpactRow; 3] = std::array::from_fn(|_| ImpactRow::default());
    for side in 0..2 {
        let sign = if side == 0 { -1. } else { 1. };
        if let Some(body) = c.bodies[side] {
            let i = body.index();
            if i >= model.body_count {
                return Err(Error::Solver("invalid impact body".into()));
            }
            let pose = state.body_q.host()?[i];
            let arm = c.point - (pose.position + pose.rotation * model.body_com.host()?[i]);
            for r in 0..3 {
                let force = directions[r] * sign;
                rows[r]
                    .ends
                    .push((body, SpatialVector::new(force, arm.cross(force))));
            }
        }
        for r in 1..3 {
            rows[r].velocity_bias += sign * directions[r].dot(c.surface_velocity[side]);
        }
    }
    Ok((rows, directions))
}

/// Apply a simultaneous midpoint-friction impact to reduced and body velocities.
///
/// The caller supplies the approaching/stationary contact active set. Poses,
/// applied forces and controls are unchanged. No contact detection, integration,
/// episode tracking or compliant-energy update is performed. Failure leaves state
/// untouched. Scalar rows share the same articulated response as contact rows.
pub fn apply_impact(
    model: &Model,
    state: &mut State,
    constraints: &[ImpactConstraint],
) -> Result<ImpactResponse> {
    if state.body_q.len() != model.body_count
        || state.body_qd.len() != model.body_count
        || state.joint_qd.len() != model.joint_dof_total
        || state.joint_q.len() != model.joint_coord_total
    {
        return Err(Error::Solver(
            "invalid articulated impact state dimensions".into(),
        ));
    }
    let mut input = state.clone_host()?;
    if input.joint_q.host()?.iter().any(|v| !v.is_finite())
        || input
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver("nonfinite articulated impact input".into()));
    }
    eval_body_velocities(model, &mut input)?;
    if input.joint_qd.host()?.iter().any(|v| !v.is_finite())
        || input
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
        || input
            .body_q
            .host()?
            .iter()
            .any(|q| !q.position.is_finite() || !q.rotation.is_finite())
    {
        return Err(Error::Solver("nonfinite articulated impact state".into()));
    }
    let mut rows = Vec::new();
    let mut frames = Vec::new();
    let mut offsets = vec![0];
    for constraint in constraints {
        match constraint {
            ImpactConstraint::Contact(c) => {
                let (contact, frame) = contact_rows(model, &input, *c)?;
                rows.extend(contact);
                frames.push(Some(frame));
            }
            ImpactConstraint::Bilateral(row)
            | ImpactConstraint::Nonnegative(row)
            | ImpactConstraint::Bounded { row, .. } => {
                rows.push(row.clone());
                frames.push(None);
            }
        }
        offsets.push(rows.len());
    }
    let flags = model.body_flags.host()?;
    let mut free = Vec::with_capacity(rows.len());
    for row in &rows {
        if !row.velocity_bias.is_finite()
            || row
                .direct
                .is_some_and(|(dof, sign)| dof >= model.joint_dof_total || !sign.is_finite())
            || row.ends.iter().any(|(body, w)| {
                body.index() >= model.body_count || !w.linear.is_finite() || !w.angular.is_finite()
            })
        {
            return Err(Error::Solver("invalid articulated impact row".into()));
        }
        let mut velocity = row.velocity_bias;
        if let Some((dof, sign)) = row.direct {
            velocity += sign * input.joint_qd.host()?[dof];
        }
        for &(body, wrench) in &row.ends {
            let i = body.index();
            if flags[i] & BodyFlags::FIXED == 0 {
                let v = input.body_qd.host()?[i];
                velocity += wrench.linear.dot(v.linear) + wrench.angular.dot(v.angular);
            }
        }
        free.push(velocity);
    }
    let mut domains = Vec::new();
    for (b, constraint) in constraints.iter().enumerate() {
        domains.push(match constraint {
            ImpactConstraint::Contact(c) => {
                let first = offsets[b];
                Domain::Contact(Contact {
                    static_friction: c.static_friction,
                    dynamic_friction: c.dynamic_friction,
                    moving: free[first + 1].hypot(free[first + 2]) > 1e-4,
                    base_normal_impulse: 0.,
                    restitution: c.restitution,
                    incoming_normal_velocity: free[first],
                })
            }
            ImpactConstraint::Bilateral(_) => Domain::Bilateral,
            ImpactConstraint::Nonnegative(_) => Domain::Nonnegative,
            ImpactConstraint::Bounded {
                impulses: [lower, upper],
                response_diagonal,
                ..
            } => {
                if !response_diagonal.is_finite() || *response_diagonal < 0.0 {
                    return Err(Error::Solver("invalid scalar feedback diagonal".into()));
                }
                Domain::Interval {
                    lower: *lower,
                    upper: *upper,
                }
            }
        });
    }
    let starts = if model.articulation_count == 0 {
        vec![0, model.joint_count as u32]
    } else {
        model.articulation_start.host()?.to_vec()
    };
    let dofs = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let mut owners = vec![None; model.body_count];
    for art in 0..starts.len() - 1 {
        for &body in &children[starts[art] as usize..starts[art + 1] as usize] {
            if flags[body as usize] & BodyFlags::IMMOVABLE == 0 {
                owners[body as usize] = Some(art);
            }
        }
    }
    let count = rows.len();
    let mut gram = vec![
        0.;
        count.checked_mul(count).ok_or_else(|| Error::Solver(
            "articulated impact capacity overflow".into()
        ))?
    ];
    let mut response = ArticulatedResponse::new(model, &input);
    let mut projections = Vec::new();
    let mut dynamic_velocity = vec![0.; count];
    for art in 0..starts.len() - 1 {
        let first = dofs[starts[art] as usize] as usize;
        let n = dofs[starts[art + 1] as usize] as usize - first;
        if n == 0
            || !rows.iter().any(|row| {
                row.ends
                    .iter()
                    .any(|(body, _)| owners[body.index()] == Some(art))
                    || row
                        .direct
                        .is_some_and(|(dof, _)| (first..first + n).contains(&dof))
            })
        {
            continue;
        }
        let mut projected = vec![vec![0.; n]; count];
        for (r, row) in rows.iter().enumerate() {
            if let Some((dof, sign)) = row.direct
                && (first..first + n).contains(&dof)
            {
                projected[r][dof - first] += sign;
            }
            for &(body, wrench) in &row.ends {
                if owners[body.index()] != Some(art) {
                    continue;
                }
                let j = response.jacobian(art, body.index())?;
                for k in 0..n {
                    projected[r][k] += wrench.linear.x * j[k]
                        + wrench.linear.y * j[n + k]
                        + wrench.linear.z * j[2 * n + k]
                        + wrench.angular.x * j[3 * n + k]
                        + wrench.angular.y * j[4 * n + k]
                        + wrench.angular.z * j[5 * n + k];
                }
            }
        }
        response.add_gram(art, n, &projected, &mut gram)?;
        let mut momentum = vec![0.; n];
        for (body, &owner) in owners.iter().enumerate() {
            if owner != Some(art) {
                continue;
            }
            let velocity = input.body_qd.host()?[body];
            let rotation = Mat3::from_quat(input.body_q.host()?[body].rotation);
            let linear = velocity.linear * model.body_mass.host()?[body];
            let angular = rotation
                * (model.body_inertia.host()?[body] * (rotation.transpose() * velocity.angular));
            let j = response.jacobian(art, body)?;
            for k in 0..n {
                momentum[k] += linear.x * j[k]
                    + linear.y * j[n + k]
                    + linear.z * j[2 * n + k]
                    + angular.x * j[3 * n + k]
                    + angular.y * j[4 * n + k]
                    + angular.z * j[5 * n + k];
            }
        }
        let energetic_rate = response.solve(art, &momentum)?;
        for r in 0..count {
            dynamic_velocity[r] += (0..n)
                .map(|k| projected[r][k] * energetic_rate[k])
                .sum::<Real>();
        }
        projections.push((art, first, projected));
    }
    let feedback_gram;
    let effective = if constraints.iter().any(|c| matches!(c, ImpactConstraint::Bounded { response_diagonal, .. } if *response_diagonal > 0.0)) {
        feedback_gram = {
            let mut matrix = gram.clone();
            for (block,constraint) in constraints.iter().enumerate() {
                if let ImpactConstraint::Bounded { response_diagonal, .. } = constraint {
                    let row = offsets[block];
                    matrix[row*count+row] += response_diagonal;
                }
            }
            matrix
        };
        &feedback_gram
    } else { &gram };
    let solution = impact_constraints::solve_midpoint(effective, &free, &domains)?;
    let mut delta = vec![0.; model.joint_dof_total];
    for (art, first, projected) in projections {
        let rhs: Vec<_> = (0..projected[0].len())
            .map(|k| {
                (0..count)
                    .map(|r| projected[r][k] * solution.impulses[r])
                    .sum()
            })
            .collect();
        for (k, value) in response.solve(art, &rhs)?.into_iter().enumerate() {
            delta[first + k] += value;
        }
    }
    let mut result = ImpactResponse {
        contact_impulses: Vec::new(),
        solution,
        relative_work: Vec::new(),
        boundary_work: Vec::new(),
    };
    for b in 0..constraints.len() {
        let first = offsets[b];
        let mut relative = 0.;
        let mut boundary = 0.;
        for r in first..offsets[b + 1] {
            let impulse = result.solution.impulses[r];
            let jump: Real = (0..count)
                .map(|k| gram[r * count + k] * result.solution.impulses[k])
                .sum();
            relative += impulse * (free[r] + 0.5 * jump);
            boundary -= impulse * (free[r] - dynamic_velocity[r]);
        }
        result.relative_work.push(relative);
        result.boundary_work.push(boundary);
        result.contact_impulses.push(frames[b].map(|frame| {
            (0..3)
                .map(|i| frame[i] * result.solution.impulses[first + i])
                .sum()
        }));
    }
    if result
        .relative_work
        .iter()
        .chain(&result.boundary_work)
        .any(|v| !v.is_finite())
        || result
            .contact_impulses
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err(Error::Solver("nonfinite articulated impact work".into()));
    }
    let mut next = input.clone_host()?;
    for (rate, jump) in next.joint_qd.host_mut()?.iter_mut().zip(delta) {
        *rate += jump;
    }
    eval_body_velocities(model, &mut next)?;
    if next.joint_qd.host()?.iter().any(|v| !v.is_finite())
        || next
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver(
            "nonfinite articulated impact velocity".into(),
        ));
    }
    *state = next;
    Ok(result)
}
