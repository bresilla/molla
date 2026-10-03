//! Common articulated response for runtime contact and scalar joint rows.

use crate::{
    articulated_response::ArticulatedResponse,
    contact_friction,
    coupled_constraints::{self, Domain},
    distance_constraints::{self, ConstraintReadings, DistanceBound},
    penalty_contact::{ContactForce, FrictionRegime},
};
use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::State;
use molla_sim::runtime::RigidScene;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn needed(scene: &RigidScene) -> Result<bool> {
    let model = scene.model();
    if crate::rigid_loop_constraints::joints(scene)
        .any(|(id, joint)| distance_constraints::enabled(scene, id, joint))
        || crate::rigid_loop_constraints::velocity_joints(scene)
            .any(|(id, joint)| distance_constraints::enabled(scene, id, joint))
        || distance_constraints::point_constraints(scene)
            .any(|(id, j, _)| distance_constraints::enabled(scene, id, j))
        || scene
            .cone_joints()
            .any(|(id, j, _)| distance_constraints::enabled(scene, id, j))
    {
        return Ok(true);
    }
    if model.shape_count == 0 && !model.requires_angular_lock_rows()? {
        return Ok(false);
    }
    for joint in 0..model.joint_count {
        if distance_constraints::has_scalar_bounds(model, joint)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn root(parents: &[usize], mut i: usize) -> usize {
    while parents[i] != i {
        i = parents[i];
    }
    i
}

fn join(parents: &mut [usize], a: usize, b: usize) {
    let a = root(parents, a);
    let b = root(parents, b);
    parents[a.max(b)] = a.min(b);
}

fn dof_articulations(
    starts: &[u32],
    offsets: &[u32],
    arts: &[usize],
    dofs: usize,
) -> Vec<Option<usize>> {
    let mut owners = vec![None; dofs];
    for &art in arts {
        let first = offsets[starts[art] as usize] as usize;
        let last = offsets[starts[art + 1] as usize] as usize;
        owners[first..last].fill(Some(art));
    }
    owners
}

fn project(
    response: &mut ArticulatedResponse<'_>,
    owners: &[Option<usize>],
    art: usize,
    ends: &[(usize, SpatialVector)],
    row: &mut [Real],
) -> Result<()> {
    let n = row.len();
    for &(body, wrench) in ends {
        if owners[body] != Some(art) {
            continue;
        }
        let j = response.jacobian(art, body)?;
        for c in 0..n {
            row[c] += j[c] * wrench.linear.x
                + j[n + c] * wrench.linear.y
                + j[2 * n + c] * wrench.linear.z
                + j[3 * n + c] * wrench.angular.x
                + j[4 * n + c] * wrench.angular.y
                + j[5 * n + c] * wrench.angular.z;
        }
    }
    Ok(())
}

pub(crate) fn apply(
    scene: &RigidScene,
    input: &State,
    output: &mut State,
    dt: Real,
    forces: &mut [ContactForce],
    selected: Option<&[bool]>,
) -> Result<ConstraintReadings> {
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let coordinates = model.joint_coord_offset.host()?;
    let kinds = model.joint_type.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let mut angular = Vec::new();
    for art in 0..model.articulation_count {
        if selected.is_some_and(|mask| !mask[art]) {
            continue;
        }
        for joint in starts[art] as usize..starts[art + 1] as usize {
            if model.joint_enabled.host()?[joint] == 0 {
                continue;
            }
            let offset = if kinds[joint] == molla_sim::JointType::Ball as u32 {
                0
            } else if kinds[joint] == molla_sim::JointType::D6 as u32 {
                3
            } else {
                continue;
            };
            let start = offsets[joint] as usize + offset;
            let lo = Vec3::from_slice(&lower[start..start + 3]);
            let hi = Vec3::from_slice(&upper[start..start + 3]);
            if (0..3).any(|axis| lo[axis].is_finite() || hi[axis].is_finite())
                && !(0..3).any(|axis| {
                    lo[axis].is_finite()
                        && hi[axis].is_finite()
                        && (hi[axis] - lo[axis]).abs() < 1e-12
                })
            {
                angular.push((start, coordinates[joint] as usize + offset, lo, hi));
            }
        }
    }
    if angular.is_empty() {
        return apply_iteration(scene, input, output, dt, forces, selected, None);
    }
    let base = output.joint_qd.host()?.to_vec();
    let mut prediction = base.clone();
    let original_forces = forces.to_vec();
    let mut maximum_error = 0.0_f64;
    for _ in 0..16 {
        output.joint_qd.host_mut()?.copy_from_slice(&base);
        forces.clone_from_slice(&original_forces);
        let readings = apply_iteration(
            scene,
            input,
            output,
            dt,
            forces,
            selected,
            Some(&prediction),
        )?;
        prediction.copy_from_slice(output.joint_qd.host()?);
        let q = input.joint_q.host()?;
        let mut feasible = true;
        maximum_error = 0.;
        for &(start, coord, lo, hi) in &angular {
            let half_delta = Vec3::from_slice(&prediction[start..start + 3]) * (0.5 * dt);
            let initial = Quat::from_xyzw(q[coord], q[coord + 1], q[coord + 2], q[coord + 3]);
            let rotation =
                Quat::from_xyzw(half_delta.x, half_delta.y, half_delta.z, 1.).normalize() * initial;
            for row in crate::joint_angles::stops(rotation, lo, hi)? {
                maximum_error = maximum_error.max(-row.error);
                feasible &= row.error.is_finite() && row.error >= -1e-12;
            }
        }
        if feasible {
            return Ok(readings);
        }
    }
    Err(Error::Solver(format!(
        "nonlinear common angular limit prediction did not converge: residual={maximum_error:e}"
    )))
}

fn apply_iteration(
    scene: &RigidScene,
    input: &State,
    output: &mut State,
    dt: Real,
    forces: &mut [ContactForce],
    selected: Option<&[bool]>,
    prediction: Option<&[Real]>,
) -> Result<ConstraintReadings> {
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let (groups, mut readings) = distance_constraints::prepare(scene, input, dt, selected)?;
    let arts: Vec<_> = (0..model.articulation_count)
        .filter(|&a| selected.is_none_or(|mask| mask[a]))
        .collect();
    let mut scalars: Vec<_> = groups.into_values().flatten().collect();
    distance_constraints::scalar_rows_with_prediction(
        model,
        input,
        &arts,
        dt,
        prediction,
        &mut scalars,
    )?;
    let dof_owners = scalars
        .iter()
        .any(|row| row.direct.is_some())
        .then(|| dof_articulations(starts, offsets, &arts, model.joint_dof_total));
    let (owners, contacts) = contact_friction::prepare(model, input, forces, dt)?;
    let contacts: Vec<_> = contacts
        .into_values()
        .flatten()
        .filter(|contact| {
            selected.is_none_or(|mask| {
                contact.ends[0]
                    .iter()
                    .filter_map(|(body, _)| owners[*body])
                    .all(|art| mask[art])
            })
        })
        .collect();
    let contact_arts: Vec<BTreeSet<usize>> = contacts
        .iter()
        .map(|c| {
            c.ends[0]
                .iter()
                .filter_map(|(body, _)| owners[*body])
                .collect()
        })
        .collect();
    let scalar_arts: Vec<BTreeSet<usize>> = scalars
        .iter()
        .map(|row| {
            let mut members: BTreeSet<_> = row
                .ends
                .iter()
                .filter_map(|(body, _)| owners[*body])
                .collect();
            if let Some((dof, _)) = row.direct {
                members.extend(
                    dof_owners
                        .as_ref()
                        .and_then(|owners| owners.get(dof))
                        .copied()
                        .flatten(),
                );
            }
            members
        })
        .collect();
    let mut parents: Vec<_> = (0..model.articulation_count + scalars.len()).collect();
    for members in &contact_arts {
        if let Some(&first) = members.first() {
            for &member in members {
                join(&mut parents, first, member);
            }
        }
    }
    for (i, members) in scalar_arts.iter().enumerate() {
        for &member in members {
            join(&mut parents, model.articulation_count + i, member);
        }
    }
    let mut islands = BTreeMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (i, members) in contact_arts.iter().enumerate() {
        if let Some(&first) = members.first() {
            islands.entry(root(&parents, first)).or_default().0.push(i);
        }
    }
    for i in 0..scalars.len() {
        islands
            .entry(root(&parents, model.articulation_count + i))
            .or_default()
            .1
            .push(i);
    }
    let mut response = ArticulatedResponse::new(model, input);
    let predicted = output.joint_qd.host()?.to_vec();
    let mut rates = predicted.clone();
    for (contact_ids, scalar_ids) in islands.into_values() {
        let scalar_start = 2 * contact_ids.len();
        let count = scalar_start + scalar_ids.len();
        let mut gram = vec![
            0.;
            count.checked_mul(count).ok_or_else(|| Error::Solver(
                "common response capacity overflow".into()
            ))?
        ];
        let mut free = Vec::with_capacity(count);
        let mut domains = Vec::with_capacity(contact_ids.len() + scalar_ids.len());
        let mut members = BTreeSet::new();
        for &id in &contact_ids {
            let contact = &contacts[id];
            let force = &forces[contact.reading];
            free.extend(contact.rates);
            domains.push(Domain::Disk {
                static_cap: force.static_friction * force.normal_force * dt,
                dynamic_cap: force.friction * force.normal_force * dt,
                moving: contact.moving,
            });
            members.extend(&contact_arts[id]);
        }
        for &id in &scalar_ids {
            let scalar = &scalars[id];
            free.push(scalar.rate - scalar.target);
            domains.push(if let Some(feedback) = scalar.feedback {
                Domain::Interval {
                    lower: feedback.impulses[0],
                    upper: feedback.impulses[1],
                }
            } else if scalar.bound == DistanceBound::Equality {
                Domain::Bilateral
            } else {
                Domain::Nonnegative
            });
            members.extend(&scalar_arts[id]);
        }
        let mut projections = Vec::new();
        for &art in &members {
            let first = offsets[starts[art] as usize] as usize;
            let n = offsets[starts[art + 1] as usize] as usize - first;
            if n == 0 {
                continue;
            }
            let mut projected = vec![vec![0.; n]; count];
            for (c, &id) in contact_ids.iter().enumerate() {
                for d in 0..2 {
                    project(
                        &mut response,
                        &owners,
                        art,
                        &contacts[id].ends[d],
                        &mut projected[2 * c + d],
                    )?;
                }
            }
            for (s, &id) in scalar_ids.iter().enumerate() {
                let row = &scalars[id];
                project(
                    &mut response,
                    &owners,
                    art,
                    &row.ends,
                    &mut projected[scalar_start + s],
                )?;
                if let Some((dof, sign)) = row.direct
                    && (first..first + n).contains(&dof)
                {
                    projected[scalar_start + s][dof - first] = sign;
                }
            }
            for r in 0..count {
                for c in 0..n {
                    free[r] += projected[r][c]
                        * (predicted[first + c] - input.joint_qd.host()?[first + c]);
                }
            }
            response.add_gram(art, n, &projected, &mut gram)?;
            projections.push((art, first, projected));
        }
        for (s, &id) in scalar_ids.iter().enumerate() {
            if let Some(feedback) = scalars[id].feedback {
                let i = scalar_start + s;
                gram[i * count + i] += feedback.diagonal;
            }
        }
        let solution = coupled_constraints::solve(&gram, &free, &domains)?;
        for (art, first, projected) in projections {
            let load: Vec<_> = (0..projected[0].len())
                .map(|c| {
                    (0..count)
                        .map(|r| projected[r][c] * solution.impulses[r])
                        .sum()
                })
                .collect();
            for (c, value) in response.solve(art, &load)?.into_iter().enumerate() {
                rates[first + c] += value;
            }
        }
        for (c, &id) in contact_ids.iter().enumerate() {
            let contact = &contacts[id];
            let reading = &mut forces[contact.reading];
            reading.tangent_impulse = contact.directions[0] * solution.impulses[2 * c]
                + contact.directions[1] * solution.impulses[2 * c + 1];
            reading.force_on_b += reading.tangent_impulse / dt;
            reading.friction_regime = if solution.sticking[c] || !solution.dynamic[c] {
                FrictionRegime::Sticking
            } else {
                FrictionRegime::Sliding
            };
            reading.applied_friction = if solution.dynamic[c] {
                reading.friction
            } else {
                reading.static_friction
            };
            reading.acted = reading.force_on_b.length_squared() > 0.;
        }
        for (s, &id) in scalar_ids.iter().enumerate() {
            let row = &scalars[id];
            let impulse = solution.impulses[scalar_start + s];
            if let Some(joint) = row.joint {
                if row.rigid {
                    let reading = readings.rigid.get_mut(&joint).unwrap();
                    let vector = row.direction * (impulse / dt);
                    if row.cone {
                        reading.angular += vector;
                    } else {
                        reading.linear += vector;
                    }
                    continue;
                }
                if row.cone {
                    let reading = readings.cones.get_mut(&joint).unwrap();
                    reading.torque += row.direction * (impulse / dt);
                    reading.active |= impulse != 0. || row.bound == DistanceBound::Equality;
                } else {
                    let reading = readings.distance.get_mut(&joint).unwrap();
                    reading.force += row.direction * (impulse / dt);
                    if impulse != 0. || row.bound == DistanceBound::Equality {
                        reading.active = row.bound;
                    }
                }
            }
        }
    }
    if rates.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver("nonfinite common constraint rate".into()));
    }
    for row in &scalars {
        if row.velocity_only
            && row.bound == DistanceBound::Equality
            && row.target == 0.
            && let Some((dof, sign)) = row.direct
        {
            let rate = rates[dof] * sign;
            let roundoff = 64. * Real::EPSILON * (1. + predicted[dof].abs());
            if !rate.is_finite() || rate.abs() > 1e-11 + roundoff {
                return Err(Error::Solver(
                    "common zero joint cap left an infeasible rate".into(),
                ));
            }
            rates[dof] = 0.;
        }
    }
    output.joint_qd.host_mut()?.copy_from_slice(&rates);
    Ok(readings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_dof_owners_match_selected_articulation_ranges() {
        let starts = [0, 2, 3, 5];
        let offsets = [0, 0, 2, 2, 5, 6];
        for arts in [vec![0, 1, 2], vec![2, 0], vec![2], vec![1], vec![]] {
            let owners = dof_articulations(&starts, &offsets, &arts, 6);
            for dof in 0..8 {
                let expected: BTreeSet<_> = arts
                    .iter()
                    .copied()
                    .filter(|&art| {
                        (offsets[starts[art] as usize] as usize
                            ..offsets[starts[art + 1] as usize] as usize)
                            .contains(&dof)
                    })
                    .collect();
                let actual: BTreeSet<_> = owners.get(dof).copied().flatten().into_iter().collect();
                assert_eq!(actual, expected, "arts={arts:?} dof={dof}");
            }
        }
        assert!(dof_articulations(&[0, 1], &[0, 0], &[0], 0).is_empty());
    }

    #[test]
    fn indexed_dof_owners_keep_many_independent_active_ranges_separate() {
        let starts: Vec<_> = (0..=1025_u32).map(|art| 3 * art).collect();
        let mut offsets = vec![0];
        for joint in 0..3075 {
            offsets.push(offsets[joint] + [0, 1, 3][joint % 3]);
        }
        let arts: Vec<_> = (0..1025).step_by(2).collect();
        let owners = dof_articulations(&starts, &offsets, &arts, 4100);
        for (dof, actual) in owners.into_iter().enumerate() {
            let expected = starts
                .windows(2)
                .enumerate()
                .find(|(_, range)| {
                    (offsets[range[0] as usize] as usize..offsets[range[1] as usize] as usize)
                        .contains(&dof)
                })
                .map(|(art, _)| art)
                .filter(|art| art % 2 == 0);
            assert_eq!(actual, expected, "dof={dof}");
        }
    }
}
