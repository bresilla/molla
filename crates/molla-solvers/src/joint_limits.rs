//! Coupled unilateral velocity reactions for scalar joint stops.

use molla_core::{Error, Result};
use molla_math::{Quat, Real, Vec3};
use molla_sim::{JointType, Model, State};

struct Stop {
    dof: usize,
    sign: Real,
    velocity: Real,
    response: Vec<Real>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn react_scalar_stops(
    model: &Model,
    state: &State,
    articulation: usize,
    acceleration: &mut [Real],
    dt: Real,
    damping: Real,
    rigorous_distance: bool,
    joint_mass: &mut crate::joint_mass::JointMass<'_>,
) -> Result<()> {
    let starts = model.articulation_start.host()?;
    let (begin, end) = if model.articulation_count == 0 {
        (0, model.joint_count)
    } else {
        (
            starts[articulation] as usize,
            starts[articulation + 1] as usize,
        )
    };
    let offsets = model.joint_dof_offset.host()?;
    let first = offsets[begin] as usize;
    let count = (offsets[end] - offsets[begin]) as usize;
    let q = state.joint_q.host()?;
    let qd = state.joint_qd.host()?;
    let coords = model.joint_coord_offset.host()?;
    let kinds = model.joint_type.host()?;
    let enabled = model.joint_enabled.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let advance = dt * (1.0 - damping * dt).max(0.0);
    if advance == 0.0 || count == 0 {
        return Ok(());
    }
    if has_mixed_stops(model, begin..end, rigorous_distance)? {
        return react_mixed_stops(
            model,
            state,
            articulation,
            acceleration,
            dt,
            damping,
            joint_mass,
        );
    }
    let mut velocity: Vec<_> = (first..first + count)
        .map(|i| qd[i] + dt * acceleration[i])
        .collect();
    let mut stops = Vec::new();
    for joint in begin..end {
        if enabled[joint] == 0
            || ![JointType::Revolute as u32, JointType::Prismatic as u32].contains(&kinds[joint])
        {
            continue;
        }
        let dof = offsets[joint] as usize;
        if lower[dof].is_finite()
            && upper[dof].is_finite()
            && (upper[dof] - lower[dof]).abs() < 1e-12
        {
            continue;
        }
        for (sign, bound) in [(1.0, lower[dof]), (-1.0, upper[dof])] {
            if bound.is_finite() {
                stops.push(Stop {
                    dof: dof - first,
                    sign,
                    velocity: sign * (bound - q[coords[joint] as usize]) / advance,
                    response: Vec::new(),
                });
            }
        }
    }
    if !stops.iter().any(|s| s.sign * velocity[s.dof] < s.velocity) {
        return Ok(());
    }
    let _span = tracing::debug_span!("scalar_stops", stops = stops.len()).entered();
    let response_span = tracing::debug_span!("scalar_stop_responses").entered();
    let factor = joint_mass.factor(articulation, count)?;
    let units: Vec<_> = stops
        .iter()
        .map(|stop| {
            let mut unit = vec![0.0; count];
            unit[stop.dof] = stop.sign;
            unit
        })
        .collect();
    for (stop, response) in stops.iter_mut().zip(factor.solve_batch(&units)?) {
        stop.response = response;
    }
    drop(response_span);
    let gram_span = tracing::debug_span!("scalar_stop_gram").entered();
    stops.retain(|s| s.sign * s.response[s.dof] > 0.0);
    let n = stops.len();
    let mut gram = vec![0.0; n * n];
    let rhs: Vec<_> = stops
        .iter()
        .map(|s| s.velocity - s.sign * velocity[s.dof])
        .collect();
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = 0.5
                * (stops[i].sign * stops[j].response[stops[i].dof]
                    + stops[j].sign * stops[i].response[stops[j].dof]);
        }
    }
    drop(gram_span);
    let impulses = tracing::debug_span!("scalar_stop_impulses")
        .in_scope(|| nonnegative_impulses(&gram, &rhs))?;
    for (stop, impulse) in stops.iter().zip(impulses) {
        for (v, response) in velocity.iter_mut().zip(&stop.response) {
            *v += impulse * response;
        }
    }
    if velocity.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver("non-finite joint limit reaction".into()));
    }
    for (i, v) in velocity.into_iter().enumerate() {
        acceleration[first + i] = (v - qd[first + i]) / dt;
    }
    Ok(())
}

pub(crate) fn has_mixed_stops(
    model: &Model,
    joints: std::ops::Range<usize>,
    rigorous_distance: bool,
) -> Result<bool> {
    let kinds = model.joint_type.host()?;
    let enabled = model.joint_enabled.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let caps = model.joint_velocity_limit.host()?;
    Ok(joints.into_iter().any(|j| {
        let first = offsets[j] as usize;
        enabled[j] != 0
            && ((kinds[j] == JointType::Ball as u32
                && molla_sim::joint_control::ball_coupled_stops(
                    &lower[first..first + 3],
                    &upper[first..first + 3],
                    &caps[first..first + 3],
                ))
                || (kinds[j] == JointType::D6 as u32
                    && molla_sim::joint_control::d6_coupled_stops(
                        &lower[first..first + 6],
                        &upper[first..first + 6],
                        &caps[first..first + 6],
                    ))
                || (rigorous_distance
                    && kinds[j] == JointType::Distance as u32
                    && molla_sim::joint_control::distance_coupled_stops(
                        &lower[first..first + 6],
                        &upper[first..first + 6],
                        &caps[first..first + 6],
                    )))
    }))
}

#[allow(clippy::too_many_arguments)]
fn react_mixed_stops(
    model: &Model,
    state: &State,
    articulation: usize,
    acceleration: &mut [Real],
    dt: Real,
    damping: Real,
    mass: &mut crate::joint_mass::JointMass<'_>,
) -> Result<()> {
    for _ in 0..16 {
        if react_mixed_stops_iteration(model, state, articulation, acceleration, dt, damping, mass)?
        {
            return Ok(());
        }
    }
    Err(Error::Solver(
        "nonlinear mixed joint limit prediction did not converge".into(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn react_mixed_stops_iteration(
    model: &Model,
    state: &State,
    articulation: usize,
    acceleration: &mut [Real],
    dt: Real,
    damping: Real,
    mass: &mut crate::joint_mass::JointMass<'_>,
) -> Result<bool> {
    use crate::coupled_constraints::Domain;
    let starts = model.articulation_start.host()?;
    let (begin, end) = if model.articulation_count == 0 {
        (0, model.joint_count)
    } else {
        (
            starts[articulation] as usize,
            starts[articulation + 1] as usize,
        )
    };
    let offsets = model.joint_dof_offset.host()?;
    let first = offsets[begin] as usize;
    let count = offsets[end] as usize - first;
    let q = state.joint_q.host()?;
    let qd = state.joint_qd.host()?;
    let coords = model.joint_coord_offset.host()?;
    let kinds = model.joint_type.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let caps = model.joint_velocity_limit.host()?;
    let enabled = model.joint_enabled.host()?;
    let flags = model.body_flags.host()?;
    let children = model.joint_child.host()?;
    let damp = (1. - damping * dt).max(0.);
    let advance = dt * damp;
    if advance == 0. || count == 0 {
        return Ok(true);
    }
    let predicted: Vec<_> = (first..first + count)
        .map(|i| qd[i] + dt * acceleration[i])
        .collect();
    let mut units = Vec::new();
    let mut targets = Vec::new();
    let mut domains = Vec::new();
    let mut add = |start: usize, gradient: &[Real], target: Real, bilateral: bool| {
        let mut unit = vec![0.; count];
        unit[start - first..start - first + gradient.len()].copy_from_slice(gradient);
        units.push(unit);
        targets.push(target);
        domains.push(if bilateral {
            Domain::Bilateral
        } else {
            Domain::Nonnegative
        });
    };
    for j in begin..end {
        if enabled[j] == 0 || flags[children[j] as usize] & molla_sim::BodyFlags::IMMOVABLE != 0 {
            continue;
        }
        let start = offsets[j] as usize;
        let c = coords[j] as usize;
        let locked = |d: usize| {
            lower[start + d].is_finite()
                && upper[start + d].is_finite()
                && (upper[start + d] - lower[start + d]).abs() < 1e-12
        };
        let scalar_axes = if kinds[j] == JointType::D6 as u32 {
            3
        } else if kinds[j] <= 1 {
            1
        } else {
            0
        };
        for d in 0..scalar_axes {
            if locked(d) {
                continue;
            }
            for (sign, bound) in [(1., lower[start + d]), (-1., upper[start + d])] {
                if bound.is_finite() {
                    add(
                        start + d,
                        &[sign],
                        sign * (bound - q[c + d]) / advance,
                        false,
                    );
                }
            }
        }
        if kinds[j] == JointType::D6 as u32 || kinds[j] == JointType::Ball as u32 {
            let angular = if kinds[j] == JointType::Ball as u32 {
                0
            } else {
                3
            };
            let initial = Quat::from_xyzw(
                q[c + angular],
                q[c + angular + 1],
                q[c + angular + 2],
                q[c + angular + 3],
            );
            let rate =
                Vec3::from_slice(&predicted[start - first + angular..start - first + angular + 3]);
            let half_delta = rate * (0.5 * advance);
            let nonlinear = !(angular..angular + 3).any(locked);
            let rotation = if nonlinear {
                Quat::from_xyzw(half_delta.x, half_delta.y, half_delta.z, 1.).normalize() * initial
            } else {
                initial
            };
            for row in crate::joint_angles::stops(
                rotation,
                Vec3::from_slice(&lower[start + angular..start + angular + 3]),
                Vec3::from_slice(&upper[start + angular..start + angular + 3]),
            )? {
                let gradient = if nonlinear {
                    (row.gradient - half_delta.cross(row.gradient))
                        / (1. + half_delta.length_squared())
                } else {
                    row.gradient
                };
                let shift = if nonlinear { gradient.dot(rate) } else { 0. };
                add(
                    start + angular,
                    &gradient.to_array(),
                    shift - row.error / advance,
                    row.bilateral,
                );
            }
        }
        if kinds[j] == JointType::Distance as u32 {
            let position = Vec3::from_slice(&q[c..c + 3]);
            let length = position.length();
            if upper[start] == 0. {
                for d in 0..3 {
                    add(start + d, &[1.], -position[d] / advance, true);
                }
            } else {
                let direction = if length > 1e-12 {
                    position / length
                } else {
                    Vec3::X
                };
                let equality = locked(0);
                for (sign, bound) in [(1., lower[start]), (-1., upper[start])] {
                    if bound.is_finite() && !(equality && sign < 0.) {
                        add(
                            start,
                            &(direction * sign).to_array(),
                            sign * (bound - length) / advance,
                            equality,
                        );
                    }
                }
            }
        }
        for d in 0..offsets[j + 1] as usize - start {
            let structural = (kinds[j] <= 1 && locked(d))
                || (kinds[j] == JointType::Ball as u32 && (0..3).all(locked))
                || (kinds[j] == JointType::D6 as u32 && locked(d) && (d < 3 || (3..6).all(locked)));
            let cap = caps[start + d];
            if structural || !cap.is_finite() {
                continue;
            }
            add(start + d, &[1.], -cap / damp, cap == 0.);
            if cap != 0. {
                add(start + d, &[-1.], -cap / damp, false);
            }
        }
    }
    if units.is_empty() {
        return Ok(true);
    }
    let free: Vec<_> = units
        .iter()
        .zip(&targets)
        .map(|(unit, target)| {
            unit.iter()
                .zip(&predicted)
                .map(|(a, b)| a * b)
                .sum::<Real>()
                - target
        })
        .collect();
    if free
        .iter()
        .zip(&targets)
        .zip(&domains)
        .all(|((&value, &target), domain)| {
            let residual = if matches!(domain, Domain::Bilateral) {
                value.abs()
            } else {
                (-value).max(0.)
            };
            value.is_finite()
                && target.is_finite()
                && residual <= 1e-10 * (1. + target.abs() + (value + target).abs())
        })
    {
        for j in begin..end {
            if enabled[j] == 0 || flags[children[j] as usize] & molla_sim::BodyFlags::IMMOVABLE != 0
            {
                continue;
            }
            for d in offsets[j] as usize..offsets[j + 1] as usize {
                if caps[d] == 0. {
                    acceleration[d] = -qd[d] / dt;
                }
            }
        }
        return Ok(true);
    }
    let factor = mass.factor(articulation, count)?;
    let responses = factor.solve_batch(&units)?;
    let n = units.len();
    let mut gram = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot = |a: &[Real], b: &[Real]| a.iter().zip(b).map(|(a, b)| a * b).sum::<Real>();
            let value = 0.5 * (dot(&units[i], &responses[j]) + dot(&units[j], &responses[i]));
            gram[i * n + j] = value;
            gram[j * n + i] = value;
        }
    }
    let impulses = match crate::coupled_constraints::solve(&gram, &free, &domains) {
        Ok(solution) => solution.impulses,
        Err(Error::Solver(message))
            if message.starts_with("common joint/contact response did not converge:") =>
        {
            projected_stop_impulses(&gram, &free, &domains, &targets)?
        }
        Err(error) => return Err(error),
    };
    for (response, impulse) in responses.iter().zip(impulses) {
        for (d, value) in response.iter().enumerate() {
            acceleration[first + d] += value * impulse / dt;
        }
    }
    for j in begin..end {
        if enabled[j] == 0 || flags[children[j] as usize] & molla_sim::BodyFlags::IMMOVABLE != 0 {
            continue;
        }
        for d in offsets[j] as usize..offsets[j + 1] as usize {
            if caps[d] != 0. {
                continue;
            }
            let rate = qd[d] + dt * acceleration[d];
            let roundoff = 64. * Real::EPSILON * (1. + qd[d].abs() + predicted[d - first].abs());
            if !rate.is_finite() || rate.abs() > 1e-10 + roundoff {
                return Err(Error::Solver(
                    "zero joint cap reaction left an infeasible rate".into(),
                ));
            }
            acceleration[d] = -qd[d] / dt;
        }
    }
    if acceleration[first..first + count]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(Error::Solver(
            "non-finite mixed joint limit reaction".into(),
        ));
    }
    Ok(false)
}

fn projected_stop_impulses(
    gram: &[Real],
    free: &[Real],
    domains: &[crate::coupled_constraints::Domain],
    targets: &[Real],
) -> Result<Vec<Real>> {
    use crate::coupled_constraints::Domain;
    let n = free.len();
    let mut impulses = vec![0.; n];
    if (0..n).any(|r| {
        gram[r * n + r] <= 0. || !matches!(domains[r], Domain::Bilateral | Domain::Nonnegative)
    }) {
        return Err(Error::Solver("invalid projected joint stop problem".into()));
    }
    for _ in 0..32768 {
        for r in 0..n {
            let value = free[r]
                + (0..n)
                    .filter(|&c| c != r)
                    .map(|c| gram[r * n + c] * impulses[c])
                    .sum::<Real>();
            impulses[r] = -value / gram[r * n + r];
            if matches!(domains[r], Domain::Nonnegative) {
                impulses[r] = impulses[r].max(0.);
            }
        }
        let converged = (0..n).all(|r| {
            let value = free[r] + (0..n).map(|c| gram[r * n + c] * impulses[c]).sum::<Real>();
            let residual = if matches!(domains[r], Domain::Nonnegative) && impulses[r] == 0. {
                (-value).max(0.)
            } else {
                value.abs()
            };
            let tolerance = 1e-10 * (1. + free[r].abs());
            value.is_finite() && impulses[r].is_finite() && residual <= tolerance
        });
        if converged {
            return Ok(impulses);
        }
    }
    let (row, residual) = (0..n)
        .map(|r| {
            let value = free[r] + (0..n).map(|c| gram[r * n + c] * impulses[c]).sum::<Real>();
            let residual = if matches!(domains[r], Domain::Nonnegative) && impulses[r] == 0. {
                (-value).max(0.)
            } else {
                value.abs()
            };
            (r, residual)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.));
    Err(Error::Solver(format!(
        "projected joint stop response did not converge: row={row} residual={residual:e} target={:e} free={:e}",
        targets[row], free[row]
    )))
}

fn nonnegative_impulses(gram: &[Real], rhs: &[Real]) -> Result<Vec<Real>> {
    let n = rhs.len();
    let mut impulses = vec![0.0; n];
    let mut active = Vec::new();
    for _ in 0..4 * n * n + 1 {
        let entering = (0..n)
            .filter(|i| !active.contains(i))
            .map(|i| {
                (
                    i,
                    rhs[i] - (0..n).map(|j| gram[i * n + j] * impulses[j]).sum::<Real>(),
                )
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((entering, residual)) = entering else {
            return checked_impulses(gram, rhs, impulses);
        };
        if residual <= 1e-10 {
            return checked_impulses(gram, rhs, impulses);
        }
        active.push(entering);
        for _ in 0..n + 1 {
            let k = active.len();
            let mut matrix = vec![0.0; k * k];
            let mut target = vec![0.0; k];
            for (i, &row) in active.iter().enumerate() {
                target[i] = rhs[row];
                for (j, &col) in active.iter().enumerate() {
                    matrix[i * k + j] = gram[row * n + col];
                }
            }
            let solved = crate::rigid_impulse::MassFactor::new(&matrix, k)?.solve(&target)?;
            let mut trial = vec![0.0; n];
            for (&i, &value) in active.iter().zip(&solved) {
                trial[i] = value;
            }
            if active.iter().all(|&i| trial[i] > 0.0) {
                impulses = trial;
                break;
            }
            let alpha = active
                .iter()
                .filter(|&&i| trial[i] <= 0.0)
                .map(|&i| {
                    if impulses[i] == 0.0 {
                        0.0
                    } else {
                        impulses[i] / (impulses[i] - trial[i])
                    }
                })
                .fold(1.0_f64, Real::min);
            for i in 0..n {
                impulses[i] += alpha * (trial[i] - impulses[i]);
            }
            active.retain(|&i| impulses[i] > 0.0);
        }
    }
    Err(Error::Solver(
        "joint limit active set did not converge".into(),
    ))
}

fn checked_impulses(gram: &[Real], rhs: &[Real], impulses: Vec<Real>) -> Result<Vec<Real>> {
    let n = rhs.len();
    for i in 0..n {
        let residual = rhs[i] - (0..n).map(|j| gram[i * n + j] * impulses[j]).sum::<Real>();
        let tolerance = 1e-8 * (1.0 + rhs[i].abs());
        if !impulses[i].is_finite()
            || impulses[i] < 0.0
            || !residual.is_finite()
            || residual > tolerance
            || (impulses[i] > 0.0 && residual.abs() > tolerance)
        {
            return Err(Error::Solver("invalid joint limit complementarity".into()));
        }
    }
    Ok(impulses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_set_releases_a_stop_coupled_to_a_stiffer_stop() {
        let result = nonnegative_impulses(&[0.01, 0.09, 0.09, 1.0], &[1.0, 2.0]).unwrap();
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert_eq!(result[1], 0.0);
        assert_eq!(nonnegative_impulses(&[1.0], &[-1.0]).unwrap(), [0.0]);
    }
}
