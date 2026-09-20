//! Coupled unilateral velocity reactions for scalar joint stops.

use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{JointType, Model, State};

struct Stop {
    dof: usize,
    sign: Real,
    velocity: Real,
    response: Vec<Real>,
}

pub(crate) fn react_scalar_stops(
    model: &Model,
    state: &State,
    articulation: usize,
    acceleration: &mut [Real],
    dt: Real,
    damping: Real,
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
    let factor = joint_mass.factor(articulation, count)?;
    for stop in &mut stops {
        let mut unit = vec![0.0; count];
        unit[stop.dof] = stop.sign;
        stop.response = factor.solve(&unit)?;
    }
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
    let impulses = nonnegative_impulses(&gram, &rhs)?;
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
