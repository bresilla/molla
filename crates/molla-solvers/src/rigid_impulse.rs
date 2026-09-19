//! Instantaneous reduced-coordinate impulse response.

use molla_core::{Error, Result};
use molla_math::{
    Real, SpatialVector,
    dense_solve::{back_sub, dense_chol, dense_eig_sym, forward_sub},
};
use molla_sim::{MassProperties, Model, State, eval_body_velocities};

/// Apply a world impulse and COM angular impulse through the articulation mass matrix.
pub fn apply_impulse(
    model: &Model,
    state: &mut State,
    body: usize,
    impulse: SpatialVector,
) -> Result<()> {
    if body >= model.body_count || !impulse.linear.is_finite() || !impulse.angular.is_finite() {
        return Err(Error::Solver("invalid body impulse".into()));
    }
    let children = model.joint_child.host()?;
    let joint = children
        .iter()
        .position(|&b| b as usize == body)
        .ok_or_else(|| Error::Solver("impulse body has no articulation".into()))?;
    let starts = model.articulation_start.host()?;
    let articulation = if model.articulation_count == 0 {
        0
    } else {
        starts
            .windows(2)
            .position(|range| range[0] as usize <= joint && joint < range[1] as usize)
            .ok_or_else(|| Error::Solver("missing impulse articulation".into()))?
    };
    let begin = if model.articulation_count == 0 {
        0
    } else {
        starts[articulation] as usize
    };
    let end = if model.articulation_count == 0 {
        model.joint_count
    } else {
        starts[articulation + 1] as usize
    };
    let offsets = model.joint_dof_offset.host()?;
    let first = offsets[begin] as usize;
    let n = (offsets[end] - offsets[begin]) as usize;
    if n == 0 {
        return Ok(());
    }
    let jacobian = crate::jacobian(
        model,
        state,
        articulation,
        body,
        model.body_com.host()?[body],
    )?;
    let wrench = [
        impulse.linear.x,
        impulse.linear.y,
        impulse.linear.z,
        impulse.angular.x,
        impulse.angular.y,
        impulse.angular.z,
    ];
    let rhs: Vec<Real> = (0..n)
        .map(|c| (0..6).map(|r| jacobian[r * n + c] * wrench[r]).sum())
        .collect();
    let matrix = crate::mass_matrix(model, state, articulation)?;
    let mut locked = Vec::new();
    for &child in &children[begin..end] {
        let i = child as usize;
        let props = MassProperties {
            mass: model.body_mass.host()?[i],
            local_com: model.body_com.host()?[i],
            inertia: model.body_inertia.host()?[i],
        };
        if props.mass == 0.0 {
            continue;
        }
        let (moments, axes) = props.principal()?;
        let floor = moments.max_element() * 1e-14;
        if moments.min_element() > floor {
            continue;
        }
        let j = crate::jacobian(model, state, articulation, i, props.local_com)?;
        for axis in 0..3 {
            if moments[axis] > floor {
                continue;
            }
            let direction = state.body_q.host()?[i].rotation * axes.col(axis);
            locked.push(
                (0..n)
                    .map(|c| {
                        direction.x * j[3 * n + c]
                            + direction.y * j[4 * n + c]
                            + direction.z * j[5 * n + c]
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
    let delta = solve_constrained_mass(&matrix, &rhs, n, &locked)?;
    let mut next = state.clone_host()?;
    for (velocity, delta) in next.joint_qd.host_mut()?[first..first + n]
        .iter_mut()
        .zip(delta)
    {
        *velocity += delta;
    }
    eval_body_velocities(model, &mut next)?;
    if next.joint_qd.host()?.iter().any(|v| !v.is_finite())
        || next
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver("body impulse overflow".into()));
    }
    *state = next;
    Ok(())
}

fn solve_constrained_mass(
    matrix: &[Real],
    rhs: &[Real],
    n: usize,
    locked: &[Vec<Real>],
) -> Result<Vec<Real>> {
    if locked.is_empty() {
        return solve_mass(matrix, rhs, n);
    }
    if matrix
        .iter()
        .chain(rhs)
        .chain(locked.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(Error::Solver(
            "non-finite constrained impulse system".into(),
        ));
    }
    let mut gram = vec![0.0; n * n];
    for row in locked {
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] += row[i] * row[j];
            }
        }
    }
    if gram.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver("impulse constraints overflow".into()));
    }
    let scale = gram.iter().copied().fold(0.0, Real::max);
    if scale == 0.0 {
        return solve_mass(matrix, rhs, n);
    }
    for v in &mut gram {
        *v /= scale;
    }
    let (values, basis) = dense_eig_sym(&gram, n)?;
    let columns: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v < 1e-12).then_some(i))
        .collect();
    let size = columns.len();
    let mut projected = vec![0.0; size * size];
    let mut force = vec![0.0; size];
    let mut weighted = vec![0.0; n * size];
    for row in 0..n {
        for (j, &b) in columns.iter().enumerate() {
            weighted[row * size + j] = (0..n)
                .map(|col| matrix[row * n + col] * basis[col * n + b])
                .sum();
        }
    }
    for (i, &a) in columns.iter().enumerate() {
        for row in 0..n {
            force[i] += basis[row * n + a] * rhs[row];
            for j in 0..size {
                projected[i * size + j] += basis[row * n + a] * weighted[row * size + j];
            }
        }
    }
    let solved = solve_mass(&projected, &force, size)?;
    Ok((0..n)
        .map(|row| {
            columns
                .iter()
                .enumerate()
                .map(|(i, &col)| basis[row * n + col] * solved[i])
                .sum()
        })
        .collect())
}

pub(crate) fn solve_mass(matrix: &[Real], rhs: &[Real], n: usize) -> Result<Vec<Real>> {
    MassFactor::new(matrix, n)?.solve(rhs)
}

/// Reusable normalized Cholesky or spectral pseudoinverse of one mass matrix.
pub(crate) struct MassFactor {
    n: usize,
    scale: Real,
    decomposition: MassDecomposition,
}

enum MassDecomposition {
    Zero,
    Cholesky(Vec<Real>),
    Spectral {
        values: Vec<Real>,
        vectors: Vec<Real>,
        floor: Real,
    },
}

impl MassFactor {
    pub(crate) fn new(matrix: &[Real], n: usize) -> Result<Self> {
        if n.checked_mul(n) != Some(matrix.len()) || matrix.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver("invalid impulse mass matrix".into()));
        }
        let scale = matrix.iter().copied().map(Real::abs).fold(0.0, Real::max);
        let decomposition = if scale == 0.0 {
            MassDecomposition::Zero
        } else {
            let normalized: Vec<Real> = matrix.iter().map(|v| v / scale).collect();
            let mut factor = normalized.clone();
            if dense_chol(&mut factor, n).is_ok()
                && (0..n).all(|i| factor[i * n + i].powi(2) > 1e-12)
            {
                MassDecomposition::Cholesky(factor)
            } else {
                let (values, vectors) = dense_eig_sym(&normalized, n)?;
                let floor = values.iter().copied().fold(0.0, Real::max) * 1e-12;
                if values.iter().any(|&value| value < -floor) {
                    return Err(Error::Solver("indefinite impulse mass matrix".into()));
                }
                MassDecomposition::Spectral {
                    values,
                    vectors,
                    floor,
                }
            }
        };
        Ok(Self {
            n,
            scale,
            decomposition,
        })
    }

    pub(crate) fn solve(&self, rhs: &[Real]) -> Result<Vec<Real>> {
        if rhs.len() != self.n || rhs.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver("invalid impulse mass right-hand side".into()));
        }
        let n = self.n;
        if matches!(self.decomposition, MassDecomposition::Zero) {
            return Ok(vec![0.0; n]);
        }
        let rhs: Vec<Real> = rhs.iter().map(|v| v / self.scale).collect();
        match &self.decomposition {
            MassDecomposition::Zero => unreachable!(),
            MassDecomposition::Cholesky(factor) => {
                back_sub(factor, &forward_sub(factor, &rhs, n)?, n)
            }
            MassDecomposition::Spectral {
                values,
                vectors,
                floor,
            } => {
                let mut result = vec![0.0; n];
                for (col, &value) in values.iter().enumerate() {
                    if value <= *floor {
                        continue;
                    }
                    let weight = (0..n)
                        .map(|row| vectors[row * n + col] * rhs[row])
                        .sum::<Real>()
                        / value;
                    for row in 0..n {
                        result[row] += vectors[row * n + col] * weight;
                    }
                }
                Ok(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_factors_match_analytic_full_and_reduced_rank_solutions() {
        for scale in [1e-100, 1.0, 1e100] {
            for singular in [false, true] {
                let matrix = if singular {
                    [2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0]
                } else {
                    [4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0]
                }
                .map(|v| v * scale);
                let factor = MassFactor::new(&matrix, 3).unwrap();
                assert_eq!(
                    matches!(factor.decomposition, MassDecomposition::Spectral { .. }),
                    singular
                );
                for rhs in [[1.0, 0.0, 0.0], [0.0, -3.0, 5.0], [9.0, -7.0, 2.0]] {
                    let expected = if singular {
                        [
                            (rhs[0] + rhs[1]) / 8.0,
                            (rhs[0] + rhs[1]) / 8.0,
                            rhs[2] / 3.0,
                        ]
                    } else {
                        [
                            (3.0 * rhs[0] - rhs[1]) / 11.0,
                            (4.0 * rhs[1] - rhs[0]) / 11.0,
                            rhs[2] / 2.0,
                        ]
                    };
                    let actual = factor.solve(&rhs).unwrap();
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!((actual * scale - expected).abs() < 1e-12);
                    }
                    assert_eq!(actual, solve_mass(&matrix, &rhs, 3).unwrap());
                }
            }
        }
    }

    #[test]
    fn factors_preserve_nullspace_cutoff_and_reject_invalid_systems() {
        let factor = MassFactor::new(&[1.0, 0.0, 0.0, 1e-14], 2).unwrap();
        assert_eq!(factor.solve(&[3.0, 4.0]).unwrap(), vec![3.0, 0.0]);
        let zero = MassFactor::new(&[0.0; 4], 2).unwrap();
        assert_eq!(zero.solve(&[3.0, 4.0]).unwrap(), vec![0.0; 2]);
        assert!(zero.solve(&[f64::NAN, 0.0]).is_err());
        assert!(zero.solve(&[0.0]).is_err());
        assert!(MassFactor::new(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
        assert!(MassFactor::new(&[f64::INFINITY], 1).is_err());
        assert!(MassFactor::new(&[1.0], 2).is_err());
        assert!(MassFactor::new(&[], usize::MAX).is_err());
        assert!(
            MassFactor::new(&[], 0)
                .unwrap()
                .solve(&[])
                .unwrap()
                .is_empty()
        );
    }
}
