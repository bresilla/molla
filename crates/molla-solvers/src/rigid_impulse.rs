//! Instantaneous reduced-coordinate impulse response.

use molla_core::{Error, Result};
use molla_math::{
    Real, SpatialVector,
    dense_solve::{back_sub, dense_chol, dense_eig_sym, forward_sub},
};
use molla_sim::{MassProperties, Model, State, eval_body_velocities};

pub(crate) mod blocks;
mod compact;
mod impact;
mod response_batch;
pub use impact::{ImpactConstraint, ImpactContact, ImpactResponse, ImpactRow, apply_impact};

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
    let matrix = crate::response_mass_matrix(model, state, articulation)?;
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

pub(crate) fn solve_constrained_mass(
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

/// Certify a normalized Cholesky spectrum above the relative null-mode cutoff.
fn cholesky_resolves_modes(factor: &[Real], n: usize) -> bool {
    let mut rows = vec![0.; n];
    let mut columns = vec![0.; n];
    for i in 0..n {
        let mut bound = 1.;
        for j in 0..i {
            bound += factor[i * n + j].abs() * rows[j];
        }
        rows[i] = bound / factor[i * n + i];
    }
    for i in (0..n).rev() {
        let mut bound = 1.;
        for j in i + 1..n {
            bound += factor[j * n + i].abs() * columns[j];
        }
        columns[i] = bound / factor[i * n + i];
    }
    let inverse_bound =
        rows.into_iter().fold(0., Real::max) * columns.into_iter().fold(0., Real::max);
    inverse_bound.is_finite() && inverse_bound * n as Real * 1e-12 < 1.
}

impl MassFactor {
    pub(crate) fn full_rank_cholesky(&self, n: usize) -> Option<(Real, &[Real])> {
        let MassDecomposition::Cholesky(factor) = &self.decomposition else {
            return None;
        };
        (self.n == n
            && self.scale.is_finite()
            && self.scale > 0.0
            && factor.iter().all(|v| v.is_finite())
            && (0..n).all(|i| factor[i * n + i] > 0.0 && factor[i * n + i].powi(2) > 1e-12))
        .then_some((self.scale, factor.as_slice()))
    }

    pub(crate) fn into_full_rank(self) -> Option<Self> {
        self.full_rank_cholesky(self.n).is_some().then_some(self)
    }

    /// Retain a normalized full-rank Cholesky with the impulse solver's pivot cutoff.
    pub(crate) fn from_cholesky(factor: Vec<Real>, n: usize, scale: Real) -> Option<Self> {
        if n.checked_mul(n) != Some(factor.len())
            || !scale.is_finite()
            || scale <= 0.0
            || factor.iter().any(|v| !v.is_finite())
            || (0..n).any(|i| factor[i * n + i] <= 0.0 || factor[i * n + i].powi(2) <= 1e-12)
            || !cholesky_resolves_modes(&factor, n)
        {
            return None;
        }
        Some(Self {
            n,
            scale,
            decomposition: MassDecomposition::Cholesky(factor),
        })
    }

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
                && cholesky_resolves_modes(&factor, n)
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

    /// Quadratic response `rhsᵀ M⁺ rhs` using the cached mass decomposition.
    pub(crate) fn quadratic(&self, rhs: &[Real]) -> Result<Real> {
        if rhs.len() != self.n || rhs.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver("invalid impulse mass right-hand side".into()));
        }
        if matches!(self.decomposition, MassDecomposition::Zero) {
            return Ok(0.0);
        }
        let root_scale = self.scale.sqrt();
        let scaled: Vec<Real> = rhs.iter().map(|v| v / root_scale).collect();
        let response: Real = match &self.decomposition {
            MassDecomposition::Zero => unreachable!(),
            MassDecomposition::Cholesky(factor) => forward_sub(factor, &scaled, self.n)?
                .iter()
                .map(|v| v * v)
                .sum(),
            MassDecomposition::Spectral {
                values,
                vectors,
                floor,
            } => values
                .iter()
                .enumerate()
                .filter(|(_, value)| **value > *floor)
                .map(|(col, value)| {
                    let weight: Real = (0..self.n)
                        .map(|row| vectors[row * self.n + col] * scaled[row])
                        .sum();
                    (weight / value.sqrt()).powi(2)
                })
                .sum(),
        };
        if !response.is_finite() {
            return Err(Error::Solver("non-finite articulated response".into()));
        }
        Ok(response)
    }

    /// Accumulate `J M⁺ Jᵀ` into a row-major response matrix.
    pub(crate) fn add_gram(&self, projected: &[Vec<Real>], response: &mut [Real]) -> Result<()> {
        let count = projected.len();
        if count.checked_mul(count) != Some(response.len())
            || projected
                .iter()
                .any(|row| row.len() != self.n || row.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Solver("invalid projected mass response".into()));
        }
        let factored = matches!(self.decomposition, MassDecomposition::Cholesky(_));
        let _span = tracing::debug_span!("loop_mass_rhs", cholesky = factored).entered();
        if let MassDecomposition::Cholesky(factor) = &self.decomposition {
            let whiten_span = tracing::debug_span!("loop_mass_whiten").entered();
            let whitened = if self.n <= 6 || count < 4 {
                projected
                    .iter()
                    .map(|row| forward_sub(factor, row, self.n))
                    .collect::<Result<Vec<_>>>()?
            } else {
                whiten_rows(factor, projected, self.n)?
            };
            drop(whiten_span);
            let _gram_span = tracing::debug_span!("loop_mass_gram").entered();
            for row in 0..count {
                for col in 0..=row {
                    let value = whitened[row]
                        .iter()
                        .zip(&whitened[col])
                        .map(|(a, b)| a * b)
                        .sum::<Real>()
                        / self.scale;
                    response[row * count + col] += value;
                    if row != col {
                        response[col * count + row] += value;
                    }
                }
            }
        } else {
            for col in 0..count {
                let solved = self.solve(&projected[col])?;
                for row in 0..count {
                    response[row * count + col] += projected[row]
                        .iter()
                        .zip(&solved)
                        .map(|(a, b)| a * b)
                        .sum::<Real>();
                }
            }
        }
        Ok(())
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

fn whiten_rows(factor: &[Real], rows: &[Vec<Real>], n: usize) -> Result<Vec<Vec<Real>>> {
    const LANES: usize = 4;
    let full_count = rows.len() / LANES * LANES;
    let mut result = Vec::with_capacity(rows.len());
    if full_count != 0 {
        result.extend((0..full_count).map(|_| vec![0.0; n]));
        let mut work = vec![[0.0; LANES]; n];
        for start in (0..full_count).step_by(LANES) {
            for i in 0..n {
                let mut values: [Real; LANES] = std::array::from_fn(|lane| rows[start + lane][i]);
                let coefficients = &factor[i * n..(i + 1) * n];
                for (&coefficient, previous) in coefficients[..i].iter().zip(&work[..i]) {
                    for lane in 0..LANES {
                        values[lane] -= coefficient * previous[lane];
                    }
                }
                let diagonal = coefficients[i];
                if diagonal.abs() < 1e-30 {
                    return Err(Error::Build(format!(
                        "forward_sub: zero diagonal at row {i}"
                    )));
                }
                for lane in 0..LANES {
                    values[lane] /= diagonal;
                    result[start + lane][i] = values[lane];
                }
                work[i] = values;
            }
        }
    }
    for row in &rows[full_count..] {
        result.push(forward_sub(factor, row, n)?);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "gram_batch_tests.rs"]
mod gram_batch_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_null_modes_do_not_become_cholesky_full_rank() {
        use molla_math::{Mat3, Quat, Vec3};
        for tilt in [0., 1e-5, 1e-4, 1e-3, 0.2] {
            let rotation =
                Mat3::from_quat(Quat::from_rotation_z(0.4) * Quat::from_rotation_y(tilt));
            let inertia =
                rotation * Mat3::from_diagonal(Vec3::new(0., 0.2, 0.2)) * rotation.transpose();
            let mut cholesky = (inertia / 0.2).to_cols_array().to_vec();
            if dense_chol(&mut cholesky, 3).is_ok() {
                assert!(MassFactor::from_cholesky(cholesky, 3, 0.2).is_none());
            }
            for scale in [1e-100, 1., 1e100] {
                let factor = MassFactor::new(&(inertia * scale).to_cols_array(), 3).unwrap();
                assert!(
                    factor.full_rank_cholesky(3).is_none(),
                    "tilt={tilt} scale={scale}"
                );
                for local in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(7., 3., -4.)] {
                    let rhs = rotation * local;
                    let solved = factor.solve(&rhs.to_array()).unwrap();
                    let actual = Vec3::from_slice(&solved) * scale;
                    let expected = rotation * Vec3::new(0., local.y / 0.2, local.z / 0.2);
                    assert!(
                        (actual - expected).length() < 1e-10,
                        "tilt={tilt} scale={scale} actual={actual:?} expected={expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn retained_cholesky_matches_fresh_factors_and_preserves_cutoff() {
        for scale in [1e-100, 1.0, 1e100] {
            let matrix = [4.0, 1.0, 1.0, 3.0].map(|v| v * scale);
            let normalizer = 4.0 * scale;
            let mut cholesky = matrix.map(|v| v / normalizer).to_vec();
            dense_chol(&mut cholesky, 2).unwrap();
            let retained = MassFactor::from_cholesky(cholesky, 2, normalizer).unwrap();
            let fresh = MassFactor::new(&matrix, 2).unwrap();
            for rhs in [[1.0, 0.0], [0.0, 1.0], [-3.0, 7.0]] {
                assert_eq!(retained.solve(&rhs).unwrap(), fresh.solve(&rhs).unwrap());
                assert_eq!(
                    retained.quadratic(&rhs).unwrap(),
                    fresh.quadratic(&rhs).unwrap()
                );
            }
        }
        for pivot in [0.0, -1.0, 1e-7, 1e-6, f64::NAN, f64::INFINITY] {
            assert!(MassFactor::from_cholesky(vec![1.0, 0.0, 0.0, pivot], 2, 1.0).is_none());
        }
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MassFactor::from_cholesky(vec![1.0], 1, scale).is_none());
        }
        assert!(MassFactor::from_cholesky(vec![1.0], 2, 1.0).is_none());
    }

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
                    let expected_quadratic: Real =
                        rhs.iter().zip(expected).map(|(a, b)| a * b).sum();
                    let quadratic = factor.quadratic(&rhs).unwrap();
                    assert!((quadratic * scale - expected_quadratic).abs() < 1e-11);
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
        assert_eq!(factor.quadratic(&[3.0, 4.0]).unwrap(), 9.0);
        let zero = MassFactor::new(&[0.0; 4], 2).unwrap();
        assert_eq!(zero.solve(&[3.0, 4.0]).unwrap(), vec![0.0; 2]);
        assert_eq!(zero.quadratic(&[3.0, 4.0]).unwrap(), 0.0);
        assert!(zero.quadratic(&[f64::NAN, 0.0]).is_err());
        assert!(zero.quadratic(&[0.0]).is_err());
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

    #[test]
    fn quadratic_response_scales_before_squaring() {
        for scale in [1e-300, 1e-200, 1.0, 1e200, 1e300] {
            let factor = MassFactor::new(&[scale], 1).unwrap();
            let actual = factor.quadratic(&[scale]).unwrap();
            assert!((actual / scale - 1.0).abs() < 1e-14);
            assert_eq!(factor.quadratic(&[0.0]).unwrap(), 0.0);
        }
        let factor = MassFactor::new(&[1.0], 1).unwrap();
        assert!(factor.quadratic(&[Real::MAX]).is_err());
    }

    #[test]
    fn gram_response_checks_dimensions_and_zero_mass() {
        for matrix in [[0.0; 4], [2.0, 0.0, 0.0, 3.0]] {
            let factor = MassFactor::new(&matrix, 2).unwrap();
            assert!(factor.add_gram(&[vec![1.0]], &mut [0.0]).is_err());
            assert!(factor.add_gram(&[vec![1.0, f64::NAN]], &mut [0.0]).is_err());
            assert!(factor.add_gram(&[vec![1.0, 2.0]], &mut []).is_err());
            factor.add_gram(&[], &mut []).unwrap();
            let mut response = [7.0; 4];
            factor
                .add_gram(&[vec![0.0; 2], vec![0.0; 2]], &mut response)
                .unwrap();
            assert_eq!(response, [7.0; 4]);
        }
    }

    #[test]
    #[ignore = "wall-clock CPU benchmark; run in release without concurrent builds"]
    fn quadratic_response_timings() {
        let n = 39;
        let rhs: Vec<Real> = (0..n).map(|i| (i % 7) as Real - 3.0).collect();
        for singular in [false, true] {
            let mut mass = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    mass[i * n + j] = if i == j { 2.0 + i as Real } else { 0.1 };
                    if singular && (i == n - 1 || j == n - 1) {
                        mass[i * n + j] = 0.0;
                    }
                }
            }
            let factor = MassFactor::new(&mass, n).unwrap();
            for batch in 0..5 {
                for gram in if batch % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = std::time::Instant::now();
                    for _ in 0..20_000 {
                        let rhs = std::hint::black_box(&rhs);
                        let value = if gram {
                            factor.quadratic(rhs).unwrap()
                        } else {
                            factor
                                .solve(rhs)
                                .unwrap()
                                .iter()
                                .zip(rhs)
                                .map(|(a, b)| a * b)
                                .sum()
                        };
                        std::hint::black_box(value);
                    }
                    eprintln!(
                        "39-DOF quadratic singular={singular} gram={gram} batch {batch}: {:.3} us/call",
                        start.elapsed().as_secs_f64() * 1e6 / 20_000.0
                    );
                }
            }
        }
    }
}
