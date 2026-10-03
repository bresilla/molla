//! Acceleration-drive mobility after bilateral joint constraints.

use crate::{
    articulated_response::ArticulatedResponse,
    distance_constraints::{DistanceBound, Row},
};
use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{State, runtime::RigidScene};

pub(crate) fn conditional_mobilities(
    gram: &[Real],
    n: usize,
    locked: &[bool],
) -> Result<Vec<Real>> {
    if n.checked_mul(n) != Some(gram.len())
        || locked.len() != n
        || gram.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Solver("invalid drive mobility matrix".into()));
    }
    let mut matrix = gram.to_vec();
    let initial: Vec<_> = (0..n).map(|i| gram[i * n + i]).collect();
    let mut eliminated = vec![false; n];
    for _ in 0..n {
        let pivot = (0..n)
            .filter(|&i| locked[i] && !eliminated[i] && initial[i] > 0.)
            .max_by(|&a, &b| {
                (matrix[a * n + a] / initial[a]).total_cmp(&(matrix[b * n + b] / initial[b]))
            });
        let Some(pivot) = pivot else {
            break;
        };
        let denominator = matrix[pivot * n + pivot];
        if denominator <= initial[pivot] * 1e-12 {
            break;
        }
        for i in 0..n {
            if i == pivot {
                continue;
            }
            for j in 0..n {
                if j != pivot {
                    matrix[i * n + j] -=
                        matrix[i * n + pivot] * (matrix[pivot * n + j] / denominator);
                }
            }
        }
        for i in 0..n {
            matrix[i * n + pivot] = 0.;
            matrix[pivot * n + i] = 0.;
        }
        eliminated[pivot] = true;
    }
    (0..n)
        .map(|i| {
            let value = matrix[i * n + i];
            if initial[i] < 0. || value < -initial[i] * 1e-10 {
                Err(Error::Solver("indefinite drive mobility matrix".into()))
            } else {
                Ok(if value <= initial[i] * 1e-12 {
                    0.
                } else {
                    value
                })
            }
        })
        .collect()
}

pub(crate) fn scale_feedback(
    scene: &RigidScene,
    input: &State,
    arts: &[usize],
    rows: &mut [Row],
) -> Result<()> {
    if !rows
        .iter()
        .any(|row| row.feedback.is_some_and(|f| f.acceleration))
    {
        return Ok(());
    }
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let count = rows.len();
    let mut gram = vec![0.; count * count];
    let mut response = ArticulatedResponse::new(model, input);
    for &art in arts {
        let begin = starts[art] as usize;
        let end = starts[art + 1] as usize;
        let n = (offsets[end] - offsets[begin]) as usize;
        let first = offsets[begin] as usize;
        let mut projected = vec![vec![0.; n]; count];
        for (r, row) in rows.iter().enumerate() {
            for &(body, wrench) in &row.ends {
                if !children[begin..end].contains(&(body as i32)) {
                    continue;
                }
                let jacobian = response.jacobian(art, body)?;
                for (c, p) in projected[r].iter_mut().enumerate() {
                    *p += jacobian[c] * wrench.linear.x
                        + jacobian[n + c] * wrench.linear.y
                        + jacobian[2 * n + c] * wrench.linear.z
                        + jacobian[3 * n + c] * wrench.angular.x
                        + jacobian[4 * n + c] * wrench.angular.y
                        + jacobian[5 * n + c] * wrench.angular.z;
                }
            }
            if let Some((dof, sign)) = row.direct
                && (first..first + n).contains(&dof)
            {
                projected[r][dof - first] = sign;
            }
        }
        response.add_gram(art, n, &projected, &mut gram)?;
    }
    let locked: Vec<_> = rows
        .iter()
        .map(|row| row.feedback.is_none() && row.bound == DistanceBound::Equality)
        .collect();
    let mobility = conditional_mobilities(&gram, count, &locked)?;
    for (row, mobility) in rows.iter_mut().zip(mobility) {
        if let Some(feedback) = &mut row.feedback
            && feedback.acceleration
        {
            feedback.diagonal *= mobility;
            if mobility == 0. {
                feedback.impulses = [0.; 2];
            }
            feedback.acceleration = false;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn redundant_locks_preserve_conditional_mobility_across_scales() {
        for scale in [1e-12, 1e-6, 1., 1e6, 1e12] {
            let matrix = [2., 0.5, 0.5, 0.5, 1., 1., 0.5, 1., 1.].map(|v| v / scale);
            let mobility = conditional_mobilities(&matrix, 3, &[false, true, true]).unwrap();
            assert!((mobility[0] * scale - 1.75).abs() < 1e-12);
            assert_eq!(&mobility[1..], &[0., 0.]);
            assert_eq!(
                conditional_mobilities(&matrix, 3, &[true, true, true]).unwrap(),
                vec![0.; 3]
            );
        }
    }
    #[test]
    fn empty_and_invalid_mobility_matrices_are_explicit() {
        assert!(conditional_mobilities(&[], 0, &[]).unwrap().is_empty());
        assert!(conditional_mobilities(&[1.], 2, &[true]).is_err());
        assert!(conditional_mobilities(&[f64::NAN], 1, &[false]).is_err());
        assert!(conditional_mobilities(&[-1.], 1, &[false]).is_err());
        assert!(conditional_mobilities(&[1., 2., 2., 1.], 2, &[true, false]).is_err());
    }
}
