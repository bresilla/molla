use super::{MassFactor, Real, Result, dense_chol, solve_mass};

pub(crate) fn solve(matrix: &[Real], rhs: &[Real], n: usize) -> Result<Vec<Real>> {
    match try_solve(matrix, rhs, n)? {
        Some(result) => Ok(result),
        None => solve_mass(matrix, rhs, n),
    }
}

fn try_solve(matrix: &[Real], rhs: &[Real], n: usize) -> Result<Option<Vec<Real>>> {
    if n < 8
        || n.checked_mul(n) != Some(matrix.len())
        || rhs.len() != n
        || matrix.iter().chain(rhs).any(|v| !v.is_finite())
    {
        return Ok(None);
    }
    let scale = matrix.iter().copied().map(Real::abs).fold(0.0, Real::max);
    if scale == 0.0 {
        return Ok(None);
    }
    let groups = components(matrix, n);
    if groups.len() == 1 {
        return Ok(None);
    }
    let _span =
        tracing::debug_span!("loop_block_factor", rows = n, blocks = groups.len()).entered();
    let mut result = vec![0.0; n];
    for group in groups {
        let size = group.len();
        let mut factor: Vec<_> = group
            .iter()
            .flat_map(|&i| group.iter().map(move |&j| matrix[i * n + j] / scale))
            .collect();
        if dense_chol(&mut factor, size).is_err() {
            tracing::debug!(fallback = "cholesky");
            return Ok(None);
        }
        let Some(factor) = MassFactor::from_cholesky(factor, size, scale) else {
            tracing::debug!(fallback = "rank");
            return Ok(None);
        };
        let b: Vec<_> = group.iter().map(|&i| rhs[i]).collect();
        for (&i, value) in group.iter().zip(factor.solve(&b)?) {
            result[i] = value;
        }
    }
    Ok(Some(result))
}

fn components(matrix: &[Real], n: usize) -> Vec<Vec<usize>> {
    let mut labels = vec![usize::MAX; n];
    let mut groups = Vec::new();
    for seed in 0..n {
        if labels[seed] != usize::MAX {
            continue;
        }
        let label = groups.len();
        let mut group = vec![seed];
        labels[seed] = label;
        let mut cursor = 0;
        while cursor < group.len() {
            let row = group[cursor];
            for col in 0..n {
                if labels[col] == usize::MAX
                    && (matrix[row * n + col] != 0.0 || matrix[col * n + row] != 0.0)
                {
                    labels[col] = label;
                    group.push(col);
                }
            }
            cursor += 1;
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(n: usize, interleaved: bool, scale: Real) -> Vec<Real> {
        let mut matrix = vec![0.0; n * n];
        let group = |i| if interleaved { i % 3 } else { i / (n / 3) };
        for i in 0..n {
            for j in 0..n {
                if group(i) == group(j) {
                    matrix[i * n + j] = scale * if i == j { 4.0 } else { 0.2 };
                }
            }
        }
        matrix
    }

    #[test]
    fn disconnected_full_rank_blocks_match_global_cholesky() {
        for n in [9, 24, 81] {
            for interleaved in [false, true] {
                for scale in [1e-100, 1.0, 1e100] {
                    let matrix = matrix(n, interleaved, scale);
                    let rhs: Vec<_> = (0..n).map(|i| (i as Real * 0.31).sin() * scale).collect();
                    let expected = solve_mass(&matrix, &rhs, n).unwrap();
                    let actual = try_solve(&matrix, &rhs, n).unwrap().unwrap();
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn global_rank_cutoff_singular_and_indefinite_systems_fall_back() {
        for diagonal in [0.0, -1.0, 1e-14, 1e-12] {
            let mut matrix = vec![0.0; 81];
            for i in 0..9 {
                matrix[i * 9 + i] = if i == 0 { diagonal } else { 1.0 };
            }
            let rhs = vec![1.0; 9];
            assert!(try_solve(&matrix, &rhs, 9).unwrap().is_none());
            match solve_mass(&matrix, &rhs, 9) {
                Ok(expected) => assert_eq!(solve(&matrix, &rhs, 9).unwrap(), expected),
                Err(_) => assert!(solve(&matrix, &rhs, 9).is_err()),
            }
        }
    }

    #[test]
    fn arbitrarily_small_cross_block_entries_are_not_discarded() {
        let mut matrix = matrix(9, false, 1.0);
        matrix[3] = 1e-100;
        matrix[27] = 1e-100;
        matrix[3 * 9 + 6] = 1e-200;
        matrix[6 * 9 + 3] = 1e-200;
        assert_eq!(components(&matrix, 9), vec![(0..9).collect::<Vec<_>>()]);
        assert!(try_solve(&matrix, &[1.0; 9], 9).unwrap().is_none());
    }
}
