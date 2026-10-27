use super::{Error, MassFactor, Real, Result};

impl MassFactor {
    pub(crate) fn add_gram_compact(
        &self,
        projected: Vec<Vec<Real>>,
        response: &mut [Real],
    ) -> Result<()> {
        if projected.iter().all(|row| row.iter().any(|&v| v != 0.0)) {
            return self.add_gram(&projected, response);
        }
        let count = projected.len();
        if count.checked_mul(count) != Some(response.len())
            || projected
                .iter()
                .any(|row| row.len() != self.n || row.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Solver("invalid projected mass response".into()));
        }
        let (indices, compact): (Vec<_>, Vec<_>) = projected
            .into_iter()
            .enumerate()
            .filter(|(_, row)| row.iter().any(|&v| v != 0.0))
            .unzip();
        let size = indices.len();
        let mut local = vec![0.0; size * size];
        self.add_gram(&compact, &mut local)?;
        for (i, &row) in indices.iter().enumerate() {
            for (j, &col) in indices.iter().enumerate() {
                response[row * count + col] += local[i * size + j];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_response_matches_dense_for_zero_sparse_and_dense_rows() {
        for n in [1, 6, 13] {
            for scale in [1e-100, 1.0, 1e100] {
                for singular in [false, true] {
                    let mut matrix = vec![0.0; n * n];
                    for i in 0..n {
                        matrix[i * n + i] = if singular && i % 3 == 0 {
                            0.0
                        } else {
                            scale * (i + 1) as Real
                        };
                        for j in 0..i {
                            if !singular || (i % 3 != 0 && j % 3 != 0) {
                                matrix[i * n + j] = scale * 0.2;
                                matrix[j * n + i] = scale * 0.2;
                            }
                        }
                    }
                    let factor = MassFactor::new(&matrix, n).unwrap();
                    for count in [0, 1, 3, 4, 17, 41] {
                        for density in [0, 1, 3, 100] {
                            let projected: Vec<Vec<Real>> = (0..count)
                                .map(|row| {
                                    (0..n)
                                        .map(|col| {
                                            if row % 7 < density {
                                                ((row * n + col + 1) as Real * 0.71).sin()
                                                    * scale.sqrt()
                                            } else {
                                                0.0
                                            }
                                        })
                                        .collect()
                                })
                                .collect();
                            let mut expected: Vec<_> = (0..count * count)
                                .map(|i| (i as Real * 0.17).cos())
                                .collect();
                            let mut actual = expected.clone();
                            factor.add_gram(&projected, &mut expected).unwrap();
                            factor.add_gram_compact(projected, &mut actual).unwrap();
                            assert_eq!(
                                actual, expected,
                                "n={n} scale={scale} singular={singular} count={count} density={density}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_rows_fail_before_modifying_response() {
        let factor = MassFactor::new(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        for rows in [
            vec![vec![]],
            vec![vec![0.0]],
            vec![vec![Real::NAN, 0.0]],
            vec![vec![0.0; 2]; 2],
        ] {
            let mut response = [7.0];
            assert!(factor.add_gram_compact(rows, &mut response).is_err());
            assert_eq!(response, [7.0]);
        }
    }
}
