use super::*;

fn compare(factor: &MassFactor, rhs: &[Vec<Real>]) {
    let actual = factor.solve_batch(rhs).unwrap();
    assert_eq!(actual.len(), rhs.len());
    for (row, (actual, rhs)) in actual.iter().zip(rhs).enumerate() {
        let expected = factor.solve(rhs).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (col, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "row={row} col={col}");
        }
    }
}

#[test]
fn batches_match_scalar_responses_bitwise() {
    for n in [0, 1, 2, 6, 7, 39, 78] {
        for scale in [1e-60, 1.0, 1e60] {
            for weak in [0.0, 1e-13, 1.0] {
                let mut matrix = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        matrix[i * n + j] = scale * if i == j { 2.0 + i as Real } else { 0.1 };
                        if weak != 1.0 && (i == n - 1 || j == n - 1) {
                            matrix[i * n + j] = if i == j { scale * weak } else { 0.0 };
                        }
                    }
                }
                let factor = MassFactor::new(&matrix, n).unwrap();
                for count in [0, 1, 2, 3, 4, 5, 7, 8, 26, 33] {
                    let rhs: Vec<Vec<Real>> = (0..count)
                        .map(|row| {
                            (0..n)
                                .map(|col| ((row * 13 + col * 7) as Real).sin())
                                .collect()
                        })
                        .collect();
                    compare(&factor, &rhs);
                    let units: Vec<Vec<Real>> = (0..count)
                        .map(|row| {
                            (0..n)
                                .map(|col| {
                                    if col == (row / 2) % n {
                                        if row % 2 == 0 { 1.0 } else { -1.0 }
                                    } else {
                                        0.0
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    compare(&factor, &units);
                }
            }
        }
    }
}

#[test]
fn diagonal_zero_and_subnormal_responses_match_bitwise() {
    for n in [1, 6, 39] {
        for scale in [0.0, 1e-60, 1.0, 1e60] {
            let mut matrix = vec![0.0; n * n];
            for i in 0..n {
                matrix[i * n + i] = scale;
            }
            let factor = MassFactor::new(&matrix, n).unwrap();
            let rhs = [-0.0, 0.0, 1e-310, -1e-310, 1.0, -1.0]
                .iter()
                .map(|&value| {
                    let mut row = vec![0.0; n];
                    row[n / 2] = value;
                    row
                })
                .collect::<Vec<_>>();
            compare(&factor, &rhs);
        }
    }
}

#[test]
fn invalid_batches_are_rejected_in_every_lane_and_tail() {
    for matrix in [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0; 4]] {
        let factor = MassFactor::new(&matrix, 2).unwrap();
        for row in 0..7 {
            for invalid in [
                vec![0.0],
                vec![0.0; 3],
                vec![Real::NAN, 0.0],
                vec![0.0, Real::INFINITY],
            ] {
                let mut rhs = vec![vec![0.0; 2]; 7];
                rhs[row] = invalid;
                assert!(factor.solve_batch(&rhs).is_err());
            }
        }
    }
}
