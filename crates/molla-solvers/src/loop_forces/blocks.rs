use super::*;

#[test]
fn bounded_active_set_matches_dense_path_through_clamps_and_releases() {
    for interleaved in [false, true] {
        let mut matrix = vec![0.0; 24 * 24];
        let group = |i| if interleaved { i % 3 } else { i / 8 };
        for i in 0..24 {
            for j in 0..24 {
                if group(i) == group(j) {
                    matrix[i * 24 + j] = if i == j { 4.0 } else { 0.2 };
                }
            }
        }
        for phase in 0..20 {
            let rhs: Vec<_> = (0..24)
                .map(|i| ((i + phase) as Real * 0.39).sin() * 10.0)
                .collect();
            let bounds: Vec<_> = (0..24)
                .map(|i| match i % 4 {
                    0 => [-0.1, 0.2],
                    1 => [0.0, Real::INFINITY],
                    2 => [0.0, 0.0],
                    _ => [Real::NEG_INFINITY, Real::INFINITY],
                })
                .collect();
            let expected = bounded_solve(&matrix, &rhs, &bounds).unwrap();
            let actual = bounded_solve_inner::<true>(&matrix, &rhs, &bounds).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
