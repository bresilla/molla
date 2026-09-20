use super::*;

fn fixture(n: usize, count: usize, singular: bool, scale: Real) -> (MassFactor, Vec<Vec<Real>>) {
    let mut mass = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            mass[i * n + j] = scale * if i == j { 2.0 + i as Real } else { 0.1 };
            if singular && (i == n - 1 || j == n - 1) {
                mass[i * n + j] = 0.0;
            }
        }
    }
    let projected = (0..count)
        .map(|row| {
            (0..n)
                .map(|col| ((row * 13 + col * 7) as Real).sin())
                .collect()
        })
        .collect();
    (MassFactor::new(&mass, n).unwrap(), projected)
}

fn scalar_reference(factor: &MassFactor, projected: &[Vec<Real>], response: &mut [Real]) {
    let count = projected.len();
    if let MassDecomposition::Cholesky(lower) = &factor.decomposition {
        let whitened: Vec<_> = projected
            .iter()
            .map(|row| forward_sub(lower, row, factor.n).unwrap())
            .collect();
        for row in 0..count {
            for col in 0..=row {
                let value = whitened[row]
                    .iter()
                    .zip(&whitened[col])
                    .map(|(a, b)| a * b)
                    .sum::<Real>()
                    / factor.scale;
                response[row * count + col] += value;
                if row != col {
                    response[col * count + row] += value;
                }
            }
        }
    } else {
        for col in 0..count {
            let solved = factor.solve(&projected[col]).unwrap();
            for row in 0..count {
                response[row * count + col] += projected[row]
                    .iter()
                    .zip(&solved)
                    .map(|(a, b)| a * b)
                    .sum::<Real>();
            }
        }
    }
}

#[test]
fn gram_batches_match_scalar_solves_bitwise() {
    for n in [0, 1, 2, 6, 7, 8, 39, 78] {
        for count in [0, 1, 2, 3, 4, 5, 7, 8, 22, 33] {
            for singular in [false, true] {
                for scale in [1e-9, 1.0, 1e9] {
                    let (factor, projected) = fixture(n, count, singular, scale);
                    let mut expected = vec![7.0; count * count];
                    let mut actual = expected.clone();
                    for _ in 0..2 {
                        scalar_reference(&factor, &projected, &mut expected);
                        factor.add_gram(&projected, &mut actual).unwrap();
                        for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "n={n} count={count} singular={singular} scale={scale} index={index}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_batches_do_not_modify_response() {
    let (factor, projected) = fixture(8, 8, false, 1.0);
    for row in 0..projected.len() {
        for invalid_length in [false, true] {
            let mut invalid = projected.clone();
            if invalid_length {
                invalid[row].pop();
            } else {
                invalid[row][0] = Real::NAN;
            }
            let mut response = vec![7.0; projected.len() * projected.len()];
            assert!(factor.add_gram(&invalid, &mut response).is_err());
            assert!(response.iter().all(|&value| value == 7.0));
        }
    }
}

#[test]
#[ignore = "release CPU Gram benchmark; run without concurrent builds"]
fn gram_batch_timings() {
    for n in [6, 39, 78] {
        for count in [1, 4, 22, 64] {
            let (factor, projected) = fixture(n, count, false, 1.0);
            let mut response = vec![0.0; count * count];
            let mut samples = Vec::new();
            for iteration in 0..110 {
                response.fill(0.0);
                let start = std::time::Instant::now();
                factor.add_gram(&projected, &mut response).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(&response);
                if iteration >= 10 {
                    samples.push(elapsed);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "gram timing: n={n} count={count} median_us={:.6} p95_us={:.6}",
                samples[50], samples[95]
            );
        }
    }
}
