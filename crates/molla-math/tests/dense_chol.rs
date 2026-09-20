use molla_math::Real;
use molla_math::dense_solve::dense_chol;

fn scalar_reference(a: &mut [Real], n: usize) -> bool {
    if a.len() != n * n {
        return false;
    }
    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j {
            let value = a[j * n + k];
            diag -= value * value;
        }
        if diag <= 0.0 {
            return false;
        }
        let root = diag.sqrt();
        a[j * n + j] = root;
        for i in j + 1..n {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = value / root;
        }
    }
    true
}

fn spd(n: usize, seed: u64, scale: Real) -> Vec<Real> {
    let mut seed = seed;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..row {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((seed >> 11) as Real / (1u64 << 53) as Real - 0.5) * scale;
            matrix[row * n + col] = value;
            matrix[col * n + row] = value;
        }
    }
    for row in 0..n {
        matrix[row * n + row] = scale
            + matrix[row * n..(row + 1) * n]
                .iter()
                .map(|v| v.abs())
                .sum::<Real>();
    }
    matrix
}

fn compare(mut matrix: Vec<Real>, n: usize) {
    let mut reference = matrix.clone();
    let expected = scalar_reference(&mut reference, n);
    assert_eq!(dense_chol(&mut matrix, n).is_ok(), expected);
    for (index, (actual, expected)) in matrix.iter().zip(reference).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "n={n} index={index}");
    }
}

#[test]
fn factors_match_scalar_order_bitwise() {
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 26, 39, 78, 128] {
        for seed in 0..8 {
            for scale in [1e-60, 1e-18, 1.0, 1e18, 1e60] {
                compare(spd(n, seed, scale), n);
            }
        }
    }
}

#[test]
fn failed_factors_preserve_partial_updates() {
    for n in [1, 2, 6, 7, 8, 9, 10, 39] {
        for pivot in 0..n {
            for value in [0.0, -1.0, -Real::INFINITY] {
                let mut matrix = spd(n, 13, 1.0);
                matrix[pivot * n + pivot] = value;
                compare(matrix, n);
            }
        }
        for value in [Real::NAN, Real::INFINITY, Real::MIN_POSITIVE, -0.0] {
            let mut matrix = spd(n, 17, 1.0);
            matrix[n * n - 1] = value;
            compare(matrix, n);
        }
        compare(vec![0.0; n * n], n);
        compare(vec![1.0; n * n], n);
    }
    compare(vec![1.0, 2.0], 2);
}

#[test]
fn exceptional_coefficients_preserve_scalar_updates() {
    for n in [7, 8, 9, 10] {
        for index in 0..n * n {
            for value in [
                Real::NAN,
                Real::INFINITY,
                Real::NEG_INFINITY,
                -0.0,
                1e-310,
                -1e-310,
            ] {
                let mut matrix = spd(n, 19, 1.0);
                matrix[index] = value;
                compare(matrix, n);
            }
        }
    }
}

#[test]
#[ignore = "release CPU factorization benchmark; run without concurrent builds"]
fn factor_scaling() {
    for n in [6, 26, 39, 78, 156] {
        let matrix = spd(n, 19, 1.0);
        let mut samples = Vec::new();
        let mut work = matrix.clone();
        for iteration in 0..110 {
            work.copy_from_slice(&matrix);
            let start = std::time::Instant::now();
            dense_chol(std::hint::black_box(&mut work), n).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&work);
            if iteration >= 10 {
                samples.push(elapsed);
            }
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "factor scaling: n={n} median_us={:.6} p95_us={:.6}",
            samples[50], samples[95]
        );
    }
}
