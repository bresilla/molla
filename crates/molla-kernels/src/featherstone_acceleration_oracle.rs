use molla_math::{Mat3, Vec3};

type Matrix = [[f64; 10]; 10];

#[path = "featherstone_acceleration_balance.rs"]
mod balance;

#[path = "generalized_mass_predictor_tests.rs"]
mod retained_predictor;

#[path = "generalized_bias_tests.rs"]
mod retained_bias;

fn position(words: &[u32], body: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(words[8 * body]) as f64,
        f32::from_bits(words[8 * body + 1]) as f64,
        f32::from_bits(words[8 * body + 2]) as f64,
    )
}

fn body_inertia(words: &[u32], body: usize) -> (f64, Mat3) {
    let value = |i| f32::from_bits(words[i]) as f64;
    let (mass, inertia) = match body {
        0 => (1500.0, 600.0),
        5 => (20.0, 1.0),
        _ => (35.0, 3.0),
    };
    let q = Vec3::new(
        value(8 * body + 4),
        value(8 * body + 5),
        value(8 * body + 6),
    );
    let w = value(8 * body + 7);
    let rotate = |v: Vec3| v + 2.0 * q.cross(q.cross(v) + w * v);
    let rotation = Mat3::from_cols(rotate(Vec3::X), rotate(Vec3::Y), rotate(Vec3::Z));
    (mass, inertia * rotation * rotation.transpose())
}

fn mass_matrix(words: &[u32]) -> Matrix {
    let position = |body| position(words, body);
    let root = position(0);
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let mut mass = [[0.0; 10]; 10];
    for body in 0..6 {
        let (m, world_inertia) = body_inertia(words, body);
        let mut linear = [Vec3::ZERO; 10];
        let mut angular = [Vec3::ZERO; 10];
        for axis in 0..3 {
            linear[axis] = axes[axis];
            linear[axis + 3] = axes[axis].cross(position(body) - root);
            angular[axis + 3] = axes[axis];
        }
        if (1..=4).contains(&body) {
            angular[5 + body] = Vec3::Z;
        }
        for i in 0..10 {
            for j in 0..10 {
                mass[i][j] +=
                    m * linear[i].dot(linear[j]) + angular[i].dot(world_inertia * angular[j]);
            }
        }
    }
    mass
}

fn solve(mut a: Matrix, mut b: [f64; 10]) -> [f64; 10] {
    for column in 0..10 {
        let pivot = (column..10)
            .max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))
            .unwrap();
        assert!(a[pivot][column].abs() > 1e-12);
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..10 {
            let factor = a[row][column] / a[column][column];
            for j in column..10 {
                a[row][j] -= factor * a[column][j];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.0; 10];
    for i in (0..10).rev() {
        x[i] = (b[i] - (i + 1..10).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x
}

fn read(path: &std::path::Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), 1000);
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

#[test]
fn acceleration_mass_oracle_solve_contract() {
    let mut a = [[0.0; 10]; 10];
    for i in 0..10 {
        a[i][i] = 2.0 + i as f64;
        for j in 0..i {
            a[i][j] = 0.01;
            a[j][i] = 0.01;
        }
    }
    let expected: [f64; 10] = std::array::from_fn(|i| i as f64 - 4.5);
    let b = std::array::from_fn(|i| (0..10).map(|j| a[i][j] * expected[j]).sum());
    let x = solve(a, b);
    assert!(
        x.iter()
            .zip(expected)
            .all(|(x, expected)| (x - expected).abs() < 1e-12)
    );
}

fn validate_fixture_geometry(base: &[u32]) {
    let offsets = [
        [0.0, 0.0, 0.0],
        [-0.75, -0.55, -0.65],
        [-0.75, -0.55, 0.65],
        [0.75, -0.55, -0.65],
        [0.75, -0.55, 0.65],
        [-1.45, -0.72, 0.0],
    ];
    for body in 0..6 {
        for axis in 0..3 {
            let expected = ([65536.0, 0.95, 0.0][axis] + offsets[body][axis]) as f32;
            assert_eq!(base[8 * body + axis], expected.to_bits());
        }
    }
    assert_eq!(&base[4..8], &[0, 0, 0, 1.0f32.to_bits()]);
}

#[test]
#[ignore = "requires frozen six-body acceleration records in MOLLA_ACCELERATION_ORACLE_CAPTURE"]
fn acceleration_capture_obeys_f64_mass_response() {
    let directory =
        std::path::PathBuf::from(std::env::var_os("MOLLA_ACCELERATION_ORACLE_CAPTURE").unwrap());
    let base = read(&directory.join("step-00-variant-0.bin"));
    validate_fixture_geometry(&base);
    let mass = mass_matrix(&base);
    for i in 0..10 {
        assert!(mass[i][i] > 0.0);
        for j in 0..10 {
            assert!((mass[i][j] - mass[j][i]).abs() < 1e-12);
        }
    }
    for axis in 0..3 {
        assert_eq!(mass[axis][axis], 1660.0);
    }
    let mut max_residual = 0.0f64;
    let mut max_acceleration_error = 0.0f64;
    let mut worst = (0, 0, 0.0, 0.0, 0.0);
    for step in 0..10 {
        let a = read(&directory.join(format!("step-{step:02}-variant-0.bin")));
        let b = read(&directory.join(format!("step-{step:02}-variant-1.bin")));
        assert_eq!(a, b);
        assert_eq!(
            &a[..117],
            &base[..117],
            "fixed poses, velocities and coordinates"
        );
        let delta: [f64; 10] = std::array::from_fn(|i| {
            f32::from_bits(a[220 + i]) as f64 - f32::from_bits(base[220 + i]) as f64
        });
        assert!(delta.iter().all(|v| v.is_finite()));
        let force: [f64; 10] = std::array::from_fn(|i| {
            step as f64 * (i + 1) as f64 * 17.0 * if i % 2 == 0 { 1.0 } else { -1.0 }
        });
        let expected = solve(mass, force);
        let scale = 1.0 + force.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for i in 0..10 {
            let absolute_residual =
                ((0..10).map(|j| mass[i][j] * delta[j]).sum::<f64>() - force[i]).abs();
            let residual = absolute_residual / scale;
            if residual > max_residual {
                let spacing = |word| {
                    let value = f32::from_bits(word).abs();
                    (f32::from_bits(value.to_bits() + 1) as f64 - value as f64) * 0.5
                };
                let quantization = (0..10)
                    .map(|j| mass[i][j].abs() * (spacing(a[220 + j]) + spacing(base[220 + j])))
                    .sum();
                worst = (step, i, absolute_residual, quantization, scale);
            }
            max_residual = max_residual.max(residual);
            max_acceleration_error = max_acceleration_error.max((delta[i] - expected[i]).abs());
        }
    }
    eprintln!(
        "independent f64 incremental mass response: max_scaled_force_residual={max_residual:e} max_acceleration_error={max_acceleration_error:e}"
    );
    eprintln!(
        "worst step={} row={} absolute_force_residual={} final_output_half_ulp_bound={} normalization={}",
        worst.0, worst.1, worst.2, worst.3, worst.4
    );
    let rate_increment_error = max_acceleration_error * (1.0f32 / 2040.0f32) as f64;
    eprintln!("independent f64 incremental rate error at fixture dt={rate_increment_error:e}");
    assert!(max_residual <= 1e-6);
    assert!(rate_increment_error <= 1e-5);
}

#[test]
#[ignore = "requires zero-bias full-basis records in MOLLA_ACCELERATION_ORACLE_CAPTURE"]
fn acceleration_capture_obeys_f64_full_operator() {
    let directory =
        std::path::PathBuf::from(std::env::var_os("MOLLA_ACCELERATION_ORACLE_CAPTURE").unwrap());
    let base = read(&directory.join("step-00-variant-0.bin"));
    assert_eq!(base, read(&directory.join("step-00-variant-1.bin")));
    validate_fixture_geometry(&base);
    for range in [48..96, 107..117, 152..172, 220..230] {
        assert!(base[range].iter().all(|&word| f32::from_bits(word) == 0.0));
    }
    let mass = mass_matrix(&base);
    let mut inverse = [[0.0; 10]; 10];
    let mut max_residual = 0.0f64;
    let mut max_rate_error = 0.0f64;
    for column in 0..10 {
        let step = column + 1;
        let a = read(&directory.join(format!("step-{step:02}-variant-0.bin")));
        assert_eq!(
            a,
            read(&directory.join(format!("step-{step:02}-variant-1.bin")))
        );
        assert_eq!(&a[..117], &base[..117]);
        let acceleration: [f64; 10] = std::array::from_fn(|i| f32::from_bits(a[220 + i]) as f64);
        assert!(acceleration.iter().all(|value| value.is_finite()));
        let mut force = [0.0; 10];
        force[column] = 100.0;
        let expected = solve(mass, force);
        for i in 0..10 {
            inverse[i][column] = acceleration[i] / 100.0;
            let residual = ((0..10).map(|j| mass[i][j] * acceleration[j]).sum::<f64>() - force[i])
                .abs()
                / 101.0;
            max_residual = max_residual.max(residual);
            max_rate_error = max_rate_error
                .max((acceleration[i] - expected[i]).abs() * (1.0f32 / 2040.0f32) as f64);
        }
    }
    let mut reciprocity = 0.0f64;
    for i in 0..10 {
        for j in 0..10 {
            reciprocity = reciprocity.max(
                (inverse[i][j] - inverse[j][i]).abs()
                    / (1.0 + inverse[i][j].abs().max(inverse[j][i].abs())),
            );
        }
    }
    let mut factor = [[0.0; 10]; 10];
    for i in 0..10 {
        for j in 0..=i {
            let remainder = (inverse[i][j] + inverse[j][i]) * 0.5
                - (0..j).map(|k| factor[i][k] * factor[j][k]).sum::<f64>();
            factor[i][j] = if i == j {
                assert!(remainder > 0.0, "positive response energy");
                remainder.sqrt()
            } else {
                remainder / factor[j][j]
            };
        }
    }
    eprintln!(
        "independent f64 full operator: max_normalized_force_residual={max_residual:e} max_dt_scaled_rate_error={max_rate_error:e} reciprocity={reciprocity:e}"
    );
    assert!(max_residual <= 1e-6);
    assert!(max_rate_error <= 1e-5);
    assert!(reciprocity <= 1e-6);
}
