use super::*;

pub(super) fn force_offset(words: &[u32], gravity: Vec3) -> [f64; 10] {
    let rate = |i| f32::from_bits(words[107 + i]) as f64;
    let root_omega = Vec3::new(rate(3), rate(4), rate(5));
    let root = position(words, 0);
    let mut result = [0.0; 10];
    for body in 0..6 {
        let (mass, inertia) = body_inertia(words, body);
        let arm = position(words, body) - root;
        let spin = if (1..=4).contains(&body) {
            Vec3::Z * rate(5 + body)
        } else {
            Vec3::ZERO
        };
        let omega = root_omega + spin;
        let acceleration = root_omega.cross(root_omega.cross(arm));
        let angular_acceleration = root_omega.cross(spin);
        let force = mass * (gravity - acceleration);
        let torque = -(inertia * angular_acceleration + omega.cross(inertia * omega));
        let root_torque = arm.cross(force) + torque;
        for axis in 0..3 {
            result[axis] += force[axis];
            result[axis + 3] += root_torque[axis];
        }
        if (1..=4).contains(&body) {
            result[5 + body] += torque.z;
        }
    }
    result
}

#[test]
fn acceleration_balance_oracle_contract() {
    let model = molla_sim::ModelBuilder::new().build().unwrap();
    assert_eq!(model.gravity.host().unwrap()[0].y, -molla_math::GRAVITY);
    let mut words = vec![0; 250];
    for body in 0..6 {
        words[8 * body] = (body as f32 * 0.25).to_bits();
        words[8 * body + 1] = (-0.5f32).to_bits();
        words[8 * body + 7] = 1.0f32.to_bits();
    }
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let static_offset = force_offset(&words, gravity);
    assert!((static_offset[1] + 1660.0 * 9.81).abs() < 1e-10);
    assert_eq!(&static_offset[6..], &[0.0; 4]);
    assert_eq!(force_offset(&words, Vec3::ZERO), [0.0; 10]);
    for dof in 0..10 {
        words[107 + dof] = (0.125 * (dof + 1) as f32).to_bits();
    }
    let once = force_offset(&words, Vec3::ZERO);
    for dof in 0..10 {
        words[107 + dof] = (2.0 * f32::from_bits(words[107 + dof])).to_bits();
    }
    let twice = force_offset(&words, Vec3::ZERO);
    for i in 0..10 {
        assert!((twice[i] - 4.0 * once[i]).abs() < 1e-10);
    }
    for dof in 0..3 {
        words[107 + dof] = 1000.0f32.to_bits();
    }
    assert_eq!(twice, force_offset(&words, Vec3::ZERO));
}

#[test]
#[ignore = "requires frozen six-body acceleration records in MOLLA_ACCELERATION_ORACLE_CAPTURE"]
fn acceleration_capture_obeys_absolute_f64_balance() {
    let directory =
        std::path::PathBuf::from(std::env::var_os("MOLLA_ACCELERATION_ORACLE_CAPTURE").unwrap());
    let base = read(&directory.join("step-00-variant-0.bin"));
    validate_fixture_geometry(&base);
    assert_eq!(molla_math::GRAVITY, 9.80665, "frozen capture gravity");
    for dof in 0..10 {
        let expected = (0.125 * (dof + 1) as f64 + 2f64.powi(-29) * (dof + 1) as f64) as f32;
        assert_eq!(base[107 + dof], expected.to_bits());
    }
    let mass = mass_matrix(&base);
    let offset = force_offset(
        &base,
        Vec3::new(0.0, -(molla_math::GRAVITY as f32) as f64, 0.0),
    );
    let mut max_residual = 0.0f64;
    let mut max_rate_error = 0.0f64;
    for step in 0..10 {
        let a = read(&directory.join(format!("step-{step:02}-variant-0.bin")));
        let b = read(&directory.join(format!("step-{step:02}-variant-1.bin")));
        assert_eq!(a, b);
        assert_eq!(&a[..117], &base[..117]);
        let acceleration: [f64; 10] = std::array::from_fn(|i| f32::from_bits(a[220 + i]) as f64);
        assert!(acceleration.iter().all(|v| v.is_finite()));
        let rhs = std::array::from_fn(|i| {
            offset[i]
                + (step + 1) as f64 * (i + 1) as f64 * 17.0 * if i % 2 == 0 { 1.0 } else { -1.0 }
        });
        let expected = solve(mass, rhs);
        let scale = 1.0 + rhs.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for i in 0..10 {
            let residual =
                ((0..10).map(|j| mass[i][j] * acceleration[j]).sum::<f64>() - rhs[i]).abs() / scale;
            max_residual = max_residual.max(residual);
            max_rate_error = max_rate_error
                .max((acceleration[i] - expected[i]).abs() * (1.0f32 / 2040.0f32) as f64);
        }
    }
    eprintln!(
        "independent Newton-Euler absolute balance: max_normalized_force_residual={max_residual:e} max_dt_scaled_rate_error={max_rate_error:e}"
    );
    assert!(max_residual <= 1e-6);
    assert!(max_rate_error <= 1e-5);
}
