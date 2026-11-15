use molla_math::{
    Mat3, Quat, Real, Vec3,
    constitutive::{ElasticMaterial, ElasticModel},
};

fn magnitude(m: Mat3) -> Real {
    m.to_cols_array()
        .iter()
        .map(|v| v.abs())
        .fold(0.0, Real::max)
}

fn materials() -> [ElasticMaterial; 4] {
    [
        ElasticModel::Corotated,
        ElasticModel::QuarticStretch,
        ElasticModel::StableNeoHookean,
        ElasticModel::BaraffWitkin,
    ]
    .map(|model| ElasticMaterial {
        model,
        mu: 2.0,
        lambda: 3.0,
    })
}

#[test]
fn diagonal_energies_match_pinned_upstream_formulas() {
    let f = Mat3::from_diagonal(Vec3::new(1.2, 0.8, 1.1));
    for (i, material) in materials().iter().enumerate() {
        let expected = [0.16, 0.004, 0.1624, 0.08][i];
        assert!((material.evaluate(f, 2, false).unwrap().energy - expected).abs() < 1.0e-12);
        if i < 3 {
            let expected = [0.195, 0.006354, 0.182704][i];
            assert!((material.evaluate(f, 3, false).unwrap().energy - expected).abs() < 1.0e-12);
        }
    }
}

#[test]
fn full_gradients_and_raw_hessians_match_finite_differences() {
    let rotate = Mat3::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.5, 0.7));
    let bases = [
        Mat3::IDENTITY,
        Mat3::IDENTITY * 0.7,
        Mat3::IDENTITY * 1.3,
        Mat3::from_diagonal(Vec3::new(1.1, 1.1 + 1.0e-10, 0.8)),
        Mat3::from_cols(
            Vec3::new(1.2, 0.1, -0.1),
            Vec3::new(0.3, 0.8, 0.2),
            Vec3::new(0.1, -0.2, 1.1),
        ),
    ];
    for material in materials() {
        for n in [2, 3] {
            if n == 3 && material.model == ElasticModel::BaraffWitkin {
                continue;
            }
            for base in bases {
                let f = rotate * base;
                let r = material.evaluate(f, n, false).unwrap();
                assert!(
                    (material.energy(f, n).unwrap() - r.energy).abs()
                        < 1.0e-11 * (1.0 + r.energy.abs())
                );
                let eps = 1.0e-6;
                for column in 0..3 * n {
                    let mut d = [0.0; 9];
                    d[column] = eps;
                    let d = Mat3::from_cols_array(&d);
                    let plus = material.evaluate(f + d, n, false).unwrap();
                    let minus = material.evaluate(f - d, n, false).unwrap();
                    let grad = (plus.energy - minus.energy) / (2.0 * eps);
                    let exact = r.gradient.to_cols_array()[column];
                    assert!(
                        (grad - exact).abs() < 2.0e-7 * (1.0 + exact.abs()),
                        "{:?} n={n} col={column}: grad {grad} != {exact}",
                        material.model
                    );
                    let fd = (plus.gradient - minus.gradient) * (0.5 / eps);
                    for row in 0..3 * n {
                        let value = r.hessian[row][column];
                        assert!(
                            (fd.to_cols_array()[row] - value).abs() < 3.0e-6 * (1.0 + value.abs()),
                            "{:?} n={n} H[{row},{column}]: {} != {value}",
                            material.model,
                            fd.to_cols_array()[row]
                        );
                        assert!((value - r.hessian[column][row]).abs() < 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn projected_tangents_are_symmetric_psd_and_objective() {
    let rotate = Mat3::from_quat(Quat::from_euler(glam::EulerRot::ZYX, -0.4, 0.3, 1.1));
    for material in materials() {
        for n in [2, 3] {
            if n == 3 && material.model == ElasticModel::BaraffWitkin {
                continue;
            }
            for scale in [0.3, 0.8, 1.0, 1.5] {
                let f = Mat3::from_cols(
                    Vec3::new(scale, 0.1, 0.0),
                    Vec3::new(0.2, scale, 0.0),
                    Vec3::Z * scale,
                );
                let r = material.evaluate(f, n, true).unwrap();
                let rotated = material.evaluate(rotate * f, n, true).unwrap();
                assert!((r.energy - rotated.energy).abs() < 1.0e-11);
                assert!(magnitude(rotated.gradient - rotate * r.gradient) < 1.0e-10);
                for k in 0..32 {
                    let values: [Real; 9] =
                        std::array::from_fn(|i| ((i * 37 + k * 13 + 7) as Real).sin());
                    let d = Mat3::from_cols_array(&values);
                    let hd = r.hessian_mul(d);
                    let quadratic: Real = hd
                        .to_cols_array()
                        .iter()
                        .zip(values)
                        .map(|(a, b)| a * b)
                        .sum();
                    assert!(
                        quadratic >= -1.0e-10,
                        "{:?} n={n}: {quadratic}",
                        material.model
                    );
                    let mut d_shell = d;
                    if n == 2 {
                        d_shell.z_axis = Vec3::ZERO;
                    }
                    assert!(
                        magnitude(
                            rotated.hessian_mul(rotate * d_shell) - rotate * r.hessian_mul(d_shell)
                        ) < 1.0e-9
                    );
                }
                for i in 0..9 {
                    for k in 0..9 {
                        assert!((r.hessian[i][k] - r.hessian[k][i]).abs() < 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn material_axes_and_domain_are_explicit() {
    let rotation = Mat3::from_rotation_z(0.6);
    let f = Mat3::from_diagonal(Vec3::new(1.3, 0.7, 1.0));
    for material in materials() {
        let e = material.evaluate(f, 2, false).unwrap().energy;
        let other = material.evaluate(f * rotation, 2, false).unwrap().energy;
        if material.model == ElasticModel::BaraffWitkin {
            assert!((e - other).abs() > 0.01);
        } else {
            assert!((e - other).abs() < 1.0e-12);
        }
        assert!(material.evaluate(Mat3::ZERO, 2, false).is_err());
        assert!(
            material
                .evaluate(Mat3::from_diagonal(Vec3::new(1.0, -1.0, 1.0)), 3, false)
                .is_err()
        );
        assert!(
            material
                .evaluate(Mat3::IDENTITY * Real::INFINITY, 2, true)
                .is_err()
        );
        assert!(material.evaluate(f, 1, false).is_err());
    }
    let material = ElasticMaterial {
        model: ElasticModel::Corotated,
        mu: 2.0,
        lambda: -0.5,
    };
    assert!(material.evaluate(f, 3, true).is_ok());
}
