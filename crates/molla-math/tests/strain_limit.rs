use molla_math::{Mat3, Quat, Vec3, constitutive::MembraneStrainLimit};

fn limit() -> MembraneStrainLimit {
    MembraneStrainLimit {
        strain: 0.5,
        shrink: [1.0; 2],
    }
}

#[test]
fn cubic_strain_derivatives_are_coupled_symmetric_and_psd() {
    let l = limit();
    let rotate = Mat3::from_quat(Quat::from_rotation_x(0.4) * Quat::from_rotation_y(-0.7));
    let right = Mat3::from_quat(Quat::from_rotation_z(0.6));
    for (a, b) in [
        (0.7, 0.8),
        (1.0, 1.0),
        (1.2, 1.2),
        (1.2, 1.20000001),
        (1.3, 0.8),
        (1.3, 1.1),
    ] {
        let diagonal = Mat3::from_diagonal(Vec3::new(a, b, 0.0));
        let analytic =
            (2.0 / 3.0) * ((a - 1.0).max(0.0).powi(3) + (b - 1.0).max(0.0).powi(3)) / l.strain;
        for f in [diagonal, rotate * diagonal, rotate * diagonal * right] {
            let r = l.evaluate(f).unwrap();
            assert!((r.energy - analytic).abs() < 1.0e-12);
            assert!((l.energy(f).unwrap() - r.energy).abs() < 1.0e-12);
            let eps = 1.0e-6;
            for j in 0..6 {
                let mut values = [0.0; 9];
                values[j] = eps;
                let d = Mat3::from_cols_array(&values);
                let fd = (l.energy(f + d).unwrap() - l.energy(f - d).unwrap()) / (2.0 * eps);
                assert!((fd - r.gradient.to_cols_array()[j]).abs() < 1.0e-7);
                let dh = (l.evaluate(f + d).unwrap().gradient
                    - l.evaluate(f - d).unwrap().gradient)
                    * (0.5 / eps);
                for i in 0..6 {
                    assert!(
                        (dh.to_cols_array()[i] - r.hessian[i][j]).abs() < 1.0e-5,
                        "{a},{b} H[{i}][{j}]"
                    );
                    assert!((r.hessian[i][j] - r.hessian[j][i]).abs() < 1.0e-12);
                }
            }
            for trial in 0..30 {
                let direction = std::array::from_fn(|i| ((i + 1 + trial * 7) as f64).sin());
                let h = r
                    .hessian_mul(Mat3::from_cols_array(&direction))
                    .to_cols_array();
                let quadratic: f64 = direction.into_iter().zip(h).map(|(a, b)| a * b).sum();
                assert!(quadratic >= -1.0e-12);
            }
            for i in 6..9 {
                assert_eq!(r.hessian[i], [0.0; 9]);
            }
        }
    }
}

#[test]
fn authored_barrier_width_and_shrink_corrected_cap_are_distinct() {
    let l = MembraneStrainLimit {
        strain: 0.1,
        shrink: [0.8, 0.9],
    };
    assert!((l.maximum_stretch().unwrap() - 1.375).abs() < 1.0e-15);
    let f = Mat3::from_diagonal(Vec3::new(1.2, 1.2, 0.0));
    assert!((l.energy(f).unwrap() - (4.0 / 3.0) * 0.2f64.powi(3) / 0.1).abs() < 1.0e-14);
    assert!(
        MembraneStrainLimit {
            shrink: [1.0; 2],
            ..l
        }
        .check(f)
        .is_err()
    );
    for bad in [
        MembraneStrainLimit { strain: 0.0, ..l },
        MembraneStrainLimit {
            strain: f64::NAN,
            ..l
        },
        MembraneStrainLimit {
            shrink: [0.0, 1.0],
            ..l
        },
        MembraneStrainLimit {
            shrink: [2.0; 2],
            ..l
        },
    ] {
        assert!(bad.maximum_stretch().is_err());
    }
}

#[test]
fn shear_and_rotated_sweeps_are_limited_by_singular_values_not_edges() {
    let l = MembraneStrainLimit {
        strain: 0.05,
        shrink: [1.0; 2],
    };
    let f = Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::ZERO);
    let end = Mat3::from_cols(Vec3::X, Vec3::new(0.2, 0.96f64.sqrt(), 0.0), Vec3::ZERO);
    assert!((end.y_axis.length() - 1.0).abs() < 1.0e-15);
    assert!((end.x_axis - end.y_axis).length() < (f.x_axis - f.y_axis).length());
    assert!(l.check(end).is_err());
    let rot = Mat3::from_quat(Quat::from_rotation_y(0.7));
    for (start, finish) in [
        (f, end),
        (rot * f, rot * end),
        (
            f,
            Mat3::from_cols(Vec3::X * 10.0, Vec3::Y * 10.0, Vec3::ZERO),
        ),
    ] {
        let df = finish - start;
        let alpha = l.step_bound(start, df, 1.0).unwrap();
        assert!(alpha > 0.0 && alpha < 1.0);
        for sample in 0..=100 {
            l.check(start + df * (alpha * sample as f64 / 100.0))
                .unwrap();
        }
    }
    assert_eq!(l.step_bound(f, Mat3::ZERO, 1.0).unwrap(), 1.0);
    assert_eq!(l.step_bound(f, f * (-1.5), 1.0).unwrap(), 1.0);
    assert!(l.step_bound(end, Mat3::ZERO, 1.0).is_err());
}
