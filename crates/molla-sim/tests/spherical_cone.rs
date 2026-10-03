use molla_math::{Quat, Vec3};
use molla_sim::{
    joint_control::SphericalConeLimits,
    spherical_cone::{basis, evaluate},
};

fn orientation(axis: Vec3, angle: f64, azimuth: f64, twist: f64) -> Quat {
    let [p, b0, b1] = basis(axis).unwrap();
    let direction = b0 * azimuth.cos() + b1 * azimuth.sin();
    Quat::from_axis_angle(p.cross(direction), angle) * Quat::from_axis_angle(p, twist)
}

#[test]
fn circular_cones_equal_primary_direction_angle_for_all_axes() {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(2.0, -3.0, 4.0)] {
        for extent in [0.1, 0.7, 1.9, 3.0] {
            for angle in [0.0, 1e-5, 0.5, 1.7, 3.13] {
                for azimuth in [0.0, 0.8, 2.3] {
                    let g = evaluate(
                        SphericalConeLimits {
                            angle0: Some(extent),
                            angle1: Some(extent),
                        },
                        axis,
                        orientation(axis, angle, azimuth, 1.3),
                    )
                    .unwrap();
                    assert!((g.angle - angle).abs() < 1e-12);
                    assert!((g.violation - (angle - extent).max(0.0)).abs() < 1e-11);
                    assert!((g.swing[0].hypot(g.swing[1]) - angle).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn asymmetric_diagonal_swing_is_elliptical_not_rectangular() {
    let limits = SphericalConeLimits {
        angle0: Some(0.4),
        angle1: Some(0.8),
    };
    let u = [0.4 * 0.8, 0.8 * 0.8];
    let q = orientation(Vec3::X, f64::hypot(u[0], u[1]), f64::atan2(u[1], u[0]), 0.0);
    let g = evaluate(limits, Vec3::X, q).unwrap();
    assert!(g.swing[0].abs() < 0.4 && g.swing[1].abs() < 0.8);
    assert!(g.rows[0].error > 0.0 && g.violation > 0.0);
    let boundary = orientation(
        Vec3::X,
        0.4_f64.hypot(0.8) / 2.0_f64.sqrt(),
        2.0_f64.atan(),
        0.0,
    );
    assert!(
        evaluate(limits, Vec3::X, boundary).unwrap().rows[0]
            .error
            .abs()
            < 1e-12
    );
}

#[test]
fn zero_and_disabled_extents_have_distinct_rows() {
    for (limits, count, bilateral) in [
        (SphericalConeLimits::default(), 0, 0),
        (
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: None,
            },
            1,
            1,
        ),
        (
            SphericalConeLimits {
                angle0: None,
                angle1: Some(0.0),
            },
            1,
            1,
        ),
        (
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
            2,
            2,
        ),
        (
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.7),
            },
            2,
            1,
        ),
        (
            SphericalConeLimits {
                angle0: None,
                angle1: Some(0.7),
            },
            1,
            0,
        ),
    ] {
        let g = evaluate(limits, Vec3::X, orientation(Vec3::X, 0.6, 0.8, 0.0)).unwrap();
        assert_eq!(g.row_count, count);
        assert_eq!(
            g.rows[..count].iter().filter(|r| r.bilateral).count(),
            bilateral
        );
    }
    let g = evaluate(
        SphericalConeLimits {
            angle0: None,
            angle1: Some(0.1),
        },
        Vec3::X,
        orientation(Vec3::X, 1.2, 0.0, 0.0),
    )
    .unwrap();
    assert!(g.violation < 1e-12 && g.rows[0].error < 0.0);
}

#[test]
fn cone_rows_match_angular_derivatives_and_do_not_oppose_twist() {
    let h = 1e-7;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(2.0, -3.0, 4.0)] {
        for angle in [0.0, 1e-5, 0.7, 2.6, std::f64::consts::PI - 1e-3] {
            for limits in [
                SphericalConeLimits {
                    angle0: Some(0.4),
                    angle1: Some(0.9),
                },
                SphericalConeLimits {
                    angle0: Some(0.0),
                    angle1: Some(0.9),
                },
                SphericalConeLimits {
                    angle0: None,
                    angle1: Some(0.0),
                },
            ] {
                let q = orientation(axis, angle, 0.7, -1.2);
                let g = evaluate(limits, axis, q).unwrap();
                for k in 0..g.row_count {
                    let row = g.rows[k];
                    assert!(row.angular.dot(q * axis.normalize()).abs() < 1e-9);
                    for omega in [Vec3::X, Vec3::Y, Vec3::Z] {
                        let plus =
                            evaluate(limits, axis, Quat::from_scaled_axis(omega * h) * q).unwrap();
                        let minus =
                            evaluate(limits, axis, Quat::from_scaled_axis(-omega * h) * q).unwrap();
                        let numerical = (plus.rows[k].error - minus.rows[k].error) / (2.0 * h);
                        let expected = row.angular.dot(omega);
                        assert!(
                            (numerical - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                            "axis={axis:?} angle={angle} row={k} derivative={numerical}, analytic={expected}"
                        );
                    }
                    for twist in [-2.9, 0.0, 1.7] {
                        let twisted = evaluate(
                            limits,
                            axis,
                            q * Quat::from_axis_angle(axis.normalize(), twist),
                        )
                        .unwrap();
                        let negated = evaluate(limits, axis, -q).unwrap();
                        assert!((twisted.rows[k].error - row.error).abs() < 1e-10);
                        assert!((negated.rows[k].angular - row.angular).length() < 1e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn antipodal_limiting_rows_are_finite_and_invalid_inputs_fail_explicitly() {
    let bounded = SphericalConeLimits {
        angle0: Some(0.4),
        angle1: Some(0.8),
    };
    let q = Quat::from_rotation_y(std::f64::consts::PI);
    let g = evaluate(bounded, Vec3::X, q).unwrap();
    assert_eq!(g.row_count, 1);
    assert!((g.violation - (std::f64::consts::PI - 0.8)).abs() < 1e-12);
    assert!((g.rows[0].angular + Vec3::Y).length() < 1e-12);
    assert!(!g.rows[0].bilateral);
    for limits in [
        SphericalConeLimits::default(),
        SphericalConeLimits {
            angle0: Some(std::f64::consts::PI),
            angle1: Some(std::f64::consts::PI),
        },
    ] {
        let g = evaluate(limits, Vec3::X, q).unwrap();
        assert_eq!(g.row_count, 0);
        assert_eq!(g.violation, 0.0);
    }
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 3.2] {
        assert!(
            SphericalConeLimits {
                angle0: Some(invalid),
                angle1: None
            }
            .validate()
            .is_err()
        );
    }
    assert!(evaluate(bounded, Vec3::ZERO, Quat::IDENTITY).is_err());
    assert!(evaluate(bounded, Vec3::X, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)).is_err());
    assert!(
        evaluate(
            SphericalConeLimits {
                angle0: Some(1e-200),
                angle1: Some(1.0)
            },
            Vec3::X,
            Quat::IDENTITY
        )
        .is_err()
    );
}

#[test]
fn antipodal_limits_preserve_twist_and_choose_the_widest_admissible_chart() {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(2., -3., 4.)] {
        let [p, b0, b1] = basis(axis).unwrap();
        for extents in [
            [Some(0.4), Some(0.9)],
            [Some(0.9), Some(0.4)],
            [Some(0.), Some(0.)],
            [Some(0.), Some(0.9)],
            [Some(0.), None],
            [Some(0.9), Some(std::f64::consts::PI)],
        ] {
            let limits = SphericalConeLimits {
                angle0: extents[0],
                angle1: extents[1],
            };
            let radius = extents.map(|a| a.unwrap_or(std::f64::consts::PI));
            let selected = usize::from(radius[1] > radius[0]);
            for twist in [0., 0.7, -2.] {
                let rotation = Quat::from_axis_angle(p.cross(b0), std::f64::consts::PI)
                    * Quat::from_axis_angle(p, twist);
                let geometry = evaluate(limits, axis, rotation).unwrap();
                let antipodal = evaluate(limits, axis, -rotation).unwrap();
                assert_eq!(geometry.row_count, antipodal.row_count);
                assert!(
                    (geometry.violation - (std::f64::consts::PI - radius[selected])).abs() < 1e-12
                );
                let n = rotation * p;
                for row in &geometry.rows[..geometry.row_count] {
                    assert!(row.error.is_finite() && row.angular.is_finite());
                    assert!(row.angular.dot(n).abs() < 1e-12);
                }
                if radius[selected] < std::f64::consts::PI {
                    let radial = p.cross([b0, b1][selected]);
                    let h = 1e-5;
                    let after =
                        evaluate(limits, axis, Quat::from_scaled_axis(-radial * h) * rotation)
                            .unwrap();
                    assert!(((geometry.violation - after.violation) / h - 1.).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn admissible_antipode_has_a_short_return_row_instead_of_a_remote_radial_stop() {
    let angle = std::f64::consts::PI - 0.001;
    let limits = SphericalConeLimits {
        angle0: None,
        angle1: Some(0.9),
    };
    let rotation = Quat::from_rotation_y(angle);
    let g = evaluate(limits, Vec3::X, rotation).unwrap();
    assert_eq!(g.row_count, 1);
    assert!((g.violation - 0.001).abs() < 1e-12);
    assert!((g.rows[0].error - 0.001).abs() < 1e-12);
    assert!((g.rows[0].angular + Vec3::Y).length() < 1e-12);
    let h = 1e-5;
    let after = evaluate(
        limits,
        Vec3::X,
        Quat::from_scaled_axis(Vec3::Y * h) * rotation,
    )
    .unwrap();
    assert!(((g.rows[0].error - after.rows[0].error) / h - 1.).abs() < 1e-8);
}
