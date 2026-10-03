//! Minimum Euclidean impulse allocation on a fixed quadratic objective level.

use crate::coupled_constraints::Domain;
use molla_core::{Error, Result};
use molla_math::Real;

fn remove_basis(vector: &mut [Real], basis: &[Vec<Real>]) {
    for _ in 0..2 {
        for b in basis {
            let dot: Real = vector.iter().zip(b).map(|(x, y)| x * y).sum();
            let squared: Real = b.iter().map(|x| x * x).sum();
            for (x, y) in vector.iter_mut().zip(b) {
                *x -= dot / squared * y;
            }
        }
    }
}

fn append_basis(mut vector: Vec<Real>, basis: &mut Vec<Vec<Real>>) {
    remove_basis(&mut vector, basis);
    let norm = vector.iter().map(|x| x * x).sum::<Real>().sqrt();
    if norm > 1e-12 {
        for x in &mut vector {
            *x /= norm;
        }
        basis.push(vector);
    }
}

pub(crate) fn canonicalize(
    gram: &[Real],
    free: &[Real],
    domains: &[Domain],
    dynamic: &[bool],
    impulse: &mut [Real],
) -> Result<()> {
    let n = impulse.len();
    if domains.iter().all(|d| matches!(d, Domain::Disk { .. })) {
        return Ok(());
    }
    let scale = impulse.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    if scale == 0. {
        return Ok(());
    }
    let gradient: Vec<Real> = gram
        .chunks_exact(n)
        .zip(free)
        .map(|(row, f)| {
            f + row
                .iter()
                .zip(impulse.iter())
                .map(|(a, x)| a * x)
                .sum::<Real>()
        })
        .collect();
    let mut fixed = Vec::new();
    let mut pinned_coordinates = vec![false; n];
    let mut r = 0;
    for (block, domain) in domains.iter().enumerate() {
        let pinned = match *domain {
            Domain::Disk {
                static_cap,
                dynamic_cap,
                ..
            } => {
                let radius = if dynamic[block] {
                    dynamic_cap
                } else {
                    static_cap
                };
                radius == 0. || gradient[r].hypot(gradient[r + 1]) > 1e-9
            }
            Domain::Nonnegative => gradient[r] > 1e-9,
            Domain::Interval { lower, upper } => lower == upper || gradient[r].abs() > 1e-9,
            Domain::Bilateral => false,
        };
        if pinned {
            for i in r..r + domain.width() {
                pinned_coordinates[i] = true;
                let mut row = vec![0.; n];
                row[i] = 1.;
                fixed.push(row);
            }
        }
        r += domain.width();
    }
    let mut basis: Vec<Vec<Real>> = Vec::new();
    for row in gram
        .chunks_exact(n)
        .chain(std::iter::once(free))
        .chain(fixed.iter().map(Vec::as_slice))
    {
        let maximum = row.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        if maximum == 0. {
            continue;
        }
        let mut vector: Vec<_> = row.iter().map(|x| x / maximum).collect();
        let norm = vector.iter().map(|x| x * x).sum::<Real>().sqrt();
        for x in &mut vector {
            *x /= norm;
        }
        append_basis(vector, &mut basis);
        if basis.len() == n {
            return Ok(());
        }
    }
    for _ in 0..domains.len() {
        let mut changed = false;
        let mut r = 0;
        for (block, domain) in domains.iter().enumerate() {
            if let Domain::Disk {
                static_cap,
                dynamic_cap,
                ..
            } = *domain
            {
                let cap = if dynamic[block] {
                    dynamic_cap
                } else {
                    static_cap
                };
                let length = impulse[r].hypot(impulse[r + 1]);
                if !pinned_coordinates[r]
                    && length > 0.
                    && (length - cap).abs() <= 8. * Real::EPSILON * cap
                {
                    let mut radial = vec![0.; n];
                    radial[r] = impulse[r] / length;
                    radial[r + 1] = impulse[r + 1] / length;
                    remove_basis(&mut radial, &basis);
                    if radial.iter().map(|x| x * x).sum::<Real>().sqrt() <= 1e-12 {
                        for i in r..r + 2 {
                            pinned_coordinates[i] = true;
                            let mut unit = vec![0.; n];
                            unit[i] = 1.;
                            append_basis(unit, &mut basis);
                        }
                        changed = true;
                    }
                }
            }
            r += domain.width();
        }
        if basis.len() == n {
            return Ok(());
        }
        if !changed {
            break;
        }
    }
    let target: Vec<Real> = basis
        .iter()
        .map(|b| {
            b.iter()
                .zip(impulse.iter())
                .map(|(a, x)| a * (x / scale))
                .sum()
        })
        .collect();
    let mut value = vec![0.; n];
    let mut correction = vec![0.; n];
    let mut normal = vec![0.; n];
    for _ in 0..32768 {
        for (b, target) in basis.iter().zip(&target) {
            let dot: Real = b.iter().zip(&value).map(|(a, x)| a * x).sum();
            for (x, a) in value.iter_mut().zip(b) {
                *x += (target - dot) * a;
            }
        }
        let mut r = 0;
        for (block, domain) in domains.iter().enumerate() {
            for i in r..r + domain.width() {
                value[i] += correction[i];
            }
            correction[r..r + domain.width()].copy_from_slice(&value[r..r + domain.width()]);
            match *domain {
                Domain::Disk {
                    static_cap,
                    dynamic_cap,
                    ..
                } => {
                    let radius = if dynamic[block] {
                        dynamic_cap
                    } else {
                        static_cap
                    } / scale;
                    let length = value[r].hypot(value[r + 1]);
                    if length > radius {
                        value[r] *= radius / length;
                        value[r + 1] *= radius / length;
                    }
                }
                Domain::Nonnegative => value[r] = value[r].max(0.),
                Domain::Interval { lower, upper } => {
                    value[r] = value[r].clamp(lower / scale, upper / scale);
                }
                Domain::Bilateral => {}
            }
            for i in r..r + domain.width() {
                if pinned_coordinates[i] {
                    value[i] = impulse[i] / scale;
                }
                correction[i] -= value[i];
            }
            r += domain.width();
        }
        for i in 0..n {
            normal[i] = value[i] + correction[i];
        }
        let mut residual: Real = 0.;
        for (b, target) in basis.iter().zip(&target) {
            let dot: Real = b.iter().zip(&value).map(|(a, x)| a * x).sum();
            residual = residual.max((dot - target).abs());
            let dot: Real = b.iter().zip(&normal).map(|(a, x)| a * x).sum();
            for (x, a) in normal.iter_mut().zip(b) {
                *x -= dot * a;
            }
        }
        for x in &normal {
            residual = residual.max(x.abs());
        }
        if !residual.is_finite() || value.iter().any(|x| !x.is_finite()) {
            break;
        }
        if residual <= 1e-11 {
            for (i, (x, value)) in impulse.iter_mut().zip(value).enumerate() {
                if !pinned_coordinates[i] {
                    *x = value * scale;
                }
            }
            return Ok(());
        }
    }
    Err(Error::Solver(
        "minimum-norm constraint allocation did not converge".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redundant_interval_rows_preserve_caps_points_and_minimum_norm() {
        for scale in [1e-12, 1., 1e12] {
            for (lower, upper, expected) in [
                (-2., 2., -0.5),
                (-0.2, 0.2, -0.2),
                (0.3, 0.3, 0.3),
                (0.2, 0.6, 0.2),
                (Real::NEG_INFINITY, Real::INFINITY, -0.5),
            ] {
                let mut impulse = [lower.max(-1.) * scale, (-1. - lower.max(-1.)) * scale];
                canonicalize(
                    &[1. / scale; 4],
                    &[1.; 2],
                    &[
                        Domain::Interval {
                            lower: lower * scale,
                            upper: upper * scale,
                        },
                        Domain::Bilateral,
                    ],
                    &[false; 2],
                    &mut impulse,
                )
                .unwrap();
                assert!((impulse[0] / scale - expected).abs() < 1e-9);
                assert!((impulse[1] / scale + 1. + expected).abs() < 1e-9);
            }
            let mut impulse = [0.2 * scale, -0.8 * scale];
            canonicalize(
                &[1. / scale; 4],
                &[-2., -1.],
                &[
                    Domain::Interval {
                        lower: -0.2 * scale,
                        upper: 0.2 * scale,
                    },
                    Domain::Bilateral,
                ],
                &[false; 2],
                &mut impulse,
            )
            .unwrap();
            assert_eq!(impulse, [0.2 * scale, -0.8 * scale]);
        }
    }

    #[test]
    fn redundant_allocation_has_analytic_interior_and_capped_solutions() {
        for scale in [1e-12, 1., 1e12] {
            for (cap, expected) in [(0.2, [-0.2, 0., -0.8]), (2., [-0.5, 0., -0.5])] {
                let gram: Vec<_> = [1., 0., 1., 0., 1., 0., 1., 0., 1.]
                    .map(|x| x / scale)
                    .to_vec();
                let disk = Domain::Disk {
                    static_cap: cap * scale,
                    dynamic_cap: cap * scale,
                    moving: true,
                };
                for initial in [[0., 0., -1.], [-cap.min(1.), 0., -1. + cap.min(1.)]] {
                    let mut impulse = initial.map(|x| x * scale);
                    canonicalize(
                        &gram,
                        &[1., 0., 1.],
                        &[disk, Domain::Bilateral],
                        &[true, false],
                        &mut impulse,
                    )
                    .unwrap();
                    for (actual, expected) in impulse.into_iter().zip(expected) {
                        assert!(
                            (actual / scale - expected).abs() < 1e-9,
                            "{actual} != {expected} at scale {scale}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn objective_linear_part_and_unilateral_feasibility_are_preserved() {
        let mut impulse = [1., 0.];
        canonicalize(
            &[1., 1., 1., 1.],
            &[-2., -1.],
            &[Domain::Nonnegative; 2],
            &[false; 2],
            &mut impulse,
        )
        .unwrap();
        assert_eq!(impulse, [1., 0.]);
        let mut impulse = [-1., 0.];
        canonicalize(
            &[1., 1., 1., 1.],
            &[1., 1.],
            &[Domain::Bilateral, Domain::Nonnegative],
            &[false; 2],
            &mut impulse,
        )
        .unwrap();
        assert!((impulse[0] + 1.).abs() < 1e-9);
        assert_eq!(impulse[1], 0.);
    }

    #[test]
    fn redundant_bilateral_rows_leave_strictly_inactive_stop_at_exact_zero() {
        let gram = [
            1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1.,
        ];
        let disk = Domain::Disk {
            static_cap: 2.,
            dynamic_cap: 2.,
            moving: true,
        };
        let mut impulse = [0., 0., 0., -1.];
        canonicalize(
            &gram,
            &[1., 0., 2., 1.],
            &[disk, Domain::Nonnegative, Domain::Bilateral],
            &[true, false, false],
            &mut impulse,
        )
        .unwrap();
        assert!((impulse[0] + 0.5).abs() < 1e-9);
        assert!((impulse[3] + 0.5).abs() < 1e-9);
        assert_eq!(impulse[2], 0.);
    }

    #[test]
    fn tangent_affine_disk_intersection_retains_its_unique_impulse() {
        let disk = Domain::Disk {
            static_cap: 1.,
            dynamic_cap: 1.,
            moving: true,
        };
        let mut impulse = [1., 0., 1.];
        canonicalize(
            &[1., 0., 0., 0., 1., 1., 0., 1., 1.],
            &[-1., -1., -1.],
            &[disk, Domain::Bilateral],
            &[true, false],
            &mut impulse,
        )
        .unwrap();
        assert!((impulse[0] - 1.).abs() < 1e-9);
        assert!(impulse[1].abs() < 1e-9);
        assert!((impulse[2] - 1.).abs() < 1e-9);
    }
}
