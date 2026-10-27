//! Double-precision block minimization over Coulomb disks and scalar impulse domains.

use molla_core::{Error, Result};
use molla_math::Real;

/// One Coulomb disk occupies two rows; each scalar domain occupies one.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Disk {
        static_cap: Real,
        dynamic_cap: Real,
        moving: bool,
    },
    Bilateral,
    Nonnegative,
    Interval {
        lower: Real,
        upper: Real,
    },
}

impl Domain {
    fn project(self, value: Real) -> Real {
        match self {
            Self::Interval { lower, upper } => value.clamp(lower, upper),
            Self::Nonnegative => value.max(0.),
            _ => value,
        }
    }

    fn residual(self, impulse: Real, velocity: Real) -> Real {
        match self {
            Self::Interval { lower, upper } if lower == upper => 0.,
            Self::Interval { lower, .. } if impulse == lower => (-velocity).max(0.),
            Self::Interval { upper, .. } if impulse == upper => velocity.max(0.),
            Self::Nonnegative if impulse == 0. => (-velocity).max(0.),
            _ => velocity.abs(),
        }
    }

    pub fn width(self) -> usize {
        if matches!(self, Self::Disk { .. }) {
            2
        } else {
            1
        }
    }
}

#[derive(Debug)]
pub struct Solution {
    pub impulses: Vec<Real>,
    /// Mode flags are indexed by domain; scalar flags remain false.
    pub dynamic: Vec<bool>,
    /// Disk slip is below the CPU velocity tolerance; scalar flags remain false.
    pub sticking: Vec<bool>,
}

/// Minimize a supplied PSD quadratic over disks and scalar impulse domains.
/// Feedback diagonals belong in `gram`; physical publication uses the original matrix.
/// Redundant impulses use the minimum Euclidean norm among primary minimizers.
pub fn solve(gram: &[Real], free: &[Real], domains: &[Domain]) -> Result<Solution> {
    let invalid = || Error::Solver("invalid common constraint problem".into());
    let n = free.len();
    if domains.iter().map(|d| d.width()).sum::<usize>() != n
        || n.checked_mul(n) != Some(gram.len())
        || gram.iter().chain(free).any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    for i in 0..n {
        if gram[i * n + i] < 0. {
            return Err(invalid());
        }
        for j in 0..i {
            let scale = gram[i * n + i].sqrt() * gram[j * n + j].sqrt();
            if (gram[i * n + j] - gram[j * n + i]).abs() > 1e-10 * scale
                || gram[i * n + j].abs() > scale * (1. + 1e-10)
            {
                return Err(invalid());
            }
        }
    }
    if n > 0 && domains.iter().all(|d| matches!(d, Domain::Bilateral)) {
        let rhs: Vec<_> = free.iter().map(|value| -value).collect();
        let mut factor = gram.to_vec();
        if let Ok(impulses) = molla_math::dense_solve::dense_chol_solve(&mut factor, &rhs, n) {
            let accurate = impulses.iter().all(|value| value.is_finite())
                && (0..n).all(|r| factor[r * n + r].powi(2) > 1e-12 * gram[r * n + r])
                && (0..n).all(|r| {
                    let (sum, magnitude) = (0..n)
                        .map(|c| gram[r * n + c] * impulses[c])
                        .fold((0., 0.), |(sum, magnitude), value| {
                            (sum + value, magnitude + value.abs())
                        });
                    let residual = free[r] + sum;
                    let scale = free[r].abs() + magnitude;
                    residual.is_finite()
                        && scale.is_finite()
                        && residual.abs() <= 128.0 * Real::EPSILON * scale
                });
            if accurate {
                return Ok(Solution {
                    impulses,
                    dynamic: vec![false; domains.len()],
                    sticking: vec![false; domains.len()],
                });
            }
        }
    }
    let mut offsets = Vec::with_capacity(domains.len());
    let mut first = 0;
    let mut dynamic = Vec::with_capacity(domains.len());
    for &domain in domains {
        if let Domain::Interval { lower, upper } = domain
            && (lower.is_nan()
                || upper.is_nan()
                || lower > upper
                || lower == Real::INFINITY
                || upper == Real::NEG_INFINITY)
        {
            return Err(invalid());
        }
        offsets.push(first);
        first += domain.width();
        dynamic.push(
            if let Domain::Disk {
                static_cap,
                dynamic_cap,
                moving,
            } = domain
            {
                if !static_cap.is_finite()
                    || !dynamic_cap.is_finite()
                    || dynamic_cap < 0.
                    || static_cap < dynamic_cap
                {
                    return Err(invalid());
                }
                moving
            } else {
                false
            },
        );
    }
    let cap = |b: usize, dynamic: &[bool]| match domains[b] {
        Domain::Disk {
            static_cap,
            dynamic_cap,
            ..
        } => {
            if dynamic[b] {
                dynamic_cap
            } else {
                static_cap
            }
        }
        _ => 0.,
    };
    let mut reference = vec![0.; n];
    for (&domain, &r) in domains.iter().zip(&offsets) {
        if domain.width() == 1 {
            reference[r] = domain.project(0.);
        }
    }
    let mut lambda = reference.clone();
    let mut sticking = vec![false; domains.len()];
    let gradient = |i: usize, lambda: &[Real]| {
        free[i] + (0..n).map(|j| gram[i * n + j] * lambda[j]).sum::<Real>()
    };
    for _ in 0..=domains.len() {
        lambda.copy_from_slice(&reference);
        let mut converged = false;
        let mut residual: Real = Real::INFINITY;
        for _ in 0..32768 {
            for (b, &domain) in domains.iter().enumerate() {
                let r = offsets[b];
                match domain {
                    Domain::Disk { .. } => {
                        let mut v = [free[r], free[r + 1]];
                        for j in 0..n {
                            if j != r && j != r + 1 {
                                v[0] += gram[r * n + j] * lambda[j];
                                v[1] += gram[(r + 1) * n + j] * lambda[j];
                            }
                        }
                        let a = [
                            gram[r * n + r],
                            gram[r * n + r + 1],
                            gram[(r + 1) * n + r + 1],
                        ];
                        let (impulse, _) = crate::contact_friction::impulse(
                            a,
                            v,
                            cap(b, &dynamic),
                            cap(b, &dynamic),
                        );
                        lambda[r..r + 2].copy_from_slice(&impulse);
                    }
                    Domain::Bilateral | Domain::Nonnegative | Domain::Interval { .. } => {
                        let diagonal = gram[r * n + r];
                        if diagonal > 0. {
                            let next = lambda[r] - gradient(r, &lambda) / diagonal;
                            lambda[r] = domain.project(next);
                        } else if let Domain::Interval { lower, upper } = domain {
                            let velocity = gradient(r, &lambda);
                            lambda[r] = domain.project(if velocity > 0. {
                                lower
                            } else if velocity < 0. {
                                upper
                            } else {
                                0.
                            });
                        }
                    }
                }
            }
            if lambda.iter().any(|v| !v.is_finite()) {
                return Err(Error::Solver("nonfinite common constraint impulse".into()));
            }
            residual = 0.;
            for (b, &domain) in domains.iter().enumerate() {
                let r = offsets[b];
                let v = gradient(r, &lambda);
                match domain {
                    Domain::Disk { .. } => {
                        let w = gradient(r + 1, &lambda);
                        sticking[b] = v.hypot(w) <= 1e-9;
                        let radius = cap(b, &dynamic);
                        if radius == 0. {
                            continue;
                        }
                        let length = lambda[r].hypot(lambda[r + 1]);
                        let curvature = gram[r * n + r].abs().max(gram[(r + 1) * n + r + 1].abs())
                            + gram[r * n + r + 1].abs();
                        if length > radius * (1. + 1e-12) {
                            return Err(Error::Solver(
                                "common friction impulse exceeds cap".into(),
                            ));
                        }
                        if length == 0. {
                            residual = residual.max(v.hypot(w));
                            continue;
                        }
                        let direction = [lambda[r] / length, lambda[r + 1] / length];
                        let multiplier = (-(v * direction[0] + w * direction[1])).max(0.);
                        residual = residual.max(
                            (v + multiplier * direction[0]).hypot(w + multiplier * direction[1]),
                        );
                        residual =
                            residual.max(((radius - length).max(0.) * curvature).min(multiplier));
                    }
                    _ => residual = residual.max(domain.residual(lambda[r], v)),
                }
            }
            if residual <= 1e-11 {
                converged = true;
                break;
            }
        }
        if !converged {
            let (row, velocity) = (0..n)
                .map(|i| (i, gradient(i, &lambda)))
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap_or((0, 0.));
            return Err(Error::Solver(format!(
                "common joint/contact response did not converge: residual={residual:e} row={row} velocity={velocity:e} free={:e}",
                free.get(row).copied().unwrap_or(0.)
            )));
        }
        let mut promoted = false;
        for (b, &domain) in domains.iter().enumerate() {
            if matches!(domain, Domain::Disk { .. }) && !dynamic[b] {
                let r = offsets[b];
                if gradient(r, &lambda).hypot(gradient(r + 1, &lambda)) > 1e-4 {
                    dynamic[b] = true;
                    promoted = true;
                }
            }
        }
        if !promoted {
            let original = lambda.clone();
            crate::constraint_allocation::canonicalize(gram, free, domains, &dynamic, &mut lambda)?;
            for i in 0..n {
                if (gradient(i, &lambda) - gradient(i, &original)).abs() > 1e-10 {
                    return Err(Error::Solver(
                        "minimum-norm allocation changed constraint response".into(),
                    ));
                }
            }
            for (b, &domain) in domains.iter().enumerate() {
                if matches!(domain, Domain::Disk { .. }) {
                    let r = offsets[b];
                    sticking[b] = gradient(r, &lambda).hypot(gradient(r + 1, &lambda)) <= 1e-9;
                }
            }
            let energy: Real = (0..n)
                .map(|i| lambda[i] * (free[i] + gradient(i, &lambda)) * 0.5)
                .sum();
            let scale: Real = (0..n).map(|i| (lambda[i] * free[i]).abs()).sum();
            let reference_energy: Real = (0..n)
                .map(|i| reference[i] * (free[i] + gradient(i, &reference)) * 0.5)
                .sum();
            if !energy.is_finite()
                || !reference_energy.is_finite()
                || energy > reference_energy + 1e-12 * (1. + scale + reference_energy.abs())
            {
                return Err(Error::Solver(
                    "common constraint objective increased".into(),
                ));
            }
            return Ok(Solution {
                impulses: lambda,
                dynamic,
                sticking,
            });
        }
    }
    Err(Error::Solver(
        "common friction mode promotion did not converge".into(),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn bilateral_full_rank_response_preserves_load_precision_across_scales() {
        for scale in [1e-12, 0.01, 1., 100., 1e12] {
            let gram = [4., 1., 1., 3.].map(|value| value / scale);
            let result =
                super::solve(&gram, &[7e-6, -9e-6], &[super::Domain::Bilateral; 2]).unwrap();
            for (actual, expected) in result.impulses.iter().zip([-3e-5 / 11., 4.3e-5 / 11.]) {
                assert!((actual / scale - expected).abs() <= 2e-18);
            }
        }
        let result = super::solve(&[1.; 4], &[0.1; 2], &[super::Domain::Bilateral; 2]).unwrap();
        for actual in result.impulses {
            assert!((actual + 0.05).abs() <= 1e-12);
        }
    }
    use super::*;

    #[test]
    fn bounded_feedback_and_friction_match_independent_active_set_oracle() {
        for scale in [1e-12, 0.01, 1., 100., 1e12] {
            for diagonal in [0., 0.5, 2.] {
                for (u, v) in [(-3., -4.), (-3., 4.), (0., 0.), (3., -4.), (3., 4.)] {
                    for (lower, upper) in [
                        (-0.3, 0.7),
                        (0.2, 0.2),
                        (-0.5, -0.2),
                        (0.2, 0.6),
                        (Real::NEG_INFINITY, 0.1),
                        (-0.1, Real::INFINITY),
                        (Real::NEG_INFINITY, Real::INFINITY),
                    ] {
                        let a = 2.;
                        let b = 0.5;
                        let d = 1.5 + diagonal;
                        let radius = 0.4;
                        let objective = |x: Real, z: Real| {
                            0.5 * (a * x * x + 2. * b * x * z + d * z * z) + u * x + v * z
                        };
                        let mut candidates = Vec::new();
                        let determinant = a * d - b * b;
                        let x = (b * v - d * u) / determinant;
                        let z = (b * u - a * v) / determinant;
                        if x.abs() <= radius && z >= lower && z <= upper {
                            candidates.push((x, z));
                        }
                        for x in [-radius, radius] {
                            candidates.push((x, (-(v + b * x) / d).clamp(lower, upper)));
                        }
                        for z in [lower, upper].into_iter().filter(|z| z.is_finite()) {
                            candidates.push(((-(u + b * z) / a).clamp(-radius, radius), z));
                        }
                        let expected = candidates
                            .into_iter()
                            .min_by(|&(x, z), &(y, w)| objective(x, z).total_cmp(&objective(y, w)))
                            .unwrap();
                        let gram = [a, 0., b, 0., 1., 0., b, 0., d].map(|g| g / scale);
                        let solution = solve(
                            &gram,
                            &[u, 0., v],
                            &[
                                Domain::Disk {
                                    static_cap: 0.6 * scale,
                                    dynamic_cap: radius * scale,
                                    moving: true,
                                },
                                Domain::Interval {
                                    lower: lower * scale,
                                    upper: upper * scale,
                                },
                            ],
                        )
                        .unwrap();
                        for (actual, expected) in
                            solution.impulses.iter().zip([expected.0, 0., expected.1])
                        {
                            assert!(
                                (actual / scale - expected).abs() < 2e-9,
                                "{actual} != {expected} at {scale}, {diagonal}, {u}, {v}, [{lower}, {upper}]"
                            );
                        }
                        assert!(solution.dynamic[0]);
                        assert!(!solution.dynamic[1]);
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_null_response_effort_and_invalid_domains_are_explicit() {
        for (free, lower, upper, expected) in [
            (7., 0.3, 0.3, 0.3),
            (7., -0.3, 0.3, -0.3),
            (-7., -0.3, 0.3, 0.3),
            (0., 0.2, 0.4, 0.2),
            (0., Real::NEG_INFINITY, Real::INFINITY, 0.),
        ] {
            assert_eq!(
                solve(&[0.], &[free], &[Domain::Interval { lower, upper }])
                    .unwrap()
                    .impulses,
                vec![expected]
            );
        }
        for (lower, upper) in [
            (0.3, -0.3),
            (Real::NAN, 0.3),
            (-0.3, Real::NAN),
            (Real::INFINITY, Real::INFINITY),
            (Real::NEG_INFINITY, Real::NEG_INFINITY),
        ] {
            assert!(solve(&[1.], &[0.], &[Domain::Interval { lower, upper }]).is_err());
        }
        assert!(
            solve(
                &[0.],
                &[1.],
                &[Domain::Interval {
                    lower: Real::NEG_INFINITY,
                    upper: Real::INFINITY
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn mixed_disk_scalar_solution_and_infeasible_rows() {
        for scale in [1e-12, 1e-6, 1., 1e6, 1e12] {
            let gram: Vec<_> = [1., 0., -1., 0., 1., 1., -1., 1., 2.]
                .into_iter()
                .map(|x| x / scale)
                .collect();
            let domains = [
                Domain::Disk {
                    static_cap: 0.0025 * scale,
                    dynamic_cap: 0.002 * scale,
                    moving: true,
                },
                Domain::Bilateral,
            ];
            let result = solve(&gram, &[0., 0.01, 0.01], &domains).unwrap();
            let disk = -0.002 / 2.0_f64.sqrt();
            assert!((result.impulses[0] / scale - disk).abs() < 1e-8);
            assert!((result.impulses[1] / scale - disk).abs() < 1e-8);
            assert!((result.impulses[2] / scale + 0.005).abs() < 1e-8);
        }
        assert!(solve(&[0.], &[0.01], &[Domain::Bilateral]).is_err());
        assert!(solve(&[0.], &[-0.01], &[Domain::Nonnegative]).is_err());
        assert_eq!(
            solve(&[0.], &[0.01], &[Domain::Nonnegative])
                .unwrap()
                .impulses,
            vec![0.]
        );
    }

    #[test]
    fn unilateral_joint_activation_and_static_promotion_use_global_solution() {
        let disk = Domain::Disk {
            static_cap: 0.0025,
            dynamic_cap: 0.002,
            moving: true,
        };
        let gram = [1., 0., -1., 0., 1., 1., -1., 1., 2.];
        let inactive = solve(&gram, &[0., 0.01, 0.01], &[disk, Domain::Nonnegative]).unwrap();
        assert_eq!(inactive.impulses[2], 0.);
        assert!((inactive.impulses[1] + 0.002).abs() < 1e-10);
        let active = solve(&gram, &[0., -0.01, -0.01], &[disk, Domain::Nonnegative]).unwrap();
        assert!((active.impulses[2] - 0.005).abs() < 1e-8);
        let rest = Domain::Disk {
            static_cap: 0.0025,
            dynamic_cap: 0.002,
            moving: false,
        };
        let held = solve(
            &[1., 0., 0., 0., 1., 1., 0., 1., 1.],
            &[0., 0.01, 0.01],
            &[rest, Domain::Bilateral],
        )
        .unwrap();
        assert!(!held.dynamic[0]);
        assert!(held.sticking[0]);
        assert!((held.impulses[1] + held.impulses[2] + 0.01).abs() < 1e-9);
        let sliding = solve(&[1., 0., 0., 1.], &[0., 0.01], &[rest]).unwrap();
        assert!(sliding.dynamic[0]);
        assert!(!sliding.sticking[0]);
        assert!((sliding.impulses[1] + 0.002).abs() < 1e-10);
    }

    #[test]
    fn satisfied_zero_response_rows_coexist_with_contact_and_faults_are_explicit() {
        let disk = Domain::Disk {
            static_cap: 0.3,
            dynamic_cap: 0.2,
            moving: true,
        };
        let result = solve(
            &[1., 0., 0., 0., 1., 0., 0., 0., 0.],
            &[1., 0., 0.],
            &[disk, Domain::Bilateral],
        )
        .unwrap();
        assert!((result.impulses[0] + 0.2).abs() < 1e-12);
        assert_eq!(result.impulses[2], 0.);
        assert!(
            solve(
                &[1., -1., -1., 1.],
                &[-1., -1.],
                &[Domain::Nonnegative, Domain::Nonnegative]
            )
            .is_err()
        );
        assert!(solve(&[1., 2., 2., 1.], &[0., 0.], &[disk]).is_err());
        assert!(solve(&[Real::NAN], &[0.], &[Domain::Bilateral]).is_err());
        assert!(solve(&[1.], &[0.], &[disk]).is_err());
    }
}
