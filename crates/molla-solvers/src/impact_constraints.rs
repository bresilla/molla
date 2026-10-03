//! Dense CPU references for coupled impact and non-associated Coulomb response.
//!
//! Callers supply event-local incoming speeds and predictor velocities in an
//! orthonormal normal/tangent frame. Contact episode tracking, response assembly,
//! time integration and redundant-impulse allocation are outside this module.
//! Newton normal restitution is not an energy certificate for coupled oblique impacts.
//! The midpoint variant pairs friction impulses with average contact velocity.

use molla_core::{Error, Result};
use molla_math::{Real, dense_solve::dense_eig_sym};

/// One contact occupies normal, tangent-1 and tangent-2 rows, in that order.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub static_friction: Real,
    pub dynamic_friction: Real,
    pub moving: bool,
    /// Known compressive predictor impulse, already included in the free velocity.
    pub base_normal_impulse: Real,
    pub restitution: Real,
    /// Normal relative velocity at impact onset; negative means approaching.
    pub incoming_normal_velocity: Real,
}

/// Scalar joint rows share the contact response matrix.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Contact(Contact),
    Bilateral,
    Nonnegative,
    Interval { lower: Real, upper: Real },
}

impl Domain {
    fn project(self, value: Real) -> Real {
        match self {
            Self::Bilateral => value,
            Self::Interval { lower, upper } => value.clamp(lower, upper),
            _ => value.max(0.0),
        }
    }

    fn residual(self, impulse: Real, velocity: Real) -> Real {
        match self {
            Self::Interval { lower, upper } if lower == upper => 0.0,
            Self::Interval { lower, .. } if impulse == lower => (-velocity).max(0.0),
            Self::Interval { upper, .. } if impulse == upper => velocity.max(0.0),
            Self::Interval { .. } | Self::Bilateral => velocity.abs(),
            _ if impulse > 0.0 => velocity.abs(),
            _ => (-velocity).max(0.0),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Contact(_) => 3,
            _ => 1,
        }
    }
}

/// Contact flags are indexed by block; scalar flags are always false.
#[derive(Debug)]
pub struct Solution {
    /// Additional impulses; known predictor impulses are not repeated here.
    pub impulses: Vec<Real>,
    pub dynamic: Vec<bool>,
    /// Zero friction-law velocity, which need not be zero final slip for midpoint impacts.
    pub sticking: Vec<bool>,
}

#[derive(Clone, Copy)]
enum FrictionVelocity {
    PostImpact,
    Midpoint,
}

impl FrictionVelocity {
    fn factor(self) -> Real {
        match self {
            Self::PostImpact => 1.,
            Self::Midpoint => 0.5,
        }
    }
}

fn validate(gram: &[Real], free: &[Real], domains: &[Domain]) -> Result<()> {
    let invalid = || Error::Solver("invalid impact constraint problem".into());
    let n = free.len();
    if domains.iter().map(|d| d.width()).sum::<usize>() != n
        || n.checked_mul(n) != Some(gram.len())
        || gram.iter().chain(free).any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    for domain in domains {
        if let Domain::Interval { lower, upper } = *domain
            && (lower.is_nan()
                || upper.is_nan()
                || lower > upper
                || lower == Real::INFINITY
                || upper == Real::NEG_INFINITY)
        {
            return Err(invalid());
        }
        if let Domain::Contact(c) = domain
            && ([
                c.static_friction,
                c.dynamic_friction,
                c.base_normal_impulse,
                c.restitution,
                c.incoming_normal_velocity,
            ]
            .iter()
            .any(|v| !v.is_finite())
                || c.dynamic_friction < 0.
                || c.static_friction < c.dynamic_friction
                || c.base_normal_impulse < 0.
                || !(0. ..=1.).contains(&c.restitution))
        {
            return Err(invalid());
        }
    }
    let mut scaled = vec![0.; gram.len()];
    for i in 0..n {
        if gram[i * n + i] < 0. {
            return Err(invalid());
        }
        for j in 0..n {
            let scale = gram[i * n + i].sqrt() * gram[j * n + j].sqrt();
            if !scale.is_finite()
                || (gram[i * n + j] - gram[j * n + i]).abs() > 1e-12 * scale
                || gram[i * n + j].abs() > scale * (1. + 1e-12)
            {
                return Err(invalid());
            }
            if scale > 0. {
                scaled[i * n + j] = gram[i * n + j] / scale;
            }
        }
    }
    let (values, basis) = dense_eig_sym(&scaled, n)?;
    if values.iter().any(|v| !v.is_finite() || *v < -1e-10) {
        return Err(invalid());
    }
    for i in 0..n {
        for j in 0..n {
            let residual = (0..n)
                .map(|k| scaled[i * n + k] * basis[k * n + j])
                .sum::<Real>()
                - basis[i * n + j] * values[j];
            if !residual.is_finite() || residual.abs() > 1e-10 * n as Real {
                return Err(Error::Solver("impact response eigensolve failed".into()));
            }
        }
    }
    Ok(())
}

fn contact_block(a: [Real; 6], free: [Real; 3], mu: Real, base: Real) -> Result<[Real; 3]> {
    let evaluate = |normal: Real| -> Result<(Real, [Real; 3], Real)> {
        let velocity = [free[1] + a[1] * normal, free[2] + a[2] * normal];
        let radius = mu * (base + normal);
        let scale = velocity[0].hypot(velocity[1]).max(1.);
        if !radius.is_finite() || !scale.is_finite() || velocity.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver("nonfinite impact disk".into()));
        }
        let (tangent, _) = crate::contact_friction::impulse(
            [a[3] / scale, a[4] / scale, a[5] / scale],
            velocity.map(|v| v / scale),
            radius,
            radius,
        );
        let response = [a[0] * normal, a[1] * tangent[0], a[2] * tangent[1]];
        let residual = free[0] + response.iter().sum::<Real>();
        let tolerance =
            1e-12 * (1. + free[0].abs() + response.iter().map(|v| v.abs()).sum::<Real>());
        if !residual.is_finite() || !tolerance.is_finite() || tangent.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Solver("nonfinite local impact response".into()));
        }
        Ok((residual, [normal, tangent[0], tangent[1]], tolerance))
    };
    let (initial, zero, _) = evaluate(0.)?;
    if initial >= 0. {
        return Ok(zero);
    }
    if a[0] == 0. {
        return Err(Error::Solver("infeasible immovable impact normal".into()));
    }
    let mut low = 0.;
    let mut high = -initial / a[0];
    let mut bracketed = false;
    for _ in 0..128 {
        let (residual, value, tolerance) = evaluate(high)?;
        if residual.abs() <= tolerance {
            return Ok(value);
        }
        if residual > 0. {
            bracketed = true;
            break;
        }
        low = high;
        high *= 2.;
    }
    if !bracketed {
        return Err(Error::Solver(
            "impact normal root could not be bracketed".into(),
        ));
    }
    for _ in 0..128 {
        let middle = 0.5 * low + 0.5 * high;
        let (residual, value, tolerance) = evaluate(middle)?;
        if residual.abs() <= tolerance {
            return Ok(value);
        }
        if middle == low || middle == high {
            break;
        }
        if residual > 0. {
            high = middle;
        } else {
            low = middle;
        }
    }
    Err(Error::Solver(
        "local impact normal response did not converge".into(),
    ))
}

fn sticking_active_set(
    gram: &[Real],
    free: &[Real],
    domains: &[Domain],
    dynamic: &[bool],
) -> Option<Vec<Real>> {
    let unilateral_count = domains
        .iter()
        .filter(|d| !matches!(d, Domain::Bilateral))
        .count();
    if domains.is_empty()
        || unilateral_count > 8
        || domains.iter().any(|d| matches!(d, Domain::Interval { .. }))
        || domains
            .iter()
            .any(|d| matches!(d, Domain::Contact(c) if c.base_normal_impulse != 0.))
    {
        return None;
    }
    let n = free.len();
    for mask in 0usize..(1 << unilateral_count) {
        let mut selected = Vec::new();
        let mut r = 0;
        let mut bit = 0;
        for (b, &domain) in domains.iter().enumerate() {
            let active = if matches!(domain, Domain::Bilateral) {
                true
            } else {
                let active = mask & (1 << bit) != 0;
                bit += 1;
                active
            };
            if active {
                selected.push(r);
                if let Domain::Contact(c) = domain {
                    let mu = if dynamic[b] {
                        c.dynamic_friction
                    } else {
                        c.static_friction
                    };
                    if mu > 0. {
                        selected.extend([r + 1, r + 2]);
                    }
                }
            }
            r += domain.width();
        }
        let size = selected.len();
        let matrix: Vec<_> = selected
            .iter()
            .flat_map(|&i| selected.iter().map(move |&j| gram[i * n + j]))
            .collect();
        let rhs: Vec<_> = selected.iter().map(|&i| -free[i]).collect();
        let Ok(solved) = crate::rigid_impulse::solve_mass(&matrix, &rhs, size) else {
            continue;
        };
        let mut lambda = vec![0.; n];
        for (&i, value) in selected.iter().zip(solved) {
            lambda[i] = value;
        }
        if lambda.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mut valid = true;
        let mut r = 0;
        for (b, &domain) in domains.iter().enumerate() {
            let bilateral = matches!(domain, Domain::Bilateral);
            let radius = if let Domain::Contact(c) = domain {
                let mu = if dynamic[b] {
                    c.dynamic_friction
                } else {
                    c.static_friction
                };
                mu * lambda[r]
            } else {
                0.
            };
            if (!bilateral && lambda[r] < 0.)
                || !radius.is_finite()
                || (matches!(domain, Domain::Contact(_))
                    && lambda[r + 1].hypot(lambda[r + 2]) > radius * (1. + 1e-12))
            {
                valid = false;
                break;
            }
            for i in r..r + domain.width() {
                let response: Real = (0..n).map(|j| gram[i * n + j] * lambda[j]).sum();
                let residual = free[i] + response;
                let scale = 1.
                    + free[i].abs()
                    + (0..n)
                        .map(|j| (gram[i * n + j] * lambda[j]).abs())
                        .sum::<Real>();
                let error = if i == r && !bilateral && lambda[r] == 0. {
                    (-residual).max(0.)
                } else if i == r || radius > 0. {
                    residual.abs()
                } else {
                    0.
                };
                if !residual.is_finite() || !scale.is_finite() || error > 1e-10 * scale {
                    valid = false;
                    break;
                }
            }
            r += domain.width();
        }
        if valid {
            return Some(lambda);
        }
    }
    None
}

/// Solve normal complementarity and maximum-dissipation tangent disks jointly.
///
/// `gram` is row-major `J M^-1 J^T`; `free` includes predictor and boundary motion.
/// Normal residual is `v_n + e * min(v_incoming, 0)`. Each tangent disk has radius
/// `mu * (base_normal_impulse + solved_normal_impulse)`. Normal complementarity
/// does not include the derivative of that radius. Nonconvergence is an error.
pub fn solve(gram: &[Real], free: &[Real], domains: &[Domain]) -> Result<Solution> {
    solve_with_velocity(gram, free, domains, FrictionVelocity::PostImpact)
}

/// Solve impact friction against the average pre/post impulse relative velocity.
///
/// Tangent residuals are `free_t + (W * impulse)_t / 2`; the normal target is
/// unchanged. Contact normals must be approaching or stationary, and incoming
/// speeds must equal `free_n`. Separating contacts are excluded by the caller's
/// impact active set. Scalar boundary work is separate. This does not integrate
/// compliant contact energy or track impact episodes.
pub fn solve_midpoint(gram: &[Real], free: &[Real], domains: &[Domain]) -> Result<Solution> {
    solve_with_velocity(gram, free, domains, FrictionVelocity::Midpoint)
}

fn solve_with_velocity(
    gram: &[Real],
    free: &[Real],
    domains: &[Domain],
    friction_velocity: FrictionVelocity,
) -> Result<Solution> {
    validate(gram, free, domains)?;
    let n = free.len();
    let mut shifted = free.to_vec();
    let mut offsets = Vec::with_capacity(domains.len());
    let mut dynamic = Vec::with_capacity(domains.len());
    let mut first = 0;
    for &domain in domains {
        offsets.push(first);
        if let Domain::Contact(c) = domain {
            if matches!(friction_velocity, FrictionVelocity::Midpoint)
                && (free[first] > 0. || c.incoming_normal_velocity != free[first])
            {
                return Err(Error::Solver(
                    "midpoint impact requires current approaching normal velocities".into(),
                ));
            }
            shifted[first] += c.restitution * c.incoming_normal_velocity.min(0.);
            if matches!(friction_velocity, FrictionVelocity::Midpoint) {
                shifted[first + 1] *= 2.;
                shifted[first + 2] *= 2.;
            }
            dynamic.push(c.moving);
        } else {
            dynamic.push(false);
        }
        first += domain.width();
    }
    if shifted.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver("nonfinite impact target".into()));
    }
    let gradient = |i: usize, lambda: &[Real]| {
        shifted[i] + (0..n).map(|j| gram[i * n + j] * lambda[j]).sum::<Real>()
    };
    let scale = |i: usize, lambda: &[Real]| {
        1. + shifted[i].abs()
            + (0..n)
                .map(|j| (gram[i * n + j] * lambda[j]).abs())
                .sum::<Real>()
    };
    let cap = |c: Contact, normal: Real, dynamic: bool| {
        let mu = if dynamic {
            c.dynamic_friction
        } else {
            c.static_friction
        };
        mu * (c.base_normal_impulse + normal)
    };
    let mut lambda = vec![0.; n];
    let mut sticking = vec![false; domains.len()];
    for _ in 0..=domains.len() {
        lambda.fill(0.);
        let mut converged = false;
        for _ in 0..32768 {
            for (b, &domain) in domains.iter().enumerate() {
                let r = offsets[b];
                if let Domain::Contact(c) = domain {
                    let mut velocity = [shifted[r], shifted[r + 1], shifted[r + 2]];
                    for j in 0..n {
                        if !(r..r + 3).contains(&j) {
                            for i in 0..3 {
                                velocity[i] += gram[(r + i) * n + j] * lambda[j];
                            }
                        }
                    }
                    let a = [
                        gram[r * n + r],
                        gram[r * n + r + 1],
                        gram[r * n + r + 2],
                        gram[(r + 1) * n + r + 1],
                        gram[(r + 1) * n + r + 2],
                        gram[(r + 2) * n + r + 2],
                    ];
                    let mu = if dynamic[b] {
                        c.dynamic_friction
                    } else {
                        c.static_friction
                    };
                    lambda[r..r + 3].copy_from_slice(&contact_block(
                        a,
                        velocity,
                        mu,
                        c.base_normal_impulse,
                    )?);
                } else {
                    let diagonal = gram[r * n + r];
                    let velocity = gradient(r, &lambda);
                    let next = if diagonal > 0. {
                        lambda[r] - velocity / diagonal
                    } else if let Domain::Interval { lower, upper } = domain {
                        if velocity > 0.0 {
                            lower
                        } else if velocity < 0.0 {
                            upper
                        } else {
                            0.0
                        }
                    } else {
                        lambda[r]
                    };
                    lambda[r] = domain.project(next);
                }
            }
            if lambda.iter().any(|v| !v.is_finite()) {
                return Err(Error::Solver("nonfinite impact impulse".into()));
            }
            let mut residual: Real = 0.;
            for (b, &domain) in domains.iter().enumerate() {
                let r = offsets[b];
                let v = gradient(r, &lambda);
                let normal_error = domain.residual(lambda[r], v);
                let normal_scale = scale(r, &lambda);
                if !v.is_finite() || !normal_scale.is_finite() {
                    return Err(Error::Solver("nonfinite impact residual".into()));
                }
                residual = residual.max(normal_error / normal_scale);
                if let Domain::Contact(c) = domain {
                    let t = r + 1;
                    let velocity = [gradient(t, &lambda), gradient(t + 1, &lambda)];
                    let tangent_scale = scale(t, &lambda).max(scale(t + 1, &lambda));
                    let radius = cap(c, lambda[r], dynamic[b]);
                    let length = lambda[t].hypot(lambda[t + 1]);
                    if !radius.is_finite()
                        || !length.is_finite()
                        || !tangent_scale.is_finite()
                        || velocity.iter().any(|v| !v.is_finite())
                    {
                        return Err(Error::Solver("nonfinite impact tangent residual".into()));
                    }
                    sticking[b] =
                        velocity[0].hypot(velocity[1]) * friction_velocity.factor() <= 1e-9;
                    if length > radius * (1. + 1e-12) {
                        residual = Real::INFINITY;
                        continue;
                    }
                    if radius == 0. {
                        continue;
                    }
                    if length == 0. {
                        residual = residual.max(velocity[0].hypot(velocity[1]) / tangent_scale);
                        continue;
                    }
                    let direction = [lambda[t] / length, lambda[t + 1] / length];
                    let multiplier =
                        (-(velocity[0] * direction[0] + velocity[1] * direction[1])).max(0.);
                    let stationarity = (velocity[0] + multiplier * direction[0])
                        .hypot(velocity[1] + multiplier * direction[1]);
                    let complementarity = multiplier * ((radius - length).max(0.) / radius);
                    residual = residual.max(stationarity.max(complementarity) / tangent_scale);
                }
            }
            if residual <= 1e-10 {
                converged = true;
                break;
            }
        }
        if !converged {
            if let Some(candidate) = sticking_active_set(gram, &shifted, domains, &dynamic) {
                lambda = candidate;
                for (b, &domain) in domains.iter().enumerate() {
                    if matches!(domain, Domain::Contact(_)) {
                        let t = offsets[b] + 1;
                        sticking[b] = gradient(t, &lambda).hypot(gradient(t + 1, &lambda))
                            * friction_velocity.factor()
                            <= 1e-9;
                    }
                }
            } else {
                return Err(Error::Solver(
                    "coupled impact response did not converge".into(),
                ));
            }
        }
        let mut promoted = false;
        for (b, &domain) in domains.iter().enumerate() {
            if matches!(domain, Domain::Contact(_)) && !dynamic[b] {
                let t = offsets[b] + 1;
                if gradient(t, &lambda).hypot(gradient(t + 1, &lambda)) * friction_velocity.factor()
                    > 1e-4
                {
                    dynamic[b] = true;
                    promoted = true;
                }
            }
        }
        if !promoted {
            return Ok(Solution {
                impulses: lambda,
                dynamic,
                sticking,
            });
        }
    }
    Err(Error::Solver(
        "impact friction mode promotion did not converge".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_scalars_match_independent_box_oracle_with_bilateral_coupling() {
        for mass in [0.01, 1.0, 100.0] {
            for speed in [-3.0, 0.0, 3.0] {
                for limits in [
                    [-0.7, 0.7],
                    [-2.0, 2.0],
                    [0.2, 0.2],
                    [0.0, 0.7],
                    [-0.7, 0.0],
                    [0.0, 0.0],
                    [Real::NEG_INFINITY, Real::INFINITY],
                    [Real::NEG_INFINITY, 0.7],
                    [-0.7, Real::INFINITY],
                ] {
                    let [lower, upper] = limits.map(|value| value * mass);
                    let gram = [2.0 / mass, 0.5 / mass, 0.5 / mass, 1.5 / mass];
                    let free = [speed, 0.4];
                    let expected = ((-speed + gram[1] * free[1] / gram[3])
                        / (gram[0] - gram[1] * gram[1] / gram[3]))
                        .clamp(lower, upper);
                    let domains = [Domain::Interval { lower, upper }, Domain::Bilateral];
                    for solution in [
                        solve(&gram, &free, &domains),
                        solve_midpoint(&gram, &free, &domains),
                    ] {
                        let actual = solution.unwrap();
                        assert!(
                            (actual.impulses[0] - expected).abs() < 1e-8 * (1.0 + expected.abs())
                        );
                        let second = (-free[1] - gram[1] * expected) / gram[3];
                        assert!((actual.impulses[1] - second).abs() < 1e-8 * (1.0 + second.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_scalars_handle_prescribed_null_effort_and_reject_invalid_domains() {
        for free in [-7.0, 0.0, 7.0] {
            let result = solve_midpoint(
                &[0.0],
                &[free],
                &[Domain::Interval {
                    lower: 0.3,
                    upper: 0.3,
                }],
            )
            .unwrap();
            assert_eq!(result.impulses, [0.3]);
        }
        for [lower, upper] in [
            [1.0, -1.0],
            [Real::NAN, 1.0],
            [0.0, Real::NAN],
            [Real::INFINITY, Real::INFINITY],
            [Real::NEG_INFINITY, Real::NEG_INFINITY],
        ] {
            assert!(solve(&[1.0], &[0.0], &[Domain::Interval { lower, upper }]).is_err());
        }
        assert!(
            solve(
                &[0.0],
                &[-1.0],
                &[Domain::Interval {
                    lower: Real::NEG_INFINITY,
                    upper: Real::INFINITY
                }]
            )
            .is_err()
        );
        assert_eq!(
            solve(
                &[0.0],
                &[-1.0],
                &[Domain::Interval {
                    lower: -0.2,
                    upper: 0.4
                }]
            )
            .unwrap()
            .impulses,
            [0.4]
        );
    }

    fn contact() -> Contact {
        Contact {
            static_friction: 0.6,
            dynamic_friction: 0.4,
            moving: false,
            base_normal_impulse: 0.,
            restitution: 0.,
            incoming_normal_velocity: -1.,
        }
    }

    fn diagonal(n: usize, value: Real) -> Vec<Real> {
        let mut result = vec![0.; n * n];
        for i in 0..n {
            result[i * n + i] = value;
        }
        result
    }

    fn close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() <= 2e-8 * expected.abs().max(1e-12),
            "actual={actual:e}, expected={expected:e}"
        );
    }

    #[test]
    fn newton_rebound_mass_speed_sweep_and_energy() {
        for mass in [1e-9, 1e-3, 1., 1e3, 1e9] {
            for speed in [1e-6, 1., 1e6] {
                for restitution in [0., 0.25, 0.5, 1.] {
                    let c = Contact {
                        restitution,
                        incoming_normal_velocity: -speed,
                        ..contact()
                    };
                    let result = solve(
                        &diagonal(3, 1. / mass),
                        &[-speed, 0., 0.],
                        &[Domain::Contact(c)],
                    )
                    .unwrap();
                    let impulse = result.impulses[0];
                    close(impulse, mass * speed * (1. + restitution));
                    let rebound = -speed + impulse / mass;
                    assert!((rebound - speed * restitution).abs() <= 1e-12 * speed);
                    assert!(
                        0.5 * mass * rebound * rebound <= 0.5 * mass * speed * speed * (1. + 1e-12)
                    );
                }
            }
        }
    }

    #[test]
    fn two_body_impact_conserves_momentum_and_measures_energy_loss() {
        for (a, b) in [(1., 1.), (1e-3, 1e3), (1e3, 1e-3)] {
            for restitution in [0., 0.5, 1.] {
                let c = Contact {
                    restitution,
                    ..contact()
                };
                let inverse_mass = 1. / a + 1. / b;
                let result = solve(
                    &diagonal(3, inverse_mass),
                    &[-1., 0., 0.],
                    &[Domain::Contact(c)],
                )
                .unwrap();
                let impulse = result.impulses[0];
                let va = 1. - impulse / a;
                let vb = impulse / b;
                close(a * va + b * vb, a);
                assert!((vb - va - restitution).abs() < 1e-12);
                let lost = 0.5 * a - 0.5 * (a * va * va + b * vb * vb);
                let expected = 0.5 * (1. - restitution * restitution) / inverse_mass;
                assert!((lost - expected).abs() < 1e-12 * a.max(b));
            }
        }
    }

    #[test]
    fn sliding_cap_includes_base_and_solved_normal_impulse() {
        for base in [0., 0.25, 2.] {
            let c = Contact {
                base_normal_impulse: base,
                restitution: 0.5,
                moving: true,
                ..contact()
            };
            let result = solve(&diagonal(3, 1.), &[-1., 10., 0.], &[Domain::Contact(c)]).unwrap();
            close(result.impulses[0], 1.5);
            close(result.impulses[1], -0.4 * (base + 1.5));
            assert!(result.dynamic[0]);
            assert!(!result.sticking[0]);
        }
    }

    #[test]
    fn normal_tangent_coupling_has_no_associated_cone_dilatancy() {
        let gram = [2., 0.5, 0., 0.5, 1., 0., 0., 0., 1.];
        let c = Contact {
            restitution: 0.5,
            moving: true,
            ..contact()
        };
        let result = solve(&gram, &[-1., 3., 0.], &[Domain::Contact(c)]).unwrap();
        close(result.impulses[0], 5. / 6.);
        close(result.impulses[1], -1. / 3.);
        close(
            -1. + 2. * result.impulses[0] + 0.5 * result.impulses[1],
            0.5,
        );
    }

    #[test]
    fn static_capture_promotion_and_initial_dynamic_mode() {
        let gram = diagonal(3, 1.);
        let held = solve(&gram, &[-1., 0.5, 0.], &[Domain::Contact(contact())]).unwrap();
        close(held.impulses[1], -0.5);
        assert!(!held.dynamic[0] && held.sticking[0]);
        let sliding = solve(&gram, &[-1., 0.8, 0.], &[Domain::Contact(contact())]).unwrap();
        close(sliding.impulses[1], -0.4);
        assert!(sliding.dynamic[0] && !sliding.sticking[0]);
        let moving = Contact {
            moving: true,
            ..contact()
        };
        let stopped = solve(&gram, &[-1., 0.3, 0.], &[Domain::Contact(moving)]).unwrap();
        close(stopped.impulses[1], -0.3);
        assert!(stopped.dynamic[0] && stopped.sticking[0]);
    }

    #[test]
    fn joint_coupling_is_solved_before_friction_mode_promotion() {
        let mut gram = diagonal(4, 1.);
        gram[7] = 0.5;
        gram[13] = 0.5;
        let c = Contact {
            static_friction: 0.1,
            dynamic_friction: 0.05,
            ..contact()
        };
        let result = solve(
            &gram,
            &[-1., 2., 0., 4.],
            &[Domain::Contact(c), Domain::Bilateral],
        )
        .unwrap();
        close(result.impulses[0], 1.);
        close(result.impulses[3], -4.);
        assert!(result.impulses[1].abs() < 1e-9);
        assert!(!result.dynamic[0] && result.sticking[0]);
        assert!(!result.dynamic[1] && !result.sticking[1]);
    }

    #[test]
    fn nonnegative_joint_couples_to_restitution_normal() {
        let mut gram = diagonal(4, 1.);
        gram[3] = 0.25;
        gram[12] = 0.25;
        let c = Contact {
            restitution: 0.5,
            ..contact()
        };
        let result = solve(
            &gram,
            &[-1., 0., 0., -1.],
            &[Domain::Contact(c), Domain::Nonnegative],
        )
        .unwrap();
        close(result.impulses[0], 4. / 3.);
        close(result.impulses[3], 2. / 3.);
    }

    #[test]
    fn separated_and_zero_load_contacts_cannot_generate_friction() {
        for restitution in [0., 0.5, 1.] {
            let c = Contact {
                restitution,
                incoming_normal_velocity: 1.,
                ..contact()
            };
            let result = solve(&diagonal(3, 1.), &[1., 5., -2.], &[Domain::Contact(c)]).unwrap();
            assert_eq!(result.impulses, [0., 0., 0.]);
        }
        let c = Contact {
            static_friction: 0.,
            dynamic_friction: 0.,
            ..contact()
        };
        let result = solve(&diagonal(3, 1.), &[-1., 5., -2.], &[Domain::Contact(c)]).unwrap();
        assert_eq!(result.impulses, [1., 0., 0.]);
    }

    #[test]
    fn tangent_basis_rotation_preserves_world_impulse() {
        let original = [2., 0.3, -0.2, 0.3, 1., 0.1, -0.2, 0.1, 0.7];
        let free = [-1., 3., -2.];
        let c = Contact {
            restitution: 0.4,
            moving: true,
            ..contact()
        };
        let reference = solve(&original, &free, &[Domain::Contact(c)]).unwrap();
        for angle in [0.3_f64, 1.2, 2.8] {
            let (s, ctheta) = angle.sin_cos();
            let rotation = [1., 0., 0., 0., ctheta, -s, 0., s, ctheta];
            let mut gram = [0.; 9];
            let mut velocity = [0.; 3];
            for i in 0..3 {
                for j in 0..3 {
                    velocity[i] += rotation[i * 3 + j] * free[j];
                    for k in 0..3 {
                        for l in 0..3 {
                            gram[i * 3 + j] +=
                                rotation[i * 3 + k] * original[k * 3 + l] * rotation[j * 3 + l];
                        }
                    }
                }
            }
            let result = solve(&gram, &velocity, &[Domain::Contact(c)]).unwrap();
            for i in 0..3 {
                let world: Real = (0..3)
                    .map(|j| rotation[j * 3 + i] * result.impulses[j])
                    .sum();
                close(world, reference.impulses[i]);
            }
        }
    }

    #[test]
    fn rank_deficient_tangent_response_remains_bounded() {
        let gram = [1., 0., 0., 0., 1., 1., 0., 1., 1.];
        let c = Contact {
            moving: true,
            ..contact()
        };
        let result = solve(&gram, &[-1., 2., 2.], &[Domain::Contact(c)]).unwrap();
        close(result.impulses[1], -0.4 / 2_f64.sqrt());
        close(result.impulses[2], result.impulses[1]);
        let zero = solve(
            &[0.; 9],
            &[0., 2., 0.],
            &[Domain::Contact(Contact {
                base_normal_impulse: 1.,
                ..c
            })],
        )
        .unwrap();
        close(zero.impulses[1], -0.4);
    }

    #[test]
    fn coupled_contacts_preserve_response_under_block_permutation() {
        let mut gram = diagonal(6, 1.);
        for (i, j, value) in [(0, 3, 0.25), (1, 4, 0.2), (0, 4, 0.1), (2, 5, -0.2)] {
            gram[i * 6 + j] = value;
            gram[j * 6 + i] = value;
        }
        for sliding in [false, true] {
            let c = Contact {
                restitution: 0.5,
                moving: sliding,
                ..contact()
            };
            let expected = if sliding {
                [1., -0.4, 0., 2., 0.8, 0.]
            } else {
                [1., -0.2, 0.1, 2., 0.3, -0.2]
            };
            let outgoing = if sliding {
                [0.5, 1., 0., 0.5, -2., 0.]
            } else {
                [0.5, 0., 0., 0.5, 0., 0.]
            };
            let free: Vec<_> = (0..6)
                .map(|i| outgoing[i] - (0..6).map(|j| gram[i * 6 + j] * expected[j]).sum::<Real>())
                .collect();
            let reference = solve(&gram, &free, &[Domain::Contact(c); 2]).unwrap();
            for i in 0..6 {
                assert!((reference.impulses[i] - expected[i]).abs() < 1e-8);
            }
            let permutation = [3, 4, 5, 0, 1, 2];
            let mut permuted = [0.; 36];
            for i in 0..6 {
                for j in 0..6 {
                    permuted[i * 6 + j] = gram[permutation[i] * 6 + permutation[j]];
                }
            }
            let free = permutation.map(|i| free[i]);
            let result = solve(&permuted, &free, &[Domain::Contact(c); 2]).unwrap();
            for i in 0..6 {
                assert!((result.impulses[i] - reference.impulses[permutation[i]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn newton_normal_restitution_alone_is_not_an_oblique_energy_certificate() {
        let gram = [1., 0.8, 0., 0.8, 1., 0., 0., 0., 1.];
        let free = [-1., -0.8, 0.];
        let c = Contact {
            restitution: 1.,
            moving: true,
            ..contact()
        };
        let result = solve(&gram, &free, &[Domain::Contact(c)]).unwrap();
        close(result.impulses[0], 2. / 0.68);
        close(result.impulses[1], -0.4 * result.impulses[0]);
        let energy_change: Real = (0..3)
            .map(|i| {
                result.impulses[i]
                    * (free[i]
                        + 0.5
                            * (0..3)
                                .map(|j| gram[i * 3 + j] * result.impulses[j])
                                .sum::<Real>())
            })
            .sum();
        assert!(
            energy_change > 0.24 && energy_change < 0.25,
            "{energy_change}"
        );
    }

    #[test]
    fn target_and_capacity_overflow_fail_closed() {
        let target = Contact {
            restitution: 1.,
            incoming_normal_velocity: -Real::MAX,
            ..contact()
        };
        assert!(
            solve(
                &diagonal(3, 1.),
                &[-Real::MAX, 0., 0.],
                &[Domain::Contact(target)]
            )
            .is_err()
        );
        let cap = Contact {
            static_friction: Real::MAX,
            dynamic_friction: Real::MAX,
            base_normal_impulse: 2.,
            ..contact()
        };
        assert!(solve(&diagonal(3, 1.), &[0.; 3], &[Domain::Contact(cap)]).is_err());
    }

    fn outgoing(gram: &[Real], free: &[Real], result: &Solution) -> Vec<Real> {
        let n = free.len();
        (0..n)
            .map(|i| {
                free[i]
                    + (0..n)
                        .map(|j| gram[i * n + j] * result.impulses[j])
                        .sum::<Real>()
            })
            .collect()
    }

    fn work(free: &[Real], after: &[Real], result: &Solution) -> Real {
        (0..free.len())
            .map(|i| 0.5 * result.impulses[i] * (free[i] + after[i]))
            .sum()
    }

    #[test]
    fn midpoint_friction_removes_the_oblique_energy_counterexample() {
        let gram = [1., 0.8, 0., 0.8, 1., 0., 0., 0., 1.];
        let free = [-1., -0.8, 0.];
        let c = Contact {
            restitution: 1.,
            moving: true,
            ..contact()
        };
        let post = solve(&gram, &free, &[Domain::Contact(c)]).unwrap();
        assert!(work(&free, &outgoing(&gram, &free, &post), &post) > 0.24);
        let midpoint = solve_midpoint(&gram, &free, &[Domain::Contact(c)]).unwrap();
        let after = outgoing(&gram, &free, &midpoint);
        close(midpoint.impulses[0], 2.);
        assert!(midpoint.impulses[1].abs() < 1e-8);
        close(after[0], 1.);
        close(after[1], 0.8);
        assert!(work(&free, &after, &midpoint).abs() < 1e-8);
    }

    #[test]
    fn midpoint_passive_oblique_impacts_dissipate_across_mass_friction_and_restitution() {
        for mass in [1e-6, 1., 1e6] {
            for (b, c) in [(0., 0.), (0.8, 0.), (-0.6, 0.5), (0.95, 0.2)] {
                let gram = [1., b, c, b, 1., 0., c, 0., 1.].map(|v| v / mass);
                for mu in [0.05, 0.4, 2.] {
                    for restitution in [0., 0.5, 1.] {
                        for slip in [-3., -0.8, 0., 0.8, 3.] {
                            let free = [-1., slip, -0.7 * slip];
                            let contact = Contact {
                                static_friction: mu * 1.5,
                                dynamic_friction: mu,
                                restitution,
                                moving: true,
                                ..contact()
                            };
                            let result = solve_midpoint(&gram, &free, &[Domain::Contact(contact)])
                                .unwrap_or_else(|e| panic!("mass={mass}, b={b}, c={c}, mu={mu}, e={restitution}, slip={slip}: {e}"));
                            let after = outgoing(&gram, &free, &result);
                            let energy = work(&free, &after, &result);
                            let tolerance = 1e-8 * mass;
                            assert!(
                                energy <= tolerance,
                                "mass={mass}, mu={mu}, e={restitution}, work={energy}"
                            );
                            assert!(after[0] + 1e-8 >= restitution);
                            assert!(result.impulses[0] >= 0.);
                            assert!(
                                result.impulses[1].hypot(result.impulses[2])
                                    <= mu * result.impulses[0] * (1. + 1e-10)
                            );
                            let tangent_work = 0.5
                                * ((free[1] + after[1]) * result.impulses[1]
                                    + (free[2] + after[2]) * result.impulses[2]);
                            assert!(tangent_work <= tolerance);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn midpoint_capture_reports_average_zero_not_final_zero() {
        let gram = diagonal(3, 1.);
        let free = [-1., 0.2, 0.];
        let result = solve_midpoint(&gram, &free, &[Domain::Contact(contact())]).unwrap();
        let after = outgoing(&gram, &free, &result);
        close(result.impulses[1], -0.4);
        close(after[1], -0.2);
        assert!(result.sticking[0] && !result.dynamic[0]);
        close(work(&free, &after, &result), -0.5);
    }

    #[test]
    fn midpoint_coupled_contacts_and_homogeneous_joints_have_nonpositive_work() {
        let mut gram = diagonal(8, 1.);
        for (i, j, value) in [
            (0, 3, 0.25),
            (1, 4, -0.2),
            (0, 4, 0.1),
            (2, 5, 0.2),
            (0, 6, 0.15),
            (1, 6, 0.2),
            (3, 7, -0.2),
        ] {
            gram[i * 8 + j] = value;
            gram[j * 8 + i] = value;
        }
        for restitution in [0., 0.5, 1.] {
            let first = Contact {
                restitution,
                moving: true,
                ..contact()
            };
            let second = Contact {
                incoming_normal_velocity: -0.7,
                ..first
            };
            let domains = [
                Domain::Contact(first),
                Domain::Contact(second),
                Domain::Bilateral,
                Domain::Nonnegative,
            ];
            let free = [-1., 0.6, -0.3, -0.7, -0.5, 0.4, 0., -0.2];
            let result = solve_midpoint(&gram, &free, &domains).unwrap();
            let after = outgoing(&gram, &free, &result);
            assert!(work(&free, &after, &result) <= 1e-9);
            for r in [0, 3] {
                let normal_work = 0.5 * result.impulses[r] * (free[r] + after[r]);
                let tangent_work = 0.5
                    * ((free[r + 1] + after[r + 1]) * result.impulses[r + 1]
                        + (free[r + 2] + after[r + 2]) * result.impulses[r + 2]);
                assert!(normal_work <= 1e-9 && tangent_work <= 1e-9);
            }
            assert!(after[6].abs() < 1e-9);
            assert!(after[7] >= -1e-9 && result.impulses[7] >= 0.);
        }
    }

    #[test]
    fn midpoint_normal_bounce_matches_authored_coefficient_and_preserves_momentum() {
        for mass in [1e-9, 1., 1e9] {
            for restitution in [0., 0.5, 1.] {
                let gram = diagonal(3, 2. / mass);
                let c = Contact {
                    restitution,
                    ..contact()
                };
                let result = solve_midpoint(&gram, &[-1., 0., 0.], &[Domain::Contact(c)]).unwrap();
                let va = 1. - result.impulses[0] / mass;
                let vb = result.impulses[0] / mass;
                close(mass * (va + vb), mass);
                assert!((vb - va - restitution).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn midpoint_rejects_stale_event_speed_and_separating_active_contacts() {
        let gram = diagonal(3, 1.);
        assert!(solve_midpoint(&gram, &[-0.5, 0., 0.], &[Domain::Contact(contact())]).is_err());
        let separating = Contact {
            incoming_normal_velocity: 1.,
            ..contact()
        };
        assert!(solve_midpoint(&gram, &[1., 0., 0.], &[Domain::Contact(separating)]).is_err());
        let stationary = Contact {
            incoming_normal_velocity: 0.,
            ..contact()
        };
        let result = solve_midpoint(&gram, &[0., 1., 0.], &[Domain::Contact(stationary)]).unwrap();
        assert_eq!(result.impulses, [0., 0., 0.]);
    }

    #[test]
    fn nearly_parallel_contact_normals_release_inactive_impulses() {
        use molla_math::{Quat, Vec3};
        let rotation = Quat::from_rotation_z(-1e-5);
        let rows = [
            Vec3::Y,
            Vec3::Z,
            Vec3::X,
            rotation * Vec3::Y,
            Vec3::Z,
            rotation * Vec3::X,
        ];
        let gram: Vec<_> = rows
            .iter()
            .flat_map(|a| rows.iter().map(|b| a.dot(*b)))
            .collect();
        let free: Vec<_> = rows
            .iter()
            .map(|row| row.dot(Vec3::new(0.1, -1., 0.)))
            .collect();
        let domains: Vec<_> = [0, 3]
            .into_iter()
            .map(|i| {
                Domain::Contact(Contact {
                    incoming_normal_velocity: free[i],
                    moving: true,
                    ..contact()
                })
            })
            .collect();
        let result = solve_midpoint(&gram, &free, &domains).unwrap();
        let after = outgoing(&gram, &free, &result);
        assert!(after[0] >= -1e-10 && after[3] >= -1e-10);
        assert!(result.impulses[0] == 0. || result.impulses[3] == 0.);
        assert!(work(&free, &after, &result) <= 1e-10);
    }

    #[test]
    fn nearly_parallel_scalar_rows_release_inactive_impulses_with_bilateral_rows() {
        let cosine = (1e-5 as Real).cos();
        let gram = [1., 0., cosine, 0., 1., 0., cosine, 0., 1.];
        let free = [-1. + 1e-6, 0.25, -1.];
        let domains = [Domain::Nonnegative, Domain::Bilateral, Domain::Nonnegative];
        let result = solve_midpoint(&gram, &free, &domains).unwrap();
        let after = outgoing(&gram, &free, &result);
        assert_eq!(result.impulses[0], 0.);
        close(result.impulses[1], -0.25);
        close(result.impulses[2], 1.);
        assert!(after[0] >= 0.);
        assert!(after[1].abs() < 1e-10 && after[2].abs() < 1e-10);
    }

    #[test]
    fn active_set_checks_inactive_rows_and_preserves_bilateral_equalities() {
        let gram = [1., 0.5, 0.5, 1.];
        let domains = [Domain::Bilateral, Domain::Nonnegative];
        let lambda = sticking_active_set(&gram, &[1., 1.], &domains, &[false; 2]).unwrap();
        assert_eq!(lambda, [-1., 0.]);
        let lambda = sticking_active_set(&gram, &[1., -1.], &domains, &[false; 2]).unwrap();
        close(lambda[0], -2.);
        close(lambda[1], 2.);
        assert!(sticking_active_set(&[1.; 4], &[1., 0.], &domains, &[false; 2]).is_none());
    }

    #[test]
    fn invalid_indefinite_and_nonfinite_inputs_are_rejected() {
        let gram = diagonal(3, 1.);
        for c in [
            Contact {
                restitution: -0.1,
                ..contact()
            },
            Contact {
                restitution: 1.1,
                ..contact()
            },
            Contact {
                incoming_normal_velocity: Real::NAN,
                ..contact()
            },
            Contact {
                static_friction: 0.1,
                ..contact()
            },
            Contact {
                dynamic_friction: -0.1,
                ..contact()
            },
            Contact {
                base_normal_impulse: -1.,
                ..contact()
            },
        ] {
            assert!(solve(&gram, &[-1., 0., 0.], &[Domain::Contact(c)]).is_err());
        }
        let indefinite = [1., -0.75, -0.75, -0.75, 1., -0.75, -0.75, -0.75, 1.];
        assert!(solve(&indefinite, &[0.; 3], &[Domain::Contact(contact())]).is_err());
        for invalid in [
            vec![0.; 8],
            vec![Real::NAN; 9],
            vec![Real::INFINITY; 9],
            diagonal(3, -1.),
        ] {
            assert!(solve(&invalid, &[0.; 3], &[Domain::Contact(contact())]).is_err());
        }
        assert!(solve(&gram, &[0.; 2], &[Domain::Contact(contact())]).is_err());
        assert!(solve(&gram, &[0.; 3], &[Domain::Bilateral]).is_err());
        let mut asymmetric = gram.clone();
        asymmetric[1] = 0.1;
        assert!(solve(&asymmetric, &[0.; 3], &[Domain::Contact(contact())]).is_err());
        assert!(solve(&gram, &[Real::NAN, 0., 0.], &[Domain::Contact(contact())]).is_err());
    }

    #[test]
    fn infeasible_immovable_rows_fail_closed_and_empty_problem_succeeds() {
        assert!(solve(&[0.; 9], &[-1., 0., 0.], &[Domain::Contact(contact())]).is_err());
        assert!(solve(&[0.], &[1.], &[Domain::Bilateral]).is_err());
        assert!(solve(&[0.], &[-1.], &[Domain::Nonnegative]).is_err());
        assert_eq!(
            solve(&[0.], &[1.], &[Domain::Nonnegative])
                .unwrap()
                .impulses,
            [0.]
        );
        assert!(solve(&[], &[], &[]).unwrap().impulses.is_empty());
    }
}
