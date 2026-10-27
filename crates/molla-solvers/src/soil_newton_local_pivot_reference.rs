use super::*;

#[path = "soil_newton_conditioned_reference.rs"]
mod conditioned;

fn polynomial(p: &[f64], x: f64) -> f64 {
    p.iter().rev().fold(0., |v, c| v * x + c)
}

fn roots_unit(p: &[f64]) -> Vec<f64> {
    let mut p = p.to_vec();
    while p.len() > 1 && p.last() == Some(&0.) {
        p.pop();
    }
    if p.len() <= 1 {
        return Vec::new();
    }
    let scale = p.iter().fold(0f64, |a, b| a.max(b.abs()));
    for c in &mut p {
        *c /= scale;
    }
    if p.len() == 2 {
        let root = -p[0] / p[1];
        return if (-1. ..=1.).contains(&root) {
            vec![root]
        } else {
            Vec::new()
        };
    }
    let derivative: Vec<_> = p
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| i as f64 * c)
        .collect();
    let mut points = roots_unit(&derivative);
    points.extend([-1., 1.]);
    points.sort_by(f64::total_cmp);
    points.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut result: Vec<_> = points
        .iter()
        .copied()
        .filter(|&x| polynomial(&p, x).abs() < 1e-12)
        .collect();
    for endpoints in points.windows(2) {
        let (mut lo, mut hi) = (endpoints[0], endpoints[1]);
        let sign = polynomial(&p, lo).is_sign_positive();
        if sign == polynomial(&p, hi).is_sign_positive() {
            continue;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if polynomial(&p, mid).is_sign_positive() == sign {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        result.push(0.5 * (lo + hi));
    }
    result.sort_by(f64::total_cmp);
    result.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    result
}

fn sliding(c: &Contact, a: &[f64], b: DVec3) -> Vec<DVec3> {
    let n = c.normal;
    let axis = if n.x.abs() < 0.8 { DVec3::X } else { DVec3::Y };
    let t1 = n.cross(axis).normalize();
    let t2 = n.cross(t1);
    let multiply = |x: DVec3| {
        DVec3::from_array(std::array::from_fn(|i| {
            (0..3).map(|j| a[3 * i + j] * x[j]).sum()
        }))
    };
    if c.mu == 0. {
        let denominator = n.dot(multiply(n));
        let lambda = -(n.dot(b) + c.gap) / denominator;
        return if denominator > 0. && lambda >= 0. && lambda.is_finite() {
            vec![lambda * n]
        } else {
            Vec::new()
        };
    }
    let directions = [n - c.mu * t1, -2. * c.mu * t2, n + c.mu * t1];
    let tangents = [t1, 2. * t2, -t1];
    let bias = n.dot(b) + c.gap;
    let mut coefficients = vec![0.; 5];
    for (i, d) in directions.into_iter().enumerate() {
        let product = multiply(d);
        let velocity = b * n.dot(product) - bias * product;
        for (j, t) in tangents.into_iter().enumerate() {
            coefficients[i + j] += n.dot(velocity.cross(t));
        }
    }
    let mut roots = roots_unit(&coefficients);
    coefficients.reverse();
    roots.extend(
        roots_unit(&coefficients)
            .into_iter()
            .filter(|r| r.abs() > 1e-15)
            .map(|r| 1. / r),
    );
    let mut angles: Vec<_> = roots
        .into_iter()
        .map(|r| {
            let angle = 2. * r.atan();
            t1 * angle.cos() + t2 * angle.sin()
        })
        .collect();
    angles.push(-t1);
    angles
        .into_iter()
        .filter_map(|t| {
            let d = n - c.mu * t;
            let denominator = n.dot(multiply(d));
            if denominator == 0. {
                return None;
            }
            let lambda = -bias / denominator;
            if lambda < 0. || !lambda.is_finite() {
                return None;
            }
            let k = lambda * d;
            let u = b + multiply(k);
            let slip = u - n * n.dot(u);
            ((n.dot(u) + c.gap).abs() < 1e-8 * (1. + u.length())
                && slip.dot(t) > 0.
                && slip.cross(t).length() < 1e-8 * (1. + slip.length()))
            .then_some(k)
        })
        .collect()
}

fn candidate_order(residual: &[f64], visited: &[bool]) -> Vec<(usize, f64)> {
    let mut order: Vec<_> = residual
        .chunks_exact(6)
        .enumerate()
        .filter(|(i, _)| !visited[*i])
        .map(|(i, r)| (i, norm(r)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    order
}

#[test]
fn global_newton_local_queue_excludes_visited_before_cap() {
    let residual: Vec<_> = (0..17)
        .flat_map(|i| [i as f64 + 1., 0., 0., 0., 0., 0.])
        .collect();
    let mut visited = vec![false; 17];
    visited[9] = true;
    let queue: Vec<_> = candidate_order(&residual, &visited)
        .into_iter()
        .take(8)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(queue, vec![16, 15, 14, 13, 12, 11, 10, 8]);
}

pub(super) fn pivot(
    problem: &Reference,
    x: &[f64],
    error: f64,
    visited: &mut [bool],
) -> Option<Vec<f64>> {
    pivot_candidates(problem, x, error, visited, PivotMode::Release, None)
}

pub(super) fn pivot_all_branches(
    problem: &Reference,
    x: &[f64],
    error: f64,
    visited: &mut [bool],
) -> Option<Vec<f64>> {
    pivot_candidates(problem, x, error, visited, PivotMode::AllBranches, None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PivotMode {
    Release,
    AllBranches,
    Settle,
}

impl PivotMode {
    fn limit(self) -> usize {
        if self == Self::Settle { 1 } else { 8 }
    }

    fn accepts(self, candidate: f64, best: f64, base: f64) -> bool {
        if self == Self::Settle {
            candidate < best && candidate < base
        } else {
            candidate < best * (1. - 1e-6)
        }
    }
}

pub(super) struct PivotCandidateTrace {
    slot: usize,
    conditioned: bool,
    counts: [usize; 6],
}

impl PivotCandidateTrace {
    pub(super) fn json(self) -> serde_json::Value {
        serde_json::json!({"slot":self.slot,"conditioned":self.conditioned,"counts":self.counts})
    }
}

pub(super) fn pivot_diagnostic(
    problem: &Reference,
    x: &[f64],
    error: f64,
    visited: &mut [bool],
    mode: PivotMode,
) -> (Option<Vec<f64>>, Vec<PivotCandidateTrace>) {
    let mut trace = Vec::new();
    let trial = pivot_candidates(problem, x, error, visited, mode, Some(&mut trace));
    (trial, trace)
}

#[test]
fn newton_prefix_oracle_pivot_policy_contract() {
    assert_eq!(PivotMode::Settle.limit(), 1);
    assert_eq!(PivotMode::Release.limit(), 8);
    assert_eq!(PivotMode::AllBranches.limit(), 8);
    for candidate in [f64::NAN, f64::INFINITY, 1., 2.] {
        assert!(!PivotMode::Settle.accepts(candidate, f64::INFINITY, 1.));
    }
    assert!(PivotMode::Settle.accepts(1. - 1e-12, f64::INFINITY, 1.));
    assert!(!PivotMode::Settle.accepts(0.9, 0.8, 1.));
    for mode in [PivotMode::Release, PivotMode::AllBranches] {
        assert!(mode.accepts(2., f64::INFINITY, 1.));
        assert!(!mode.accepts(1. - 1e-12, 1., 2.));
    }
}

fn pivot_candidates(
    problem: &Reference,
    x: &[f64],
    error: f64,
    visited: &mut [bool],
    mode: PivotMode,
    mut trace: Option<&mut Vec<PivotCandidateTrace>>,
) -> Option<Vec<f64>> {
    let all_branches = mode != PivotMode::Release;
    if visited.iter().filter(|&&v| v).count() >= 8 {
        return None;
    }
    let q = problem.rates(x, true);
    let linear = problem
        .conditioned_pivot
        .then(|| Linearization::new(problem, x));
    let residual = problem.residual(x);
    let order = candidate_order(&residual, visited);
    if std::env::var_os("MOLLA_NEWTON_LOCAL_TRACE").is_some() {
        eprintln!(
            "local release queue={:?}",
            order
                .iter()
                .take(16)
                .map(|&(i, r)| (problem.contacts[i].index, r))
                .collect::<Vec<_>>()
        );
    }
    let mut best = f64::INFINITY;
    let mut selected = None;
    for (index, _) in order.into_iter().take(mode.limit()) {
        let mut counts = [0usize; 6];
        let c = &problem.contacts[index];
        let old = &x[6 * index..6 * index + 6];
        let conditioned = linear
            .as_ref()
            .and_then(|linear| conditioned::Conditioned::new(linear, x, &q, index));
        let n = q.len();
        let response: Vec<_> = c.prepared.clone().unwrap_or_else(|| {
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| problem.inverse[i * n + j] * c.rows[j])
                        .sum::<DVec3>()
                })
                .collect()
        });
        let columns: [DVec3; 3] = conditioned.as_ref().map(|c| c.columns).unwrap_or_else(|| {
            std::array::from_fn(|j| {
                c.rows
                    .iter()
                    .zip(&response)
                    .map(|(row, r)| *row * r[j])
                    .sum()
            })
        });
        let other = conditioned.as_ref().map(|c| c.other).unwrap_or_else(|| {
            c.rows
                .iter()
                .zip(&q)
                .map(|(row, q)| *row * *q)
                .sum::<DVec3>()
                + (0..3).map(|j| columns[j] * old[j]).sum::<DVec3>()
        });
        let walls = (0..3).fold(0u32, |bits, i| {
            bits | if c.wall[i] != 0. { 1 << i } else { 0 }
        });
        for mask in 0..8u32 {
            if mask & !walls != 0 {
                continue;
            }
            let soil = DVec3::from_array(std::array::from_fn(|i| {
                if mask & (1 << i) != 0 { 0. } else { c.soil[i] }
            }));
            let a: Vec<_> = (0..3)
                .flat_map(|i| {
                    (0..3).map(move |j| {
                        columns[j][i]
                            + if i == j && mask & (1 << i) == 0 {
                                1. / c.mass
                            } else {
                                0.
                            }
                    })
                })
                .collect();
            let rhs = other - soil - c.normal * c.gap;
            let mut candidates = sliding(c, &a, soil - other);
            if c.normal.dot(soil - other) + c.gap >= 0. {
                candidates.push(DVec3::ZERO);
            }
            if let Some(k) = solve_dense(a.clone(), rhs.to_array().to_vec()) {
                candidates.push(xyz(&k));
            }
            for k in candidates {
                counts[0] += 1;
                let lambda = c.normal.dot(k);
                if lambda < 0.
                    || (k - c.normal * lambda).length() > c.mu * lambda + 1e-12 * (1. + k.length())
                {
                    continue;
                }
                counts[1] += 1;
                let w = DVec3::from_array(std::array::from_fn(|i| {
                    if mask & (1 << i) != 0 {
                        -c.mass * c.soil[i] - k[i]
                    } else {
                        0.
                    }
                }));
                let soil = c.soil + (k + w) / c.mass;
                if (0..3).any(|i| c.wall[i] * w[i] < 0. || c.wall[i] * soil[i] < -1e-12) {
                    continue;
                }
                counts[2] += 1;
                let relative = soil - other + (0..3).map(|j| columns[j] * k[j]).sum::<DVec3>();
                let separation = c.normal.dot(relative) + c.gap;
                let slip = relative - c.normal * c.normal.dot(relative);
                let tangent = k - c.normal * lambda;
                if separation < -1e-8 * (1. + relative.length() + c.gap)
                    || (lambda * separation).abs()
                        > 1e-8 * (1. + k.length()) * (1. + relative.length() + c.gap)
                    || (slip.length() > 1e-10
                        && (tangent + c.mu * lambda * slip.normalize()).length()
                            > 1e-8 * (1. + k.length()))
                {
                    continue;
                }
                counts[3] += 1;
                if !all_branches && !(0..3).any(|i| c.wall[i] * old[3 + i] > 1e-6 && w[i] == 0.) {
                    continue;
                }
                counts[4] += 1;
                let mut trial = conditioned
                    .as_ref()
                    .map(|m| m.trial(problem, x, k, w))
                    .unwrap_or_else(|| x.to_vec());
                trial[6 * index..6 * index + 6].copy_from_slice(&six(k, w));
                problem.canonicalize(&mut trial);
                let candidate = norm(&problem.residual(&trial));
                if mode.accepts(candidate, best, error) {
                    counts[5] += 1;
                    best = candidate;
                    selected = Some((index, trial, conditioned.is_some()));
                }
            }
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(PivotCandidateTrace {
                slot: c.index,
                conditioned: conditioned.is_some(),
                counts,
            });
        }
        if std::env::var_os("MOLLA_NEWTON_LOCAL_TRACE").is_some() {
            eprintln!(
                "local pivot candidates slot={} conditioned={} all_branches={all_branches} generated_cone_wall_contact_release_best={counts:?}",
                c.index,
                conditioned.is_some()
            );
        }
    }
    selected.map(|(index, trial, conditioned)| {
        eprintln!("local release accepted slot={} conditioned={conditioned} residual={best:e} previous={error:e}", problem.contacts[index].index);
        visited[index] = true;
        trial
    })
}

#[test]
fn global_newton_local_polynomial_roots() {
    let roots = roots_unit(&[0.03125, -0.21875, -0.5625, 0.25, 1.]);
    for root in roots {
        assert!(polynomial(&[0.03125, -0.21875, -0.5625, 0.25, 1.], root).abs() < 1e-10);
    }
    let roots = roots_unit(&[0.25, -1., 1.]);
    assert_eq!(roots.len(), 1);
    assert!((roots[0] - 0.5).abs() < 1e-10);
    let roots = roots_unit(&[0.0625, 0., -0.625, 0., 1.]);
    assert_eq!(roots.len(), 4);
}

#[test]
fn global_newton_local_sliding_oracle() {
    let f = pair(true, 0.3);
    let mut p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let c = &mut p.contacts[0];
    let a = [2., 0.2, -0.1, 0.2, 0.7, 0.05, -0.1, 0.05, 1.3];
    let multiply = |x: DVec3| {
        DVec3::from_array(std::array::from_fn(|i| {
            (0..3).map(|j| a[3 * i + j] * x[j]).sum()
        }))
    };
    for n in [
        DVec3::Y,
        DVec3::new(0.0238544, -0.9997154, 1.4e-7).normalize(),
    ] {
        c.normal = n;
        c.gap = 0.2;
        let t1 = n.cross(DVec3::X).normalize();
        let t2 = n.cross(t1);
        for mu in [0., 0.1, 0.3, 1.1] {
            c.mu = mu;
            for angle in [
                0.,
                1.2,
                std::f64::consts::PI,
                std::f64::consts::PI + 1e-6,
                -1.4,
            ] {
                let t = t1 * angle.cos() + t2 * angle.sin();
                let expected = 2. * (n - mu * t);
                let b = 0.7 * t - c.gap * n - multiply(expected);
                let found = sliding(c, &a, b);
                assert!(
                    found.iter().any(|k| (*k - expected).length() < 1e-7),
                    "mu={mu} angle={angle} roots={found:?}"
                );
            }
        }
    }
}
