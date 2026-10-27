use super::*;

struct Factor {
    n: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

impl Factor {
    fn new(mut lu: Vec<f64>, n: usize) -> Option<Self> {
        if n == 0 || lu.len() != n.checked_mul(n)? || !lu.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut pivots = Vec::with_capacity(n);
        for k in 0..n {
            let pivot =
                (k..n).max_by(|&a, &b| lu[a * n + k].abs().total_cmp(&lu[b * n + k].abs()))?;
            if lu[pivot * n + k].abs() <= 1e-30 {
                return None;
            }
            pivots.push(pivot);
            for j in 0..n {
                lu.swap(k * n + j, pivot * n + j);
            }
            for i in k + 1..n {
                let diagonal = lu[k * n + k];
                lu[i * n + k] /= diagonal;
                for j in k + 1..n {
                    let term = lu[i * n + k] * lu[k * n + j];
                    lu[i * n + j] -= term;
                }
            }
        }
        lu.iter()
            .all(|v| v.is_finite())
            .then_some(Self { n, lu, pivots })
    }

    fn solve(&self, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
        let n = self.n;
        if rhs.len() != n || !rhs.iter().all(|v| v.is_finite()) {
            return None;
        }
        for (k, &pivot) in self.pivots.iter().enumerate() {
            rhs.swap(k, pivot);
        }
        for i in 0..n {
            for j in 0..i {
                let term = self.lu[i * n + j] * rhs[j];
                rhs[i] -= term;
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                let term = self.lu[i * n + j] * rhs[j];
                rhs[i] -= term;
            }
            rhs[i] /= self.lu[i * n + i];
        }
        rhs.iter().all(|v| v.is_finite()).then_some(rhs)
    }
}

fn project(normal: DVec3, mu: f64, value: DVec3) -> DVec3 {
    let s = normal.dot(value);
    let tangent = value - normal * s;
    let t = tangent.length();
    if mu == 0. {
        return normal * s.max(0.);
    }
    if s >= 0. && t <= mu * s {
        return value;
    }
    if s + mu * t <= 0. {
        return DVec3::ZERO;
    }
    let height = (s + mu * t) / (1. + mu * mu);
    normal * height + tangent * (mu * height / t)
}

fn velocity(problem: &Reference, impulse: &[f64], free: bool) -> Vec<f64> {
    let q = problem.rates(impulse, free);
    problem
        .contacts
        .iter()
        .zip(impulse.chunks_exact(6))
        .flat_map(|(c, x)| {
            let (soil, relative) = Reference::velocities(c, x, &q, free);
            six(
                relative + if free { c.normal * c.gap } else { DVec3::ZERO },
                soil,
            )
        })
        .collect()
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0., f64::max)
}

struct Shifted {
    core: Factor,
    responses: Vec<Vec<DVec3>>,
    local: Vec<(DVec3, DVec3)>,
    rho: Vec<f64>,
}

impl Shifted {
    fn new(problem: &Reference) -> Option<Self> {
        let n = problem.free.len();
        if !(1..=32).contains(&n) || problem.contacts.len() > 16384 {
            return None;
        }
        let responses: Vec<_> = problem
            .contacts
            .iter()
            .map(|c| {
                c.prepared.clone().unwrap_or_else(|| {
                    (0..n)
                        .map(|i| {
                            (0..n)
                                .map(|j| c.rows[j] * problem.inverse[i * n + j])
                                .sum::<DVec3>()
                        })
                        .collect()
                })
            })
            .collect();
        let rho: Vec<_> = problem.contacts.iter().map(|c| 1. / c.mass).collect();
        if rho.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return None;
        }
        let local: Vec<_> = problem
            .contacts
            .iter()
            .zip(&rho)
            .map(|(c, &rho)| {
                let b = 1. / c.mass;
                let divisor = rho * (rho + 2. * b);
                let mut a = DVec3::splat((b + rho) / divisor);
                let mut cross = DVec3::splat(-b / divisor);
                for axis in 0..3 {
                    if c.wall[axis] == 0. {
                        a[axis] = 1. / (b + rho);
                        cross[axis] = 0.;
                    }
                }
                (a, cross)
            })
            .collect();
        let mut core = vec![0.; n * n];
        for i in 0..n {
            core[i * n + i] = 1.;
            for (index, c) in problem.contacts.iter().enumerate() {
                for j in 0..n {
                    core[i * n + j] += responses[index][i].dot(local[index].0 * c.rows[j]);
                }
            }
        }
        Some(Self {
            core: Factor::new(core, n)?,
            responses,
            local,
            rho,
        })
    }

    fn solve(&self, problem: &Reference, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
        if rhs.len() != 6 * problem.contacts.len() {
            return None;
        }
        let mut projected = vec![0.; self.core.n];
        for (index, x) in rhs.chunks_exact_mut(6).enumerate() {
            let (a, b) = self.local[index];
            let k = xyz(x);
            let mut w = xyz(&x[3..]);
            for axis in 0..3 {
                if problem.contacts[index].wall[axis] == 0. {
                    w[axis] = 0.;
                }
            }
            let yk = a * k + b * w;
            let yw = b * k + a * w;
            x.copy_from_slice(&six(yk, yw));
            for (i, value) in projected.iter_mut().enumerate() {
                *value += self.responses[index][i].dot(yk);
            }
        }
        let correction = self.core.solve(projected)?;
        for ((c, x), &(a, b)) in problem
            .contacts
            .iter()
            .zip(rhs.chunks_exact_mut(6))
            .zip(&self.local)
        {
            let point = c
                .rows
                .iter()
                .zip(&correction)
                .map(|(row, value)| *row * *value)
                .sum::<DVec3>();
            for axis in 0..3 {
                x[axis] -= a[axis] * point[axis];
                x[3 + axis] -= b[axis] * point[axis];
            }
        }
        rhs.iter().all(|v| v.is_finite()).then_some(rhs)
    }
}

pub(super) fn trial(problem: &Reference) -> Option<(Vec<f64>, Vec<serde_json::Value>, Vec<f64>)> {
    let n = problem.contacts.len().checked_mul(6)?;
    if n == 0 {
        return Some((vec![], vec![], vec![]));
    }
    let factor = Shifted::new(problem)?;
    let rho = &factor.rho;
    let mut free = velocity(problem, &vec![0.; n], true);
    for (c, v) in problem.contacts.iter().zip(free.chunks_exact_mut(6)) {
        for axis in 0..3 {
            if c.wall[axis] == 0. {
                v[3 + axis] = 0.;
            }
        }
    }
    let mut z = vec![0.; n];
    let mut dual = vec![0.; n];
    let mut trace = Vec::new();
    for iteration in 1..=64 {
        let rates = problem.rates(&z, true);
        let mut rhs: Vec<_> = (0..n)
            .map(|i| rho[i / 6] * (z[i] - dual[i]) - free[i])
            .collect();
        for (index, c) in problem.contacts.iter().enumerate() {
            let (_, relative) =
                Reference::velocities(c, &z[6 * index..6 * index + 6], &rates, true);
            let slip = relative - c.normal * c.normal.dot(relative);
            let correction = c.normal * (c.mu * slip.length());
            for axis in 0..3 {
                rhs[6 * index + axis] -= correction[axis];
            }
        }
        let x = factor.solve(problem, rhs)?;
        let old = z.clone();
        for (index, c) in problem.contacts.iter().enumerate() {
            let at = 6 * index;
            let point = DVec3::new(
                x[at] + dual[at],
                x[at + 1] + dual[at + 1],
                x[at + 2] + dual[at + 2],
            );
            let cone = project(c.normal, c.mu, point);
            for axis in 0..3 {
                z[at + axis] = cone[axis];
                z[at + 3 + axis] = c.wall[axis]
                    * (c.wall[axis] * (x[at + 3 + axis] + dual[at + 3 + axis])).max(0.);
            }
        }
        for i in 0..n {
            dual[i] += x[i] - z[i];
        }
        if !z.iter().chain(&dual).all(|v| v.is_finite()) {
            return None;
        }
        if [1, 2, 4, 8, 16, 32, 64].contains(&iteration) {
            let primal = (0..n).map(|i| (x[i] - z[i]).abs()).fold(0., f64::max)
                / (1. + max_abs(&x) + max_abs(&z));
            let dual_residual = (0..n)
                .map(|i| rho[i / 6] * (z[i] - old[i]).abs())
                .fold(0., f64::max)
                / (1. + max_abs(&free));
            trace.push(serde_json::json!({"iteration":iteration,"physical":problem.physical_error(&z),"primal":primal,"dual":dual_residual}));
        }
    }
    Some((z, trace, rho.clone()))
}

pub(super) fn diagnose(
    problem: &Reference,
    response: &[[f32; 4]],
    archived: &[u8],
    hash: &str,
    output: &std::path::Path,
) {
    assert!(!output.exists(), "fresh ADMM output required");
    let mut prepared = problem.clone();
    prepared.use_prepared_response(response);
    let expected: Vec<_> = problem
        .contacts
        .iter()
        .flat_map(|c| {
            (0..6).map(move |i| {
                let at = 16 * c.index + if i < 3 { i } else { i + 1 };
                let word =
                    |j| f32::from_le_bytes(archived[4 * j..4 * j + 4].try_into().unwrap()) as f64;
                word(at) + word(at + 8)
            })
        })
        .collect();
    let (x, trace, rho) = trial(&prepared).expect("bounded finite ADMM algebra");
    let (repeated, _, _) = trial(&prepared).unwrap();
    assert_eq!(x, repeated);
    let rate_delta = prepared
        .rates(&x, true)
        .iter()
        .zip(prepared.rates(&expected, true))
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let state_delta = x
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let physical = problem.physical_error(&x);
    let prepared_physical = prepared.physical_error(&x);
    let archived_residual = prepared.residual(&expected);
    let linear = Linearization::new(&prepared, &expected);
    let correction = condensed::undamped(
        &linear,
        &archived_residual.iter().map(|v| -v).collect::<Vec<_>>(),
    );
    let independent = correction.map(|step| {
        let refined: Vec<_> = expected.iter().zip(&step).map(|(a, b)| a + b).collect();
        let linear_error = max_abs(&linear.product(&step).iter().zip(&archived_residual).map(|(a, b)| a + b).collect::<Vec<_>>());
        let delta = max_abs(&refined.iter().zip(&x).map(|(a, b)| a - b).collect::<Vec<_>>());
        serde_json::json!({"method":"one projected-contact Newton step from archive, independent condensed Jacobian","linear_residual_max":linear_error,"physical":problem.physical_error(&refined),"prepared_physical":prepared.physical_error(&refined),"residual_max":max_abs(&prepared.residual(&refined)),"admm_delta":delta,"archive_delta":max_abs(&step)})
    });
    let worst = (0..x.len()).max_by(|&a, &b| {
        (x[a] - expected[a])
            .abs()
            .total_cmp(&(x[b] - expected[b]).abs())
    });
    let worst_contact = worst.map(|i| {
        let c = &prepared.contacts[i / 6];
        let at = 6 * (i / 6);
        let q = prepared.rates(&expected, true);
        let (_, relative) = Reference::velocities(c, &expected[at..at+6], &q, true);
        serde_json::json!({"index":c.index,"coordinate":i%6,"mass":c.mass,"normal":c.normal.to_array(),"gap":c.gap,"mu":c.mu,"archive_impulse":&expected[at..at+6],"admm_impulse":&x[at..at+6],"archive_separation":c.normal.dot(relative)+c.gap,"archive_slip":(relative-c.normal*c.normal.dot(relative)).length()})
    });
    let reduced_unknowns = problem
        .contacts
        .iter()
        .map(|c| 3 + c.wall.iter().filter(|&&v| v != 0.).count())
        .sum::<usize>();
    let result = serde_json::json!({"packet":hash,"scope":"CPU-only cold ADMM mechanism diagnostic; not GPU or public-buffer acceptance","acceptance":false,"active_contacts":problem.contacts.len(),"linear_core_dofs":problem.free.len(),"contact_unknowns":6*problem.contacts.len(),"reduced_unknowns":reduced_unknowns,"linear_method":"mass-scaled restricted Woodbury","iterations":64,"factorizations":1,"metric":"inverse-soil-mass","rho_range":[rho.iter().copied().fold(f64::INFINITY,f64::min),rho.iter().copied().fold(0.,f64::max)],"physical":physical,"prepared_physical":prepared_physical,"physical_bound":1e-6,"state_delta":state_delta,"rate_delta":rate_delta,"comparison_bound":2e-5,"mechanism_gate":physical<=1e-6&&prepared_physical<=1e-6&&state_delta<=2e-5&&rate_delta<=2e-5,"exact_cpu_repeat":true,"trace":trace});
    let mut result = result;
    result["reference_audit"] = serde_json::json!({"archive_physical":problem.physical_error(&expected),"archive_prepared_physical":prepared.physical_error(&expected),"archive_residual_max":max_abs(&archived_residual),"admm_residual_max":max_abs(&prepared.residual(&x)),"independent":independent,"worst_contact":worst_contact});
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("ADMM_DIAGNOSTIC {result}");
}

#[test]
fn admm_reference_woodbury_matches_full_shifted_operator() {
    for mask in 0..29 {
        let f = pair(mask != 0, 0.3);
        let physical = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
        for prepared in [false, true] {
            let mut p = physical.clone();
            if mask >= 2 {
                for (index, c) in p.contacts.iter_mut().enumerate() {
                    let mut signs = (mask - 2 + index) % 27;
                    for axis in 0..3 {
                        c.wall[axis] = (signs % 3) as f64 - 1.;
                        signs /= 3;
                    }
                }
            }
            if prepared {
                let mut responses = Vec::new();
                for c in &p.contacts {
                    for i in 0..p.free.len() {
                        let value = (0..p.free.len())
                            .map(|j| c.rows[j] * p.inverse[i * p.free.len() + j])
                            .sum::<DVec3>();
                        responses.push([value.x as f32, value.y as f32, value.z as f32, 0.]);
                    }
                }
                responses[0][1] += 1e-4;
                p.use_prepared_response(&responses);
            }
            let low = Shifted::new(&p).unwrap();
            let n = 6 * p.contacts.len();
            let retained: Vec<_> = (0..n)
                .filter(|i| i % 6 < 3 || p.contacts[i / 6].wall[i % 6 - 3] != 0.)
                .collect();
            let m = retained.len();
            let mut dense = vec![0.; m * m];
            for (column, &j) in retained.iter().enumerate() {
                let mut unit = vec![0.; n];
                unit[j] = 1.;
                let values = velocity(&p, &unit, false);
                for (row, &i) in retained.iter().enumerate() {
                    dense[row * m + column] = values[i];
                }
                dense[column * m + column] += low.rho[j / 6];
            }
            let full = Factor::new(dense.clone(), m).unwrap();
            for offset in [0., 0.25, 100.] {
                let rhs: Vec<_> = (0..n).map(|i| (i as f64 + offset).sin()).collect();
                let actual = low.solve(&p, rhs.clone()).unwrap();
                let expected = full
                    .solve(retained.iter().map(|&i| rhs[i]).collect())
                    .unwrap();
                for (row, &i) in retained.iter().enumerate() {
                    assert!((actual[i] - expected[row]).abs() < 1e-12);
                    assert!(
                        (retained
                            .iter()
                            .enumerate()
                            .map(|(col, &j)| dense[row * m + col] * actual[j])
                            .sum::<f64>()
                            - rhs[i])
                            .abs()
                            < 1e-12
                    );
                }
                for i in (0..n).filter(|i| !retained.contains(i)) {
                    assert_eq!(actual[i], 0.);
                }
            }
        }
    }
}

#[test]
fn admm_reference_mass_metric_preserves_analytic_contacts() {
    let f = pair(false, 0.3);
    for mu in [0., 0.1, 0.3] {
        let mut p = Reference::new(&f, vec![0.; f.free.len()]);
        let base = p.contacts[0].clone();
        p.contacts = [-6, -3, 0, 3, 6]
            .map(|power| {
                let mut c = base.clone();
                c.mass = 10f64.powi(power);
                c.normal = DVec3::Y;
                c.soil = DVec3::new(0.2, -1., 0.1);
                c.gap = 0.;
                c.mu = mu;
                c.wall = [0.; 3];
                c.rows.fill(DVec3::ZERO);
                c.prepared = None;
                c
            })
            .to_vec();
        let (x, _, _) = trial(&p).unwrap();
        for (c, actual) in p.contacts.iter().zip(x.chunks_exact(6)) {
            let tangent = DVec3::new(0.2, 0., 0.1);
            let expected = c.mass * (DVec3::Y - tangent * (mu / tangent.length()).min(1.));
            assert!(
                (xyz(actual) - expected).length() / c.mass < 1e-10,
                "mu={mu} mass={}",
                c.mass
            );
            assert_eq!(xyz(&actual[3..]), DVec3::ZERO);
        }
        assert!(p.physical_error(&x) < 1e-6);
    }
}

#[test]
fn admm_reference_independent_newton_crosscheck() {
    let f = pair(false, 0.);
    let p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
    let (root, _, _) = trial(&p).unwrap();
    let perturbed: Vec<_> = root.iter().map(|v| v * 1.001).collect();
    let residual = p.residual(&perturbed);
    let linear = Linearization::new(&p, &perturbed);
    let step =
        condensed::undamped(&linear, &residual.iter().map(|v| -v).collect::<Vec<_>>()).unwrap();
    assert!(
        max_abs(
            &linear
                .product(&step)
                .iter()
                .zip(&residual)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>()
        ) < 1e-12
    );
    let refined: Vec<_> = perturbed.iter().zip(step).map(|(a, b)| a + b).collect();
    assert!(p.physical_error(&refined) < 1e-10);
    assert!(
        max_abs(
            &refined
                .iter()
                .zip(root)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()
        ) < 1e-10
    );
}

#[test]
fn admm_reference_factor_reuses_pivoted_system() {
    let a = vec![0., 2., 1., 3., 1., -1., 1., 0., 2.];
    let factor = Factor::new(a.clone(), 3).unwrap();
    for rhs in [vec![1., 2., 3.], vec![-5., 0., 7.]] {
        let x = factor.solve(rhs.clone()).unwrap();
        let expected = solve_dense(a.clone(), rhs.clone()).unwrap();
        for i in 0..3 {
            assert!((x[i] - expected[i]).abs() < 1e-12);
            assert!(((0..3).map(|j| a[i * 3 + j] * x[j]).sum::<f64>() - rhs[i]).abs() < 1e-12);
        }
    }
    assert!(Factor::new(vec![0.; 4], 2).is_none());
    assert!(Factor::new(vec![f64::NAN], 1).is_none());
    assert!(factor.solve(vec![1.]).is_none());
}

#[test]
fn admm_reference_cone_projection_and_contact_laws() {
    for normal in [DVec3::Y, DVec3::new(1., 2., -3.).normalize()] {
        for mu in [0., 0.1, 0.5, 1., 2.] {
            for i in -8..9 {
                for j in -8..9 {
                    let v = DVec3::new(i as f64 * 0.3, j as f64 * 0.2, 0.7);
                    let p = project(normal, mu, v);
                    let height = normal.dot(p);
                    assert!(height >= -1e-12);
                    assert!((p - normal * height).length() <= mu * height + 1e-12);
                    assert!((project(normal, mu, p) - p).length() < 1e-12);
                    assert!((v - p).dot(p).abs() < 1e-11);
                    let dual = p - v;
                    assert!(
                        normal.dot(dual) + 1e-12
                            >= mu * (dual - normal * normal.dot(dual)).length()
                    );
                }
            }
        }
    }
    for mu in [0., 0.3] {
        let f = pair(false, mu);
        let p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
        let (x, _, _) = trial(&p).unwrap();
        assert!(
            p.physical_error(&x) < 1e-6,
            "mu={mu} error={}",
            p.physical_error(&x)
        );
        verify_solution(&f, &p, &x);
    }
}
