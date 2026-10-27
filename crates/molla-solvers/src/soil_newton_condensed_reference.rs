use super::*;

pub(super) fn undamped(linear: &Linearization<'_>, rhs: &[f64]) -> Option<Vec<f64>> {
    let n = linear.problem.free.len();
    if !(1..=32).contains(&n) || rhs.len() != 6 * linear.problem.contacts.len() {
        return None;
    }
    let mut core = vec![0.; n * n];
    let mut projected = vec![0.; n];
    let mut cache = Vec::with_capacity(linear.problem.contacts.len());
    for i in 0..n {
        core[i * n + i] = 1.;
    }
    for (index, c) in linear.problem.contacts.iter().enumerate() {
        let d: Vec<_> = linear.direct[index].iter().flatten().copied().collect();
        let mut inverse = [[0.; 6]; 6];
        for j in 0..6 {
            let mut unit = vec![0.; 6];
            unit[j] = 1.;
            let column = solve_dense(d.clone(), unit)?;
            for i in 0..6 {
                inverse[i][j] = column[i];
            }
        }
        let g: [[f64; 3]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|a| {
                (0..6)
                    .map(|k| inverse[i][k] * linear.machine[index][k][a])
                    .sum()
            })
        });
        let base: [f64; 6] =
            std::array::from_fn(|i| (0..6).map(|k| inverse[i][k] * rhs[6 * index + k]).sum());
        for i in 0..n {
            let response = c.prepared.as_ref().map(|r| r[i]).unwrap_or_else(|| {
                (0..n)
                    .map(|j| c.rows[j] * linear.problem.inverse[i * n + j])
                    .sum()
            });
            for a in 0..3 {
                projected[i] += response[a] * base[a];
            }
            for j in 0..n {
                for a in 0..3 {
                    for b in 0..3 {
                        core[i * n + j] -= response[a] * g[a][b] * c.rows[j][b];
                    }
                }
            }
        }
        cache.push((g, base));
    }
    let correction = solve_dense(core, projected)?;
    let mut result = Vec::with_capacity(rhs.len());
    for (c, (g, base)) in linear.problem.contacts.iter().zip(cache) {
        let point: DVec3 = (0..n).map(|i| c.rows[i] * correction[i]).sum();
        for i in 0..6 {
            result.push(base[i] + (0..3).map(|a| g[i][a] * point[a]).sum::<f64>());
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn inverse_spd(a: Block) -> Option<Block> {
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let value = a[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            lower[i][j] = if i == j {
                if !value.is_finite() || value <= 0. {
                    return None;
                }
                value.sqrt()
            } else {
                value / lower[j][j]
            };
        }
    }
    let mut result = [[0.; 6]; 6];
    for column in 0..6 {
        let mut x = [0.; 6];
        for i in 0..6 {
            let rhs = if i == column { 1. } else { 0. };
            x[i] = (rhs - (0..i).map(|j| lower[i][j] * x[j]).sum::<f64>()) / lower[i][i];
        }
        for i in (0..6).rev() {
            x[i] = (x[i] - (i + 1..6).map(|j| lower[j][i] * x[j]).sum::<f64>()) / lower[i][i];
            result[i][column] = x[i];
        }
    }
    result
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(result)
}

fn expanded(linear: &Linearization<'_>, rhs: &[f64], damping: f64) -> Option<Vec<f64>> {
    let n = linear.problem.free.len();
    let width = 2 * n;
    if !(1..=32).contains(&n) || !damping.is_finite() || damping <= 0. {
        return None;
    }
    let mut core = vec![0.; width * width];
    let mut projected = vec![0.; width];
    let mut base = vec![0.; rhs.len()];
    let mut inverse_columns = vec![0.; rhs.len() * width];
    for i in 0..n {
        core[i * width + n + i] = 1.;
        core[(n + i) * width + i] = 1.;
    }
    for (index, c) in linear.problem.contacts.iter().enumerate() {
        let d = &linear.direct[index];
        let m = &linear.machine[index];
        let e: Block = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6).map(|k| d[k][i] * d[k][j]).sum::<f64>() + if i == j { damping } else { 0. }
            })
        });
        let inverse = inverse_spd(e)?;
        let mut u = vec![0.; 6 * n];
        let mut columns = vec![0.; 6 * width];
        for i in 0..6 {
            for j in 0..n {
                u[i * n + j] = (0..3).map(|axis| m[i][axis] * c.rows[j][axis]).sum();
            }
        }
        for i in 0..n {
            for j in 0..n {
                core[i * width + j] -= (0..6).map(|k| u[k * n + i] * u[k * n + j]).sum::<f64>();
            }
        }
        for i in 0..6 {
            for j in 0..n {
                columns[i * width + j] = (0..6).map(|k| d[k][i] * u[k * n + j]).sum();
                if i < 3 {
                    columns[i * width + n + j] = -if let Some(response) = &c.prepared {
                        response[j][i]
                    } else {
                        (0..n)
                            .map(|k| linear.problem.inverse[j * n + k] * c.rows[k][i])
                            .sum()
                    };
                }
            }
        }
        let start = index * 6;
        for i in 0..6 {
            base[start + i] = (0..6).map(|k| inverse[i][k] * rhs[start + k]).sum();
            for j in 0..width {
                inverse_columns[(start + i) * width + j] =
                    (0..6).map(|k| inverse[i][k] * columns[k * width + j]).sum();
            }
        }
        for i in 0..width {
            projected[i] += (0..6)
                .map(|k| columns[k * width + i] * base[start + k])
                .sum::<f64>();
            for j in 0..width {
                core[i * width + j] += (0..6)
                    .map(|k| columns[k * width + i] * inverse_columns[(start + k) * width + j])
                    .sum::<f64>();
            }
        }
    }
    let correction = solve_dense(core, projected)?;
    for (i, value) in base.iter_mut().enumerate() {
        *value -= (0..width)
            .map(|j| inverse_columns[i * width + j] * correction[j])
            .sum::<f64>();
    }
    base.iter().all(|v| v.is_finite()).then_some(base)
}

struct ContactCache {
    g: [[f64; 3]; 6],
    h: [[f64; 3]; 6],
    base: [f64; 6],
    response: Vec<DVec3>,
}

pub(super) fn regularized(
    linear: &Linearization<'_>,
    rhs: &[f64],
    damping: f64,
) -> Option<Vec<f64>> {
    let n = linear.problem.free.len();
    let width = 2 * n;
    if !(1..=32).contains(&n)
        || rhs.len() != 6 * linear.problem.contacts.len()
        || !damping.is_finite()
        || damping <= 0.
    {
        return None;
    }
    let mut core = vec![0.; width * width];
    let mut projected = vec![0.; width];
    let mut caches = Vec::with_capacity(linear.problem.contacts.len());
    for i in 0..n {
        core[i * width + n + i] = 1.;
        core[(n + i) * width + i] = 1.;
    }
    for (index, c) in linear.problem.contacts.iter().enumerate() {
        let d = &linear.direct[index];
        let m = &linear.machine[index];
        let e: Block = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6).map(|k| d[k][i] * d[k][j]).sum::<f64>() + if i == j { damping } else { 0. }
            })
        });
        let f: Block = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6).map(|k| d[i][k] * d[j][k]).sum::<f64>() + if i == j { damping } else { 0. }
            })
        });
        let ei = inverse_spd(e)?;
        let fi = inverse_spd(f)?;
        let dtb: [[f64; 3]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| d[k][i] * m[k][j]).sum())
        });
        let g: [[f64; 3]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| ei[i][k] * dtb[k][j]).sum())
        });
        let h = std::array::from_fn(|i| std::array::from_fn(|j| ei[i][j]));
        let base: [f64; 6] =
            std::array::from_fn(|i| (0..6).map(|k| ei[i][k] * rhs[6 * index + k]).sum());
        let fim: [[f64; 3]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| fi[i][k] * m[k][j]).sum())
        });
        let t: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| -damping * (0..6).map(|k| m[k][i] * fim[k][j]).sum::<f64>())
        });
        let r1: [f64; 3] = std::array::from_fn(|i| (0..6).map(|k| dtb[k][i] * base[k]).sum());
        let response: Vec<DVec3> = c.prepared.clone().unwrap_or_else(|| {
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| c.rows[j] * linear.problem.inverse[i * n + j])
                        .sum()
                })
                .collect()
        });
        for i in 0..n {
            for a in 0..3 {
                projected[i] += c.rows[i][a] * r1[a];
                projected[n + i] -= response[i][a] * base[a];
            }
            for j in 0..n {
                for a in 0..3 {
                    for b in 0..3 {
                        core[i * width + j] += c.rows[i][a] * t[a][b] * c.rows[j][b];
                        let cross = c.rows[i][a] * g[b][a] * response[j][b];
                        core[i * width + n + j] -= cross;
                        core[(n + j) * width + i] -= cross;
                        core[(n + i) * width + n + j] += response[i][a] * h[a][b] * response[j][b];
                    }
                }
            }
        }
        caches.push(ContactCache {
            g,
            h,
            base,
            response,
        });
    }
    let correction = solve_dense(core, projected)?;
    let mut result = Vec::with_capacity(rhs.len());
    for (c, cache) in linear.problem.contacts.iter().zip(caches) {
        let jeta: DVec3 = (0..n).map(|i| c.rows[i] * correction[i]).sum();
        let reta: DVec3 = (0..n).map(|i| cache.response[i] * correction[n + i]).sum();
        for i in 0..6 {
            result.push(
                cache.base[i]
                    + (0..3)
                        .map(|a| -cache.g[i][a] * jeta[a] + cache.h[i][a] * reta[a])
                        .sum::<f64>(),
            );
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[test]
fn global_newton_condensed_rank_deficient_blocks() {
    let fixture = pair(true, 0.3);
    let problem = Reference::new(&fixture, fixture.free.iter().map(|&v| v as f64).collect());
    let mut linear = Linearization::new(&problem, &vec![0.; 6 * problem.contacts.len()]);
    linear.direct.fill([[0.; 6]; 6]);
    let rhs: Vec<_> = (0..6 * problem.contacts.len())
        .map(|i| (i as f64 * 0.37).sin())
        .collect();
    for damping in [1e-3, 0.1, 10.] {
        let result = regularized(&linear, &rhs, damping).unwrap();
        let product = linear.transpose(&linear.product(&result));
        let relative = norm(
            &product
                .iter()
                .zip(&result)
                .zip(&rhs)
                .map(|((p, x), b)| p + damping * x - b)
                .collect::<Vec<_>>(),
        ) / norm(&rhs);
        assert!(
            relative < 1e-8,
            "rank-deficient normal residual={relative:e}"
        );
    }
    linear.direct[0][0][0] = f64::NAN;
    assert!(regularized(&linear, &rhs, 0.1).is_none());
}

#[test]
fn global_newton_condensed_damping_matches_dense() {
    for fixture in [
        pair(false, 0.),
        pair(true, 0.3),
        articulated(7),
        wide_fixture(),
    ] {
        for prepared in [false, true] {
            let mut problem =
                Reference::new(&fixture, fixture.free.iter().map(|&v| v as f64).collect());
            let n = problem.free.len();
            for c in &mut problem.contacts {
                c.fischer = true;
                c.smoothing = 0.001;
                if prepared {
                    c.prepared = Some(
                        (0..n)
                            .map(|i| {
                                DVec3::from_array(std::array::from_fn(|axis| {
                                    (0..n)
                                        .map(|j| problem.inverse[i * n + j] * c.rows[j][axis])
                                        .sum::<f64>() as f32
                                        as f64
                                }))
                            })
                            .collect(),
                    );
                }
            }
            let x = vec![0.01; 6 * problem.contacts.len()];
            let linear = Linearization::new(&problem, &x);
            let f = problem.residual(&x);
            let rhs: Vec<_> = linear.transpose(&f).iter().map(|v| -v).collect();
            let size = rhs.len();
            for damping in [1e-4, 0.1, 10.] {
                let result = regularized(&linear, &rhs, damping).unwrap();
                let other = expanded(&linear, &rhs, damping).unwrap();
                assert!(
                    norm(
                        &result
                            .iter()
                            .zip(&other)
                            .map(|(a, b)| a - b)
                            .collect::<Vec<_>>()
                    ) / (1. + norm(&other))
                        < 1e-7
                );
                let mut dense = vec![0.; size * size];
                for j in 0..size {
                    let mut unit = vec![0.; size];
                    unit[j] = 1.;
                    let column = linear.transpose(&linear.product(&unit));
                    for i in 0..size {
                        dense[i * size + j] = column[i] + if i == j { damping } else { 0. };
                    }
                }
                let expected = solve_dense(dense, rhs.clone()).unwrap();
                let error = norm(
                    &result
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                ) / (1. + norm(&expected));
                assert!(
                    error < 1e-7,
                    "prepared={prepared} damping={damping} error={error:e}"
                );
                let product = linear.transpose(&linear.product(&result));
                let residual = norm(
                    &product
                        .iter()
                        .zip(&result)
                        .zip(&rhs)
                        .map(|((p, x), b)| p + damping * x - b)
                        .collect::<Vec<_>>(),
                ) / norm(&rhs);
                assert!(residual < 1e-8, "normal residual={residual:e}");
            }
            for damping in [0., -1., f64::NAN, f64::INFINITY] {
                assert!(regularized(&linear, &rhs, damping).is_none());
            }
        }
    }
}
