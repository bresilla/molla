use super::*;

pub(super) struct Conditioned {
    pub columns: [DVec3; 3],
    pub other: DVec3,
    selected: usize,
    bias: Vec<f64>,
    response: Vec<DVec3>,
    g: Vec<[[f64; 3]; 6]>,
    base: Vec<[f64; 6]>,
}

fn response(problem: &Reference, c: &Contact) -> Vec<DVec3> {
    c.prepared.clone().unwrap_or_else(|| {
        let n = problem.free.len();
        (0..n)
            .map(|i| (0..n).map(|j| problem.inverse[i * n + j] * c.rows[j]).sum())
            .collect()
    })
}

impl Conditioned {
    pub fn new(linear: &Linearization<'_>, x: &[f64], q: &[f64], selected: usize) -> Option<Self> {
        let problem = linear.problem;
        let n = q.len();
        let mut core = vec![0.; n * n];
        for i in 0..n {
            core[i * n + i] = 1.;
        }
        let residual = problem.residual(x);
        let mut projected = vec![0.; n];
        let mut g = vec![[[0.; 3]; 6]; problem.contacts.len()];
        let mut base = vec![[0.; 6]; problem.contacts.len()];
        for (index, c) in problem.contacts.iter().enumerate() {
            if index == selected {
                continue;
            }
            let d: Vec<_> = linear.direct[index].iter().flatten().copied().collect();
            for axis in 0..3 {
                let rhs: Vec<_> = linear.machine[index].iter().map(|r| r[axis]).collect();
                let column = solve_dense(d.clone(), rhs)?;
                for i in 0..6 {
                    g[index][i][axis] = column[i];
                }
            }
            base[index] = solve_dense(
                d,
                residual[6 * index..6 * index + 6]
                    .iter()
                    .map(|f| -f)
                    .collect(),
            )?
            .try_into()
            .ok()?;
            let r = response(problem, c);
            for i in 0..n {
                projected[i] -= (0..3).map(|a| r[i][a] * base[index][a]).sum::<f64>();
                for j in 0..n {
                    for a in 0..3 {
                        for b in 0..3 {
                            core[i * n + j] -= r[i][a] * g[index][a][b] * c.rows[j][b];
                        }
                    }
                }
            }
        }
        let bias = solve_dense(core.clone(), projected)?;
        let c = &problem.contacts[selected];
        let bare = response(problem, c);
        let mut response = vec![DVec3::ZERO; n];
        for axis in 0..3 {
            let column = solve_dense(core.clone(), bare.iter().map(|r| r[axis]).collect())?;
            for i in 0..n {
                response[i][axis] = column[i];
            }
        }
        let columns = std::array::from_fn(|axis| {
            c.rows
                .iter()
                .zip(&response)
                .map(|(row, r)| *row * r[axis])
                .sum()
        });
        let old = xyz(&x[6 * selected..]);
        let other = (0..n)
            .map(|i| c.rows[i] * (q[i] + bias[i] + response[i].dot(old)))
            .sum();
        Some(Self {
            columns,
            other,
            selected,
            bias,
            response,
            g,
            base,
        })
    }

    pub fn trial(&self, problem: &Reference, x: &[f64], k: DVec3, w: DVec3) -> Vec<f64> {
        let dk = k - xyz(&x[6 * self.selected..]);
        let dq: Vec<_> = self
            .bias
            .iter()
            .zip(&self.response)
            .map(|(b, r)| b - r.dot(dk))
            .collect();
        let mut trial = x.to_vec();
        for (index, c) in problem.contacts.iter().enumerate() {
            if index == self.selected {
                continue;
            }
            let point: DVec3 = c.rows.iter().zip(&dq).map(|(row, q)| *row * *q).sum();
            for i in 0..6 {
                trial[6 * index + i] += self.base[index][i]
                    - (0..3).map(|a| self.g[index][i][a] * point[a]).sum::<f64>();
            }
        }
        trial[6 * self.selected..6 * self.selected + 6].copy_from_slice(&six(k, w));
        trial
    }
}

#[test]
fn global_newton_conditioned_local_matches_dense() {
    for f in [pair(true, 0.3), articulated(7), wide_fixture()] {
        for prepared in [false, true] {
            let mut p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
            if prepared {
                let mut packed = vec![[0f32; 4]; f.contacts.len() * p.free.len()];
                for c in &p.contacts {
                    for (i, r) in response(&p, c).into_iter().enumerate() {
                        packed[c.index * p.free.len() + i] =
                            [r.x as f32, r.y as f32, r.z as f32, 0.];
                    }
                }
                p.use_prepared_response(&packed);
            }
            for c in &mut p.contacts {
                c.fischer = true;
                c.smoothing = 0.01;
            }
            let x = vec![0.1; 6 * p.contacts.len()];
            let q = p.rates(&x, true);
            let linear = Linearization::new(&p, &x);
            for selected in [0, p.contacts.len() - 1] {
                let model = Conditioned::new(&linear, &x, &q, selected).unwrap();
                let k = DVec3::new(0.17, -0.01, 0.12);
                let w = DVec3::new(0.04, 0.15, 0.1);
                let trial = model.trial(&p, &x, k, w);
                let delta: Vec<_> = trial.iter().zip(&x).map(|(a, b)| a - b).collect();
                let mut chosen = vec![0.; x.len()];
                chosen[6 * selected..6 * selected + 6]
                    .copy_from_slice(&delta[6 * selected..6 * selected + 6]);
                let residual = p.residual(&x);
                let forcing = linear.product(&chosen);
                let indices: Vec<_> = (0..x.len()).filter(|i| i / 6 != selected).collect();
                let size = indices.len();
                let mut dense = vec![0.; size * size];
                for (j, &index) in indices.iter().enumerate() {
                    let mut unit = vec![0.; x.len()];
                    unit[index] = 1.;
                    let column = linear.product(&unit);
                    for (i, &row) in indices.iter().enumerate() {
                        dense[i * size + j] = column[row];
                    }
                }
                let rhs: Vec<_> = indices.iter().map(|&i| -residual[i] - forcing[i]).collect();
                let expected = solve_dense(dense, rhs).unwrap();
                let error: Vec<_> = indices
                    .iter()
                    .zip(&expected)
                    .map(|(&i, v)| delta[i] - v)
                    .collect();
                assert!(norm(&error) / (1. + norm(&expected)) < 1e-8);
                let product = linear.product(&delta);
                let error: Vec<_> = indices.iter().map(|&i| product[i] + residual[i]).collect();
                assert!(norm(&error) / (1. + norm(&residual)) < 1e-8);
                let actual_q = p.rates(&trial, true);
                let predicted: DVec3 =
                    model.other - (0..3).map(|a| model.columns[a] * k[a]).sum::<DVec3>();
                let actual: DVec3 = p.contacts[selected]
                    .rows
                    .iter()
                    .zip(actual_q)
                    .map(|(r, q)| *r * q)
                    .sum();
                assert!((actual - predicted).length() / (1. + actual.length()) < 1e-8);
            }
        }
    }
}

#[test]
fn global_newton_conditioned_local_singular_blocks() {
    let f = pair(true, 0.3);
    let mut p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
    for c in &mut p.contacts {
        c.fischer = true;
        c.smoothing = 0.01;
    }
    let x = vec![0.1; 6 * p.contacts.len()];
    let q = p.rates(&x, true);
    let mut linear = Linearization::new(&p, &x);
    linear.direct[0] = [[0.; 6]; 6];
    assert!(Conditioned::new(&linear, &x, &q, 0).is_some());
    assert!(Conditioned::new(&linear, &x, &q, 1).is_none());
}
