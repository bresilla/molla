use super::*;

#[path = "soil_newton_operator_tests.rs"]
mod gpu_operator;

#[path = "soil_newton_cone_reference.rs"]
mod cone;

#[path = "soil_newton_local_pivot_reference.rs"]
mod local_pivot;

#[path = "soil_newton_condensed_reference.rs"]
mod condensed;

#[path = "soil_newton_admm_reference.rs"]
mod admm;

type Block = [[f64; 6]; 6];

fn fischer(a: f64, b: f64) -> (f64, f64, f64) {
    let r = a.hypot(b);
    if r == 0. {
        return (
            0.,
            1. - std::f64::consts::FRAC_1_SQRT_2,
            1. - std::f64::consts::FRAC_1_SQRT_2,
        );
    }
    let value = if a + b > 0. {
        2. * a * (b / (a + b + r))
    } else {
        a + b - r
    };
    let da = if a >= 0. {
        (b / r) * (b / (r + a))
    } else {
        1. - a / r
    };
    let db = if b >= 0. {
        (a / r) * (a / (r + b))
    } else {
        1. - b / r
    };
    (value, da, db)
}

fn smooth_fischer(a: f64, b: f64, epsilon: f64) -> (f64, f64, f64) {
    if epsilon == 0. {
        return fischer(a, b);
    }
    let t = std::f64::consts::SQRT_2 * epsilon;
    let r = a.hypot(b).hypot(t);
    let value = if a + b > 0. {
        2. * (a * (b / (a + b + r)) - epsilon * (epsilon / (a + b + r)))
    } else {
        a + b - r
    };
    let da = if a >= 0. {
        (b / r) * (b / (r + a)) + (t / r) * (t / (r + a))
    } else {
        1. - a / r
    };
    let db = if b >= 0. {
        (a / r) * (a / (r + b)) + (t / r) * (t / (r + b))
    } else {
        1. - b / r
    };
    (value, da, db)
}

#[derive(Clone)]
struct Contact {
    fischer: bool,
    conical: bool,
    smoothing: f64,
    prepared: Option<Vec<DVec3>>,
    index: usize,
    normal: DVec3,
    rho: f64,
    mass: f64,
    mu: f64,
    gap: f64,
    soil: DVec3,
    wall: [f64; 3],
    rows: Vec<DVec3>,
}

#[derive(Clone)]
struct Reference {
    mass: Vec<f32>,
    inverse: Vec<f64>,
    free: Vec<f64>,
    contacts: Vec<Contact>,
    local_pivot: bool,
    condensed_damping: bool,
    condensed_newton: bool,
    stagnation_smoothing: bool,
    conditioned_pivot: bool,
}

struct Linearization<'a> {
    problem: &'a Reference,
    direct: Vec<Block>,
    machine: Vec<[[f64; 3]; 6]>,
}

impl<'a> Linearization<'a> {
    fn new(problem: &'a Reference, x: &[f64]) -> Self {
        let q = problem.rates(x, true);
        let mut direct = Vec::new();
        let mut machine = Vec::new();
        for (c, x) in problem.contacts.iter().zip(x.chunks_exact(6)) {
            let (s, r) = Reference::velocities(c, x, &q, true);
            let mut d = [[0.0; 6]; 6];
            let mut m = [[0.0; 3]; 6];
            for j in 0..6 {
                let mut dx = [0.0; 6];
                dx[j] = 1.0;
                let ds = (xyz(&dx) + xyz(&dx[3..])) / c.mass;
                let column = Reference::contact_derivative(c, x, s, r, &dx, ds, ds);
                for i in 0..6 {
                    d[i][j] = column[i];
                }
            }
            for j in 0..3 {
                let mut dr = DVec3::ZERO;
                dr[j] = -1.0;
                let column = Reference::contact_derivative(c, x, s, r, &[0.0; 6], DVec3::ZERO, dr);
                for i in 0..6 {
                    m[i][j] = column[i];
                }
            }
            direct.push(d);
            machine.push(m);
        }
        Self {
            problem,
            direct,
            machine,
        }
    }

    fn product(&self, dx: &[f64]) -> Vec<f64> {
        let dq = self.problem.rates(dx, false);
        let mut result = apply_blocks(&self.direct, dx);
        for ((c, m), out) in self
            .problem
            .contacts
            .iter()
            .zip(&self.machine)
            .zip(result.chunks_exact_mut(6))
        {
            let v = c
                .rows
                .iter()
                .zip(&dq)
                .map(|(r, q)| *r * *q)
                .sum::<DVec3>()
                .to_array();
            for i in 0..6 {
                out[i] += dot(&m[i], &v);
            }
        }
        result
    }

    fn transpose(&self, y: &[f64]) -> Vec<f64> {
        let n = self.problem.free.len();
        let mut rates = vec![0.0; n];
        let mut result = vec![0.0; y.len()];
        for ((((c, d), m), y), out) in self
            .problem
            .contacts
            .iter()
            .zip(&self.direct)
            .zip(&self.machine)
            .zip(y.chunks_exact(6))
            .zip(result.chunks_exact_mut(6))
        {
            for j in 0..6 {
                out[j] = (0..6).map(|i| d[i][j] * y[i]).sum();
            }
            let v = DVec3::from_array(std::array::from_fn(|j| {
                (0..6).map(|i| m[i][j] * y[i]).sum()
            }));
            for (rate, row) in rates.iter_mut().zip(&c.rows) {
                *rate += row.dot(v);
            }
        }
        let dual: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.problem.inverse[j * n + i] * rates[j])
                    .sum::<f64>()
            })
            .collect();
        for (c, out) in self.problem.contacts.iter().zip(result.chunks_exact_mut(6)) {
            let (rows, dual) = match &c.prepared {
                Some(response) => (response, &rates),
                None => (&c.rows, &dual),
            };
            let v = rows.iter().zip(dual).map(|(r, q)| *r * *q).sum::<DVec3>();
            for i in 0..3 {
                out[i] -= v[i];
            }
        }
        result
    }
}

fn regularized_direction(linear: &Linearization<'_>, f: &[f64], damping: f64) -> (Vec<f64>, usize) {
    let mut residual: Vec<_> = linear.transpose(f).iter().map(|v| -v).collect();
    let mut products = 1;
    if linear.problem.condensed_damping {
        if let Some(result) = condensed::regularized(linear, &residual, damping) {
            let product = linear.transpose(&linear.product(&result));
            products += 2;
            let error = norm(
                &product
                    .iter()
                    .zip(&result)
                    .zip(&residual)
                    .map(|((p, x), b)| p + damping * x - b)
                    .collect::<Vec<_>>(),
            ) / norm(&residual);
            eprintln!("condensed damping={damping:e} normal_relative={error:e}");
            if error <= 1e-8 {
                return (result, products);
            }
        }
    }
    let mut direction = residual.clone();
    let mut result = vec![0.0; f.len()];
    let initial = dot(&residual, &residual);
    let mut square = initial;
    for _ in 0..64 {
        if square <= 1e-16 * initial {
            break;
        }
        let mut product = linear.transpose(&linear.product(&direction));
        products += 2;
        for (p, d) in product.iter_mut().zip(&direction) {
            *p += damping * d;
        }
        let denominator = dot(&direction, &product);
        if denominator <= 0.0 || !denominator.is_finite() {
            break;
        }
        let alpha = square / denominator;
        for i in 0..result.len() {
            result[i] += alpha * direction[i];
            residual[i] -= alpha * product[i];
        }
        let next = dot(&residual, &residual);
        for i in 0..direction.len() {
            direction[i] = residual[i] + (next / square) * direction[i];
        }
        square = next;
    }
    (result, products)
}

fn solve_dense(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))?;
        if !a[pivot * n + k].is_finite() || a[pivot * n + k].abs() < 1e-20 {
            return None;
        }
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let ratio = a[i * n + k] / a[k * n + k];
            for j in k + 1..n {
                a[i * n + j] -= ratio * a[k * n + j];
            }
            b[i] -= ratio * b[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[i * n + j] * b[j];
        }
        b[i] /= a[i * n + i];
    }
    b.iter().all(|x| x.is_finite()).then_some(b)
}

fn xyz(x: &[f64]) -> DVec3 {
    DVec3::new(x[0], x[1], x[2])
}
fn six(k: DVec3, w: DVec3) -> [f64; 6] {
    [k.x, k.y, k.z, w.x, w.y, w.z]
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

impl Reference {
    fn canonicalize(&self, x: &mut [f64]) {
        for (c, x) in self.contacts.iter().zip(x.chunks_exact_mut(6)) {
            for i in 0..3 {
                if c.wall[i] == 0.0 {
                    x[3 + i] = 0.0;
                }
            }
        }
    }
    fn new(f: &Fixture, free: Vec<f64>) -> Self {
        let n = free.len();
        assert!(free.iter().all(|v| v.is_finite()));
        assert!(f.mass.iter().all(|v| v.is_finite()));
        let mut inverse = vec![0.0; n * n];
        for j in 0..n {
            let mut unit = vec![0.0; n];
            unit[j] = 1.0;
            let column = solve_dense(f.mass.iter().map(|&x| x as f64).collect(), unit).unwrap();
            for i in 0..n {
                inverse[i * n + j] = column[i];
            }
        }
        let contacts = f
            .contacts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.soil_mass[3] > 0.0)
            .map(|(index, c)| {
                let rows: Vec<_> = f.rows[index * n..(index + 1) * n]
                    .iter()
                    .map(|&r| vec(r))
                    .collect();
                let mut trace = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        trace += inverse[i * n + j] * rows[i].dot(rows[j]);
                    }
                }
                Contact {
                    fischer: false,
                    conical: false,
                    smoothing: 0.,
                    prepared: None,
                    index,
                    normal: vec(c.normal_gap).normalize(),
                    rho: 1.0 / (1.0 / c.soil_mass[3] as f64 + trace),
                    mass: c.soil_mass[3] as f64,
                    mu: c.friction[0] as f64,
                    gap: c.normal_gap[3] as f64,
                    soil: vec(c.soil_mass),
                    wall: std::array::from_fn(|axis| match (c.boundary[0] >> (2 * axis)) & 3 {
                        1 => 1.0,
                        2 => -1.0,
                        _ => 0.0,
                    }),
                    rows,
                }
            })
            .collect();
        Self {
            mass: f.mass.clone(),
            inverse,
            free,
            contacts,
            local_pivot: false,
            condensed_damping: false,
            condensed_newton: false,
            stagnation_smoothing: false,
            conditioned_pivot: false,
        }
    }

    fn use_prepared_response(&mut self, response: &[[f32; 4]]) {
        let n = self.free.len();
        for c in &mut self.contacts {
            let prepared: Vec<_> = response[c.index * n..(c.index + 1) * n]
                .iter()
                .map(|r| vec(*r))
                .collect();
            let trace: f64 = c
                .rows
                .iter()
                .zip(&prepared)
                .map(|(row, r)| row.dot(*r))
                .sum();
            c.rho = 1. / (1. / c.mass + trace);
            c.prepared = Some(prepared);
        }
    }

    fn rates(&self, x: &[f64], free: bool) -> Vec<f64> {
        let n = self.free.len();
        if self.contacts.iter().any(|c| c.prepared.is_some()) {
            let mut q = if free { self.free.clone() } else { vec![0.; n] };
            for (contact, local) in self.contacts.iter().zip(x.chunks_exact(6)) {
                for (rate, response) in q.iter_mut().zip(contact.prepared.as_ref().unwrap()) {
                    *rate -= response.dot(xyz(local));
                }
            }
            return q;
        }
        let mut momentum = vec![0.0; n];
        for (c, x) in self.contacts.iter().zip(x.chunks_exact(6)) {
            for (i, row) in c.rows.iter().enumerate() {
                momentum[i] -= row.dot(xyz(x));
            }
        }
        (0..n)
            .map(|i| {
                (if free { self.free[i] } else { 0.0 })
                    + (0..n)
                        .map(|j| self.inverse[i * n + j] * momentum[j])
                        .sum::<f64>()
            })
            .collect()
    }

    fn velocities(c: &Contact, x: &[f64], q: &[f64], free: bool) -> (DVec3, DVec3) {
        let soil = (if free { c.soil } else { DVec3::ZERO }) + (xyz(x) + xyz(&x[3..])) / c.mass;
        let machine = c.rows.iter().zip(q).map(|(r, q)| *r * *q).sum::<DVec3>();
        (soil, soil - machine)
    }

    fn contact_residual(c: &Contact, x: &[f64], soil: DVec3, relative: DVec3) -> [f64; 6] {
        let k = xyz(x);
        let w = xyz(&x[3..]);
        let lambda = c.normal.dot(k);
        let tangent = k - c.normal * lambda;
        let slip = relative - c.normal * c.normal.dot(relative);
        let y = tangent - c.rho * slip;
        let radius = y.length();
        let cap = c.mu * lambda.max(0.0);
        let projected = if radius > cap { y * (cap / radius) } else { y };
        let separation = c.rho * (c.normal.dot(relative) + c.gap);
        let normal = if c.fischer {
            smooth_fischer(lambda, separation, c.smoothing).0
        } else {
            lambda - (lambda - separation).max(0.)
        };
        let fk = if c.conical && c.mu > 0. {
            cone::contact(c, k, relative, DVec3::ZERO, DVec3::ZERO).0
        } else {
            c.normal * normal + tangent - projected
        };
        let fw = DVec3::from_array(std::array::from_fn(|i| {
            if c.fischer && c.wall[i] != 0. {
                return c.wall[i]
                    * smooth_fischer(c.wall[i] * w[i], c.wall[i] * c.mass * soil[i], c.smoothing)
                        .0;
            }
            w[i] - c.wall[i] * (c.wall[i] * (w[i] - c.mass * soil[i])).max(0.0)
        }));
        six(fk, fw)
    }

    fn contact_derivative(
        c: &Contact,
        x: &[f64],
        soil: DVec3,
        relative: DVec3,
        dx: &[f64],
        ds: DVec3,
        dr: DVec3,
    ) -> [f64; 6] {
        let k = xyz(x);
        let w = xyz(&x[3..]);
        let dk = xyz(dx);
        let dw = xyz(&dx[3..]);
        let lambda = c.normal.dot(k);
        let dl = c.normal.dot(dk);
        let tangent = k - c.normal * lambda;
        let dt = dk - c.normal * dl;
        let slip = relative - c.normal * c.normal.dot(relative);
        let dslip = dr - c.normal * c.normal.dot(dr);
        let y = tangent - c.rho * slip;
        let dy = dt - c.rho * dslip;
        let radius = y.length();
        let cap = c.mu * lambda.max(0.0);
        let dp = if c.mu == 0.0 {
            DVec3::ZERO
        } else if radius > cap {
            let unit = y / radius;
            let dc = if lambda > 0.0 { c.mu * dl } else { 0.0 };
            unit * dc + (cap / radius) * (dy - unit * unit.dot(dy))
        } else {
            dy
        };
        let dn = if c.fischer {
            let (_, a, b) = smooth_fischer(
                lambda,
                c.rho * (c.normal.dot(relative) + c.gap),
                c.smoothing,
            );
            a * dl + b * c.rho * c.normal.dot(dr)
        } else if lambda - c.rho * (c.normal.dot(relative) + c.gap) > 0.0 {
            c.rho * c.normal.dot(dr)
        } else {
            dl
        };
        let fk = if c.conical && c.mu > 0. {
            cone::contact(c, k, relative, dk, dr).1
        } else {
            c.normal * dn + dt - dp
        };
        let fw = DVec3::from_array(std::array::from_fn(|i| {
            if c.fischer && c.wall[i] != 0. {
                let (_, a, b) =
                    smooth_fischer(c.wall[i] * w[i], c.wall[i] * c.mass * soil[i], c.smoothing);
                return a * dw[i] + b * c.mass * ds[i];
            }
            if c.wall[i] * (w[i] - c.mass * soil[i]) > 0.0 {
                c.mass * ds[i]
            } else {
                dw[i]
            }
        }));
        six(fk, fw)
    }

    fn residual(&self, x: &[f64]) -> Vec<f64> {
        let q = self.rates(x, true);
        self.contacts
            .iter()
            .zip(x.chunks_exact(6))
            .flat_map(|(c, x)| {
                let (s, r) = Self::velocities(c, x, &q, true);
                Self::contact_residual(c, x, s, r)
            })
            .collect()
    }

    fn physical_error(&self, x: &[f64]) -> f64 {
        let q = self.rates(x, true);
        let mut worst = 0.0_f64;
        for (c, x) in self.contacts.iter().zip(x.chunks_exact(6)) {
            let k = xyz(x);
            let w = xyz(&x[3..]);
            let (soil, relative) = Self::velocities(c, x, &q, true);
            let machine = soil - relative;
            let lambda = c.normal.dot(k);
            let separation = c.normal.dot(relative) + c.gap;
            let tangent = k - c.normal * lambda;
            let slip = relative - c.normal * c.normal.dot(relative);
            let ps = 1.0 + k.length() + w.length();
            let vs = 1.0 + soil.length() + machine.length();
            let ns = 1.0_f64
                .max(c.normal.dot(soil).abs())
                .max(c.normal.dot(machine).abs())
                .max(c.gap);
            for error in [
                -lambda / ps,
                -separation / ns,
                (lambda * separation).abs() / (ps * vs),
                (tangent.length() - c.mu * lambda) / ps,
            ] {
                worst = worst.max(error);
            }
            if slip.length() > 1e-6 * vs {
                worst = worst.max((tangent + c.mu * lambda * slip.normalize()).length() / ps);
            }
            for i in 0..3 {
                if c.wall[i] == 0.0 {
                    worst = worst.max(w[i].abs() / ps);
                } else {
                    worst = worst
                        .max(-c.wall[i] * w[i] / ps)
                        .max(-c.wall[i] * soil[i] / vs)
                        .max((w[i] * soil[i]).abs() / (ps * vs));
                }
            }
        }
        if !x.iter().chain(q.iter()).all(|v| v.is_finite()) {
            return f64::INFINITY;
        }
        worst
    }

    fn product(&self, x: &[f64], dx: &[f64]) -> Vec<f64> {
        let q = self.rates(x, true);
        let dq = self.rates(dx, false);
        self.contacts
            .iter()
            .zip(x.chunks_exact(6))
            .zip(dx.chunks_exact(6))
            .flat_map(|((c, x), dx)| {
                let (s, r) = Self::velocities(c, x, &q, true);
                let (ds, dr) = Self::velocities(c, dx, &dq, false);
                Self::contact_derivative(c, x, s, r, dx, ds, dr)
            })
            .collect()
    }

    fn preconditioner(&self, x: &[f64]) -> Vec<Block> {
        let q = self.rates(x, true);
        let n = q.len();
        self.contacts
            .iter()
            .zip(x.chunks_exact(6))
            .map(|(c, x)| {
                let (s, r) = Self::velocities(c, x, &q, true);
                let mut block = [[0.0; 6]; 6];
                for j in 0..6 {
                    let mut dx = [0.0; 6];
                    dx[j] = 1.0;
                    let dq: Vec<_> = (0..n)
                        .map(|i| {
                            if let Some(response) = &c.prepared {
                                return -response[i].dot(xyz(&dx));
                            }
                            -(0..n)
                                .map(|k| self.inverse[i * n + k] * c.rows[k].dot(xyz(&dx)))
                                .sum::<f64>()
                        })
                        .collect();
                    let (ds, dr) = Self::velocities(c, &dx, &dq, false);
                    let column = Self::contact_derivative(c, x, s, r, &dx, ds, dr);
                    for i in 0..6 {
                        block[i][j] = column[i];
                    }
                }
                let mut inverse = [[0.0; 6]; 6];
                for j in 0..6 {
                    let mut unit = vec![0.0; 6];
                    unit[j] = 1.0;
                    let Some(column) = solve_dense(block.into_iter().flatten().collect(), unit)
                    else {
                        return std::array::from_fn(|i| {
                            std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })
                        });
                    };
                    for i in 0..6 {
                        inverse[i][j] = column[i];
                    }
                }
                inverse
            })
            .collect()
    }
}

fn apply_blocks(blocks: &[Block], x: &[f64]) -> Vec<f64> {
    blocks
        .iter()
        .zip(x.chunks_exact(6))
        .flat_map(|(b, x)| b.iter().map(|row| dot(row, x)))
        .collect()
}

fn gmres(a: impl Fn(&[f64]) -> Vec<f64>, b: &[f64], limit: usize) -> (Vec<f64>, usize, f64) {
    let beta = norm(b);
    if beta == 0.0 {
        return (vec![0.0; b.len()], 0, 0.0);
    }
    let mut basis = vec![b.iter().map(|x| x / beta).collect::<Vec<_>>()];
    let mut h = vec![vec![0.0; limit]; limit + 1];
    let mut cs = vec![0.0; limit];
    let mut sn = cs.clone();
    let mut g = vec![0.0; limit + 1];
    g[0] = beta;
    let mut count = 0;
    for j in 0..limit {
        let mut v = a(&basis[j]);
        for _ in 0..2 {
            for i in 0..=j {
                let p = dot(&basis[i], &v);
                h[i][j] += p;
                for (v, q) in v.iter_mut().zip(&basis[i]) {
                    *v -= p * q;
                }
            }
        }
        h[j + 1][j] = norm(&v);
        let breakdown = h[j + 1][j] < 1e-14;
        if !breakdown {
            basis.push(v.iter().map(|x| x / h[j + 1][j]).collect());
        }
        for i in 0..j {
            let v = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
            h[i][j] = v;
        }
        let length = h[j][j].hypot(h[j + 1][j]);
        if length == 0.0 {
            break;
        }
        cs[j] = h[j][j] / length;
        sn[j] = h[j + 1][j] / length;
        h[j][j] = length;
        h[j + 1][j] = 0.0;
        g[j + 1] = -sn[j] * g[j];
        g[j] *= cs[j];
        count = j + 1;
        if breakdown || g[j + 1].abs() <= 1e-8 * beta {
            break;
        }
    }
    let mut y = g[..count].to_vec();
    for i in (0..count).rev() {
        for j in i + 1..count {
            y[i] -= h[i][j] * y[j];
        }
        y[i] /= h[i][i];
    }
    let mut x = vec![0.0; b.len()];
    for i in 0..count {
        for (x, q) in x.iter_mut().zip(&basis[i]) {
            *x += y[i] * q;
        }
    }
    let residual = norm(&a(&x).iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>()) / beta;
    (x, count, residual)
}

fn solve(problem: &Reference, x: Vec<f64>) -> Result<(Vec<f64>, usize, usize)> {
    solve_continuation(problem, x, 0.)
}

fn diagnose_newton_state(problem: &Reference, x: &[f64]) {
    let q = problem.rates(x, true);
    let residual = problem.residual(x);
    let mut order: Vec<_> = residual
        .chunks_exact(6)
        .enumerate()
        .map(|(i, r)| (i, norm(r)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, error) in order.into_iter().take(8) {
        let c = &problem.contacts[i];
        let local = &x[6 * i..6 * i + 6];
        let (soil, relative) = Reference::velocities(c, local, &q, true);
        let k = xyz(local);
        let w = xyz(&local[3..]);
        let lambda = c.normal.dot(k);
        let slip = relative - c.normal * c.normal.dot(relative);
        let tangent = k - c.normal * lambda;
        eprintln!(
            "Newton failed slot={} error={error:e} mass={} rho={} mu={} smoothing={} normal={:?} wall={:?} lambda={lambda:e} separation={:e} tangent={tangent:?} slip={slip:?} w={w:?} soil={soil:?}",
            c.index,
            c.mass,
            c.rho,
            c.mu,
            c.smoothing,
            c.normal,
            c.wall,
            c.normal.dot(relative) + c.gap
        );
    }
}

fn solve_continuation(
    problem: &Reference,
    mut x: Vec<f64>,
    mut smoothing: f64,
) -> Result<(Vec<f64>, usize, usize)> {
    assert!(smoothing.is_finite() && smoothing >= 0.);
    let mut work = problem.clone();
    let mut visited = vec![false; problem.contacts.len()];
    let mut products = 0;
    let mut stalled = 0;
    for step in 0..64 {
        for c in &mut work.contacts {
            c.smoothing = smoothing;
        }
        let problem = &work;
        let f = problem.residual(&x);
        let error = norm(&f);
        if !error.is_finite() {
            return Err(Error::Solver("global Newton nonfinite residual".into()));
        }
        if problem.physical_error(&x) <= 1e-8 {
            return Ok((x, step, products));
        }
        let linear = Linearization::new(problem, &x);
        let rhs: Vec<_> = f.iter().map(|v| -v).collect();
        let direct = problem
            .condensed_newton
            .then(|| condensed::undamped(&linear, &rhs))
            .flatten()
            .and_then(|d| {
                products += 1;
                let relative = norm(
                    &linear
                        .product(&d)
                        .iter()
                        .zip(&rhs)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                ) / norm(&rhs);
                eprintln!("condensed Newton step={step} relative={relative:e}");
                (relative <= 1e-5).then_some((d, 0, relative))
            });
        let (d, iterations, relative) = direct.unwrap_or_else(|| {
            let pre = problem.preconditioner(&x);
            let (y, iterations, relative) =
                gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
            products += iterations + 1;
            (apply_blocks(&pre, &y), iterations, relative)
        });
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..16 {
            let mut trial: Vec<_> = x.iter().zip(&d).map(|(x, d)| x + alpha * d).collect();
            problem.canonicalize(&mut trial);
            let merit = norm(&problem.residual(&trial));
            if merit < error * (1.0 - 1e-4 * alpha) {
                accepted = Some(trial);
                break;
            }
            alpha *= 0.5;
        }
        if accepted.is_none() && problem.local_pivot {
            accepted = local_pivot::pivot(problem, &x, error, &mut visited);
        }
        if accepted.is_none() {
            let mut damping = error / (1.0 + norm(&x));
            for attempt in 0..8 {
                let (d, used) = regularized_direction(&linear, &f, damping);
                products += used;
                let mut trial: Vec<_> = x.iter().zip(&d).map(|(x, d)| x + d).collect();
                problem.canonicalize(&mut trial);
                let candidate = norm(&problem.residual(&trial));
                eprintln!(
                    "regularized attempt={attempt} damping={damping:e} residual={candidate:e}"
                );
                if candidate < error {
                    accepted = Some(trial);
                    break;
                }
                damping *= 10.0;
            }
        }
        eprintln!(
            "global Newton step={step} residual={error:e} physical={:e} Krylov={iterations} linear_relative={relative:e} direction={:e} alpha={alpha:e}",
            problem.physical_error(&x),
            norm(&d)
        );
        if accepted.is_none() {
            diagnose_newton_state(problem, &x);
        }
        x = accepted.ok_or_else(|| {
            Error::Solver(format!("global Newton line search at {step}: {error:e}"))
        })?;
        let next_error = norm(&problem.residual(&x));
        stalled = if next_error >= 0.99 * error {
            stalled + 1
        } else {
            0
        };
        if smoothing > 0.
            && (next_error <= smoothing || (problem.stagnation_smoothing && stalled >= 2))
        {
            smoothing *= 0.1;
            stalled = 0;
            if smoothing < 1e-10 {
                smoothing = 0.;
            }
            eprintln!("global Newton smoothing={smoothing:e}");
        }
    }
    diagnose_newton_state(&work, &x);
    Err(Error::Solver("global Newton iteration limit".into()))
}

fn verify_solution(f: &Fixture, problem: &Reference, x: &[f64]) {
    let q = problem.rates(x, true);
    let mut outputs = vec![SoilWallFrictionOutput::default(); f.contacts.len()];
    for (c, x) in problem.contacts.iter().zip(x.chunks_exact(6)) {
        let k = xyz(x);
        let w = xyz(&x[3..]);
        let (soil, relative) = Reference::velocities(c, x, &q, true);
        let bits = (0..3).fold(0, |bits, i| {
            bits | if c.wall[i] * w[i] > 0.0 { 1 << i } else { 0 }
        });
        outputs[c.index] = SoilWallFrictionOutput {
            machine_impulse_normal: pack(-k, c.normal.dot(k)),
            soil_velocity_energy: pack(
                soil,
                0.5 * c.mass * (soil.length_squared() - c.soil.length_squared()),
            ),
            machine_velocity_residual: pack(soil - relative, 0.0),
            wall_impulse: pack(w, 0.0),
            status: [0, 0, 0, bits],
        };
    }
    verify_precise_free_tolerance(f, &problem.free, &q, &outputs, 1e-6);
    for c in &problem.contacts {
        let normal = vec(f.contacts[c.index].normal_gap);
        let soil = normal.dot(vec(outputs[c.index].soil_velocity_energy));
        let machine = c
            .rows
            .iter()
            .zip(&q)
            .map(|(r, q)| normal.dot(*r) * q)
            .sum::<f64>();
        let scale = 1.0_f64.max(soil.abs()).max(machine.abs()).max(c.gap);
        assert!(
            soil - machine + c.gap >= -1e-6 * scale,
            "strict normal gate at {}",
            c.index
        );
    }
}

#[test]
fn global_newton_directional_derivative() {
    directional_derivative(false, false, 0.);
}

#[test]
fn global_newton_fischer_directional_derivative() {
    directional_derivative(true, false, 0.);
}

#[test]
fn global_newton_prepared_fischer_directional_derivative() {
    directional_derivative(true, true, 0.);
}

#[test]
fn global_newton_smoothed_directional_derivative() {
    directional_derivative(true, false, 0.01);
    directional_derivative(true, true, 0.01);
}

#[test]
fn global_newton_smoothed_fischer() {
    for epsilon in [1e-8, 0.01, 1.] {
        for (a, b) in [(0., 0.), (0.2, 0.3), (-0.2, 0.3), (0.2, -0.3), (-0.2, -0.3)] {
            let (f, da, db) = smooth_fischer(a, b, epsilon);
            let expected = a + b - a.hypot(b).hypot(std::f64::consts::SQRT_2 * epsilon);
            assert!((f - expected).abs() <= 1e-14 * (1. + expected.abs()));
            let h = 1e-5 * epsilon;
            let numerical_a = (smooth_fischer(a + h, b, epsilon).0
                - smooth_fischer(a - h, b, epsilon).0)
                / (2. * h);
            let numerical_b = (smooth_fischer(a, b + h, epsilon).0
                - smooth_fischer(a, b - h, epsilon).0)
                / (2. * h);
            let tolerance = 1e-9 + 1e-15 / h;
            assert!((da - numerical_a).abs() < tolerance);
            assert!((db - numerical_b).abs() < tolerance);
        }
        assert!(smooth_fischer(epsilon, epsilon, epsilon).0.abs() < 1e-14 * epsilon);
    }
    assert_eq!(smooth_fischer(0., 0., 0.), fischer(0., 0.));
    let (f, da, db) = smooth_fischer(1e20, 1e-10, 0.01);
    assert!((f / 1e-10 - 1.).abs() < 2e-14);
    assert!(da > 0. && db > 0.);
}

#[test]
fn global_newton_fischer_complementarity() {
    for (a, b) in [(0., 0.), (0., 3.), (4., 0.), (1e30, 0.)] {
        assert_eq!(fischer(a, b).0, 0.);
    }
    for (a, b) in [
        (0.2, 0.3),
        (-0.2, 0.3),
        (0.2, -0.3),
        (-0.2, -0.3),
        (0., -3.),
    ] {
        let (f, da, db) = fischer(a, b);
        assert!((f - (a + b - a.hypot(b))).abs() < 1e-14);
        let eps = 1e-6;
        assert!((da - (fischer(a + eps, b).0 - fischer(a - eps, b).0) / (2. * eps)).abs() < 1e-8);
        assert!((db - (fischer(a, b + eps).0 - fischer(a, b - eps).0) / (2. * eps)).abs() < 1e-8);
    }
    assert!((fischer(1e20, 1e-10).0 / 1e-10 - 1.).abs() < 1e-14);
}

fn directional_derivative(fischer: bool, prepared: bool, smoothing: f64) {
    directional_derivative_conical(fischer, prepared, smoothing, false);
}

#[test]
fn global_newton_conical_contact_derivative() {
    for prepared in [false, true] {
        for smoothing in [0., 0.01] {
            directional_derivative_conical(true, prepared, smoothing, true);
        }
    }
}

fn directional_derivative_conical(fischer: bool, prepared: bool, smoothing: f64, conical: bool) {
    let f = articulated(7);
    let mut problem = Reference::new(&f, f.free.iter().map(|&x| x as f64).collect());
    for contact in &mut problem.contacts {
        contact.fischer = fischer;
        contact.conical = conical;
        contact.smoothing = smoothing;
    }
    if prepared {
        let n = problem.free.len();
        let mut response = Vec::new();
        for c in &problem.contacts {
            for i in 0..n {
                response.push(pack(
                    (0..n)
                        .map(|j| c.rows[j] * problem.inverse[i * n + j])
                        .sum::<DVec3>(),
                    0.,
                ));
            }
        }
        problem.use_prepared_response(&response);
    }
    for seed in 0..8 {
        let x: Vec<_> = (0..42)
            .map(|i| 0.31 * ((i * 17 + seed * 11) as f64 + 0.2).sin())
            .collect();
        let d: Vec<_> = (0..42)
            .map(|i| ((i * 13 + seed * 7) as f64).cos())
            .collect();
        let analytic = problem.product(&x, &d);
        let linear = Linearization::new(&problem, &x);
        let cached = linear.product(&d);
        let dual: Vec<_> = (0..x.len()).map(|i| (i as f64 + 0.7).sin()).collect();
        let left = dot(&cached, &dual);
        let right = dot(&d, &linear.transpose(&dual));
        assert!((left - right).abs() <= 1e-12 * (1.0 + left.abs() + right.abs()));
        for (a, b) in analytic.iter().zip(&cached) {
            assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()));
        }
        let eps = 1e-6;
        let plus = problem.residual(
            &x.iter()
                .zip(&d)
                .map(|(x, d)| x + eps * d)
                .collect::<Vec<_>>(),
        );
        let minus = problem.residual(
            &x.iter()
                .zip(&d)
                .map(|(x, d)| x - eps * d)
                .collect::<Vec<_>>(),
        );
        for (i, actual) in analytic.iter().enumerate() {
            let expected = (plus[i] - minus[i]) / (2.0 * eps);
            assert!(
                (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
                "axis {i}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn global_newton_physical_fixtures() {
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [
        pair(false, 0.0),
        pair(false, 0.3),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
    ] {
        let problem = Reference::new(&f, f.free.iter().map(|&x| x as f64).collect());
        let (x, steps, products) = solve(&problem, vec![0.0; 6 * problem.contacts.len()]).unwrap();
        verify_solution(&f, &problem, &x);
        eprintln!(
            "global Newton solved contacts={} steps={steps} products={products}",
            problem.contacts.len()
        );
    }
}

#[test]
fn global_newton_krylov_and_regularization() {
    let a = |x: &[f64]| vec![4.0 * x[0] + x[1], -2.0 * x[0] + 3.0 * x[1]];
    let rhs = a(&[2.0, -3.0]);
    let (x, steps, error) = gmres(a, &rhs, 8);
    assert_eq!(steps, 2);
    assert!(error < 1e-12 && (x[0] - 2.0).abs() < 1e-12 && (x[1] + 3.0).abs() < 1e-12);
    let f = articulated(7);
    let problem = Reference::new(&f, f.free.iter().map(|&x| x as f64).collect());
    let x = vec![0.0; 42];
    let linear = Linearization::new(&problem, &x);
    let residual = problem.residual(&x);
    let (d, products) = regularized_direction(&linear, &residual, 1e-3);
    assert!((1..=129).contains(&products));
    let mut linear_error = linear.product(&d);
    for (value, r) in linear_error.iter_mut().zip(&residual) {
        *value += r;
    }
    let normal = linear.transpose(&linear_error);
    let defect: Vec<_> = normal.iter().zip(&d).map(|(a, d)| a + 1e-3 * d).collect();
    assert!(norm(&defect) < 1e-6 * norm(&linear.transpose(&residual)));
    assert!(dot(&linear.transpose(&residual), &d) < 0.0);
    let mut trial: Vec<_> = x.iter().zip(d).map(|(x, d)| x + 0.01 * d).collect();
    problem.canonicalize(&mut trial);
    assert!(norm(&problem.residual(&trial)) < norm(&residual));
}

#[test]
#[ignore = "requires saved actual shared inputs"]
fn global_newton_captured_inputs() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let problem = Reference::new(&f, combined(&pairs));
    let now = std::time::Instant::now();
    let (x, steps, products) = solve(&problem, vec![0.0; 6 * problem.contacts.len()]).unwrap();
    verify_solution(&f, &problem, &x);
    eprintln!(
        "global Newton captured contacts={} steps={steps} products={products} wall_ms={}",
        problem.contacts.len(),
        now.elapsed().as_secs_f64() * 1000.0
    );
}
