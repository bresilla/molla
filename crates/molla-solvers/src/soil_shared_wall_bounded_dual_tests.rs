use super::*;

#[test]
fn shared_wall_single_float_rate_capacity_cpu() {
    let mass = 1_000_000.0_f64;
    let impulse = (mass / (mass + 1.0)) as f32;
    let exact_rate = 1.0 - impulse as f64 / mass;
    let high = exact_rate as f32;
    let low = (exact_rate - high as f64) as f32;
    let combined = high as f64 + low as f64;
    let rounded_residual = mass * (high as f64 - 1.0) + impulse as f64;
    let precise_residual = mass * (combined - 1.0) + impulse as f64;
    let cap = 0.25e-6 * (1.0 + impulse as f64);
    let nearest_distance = (high as f64 - exact_rate).abs();
    assert!((high.next_down() as f64 - exact_rate).abs() > nearest_distance);
    assert!((high.next_up() as f64 - exact_rate).abs() > nearest_distance);
    assert!(rounded_residual.abs() - cap > 0.01);
    assert!(precise_residual.abs() < 1e-6);
    let k = DVec3::X * impulse as f64;
    let f = Fixture {
        mass: vec![mass as f32],
        free: vec![1.0],
        rows: vec![pack(DVec3::X, 0.0)],
        contacts: vec![SoilSharedWallContact {
            soil_mass: [0.0, 0.0, 0.0, 1.0],
            normal_gap: pack(DVec3::X, 0.0),
            friction: [0.0; 4],
            boundary: [0; 4],
        }],
    };
    let output = SoilWallFrictionOutput {
        machine_impulse_normal: pack(-k, impulse as f64),
        soil_velocity_energy: pack(k, 0.5 * k.length_squared()),
        machine_velocity_residual: pack(DVec3::X * combined, 0.0),
        ..Default::default()
    };
    verify_precise_tolerance(&f, &[combined], &[output], 1e-6);
    eprintln!(
        "single-float momentum residual={rounded_residual}; high+low residual={precise_residual}"
    );
}

fn ball_plane(v: DVec3, cap: f64, normal: DVec3, lambda: f64) -> (DVec3, [[f64; 3]; 3]) {
    let size = v.length();
    let scale = if size > cap { cap / size } else { 1.0 };
    let mut p = v * scale;
    let mut derivative = [[0.0; 3]; 3];
    if p.dot(normal) < -lambda {
        let tangent = v - v.dot(normal) * normal;
        let length = tangent.length();
        let radius = (cap * cap - lambda * lambda).max(0.0).sqrt();
        let scale = if length > radius {
            radius / length
        } else {
            1.0
        };
        p = -lambda * normal + scale * tangent;
        for i in 0..3 {
            for j in 0..3 {
                derivative[i][j] = scale * (f64::from(i == j) - normal[i] * normal[j]);
                if length > radius {
                    derivative[i][j] -= scale * (tangent[i] / length) * (tangent[j] / length);
                }
            }
        }
    } else {
        for i in 0..3 {
            for j in 0..3 {
                derivative[i][j] = scale * f64::from(i == j);
                if size > cap {
                    derivative[i][j] -= scale * (v[i] / size) * (v[j] / size);
                }
            }
        }
    }
    (p, derivative)
}

pub(super) fn project(f: &Fixture, rates: &[f32], outputs: &mut [SoilWallFrictionOutput]) {
    let n = rates.len();
    let mut residual = vec![0.0; n];
    let mut gates = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
            residual[i] += p;
            gates[i] += p.abs();
        }
        for (c, o) in outputs.iter().enumerate() {
            let row = vec(f.rows[c * n + i]);
            residual[i] += row.dot(-vec(o.machine_impulse_normal));
            gates[i] += row.dot(vec(o.machine_impulse_normal)).abs();
        }
        gates[i] *= 0.98e-6;
    }
    let mut caps = vec![0.0; outputs.len()];
    for (c, o) in outputs.iter().enumerate() {
        let mass = f.contacts[c].soil_mass[3] as f64;
        let old = -vec(o.machine_impulse_normal);
        if mass == 0.0 || old.dot(vec(f.contacts[c].normal_gap).normalize()) <= 0.0 {
            continue;
        }
        let machine = (0..n)
            .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
            .sum::<DVec3>();
        caps[c] = (1.0 - 8.0 * f32::EPSILON as f64)
            * (0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length())).min(
                0.25e-6 * mass * (1.0 + vec(o.soil_velocity_energy).length() + machine.length()),
            );
    }
    let evaluate = |w: &[f64]| {
        let mut deltas = vec![DVec3::ZERO; outputs.len()];
        let mut gradient = residual.clone();
        let mut hessian = vec![0.0; n * n];
        let mut objective = (0..n)
            .map(|i| gates[i] * w[i].abs() - residual[i] * w[i])
            .sum::<f64>();
        for (c, cap) in caps.iter().enumerate() {
            if *cap == 0.0 {
                continue;
            }
            let mass = f.contacts[c].soil_mass[3] as f64;
            let v = -mass * (0..n).map(|i| w[i] * vec(f.rows[c * n + i])).sum::<DVec3>();
            let normal = vec(f.contacts[c].normal_gap).normalize();
            let lambda = (-vec(outputs[c].machine_impulse_normal)).dot(normal);
            let (p, derivative) = ball_plane(v, *cap, normal, lambda);
            deltas[c] = p;
            objective += (v.dot(p) - 0.5 * p.length_squared()) / mass;
            for i in 0..n {
                let a = vec(f.rows[c * n + i]);
                gradient[i] += a.dot(p);
                for j in 0..n {
                    let b = vec(f.rows[c * n + j]);
                    let db = DVec3::from_array(std::array::from_fn(|k| {
                        (0..3).map(|l| derivative[k][l] * b[l]).sum()
                    }));
                    hessian[i * n + j] += mass * a.dot(db);
                }
            }
        }
        (objective, gradient, hessian, deltas)
    };
    let mut weights = vec![0.0; n];
    let mut solution = None;
    for iteration in 0..8 {
        let (objective, gradient, mut hessian, deltas) = evaluate(&weights);
        let worst = (0..n)
            .map(|i| gradient[i].abs() / gates[i])
            .fold(0.0_f64, f64::max);
        eprintln!("bounded dual iteration={iteration} ratio={worst} objective={objective}");
        if 0.98 * worst <= 1.0 {
            solution = Some(deltas);
            break;
        }
        let mut rhs = vec![0.0; n];
        let mut signs = vec![0.0; n];
        for i in 0..n {
            signs[i] = if weights[i] != 0.0 {
                weights[i].signum()
            } else if gradient[i].abs() > gates[i] {
                gradient[i].signum()
            } else {
                0.0
            };
            rhs[i] = gradient[i] - gates[i] * signs[i];
        }
        for i in 0..n {
            if signs[i] == 0.0 {
                rhs[i] = 0.0;
                for j in 0..n {
                    hessian[i * n + j] = 0.0;
                    hessian[j * n + i] = 0.0;
                }
            }
        }
        let Some(step) = coupled_weights(&hessian, &rhs) else {
            return;
        };
        let mut fraction = 1.0;
        let mut found = false;
        for _ in 0..8 {
            let candidate: Vec<_> = (0..n)
                .map(|i| {
                    let q = weights[i] + fraction * step[i];
                    if signs[i] * q > 0.0 { q } else { 0.0 }
                })
                .collect();
            if evaluate(&candidate).0 < objective {
                weights = candidate;
                found = true;
                break;
            }
            fraction *= 0.5;
        }
        if !found {
            eprintln!("bounded dual line search rejected");
            return;
        }
    }
    if solution.is_none() {
        let (_, gradient, _, deltas) = evaluate(&weights);
        if (0..n).all(|i| 0.98 * gradient[i].abs() <= gates[i]) {
            solution = Some(deltas);
        }
    }
    let Some(deltas) = solution else {
        eprintln!("bounded dual exhausted");
        return;
    };
    for (c, o) in outputs.iter_mut().enumerate() {
        if caps[c] == 0.0 {
            continue;
        }
        assert!(deltas[c].length() <= caps[c] / (1.0 - 8.0 * f32::EPSILON as f64));
        let k = -vec(o.machine_impulse_normal) + deltas[c];
        let lambda = k.dot(vec(f.contacts[c].normal_gap).normalize());
        assert!(lambda >= 0.0);
        o.machine_impulse_normal = pack(-k, lambda);
        for axis in 0..3 {
            o.soil_velocity_energy[axis] = f.contacts[c].soil_mass[axis]
                + (-o.machine_impulse_normal[axis] + o.wall_impulse[axis])
                    / f.contacts[c].soil_mass[3];
        }
    }
}

#[test]
fn shared_wall_bounded_ball_plane_cpu() {
    for v in [
        DVec3::new(0.1, 0.2, 0.3),
        DVec3::new(-2.0, 0.5, 0.3),
        DVec3::new(-2.0, 3.0, 1.0),
        DVec3::new(-0.4, 0.5, 0.0),
        DVec3::new(0.4, 3.0, 1.0),
    ] {
        let (p, d) = ball_plane(v, 1.0, DVec3::X, 0.2);
        assert!(p.length() <= 1.0 + 1e-14 && p.x >= -0.2);
        for j in 0..3 {
            let mut a = v;
            let mut b = v;
            a[j] += 1e-6;
            b[j] -= 1e-6;
            let difference =
                (ball_plane(a, 1.0, DVec3::X, 0.2).0 - ball_plane(b, 1.0, DVec3::X, 0.2).0) / 2e-6;
            for i in 0..3 {
                assert!((difference[i] - d[i][j]).abs() < 1e-8);
            }
        }
    }
}
