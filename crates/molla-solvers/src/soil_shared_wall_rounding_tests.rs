use super::*;

#[path = "soil_shared_wall_bounded_dual_tests.rs"]
mod bounded_dual;

fn represented_rates(f: &Fixture, initial: &[f64], outputs: &[SoilWallFrictionOutput]) -> Vec<f32> {
    let n = f.free.len();
    let mut rates: Vec<_> = initial.iter().map(|q| *q as f32).collect();
    let mut impulse = vec![0.0; n];
    let mut gate = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            gate[i] +=
                (f.mass[i * n + j] as f64).abs() * (rates[j] as f64 - f.free[j] as f64).abs();
        }
        for (c, output) in outputs.iter().enumerate() {
            let row = vec(f.rows[c * n + i]);
            let k = -vec(output.machine_impulse_normal);
            impulse[i] += row.dot(k);
            gate[i] += row.abs().dot(k.abs());
        }
        gate[i] *= 1e-6;
    }
    let residual = |q: &[f32]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                impulse[i]
                    + (0..n)
                        .map(|j| f.mass[i * n + j] as f64 * (q[j] as f64 - f.free[j] as f64))
                        .sum::<f64>()
            })
            .collect()
    };
    for _ in 0..16 {
        let r = residual(&rates);
        if r.iter().zip(&gate).all(|(r, g)| r.abs() <= *g) {
            break;
        }
        let score = |v: &[f64]| {
            v.iter()
                .zip(&gate)
                .map(|(v, g)| (v / g).abs())
                .fold(0.0_f64, f64::max)
        };
        let mut best = score(&r);
        let mut selected = None;
        for j in 0..n {
            let interval = |score: f64| {
                let radius = 1e-6 * (1.0 + initial[j].abs());
                let mut low = initial[j] - radius - rates[j] as f64;
                let mut high = initial[j] + radius - rates[j] as f64;
                for i in 0..n {
                    let a = f.mass[i * n + j] as f64;
                    if a == 0.0 {
                        if r[i].abs() > score * gate[i] {
                            return None;
                        }
                        continue;
                    }
                    let x = (-score * gate[i] - r[i]) / a;
                    let y = (score * gate[i] - r[i]) / a;
                    low = low.max(x.min(y));
                    high = high.min(x.max(y));
                    if low > high {
                        return None;
                    }
                }
                Some((low, high))
            };
            let mut low = 0.0;
            let mut high = score(&r);
            for _ in 0..12 {
                let middle = 0.5 * (low + high);
                if interval(middle).is_some() {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            let Some((low, high)) = interval(high) else {
                continue;
            };
            let center = (rates[j] as f64 + 0.0_f64.clamp(low, high)) as f32;
            for q in [center, center.next_down(), center.next_up()] {
                if !q.is_finite() || (q as f64 - initial[j]).abs() > 1e-6 * (1.0 + initial[j].abs())
                {
                    continue;
                }
                let mut candidate = rates.clone();
                candidate[j] = q;
                let value = score(&residual(&candidate));
                if value < best {
                    best = value;
                    selected = Some((j, q));
                }
            }
        }
        let Some((j, q)) = selected else {
            break;
        };
        rates[j] = q;
    }
    rates
}

#[test]
#[ignore = "CPU only; requires saved input/solve and fresh output paths"]
fn shared_wall_captured_represented_rates_cpu() {
    let path =
        |key| std::path::PathBuf::from(std::env::var_os(key).expect("capture path required"));
    let f = captured_fixture(&path("SOIL_WALL_INPUT_REPLAY"));
    let saved = path("SOIL_WALL_SAVED_SOLVE");
    let output = path("SOIL_WALL_PROBE_OUTPUT");
    std::fs::create_dir(&output).unwrap();
    let n = f.free.len();
    let status = std::fs::read(saved.join("status.bin")).unwrap();
    let rates: Vec<_> = status[32..32 + 8 * n]
        .chunks_exact(8)
        .map(|b| {
            f32::from_ne_bytes(b[..4].try_into().unwrap()) as f64
                + f32::from_ne_bytes(b[4..].try_into().unwrap()) as f64
        })
        .collect();
    let bytes = std::fs::read(saved.join("outputs.bin")).unwrap();
    let mut outputs: Vec<SoilWallFrictionOutput> = bytes
        .chunks_exact(80)
        .take(f.contacts.len())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let before = captured_momentum_diagnostics(&f, &rates, &outputs);
    let impulse_projection = std::env::var_os("SOIL_WALL_CPU_IMPULSE_PROJECTION").is_some();
    let corrected = if std::env::var_os("SOIL_WALL_CPU_NEIGHBOR_RECONCILIATION").is_some() {
        neighbor_reconciliation(&f, &rates, &mut outputs)
    } else if impulse_projection {
        let q: Vec<_> = rates.iter().map(|q| *q as f32).collect();
        if std::env::var_os("SOIL_WALL_CPU_COUPLED_PROJECTION").is_some() {
            project_coupled(&f, &q, &mut outputs);
        } else {
            project_impulses(&f, &q, &mut outputs);
        }
        q
    } else {
        represented_rates(&f, &rates, &outputs)
    };
    let report = serde_json::json!({"acceptance": false,
        "impulse_projection": impulse_projection,
        "coupled_projection": std::env::var_os("SOIL_WALL_CPU_COUPLED_PROJECTION").is_some(),
        "redistribution": std::env::var_os("SOIL_WALL_CPU_REDISTRIBUTE").is_some(),
        "momentum_slabs": std::env::var_os("SOIL_WALL_CPU_MOMENTUM_SLABS").is_some(),
        "bounded_dual": std::env::var_os("SOIL_WALL_CPU_BOUNDED_DUAL").is_some(),
        "neighbor_reconciliation": std::env::var_os("SOIL_WALL_CPU_NEIGHBOR_RECONCILIATION").is_some(),
        "scope": "CPU prototype of bounded represented-rate search; not production or GPU acceptance",
        "before": before,
        "after": captured_momentum_diagnostics(&f, &corrected.iter().map(|q| *q as f64).collect::<Vec<_>>(), &outputs),
        "rates_before": rates, "rates_after": corrected});
    std::fs::write(
        output.join("projection.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    for (c, out) in outputs.iter_mut().enumerate() {
        if f.contacts[c].soil_mass[3] == 0.0 {
            continue;
        }
        let machine = (0..n)
            .map(|i| vec(f.rows[c * n + i]) * corrected[i] as f64)
            .sum::<DVec3>();
        out.machine_velocity_residual = pack(machine, 0.0);
        let soil = vec(out.soil_velocity_energy);
        out.soil_velocity_energy[3] = (0.5
            * f.contacts[c].soil_mass[3] as f64
            * (soil.length_squared() - vec(f.contacts[c].soil_mass).length_squared()))
            as f32;
    }
    verify_tolerance(&f, &corrected, &outputs, 1e-6);
}

fn neighbor_reconciliation(
    f: &Fixture,
    initial: &[f64],
    outputs: &mut [SoilWallFrictionOutput],
) -> Vec<f32> {
    let nearest: Vec<_> = initial.iter().map(|q| *q as f32).collect();
    let original = outputs.to_vec();
    for trial in 0..=2 * initial.len() {
        let mut rates = nearest.clone();
        if trial > 0 {
            let i = (trial - 1) / 2;
            rates[i] = if trial % 2 == 1 {
                rates[i].next_down()
            } else {
                rates[i].next_up()
            };
            if !rates[i].is_finite()
                || (rates[i] as f64 - initial[i]).abs() > 1e-6 * (1.0 + initial[i].abs())
            {
                continue;
            }
        }
        let mut candidate = original.clone();
        project_coupled(f, &rates, &mut candidate);
        let report = captured_momentum_diagnostics(
            f,
            &rates.iter().map(|q| *q as f64).collect::<Vec<_>>(),
            &candidate,
        );
        if report["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["rounded_gate_ratio"].as_f64().is_some_and(|r| r <= 1.0))
        {
            eprintln!(
                "neighbor reconciliation selected trial={trial} (zero=nearest, then down/up per DOF)"
            );
            outputs.copy_from_slice(&candidate);
            return rates;
        }
    }
    eprintln!("neighbor reconciliation found no momentum-feasible candidate");
    nearest
}

fn coupled_weights(matrix: &[f64], residual: &[f64]) -> Option<Vec<f64>> {
    let n = residual.len();
    if n == 0
        || n > 32
        || matrix.len() != n * n
        || matrix.iter().chain(residual).any(|x| !x.is_finite())
    {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = residual.to_vec();
    let mut order: Vec<_> = (0..n).collect();
    let mut scale = vec![1.0; n];
    for i in 0..n {
        if a[i * n + i] < 0.0 {
            return None;
        }
        if a[i * n + i] > 0.0 {
            scale[i] = a[i * n + i].sqrt();
        }
        b[i] /= scale[i];
    }
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = a[i * n + j] / scale[i] / scale[j];
        }
    }
    let mut rank = 0;
    for k in 0..n {
        let p = (k..n)
            .max_by(|i, j| a[i * n + i].total_cmp(&a[j * n + j]))
            .unwrap();
        if a[p * n + p] <= 1e-6 {
            break;
        }
        for j in 0..n {
            a.swap(k * n + j, p * n + j);
        }
        for i in 0..n {
            a.swap(i * n + k, i * n + p);
        }
        b.swap(k, p);
        order.swap(k, p);
        let pivot = a[k * n + k];
        for j in 0..n {
            a[k * n + j] /= pivot;
        }
        b[k] /= pivot;
        for i in 0..n {
            if i == k {
                continue;
            }
            let multiplier = a[i * n + k];
            for j in 0..n {
                a[i * n + j] -= multiplier * a[k * n + j];
            }
            b[i] -= multiplier * b[k];
        }
        rank += 1;
    }
    let mut weights = vec![0.0; n];
    for i in 0..rank {
        weights[order[i]] = b[i] / scale[order[i]];
    }
    weights.iter().all(|x| x.is_finite()).then_some(weights)
}

#[test]
#[ignore = "CPU only; requires captured input, solve, and fresh output paths"]
fn shared_wall_captured_correction_capacity_cpu() {
    let rounding_envelope = std::env::var_os("SOIL_WALL_CPU_CAPACITY_ROUNDING").is_some();
    let path =
        |key| std::path::PathBuf::from(std::env::var_os(key).expect("capture path required"));
    let f = captured_fixture(&path("SOIL_WALL_INPUT_REPLAY"));
    let saved = path("SOIL_WALL_SAVED_SOLVE");
    let output = path("SOIL_WALL_PROBE_OUTPUT");
    std::fs::create_dir(&output).unwrap();
    let n = f.free.len();
    let status = std::fs::read(saved.join("status.bin")).unwrap();
    let rates: Vec<_> = status[32..32 + 8 * n]
        .chunks_exact(8)
        .map(|b| {
            (f32::from_ne_bytes(b[..4].try_into().unwrap()) as f64
                + f32::from_ne_bytes(b[4..].try_into().unwrap()) as f64) as f32
        })
        .collect();
    let bytes = std::fs::read(saved.join("outputs.bin")).unwrap();
    let outputs: Vec<SoilWallFrictionOutput> = bytes
        .chunks_exact(80)
        .take(f.contacts.len())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let mut residual = vec![0.0; n];
    let mut gates = vec![1.0; n];
    let mut caps = vec![0.0; outputs.len()];
    for i in 0..n {
        for j in 0..n {
            let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
            residual[i] += p;
            gates[i] += p.abs();
        }
        for (c, o) in outputs.iter().enumerate() {
            let row = vec(f.rows[c * n + i]);
            residual[i] += row.dot(-vec(o.machine_impulse_normal));
            gates[i] += row.abs().dot(vec(o.machine_impulse_normal).abs());
        }
        gates[i] *= 1e-6;
    }
    for (c, o) in outputs.iter().enumerate() {
        let mass = f.contacts[c].soil_mass[3] as f64;
        let old = -vec(o.machine_impulse_normal);
        if mass == 0.0 || old.dot(vec(f.contacts[c].normal_gap).normalize()) <= 0.0 {
            continue;
        }
        let machine = (0..n)
            .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
            .sum::<DVec3>();
        caps[c] = (0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length()))
            .min(0.25e-6 * mass * (1.0 + vec(o.soil_velocity_energy).length() + machine.length()));
        if rounding_envelope {
            let ulps = DVec3::from_array(std::array::from_fn(|axis| {
                let magnitude = (old[axis].abs() + caps[c]) as f32;
                magnitude.next_up() as f64 - magnitude as f64
            }));
            caps[c] = caps[c] * (1.0 + 8.0 * f32::EPSILON as f64) + ulps.length();
            assert!(caps[c].is_finite());
        }
        for i in 0..n {
            gates[i] += 1e-6 * caps[c] * vec(f.rows[c * n + i]).length();
        }
    }
    let separation = |w: &[f64]| {
        let mut value = (0..n)
            .map(|i| residual[i] * w[i] - gates[i] * w[i].abs())
            .sum::<f64>();
        for (c, cap) in caps.iter().enumerate() {
            if *cap == 0.0 {
                continue;
            }
            let g = (0..n).map(|i| w[i] * vec(f.rows[c * n + i])).sum::<DVec3>();
            let normal = vec(f.contacts[c].normal_gap).normalize();
            let lambda = (-vec(outputs[c].machine_impulse_normal)).dot(normal);
            let gn = g.dot(normal);
            let support = if !rounding_envelope && *cap * gn > lambda * g.length() {
                lambda * gn
                    + (cap * cap - lambda * lambda).max(0.0).sqrt() * (g - gn * normal).length()
            } else {
                cap * g.length()
            };
            value -= support;
        }
        value
    };
    let selected = (0..n)
        .max_by(|i, j| {
            (residual[*i] / gates[*i])
                .abs()
                .total_cmp(&(residual[*j] / gates[*j]).abs())
        })
        .unwrap();
    let mut weights = vec![0.0; n];
    weights[selected] = residual[selected].signum();
    for _ in 0..8 {
        for i in 0..n {
            if i == selected {
                continue;
            }
            let mut lo = -16.0;
            let mut hi = 16.0;
            let old = weights[i];
            let score = separation(&weights);
            for _ in 0..48 {
                let a = lo + (hi - lo) / 3.0;
                let b = hi - (hi - lo) / 3.0;
                weights[i] = a;
                let sa = separation(&weights);
                weights[i] = b;
                if sa < separation(&weights) {
                    lo = a;
                } else {
                    hi = b;
                }
            }
            weights[i] = 0.5 * (lo + hi);
            if separation(&weights) < score {
                weights[i] = old;
            }
        }
    }
    let gap = separation(&weights);
    let report = serde_json::json!({"acceptance":false,
        "scope":"f64 numerical separation bound at fixed nearest rates; optional enlarged balls include a full impulse ULP per axis and 8-epsilon norm margin, dropping normal halfspaces; not an exhaustive represented-rate search or interval-arithmetic proof",
        "rounding_envelope":rounding_envelope,
        "residual":residual,"gates_upper_bound":gates,"weights":weights,
        "separation_gap":gap,"positive_gap":gap>0.0});
    std::fs::write(
        output.join("capacity.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("bounded correction separation gap={gap} weights={weights:?}");
}

fn project_coupled(f: &Fixture, rates: &[f32], outputs: &mut [SoilWallFrictionOutput]) {
    if std::env::var_os("SOIL_WALL_CPU_BOUNDED_DUAL").is_some() {
        bounded_dual::project(f, rates, outputs);
        return;
    }
    let n = rates.len();
    let active: Vec<_> = outputs
        .iter()
        .enumerate()
        .map(|(c, o)| {
            f.contacts[c].soil_mass[3] > 0.0
                && (-vec(o.machine_impulse_normal)).dot(vec(f.contacts[c].normal_gap).normalize())
                    > 0.0
        })
        .collect();
    let mut gram = vec![0.0; n * n];
    let mut residual = vec![0.0; n];
    let mut scale = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
            residual[i] += p;
            scale[i] += p.abs();
        }
        for (c, o) in outputs.iter().enumerate() {
            residual[i] += vec(f.rows[c * n + i]).dot(-vec(o.machine_impulse_normal));
            scale[i] += vec(f.rows[c * n + i])
                .abs()
                .dot(vec(o.machine_impulse_normal).abs());
            if active[c] {
                for j in 0..n {
                    gram[i * n + j] += f.contacts[c].soil_mass[3] as f64
                        * vec(f.rows[c * n + i]).dot(vec(f.rows[c * n + j]));
                }
            }
        }
    }
    let interior = if std::env::var_os("SOIL_WALL_CPU_MOMENTUM_SLABS").is_some() {
        0.9e-6
    } else {
        0.5e-6
    };
    for i in 0..n {
        residual[i] -= residual[i].clamp(-interior * scale[i], interior * scale[i]);
    }
    if std::env::var_os("SOIL_WALL_CPU_REDISTRIBUTE").is_some() {
        redistribute_coupled(f, rates, outputs, &active, &residual);
        return;
    }
    let Some(weights) = coupled_weights(&gram, &residual) else {
        return;
    };
    eprintln!("coupled residual={residual:?} weights={weights:?}");
    let mut step = 1.0_f64;
    for (c, o) in outputs.iter().enumerate() {
        if !active[c] {
            continue;
        }
        let mass = f.contacts[c].soil_mass[3] as f64;
        let delta = -mass
            * (0..n)
                .map(|i| weights[i] * vec(f.rows[c * n + i]))
                .sum::<DVec3>();
        let length = delta.length();
        if length == 0.0 {
            continue;
        }
        let old = -vec(o.machine_impulse_normal);
        let machine = (0..n)
            .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
            .sum::<DVec3>();
        step = step.min(0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length()) / length);
        step = step.min(
            0.25e-6 * mass * (1.0 + vec(o.soil_velocity_energy).length() + machine.length())
                / length,
        );
        let normal = vec(f.contacts[c].normal_gap).normalize();
        if delta.dot(normal) < 0.0 {
            step = step.min(-old.dot(normal) / delta.dot(normal));
        }
    }
    if step < 1.0 {
        step *= 0.9;
    }
    eprintln!("coupled bounded step={step}");
    let mut candidate = outputs.to_vec();
    for (c, o) in candidate.iter_mut().enumerate() {
        if !active[c] {
            continue;
        }
        let mass = f.contacts[c].soil_mass[3] as f64;
        let delta = -step
            * mass
            * (0..n)
                .map(|i| weights[i] * vec(f.rows[c * n + i]))
                .sum::<DVec3>();
        let old = -vec(o.machine_impulse_normal);
        let machine = (0..n)
            .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
            .sum::<DVec3>();
        if delta.length() > 0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length())
            || delta.length() / mass
                > 0.25e-6 * (1.0 + vec(o.soil_velocity_energy).length() + machine.length())
        {
            eprintln!(
                "coupled bound rejected contact={c} delta={} speed={} old={old:?}",
                delta.length(),
                delta.length() / mass
            );
            return;
        }
        let k = old + delta;
        let lambda = k.dot(vec(f.contacts[c].normal_gap).normalize());
        if lambda < 0.0 {
            eprintln!("coupled normal rejected contact={c} lambda={lambda}");
            return;
        }
        o.machine_impulse_normal = pack(-k, lambda);
        for axis in 0..3 {
            o.soil_velocity_energy[axis] = f.contacts[c].soil_mass[axis]
                + (-o.machine_impulse_normal[axis] + o.wall_impulse[axis])
                    / f.contacts[c].soil_mass[3];
        }
    }
    outputs.copy_from_slice(&candidate);
}

fn redistribute_coupled(
    f: &Fixture,
    rates: &[f32],
    outputs: &mut [SoilWallFrictionOutput],
    active: &[bool],
    residual: &[f64],
) {
    let n = rates.len();
    let mut free = active.to_vec();
    let mut deltas = vec![DVec3::ZERO; outputs.len()];
    let slabs = std::env::var_os("SOIL_WALL_CPU_MOMENTUM_SLABS").is_some();
    let mut constrained: Vec<_> = residual.iter().map(|r| !slabs || *r != 0.0).collect();
    let mut target = residual.to_vec();
    let mut solved = false;
    for pass in 0..8 {
        let mut gram = vec![0.0; n * n];
        let mut remaining = target.clone();
        for c in 0..outputs.len() {
            if !active[c] {
                continue;
            }
            for i in 0..n {
                let row = vec(f.rows[c * n + i]);
                if free[c] {
                    for j in 0..n {
                        gram[i * n + j] +=
                            f.contacts[c].soil_mass[3] as f64 * row.dot(vec(f.rows[c * n + j]));
                    }
                } else {
                    remaining[i] += row.dot(deltas[c]);
                }
            }
        }
        for i in 0..n {
            if !constrained[i] {
                remaining[i] = 0.0;
                for j in 0..n {
                    gram[i * n + j] = 0.0;
                    gram[j * n + i] = 0.0;
                }
            }
        }
        let Some(weights) = coupled_weights(&gram, &remaining) else {
            return;
        };
        let mut saturated = 0;
        for (c, o) in outputs.iter().enumerate() {
            if !free[c] {
                continue;
            }
            let mass = f.contacts[c].soil_mass[3] as f64;
            let delta = -mass
                * (0..n)
                    .map(|i| weights[i] * vec(f.rows[c * n + i]))
                    .sum::<DVec3>();
            let old = -vec(o.machine_impulse_normal);
            let machine = (0..n)
                .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
                .sum::<DVec3>();
            let size = delta.length();
            let mut step = 1.0_f64;
            if size > 0.0 {
                step =
                    step.min(0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length()) / size);
                step = step.min(
                    0.25e-6
                        * mass
                        * (1.0 + vec(o.soil_velocity_energy).length() + machine.length())
                        / size,
                );
            }
            let normal = vec(f.contacts[c].normal_gap).normalize();
            if delta.dot(normal) < 0.0 {
                step = step.min(-old.dot(normal) / delta.dot(normal));
            }
            if step < 1.0 {
                deltas[c] = (0.9 * step) * delta;
                free[c] = false;
                saturated += 1;
            } else {
                deltas[c] = delta;
            }
        }
        eprintln!("redistribution pass={pass} saturated={saturated}");
        {
            let mut added = 0;
            for i in 0..n {
                if constrained[i] {
                    continue;
                }
                let mut value = 0.0;
                let mut scale = 1.0;
                let mut correction = 0.0;
                for j in 0..n {
                    let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
                    value += p;
                    scale += p.abs();
                }
                for (c, o) in outputs.iter().enumerate() {
                    let row = vec(f.rows[c * n + i]);
                    value += row.dot(-vec(o.machine_impulse_normal));
                    correction += row.dot(deltas[c]);
                    scale += row.abs().dot(vec(o.machine_impulse_normal).abs());
                }
                if (value + correction).abs() > 1e-6 * scale {
                    constrained[i] = true;
                    target[i] = value - (value + correction).signum() * 0.9e-6 * scale;
                    eprintln!("redistribution constrains row={i} target={}", target[i]);
                    added += 1;
                }
            }
            if added > 0 {
                eprintln!("redistribution adds {added} momentum rows");
                free.copy_from_slice(active);
                deltas.fill(DVec3::ZERO);
                continue;
            }
            if saturated == 0 {
                solved = true;
                break;
            }
        }
    }
    if !solved {
        eprintln!("redistribution exhausted bounded active-set passes");
        return;
    }
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for i in 0..n {
        let mut value = 0.0;
        let mut scale = 1.0;
        let mut slope = 0.0;
        for j in 0..n {
            let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
            value += p;
            scale += p.abs();
        }
        for (c, o) in outputs.iter().enumerate() {
            let row = vec(f.rows[c * n + i]);
            value += row.dot(-vec(o.machine_impulse_normal));
            scale += row.abs().dot(vec(o.machine_impulse_normal).abs());
            slope += row.dot(deltas[c]);
        }
        let gate = 1e-6 * scale;
        if slope == 0.0 {
            if value.abs() > gate {
                return;
            }
        } else {
            let a = (-gate - value) / slope;
            let b = (gate - value) / slope;
            low = low.max(a.min(b));
            high = high.min(a.max(b));
        }
    }
    if !low.is_finite() || !high.is_finite() || low > high {
        eprintln!("redistribution rejects interval=[{low},{high}]");
        return;
    }
    let fraction = 0.5 * (low + high);
    eprintln!("redistribution feasible interval=[{low},{high}] fraction={fraction}");
    for delta in &mut deltas {
        *delta *= fraction;
    }
    let mut candidate = outputs.to_vec();
    for (c, o) in candidate.iter_mut().enumerate() {
        if !active[c] {
            continue;
        }
        let mass = f.contacts[c].soil_mass[3] as f64;
        let old = -vec(o.machine_impulse_normal);
        let machine = (0..n)
            .map(|i| rates[i] as f64 * vec(f.rows[c * n + i]))
            .sum::<DVec3>();
        assert!(
            deltas[c].length() <= 0.25e-6 * (1.0 + old.length() + vec(o.wall_impulse).length())
        );
        assert!(
            deltas[c].length() / mass
                <= 0.25e-6 * (1.0 + vec(o.soil_velocity_energy).length() + machine.length())
        );
        let k = old + deltas[c];
        let lambda = k.dot(vec(f.contacts[c].normal_gap).normalize());
        if lambda < 0.0 || !k.is_finite() {
            return;
        }
        o.machine_impulse_normal = pack(-k, lambda);
        for axis in 0..3 {
            o.soil_velocity_energy[axis] = f.contacts[c].soil_mass[axis]
                + (-o.machine_impulse_normal[axis] + o.wall_impulse[axis])
                    / f.contacts[c].soil_mass[3];
        }
    }
    outputs.copy_from_slice(&candidate);
}

#[test]
fn shared_wall_redistributed_bounded_cpu() {
    let mut f = Fixture {
        mass: vec![1.0],
        free: vec![0.0],
        rows: Vec::new(),
        contacts: Vec::new(),
    };
    let mut outputs = Vec::new();
    for c in 0..21 {
        let mass = if c == 0 { 100.0 } else { 1.0 };
        let normal = if c == 0 || c % 2 == 1 {
            DVec3::X
        } else {
            -DVec3::X
        };
        let lambda = if c == 0 { 3e-6 } else { 1e-4 };
        let k = lambda * normal;
        f.rows.push(pack(DVec3::X, 0.0));
        f.contacts.push(SoilSharedWallContact {
            soil_mass: pack(-k / mass, mass),
            normal_gap: pack(normal, 0.0),
            friction: [0.0; 4],
            boundary: [0; 4],
        });
        outputs.push(SoilWallFrictionOutput {
            machine_impulse_normal: pack(-k, lambda),
            ..Default::default()
        });
    }
    let original = outputs.clone();
    let residual = outputs
        .iter()
        .map(|o| -(o.machine_impulse_normal[0] as f64))
        .sum::<f64>();
    let scale = 1.0
        + outputs
            .iter()
            .map(|o| o.machine_impulse_normal[0].abs() as f64)
            .sum::<f64>();
    assert!(residual > 1e-6 * scale);
    redistribute_coupled(
        &f,
        &[0.0],
        &mut outputs,
        &[true; 21],
        &[residual - 0.5e-6 * scale],
    );
    verify_tolerance(&f, &[0.0], &outputs, 1e-6);
    let heavy =
        (vec(outputs[0].machine_impulse_normal) - vec(original[0].machine_impulse_normal)).length();
    let light: f64 = (1..21)
        .map(|c| {
            (vec(outputs[c].machine_impulse_normal) - vec(original[c].machine_impulse_normal))
                .length()
        })
        .sum();
    assert!(heavy <= 0.25e-6 * (1.0 + vec(original[0].machine_impulse_normal).length()));
    assert!(light > heavy);
}

#[test]
fn shared_wall_coupled_rank_and_scaling_cpu() {
    let zero = coupled_weights(&[0.0; 4], &[0.0; 2]).unwrap();
    assert_eq!(zero, vec![0.0; 2]);
    let rank_one = coupled_weights(&[1.0, 1.0, 1.0, 1.0], &[2.0, 2.0]).unwrap();
    assert!((rank_one.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    let scaled = coupled_weights(&[1e-12, 0.0, 0.0, 1e12], &[2e-12, 3e12]).unwrap();
    assert!((scaled[0] - 2.0).abs() < 1e-12 && (scaled[1] - 3.0).abs() < 1e-12);
    assert!(coupled_weights(&[f64::NAN], &[0.0]).is_none());
    assert!(coupled_weights(&[-1.0], &[0.0]).is_none());
}

fn project_impulses(f: &Fixture, rates: &[f32], outputs: &mut [SoilWallFrictionOutput]) {
    let n = rates.len();
    for _ in 0..1 {
        let mut worst = 1.0;
        let mut selected = None;
        for i in 0..n {
            let mut r = 0.0;
            let mut scale = 1.0;
            for j in 0..n {
                let p = f.mass[i * n + j] as f64 * (rates[j] as f64 - f.free[j] as f64);
                r += p;
                scale += p.abs();
            }
            for (c, out) in outputs.iter().enumerate() {
                let row = vec(f.rows[c * n + i]);
                let k = -vec(out.machine_impulse_normal);
                r += row.dot(k);
                scale += row.abs().dot(k.abs());
            }
            let ratio = r.abs() / (1e-6 * scale);
            if ratio > worst {
                worst = ratio;
                selected = Some((i, r));
            }
        }
        let Some((i, r)) = selected else {
            return;
        };
        let active = |c: usize, out: &SoilWallFrictionOutput| {
            f.contacts[c].soil_mass[3] > 0.0
                && (-vec(out.machine_impulse_normal)).dot(vec(f.contacts[c].normal_gap).normalize())
                    > 0.0
        };
        let denominator = outputs
            .iter()
            .enumerate()
            .filter(|(c, o)| active(*c, o))
            .map(|(c, _)| {
                f.contacts[c].soil_mass[3] as f64 * vec(f.rows[c * n + i]).length_squared()
            })
            .sum::<f64>();
        if !denominator.is_finite() || denominator <= 0.0 {
            return;
        }
        let mut candidate = outputs.to_vec();
        for (c, out) in candidate.iter_mut().enumerate() {
            if !active(c, out) {
                continue;
            }
            let mass = f.contacts[c].soil_mass[3] as f64;
            let delta = -(mass * r / denominator) * vec(f.rows[c * n + i]);
            let old = -vec(out.machine_impulse_normal);
            let wall = vec(out.wall_impulse);
            let machine = (0..n)
                .map(|j| vec(f.rows[c * n + j]) * rates[j] as f64)
                .sum::<DVec3>();
            if delta.length() > 0.25e-6 * (1.0 + old.length() + wall.length())
                || delta.length() / mass
                    > 0.25e-6 * (1.0 + vec(out.soil_velocity_energy).length() + machine.length())
            {
                return;
            }
            let k = old + delta;
            let lambda = k.dot(vec(f.contacts[c].normal_gap).normalize());
            if lambda < 0.0 {
                return;
            }
            out.machine_impulse_normal = pack(-k, lambda);
            for axis in 0..3 {
                out.soil_velocity_energy[axis] = f.contacts[c].soil_mass[axis]
                    + (-out.machine_impulse_normal[axis] + out.wall_impulse[axis])
                        / f.contacts[c].soil_mass[3];
            }
        }
        outputs.copy_from_slice(&candidate);
    }
}

#[test]
#[ignore = "requires GPU and SOIL_WALL_TEST_REPRESENTED_RATES; guarded projection rollback"]
fn shared_wall_split_represented_rejects_large_correction() {
    assert!(std::env::var_os("SOIL_WALL_TEST_REPRESENTED_RATES").is_some());
    let engine = Engine::new_gpu().unwrap();
    let mut f = pair(false, 0.3);
    for value in &mut f.mass {
        *value *= 1e6;
    }
    let options = [true, true, true, false, false];
    let flags = [true, true, true, false, true];
    let reference = create_ordered(&engine, &f, 64, options, flags);
    let candidate = create_preparation(&engine, &f, 64, options, flags, 0, true);
    let a = run(&engine, &reference, 3, 2, true);
    let b = run(&engine, &candidate, 3, 2, true);
    assert_eq!(a.2[0], SHARED_WALL_SWEEPS);
    assert_eq!(a, b);
    assert_eq!(run(&engine, &candidate, 3, 2, false), b);
    assert_eq!(run(&engine, &candidate, 3, 2, true), b);
}

#[test]
#[ignore = "CPU only; requires captured shared input/output and fresh report path"]
fn shared_wall_captured_coverage_cpu() {
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_WALL_INPUT_REPLAY").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_WALL_PROBE_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let f = captured_fixture(&root);
    let n = f.free.len();
    let saved = std::env::var_os("SOIL_WALL_SAVED_SOLVE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.clone());
    let bytes = std::fs::read(saved.join("rates.bin")).unwrap();
    let rates: Vec<_> = bytes
        .chunks_exact(4)
        .take(n)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()) as f64)
        .collect();
    let bytes = std::fs::read(saved.join("outputs.bin")).unwrap();
    let outputs: Vec<SoilWallFrictionOutput> = bytes
        .chunks_exact(80)
        .take(f.contacts.len())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let mut walls = Vec::new();
    let mut normals = Vec::new();
    for (c, o) in outputs.iter().enumerate() {
        let contact = f.contacts[c];
        if contact.soil_mass[3] == 0.0 {
            continue;
        }
        let velocity = vec(o.soil_velocity_energy);
        for axis in 0..3 {
            let flag = (contact.boundary[0] >> (2 * axis)) & 3;
            if (flag == 1 && velocity[axis] < 0.0) || (flag == 2 && velocity[axis] > 0.0) {
                walls.push(serde_json::json!({"slot": c, "axis": axis, "flag": flag, "velocity": velocity[axis]}));
            }
        }
        let normal = vec(contact.normal_gap);
        let soil = normal.dot(velocity);
        let machine = (0..n)
            .map(|i| normal.dot(vec(f.rows[c * n + i])) * rates[i])
            .sum::<f64>();
        let gap = contact.normal_gap[3] as f64;
        let slack = soil - machine + gap;
        let gate = 1e-6 * 1.0_f64.max(soil.abs()).max(machine.abs()).max(gap);
        if slack < -gate {
            normals.push(serde_json::json!({"slot": c, "slack_f64": slack, "gate": gate, "soil": soil, "machine": machine, "gap": gap}));
        }
    }
    std::fs::write(output.join("coverage.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "acceptance": false, "scope": "CPU audit of selected shared contacts only; not the full-grid candidate certificate or exact GPU arithmetic",
        "wall_failures": walls, "normal_failures": normals
    })).unwrap()).unwrap();
}
