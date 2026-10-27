use super::*;

fn nonsingular_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))?;
        if a[pivot * n + k].abs() < 1e-10 {
            return None;
        }
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let f = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i];
    }
    Some(x)
}

fn exhaustive_qp(h: &[f64], rhs: &[f64], constraints: &[Vec<f64>]) -> Vec<f64> {
    let n = rhs.len();
    let mut best: Option<(f64, Vec<f64>)> = None;
    for mask in 0..1usize << constraints.len() {
        let active: Vec<_> = constraints
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, a)| a)
            .collect();
        if active.len() > n {
            continue;
        }
        let size = n + active.len();
        let mut a = vec![0.; size * size];
        let mut b = vec![0.; size];
        for i in 0..n {
            b[i] = rhs[i];
            for j in 0..n {
                a[i * size + j] = h[i * n + j];
            }
            for (k, row) in active.iter().enumerate() {
                a[i * size + n + k] = -row[i];
                a[(n + k) * size + i] = row[i];
            }
        }
        let Some(solution) = nonsingular_solve(a, b) else {
            continue;
        };
        if solution[n..].iter().any(|&v| v < -1e-8)
            || constraints
                .iter()
                .any(|a| a.iter().zip(&solution).map(|(a, x)| a * x).sum::<f64>() < -1e-8)
        {
            continue;
        }
        let x = &solution[..n];
        let energy = (0..n)
            .map(|i| 0.5 * x[i] * (0..n).map(|j| h[i * n + j] * x[j]).sum::<f64>() - rhs[i] * x[i])
            .sum::<f64>();
        if best.as_ref().is_none_or(|(old, _)| energy < *old) {
            best = Some((energy, x.to_vec()));
        }
    }
    best.expect("homogeneous constraints have a feasible zero solution")
        .1
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn boundary_projection_matches_exhaustive_kkt_and_accounts_external_impulse() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mass_values = [2.0f32, 0.2, 0.1, 0.2, 1.5, -0.1, 0.1, -0.1, 1.0];
    let jac_values = [
        1.0f32, 0.2, 0., 0., 1., 0.3, 0.1, 0., 1., 0.3, 0., 0.2, 0., 0.4, 0., 0.2, 0., 0.3,
    ];
    let node_buffer = buffer(&engine, &[0; 6 * 32]);
    let spans = buffer(&engine, bytemuck::cast_slice(&[[0u32, 6]]));
    let mass = buffer(&engine, bytemuck::cast_slice(&mass_values));
    let jac = buffer(&engine, bytemuck::cast_slice(&jac_values));
    let free = buffer(&engine, &[0; 12]);
    let shared = match std::env::var("SOIL_PROJECTION_SHARED_BOUNDARY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_SHARED_BOUNDARY must be absent or 1"),
    };
    let cached = match std::env::var("SOIL_PROJECTION_CACHED_BOUNDARY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_CACHED_BOUNDARY must be absent or 1"),
    };
    assert!(!(shared && cached));
    let projection = SoilContactProjectionGpu::prepare(
        &engine,
        SoilProjectionInputs {
            nodes: &node_buffer,
            spans: &spans,
            mass: &mass,
            jacobians: &jac,
            free_rates: &free,
        },
        SoilProjectionCapacity {
            dofs: 3,
            bodies: 1,
            nodes: 6,
            max_owned_bytes: 240,
        },
        true,
        if shared { 3 } else { 32 },
        if cached {
            ProjectionShaderMode::CachedBoundary
        } else if shared {
            ProjectionShaderMode::SharedBoundary
        } else {
            ProjectionShaderMode::Sized
        },
    )
    .unwrap();
    let m: Vec<_> = mass_values.iter().map(|&x| f64::from(x)).collect();
    let j: Vec<_> = jac_values.iter().map(|&x| f64::from(x)).collect();
    for case in 0..76 {
        let rates: Vec<f32> = (0..3)
            .map(|i| (((case * 7 + i * 13) % 19) as f32 - 9.0) * 0.37)
            .collect();
        let mut nodes = Vec::new();
        let mut h = m.clone();
        let mut rhs: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|k| m[i * 3 + k] * f64::from(rates[k])).sum())
            .collect();
        let mut constraints = Vec::new();
        let mut initial_energy = (0..3)
            .map(|i| {
                0.5 * f64::from(rates[i])
                    * (0..3)
                        .map(|k| m[i * 3 + k] * f64::from(rates[k]))
                        .sum::<f64>()
            })
            .sum::<f64>();
        for k in 0..6 {
            let face = if case % 4 == 0 { k % 2 } else { k };
            let enabled = case % 4 != 1 || k < 3;
            let arm = if case % 4 == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(
                    k as f64 * 0.13 - 0.3,
                    0.2 - k as f64 * 0.07,
                    (k % 3) as f64 * 0.1,
                )
            };
            let p = Vec3::new(0.07 * (k as f64 - 2.), -0.03 * k as f64, 0.02);
            let node = SoilProjectionNode {
                arm_mass: [
                    arm.x as f32,
                    arm.y as f32,
                    arm.z as f32,
                    0.2 + k as f32 * 0.1,
                ],
                momentum: [
                    p.x as f32,
                    p.y as f32,
                    p.z as f32,
                    if enabled { (1u32 << face) as f32 } else { 0. },
                ],
            };
            let arm = Vec3::new(
                node.arm_mass[0] as f64,
                node.arm_mass[1] as f64,
                node.arm_mass[2] as f64,
            );
            let p = Vec3::new(
                node.momentum[0] as f64,
                node.momentum[1] as f64,
                node.momentum[2] as f64,
            );
            let cols = point_columns(&j, 3, arm);
            let mut normal = [0.; 3];
            normal[face / 2] = if face % 2 == 0 { 1. } else { -1. };
            let normal = Vec3::new(normal[0], normal[1], normal[2]);
            if enabled {
                constraints.push(cols.iter().map(|v| v.dot(normal)).collect::<Vec<_>>());
            }
            for i in 0..3 {
                rhs[i] += cols[i].dot(p);
                for k in 0..3 {
                    h[i * 3 + k] += f64::from(node.arm_mass[3]) * cols[i].dot(cols[k]);
                }
            }
            initial_energy += 0.5 * p.dot(p) / f64::from(node.arm_mass[3]);
            nodes.push(node);
        }
        let expected = exhaustive_qp(&h, &rhs, &constraints);
        engine
            .queue()
            .unwrap()
            .write_buffer(&node_buffer, 0, bytemuck::cast_slice(&nodes));
        engine
            .queue()
            .unwrap()
            .write_buffer(&free, 0, bytemuck::cast_slice(&rates));
        submit(&engine, &projection, true);
        assert_eq!(
            read(&engine, projection.status_buffer())[0].to_bits(),
            0,
            "case {case}"
        );
        let actual = read(&engine, projection.rates_buffer());
        for i in 0..3 {
            assert!(
                (f64::from(actual[i]) - expected[i]).abs() < 2e-5,
                "case {case}: {actual:?} vs {expected:?}"
            );
        }
        let ground = read(&engine, projection.boundary_impulses_buffer());
        let impulse = read(&engine, projection.impulses_buffer());
        for i in 0..3 {
            let delta = (0..3)
                .map(|k| m[i * 3 + k] * f64::from(actual[k] - rates[k]))
                .sum::<f64>();
            let observed = (0..6)
                .map(|k| j[k * 3 + i] * f64::from(impulse[if k < 3 { k } else { k + 1 }]))
                .sum::<f64>();
            assert!(
                (delta - observed).abs() < 2e-5,
                "machine impulse case {case} coordinate {i}: {delta} vs {observed}; ground={ground:?}, impulse={impulse:?}, rates={actual:?}"
            );
            let ground_generalized = (0..6)
                .map(|k| j[k * 3 + i] * f64::from(ground[if k < 3 { k } else { k + 1 }]))
                .sum::<f64>();
            let residual = (0..3)
                .map(|k| h[i * 3 + k] * f64::from(actual[k]))
                .sum::<f64>()
                - rhs[i]
                - ground_generalized;
            assert!(
                residual.abs() < 2e-5,
                "external impulse case {case}: {residual}"
            );
        }
        let final_energy = (0..3)
            .map(|i| {
                0.5 * f64::from(actual[i])
                    * (0..3)
                        .map(|k| h[i * 3 + k] * f64::from(actual[k]))
                        .sum::<f64>()
            })
            .sum::<f64>();
        assert!(final_energy <= initial_energy + 2e-5);
        for constraint in &constraints {
            assert!(
                constraint
                    .iter()
                    .zip(&actual)
                    .map(|(a, x)| a * f64::from(*x))
                    .sum::<f64>()
                    >= -2e-5
            );
        }
    }
    for invalid_mask in [-1.0, 0.5, 64.0, f32::NAN] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&node_buffer, 28, bytemuck::bytes_of(&invalid_mask));
        submit(&engine, &projection, true);
        assert_ne!(
            read(&engine, projection.status_buffer())[0].to_bits() & SOIL_PROJECTION_INPUT,
            0
        );
    }
}
