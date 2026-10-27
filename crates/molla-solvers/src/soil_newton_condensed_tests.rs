use super::*;

#[test]
#[ignore = "requires GPU; shared hybrid Krylov budget after singular direct rejection"]
fn resident_newton_hybrid_budget() {
    let engine = Engine::new_gpu().unwrap();
    let mut f = wide_fixture();
    f.mass.fill(0.);
    for i in 0..32 {
        f.mass[32 * i + i] = 1.;
    }
    f.free.fill(0.);
    f.contacts = vec![f.contacts[0]; 17];
    f.rows = vec![[0.; 4]; 17 * 32];
    for c in 0..17 {
        f.rows[32 * c + c][0] = 1.;
        f.rows[32 * c + (c + 1) % 17][0] = 0.95;
    }
    let p = Reference::new(&f, vec![0.; 32]);
    assert_eq!(p.contacts.len(), 17);
    let zero = vec![0.; 102];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    let free = storage(&engine, &vec![[0f32; 2]; 32]);
    let mass = storage(&engine, &p.mass);
    let [rows, response, contacts] = op.point_inputs();
    for budget in [1, 8, 64] {
        let cap = SoilNewtonCapacity {
            krylov_iterations: budget,
            ..capacity(32, 17)
        };
        let solver = ExperimentalSoilNewtonGpu::new_hybrid_budget_probe(
            &engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: rows,
                point_response: response,
                contacts,
                upstream_status: op.status_buffer(),
            },
            cap,
        )
        .unwrap();
        let status = run(&engine, &solver, true);
        let [_, direction] = solver.linear_probe_buffers();
        let direction = values(&engine, direction);
        let basis = values(&engine, solver.basis_buffer());
        let d: Vec<f64> = (0..102)
            .map(|i| {
                let c = i / 6;
                let axis = i % 6;
                let axis = if axis < 3 { axis } else { axis + 1 };
                direction[8 * c + axis] as f64
                    + basis[16 * budget as usize * 17 + 16 * c + 8 + axis] as f64
            })
            .collect();
        let residual: Vec<f64> = (0..102)
            .map(|i| (0.1_f32 * (i + 1) as f32) as f64)
            .collect();
        let error: Vec<f64> = (0..102)
            .map(|i| {
                if i % 6 != 0 {
                    return d[i] + residual[i];
                }
                let c = i / 6;
                residual[i]
                    - (0..17)
                        .map(|k| {
                            let coupling: f64 = (0..32)
                                .map(|j| {
                                    f.rows[c * 32 + j][0] as f64 * f.rows[k * 32 + j][0] as f64
                                })
                                .sum();
                            coupling * d[6 * k]
                        })
                        .sum::<f64>()
            })
            .collect();
        let relative = norm(&error) / norm(&residual);
        eprintln!("hybrid budget={budget} status={status:?} independent_relative={relative:e}");
        assert_eq!(status[3], 1, "direct rejection must be exercised");
        assert!(status[2] <= budget);
        if budget <= 8 {
            assert_eq!(status[0], 8);
            assert_eq!(status[2], budget);
            assert_eq!(status[7], 0);
            assert!(relative > 1e-5);
        } else {
            assert_eq!(status[0], 0);
            assert!(status[2] > 8);
            assert_eq!(status[7], 1);
            assert!(relative <= 1e-5);
        }
    }
}

#[test]
#[ignore = "requires GPU; condensed direction versus independent dense normal equations"]
fn resident_newton_condensed_damped_oracle() {
    condensed_oracle(false);
}

#[test]
#[ignore = "requires GPU; direct Newton direction and singular-block GMRES fallback"]
fn resident_newton_direct_oracle() {
    condensed_oracle(true);
}

fn condensed_oracle(undamped: bool) {
    let engine = Engine::new_gpu().unwrap();
    let mut fixtures = vec![
        (pair(true, 0.3), 0.1_f32, false),
        (articulated(65), 0.01, false),
        (wide_fixture(), 0.1, false),
    ];
    if undamped {
        let mut f = pair(true, 0.3);
        f.contacts.truncate(1);
        f.rows.truncate(f.free.len());
        fixtures.push((f, 0., true));
    }
    let selected = std::env::var("MOLLA_NEWTON_DIRECT_ORACLE_CASE")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    assert!(selected.is_none_or(|i| undamped && i < fixtures.len()));
    for (index, (f, damping, singular)) in fixtures.into_iter().enumerate() {
        if selected.is_some_and(|i| i != index) {
            continue;
        }
        let mut p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let zero = vec![0.; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let pairs: Vec<_> = p
            .free
            .iter()
            .map(|&q| [q as f32, (q - q as f32 as f64) as f32])
            .collect();
        let free = storage(&engine, &pairs);
        let mass = storage(&engine, &p.mass);
        let [rows, response, contacts] = op.point_inputs();
        let input = || SoilNewtonInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: rows,
            point_response: response,
            contacts,
            upstream_status: op.status_buffer(),
        };
        let cap = capacity(p.free.len(), f.contacts.len());
        let bytes = ExperimentalSoilNewtonGpu::required_condensed_bytes(cap).unwrap();
        let construct = |cap| {
            if undamped {
                ExperimentalSoilNewtonGpu::new_direct_probe(&engine, input(), cap, singular)
            } else {
                ExperimentalSoilNewtonGpu::new_condensed_probe(&engine, input(), cap, damping)
            }
        };
        let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            construct(SoilNewtonCapacity {
                max_owned_bytes: bytes - 1,
                ..cap
            })
            .is_err()
        );
        assert_eq!(allocation.finish().observations, 0);
        let solver = construct(SoilNewtonCapacity {
            max_owned_bytes: bytes,
            ..cap
        })
        .unwrap();
        assert_eq!(solver.owned_buffer_bytes(), bytes);
        let status = run(&engine, &solver, true);
        let [blocks, direction] = solver.linear_probe_buffers();
        let blocks = values(&engine, blocks);
        let direction = values(&engine, direction);
        let basis = values(&engine, solver.basis_buffer());
        let response: Vec<[f32; 4]> = values(&engine, response)
            .chunks_exact(4)
            .map(|v| v.try_into().unwrap())
            .collect();
        p.use_prepared_response(&response);
        let mut d = Vec::new();
        let mut residual = Vec::new();
        let mut direct = Vec::new();
        let mut machine = Vec::new();
        for c in &p.contacts {
            d.extend((0..6).map(|i| {
                let i = if i < 3 { i } else { i + 1 };
                direction[8 * c.index + i] as f64
                    + basis[16 * cap.krylov_iterations as usize * f.contacts.len().max(1)
                        + 16 * c.index
                        + 8
                        + i] as f64
            }));
            residual.extend((0..6).map(|i| blocks[114 * c.index + i] as f64));
            let coefficient = |offset| {
                blocks[114 * c.index + 6 + 2 * offset] as f64
                    + blocks[114 * c.index + 7 + 2 * offset] as f64
            };
            direct.push(std::array::from_fn(|i| {
                std::array::from_fn(|j| coefficient(6 * i + j))
            }));
            machine.push(std::array::from_fn(|i| {
                std::array::from_fn(|j| coefficient(36 + 3 * i + j))
            }));
        }
        let linear = Linearization {
            problem: &p,
            direct,
            machine,
        };
        let rhs: Vec<_> = if undamped {
            residual.iter().map(|v| -v).collect()
        } else {
            linear.transpose(&residual).iter().map(|v| -v).collect()
        };
        let size = d.len();
        let mut normal = vec![0.; size * size];
        for j in 0..size {
            let mut unit = vec![0.; size];
            unit[j] = 1.;
            let column = if undamped {
                linear.product(&unit)
            } else {
                linear.transpose(&linear.product(&unit))
            };
            for i in 0..size {
                normal[i * size + j] = column[i]
                    + if !undamped && i == j {
                        damping as f64
                    } else {
                        0.
                    };
            }
        }
        let expected = solve_dense(normal, rhs.clone()).unwrap();
        let error = norm(
            &d.iter()
                .zip(&expected)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        ) / (1. + norm(&expected));
        let product = if undamped {
            linear.product(&d)
        } else {
            linear.transpose(&linear.product(&d))
        };
        let relative = norm(
            &product
                .iter()
                .zip(&d)
                .zip(&rhs)
                .map(|((v, x), b)| v + if undamped { 0. } else { damping as f64 * x } - b)
                .collect::<Vec<_>>(),
        ) / norm(&rhs);
        eprintln!(
            "condensed oracle undamped={undamped} singular={singular} dofs={} contacts={} status={status:?} relative={relative:e} direction={error:e}",
            p.free.len(),
            p.contacts.len()
        );
        assert_eq!(status[0], 0);
        assert_eq!(status[3], u32::from(singular));
        if !singular {
            assert_eq!(status[2], 1);
        } else {
            assert!((1..=64).contains(&status[2]));
        }
        assert!(relative <= 1e-5);
        assert!(error <= 1e-5);
    }
}
