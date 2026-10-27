use super::*;

#[test]
#[ignore = "requires GPU; conditional correction versus dense unselected elimination"]
fn resident_newton_conditioned_oracle() {
    let engine = Engine::new_gpu().unwrap();
    for mut f in [pair(true, 0.3), articulated(65), wide_fixture()] {
        let n = f.free.len();
        if n > 3 {
            f.contacts[0].soil_mass[3] = 0.;
            f.contacts[2].soil_mass[3] = 0.;
        }
        let mut p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let zero = vec![0.; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let free = storage(
            &engine,
            &p.free.iter().map(|&v| [v as f32, 0.]).collect::<Vec<_>>(),
        );
        let mass = storage(&engine, &p.mass);
        let [rows, response, contacts] = op.point_inputs();
        let prepared: Vec<[f32; 4]> = values(&engine, response)
            .chunks_exact(4)
            .map(|v| v.try_into().unwrap())
            .collect();
        p.use_prepared_response(&prepared);
        let selected = p.contacts.last().unwrap().index;
        let first = p.contacts[0].index;
        for singular in [None, Some(selected), Some(first)] {
            let solver = ExperimentalSoilNewtonGpu::new_conditioned_probe(
                &engine,
                SoilNewtonInputs {
                    mass: &mass,
                    free_rates: &free,
                    point_rows: rows,
                    point_response: response,
                    contacts,
                    upstream_status: op.status_buffer(),
                },
                capacity(n, f.contacts.len()),
                selected as u32,
                singular.map(|i| i as u32),
            )
            .unwrap();
            let mut previous = None;
            for (iteration, reset) in [true, false, true].into_iter().enumerate() {
                if iteration == 2 {
                    for buffer in solver.scratch_buffers() {
                        engine.queue().unwrap().write_buffer(
                            buffer,
                            0,
                            bytemuck::cast_slice(&vec![f32::NAN; buffer.size() as usize / 4]),
                        );
                    }
                }
                let status = run(&engine, &solver, reset);
                let states = values(&engine, solver.state_buffer());
                let blocks = values(&engine, solver.linear_probe_buffers()[0]);
                if let Some(ref old) = previous {
                    assert_eq!(&states, old);
                }
                previous = Some(states.clone());
                let mut delta = Vec::new();
                let mut residual = Vec::new();
                let mut direct = Vec::new();
                let mut machine = Vec::new();
                for c in &p.contacts {
                    delta.extend((0..6).map(|i| {
                        let offset = 16 * c.index + if i < 3 { i } else { i + 1 };
                        states[offset] as f64 + states[offset + 8] as f64 - 0.1_f32 as f64
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
                if singular == Some(first) {
                    assert_eq!(status[0], 8);
                    assert!(delta.iter().all(|&v| v == 0.));
                    continue;
                }
                assert_eq!(status[0], 0);
                let selected_offset = delta.len() - 6;
                for (j, x) in [0.17_f32, -0.01, 0.12, 0.04, 0.15, 0.1]
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(delta[selected_offset + j], x as f64 - 0.1_f32 as f64);
                }
                let linear = Linearization {
                    problem: &p,
                    direct,
                    machine,
                };
                let size = selected_offset;
                let mut selected_delta = vec![0.; delta.len()];
                selected_delta[size..].copy_from_slice(&delta[size..]);
                let coupling = linear.product(&selected_delta);
                let rhs: Vec<_> = (0..size).map(|i| -residual[i] - coupling[i]).collect();
                let mut matrix = vec![0.; size * size];
                for j in 0..size {
                    let mut unit = vec![0.; delta.len()];
                    unit[j] = 1.;
                    let column = linear.product(&unit);
                    for i in 0..size {
                        matrix[i * size + j] = column[i];
                    }
                }
                let expected = solve_dense(matrix, rhs.clone()).unwrap();
                let direction_error = norm(
                    &(0..size)
                        .map(|i| delta[i] - expected[i])
                        .collect::<Vec<_>>(),
                ) / (1. + norm(&expected));
                let product = linear.product(&delta);
                let relative = norm(
                    &(0..size)
                        .map(|i| product[i] + residual[i])
                        .collect::<Vec<_>>(),
                ) / norm(&rhs).max(1e-30);
                eprintln!(
                    "conditional oracle n={n} active={} selected={selected} singular={singular:?} reset={reset} relative={relative:e} direction={direction_error:e}",
                    p.contacts.len()
                );
                assert!(relative <= 1e-5);
                assert!(direction_error <= 1e-5);
            }
        }
    }
}
