use super::*;

fn snapshot(engine: &Engine, solver: &ExperimentalSoilNewtonGpu) -> Vec<(&'static str, Vec<u8>)> {
    let mut result: Vec<_> = [
        ("states", solver.state_buffer()),
        ("rates", solver.rates_buffer()),
        ("outputs", solver.outputs_buffer()),
        ("status", solver.status_buffer()),
        ("publication", solver.publication_status_buffer()),
        ("direction", solver.direction_buffer()),
        ("saved", solver.forward_probe_output()),
    ]
    .map(|(name, buffer)| (name, capture::read(engine, buffer)))
    .into();
    result.insert(
        5,
        (
            "blocks",
            bytemuck::cast_slice(&solver.logical_scratch_values(engine, 0)).to_vec(),
        ),
    );
    result
}

pub(super) fn compare_prefixes(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    fixture: &Fixture,
    reference: &Reference,
    output: &std::path::Path,
) {
    assert!(!output.exists(), "refusing to overwrite prefix evidence");
    assert!(cap.newton_iterations <= 64);
    std::fs::create_dir_all(output).unwrap();
    let copy = || SoilNewtonInputs {
        mass: input.mass,
        free_rates: input.free_rates,
        point_rows: input.point_rows,
        point_response: input.point_response,
        contacts: input.contacts,
        upstream_status: input.upstream_status,
    };
    if let Ok(contact) = std::env::var("MOLLA_NEWTON_PRECISION_LOCAL_CONTACT") {
        let contact: u32 = contact.parse().unwrap();
        let reference = std::path::PathBuf::from(
            std::env::var_os("MOLLA_NEWTON_PRECISION_PREFIX_REFERENCE").unwrap(),
        );
        for mode in 0..2 {
            let solver = ExperimentalSoilNewtonGpu::new_precise_local_probe(
                engine,
                copy(),
                cap,
                mode == 1,
                contact,
            )
            .unwrap();
            assert_eq!(run(engine, &solver, true)[0], 0);
            let mut sample = snapshot(engine, &solver);
            sample.push((
                "local",
                capture::read(engine, solver.precision_local_buffer()),
            ));
            let root = output.join(format!("local-mode-{mode}"));
            std::fs::create_dir(&root).unwrap();
            for (name, bytes) in &sample {
                std::fs::write(root.join(format!("{name}.bin")), bytes).unwrap();
            }
            let reference_mode =
                if std::env::var_os("MOLLA_NEWTON_PRECISION_LOCAL_EXPECT_CONTROL").is_some() {
                    0
                } else {
                    mode
                };
            let baseline =
                std::fs::read(reference.join(format!("prefix-0-mode-{reference_mode}/blocks.bin")))
                    .unwrap();
            let begin = contact as usize * 456;
            let matches = sample[5].1[begin..begin + 456] == baseline[begin..begin + 456];
            eprintln!(
                "isolated contact={contact} mode={mode} reference_mode={reference_mode} exact_prefix_block={matches}"
            );
            assert!(matches, "isolated block differs from integrated prefix");
        }
        return;
    }
    let baseline = if std::env::var_os("MOLLA_NEWTON_PRECISION_PROTECTED_BASELINE").is_some() {
        assert!(crate::soil_newton_gpu::protected_enabled(engine));
        ExperimentalSoilNewtonGpu::new_fischer_conditioned(engine, copy(), cap).unwrap()
    } else {
        ExperimentalSoilNewtonGpu::new_precise_control_probe(engine, copy(), cap).unwrap()
    };
    let solvers = [
        baseline,
        ExperimentalSoilNewtonGpu::new_precise_candidate_probe(engine, copy(), cap).unwrap(),
    ];
    for limit in 0..=cap.newton_iterations {
        let samples: Vec<_> = solvers
            .iter()
            .enumerate()
            .map(|(mode, solver)| {
                solver.set_iteration_prefix_probe(engine, limit);
                let status = run(engine, solver, true);
                assert!(
                    matches!(status[0], 0 | 8),
                    "mode={mode} limit={limit} status={status:?}"
                );
                let saved = snapshot(engine, solver);
                assert_eq!(run(engine, solver, true), status);
                assert_eq!(
                    capture::read(engine, solver.state_buffer()),
                    saved[0].1,
                    "prefix state repeat mode={mode} limit={limit}"
                );
                saved
            })
            .collect();
        let mut differences = String::new();
        for ((name, a), (_, b)) in samples[0].iter().zip(&samples[1]) {
            assert_eq!(a.len(), b.len());
            let changed: Vec<_> = a
                .chunks_exact(4)
                .zip(b.chunks_exact(4))
                .enumerate()
                .filter_map(|(i, (a, b))| (a != b).then_some(i))
                .collect();
            differences.push_str(&format!(
                "prefix={limit} buffer={name} different_words={} first={:?}\n",
                changed.len(),
                changed.first()
            ));
        }
        eprint!("{differences}");
        std::fs::write(output.join(format!("prefix-{limit}.txt")), differences).unwrap();
        let differs = samples[0][0].1 != samples[1][0].1;
        if limit == 0 || differs || limit == cap.newton_iterations {
            for (mode, sample) in samples.iter().enumerate() {
                let root = output.join(format!("prefix-{limit}-mode-{mode}"));
                std::fs::create_dir(&root).unwrap();
                for (name, bytes) in sample {
                    std::fs::write(root.join(format!("{name}.bin")), bytes).unwrap();
                }
            }
        }
        if differs {
            eprintln!(
                "first differing prefix={limit}; mode 0=control, 1=candidate; diagnostic only"
            );
            return;
        }
        let converged = samples.iter().all(|sample| sample[3].1[..4] == [0; 4]);
        if converged {
            assert_eq!(&samples[0][..5], &samples[1][..5]);
            for solver in &solvers {
                super::super::publication::verify_outputs(
                    engine,
                    solver.outputs_buffer(),
                    solver.rates_buffer(),
                    fixture,
                    reference,
                );
                solver.set_iteration_prefix_probe(engine, cap.newton_iterations);
            }
            eprintln!(
                "all prefixes through convergence={limit} match exposed outputs; independent publication passes"
            );
            if std::env::var_os("MOLLA_NEWTON_PRECISION_TIMING").is_some() {
                let expected: [u32; 8] = bytemuck::pod_read_unaligned(&samples[0][3].1[..32]);
                for (mode, solver) in solvers.iter().enumerate() {
                    for warmup in 0..2 {
                        eprintln!("precision-cost warmup={warmup} mode={mode}");
                        assert_eq!(run(engine, solver, true), expected);
                    }
                }
                for pair in 0..8 {
                    for order in 0..2 {
                        let mode = (pair + order) % 2;
                        eprintln!("precision-cost sample={} mode={mode}", 2 * pair + order);
                        assert_eq!(run(engine, &solvers[mode], true), expected);
                    }
                }
                for solver in &solvers {
                    assert_eq!(&snapshot(engine, solver)[..5], &samples[0][..5]);
                }
                eprintln!(
                    "precision-cost exact reports/states/rates/outputs/publication match; mode 0=control, 1=candidate"
                );
            }
            return;
        }
    }
    eprintln!("all bounded prefixes match states; diagnostic only");
}
