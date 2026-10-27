use super::*;

#[path = "soil_newton_specialized_admission_tests.rs"]
mod specialized_admission;

pub(super) fn snapshot(engine: &Engine, solver: &ExperimentalSoilNewtonGpu) -> [Vec<u8>; 5] {
    [
        solver.state_buffer(),
        solver.rates_buffer(),
        solver.outputs_buffer(),
        solver.status_buffer(),
        solver.publication_status_buffer(),
    ]
    .map(|buffer| capture::read(engine, buffer))
}

pub(super) fn states(bytes: &[u8], p: &Reference) -> Vec<f64> {
    let floats: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    p.contacts
        .iter()
        .flat_map(|c| {
            (0..6).map(|i| {
                let k = 16 * c.index + if i < 3 { i } else { i + 1 };
                floats[k] + floats[k + 8]
            })
        })
        .collect()
}

pub(super) fn public_delta(a: &[Vec<u8>; 5], b: &[Vec<u8>; 5], p: &Reference) -> f64 {
    let mut error = 0.0_f64;
    let mut observe = |a: f64, b: f64| {
        assert!(a.is_finite() && b.is_finite());
        error = error.max((a - b).abs());
    };
    for (a, b) in states(&a[0], p).into_iter().zip(states(&b[0], p)) {
        observe(a, b);
    }
    let f = |b: &[u8]| f64::from(f32::from_le_bytes(b.try_into().unwrap()));
    assert_eq!(a[1].len(), b[1].len());
    for (a, b) in a[1].chunks_exact(8).zip(b[1].chunks_exact(8)) {
        observe(f(&a[..4]) + f(&a[4..]), f(&b[..4]) + f(&b[4..]));
    }
    assert_eq!(a[2].len(), b[2].len());
    for (a, b) in a[2].chunks_exact(80).zip(b[2].chunks_exact(80)) {
        for (a, b) in a[..64].chunks_exact(4).zip(b[..64].chunks_exact(4)) {
            observe(f(a), f(b));
        }
        assert_eq!(&a[64..], &b[64..], "published contact flags");
    }
    error
}

pub(super) fn compare(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    f: &Fixture,
    p: &Reference,
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let kind = crate::soil_newton_gpu::distributed::mixed::kind();
    std::fs::create_dir_all(output).unwrap();
    let root = output.join(&hash);
    std::fs::create_dir(&root).expect("fresh mixed solve output");
    std::fs::write(root.join("original-newton.bin"), bytes).unwrap();
    if specialized_admission::selected(
        &kind,
        std::env::var("MOLLA_NEWTON_SPECIALIZED_ADMISSION")
            .ok()
            .as_deref(),
    ) {
        specialized_admission::run_candidate(engine, input, f, p, &root, &hash);
        return;
    }
    let build = |candidate| {
        crate::soil_newton_gpu::distributed::mixed::new(
            engine,
            SoilNewtonInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: input.point_rows,
                point_response: input.point_response,
                contacts: input.contacts,
                upstream_status: input.upstream_status,
            },
            capacity(p.free.len(), f.contacts.len()),
            candidate,
        )
        .unwrap()
    };
    let solvers = if std::env::var_os("MOLLA_NEWTON_PARALLEL_PREPARATION").is_some() {
        assert!(matches!(
            kind.as_str(),
            "float-controls-specialized"
                | "soa-scratch"
                | "basis-tiles"
                | "block-initialization"
                | "vector-soa"
        ));
        assert_eq!(
            std::env::var("MOLLA_NEWTON_PARALLEL_PREPARATION").unwrap(),
            "1"
        );
        std::thread::scope(|scope| {
            let control = scope.spawn(|| build(false));
            let candidate = scope.spawn(|| build(true));
            [control.join().unwrap(), candidate.join().unwrap()]
        })
    } else {
        [false, true].map(build)
    };
    let mut saved = Vec::new();
    let mut records = Vec::new();
    #[cfg(feature = "retained-local-newton")]
    let mut direct_quality = None;
    for (variant, solver) in solvers.iter().enumerate() {
        eprintln!("mixed-solve case={hash} warmup variant={variant}");
        let report = run(engine, solver, true);
        let buffers = snapshot(engine, solver);
        for (name, bytes) in ["states", "rates", "outputs", "status", "publication"]
            .into_iter()
            .zip(&buffers)
        {
            std::fs::write(root.join(format!("v{variant}-{name}.bin")), bytes).unwrap();
        }
        let publication: [u32; 8] = bytemuck::pod_read_unaligned(&buffers[4]);
        let x = states(&buffers[0], p);
        let physical = p.physical_error(&x);
        let valid = report[0] == 0
            && publication[0] == 0
            && x.iter().all(|x| x.is_finite())
            && physical <= 1e-6;
        records.push(serde_json::json!({"variant":variant,"kind":kind,"report":report,"publication":publication,"physical":physical,"valid":valid}));
        std::fs::write(
            root.join("results.json"),
            serde_json::to_vec_pretty(&records).unwrap(),
        )
        .unwrap();
        assert!(
            valid,
            "mixed variant={variant} report={report:?} physical={physical:e}"
        );
        super::super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            f,
            p,
        );
        verify_solution(f, p, &x);
        if matches!(
            kind.as_str(),
            "float-controls-shared"
                | "float-controls-selective"
                | "float-controls-specialized"
                | "soa-scratch"
                | "basis-tiles"
                | "block-initialization"
                | "vector-soa"
        ) && variant == 0
        {
            let reference = std::path::PathBuf::from(
                std::env::var_os("MOLLA_FLOAT_CONTROLS_REFERENCE")
                    .expect("explicit frozen original-control outputs"),
            );
            let differences: Vec<_> = ["states", "rates", "outputs", "status", "publication"]
                .into_iter()
                .zip(&buffers)
                .map(|(name, actual)| {
                    let expected =
                        std::fs::read(reference.join(&hash).join(format!("v1-{name}.bin")))
                            .unwrap();
                    assert_eq!(actual.len(), expected.len());
                    let count = actual
                        .chunks_exact(4)
                        .zip(expected.chunks_exact(4))
                        .filter(|(a, b)| a != b)
                        .count();
                    serde_json::json!({"buffer":name,"different_words":count,"bytes":actual.len()})
                })
                .collect();
            std::fs::write(
                root.join("archived-control.json"),
                serde_json::to_vec_pretty(&differences).unwrap(),
            )
            .unwrap();
            assert!(
                differences.iter().all(|d| d["different_words"] == 0),
                "strict shared control differs from archived baseline: {differences:?}"
            );
        }
        if kind == "exact-inverse-cache-shared" {
            let bytes = capture::read(engine, solver.precision_local_buffer());
            assert!(bytes.len() >= 256 * 8);
            let mut hits = 0u64;
            let mut builds = 0u64;
            for pair in bytes[..256 * 8].chunks_exact(8) {
                hits += u64::from(u32::from_le_bytes(pair[..4].try_into().unwrap()));
                builds += u64::from(u32::from_le_bytes(pair[4..].try_into().unwrap()));
            }
            assert!(builds > 0 && hits + builds < 10_000_000);
            if variant == 0 {
                assert_eq!(hits, 0);
            }
            std::fs::write(root.join(format!("v{variant}-inverse-cache.json")), serde_json::to_vec_pretty(&serde_json::json!({"hits":hits,"builds":builds,"scope":"lane-local exact key, counters exported after complete solve; not production telemetry"})).unwrap()).unwrap();
        }
        if kind == "common-handoff" && variant == 1 {
            let bytes = capture::read(engine, solver.precision_local_buffer());
            let word = |pair: usize, component: usize| {
                u32::from_le_bytes(
                    bytes[(4100 + pair) * 8 + component * 4..(4100 + pair) * 8 + component * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            let point = word(0, 0);
            assert!(point <= 2);
            if point == 1 {
                assert_eq!(word(1, 0), 0);
                assert_eq!(word(9, 1), 8);
            }
            if point == 2 {
                assert_eq!(word(1, 0), 2);
            }
            let checkpoint = if point == 0 {
                serde_json::json!({"point":0,"common_completed":true})
            } else {
                serde_json::json!({"point":point,"common_completed":false,"phase":word(1,0),"accepted_steps_before_handoff":word(1,1),"prefix_iterations_at_handoff":word(9,1)})
            };
            std::fs::write(
                root.join("handoff.json"),
                serde_json::to_vec_pretty(&checkpoint).unwrap(),
            )
            .unwrap();
        }
        #[cfg(feature = "retained-local-newton")]
        if kind == "direct-quality-trace" && variant == 1 {
            let trace = crate::soil_newton_gpu::direct_quality::decode(&capture::read(
                engine,
                solver.precision_local_buffer(),
            ));
            std::fs::write(
                root.join("direct-quality.json"),
                serde_json::to_vec_pretty(&trace).unwrap(),
            )
            .unwrap();
            direct_quality = Some(trace);
        }
        saved.push(buffers);
    }
    let delta = public_delta(&saved[0], &saved[1], p);
    std::fs::write(root.join("comparison.json"),serde_json::to_vec_pretty(&serde_json::json!({"max_absolute_public_delta":delta,"bound":2e-5,"passed":delta<=2e-5})).unwrap()).unwrap();
    println!("mixed-solve case={hash} public_delta={delta:e} bound=2e-5");
    assert!(delta <= 2e-5, "mixed public comparison");
    if matches!(
        kind.as_str(),
        "direct-quality-trace"
            | "unit-division"
            | "fixed-width"
            | "conditioned-cache"
            | "residual-value"
            | "stream-block"
            | "mobility-cache"
            | "slot-constants"
            | "common-handoff"
            | "accepted-rates-shared"
            | "grouped-norm-shared"
            | "complementarity-shared"
            | "private-samples-shared"
            | "certification-cost-shared"
            | "active-indices-shared"
            | "subgroup-reduction-shared"
            | "exact-inverse-cache-shared"
            | "soa-scratch"
            | "basis-tiles"
            | "block-initialization"
            | "vector-soa"
    ) {
        assert_eq!(
            saved[0], saved[1],
            "diagnostic must preserve every public byte"
        );
    }
    for pair in 0..8 {
        for order in 0..2 {
            let variant = (pair + order) % 2;
            let solver = &solvers[variant];
            eprintln!("mixed-solve case={hash} pair={pair} variant={variant}");
            let report = run(engine, solver, true);
            assert_eq!(report[0], 0);
            assert_eq!(snapshot(engine, solver), saved[variant], "mixed repeat");
            #[cfg(feature = "retained-local-newton")]
            if let Some(expected) = &direct_quality {
                if variant == 1 {
                    assert_eq!(
                        &crate::soil_newton_gpu::direct_quality::decode(&capture::read(
                            engine,
                            solver.precision_local_buffer()
                        )),
                        expected,
                        "direct quality repeat"
                    );
                }
            }
        }
    }
    std::fs::write(
        root.join("repeats-complete.txt"),
        "16 exact repeat checks\n",
    )
    .unwrap();
}
