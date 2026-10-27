use super::*;

pub(super) fn selected(kind: &str, mode: Option<&str>) -> bool {
    match mode {
        None => false,
        Some("1") if kind == "float-controls-specialized" => true,
        _ => panic!("candidate-only admission requires float-controls-specialized and explicit 1"),
    }
}

#[test]
fn newton_specialized_admission_is_explicit() {
    for kind in [
        "inner",
        "float-controls-specialized",
        "float-controls-selective",
    ] {
        assert!(!selected(kind, None));
        for mode in ["", "0", "true", "2"] {
            assert!(std::panic::catch_unwind(|| selected(kind, Some(mode))).is_err());
        }
    }
    assert!(selected("float-controls-specialized", Some("1")));
    assert!(std::panic::catch_unwind(|| selected("inner", Some("1"))).is_err());
}

pub(super) fn run_candidate(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    f: &Fixture,
    p: &Reference,
    root: &std::path::Path,
    hash: &str,
) {
    assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_TIMESTAMPS").is_none());
    let reference = std::path::PathBuf::from(
        std::env::var_os("MOLLA_FLOAT_CONTROLS_REFERENCE")
            .expect("explicit frozen control output directory"),
    );
    let fields = ["states", "rates", "outputs", "status", "publication"];
    let archived: [Vec<u8>; 5] = fields
        .map(|name| std::fs::read(reference.join(hash).join(format!("v1-{name}.bin"))).unwrap());
    let solver = crate::soil_newton_gpu::distributed::mixed::new(
        engine,
        input,
        capacity(p.free.len(), f.contacts.len()),
        true,
    )
    .unwrap();
    let report = run(engine, &solver, true);
    let actual = snapshot(engine, &solver);
    let publication: [u32; 8] = bytemuck::pod_read_unaligned(&actual[4]);
    let x = states(&actual[0], p);
    let physical = p.physical_error(&x);
    let delta = public_delta(&actual, &archived, p);
    let differences: Vec<_> = fields
        .into_iter()
        .zip(actual.iter().zip(&archived))
        .map(|(name, (a, b))| {
            assert_eq!(a.len(), b.len());
            std::fs::write(root.join(format!("candidate-{name}.bin")), a).unwrap();
            serde_json::json!({"buffer":name,"bytes":a.len(),"different_words":
            a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(a,b)|a!=b).count()})
        })
        .collect();
    let valid = report[0] == 0
        && publication[0] == 0
        && x.iter().all(|x| x.is_finite())
        && physical <= 1e-6
        && delta <= 2e-5;
    let exact = actual == archived;
    std::fs::write(root.join("admission.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "kind":"float-controls-specialized","scope":"candidate-only numerical admission against frozen control; no paired timing",
        "report":report,"publication":publication,"physical":physical,"physical_bound":1e-6,
        "max_public_delta":delta,"public_bound":2e-5,"valid":valid,"exact":exact,"differences":differences
    })).unwrap()).unwrap();
    assert!(
        valid,
        "candidate report={report:?} physical={physical:e} public={delta:e}"
    );
    super::super::super::publication::verify_outputs(
        engine,
        solver.outputs_buffer(),
        solver.rates_buffer(),
        f,
        p,
    );
    verify_solution(f, p, &x);
    assert!(
        exact,
        "specialized candidate differs from frozen current control"
    );
    for _ in 0..16 {
        assert_eq!(run(engine, &solver, true), report);
        assert!(
            snapshot(engine, &solver) == actual,
            "candidate self-repeat differs"
        );
    }
    std::fs::write(
        root.join("admission-repeats.txt"),
        "16 exact candidate-only repeat checks\n",
    )
    .unwrap();
    println!(
        "specialized admission packet={hash} physical={physical:e} public={delta:e}: archived exact, 16 repeats exact; no paired timing"
    );
}
