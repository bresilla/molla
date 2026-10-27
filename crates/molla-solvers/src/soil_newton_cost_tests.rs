use super::*;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
pub(super) fn compare_delta_cost(
    engine: &Engine,
    baseline: &ExperimentalSoilNewtonGpu,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    fixture: &Fixture,
    reference: &Reference,
) {
    let schedule = std::env::var("MOLLA_NEWTON_SCHEDULE_COMPARE").ok();
    let schur = std::env::var_os("MOLLA_NEWTON_SCHUR_COMPARE").is_some();
    let schur_inverse = std::env::var_os("MOLLA_NEWTON_SCHUR_INVERSE_COMPARE").is_some();
    let reuse = std::env::var_os("MOLLA_NEWTON_SCHUR_REUSE_COMPARE").is_some();
    assert!(
        !reuse || schur_inverse,
        "reuse comparison requires Schur inverse selection"
    );
    let cgs2 = std::env::var_os("MOLLA_NEWTON_CGS2_COMPARE").is_some();
    if schedule.is_some() || schur || schur_inverse || cgs2 {
        assert!(
            usize::from(schedule.is_some())
                + usize::from(schur)
                + usize::from(schur_inverse)
                + usize::from(cgs2)
                == 1,
            "algorithm comparisons must be isolated"
        );
        for other in [
            "DELTA",
            "PARALLEL",
            "VOLATILE",
            "TILED",
            "CANDIDATE",
            "NONLINEAR_CACHE",
            "PHASE",
            "SPARSE",
            "INVERSE",
            "SHARED_INVERSE",
            "NOINLINE",
            "ROW",
            "COUNT",
            "OPERATION",
            "NORM",
            "SLICED",
        ] {
            assert!(
                std::env::var_os(format!("MOLLA_NEWTON_{other}_COMPARE")).is_none(),
                "algorithm comparison must be isolated"
            );
        }
        let label = if cgs2 {
            "cgs2".to_owned()
        } else if schur {
            "schur-contact".to_owned()
        } else if reuse {
            "schur-reuse".to_owned()
        } else if schur_inverse {
            "schur-inverse".to_owned()
        } else {
            let prefix = schedule.unwrap();
            assert!(matches!(prefix.as_str(), "0" | "4"));
            format!("schedule-{prefix}")
        };
        compare_algorithm_cost(engine, baseline, input, cap, fixture, reference, &label);
        return;
    }
    let parallel = std::env::var_os("MOLLA_NEWTON_PARALLEL_COMPARE").is_some();
    let volatile = std::env::var_os("MOLLA_NEWTON_VOLATILE_COMPARE").is_some();
    let tiled = std::env::var_os("MOLLA_NEWTON_TILED_COMPARE").is_some();
    let candidates = std::env::var_os("MOLLA_NEWTON_CANDIDATE_COMPARE").is_some();
    let nonlinear_cache = std::env::var_os("MOLLA_NEWTON_NONLINEAR_CACHE_COMPARE").is_some();
    let phase = std::env::var("MOLLA_NEWTON_PHASE_COMPARE").ok();
    let sparse = std::env::var_os("MOLLA_NEWTON_SPARSE_COMPARE").is_some();
    let inverse = std::env::var_os("MOLLA_NEWTON_INVERSE_COMPARE").is_some();
    let shared = std::env::var_os("MOLLA_NEWTON_SHARED_INVERSE_COMPARE").is_some();
    let noinline = std::env::var_os("MOLLA_NEWTON_NOINLINE_COMPARE").is_some();
    let row = std::env::var_os("MOLLA_NEWTON_ROW_COMPARE").is_some();
    let counts = std::env::var_os("MOLLA_NEWTON_COUNT_COMPARE").is_some();
    let operation = std::env::var("MOLLA_NEWTON_OPERATION_COMPARE").ok();
    let norm = std::env::var_os("MOLLA_NEWTON_NORM_COMPARE").is_some();
    let quaternary = std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("quaternary");
    let private_mgs = std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("private-mgs");
    let active_publication =
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-active");
    let zero_publication =
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-zero");
    let publication_products =
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-products");
    let adaptive_publication =
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-adaptive");
    let validation_publication =
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-validation");
    let sliced = std::env::var_os("MOLLA_NEWTON_SLICED_COMPARE").is_some();
    if sliced {
        assert!(cfg!(feature = "norm-tree-newton"));
        for other in [
            "DELTA",
            "PARALLEL",
            "VOLATILE",
            "TILED",
            "CANDIDATE",
            "NONLINEAR_CACHE",
            "PHASE",
            "SPARSE",
            "INVERSE",
            "SHARED_INVERSE",
            "NOINLINE",
            "ROW",
            "COUNT",
            "OPERATION",
            "NORM",
        ] {
            assert!(
                std::env::var_os(format!("MOLLA_NEWTON_{other}_COMPARE")).is_none(),
                "sliced comparison must be isolated"
            );
        }
    }
    if norm {
        assert!(matches!(
            std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
            Ok("1"
                | "quaternary"
                | "private-mgs"
                | "publication-active"
                | "publication-zero"
                | "publication-products"
                | "publication-adaptive"
                | "publication-validation")
        ));
        assert!(!quaternary || cfg!(feature = "norm-tree-newton"));
        assert!(!active_publication || cfg!(feature = "norm-tree-newton"));
        assert!(!zero_publication || cfg!(feature = "active-publication-newton"));
        assert!(!publication_products || cfg!(feature = "active-publication-newton"));
        assert!(!adaptive_publication || cfg!(feature = "retained-local-newton"));
        assert!(!validation_publication || cfg!(feature = "retained-local-newton"));
        assert!(!private_mgs || cfg!(feature = "norm-tree-newton"));
        assert!(!row && !counts && operation.is_none());
    }
    if let Some(scope) = &operation {
        assert!(matches!(
            scope.as_str(),
            "inner2" | "inner4" | "norm2" | "norm4" | "prepare2" | "project2"
        ));
        assert!(!row && !counts);
    }
    assert!(!(row && counts));
    if counts || operation.is_some() || norm {
        assert!(
            cfg!(feature = "schur-newton"),
            "operation counts require method 19"
        );
    }
    if row || counts || operation.is_some() || norm {
        for other in [
            "DELTA",
            "PARALLEL",
            "VOLATILE",
            "TILED",
            "CANDIDATE",
            "NONLINEAR_CACHE",
            "PHASE",
            "SPARSE",
            "INVERSE",
            "SHARED_INVERSE",
            "NOINLINE",
        ] {
            assert!(
                std::env::var_os(format!("MOLLA_NEWTON_{other}_COMPARE")).is_none(),
                "row-product comparison must be isolated"
            );
        }
    }
    assert!(
        !noinline
            || (!shared
                && !inverse
                && !sparse
                && phase.is_none()
                && !nonlinear_cache
                && !candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !shared
            || (!inverse
                && !sparse
                && phase.is_none()
                && !nonlinear_cache
                && !candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !inverse
            || (!sparse
                && phase.is_none()
                && !nonlinear_cache
                && !candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !sparse
            || (phase.is_none()
                && !nonlinear_cache
                && !candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    if let Some(repeats) = &phase {
        assert!(matches!(
            repeats.as_str(),
            "2" | "4" | "linear2" | "linear4"
        ));
        assert!(
            !nonlinear_cache
                && !candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none()
        );
    }
    assert!(
        !nonlinear_cache
            || (!candidates
                && !tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !candidates
            || (!tiled
                && !volatile
                && !parallel
                && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !tiled
            || (!volatile && !parallel && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none())
    );
    assert!(
        !volatile || (!parallel && std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none()),
        "choose one rounding/recovery comparison"
    );
    assert!(
        !parallel || std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_none(),
        "choose one recovery comparison"
    );
    let phase_label = phase.as_ref().map(|r| format!("phase-{r}-cost"));
    let operation_label = operation
        .as_ref()
        .map(|scope| format!("operation-{scope}-cost"));
    let label = if sliced {
        "sliced-cost"
    } else if validation_publication {
        "publication-validation-cost"
    } else if adaptive_publication {
        "publication-adaptive-cost"
    } else if publication_products {
        "publication-products-cost"
    } else if zero_publication {
        "zero-publication-cost"
    } else if active_publication {
        "active-publication-cost"
    } else if quaternary {
        "norm-quaternary-cost"
    } else if private_mgs {
        "private-mgs-cost"
    } else if norm {
        "norm-tree-cost"
    } else if let Some(label) = operation_label.as_deref() {
        label
    } else if counts {
        "operation-counts-cost"
    } else if row {
        "row-products-cost"
    } else if noinline {
        "noinline-cost"
    } else if shared {
        "shared-inverse-cost"
    } else if inverse {
        "inverse-skip-cost"
    } else if sparse {
        "sparse-products-cost"
    } else if let Some(label) = phase_label.as_deref() {
        label
    } else if nonlinear_cache {
        "nonlinear-cache-cost"
    } else if candidates {
        "candidate-cache-cost"
    } else if tiled {
        "tiled-core-cost"
    } else if volatile {
        "volatile-round-cost"
    } else if parallel {
        "parallel-release-cost"
    } else {
        "delta-cache-cost"
    };
    assert_eq!(
        baseline.failure_method(),
        if adaptive_publication || validation_publication {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedRetainedLocal256
        } else if zero_publication || publication_products {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedActivePublication256
        } else if sliced || quaternary || private_mgs || active_publication {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedNorm256
        } else if (noinline || row || counts || operation.is_some() || norm)
            && cfg!(feature = "schur-newton")
        {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedSchur256
        } else if tiled
            || candidates
            || nonlinear_cache
            || phase.is_some()
            || sparse
            || inverse
            || shared
            || noinline
            || row
        {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedVolatile256
        } else if volatile {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedParallel256
        } else if parallel {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedCached256
        } else {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedWide256
        },
        "recovery timing requires its exact control method"
    );
    let candidate = if sliced
        || volatile
        || tiled
        || candidates
        || nonlinear_cache
        || phase.is_some()
        || sparse
        || inverse
        || shared
        || noinline
        || row
        || counts
        || operation.is_some()
        || norm
    {
        ExperimentalSoilNewtonGpu::new_volatile_rounding_probe(engine, input, cap)
    } else if parallel {
        ExperimentalSoilNewtonGpu::new_parallel_release_probe(engine, input, cap)
    } else {
        ExperimentalSoilNewtonGpu::new_cached_delta_probe(engine, input, cap)
    }
    .unwrap();
    let solvers = [baseline, &candidate];
    let snapshot = |solver: &ExperimentalSoilNewtonGpu| {
        [
            solver.state_buffer(),
            solver.rates_buffer(),
            solver.outputs_buffer(),
            solver.publication_status_buffer(),
            solver.status_buffer(),
        ]
        .map(|b| capture::read(engine, b))
    };
    let assert_snapshot = |actual: [Vec<u8>; 5], expected: &[Vec<u8>; 5]| {
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.len(), b.len(), "buffer {index} length");
            assert!(
                a == b,
                "buffer {index} differs at byte {:?}",
                a.iter().zip(b).position(|(x, y)| x != y)
            );
        }
    };
    eprintln!("{label} warmup variant=0");
    let expected = run(engine, baseline, true);
    assert_eq!(expected[0], 0);
    let bytes = snapshot(baseline);
    if zero_publication {
        let active = fixture
            .contacts
            .iter()
            .filter(|c| c.soil_mass[3] > 0.0)
            .count();
        let zero = bytes[2]
            .chunks_exact(80)
            .zip(&fixture.contacts)
            .filter(|(b, c)| {
                c.soil_mass[3] > 0.0
                    && b[..12]
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()) == 0.0)
            })
            .count();
        eprintln!("{label} impulse-coverage active={active} zero={zero}");
    }
    eprintln!("{label} warmup variant=1");
    assert_eq!(run(engine, &candidate, true), expected);
    assert_snapshot(snapshot(&candidate), &bytes);
    let read_counts = || {
        crate::soil_newton_parallel_release::operation_counts::decode(&capture::read(
            engine,
            candidate.precision_local_buffer(),
        ))
    };
    let expected_counts = counts.then(|| {
        let values = read_counts();
        assert!(values[0] > 0 && values[26] > 0);
        assert!(values[30] <= cap.contacts);
        let names: Vec<_> = crate::soil_newton_parallel_release::operation_counts::FUNCTIONS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        eprintln!("operation-counts names={names:?} values={values:?}");
        values
    });
    for pair in 0..8 {
        for order in 0..2 {
            let variant = (pair + order) % 2;
            let solver = solvers[variant];
            eprintln!("{label} sample={} variant={variant}", 2 * pair + order);
            assert_eq!(run(engine, solver, true), expected);
            assert_snapshot(snapshot(solver), &bytes);
            if variant == 1 {
                if let Some(expected) = expected_counts {
                    assert_eq!(
                        read_counts(),
                        expected,
                        "operation counts must repeat exactly"
                    );
                }
            }
            super::super::publication::verify_outputs(
                engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                fixture,
                reference,
            );
        }
    }
    eprintln!(
        "{label} reports={expected:?}; all five buffers byte-identical between variants and repeats; independent physical publication each sample; variant 0=runtime control, 1=prototype; lanes={}; contact_constants={}; parallel_core={}",
        crate::soil_newton_gpu::execution_lanes(true),
        sliced
            || norm
            || operation.is_some()
            || counts
            || row
            || noinline
            || shared
            || inverse
            || sparse
            || phase.is_some()
            || nonlinear_cache
            || candidates
            || tiled
            || volatile
            || parallel
            || std::env::var_os("MOLLA_NEWTON_STATIC_CONTACT_CACHE").is_some(),
        sliced
            || norm
            || operation.is_some()
            || counts
            || row
            || noinline
            || shared
            || inverse
            || sparse
            || phase.is_some()
            || nonlinear_cache
            || candidates
            || tiled
            || volatile
            || (parallel && std::env::var_os("MOLLA_NEWTON_PARALLEL_CORE").is_some())
    );
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
fn compare_algorithm_cost(
    engine: &Engine,
    baseline: &ExperimentalSoilNewtonGpu,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    fixture: &Fixture,
    reference: &Reference,
    label: &str,
) {
    assert_eq!(cap.newton_iterations, 64);
    assert_eq!(cap.krylov_iterations, 64);
    assert_eq!(
        baseline.failure_method(),
        if (label == "cgs2" || label == "schur-reuse") && cfg!(feature = "schur-newton") {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedSchur256
        } else {
            crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedVolatile256
        }
    );
    let candidate =
        ExperimentalSoilNewtonGpu::new_volatile_rounding_probe(engine, input, cap).unwrap();
    let solvers = [baseline, &candidate];
    let snapshot = |solver: &ExperimentalSoilNewtonGpu| {
        [
            solver.state_buffer(),
            solver.rates_buffer(),
            solver.outputs_buffer(),
            solver.publication_status_buffer(),
            solver.status_buffer(),
        ]
        .map(|buffer| capture::read(engine, buffer))
    };
    let expected = solvers.map(|solver| {
        eprintln!(
            "{label}-cost warmup variant={}",
            usize::from(std::ptr::eq(solver, &candidate))
        );
        check(engine, solver, fixture, reference);
        snapshot(solver)
    });
    if label == "schur-reuse" {
        assert_eq!(
            expected[0], expected[1],
            "shared assembly must preserve all five buffers exactly"
        );
    }
    for (variant, bytes) in expected.iter().enumerate() {
        let report: [u32; 8] = bytemuck::pod_read_unaligned(&bytes[4][..32]);
        eprintln!("{label}-cost variant={variant} reports={report:?}");
    }
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_ALGORITHM_TRACE") {
        let root = std::path::PathBuf::from(output);
        std::fs::create_dir(&root).unwrap();
        for (variant, buffers) in expected.iter().enumerate() {
            let path = root.join(format!("variant-{variant}"));
            std::fs::create_dir(&path).unwrap();
            for (name, bytes) in ["states", "rates", "outputs", "publication", "status"]
                .into_iter()
                .zip(buffers)
            {
                std::fs::write(path.join(format!("{name}.bin")), bytes).unwrap();
            }
        }
        std::fs::write(root.join("scope.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "acceptance":false,"control_backend":baseline.failure_method().name(),
            "candidate":label,"candidate_is_production_profile":false,
            "dofs":cap.dofs,"contacts":cap.contacts,"newton_limit":cap.newton_iterations,
            "krylov_limit":cap.krylov_iterations,
            "scope":"cold common-input algorithm snapshots after physical checks; repeat results remain in process log; cross-variant agreement is not asserted by this trace"
        })).unwrap()).unwrap();
    }
    for pair in 0..8 {
        for order in 0..2 {
            let variant = (pair + order) % 2;
            let solver = solvers[variant];
            eprintln!("{label}-cost sample={} variant={variant}", 2 * pair + order);
            let report = run(engine, solver, true);
            assert_eq!(report[0], 0, "{label} variant={variant} report={report:?}");
            for (buffer, (actual, expected)) in
                snapshot(solver).iter().zip(&expected[variant]).enumerate()
            {
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual == expected,
                    "{label} variant={variant} buffer={buffer} repeat mismatch"
                );
            }
            super::super::publication::verify_outputs(
                engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                fixture,
                reference,
            );
        }
    }
    eprintln!(
        "{label}-cost verified: independent physical/state/publication gates; own five-buffer repeat equality; cross-variant equality NOT required; variant 0=runtime {:?}, 1={label}",
        baseline.failure_method()
    );
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
pub(super) fn compare_width_cost(
    engine: &Engine,
    baseline: &ExperimentalSoilNewtonGpu,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    fixture: &Fixture,
    reference: &Reference,
) {
    assert_eq!(
        baseline.failure_method(),
        crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedStalled,
        "width timing control must be the 64-lane runtime"
    );
    let wide = ExperimentalSoilNewtonGpu::new_wide_probe(engine, input, cap).unwrap();
    let solvers = [baseline, &wide];
    let snapshot = |solver: &ExperimentalSoilNewtonGpu| {
        [
            solver.state_buffer(),
            solver.rates_buffer(),
            solver.outputs_buffer(),
            solver.publication_status_buffer(),
        ]
        .map(|b| capture::read(engine, b))
    };
    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for (variant, solver) in solvers.iter().enumerate() {
        eprintln!("width-cost warmup variant={variant}");
        let report = run(engine, solver, true);
        assert_eq!(report[0], 0);
        super::super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            fixture,
            reference,
        );
        snapshots.push(snapshot(solver));
        reports.push(report);
    }
    for pair in 0..8 {
        for order in 0..2 {
            let variant = (pair + order) % 2;
            let solver = solvers[variant];
            eprintln!("width-cost sample={} variant={variant}", 2 * pair + order);
            assert_eq!(run(engine, solver, true), reports[variant]);
            assert_eq!(snapshot(solver), snapshots[variant]);
            super::super::publication::verify_outputs(
                engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                fixture,
                reference,
            );
        }
    }
    eprintln!(
        "width-cost reports={reports:?}; identical archived input, per-variant bit-stable repeats, independent physical publication each sample; variant 0=64 lanes, 1={} lanes",
        std::env::var("MOLLA_NEWTON_WIDE_LANES").unwrap()
    );
}

pub(super) fn compare_release_cost(
    engine: &Engine,
    baseline: &ExperimentalSoilNewtonGpu,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    fixture: &Fixture,
    reference: &Reference,
) {
    let ablation =
        ExperimentalSoilNewtonGpu::new_without_release_probe(engine, input, cap).unwrap();
    let solvers = [baseline, &ablation];
    let expected = run(engine, baseline, true);
    assert_eq!(expected[0], 0);
    let snapshot = |solver: &ExperimentalSoilNewtonGpu| {
        [
            solver.state_buffer(),
            solver.rates_buffer(),
            solver.outputs_buffer(),
            solver.publication_status_buffer(),
        ]
        .map(|buffer| capture::read(engine, buffer))
    };
    let bytes = snapshot(baseline);
    for (variant, solver) in solvers.iter().enumerate() {
        for warmup in 0..2 {
            eprintln!("release-cost warmup={warmup} variant={variant}");
            assert_eq!(run(engine, solver, true), expected);
            assert_eq!(snapshot(solver), bytes);
        }
        super::super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            fixture,
            reference,
        );
    }
    for pair in 0..8 {
        for order in 0..2 {
            let variant = (pair + order) % 2;
            eprintln!("release-cost sample={} variant={variant}", 2 * pair + order);
            assert_eq!(run(engine, solvers[variant], true), expected);
        }
    }
    for solver in solvers {
        assert_eq!(snapshot(solver), bytes);
    }
    eprintln!(
        "release-cost exact reports/states/rates/outputs/publication match; variant 0=full, 1=no-release diagnostic only"
    );
}
