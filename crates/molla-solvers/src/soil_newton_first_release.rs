use super::*;

#[path = "soil_newton_smoothed_trace.rs"]
pub(crate) mod smoothed_trace;

#[path = "soil_newton_smoothed_settle.rs"]
pub(crate) mod smoothed_settle;

#[path = "soil_newton_smoothed_stages.rs"]
pub(crate) mod smoothed_stages;

const ANCHOR: &str = "if fischer_smoothing==0.0 && smoothing_stalled==1u && workgroupUniformLoad(&outer.releases)>0u {";
const GUARD: &str = "!first_release_attempted && fischer_smoothing==0.0 && smoothing_stalled>=2u && workgroupUniformLoad(&outer.releases)==0u";
const DECLARATION: &str = "\nvar<private> first_release_attempted:bool=false;\n";

pub(crate) fn source(mut source: String) -> String {
    let inserted = format!(
        "if {GUARD} {{\nfirst_release_attempted=true;\nif conditioned_release() {{step++;if lane==0u {{report.steps=step;report.trials++;}}settle_pending=true;phase=0u;continue;}}\n}}\n{ANCHOR}"
    );
    replace(&mut source, ANCHOR, &inserted);
    source.push_str(DECLARATION);
    source
}

pub(crate) fn new_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    new_with_source(engine, input, capacity, source)
}

const SHARED_GUARD: &str = "if fischer_smoothing==0.0 && ((smoothing_stalled==1u && workgroupUniformLoad(&outer.releases)>0u) || (params.pad2==1u && !first_release_attempted && smoothing_stalled>=2u && workgroupUniformLoad(&outer.releases)==0u)) {\nif workgroupUniformLoad(&outer.releases)==0u {first_release_attempted=true;}";

fn shared_source(mut source: String) -> String {
    assert!(!source.contains("params.pad2"));
    replace(&mut source, ANCHOR, SHARED_GUARD);
    source.push_str(DECLARATION);
    source
}

const STALL_UPDATE: &str =
    "smoothing_stalled=select(0u,smoothing_stalled+1u,residual>=0.99*previous);";
const SMOOTHED_UPDATE: &str = "smoothing_stalled=select(0u,smoothing_stalled+1u,residual>=0.99*previous);\nsmoothed_release_pending=fischer_smoothing>0.0 && residual>fischer_smoothing && smoothing_stalled>=2u;";
const SMOOTHED_GUARD: &str = "if (fischer_smoothing==0.0 && smoothing_stalled==1u && workgroupUniformLoad(&outer.releases)>0u) || (smoothed_release_pending && !smoothed_release_attempted && fischer_smoothing>0.0 && workgroupUniformLoad(&outer.releases)==0u) {\nif fischer_smoothing>0.0 {smoothed_release_attempted=true;smoothed_release_pending=false;}";
const SMOOTHED_DECLARATIONS: &str = "\nvar<private> smoothed_release_pending:bool=false;\nvar<private> smoothed_release_attempted:bool=false;\n";

fn smoothed_source(mut source: String) -> String {
    replace(&mut source, STALL_UPDATE, SMOOTHED_UPDATE);
    replace(&mut source, ANCHOR, SMOOTHED_GUARD);
    source.push_str(SMOOTHED_DECLARATIONS);
    source
}

pub(crate) fn new_smoothed_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    new_with_source(engine, input, capacity, smoothed_source)
}

#[test]
fn newton_smoothed_release_contract() {
    let baseline = ExperimentalSoilNewtonGpu::conditioned_runtime_program(
        shader_source_for(NewtonVariant::Fischer),
        true,
        256,
        SoilNewtonCapacity {
            dofs: 10,
            contacts: 4096,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    );
    let candidate = smoothed_source(baseline.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, SMOOTHED_UPDATE, STALL_UPDATE);
    replace(&mut restored, SMOOTHED_GUARD, ANCHOR);
    replace(&mut restored, SMOOTHED_DECLARATIONS, "");
    assert_eq!(restored, baseline);
    for marker in [
        "conditioned_release()",
        "fischer_smoothing*=0.1;",
        "fischer_smoothing<1e-10",
        "backtrack<16u",
        "attempt==8u",
        "outer.releases)>=8u",
        "if visited {continue;}",
        "if certify_candidate(lane) {return;}",
        "@binding(",
        "var<workgroup>",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    for smoothing in [0.0f32, 1e-10, 1e-8, 1e-4, 0.01] {
        for residual in [0.0f32, 1e-8, 1e-4, 0.01, 1.0] {
            for stalled in 0..4 {
                let pending = smoothing > 0.0 && residual > smoothing && stalled >= 2;
                for attempted in [false, true] {
                    for releases in 0..9 {
                        let eligible = pending && !attempted && smoothing > 0.0 && releases == 0;
                        if eligible {
                            assert!(stalled >= 2 && residual > smoothing);
                            assert!(!attempted && releases == 0);
                        }
                    }
                }
                if residual <= smoothing || stalled < 2 || smoothing == 0.0 {
                    assert!(!pending);
                }
            }
        }
    }
    let at = candidate.find(SMOOTHED_GUARD).unwrap();
    assert!(candidate[..at].contains("step==params.newton_iterations"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

pub(crate) fn new_shared_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    new_with_source(engine, input, capacity, shared_source)
}

pub(crate) fn set_policy(solver: &ExperimentalSoilNewtonGpu, engine: &Engine, enabled: bool) {
    engine
        .queue()
        .unwrap()
        .write_buffer(&solver.scratch[7], 28, &u32::from(enabled).to_le_bytes());
}

fn new_with_source(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
    transform: fn(String) -> String,
) -> Result<ExperimentalSoilNewtonGpu> {
    let (lanes, storage) = execution_limits(engine);
    let limits = engine.device().ok_or_else(invalid)?.limits();
    if !protected_enabled(engine)
        || lanes != 256
        || limits.max_compute_invocations_per_workgroup < lanes
        || limits.max_compute_workgroup_size_x < lanes
        || limits.max_compute_workgroup_storage_size < storage
    {
        return Err(invalid());
    }
    let mut solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        capacity,
        NewtonVariant::Fischer,
        |s| {
            transform(ExperimentalSoilNewtonGpu::conditioned_runtime_program(
                s, true, lanes, capacity,
            ))
        },
        "newton",
        true,
    )?;
    solver.failure_method = NewtonFailureMethod::FischerConditionedProtectedSoaScratch256;
    Ok(solver)
}

#[test]
fn newton_first_release_shared_contract() {
    let baseline = ExperimentalSoilNewtonGpu::conditioned_runtime_program(
        shader_source_for(NewtonVariant::Fischer),
        true,
        256,
        SoilNewtonCapacity {
            dofs: 10,
            contacts: 4096,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    );
    let shared = shared_source(baseline.clone());
    let mut restored = shared.clone();
    replace(&mut restored, SHARED_GUARD, ANCHOR);
    replace(&mut restored, DECLARATION, "");
    assert_eq!(restored, baseline);
    for marker in ["conditioned_release()", "@binding(", "var<workgroup>"] {
        assert_eq!(
            baseline.matches(marker).count(),
            shared.matches(marker).count()
        );
    }
    assert_eq!(shared.matches("params.pad2").count(), 1);
    for enabled in [false, true] {
        for attempted in [false, true] {
            for smoothed in [false, true] {
                for stalled in 0..4 {
                    for releases in 0..9 {
                        let original = !smoothed && stalled == 1 && releases > 0;
                        let first =
                            enabled && !attempted && !smoothed && stalled >= 2 && releases == 0;
                        let combined = !smoothed
                            && ((stalled == 1 && releases > 0)
                                || (enabled && !attempted && stalled >= 2 && releases == 0));
                        assert_eq!(combined, original || first);
                        assert!(!(original && first));
                        assert_eq!(attempted || (combined && releases == 0), attempted || first);
                    }
                }
            }
        }
    }
    crate::soil_newton_protected::compile(&shared, "newton").unwrap();
}

#[test]
fn newton_first_release_guard_contract() {
    let baseline = ExperimentalSoilNewtonGpu::conditioned_runtime_program(
        shader_source_for(NewtonVariant::Fischer),
        true,
        256,
        SoilNewtonCapacity {
            dofs: 10,
            contacts: 4096,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    );
    let candidate = source(baseline.clone());
    let start = candidate.find(&format!("if {GUARD}")).unwrap();
    let end = start + candidate[start..].find(ANCHOR).unwrap();
    let mut restored = candidate.clone();
    restored.replace_range(start..end, "");
    replace(&mut restored, DECLARATION, "");
    assert_eq!(restored, baseline);
    assert!(candidate[..start].contains("step==params.newton_iterations"));
    for marker in [
        "backtrack<16u",
        "attempt==8u",
        "outer.releases)>=8u",
        "if visited {continue;}",
        "if certify_candidate(lane) {return;}",
        "var<workgroup>",
        "@binding(",
    ] {
        assert_eq!(
            baseline.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    for attempted in [false, true] {
        for smoothed in [false, true] {
            for stalled in 0..4 {
                for releases in 0..9 {
                    let eligible = !attempted && !smoothed && stalled >= 2 && releases == 0;
                    if eligible {
                        assert!(!attempted && !smoothed && releases == 0 && stalled >= 2);
                    } else {
                        assert!(attempted || smoothed || stalled < 2 || releases != 0);
                    }
                }
            }
        }
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
