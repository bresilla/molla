use super::*;

const CERTIFY: &str = "if certify_candidate(lane) {return;}";
const CALL: &str = "if conditioned_release() {step++;";
const RECOVERED: &str =
    "if conditioned_release() {if fischer_smoothing>0.0 {stage_recovered=true;}step++;";
const DECLARE: &str = "\nvar<private> stage_recovered:bool=false;\n";
const TRACE: &str = "if certify_candidate(lane) {if lane==0u {report.diagnostic_limit=1u|select(0u,2u,settle_pending)|select(0u,4u,fischer_smoothing==0.0)|select(0u,8u,step<params.newton_iterations)|select(0u,16u,stage_recovered)|(smoothed_stage_attempts<<8u)|(outer.releases<<16u);}return;}";
const COMPLETE: &str =
    "if certify_candidate(lane) {if !stage_recovered || fischer_smoothing==0.0 {return;}}";

fn completion_source(source: String) -> String {
    let mut source = trace_source(source);
    replace(&mut source, TRACE, COMPLETE);
    source
}

pub(crate) fn new_completion_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    new_with_source(engine, input, capacity, |s| {
        completion_source(super::source(smoothed_source(s)))
    })
}

fn trace_source(mut source: String) -> String {
    assert_eq!(source.matches("report.diagnostic_limit").count(), 1);
    let start = source.find(GUARD).unwrap();
    let at = start + source[start..].find(CALL).unwrap();
    source.replace_range(at..at + CALL.len(), RECOVERED);
    replace(&mut source, CERTIFY, TRACE);
    source.push_str(DECLARE);
    source
}

pub(crate) fn new_trace_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    assert!(std::env::var_os("MOLLA_NEWTON_DIAGNOSTIC_INNER_LIMIT").is_none());
    new_with_source(engine, input, capacity, |s| {
        trace_source(super::source(smoothed_source(s)))
    })
}

fn decode(word: u32) -> (bool, bool, bool, bool, u32, u32) {
    assert_eq!(word & !0x000f0f1f, 0);
    assert_eq!(word & 1, 1);
    let attempts = (word >> 8) & 15;
    let releases = (word >> 16) & 15;
    assert!(attempts <= 8 && releases <= 8);
    let recovered = word & 16 != 0;
    assert!(!recovered || (attempts > 0 && releases > 0));
    (
        word & 2 != 0,
        word & 4 != 0,
        word & 8 != 0,
        recovered,
        attempts,
        releases,
    )
}

#[test]
fn newton_finish_pending_trace_contract() {
    let baseline = super::source(smoothed_source(
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
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
        ),
    ));
    let candidate = trace_source(baseline.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, RECOVERED, CALL);
    replace(&mut restored, TRACE, CERTIFY);
    replace(&mut restored, DECLARE, "");
    assert_eq!(restored, baseline);
    for marker in [
        "conditioned_release()",
        "certify_candidate(lane)",
        "@binding(",
        "var<workgroup>",
        "fischer_smoothing*=0.1;",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    assert_eq!(decode(1), (false, false, false, false, 0, 0));
    assert_eq!(
        decode(31 | (8 << 8) | (8 << 16)),
        (true, true, true, true, 8, 8)
    );
    for word in [0, 33, 17, 1 | (9 << 8), 1 | (9 << 16)] {
        assert!(std::panic::catch_unwind(|| decode(word)).is_err());
    }
}

#[test]
fn newton_finish_pending_completion_contract() {
    let baseline = super::source(smoothed_source(
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
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
        ),
    ));
    let candidate = completion_source(baseline.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, RECOVERED, CALL);
    replace(&mut restored, COMPLETE, CERTIFY);
    replace(&mut restored, DECLARE, "");
    assert_eq!(restored, baseline);
    for marker in [
        "conditioned_release()",
        "certify_candidate(lane)",
        "report.diagnostic_limit",
        "@binding(",
        "var<workgroup>",
        "fischer_smoothing*=0.1;",
        "fischer_smoothing<1e-10",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
        "backtrack<16u",
        "attempt==8u",
        "if visited {continue;}",
        "(!branch_settle || score<branch_merit)",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    for certified in [false, true] {
        for recovered in [false, true] {
            for smoothing in [0.0f32, 1e-10, 0.001] {
                let returns = certified && (!recovered || smoothing == 0.0);
                if !recovered || smoothing == 0.0 {
                    assert_eq!(returns, certified);
                }
                if recovered && smoothing > 0.0 {
                    assert!(!returns);
                }
                if returns {
                    assert!(certified);
                }
            }
        }
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
