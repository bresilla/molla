use super::*;

const CALL: &str = "if conditioned_release() {step++;";
const SETTLE: &str = "branch_settle=fischer_smoothing>0.0;branch_merit=merit;\nlet first_settled=conditioned_release();branch_settle=false;\nif first_settled {step++;";

fn source(mut source: String) -> String {
    let start = source.find(SMOOTHED_GUARD).unwrap();
    let call = start + source[start..].find(CALL).unwrap();
    source.replace_range(call..call + CALL.len(), SETTLE);
    source
}

pub(crate) fn new_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    new_with_source(engine, input, capacity, |s| source(smoothed_source(s)))
}

#[test]
fn newton_smoothed_settle_contract() {
    let baseline = smoothed_source(ExperimentalSoilNewtonGpu::conditioned_runtime_program(
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
    ));
    let candidate = source(baseline.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, SETTLE, CALL);
    assert_eq!(restored, baseline);
    for marker in [
        "conditioned_release()",
        "@binding(",
        "var<workgroup>",
        "fischer_smoothing*=0.1;",
        "outer.releases)>=8u",
        "backtrack<16u",
        "attempt==8u",
        "if certify_candidate(lane) {return;}",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    for guard in [
        "if branch_settle && rank>0u {break;}",
        "(!branch_settle || score<branch_merit)",
        "if visited {continue;}",
        "smoothed_release_pending && !smoothed_release_attempted",
        "smoothed_release_attempted=true;smoothed_release_pending=false;",
    ] {
        assert!(candidate.contains(guard), "{guard}");
    }
    assert!(SETTLE.contains("conditioned_release();branch_settle=false;"));
    let call = candidate.find(SETTLE).unwrap();
    assert!(candidate[..call].contains("step==params.newton_iterations"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
