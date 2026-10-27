use super::*;

#[path = "soil_newton_finish_pending.rs"]
pub(crate) mod finish_pending;

const GUARD: &str = "if (fischer_smoothing==0.0 && smoothing_stalled==1u && workgroupUniformLoad(&outer.releases)>0u) || (smoothed_release_pending && smoothed_stage_attempts<8u && fischer_smoothing>0.0 && fischer_smoothing!=smoothed_stage_last && workgroupUniformLoad(&outer.releases)==0u) {\nif fischer_smoothing>0.0 {smoothed_stage_attempts++;smoothed_stage_last=fischer_smoothing;smoothed_release_pending=false;}";
const DECLARE: &str = "\nvar<private> smoothed_release_pending:bool=false;\nvar<private> smoothed_stage_attempts:u32=0u;\nvar<private> smoothed_stage_last:f32=0.0;\n";

fn source(mut source: String) -> String {
    replace(&mut source, SMOOTHED_GUARD, GUARD);
    replace(&mut source, SMOOTHED_DECLARATIONS, DECLARE);
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
fn newton_smoothed_stages_contract() {
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
    replace(&mut restored, GUARD, SMOOTHED_GUARD);
    replace(&mut restored, DECLARE, SMOOTHED_DECLARATIONS);
    assert_eq!(restored, baseline);
    assert!(!candidate.contains("smoothed_release_attempted"));
    for marker in [
        "conditioned_release()",
        "@binding(",
        "var<workgroup>",
        "fischer_smoothing*=0.1;",
        "fischer_smoothing<1e-10",
        "outer.releases)>=8u",
        "backtrack<16u",
        "attempt==8u",
        "if visited {continue;}",
        "if certify_candidate(lane) {return;}",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    let eligible = |pending, count, smoothing, last, releases| {
        pending && count < 8 && smoothing > 0.0f32 && smoothing != last && releases == 0
    };
    let (mut count, mut smoothing, mut last) = (0, 0.01f32, 0.0f32);
    for _ in 0..16 {
        smoothing *= 0.1;
        if smoothing < 1e-10 {
            smoothing = 0.0;
        }
        for _ in 0..4 {
            if eligible(true, count, smoothing, last, 0) {
                count += 1;
                last = smoothing;
            }
            assert!(!eligible(true, count, smoothing, last, 0));
            assert!(!eligible(false, count, smoothing, 0.0, 0));
            assert!(!eligible(true, count, smoothing, 0.0, 1));
        }
        assert!(count <= 8);
    }
    assert!(count > 1);
    assert!(!eligible(true, 8, 0.01, 0.0, 0));
    assert!(candidate[..candidate.find(GUARD).unwrap()].contains("step==params.newton_iterations"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
