use super::*;

const DECLARE: &str = "\nvar<private> smoothed_trace_active:bool=false;\n";
const MARK: &str = "\nsmoothed_trace_active=fischer_smoothing>0.0;\nif smoothed_trace_active && lane==0u {report.diagnostic_limit=step+1u;}";
const CALL: &str = "if conditioned_release() {step++;";
const TRACED_CALL: &str = "let traced_released=conditioned_release();let traced_attempt=smoothed_trace_active;smoothed_trace_active=false;\nif traced_released {if traced_attempt && lane==0u {report.diagnostic_limit|=0x80000000u;}step++;";
const CACHE: &str = "conditioned_cache();";
const TRACED_CACHE: &str = "conditioned_cache();\nif smoothed_trace_active && lane==0u {var invalid_count=0u;for(var slot=0u;slot<status.active_count;slot++) {let c=active_indices[slot];if condensed[4160u+54u*c+53u].x!=1.0 {invalid_count++;}}report.diagnostic_limit|=invalid_count<<8u;}";

fn source(mut source: String) -> String {
    assert_eq!(source.matches("report.diagnostic_limit").count(), 1);
    let guard = format!("{SMOOTHED_GUARD}{MARK}");
    replace(&mut source, SMOOTHED_GUARD, &guard);
    let start = source.find(&guard).unwrap();
    let call = start + source[start..].find(CALL).unwrap();
    source.replace_range(call..call + CALL.len(), TRACED_CALL);
    replace(&mut source, CACHE, TRACED_CACHE);
    source.push_str(DECLARE);
    source
}

pub(crate) fn new_probe(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    capacity: SoilNewtonCapacity,
) -> Result<ExperimentalSoilNewtonGpu> {
    assert!(capacity.newton_iterations < 256 && capacity.contacts <= 4096);
    assert!(std::env::var_os("MOLLA_NEWTON_DIAGNOSTIC_INNER_LIMIT").is_none());
    new_with_source(engine, input, capacity, |s| source(smoothed_source(s)))
}

pub(crate) fn decode(word: u32, active: usize, limit: u32) -> Option<serde_json::Value> {
    if word == 0 {
        return None;
    }
    assert_eq!(word & !0x801fffff, 0);
    let encoded_step = word & 255;
    assert!(encoded_step > 0 && encoded_step <= limit);
    let invalid = (word >> 8) & 8191;
    assert!(invalid as usize <= active);
    Some(serde_json::json!({
        "attempt_at_step": encoded_step - 1,
        "invalid_local_caches": invalid,
        "recovered": word & 0x80000000 != 0,
    }))
}

#[test]
fn newton_smoothed_trace_contract() {
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
    let traced = source(baseline.clone());
    let mut restored = traced.clone();
    replace(
        &mut restored,
        &format!("{SMOOTHED_GUARD}{MARK}"),
        SMOOTHED_GUARD,
    );
    replace(&mut restored, TRACED_CALL, CALL);
    replace(&mut restored, TRACED_CACHE, CACHE);
    replace(&mut restored, DECLARE, "");
    assert_eq!(restored, baseline);
    for marker in [
        "conditioned_release()",
        "@binding(",
        "var<workgroup>",
        "fischer_smoothing*=0.1;",
        "if certify_candidate(lane) {return;}",
    ] {
        assert_eq!(
            traced.matches(marker).count(),
            baseline.matches(marker).count()
        );
    }
    crate::soil_newton_protected::compile(&traced, "newton").unwrap();
    assert!(decode(0, 219, 64).is_none());
    assert_eq!(
        decode(17 | (2 << 8), 219, 64).unwrap(),
        serde_json::json!({
            "attempt_at_step": 16, "invalid_local_caches": 2, "recovered": false,
        })
    );
    assert_eq!(
        decode(0x80000001 | (4096 << 8), 4096, 64).unwrap()["recovered"],
        true
    );
    for word in [0x80000000, 1 << 21, 65, 1 | (220 << 8)] {
        assert!(std::panic::catch_unwind(|| decode(word, 219, 64)).is_err());
    }
}
