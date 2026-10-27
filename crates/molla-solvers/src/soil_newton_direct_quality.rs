use super::*;

#[path = "soil_newton_direct_quality_tests.rs"]
mod tests;

pub(crate) const SHARED_BYTES: u32 = 128 * 16 + 8;
pub(crate) const EXPORTED_BYTES: usize = 8 + 128 * 16;

fn source(mut input: String) -> String {
    let start = input.find("fn direct_direction()->bool {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    let mut direct = input[start..end].to_owned();
    replace(
        &mut direct,
        "let error=true_direction_error();",
        "let error=true_direction_error();record_direct_quality(error);",
    );
    input.replace_range(start..end, &direct);
    replace(
        &mut input,
        "@compute @workgroup_size(64)\nfn newton(@builtin(local_invocation_index) id:u32) {",
        "fn quality_traced_newton(id:u32) {",
    );
    input.push_str(include_str!("../shaders/soil_newton_direct_quality.wgsl"));
    input
}

fn program(input: String, candidate: bool) -> String {
    let input = retained_conditioned::source(retained_conditioned::baseline(input));
    let input = if candidate { source(input) } else { input };
    format!(
        "{}{}{}256\n{input}",
        crate::soil_newton_protected::PREFIX,
        crate::soil_newton_protected::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    )
}

pub(crate) fn new(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    candidate: bool,
) -> Result<ExperimentalSoilNewtonGpu> {
    let (lanes, storage) = execution_limits(engine);
    let limits = engine.device().ok_or_else(invalid)?.limits();
    let extra = if candidate { SHARED_BYTES } else { 0 };
    if lanes != 256
        || !protected_enabled(engine)
        || limits.max_compute_workgroup_storage_size < storage + extra
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
    {
        return Err(invalid());
    }
    ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        cap,
        NewtonVariant::Fischer,
        |input| program(input, candidate),
        "newton",
        true,
    )
}

pub(crate) fn decode(bytes: &[u8]) -> Vec<[f32; 4]> {
    assert!(bytes.len() >= EXPORTED_BYTES, "truncated direct quality");
    let float = |i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let count = float(0);
    assert!(count.is_finite() && count >= 0.0 && count <= 128.0 && count.fract() == 0.0);
    assert_eq!(float(1), 0.0, "direct quality trace overflow");
    let records = (0..count as usize)
        .map(|i| {
            let row = std::array::from_fn(|j| float(2 + 4 * i + j));
            assert!(
                row.iter().all(|x| x.is_finite()),
                "nonfinite direct quality"
            );
            assert!(
                row[0] >= 0.0
                    && row[1] >= 0.0
                    && row[2] >= -1.0
                    && row[2].fract() == 0.0
                    && row[3] >= 0.0
                    && row[3].fract() == 0.0
            );
            row
        })
        .collect();
    for i in 2 + 4 * count as usize..2 + 4 * 128 {
        assert_eq!(float(i), 0.0, "stale direct quality tail");
    }
    records
}

#[test]
fn newton_direct_quality_source_and_decode_contract() {
    let control = retained_conditioned::source(retained_conditioned::baseline(shader_source_for(
        NewtonVariant::Fischer,
    )));
    let candidate = source(control.clone());
    let suffix = include_str!("../shaders/soil_newton_direct_quality.wgsl");
    let restored = candidate
        .strip_suffix(suffix)
        .unwrap()
        .replace(
            "let error=true_direction_error();record_direct_quality(error);",
            "let error=true_direction_error();",
        )
        .replace(
            "fn quality_traced_newton(id:u32) {",
            "@compute @workgroup_size(64)\nfn newton(@builtin(local_invocation_index) id:u32) {",
        );
    assert_eq!(restored, control);
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    crate::soil_newton_protected::compile(
        &program(shader_source_for(NewtonVariant::Fischer), true),
        "newton",
    )
    .unwrap();
    let zero = vec![0u8; EXPORTED_BYTES];
    assert!(decode(&zero).is_empty());
    let mut one = zero.clone();
    one[..4].copy_from_slice(&1f32.to_le_bytes());
    one[8..12].copy_from_slice(&0.125f32.to_le_bytes());
    assert_eq!(decode(&one), vec![[0.125, 0.0, 0.0, 0.0]]);
    for (offset, value) in [(0, 129.0), (0, 0.5), (4, 1.0), (8, f32::NAN), (24, 1.0)] {
        let mut bad = one.clone();
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(std::panic::catch_unwind(|| decode(&bad)).is_err());
    }
    assert!(std::panic::catch_unwind(|| decode(&zero[..EXPORTED_BYTES - 1])).is_err());
}
