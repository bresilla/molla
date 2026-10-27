use super::*;

#[path = "soil_newton_retained_direct_tests.rs"]
mod tests;

pub(super) fn source(mut input: String) -> String {
    let from = "            } else {\n                sum=vec2<f32>(response[chosen*width+row_index][column-n-1u],0.0);\n            }";
    replace(
        &mut input,
        from,
        &from.replacen("} else {", "} else if chosen!=0xffffffffu {", 1),
    );
    replace(
        &mut input,
        "if lane<3u {condensed[4090u+lane]=vec2<f32>(states[chosen].high.k[lane],states[chosen].low.k[lane]);}",
        "if lane<3u {condensed[4090u+lane]=vec2<f32>(0.0);if chosen!=0xffffffffu {condensed[4090u+lane]=vec2<f32>(states[chosen].high.k[lane],states[chosen].low.k[lane]);}}",
    );
    let start = input.find("fn direct_direction()->bool {").unwrap();
    let first = start + input[start..].find("    for(var slot=lane;slot<status.active_count;slot+=64u) {\n        if !direct_contact").unwrap();
    let end = first
        + input[first..]
            .find("    let error=true_direction_error();")
            .unwrap();
    input.replace_range(
        first..end,
        "    if !retained_direct_solve() {return false;}\n",
    );
    input.push_str(include_str!("../shaders/soil_newton_retained_direct.wgsl"));
    input
}

pub(super) fn program(input: String, candidate: bool) -> String {
    program_with_policy(input, candidate, false)
}

fn lazy_source(mut input: String) -> String {
    let start = input.find("fn direct_direction()->bool {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    let mut original = input[start..end].to_owned();
    replace(
        &mut original,
        "fn direct_direction()->bool {",
        "fn initial_direct_direction()->bool {\n    retained_direct_ready=false;",
    );
    replace(
        &mut original,
        "        if !direct_contact(active_indices[slot]) {atomicStore(&condensed_failed,1u);}",
        "        let c=active_indices[slot];let valid=direct_contact(c);condensed[4160u+54u*c+53u]=vec2<f32>(select(0.0,1.0,valid),0.0);if !valid {atomicStore(&condensed_failed,1u);}",
    );
    replace(
        &mut original,
        "    storageBarrier();if !condensed_ok() {return false;}",
        "    retained_direct_ready=true;storageBarrier();if !condensed_ok() {return false;}",
    );
    let cache = input.find("fn conditioned_cache() {").unwrap();
    let end = cache + input[cache..].find("\n}").unwrap() + 2;
    let mut validation = input[cache..end].to_owned();
    replace(
        &mut validation,
        "fn conditioned_cache() {",
        "fn retained_validate_columns() {",
    );
    replace(
        &mut validation,
        "var valid=direct_contact(c);",
        "var valid=condensed[offset+53u].x==1.0;",
    );
    input = source(input);
    let start = input.find("fn direct_direction()->bool {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    input.replace_range(start..end,r#"fn direct_direction()->bool {
    if initial_direct_direction() {return true;}
    if !retained_direct_ready {return false;}
    prepare_condensed();
    if lane==0u {atomicStore(&condensed_failed,0u);control.beta=refinement_norm;status.beta=refinement_norm;}
    storageBarrier();
    if !retained_direct_solve() {return false;}
    let error=true_direction_error();
    if lane==0u {status.iterations=1u;}
    return error<=params.tolerance;
}"#);
    replace(
        &mut input,
        "fn retained_direct_solve()->bool {\n    conditioned_cache();",
        "fn retained_direct_solve()->bool {\n    retained_validate_columns();",
    );
    replace(
        &mut input,
        "        direct_core();if !condensed_ok() {return false;}\n        if !condensed_factor(params.width) {return false;}\n        direct_recover();",
        "        return false;",
    );
    input.push_str(&format!(
        "\nvar<private> retained_direct_ready:bool=false;\n{original}\n{validation}\n"
    ));
    input
}

fn quality_source(input: String) -> String {
    let mut input = lazy_source(input);
    let start = input.find("fn direct_direction()->bool {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    input.replace_range(
        start..end,
        include_str!("../shaders/soil_newton_direct_quality_correction.wgsl"),
    );
    let metric = include_str!("../shaders/soil_newton_direct_quality.wgsl");
    let start = metric.find("fn direct_contact_relative(").unwrap();
    let end = start + metric[start..].find("\n}").unwrap() + 2;
    input.push_str(&metric[start..end]);
    input
}

pub(super) fn quality_program(input: String, candidate: bool) -> String {
    let input = retained_conditioned::source(retained_conditioned::baseline(input));
    let input = if candidate {
        quality_source(input)
    } else {
        input
    };
    format!(
        "{}{}{}256\n{input}",
        crate::soil_newton_protected::PREFIX,
        crate::soil_newton_protected::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    )
}

pub(super) fn program_with_policy(input: String, candidate: bool, lazy: bool) -> String {
    let input = retained_conditioned::source(retained_conditioned::baseline(input));
    let input = if candidate {
        if lazy {
            lazy_source(input)
        } else {
            source(input)
        }
    } else {
        input
    };
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
    lazy: bool,
) -> Result<ExperimentalSoilNewtonGpu> {
    let (lanes, storage) = execution_limits(engine);
    let limits = engine.device().ok_or_else(invalid)?.limits();
    if lanes != 256
        || !protected_enabled(engine)
        || limits.max_compute_workgroup_storage_size < storage
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
        |input| {
            if super::distributed::mixed::kind() == "retained-direct-quality" {
                quality_program(input, candidate)
            } else {
                program_with_policy(input, candidate, lazy)
            }
        },
        "newton",
        true,
    )
}

#[test]
fn newton_retained_direct_quality_source_contract() {
    let control = retained_conditioned::source(retained_conditioned::baseline(shader_source_for(
        NewtonVariant::Fischer,
    )));
    let candidate = quality_source(control.clone());
    for marker in [
        "var<workgroup>",
        "var<storage",
        "backtrack<16u",
        "attempt==8u",
        "outer.releases)>=8u",
    ] {
        assert_eq!(
            control.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    for marker in [
        "} else {prepare_condensed();}",
        "direction[count+c]=direction[c]",
        "direction[2u*count+c]=basis[params.iterations*count+c].z",
        "direction[c]=direction[count+c]",
        "basis[params.iterations*count+c].z=direction[2u*count+c]",
        "if error<=params.tolerance",
        "if !accepted || after<before",
    ] {
        assert!(candidate.contains(marker), "missing {marker}");
    }
    crate::soil_newton_protected::compile(
        &quality_program(shader_source_for(NewtonVariant::Fischer), true),
        "newton",
    )
    .unwrap();
}

#[test]
fn newton_retained_direct_lazy_source_contract() {
    let control = retained_conditioned::source(retained_conditioned::baseline(shader_source_for(
        NewtonVariant::Fischer,
    )));
    let candidate = lazy_source(control.clone());
    for marker in [
        "var<workgroup>",
        "var<storage",
        "backtrack<16u",
        "attempt==8u",
    ] {
        assert_eq!(
            control.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    assert!(candidate.contains("if initial_direct_direction() {return true;}"));
    assert!(candidate.contains("if !retained_direct_ready {return false;}"));
    let validation = candidate
        .split_once("fn retained_validate_columns()")
        .unwrap()
        .1;
    assert!(!validation.contains("direct_contact("));
    crate::soil_newton_protected::compile(
        &program_with_policy(shader_source_for(NewtonVariant::Fischer), true, true),
        "newton",
    )
    .unwrap();
}

#[test]
fn newton_retained_direct_source_contract() {
    let control = retained_conditioned::source(retained_conditioned::baseline(shader_source_for(
        NewtonVariant::Fischer,
    )));
    let candidate = source(control.clone());
    for name in [
        "hybrid_direction",
        "true_direction_error",
        "conditioned_factor",
        "solve_bounded",
    ] {
        let body = |s: &str| {
            let start = s.find(&format!("fn {name}(")).unwrap();
            let end = start + s[start..].find("\n}").unwrap() + 2;
            s[start..end].to_owned()
        };
        assert_eq!(body(&candidate), body(&control), "unchanged {name}");
    }
    for marker in [
        "var<workgroup>",
        "var<storage",
        "backtrack<16u",
        "attempt==8u",
        "outer.releases)>=8u",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    for width in 1..=32 {
        let n = width + 6;
        assert!(n * (n + 4) <= 1600);
        assert!(1600 + n * (n + 4) < 4000);
    }
    assert!(candidate.contains("if invalid!=1u {return false;}"));
    crate::soil_newton_protected::compile(
        &program(shader_source_for(NewtonVariant::Fischer), true),
        "newton",
    )
    .unwrap();
}
