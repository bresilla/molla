use super::*;

const ENTRY: &str =
    "@compute @workgroup_size(64)\nfn newton(@builtin(local_invocation_index) id:u32) {";

pub(crate) fn source(input: String) -> String {
    let mut input = active_indices::source(complementarity::baseline(input), false);
    let start = input.find("fn prepare_linear() {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    let mut prepare = input[start..end].to_owned();
    replace(&mut prepare, "inverses[c]=Inverse();", "");
    replace(
        &mut prepare,
        "if !build_inverse(c) {",
        "if !exact_cached_inverse(c) {",
    );
    input.replace_range(start..end, &prepare);
    replace(&mut input, ENTRY, "fn exact_cache_solve(id:u32) {");
    input.push_str(include_str!(
        "../shaders/soil_newton_exact_inverse_cache.wgsl"
    ));
    input.push_str(&format!("\n{ENTRY}\n    exact_cache_solve(id);\n    condensed[id]=bitcast<vec2<f32>>(vec2<u32>(exact_inverse_hits,exact_inverse_builds));\n}}\n"));
    input
}

#[test]
fn newton_exact_inverse_cache_source_contract() {
    let base = active_indices::source(
        complementarity::baseline(shader_source_for(NewtonVariant::Fischer)),
        false,
    );
    let candidate = source(shader_source_for(NewtonVariant::Fischer));
    for name in [
        "build_inverse",
        "schur_build_inverse",
        "dense_build_inverse",
        "hybrid_direction",
        "evaluate",
        "precondition",
        "certify_candidate",
    ] {
        let body = |text: &str| {
            let start = text.find(&format!("fn {name}(")).unwrap();
            let end = start + text[start..].find("\n}").unwrap() + 2;
            text[start..end].to_owned()
        };
        assert_eq!(body(&candidate), body(&base), "{name}");
    }
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            candidate.matches(declaration).count(),
            base.matches(declaration).count()
        );
    }
    assert!(candidate.contains("exact_inverse_contact==c"));
    assert!(candidate.contains("i<36u"));
    assert!(candidate.contains("i<18u"));
    assert!(candidate.contains("if params.pad0==1u && accepted"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_EXACT_INVERSE_SHADERS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), base).unwrap();
        std::fs::write(root.join("shared.wgsl"), candidate).unwrap();
    }
}
