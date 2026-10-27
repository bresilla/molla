use super::*;

pub(crate) fn source(mut input: String) -> String {
    let start = input.find("fn prepare_linear() {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    let mut prepare = input[start..end].to_owned();
    replace(
        &mut prepare,
        "inverses[c]=Inverse();",
        "if !inverse_reuse_ready {inverses[c]=Inverse();}",
    );
    replace(
        &mut prepare,
        "if !build_inverse(c) {",
        "if !inverse_reuse_ready && !build_inverse(c) {",
    );
    let tail = prepare.rfind("    storageBarrier();").unwrap();
    prepare.insert_str(tail, "    inverse_reuse_ready=true;\n");
    input.replace_range(start..end, &prepare);
    let start = input.find("fn hybrid_direction(id:u32) {").unwrap();
    let end = start + input[start..].find("\n}").unwrap() + 2;
    let mut hybrid = input[start..end].to_owned();
    replace(
        &mut hybrid,
        "if fault!=8u {return;}",
        "if fault!=0u {inverse_reuse_ready=false;}\n    if fault!=8u {return;}",
    );
    input.replace_range(start..end, &hybrid);
    input.push_str("\nvar<private> inverse_reuse_ready:bool=false;\n");
    input
}

#[test]
fn newton_lagged_preconditioner_contract() {
    let control = slot_constants::baseline(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone());
    let body = |s: &str, name: &str| {
        let start = s.find(&format!("fn {name}(")).unwrap();
        let end = start + s[start..].find("\n}").unwrap() + 2;
        s[start..end].to_owned()
    };
    for name in [
        "newton",
        "fischer_store_block",
        "nonlinear_contact",
        "evaluate",
        "true_direction_error",
        "build_inverse",
        "certify_candidate",
        "conditioned_release",
        "precondition",
    ] {
        assert_eq!(body(&control, name), body(&candidate, name));
    }
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    let prepare = body(&candidate, "prepare_linear");
    assert!(prepare.contains("if !inverse_reuse_ready {inverses[c]=Inverse();}"));
    assert!(prepare.contains("if !inverse_reuse_ready && !build_inverse(c) {"));
    assert!(
        prepare.find("build_inverse(c)").unwrap()
            < prepare.find("inverse_reuse_ready=true;").unwrap()
    );
    let hybrid = body(&candidate, "hybrid_direction");
    assert!(
        hybrid.find("inverse_reuse_ready=false;").unwrap()
            < hybrid.find("prepare_linear();solve_bounded").unwrap()
    );
    assert!(hybrid.contains("solve_bounded(id,min(8u,params.iterations));"));
    assert!(hybrid.contains("solve_bounded(id,params.iterations-used);"));
    assert!(candidate.contains("var<private> inverse_reuse_ready:bool=false;"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_LAGGED_PRECONDITIONER_SHADERS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        for (name, shader) in [("control", control), ("candidate", candidate)] {
            std::fs::write(root.join(format!("{name}.wgsl")), shader).unwrap();
        }
    }
}
