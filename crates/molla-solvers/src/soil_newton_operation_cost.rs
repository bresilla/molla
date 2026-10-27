pub(crate) fn source(mut source: String, scope: &str) -> String {
    if scope == "project2" {
        super::replace(
            &mut source,
            "fn project_rates(kind:u32) {",
            "fn measured_project_rates(kind:u32) {",
        );
        source.push_str("\nfn project_rates(kind:u32) {\n    for(var operation_repeat=0u;operation_repeat<2u;operation_repeat++) {measured_project_rates(kind);}\n}\n");
        return source;
    }
    if scope == "prepare2" {
        super::replace(
            &mut source,
            "fn prepare_linear() {",
            "fn measured_prepare_linear() {",
        );
        source.push_str("\nfn prepare_linear() {\n    for(var operation_repeat=0u;operation_repeat<2u;operation_repeat++) {measured_prepare_linear();}\n}\n");
        return source;
    }
    let (name, args, call, repeats) = match scope {
        "inner2" => ("inner_product", "a:u32,ai:u32,b:u32,bi:u32", "a,ai,b,bi", 2),
        "inner4" => ("inner_product", "a:u32,ai:u32,b:u32,bi:u32", "a,ai,b,bi", 4),
        "norm2" => ("scratch_norm", "", "", 2),
        "norm4" => ("scratch_norm", "", "", 4),
        _ => panic!("unsupported operation cost scope"),
    };
    super::replace(
        &mut source,
        &format!("fn {name}({args})->f32 {{"),
        &format!("fn measured_{name}({args})->f32 {{"),
    );
    source.push_str(&format!("\nfn {name}({args})->f32 {{\n    var result=0.0;\n    for(var operation_repeat=0u;operation_repeat<{repeats}u;operation_repeat++) {{result=measured_{name}({call});}}\n    return result;\n}}\n"));
    source
}

#[test]
fn newton_operation_cost_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::schur_inverse::source(super::core_source(super::source(
        Solver::cached_contact_source(Solver::cached_delta_source(Solver::stalled_release_source(
            shader_source_for(NewtonVariant::Fischer),
        ))),
    )));
    for scope in ["inner2", "inner4", "norm2", "norm4", "prepare2", "project2"] {
        let candidate = source(control.clone(), scope);
        let name = if scope.starts_with("inner") {
            "inner_product"
        } else if scope.starts_with("norm") {
            "scratch_norm"
        } else if scope.starts_with("prepare") {
            "prepare_linear"
        } else {
            "project_rates"
        };
        let (base, _) = candidate.rsplit_once(&format!("\nfn {name}(")).unwrap();
        assert_eq!(
            base.replace(&format!("fn measured_{name}("), &format!("fn {name}(")),
            control
        );
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{candidate}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_OPERATION_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{scope}.wgsl")), candidate).unwrap();
            std::fs::write(
                root.join(format!("{scope}.spv")),
                binary
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}
