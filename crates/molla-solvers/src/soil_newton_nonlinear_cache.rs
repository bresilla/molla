pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn nonlinear_contact(c:u32)").unwrap();
    let end = start + source[start..].find("\nfn fischer_root(").unwrap();
    let mut local = source[start..end].to_owned();
    super::replace(
        &mut local,
        "let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);var separation_pair=vec2<f32>(0.0);",
        "nonlinear_constants(c);let normal_length=nonlinear_cache_length;let trace=nonlinear_cache_trace;var separation_pair=vec2<f32>(0.0);",
    );
    super::replace(
        &mut local,
        "fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length)",
        "nonlinear_cache_normal[axis]",
    );
    super::replace(
        &mut local,
        "trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));",
        "",
    );
    super::replace(
        &mut local,
        "fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace))",
        "nonlinear_cache_rho",
    );
    source.replace_range(start..end, &local);
    source.push_str(include_str!("../shaders/soil_newton_nonlinear_cache.wgsl"));
    source
}

#[test]
fn newton_nonlinear_cache_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let candidate = source(control.clone());
    assert!(candidate.contains("nonlinear_cache_valid && nonlinear_cache_contact==c"));
    assert!(candidate.contains("var<private> nonlinear_cache_valid:bool=false;"));
    assert_eq!(
        control.matches("var<workgroup>").count(),
        candidate.matches("var<workgroup>").count()
    );
    let local = candidate
        .split_once("fn nonlinear_contact(c:u32)")
        .unwrap()
        .1
        .split_once("\nfn fischer_root(")
        .unwrap()
        .0;
    assert!(
        local.find("abs(normal_size-1.0)>1e-4").unwrap()
            < local.find("nonlinear_constants(c);").unwrap()
    );
    assert!(local.contains("if !finite(scale) || scale<=0.0"));
    assert!(!local.contains("trace=rate_add"));
    assert!(candidate.contains("project_rates(1u);"));
    assert!(candidate.contains("DIRECT_LAZY_JACOBIAN:bool=true"));
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_NONLINEAR_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("candidate.wgsl"), &candidate).unwrap();
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}
