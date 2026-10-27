pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn build_inverse(c:u32)->bool {").unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let mut body = source[start..end].to_owned();
    assert_eq!(
        body.matches("var matrix:array<array<f32,12>,6>;").count(),
        1
    );
    super::replace(
        &mut body,
        "for (var j=0u;j<12u;j++) { matrix[k][j]/=diagonal; }",
        "for (var j=0u;j<12u;j++) { matrix[k][j]/=diagonal; }\n        var pivot_finite=true;for(var j=0u;j<12u;j++) {pivot_finite=pivot_finite && finite(matrix[k][j]);}",
    );
    super::replace(
        &mut body,
        "let ratio=matrix[i][k];\n            for (var j=0u;j<12u;j++)",
        "let ratio=matrix[i][k];\n            if (bitcast<u32>(ratio)&0x7fffffffu)==0u && pivot_finite {\n                var row_finite=true;for(var j=0u;j<12u;j++) {row_finite=row_finite && finite(matrix[i][j]);}\n                if row_finite {\n                    for(var j=0u;j<12u;j++) {if abs(matrix[i][j])<bitcast<f32>(0x00800000u) {matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-ratio,vec2<f32>(matrix[k][j],0.0))));}}\n                    continue;\n                }\n            }\n            for (var j=0u;j<12u;j++)",
    );
    source.replace_range(start..end, &body);
    source
}

#[test]
fn newton_inverse_skip_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let candidate = source(control.clone());
    assert!(candidate.contains("(bitcast<u32>(ratio)&0x7fffffffu)==0u && pivot_finite"));
    assert!(candidate.contains("if row_finite"));
    assert!(candidate.contains("abs(matrix[i][j])<bitcast<f32>(0x00800000u)"));
    assert_eq!(
        control.matches("var<workgroup>").count(),
        candidate.matches("var<workgroup>").count()
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
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_INVERSE_ARTIFACTS") {
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
