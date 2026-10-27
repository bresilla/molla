pub(crate) fn source(mut source: String) -> String {
    let start = source
        .find("fn dense_build_inverse(c:u32)->bool {")
        .unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let dense = source[start..end].to_owned();
    let (assembly, elimination) = dense.split_once("    for (var k=0u;k<6u;k++) {").unwrap();
    let assembly = assembly
        .strip_prefix("fn dense_build_inverse(c:u32)->bool {\n")
        .unwrap();
    let assembled_dense = format!(
        "fn dense_build_inverse(c:u32,assembled:array<array<f32,12>,6>)->bool {{\n    var matrix=assembled;\n    for (var k=0u;k<6u;k++) {{{elimination}"
    );
    source.replace_range(start..end, &assembled_dense);
    let start = source
        .find("fn schur_build_inverse(c:u32)->bool {")
        .unwrap();
    let end = start + source[start..].find("    var ratio:").unwrap();
    source.replace_range(
        start..end,
        r#"fn schur_build_inverse(c:u32,assembled:array<array<f32,12>,6>)->bool {
    var upper:array<f32,18>;var lower:array<f32,3>;var diagonal:array<f32,3>;
    for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) {
        let x=assembled[i][j];
        if i<3u {upper[i*6u+j]=x;}
        else if j==i {diagonal[i-3u]=x;}
        else if j==i-3u {lower[i-3u]=x;}
        else if x!=0.0 {return false;}
    } }
"#,
    );
    let wrapper = "fn build_inverse(c:u32)->bool {\n    if schur_build_inverse(c) {return true;}\n    return dense_build_inverse(c);\n}";
    assert_eq!(source.matches(wrapper).count(), 1);
    source.replace(wrapper, &format!(
        "fn build_inverse(c:u32)->bool {{\n{assembly}    if schur_build_inverse(c,matrix) {{return true;}}\n    return dense_build_inverse(c,matrix);\n}}"
    ))
}

#[test]
fn newton_schur_reuse_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let base = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let control = super::source(base);
    let candidate = source(control.clone());
    assert_eq!(
        candidate
            .matches("var mobility:array<vec3<f32>,3>;")
            .count(),
        1
    );
    assert_eq!(
        control.matches("var mobility:array<vec3<f32>,3>;").count(),
        2
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert!(candidate.contains("if lower[i]!=0.0 {return false;}"));
    let tail = |s: &str| {
        s.split_once("    for (var k=0u;k<6u;k++) {")
            .unwrap()
            .1
            .split_once("\n@compute")
            .unwrap()
            .0
            .to_owned()
    };
    assert_eq!(tail(&candidate), tail(&control));
    crate::soil_newton_protected::compile(
        &format!(
            "{}{}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            candidate
        ),
        "newton",
    )
    .unwrap();
}
