use super::*;

const NORMALIZE: &str = "        for(var j=0u;j<9u;j++) {matrix[k][j]/=divisor;}";
const ELIMINATE: &str = "            for(var j=0u;j<9u;j++) {matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-r,vec2<f32>(matrix[k][j],0.0))));}";

pub(crate) fn source(mut input: String, shared: bool) -> String {
    let start = input.find("fn schur_build_inverse(c:u32)->bool {").unwrap();
    let end = start + input[start..].find("\nfn build_inverse(").unwrap();
    let mut body = input[start..end].to_owned();
    let policy = if shared { "params.pad0==1u" } else { "true" };
    replace(
        &mut body,
        NORMALIZE,
        &format!(
            "        let suffix={policy};\n        let first=select(0u,k+1u,suffix);\n        if suffix {{matrix[k][k]=1.0;}}\n        for(var j=first;j<9u;j++) {{matrix[k][j]/=divisor;}}"
        ),
    );
    replace(
        &mut body,
        ELIMINATE,
        "            for(var j=first;j<9u;j++) {matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-r,vec2<f32>(matrix[k][j],0.0))));}\n            if suffix {matrix[i][k]=0.0;}",
    );
    input.replace_range(start..end, &body);
    input
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_pivot_suffix_source_contract() {
    let control = adjugate::baseline(shader_source_for(NewtonVariant::Fischer));
    assert!(!control.contains("params.pad0"));
    let candidate = source(control.clone(), false);
    let shared = source(control.clone(), true);
    assert_eq!(shared.matches("params.pad0").count(), 1);
    assert_eq!(candidate, shared.replace("params.pad0==1u", "true"));
    assert_eq!(
        shared
            .replace("        let suffix=params.pad0==1u;\n        let first=select(0u,k+1u,suffix);\n        if suffix {matrix[k][k]=1.0;}\n        for(var j=first;j<9u;j++) {matrix[k][j]/=divisor;}", NORMALIZE)
            .replace("            for(var j=first;j<9u;j++) {matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-r,vec2<f32>(matrix[k][j],0.0))));}\n            if suffix {matrix[i][k]=0.0;}", ELIMINATE),
        control
    );
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    for (name, input) in [
        ("control", control),
        ("candidate", candidate),
        ("shared", shared),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_PIVOT_SUFFIX_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), &input).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}
