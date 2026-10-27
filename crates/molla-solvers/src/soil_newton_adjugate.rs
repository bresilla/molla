use super::*;

pub(crate) fn source(mut input: String) -> String {
    let start = input.find("fn schur_build_inverse(c:u32)->bool {").unwrap();
    let end = start + input[start..].find("\nfn build_inverse(").unwrap();
    let mut body = input[start..end].to_owned();
    replace(
        &mut body,
        "    for(var k=0u;k<3u;k++) {",
        "    if adjugate_inverse(c,matrix,ratio,diagonal) {return true;}\n    for(var k=0u;k<3u;k++) {",
    );
    input.replace_range(start..end, &body);
    input.push_str(include_str!("../shaders/soil_newton_adjugate.wgsl"));
    input
}

pub(crate) fn baseline(input: String) -> String {
    assert!(cfg!(feature = "stagnant-branch-newton"));
    ExperimentalSoilNewtonGpu::conditioned_program(
        input,
        true,
        256,
        SoilNewtonCapacity {
            dofs: 10,
            contacts: 4096,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    )
}

pub(crate) fn shared_source(input: String) -> String {
    assert!(!input.contains("params.pad0"));
    let mut result = source(input);
    replace(
        &mut result,
        "if adjugate_inverse(c,matrix,ratio,diagonal)",
        "if params.pad0==1u && adjugate_inverse(c,matrix,ratio,diagonal)",
    );
    result
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_adjugate_source_contract() {
    let control = baseline(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone());
    let injected = "    if adjugate_inverse(c,matrix,ratio,diagonal) {return true;}\n";
    assert_eq!(
        candidate
            .strip_suffix(include_str!("../shaders/soil_newton_adjugate.wgsl"))
            .unwrap()
            .replace(injected, ""),
        control
    );
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    assert!(candidate.contains("abs(det)<1e-5"));
    assert!(candidate.contains("error>1e-6"));
    assert!(candidate.contains("return dense_build_inverse(c);"));
    let shared = shared_source(control.clone());
    assert_eq!(shared.matches("params.pad0").count(), 1);
    for (name, input) in [
        ("control", control),
        ("candidate", candidate),
        ("shared", shared),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_ADJUGATE_SHADERS") {
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
