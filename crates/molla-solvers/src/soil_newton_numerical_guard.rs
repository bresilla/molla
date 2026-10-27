use super::*;

const GUARD: &str = include_str!("../shaders/soil_publication_numerical_guard.wgsl");

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    let marker = "        var wall_bits=0u;";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(marker, &format!("{GUARD}{marker}"));
    source
}

#[test]
fn newton_numerical_guard_source_and_compiler() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    for marker in [
        "@binding(",
        "var<workgroup>",
        "var<storage",
        "fn certify_candidate(",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    assert_eq!(candidate.matches(GUARD).count(), 1);
    assert_eq!(candidate.replace(GUARD, ""), control);
    assert!(GUARD.contains("min(lambda,rho*strict.x)"));
    assert!(GUARD.contains("!all(abs(residual)<=vec3<f32>(1e-6))"));
    let words = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(path) = std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD_ARTIFACTS") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(path.join("candidate.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}
