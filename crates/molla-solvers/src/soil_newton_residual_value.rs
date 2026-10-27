use super::*;

pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn fischer_precise(").unwrap();
    let end = start + source[start..].find("    var da=").unwrap();
    let mut value = source[start..end].to_owned();
    let header = "fn fischer_precise(a:vec2<f32>,b:vec2<f32>)->array<vec2<f32>,3>";
    assert!(value.starts_with(header));
    value = value.replacen(
        header,
        "fn fischer_value(a:vec2<f32>,b:vec2<f32>)->vec2<f32>",
        1,
    );
    let zero = "return array<vec2<f32>,3>(vec2<f32>(0.0),vec2<f32>(0.2928932188,-1.210161704e-8),vec2<f32>(0.2928932188,-1.210161704e-8));";
    assert_eq!(value.matches(zero).count(), 1);
    value = value.replace(zero, "return vec2<f32>(0.0);");
    value.push_str("    return rate_mul(scale,residual);\n}\n");
    let start = source.find("fn fischer_residual(").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let residual = &source[start..end];
    assert_eq!(residual.matches("fischer_precise(").count(), 2);
    assert_eq!(residual.matches("))[0]").count(), 2);
    let residual = residual
        .replace("fischer_precise(", "fischer_value(")
        .replace("))[0]", "))");
    source.replace_range(start..end, &residual);
    source.push_str(&value);
    source
}

#[test]
fn newton_residual_value_contract() {
    let control = shared_core::source(native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    ));
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    for name in [
        "fischer_precise",
        "fischer_linear",
        "fischer_block",
        "reduce_fischer_smoothing",
        "conditioned_release",
    ] {
        let body = |s: &str| {
            let start = s.find(&format!("fn {name}(")).unwrap();
            let end = start + s[start..].find("\n}").unwrap() + 2;
            s[start..end].to_owned()
        };
        assert_eq!(body(&control), body(&candidate));
    }
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    assert_eq!(candidate.matches("fn fischer_value(").count(), 1);
    assert!(candidate.contains("fischer_smoothing*=0.1;"));
}
