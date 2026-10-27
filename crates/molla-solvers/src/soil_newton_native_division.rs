use super::*;

pub(crate) fn source(mut source: String) -> Result<String> {
    let old = "fn fischer_div(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {\n    let high=rate_round(a.x/b.x);\n    let remainder=rate_add(a,-rate_mul(high,b));\n    return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/value(b),0.0));\n}";
    replace_once(
        &mut source,
        old,
        "fn fischer_div(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {\n    let quotient=(f64(a.x)+f64(a.y))/(f64(b.x)+f64(b.y));\n    let high=f32(quotient);\n    return vec2<f32>(high,f32(quotient-f64(high)));\n}",
    )?;
    Ok(source)
}

#[test]
fn newton_native_division_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, shader_source_for};
    let program = crate::soil_newton_gpu::distributed::mixed::native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    );
    let protected = super::source(&program).unwrap();
    let changed = source(protected.clone()).unwrap();
    let strip = |s: &str| {
        let start = s.find("fn fischer_div(").unwrap();
        let end = start + s[start..].find("\n}").unwrap() + 2;
        format!("{}{}", &s[..start], &s[end..])
    };
    assert_eq!(strip(&protected), strip(&changed));
    assert!(source(changed).is_err());
    let control = compile(&program, "newton").unwrap();
    let candidate = compile(&format!("{NATIVE_DIVISION_MARKER}{program}"), "newton").unwrap();
    let f64_type = |words: &[u32]| {
        instructions(words)
            .unwrap()
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 22 && op.len() == 3 && op[2] == 64)
    };
    assert!(!f64_type(&control));
    assert!(f64_type(&candidate));
    assert!(
        compile(
            &format!("{NATIVE_DIVISION_MARKER}{NATIVE_DIVISION_MARKER}{program}"),
            "newton"
        )
        .is_err()
    );
    if let Some(root) = std::env::var_os("MOLLA_NATIVE_DIVISION_SHADERS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        for (name, words) in [("control", control), ("candidate", candidate)] {
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
        }
        std::fs::write(root.join("original.wgsl"), program).unwrap();
        std::fs::write(root.join("candidate.wgsl"), source(protected).unwrap()).unwrap();
    }
}
