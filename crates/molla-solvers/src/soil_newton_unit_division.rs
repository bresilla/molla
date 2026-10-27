use super::*;

const ORIGINAL: &str = "fn fischer_div(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {\n    let high=rate_round(a.x/b.x);\n    let remainder=rate_add(a,-rate_mul(high,b));\n    return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/value(b),0.0));\n}";
const FAST: &str = "fn fischer_div(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {\n    var result:vec2<f32>;\n    if abs(b.x)==1.0 && b.y==0.0 && all(abs(a)<=vec2<f32>(3.402823e38)) && (a.x==0.0 || abs(a.x)>=1.1754943508222875e-38) && (a.y==0.0 || abs(a.y)>=1.1754943508222875e-38) {\n        result=rate_add(vec2<f32>(a.x*b.x,0.0),vec2<f32>(a.y*b.x,0.0));\n    } else {\n        let high=rate_round(a.x/b.x);\n        let remainder=rate_add(a,-rate_mul(high,b));\n        result=rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/value(b),0.0));\n    }\n    return result;\n}";

pub(crate) fn source(mut source: String) -> Result<String> {
    if source.contains(FAST) {
        return Err(invalid("unit division already transformed"));
    }
    replace_once(&mut source, ORIGINAL, FAST)?;
    Ok(source)
}

#[test]
fn newton_unit_division_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, shader_source_for};
    let program = crate::soil_newton_gpu::distributed::mixed::native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    );
    let protected = super::source(&program).unwrap();
    assert!(uses_compiler(&program).unwrap());
    assert!(!uses_compiler("unmarked source").unwrap());
    assert!(uses_compiler(&format!("{UNIT_DIVISION_MARKER}{program}")).is_err());
    let candidate_program = crate::soil_newton_gpu::distributed::mixed::unit_division_program(
        shader_source_for(NewtonVariant::Fischer),
        true,
    );
    assert!(uses_compiler(&candidate_program).unwrap());
    assert_eq!(candidate_program.matches(UNIT_DIVISION_MARKER).count(), 1);
    let candidate = source(protected.clone()).unwrap();
    assert_eq!(candidate.replace(FAST, ORIGINAL), protected);
    assert_eq!(FAST.matches("return ").count(), 1);
    assert!(source(candidate).is_err());
    compile(&candidate_program, "newton").unwrap();
    for marker in [UNIT_DIVISION_MARKER, NATIVE_DIVISION_MARKER] {
        assert!(
            compile(
                &format!("{marker}{UNIT_DIVISION_MARKER}{program}"),
                "newton"
            )
            .is_err()
        );
    }
}
