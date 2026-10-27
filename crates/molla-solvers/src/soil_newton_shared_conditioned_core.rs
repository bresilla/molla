#[cfg(all(test, feature = "retained-local-newton"))]
use super::{NewtonVariant, distributed::mixed::native_division_program, shader_source_for};

pub(crate) fn source(mut source: String) -> String {
    let marker = "fn conditioned_cache() {";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(
        marker,
        &format!("{marker}\n    conditioned_shared_ready=false;"),
    );
    let marker = "    conditioned_core_width=n;\n";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(
        marker,
        &format!("{marker}    conditioned_shared_build();\n"),
    );
    let start = source.find("fn conditioned_prepare(").unwrap();
    let first = start
        + source[start..]
            .find("                sum=vec2<f32>(select(0.0,1.0,column==row_index),0.0);")
            .unwrap();
    let end = first
        + source[first..]
            .find("            } else if column<n {")
            .unwrap();
    let original = &source[first..end];
    let cached = format!(
        "                if conditioned_shared_ready {{
                    let base_column=select(column,width,column==n);
                    sum=basis[row_index*(width+1u)+base_column].v.k.xy;
                    if condensed[4160u+54u*chosen+53u].x==1.0 {{
                        sum=conditioned_shared_accumulate(sum,chosen,row_index,base_column,-1.0);
                    }}
                }}
                if !conditioned_shared_ready || !all(abs(sum)<=vec2<f32>(3.402823e38)) {{
{original}                }}
"
    );
    source.replace_range(first..end, &cached);
    source.push_str(include_str!(
        "../shaders/soil_newton_shared_conditioned_core.wgsl"
    ));
    source
}

#[test]
#[cfg(feature = "retained-local-newton")]
fn newton_shared_conditioned_core_contract() {
    let control = native_division_program(shader_source_for(NewtonVariant::Fischer), false);
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    assert!(candidate.contains("fn conditioned_cache() {\n    conditioned_shared_ready=false;"));
    for width in 1usize..=32 {
        for iterations in 1usize..=64 {
            for contacts in [0usize, 1, 2, 31, 32, 255, 256, 257, 4096] {
                let cells = width * (width + 1);
                let reserved = iterations * contacts.max(1);
                if cells <= reserved {
                    assert!(cells - 1 < reserved);
                }
            }
        }
    }
    for name in [
        "conditioned_factor",
        "conditioned_state",
        "krylov_cycle",
        "least_squares",
    ] {
        let body = |s: &str| {
            let start = s.find(&format!("fn {name}(")).unwrap();
            let end = start + s[start..].find("\n}").unwrap() + 2;
            s[start..end].to_owned()
        };
        assert_eq!(body(&control), body(&candidate));
    }
}
