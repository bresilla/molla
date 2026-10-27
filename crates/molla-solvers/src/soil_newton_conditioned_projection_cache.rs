#[cfg(all(test, feature = "retained-local-newton"))]
use super::{NewtonVariant, distributed::mixed::native_division_program, shader_source_for};

pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn conditioned_prepare(").unwrap();
    let end = start
        + source[start..]
            .find("\nfn retained_conditioned_state(")
            .unwrap();
    let mut prepare = source[start..end].to_owned();
    let marker = "    conditioned_core_width=n;\n";
    assert_eq!(prepare.matches(marker).count(), 1);
    prepare = prepare.replace(
        marker,
        &format!(
            "{marker}{}",
            include_str!("../shaders/soil_newton_conditioned_projection_cache.wgsl")
        ),
    );
    let first = prepare
        .find("                        for(var a=0u;a<3u;a++) {\n")
        .unwrap();
    let last = first
        + prepare[first..]
            .find("                        let row=rows[")
            .unwrap();
    let original = &prepare[first..last];
    let replacement = format!(
        "                        if width<=params.iterations {{
                            let cached=basis[row_index*max(params.contacts,1u)+c].v;
                            for(var a=0u;a<3u;a++) {{projected[a]=vec2<f32>(cached.k[a],cached.w[a]);}}
                        }} else {{
{original}                        }}
"
    );
    prepare.replace_range(first..last, &replacement);
    source.replace_range(start..end, &prepare);
    source
}

#[test]
#[cfg(feature = "retained-local-newton")]
fn newton_conditioned_projection_cache_contract() {
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
    assert_eq!(candidate.matches("if width<=params.iterations").count(), 2);
    for width in 1usize..=32 {
        for iterations in 1usize..=64 {
            for contacts in [0usize, 1, 255, 256, 257, 4096] {
                let stride = contacts.max(1);
                if width <= iterations && contacts != 0 {
                    let last = (width - 1) * stride + contacts - 1;
                    assert!(last < iterations * stride);
                    assert!(last < (iterations + 1) * stride);
                }
            }
        }
    }
    for name in [
        "conditioned_release",
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
