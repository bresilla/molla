use super::*;

pub(crate) fn source(mut source: String) -> String {
    let mut assembly = None;
    for name in ["schur_build_inverse", "dense_build_inverse"] {
        let start = source
            .find(&format!("fn {name}(c:u32)->bool {{\n"))
            .unwrap();
        let start = start + source[start..].find("    var mobility:").unwrap();
        let expected = "    var mobility:array<vec3<f32>,3>;\n    for (var axis=0u;axis<3u;axis++) { for (var j=0u;j<3u;j++) {\n        var sum=vec2<f32>(0.0);\n        for (var i=0u;i<params.width;i++) { sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][j],0.0))); }\n        mobility[axis][j]=value(sum);\n    } }\n";
        let end = start + expected.len();
        assert_eq!(&source[start..end], expected);
        if let Some(previous) = &assembly {
            assert_eq!(previous, expected);
        } else {
            assembly = Some(expected.to_owned());
        }
        source.replace_range(start..end, "    let mobility=cached_mobility(c);\n");
    }
    source.push_str(&format!(
        "\nfn compute_mobility(c:u32)->array<vec3<f32>,3> {{\n{}    return mobility;\n}}\n",
        assembly.unwrap()
    ));
    source.push_str(
        r#"
var<private> mobility_valid:bool=false;
var<private> mobility_contact:u32;
var<private> mobility_value:array<vec3<f32>,3>;
fn cached_mobility(c:u32)->array<vec3<f32>,3> {
    if !mobility_valid || mobility_contact!=c {
        mobility_value=compute_mobility(c);mobility_contact=c;mobility_valid=true;
    }
    return mobility_value;
}
"#,
    );
    source
}

#[test]
fn newton_mobility_cache_contract() {
    let control = stream_block::source(shared_core::source(native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    )));
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    let body = |s: &str, name: &str| {
        let a = s.find(&format!("fn {name}(")).unwrap();
        let b = a + s[a..].find("\n}").unwrap() + 2;
        s[a..b].to_owned()
    };
    for name in [
        "fischer_linear",
        "fischer_store_block",
        "prepare_linear",
        "hybrid_direction",
        "build_inverse",
        "conditioned_release",
    ] {
        assert_eq!(body(&control, name), body(&candidate, name));
    }
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    assert_eq!(
        candidate
            .matches("let mobility=cached_mobility(c);")
            .count(),
        2
    );
    assert!(candidate.contains("var<private> mobility_valid:bool=false;"));
    assert!(candidate.contains("!mobility_valid || mobility_contact!=c"));
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_RESOURCE_EXPORT") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        let words = crate::soil_newton_protected::compile(&control, "newton").unwrap();
        std::fs::write(root.join("method26.wgsl"), &control).unwrap();
        std::fs::write(
            root.join("method26.spv"),
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}
