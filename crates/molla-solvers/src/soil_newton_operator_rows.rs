use super::*;

fn original(direction: bool) -> String {
    let body = if direction {
        "for(var i=0u;i<6u;i++) {\n            var sum=vec2<f32>(0.0);\n            for(var j=0u;j<6u;j++) {sum=rate_add(sum,fischer_product(soa_direct(c,6u*i+j),direction_pair(c,j)));}\n            for(var j=0u;j<3u;j++) {sum=rate_add(sum,fischer_product(soa_machine(c,3u*i+j),point[j]));}\n            result[i]=value(sum);\n        }"
    } else {
        "for (var i=0u;i<6u;i++) {\n            var sum=vec2<f32>(0.0);\n            for (var j=0u;j<6u;j++) {sum=rate_add(sum,fischer_product(soa_direct(c,6u*i+j),vec2<f32>(component(vector[c],j),0.0)));}\n            for (var j=0u;j<3u;j++) {sum=rate_add(sum,fischer_product(soa_machine(c,3u*i+j),point[j]));}\n            result[i]=value(sum);\n        }"
    };
    body.to_owned()
}

fn rows(direction: bool) -> String {
    let mut text = String::new();
    for i in 0..6 {
        text.push_str(&format!("var operator_row_{i}=vec2<f32>(0.0);\n"));
    }
    text.push_str("for(var j=0u;j<6u;j++) {\n");
    text.push_str(if direction {
        "let input=direction_pair(c,j);\n"
    } else {
        "let input=vec2<f32>(component(vector[c],j),0.0);\n"
    });
    for i in 0..6 {
        text.push_str(&format!("operator_row_{i}=rate_add(operator_row_{i},fischer_product(soa_direct(c,{}u+j),input));\n", 6*i));
    }
    text.push_str("}\nfor(var j=0u;j<3u;j++) {\nlet input=point[j];\n");
    for i in 0..6 {
        text.push_str(&format!("operator_row_{i}=rate_add(operator_row_{i},fischer_product(soa_machine(c,{}u+j),input));\n", 3*i));
    }
    text.push_str("}\n");
    for i in 0..6 {
        text.push_str(&format!("result[{i}u]=value(operator_row_{i});\n"));
    }
    text
}

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    for direction in [false, true] {
        let original = original(direction);
        replace(
            &mut source,
            &original,
            &format!(
                "if params.pad2==0u {{\n{original}\n}} else {{\n{}\n}}",
                rows(direction)
            ),
        );
    }
    source
}

#[test]
fn newton_operator_rows_source_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    for direction in [false, true] {
        let original = original(direction);
        replace(
            &mut restored,
            &format!(
                "if params.pad2==0u {{\n{original}\n}} else {{\n{}\n}}",
                rows(direction)
            ),
            &original,
        );
    }
    assert_eq!(restored, control);
    for token in [
        "@binding(",
        "var<storage",
        "var<workgroup>",
        "var<private>",
        "workgroupBarrier();",
        "storageBarrier();",
    ] {
        assert_eq!(
            candidate.matches(token).count(),
            control.matches(token).count()
        );
    }
    let words = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), control).unwrap();
        std::fs::write(root.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(root.join("candidate.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}

#[test]
fn newton_operator_rows_operand_order() {
    for direction in [false, true] {
        let text = rows(direction);
        assert_eq!(text.matches("let input=").count(), 2);
        assert_eq!(text.matches("fischer_product(").count(), 12);
        assert_eq!(text.matches("rate_add(").count(), 12);
        assert!(!text.contains("rate_mul("));
        let mut baseline = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let mut interleaved = baseline.clone();
        for i in 0..6 {
            for j in 0..6 {
                baseline[i].push((0, 6 * i + j, j));
            }
            for j in 0..3 {
                baseline[i].push((1, 3 * i + j, j));
            }
            assert!(text.contains(&format!("operator_row_{i}=rate_add(operator_row_{i},fischer_product(soa_direct(c,{}u+j),input));",6*i)));
            assert!(text.contains(&format!("operator_row_{i}=rate_add(operator_row_{i},fischer_product(soa_machine(c,{}u+j),input));",3*i)));
        }
        for j in 0..6 {
            for i in 0..6 {
                interleaved[i].push((0, 6 * i + j, j));
            }
        }
        for j in 0..3 {
            for i in 0..6 {
                interleaved[i].push((1, 3 * i + j, j));
            }
        }
        assert_eq!(baseline, interleaved);
    }
}
