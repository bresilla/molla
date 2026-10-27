use super::replace;

fn rewrite(source: &mut String, name: &str, edit: impl FnOnce(&mut String)) {
    let marker = format!("fn {name}(");
    assert_eq!(source.matches(&marker).count(), 1);
    let start = source.find(&marker).unwrap();
    let end = start + source[start..].find("\nfn ").unwrap();
    let mut body = source[start..end].to_owned();
    edit(&mut body);
    source.replace_range(start..end, &body);
}

pub(crate) fn source(mut source: String) -> String {
    let linear = include_str!("../shaders/soil_newton_fischer_linear.wgsl");
    let block = &linear[linear.find("fn fischer_block(").unwrap()..];
    assert_eq!(
        source.matches(block).count(),
        1,
        "Fischer zero-pattern construction contract"
    );
    for (name, argument, valid) in [
        (
            "apply_operator",
            "vec2<f32>(component(vector[c],j),0.0)",
            "valid_six(vector[c])",
        ),
        (
            "apply_direction",
            "direction_pair(c,j)",
            "sparse_direction_finite(c)",
        ),
    ] {
        rewrite(&mut source, name, |body| {
            replace(
                body,
                "var result:array<f32,6>;",
                &format!(
                    "var sparse={valid};\n        for(var a=0u;a<3u;a++) {{sparse=sparse && all(abs(point[a])<=vec2<f32>(3.402823e38));}}\n        var result:array<f32,6>;"
                ),
            );
            let direct =
                format!("sum=rate_add(sum,fischer_product(blocks[c].direct[6u*i+j],{argument}));");
            replace(
                body,
                &direct,
                &format!("if !sparse || i<3u || j==i || j+3u==i {{{direct}}}"),
            );
            let machine = "sum=rate_add(sum,fischer_product(blocks[c].machine[3u*i+j],point[j]));";
            replace(body, machine, &format!("if !sparse || i<3u {{{machine}}}"));
        });
    }
    rewrite(&mut source, "apply_transpose", |body| {
        replace(
            body,
            "let c=active_indices[slot];var point=Six();",
            "let c=active_indices[slot];let sparse=valid_six(vector[c]);var point=Six();",
        );
        replace(
            body,
            "let c=active_indices[slot];var result:array<f32,6>;",
            "let c=active_indices[slot];let sparse=valid_six(vector[c]);var result:array<f32,6>;",
        );
        let machine = "sum=rate_add(sum,fischer_product(blocks[c].machine[i*3u+axis],vec2<f32>(component(vector[c],i),0.0)));";
        replace(body, machine, &format!("if !sparse || i<3u {{{machine}}}"));
        let direct = "sum=rate_add(sum,fischer_product(blocks[c].direct[j*6u+i],vec2<f32>(component(vector[c],j),0.0)));";
        replace(
            body,
            direct,
            &format!("if !sparse || j<3u || j==3u+i%3u {{{direct}}}"),
        );
    });
    source.push_str("\nfn sparse_direction_finite(c:u32)->bool {\n    for(var i=0u;i<6u;i++) {if !all(abs(direction_pair(c,i))<=vec2<f32>(3.402823e38)) {return false;}}\n    return true;\n}\n");
    source
}

#[test]
fn newton_sparse_products_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        control.matches("var<workgroup>").count(),
        candidate.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate
            .matches("if !sparse || i<3u || j==i || j+3u==i")
            .count(),
        2
    );
    assert_eq!(
        candidate
            .matches("if !sparse || j<3u || j==3u+i%3u")
            .count(),
        1
    );
    for i in 0..6 {
        for j in 0..6 {
            assert_eq!(
                i < 3 || j == i || j + 3 == i,
                i < 3 || (i >= 3 && (j == i - 3 || j == i))
            );
            assert_eq!(
                j < 3 || j == 3 + i % 3,
                j < 3 || (j >= 3 && (i == j - 3 || i == j))
            );
        }
    }
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_SPARSE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("candidate.wgsl"), &candidate).unwrap();
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}
