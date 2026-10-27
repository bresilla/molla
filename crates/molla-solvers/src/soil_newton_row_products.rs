pub(crate) const SHARED_BYTES: u32 = 6152;

pub(crate) fn source(mut source: String) -> String {
    assert!(!source.contains("row_product_points"));
    for (name, mode, argument) in [
        (
            "apply_operator",
            "0u",
            "vec2<f32>(component(vector[c],j),0.0)",
        ),
        ("apply_direction", "3u", "direction_pair(c,j)"),
    ] {
        let marker = format!("fn {name}() {{");
        assert_eq!(source.matches(&marker).count(), 1);
        let start = source.find(&marker).unwrap();
        let mut depth = 0;
        let end = start
            + source[start..]
                .char_indices()
                .find_map(|(at, ch)| {
                    if ch == '{' {
                        depth += 1;
                    }
                    if ch == '}' {
                        depth -= 1;
                        if depth == 0 {
                            return Some(at + 1);
                        }
                    }
                    None
                })
                .expect("closed product function");
        let original = &source[start..end];
        assert!(original.contains(&format!("project_rates({mode});")));
        assert!(original.contains("point[axis]=sum;"));
        assert!(
            original.contains("sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],rates[i]));")
        );
        assert!(original.contains(&format!(
            "sum=rate_add(sum,fischer_product(blocks[c].direct[6u*i+j],{argument}));"
        )));
        assert!(
            original
                .contains("sum=rate_add(sum,fischer_product(blocks[c].machine[3u*i+j],point[j]));")
        );
        assert!(original.contains("scratch[c]=packed(result);"));
        let replacement = include_str!("../shaders/soil_newton_row_products.wgsl")
            .replace("ROW_PRODUCT", name)
            .replace("PROJECTION_MODE", mode)
            .replace("DIRECT_ARGUMENT", argument);
        source.replace_range(start..end, &replacement);
    }
    source.push_str("\nvar<workgroup> row_product_points:array<vec2<f32>,768>;\nvar<workgroup> row_product_count:u32;\n");
    source
}

pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    let mut input = if candidate { source(input) } else { input };
    input.push_str(r#"
@compute @workgroup_size(256)
fn row_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {status.active_count=contacts[0].boundary.x;}
    for(var c=lane;c<params.contacts;c+=64u) {
        active_indices[c]=(2u*c+7u)%params.contacts;
        var b=Block();var x=Six();var low=Six();
        for(var i=0u;i<6u;i++) {
            let v=f32(i32((c*3u+i*5u)%17u)-8)/16.0;
            let l=f32(i32((c+i)%5u)-2)/268435456.0;
            if i<3u {x.k[i]=v;low.k[i]=l;}else {x.w[i-3u]=v;low.w[i-3u]=l;}
            for(var j=0u;j<6u;j++) {b.direct[i*6u+j]=vec2<f32>(f32(i32((c+i*3u+j)%19u)-9)/32.0,f32(i32((i+j)%3u)-1)/134217728.0);}
            for(var j=0u;j<3u;j++) {b.machine[i*3u+j]=vec2<f32>(f32(i32((c+i+j*7u)%13u)-6)/64.0,f32(i32((i+j)%3u)-1)/134217728.0);}
        }
        blocks[c]=b;vector[c]=x;direction[c]=x;basis[params.iterations*params.contacts+c].z=low;
        scratch[c]=Six(vec4<f32>(17.0),vec4<f32>(19.0));
        states[c]=State(Six(),Six());
    }
    storageBarrier();apply_operator();
    for(var c=lane;c<params.contacts;c+=64u) {states[c].high=scratch[c];scratch[c]=Six(vec4<f32>(17.0),vec4<f32>(19.0));}
    storageBarrier();apply_direction();
    for(var c=lane;c<params.contacts;c+=64u) {states[c].low=scratch[c];}
    storageBarrier();
    if lane==0u {atomicStore(&report.fault,atomicLoad(&status.fault));}
}
"#);
    input
}

#[path = "soil_newton_row_products_tests.rs"]
mod tests;

#[test]
fn newton_row_products_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::schur_inverse::source(super::core_source(super::source(
        Solver::cached_contact_source(Solver::cached_delta_source(Solver::stalled_release_source(
            shader_source_for(NewtonVariant::Fischer),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count() + 2
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    assert_eq!(
        candidate
            .matches("workgroupUniformLoad(&row_product_count)")
            .count(),
        2
    );
    for (name, shader) in [("control", control), ("candidate", candidate)] {
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{shader}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_ROW_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), shader).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                binary
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}
