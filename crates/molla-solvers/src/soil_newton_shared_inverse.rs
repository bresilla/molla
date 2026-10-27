use super::replace;

pub(crate) fn source(mut source: String) -> String {
    assert_eq!(
        source.matches("var<workgroup> h:array<f32,4160>;").count(),
        1
    );
    assert_eq!(
        source
            .matches("for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}")
            .count(),
        1
    );
    let start = source.find("fn build_inverse(c:u32)->bool {").unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let mut inverse = source[start..end].to_owned();
    replace(
        &mut inverse,
        "fn build_inverse(c:u32)->bool {",
        "fn shared_build_inverse(c:u32)->bool {\n    if lane>=56u {return false;}let base=73u*lane;",
    );
    replace(&mut inverse, "var matrix:array<array<f32,12>,6>;", "");
    replace(
        &mut inverse,
        "if pivot!=k { let row=matrix[k]; matrix[k]=matrix[pivot]; matrix[pivot]=row; }",
        "if pivot!=k {for(var j=0u;j<12u;j++) {let entry=matrix[k][j];matrix[k][j]=matrix[pivot][j];matrix[pivot][j]=entry;}}",
    );
    for (old, new) in [
        ("matrix[i][j+6u]", "h[base+12u*i+j+6u]"),
        ("matrix[i][j]", "h[base+12u*i+j]"),
        ("matrix[k][k]", "h[base+12u*k+k]"),
        ("matrix[i][k]", "h[base+12u*i+k]"),
        ("matrix[k][j]", "h[base+12u*k+j]"),
        ("matrix[pivot][j]", "h[base+12u*pivot+j]"),
    ] {
        assert!(inverse.contains(old), "missing shared matrix access {old}");
        inverse = inverse.replace(old, new);
    }
    assert!(!inverse.contains("matrix["));
    let start = source.find("fn prepare_linear() {").unwrap();
    let end = start + source[start..].find("\nfn trial(").unwrap();
    let mut prepare = source[start..end].to_owned();
    replace(
        &mut prepare,
        "for (var slot=lane;slot<status.active_count;slot+=64u) {",
        "if lane<56u {for (var slot=lane;slot<status.active_count;slot+=56u) {",
    );
    replace(
        &mut prepare,
        "if !build_inverse(c) {",
        "if !shared_build_inverse(c) {",
    );
    replace(
        &mut prepare,
        "    }\n    storageBarrier();\n}",
        "    }}\n    storageBarrier();\n}",
    );
    assert_eq!(prepare.matches("storageBarrier();").count(), 2);
    prepare = prepare.replace("storageBarrier();", "storageBarrier();workgroupBarrier();");
    source.replace_range(start..end, &prepare);
    source.push_str(&inverse);
    source
}

#[test]
fn newton_shared_inverse_source_contract() {
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
    assert!(candidate.contains("slot+=56u"));
    assert!(candidate.contains("if lane>=56u {return false;}let base=73u*lane;"));
    let mut occupied = std::collections::BTreeSet::new();
    for lane in 0..56 {
        for row in 0..6 {
            for col in 0..12 {
                let at = 73 * lane + 12 * row + col;
                assert!(at < 4160);
                assert!(occupied.insert(at));
            }
        }
    }
    assert_eq!(occupied.len(), 56 * 72);
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_SHARED_INVERSE_ARTIFACTS") {
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
