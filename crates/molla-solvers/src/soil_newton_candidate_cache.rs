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
    let start = source.find("fn release_local(").unwrap();
    let end = start + source[start..].find("fn release_candidates(").unwrap();
    let mut prepare = source[start..end].to_owned();
    replace(
        &mut prepare,
        "fn release_local(c:u32,mask:u32)->ReleaseChoice {",
        "fn release_prepare_candidates(c:u32,mask:u32)->ReleaseChoice {\n    h[80u*mask]=0.0;",
    );
    replace(
        &mut prepare,
        "for(var i=0u;i<candidates.count;i++) {release_consider(c,mask,candidates.values[i],other,machine,&best);}",
        r#"let base=80u*mask;h[base]=f32(candidates.count);
    for(var a=0u;a<3u;a++) {
        h[base+1u+a]=other[a];
        for(var j=0u;j<3u;j++) {h[base+4u+3u*j+a]=machine[j][a];}
    }
    for(var i=0u;i<candidates.count;i++) {
        for(var a=0u;a<3u;a++) {h[base+13u+3u*i+a]=candidates.values[i][a];}
    }"#,
    );
    replace(
        &mut source,
        "    if lane<160u {candidate=release_local_candidate(c,lane/20u,lane%20u);}",
        "    if lane<8u {let unused=release_prepare_candidates(c,lane);}\n    workgroupBarrier();\n    if lane<160u {candidate=release_cached_candidate(c,lane/20u,lane%20u);}",
    );
    source.push_str(&prepare);
    source.push_str(include_str!("../shaders/soil_newton_candidate_cache.wgsl"));
    source
}

#[test]
fn newton_candidate_cache_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let original = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let control = super::core_source(super::source(original));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert!(candidate.contains("h[80u*mask]=0.0;"));
    assert!(candidate.contains("if lane<8u {let unused=release_prepare_candidates(c,lane);}"));
    assert!(candidate.contains("b.x==a.x && b.y<a.y"));
    for name in ["release_candidates", "release_consider"] {
        let body = |s: &str| {
            s.split_once(&format!("fn {name}("))
                .unwrap()
                .1
                .split_once("\nfn ")
                .unwrap()
                .0
                .to_owned()
        };
        assert_eq!(body(&control), body(&candidate));
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
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_CANDIDATE_ARTIFACTS") {
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
