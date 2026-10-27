pub(crate) fn source(mut source: String) -> String {
    assert!(
        source.contains("fn fischer_block("),
        "Schur contact requires Fischer blocks"
    );
    super::replace(
        &mut source,
        "fn direct_contact(c:u32)->bool {",
        "fn dense_direct_contact(c:u32)->bool {",
    );
    source.push_str(include_str!("../shaders/soil_newton_schur_contact.wgsl"));
    source
}

#[test]
fn newton_schur_contact_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let candidate = source(control.clone());
    let appended = include_str!("../shaders/soil_newton_schur_contact.wgsl");
    assert_eq!(
        candidate.strip_suffix(appended).unwrap().replace(
            "fn dense_direct_contact(c:u32)->bool {",
            "fn direct_contact(c:u32)->bool {"
        ),
        control
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert!(appended.contains("return dense_direct_contact(c);"));
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_SCHUR_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("schur-contact.wgsl"), &candidate).unwrap();
        std::fs::write(
            root.join("schur-contact.spv"),
            binary
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
#[should_panic(expected = "Schur contact requires Fischer blocks")]
fn newton_schur_contact_rejects_other_models() {
    source(String::new());
}
