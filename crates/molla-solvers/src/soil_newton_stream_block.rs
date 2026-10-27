use super::replace;
#[cfg(all(test, feature = "retained-local-newton"))]
use super::{
    Engine, ExperimentalSoilNewtonGpu, NewtonVariant, SoilNewtonCapacity, SoilNewtonInputs,
    distributed::mixed::{native_division_program, shared_core},
    shader_source_for,
};

#[cfg(all(test, feature = "retained-local-newton"))]
#[path = "soil_newton_stream_block_tests.rs"]
mod tests;

pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn fischer_block(").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let block = source[start..end]
        .replace(
            "fn fischer_block(local:Local,residual:array<f32,6>)->Block",
            "fn fischer_store_block(c:u32,local:Local)",
        )
        .replace("    var block:Block;block.residual=residual;\n", "")
        .replace("block.", "blocks[c].")
        .replace("    return block;\n", "");
    assert!(!block.contains("->Block"));
    assert!(!block.contains("var block"));
    source.replace_range(start..end, &block);

    let start = source.find("fn nonlinear_contact(").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let mut contact = source[start..end].to_owned();
    assert_eq!(contact.matches("blocks[c]=Block();").count(), 1);
    replace(
        &mut contact,
        "var block:Block; block.residual=fischer_residual(a);",
        "blocks[c].residual=fischer_residual(a);",
    );
    replace(
        &mut contact,
        "if direct_linearize {block=fischer_block(a,block.residual);}",
        "if direct_linearize {fischer_store_block(c,a);}",
    );
    let validation = contact
        .find("    for (var i=0u;i<6u;i++) { if !finite(block.residual[i])")
        .unwrap();
    let mut checks = contact[validation..].to_owned();
    assert_eq!(
        checks
            .matches("atomicOr(&status.fault,2u); return;")
            .count(),
        3
    );
    checks = checks
        .replace("    blocks[c]=block;\n", "")
        .replace("block.", "blocks[c].")
        .replace(
            "atomicOr(&status.fault,2u); return;",
            "blocks[c]=Block();atomicOr(&status.fault,2u); return;",
        );
    contact.replace_range(validation.., &checks);
    assert!(!contact.contains("block."));
    source.replace_range(start..end, &contact);
    source
}

#[test]
#[cfg(feature = "retained-local-newton")]
fn newton_stream_block_contract() {
    let control = shared_core::source(native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    ));
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    let body = |s: &str, name: &str| {
        let start = s.find(&format!("fn {name}(")).unwrap();
        let end = start + s[start..].find("\n}").unwrap() + 2;
        s[start..end].to_owned()
    };
    for name in [
        "fischer_precise",
        "fischer_linear",
        "evaluate",
        "reduce_fischer_smoothing",
        "conditioned_release",
    ] {
        assert_eq!(body(&control, name), body(&candidate, name));
    }
    let restored = body(&candidate, "fischer_store_block")
        .replace(
            "fn fischer_store_block(c:u32,local:Local)",
            "fn fischer_block(local:Local,residual:array<f32,6>)->Block",
        )
        .replace("blocks[c].", "block.")
        .replace(
            "    for(var i=0u;i<3u;i++) {",
            "    var block:Block;block.residual=residual;\n    for(var i=0u;i<3u;i++) {",
        )
        .trim_end_matches('}')
        .to_owned()
        + "    return block;\n}";
    assert_eq!(body(&control, "fischer_block"), restored);
    let contact = body(&candidate, "nonlinear_contact");
    assert_eq!(contact.matches("blocks[c]=Block();").count(), 4);
    assert!(contact.find("blocks[c]=Block();").unwrap() < contact.find("let node=").unwrap());
    assert!(contact.contains("if direct_linearize {fischer_store_block(c,a);}"));
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
}
