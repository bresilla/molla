use super::*;

const INITIALIZE: &str = "\nfn initialize_linear_block(c:u32) {\n    if !direct_linearize {soa_clear_block(c);return;}\n    for(var i=3u;i<6u;i++) {\n        for(var j=0u;j<6u;j++) {\n            if j!=i && j+3u!=i {soa_set_direct(c,6u*i+j,vec2<f32>(0.0));}\n        }\n    }\n    for(var i=9u;i<18u;i++) {soa_set_machine(c,i,vec2<f32>(0.0));}\n}\n";

pub(crate) fn program(input: String, candidate: bool) -> String {
    let mut source = soa_scratch::program(input, true);
    if !candidate {
        return source;
    }
    let start = source.find("fn nonlinear_contact(").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let mut body = source[start..end].to_owned();
    replace(&mut body, "    soa_clear_block(c);\n", "");
    replace(
        &mut body,
        "if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }",
        "if c>=params.contacts || atomicLoad(&status.fault)!=0u { soa_clear_block(c);return; }",
    );
    replace(
        &mut body,
        "if node.soil_mass.w==0.0 { return; }",
        "if node.soil_mass.w==0.0 { soa_clear_block(c);return; }",
    );
    assert_eq!(body.matches("{ atomicOr(&status.fault,").count(), 3);
    body = body.replace(
        "{ atomicOr(&status.fault,",
        "{ soa_clear_block(c);atomicOr(&status.fault,",
    );
    replace(
        &mut body,
        "    a.paired_rho=",
        "    initialize_linear_block(c);\n    a.paired_rho=",
    );
    source.replace_range(start..end, &body);
    source.push_str(INITIALIZE);
    source
}

#[test]
fn newton_block_initialization_write_coverage() {
    let mut direct = [0u32; 36];
    let mut machine = [0u32; 18];
    for i in 3..6 {
        for j in 0..6 {
            if j != i && j + 3 != i {
                direct[6 * i + j] += 1;
            }
        }
    }
    machine[9..].fill(1);
    assert_eq!(direct.iter().sum::<u32>(), 12);
    assert_eq!(machine.iter().sum::<u32>(), 9);
    for i in 0..3 {
        for j in 0..3 {
            direct[6 * i + j] += 1;
            direct[6 * i + 3 + j] += 1;
            machine[3 * i + j] += 1;
        }
        direct[6 * (3 + i) + i] += 1;
        direct[6 * (3 + i) + 3 + i] += 1;
    }
    assert_eq!(direct, [1; 36]);
    assert_eq!(machine, [1; 18]);
}

#[test]
fn newton_block_initialization_source_contract() {
    let control = program(shader_source_for(NewtonVariant::Fischer), false);
    let candidate = program(shader_source_for(NewtonVariant::Fischer), true);
    let mut restored = candidate.strip_suffix(INITIALIZE).unwrap().to_owned();
    replace(&mut restored, "    initialize_linear_block(c);\n", "");
    restored = restored.replace("{ soa_clear_block(c);return; }", "{ return; }");
    restored = restored.replacen(
        "{ soa_clear_block(c);atomicOr(&status.fault,",
        "{ atomicOr(&status.fault,",
        3,
    );
    replace(
        &mut restored,
        "    if c>=max(params.contacts,1u) { return; }\n    if c>=params.contacts",
        "    if c>=max(params.contacts,1u) { return; }\n    soa_clear_block(c);\n    if c>=params.contacts",
    );
    assert_eq!(control, restored);
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let words = crate::soil_newton_protected::compile(&source, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_BLOCK_INITIALIZATION_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
        }
    }
}
