use super::*;

const DECLARATION: &str = "\nvar<private> publication_inputs_cached:bool=false;\n";
const ENTRY: &str = "fn newton(@builtin(local_invocation_index) id:u32) {";
const CONTACT: &str = "!finite4(node.soil_mass) || !finite4(node.normal_gap) || abs(dot(node.normal_gap.xyz,node.normal_gap.xyz)-1.0)>1e-4 || node.normal_gap.w<0.0\n            || !finite4(node.friction) || node.friction.x<0.0 || any(node.friction.yzw!=vec3<f32>(0.0)) || (node.boundary.x&~63u)!=0u || any(node.boundary.yzw!=vec3<u32>(0u))";
const ROWS: &str = "        for (var i=0u;i<params.width;i++) {\n            let a=rows[c*params.width+i];let b=response[c*params.width+i];\n            if !finite4(a) || !finite4(b) || a.w!=0.0 || b.w!=0.0 {publication_fault(1u,c+1u);}\n        }";
const MASS: &str = "        if !all(abs(free_rates[lane])<=vec2<f32>(3.402823e38)) || !finite(value(free_rates[lane])) {publication_fault(1u,0u);}\n        for (var j=0u;j<params.width;j++) {\n            let m=mass[lane*params.width+j];\n            if !finite(m) || (lane==j && m<=0.0) || abs(m-mass[j*params.width+lane])>1e-6*max(1.0,abs(m)) {publication_fault(1u,0u);}\n        }";
const INPUTS: &str = "    let invalid_inputs=workgroupUniformLoad(&publication_stop)!=0u;";

fn changes() -> Vec<(String, String)> {
    vec![
        (ENTRY.into(), format!("{ENTRY}\n    publication_inputs_cached=false;")),
        ("fn publication_certify(lane:u32) {".into(), "fn publication_certify(lane:u32) {\n    let cached=publication_inputs_cached && params.pad2!=0u;".into()),
        ("if !finite(node.soil_mass.w) || node.soil_mass.w<0.0 {publication_fault(1u,c+1u);continue;}".into(), "if !cached && (!finite(node.soil_mass.w) || node.soil_mass.w<0.0) {publication_fault(1u,c+1u);continue;}".into()),
        (CONTACT.into(), format!("(!cached && ({CONTACT}))")),
        ("if ((node.boundary.x>>(2u*axis))&3u)==3u {publication_fault(1u,c+1u);}".into(), "if !cached && ((node.boundary.x>>(2u*axis))&3u)==3u {publication_fault(1u,c+1u);}".into()),
        (ROWS.into(), format!("        if !cached {{\n{ROWS}\n        }}")),
        (MASS.into(), format!("        if !cached {{\n{MASS}\n        }}")),
        (INPUTS.into(), format!("{INPUTS}\n    if !invalid_inputs {{publication_inputs_cached=true;}}")),
    ]
}

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    for (old, new) in changes() {
        replace(&mut source, &old, &new);
    }
    source.push_str(DECLARATION);
    source
}

#[test]
fn newton_publication_input_cache_source_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, DECLARATION, "");
    for (old, new) in changes().into_iter().rev() {
        replace(&mut restored, &new, &old);
    }
    assert_eq!(restored, control);
    for marker in [
        "@binding(",
        "var<storage",
        "var<workgroup>",
        "rate_mul(",
        "rate_add(",
        "publication_fault(",
        "workgroupBarrier();",
        "storageBarrier();",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    for name in ["mass", "free_rates", "rows", "response", "contacts"] {
        assert!(control.contains(&format!("var<storage,read> {name}:")));
    }
    assert!(candidate.contains("|| !valid_six(x.high) || !valid_six(x.low)"));
    let suffix = |s: &str| s.split_once("    let invalid_rates=").unwrap().1.to_owned();
    assert_eq!(
        suffix(&control),
        suffix(&candidate).strip_suffix(DECLARATION).unwrap()
    );
    let binary = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_INPUT_CACHE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), control).unwrap();
        std::fs::write(root.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
fn newton_publication_input_cache_validation_scope() {
    for mode in [false, true] {
        for immutable_fault in [0u32, 1, 2, 3] {
            let mut cached = false;
            let mut validations = 0;
            let faults = [1u32, 2, 0, 1, 0, 2, 3, 0];
            for dynamic_fault in faults {
                let static_fault = if cached && mode {
                    0
                } else {
                    validations += 1;
                    immutable_fault
                };
                let combined = static_fault | dynamic_fault;
                assert_eq!(combined, immutable_fault | dynamic_fault);
                if combined == 0 {
                    cached = true;
                }
                assert!(!cached || immutable_fault == 0);
            }
            assert_eq!(
                validations,
                if mode && immutable_fault == 0 {
                    3
                } else {
                    faults.len()
                }
            );
        }
    }
}
