use super::*;

fn replace(input: &mut String, old: &str, new: &str) {
    assert_eq!(
        input.matches(old).count(),
        1,
        "slot constants source contract: {old}"
    );
    *input = input.replace(old, new);
}

pub(crate) fn baseline(input: String) -> String {
    stream_block::source(shared_core::source(native_division_program(input, false)))
}

pub(crate) fn source(mut input: String) -> String {
    replace(
        &mut input,
        "fn nonlinear_contact(c:u32) {",
        "fn nonlinear_contact(c:u32,slot:u32) {",
    );
    replace(
        &mut input,
        "nonlinear_contact(c);",
        "nonlinear_contact(c,slot);",
    );
    let start = input.find("fn nonlinear_contact(").unwrap();
    let end = start + input[start..].find("\nfn fischer_root(").unwrap();
    let mut local = input[start..end].to_owned();
    replace(
        &mut local,
        "let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);var separation_pair=vec2<f32>(0.0);",
        "let cached=slot_constants_enabled();var normal_length:vec2<f32>;var trace=vec2<f32>(0.0);if cached {normal_length=slot_constant(slot,0u);trace=slot_constant(slot,1u);} else {normal_length=fischer_length(node.normal_gap.xyz);}var separation_pair=vec2<f32>(0.0);",
    );
    replace(
        &mut local,
        "a.paired_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);",
        "if cached {a.paired_normal[axis]=slot_constant(slot,2u+axis);} else {a.paired_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);}",
    );
    replace(
        &mut local,
        "trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));",
        "if !cached {trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));}",
    );
    replace(
        &mut local,
        "a.paired_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace));",
        "if cached {a.paired_rho=slot_constant(slot,5u);} else {a.paired_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace));}",
    );
    input.replace_range(start..end, &local);
    let phases = "    // Phases: base, Newton trial, damping base, damped trial, final restore.";
    replace(
        &mut input,
        phases,
        &format!(
            "    if slot_constants_enabled() {{\n        for(var slot=lane;slot<status.active_count;slot+=64u) {{build_slot_constants(slot);}}\n    }}\n    storageBarrier();\n{phases}"
        ),
    );
    input.push_str(include_str!("../shaders/soil_newton_slot_constants.wgsl"));
    input
}

#[test]
fn newton_slot_constants_contract() {
    let control = baseline(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    for declaration in ["var<storage", "var<workgroup>", "var<private>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    for name in [
        "fischer_residual",
        "fischer_store_block",
        "hybrid_direction",
        "conditioned_release",
        "prepare_linear",
    ] {
        let body = |s: &str| {
            let a = s.find(&format!("fn {name}(")).unwrap();
            let b = a + s[a..].find("\n}").unwrap() + 2;
            s[a..b].to_owned()
        };
        assert_eq!(body(&control), body(&candidate));
    }
    for capacity in [0_u32, 1, 12, 13, 63, 64, 255, 256, 257, 4096] {
        for count in 0..=capacity {
            if count > 0 && count <= capacity / 13 {
                let mut words = vec![u32::MAX; capacity as usize];
                for i in 0..count {
                    words[i as usize] = capacity - i - 1;
                }
                let prefix = words[..count as usize].to_vec();
                for slot in 0..count {
                    for i in 0..12 {
                        words[(count + 12 * slot + i) as usize] = slot;
                    }
                }
                assert_eq!(&words[..count as usize], &prefix);
                assert!(u64::from(count) * 13 <= u64::from(capacity));
            }
        }
    }
    for capacity in [u32::MAX, u32::MAX - 1] {
        let count = capacity / 13;
        assert!(u64::from(count) + 12 * u64::from(count) <= u64::from(capacity));
    }
    let local = candidate
        .split_once("fn nonlinear_contact(")
        .unwrap()
        .1
        .split_once("\nfn fischer_root(")
        .unwrap()
        .0;
    assert!(
        local.find("abs(normal_size-1.0)>1e-4").unwrap()
            < local.find("slot_constants_enabled()").unwrap()
    );
    assert!(local.contains("if !finite(scale) || scale<=0.0"));
    if let Some(root) = std::env::var_os("MOLLA_SLOT_CONSTANTS_SHADERS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        for (name, source) in [("control", control), ("candidate", candidate)] {
            let binary = crate::soil_newton_protected::compile(&source, "newton").unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                binary
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}
