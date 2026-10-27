use super::*;

const ISOLATED: &str = "\n@compute @workgroup_size(256)\nfn certification_probe(@builtin(local_invocation_index) id:u32) {\n    lane=id;rate_lane=id;\n    if lane==0u {atomicStore(&report.fault,0u);var count=0u;for(var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {active_indices[count]=c;count++;}}status.active_count=count;}\n    storageBarrier();\n    for(var copy=0u;copy<1u+params.pad0;copy++) {certify_candidate(lane);}\n}\n";

pub(crate) fn isolated_source(input: String) -> String {
    format!("{}{ISOLATED}", complementarity::baseline(input))
}

pub(crate) fn isolated_new(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    copies: u32,
) -> Result<ExperimentalSoilNewtonGpu> {
    assert!(matches!(copies, 1 | 2));
    let solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        cap,
        NewtonVariant::Fischer,
        isolated_source,
        "certification_probe",
        true,
    )?;
    engine
        .queue()
        .unwrap()
        .write_buffer(&solver.scratch[7], 20, &(copies - 1).to_le_bytes());
    Ok(solver)
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_certification_isolated_source_contract() {
    let control = complementarity::baseline(shader_source_for(NewtonVariant::Fischer));
    let input = isolated_source(shader_source_for(NewtonVariant::Fischer));
    assert_eq!(input.strip_suffix(ISOLATED).unwrap(), control);
    assert!(!ISOLATED.contains("evaluate(") && !ISOLATED.contains("prepare_linear("));
    assert!(!ISOLATED.contains("states["));
    let binary = crate::soil_newton_protected::compile(&input, "certification_probe").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_CERTIFICATION_COST_SHADERS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("isolated.wgsl"), input).unwrap();
        std::fs::write(root.join("isolated.spv"), bytemuck::cast_slice(&binary)).unwrap();
    }
}

const ONCE: &str = "if certify_candidate(lane) {return;}";
const REPEATED: &str = "var certified=false;for(var certification_copy=0u;certification_copy<1u+params.pad0;certification_copy++) {certified=certify_candidate(lane);}if certified {return;}";

pub(crate) fn source(mut input: String) -> String {
    replace(&mut input, ONCE, REPEATED);
    input
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_certification_cost_source_contract() {
    if std::env::var_os("MOLLA_NEWTON_MIXED_KIND").is_some() {
        assert_eq!(kind(), "certification-cost-shared");
    }
    let control = complementarity::baseline(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone());
    assert_eq!(candidate.replace(REPEATED, ONCE), control);
    for marker in [
        "certify_candidate(lane)",
        "var<storage",
        "var<workgroup>",
        "var<private>",
        "@binding(",
        "fn ",
    ] {
        assert_eq!(
            control.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    for (name, input) in [("control", control), ("shared", candidate)] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_CERTIFICATION_COST_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), input).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}
