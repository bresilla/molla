#[cfg(test)]
use super::distributed::mixed::{complementarity, kind};
use super::*;

const SERIAL: &str = "        var count=0u;\n        for (var c=0u;c<params.contacts;c++) {\n            let mass=contacts[c].soil_mass.w;\n            if !finite(mass) || mass<0.0 {atomicOr(&status.fault,1u);}\n            if mass>0.0 {active_indices[count]=c;count++;}\n        }\n        status.active_count=count;";
const RESET: &str = "        atomicStore(&status.fault,upstream[0]);report.steps=0u;report.linear_iterations=0u;report.trials=0u;report.damping=0.0;report.diagnostic_limit=0u;";
const PARALLEL: &str = include_str!("../shaders/soil_newton_active_indices.wgsl");

fn additions(shared: bool) -> String {
    let policy = if shared { "params.pad0==1u" } else { "true" };
    format!(
        "\n{PARALLEL}\nfn build_ordered_indices() {{\n    if {policy} {{parallel_active_indices();}}else {{if lane==0u {{\n{SERIAL}\n    }}}}\n}}\n"
    )
}

pub(crate) fn source(mut input: String, shared: bool) -> String {
    assert!(input.contains("// NEWTON_EXECUTION_LANES:256\n"));
    replace(
        &mut input,
        &format!("    if lane==0u {{\n{RESET}\n{SERIAL}\n    }}"),
        &format!(
            "    if lane==0u {{\n{RESET}\n    }}\n    storageBarrier();build_ordered_indices();"
        ),
    );
    input.push_str(&additions(shared));
    input
}

#[cfg(test)]
fn probe_source(input: String, publication: bool) -> String {
    let mut input = source(complementarity::baseline(input), true);
    let initialize = if publication {
        ""
    } else {
        "for(var c=lane;c<max(params.contacts,1u);c+=256u) {active_indices[c]=0xffffffffu;}"
    };
    let output = if publication {
        "certify_candidate(lane);"
    } else {
        "if lane==0u {let count=value(rate_mul(1.0,rate_add(vec2<f32>(f32(status.active_count),0.0),vec2<f32>(0.0))));condensed[0]=vec2<f32>(f32(atomicLoad(&status.fault)),count);}for(var c=lane;c<max(params.contacts,1u);c+=256u) {let index=active_indices[c];condensed[1u+c]=rate_mul(1.0,rate_add(vec2<f32>(select(-1.0,f32(index),index!=0xffffffffu),0.0),vec2<f32>(0.0)));}"
    };
    input.push_str(&format!("\n@compute @workgroup_size(256)\nfn index_probe(@builtin(local_invocation_index) id:u32) {{\n    lane=id;rate_lane=id;{initialize}\n    if lane==0u {{atomicStore(&report.fault,0u);atomicStore(&status.fault,upstream[0]);}}\n    storageBarrier();build_ordered_indices();storageBarrier();{output}\n}}\n"));
    input
}

#[cfg(test)]
pub(crate) fn probe_new(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    candidate: bool,
    publication: bool,
) -> Result<ExperimentalSoilNewtonGpu> {
    let solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        cap,
        NewtonVariant::Fischer,
        |s| probe_source(s, publication),
        "index_probe",
        true,
    )?;
    engine.queue().unwrap().write_buffer(
        &solver.scratch[7],
        20,
        &u32::from(candidate).to_le_bytes(),
    );
    Ok(solver)
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_active_indices_source_contract() {
    if std::env::var_os("MOLLA_NEWTON_MIXED_KIND").is_some() {
        assert_eq!(kind(), "active-indices-shared");
    }
    let control = complementarity::baseline(shader_source_for(NewtonVariant::Fischer));
    #[cfg(all(
        feature = "parallel-indices-newton",
        not(feature = "soa-scratch-newton")
    ))]
    assert_eq!(
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
            shader_source_for(NewtonVariant::Fischer),
            true,
            256,
            SoilNewtonCapacity {
                dofs: 10,
                contacts: 4096,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
        ),
        source(control.clone(), false)
    );
    let shared = source(control.clone(), true);
    let restored = shared.replace(&additions(true), "").replace(
        &format!(
            "    if lane==0u {{\n{RESET}\n    }}\n    storageBarrier();build_ordered_indices();"
        ),
        &format!("    if lane==0u {{\n{RESET}\n{SERIAL}\n    }}"),
    );
    assert_eq!(restored, control);
    for marker in ["var<storage", "var<workgroup>", "var<private>", "@binding("] {
        assert_eq!(
            control.matches(marker).count(),
            shared.matches(marker).count()
        );
    }
    for (name, input, entry) in [
        ("shared", shared, "newton"),
        ("candidate", source(control, false), "newton"),
        (
            "indices",
            probe_source(shader_source_for(NewtonVariant::Fischer), false),
            "index_probe",
        ),
        (
            "publication",
            probe_source(shader_source_for(NewtonVariant::Fischer), true),
            "index_probe",
        ),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, entry).unwrap();
        if let Some(root) = std::env::var_os("MOLLA_ACTIVE_INDICES_SHADERS") {
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

#[test]
fn newton_active_indices_integer_contract() {
    let pack = |v: u32| [(v & 65535) as f32, (v >> 16) as f32];
    let unpack = |v: [f32; 2]| v[0] as u32 | ((v[1] as u32) << 16);
    for v in (0..=131073).chain([0x00ffffff, 0x01000000, 0x7fffffff, 0xfffffffe, u32::MAX]) {
        assert_eq!(unpack(pack(v)), v);
    }
    for n in (0..=4097u32).chain([65535, 65536, 1 << 24, u32::MAX]) {
        let mut end = 0;
        for lane in 0..256u32 {
            let first = (n / 256) * lane + lane.min(n % 256);
            assert_eq!(first, end);
            end = first + n / 256 + u32::from(lane < n % 256);
            assert!(end <= n);
        }
        assert_eq!(end, n);
    }
}

#[cfg(all(test, feature = "stagnant-branch-newton"))]
#[path = "soil_newton_active_indices_tests.rs"]
mod tests;
