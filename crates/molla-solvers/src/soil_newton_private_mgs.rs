const CALL: &str = "        private_mgs_orthogonalize(j);\n";

fn original() -> &'static str {
    let source = include_str!("../shaders/soil_newton_krylov.wgsl");
    let start = source.find("        for (var orthogonal_pass=0u;").unwrap();
    let end = start
        + source[start..]
            .find("        let magnitude=scratch_norm();")
            .unwrap();
    &source[start..end]
}

pub(crate) fn source(mut input: String) -> String {
    let old = original();
    super::replace(&mut input, old, CALL);
    let mut helper = include_str!("../shaders/soil_newton_private_mgs.wgsl").to_owned();
    assert_eq!(helper.matches("__FALLBACK__").count(), 1);
    helper = helper.replace("__FALLBACK__", old);
    input.push_str(&helper);
    input
}

pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    let mut input = if candidate { source(input) } else { input };
    let body = if candidate { CALL } else { original() };
    input.push_str(&format!(
        r#"
@compute @workgroup_size(256)
fn norm_probe(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;
    if lane==0u {{status.active_count=contacts[0].boundary.x;}}
    let j=contacts[0].boundary.y;
    for(var i=lane;i<4160u;i+=64u) {{h[i]=0.0;}}
    for(var c=lane;c<params.contacts;c+=64u) {{
        active_indices[c]=params.contacts-1u-c;
        let seed=Six(contacts[c].soil_mass,contacts[c].normal_gap);scratch[c]=seed;
        for(var index=0u;index<64u;index++) {{
            let factor=f32(1u+(index*5u+c*3u)%7u)/8.0;
            basis[index*max(params.contacts,1u)+c].v=scaled(seed,factor);
        }}
    }}
    storageBarrier();workgroupBarrier();
{body}
    workgroupBarrier();
    for(var i=lane;i<64u;i+=64u) {{condensed[i]=vec2<f32>(h[i*64u+j],0.0);}}
}}
"#
    ));
    input
}

#[test]
fn newton_private_mgs_preserves_fallback_arithmetic_and_budgets() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::norm_tree::source(super::schur_inverse::source(super::core_source(
        super::source(Solver::cached_contact_source(Solver::cached_delta_source(
            Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
        ))),
    )));
    let candidate = source(control.clone());
    let (body, helper) = candidate.split_once("\nfn private_mgs_product(").unwrap();
    assert_eq!(body.replace(CALL, original()), control);
    assert!(helper.contains(original()));
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    for candidate in [false, true] {
        let probe = probe_source(control.clone(), candidate);
        let marked = format!(
            "{}{}256\n{probe}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        );
        crate::soil_newton_protected::compile(&marked, "norm_probe").unwrap();
    }
    for (name, shader) in [("control", control), ("candidate", candidate)] {
        let marked = format!(
            "{}{}256\n{shader}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        );
        let words = crate::soil_newton_protected::compile(&marked, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_PRIVATE_MGS_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), marked).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

#[path = "soil_newton_private_mgs_tests.rs"]
mod tests;
