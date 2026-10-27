pub(crate) fn source(mut source: String) -> String {
    super::replace(
        &mut source,
        r#"    parts[lane]=vec2<f32>(largest,select(0.0,1.0,invalid));workgroupBarrier();
    if lane==0u {
        var scale=0.0;var bad=false;
        for (var i=0u;i<64u;i++) {scale=max(scale,parts[i].x);bad=bad || parts[i].y!=0.0;}
        control.scalar=select(scale,-1.0,bad);
    }"#,
        r#"    parts[lane]=vec2<f32>(largest,select(0.0,1.0,invalid));workgroupBarrier();
    for(var norm_stride=128u;norm_stride>0u;norm_stride/=2u) {
        if lane<norm_stride {parts[lane]=max(parts[lane],parts[lane+norm_stride]);}
        workgroupBarrier();
    }
    if lane==0u {control.scalar=select(parts[0].x,-1.0,parts[0].y!=0.0);}"#,
    );
    source
}

#[cfg(test)]
pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-products") {
        return super::publication_products::probe_source(input, candidate);
    }
    if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-zero") {
        return super::zero_publication::probe_source(input, candidate);
    }
    if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-active") {
        return super::active_publication::probe_source(input, candidate);
    }
    if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("private-mgs") {
        return super::private_mgs::probe_source(input, candidate);
    }
    let quaternary = std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("quaternary");
    let mut input = if candidate || quaternary {
        source(input)
    } else {
        input
    };
    if candidate && quaternary {
        input = super::norm_quaternary::source(input);
    }
    input.push_str(
        r#"
@compute @workgroup_size(256)
fn norm_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {status.active_count=contacts[0].boundary.x;}
    for(var c=lane;c<params.contacts;c+=64u) {
        active_indices[c]=c;scratch[c]=Six(contacts[c].soil_mass,contacts[c].normal_gap);
    }
    storageBarrier();let result=scratch_norm();
    if lane==0u {condensed[0]=vec2<f32>(result,0.0);}
}
"#,
    );
    if quaternary {
        super::replace(
            &mut input,
            "if lane==0u {condensed[0]=vec2<f32>(result,0.0);}",
            "let dot=inner_product(0u,0u,0u,0u);if lane==0u {condensed[0]=vec2<f32>(result,dot);}",
        );
    }
    input
}

#[test]
fn newton_norm_tree_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::schur_inverse::source(super::core_source(super::source(
        Solver::cached_contact_source(Solver::cached_delta_source(Solver::stalled_release_source(
            shader_source_for(NewtonVariant::Fischer),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    for (name, shader) in [("control", control), ("candidate", candidate)] {
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{shader}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_NORM_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), shader).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                binary
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

#[path = "soil_newton_norm_tree_tests.rs"]
#[cfg(test)]
mod tests;
