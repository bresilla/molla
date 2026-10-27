#[path = "soil_newton_schur_reuse.rs"]
#[cfg(test)]
pub(crate) mod reuse;

pub(crate) fn source(mut source: String) -> String {
    assert!(
        source.contains("fn fischer_block("),
        "Schur inverse requires Fischer blocks"
    );
    let candidate = include_str!("../shaders/soil_newton_schur_inverse.wgsl");
    let start = source.find("fn build_inverse(c:u32)->bool {").unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let original = &source[start..end];
    let mobility = original
        .split_once("    var mobility:")
        .unwrap()
        .1
        .split_once("    var matrix:")
        .unwrap()
        .0;
    assert!(
        candidate.contains(mobility),
        "preconditioner mobility assembly changed"
    );
    let sum = original
        .split_once("        var sum=blocks[c].direct")
        .unwrap()
        .1
        .split_once("        matrix[i][j]=value(sum);")
        .unwrap()
        .0;
    assert!(
        candidate.contains(sum),
        "preconditioner coefficient assembly changed"
    );
    super::replace(
        &mut source,
        "fn build_inverse(c:u32)->bool {",
        "fn dense_build_inverse(c:u32)->bool {",
    );
    source.push_str(candidate);
    source
}

#[test]
fn newton_schur_inverse_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let candidate = source(control.clone());
    let appended = include_str!("../shaders/soil_newton_schur_inverse.wgsl");
    assert!(appended.contains("if lower[i]!=0.0 {return false;}"));
    assert!(appended.find("if lower[i]!=0.0").unwrap() < appended.find("ratio[i]=").unwrap());
    assert_eq!(
        candidate.strip_suffix(appended).unwrap().replace(
            "fn dense_build_inverse(c:u32)->bool {",
            "fn build_inverse(c:u32)->bool {"
        ),
        control
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_SCHUR_INVERSE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("schur-inverse.wgsl"), &candidate).unwrap();
        std::fs::write(
            root.join("schur-inverse.spv"),
            binary
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
#[should_panic(expected = "Schur inverse requires Fischer blocks")]
fn newton_schur_inverse_rejects_other_models() {
    source(String::new());
}

#[cfg(test)]
pub(crate) fn probe_source(mut input: String, candidate: bool) -> String {
    if candidate {
        input = source(input);
        if std::env::var_os("MOLLA_NEWTON_SCHUR_REUSE_COMPARE").is_some() {
            input = reuse::source(input);
            input = input.replace(
                "if schur_build_inverse(c,matrix) {return true;}",
                "probe_reduced=schur_build_inverse(c,matrix);if probe_reduced {return true;}",
            );
        } else {
            super::replace(
                &mut input,
                "if schur_build_inverse(c) {return true;}",
                "probe_reduced=schur_build_inverse(c);if probe_reduced {return true;}",
            );
        }
    }
    input.push_str(r#"
var<private> probe_reduced:bool=false;
@compute @workgroup_size(256)
fn inverse_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    for(var c=lane;c<params.contacts;c+=64u) {
        let mode=c%7u;var b=Block();
        if mode!=5u {
            for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {
                b.direct[i*6u+j]=vec2<f32>(select(0.125,4.0+f32(c%5u)*0.125,i==j),0.0);
                b.direct[i*6u+3u+j]=vec2<f32>(select(0.0625,0.25,i==j),0.0);
                b.machine[i*3u+j]=vec2<f32>(select(0.03125,0.125,i==j),0.0);
            }}
            for(var i=0u;i<3u;i++) {
                b.direct[(3u+i)*6u+i]=vec2<f32>(select(0.25,0.0,mode==1u || (mode==2u && i==0u)),0.0);
                b.direct[(3u+i)*6u+3u+i]=vec2<f32>(1.0+f32(i)*0.125,0.0);
            }
            if mode==3u {b.direct[19u]=vec2<f32>(0.125,0.0);}
            if mode==4u {b.direct[21u]=vec2<f32>(0.0);}
            if mode==6u {b.direct[0u]=vec2<f32>(bitcast<f32>(0x7fc00000u),0.0);}
        }
        blocks[c]=b;inverses[c]=Inverse();probe_reduced=false;
        let accepted=build_inverse(c);
        condensed[c]=vec2<f32>(select(0.0,1.0,accepted),select(0.0,1.0,probe_reduced));
    }
    storageBarrier();
    if lane==0u {atomicStore(&report.fault,atomicLoad(&status.fault));}
}
"#);
    format!("{}{}", crate::soil_newton_protected::VOLATILE_MARKER, input)
}

#[path = "soil_newton_schur_inverse_tests.rs"]
#[cfg(test)]
mod tests;
