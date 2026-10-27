pub(crate) const OFFSET: u64 = 4100 * 8;
pub(crate) const BYTES: u64 = 8 * 8;

pub(crate) fn source(mut source: String) -> String {
    assert!(!source.contains("condensed[4100"));
    super::replace(
        &mut source,
        "fn newton(@builtin(local_invocation_index) id:u32) {\n    lane=id;rate_lane=id;",
        r#"fn newton(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {
        control.stop=bitcast<u32>(condensed[4100u].x);
        condensed[4100u]=vec2<f32>(bitcast<f32>(2u),0.0);
    }
    let slice_mode=workgroupUniformLoad(&control.stop);
    if slice_mode==2u {return;}
    if slice_mode==0u {"#,
    );
    super::replace(
        &mut source,
        "    // Phases: base, Newton trial, damping base, damped trial, final restore.\n    var phase=0u;var step=0u;var merit=0.0;var state_norm=0.0;",
        r#"    } else {
        fischer_smoothing=condensed[4102u].x;
        smoothing_stalled=bitcast<u32>(condensed[4102u].y);
        conditioned_model=bitcast<u32>(condensed[4103u].x)!=0u;
        if lane==0u {
            outer=Outer();outer.releases=bitcast<u32>(condensed[4101u].y);
            for(var i=0u;i<8u;i++) {release_visited[i]=bitcast<u32>(condensed[4104u+i/2u][i%2u]);}
        }
    }
    if lane==0u {control.count=bitcast<u32>(condensed[4101u].x);}
    var phase=0u;var step=workgroupUniformLoad(&control.count);var merit=0.0;var state_norm=0.0;"#,
    );
    let acceptance = "reduce_fischer_smoothing(candidate,merit);continue;";
    assert_eq!(source.matches(acceptance).count(), 2);
    source = source.replace(
        acceptance,
        "reduce_fischer_smoothing(candidate,merit);save_outer_slice(step);return;",
    );
    super::replace(
        &mut source,
        "phase=0u;continue;\n}",
        "save_outer_slice(step);return;\n}",
    );
    source.push_str(r#"
fn save_outer_slice(step:u32) {
    if lane==0u {
        condensed[4100u]=vec2<f32>(bitcast<f32>(1u),0.0);
        condensed[4101u]=vec2<f32>(bitcast<f32>(step),bitcast<f32>(outer.releases));
        condensed[4102u]=vec2<f32>(fischer_smoothing,bitcast<f32>(smoothing_stalled));
        condensed[4103u]=vec2<f32>(bitcast<f32>(select(0u,1u,conditioned_model)),0.0);
        for(var i=0u;i<4u;i++) {
            condensed[4104u+i]=vec2<f32>(bitcast<f32>(release_visited[2u*i]),bitcast<f32>(release_visited[2u*i+1u]));
        }
    }
    storageBarrier();
}
"#);
    source
}

#[test]
fn newton_sliced_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::norm_tree::source(super::schur_inverse::source(super::core_source(
        super::source(Solver::cached_contact_source(Solver::cached_delta_source(
            Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate.matches("save_outer_slice(step);return;").count(),
        3
    );
    assert!(OFFSET + BYTES <= 4160 * 8);
    for name in [
        "evaluate",
        "hybrid_direction",
        "conditioned_release",
        "certify_candidate",
    ] {
        let body = |s: &str| {
            let a = s.find(&format!("fn {name}(")).unwrap();
            let b = a + s[a..].find("\n}").unwrap();
            s[a..b].to_owned()
        };
        assert_eq!(body(&candidate), body(&control));
    }
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{source}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_SLICED_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
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
