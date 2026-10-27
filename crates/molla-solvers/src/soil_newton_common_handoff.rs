use super::*;

fn body_range(source: &str, name: &str) -> std::ops::Range<usize> {
    let a = source.find(&format!("fn {name}(")).unwrap();
    let mut depth = 0;
    let b = source[a..]
        .char_indices()
        .find_map(|(at, ch)| {
            if ch == '{' {
                depth += 1;
            }
            if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(a + at + 1);
                }
            }
            None
        })
        .expect("closed function");
    a..b
}

pub(crate) fn baseline(source: String) -> String {
    stream_block::source(shared_core::source(native_division_program(source, false)))
}

pub(crate) fn source(mut source: String, common: bool) -> String {
    assert!(!source.contains("condensed[4100"));
    let range = body_range(&source, "newton");
    let original = source[range.clone()].to_owned();
    let locals = "    var phase=0u;var step=0u;var merit=0.0;var state_norm=0.0;\n    var alpha=1.0;var backtrack=0u;var damping=0.0;var attempt=0u;";
    let evaluation = "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);let candidate=evaluate();";
    let hybrid = "state_norm=scratch_norm();prepare_linear();hybrid_direction(lane);";
    let tail_start = "        if attempt==0u && conditioned_release() {";
    let mut entry = original.clone();
    if common {
        replace(
            &mut entry,
            "    lane=id;rate_lane=id;",
            "    lane=id;rate_lane=id;\n    if lane==0u {condensed[4100u]=vec2<f32>(0.0);}",
        );
        replace(
            &mut entry,
            hybrid,
            "state_norm=scratch_norm();prepare_linear();solve_bounded(lane,min(8u,params.iterations));\n            if lane==0u {control.stop=atomicLoad(&status.fault);}\n            if workgroupUniformLoad(&control.stop)==8u {save_common_handoff(1u,phase,step,merit,state_norm,alpha,backtrack,damping,attempt);return;}",
        );
        let tail = entry.find(tail_start).unwrap();
        entry.truncate(tail);
        entry.push_str("        save_common_handoff(2u,phase,step,merit,state_norm,alpha,backtrack,damping,attempt);return;\n    }\n}");
    } else {
        let initialization = entry.find(locals).unwrap();
        entry.replace_range(
            ..initialization,
            r#"fn newton(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {control.stop=bitcast<u32>(condensed[4100u].x);}
    var resume_point=workgroupUniformLoad(&control.stop);
    if resume_point==0u {return;}
    fischer_smoothing=condensed[4105u].x;smoothing_stalled=bitcast<u32>(condensed[4105u].y);
    conditioned_model=bitcast<u32>(condensed[4106u].x)!=0u;
    direct_linearize=bitcast<u32>(condensed[4106u].y)!=0u;
    if lane<params.width {rates[lane]=condensed[4120u+lane];}
    if lane==0u {
        outer.stop=bitcast<u32>(condensed[4107u].x);outer.accepted=bitcast<u32>(condensed[4107u].y);
        outer.error=condensed[4108u].x;outer.physical=condensed[4108u].y;
        outer.releases=bitcast<u32>(condensed[4109u].x);
        for(var i=0u;i<8u;i++) {release_visited[i]=bitcast<u32>(condensed[4110u+i/2u][i%2u]);}
    }
    workgroupBarrier();
"#,
        );
        replace(
            &mut entry,
            locals,
            r#"    var phase=bitcast<u32>(condensed[4101u].x);var step=bitcast<u32>(condensed[4101u].y);
    var merit=condensed[4102u].x;var state_norm=condensed[4102u].y;
    var alpha=condensed[4103u].x;var backtrack=bitcast<u32>(condensed[4103u].y);
    var damping=condensed[4104u].x;var attempt=bitcast<u32>(condensed[4104u].y);"#,
        );
        let resumed_evaluation = evaluation.replace("let candidate=", "candidate=");
        replace(
            &mut entry,
            evaluation,
            &format!("var candidate=0.0;if resume_point==0u {{{resumed_evaluation}}}"),
        );
        replace(
            &mut entry,
            "        if phase==0u {\n            merit=candidate;",
            "        if phase==0u {\n            if resume_point==1u {hybrid_handoff_tail(lane);resume_point=0u;} else {\n            merit=candidate;",
        );
        replace(&mut entry, hybrid, &format!("{hybrid}\n            }}"));
        replace(
            &mut entry,
            tail_start,
            &format!("        resume_point=0u;\n{tail_start}"),
        );
        let hybrid_body = source[body_range(&source, "hybrid_direction")].to_owned();
        let tail = hybrid_body
            .replace("fn hybrid_direction(", "fn hybrid_handoff_tail(")
            .replace("    solve_bounded(id,min(8u,params.iterations));\n", "");
        assert!(!tail.contains("solve_bounded(id,min(8u"));
        source.push_str(&format!("\n{tail}\n"));
    }
    source.replace_range(range, &entry);
    if common {
        source.push_str(include_str!("../shaders/soil_newton_common_handoff.wgsl"));
    }
    source
}

pub(crate) fn attach(engine: &Engine, owner: &mut ExperimentalSoilNewtonGpu) -> Result<()> {
    let device = engine.device().ok_or_else(invalid)?;
    let input = source(baseline(shader_source_for(NewtonVariant::Fischer)), true);
    let module = crate::soil_newton_protected::shader(device, &input, "newton")?;
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("common Newton handoff"),
        bind_group_layouts: &[Some(&owner.pipeline.get_bind_group_layout(0))],
        immediate_size: 0,
    });
    owner.prefix_pipeline = Some(device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some("common Newton path"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("newton"),
            compilation_options: Default::default(),
            cache: None,
        },
    ));
    Ok(())
}

#[test]
fn newton_common_handoff_contract() {
    let nested = "fn probe() {\nif true {\n}\n}\nfn next() {}";
    assert_eq!(
        &nested[body_range(nested, "probe")],
        "fn probe() {\nif true {\n}\n}"
    );
    let control = baseline(shader_source_for(NewtonVariant::Fischer));
    for common in [false, true] {
        let candidate = source(control.clone(), common);
        assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
        let binary = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
        for declaration in ["var<storage", "var<workgroup>", "@binding("] {
            assert_eq!(
                control.matches(declaration).count(),
                candidate.matches(declaration).count()
            );
        }
        for name in [
            "hybrid_direction",
            "solve_bounded",
            "conditioned_release",
            "fischer_store_block",
            "certify_candidate",
        ] {
            assert_eq!(
                &control[body_range(&control, name)],
                &candidate[body_range(&candidate, name)]
            );
        }
        if let Some(root) = std::env::var_os("MOLLA_COMMON_HANDOFF_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            let name = if common { "common" } else { "fallback" };
            std::fs::write(root.join(format!("{name}.wgsl")), candidate).unwrap();
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
    assert!(4093 <= 4100 && 4114 <= 4120 && 4120 + 32 <= 4160);
}
