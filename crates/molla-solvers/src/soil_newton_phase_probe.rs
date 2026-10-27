pub(crate) fn source(mut source: String, repeats: u32, linear: bool) -> String {
    assert!(matches!(repeats, 2 | 4));
    assert!(source.contains("project_rates(1u);"));
    assert!(source.contains("DIRECT_LAZY_JACOBIAN:bool=true;"));
    if linear {
        super::replace(
            &mut source,
            "storageBarrier();state_norm=scratch_norm();prepare_linear();hybrid_direction(lane);",
            &format!(
                "storageBarrier();state_norm=scratch_norm();\n            for(var phase_repeat=1u;phase_repeat<{repeats}u;phase_repeat++) {{prepare_linear();hybrid_direction(lane);}}\n            prepare_linear();hybrid_direction(lane);"
            ),
        );
    } else {
        super::replace(
            &mut source,
            "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);let candidate=evaluate();",
            &format!(
                "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);\n        for(var phase_repeat=1u;phase_repeat<{repeats}u;phase_repeat++) {{evaluate();}}\n        let candidate=evaluate();"
            ),
        );
    }
    source
}

#[test]
fn newton_phase_probe_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    let evaluate = |s: &str| {
        s.split_once("fn evaluate()->f32 {")
            .unwrap()
            .1
            .split_once("fn prepare_linear()")
            .unwrap()
            .0
            .to_owned()
    };
    for (repeats, linear) in [(2, false), (4, false), (2, true), (4, true)] {
        let candidate = source(control.clone(), repeats, linear);
        assert_eq!(evaluate(&control), evaluate(&candidate));
        assert_eq!(candidate.matches("phase_repeat=1u").count(), 1);
        assert!(candidate.contains("backtrack<16u"));
        assert!(candidate.contains("attempt==8u"));
        assert!(candidate.contains("outer.releases)>=8u"));
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{candidate}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_PHASE_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            let phase = if linear { "linear" } else { "evaluate" };
            std::fs::write(root.join(format!("{phase}-{repeats}.wgsl")), &candidate).unwrap();
            std::fs::write(
                root.join(format!("{phase}-{repeats}.spv")),
                binary
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}
