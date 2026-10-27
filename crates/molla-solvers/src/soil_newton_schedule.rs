pub(crate) fn source(mut source: String, prefix: u32) -> String {
    assert!(matches!(prefix, 0 | 4), "unsupported experimental prefix");
    let old = include_str!("../shaders/soil_newton_hybrid.wgsl");
    let new = if prefix == 0 {
        super::replace(
            &mut source,
            "state_norm=scratch_norm();prepare_linear();hybrid_direction(lane);",
            "state_norm=scratch_norm();prepare_condensed();hybrid_direction(lane);",
        );
        "fn hybrid_direction(id:u32) {\n    if direct_direction() {return;}\n    prepare_linear();solve_bounded(id,params.iterations);\n}\n".to_owned()
    } else {
        assert_eq!(old.matches("min(8u,params.iterations)").count(), 1);
        old.replace("min(8u,params.iterations)", "min(4u,params.iterations)")
    };
    super::replace(&mut source, old, &new);
    source
}

#[test]
fn newton_schedule_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::core_source(super::source(Solver::cached_contact_source(
        Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
            NewtonVariant::Fischer,
        ))),
    )));
    for prefix in [0, 4] {
        let candidate = source(control.clone(), prefix);
        assert_eq!(
            candidate.matches("var<workgroup>").count(),
            control.matches("var<workgroup>").count()
        );
        for unchanged in [
            "fn prepare_linear()",
            "fn prepare_condensed()",
            "fn direct_direction()",
            "fn solve_bounded(",
        ] {
            let body = |s: &str| {
                s[s.find(unchanged).unwrap()..]
                    .split_once("\n}")
                    .unwrap()
                    .0
                    .to_owned()
            };
            assert_eq!(body(&candidate), body(&control));
        }
        assert!(candidate.contains("backtrack<16u"));
        assert!(candidate.contains("attempt==8u"));
        assert!(candidate.contains("outer.releases)>=8u"));
        if prefix == 0 {
            assert!(!candidate.contains("prepare_linear();hybrid_direction(lane);"));
            assert!(candidate.contains("prepare_condensed();hybrid_direction(lane);"));
        } else {
            assert!(candidate.contains("params.iterations-used"));
        }
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{candidate}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_SCHEDULE_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("prefix-{prefix}.wgsl")), &candidate).unwrap();
            std::fs::write(
                root.join(format!("prefix-{prefix}.spv")),
                binary
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

#[test]
#[should_panic(expected = "unsupported experimental prefix")]
fn newton_schedule_rejects_unknown_prefix() {
    source(String::new(), 64);
}
