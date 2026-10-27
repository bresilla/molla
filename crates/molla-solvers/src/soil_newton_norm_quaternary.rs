const ORIGINAL: &str = "for(var norm_stride=128u;norm_stride>0u;norm_stride/=2u) {\n        if lane<norm_stride {parts[lane]=max(parts[lane],parts[lane+norm_stride]);}\n        workgroupBarrier();\n    }";
const QUATERNARY: &str = "for(var norm_stride=64u;norm_stride>0u;norm_stride/=4u) {\n        if lane<norm_stride {parts[lane]=max(max(parts[lane],parts[lane+norm_stride]),max(parts[lane+2u*norm_stride],parts[lane+3u*norm_stride]));}\n        workgroupBarrier();\n    }";

pub(crate) fn source(mut input: String) -> String {
    super::replace(&mut input, ORIGINAL, QUATERNARY);
    input
}

#[test]
fn newton_norm_quaternary_preserves_paired_arithmetic_and_budgets() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let original = super::norm_tree::source(super::schur_inverse::source(super::core_source(
        super::source(Solver::cached_contact_source(Solver::cached_delta_source(
            Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
        ))),
    )));
    let candidate = source(original.clone());
    assert_eq!(candidate.replace(QUATERNARY, ORIGINAL), original);
    for (name, shader) in [("control", original), ("candidate", candidate)] {
        let marked = format!(
            "{}{}256\n{shader}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER,
        );
        let binary = crate::soil_newton_protected::compile(&marked, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_QUATERNARY_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), marked).unwrap();
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
