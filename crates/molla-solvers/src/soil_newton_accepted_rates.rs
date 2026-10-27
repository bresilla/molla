use super::*;

const DECLARATION: &str = "\nvar<private> accepted_rates_ready:bool=false;\n";
const EVALUATE: &str = "fn evaluate()->f32 {\n    project_rates(1u);";
const REUSE: &str = "fn evaluate()->f32 {\n    if !accepted_rates_ready {project_rates(1u);}\n    accepted_rates_ready=false;";
const ACCEPT: &str = "reduce_fischer_smoothing(candidate,merit);continue;";

pub(crate) fn source(mut input: String, shared: bool) -> String {
    assert_eq!(input.matches(EVALUATE).count(), 1);
    assert_eq!(input.matches(ACCEPT).count(), 2);
    let policy = if shared { "params.pad0==1u" } else { "true" };
    replace(&mut input, EVALUATE, REUSE);
    input = input.replace(ACCEPT, &format!("accepted_rates_ready={policy};{ACCEPT}"));
    input.push_str(DECLARATION);
    input
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_accepted_rates_source_contract() {
    if std::env::var_os("MOLLA_NEWTON_MIXED_KIND").is_some() {
        assert_eq!(kind(), "accepted-rates-shared");
    }
    let baseline = grouped_sum::source_eight(
        adjugate::baseline(shader_source_for(NewtonVariant::Fischer)),
        false,
    );
    let shared = source(baseline.clone(), true);
    let mut restored = shared.clone();
    replace(&mut restored, DECLARATION, "");
    replace(&mut restored, REUSE, EVALUATE);
    restored = restored.replace(
        &format!("accepted_rates_ready=params.pad0==1u;{ACCEPT}"),
        ACCEPT,
    );
    assert_eq!(restored, baseline);
    assert_eq!(shared.matches("accepted_rates_ready=").count(), 3);
    for name in [
        "project_rates",
        "nonlinear_contact",
        "scratch_norm",
        "reduce_fischer_smoothing",
    ] {
        let start = baseline.find(&format!("fn {name}(")).unwrap();
        let suffix = &baseline[start..];
        let end = suffix.find("\nfn ").unwrap_or(suffix.len());
        assert!(shared.contains(&suffix[..end]));
        if name != "project_rates" {
            assert!(!suffix[..end].contains("rates[lane]="));
            assert!(!suffix[..end].contains("project_rates("));
        }
    }
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            baseline.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    for (name, input) in [
        ("control", baseline),
        ("shared", shared),
        ("candidate", source(restored, false)),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_ACCEPTED_RATES_SHADERS") {
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
