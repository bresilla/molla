use super::*;

const NORMAL: &str = "fischer_precise(lambda,fischer_product(a.paired_rho,a.paired_separation))";
const WALL: &str = "fischer_precise(rate_mul(a.wall[i],a.paired_w[i]),rate_mul(a.wall[i]*a.mass,a.paired_soil[i]))";
const DECLARATIONS: &str = "\nvar<private> contact_normal_sample:array<vec2<f32>,3>;\nvar<private> contact_wall_samples:array<array<vec2<f32>,3>,3>;\n";

fn changes(shared: bool) -> Vec<(String, String)> {
    let policy = if shared { "params.pad0==1u" } else { "true" };
    vec![
        (
            format!("let normal={NORMAL}[0];"),
            format!(
                "let normal_sample={NORMAL};if {policy} {{contact_normal_sample=normal_sample;}}let normal=normal_sample[0];"
            ),
        ),
        (
            format!("if a.wall[i]!=0.0 {{w[i]=a.wall[i]*value({WALL}[0]);}}"),
            format!(
                "if a.wall[i]!=0.0 {{let wall_sample={WALL};if {policy} {{contact_wall_samples[i]=wall_sample;}}w[i]=a.wall[i]*value(wall_sample[0]);}}"
            ),
        ),
        (
            format!("let normal={NORMAL};"),
            format!(
                "var normal:array<vec2<f32>,3>;if {policy} {{normal=contact_normal_sample;}}else {{normal={NORMAL};}}"
            ),
        ),
        (
            format!("let wall={WALL};"),
            format!(
                "var wall:array<vec2<f32>,3>;if {policy} {{wall=contact_wall_samples[i];}}else {{wall={WALL};}}"
            ),
        ),
    ]
}

pub(crate) fn source(mut input: String, shared: bool) -> String {
    assert!(input.contains("blocks[c].residual=fischer_residual(a);\n    if direct_linearize {fischer_store_block(c,a);}"));
    assert_eq!(input.matches("fischer_linear(").count(), 2);
    assert_eq!(input.matches("fischer_store_block(").count(), 2);
    for (from, to) in changes(shared) {
        replace(&mut input, &from, &to);
    }
    input.push_str(DECLARATIONS);
    input
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_private_samples_source_contract() {
    if std::env::var_os("MOLLA_NEWTON_MIXED_KIND").is_some() {
        assert_eq!(kind(), "private-samples-shared");
    }
    let control = complementarity::baseline(shader_source_for(NewtonVariant::Fischer));
    let shared = source(control.clone(), true);
    let mut restored = shared.clone();
    replace(&mut restored, DECLARATIONS, "");
    for (from, to) in changes(true).into_iter().rev() {
        replace(&mut restored, &to, &from);
    }
    assert_eq!(restored, control);
    for declaration in ["var<storage", "var<workgroup>", "@binding(", "fn "] {
        assert_eq!(
            control.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    assert_eq!(
        control.matches("var<private>").count() + 2,
        shared.matches("var<private>").count()
    );
    for (name, input) in [
        ("control", control.clone()),
        ("shared", shared),
        ("candidate", source(control, false)),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_PRIVATE_SAMPLES_SHADERS") {
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
