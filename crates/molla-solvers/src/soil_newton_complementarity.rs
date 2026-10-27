use super::*;

const NORMAL: &str = "fischer_precise(lambda,fischer_product(a.paired_rho,a.paired_separation))";
const WALL: &str = "fischer_precise(rate_mul(a.wall[i],a.paired_w[i]),rate_mul(a.wall[i]*a.mass,a.paired_soil[i]))";

fn additions() -> String {
    format!(
        "\nstruct FischerSamples {{ normal:array<vec2<f32>,3>, wall:array<array<vec2<f32>,3>,3> }}\nfn fischer_samples(a:Local)->FischerSamples {{\n    var result:FischerSamples;let lambda=fischer_dot(a.paired_normal,a.paired_k);\n    result.normal={NORMAL};\n    for(var i=0u;i<3u;i++) {{if a.wall[i]!=0.0 {{result.wall[i]={WALL};}}}}\n    return result;\n}}\nfn fischer_residual(a:Local)->array<f32,6> {{return fischer_residual_sampled(a,fischer_samples(a));}}\n"
    )
}

fn changes(shared: bool) -> Vec<(String, String)> {
    let policy = if shared { "params.pad0==1u" } else { "true" };
    [
        ("fn fischer_residual(a:Local)->array<f32,6>".into(), "fn fischer_residual_sampled(a:Local,samples:FischerSamples)->array<f32,6>".into()),
        (format!("let normal={NORMAL}[0];"), "let normal=samples.normal[0];".into()),
        (format!("a.wall[i]*value({WALL}[0])"), "a.wall[i]*value(samples.wall[i][0])".into()),
        ("fn fischer_linear(a:Local)->FischerLinear".into(), "fn fischer_linear(a:Local,samples:FischerSamples)->FischerLinear".into()),
        (format!("let normal={NORMAL};"), format!("var normal=samples.normal;if !({policy}) {{normal={NORMAL};}}")),
        (format!("let wall={WALL};"), format!("var wall=samples.wall[i];if !({policy}) {{wall={WALL};}}")),
        ("fn fischer_store_block(c:u32,local:Local)".into(), "fn fischer_store_block(c:u32,local:Local,samples:FischerSamples)".into()),
        ("let a=fischer_linear(local);".into(), "let a=fischer_linear(local,samples);".into()),
        ("blocks[c].residual=fischer_residual(a);".into(), "let samples=fischer_samples(a);blocks[c].residual=fischer_residual_sampled(a,samples);".into()),
        ("fischer_store_block(c,a);".into(), "fischer_store_block(c,a,samples);".into()),
    ].into()
}

pub(crate) fn source(mut input: String, shared: bool) -> String {
    for (from, to) in changes(shared) {
        replace(&mut input, &from, &to);
    }
    input.push_str(&additions());
    input
}

pub(crate) fn baseline(input: String) -> String {
    grouped_sum::source_eight(adjugate::baseline(input), false)
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_complementarity_source_contract() {
    if std::env::var_os("MOLLA_NEWTON_MIXED_KIND").is_some() {
        assert_eq!(kind(), "complementarity-shared");
    }
    let control = baseline(shader_source_for(NewtonVariant::Fischer));
    let shared = source(control.clone(), true);
    let mut restored = shared.clone();
    replace(&mut restored, &additions(), "");
    for (from, to) in changes(true).into_iter().rev() {
        replace(&mut restored, &to, &from);
    }
    assert_eq!(restored, control);
    assert_eq!(shared.matches("fischer_linear(").count(), 2);
    assert_eq!(shared.matches("fischer_store_block(").count(), 2);
    let samples = shared
        .find("let samples=fischer_samples(a);blocks[c]")
        .unwrap();
    assert!(shared[samples..].starts_with("let samples=fischer_samples(a);blocks[c].residual=fischer_residual_sampled(a,samples);\n    if direct_linearize {fischer_store_block(c,a,samples);}"));
    for declaration in ["var<storage", "var<workgroup>", "var<private>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    for (name, input) in [
        ("control", control.clone()),
        ("shared", shared),
        ("candidate", source(control, false)),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_COMPLEMENTARITY_SHADERS") {
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
