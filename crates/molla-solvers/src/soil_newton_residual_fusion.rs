use super::*;

fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    &source[start..end]
}

fn transform(source: &str) -> (String, String, String) {
    let direction = body(source, "apply_direction");
    let norm = body(source, "scratch_norm");
    let error = body(source, "true_direction_error");
    let mut fused = direction.replacen(
        "fn apply_direction() {",
        "fn fused_direction_norm()->f32 {\n    storageBarrier();\n    var largest=0.0; var invalid=false;",
        1,
    );
    let assignment = "result[i]=value(sum);";
    assert_eq!(fused.matches(assignment).count(), 1);
    fused = fused.replace(assignment, "let applied=value(sum);\n            let v=value(rate_add(vec2<f32>(applied,0.0),vec2<f32>(soa_residual(c,i),0.0)));\n            result[i]=v;\n            if !finite(v) {invalid=true;}largest=max(largest,abs(v));");
    let norm_tail = &norm[norm.find("    parts[lane]=vec2<f32>(largest,").unwrap()..];
    fused.truncate(fused.len() - 1);
    fused.push_str(norm_tail);
    let first = error
        .find("    storageBarrier();apply_direction();")
        .unwrap();
    let last = error.find("    if lane==0u").unwrap();
    let original = error[first..last].to_owned();
    assert_eq!(original.matches("let error=scratch_norm();").count(), 1);
    let replacement = format!(
        "    var error=0.0;\n    if params.pad2==0u {{\n{}    }} else {{error=fused_direction_norm();}}\n",
        original.replace("let error=scratch_norm();", "error=scratch_norm();")
    );
    (original, replacement, fused)
}

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    let (original, replacement, fused) = transform(&source);
    replace(&mut source, &original, &replacement);
    source.push('\n');
    source.push_str(&fused);
    source
}

#[test]
fn newton_residual_fusion_source_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let (original, replacement, fused) = transform(&control);
    let mut restored = candidate
        .strip_suffix(&format!("\n{fused}"))
        .unwrap()
        .to_owned();
    replace(&mut restored, &replacement, &original);
    assert_eq!(restored, control);
    for name in [
        "apply_direction",
        "apply_operator",
        "scratch_norm",
        "recover_direction",
        "project_rates",
    ] {
        assert_eq!(body(&candidate, name), body(&control, name));
    }
    for token in ["@binding(", "var<storage", "var<workgroup>", "var<private>"] {
        assert_eq!(
            candidate.matches(token).count(),
            control.matches(token).count()
        );
    }
    let words = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_RESIDUAL_FUSION_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), control).unwrap();
        std::fs::write(root.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(root.join("candidate.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}

#[test]
fn newton_residual_fusion_rounding_and_traversal_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let (original, _, fused) = transform(&control);
    assert!(fused.contains("let applied=value(sum);"));
    assert!(fused.contains(
        "let v=value(rate_add(vec2<f32>(applied,0.0),vec2<f32>(soa_residual(c,i),0.0)));"
    ));
    let direction = body(&control, "apply_direction");
    let norm = body(&control, "scratch_norm");
    let visits = |text: &str| text.matches("slot<status.active_count").count();
    assert_eq!(visits(direction) + visits(&original) + visits(norm), 4);
    assert_eq!(visits(&fused), 2);
    assert_eq!(
        direction.matches("storageBarrier();").count()
            + original.matches("storageBarrier();").count(),
        3
    );
    assert_eq!(fused.matches("storageBarrier();").count(), 2);
    assert_eq!(fused.matches("scratch[c]=packed(result);").count(), 1);
    assert_eq!(fused.matches("project_rates(3u);").count(), 1);
    let marker = "    parts[lane]=vec2<f32>(largest,";
    assert_eq!(
        &fused[fused.find(marker).unwrap()..],
        &norm[norm.find(marker).unwrap()..]
    );
}
