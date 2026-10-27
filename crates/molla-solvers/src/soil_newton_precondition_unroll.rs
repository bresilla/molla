use super::*;

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    let start = source.find("fn precondition(c:u32,x:Six)->Six {").unwrap();
    let end = start + source[start..].find("\nfn build_inverse(").unwrap();
    let original = source[start..end].replacen("fn precondition(", "fn precondition_loop(", 1);
    let fields = ["k.x", "k.y", "k.z", "w.x", "w.y", "w.z"];
    let mut body =
        String::from("fn precondition_unrolled(c:u32,x:Six)->Six {\n    var result=Six();\n");
    for (row, output) in fields.iter().enumerate() {
        body.push_str("    {var sum=vec2<f32>(0.0);\n");
        for (column, input) in fields.iter().enumerate() {
            body.push_str(&format!(
                "        sum=rate_add(sum,rate_mul(soa_inverse(c,{}u),vec2<f32>(x.{input},0.0)));\n",
                6 * row + column
            ));
        }
        body.push_str(&format!("        result.{output}=value(sum);}}\n"));
    }
    body.push_str("    return result;\n}\nfn precondition(c:u32,x:Six)->Six {\n    if params.pad2==0u {return precondition_loop(c,x);}\n    return precondition_unrolled(c,x);\n}\n");
    source.replace_range(start..end, &format!("{original}\n{body}"));
    source
}

#[test]
fn newton_precondition_unroll_preserves_all_terms() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let start = candidate.find("fn precondition_unrolled(").unwrap();
    let end = start + candidate[start..].find("\nfn precondition(").unwrap();
    let unrolled = &candidate[start..end];
    assert_eq!(unrolled.matches("sum=rate_add(sum,rate_mul(").count(), 36);
    assert_eq!(unrolled.matches("=value(sum);").count(), 6);
    for index in 0..36 {
        assert_eq!(
            unrolled
                .matches(&format!("soa_inverse(c,{index}u)"))
                .count(),
            1
        );
    }
    for marker in [
        "@binding(",
        "storageBarrier()",
        "workgroupBarrier()",
        "publication_fault(",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
