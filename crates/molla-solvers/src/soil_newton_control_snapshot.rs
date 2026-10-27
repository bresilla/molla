use super::*;

const HYBRID: &str = "    let used=workgroupUniformLoad(&control.count);\n    let fault=workgroupUniformLoad(&control.stop);";
const HYBRID_SNAPSHOT: &str = "    var used=0u;var fault=0u;\n    if params.pad2==0u {\n        used=workgroupUniformLoad(&control.count);\n        fault=workgroupUniformLoad(&control.stop);\n    } else {\n        let snapshot=workgroupUniformLoad(&control);\n        used=snapshot.count;fault=snapshot.stop;\n    }";
const REFINE: &str = "        if workgroupUniformLoad(&control.stop)!=0u {return;}\n        if workgroupUniformLoad(&control.relative)<=params.tolerance {return;}\n        if used>=limit || workgroupUniformLoad(&control.count)==0u {\n            if lane==0u {atomicOr(&status.fault,8u);}return;\n        }";
const REFINE_SNAPSHOT: &str = "        let snapshot=workgroupUniformLoad(&control);\n        if snapshot.stop!=0u {return;}\n        if snapshot.relative<=params.tolerance {return;}\n        if used>=limit || snapshot.count==0u {\n            if lane==0u {atomicOr(&status.fault,8u);}return;\n        }";

fn refinement() -> String {
    format!(
        "        if params.pad2==0u {{\n{REFINE}\n        }} else {{\n{REFINE_SNAPSHOT}\n        }}"
    )
}

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    replace(&mut source, HYBRID, HYBRID_SNAPSHOT);
    replace(&mut source, REFINE, &refinement());
    source
}

#[test]
fn newton_control_snapshot_preserves_arithmetic_and_storage() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, HYBRID_SNAPSHOT, HYBRID);
    replace(&mut restored, &refinement(), REFINE);
    assert_eq!(restored, control);
    for marker in [
        "var<storage",
        "var<workgroup>",
        "@binding(",
        "rate_add(",
        "rate_mul(",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    assert_eq!(
        candidate
            .matches("let snapshot=workgroupUniformLoad(&control);")
            .count(),
        2
    );
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_control_snapshot_preserves_stopping_predicates() {
    for stop in [0, 1, 2, 8, u32::MAX] {
        for relative in [f32::NAN, f32::INFINITY, -1., 0., 1e-7, 1e-6, 1.] {
            for count in [0, 1, 64] {
                for used in [0, 63, 64] {
                    let original = if stop != 0 {
                        1
                    } else if relative <= 1e-6 {
                        2
                    } else if used >= 64 || count == 0 {
                        3
                    } else {
                        0
                    };
                    let snapshot = (stop, relative, count);
                    let copied = if snapshot.0 != 0 {
                        1
                    } else if snapshot.1 <= 1e-6 {
                        2
                    } else if used >= 64 || snapshot.2 == 0 {
                        3
                    } else {
                        0
                    };
                    assert_eq!(original, copied);
                }
            }
        }
    }
}
