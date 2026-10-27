use super::*;

const ORIGINAL: &str = "    for(var norm_stride=128u;norm_stride>0u;norm_stride/=2u) {\n        if lane<norm_stride {parts[lane]=max(parts[lane],parts[lane+norm_stride]);}\n        workgroupBarrier();\n    }";
const GROUPED: &str = "
        for(var norm_group=32u;norm_group>0u;norm_group/=8u) {
            if lane<norm_group {
                let a=max(parts[lane],parts[lane+4u*norm_group]);
                let b=max(parts[lane+norm_group],parts[lane+5u*norm_group]);
                let c=max(parts[lane+2u*norm_group],parts[lane+6u*norm_group]);
                let d=max(parts[lane+3u*norm_group],parts[lane+7u*norm_group]);
                parts[lane]=max(max(a,c),max(b,d));
            }
            workgroupBarrier();
        }
        if lane==0u {parts[0]=max(max(parts[0],parts[2]),max(parts[1],parts[3]));}
        workgroupBarrier();
";

pub(crate) fn baseline(input: String) -> String {
    active_indices::source(complementarity::baseline(input), false)
}

pub(crate) fn source(mut input: String, shared: bool) -> String {
    assert!(input.contains("// NEWTON_EXECUTION_LANES:256\n"));
    assert!(!input.contains("params.pad0"));
    let policy = if shared { "params.pad0==1u" } else { "true" };
    replace(
        &mut input,
        ORIGINAL,
        &format!("    if {policy} {{{GROUPED}    }} else {{\n{ORIGINAL}\n    }}"),
    );
    input
}

#[test]
fn newton_grouped_norm_tree_and_ownership() {
    let mut control: Vec<_> = (0..256).map(|i| i.to_string()).collect();
    let mut grouped = control.clone();
    for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            control[lane] = format!("max({},{})", control[lane], control[lane + stride]);
        }
    }
    for stride in [32, 4] {
        for lane in 0..stride {
            let pair = |offset| {
                format!(
                    "max({},{})",
                    grouped[lane + offset * stride],
                    grouped[lane + (offset + 4) * stride]
                )
            };
            grouped[lane] = format!(
                "max(max({},{}),max({},{}))",
                pair(0),
                pair(2),
                pair(1),
                pair(3)
            );
        }
    }
    let result = format!(
        "max(max({},{}),max({},{}))",
        grouped[0], grouped[2], grouped[1], grouped[3]
    );
    assert_eq!(control[0], result);
    for stride in [32, 4] {
        let mut reads = vec![0; stride * 8];
        for lane in 0..stride {
            for offset in 0..8 {
                reads[lane + offset * stride] += 1;
            }
        }
        assert!(reads.iter().all(|&count| count == 1));
    }
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_grouped_norm_source_contract() {
    let control = baseline(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone(), true);
    assert_eq!(
        candidate.replace(
            &format!("    if params.pad0==1u {{{GROUPED}    }} else {{\n{ORIGINAL}\n    }}"),
            ORIGINAL
        ),
        control
    );
    for declaration in ["var<storage", "var<workgroup>", "var<private>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            candidate.matches(declaration).count()
        );
    }
    for (name, source) in [
        ("control", control.clone()),
        ("candidate", self::source(control, false)),
        ("shared", candidate),
    ] {
        let binary = crate::soil_newton_protected::compile(&source, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_GROUPED_NORM_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}
