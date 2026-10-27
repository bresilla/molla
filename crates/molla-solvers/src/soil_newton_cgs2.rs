pub(crate) fn source(mut source: String) -> String {
    let krylov = include_str!("../shaders/soil_newton_krylov.wgsl");
    let start = krylov.find("        for (var orthogonal_pass=0u;").unwrap();
    let end = start
        + krylov[start..]
            .find("        let magnitude=scratch_norm();")
            .unwrap();
    super::replace(
        &mut source,
        &krylov[start..end],
        "        cgs2_orthogonalize(j);\n",
    );
    assert_eq!(
        source
            .matches("var<workgroup> coefficients:array<f32,64>;")
            .count(),
        1
    );
    assert!(source.contains("coefficients[i]=value(sum)/h[i*64u+i];"));
    source.push_str(include_str!("../shaders/soil_newton_cgs2.wgsl"));
    source
}

#[test]
fn newton_cgs2_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::schur_inverse::source(super::core_source(super::source(
        Solver::cached_contact_source(Solver::cached_delta_source(Solver::stalled_release_source(
            shader_source_for(NewtonVariant::Fischer),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    assert!(candidate.contains("sweep<2u"));
    for (name, shader) in [("control", control), ("candidate", candidate)] {
        let binary = crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{shader}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "newton",
        )
        .unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_CGS2_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), shader).unwrap();
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

#[test]
fn newton_cgs2_lane_groups_are_disjoint_and_complete() {
    for count in 1..=64 {
        let mut width = 256;
        while width * count > 256 {
            width /= 2;
        }
        assert!(width >= 4);
        for lane in 0..256 {
            let group = lane / width;
            let member = lane % width;
            let mut stride = width / 2;
            while stride > 0 {
                if member < stride {
                    assert_eq!((lane + stride) / width, group);
                    assert!(lane + stride < 256);
                }
                stride /= 2;
            }
        }
        for contacts in [0usize, 1, 255, 256, 257, 601] {
            for group in 0..count {
                let mut visited = vec![0u32; contacts];
                for lane in group * width..(group + 1) * width {
                    for c in (lane % width..contacts).step_by(width) {
                        visited[c] += 1;
                    }
                }
                assert!(visited.iter().all(|&v| v == 1));
            }
        }
    }
}
