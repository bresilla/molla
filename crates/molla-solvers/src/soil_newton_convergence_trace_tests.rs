use super::*;

#[test]
fn newton_stagnant_branch_preserves_caps_and_checks() {
    let baseline = ExperimentalSoilNewtonGpu::post_release_settle_source(
        ExperimentalSoilNewtonGpu::stalled_release_source(
            crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
        ),
    );
    let candidate = ExperimentalSoilNewtonGpu::stagnant_branch_source(baseline.clone());
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
        "score<branch_merit",
        "if visited {continue;}",
    ] {
        assert_eq!(
            candidate.matches(bound).count(),
            baseline.matches(bound).count()
        );
    }
    assert!(
        candidate.find("step==params.newton_iterations").unwrap()
            < candidate
                .find("if fischer_smoothing==0.0 && smoothing_stalled==1u")
                .unwrap()
    );
    assert!(candidate.contains("if branch_settle && rank>0u {break;}"));
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_post_release_priority_preserves_caps_and_merit() {
    let baseline = ExperimentalSoilNewtonGpu::post_release_settle_source(
        ExperimentalSoilNewtonGpu::stalled_release_source(
            crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
        ),
    );
    let candidate = ExperimentalSoilNewtonGpu::post_release_priority_source(baseline.clone());
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
        "score<branch_merit",
        "if visited {continue;}",
    ] {
        assert_eq!(
            candidate.matches(bound).count(),
            baseline.matches(bound).count()
        );
    }
    assert!(candidate.contains(
        "if branch_settle && workgroupUniformLoad(&release_contact)!=0xffffffffu {break;}"
    ));
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_post_release_settle_preserves_caps_and_checks() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::post_release_settle_source(baseline.clone());
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
    ] {
        assert_eq!(
            candidate.matches(bound).count(),
            baseline.matches(bound).count()
        );
    }
    assert!(
        candidate.find("step==params.newton_iterations").unwrap()
            < candidate
                .find("if settle_pending && fischer_smoothing==0.0")
                .unwrap()
    );
    assert!(candidate.contains("(!branch_settle || score<branch_merit)"));
    assert!(candidate.contains("if visited {continue;}"));
    assert!(candidate.contains("release_visited[outer.releases]=chosen;outer.releases++;"));
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_conditioned_refinement_preserves_outer_limits() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::conditioned_refine_source(baseline.clone());
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
    ] {
        assert_eq!(
            candidate.matches(bound).count(),
            baseline.matches(bound).count()
        );
    }
    assert!(candidate.contains("var permutation:array<u32,6>"));
    assert!(candidate.contains("residual[permutation[i]]"));
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
#[cfg(all(feature = "publication-products-newton", not(target_arch = "wasm32")))]
fn newton_conditioned_pivot_inspection_compiles() {
    use crate::soil_newton_parallel_release as p;
    let source = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let source = ExperimentalSoilNewtonGpu::cached_contact_source(
        ExperimentalSoilNewtonGpu::cached_delta_source(source),
    );
    let source = p::publication_products::source(p::active_publication::source(
        p::norm_tree::source(p::schur_inverse::source(p::core_source(p::source(source)))),
    ));
    let candidate = ExperimentalSoilNewtonGpu::pivot_inspect_source(source, 113);
    assert!(candidate.contains("let c=113u;"));
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_fallback_branch_preserves_limits_and_checks() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::fallback_branch_pivot_source(baseline.clone());
    let old = "if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}";
    let new = "if (!releases && fischer_smoothing!=0.0) || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}";
    assert_eq!(candidate.replace(new, old), baseline);
    assert_eq!(
        candidate
            .matches("attempt==0u && conditioned_release()")
            .count(),
        1
    );
    for smoothing in [0.0, 1e-9, 0.001, 0.01] {
        for releases in [false, true] {
            assert_eq!(
                !(!releases && smoothing != 0.0),
                releases || smoothing == 0.0
            );
        }
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_branch_pivot_preserves_limits_and_physical_checks() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::branch_pivot_source(baseline.clone());
    let old = "if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}";
    let new = "if (!releases && !(fischer_smoothing==0.0 && smoothing_stalled>=2u)) || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}";
    assert_eq!(candidate.replace(new, old), baseline);
    for smoothed in [false, true] {
        for stalled in 0..=3 {
            for releases in [false, true] {
                let accepted = releases || (!smoothed && stalled >= 2);
                assert_eq!(
                    accepted,
                    releases || (!smoothed && matches!(stalled, 2 | 3))
                );
            }
        }
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_nonmonotone_stall_keeps_publication_and_limits() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::nonmonotone_stall_source(baseline.clone());
    let old = "candidate<merit*(1.0-select(1e-4*alpha,0.01,fischer_smoothing==0.0 && smoothing_stalled>=2u))";
    let new = "candidate<merit*select(1.0-1e-4*alpha,1.0-0.1*alpha+0.1/f32(step+1u),fischer_smoothing==0.0 && smoothing_stalled>=2u)";
    assert_eq!(candidate.replace(new, old), baseline);
    for step in 1..=64 {
        let allowance = 0.1 / f64::from(step);
        assert!(allowance > 0. && allowance <= 0.1);
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_revisited_release_preserves_work_bounds() {
    let baseline = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    let candidate = ExperimentalSoilNewtonGpu::revisited_release_source(baseline.clone());
    let marker = "fn conditioned_release()->bool {";
    assert_eq!(
        baseline.split_once(marker).unwrap().0,
        candidate.split_once(marker).unwrap().0
    );
    let body = candidate.split_once(marker).unwrap().1;
    assert!(
        body.contains("if visited && (fischer_smoothing>0.0 || smoothing_stalled<2u) {continue;}")
    );
    assert!(body.contains("outer.releases)>=8u"));
    assert!(body.contains("release_visited[outer.releases]=chosen;outer.releases++;"));
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
    ] {
        assert_eq!(
            baseline.matches(bound).count(),
            candidate.matches(bound).count()
        );
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}

#[test]
fn newton_stalled_release_shader_structure() {
    let source = ExperimentalSoilNewtonGpu::stalled_release_source(
        crate::soil_newton_gpu::shader_source_for(NewtonVariant::Fischer),
    );
    assert_eq!(
        source
            .matches("select(1e-4*alpha,0.01,fischer_smoothing==0.0 && smoothing_stalled>=2u)")
            .count(),
        1
    );
    assert!(source.contains("backtrack<16u"));
    assert!(source.contains("attempt==8u"));
    assert!(source.contains("outer.releases)>=8u"));
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    crate::soil_newton_protected::compile(&source, "newton").unwrap();
}

#[path = "soil_newton_prefix_oracle.rs"]
mod prefix_oracle;

pub(super) fn pivot_prefixes(
    reference: &Reference,
    slots: usize,
    input: &std::path::Path,
    output: &std::path::Path,
) {
    prefix_oracle::run(reference, slots, input, output);
}

fn control_snapshot(
    report: &[u32; 8],
    limit: u32,
    words: Option<&[f32]>,
) -> Option<serde_json::Value> {
    if report[0] == 0 {
        assert!(
            words.is_none(),
            "completed solve has no prefix-cap control snapshot"
        );
        return None;
    }
    assert_eq!(report[0], 8);
    let words = words.expect("prefix-cap snapshot required");
    assert_eq!(words.len(), 20);
    assert!(words.iter().all(|w| w.is_finite()));
    assert_eq!(words[3], limit as f32);
    assert!(words[0] >= 0. && words[0] <= 0.01);
    assert!(words[1] >= 0. && words[1].fract() == 0.);
    assert!(words[2] >= 0. && words[2] <= 8. && words[2].fract() == 0.);
    for (i, v) in words[4..].chunks_exact(2).enumerate() {
        assert_eq!(v[1], if i < words[2] as usize { 1. } else { 0. });
        if v[1] == 1. {
            assert!(v[0] >= 0. && v[0].fract() == 0.);
        }
    }
    Some(
        serde_json::json!({"smoothing":words[0],"stalled_steps":words[1],
        "releases":words[2],"visited":words[4..].chunks_exact(2)
            .filter(|v|v[1]==1.).map(|v|v[0] as u32).collect::<Vec<_>>()}),
    )
}

#[test]
fn newton_convergence_control_snapshot_contract() {
    let completed = [0u32; 8];
    assert!(control_snapshot(&completed, 64, None).is_none());
    let mut cap = completed;
    cap[0] = 8;
    let mut words = [0f32; 20];
    words[..6].copy_from_slice(&[0.001, 2., 1., 16., 115., 1.]);
    let value = control_snapshot(&cap, 16, Some(&words)).unwrap();
    assert_eq!(value["visited"], serde_json::json!([115]));
    assert!(std::panic::catch_unwind(|| control_snapshot(&completed, 64, Some(&words))).is_err());
    assert!(std::panic::catch_unwind(|| control_snapshot(&cap, 16, None)).is_err());
    for (index, value) in [
        (0, f32::NAN),
        (0, -1.),
        (1, 0.5),
        (2, 9.),
        (3, 15.),
        (4, -1.),
        (5, 0.),
        (7, 1.),
    ] {
        let mut invalid = words;
        invalid[index] = value;
        assert!(std::panic::catch_unwind(|| control_snapshot(&cap, 16, Some(&invalid))).is_err());
    }
}

pub(super) fn trace(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    reference: &Reference,
    output: &std::path::Path,
    seed: Option<&[u8]>,
) {
    assert!(
        !output.exists(),
        "refusing to overwrite convergence evidence"
    );
    std::fs::create_dir_all(output).unwrap();
    let mut rows = Vec::new();
    let limits: Vec<u32> = if std::env::var_os("MOLLA_NEWTON_CONVERGENCE_LATE").is_some() {
        (48..=64).collect()
    } else {
        vec![0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    };
    for limit in limits {
        solver.set_iteration_prefix_probe(engine, limit);
        if let Some(seed) = seed {
            engine
                .queue()
                .unwrap()
                .write_buffer(solver.state_buffer(), 0, seed);
            engine.queue().unwrap().write_buffer(
                solver.status_buffer(),
                0,
                bytemuck::cast_slice(&[0u32; 8]),
            );
        }
        let report = run(engine, solver, seed.is_none());
        assert!(matches!(report[0], 0 | 8), "prefix={limit} {report:?}");
        let bytes = capture::read(engine, solver.state_buffer());
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let x: Vec<f64> = reference
            .contacts
            .iter()
            .flat_map(|c| {
                let values = &values;
                (0..6).map(move |i| {
                    let at = 16 * c.index + if i < 3 { i } else { i + 1 };
                    f64::from(values[at]) + f64::from(values[at + 8])
                })
            })
            .collect();
        let residual = reference.residual(&x);
        let mut ranked: Vec<_> = residual
            .chunks_exact(6)
            .enumerate()
            .map(|(i, r)| (reference.contacts[i].index, norm(r)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let control = if std::env::var_os("MOLLA_NEWTON_CONVERGENCE_CONTROL").is_some() {
            let words = (report[0] == 8).then(|| {
                capture::read(engine, solver.precision_local_buffer())[..80]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            });
            control_snapshot(&report, limit, words.as_deref())
        } else {
            None
        };
        let row = serde_json::json!({
            "limit": limit, "report": report,
            "gpu_residual": f32::from_bits(report[4]),
            "gpu_physical": f32::from_bits(report[5]),
            "damping": f32::from_bits(report[6]),
            "reference_physical": reference.physical_error(&x),
            "reference_residual": norm(&residual),
            "worst_residual_contacts": &ranked[..ranked.len().min(4)],
            "control":control,
        });
        eprintln!("convergence-prefix {row}");
        rows.push(row);
        std::fs::write(output.join(format!("states-{limit}.bin")), bytes).unwrap();
        std::fs::write(
            output.join("prefixes.json"),
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
    }
    solver.set_iteration_prefix_probe(engine, 64);
}
