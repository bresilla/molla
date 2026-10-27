const DENSE: &str = "for (var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {";
const ACTIVE: &str = "for (var publication_slot=0u;publication_slot<status.active_count;publication_slot++) {let c=active_indices[publication_slot];if contacts[c].soil_mass.w>0.0 {";

pub(crate) fn source(mut source: String) -> String {
    let publication = crate::soil_newton_publish_gpu::resident_source();
    assert_eq!(publication.matches(DENSE).count(), 2);
    let optimized = publication.replace(DENSE, ACTIVE);
    super::replace(&mut source, &publication, &optimized);
    source
}

#[cfg(test)]
pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    let mut input = if candidate { source(input) } else { input };
    input.push_str(r#"
@compute @workgroup_size(256)
fn norm_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {
        status.active_count=0u;
        for(var c=0u;c<params.contacts;c++) {
            if contacts[c].soil_mass.w>0.0 {active_indices[status.active_count]=c;status.active_count++;}
        }
    }
    storageBarrier();let accepted=certify_candidate(lane);
    if lane==0u {report.steps=select(0u,1u,accepted);}
}
"#);
    input
}

#[test]
fn newton_active_publication_preserves_checks_and_sum_order() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::norm_tree::source(super::schur_inverse::source(super::core_source(
        super::source(Solver::cached_contact_source(Solver::cached_delta_source(
            Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(candidate.matches(ACTIVE).count(), 2);
    assert_eq!(candidate.replace(ACTIVE, DENSE), control);
    assert!(control.contains("if mass>0.0 {active_indices[count]=c;count++;}"));
    for capacity in [0, 1, 63, 64, 65, 255, 256, 257, 4096] {
        let masses: Vec<_> = (0..capacity)
            .map(|c| if c % 7 == 0 { 1.0f32 } else { 0.0 })
            .collect();
        let dense: Vec<_> = masses
            .iter()
            .enumerate()
            .filter(|(_, m)| **m > 0.0)
            .map(|(c, _)| c)
            .collect();
        let indices = dense.clone();
        let active: Vec<_> = (0..indices.len())
            .map(|slot| indices[slot])
            .filter(|&c| masses[c] > 0.0)
            .collect();
        assert_eq!(active, dense);
    }
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    for optimized in [false, true] {
        let probe = probe_source(control.clone(), optimized);
        crate::soil_newton_protected::compile(
            &format!(
                "{}{}256\n{probe}",
                crate::soil_newton_protected::VOLATILE_MARKER,
                crate::soil_newton_width::MARKER
            ),
            "norm_probe",
        )
        .unwrap();
    }
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_ACTIVE_PUBLICATION_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[path = "soil_newton_active_publication_tests.rs"]
#[cfg(test)]
mod tests;
