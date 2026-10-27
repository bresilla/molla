const ORIGINAL: [&str; 2] = [
    "for (var axis=0u;axis<3u;axis++) {sum=rate_add(sum,rate_mul(-response[c*params.width+lane][axis],vec2<f32>(k[axis],0.0)));}",
    "for (var axis=0u;axis<3u;axis++) {reaction=rate_add(reaction,rate_mul(rows[c*params.width+i][axis],vec2<f32>(k[axis],0.0)));}",
];

fn optimized(original: &str) -> String {
    format!(
        "let publication_axes=select(3u,1u,all(k==vec3<f32>(0.0)));{}",
        original.replace("axis<3u", "axis<publication_axes")
    )
}

pub(crate) fn source(mut source: String) -> String {
    for original in ORIGINAL {
        super::replace(&mut source, original, &optimized(original));
    }
    source
}

pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    let input = super::active_publication::probe_source(input, true);
    if candidate { source(input) } else { input }
}

#[test]
fn newton_zero_publication_preserves_validation_and_first_axis() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::active_publication::source(super::norm_tree::source(
        super::schur_inverse::source(super::core_source(super::source(
            Solver::cached_contact_source(Solver::cached_delta_source(
                Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
            )),
        ))),
    ));
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    for original in ORIGINAL {
        assert_eq!(restored.matches(&optimized(original)).count(), 1);
        restored = restored.replace(&optimized(original), original);
    }
    assert_eq!(restored, control);
    crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
}
