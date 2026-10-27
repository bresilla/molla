#[cfg(test)]
use super::ExperimentalSoilNewtonGpu;
use super::replace;

#[cfg(test)]
#[path = "soil_newton_retained_conditioned_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) fn baseline(source: String) -> String {
    use crate::soil_newton_parallel_release as p;
    let source = ExperimentalSoilNewtonGpu::stalled_release_source(source);
    let source = ExperimentalSoilNewtonGpu::cached_contact_source(
        ExperimentalSoilNewtonGpu::cached_delta_source(source),
    );
    p::publication_products::source(p::active_publication::source(p::norm_tree::source(
        p::schur_inverse::source(p::core_source(p::source(source))),
    )))
}

#[cfg(test)]
pub(super) fn publication_adaptive_probe(input: String, candidate: bool, edge: bool) -> String {
    let mut input = source(baseline(input));
    if candidate {
        input = crate::soil_newton_parallel_release::publication_products::adaptive_source(input);
    }
    if edge {
        input = crate::soil_newton_parallel_release::active_publication::probe_source(input, false);
    }
    format!(
        "{}{}{}256\n{input}",
        crate::soil_newton_protected::PREFIX,
        crate::soil_newton_protected::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    )
}

#[test]
fn newton_publication_adaptive_composed_compile() {
    let base = source(baseline(super::shader_source_for(
        super::NewtonVariant::Fischer,
    )));
    let candidate =
        crate::soil_newton_parallel_release::publication_products::adaptive_source(base.clone());
    assert_eq!(
        base.matches("var<workgroup>").count(),
        candidate.matches("var<workgroup>").count()
    );
    assert_eq!(
        base.matches("var<storage").count(),
        candidate.matches("var<storage").count()
    );
    assert!(candidate.contains("for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}"));
    for edge in [false, true] {
        let shader = publication_adaptive_probe(
            super::shader_source_for(super::NewtonVariant::Fischer),
            true,
            edge,
        );
        crate::soil_newton_protected::compile(&shader, if edge { "norm_probe" } else { "newton" })
            .unwrap();
    }
}

pub(crate) fn source(mut source: String) -> String {
    let start = source.find("fn conditioned_factor()->bool {").unwrap();
    let end = start + source[start..].find("\nfn conditioned_prepare(").unwrap();
    let mut factor = source[start..end].to_owned();
    replace(
        &mut factor,
        "let n=params.width;let stride=n+4u;",
        "let n=conditioned_core_width;let stride=n+4u;",
    );
    assert_eq!(factor.matches("1152u").count(), 2);
    factor = factor.replace("1152u", "1600u");
    source.replace_range(start..end, &factor);
    let start = source.find("fn conditioned_prepare(").unwrap();
    let end = start + source[start..].find("\nfn conditioned_delta(").unwrap();
    source.replace_range(
        start..end,
        include_str!("../shaders/soil_newton_retained_conditioned.wgsl"),
    );
    replace(
        &mut source,
        "let n=params.width;let offset=i*(n+4u)+n;",
        "let n=conditioned_core_width;let offset=i*(n+4u)+n;",
    );
    replace(
        &mut source,
        "if c==chosen {return State(next,Six());}",
        "if c==chosen {return State(next,Six());}\n    if c==conditioned_retained {return retained_conditioned_state(c,next);}",
    );
    for function in ["fn release_local(", "fn release_local_candidate("] {
        let start = source.find(function).unwrap();
        let end = start + source[start..].find("\nfn ").unwrap();
        let mut local = source[start..end].to_owned();
        for (old, new, count) in [
            (
                "let width=params.width;let stride=width+4u;",
                "let width=params.width;let core=conditioned_core_width;let stride=core+4u;",
                1,
            ),
            ("condensed[i*stride+width]", "condensed[i*stride+core]", 1),
            (
                "condensed[i*stride+width+1u+j]",
                "condensed[i*stride+core+1u+j]",
                2,
            ),
        ] {
            assert_eq!(local.matches(old).count(), count, "{function}: {old}");
            local = local.replace(old, new);
        }
        source.replace_range(start..end, &local);
    }
    source
}

#[test]
fn retained_conditioned_core_bounds_and_compile() {
    use crate::soil_newton_parallel_release as p;
    for width in 1..=32usize {
        for retained in [0, 6] {
            let core = width + retained;
            assert!(core * (core + 4) <= 1600);
            assert!(1600 + core * (core + 4) < 4000);
        }
    }
    let base = ExperimentalSoilNewtonGpu::stalled_release_source(super::shader_source_for(
        super::NewtonVariant::Fischer,
    ));
    let base = ExperimentalSoilNewtonGpu::cached_contact_source(
        ExperimentalSoilNewtonGpu::cached_delta_source(base),
    );
    let base = p::publication_products::source(p::active_publication::source(
        p::norm_tree::source(p::schur_inverse::source(p::core_source(p::source(base)))),
    ));
    let candidate = source(base.clone());
    for bound in [
        "backtrack<16u",
        "attempt==8u",
        "step==params.newton_iterations",
        "outer.releases)>=8u",
    ] {
        assert_eq!(
            candidate.matches(bound).count(),
            base.matches(bound).count()
        );
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
