use super::*;

pub(crate) fn source(source: String) -> String {
    let marker = "fischer_smoothing*=0.1;";
    assert_eq!(source.matches(marker).count(), 1);
    source.replace(marker, "fischer_smoothing*=0.01;")
}

#[test]
fn newton_fast_continuation_contract() {
    let control = shared_core::source(native_division_program(
        shader_source_for(NewtonVariant::Fischer),
        false,
    ));
    let candidate = source(control.clone());
    assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    assert_eq!(
        candidate.replace("fischer_smoothing*=0.01;", "fischer_smoothing*=0.1;"),
        control
    );
    assert!(candidate.contains("fischer_smoothing<1e-10"));
    assert!(candidate.contains("residual<=fischer_smoothing || smoothing_stalled>=2u"));
    for initial in [0.0f32, 1e-12, 1e-10, 1e-8, 0.01, 0.1, 1.0] {
        let mut epsilon = initial;
        for _ in 0..16 {
            let old = epsilon;
            epsilon *= 0.01;
            if epsilon < 1e-10 {
                epsilon = 0.0;
            }
            assert!(epsilon.is_finite() && epsilon >= 0.0 && epsilon <= old);
        }
        assert_eq!(epsilon, 0.0);
    }
}
