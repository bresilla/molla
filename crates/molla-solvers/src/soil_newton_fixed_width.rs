use super::*;
use crate::soil_newton_protected::FIXED_WIDTH_MARKER;

pub(super) fn source(source: String, width: u32) -> Result<String> {
    if !(1..=32).contains(&width)
        || !source.contains("params.width")
        || source.contains(FIXED_WIDTH_MARKER)
    {
        return Err(invalid());
    }
    if !crate::soil_newton_protected::uses_compiler(&source)? {
        return Err(invalid());
    }
    Ok(format!("{source}\n{FIXED_WIDTH_MARKER}{width}\n"))
}

pub(crate) fn specialize(source: String) -> Result<String> {
    if source.matches(FIXED_WIDTH_MARKER).count() != 1 || !source.contains("params.width") {
        return Err(invalid());
    }
    let width = source
        .split_once(FIXED_WIDTH_MARKER)
        .unwrap()
        .1
        .lines()
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|w| (1..=32).contains(w))
        .ok_or_else(invalid)?;
    Ok(source.replace("params.width", &format!("{width}u")))
}

#[test]
fn newton_fixed_width_source_contract() {
    let control = native_division_program(shader_source_for(NewtonVariant::Fischer), false);
    for width in [1, 3, 10, 32] {
        let candidate = source(control.clone(), width).unwrap();
        assert!(crate::soil_newton_protected::uses_compiler(&candidate).unwrap());
        let applied = specialize(candidate.clone()).unwrap();
        assert!(!applied.contains("params.width"));
        assert_eq!(
            applied,
            candidate.replace("params.width", &format!("{width}u"))
        );
        for unchanged in [
            "params.contacts",
            "params.iterations",
            "params.tolerance",
            "workgroupBarrier",
            "storageBarrier",
        ] {
            assert_eq!(
                applied.matches(unchanged).count(),
                control.matches(unchanged).count()
            );
        }
        crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
        assert!(specialize(applied).is_err());
        assert!(source(candidate, width).is_err());
    }
    for width in [0, 33, u32::MAX] {
        assert!(source(control.clone(), width).is_err());
    }
    assert!(source("params.width".into(), 10).is_err());
    for marker in ["0", "33", "unknown", "10\n// NEWTON_FIXED_DOF_WIDTH:10"] {
        assert!(
            crate::soil_newton_protected::compile(
                &format!("{control}\n{FIXED_WIDTH_MARKER}{marker}\n"),
                "newton"
            )
            .is_err()
        );
    }
}
