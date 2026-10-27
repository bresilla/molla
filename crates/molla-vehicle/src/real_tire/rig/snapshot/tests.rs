use super::TireRigAcceptanceCheck as Check;

#[test]
fn acceptance_checks_reject_nonfinite_observations_and_bounds() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let checks = [
            Check::at_least("lower", "", "", invalid, -1.0),
            Check::at_least("lower_bound", "", "", 0.0, invalid),
            Check::at_most("upper", "", "", invalid, 1.0),
            Check::at_most("upper_bound", "", "", 0.0, invalid),
            Check::between("range", "", "", invalid, -1.0, 1.0),
            Check::between("range_lower", "", "", 0.0, invalid, 1.0),
            Check::between("range_upper", "", "", 0.0, -1.0, invalid),
            Check::between("range_both", "", "", 0.0, invalid, invalid),
        ];
        for check in checks {
            assert!(!check.passed, "{check:?}");
            assert!(check.is_finite(), "{check:?}");
            assert!(check.to_text_line().contains("passed=false"));
        }
    }
}

#[test]
fn acceptance_checks_keep_finite_inclusive_bounds_and_sorted_ranges() {
    for observed in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        assert_eq!(
            Check::at_least("lower", "", "", observed, -1.0).passed,
            observed >= -1.0
        );
        assert_eq!(
            Check::at_most("upper", "", "", observed, 1.0).passed,
            observed <= 1.0
        );
        for (lo, hi) in [(-1.0, 1.0), (1.0, -1.0)] {
            let check = Check::between("range", "", "", observed, lo, hi);
            assert_eq!(check.passed, (-1.0..=1.0).contains(&observed));
            assert_eq!(check.min_value, Some(-1.0));
            assert_eq!(check.max_value, Some(1.0));
        }
    }
}
