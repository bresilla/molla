use super::*;
use std::sync::Arc;

#[test]
fn a_tyre_geometry_s_profile_is_built_once_and_matches_a_fresh_one() {
    let first = brush_profile(0.413, 0.21, 64, 3);
    assert!(Arc::ptr_eq(&first, &brush_profile(0.413, 0.21, 64, 3)));
    assert_eq!(*first, *tread_points(0.413, 0.21, 64, 3));
    for (radius, width, segments, strips) in [
        (0.414, 0.21, 64, 3),
        (0.413, 0.22, 64, 3),
        (0.413, 0.21, 32, 3),
        (0.413, 0.21, 64, 2),
    ] {
        let other = brush_profile(radius, width, segments, strips);
        assert!(!Arc::ptr_eq(&first, &other));
        assert_eq!(*other, *tread_points(radius, width, segments, strips));
    }
}
