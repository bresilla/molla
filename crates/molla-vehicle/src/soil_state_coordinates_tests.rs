use super::*;
use crate::{SoilTileCoord, SoilTileLayout};

#[test]
fn split_coordinates_preserve_boundary_sides_and_cancellation() {
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let frame = [1e12, -1e12, 0.0];
    let local = [-1e-6, -1e-6, -f64::from_bits(1)];
    assert_eq!(
        layout.tile_at_local(local, frame).unwrap(),
        SoilTileCoord([999_999_999_999, -1_000_000_000_001, -1])
    );
    assert_ne!(
        layout
            .tile_at(std::array::from_fn(|i| frame[i] + local[i]))
            .unwrap(),
        layout.tile_at_local(local, frame).unwrap()
    );
    let huge = [2f64.powi(100); 3];
    let shifted = SoilTileLayout::new(huge, 1.0).unwrap();
    assert_eq!(
        shifted.tile_at_local([-1e-100, 0.0, 1e-100], huge).unwrap(),
        SoilTileCoord([-1, 0, 0])
    );
    assert_eq!(
        layout.tile_at_local([-0.0; 3], [0.0; 3]).unwrap(),
        SoilTileCoord([0; 3])
    );
}

#[test]
fn split_coordinates_match_independent_scaled_integer_oracle() {
    let scale = 1i128 << 40;
    let frames = [
        0,
        1,
        -1,
        3,
        -3,
        1i64 << 20,
        -(1i64 << 20),
        1i64 << 40,
        -(1i64 << 40),
        1i64 << 50,
        -(1i64 << 50),
    ];
    let offsets = [-scale, -scale + 1, -129, -1, 0, 1, 129, scale - 1, scale];
    let edges = [
        1,
        3,
        127,
        scale - 1,
        scale,
        scale + 1,
        3 * scale + 17,
        16 * scale,
    ];
    let mut checked = 0;
    for frame in frames {
        for origin in [0, frame, frame - 1, frame + 1, -frame] {
            for local in offsets {
                for edge in edges {
                    let numerator = (i128::from(frame) - i128::from(origin)) * scale + local;
                    let expected = numerator.div_euclid(edge);
                    let actual = coordinate(
                        local as f64 / scale as f64,
                        frame as f64,
                        origin as f64,
                        edge as f64 / scale as f64,
                    );
                    if expected.abs() < (1i128 << 52) {
                        assert_eq!(
                            actual.unwrap(),
                            expected as i64,
                            "frame={frame} origin={origin} local={local} edge={edge}"
                        );
                    } else {
                        assert_eq!(actual, Err(SoilStateError::UnrepresentablePosition));
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 3960);
}

#[test]
fn split_coordinates_cover_subnormal_edges_and_tile_limits() {
    let tiny = f64::from_bits(1);
    for n in -17..=17 {
        assert_eq!(coordinate(n as f64 * tiny, 0.0, 0.0, tiny).unwrap(), n);
    }
    let limit = (1u64 << 52) as f64;
    assert_eq!(
        coordinate(-0.25, limit, 0.0, 1.0).unwrap(),
        (1i64 << 52) - 1
    );
    assert_eq!(
        coordinate(0.25, -limit, 0.0, 1.0),
        Err(SoilStateError::UnrepresentablePosition)
    );
    assert_eq!(
        coordinate(1.0, -limit, 0.0, 1.0).unwrap(),
        -(1i64 << 52) + 1
    );
    assert_eq!(
        coordinate(0.0, limit, 0.0, 1.0),
        Err(SoilStateError::UnrepresentablePosition)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (local, frame, origin) in [(bad, 0.0, 0.0), (0.0, bad, 0.0), (0.0, 0.0, bad)] {
            assert_eq!(
                coordinate(local, frame, origin, 1.0),
                Err(SoilStateError::UnrepresentablePosition)
            );
        }
    }
    assert_eq!(
        coordinate(f64::MAX, f64::MAX, 0.0, 1.0),
        Err(SoilStateError::UnrepresentablePosition)
    );
    assert_eq!(
        coordinate(1.5e308, 0.0, 0.0, 1e308),
        Err(SoilStateError::UnrepresentablePosition)
    );
}

#[test]
fn split_coordinates_preserve_membership_under_exact_rebasing() {
    let layout = SoilTileLayout::new([1e12, -1e12, 2e12], 0.125).unwrap();
    for i in -257..=257 {
        let local = [i as f64 / 1024.0, i as f64 / 2048.0, -i as f64 / 4096.0];
        let frame = [1e12, -1e12, 2e12];
        let expected = layout.tile_at_local(local, frame).unwrap();
        for delta in [-16.0, -0.5, 0.0, 0.5, 16.0] {
            assert_eq!(
                layout
                    .tile_at_local(local.map(|v| v - delta), frame.map(|v| v + delta))
                    .unwrap(),
                expected
            );
        }
    }
}
