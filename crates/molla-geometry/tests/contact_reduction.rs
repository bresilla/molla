//! Stage 28 — contact-reduction integration tests.
//!
//! Cross-checks the canonical "8 contacts → 4 manifold" reducer and
//! the temporal-coherence path that warm-starts friction across
//! frames.

use molla_geometry::contact_reduction::{Contact, reduce_contacts, reduce_with_history};
use molla_math::Vec3;

fn c(a: u32, b: u32, id: u64, point: Vec3, depth: f64) -> Contact {
    Contact {
        a_shape: a,
        b_shape: b,
        id,
        point,
        normal: Vec3::Y,
        depth,
    }
}

#[test]
fn eight_contacts_one_pair_reduce_to_four_canonical() {
    // 8 contacts on the same shape pair, scattered over a 1×1 patch
    // in the XZ plane. The reducer should pick:
    //   1. The deepest contact (verbatim).
    //   2. Three more that span the patch — minimum pair distance > 0.
    let cs = vec![
        c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.010),
        c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.012),
        c(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.011),
        c(0, 1, 4, Vec3::new(1.0, 0.0, 1.0), 0.090), // king
        c(0, 1, 5, Vec3::new(0.5, 0.0, 0.5), 0.005),
        c(0, 1, 6, Vec3::new(0.2, 0.0, 0.7), 0.005),
        c(0, 1, 7, Vec3::new(0.7, 0.0, 0.3), 0.005),
        c(0, 1, 8, Vec3::new(0.4, 0.0, 0.1), 0.005),
    ];
    let kept = reduce_contacts(&cs, 4);
    assert_eq!(kept.len(), 4, "must reduce to exactly 4 contacts");

    // Deepest must be first (Newton's "deepest survives" guarantee).
    assert_eq!(kept[0].id, 4, "deepest king must be kept first");
    assert!(
        (kept[0].depth - 0.090).abs() < 1e-12,
        "deepest depth preserved verbatim"
    );

    // The patch should be spanned: every pair of kept contacts is at
    // least 0.5 apart (the midpoint cluster is collapsed).
    for i in 0..kept.len() {
        for j in (i + 1)..kept.len() {
            let d = (kept[i].point - kept[j].point).length();
            assert!(
                d > 0.49,
                "kept[{i}] = {:?} and kept[{j}] = {:?} only {d} apart \
                 → reducer collapsed instead of spreading",
                kept[i].point,
                kept[j].point,
            );
        }
    }
}

#[test]
fn reducer_passthrough_when_under_cap() {
    let cs = vec![
        c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.01),
        c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.02),
        c(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.03),
    ];
    let kept = reduce_contacts(&cs, 4);
    assert_eq!(kept.len(), 3);
    // Even under cap, the deepest is reordered to the front.
    assert_eq!(kept[0].id, 3, "deepest comes first even under cap");
}

#[test]
fn reducer_groups_pairs_independently() {
    // Two unrelated pairs, 4 contacts each. cap=2 → 4 total.
    let cs = vec![
        c(0, 1, 11, Vec3::new(0.0, 0.0, 0.0), 0.01),
        c(0, 1, 12, Vec3::new(1.0, 0.0, 0.0), 0.02),
        c(0, 1, 13, Vec3::new(0.0, 0.0, 1.0), 0.03),
        c(0, 1, 14, Vec3::new(1.0, 0.0, 1.0), 0.04),
        c(2, 3, 21, Vec3::new(5.0, 0.0, 0.0), 0.05),
        c(2, 3, 22, Vec3::new(6.0, 0.0, 0.0), 0.04),
        c(2, 3, 23, Vec3::new(5.0, 0.0, 1.0), 0.03),
        c(2, 3, 24, Vec3::new(6.0, 0.0, 1.0), 0.02),
    ];
    let kept = reduce_contacts(&cs, 2);
    assert_eq!(kept.len(), 4, "2 pairs × 2 cap = 4 kept");
    let pair01_count = kept.iter().filter(|c| c.a_shape == 0).count();
    let pair23_count = kept.iter().filter(|c| c.a_shape == 2).count();
    assert_eq!(pair01_count, 2);
    assert_eq!(pair23_count, 2);
}

#[test]
fn dense_contact_patch_reduces_to_deepest_spanning_quad() {
    // Dense 11×11 patch on one shape pair. This is the hard-contact
    // stress case from PLAN Tier 4.1: the reducer must not feed a solver
    // hundreds of redundant face contacts, and the four survivors must
    // still span the patch.
    let mut contacts = Vec::new();
    let mut id = 1_u64;
    for iz in 0..11 {
        for ix in 0..11 {
            let x = -1.0 + ix as f64 * 0.2;
            let z = -1.0 + iz as f64 * 0.2;
            let r2 = x * x + z * z;
            let depth = 0.01 + 0.002 * (2.0 - r2).max(0.0);
            contacts.push(c(4, 7, id, Vec3::new(x, 0.0, z), depth));
            id += 1;
        }
    }
    // One intentionally deepest off-centre point should survive verbatim.
    contacts.push(c(4, 7, 9_999, Vec3::new(0.1, 0.0, 0.1), 0.10));

    let kept = reduce_contacts(&contacts, 4);
    assert_eq!(kept.len(), 4, "dense patch must reduce to 4 contacts");
    assert_eq!(kept[0].id, 9_999, "deepest contact must survive first");
    assert!((kept[0].depth - 0.10).abs() < 1.0e-12);

    let min_x = kept.iter().map(|p| p.point.x).fold(f64::INFINITY, f64::min);
    let max_x = kept
        .iter()
        .map(|p| p.point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_z = kept.iter().map(|p| p.point.z).fold(f64::INFINITY, f64::min);
    let max_z = kept
        .iter()
        .map(|p| p.point.z)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_x - min_x > 1.5,
        "reduced dense patch did not span X: kept={kept:?}"
    );
    assert!(
        max_z - min_z > 1.5,
        "reduced dense patch did not span Z: kept={kept:?}"
    );
}

#[test]
fn temporal_coherence_survives_a_subset_of_frames() {
    // Frame 1: 4 contacts on the corners of a unit square.
    let frame1 = vec![
        c(0, 1, 1001, Vec3::new(0.0, 0.0, 0.0), 0.01),
        c(0, 1, 1002, Vec3::new(1.0, 0.0, 0.0), 0.01),
        c(0, 1, 1003, Vec3::new(0.0, 0.0, 1.0), 0.01),
        c(0, 1, 1004, Vec3::new(1.0, 0.0, 1.0), 0.01),
    ];

    // Frame 2: same 4 corners with sub-millimetre drift, 4 mid-edge
    // points added. With cap=4 the reducer keeps the corners (max
    // spread); the matcher should attach all 4 prev ids to them.
    let frame2 = vec![
        c(0, 1, 0, Vec3::new(0.0008, 0.0, 0.0), 0.011),
        c(0, 1, 0, Vec3::new(1.0001, 0.0, 0.0), 0.012),
        c(0, 1, 0, Vec3::new(0.0, 0.0, 1.0007), 0.011),
        c(0, 1, 0, Vec3::new(0.9998, 0.0, 1.0009), 0.013),
        c(0, 1, 0, Vec3::new(0.5, 0.0, 0.0), 0.005),
        c(0, 1, 0, Vec3::new(0.5, 0.0, 1.0), 0.005),
        c(0, 1, 0, Vec3::new(0.0, 0.0, 0.5), 0.005),
        c(0, 1, 0, Vec3::new(1.0, 0.0, 0.5), 0.005),
    ];

    // First call: warm-start frame2 against frame1.
    let reduced2 = reduce_with_history(&frame1, &frame2, 4, 0.05);
    assert_eq!(reduced2.len(), 4);
    let inherited2: Vec<u64> = reduced2.iter().map(|c| c.id).filter(|&i| i != 0).collect();
    assert_eq!(
        inherited2.len(),
        4,
        "all 4 reduced contacts of frame2 should inherit; got ids {:?}",
        reduced2.iter().map(|c| c.id).collect::<Vec<_>>()
    );
    let mut sorted = inherited2.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted, vec![1001, 1002, 1003, 1004]);

    // Frame 3: same shape, slightly more drift. Pass frame2 (post-
    // reduction) as the prev table — IDs should still survive.
    let frame3 = vec![
        c(0, 1, 0, Vec3::new(0.0015, 0.0, 0.0), 0.012),
        c(0, 1, 0, Vec3::new(1.0010, 0.0, 0.0), 0.013),
        c(0, 1, 0, Vec3::new(0.0010, 0.0, 1.0015), 0.012),
        c(0, 1, 0, Vec3::new(0.9985, 0.0, 1.0020), 0.014),
    ];
    let reduced3 = reduce_with_history(&reduced2, &frame3, 4, 0.05);
    assert_eq!(reduced3.len(), 4);
    let inherited3: Vec<u64> = reduced3.iter().map(|c| c.id).filter(|&i| i != 0).collect();
    let mut sorted3 = inherited3.clone();
    sorted3.sort_unstable();
    sorted3.dedup();
    assert_eq!(
        sorted3,
        vec![1001, 1002, 1003, 1004],
        "frame3 should still inherit the original ids via frame2"
    );
}

#[test]
fn temporal_coherence_drops_unmatched_when_drift_exceeds_radius() {
    let prev = vec![c(0, 1, 999, Vec3::ZERO, 0.01)];
    let cur = vec![c(0, 1, 7, Vec3::new(10.0, 0.0, 0.0), 0.01)];
    let kept = reduce_with_history(&prev, &cur, 4, 0.05);
    assert_eq!(kept.len(), 1);
    assert_eq!(
        kept[0].id, 7,
        "drift > match_radius → keep our own id, don't inherit"
    );
}
