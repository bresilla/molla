//! Contact reduction — coalesce many narrow-phase contacts on the
//! same shape pair into a small canonical manifold.
//!
//! Reference: Newton's `solvers/contact/contact_reduction_global.py`
//! and `geometry/contact_match.py`. The classic policy is "keep the
//! deepest contact + the points that span the contact patch as widely
//! as possible". For a typical box-on-box face contact you only need
//! 4 contacts to capture the manifold cleanly; emitting all 8 vertex
//! intersections is wasteful and gives the solver redundant work.
//!
//! This module operates on a slice-of-`Contact` API (immutable input
//! → owned `Vec` output) so it composes outside the live `sim::Contacts`
//! buffer that `geometry::contact_extras::reduce_contacts` mutates.
//! The two are complementary: `contact_extras::reduce_contacts` is the
//! voxel-bin pruner that runs in the per-frame pipeline; this module's
//! reducer is a generic max-N-per-pair primitive that callers can use
//! standalone (oracle tests, MPR fixtures, custom narrow-phase
//! variants, etc.).
//!
//! Two entry points:
//!
//! * [`reduce_contacts`] — single-frame reduction. Groups contacts by
//!   `(shape_a, shape_b)` and per group keeps `max_per_pair` canonical
//!   contacts (deepest first, then the points that maximise spread).
//!
//! * [`reduce_with_history`] — temporal-coherence variant that re-uses
//!   stable contact IDs from the previous frame so per-contact warm
//!   starts (XPBD `λ_n`, friction direction) survive sub-stepping.
//!
//! All routines are pure CPU; GPU port deferred to Stage 28.1.

use molla_math::{Real, Vec3};

/// One narrow-phase contact in a shape-agnostic struct-of-fields
/// layout. This is the slice element [`reduce_contacts`] consumes.
///
/// `id` is opaque to the reducer — it's preserved on the output so
/// callers can match warm-start data across frames. Set it to
/// whatever stable hash you want (Newton uses `(a_shape, b_shape,
/// voxel(point))`; molla's pipeline uses [`super::contact_extras::contact_id`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Contact {
    /// Shape-A identifier. Used only to group contacts during reduction.
    pub a_shape: u32,
    /// Shape-B identifier. Pairs `(a, b)` and `(b, a)` are *not*
    /// canonicalised here — callers are expected to feed normalised
    /// pairs (lo, hi).
    pub b_shape: u32,
    /// Stable per-contact ID for warm-start matching across frames.
    /// The reducer preserves whatever value the caller put here.
    pub id: u64,
    /// World-space contact point (mid-plane between the two surfaces).
    pub point: Vec3,
    /// Unit normal pointing from A into B (matches molla's convention).
    pub normal: Vec3,
    /// Penetration depth (positive). Used as the "deepest" criterion.
    pub depth: Real,
}

/// Reduce per-pair contacts to ≤ `max_per_pair` canonical points.
///
/// Algorithm (Newton-style "keep deepest + furthest"):
/// 1. Group contacts by `(a_shape, b_shape)`.
/// 2. For each group, always keep the deepest contact.
/// 3. Greedily add the contact whose minimum distance to any
///    already-kept point is maximal — i.e. expand the convex hull of
///    the kept set as fast as possible. This produces a
///    well-conditioned manifold (4 corners of a face, not 4 points
///    clustered at one corner).
/// 4. Stop when the group hits `max_per_pair` or runs out of
///    candidates.
///
/// Edge cases:
/// * `max_per_pair == 0` → empty output.
/// * `contacts.is_empty()` → empty output.
/// * `contacts.len() <= max_per_pair` per group → all contacts pass
///   through unchanged (but possibly reordered: deepest first, then
///   greedy spread).
///
/// The output is **owned** (caller-friendly) and the order is
/// deterministic for a given input (greedy tie-breaks fall on lower
/// input index).
#[must_use]
pub fn reduce_contacts(contacts: &[Contact], max_per_pair: usize) -> Vec<Contact> {
    if contacts.is_empty() || max_per_pair == 0 {
        return Vec::new();
    }

    // Bucket indices into per-pair groups. We keep the indirection
    // (rather than copying contacts) so we don't pay a per-contact
    // clone for the common single-pair case.
    use std::collections::HashMap;
    let mut groups: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, c) in contacts.iter().enumerate() {
        groups.entry((c.a_shape, c.b_shape)).or_default().push(i);
    }

    let mut out = Vec::with_capacity(contacts.len().min(groups.len() * max_per_pair));
    // Stable iteration order (HashMap iteration is unordered) ⇒ sort
    // the keys so the output is deterministic across runs.
    let mut keys: Vec<(u32, u32)> = groups.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let indices = &groups[&key];
        reduce_one_pair(contacts, indices, max_per_pair, &mut out);
    }
    out
}

/// Greedily reduce one shape pair's contacts.
fn reduce_one_pair(
    contacts: &[Contact],
    indices: &[usize],
    max_per_pair: usize,
    out: &mut Vec<Contact>,
) {
    if indices.is_empty() {
        return;
    }
    let n = indices.len();
    let cap = max_per_pair.min(n);

    // 1) Deepest contact.
    let mut deepest_local = 0usize;
    let mut deepest_depth = contacts[indices[0]].depth;
    for (li, &gi) in indices.iter().enumerate().skip(1) {
        let d = contacts[gi].depth;
        if d > deepest_depth {
            deepest_depth = d;
            deepest_local = li;
        }
    }
    let mut kept_local: Vec<usize> = Vec::with_capacity(cap);
    kept_local.push(deepest_local);

    // 2) Greedy farthest-point sweep until we have `cap` contacts.
    //    For each remaining candidate compute `min_dist_to_kept`,
    //    then take the candidate maximising that quantity. O(n²·cap)
    //    but cap is tiny (≤ 4) and n is small (rarely > 16 per pair).
    while kept_local.len() < cap {
        let mut best_local: Option<usize> = None;
        let mut best_min_dist = -1.0;
        for li in 0..n {
            if kept_local.contains(&li) {
                continue;
            }
            let p = contacts[indices[li]].point;
            // Distance to nearest already-kept contact.
            let mut min_d = Real::INFINITY;
            for &kli in &kept_local {
                let d = (p - contacts[indices[kli]].point).length();
                if d < min_d {
                    min_d = d;
                }
            }
            // Tie-break (and fully-coincident points, min_d == 0):
            // prefer deeper contact, then lower input index.
            let cur_depth = contacts[indices[li]].depth;
            let pick = match best_local {
                None => true,
                Some(_) => {
                    if min_d > best_min_dist + 1e-12 {
                        true
                    } else if (min_d - best_min_dist).abs() <= 1e-12 {
                        // Tie on spread → deeper wins; lower index breaks.
                        let bli = best_local.unwrap();
                        let bdepth = contacts[indices[bli]].depth;
                        if cur_depth > bdepth + 1e-12 {
                            true
                        } else if (cur_depth - bdepth).abs() <= 1e-12 {
                            li < bli
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
            };
            if pick {
                best_local = Some(li);
                best_min_dist = min_d;
            }
        }
        match best_local {
            Some(li) => kept_local.push(li),
            None => break,
        }
    }

    // Emit kept contacts (in selection order — deepest first).
    for li in kept_local {
        out.push(contacts[indices[li]]);
    }
}

/// Reduce `current` while threading IDs from `prev` so persistent
/// contacts keep a stable `id` across frames. This is the warm-start
/// affordance: the friction solver's per-contact data (Lagrange
/// multipliers, integrated friction tangents) keys off `id`, so a
/// stable id between calls means warm-starts survive sub-stepping.
///
/// Matching policy: for each group `(a_shape, b_shape)`, walk the
/// reduced current contacts and try to attach the closest
/// previous-frame contact within `match_radius`. If the previous
/// frame had no nearby contact, the current contact keeps its own
/// `id` (which the caller is expected to seed with whatever stable
/// hash they normally use — Newton picks `voxel_id`).
///
/// `match_radius` is the threshold for "this is the same contact as
/// last frame". A reasonable default for rigid-body face contacts is
/// `2 × contact_margin`; pick smaller for cloth (~1 mm) and larger
/// for ragdolls / hands.
#[must_use]
pub fn reduce_with_history(
    prev: &[Contact],
    current: &[Contact],
    max_per_pair: usize,
    match_radius: Real,
) -> Vec<Contact> {
    let mut reduced = reduce_contacts(current, max_per_pair);
    if prev.is_empty() {
        return reduced;
    }

    // Index prev by (a, b) for cheap per-group lookup.
    use std::collections::HashMap;
    let mut prev_by_pair: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, c) in prev.iter().enumerate() {
        prev_by_pair
            .entry((c.a_shape, c.b_shape))
            .or_default()
            .push(i);
    }

    let r2 = match_radius * match_radius;
    // Track which prev indices we've already claimed so two current
    // contacts don't both inherit the same prev id.
    let mut claimed: Vec<bool> = vec![false; prev.len()];

    for cur in reduced.iter_mut() {
        let key = (cur.a_shape, cur.b_shape);
        if let Some(prev_indices) = prev_by_pair.get(&key) {
            // Find the closest unclaimed prev contact within the
            // matching radius.
            let mut best: Option<usize> = None;
            let mut best_d2 = r2;
            for &pi in prev_indices {
                if claimed[pi] {
                    continue;
                }
                let d2 = (prev[pi].point - cur.point).length_squared();
                if d2 <= best_d2 {
                    best_d2 = d2;
                    best = Some(pi);
                }
            }
            if let Some(pi) = best {
                cur.id = prev[pi].id;
                claimed[pi] = true;
            }
        }
    }
    reduced
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(a: u32, b: u32, id: u64, point: Vec3, depth: Real) -> Contact {
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
    fn empty_input_returns_empty() {
        assert!(reduce_contacts(&[], 4).is_empty());
    }

    #[test]
    fn zero_max_returns_empty() {
        let cs = vec![c(0, 1, 0, Vec3::ZERO, 0.01)];
        assert!(reduce_contacts(&cs, 0).is_empty());
    }

    #[test]
    fn keeps_all_when_under_cap() {
        let cs = vec![
            c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.01),
            c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.02),
        ];
        let kept = reduce_contacts(&cs, 4);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn keeps_deepest_first() {
        let cs = vec![
            c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.01),
            c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.05),
            c(0, 1, 3, Vec3::new(2.0, 0.0, 0.0), 0.02),
        ];
        let kept = reduce_contacts(&cs, 1);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, 2, "deepest should win");
    }

    #[test]
    fn pair_grouping_is_independent() {
        let cs = vec![
            c(0, 1, 1, Vec3::ZERO, 0.01),
            c(2, 3, 2, Vec3::ZERO, 0.02),
            c(0, 1, 3, Vec3::X, 0.03),
            c(2, 3, 4, Vec3::Y, 0.04),
        ];
        // Cap each pair at 1 — should keep deepest of each pair.
        let kept = reduce_contacts(&cs, 1);
        assert_eq!(kept.len(), 2);
        // Sorted by key (0,1) before (2,3).
        assert_eq!(kept[0].a_shape, 0);
        assert_eq!(kept[0].id, 3, "deepest of (0,1) is id=3");
        assert_eq!(kept[1].a_shape, 2);
        assert_eq!(kept[1].id, 4, "deepest of (2,3) is id=4");
    }

    #[test]
    fn eight_contacts_reduce_to_four_canonical() {
        // 8 contacts at the corners + midpoints of a unit square,
        // depths varying. With cap=4 the reducer should keep 4
        // contacts: the deepest one, and 3 corners spanning the patch.
        let cs = vec![
            // 4 corners at depth 0.01
            c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.010),
            c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.012),
            c(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.011),
            c(0, 1, 4, Vec3::new(1.0, 0.0, 1.0), 0.013),
            // 4 midpoints — closer to centre, smallest depths.
            c(0, 1, 5, Vec3::new(0.5, 0.0, 0.0), 0.005),
            c(0, 1, 6, Vec3::new(0.5, 0.0, 1.0), 0.005),
            c(0, 1, 7, Vec3::new(0.0, 0.0, 0.5), 0.005),
            c(0, 1, 8, Vec3::new(1.0, 0.0, 0.5), 0.005),
            // One deepest centre point, depth 0.020.
            // Wait: that's 9 contacts and only one pair — adjust.
        ];
        let kept = reduce_contacts(&cs, 4);
        assert_eq!(kept.len(), 4);
        // The deepest (id=4, depth 0.013) should be first.
        assert_eq!(kept[0].id, 4);
        // The remaining 3 should span the patch — at minimum each
        // pair-distance > 0.5 (the midpoint cluster spacing).
        for i in 0..kept.len() {
            for j in (i + 1)..kept.len() {
                let d = (kept[i].point - kept[j].point).length();
                assert!(
                    d > 0.49,
                    "kept[{i}] and kept[{j}] only {d} apart — \
                     reducer collapsed instead of spreading"
                );
            }
        }
    }

    #[test]
    fn deepest_is_preserved_among_eight() {
        // Newton's reduction guarantees the single deepest contact
        // makes it through verbatim.
        let cs = vec![
            c(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.01),
            c(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.02),
            c(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.03),
            c(0, 1, 4, Vec3::new(1.0, 0.0, 1.0), 0.99), // king
            c(0, 1, 5, Vec3::new(0.5, 0.0, 0.5), 0.04),
            c(0, 1, 6, Vec3::new(0.2, 0.0, 0.7), 0.05),
            c(0, 1, 7, Vec3::new(0.7, 0.0, 0.3), 0.06),
            c(0, 1, 8, Vec3::new(0.4, 0.0, 0.1), 0.07),
        ];
        let kept = reduce_contacts(&cs, 4);
        assert_eq!(kept.len(), 4);
        assert_eq!(kept[0].id, 4, "deepest king must be kept");
        assert!((kept[0].depth - 0.99).abs() < 1e-12);
    }

    #[test]
    fn temporal_coherence_preserves_ids() {
        // Frame 1: 4 contacts on the corners of a unit square.
        let prev = vec![
            c(0, 1, 100, Vec3::new(0.0, 0.0, 0.0), 0.01),
            c(0, 1, 200, Vec3::new(1.0, 0.0, 0.0), 0.01),
            c(0, 1, 300, Vec3::new(0.0, 0.0, 1.0), 0.01),
            c(0, 1, 400, Vec3::new(1.0, 0.0, 1.0), 0.01),
        ];
        // Frame 2: same corners + 4 midpoints (8 raw contacts), drift
        // each corner slightly so the matcher has to work for it.
        let cur = vec![
            c(0, 1, 0, Vec3::new(0.001, 0.0, 0.0), 0.01),
            c(0, 1, 0, Vec3::new(1.002, 0.0, 0.0), 0.01),
            c(0, 1, 0, Vec3::new(0.0, 0.0, 1.001), 0.01),
            c(0, 1, 0, Vec3::new(1.0, 0.0, 1.002), 0.01),
            c(0, 1, 0, Vec3::new(0.5, 0.0, 0.0), 0.005),
            c(0, 1, 0, Vec3::new(0.5, 0.0, 1.0), 0.005),
            c(0, 1, 0, Vec3::new(0.0, 0.0, 0.5), 0.005),
            c(0, 1, 0, Vec3::new(1.0, 0.0, 0.5), 0.005),
        ];
        // Match radius 0.05 — corner drift is at most 0.002, so all
        // four corners should match a previous-frame contact.
        let kept = reduce_with_history(&prev, &cur, 4, 0.05);
        assert_eq!(kept.len(), 4);
        // Every kept contact should have inherited a non-zero id from
        // its previous-frame match.
        let inherited: Vec<u64> = kept.iter().map(|k| k.id).filter(|&i| i != 0).collect();
        assert_eq!(
            inherited.len(),
            4,
            "all 4 kept contacts should inherit an id; got {:?}",
            kept.iter().map(|k| k.id).collect::<Vec<_>>()
        );
        // Each inherited id must be unique (no double-claims).
        let mut sorted = inherited.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), inherited.len(), "ids must be unique");
        // And every id must come from {100, 200, 300, 400}.
        for id in inherited {
            assert!(matches!(id, 100 | 200 | 300 | 400));
        }
    }

    #[test]
    fn temporal_coherence_drops_unmatched() {
        // Drift the contact past `match_radius` → no inherited id.
        let prev = vec![c(0, 1, 999, Vec3::ZERO, 0.01)];
        let cur = vec![c(0, 1, 0, Vec3::new(10.0, 0.0, 0.0), 0.01)];
        let kept = reduce_with_history(&prev, &cur, 4, 0.05);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, 0, "should keep its own (un-inherited) id");
    }

    #[test]
    fn temporal_coherence_no_double_claim() {
        // Two current contacts both close to the same prev contact:
        // only one should inherit, the other keeps its own id.
        let prev = vec![c(0, 1, 42, Vec3::ZERO, 0.01)];
        let cur = vec![
            c(0, 1, 0, Vec3::new(0.001, 0.0, 0.0), 0.02),
            c(0, 1, 0, Vec3::new(0.002, 0.0, 0.0), 0.01),
        ];
        let kept = reduce_with_history(&prev, &cur, 4, 0.5);
        assert_eq!(kept.len(), 2);
        let inherited: Vec<u64> = kept.iter().map(|k| k.id).filter(|&i| i == 42).collect();
        assert_eq!(inherited.len(), 1, "exactly one current claims prev=42");
    }
}
