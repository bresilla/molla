//! Brute-force broad phase. Generates every shape-pair whose AABBs
//! overlap. SAP follows once N starts hurting.

use crate::math::Real;
use crate::sim::{Model, State};

use super::shape_world::shape_world_aabb;

/// One candidate pair from the broad phase: `(shape_a, shape_b)` with
/// `shape_a < shape_b`. The narrow phase consumes this list to produce
/// real contacts.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ShapePair {
    pub a: u32,
    pub b: u32,
}

/// O(N²) AABB-overlap pair generator. Honest about the complexity in
/// the name. Good enough up to a few hundred shapes; we'll swap in
/// SAP / BVH when benches start hurting.
///
/// Filters:
/// - Skips pairs where both shapes are static (both `shape_body == -1`)
///   — static-static pairs never produce useful contacts.
/// - Skips pairs from disjoint worlds (per-body world index).
/// - Always-emits pairs against half-space planes (their AABB is
///   infinite, so the AABB check would be vacuous; we'd still need
///   the narrow-phase check).
pub fn brute_force_pairs(
    model: &Model,
    state: &State,
    margin: Real,
    out: &mut Vec<ShapePair>,
) {
    out.clear();
    let n = model.shape_count;
    if n < 2 {
        return;
    }

    let shape_body = model.shape_body.host().expect("shape_body");
    let body_world = model.body_world.host().expect("body_world");

    // Pre-compute all AABBs once; halves the recomputation cost.
    let aabbs: Vec<_> = (0..n)
        .map(|i| shape_world_aabb(model, state, i, margin))
        .collect();

    for i in 0..n {
        let bi = shape_body[i];
        let wi = if bi < 0 { -1 } else { body_world[bi as usize] };

        for j in (i + 1)..n {
            let bj = shape_body[j];

            // Both static — no relative motion, no contact to add.
            if bi < 0 && bj < 0 {
                continue;
            }

            // Self-collision filter: same body, skip.
            if bi >= 0 && bi == bj {
                continue;
            }

            // World filter — both shapes must live in the same world,
            // or one of them must be global (-1 world index).
            let wj = if bj < 0 { -1 } else { body_world[bj as usize] };
            if wi >= 0 && wj >= 0 && wi != wj {
                continue;
            }

            if aabbs[i].intersects(&aabbs[j]) {
                out.push(ShapePair {
                    a: i as u32,
                    b: j as u32,
                });
            }
        }
    }
}
