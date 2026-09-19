//! Broad-phase pair generation. Two flavours: a brute-force O(N²) pass
//! kept for small-scene tests / parity, and Sweep-and-Prune (SAP) which
//! is the default for the simulation pipeline.
//!
//! SAP sorts shape AABBs by the axis with the highest variance, then
//! sweeps once and emits any pair whose `[min, max]` intervals overlap
//! on that axis (cheap precondition) and whose 3D AABBs intersect
//! (cheap final check). The result matches brute-force exactly.
//!
//! Half-space planes have an infinite AABB by construction — they are
//! emitted against every non-static shape regardless of axis.

use molla_math::{Real, Vec3};
use molla_sim::{Model, ShapeGeometry, State};

/// Pair-level filter: returns true if the broad phase should keep
/// the pair `(i, j)`. Combines static-shape, self-collision, world,
/// collision-group bitmask, and explicit-exclusion checks.
#[inline]
pub fn pair_passes_filters(
    model: &Model,
    shape_body: &[i32],
    body_world: &[i32],
    i: usize,
    j: usize,
) -> bool {
    let bi = shape_body[i];
    let bj = shape_body[j];
    if let Ok(flags) = model.shape_flags.host()
        && (flags[i] & molla_sim::ShapeFlags::COLLIDES == 0
            || flags[j] & molla_sim::ShapeFlags::COLLIDES == 0)
    {
        return false;
    }
    if bi < 0 && bj < 0 {
        return false;
    }
    if bi >= 0 && bi == bj {
        return false;
    }
    let wi = if bi < 0 { -1 } else { body_world[bi as usize] };
    let wj = if bj < 0 { -1 } else { body_world[bj as usize] };
    if wi >= 0 && wj >= 0 && wi != wj {
        return false;
    }
    // Group bitmask.
    if let Ok(groups) = model.shape_collision_group.host() {
        let gi = groups.get(i).copied().unwrap_or(0xffff_ffff);
        let gj = groups.get(j).copied().unwrap_or(0xffff_ffff);
        let masks = model
            .shape_collision_mask
            .host()
            .expect("shape_collision_mask");
        if gi & masks[j] == 0 || gj & masks[i] == 0 {
            return false;
        }
    }
    // Explicit exclusion list (sorted, so binary search).
    let pairs = &model.shape_collision_filter_pairs;
    if !pairs.is_empty() {
        let lo = i.min(j) as u32;
        let hi = i.max(j) as u32;
        if pairs.binary_search(&(lo, hi)).is_ok() {
            return false;
        }
    }
    true
}

use crate::aabb::Aabb;
use crate::shape_world::shape_world_aabb;

/// One candidate pair from the broad phase: `(shape_a, shape_b)` with
/// `shape_a < shape_b`. The narrow phase consumes this list to produce
/// real contacts.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ShapePair {
    pub a: u32,
    pub b: u32,
}

/// Cached scratch state for the broad phase. Reuse one instance across
/// frames to avoid reallocating the per-shape AABB cache + sort indices.
#[derive(Debug, Default)]
pub struct BroadPhase {
    aabbs: Vec<Aabb>,
    /// Indices of finite-AABB shapes, sorted by `aabbs[i].min[axis]`.
    finite_idx: Vec<u32>,
    /// Indices of infinite-AABB shapes (planes). They get paired against
    /// every non-static finite shape.
    infinite_idx: Vec<u32>,
}

impl BroadPhase {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sweep-and-prune pair generation. Output appears in `out`,
    /// already deduplicated and with `a < b`.
    pub fn sap_pairs(
        &mut self,
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

        self.aabbs.clear();
        self.aabbs.reserve(n);
        for i in 0..n {
            self.aabbs.push(shape_world_aabb(model, state, i, margin));
        }

        // Partition: planes (infinite AABB) vs finite shapes.
        self.finite_idx.clear();
        self.infinite_idx.clear();
        for i in 0..n {
            if matches!(model.shape_geometry[i], ShapeGeometry::Plane { .. }) {
                self.infinite_idx.push(i as u32);
            } else {
                self.finite_idx.push(i as u32);
            }
        }

        if self.finite_idx.is_empty() {
            return;
        }

        // Pick the axis with the largest variance of centroid positions
        // — that minimises sweep collisions on average. Cheap to
        // estimate using min+max range.
        let axis = best_sweep_axis(&self.aabbs, &self.finite_idx);

        // Sort finite shape indices by AABB min on the chosen axis.
        self.finite_idx.sort_by(|&a, &b| {
            let ma = aabb_axis(&self.aabbs[a as usize], axis, true);
            let mb = aabb_axis(&self.aabbs[b as usize], axis, true);
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sweep — for each shape, advance until the trailing shape's
        // max < this shape's min on the sweep axis, then test the AABB
        // against everything in the active set.
        let m = self.finite_idx.len();
        for ii in 0..m {
            let i = self.finite_idx[ii] as usize;
            let aabb_i = self.aabbs[i];
            let bi = shape_body[i];
            let wi = if bi < 0 { -1 } else { body_world[bi as usize] };

            for jj in (ii + 1)..m {
                let j = self.finite_idx[jj] as usize;
                let aabb_j = self.aabbs[j];

                // Sweep prune: if j's min on axis > i's max, all later
                // shapes are also non-overlapping (since j..m is sorted).
                if aabb_axis(&aabb_j, axis, true) > aabb_axis(&aabb_i, axis, false) {
                    break;
                }

                if !pair_passes_filters(model, shape_body, body_world, i, j) {
                    continue;
                }
                if !aabb_i.intersects(&aabb_j) {
                    continue;
                }
                // Suppress unused-vars warnings in this branch.
                let _ = (bi, wi);
                let (a, b) = if (i as u32) < (j as u32) {
                    (i as u32, j as u32)
                } else {
                    (j as u32, i as u32)
                };
                out.push(ShapePair { a, b });
            }
        }

        // Pair every infinite-AABB shape (planes) against every finite
        // shape that's not static. Cheap O(P · F) — there's typically
        // one plane.
        for &p in &self.infinite_idx {
            for &f in &self.finite_idx {
                if !pair_passes_filters(model, shape_body, body_world, p as usize, f as usize) {
                    continue;
                }
                let (a, b) = if p < f { (p, f) } else { (f, p) };
                out.push(ShapePair { a, b });
            }
        }
    }
}

/// Brute-force O(N²) AABB-overlap pair generator. Kept for parity tests
/// and small scenes. Production code should call [`BroadPhase::sap_pairs`].
pub fn brute_force_pairs(model: &Model, state: &State, margin: Real, out: &mut Vec<ShapePair>) {
    out.clear();
    let n = model.shape_count;
    if n < 2 {
        return;
    }

    let shape_body = model.shape_body.host().expect("shape_body");
    let body_world = model.body_world.host().expect("body_world");

    let aabbs: Vec<_> = (0..n)
        .map(|i| shape_world_aabb(model, state, i, margin))
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if !pair_passes_filters(model, shape_body, body_world, i, j) {
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

/// Centroid range along each world axis. Finite AABBs only.
fn best_sweep_axis(aabbs: &[Aabb], finite_idx: &[u32]) -> usize {
    let mut min = Vec3::splat(Real::INFINITY);
    let mut max = Vec3::splat(-Real::INFINITY);
    for &k in finite_idx {
        let c = aabbs[k as usize];
        let mid = (c.min + c.max) * 0.5;
        min = min.min(mid);
        max = max.max(mid);
    }
    let r = max - min;
    if r.x >= r.y && r.x >= r.z {
        0
    } else if r.y >= r.z {
        1
    } else {
        2
    }
}

#[inline]
fn aabb_axis(a: &Aabb, axis: usize, is_min: bool) -> Real {
    let v = if is_min { a.min } else { a.max };
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_sim::ModelBuilder;

    fn build_two_sphere_scene() -> (Model, State) {
        use molla_math::Transform;
        use molla_sim::BodyParams;
        let mut b = ModelBuilder::new();
        let body_a = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 0.0, 0.0)));
        b.add_shape_sphere(body_a, 0.5, Transform::IDENTITY)
            .unwrap();
        let body_b = b.add_body(BodyParams::new().with_position(Vec3::new(2.0, 0.0, 0.0)));
        b.add_shape_sphere(body_b, 0.5, Transform::IDENTITY)
            .unwrap();
        let body_c = b.add_body(BodyParams::new().with_position(Vec3::new(0.6, 0.0, 0.0)));
        b.add_shape_sphere(body_c, 0.5, Transform::IDENTITY)
            .unwrap();
        let m = b.build().unwrap();
        let s = m.state().unwrap();
        (m, s)
    }

    #[test]
    fn sap_matches_brute_force_on_small_scene() {
        let (model, state) = build_two_sphere_scene();
        let mut bf = Vec::new();
        brute_force_pairs(&model, &state, 0.0, &mut bf);

        let mut sap = BroadPhase::new();
        let mut sap_out = Vec::new();
        sap.sap_pairs(&model, &state, 0.0, &mut sap_out);

        let mut bf_sorted: Vec<_> = bf.iter().map(|p| (p.a, p.b)).collect();
        let mut sap_sorted: Vec<_> = sap_out.iter().map(|p| (p.a, p.b)).collect();
        bf_sorted.sort();
        sap_sorted.sort();
        assert_eq!(bf_sorted, sap_sorted);
    }
}
