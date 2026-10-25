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

use molla_math::{Real, Transform};
use molla_sim::runtime::SceneStamp;
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
    /// Shape centres along one axis, for choosing the sweep axis.
    centres: Vec<Real>,
    /// The compiled model the static shapes below were sorted for, the
    /// margin and poses they were bounded at, and the sweep axis.
    cached: Option<SceneStamp>,
    margin: Real,
    static_poses: Vec<Transform>,
    axis: usize,
    /// Static shapes in classes by their length along the sweep axis, each
    /// class sorted by `aabbs[i].min[axis]` beside its longest length, and
    /// the static shapes too long for any class.
    statics: Vec<(Real, Vec<u32>)>,
    long_statics: Vec<u32>,
    /// Shapes on bodies, sorted by `aabbs[i].min[axis]` each call.
    moving: Vec<u32>,
}

/// The longest static shape along the sweep axis, metres, of each class;
/// longer ones are tested against every moving shape.
const STATIC_CLASSES: [Real; 4] = [1.0, 4.0, 16.0, 64.0];

impl BroadPhase {
    pub(crate) fn shape_aabb(&self, shape: usize) -> Aabb {
        self.aabbs[shape]
    }

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
        self.sap_pairs_with_bounds(model, out, |i| shape_world_aabb(model, state, i, margin));
    }

    pub(crate) fn sap_pairs_prepared(
        &mut self,
        bounds: &crate::shape_world::PreparedShapeBounds<'_>,
        state: &State,
        margin: Real,
        out: &mut Vec<ShapePair>,
    ) {
        match bounds.stamp() {
            Some(stamp) => self.sap_pairs_cached(bounds, state, margin, out, stamp),
            None => self.sap_pairs_with_bounds(bounds.model(), out, |i| {
                bounds.world_aabb(state, i, margin)
            }),
        }
    }

    /// Sweep-and-prune that keeps the static shapes bounded and sorted
    /// while the scene keeps its identity: each call bounds and sorts only
    /// the shapes on bodies, sweeps them against each other, and tests each
    /// against the static shapes within reach of it along the sweep axis.
    fn sap_pairs_cached(
        &mut self,
        bounds: &crate::shape_world::PreparedShapeBounds<'_>,
        state: &State,
        margin: Real,
        out: &mut Vec<ShapePair>,
        stamp: &SceneStamp,
    ) {
        out.clear();
        let model = bounds.model();
        let n = model.shape_count;
        if n < 2 {
            return;
        }
        let shape_body = model.shape_body.host().expect("shape_body");
        let body_world = model.body_world.host().expect("body_world");
        let poses = model.shape_transform.host().expect("shape_transform");
        let unmoved = self
            .statics
            .iter()
            .flat_map(|(_, class)| class)
            .chain(&self.long_statics)
            .zip(&self.static_poses)
            .all(|(&k, pose)| poses.get(k as usize) == Some(pose));
        if self.cached.as_ref() != Some(stamp)
            || self.aabbs.len() != n
            || self.margin != margin
            || !unmoved
        {
            self.aabbs.clear();
            self.aabbs
                .extend((0..n).map(|i| bounds.world_aabb(state, i, margin)));
            self.finite_idx.clear();
            self.infinite_idx.clear();
            for i in 0..n {
                if matches!(model.shape_geometry[i], ShapeGeometry::Plane { .. }) {
                    self.infinite_idx.push(i as u32);
                } else {
                    self.finite_idx.push(i as u32);
                }
            }
            let axis = best_sweep_axis(&self.aabbs, &self.finite_idx, &mut self.centres);
            self.statics = STATIC_CLASSES.iter().map(|_| (0.0, Vec::new())).collect();
            self.long_statics.clear();
            for &k in &self.finite_idx {
                if shape_body[k as usize] >= 0 {
                    continue;
                }
                let a = &self.aabbs[k as usize];
                let length = aabb_axis(a, axis, false) - aabb_axis(a, axis, true);
                match STATIC_CLASSES.iter().position(|&longest| length <= longest) {
                    Some(c) => {
                        let (reach, class) = &mut self.statics[c];
                        *reach = reach.max(length);
                        class.push(k);
                    }
                    None => self.long_statics.push(k),
                }
            }
            for (_, class) in &mut self.statics {
                class.sort_by(|&a, &b| {
                    aabb_axis(&self.aabbs[a as usize], axis, true).total_cmp(&aabb_axis(
                        &self.aabbs[b as usize],
                        axis,
                        true,
                    ))
                });
            }
            self.static_poses.clear();
            self.static_poses.extend(
                self.statics
                    .iter()
                    .flat_map(|(_, class)| class)
                    .chain(&self.long_statics)
                    .map(|&k| poses[k as usize]),
            );
            self.axis = axis;
            self.margin = margin;
            self.cached = Some(stamp.clone());
        } else {
            for i in 0..n {
                if shape_body[i] >= 0 {
                    self.aabbs[i] = bounds.world_aabb(state, i, margin);
                }
            }
        }
        let axis = self.axis;
        self.moving.clear();
        self.moving.extend(
            self.finite_idx
                .iter()
                .copied()
                .filter(|&k| shape_body[k as usize] >= 0),
        );
        self.moving.sort_by(|&a, &b| {
            aabb_axis(&self.aabbs[a as usize], axis, true).total_cmp(&aabb_axis(
                &self.aabbs[b as usize],
                axis,
                true,
            ))
        });
        let push = |out: &mut Vec<ShapePair>, i: u32, j: u32| {
            out.push(ShapePair {
                a: i.min(j),
                b: i.max(j),
            })
        };

        for (ii, &i) in self.moving.iter().enumerate() {
            let aabb_i = self.aabbs[i as usize];
            for &j in &self.moving[ii + 1..] {
                let aabb_j = self.aabbs[j as usize];
                if aabb_axis(&aabb_j, axis, true) > aabb_axis(&aabb_i, axis, false) {
                    break;
                }
                if aabb_i.intersects(&aabb_j)
                    && pair_passes_filters(model, shape_body, body_world, i as usize, j as usize)
                {
                    push(out, i, j);
                }
            }
            let hi = aabb_axis(&aabb_i, axis, false);
            for (reach, class) in &self.statics {
                let lo = aabb_axis(&aabb_i, axis, true) - reach;
                let start =
                    class.partition_point(|&s| aabb_axis(&self.aabbs[s as usize], axis, true) < lo);
                for &s in &class[start..] {
                    let aabb_s = self.aabbs[s as usize];
                    if aabb_axis(&aabb_s, axis, true) > hi {
                        break;
                    }
                    if aabb_i.intersects(&aabb_s)
                        && pair_passes_filters(
                            model, shape_body, body_world, i as usize, s as usize,
                        )
                    {
                        push(out, i, s);
                    }
                }
            }
            for &s in &self.long_statics {
                if aabb_i.intersects(&self.aabbs[s as usize])
                    && pair_passes_filters(model, shape_body, body_world, i as usize, s as usize)
                {
                    push(out, i, s);
                }
            }
        }

        for &p in &self.infinite_idx {
            for &f in &self.finite_idx {
                if pair_passes_filters(model, shape_body, body_world, p as usize, f as usize) {
                    push(out, p, f);
                }
            }
        }
    }

    fn sap_pairs_with_bounds(
        &mut self,
        model: &Model,
        out: &mut Vec<ShapePair>,
        bounds: impl Fn(usize) -> Aabb,
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
            self.aabbs.push(bounds(i));
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

        // Sweep the axis the shapes spread furthest along, as their middle
        // half spreads: a few far-off shapes do not pick it.
        let axis = best_sweep_axis(&self.aabbs, &self.finite_idx, &mut self.centres);

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

/// The axis along which the middle half of the finite shapes' centres spreads furthest.
fn best_sweep_axis(aabbs: &[Aabb], finite_idx: &[u32], centres: &mut Vec<Real>) -> usize {
    let mut best = (0, -1.0);
    for axis in 0..3 {
        centres.clear();
        centres.extend(
            finite_idx
                .iter()
                .map(|&k| {
                    (aabb_axis(&aabbs[k as usize], axis, true)
                        + aabb_axis(&aabbs[k as usize], axis, false))
                        * 0.5
                })
                .filter(|c| c.is_finite()),
        );
        if centres.len() < 2 {
            continue;
        }
        let (q1, q3) = (centres.len() / 4, centres.len() * 3 / 4);
        let low = *centres.select_nth_unstable_by(q1, Real::total_cmp).1;
        let high = *centres.select_nth_unstable_by(q3, Real::total_cmp).1;
        if high - low > best.1 {
            best = (axis, high - low);
        }
    }
    best.0
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
    use molla_math::Vec3;
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

    #[test]
    fn the_sweep_runs_along_the_spread_not_toward_a_far_off_shape() {
        let cube = |c: Vec3| Aabb {
            min: c - Vec3::splat(0.5),
            max: c + Vec3::splat(0.5),
        };
        let mut aabbs: Vec<Aabb> = (0..40)
            .map(|k| cube(Vec3::new((k % 3) as Real, (k % 2) as Real, 4.0 * k as Real)))
            .collect();
        aabbs.push(cube(Vec3::new(0.0, -5.0e4, 0.0)));
        aabbs.push(cube(Vec3::new(0.0, 5.0e4, 0.0)));
        let all: Vec<u32> = (0..aabbs.len() as u32).collect();
        assert_eq!(
            best_sweep_axis(&aabbs, &all, &mut Vec::new()),
            2,
            "the shapes line up along Z"
        );
    }
}
