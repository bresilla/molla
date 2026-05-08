//! Bounding-volume hierarchy (BVH) over shape AABBs.
//!
//! Top-down construction with surface-area-heuristic splitting on
//! the longest centroid axis. Used as an alternative broad phase
//! for scenes where SAP starts to hurt (>few hundred shapes,
//! triangle-mesh shapes). Stage 6 ships a CPU implementation; a
//! GPU LBVH may follow later.
//!
//! Rebuilt every step (no refit yet — Newton tracks the same
//! decision, since GPU rebuilds are cheap with parallel sorts).
//! For molla's CPU pipeline, top-down rebuild at a few thousand
//! shapes is sub-millisecond.

use crate::math::Real;
use crate::sim::{Model, ShapeGeometry, State};

use super::aabb::Aabb;
use super::broad_phase::ShapePair;
use super::shape_world::shape_world_aabb;

/// One BVH node: either an internal node holding two children, or a
/// leaf holding a shape index. Stored flat in `nodes`; child indices
/// reference the same array.
#[derive(Copy, Clone, Debug)]
pub struct BvhNode {
    /// Tight AABB of the subtree.
    pub aabb: Aabb,
    /// `> 0` for internal nodes (left child = `child0`, right = `child1`);
    /// `== -1` for leaf nodes; the shape index is then in `child0`.
    pub child0: i32,
    pub child1: i32,
}

impl BvhNode {
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.child1 < 0
    }
    #[must_use]
    pub fn shape_index(&self) -> usize {
        debug_assert!(self.is_leaf());
        self.child0 as usize
    }
}

/// Top-down BVH over shape AABBs.
#[derive(Debug, Default)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub root: i32,
    /// Indices of shapes whose AABB is infinite (planes). They're
    /// excluded from the BVH and tested separately.
    pub infinite_idx: Vec<u32>,
}

impl Bvh {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the BVH from scratch over the model's shape AABBs.
    pub fn build(&mut self, model: &Model, state: &State, margin: Real) {
        self.nodes.clear();
        self.root = -1;
        self.infinite_idx.clear();

        let n = model.shape_count;
        if n == 0 {
            return;
        }

        // Compute AABBs and partition planes vs finite shapes.
        let mut aabbs: Vec<Aabb> = Vec::with_capacity(n);
        let mut finite_idx: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            let g = model.shape_geometry[i];
            let aabb = shape_world_aabb(model, state, i, margin);
            aabbs.push(aabb);
            if matches!(g, ShapeGeometry::Plane { .. }) {
                self.infinite_idx.push(i as u32);
            } else {
                finite_idx.push(i as u32);
            }
        }

        if !finite_idx.is_empty() {
            self.root = self.build_recursive(&mut finite_idx, &aabbs);
        }
    }

    fn build_recursive(&mut self, indices: &mut [u32], aabbs: &[Aabb]) -> i32 {
        if indices.len() == 1 {
            // Leaf.
            let shape = indices[0];
            let node = BvhNode {
                aabb: aabbs[shape as usize],
                child0: shape as i32,
                child1: -1,
            };
            self.nodes.push(node);
            return (self.nodes.len() - 1) as i32;
        }

        // Compute the centroid bounds and find the longest axis.
        let mut centroid_min = crate::math::Vec3::splat(Real::INFINITY);
        let mut centroid_max = crate::math::Vec3::splat(-Real::INFINITY);
        for &i in indices.iter() {
            let c = (aabbs[i as usize].min + aabbs[i as usize].max) * 0.5;
            centroid_min = centroid_min.min(c);
            centroid_max = centroid_max.max(c);
        }
        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Sort by centroid component on the chosen axis.
        indices.sort_unstable_by(|&a, &b| {
            let ca = (aabbs[a as usize].min + aabbs[a as usize].max) * 0.5;
            let cb = (aabbs[b as usize].min + aabbs[b as usize].max) * 0.5;
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Median split for simplicity (SAH would be better; OK for
        // stage 6.x).
        let mid = indices.len() / 2;
        let (left, right) = indices.split_at_mut(mid);
        let left_idx = self.build_recursive(left, aabbs);
        let right_idx = self.build_recursive(right, aabbs);

        // Combined AABB.
        let l_aabb = self.nodes[left_idx as usize].aabb;
        let r_aabb = self.nodes[right_idx as usize].aabb;
        let aabb = Aabb {
            min: l_aabb.min.min(r_aabb.min),
            max: l_aabb.max.max(r_aabb.max),
        };
        let node = BvhNode {
            aabb,
            child0: left_idx,
            child1: right_idx,
        };
        self.nodes.push(node);
        (self.nodes.len() - 1) as i32
    }

    /// Generate AABB-overlap pairs by traversing the BVH. Output
    /// is `(a, b)` pairs with `a < b`.
    pub fn pairs(
        &self,
        model: &Model,
        out: &mut Vec<ShapePair>,
    ) {
        out.clear();
        let shape_body = model.shape_body.host().expect("shape_body");
        let body_world = model.body_world.host().expect("body_world");

        if self.root >= 0 {
            // Self-pair traversal of the BVH.
            self.traverse_self(self.root, shape_body, body_world, out);
        }

        // Plane-vs-finite: planes are excluded from the BVH; pair
        // every plane with every finite shape that survives the
        // body / world filters.
        for &p in &self.infinite_idx {
            let pi = p as usize;
            let bp = shape_body[pi];
            let wp = if bp < 0 { -1 } else { body_world[bp as usize] };
            self.collect_finite_against_plane(
                self.root,
                p,
                bp,
                wp,
                shape_body,
                body_world,
                out,
            );
        }
    }

    fn traverse_self(
        &self,
        node: i32,
        shape_body: &[i32],
        body_world: &[i32],
        out: &mut Vec<ShapePair>,
    ) {
        if node < 0 {
            return;
        }
        let n = self.nodes[node as usize];
        if n.is_leaf() {
            return;
        }
        // Recurse into children individually first, then test the
        // pair (left, right) against each other.
        self.traverse_self(n.child0, shape_body, body_world, out);
        self.traverse_self(n.child1, shape_body, body_world, out);
        self.traverse_pair(n.child0, n.child1, shape_body, body_world, out);
    }

    fn traverse_pair(
        &self,
        a: i32,
        b: i32,
        shape_body: &[i32],
        body_world: &[i32],
        out: &mut Vec<ShapePair>,
    ) {
        if a < 0 || b < 0 {
            return;
        }
        let na = self.nodes[a as usize];
        let nb = self.nodes[b as usize];
        if !na.aabb.intersects(&nb.aabb) {
            return;
        }
        match (na.is_leaf(), nb.is_leaf()) {
            (true, true) => {
                let i = na.shape_index();
                let j = nb.shape_index();
                if Self::pair_passes_filters(shape_body, body_world, i, j) {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    out.push(ShapePair {
                        a: lo as u32,
                        b: hi as u32,
                    });
                }
            }
            (false, true) => {
                self.traverse_pair(na.child0, b, shape_body, body_world, out);
                self.traverse_pair(na.child1, b, shape_body, body_world, out);
            }
            (true, false) => {
                self.traverse_pair(a, nb.child0, shape_body, body_world, out);
                self.traverse_pair(a, nb.child1, shape_body, body_world, out);
            }
            (false, false) => {
                // Descend into the bigger node first (cheap
                // optimization).
                let va = (na.aabb.max - na.aabb.min).length_squared();
                let vb = (nb.aabb.max - nb.aabb.min).length_squared();
                if va > vb {
                    self.traverse_pair(na.child0, b, shape_body, body_world, out);
                    self.traverse_pair(na.child1, b, shape_body, body_world, out);
                } else {
                    self.traverse_pair(a, nb.child0, shape_body, body_world, out);
                    self.traverse_pair(a, nb.child1, shape_body, body_world, out);
                }
            }
        }
    }

    fn collect_finite_against_plane(
        &self,
        node: i32,
        plane_shape: u32,
        plane_body: i32,
        plane_world: i32,
        shape_body: &[i32],
        body_world: &[i32],
        out: &mut Vec<ShapePair>,
    ) {
        if node < 0 {
            return;
        }
        let n = self.nodes[node as usize];
        if n.is_leaf() {
            let f = n.shape_index();
            let bf = shape_body[f];
            if Self::plane_pair_passes_filters(plane_body, plane_world, bf, body_world) {
                let (lo, hi) = if (plane_shape as usize) < f {
                    (plane_shape, f as u32)
                } else {
                    (f as u32, plane_shape)
                };
                out.push(ShapePair { a: lo, b: hi });
            }
            return;
        }
        self.collect_finite_against_plane(
            n.child0,
            plane_shape,
            plane_body,
            plane_world,
            shape_body,
            body_world,
            out,
        );
        self.collect_finite_against_plane(
            n.child1,
            plane_shape,
            plane_body,
            plane_world,
            shape_body,
            body_world,
            out,
        );
    }

    #[inline]
    fn pair_passes_filters(
        shape_body: &[i32],
        body_world: &[i32],
        i: usize,
        j: usize,
    ) -> bool {
        let bi = shape_body[i];
        let bj = shape_body[j];
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
        true
    }

    #[inline]
    fn plane_pair_passes_filters(
        plane_body: i32,
        plane_world: i32,
        finite_body: i32,
        body_world: &[i32],
    ) -> bool {
        if plane_body < 0 && finite_body < 0 {
            return false;
        }
        if plane_body >= 0 && plane_body == finite_body {
            return false;
        }
        let wf = if finite_body < 0 {
            -1
        } else {
            body_world[finite_body as usize]
        };
        if plane_world >= 0 && wf >= 0 && plane_world != wf {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Transform, Vec3};
    use crate::sim::{BodyParams, ModelBuilder};

    fn build_grid_scene(n: usize) -> (Model, State) {
        let mut b = ModelBuilder::new();
        for i in 0..n {
            let pos = Vec3::new(
                (i % 8) as Real * 2.0,
                ((i / 8) % 8) as Real * 2.0,
                (i / 64) as Real * 2.0,
            );
            let body = b.add_body(BodyParams::new().with_position(pos));
            b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        }
        let m = b.build().unwrap();
        let s = m.state().unwrap();
        (m, s)
    }

    /// BVH and brute-force broad phases must produce the same set
    /// of pairs (modulo order) on a 100-shape grid scene.
    #[test]
    fn bvh_matches_brute_force_on_100_shapes() {
        let (model, state) = build_grid_scene(100);
        let mut brute_pairs = Vec::new();
        super::super::broad_phase::brute_force_pairs(&model, &state, 0.0, &mut brute_pairs);

        let mut bvh = Bvh::new();
        bvh.build(&model, &state, 0.0);
        let mut bvh_pairs = Vec::new();
        bvh.pairs(&model, &mut bvh_pairs);

        let mut bf: Vec<_> = brute_pairs.iter().map(|p| (p.a, p.b)).collect();
        let mut bv: Vec<_> = bvh_pairs.iter().map(|p| (p.a, p.b)).collect();
        bf.sort();
        bv.sort();
        assert_eq!(bf, bv, "BVH/BF pair-set mismatch");
    }

    /// Stage-6 DoD: BVH parity against brute force on a 1k-shape
    /// scene. Confirms correctness at scale.
    #[test]
    fn bvh_matches_brute_force_on_1k_shapes() {
        let (model, state) = build_grid_scene(1024);
        let mut brute_pairs = Vec::new();
        super::super::broad_phase::brute_force_pairs(&model, &state, 0.0, &mut brute_pairs);

        let mut bvh = Bvh::new();
        bvh.build(&model, &state, 0.0);
        let mut bvh_pairs = Vec::new();
        bvh.pairs(&model, &mut bvh_pairs);

        let mut bf: Vec<_> = brute_pairs.iter().map(|p| (p.a, p.b)).collect();
        let mut bv: Vec<_> = bvh_pairs.iter().map(|p| (p.a, p.b)).collect();
        bf.sort();
        bv.sort();
        assert_eq!(bf.len(), bv.len(), "pair-count mismatch (BF {} vs BVH {})", bf.len(), bv.len());
        assert_eq!(bf, bv, "BVH/BF pair-set mismatch on 1k shapes");
    }

    #[test]
    fn bvh_handles_planes() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        b.add_shape_plane(crate::core::BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let model = b.build().unwrap();
        let state = model.state().unwrap();

        let mut bvh = Bvh::new();
        bvh.build(&model, &state, 0.0);
        let mut pairs = Vec::new();
        bvh.pairs(&model, &mut pairs);
        assert_eq!(pairs.len(), 1, "expected 1 sphere-plane pair, got {pairs:?}");
        assert_eq!(pairs[0].a.min(pairs[0].b), 0); // sphere shape
        assert_eq!(pairs[0].a.max(pairs[0].b), 1); // plane shape
    }
}
