//! Cross-mesh soft-tri BVH wrapper — Stage 33.
//!
//! Wraps a top-down AABB BVH over the union of every soft surface's
//! triangles. The contact gather pass walks per-vertex queries through
//! this BVH; per-substep refit keeps the topology fixed (cheap
//! AABB-only update) while the rebuild cost only re-pays itself if the
//! soft mesh count changes.
//!
//! v1 ships CPU-only. The GPU port (Stage 33.2) will replace the
//! traversal with the existing [`crate::kernels::lbvh`] dispatcher and
//! reuse the AABB array prepared here.

use crate::math::{Real, Vec3};
use crate::sim::{Model, State};

use super::aabb::Aabb;

/// Internal node / leaf for a flat top-down BVH. Internal nodes
/// reference children via `[i32; 2]` indices into the same array; for
/// leaves the second slot is `-1` and the first slot stores the
/// global tri index.
#[derive(Copy, Clone, Debug)]
struct SoftBvhNode {
    aabb: Aabb,
    /// Internal: child0 index. Leaf: global tri index (into
    /// `Model.soft_surface_tris`).
    child0: i32,
    /// Internal: child1 index. Leaf: `-1`.
    child1: i32,
}

impl SoftBvhNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.child1 < 0
    }
    #[inline]
    fn tri_index(&self) -> u32 {
        debug_assert!(self.is_leaf());
        self.child0 as u32
    }
}

/// CPU BVH over every soft-surface triangle. Owns:
/// - `tri_aabbs`: per-tri AABBs in the same order as
///   `Model.soft_surface_tris`. Refit reuses these slots.
/// - `nodes`: flat node array with `root` indexing the topmost node.
/// - `tri_to_surface`: per-tri lookup → which `SoftSurface` index the
///   triangle belongs to. Used by the contact gather to filter out
///   same-surface contacts (self-collision land in Stage 33.3).
#[derive(Debug, Default)]
pub struct SoftBvh {
    pub tri_aabbs: Vec<Aabb>,
    nodes: Vec<SoftBvhNode>,
    root: i32,
    /// Surface ownership per tri; same length as `tri_aabbs`.
    pub tri_to_surface: Vec<u32>,
}

impl SoftBvh {
    /// Build the BVH from the model's `soft_surfaces` registry.
    /// `state.particle_q` supplies vertex positions.
    pub fn build(model: &Model, state: &State) -> Self {
        let mut bvh = Self::default();
        bvh.rebuild(model, state);
        bvh
    }

    fn rebuild(&mut self, model: &Model, state: &State) {
        self.nodes.clear();
        self.tri_aabbs.clear();
        self.tri_to_surface.clear();
        self.root = -1;

        if model.soft_surfaces.is_empty() {
            return;
        }
        let q = match state.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        if tris.is_empty() {
            return;
        }

        // Per-tri AABB + surface ownership. Tris are stored flat
        // across all surfaces, so we walk the registry in order.
        self.tri_aabbs.reserve(tris.len());
        self.tri_to_surface.reserve(tris.len());
        for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
            let lo = s.tri_offset as usize;
            let hi = lo + s.tri_count as usize;
            for t in lo..hi {
                let tri = tris[t];
                let aabb = tri_aabb(&tri, q);
                self.tri_aabbs.push(aabb);
                self.tri_to_surface.push(s_idx as u32);
            }
        }

        if self.tri_aabbs.is_empty() {
            return;
        }
        let mut indices: Vec<u32> = (0..self.tri_aabbs.len() as u32).collect();
        let aabbs = self.tri_aabbs.clone();
        self.root = self.build_recursive(&mut indices, &aabbs);
    }

    /// Stage 33.4 — refit the BVH using SWEPT AABBs (the union of
    /// every tri's `t0` AABB and `t1` AABB). The resulting BVH bounds
    /// the *trajectory* of every triangle over the upcoming substep,
    /// so a swept-point query (see [`Self::query_point_swept`]) can
    /// find every candidate tri the vertex might collide with along
    /// its own trajectory. Used by the VBD pre-substep CCD pass.
    ///
    /// Falls back to a full rebuild when the tri count has drifted
    /// (rare — the soft surface registry is fixed at build time).
    pub fn refit_swept(&mut self, state_t0: &State, state_t1: &State, model: &Model) {
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        if tris.len() != self.tri_aabbs.len() {
            // Topology changed — full rebuild from t1 (best we can
            // do; the topology drifted out from under us).
            self.rebuild(model, state_t1);
            return;
        }
        let q0 = match state_t0.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        let q1 = match state_t1.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        for (i, tri) in tris.iter().enumerate() {
            let a0 = tri_aabb(tri, q0);
            let a1 = tri_aabb(tri, q1);
            self.tri_aabbs[i] = aabb_union(a0, a1);
        }
        if self.root >= 0 {
            self.refit_node(self.root);
        }
    }

    /// Stage 33.4 — query: indices of every tri whose AABB intersects
    /// the swept-point AABB (the union of `p_t0` and `p_t1`, inflated
    /// by `margin`). Pair with [`Self::refit_swept`] to find every tri
    /// whose trajectory may collide with the point's trajectory.
    /// Returns global tri indices.
    #[must_use]
    pub fn query_point_swept(&self, p_t0: Vec3, p_t1: Vec3, margin: Real) -> Vec<u32> {
        let mut out = Vec::new();
        if self.root < 0 {
            return out;
        }
        // Swept-point AABB: union of `p_t0` and `p_t1`. The query
        // returns every tri whose tri-aabb is within `margin` of any
        // point on the line segment `[p_t0, p_t1]` — actually more
        // conservative than that (we use the AABB of the segment
        // not the segment itself), which is fine: candidates are
        // refined downstream by the per-tri `vt_ccd` test.
        let q_min = p_t0.min(p_t1);
        let q_max = p_t0.max(p_t1);
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(node_idx) = stack.pop() {
            if node_idx < 0 {
                continue;
            }
            let n = self.nodes[node_idx as usize];
            if !aabb_intersects_swept(&n.aabb, q_min, q_max, margin) {
                continue;
            }
            if n.is_leaf() {
                out.push(n.tri_index());
            } else {
                stack.push(n.child0);
                stack.push(n.child1);
            }
        }
        out
    }

    /// Recompute every per-tri AABB from `state.particle_q` then
    /// refit the existing tree topology bottom-up. Cheap when the
    /// mesh layout is unchanged — the recommended per-substep call
    /// path. Falls back to a full rebuild if the tri count has
    /// drifted (e.g. a new surface was registered).
    pub fn refit(&mut self, model: &Model, state: &State) {
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        if tris.len() != self.tri_aabbs.len() {
            // Topology changed — full rebuild.
            self.rebuild(model, state);
            return;
        }
        let q = match state.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        for (i, tri) in tris.iter().enumerate() {
            self.tri_aabbs[i] = tri_aabb(tri, q);
        }
        // Refit bottom-up: post-order walk from root, recomputing each
        // internal node's AABB as the union of its children's.
        if self.root >= 0 {
            self.refit_node(self.root);
        }
    }

    fn refit_node(&mut self, node_idx: i32) {
        if node_idx < 0 {
            return;
        }
        let n = self.nodes[node_idx as usize];
        if n.is_leaf() {
            // Leaf AABB lives in `tri_aabbs`; copy through.
            let tri_idx = n.tri_index() as usize;
            self.nodes[node_idx as usize].aabb = self.tri_aabbs[tri_idx];
            return;
        }
        let c0 = n.child0;
        let c1 = n.child1;
        self.refit_node(c0);
        self.refit_node(c1);
        let a0 = self.nodes[c0 as usize].aabb;
        let a1 = self.nodes[c1 as usize].aabb;
        self.nodes[node_idx as usize].aabb = aabb_union(a0, a1);
    }

    /// Query: indices of every tri whose (margin-inflated) AABB
    /// contains `p`. Returns global tri indices (into
    /// `Model.soft_surface_tris`).
    #[must_use]
    pub fn query_point(&self, p: Vec3, margin: Real) -> Vec<u32> {
        let mut out = Vec::new();
        if self.root < 0 {
            return out;
        }
        // Iterative DFS so deep trees don't blow the stack.
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(node_idx) = stack.pop() {
            if node_idx < 0 {
                continue;
            }
            let n = self.nodes[node_idx as usize];
            if !aabb_contains_point(&n.aabb, p, margin) {
                continue;
            }
            if n.is_leaf() {
                out.push(n.tri_index());
            } else {
                stack.push(n.child0);
                stack.push(n.child1);
            }
        }
        out
    }

    fn build_recursive(&mut self, indices: &mut [u32], aabbs: &[Aabb]) -> i32 {
        if indices.len() == 1 {
            let tri = indices[0];
            let node = SoftBvhNode {
                aabb: aabbs[tri as usize],
                child0: tri as i32,
                child1: -1,
            };
            self.nodes.push(node);
            return (self.nodes.len() - 1) as i32;
        }
        // Centroid bounds → split on longest axis (median split).
        let mut centroid_min = Vec3::splat(Real::INFINITY);
        let mut centroid_max = Vec3::splat(-Real::INFINITY);
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
        let mid = indices.len() / 2;
        let (left, right) = indices.split_at_mut(mid);
        let l = self.build_recursive(left, aabbs);
        let r = self.build_recursive(right, aabbs);
        let l_aabb = self.nodes[l as usize].aabb;
        let r_aabb = self.nodes[r as usize].aabb;
        let node = SoftBvhNode {
            aabb: aabb_union(l_aabb, r_aabb),
            child0: l,
            child1: r,
        };
        self.nodes.push(node);
        (self.nodes.len() - 1) as i32
    }
}

#[inline]
fn tri_aabb(tri: &[u32; 3], q: &[Vec3]) -> Aabb {
    let v0 = q[tri[0] as usize];
    let v1 = q[tri[1] as usize];
    let v2 = q[tri[2] as usize];
    Aabb {
        min: v0.min(v1).min(v2),
        max: v0.max(v1).max(v2),
    }
}

#[inline]
fn aabb_union(a: Aabb, b: Aabb) -> Aabb {
    Aabb {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    }
}

#[inline]
fn aabb_contains_point(a: &Aabb, p: Vec3, margin: Real) -> bool {
    p.x >= a.min.x - margin
        && p.x <= a.max.x + margin
        && p.y >= a.min.y - margin
        && p.y <= a.max.y + margin
        && p.z >= a.min.z - margin
        && p.z <= a.max.z + margin
}

/// Stage 33.4 — true iff the tri AABB `a` intersects the swept-point
/// AABB `[q_min, q_max]` after `margin` inflation on every axis.
#[inline]
fn aabb_intersects_swept(a: &Aabb, q_min: Vec3, q_max: Vec3, margin: Real) -> bool {
    a.min.x - margin <= q_max.x
        && a.max.x + margin >= q_min.x
        && a.min.y - margin <= q_max.y
        && a.max.y + margin >= q_min.y
        && a.min.z - margin <= q_max.z
        && a.max.z + margin >= q_min.z
}
