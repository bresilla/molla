//! Per-mesh AABB BVH for accelerating shape-vs-trimesh narrow phase
//! and ray-vs-trimesh queries. Without this, every shape-mesh pair
//! tests O(triangle_count) primitives — fine for ~50-tri shapes,
//! ruinous past 1k.
//!
//! Build is top-down median-split along the longest axis. Query
//! patterns:
//!   * `triangles_overlapping_aabb(min, max, out)` — for shape-vs-mesh.
//!   * `ray_cast(origin, dir, max_t)` — for raycast (host fallback;
//!     the WGSL raycast still iterates triangles for now).
//!
//! Stored as a flat `Vec<BvhNode>` so it's straightforward to port
//! to the GPU later (a contiguous array is cache-friendly).

use molla_math::{Real, Vec3};

/// One node of the per-mesh BVH. Internal nodes have two child
/// indices; leaves carry a contiguous slice of `tri_count` triangle
/// indices starting at `prim_start` in the mesh's `indices` array.
#[derive(Copy, Clone, Debug)]
pub struct BvhNode {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    /// Index of the first triangle in this leaf (within
    /// `triangle_order`). Meaningful only when `is_leaf == 1`.
    pub prim_start: u32,
    /// Number of triangles in this leaf. `0` for internal nodes.
    pub tri_count: u32,
    /// Left child index for internal nodes. `0` for leaves.
    pub left: u32,
    /// Right child index for internal nodes.
    pub right: u32,
    /// `1` = leaf, `0` = internal.
    pub is_leaf: u32,
}

impl BvhNode {
    pub const INVALID: u32 = u32::MAX;
}

/// Per-mesh BVH. Owns a flat node array + a triangle-index
/// permutation (`triangle_order`) so leaves can reference contiguous
/// triangle ranges.
#[derive(Debug, Clone, Default)]
pub struct MeshBvh {
    pub nodes: Vec<BvhNode>,
    /// Triangle indices in BVH-leaf order. Each leaf's range
    /// `[prim_start, prim_start + tri_count)` indexes into this.
    pub triangle_order: Vec<u32>,
}

impl MeshBvh {
    /// Exact median-split node count, or `None` when it exceeds `usize`.
    pub fn planned_node_count(triangles: usize, max_leaf_size: usize) -> Option<usize> {
        if triangles == 0 {
            return Some(0);
        }
        let leaf = max_leaf_size.max(1);
        let mut level = 1usize;
        while triangles / level > leaf {
            level = level.checked_mul(2)?;
        }
        let splits = if triangles / level == leaf {
            triangles % level
        } else {
            0
        };
        level
            .checked_add(splits)?
            .checked_sub(1)?
            .checked_mul(2)?
            .checked_add(1)
    }

    /// Build a BVH for the given mesh. `vertices` is the mesh's
    /// vertex array; `indices` is the flat triangle list (each entry
    /// is a `[v0, v1, v2]` triple). `max_leaf_size` triangles per
    /// leaf — 4 is a reasonable default (Embree's hint).
    pub fn build(vertices: &[Vec3], indices: &[[u32; 3]], max_leaf_size: usize) -> Self {
        let n = indices.len();
        if n == 0 {
            return Self::default();
        }
        let max_leaf = max_leaf_size.max(1);
        let node_count =
            Self::planned_node_count(n, max_leaf).expect("mesh BVH node count exceeds usize");
        // Per-triangle centroids + AABBs to drive the split.
        let mut centroid = Vec::with_capacity(n);
        let mut aabb = Vec::with_capacity(n);
        for tri in indices {
            let v0 = vertices[tri[0] as usize];
            let v1 = vertices[tri[1] as usize];
            let v2 = vertices[tri[2] as usize];
            centroid.push((v0 + v1 + v2) / 3.0);
            aabb.push((v0.min(v1).min(v2), v0.max(v1).max(v2)));
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(node_count);
        // Reserve root (index 0).
        nodes.push(BvhNode {
            aabb_min: Vec3::splat(Real::INFINITY),
            aabb_max: Vec3::splat(-Real::INFINITY),
            prim_start: 0,
            tri_count: 0,
            left: 0,
            right: 0,
            is_leaf: 0,
        });
        build_recursive(&centroid, &aabb, &mut order, 0, n, 0, max_leaf, &mut nodes);
        Self {
            nodes,
            triangle_order: order,
        }
    }

    /// Append every triangle index whose AABB overlaps the query AABB.
    /// Used by shape-vs-trimesh narrow phase.
    pub fn triangles_overlapping_aabb(&self, query_min: Vec3, query_max: Vec3, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if !aabb_overlap(node.aabb_min, node.aabb_max, query_min, query_max) {
                continue;
            }
            if node.is_leaf == 1 {
                let lo = node.prim_start as usize;
                let hi = lo + node.tri_count as usize;
                out.extend_from_slice(&self.triangle_order[lo..hi]);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// Closest-hit ray cast. Returns `(triangle_index, t, normal)`
    /// where `t ∈ [0, max_t]` and `normal` is the (unnormalised)
    /// outward triangle normal at the hit.
    pub fn ray_cast(
        &self,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
        origin: Vec3,
        dir: Vec3,
        max_t: Real,
    ) -> Option<(u32, Real, Vec3)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(u32, Real, Vec3)> = None;
        let mut best_t = max_t;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        let inv_dir = Vec3::new(
            if dir.x != 0.0 {
                1.0 / dir.x
            } else {
                Real::INFINITY
            },
            if dir.y != 0.0 {
                1.0 / dir.y
            } else {
                Real::INFINITY
            },
            if dir.z != 0.0 {
                1.0 / dir.z
            } else {
                Real::INFINITY
            },
        );
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if !ray_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, best_t) {
                continue;
            }
            if node.is_leaf == 1 {
                let lo = node.prim_start as usize;
                let hi = lo + node.tri_count as usize;
                for &tri_idx in &self.triangle_order[lo..hi] {
                    let tri = indices[tri_idx as usize];
                    let v0 = vertices[tri[0] as usize];
                    let v1 = vertices[tri[1] as usize];
                    let v2 = vertices[tri[2] as usize];
                    if let Some((t, n)) = ray_triangle(origin, dir, v0, v1, v2)
                        && t >= 0.0
                        && t <= best_t
                        && (best.is_none() || t < best_t)
                    {
                        best_t = t;
                        best = Some((tri_idx, t, n));
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// Total bounding box of the mesh (the root node's AABB).
    pub fn root_aabb(&self) -> (Vec3, Vec3) {
        if self.nodes.is_empty() {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (self.nodes[0].aabb_min, self.nodes[0].aabb_max)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_recursive(
    centroid: &[Vec3],
    aabb: &[(Vec3, Vec3)],
    order: &mut [u32],
    range_lo: usize,
    range_hi: usize,
    node_idx: usize,
    max_leaf: usize,
    nodes: &mut Vec<BvhNode>,
) {
    // Compute this node's AABB by union over its triangles' AABBs.
    let mut min = Vec3::splat(Real::INFINITY);
    let mut max = Vec3::splat(-Real::INFINITY);
    for &i in order[range_lo..range_hi].iter() {
        let (lo, hi) = aabb[i as usize];
        min = min.min(lo);
        max = max.max(hi);
    }
    let count = range_hi - range_lo;
    if count <= max_leaf {
        nodes[node_idx] = BvhNode {
            aabb_min: min,
            aabb_max: max,
            prim_start: range_lo as u32,
            tri_count: count as u32,
            left: 0,
            right: 0,
            is_leaf: 1,
        };
        return;
    }

    // Split along longest axis at the median centroid.
    let extent = max - min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let slice = &mut order[range_lo..range_hi];
    slice.sort_unstable_by(|a, b| {
        let ca = component(centroid[*a as usize], axis);
        let cb = component(centroid[*b as usize], axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = range_lo + count / 2;

    // Reserve indices for the two children, then recurse.
    let left_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb_min: Vec3::ZERO,
        aabb_max: Vec3::ZERO,
        prim_start: 0,
        tri_count: 0,
        left: 0,
        right: 0,
        is_leaf: 0,
    });
    let right_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb_min: Vec3::ZERO,
        aabb_max: Vec3::ZERO,
        prim_start: 0,
        tri_count: 0,
        left: 0,
        right: 0,
        is_leaf: 0,
    });

    nodes[node_idx] = BvhNode {
        aabb_min: min,
        aabb_max: max,
        prim_start: 0,
        tri_count: 0,
        left: left_idx,
        right: right_idx,
        is_leaf: 0,
    };

    build_recursive(
        centroid,
        aabb,
        order,
        range_lo,
        mid,
        left_idx as usize,
        max_leaf,
        nodes,
    );
    build_recursive(
        centroid,
        aabb,
        order,
        mid,
        range_hi,
        right_idx as usize,
        max_leaf,
        nodes,
    );
}

fn component(v: Vec3, axis: usize) -> Real {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn aabb_overlap(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool {
    a_min.x <= b_max.x
        && a_max.x >= b_min.x
        && a_min.y <= b_max.y
        && a_max.y >= b_min.y
        && a_min.z <= b_max.z
        && a_max.z >= b_min.z
}

fn ray_aabb(origin: Vec3, inv_dir: Vec3, min: Vec3, max: Vec3, max_t: Real) -> bool {
    let (mut lo, mut hi) = (0.0_f64, max_t);
    for i in 0..3 {
        if inv_dir[i].is_infinite() {
            if origin[i] < min[i] || origin[i] > max[i] {
                return false;
            }
        } else {
            let a = (min[i] - origin[i]) * inv_dir[i];
            let b = (max[i] - origin[i]) * inv_dir[i];
            lo = lo.max(a.min(b));
            hi = hi.min(a.max(b));
            if lo > hi {
                return false;
            }
        }
    }
    true
}

/// Möller-Trumbore ray-triangle intersection. Returns `(t, normal)`
/// on hit; `t` along ray. Normal is unnormalised cross product
/// `(v1 − v0) × (v2 − v0)` — caller normalises if needed.
pub(crate) fn ray_triangle(
    origin: Vec3,
    dir: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(Real, Vec3)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() <= 1e-14 * edge1.length() * edge2.length() * dir.length() {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(-1e-12..=1.0 + 1e-12).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }
    let t = f * edge2.dot(q);
    let normal = edge1.cross(edge2);
    Some((t, normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_mesh() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        // Eight unit cube vertices and 12 triangles (2 per face).
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let i = vec![
            [0, 1, 2],
            [0, 2, 3], // -Z face
            [4, 6, 5],
            [4, 7, 6], // +Z face
            [0, 4, 5],
            [0, 5, 1], // -Y face
            [3, 2, 6],
            [3, 6, 7], // +Y face
            [0, 3, 7],
            [0, 7, 4], // -X face
            [1, 5, 6],
            [1, 6, 2], // +X face
        ];
        (v, i)
    }

    #[test]
    fn build_returns_root_aabb_matches_mesh() {
        let (v, i) = unit_cube_mesh();
        let bvh = MeshBvh::build(&v, &i, 4);
        let (lo, hi) = bvh.root_aabb();
        assert!((lo - Vec3::splat(-0.5)).length() < 1e-9);
        assert!((hi - Vec3::splat(0.5)).length() < 1e-9);
    }

    #[test]
    fn planned_capacity_matches_actual_trees_across_leaf_sizes() {
        let vertices = [Vec3::ZERO, Vec3::X, Vec3::Z];
        for leaf in [0, 1, 2, 3, 4, 7, 8, 16, 129] {
            for count in [0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 49, 129, 257, 4097] {
                let indices = vec![[0, 1, 2]; count];
                let tree = MeshBvh::build(&vertices, &indices, leaf);
                assert_eq!(
                    MeshBvh::planned_node_count(count, leaf),
                    Some(tree.nodes.len()),
                    "count={count}, leaf={leaf}"
                );
                assert_eq!(
                    tree.nodes.capacity(),
                    tree.nodes.len(),
                    "count={count}, leaf={leaf}"
                );
                assert_eq!(tree.triangle_order.len(), count);
            }
        }
        assert_eq!(MeshBvh::planned_node_count(usize::MAX, 1), None);
        assert_eq!(
            MeshBvh::planned_node_count(usize::MAX / 2 + 1, 1),
            Some(usize::MAX)
        );
        assert_eq!(MeshBvh::planned_node_count(usize::MAX, usize::MAX), Some(1));
    }

    #[test]
    fn aabb_query_finds_overlapping_triangles_only() {
        let (v, i) = unit_cube_mesh();
        let bvh = MeshBvh::build(&v, &i, 2);
        // Query the +X face only: shrink the AABB to x in [0.4, 0.6].
        let mut hits = Vec::new();
        bvh.triangles_overlapping_aabb(
            Vec3::new(0.4, -1.0, -1.0),
            Vec3::new(0.6, 1.0, 1.0),
            &mut hits,
        );
        // +X face triangles are indices 10 and 11.
        let mut got: Vec<u32> = hits.into_iter().collect();
        got.sort();
        got.dedup();
        assert!(got.contains(&10));
        assert!(got.contains(&11));
    }

    #[test]
    fn ray_cast_hits_unit_cube_face() {
        let (v, i) = unit_cube_mesh();
        let bvh = MeshBvh::build(&v, &i, 4);
        // Ray from (-2, 0, 0) along +X should hit the -X face at t=1.5.
        let hit = bvh.ray_cast(&v, &i, Vec3::new(-2.0, 0.0, 0.0), Vec3::X, 100.0);
        let (_, t, _) = hit.expect("expected hit on -X face");
        assert!((t - 1.5).abs() < 1e-9, "got t={t}, expected 1.5");
    }

    #[test]
    fn ray_cast_misses_returns_none() {
        let (v, i) = unit_cube_mesh();
        let bvh = MeshBvh::build(&v, &i, 4);
        let hit = bvh.ray_cast(&v, &i, Vec3::new(-2.0, 5.0, 0.0), Vec3::X, 100.0);
        assert!(hit.is_none());
    }

    #[test]
    fn aabb_query_prunes_strip_mesh() {
        // 64 disjoint unit-square triangles spread along +X — each
        // occupies a tight AABB at x ∈ [k, k+1]. A point query in
        // x ∈ [30, 31] should hit only the local triangles.
        let n = 64usize;
        let mut v: Vec<Vec3> = Vec::new();
        let mut i: Vec<[u32; 3]> = Vec::new();
        for k in 0..n {
            let base = v.len() as u32;
            v.push(Vec3::new(k as Real, 0.0, 0.0));
            v.push(Vec3::new((k + 1) as Real, 0.0, 0.0));
            v.push(Vec3::new(k as Real, 0.0, 1.0));
            i.push([base, base + 1, base + 2]);
        }
        let bvh = MeshBvh::build(&v, &i, 4);
        let mut hits = Vec::new();
        bvh.triangles_overlapping_aabb(
            Vec3::new(30.0, -0.5, -0.5),
            Vec3::new(31.0, 0.5, 1.5),
            &mut hits,
        );
        assert!(
            hits.len() < 8,
            "BVH should prune to local triangles — got {} hits",
            hits.len()
        );
    }
}
