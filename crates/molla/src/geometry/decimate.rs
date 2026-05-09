//! Quadric Error Metric (QEM) mesh decimation.
//!
//! Stage-21 deliverable per `PLAN.md` §21 / `GAP.md` §4. Produces LOD
//! chains for collision vs render meshes — a render mesh might keep
//! 100k tris while the collider drops to 5k.
//!
//! Algorithm: Garland-Heckbert "Surface Simplification Using Quadric
//! Error Metrics" (SIGGRAPH '97). Each vertex carries the sum of the
//! squared-distance quadrics of its incident faces; collapsing edge
//! `(u, v)` replaces both endpoints with a single new vertex whose
//! position minimises the combined quadric (or, when that's
//! ill-conditioned, the midpoint). The error of any candidate
//! collapse is `vᵀ Q v` for the chosen position, and we always pop
//! the lowest-error edge from a priority queue.
//!
//! Topology guards:
//! - Skip collapses that flip any incident face's normal (prevents
//!   self-intersection).
//! - Skip collapses whose merged 1-ring would create a non-manifold
//!   edge (two faces sharing more than 2 vertices).
//!
//! The implementation is straightforward and CPU-only; it's intended
//! for asset-load workloads. Input meshes should be reasonably
//! manifold for clean output. Open meshes work but the boundary
//! drifts (we don't add boundary-preservation quadrics — a known
//! Garland-Heckbert simplification).

use crate::math::{Real, Vec3};
use crate::sim::TriMesh;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Decimate `mesh` to approximately `target_tri_count` triangles via
/// QEM edge collapses. Returns a new mesh; the input is unchanged.
///
/// If `target_tri_count >= mesh.indices.len()` the mesh is returned
/// (rebuilt with a fresh BVH). If `target_tri_count == 0` the mesh
/// is reduced as far as the topology guards allow (could be more
/// than 0 tris if every remaining collapse would cause a flip).
pub fn decimate(mesh: &TriMesh, target_tri_count: usize) -> TriMesh {
    let mut state = DecimateState::new(mesh);

    while state.alive_tri_count() > target_tri_count {
        if !state.collapse_one() {
            break; // no more legal collapses
        }
    }

    state.into_trimesh()
}

// ── Quadric algebra ──────────────────────────────────────────────
// A 4x4 symmetric quadric matrix Q = nnᵀ where n = (a, b, c, d)
// (plane equation `ax+by+cz+d=0`). Stored as the 10 unique entries.
#[derive(Copy, Clone, Debug, Default)]
struct Quadric {
    // Symmetric 4x4 — store upper triangle row-major.
    // Q = [ q00 q01 q02 q03 ]
    //     [ q01 q11 q12 q13 ]
    //     [ q02 q12 q22 q23 ]
    //     [ q03 q13 q23 q33 ]
    q00: Real, q01: Real, q02: Real, q03: Real,
    q11: Real, q12: Real, q13: Real,
    q22: Real, q23: Real,
    q33: Real,
}

impl Quadric {
    fn from_plane(a: Real, b: Real, c: Real, d: Real) -> Self {
        Self {
            q00: a * a, q01: a * b, q02: a * c, q03: a * d,
            q11: b * b, q12: b * c, q13: b * d,
            q22: c * c, q23: c * d,
            q33: d * d,
        }
    }

    fn add(&mut self, other: &Quadric) {
        self.q00 += other.q00; self.q01 += other.q01;
        self.q02 += other.q02; self.q03 += other.q03;
        self.q11 += other.q11; self.q12 += other.q12;
        self.q13 += other.q13; self.q22 += other.q22;
        self.q23 += other.q23; self.q33 += other.q33;
    }

    /// Evaluate `vᵀ Q v` for `v = (x, y, z, 1)`.
    fn evaluate(&self, p: Vec3) -> Real {
        let x = p.x; let y = p.y; let z = p.z;
        self.q00 * x * x + 2.0 * self.q01 * x * y + 2.0 * self.q02 * x * z + 2.0 * self.q03 * x
            + self.q11 * y * y + 2.0 * self.q12 * y * z + 2.0 * self.q13 * y
            + self.q22 * z * z + 2.0 * self.q23 * z
            + self.q33
    }
}

#[derive(Debug)]
struct Tri {
    v: [u32; 3],
    deleted: bool,
}

struct DecimateState {
    /// Original vertex positions (immutable across collapses; new
    /// vertices appended to the end as collapses happen).
    positions: Vec<Vec3>,
    quadrics: Vec<Quadric>,
    /// Vertex → list of triangle indices it participates in.
    vert_tris: Vec<Vec<u32>>,
    /// Whether vertex is still alive.
    alive: Vec<bool>,
    tris: Vec<Tri>,
    alive_tris: usize,
}

impl DecimateState {
    fn new(mesh: &TriMesh) -> Self {
        let positions = mesh.vertices.clone();
        let n = positions.len();
        let mut quadrics = vec![Quadric::default(); n];
        let mut tris: Vec<Tri> = Vec::with_capacity(mesh.indices.len());
        let mut vert_tris: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (ti, idx) in mesh.indices.iter().enumerate() {
            let v0 = positions[idx[0] as usize];
            let v1 = positions[idx[1] as usize];
            let v2 = positions[idx[2] as usize];
            let n_vec = (v1 - v0).cross(v2 - v0);
            let len = n_vec.length();
            if len > 1.0e-20 {
                let nrm = n_vec / len;
                let d = -nrm.dot(v0);
                let q = Quadric::from_plane(nrm.x, nrm.y, nrm.z, d);
                quadrics[idx[0] as usize].add(&q);
                quadrics[idx[1] as usize].add(&q);
                quadrics[idx[2] as usize].add(&q);
            }
            tris.push(Tri { v: [idx[0], idx[1], idx[2]], deleted: false });
            vert_tris[idx[0] as usize].push(ti as u32);
            vert_tris[idx[1] as usize].push(ti as u32);
            vert_tris[idx[2] as usize].push(ti as u32);
        }
        let alive_tris = tris.len();
        Self {
            positions,
            quadrics,
            vert_tris,
            alive: vec![true; n],
            tris,
            alive_tris,
        }
    }

    fn alive_tri_count(&self) -> usize {
        self.alive_tris
    }

    /// Try to perform one collapse. Returns false if no legal
    /// collapse remains.
    fn collapse_one(&mut self) -> bool {
        // Build (or refresh) the priority queue of candidate edges.
        // For simplicity (and to keep this within budget) we rebuild
        // it every call — sound but O(E log E) per collapse. For
        // production-grade decimators a stale-entry incremental queue
        // wins; the meshes we target here are small enough that the
        // simpler code wins on time-to-correct.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for t in &self.tris {
            if t.deleted {
                continue;
            }
            for &(u, v) in &[(t.v[0], t.v[1]), (t.v[1], t.v[2]), (t.v[2], t.v[0])] {
                let key = if u < v { (u, v) } else { (v, u) };
                if !seen.insert(key) {
                    continue;
                }
                let target = self.optimal_position(key.0, key.1);
                let mut q = self.quadrics[key.0 as usize];
                q.add(&self.quadrics[key.1 as usize]);
                let cost = q.evaluate(target).max(0.0);
                heap.push(HeapEntry {
                    cost,
                    u: key.0,
                    v: key.1,
                    target,
                });
            }
        }
        // Pop until we find one whose collapse is legal.
        while let Some(entry) = heap.pop() {
            if !self.alive[entry.u as usize] || !self.alive[entry.v as usize] {
                continue;
            }
            if self.collapse_would_flip(entry.u, entry.v, entry.target)
                || self.collapse_is_nonmanifold(entry.u, entry.v)
            {
                continue;
            }
            self.do_collapse(entry.u, entry.v, entry.target);
            return true;
        }
        false
    }

    fn optimal_position(&self, u: u32, v: u32) -> Vec3 {
        // Solve the upper-left 3x3 of (Qu + Qv) for the minimum-error
        // position. If singular, fall back to midpoint.
        let mut q = self.quadrics[u as usize];
        q.add(&self.quadrics[v as usize]);
        let m = [
            [q.q00, q.q01, q.q02],
            [q.q01, q.q11, q.q12],
            [q.q02, q.q12, q.q22],
        ];
        let b = [-q.q03, -q.q13, -q.q23];
        match solve3x3(m, b) {
            Some(x) => Vec3::new(x[0], x[1], x[2]),
            None => (self.positions[u as usize] + self.positions[v as usize]) * 0.5,
        }
    }

    fn collapse_would_flip(&self, u: u32, v: u32, target: Vec3) -> bool {
        // For every triangle incident on `u` or `v` that doesn't
        // contain the *other* endpoint (i.e. won't be deleted), check
        // that its post-collapse normal still points the same way.
        for &endpoint in &[u, v] {
            let other = if endpoint == u { v } else { u };
            for &ti in &self.vert_tris[endpoint as usize] {
                let t = &self.tris[ti as usize];
                if t.deleted {
                    continue;
                }
                if t.v.contains(&other) {
                    continue; // would be deleted by collapse
                }
                // Replace `endpoint` with `target` and check normal.
                let v_old: [Vec3; 3] = std::array::from_fn(|k| self.positions[t.v[k] as usize]);
                let v_new: [Vec3; 3] = std::array::from_fn(|k| {
                    if t.v[k] == endpoint {
                        target
                    } else {
                        self.positions[t.v[k] as usize]
                    }
                });
                let n_old = (v_old[1] - v_old[0]).cross(v_old[2] - v_old[0]);
                let n_new = (v_new[1] - v_new[0]).cross(v_new[2] - v_new[0]);
                // If either normal degenerates, reject — collapse
                // would create a sliver.
                if n_new.length_squared() < 1.0e-20 {
                    return true;
                }
                if n_old.dot(n_new) <= 0.0 {
                    return true;
                }
            }
        }
        false
    }

    fn collapse_is_nonmanifold(&self, u: u32, v: u32) -> bool {
        // Manifold check: the open neighbour ring of u and v intersect
        // in at most 2 vertices (the two faces sharing edge uv). If
        // they share more, collapsing creates a non-manifold edge.
        let mut neighbours_u: HashSet<u32> = HashSet::new();
        for &ti in &self.vert_tris[u as usize] {
            let t = &self.tris[ti as usize];
            if t.deleted {
                continue;
            }
            for &w in &t.v {
                if w != u && w != v {
                    neighbours_u.insert(w);
                }
            }
        }
        let mut shared = 0;
        for &ti in &self.vert_tris[v as usize] {
            let t = &self.tris[ti as usize];
            if t.deleted {
                continue;
            }
            for &w in &t.v {
                if w != u && w != v && neighbours_u.contains(&w) {
                    shared += 1;
                }
            }
        }
        // Each face sharing edge uv contributes its 3rd vertex twice
        // (once via u's ring, once via v's). We compare against 4 to
        // permit up to 2 shared faces (manifold case).
        shared > 4
    }

    fn do_collapse(&mut self, u: u32, v: u32, target: Vec3) {
        // Move `u` to the target, kill `v`. Every triangle on `v`
        // either becomes degenerate (if it contained `u`) or has its
        // `v` index rewritten to `u`.
        self.positions[u as usize] = target;
        let mut q = self.quadrics[u as usize];
        q.add(&self.quadrics[v as usize]);
        self.quadrics[u as usize] = q;
        self.alive[v as usize] = false;
        let v_tris = std::mem::take(&mut self.vert_tris[v as usize]);
        for ti in v_tris {
            let tri = &mut self.tris[ti as usize];
            if tri.deleted {
                continue;
            }
            if tri.v.contains(&u) {
                tri.deleted = true;
                self.alive_tris -= 1;
                // Remove this `ti` from u's vert_tris.
                self.vert_tris[u as usize].retain(|x| *x != ti);
            } else {
                for slot in tri.v.iter_mut() {
                    if *slot == v {
                        *slot = u;
                    }
                }
                self.vert_tris[u as usize].push(ti);
            }
        }
    }

    fn into_trimesh(self) -> TriMesh {
        // Compact: pick alive vertices, remap indices.
        let mut new_index_of: HashMap<u32, u32> = HashMap::new();
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for t in &self.tris {
            if t.deleted {
                continue;
            }
            let mut idx = [0u32; 3];
            for (k, &old) in t.v.iter().enumerate() {
                let entry = new_index_of.entry(old).or_insert_with(|| {
                    vertices.push(self.positions[old as usize]);
                    (vertices.len() - 1) as u32
                });
                idx[k] = *entry;
            }
            // Drop fully degenerate (any two indices equal).
            if idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2] {
                continue;
            }
            indices.push(idx);
        }
        let mut m = TriMesh {
            vertices,
            indices,
            ..Default::default()
        };
        m.build_bvh(4);
        m
    }
}

// ── Heap with min-cost ordering. `BinaryHeap` is max-heap; flip the
// sign of `cost` so the smallest-cost entry pops first.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    cost: Real,
    u: u32,
    v: u32,
    target: Vec3,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap via reversed comparison; treat NaN as last.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn solve3x3(m: [[Real; 3]; 3], b: [Real; 3]) -> Option<[Real; 3]> {
    // Cramer's rule with a singularity guard.
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1.0e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let dx = b[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (b[1] * m[2][2] - m[1][2] * b[2])
        + m[0][2] * (b[1] * m[2][1] - m[1][1] * b[2]);
    let dy = m[0][0] * (b[1] * m[2][2] - m[1][2] * b[2])
        - b[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * b[2] - b[1] * m[2][0]);
    let dz = m[0][0] * (m[1][1] * b[2] - b[1] * m[2][1])
        - m[0][1] * (m[1][0] * b[2] - b[1] * m[2][0])
        + b[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    Some([dx * inv, dy * inv, dz * inv])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_mesh() -> TriMesh {
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
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 6, 2], [3, 7, 6],
            [0, 7, 3], [0, 4, 7],
            [1, 2, 6], [1, 6, 5],
        ];
        let mut m = TriMesh { vertices: v, indices: i, ..Default::default() };
        m.build_bvh(4);
        m
    }

    #[test]
    fn decimate_returns_something_for_cube() {
        let m = cube_mesh();
        let d = decimate(&m, 8);
        // Cube has 12 tris; aim for 8 — the result should be ≤ 12
        // and ≥ 4 (a tetrahedron is the floor).
        assert!(d.indices.len() <= 12);
        assert!(d.indices.len() >= 4, "got {} tris", d.indices.len());
    }

    #[test]
    fn decimate_with_high_target_returns_input_size() {
        let m = cube_mesh();
        let d = decimate(&m, 100);
        assert_eq!(d.indices.len(), 12);
    }
}
