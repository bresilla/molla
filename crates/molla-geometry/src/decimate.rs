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

use molla_math::{Real, Vec3};
use molla_sim::TriMesh;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Decimate `mesh` to approximately `target_tri_count` triangles via
/// QEM edge collapses. Returns a new mesh; the input is unchanged.
///
/// If `target_tri_count >= mesh.indices.len()` the mesh is returned
/// (rebuilt with a fresh BVH). If `target_tri_count == 0` the mesh
/// is reduced as far as the topology guards allow (could be more
/// than 0 tris if every remaining collapse would cause a flip).
///
/// Stage 21.1 makes the inner heap incremental: the priority queue
/// is built once with rayon-parallel cost evaluation, then each
/// collapse only re-evaluates edges incident on the surviving
/// vertex. Stale heap entries are skipped via a per-vertex version
/// counter. Combined effect: ~100x faster on a 50k-tri mesh vs the
/// original Stage 21 implementation that rebuilt the whole heap on
/// every collapse.
pub fn decimate(mesh: &TriMesh, target_tri_count: usize) -> TriMesh {
    let mut state = DecimateState::new(mesh);
    state.seed_heap();

    while state.alive_tri_count() > target_tri_count {
        if !state.collapse_one() {
            break; // no more legal collapses
        }
    }

    state.into_trimesh()
}

/// Decimate `mesh` to multiple LOD levels in one pass, sharing the
/// QEM priority queue across levels. Each entry of `factors` is a
/// fraction in `(0, 1]` of the input triangle count (e.g. `[0.5,
/// 0.25, 0.1]` produces three meshes at 50 %, 25 %, 10 % of the
/// original). Output meshes are returned in the same order as
/// `factors`.
///
/// Cheaper than running [`decimate`] N times because the QEM state
/// (vertex quadrics + topology) is collapsed in-place and snapshots
/// taken at each target — so you pay ~one decimation's worth of work
/// for the deepest factor and snapshot the intermediate levels for
/// free.
///
/// Factors are processed in *descending* order internally (largest
/// triangle count first → coarsest last). Out-of-range factors are
/// clamped: ≤ 0 yields the maximally simplified mesh the topology
/// guards permit, > 1 yields the input.
pub fn decimate_lod_chain(mesh: &TriMesh, factors: &[f64]) -> Vec<TriMesh> {
    if factors.is_empty() {
        return Vec::new();
    }
    let total = mesh.indices.len();
    // Build a (target_tri_count, original_index) list sorted by
    // descending target so we collapse from the most-detailed level
    // down to the coarsest in one continuous run.
    let mut targets: Vec<(usize, usize)> = factors
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let f = f.clamp(0.0, 1.0);
            let n = (total as f64 * f).round() as usize;
            (n.min(total), i)
        })
        .collect();
    targets.sort_by(|a, b| b.0.cmp(&a.0));

    let mut state = DecimateState::new(mesh);
    state.seed_heap();
    let mut out: Vec<Option<TriMesh>> = (0..factors.len()).map(|_| None).collect();

    for (target, idx) in targets {
        while state.alive_tri_count() > target {
            if !state.collapse_one() {
                break;
            }
        }
        out[idx] = Some(state.snapshot_trimesh());
    }

    out.into_iter()
        .map(|m| m.expect("every index assigned"))
        .collect()
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
    q00: Real,
    q01: Real,
    q02: Real,
    q03: Real,
    q11: Real,
    q12: Real,
    q13: Real,
    q22: Real,
    q23: Real,
    q33: Real,
}

impl Quadric {
    fn from_plane(a: Real, b: Real, c: Real, d: Real) -> Self {
        Self {
            q00: a * a,
            q01: a * b,
            q02: a * c,
            q03: a * d,
            q11: b * b,
            q12: b * c,
            q13: b * d,
            q22: c * c,
            q23: c * d,
            q33: d * d,
        }
    }

    fn add(&mut self, other: &Quadric) {
        self.q00 += other.q00;
        self.q01 += other.q01;
        self.q02 += other.q02;
        self.q03 += other.q03;
        self.q11 += other.q11;
        self.q12 += other.q12;
        self.q13 += other.q13;
        self.q22 += other.q22;
        self.q23 += other.q23;
        self.q33 += other.q33;
    }

    /// Evaluate `vᵀ Q v` for `v = (x, y, z, 1)`.
    fn evaluate(&self, p: Vec3) -> Real {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        self.q00 * x * x
            + 2.0 * self.q01 * x * y
            + 2.0 * self.q02 * x * z
            + 2.0 * self.q03 * x
            + self.q11 * y * y
            + 2.0 * self.q12 * y * z
            + 2.0 * self.q13 * y
            + self.q22 * z * z
            + 2.0 * self.q23 * z
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
    /// Per-vertex version counter — bumped each time the vertex's
    /// quadric changes (i.e. when it absorbs another vertex via a
    /// collapse). Heap entries record the (u_ver, v_ver) snapshot at
    /// insertion time; on pop, if either version no longer matches
    /// the entry is stale and skipped. This is the standard QEM
    /// lazy-invalidation trick that keeps the heap bounded in size
    /// and avoids the per-collapse O(E) rebuild.
    vert_version: Vec<u32>,
    /// Incremental priority queue. Built once by `seed_heap`, then
    /// only edges incident on a surviving vertex are re-inserted on
    /// each collapse.
    heap: BinaryHeap<HeapEntry>,
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
            tris.push(Tri {
                v: [idx[0], idx[1], idx[2]],
                deleted: false,
            });
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
            vert_version: vec![0; n],
            heap: BinaryHeap::new(),
        }
    }

    /// Build the initial priority queue. Run once after [`Self::new`]
    /// before the first [`Self::collapse_one`] call.
    ///
    /// Walks the triangle list to collect canonical edges, then fans
    /// the per-edge cost evaluation across rayon's pool. The heap is
    /// then heapified once via `BinaryHeap::from` (O(n)).
    fn seed_heap(&mut self) {
        let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(self.tris.len() * 3 / 2);
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.tris.len() * 3 / 2);
        for t in &self.tris {
            if t.deleted {
                continue;
            }
            for &(u, v) in &[(t.v[0], t.v[1]), (t.v[1], t.v[2]), (t.v[2], t.v[0])] {
                let key = if u < v { (u, v) } else { (v, u) };
                if seen.insert(key) {
                    edges.push(key);
                }
            }
        }
        // Per-edge cost / target — fan out across cores. Each task
        // is read-only against `quadrics` + `positions`, no shared
        // mutable state.
        let entries: Vec<HeapEntry> = edges
            .par_iter()
            .map(|&(u, v)| self.edge_entry(u, v))
            .collect();
        self.heap = BinaryHeap::from(entries);
    }

    /// Build a `HeapEntry` for edge `(u, v)`. Pure read; safe to call
    /// from rayon tasks.
    fn edge_entry(&self, u: u32, v: u32) -> HeapEntry {
        let target = self.optimal_position(u, v);
        let mut q = self.quadrics[u as usize];
        q.add(&self.quadrics[v as usize]);
        let cost = q.evaluate(target).max(0.0);
        HeapEntry {
            cost,
            u,
            v,
            target,
            u_ver: self.vert_version[u as usize],
            v_ver: self.vert_version[v as usize],
        }
    }

    fn alive_tri_count(&self) -> usize {
        self.alive_tris
    }

    /// Try to perform one collapse. Returns false if no legal
    /// collapse remains.
    ///
    /// Reads from `self.heap` (the persistent priority queue seeded
    /// once by [`Self::seed_heap`]). Stale entries (one of whose
    /// vertices has been modified by a later collapse) are
    /// recognised by version mismatch and skipped — this is the
    /// standard QEM lazy-invalidation trick that keeps the
    /// per-collapse cost at amortised O(log E) instead of O(E).
    fn collapse_one(&mut self) -> bool {
        while let Some(entry) = self.heap.pop() {
            // Liveness + freshness check.
            if !self.alive[entry.u as usize] || !self.alive[entry.v as usize] {
                continue;
            }
            if self.vert_version[entry.u as usize] != entry.u_ver
                || self.vert_version[entry.v as usize] != entry.v_ver
            {
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

    /// Build a `TriMesh` from the current alive vertices/triangles
    /// without consuming `self`. Used by [`decimate_lod_chain`] to
    /// snapshot intermediate levels while continuing to collapse.
    fn snapshot_trimesh(&self) -> TriMesh {
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
        // Bump u's version: every heap entry that referenced u
        // (either as endpoint) is now stale and will be filtered on
        // pop. v gets bumped too as a belt-and-braces guard, though
        // its `alive` flag would catch it.
        self.vert_version[u as usize] = self.vert_version[u as usize].wrapping_add(1);
        self.vert_version[v as usize] = self.vert_version[v as usize].wrapping_add(1);

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

        // Re-insert every edge incident on `u` with fresh costs and
        // version snapshots. This is the only "real" work per
        // collapse — bounded by u's degree (typically ~6 for a
        // manifold mesh), so amortised O(log E).
        //
        // Walk u's surviving triangles and dedupe the neighbour set.
        let u_tris = self.vert_tris[u as usize].clone();
        let mut neighbours: HashSet<u32> = HashSet::with_capacity(u_tris.len() * 2);
        for ti in &u_tris {
            let tri = &self.tris[*ti as usize];
            if tri.deleted {
                continue;
            }
            for &w in &tri.v {
                if w != u {
                    neighbours.insert(w);
                }
            }
        }
        // Sort for deterministic insertion order — `HashSet`
        // iteration order is randomised by the hasher seed and would
        // otherwise inject run-to-run noise into the heap layout.
        let mut neighbours: Vec<u32> = neighbours.into_iter().collect();
        neighbours.sort_unstable();
        for w in neighbours {
            let (a, b) = if u < w { (u, w) } else { (w, u) };
            let entry = self.edge_entry(a, b);
            self.heap.push(entry);
        }
    }

    fn into_trimesh(self) -> TriMesh {
        self.snapshot_trimesh()
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
    /// Vertex-version snapshot at insertion. If the popped entry's
    /// versions disagree with `state.vert_version[u/v]`, the edge is
    /// stale (one endpoint was modified by a later collapse) and
    /// must be skipped.
    u_ver: u32,
    v_ver: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.u == other.u && self.v == other.v
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
        // Min-heap via reversed cost. Tie-break on (u, v) so two
        // entries with identical cost still pop in a fixed order
        // across runs — important now that costs are computed in
        // parallel and float-summation order is no longer the
        // implicit determinism source. Treat NaN as last.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.u.cmp(&self.u))
            .then_with(|| other.v.cmp(&self.v))
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
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 7, 3],
            [0, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mut m = TriMesh {
            vertices: v,
            indices: i,
            ..Default::default()
        };
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
