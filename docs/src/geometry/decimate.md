# Mesh decimation

`crates/molla/src/geometry/decimate.rs` implements Garland-Heckbert
Quadric Error Metric (QEM) edge-collapse decimation. Stage-21
deliverable per `PLAN.md` §21 — produces LOD chains for collision
vs render meshes (e.g. render mesh keeps 100k tris, collider drops
to 5k).

## Public API

```rust
pub fn decimate(mesh: &TriMesh, target_tri_count: usize) -> TriMesh;
pub fn decimate_lod_chain(mesh: &TriMesh, factors: &[f64]) -> Vec<TriMesh>;
```

`decimate` reduces `mesh` to approximately `target_tri_count`
triangles. Returns a new mesh; the input is unchanged. If
`target_tri_count >= mesh.indices.len()` the mesh is returned
verbatim (with a fresh BVH). If `target_tri_count == 0` the mesh
is reduced as far as the topology guards allow — that could be
more than 0 tris if every remaining collapse would cause a flip.

`decimate_lod_chain` produces multiple LOD levels in one pass,
sharing the QEM priority queue across levels. Each entry of
`factors` is a fraction in `(0, 1]` of the input triangle count
(e.g. `[0.5, 0.25, 0.1]` produces three meshes at 50%, 25%, 10%
of the original). Output meshes are returned in the same order as
`factors`. Cheaper than calling `decimate` N times because the QEM
state (vertex quadrics + topology) is collapsed in-place and
snapshots are taken at each target — so you pay one decimation's
worth of work for the deepest factor and snapshot the
intermediate levels for free.

## Algorithm

Garland-Heckbert "Surface Simplification Using Quadric Error
Metrics" (SIGGRAPH '97). Each vertex carries the sum of the
squared-distance quadrics of its incident faces. A quadric
`Q ∈ ℝ⁴ˣ⁴` represents the sum of squared distances from a point
to a set of planes; collapsing edge `(u, v)` replaces both
endpoints with a single new vertex whose position minimises the
combined quadric `Q_u + Q_v`. The error of any candidate collapse
is `vᵀ Q v` for the chosen position.

The collapser repeatedly pops the lowest-error edge from a
priority queue:

1. **Initial quadrics.** For each face `f` with plane `n · x = d`
   (unit normal `n`, offset `d`), build the `4 × 4` outer-product
   quadric `Q_f = [n; d] · [n; d]ᵀ`. Sum into each incident vertex.
2. **Edge queue.** For every edge `(u, v)`, compute the combined
   quadric `Q_uv = Q_u + Q_v`, find the position `p*` that
   minimises `pᵀ Q_uv p` (solve the 3×3 linear system from
   `Q_uv`'s top-left 3×3 block, fall back to the midpoint when
   the system is singular), and push `(error, u, v, p*)` onto a
   min-heap.
3. **Collapse loop.** Pop the lowest-error edge:
   * Run topology guards (see below). Skip if any guard fails.
   * Replace `u` and `v` with a new vertex at `p*`. Update incident
     face indices.
   * Sum quadrics: `Q_new = Q_u + Q_v`.
   * Re-evaluate every edge incident to `new`; push fresh entries
     onto the heap.
4. **Stale-entry skip.** Each vertex carries a version counter
   bumped on every collapse. Heap entries store the version they
   were inserted at; popped entries whose recorded version doesn't
   match the current vertex version are stale and discarded.

The version counter is what makes the heap **incremental**. The
Stage 21 v1 implementation rebuilt the whole heap on every
collapse — Stage 21.1 made the inner heap incremental with
rayon-parallel cost evaluation on the initial seed pass.
Combined effect: ~100× faster on a 50k-tri mesh.

## Topology guards

Two guards filter out collapses that would corrupt the mesh:

* **Normal-flip check.** Skip collapses that flip any incident
  face's normal — i.e. for every face still using `u` or `v`,
  verify the post-collapse face normal points the same direction
  as the pre-collapse normal. Prevents self-intersection through
  the surface.
* **Non-manifold check.** Skip collapses whose merged 1-ring would
  create a non-manifold edge (two faces sharing more than 2
  vertices). Standard "link condition" from Dey, Edelsbrunner &
  Guha 1998 — the link of edge `(u, v)` (the boundary of the
  union of `star(u) ∪ star(v)`) must equal `link(u) ∩ link(v)`.

Both guards fire conservatively — if the topology near the edge
is ambiguous (degenerate triangles, dangling edges), the collapse
is refused. The result is that on broken inputs decimation stops
short of the target rather than producing garbage.

Boundary preservation is a known Garland-Heckbert simplification:
molla doesn't add the boundary-preservation quadrics, so open
meshes drift at the boundary as decimation proceeds. Closed
manifolds (the common case for collision authoring) decimate
cleanly.

## Edge-collapse strategy details

`decimate.rs::DecimateState` owns:

* `vertex_quadrics: Vec<Mat4>` — sum of incident face quadrics.
* `vertex_versions: Vec<u32>` — bumped on every collapse to
  invalidate stale heap entries.
* `vertex_alive: Vec<bool>` — false after a vertex is removed.
* `face_indices: Vec<[u32; 3]>` — per-face vertex indices (with
  sentinel `u32::MAX` for collapsed faces; compacted before
  output).
* `heap: BinaryHeap<HeapEntry>` — min-heap on cost (Rust's
  `BinaryHeap` is a max-heap, so cost is stored as `Reverse(cost)`).
* `vertex_incident_faces: Vec<HashSet<u32>>` — adjacency cache
  rebuilt on collapse.

`collapse_one()` is the work unit: pop, validate (version check,
guards), compute the new vertex position via the quadric minimiser
or midpoint fallback, update adjacency, bump versions, re-seed
incident edges. Each call performs `O(deg(new_vertex))` heap pushes
where `deg` is the vertex valence — typically 6 for a closed
manifold, so the per-collapse cost is small.

`into_trimesh()` is the final pass: walk the alive vertices and
faces, compact indices, return a new `TriMesh`. The output `BVH`
is rebuilt from the compacted geometry — collision queries against
the decimated mesh are immediately ready.

## Performance

QEM decimation is `O(N log N)` in practice (heap operations
dominate, each costing `O(log N)`). On a 50k-tri input reduced to
1k, the implementation completes in ~50 ms on one core (Stage
21.1 measurement). `decimate_lod_chain` with three factors
(0.5 / 0.25 / 0.1) on the same input costs ~52 ms — the marginal
cost of snapshotting intermediate levels is a few percent.

Best paired with the [convex hull builder](./hull.md) for
collision authoring: hull the input cloud first, then decimate
the hull mesh to keep the vertex count in the ~50 range where
`support_convex_hull`'s linear-scan support function performs
well.
