# Convex hull builder

`crates/molla/src/geometry/hull.rs` implements an incremental
3D QuickHull suitable for asset-load workloads. The output feeds
directly into `ShapeGeometry::ConvexHull { hull_id }` via
`ModelBuilder::add_hull`. CPU-only — per the GPU-first carve-out,
hull construction is tooling and runs once, not per substep.

## Public API

```rust
pub struct ConvexHullMesh {
    pub vertices: Vec<Vec3>,
    pub faces:    Vec<[u32; 3]>,
}

impl ConvexHullMesh {
    pub fn is_empty(&self) -> bool;
    pub fn edges(&self) -> Vec<(u32, u32)>;
}

pub fn convex_hull(points: &[Vec3]) -> ConvexHullMesh;
```

`ConvexHullMesh::edges` materialises the unique undirected edge list
(`u < v`) used by the viewer's hull-overlay wireframe. The vertex
list alone is what
[`ModelBuilder::add_hull`](../sim/builder.md) consumes — the face
data is for visualisation and topology-aware callers.

## Algorithm

Classic incremental QuickHull, as outlined in the module header at
`hull.rs:1-23`:

1. **Seed tetrahedron.** Pick four extreme points (per-axis min/max
   plus the farthest point from the resulting triangle) and check
   their signed volume is non-degenerate.
2. **Partition.** Bucket every remaining point by the face it sits
   "outside" of. A point is *outside* face `f` iff
   `f.normal · (p - f.anchor) > eps`.
3. **Iterate.** Repeatedly:
   * Pick the face with the largest outside set.
   * Find the farthest outside point `p_far`.
   * Walk the BFS / DFS frontier of faces `p_far` can see (its
     **horizon**) and delete them.
   * Stitch the silhouette edges to `p_far` as a fan of new
     triangles.
   * Re-partition the deleted faces' outside sets against the new
     fan.
4. **Done.** When every face's outside set is empty, the surviving
   faces are the hull.

The result is a triangulated manifold: each edge is shared by
exactly two faces; `ConvexHullMesh::edges` relies on this and would
return duplicates on a non-manifold output.

## Robustness

Numerical robustness uses an ε *scaled to the input diameter*
(`hull.rs:99-107`). Without this, hulls of geometry expressed in mm
behave differently from the same geometry in metres — fixed-ε
tolerances either reject valid coplanar candidates or accept
near-coplanar noise.

Degenerate inputs are handled defensively:

* **`< 4` points** — return the unique input vertices verbatim. The
  caller decides what to do (typically: bail out, this isn't a
  hull).
* **Colinear / coplanar inputs** — fall back to a 2D hull projected
  on the dominant plane (we find the projection axis by picking the
  smallest eigen-direction of the centred covariance — equivalent to
  Newton's `geometry/hull.py` fallback). If even that fails (all
  points within `eps` of each other), return the unique points and
  no faces.
* **Coplanar at the *tetra-seed* step** — bail back to the 2D
  fallback rather than emit a zero-volume tetra (which would feed
  GJK a flat Minkowski difference).

These guards mean `convex_hull(points).is_empty() == true` is a
reliable "input was degenerate" signal. Callers that need to detect
this (e.g. the URDF importer) check `faces.is_empty()` after the
call.

## Cost / vertex caps

QuickHull is `O(N · log N)` on average and `O(N²)` worst-case
(adversarial inputs that hit every face). Practical cost on a 10k-
point mesh-collision cloud: a few ms on one core, which is fine for
asset-load. For runtime hull rebuilds (point clouds streamed at
60 Hz) the implementation would need an incremental update path
that molla doesn't ship.

The `ShapeGeometry::ConvexHull` consumer uses a linear-scan support
function (`support_convex_hull` in `support.rs:165-176`) — the cost
is `O(V)` per support query. For hulls up to ~50 vertices that's
cheaper than a DOP / hill-climb-with-edge-graph; past that the
linear scan dominates. The hull builder doesn't cap output vertices
itself, but downstream callers commonly run the result through
[mesh decimation](./decimate.md) to get a low-V hull for narrow-
phase use.
