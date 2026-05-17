# EPA

The Expanding Polytope Algorithm picks up where [GJK](./gjk.md)
leaves off — given a 4-vertex Minkowski-difference simplex that
contains the origin, EPA grows the simplex face-by-face until the
face nearest the origin is on the polytope boundary. The
distance from origin to that face is the **penetration depth**;
the face normal is the **contact normal**.

Reference: van den Bergen, *Proximity Queries and Penetration
Depth Computation on 3D Game Objects* (2001).

CPU implementation: `crates/molla/src/geometry/epa.rs`.
GPU port (Stage 15): `crates/molla-geometry/shaders/epa.wgsl`.

## Public API

```rust
pub struct EpaContact {
    pub point_a: Vec3,
    pub point_b: Vec3,
    pub normal:  Vec3, // unit, points from A into B
    pub depth:   Real, // ≥ 0
}

pub fn epa(
    geom_a: ShapeGeometry, xf_a: &Transform,
    geom_b: ShapeGeometry, xf_b: &Transform,
    simplex: &GjkSimplex,
) -> Option<EpaContact>;

pub fn epa_with_hulls(
    geom_a: ShapeGeometry, xf_a: &Transform,
    geom_b: ShapeGeometry, xf_b: &Transform,
    simplex: &GjkSimplex,
    hulls:   &[Vec<Vec3>],
) -> Option<EpaContact>;
```

`Option` is the safety hatch — EPA returns `None` only when the
initial polytope build degenerates beyond recovery (every cardinal
fallback face has zero distance to origin, indicating both shapes
collapse to a point in Minkowski space). Production scenes never
hit this; tests use the `Option` to assert no NaNs leak through.

## Algorithm

1. **Initial polytope.** Pad the GJK simplex (which may be 1, 2,
   3, or 4 vertices) into a 4-vertex tetrahedron strictly
   containing the origin. If the simplex pad fails or produces a
   tetra with the origin on a face,
   `build_initial_faces_or_fallback` (`epa.rs:181-220`) falls back
   to four cardinal-axis support points.
2. **Find nearest face.** Walk the polytope's faces; pick the one
   whose plane is closest to the origin along its outward normal.
3. **Expand.** Take a new support point along that face's normal.
   If the new support sits within `TOL = 1e-6` of the face plane,
   the face is on the polytope boundary → **converged**. Return
   the contact derived from this face.
4. **Otherwise expand the polytope:**
   * Find every face whose outward normal can "see" the new
     vertex (the **visible** faces).
   * Collect the **silhouette** (edges that appear in exactly one
     visible face) as directed edges.
   * Delete the visible faces.
   * Stitch a fan of new triangles from each silhouette edge to
     the new vertex.
5. Loop until convergence or `MAX_ITERS = 64`.

The silhouette computation uses the standard "cancel directed
edges" trick (`epa.rs:124-141`):

```rust
let mut directed: Vec<(usize, usize)> = Vec::new();
for &fi in &visible_faces {
    let f = &faces[fi];
    for k in 0..3 {
        let a = f.indices[k];
        let b = f.indices[(k + 1) % 3];
        // If the reverse edge (b → a) was just inserted, cancel
        // it — that means two adjacent visible faces shared this
        // edge, so it's not on the silhouette.
        if let Some(p) = directed.iter().position(|&(x, y)| x == b && y == a) {
            directed.remove(p);
        } else {
            directed.push((a, b));
        }
    }
}
```

Each surviving directed edge `(a, b)` becomes a new triangle
`(a, b, new_idx)` whose outward winding follows from the original
face's CCW orientation.

## Contact extraction

The converged face has three vertex indices into `vertices`
(carrying the per-shape `a`/`b` support points). Project the origin
onto the face plane and recover barycentric weights `(u, v, w)`;
mix the per-shape support points by the same weights to get
`point_a` / `point_b` on the original shapes. The contact normal
is the face's outward normal; the depth is the plane distance from
the origin.

This is the same pattern as GJK's disjoint-case closest-point
recovery — preserving the per-shape support data in
`SupportPoint::a` / `SupportPoint::b` is what makes it possible.

## Fallback / degenerate inputs

EPA needs a non-degenerate tetra to start. The simplex coming in
may be just a vertex (radially-symmetric shapes whose Minkowski
difference passes through the origin) or coplanar (a flat plate
normal to the Minkowski-axis collapses both supports to the same
point). `build_initial_faces_or_fallback` does two things:

1. Try padding the simplex with cardinal-axis support points
   until it's a non-degenerate tetra strictly containing the
   origin.
2. If that fails (e.g. two co-linear sphere centres → simplex on
   the X axis), discard the simplex and build a fresh 4-vertex
   tetra from `±X`, `±Y`, `±Z` support points.

Either path then triggers a "reject if origin lies on a face"
check; on rejection the routine retries with a different vertex
ordering or returns `None`.

The iteration cap (`MAX_ITERS = 64`) prevents infinite expansion
on numerically pathological inputs — EPA emits the current best
face if it can't converge. This is rare; `MprCpu` / `MprGpu`
backends in the [narrow phase](./narrow_phase.md) are the
recommended escape hatch when EPA stalls in deep-penetration
regimes.

## GPU port (Stage 15)

`shaders/epa.wgsl` runs one thread per pair (matching
[GJK](./gjk.md)'s layout). Per-thread fixed-size arrays cap the
polytope at `MAX_VERTICES = 32` and `MAX_FACES = 64` — enough for
~95% of realistic contacts. Pairs that overflow or fail to
converge emit `status = STATUS_FALLBACK = 2`; the host's
collision pipeline then re-runs them through CPU EPA.

The output stride is 16 `f32` per pair, identical to the [MPR](./mpr.md)
output so callers can share decode helpers:

```text
[0..3]   normal.xyz, depth
[4..7]   point_a.xyz, _
[8..11]  point_b.xyz, _
[12..15] status_bits_as_f32, _, _, _
```

This format (depth in slot `[3]`, status in slot `[12]`) lets the
host detect FALLBACK by reading just one of the 16 floats.
