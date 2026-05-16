# GJK

Gilbert–Johnson–Keerthi finds whether two convex shapes overlap.
On overlap it yields the 4-vertex Minkowski-difference simplex that
[EPA](./epa.md) needs for the penetration-depth pass. On
disjoint inputs it additionally returns the closest points on each
shape and the distance between them.

References:

* Ericson, *Real-Time Collision Detection* (2005), §9.5.
* van den Bergen, *A Fast and Robust GJK Implementation* (1999).

CPU implementation: `crates/molla/src/geometry/gjk.rs`. GPU
boolean-overlap port: `crates/molla-geometry/shaders/gjk.wgsl` (Stage 15).

## Public API

```rust
pub struct GjkDistance {
    pub point_a:  Vec3,
    pub point_b:  Vec3,
    pub distance: Real,
}

pub enum GjkResult {
    Disjoint(GjkDistance),
    Overlap(GjkSimplex),
}

pub struct GjkSimplex {
    pub points: Vec<SupportPoint>,
}

pub struct SupportPoint {
    pub w: Vec3, // support_A(d) − support_B(−d)  (Minkowski-diff)
    pub a: Vec3, // support_A(d)                 (world)
    pub b: Vec3, // support_B(−d)                (world)
}

pub fn gjk(
    geom_a: ShapeGeometry, xf_a: &Transform,
    geom_b: ShapeGeometry, xf_b: &Transform,
) -> GjkResult;

pub fn gjk_with_hulls(
    geom_a: ShapeGeometry, xf_a: &Transform,
    geom_b: ShapeGeometry, xf_b: &Transform,
    hulls:  &[Vec<Vec3>],
) -> GjkResult;
```

`SupportPoint` keeps the per-shape support points alongside the
Minkowski-difference vertex. That lets the disjoint-case path
recover world-space closest points on the originals via the
simplex's barycentric weights without re-running the support map.

## Algorithm

GJK walks a simplex (up to 4 vertices) through Minkowski-difference
space, searching toward the origin. Each iteration:

1. Compute `v` = closest point on the current simplex to the
   origin in Minkowski space.
2. Search direction is `-v`. Take a new support point along that
   direction in the Minkowski difference.
3. Termination check: if the new support is no closer to the
   origin than `v` already is, the simplex has converged → return
   `Disjoint`.
4. Otherwise add the new point and reduce the simplex to the
   smallest sub-feature (vertex / edge / triangle / tetrahedron)
   containing the origin's projection.
5. If the tetrahedron contains the origin → return `Overlap`.

The termination criterion (Ericson §9.5) is

```text
v·v - v·w ≤ ε · v·v        where w is the new support point.
```

i.e. the new support cannot push the closest point closer to the
origin than the current `v`.

```rust
// gjk.rs:111-128 (excerpt)
let v_dot_v = v.length_squared();
let new_pt  = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, -v);
let v_dot_w = v.dot(new_pt.w);
if v_dot_v - v_dot_w <= 1e-10 * v_dot_v {
    let (pa, pb, dist) = closest_on_simplex_with_points(&simplex);
    return GjkResult::Disjoint(GjkDistance { point_a: pa, point_b: pb, distance: dist });
}
```

## Simplex update

`do_simplex` routes by current simplex size:

* **2 vertices (line).** `do_simplex_line` (`gjk.rs:204-219`):
  classify origin against the edge — if "behind" the newer vertex,
  drop the older one; otherwise keep both and search perpendicular
  via the triple product `(ab × ao) × ab`.
* **3 vertices (triangle).** `do_simplex_triangle` (`gjk.rs:221-255`):
  test the origin against each edge plane; either drop a vertex
  and recurse, or pick the half-space above/below the triangle and
  search the face normal.
* **4 vertices (tetrahedron).** `do_simplex_tetra`
  (`gjk.rs:257-293`): test against the three faces incident to the
  newest vertex (ABC / ACD / ADB); drop the opposite vertex if any
  is on the origin's side, otherwise the tetra contains the
  origin → `ContainsOrigin`.

The simplex always keeps the newest support point at the back of
the `Vec` and uses index conventions
`[a]` / `[b, a]` / `[c, b, a]` / `[d, c, b, a]` (newest last) so
the routines can address by position without tracking which vertex
is which.

## Closest-point recovery

When GJK terminates `Disjoint` with 1, 2, or 3 simplex vertices,
`closest_on_simplex_with_points` projects the origin onto the
simplex sub-feature, recovers barycentric weights, and combines
the per-shape `a`/`b` support points by the same weights:

```rust
// gjk.rs:350-373 (excerpt)
fn closest_on_simplex_with_points(s: &[SupportPoint]) -> (Vec3, Vec3, Real) {
    match s.len() {
        1 => (s[0].a, s[0].b, s[0].w.length()),
        2 => {
            let t  = project_t_onto_segment(s[0].w, s[1].w);
            let pa = s[0].a + (s[1].a - s[0].a) * t;
            let pb = s[0].b + (s[1].b - s[0].b) * t;
            (pa, pb, ((s[0].w + (s[1].w - s[0].w) * t).length()))
        }
        3 => /* barycentric on triangle */,
        4 => /* should be Overlap; fall back to closest face */,
    }
}
```

This is what makes GJK *useful* for non-contact queries
(distance-to-shape pickers, swept tests, IK reachability): the
"disjoint" branch is genuinely informative, not just a no-overlap
flag.

## Iteration cap

`MAX_ITERS = 64` (`gjk.rs:90`). On adversarially-shaped inputs
(numerically near-tangent surfaces) the no-progress termination
fails to trigger and GJK can spin. The cap means it bails out and
treats the situation as "touching" — handing off whatever closest
points the current simplex produces. Empirically the cap is hit
only on synthetic stress tests; production scenes converge in
≤ 8 iterations.

## GPU port (Stage 15)

`shaders/gjk.wgsl` runs the boolean overlap variant — one thread
per shape pair from the broad phase. The per-pair output is a
`u32` flag (0 = disjoint, 1 = overlap). When the host has bound a
real simplex output buffer the kernel additionally writes the
4-vertex Minkowski-difference simplex into
`gjk_simplex[i]` (12 × `vec4<f32>` per pair: 4 vertices × 3 vec3
each for `w`, `a`, `b`) for EPA to bootstrap from.

The kernel inlines support functions for Sphere / Box / Capsule /
Cylinder / Cone / Ellipsoid. `ConvexHull` and `TriMesh` stay on
CPU — they would need additional storage for hull vertex tables
and per-mesh BVH traversal that's not worth the per-pair-kernel
complexity at current pair counts. Plane is also CPU-handled
because GJK doesn't terminate against half-spaces.

This kernel feeds the [GPU prefilter](./narrow_phase.md) — its
job is to rapidly cull non-overlapping pairs and pass only the
overlapping ones to the full GPU EPA pass.
