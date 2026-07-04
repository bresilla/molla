# Shape primitives

The collision pipeline is parameterised over a single enum
([`ShapeGeometry`](https://github.com/anthropics/molla/blob/main/crates/molla/src/sim/builder.rs#L17)
in `crates/molla/src/sim/builder.rs:17`) covering every primitive
molla knows how to collide. Each variant ships a closed-form support
function (`crates/molla/src/geometry/support.rs`) so GJK
([GJK](./gjk.md)), EPA ([EPA](./epa.md)), and MPR ([MPR](./mpr.md))
can drive every convex shape uniformly.

```rust
pub enum ShapeGeometry {
    Sphere      { radius: Real },
    Plane       { normal: Vec3 },
    Box         { half_extents: Vec3 },
    Capsule     { half_height: Real, radius: Real },
    Cylinder    { half_height: Real, radius: Real },
    Cone        { half_height: Real, radius: Real },
    Ellipsoid   { half_axes: Vec3 },
    Triangle    { v0: Vec3, v1: Vec3, v2: Vec3 },
    TriMesh     { mesh_id: u32 },
    ConvexHull  { hull_id: u32 },
    Heightfield { field_id: u32 },
    Sdf         { sdf_id: u32 },
}
```

`Capsule` / `Cylinder` / `Cone` are oriented along the shape-local
`+Y` axis. The body's `Transform` rotates them into world frame.

## Support functions

Every convex variant has an analytic
`support(direction) -> point` map declared in
`crates/molla/src/geometry/support.rs`. The signatures (slightly
abridged):

```rust
pub fn support(geom: ShapeGeometry, xf: &Transform, dir_world: Vec3) -> Vec3;
pub fn support_with_hulls(geom: ShapeGeometry, xf: &Transform, dir_world: Vec3, hulls: &[Vec<Vec3>]) -> Vec3;

pub fn support_sphere     (radius: Real, dir: Vec3) -> Vec3;
pub fn support_box        (half: Vec3,   dir: Vec3) -> Vec3;
pub fn support_capsule    (half_height: Real, radius: Real, dir: Vec3) -> Vec3;
pub fn support_cylinder   (half_height: Real, radius: Real, dir: Vec3) -> Vec3;
pub fn support_cone       (half_height: Real, radius: Real, dir: Vec3) -> Vec3;
pub fn support_ellipsoid  (half_axes: Vec3,   dir: Vec3) -> Vec3;
pub fn support_convex_hull(vertices: &[Vec3], dir: Vec3) -> Vec3;
pub fn support_triangle   (v0: Vec3, v1: Vec3, v2: Vec3, dir: Vec3) -> Vec3;
pub fn support_minkowski  (
    geom_a: ShapeGeometry, xf_a: &Transform,
    geom_b: ShapeGeometry, xf_b: &Transform,
    dir_world: Vec3,
) -> Vec3;
```

The public `support` rotates `dir_world` into shape-local space via
`xf.rotation.conjugate()`, evaluates the analytic map, and rotates
the result back. Degenerate (zero-length) directions short-circuit
to the shape origin (`crates/molla/src/geometry/support.rs:29-34`).

### Per-shape formulas

* **Sphere** — `r · normalize(dir)`. Trivial.
* **Box** — pick the corner whose component signs match `dir`:
  `Vec3::new(±h.x, ±h.y, ±h.z)` with the sign of each component
  copied from `dir`.
* **Capsule** — end-cap centre selected by `sign(dir.y)`, plus a
  sphere support of the cap radius along `dir`. Three lines of code
  (`support.rs:93-97`).
* **Cylinder** — `±half_height` along Y plus the radial edge
  `radius · dir.xz / |dir.xz|`. The degenerate `dir = ±Y` case picks
  `+X` as a stable fallback so GJK doesn't NaN.
* **Cone** — compares `apex_dot = half_height·dir.y` against
  `base_edge·dir`. Whichever is larger wins; the apex is at
  `(0, +half_height, 0)`, the base disc at `y = -half_height`. Note
  the standard graphics convention (apex up) — some CAD tools use
  apex-down.
* **Ellipsoid** — closed form
  `(a²·dx, b²·dy, c²·dz) / |(a·dx, b·dy, c·dz)|`. Reduces to a sphere
  when all three semi-axes match (`support.rs:144-158`). Mirrors
  Newton's `geometry/ellipsoid.py`.
* **ConvexHull** — linear scan `argmax_v v·dir` over the hull's
  vertices. O(N) per query; adequate for hulls up to ~50 verts
  (Stage 21 [hull builder](./hull.md) caps at that). For larger
  hulls a DOP / hill-climb with edge graph would be needed but the
  CPU pipeline does not currently use one.
* **Triangle** — `argmax` of three vertex dot products. Used by the
  per-triangle narrow-phase dispatch when a `TriMesh` leaf is
  emitted.

`Plane`, `TriMesh`, `Heightfield`, and `Sdf` return `Vec3::ZERO`
from `support` — these shapes are routed through dedicated narrow-
phase paths because their support map is unbounded (plane) or
discontinuous (heightfield).

## Distance / closest-point operations

`pair_contact` in [narrow_phase.rs](./narrow_phase.md) ships
closed-form distance & contact routines for the cheap pairs:

* **Sphere–sphere** — `sphere_sphere` (`narrow_phase.rs:392`). One
  subtract + length check + normal divide.
* **Sphere–plane** — project the centre onto the plane normal,
  subtract the radius.
* **Sphere–box** — clamp the sphere centre to the box's local AABB
  (Mat3 rotate-into-local), recover the world contact point.
* **Sphere–capsule** — `closest_point_on_segment` against the
  capsule's spine, then a sphere-sphere reduction.
* **Capsule–capsule** — `closest_segment_segment` for the two spines
  (`narrow_phase.rs:1113`) followed by a sphere-sphere reduction on
  the cap radii.
* **Capsule–plane** — two end-caps tested independently, up to two
  manifold points.
* **Box–plane** — eight corners against the plane half-space; keep
  the four deepest (`box_plane`, `narrow_phase.rs:522`).
* **Box–box** — full SAT with 15 separating axes (3 + 3 face
  normals plus 9 edge-edge cross products), face / edge / face-edge
  manifold extraction (`box_box_sat`, `narrow_phase.rs:594`).
* **Ellipsoid–sphere** / **Ellipsoid–plane** — closest-point on
  ellipsoid via the first-octant Newton iteration in
  `closest_point_on_ellipsoid` (`narrow_phase.rs:1196`).

Everything else (cylinder–box, capsule–box, cone–\*, hull–\*,
ellipsoid–{box, capsule, cylinder, ellipsoid}, triangle–\*) routes
through `gjk_epa_pair_with_hulls` (`narrow_phase.rs:1385`). Plane
fallbacks (`cylinder_plane_fallback`, `cone_plane_fallback`,
`hull_plane_fallback`) exist because plane is a half-space and GJK
isn't well-defined against it.

## Triangle meshes, heightfields, SDFs

`TriMesh` / `Heightfield` / `Sdf` are *first-priority* in the
dispatch (`narrow_phase.rs:314-331`). They have to come before the
generic GJK arm because their support maps are not well-defined —
the routing would otherwise feed a broken support into GJK.

* `TriMesh` → `trimesh_pair` walks the mesh BVH ([BVH](./bvh.md)) and
  dispatches per-triangle narrow-phase against the other shape.
* `Heightfield` → expands the AABB into the height grid, generates
  the touching triangles on the fly, and dispatches as for TriMesh.
* `Sdf` → ball-marches against the SDF gradient (analytic SDFs only;
  `SdfGrid` voxelised fields go through the
  [hydroelastic](./hydroelastic.md) path).


## Helical thread SDF

`SdfShape::HelicalThread { major, minor, pitch, half_height }` — a circle
of radius `minor` swept along the helix of radius `major` with pitch
`pitch`, capped to `|y| ≤ half_height`. Distance is evaluated in the
wrapped screw-profile plane `(r − major, y − pitch·θ/τ mod pitch)` — the
standard swept-circle screw SDF, exact in the profile plane. The oracle
that distinguishes it from a torus-ring stack is **screw symmetry**:
rotating by α about +Y while translating by `pitch·α/τ` leaves the
distance field invariant (checked to 1e-9 over 24 angles in
`heightfield_and_sdf.rs`) — rings have no axial coupling, so a bolt can
never *thread* through them.
