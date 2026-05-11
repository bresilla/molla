# Narrow phase

The narrow phase consumes the broad-phase pair list and produces
real contacts. Entry point is `pair_contact` in
`crates/molla/src/geometry/narrow_phase.rs:48`:

```rust
pub fn pair_contact(
    model:               &Model,
    state:               &State,
    pair:                ShapePair,
    default_friction:    Real,
    default_restitution: Real,
    out:                 &mut Vec<ContactManifold>,
);

pub struct ContactManifold {
    pub a_shape:             u32,
    pub b_shape:             u32,
    pub point:               Vec3,
    pub normal:              Vec3, // unit, points A → B
    pub depth:               Real, // ≥ 0 on output
    pub friction:            Real,
    pub rolling_friction:    Real,
    pub torsional_friction:  Real,
    pub restitution:         Real,
}

pub const MAX_MANIFOLD_POINTS: usize = 4;
```

The contact normal points from `a` toward `b`. Solvers correct
`b += depth * normal` and `a -= depth * normal`. `depth > 0` is the
overlap; values arrive non-negative.

## Per-pair material rule

`pair_material` (`narrow_phase.rs:382`) combines per-shape friction
/ restitution / rolling / torsional values via `max`, matching
Newton's pair rule. The shape arrays are guaranteed finite at this
point — the builder substitutes the `NaN` "use default" sentinels
with the configured defaults at finalize time. Pairs that exceed
the array (which would indicate an upstream bug) fall back to the
caller-supplied `default_*`.

## Dispatch table

`pair_contact` is one giant `match` on `(geom_a, geom_b)`. The
order matters: special primitives (`Plane`, `TriMesh`,
`Heightfield`, `Sdf`) are dispatched **before** the generic GJK +
EPA arm because GJK is not well-defined against unbounded /
discontinuous supports.

### Fast paths (closed-form)

* `Sphere`–`Sphere` → `sphere_sphere`
* `Sphere`–`Plane` → `sphere_plane` (with `flip_manifold` reverse)
* `Sphere`–`Box` → `sphere_box`
* `Box`–`Plane` → `box_plane` (up to 4 manifold points — keep four
  deepest corners)
* `Box`–`Box` → `box_box_sat` (15-axis SAT with face/edge/face-edge
  manifold extraction, `narrow_phase.rs:594`)
* `Sphere`–`Capsule` → `sphere_capsule`
* `Capsule`–`Plane` → `capsule_plane` (up to 2 manifold points,
  one per cap)
* `Capsule`–`Capsule` → `capsule_capsule` via
  `closest_segment_segment`
* `Sphere`–`Ellipsoid` → `ellipsoid_sphere`
* `Plane`–`Ellipsoid` → `ellipsoid_plane`

Each routine has a "reversed" sibling that calls into the canonical
form then maps the output via `flip_manifold` so the `a`/`b` order
on the input pair is preserved on the output.

### Plane fallbacks

* `Cylinder`–`Plane` → `cylinder_plane_fallback`
* `Cone`–`Plane` → `cone_plane_fallback`
* `ConvexHull`–`Plane` → `hull_plane_fallback`

These exist because plane is a half-space — GJK can't terminate
against it (the support function is unbounded). The fallback
walks the convex shape's "lowest" support points along the plane
normal directly.

### Mesh / heightfield / SDF (priority-routed)

`(TriMesh, _)` / `(_, TriMesh)` → `trimesh_pair`. The mesh-side
[BVH](./bvh.md) is walked: query the mesh BVH with the other
shape's world AABB, recover the triangle list, dispatch
per-triangle narrow phase.

`(Heightfield, _)` / `(_, Heightfield)` → `heightfield_pair`.
Expands the AABB into the grid coordinate system, generates
touching triangles on the fly (two per quad cell), routes through
the same triangle-based GJK+EPA path as TriMesh.

`(Sdf, _)` / `(_, Sdf)` → `sdf_pair`. Ball-marches against the
analytic SDF gradient. (Voxelised SDF grids go through the
[hydroelastic](./hydroelastic.md) path instead, since they're the
"soft body has a pressure field" use case.)

### GJK + EPA

Everything else (capsule-box, cylinder-cylinder, ellipsoid-box,
triangle-\*, cone-\*, hull-\*) routes through
`gjk_epa_pair_with_hulls` (`narrow_phase.rs:1385`). [GJK](./gjk.md)
either returns "disjoint" (no contact) or hands its 4-vertex
simplex to [EPA](./epa.md) for the penetration-depth pass. The
`hulls` parameter resolves `ShapeGeometry::ConvexHull { hull_id }`
support queries against the model's hull table.

## `NarrowPhaseBackend` (Stage 28.1 v2)

The backend is selectable at runtime via
`CollisionPipeline::set_narrow_phase_backend`:

```rust
pub enum NarrowPhaseBackend {
    Gjk,    // (default) CPU GJK + EPA via pair_contact
    MprCpu, // CPU MPR portal refinement (geometry/mpr.rs)
    MprGpu, // GPU MPR batch dispatch (kernels/mpr.rs)
}
```

* **`Gjk`** — the default. Best general-purpose choice; no GPU
  dependency, exhaustive shape-type support.
* **`MprCpu`** — prefer when EPA's polytope expansion stalls (deep
  penetration scenes — cloth reset states, MPM coupling). MPR
  returns depth + normal in one algorithm.
* **`MprGpu`** — batches all eligible convex pairs through one
  dispatch on the [`MprKernel`](./mpr.md). Wins on pair counts ≥
  256 where the per-dispatch overhead amortizes.

Pairs the chosen backend cannot resolve (Plane, TriMesh,
Heightfield, SDF, Triangle on either side, or GPU `FALLBACK`
status) fall through to the CPU GJK + EPA path transparently —
callers always get a complete manifold set.

When the full GPU narrow phase is enabled
(`CollisionPipeline::enable_gpu_full_narrow_phase`), it takes
precedence over the MPR backend choice. Both run on the GPU but
only one at a time — they're alternative implementations of the
same convex-pair contract.

## Output contract

`pair_contact` pushes 0..N `ContactManifold` entries onto `out`,
where `N ≤ MAX_MANIFOLD_POINTS = 4`. Box–box and box–plane can
produce 4; cylinder–plane / capsule–plane up to 2; everything else
0 or 1. The material fields (`friction`, `rolling_friction`,
`torsional_friction`, `restitution`) are stamped on every emitted
manifold by the post-loop at `narrow_phase.rs:368-374` — the
geometry routines themselves don't touch them.
