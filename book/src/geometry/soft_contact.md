# Soft contact pipeline

Stage 33 ships cross-mesh soft-vs-soft vertex-triangle (VT) and
edge-edge (EE) contact for cloth and tet meshes, plus continuous
collision detection (CCD) for tunnelling-prevention at large
substeps. The pipeline composes:

```
SoftSurface  ──▶  SoftBvh  ──▶  VT / EE contact gather
   ▲                              │
   └── ClothMeshId / TetMeshId    └── soft_ccd::vt_ccd → root-finder
```

Raw registry descriptors live under `crates/molla/src/sim/soft_surface.rs`
because they are model data. Geometry re-exports those descriptors for
compatibility and owns the derived acceleration/contact pieces in
`crates/molla/src/geometry/`: `soft_surface.rs` (`boundary_facets`),
`soft_bvh.rs`, and `soft_ccd.rs`.

## `SoftSurface` registry

Cloth tris and tet boundary tris are unified under a single
"soft surface" so the downstream contact gather can treat them
interchangeably:

```rust
pub enum SoftSurfaceKind {
    TetMesh, // boundary triangulation of a volumetric tet mesh
    Cloth,   // tessellation tris (already in Model.tri_indices)
}

pub struct ClothMeshId(pub u32);

pub const INVALID_VERT: u32 = u32::MAX;

pub struct SoftSurface {
    pub kind:                SoftSurfaceKind,
    pub vertex_offset:       u32,                  // index into Model.particle_q
    pub vertex_count:        u32,
    pub tri_offset:          u32,                  // index into Model.soft_surface_tris
    pub tri_count:           u32,
    pub edge_offset:         u32,                  // Stage 33.1 — EE contact
    pub edge_count:          u32,
    pub tet_mesh_id:         Option<TetMeshId>,
    pub cloth_mesh_id:       Option<ClothMeshId>,
    pub self_collision:      bool,                 // Stage 33.3
    pub friction:            Real,                 // Stage 33.5
    pub restitution:         Real,                 // Stage 33.5
}

pub fn compute_tet_surface_triangles(tets: &[[u32; 4]]) -> Vec<[u32; 3]>;
pub fn compute_surface_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]>;
pub fn boundary_facets(mesh: &FemMesh) -> Vec<[u32; 3]>;
```

`compute_tet_surface_triangles` extracts the boundary triangulation
of a tet mesh via the standard "tris that belong to exactly one
tet" pass — group by sorted vertex triple, keep tris with count 1.
The original CCW winding (from outside the tet) is preserved so
downstream normal queries get the outward direction.

A key invariant: the resulting tri list is **sorted by sorted-key**
(`sim/soft_surface.rs`). `HashMap` iteration order isn't
stable across runs; without the sort, the per-tri global IDs that
downstream code (LBVH, soft-VT contact gather, friction warm-start
LUT keyed on `(vertex_global_id, tri_global_id)`) computes would
shuffle frame to frame — breaking warm-starts and parity tests.

`compute_surface_edges` returns the unique undirected edges as
`[min, max]` pairs. Used by Stage 33.1's EE contact gather.

## `SoftBvh` — cross-mesh BVH

```rust
pub struct SoftBvh {
    pub tri_aabbs:      Vec<Aabb>,
    pub tri_to_surface: Vec<u32>, // global tri idx → SoftSurface idx
    // internal nodes hidden
}

impl SoftBvh {
    pub fn build(model: &Model, state: &State) -> Self;
    pub fn refit(&mut self, model: &Model, state: &State);
    pub fn query_aabb(&self, min: Vec3, max: Vec3, out: &mut Vec<u32>);
}
```

The BVH spans **every soft-surface triangle** across every cloth
and tet mesh — one tree for the whole soft-body world. The
contact gather walks per-vertex queries through this BVH; the
per-tri global IDs index back into `Model.soft_surface_tris` and
the surface ownership in `tri_to_surface` lets the gather filter
out same-surface pairs (Stage 33's cross-mesh-only default) or
allow them (Stage 33.3 self-collision, gated by
`SoftSurface::self_collision`).

Refit-only update is cheap (AABB-only, no topology change). The
full rebuild only re-pays itself when the soft mesh count
changes — which it doesn't, mid-rollout.

GPU port (Stage 33.2) is the existing `crate::kernels::lbvh`
dispatcher reusing the AABB array prepared here. The CPU oracle
remains live for the per-vertex contact gather logic.

## VT + EE contact

The contact gather pass walks per-vertex queries through the
`SoftBvh`:

1. **AABB query** — expand the vertex's swept AABB by the contact
   margin, recover the candidate triangle list.
2. **Per-tri test** — for each candidate, classify the closest
   point on the triangle (3 vertices, 3 edges, 1 face — the
   standard 7-region split from `closest_point_on_triangle` in
   [sdf.md](./sdf.md)). Emit a VT contact when the projected
   distance falls below the margin.
3. **Same-surface filter** — for non-self-collision surfaces,
   skip pairs where `tri_to_surface[tri_idx]` matches the
   vertex's owning surface. For self-collision surfaces, also
   skip triangles *topologically* incident to the vertex (looked
   up via `Model.soft_vertex_incident_tris`) to avoid trivial
   "I touch myself" emissions.

Stage 33.1 EE contact reuses the same BVH plus the per-surface
edge lists from `compute_surface_edges`. Edge-edge contact is
emitted when the segment-segment closest-points pair falls below
the margin AND the projections are in the interior of both
edges (Bridson's classic test).

## VT-CCD root finding (Stage 33.4)

Per-substep VT contact misses **fast tunnelling** — a vertex
moving fast enough to skip past a triangle between consecutive
positions doesn't trigger the discrete test. Stage 33.4 adds the
continuous variant:

```rust
pub fn vt_ccd(
    v_at_t0:   Vec3,
    v_at_t1:   Vec3,
    tri_at_t0: [Vec3; 3],
    tri_at_t1: [Vec3; 3],
) -> Option<Real>; // earliest t ∈ [0, 1] of valid contact, or None

pub fn vt_ccd_normal(
    v0: Vec3, v1: Vec3,
    tri0: [Vec3; 3], tri1: [Vec3; 3],
    t: Real,
) -> Vec3; // outward triangle normal at TOI t
```

### Math

Given a vertex and a triangle that each move *linearly* between
`t = 0` and `t = 1`, we want the earliest `t* ∈ [0, 1]` at which
the vertex passes through the triangle plane AND the projection
lies inside the triangle. The Bridson (2003) formulation
expresses the signed distance from vertex to triangle plane as

```text
f(t) = ((B(t) - A(t)) × (C(t) - A(t))) · (V(t) - A(t))
```

which is a **cubic** in `t` (each of `(B-A)`, `(C-A)`, `(V-A)` is
linear in `t`; the cross product is quadratic; the dot is cubic).
Roots of `f(t) = 0` are the *coplanar instants*; we filter each
root by whether the barycentric projection sits inside the
triangle.

### Root-finder strategy

Two stages (`soft_ccd.rs:101-167`):

1. **Sign-change sweep.** Sample `f(t)` at 20 equally-spaced
   sub-intervals across `[0, 1]`. Each sign change pins down at
   least one root in that sub-interval; bisect to refine.
2. **Bisection.** Standard binary halving for up to 60 iterations
   (`2⁻⁶⁰ ≈ 8.7e-19`, far below the `ROOT_EPS = 1e-7` tolerance).
   Exits early when `|f(t)| < ROOT_EPS` or the interval shrinks
   below `ROOT_EPS`.

```rust
// soft_ccd.rs:129-166 (excerpt)
const N: usize = 20;
let mut prev_f = f(0.0);
for k in 1..=N {
    let t = k as Real / N as Real;
    let curr_f = f(t);
    if (prev_f >= 0.0) != (curr_f >= 0.0) {
        if let Some(t_root) = bisect(&f, prev_t, prev_f, t, curr_f) {
            if let Some(t_hit) = check_root(t_root, ...) {
                return Some(t_hit);  // earliest valid root wins
            }
        }
    }
    prev_t = t;
    prev_f = curr_f;
}
```

The inside-triangle filter (`check_root`) computes barycentric
coords at the candidate `t` and rejects projections that fall
outside the triangle (with a small slack `BARY_SLACK = 1e-4`
to admit grazing edge contacts).

### Termination + slack

`ROOT_EPS = 1e-7` is the practical convergence threshold (tighter
just burns iterations — the snap-back in the integrator uses
`TOI - eps` with a `1e-4` eps anyway).

`BARY_SLACK = 1e-4` is generous enough to admit legitimate
grazing-edge contacts but tight enough to reject projections that
sit well outside the triangle. This is the dominant source of
"missed" CCD contacts; setting it tighter rejects valid edge cases
and setting it looser pulls in spurious near-misses.

v1 ships VT-CCD only; EE-CCD (the 4D quartic case) is deferred to
Stage 33.4.1.

## Friction + restitution (Stage 33.2.1 / 33.5)

Per-surface friction and restitution coefficients live on
`SoftSurface::friction` and `SoftSurface::restitution`. Pair-rules:

* **Friction** combines via geometric mean:
  `μ = sqrt(μ_a · μ_b)`. Zero on either side disables tangential
  resistance for the contact.
* **Restitution** combines via `min(e_a, e_b)` — molla's
  pessimistic rule; one inelastic surface dominates.

Defaults per kind: `Cloth { friction = 0.4, restitution = 0.0 }`,
`TetMesh { friction = 0.4, restitution = 0.1 }`. Settable via
the `SoftSurface` getters/setters at builder time.

## Cross-mesh path

The end-to-end cross-mesh contact path (cloth-vs-cloth,
cloth-vs-tet, tet-vs-tet) hands off through:

1. `SoftBvh::refit` updates the per-tri AABBs from
   `state.particle_q`.
2. Per-vertex broad-phase query produces a candidate tri list.
3. Per-tri VT test produces a discrete-time contact OR
4. `vt_ccd` produces a continuous-time-of-impact (TOI) when
   the discrete test misses but the swept AABB indicates a
   crossing.
5. Emitted contacts feed [XPBD](../solvers/xpbd.md) /
   [VBD](../solvers/vbd.md) penalty terms with friction +
   restitution per the rules above.

Per-substep refit + per-vertex tree traversal makes the dominant
cost `O(N_vertices × log(N_tris))`. For two ~150-tet jelly cubes
the per-step cost is well under 0.5 ms on CPU; GPU port (Stage
33.2) brings it under 50 µs.
