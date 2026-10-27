# MPR (Minkowski Portal Refinement)

MPR is an alternative narrow-phase to [GJK](./gjk.md) + [EPA](./epa.md).
Like GJK it operates on the Minkowski difference `A − B`, but
instead of building a simplex around the origin and then
*expanding* it (the EPA step), it constructs a **portal** — a
triangle on the boundary of the Minkowski difference whose
extension hits the origin — and refines it until the portal face
is tight against the origin.

Two practical wins over GJK + EPA:

1. **One algorithm, both answers.** GJK only resolves yes/no
   overlap; EPA computes depth + normal as a separate pass. MPR
   returns depth + normal in a single algorithm.
2. **Robust on deep penetration.** EPA's polytope expansion can
   diverge when the GJK simplex is highly distorted (50%
   interpenetration, concentric cylinders, MPM reset states). MPR's
   portal refinement is a single smooth descent and tends to
   converge where EPA stalls.

References:

* Snethen (2007), the original MPR paper.
* Newhouse, *Game Programming Gems 7* (2008) ch. 2.5,
  "XenoCollide: Complex Collision Made Simple".

Stage 28 ships the CPU implementation at
`crates/molla/src/geometry/mpr.rs`; Stage 28.1 adds the WGSL port
in `crates/molla-geometry/shaders/mpr.wgsl`.

## Public API

```rust
pub struct Manifold {
    pub point_a: Vec3,
    pub point_b: Vec3,
    pub normal:  Vec3, // unit, points A → B
    pub depth:   Real,
}

pub trait SupportFn {
    fn support(&self, dir: Vec3) -> Vec3;
}

pub struct ShapeSupport<'a> {
    geom:  ShapeGeometry,
    xf:    &'a Transform,
    hulls: &'a [Vec<Vec3>],
}

impl<'a> ShapeSupport<'a> {
    pub fn new(geom: ShapeGeometry, xf: &'a Transform) -> Self;
    pub fn with_hulls(geom: ShapeGeometry, xf: &'a Transform, hulls: &'a [Vec<Vec3>]) -> Self;
}

pub fn mpr_collide<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(
    a:         &A,
    b:         &B,
    tolerance: Real,
    max_iters: u32,
) -> Option<Manifold>;
```

The `SupportFn` trait abstracts over analytic primitives, hulls,
and closures — the algorithm only ever queries a shape through it.
The output mirrors [`EpaContact`](./epa.md) so callers can swap GJK
+ EPA ↔ MPR freely.

## Portal-loop algorithm

The portal loop has five phases (`mpr.rs:150-257`):

**Phase 1 — Interior point `v0`.** Compute a point that's
guaranteed inside the Minkowski difference. Strategy: average
opposing supports per shape to get each shape's *centroid
estimate* (always interior to a convex shape), then return the
difference. Using six cardinal-axis directions
(`±X`, `±Y`, `±Z`, `mpr.rs:119-137`) avoids the degenerate case
where `+X` / `−X` supports of a flat plate collapse to the same
point.

**Phase 2 — First portal vertex `v1`.** Take a support along the
ray from `v0` toward the origin, i.e. along `-v0`. If
`v1 · (-v0) ≤ 0`, the origin is outside the supporting half-space →
no overlap; return `None`.

**Phase 3 — Second portal vertex `v2`.** Search direction is
perpendicular to the segment `(v0, v1)` and in the half-plane
containing the origin: `v0 × v1`. Take a support along that
direction; reject if it's not past the origin half-plane.

**Phase 4 — Third portal vertex `v3` (the "origin-encloses"
loop).** Form a candidate face from `(v0, v1, v2)` with normal
pointing away from `v0` (toward the boundary). Take a support `v3`
past that face. Test whether the ray from `v0` to the origin
pierces the triangle `(v1, v2, v3)`:

* If the origin is "outside" edge `(v1, v3)` of the v0–v1–v3
  plane, replace `v2` with `v3` and retry.
* If the origin is outside edge `(v2, v3)` of the v0–v2–v3
  plane, replace `v1` with `v3` and retry.
* Otherwise the triangle `(v1, v2, v3)` brackets the
  origin-direction ray → proceed to refinement.

The test uses sign comparisons on `v0.w.cross(v3.w).dot(v_i.w)`,
which is proportional to the signed volume of the tetrahedron
`(origin, v0, v3, v_i)`.

**Phase 5 — Portal refinement.** Each iteration:

```text
n        = (v2 - v1) × (v3 - v1)             // portal-face normal
if n · (v1 - v0) < 0: n = -n                  // outward orient
face_dist = v1 · n̂                            // origin-to-plane distance
v4        = support(n̂)                        // sample past the portal
new_dist  = v4 · n̂

if new_dist - face_dist < tolerance:          // ← termination
    return extract_contact(v1, v2, v3, n̂)
```

If termination doesn't fire, replace whichever of `v1` / `v2` /
`v3` keeps the new sub-tetrahedron `(v0, v1, v2, v3)` bracketing
the ray `v0 → origin`. The replacement test is the standard
Snethen 2007 §3.3 / Newhouse 2008 §2.5 criterion — compare signs
of `(v4 × v0) · v_i` (proportional to the signed volume of the
tetra `(origin, v0, v4, v_i)`):

```rust
// mpr.rs:323-344
let cross_v4_v0 = v4.w.cross(v0.w);
let d1 = v1.w.dot(cross_v4_v0);
let d2 = v2.w.dot(cross_v4_v0);
let d3 = v3.w.dot(cross_v4_v0);
if d1 >= 0.0 {
    if d2 >= 0.0 { v3 = v4; } else { v2 = v4; }
} else {
    if d3 >= 0.0 { v1 = v4; } else { v3 = v4; }
}
```

The convergence criterion (`new_dist - face_dist < tolerance`) is
the right one for the MPR loop: it says "the next support point is
not significantly further past the portal face than the portal
face is from the origin", i.e. the polytope can't expand
meaningfully in that direction any more. Newton uses `1e-6`;
values down to `1e-9` improve accuracy at the cost of a couple
extra iters.

## When MPR beats GJK/EPA

* **Cylinder-cylinder, capsule-capsule.** The Minkowski difference
  of two cylinders along arbitrary axes has many near-flat regions
  that EPA's polytope expansion picks up coplanar faces from,
  stalling convergence. MPR's smooth portal descent handles them
  cleanly.
* **Deep penetration.** When the GJK simplex contains the origin
  with one vertex much further than the others, EPA's first
  polytope is highly distorted. MPR doesn't need the GJK simplex
  at all — its interior point comes from a centroid estimate.
* **Reset states / large penetration jumps.** Cloth that snaps to
  a new rest pose, MPM particles re-seeded inside a rigid body —
  scenarios where the previous frame's contact info is invalid.

## Manifold extraction

After convergence, `extract_contact` (`mpr.rs:354-374`) projects
the origin onto the portal plane, computes barycentric coordinates
of that projection on triangle `(v1, v2, v3)`, and mixes the
per-shape support points by the same weights to recover
`point_a` / `point_b` on the original shapes.

The normal is the converged outward portal-face normal (which is
the contact normal `A → B` in world coordinates because the
Minkowski difference is `A − B`). On degenerate portals (zero-area
triangle), the routine falls back to the last valid normal seen
during the loop.

## GPU port (Stage 28.1)

`shaders/mpr.wgsl` runs **one thread per pair** matching the GJK /
EPA kernels — the portal loop is purely scalar per pair (3
vertices + 1 interior point + one new sample per iteration), so
workgroup-level parallelism wouldn't help. Output stride is 16
`f32` per pair, identical to EPA so the host can share decode
helpers:

```text
[0..3]   normal.xyz, depth
[4..7]   point_a.xyz, _
[8..11]  point_b.xyz, _
[12..15] status_bits_as_f32, _, _, _
```

Termination criterion matches CPU verbatim
(`new_dist - face_dist < tolerance`). If the iteration cap is hit
the kernel emits the current portal with the last valid normal as
`STATUS_OK`, **not** `FALLBACK` — callers can inspect the depth
and decide whether to accept (the v1 GPU result agrees with CPU to
~6 decimal digits even on hard cases). Pairs that genuinely can't
proceed (degenerate Minkowski difference) emit `STATUS_FALLBACK`
and the host re-runs them on the CPU GJK + EPA path.

Enable with `CollisionPipeline::set_narrow_phase_backend(NarrowPhaseBackend::MprGpu)`
plus `enable_mpr_gpu_backend(&engine, capacity, tolerance)`. The
backend is at runtime parity with the CPU MPR backend; the GPU
wins on pair counts ≥ ~256 where the per-dispatch overhead
amortises.
