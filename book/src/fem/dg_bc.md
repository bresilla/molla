# DG boundary conditions

DG fields can't use **strong** Dirichlet pinning because the per-cell
DoF layout makes "which DoF gets pinned at a shared boundary vertex"
ambiguous. The textbook answer is **Nitsche weak Dirichlet** — a
penalty + consistency boundary block added to the system matrix, plus
a matching RHS contribution.

## The weak-Dirichlet variant

```rust
BoundaryCondition::WeakDirichletDG {
    vertex: u32,
    value: Real,
    sigma: Real,
}
```

Apply via `apply_dg_bcs` (described in
[boundary conditions](./bc.md#dg-nitsche-weak-dirichlet)). The
relevant math:

\\[
a_{\partial\Omega}(u, v) = -\alpha \int_{\partial\Omega} (\nabla u \cdot \mathbf{n}) v \, dS
                          - \alpha \int_{\partial\Omega} u (\nabla v \cdot \mathbf{n}) \, dS
                          + \frac{\sigma \alpha}{h_F} \int_{\partial\Omega} u \, v \, dS
\\]

with matching RHS

\\[
\ell_{\partial\Omega}(v) = -\alpha \int_{\partial\Omega} u_D (\nabla v \cdot \mathbf{n}) \, dS
                         + \frac{\sigma \alpha}{h_F} \int_{\partial\Omega} u_D v \, dS,
\\]

and \\( u_D \\) is linearly interpolated between the per-vertex
`value`s on each boundary face.

## Default penalty

```rust
pub fn default_nitsche_sigma(degree: usize) -> Real {
    (degree as Real) * ((degree + 1) as Real) * 10.0
}
```

That's the Arnold-Brezzi-Cockburn-Marini choice from the standard DG
literature: large enough to keep the boundary block coercive across
reasonable mesh sizes, small enough to avoid conditioning blow-up.
Concretely:

| Degree | `default_nitsche_sigma` |
|--------|------------------------|
| 0      | 0   (no slope; pin via flux upstream) |
| 1      | 20  |
| 2      | 60  |

The factor of 10 has some slack — you can dial it down to 5–8 for
better conditioning on smooth problems, or up to 15–20 for very
non-conforming meshes where the trace inequality constant grows. The
underlying coercivity bound is

\\[
\sigma > c_{\text{trace}}(d) \cdot |\partial K| / |K|
\\]

with \\( c_{\text{trace}} \\) a polynomial in `degree`. The factor-of-10
choice is conservatively above this for all but pathological meshes.

## How the applier works

`apply_dg_bcs` runs four passes:

1. Index BCs by vertex: build a `HashMap<u32, DgVertexBc>`.
2. Identify **boundary faces** — faces owned by exactly one cell.
   (Internally, `apply_dg_bcs` rebuilds the face → cells map; this is
   `O(cell_count · 4)` for tets and is only done once at solver
   setup.)
3. For every boundary face where *every* vertex has a
   `WeakDirichletDG` entry, compute the Nitsche boundary block via
   `compute_nitsche_boundary_block` and accumulate it into the
   global matrix triplets. Per-face σ is the **maximum** of the
   vertex sigmas.
4. Rebuild the `BsrMatrix` via column-scans + new triplets. This is
   `O(n²)` in the matrix dimension but happens only once at setup
   time — not per-step.

Faces with partial constraints (some vertices have
`WeakDirichletDG`, others don't) are silently skipped — you can't
Nitsche-pin half a face.

## Periodic face pairs

```rust
pub fn periodic_face_pairs(
    mesh: &FemMesh,
    axis: usize,
) -> Vec<PeriodicPair>;
```

For a periodic-axis SWE / Burgers test, this finds every boundary
face whose vertices have a matching opposite-side counterpart along
the given axis, and pairs them into a list of "synthetic interior
faces" that the flux assembler then treats like normal interior
faces — so a shock rolling out the right boundary re-enters on the
left without any special-casing in the RHS closure.

Mesh requirements: the grid builders `tri_grid_2d` and `tet_grid_3d`
produce face-matched periodic geometry on their axes. For
unstructured meshes the caller has to ensure the face pairs match up
exactly.

The viewer scene `DgBurgersShock` (#96) uses this to roll a shock
periodically; the algorithm correctness is checked by the
`periodic_face_pairs_finds_matching_boundary_edges_on_grid` unit
test.

## Why not implement strong Dirichlet for DG?

The "natural" strong-Dirichlet adaptation to DG would be:

- Find every boundary vertex.
- Find every cell that owns a DoF at that vertex.
- Pin all those DoFs to the prescribed value.

That works for boundary *vertices* of the DG geometry, but
- It overconstrains the system — a P2 DG element has 6/10 DoFs per
  cell, and pinning 3/4 per boundary face plus the cell vertices
  leaves no slack for the interior.
- It breaks the local-conservation property — strong pinning at the
  boundary forces the integrated flux through the boundary to whatever
  the post-pin matrix produces, which usually isn't what was intended.
- The discrete trace inequality (that makes Nitsche coercive)
  guarantees the weak form *converges to the strong* as \\( \sigma \\)
  grows — so weak Dirichlet *is* the right answer; you just pay the
  penalty in coercivity instead of in DoF count.

For CG fields, strong pinning is still preferred (`apply_bcs`); for
DG you always use `apply_dg_bcs` plus `WeakDirichletDG` entries.

## Cross-references

- [Boundary conditions](./bc.md#dg-nitsche-weak-dirichlet) — the
  applier and the enum variant.
- [DG basis](./dg_basis.md) — what DoF count a face touches.
- [DG flux](./dg_flux.md) —
  [`compute_nitsche_boundary_block`](./dg_flux.md#nitsche-boundary-block)
  is the per-face workhorse.
- [DG time](./dg_time.md) — the SSP-RK stepper that integrates the
  boundary-augmented system.
