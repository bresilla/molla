# DG basis and per-cell DoFs

The `FunctionSpace::DGNodal { degree }` space (Stage 39B + 43) attaches
its DoFs to **cells**, not vertices or edges. A vertex shared by `k`
cells contributes `k · ndof_per_cell` DoFs total — one set per owning
cell — and inter-cell coupling happens only via numerical flux at
faces.

## Per-cell DoF count

```rust
pub fn dg_ndof_per_cell(cell: &FemCell, degree: usize) -> usize;
```

| Degree | Triangle (`(d+1)(d+2)/2`) | Tetrahedron (`(d+1)(d+2)(d+3)/6`) |
|--------|---------------------------|-----------------------------------|
| 0 (P0) | 1                         | 1                                 |
| 1 (P1) | 3                         | 4                                 |
| 2 (P2) | 6                         | 10                                |

v1 panics for `degree > 2` — the reference shape-function tables are
only built for those three orders, and the DG element kernels in
[`compute_element_dg_*`](./dg.md#the-dg-semi-discrete-system) match.

## Reference shape functions

On the interior of one cell, the DG nodal basis equals the matching CG
nodal basis evaluated *only on that cell*:

| Degree | Tri basis                                             | Tet basis              |
|--------|-------------------------------------------------------|------------------------|
| 0      | `1` (constant)                                        | `1`                    |
| 1      | `(λ₀, λ₁, λ₂)`                                        | `(λ₀, λ₁, λ₂, λ₃)`     |
| 2      | `[c0, c1, c2, e01, e12, e20]` (same as [P2](./p2.md)) | `[c0..c3, e01..e23]`   |

The corner / edge ordering for P2 matches `TRI_EDGES` / `TET_EDGES`
exactly. This is what makes the kernel reuse possible:

```rust
match (cell, degree) {
    (FemCell::Tri(idx), 0) => vec![vec![tri_area_2d(...)]],
    (_, 1) => compute_element_mass(cell, vertices),         // CG-P1 reuse
    (_, 2) => compute_element_p2_mass(cell, vertices),      // CG-P2 reuse
    _ => unreachable!(),
}
```

— `compute_element_dg_mass` and `compute_element_dg_stiffness` simply
forward to the CG kernels for `degree ∈ {1, 2}` because the *element-
interior* integrands are identical. The DoF map and the inter-element
flux blocks are what makes the math behave like DG.

## DG DoF offsets

```rust
let mut offsets = Vec::with_capacity(mesh.cells.len() + 1);
let mut running = 0;
for c in &mesh.cells {
    offsets.push(running);
    running += dg_ndof_per_cell(c, degree);
}
offsets.push(running);
```

For a homogeneous mesh (all tets or all tris), this collapses to
`offsets[c] = c * ndof_per_cell`. For mixed meshes (currently the v1
builders don't produce them, but the function-space supports them),
the offsets grow non-uniformly.

The `dg_nodal_positions` helper recovers the *world-space* position of
each cell-local DoF:

```rust
pub fn dg_nodal_positions(cell: &FemCell, vertices: &[Vec3], degree: usize) -> Vec<Vec3>;
```

— useful for the limiters (which need cell averages and neighbour
cell values at the parent face's quadrature points) and for the
viewer (which renders the per-cell solution at each nodal location).

## Lagrange vs Bernstein basis

Molla's DG uses the **Lagrange nodal basis** — DoFs are values of the
field at specific reference points. Pros: trivial interpolation (each
DoF is the field at its node); same code as CG-P2 internally.
Cons: the basis is *not* positivity-preserving — a DG-P2 reconstruction
of a non-negative field may produce small negative values between
nodes, which is why limiters exist.

An alternative is the **Bernstein basis** — DoFs are Bernstein-Bézier
control coefficients. Pros: positivity preserving when DoFs are
non-negative; easier to limit. Cons: requires extra change-of-basis
plumbing and the reference math is more involved. Molla has not
needed Bernstein so far; if a future stage adds limiter-free
positivity preservation, this is where it would land.

## Why per-cell DoFs?

Three reasons, each one tied to a property:

1. **Local conservation.** Each cell's residual is computed from its
   own DoFs only (plus face-flux contributions). Mass conservation is
   a per-cell identity, not a global integral.
2. **Block-diagonal mass.** Inverting the mass matrix at every RHS
   evaluation (for the explicit time stepper) costs
   \\( O(N) \\) per cell, with no global solve. This is the killer
   feature for hyperbolic problems where the explicit RK time
   stepping needs to evaluate `du/dt = M⁻¹ R(u)` hundreds of times per
   second.
3. **Sharp discontinuities.** Cells can carry different values at
   shared vertices, so a shock can sit in one cell instead of
   spreading across many. Combined with a slope limiter (see
   [dg_limiters](./dg_limiters.md)) this captures shocks in
   essentially one cell.

## Memory cost

For a 16³ tet mesh (~5 k vertices, ~25 k cells):

| Space                  | DoF count | Storage (f64) |
|------------------------|-----------|---------------|
| CG-P1                  | ~5 k      | ~40 kB        |
| CG-P2                  | ~30 k     | ~240 kB       |
| DG-P0                  | ~25 k     | ~200 kB       |
| DG-P1                  | ~100 k    | ~800 kB       |
| DG-P2                  | ~250 k    | ~2 MB         |

DG-P2 has ~50× more DoFs than CG-P1 on the same mesh — but the
explicit time stepper with block-diagonal mass + per-cell parallelism
makes it competitive for hyperbolic problems where the implicit CG
path would need a much smaller \\( \Delta t \\) for stability anyway.

## Cross-references

- [DG overview](./dg.md) — semi-discrete weak form and the chapter
  map.
- [Numerical fluxes](./dg_flux.md) — the inter-cell coupling.
- [Time integrators](./dg_time.md) — SSP-RK steppers that pair with
  the block-diagonal mass.
- [DG BC](./dg_bc.md) — Nitsche weak Dirichlet (DG can't use strong
  Dirichlet because the per-cell DoF makes "which DoF to pin at a
  shared vertex" ambiguous).
- [Limiters](./dg_limiters.md) — the post-stage hooks that clip the
  per-cell reconstruction.
