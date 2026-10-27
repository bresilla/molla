# Discontinuous Galerkin (DG)

The DG cluster — `crates/molla-fem/src/dg_element.rs` (~2700 lines)
and `dg_time_integrator.rs` (~200 lines) — is the second-largest
chapter set in the FEM stack. DG generalises classical continuous
Galerkin (CG) by **dropping inter-element continuity**: a vertex shared
by `k` cells gets `k · ndof_per_cell` DoFs total, one set per owning
cell, and coupling between cells happens via **numerical flux**
integrals on each shared face.

The DG framework is split into six sub-chapters in this docset:

- [Basis and per-cell DoFs](./dg_basis.md) — `DGNodal { degree }`,
  P0 / P1 / P2 layouts, `dg_ndof_per_cell`, and the Lagrange-vs-
  Bernstein basis trade-off.
- [Numerical fluxes](./dg_flux.md) — the `NumericalFlux` enum:
  `Central`, `Upwind`, `LaxFriedrichs`, `Roe`, plus the SWE specials
  (HLL, HLLC, RoeSwe).
- [Slope limiters](./dg_limiters.md) — Barth-Jespersen scalar limiter
  and Krivodonova characteristic-wise limiter.
- [Time integration](./dg_time.md) — SSP-RK2 / SSP-RK3 in the
  low-storage Shu-Osher form.
- [Boundary conditions](./dg_bc.md) — Nitsche weak Dirichlet,
  periodic face pairs.
- [GPU port](./dg_gpu.md) — Stage 43 v4 status note.

This page gives the high-level overview that ties them together.

## Why DG?

Three properties make DG the canonical choice for **hyperbolic** PDEs
(conservation laws, advection, shallow-water, Euler):

1. **Local conservation.** Each cell's DoFs satisfy a discrete
   conservation law independently — the face flux is what couples them,
   and the flux is consistent. No mass leaks between elements through
   numerical error.
2. **Sharp discontinuity capture.** Per-cell DoFs let the solution
   jump at faces; combined with a slope limiter you can resolve
   shocks in one cell instead of smearing them over many.
3. **Embarrassingly parallel assembly.** No DoF is shared, so the
   element loop has no atomic-add hazard — the per-face flux block is
   the only cross-cell coupling, and that's a tightly scoped read of
   two cells' DoFs.

DG also has costs — `k×` more DoFs per shared vertex than CG, more
complex BCs (Nitsche / weak Dirichlet — see [dg_bc](./dg_bc.md)), and
limiters that need careful tuning — so molla uses CG for elliptic /
parabolic problems (heat, elasticity, Stokes) and DG for the
hyperbolic specials (Burgers shock, SWE dam-break).

## Stage timeline

| Stage    | Capability                                                  |
|----------|-------------------------------------------------------------|
| 39B v1   | P0 / P1 DG, SIP Laplacian flux, mass/stiffness per cell     |
| 43 v1    | `NumericalFlux::{Central, Upwind, LaxFriedrichs, Roe}`      |
| 43 v2    | SSP-RK2 / RK3 time integration; Nitsche weak Dirichlet      |
| 43 v3    | P2 DG, SWE-specific HLL / HLLC / RoeSwe, Krivodonova limiter |
| 43 v4    | GPU port of HLL / HLLC / RoeSwe (queued)                    |
| C2       | 2D compressible Euler physical / Rusanov / HLL flux helpers |

## The DG semi-discrete system

For a scalar conservation law \\( u_t + \nabla \cdot F(u) = 0 \\) the
DG weak form per cell \\( K \\) reads

\\[
\int_K \phi_i u_t \, dx
  - \int_K \nabla \phi_i \cdot F(u) \, dx
  + \int_{\partial K} \phi_i \, \hat{F}(u^-, u^+, \mathbf{n}) \, dS = 0,
\\]

where \\( \hat{F} \\) is the **numerical flux** that couples the two
sides of each face. The DG-mass matrix is **block-diagonal** (no
shared DoFs), so the inverse mass is cheap to apply — typically the
RHS routine inverts it on-the-fly inside `du/dt = M⁻¹ R(u)` before
handing to the [time integrator](./dg_time.md).

The `dg_element.rs` API exposes the building blocks:

For compressible-flow follow-ons, the Euler helpers mirror the SWE
pointwise flux API:

```rust
pub fn euler_pressure_2d(u: [Real; 4], gamma: Real) -> Real;
pub fn euler_flux_dot_n_2d(u: [Real; 4], n: Vec3, gamma: Real) -> [Real; 4];
pub fn compute_euler_numerical_flux_rusanov_2d(... ) -> [Real; 4];
pub fn compute_euler_numerical_flux_hll_2d(... ) -> [Real; 4];
```

The state is conservative `(ρ, ρu, ρv, E)`, with an ideal-gas EOS.

```rust
pub fn compute_element_dg_mass(cell, vertices, degree) -> Vec<Vec<Real>>;
pub fn compute_element_dg_stiffness(cell, vertices, coeff, degree) -> Vec<Vec<Real>>;
pub fn compute_dg_numerical_flux_block(...) -> ...;
pub fn compute_dg_flux_jump_block(...) -> ...;            // SIP elliptic
pub fn compute_nitsche_boundary_block(...) -> (Vec<Vec<Real>>, Vec<Real>);
pub fn compute_swe_numerical_flux_hll(...) -> ...;
pub fn compute_swe_numerical_flux_hllc(...) -> ...;
pub fn compute_swe_numerical_flux_roe(...) -> ...;
pub fn limit_solution(...);                               // Barth-Jespersen
pub fn limit_solution_characteristic(...);                // Krivodonova
pub fn compute_interior_faces(mesh) -> Vec<InteriorFace>;
pub fn periodic_face_pairs(mesh, axis) -> Vec<PeriodicPair>;
pub fn dg_nodal_positions(cell, vertices, degree) -> Vec<Vec3>;
pub fn assemble_dg_global<C, F>(...);
```

`assemble_dg_global` is the DG analogue of `assemble_global` — it
visits every cell + every interior face and scatters into a per-cell
BSR with `B = 1`.

## Per-cell DoF layout

For `DGNodal { degree }` the cell offsets are a running sum:

```text
offset[c]     = Σ_{c' < c} dg_ndof_per_cell(cells[c'], degree)
offset[c + 1] = offset[c] + dg_ndof_per_cell(cells[c], degree)
```

— so cell `c`'s local DoF `i` lives at global index
`offset[c] + i`. This is what [adjustment-set rebuilding](./amr.md)
and the
[DG-AMR face match](./amr.md#dg-amr-face-matching) routines key off.

### Worked example — DG-P1 on a 4-triangle mesh

A four-triangle mesh sharing one center vertex would have
\\( 4 \text{ cells} \times 3 \text{ DoFs/cell} = 12 \\) total DG-P1
DoFs, vs only \\( 5 \\) shared CG-P1 DoFs:

```text
CG-P1 (5 DoFs):       DG-P1 (12 DoFs):

      v1                   v1_0 ─── v1_1
     /  \                  / \     / \
   v2 ── v0 ── v3        v2_0 v0_0 v0_1 v3_1
     \  /                  \   ╳   /
      v4                  v2_2 v0_2 v0_3 v4_3
                            \ /     \ /
                             v4_2 ─── v4_3
```

Each "ghost" DoF (e.g. `v0_0`) belongs to its owning cell only —
shared vertices each show up four times (once per cell) and the four
copies are coupled only through the four edges meeting at the center.

This shape generalises directly to tets in 3D; numbers scale as
4 DoFs/cell (P1 tet), 10 DoFs/cell (P2 tet).

## Glossary

- **Average** \\( \{w\} = \frac{1}{2}(w^- + w^+) \\) — mean of the
  two-sided trace of a field on an interior face.
- **Jump** \\( [w] = w^- - w^+ \\) — scalar jump across the face;
  vector form \\( [w] \, \mathbf{n} \\) for normal-flux work.
- **Numerical flux** \\( \hat F(u^-, u^+, \mathbf{n}) \\) — the single
  flux value used at every quadrature point on a face, built from the
  two one-sided traces and the outward normal.
- **SIP** — Symmetric Interior Penalty; the standard symmetric DG
  formulation for elliptic operators. See
  [`compute_dg_flux_jump_block`](./dg_flux.md#sip-flux-for-elliptic-diffusion).

Read on for the deep dives.
