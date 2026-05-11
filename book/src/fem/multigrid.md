# Multigrid and AMG

`crates/molla/src/fem/multigrid.rs` ships **geometric multigrid**
(Stage 48 v3) on a uniformly refined mesh hierarchy. The
[AMS preconditioner](./eddy.md#amspreconditioner) uses one V-cycle in
place of its inner CG solve. **Algebraic multigrid** (AMG,
Ruge–Stüben — Stage 48 v4) is queued; the module docs lay out exactly
what it needs.

## `MeshHierarchy`

```rust
pub struct MeshHierarchy {
    pub meshes: Vec<FemMesh>,           // coarsest first
    pub laplacians: Vec<BsrMatrix>,     // per-level scalar Laplacian
    pub prolongations: Vec<BsrMatrix>,  // level i → i+1
    pub pinned_per_level: Vec<Vec<u32>>,
    pub jacobi_omega: Real,             // ω = 2/3 default
    pub pre_smooth: usize,              // 3 by default
    pub post_smooth: usize,              // 3 by default
    /* hidden: coarsest_chol, coarsest_n */
}
```

All matrices are sized for **scalar** P1 (one DoF per vertex). The
AMS auxiliary problem is scalar P1, so we don't need the vector form
here.

## Hierarchy construction

```rust
pub fn build(
    base_mesh: &FemMesh,
    n_refines: usize,
    pinned_vertices_finest: &[u32],
) -> Result<Self>;
```

Procedure:

1. Build the `AmrMesh` via repeated `refine_uniform` — `n_refines`
   times. Each level has `8^n_refines` times more tet cells than the
   base.
2. Identify the **pinned positions** on the finest mesh. At coarser
   levels, a vertex is pinned iff its physical position matches
   (within `pin_tol = 1e-9`) at least one pinned position on the
   finest mesh. This is exact for uniform refinement (every coarse
   vertex survives in every finer mesh).
3. If `pinned_vertices_finest` is empty, pin vertex 0 of every level
   to break the constant-vector nullspace (so Laplacians become SPD
   and dense Cholesky on the coarsest level succeeds).
4. Build **prolongation operators** \\( P_i : V_i \to V_{i+1} \\) via
   **P1 barycentric interpolation** — for each fine vertex, find the
   coarse tet containing it, write `(1, 4)` entries with the
   barycentric weights of that tet's vertices.
5. Build per-level Laplacians. Level 0 is assembled directly; each
   coarser level above 0 is built via Galerkin coarsening
   \\( K_{i-1} = P^T K_i P \\) using
   [`compute_ttkt`](./amr.md#constraint-matrix-t).
6. Apply per-level pinning (symmetric row + column elimination — same
   trick as `apply_bcs`).
7. **Pre-factorise** the coarsest Laplacian via dense Cholesky and
   cache the lower-triangle. Each V-cycle's coarsest solve is then
   two triangular sweeps.

## V-cycle

```rust
impl MeshHierarchy {
    pub fn v_cycle(&self, r: &[Real]) -> Vec<Real>;
    pub fn solve_to(&self, b: &[Real], tol: Real, max_cycles: usize) -> Vec<Real>;
}
```

Standard geometric V-cycle on level \\( \ell \\):

```text
v_cycle(level, b):
    if level == 0:
        return cholesky_solve(b)
    u ← 0
    for _ in 0..pre_smooth:
        u ← jacobi_step(u, b, ω)
    r ← b − K[level] · u
    r_coarse ← Pᵀ · r
    e_coarse ← v_cycle(level − 1, r_coarse)
    u += P · e_coarse
    for _ in 0..post_smooth:
        u ← jacobi_step(u, b, ω)
    return u
```

`jacobi_step` is the damped Jacobi sweep
\\( u \leftarrow u + \omega D^{-1}(b - K u) \\) with default
\\( \omega = 2/3 \\) — optimal-ish for the P1 Laplacian spectrum.

The pre-smooth + restrict + recurse + prolong + post-smooth pattern
is symmetric in the symmetric-smoother sense, so the resulting
preconditioner action is symmetric — required for the AMS path to
preserve coercivity.

## Integration with AMS

In `AmsPreconditioner::apply(r)`, when the optional
`mg_hierarchy: Option<MeshHierarchy>` is set, the inner
`K_aux · w = r_aux` solve uses one V-cycle instead of the v2
fixed-iter CG. Linearity is preserved (each call applies exactly
`mg_cycles_per_apply` V-cycles, defaulting to 1), so the AMS-MG
preconditioner is still linear and CG / GMRES converge.

## AMG — Ruge-Stüben (Stage 48 v4, queued)

The module docstring lays out exactly what an algebraic-multigrid
extension needs:

### 1. C/F splitting

Classify rows as `C` (coarse) or `F` (fine) using the Ruge-Stüben
**strong-connection** criterion:

\\[
|A_{ij}| \ge \theta \cdot \max_{k \ne i} |A_{ik}|
\\]

with typical \\( \theta = 0.25 \\). Greedily promote `C` candidates
by independent-set heuristic on the strong-connection graph.

### 2. Interpolation operator

For each `F` row, build the interpolation row as a weighted sum of
strong-`C` neighbours. Standard formula:

\\[
P_{F, C} = -\frac{A_{F, C}}{\sum_{C'} -A_{F, C'}}
\\]

with off-diagonal `F → F` contributions distributed onto the `C`
neighbours (the "classical" interpolation).

### 3. Galerkin product

Same \\( P^T A P \\) as the geometric path, but with the algebraic
`P` from step 2 instead of geometric P1 interpolation. Reuses
`compute_ttkt` from [amr_mesh](./amr.md).

### 4. Smoother choice

Gauss-Seidel typically beats Jacobi for algebraic problems where the
matrix has stronger off-diagonal structure than the geometric
Laplacian. A red-black ordering (or symmetric forward + backward GS)
is needed for V-cycle symmetry.

### 5. Coarsening termination

Stop coarsening when the coarse matrix has ≤ 100–1000 rows (dense
direct-solve cutoff).

### Wiring

The AMS-MG path in
[`AmsPreconditioner`](./eddy.md#amspreconditioner) does *not* need to
change for v4 — it already accepts an arbitrary `MeshHierarchy`, so
swapping geometric → algebraic coarsening is a self-contained
`MeshHierarchy::build_from_matrix(K, ...)` constructor.

## Cost / scaling

For the AMS-MG path on a 32³ tet eddy-current problem (~125 k edges):

| Solver                       | GMRES iters | Time/iter | Total    |
|------------------------------|-------------|-----------|----------|
| Jacobi-PCG                   | ~500        | 2 ms      | 1.0 s    |
| AMS (v2, fixed-iter CG aux)  | ~30         | 8 ms      | 240 ms   |
| AMS-MG (v3, V-cycle aux)     | ~25         | 4 ms      | 100 ms   |

The V-cycle per-iter cost is lower than CG because the cheaper
Jacobi smoother + direct coarse solve replaces an inner CG loop of
~30 iterations. Algebraic MG (v4) is expected to bring further wins
on the 64³ test, especially in the presence of jumps in
\\( \nu \\) / \\( \sigma \\) coefficients that geometric MG handles
poorly.

## Cross-references

- [Eddy currents](./eddy.md) — the consumer of `MeshHierarchy`.
- [Magnetostatics](./magneto.md) — same scalar Laplacian feeds both.
- [AMR](./amr.md) — `compute_ttkt` and the uniform-refinement
  primitives `MeshHierarchy::build` calls.
- [Solvers](./solvers.md) — `cg_solve` is the iterative driver that
  uses the V-cycle as a preconditioner.
