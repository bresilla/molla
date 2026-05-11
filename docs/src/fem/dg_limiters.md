# DG slope limiters

A DG-P1 or DG-P2 reconstruction can develop spurious oscillations
near discontinuities ("Gibbs phenomenon" of finite-element flavour).
Slope limiters clip the per-cell reconstruction so it stays within a
monotone envelope derived from the cell averages of the
neighbourhood. Molla ships two:

- **Barth-Jespersen** (Stage 43 v1) — scalar, per-component, robust
  default.
- **Krivodonova characteristic-wise** (Stage 43 v3) — projects onto
  the eigenvector basis of the local Roe-averaged Jacobian, then
  limits each characteristic field separately.

## Barth-Jespersen — `limit_solution`

```rust
pub fn limit_solution(
    mesh: &FemMesh,
    u_dg: &mut [Real],
    degree: usize,
    cell_neighbours: &[Vec<u32>],
    physical_min: Real,
    physical_max: Real,
) -> Vec<Real>;  // per-cell phi factor
```

The algorithm — applied per cell \\( K \\) — is:

1. **Cell average** \\( \bar u = \frac{1}{N} \sum_{i \in K} u_i \\) (arithmetic
   mean of the per-cell Lagrange-node values; equal-weight average is
   conventional in production DG codes; for P2 it differs from the
   exact mass-weighted average by O(h²), within the limiter's
   accuracy budget).
2. **Neighbour envelope** \\( u_{\min} = \min_{n \in N(K)} \bar u_n \\),
   \\( u_{\max} = \max_{n \in N(K)} \bar u_n \\) including \\( K \\)
   itself, then clamped to the user-supplied `physical_min` /
   `physical_max` envelope.
3. **Per-node clip factor**

   \\[
   y_i = \begin{cases}
   \min(1, (u_{\max} - \bar u) / (u_i - \bar u)) & \text{if } u_i > \bar u \\
   \min(1, (u_{\min} - \bar u) / (u_i - \bar u)) & \text{if } u_i < \bar u \\
   1 & \text{if } u_i = \bar u
   \end{cases}
   \\]

4. **Cell-wide factor** \\( \phi = \min_{i \in K} y_i \in [0, 1] \\).
5. **Apply**: \\( u_i \leftarrow \bar u + \phi (u_i - \bar u) \\).
   Then hard-clamp each node to `[physical_min, physical_max]` as a
   last-resort guard.

The returned `Vec<Real>` of length `cell_count` carries the per-cell
\\( \phi \\) values for diagnostics — a cell with \\( \phi = 1 \\) is
unlimited (smooth) and a cell with \\( \phi = 0 \\) collapses to a
constant.

### When to limit

The textbook discipline (Cockburn–Shu) is to limit **after every
SSP-RK substage**. The [time integrators](./dg_time.md) take a
`post_stage` closure that the caller wires to a limit-solution call:

```rust
let neighbours = build_cell_neighbours(&mesh);
ssp_rk3_step(&mut u, t, dt, |u, _t| rhs(u), |u| {
    limit_solution(&mesh, u, degree, &neighbours, 0.0, Real::INFINITY);
});
```

For P0, the limiter just clamps to the physical envelope (no slope to
limit) and returns `phi = 1` for every cell.

## Krivodonova — `limit_solution_characteristic`

For **systems** of conservation laws (SWE in molla's case), limiting
each conservative variable independently leaks information between
characteristic fields and produces over-/under-shoot in the wrong
direction. Krivodonova (2007) solved this by limiting in
**characteristic variables**:

```rust
pub fn limit_solution_characteristic(
    mesh: &FemMesh,
    u_dg: &mut [Real],
    degree: usize,
    cell_neighbours: &[Vec<u32>],
    n_components: usize,
    physical_min: Real,
    physical_max: Real,
    g: Real,
);
```

The algorithm:

1. **Compute conservative cell averages** \\( \bar U_K \\) (same as
   Barth–Jespersen, just per-component).
2. **For each cell \\( K \\)**, build a single **Roe-average state**
   from `K` and its neighbours.
3. **Pick a "principal" normal** — for tri / tet meshes molla uses
   the area-weighted average of the outward face normals (so the
   limiter projects onto the eigenvector basis of the
   *flux Jacobian in the dominant flow direction*).
4. **Eigendecompose** the Roe-averaged Jacobian \\( \hat A(\hat U) \cdot \mathbf{n}_K \\)
   to get the right eigenvectors \\( R \\) and left eigenvectors
   \\( L \\). For SWE, the eigenvalues are
   \\( (\hat u - \hat c, \, \hat u, \, \hat u + \hat c) \\) along
   \\( \mathbf{n}_K \\).
5. **Project** the conservative slopes into characteristic space:
   \\( W = L \cdot (U_i - \bar U_K) \\) per node, plus the neighbour
   differences \\( L \cdot (\bar U_n - \bar U_K) \\).
6. **Apply Barth-Jespersen per characteristic component** of \\( W \\).
7. **Project back** with \\( R \\) and overwrite the conservative
   nodal values.

The wet-bed floor in `compute_swe_numerical_flux_roe`'s logic is
mirrored here — when \\( h \\) drops below a small \\( \epsilon \\),
the wave-speed denominators are clamped and the limiter falls back to
conservative-variable BJ for that face.

This produces noticeably **sharper** contacts than scalar BJ on the
dam-break test (`DgShallowWaterDamBreak`, viewer scene #97): the post-
shock height drop is captured in 1–2 cells instead of 3–5.

## Limiter sanity tests

| Test                                | What it checks                          |
|-------------------------------------|-----------------------------------------|
| `limit_solution_constant_field`     | \\( \phi = 1 \\) everywhere, no change. |
| `limit_solution_clamps_overshoot`   | A node overshooting the envelope gets clipped to it. |
| `limit_solution_characteristic_swe_dam_break` | Sharper shocks than scalar BJ on the same geometry. |

## Limitations

Barth-Jespersen has the known property that it **reduces to first
order** at smooth extrema — a smooth peak gets clipped because the
neighbour envelope sees a local max there and the limiter activates.
For DG-P2 this means the formal third-order accuracy degrades to
first-order near extrema; the practical fix is the Krivodonova
characteristic limiter (which detects monotone-vs-extremum locally
via the projected slopes), or a more sophisticated **WENO-style**
limiter that's queued for a future stage.

## Cross-references

- [DG flux](./dg_flux.md) — the inter-element coupling that the
  limiter runs after.
- [DG time](./dg_time.md) — SSP-RK time stepper's `post_stage`
  closure is where you'd plug the limiter in.
- [DG GPU](./dg_gpu.md) — the GPU port of the SWE fluxes; the limiter
  currently stays on CPU pre/post the GPU dispatch.
