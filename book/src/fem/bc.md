# Boundary conditions

Every assembled matrix needs boundary data folded into the linear
system before CG / GMRES will produce a meaningful answer. Molla's BC
machinery lives in `crates/molla-fem/src/bc.rs` and supports five
variants under a single enum:

```rust
pub enum BoundaryCondition {
    Dirichlet        { vertex: u32, value: Real },
    DirichletVector  { vertex: u32, values: Vec<Real> },
    Neumann          { face: [u32; 3], flux: Real },
    NeumannTraction  { face: [u32; 3], traction: [Real; 3] },
    WeakDirichletDG  { vertex: u32, value: Real, sigma: Real },
}
```

All BCs are addressed by **global DoF index** (for vector spaces:
`dof = vertex * dim + component`). The caller decides which vertices
get pinned given the mesh — there's no automatic boundary-detection
in the BC reducer itself.

## Symmetric Dirichlet reduction — the canonical reducer

`apply_dirichlet` (scalar) and `apply_bcs` (vector-aware) are the
workhorses. `apply_dirichlet` is a thin wrapper:

```rust
pub fn apply_dirichlet(k: &mut BsrMatrix, f: &mut [Real], bcs: &[BoundaryCondition]) {
    apply_bcs(k, f, bcs, &[], 1);
}
```

`apply_bcs` runs four passes in order:

1. **Pass 1 — integrate `NeumannTraction` into `f`**. For every
   `NeumannTraction { face, traction }`, compute the 3D triangle area
   `0.5 · |b - a × c - a|` and add `(area / 3) · traction[c]` to the
   RHS at each face vertex's component DoF. This is the consistent P1
   surface-integral lumping of a uniform traction. Order matters —
   Dirichlet overwrites the RHS, so any traction silently assigned to
   a constrained DoF is dropped (textbook "Dirichlet wins").

2. **Pass 2 — collect pinned DoFs into a flat `Vec<Option<Real>>`**.
   We need the full pin set up-front so the column-elimination pass
   can shift the RHS in one sweep. `DirichletVector` expands to one
   pin per component DoF in interleaved layout.

3. **Pass 3 — symmetric column elimination**. For every non-pinned
   row \\( i \\), look at every entry \\( K_{ij} \\) where \\( j \\) is
   pinned with value \\( v_j \\); shift
   \\( f_i \mathrel{-}= K_{ij} v_j \\) and zero \\( K_{ij} \\). This is
   what restores the symmetry that the textbook row-only "pin" breaks:
   without it, a non-zero Dirichlet value would leak its contribution
   asymmetrically.

4. **Pass 4 — pin each constrained row**. Zero every off-diagonal
   entry of the row, set the diagonal to 1, set `f[r] = value`.

The end result is an SPD reduced matrix when the parent is SPD, and a
RHS that produces the correct interior values for non-zero Dirichlet
data without further bookkeeping. CG / PCG retains its convergence
guarantee on the reduced system. The test
`dirichlet_symmetric_reduction_preserves_symmetry` pins this down on
a tridiagonal Laplacian-like fixture.

### Why symmetric reduction matters

Some textbooks teach the simpler asymmetric reducer ("zero the row, set
the diagonal to 1"). That works for *zero* Dirichlet values but not for
non-zero ones — and it breaks symmetry, so CG drifts off its
convergence guarantee. The molla reducer always does the column shift
+ matrix zero so the resulting system is symmetric, SPD if the parent
was, and correct for arbitrary Dirichlet values.

```text
Before pinning vertex 0 to v = 2:    After symmetric reduction:

  [ 3  -1   0   0 ] [u₀]   [1]        [ 1   0   0   0 ] [u₀]   [ 2 ]
  [-1   3  -1   0 ] [u₁]   [1]        [ 0   3  -1   0 ] [u₁]   [ 3 ]    ← shift +2
  [ 0  -1   3  -1 ] [u₂]   [1]        [ 0  -1   3  -1 ] [u₂]   [ 1 ]
  [ 0   0  -1   3 ] [u₃] = [1]        [ 0   0  -1   3 ] [u₃] = [ 1 ]
```

After CG, `u₀ = 2` exactly (identity row pin) and the interior values
are the correct response to the BC shift.

## `Neumann` is currently a no-op

The scalar `Neumann { face, flux }` variant is recorded in the BC list
but not integrated into the RHS — the v1 weak form already drops the
boundary integral for the zero-flux case, which is the homogeneous
default. The `neumann_is_a_noop_in_v1` test pins this. For non-zero
fluxes, callers should integrate the surface contribution themselves
(or use `NeumannTraction` for the vector-valued elasticity case,
which is fully wired).

## `NeumannTraction` for elasticity

```rust
NeumannTraction { face: [u32; 3], traction: [Real; 3] }
```

is the workhorse for "push this face with that traction vector" in
elasticity scenes. The reducer integrates the consistent P1 surface
integral and folds the resulting nodal forces into `f` before
Dirichlet pinning. The adjoint chapter shows how this entry shape
makes the gradient `∂J/∂t` a one-liner.

## DG Nitsche weak-Dirichlet

```rust
WeakDirichletDG {
    vertex: u32,
    value: Real,
    sigma: Real,
}
```

is fundamentally different — it never overwrites a row. Instead it
modifies the global DG matrix by **adding** the symmetric
interior-penalty boundary block

\\[
a_{\partial\Omega}(u, v) = -\alpha \int_{\partial\Omega} (\nabla u \cdot \mathbf{n}) v \, dS
                          - \alpha \int_{\partial\Omega} u (\nabla v \cdot \mathbf{n}) \, dS
                          + \frac{\sigma \alpha}{h_F} \int_{\partial\Omega} u \, v \, dS
\\]

and the matching RHS

\\[
\ell_{\partial\Omega}(v) = -\alpha \int_{\partial\Omega} u_D (\nabla v \cdot \mathbf{n}) \, dS
                         + \frac{\sigma \alpha}{h_F} \int_{\partial\Omega} u_D v \, dS,
\\]

where \\( u_D \\) is the prescribed Dirichlet trace (linearly
interpolated between the per-vertex `value`s on each boundary face).

Apply via `apply_dg_bcs`:

```rust
pub fn apply_dg_bcs(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
    mesh: &FemMesh,
    degree: usize,
    alpha: Real,
) -> Result<()>;
```

Faces where not every vertex carries a `WeakDirichletDG` entry are
skipped; per-face \\( \sigma \\) is the max of the constituent vertex
sigmas. The default penalty
\\( \sigma_{\text{rec}} = \text{degree} \cdot (\text{degree} + 1) \cdot 10 \\)
is exposed by:

```rust
pub fn default_nitsche_sigma(degree: usize) -> Real;
```

That value sits in the "large enough for coercivity, small enough to
avoid conditioning blow-up" sweet spot from the literature
(Arnold-Brezzi-Cockburn-Marini).

The regular `apply_bcs` silently drops `WeakDirichletDG` entries so
the same `Vec<BoundaryCondition>` can carry both CG and DG-bound
variants. Internally, `apply_dg_bcs` re-extracts triplets from the
matrix via column-scans, merges in the boundary triplets, and rebuilds
the BSR — a `O(n²)` operation for `n` rows. It's only used at solver
setup, not per-iteration, so the cost is acceptable.

### When to use weak vs strong Dirichlet

| Property               | Strong (`Dirichlet`)         | Weak (`WeakDirichletDG`)         |
|------------------------|------------------------------|----------------------------------|
| Continuity preserved   | yes (DoF pinned exactly)     | only as `σ → ∞`                  |
| Works on DG fields     | only on per-cell DoFs        | yes (face integral)              |
| Affects matrix pattern | no (zeroes existing entries) | yes (rebuilds BSR with new block)|
| Conditioning           | identical to parent          | grows linearly with `σ`          |

Strong Dirichlet is the right answer for CG fields. Weak Dirichlet is
the right answer for DG, where strong pinning has no clean semantics
because the DoFs are *per cell* — you'd have to pick which cell's DoF
gets pinned at a boundary vertex.

## Cross-references

- [Solvers](./solvers.md) — every solver calls `apply_bcs` or
  `apply_dirichlet` before handing the reduced matrix to CG.
- [Adjoint](./adjoint.md) — the adjoint solve inherits the same BC
  reduction; `NeumannTraction` entries are stripped before pinning so
  they don't double-count into the adjoint RHS.
- [DG basis](./dg_basis.md) and [DG BC](./dg_bc.md) — where the
  Nitsche path is used in anger.
- [Signorini](./signorini.md) — non-penetration constraints are an
  *inequality* version of Dirichlet; the Uzawa loop builds on top of
  the standard BC reducer.
