# Global assembly

`crates/molla-fem/src/assembly.rs` is where the [element kernels](./elements.md)
meet the sparse `BsrMatrix`. The whole module is ~160 lines because
all the heavy lifting lives in the kernels (per-cell math) and in
`BsrMatrix::from_scalar_triplets` (CSR construction with dedupe and
accumulation).

## The visitor pattern

```rust
pub fn assemble_global<F>(
    mesh: &FemMesh,
    space: FunctionSpace,
    element_visitor: F,
) -> Result<BsrMatrix>
where
    F: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>;
```

`element_visitor` is *any* function from
`(cell, vertices) → local-matrix`. v1 ships:

```rust
pub fn assemble_mass(mesh: &FemMesh, space: FunctionSpace) -> Result<BsrMatrix>;
pub fn assemble_stiffness(mesh: &FemMesh, space: FunctionSpace, coeff: Real) -> Result<BsrMatrix>;
```

as one-liners over `compute_element_mass` and
`compute_element_stiffness`. The visitor pattern means
[elasticity](./elements.md#linear-isotropic-elasticity), Stokes,
convection-diffusion, eddy currents — every operator listed in this
chapter set — plugs into the same scatter loop without changing the
top-level API.

The visitor receives the full vertex array (not just the cell's
vertices) because some kernels need geometric quantities that depend
on neighbouring cells, and because the visitor cost of indexing into
a `&[Vec3]` is negligible.

## The scatter loop

For each cell:

1. Call the visitor → `local: Vec<Vec<Real>>` of size
   `(n_v · d) × (n_v · d)`.
2. For each local entry `local[i*d + ci][j*d + cj]`:
   - Map local node `i` → global vertex `vi`, local node `j` → global
     vertex `vj`.
   - Compute `(row_global, col_global) = (vi · d + ci, vj · d + cj)`.
   - Push `(row_global, col_global, value)` into a flat
     `Vec<(usize, usize, Real)>` if `value ≠ 0`.

After all cells are visited, the triplet vector is handed to
`BsrMatrix::from_scalar_triplets(n, n, &triplets)`, which sorts,
dedupes coincident `(row, col)` slots, **accumulates** their values,
and produces the final CSR `(row_offsets, col_idx, values)`.

The triplet vector is pre-allocated with `mesh.cells.len() * 16 * d * d`
capacity to avoid reallocations in the common P1 / P1Vector case.

### CSR layout convention

Every assembled matrix uses `BsrMatrix` with **`B = 1`** — block size
one, i.e. plain scalar CSR. The choice is deliberate:

- CG / PCG / GMRES only need `mv` and `diagonal()`, both of which work
  bit-identically on `B = 1` BSR vs a hand-rolled CSR.
- The GPU port (`FemGpuData::upload`) is a direct flat-array mirror
  of the same `(row_offsets, col_idx, values)` triple.
- Vector-valued fields are handled by the **interleaved DoF layout**
  — vertex `v` component `c` lives at scalar DoF `v · d + c` — so the
  CSR row count for elasticity is `vert_count * 3`, not
  `vert_count` with `B = 3` block padding. The interleaved variant
  is what every kernel and BC reducer assumes.

### Cost

The COO → CSR step inside `from_scalar_triplets` is
\\( O(\mathrm{nnz} \cdot \log \mathrm{nnz}) \\) (sort + scan), plenty
fast at v1 mesh sizes — at 16³ tets (~5 k vertices, ~25 k cells,
~100 k triplets) it runs in under 10 ms. Larger meshes would benefit
from a row-grouped pre-pass but the current implementation has not
needed it.

## Nedelec edge-element route

The Nedelec basis attaches DoFs to mesh **edges**, not vertices, and
the local → global scatter applies an orientation sign. `assemble_global`
detects the `FunctionSpace::NedelecEdge { order }` variant up-front
and routes to a dedicated helper:

```rust
if let FunctionSpace::NedelecEdge { order } = space {
    let map = NedelecDofMap::build(mesh);
    return assemble_global_nedelec(mesh, &map, element_visitor);
}
```

That assembler is described in the
[magnetostatics chapter](./magneto.md#assembly).

## Matvec wrapper

```rust
pub fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real>;
```

is an allocating wrapper around `a.mv(x, &mut y)`. The closure-driven
[`cg_solve`](../math/iterative.md) and `gmres` solvers want
\\( y \leftarrow A x \\) as an `Fn(&[Real]) -> Vec<Real>`, so every
solver in the FEM module uses `|x| matvec(&k, x)` as its operator
closure. The intermediate `Vec<Real>` allocation per iteration is
~~heap pressure but turns out to be negligible compared to the matvec
itself for the mesh sizes molla cares about.

## Worked example — heat equation steady state

The "all-boundary Dirichlet `= 5` ⇒ interior is `5`" contract test
fits in 15 lines:

```rust
let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
let space = FunctionSpace::P1Nodal;
let k = assemble_stiffness(&mesh, space, 1.0)?;
let n = space.ndof(&mesh);
let f = vec![0.0; n];

let bcs: Vec<_> = mesh.vertices.iter().enumerate()
    .filter(|(_, p)| /* on boundary */ )
    .map(|(i, _)| BoundaryCondition::Dirichlet { vertex: i as u32, value: 5.0 })
    .collect();

let u = solve_steady_state(&k, &f, &bcs, 500, 1e-10);
assert!(u.iter().all(|&v| (v - 5.0).abs() < 1e-6));
```

The `assemble_stiffness` call invokes
`assemble_global(mesh, P1Nodal, |c, v| compute_element_stiffness(c, v, 1.0))`,
which walks all 32 triangles, calls the kernel on each
(\\( 3 \times 3 \\) blocks), scatters into 100 triplets (5 × 5 = 25
vertex grid → at most 5²·3³ entries), dedupes, and returns a CSR
matrix with ~85 nnz. The solver chapter picks up from there.

## Cross-references

- [Element kernels](./elements.md) — what each visitor computes.
- [Function space](./mesh.md#function-spaces) — what `space` parameter
  the assembler accepts.
- [Boundary conditions](./bc.md) — the BC reducer runs on the
  assembled matrix to produce the SPD reduced system.
- [Solvers](./solvers.md) — where CG meets the assembled `BsrMatrix`.
- [P2 assembly](./p2.md) — the P2 variant `assemble_global_p2` uses a
  dedicated `P2DofMap` because corner + edge-midpoint DoFs share one
  address space.
- [AMR assembly](./amr.md) — `assemble_global_amr` adds a Galerkin
  constraint reduction `Kc = T^T K T` for non-conforming refinement.
