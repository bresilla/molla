# Meshes and function spaces

The geometric substrate of the FEM stack lives in
`crates/molla-fem/src/mesh.rs` — a flat `Vec<Vec3>` of vertex
positions and a `Vec<FemCell>` of cell topologies — together with the
`FunctionSpace` enum in `function_space.rs` that decides how degrees of
freedom (DoFs) are attached to the mesh.

## `FemCell`

```rust
pub enum FemCell {
    Tri([u32; 3]),
    Tet([u32; 4]),
}
```

A single FEM cell is either a triangle (three vertex indices) or a
tetrahedron (four). The enum has only two variants in v1; adding
wedge / hex / pyramid is purely additive — every kernel pattern-matches
on it and the new arms fall in naturally. Helper methods
`vert_count()` and `verts()` give scatter loops the slice form they
need.

For 2D problems we embed the triangles in the `z = 0` plane and the
element kernels project on `(x, y)` only — so `tri_grid_2d` builds a
mesh that the same `Vec3`-based assembler can handle without changing
the storage schema.

## `FemMesh`

```rust
pub struct FemMesh {
    pub vertices: Vec<Vec3>,
    pub cells: Vec<FemCell>,
}
```

That is the entire schema. There is no per-cell material slot in v1 —
material parameters travel as separate `Vec<Real>` / `Vec<Vec3>`
arrays alongside the mesh, which keeps the elasticity / Stokes / eddy
solvers from each demanding their own struct shape.

### Grid builders

Two builders cover almost every test fixture in the codebase:

```rust
FemMesh::tri_grid_2d(nx: usize, ny: usize, w: Real, h: Real)
FemMesh::tet_grid_3d(nx: usize, ny: usize, nz: usize, w: Real, h: Real, d: Real)
```

- `tri_grid_2d` produces an `nx × ny` quad grid split into two
  right-triangles per quad along the `(low-left, top-right)` diagonal.
  Both triangles are CCW-oriented so `tri_area_2d` returns positive.
- `tet_grid_3d` produces an `nx × ny × nz` hex grid where every hex is
  split into 6 tetrahedra via the **Kuhn / Coxeter** body-diagonal
  decomposition. All six tets share the `v000 → v111` body diagonal
  and have positive volume under the standard
  \\((b-a) \cdot ((c-a) \times (d-a)) > 0\\) convention. The
  `tet_grid_3d_positive_volumes` unit test asserts this on every cell.

The grid builders are deliberately the *only* topology-generating
helpers in the v1 FEM module. More exotic FEM geometries are built
procedurally by composing these builders; external mesh importers are
outside the physics-only scope.

## Edge enumeration — `compute_edges`

```rust
pub fn compute_edges(mesh: &FemMesh) -> (Vec<[u32; 2]>, HashMap<[u32; 2], u32>)
```

Several function spaces — P2 (edge-midpoint DoFs), Nedelec edge
elements (one DoF per edge), and the AMS preconditioner (uses the
discrete-gradient `G` on edges) — need to enumerate the **unique
undirected edges** of the mesh and assign each a stable ID. The helper
returns `(edges, edge_id)` where each edge is stored as
`[min, max]` and the `HashMap` gives `O(1)` reverse lookup.

A triangle contributes 3 edges; a tet contributes 6. The single-tet
`compute_edges_single_tet_has_six_edges` test pins this down.

## Function spaces

The `FunctionSpace` enum decides how DoFs are attached:

```rust
pub enum FunctionSpace {
    P1Nodal,                    // 1 scalar / vertex
    P2Nodal,                    // 1 scalar / vertex + 1 / edge
    P1Vector { dim: usize },    // dim scalars / vertex (elasticity)
    P2Vector { dim: usize },    // Taylor-Hood velocity
    DGNodal { degree: usize },  // per-cell DoFs (DG)
    NedelecEdge { order: usize }, // 1 DoF / edge (H(curl))
}
```

`ndof(mesh)` returns the global DoF count for each — `vert_count`,
`vert_count + edge_count`, `vert_count · dim`, etc. The vector
spaces use **interleaved** storage: DoF `v · d + c` for vertex `v`
component `c`. This is what every assembler / BC applier in the FEM
module assumes and is checked at every layer via `debug_assert_eq!`.

DG has its own per-cell DoF count exposed by
`dg_ndof_per_cell(cell, degree)`:

- Triangle: `(deg+1)(deg+2)/2` — 1 (P0), 3 (P1), 6 (P2).
- Tetrahedron: `(deg+1)(deg+2)(deg+3)/6` — 1, 4, 10.

For DG, a vertex shared by `k` cells contributes `k · ndof_per_cell`
total DoFs — one set per owning cell — which is the whole point of DG
(no inter-cell sharing of DoFs; coupling happens via numerical flux
across faces).

### Example DoF counts on a 2 × 2 × 2 tet grid

| Space                 | DoFs                       |
|-----------------------|----------------------------|
| `P1Nodal`             | 27  (= 3³ vertices)        |
| `P2Nodal`             | 125 (27 + 98 edges)        |
| `P1Vector { dim: 3 }` | 81                         |
| `P2Vector { dim: 3 }` | 375                        |
| `DGNodal { 0 }`       | 48  (6 tets × 8 hexes)     |
| `DGNodal { 1 }`       | 192 (4 per tet)            |
| `NedelecEdge { 1 }`   | 98  (unique edges)         |

The `function_space.rs` test suite locks in the formulas (`p2_nodal_*`,
`p1_vector_*`, `dg_p0_ndof_equals_cell_count`, ...).

## Cross-references

- The element kernels that operate on `FemCell` live in
  [elements](./elements.md).
- The assembly walker that scatters into the global `BsrMatrix` is
  described in [assembly](./assembly.md).
- The P2 mapping from `(cell, local_node_idx) → global_dof` is in
  [p2](./p2.md).
- The Nedelec mapping from `(cell, local_edge_idx) → (global_dof, sign)`
  is in [magneto](./magneto.md).
- The AMR variant — vertex pool grows under uniform / non-uniform
  refinement — is described in [amr](./amr.md).
