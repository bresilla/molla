# Adaptive Mesh Refinement (AMR)

`crates/molla-fem/src/amr_mesh.rs` (~1750 lines) is the second-largest
single source file in the FEM stack. It builds a refinement-tree on
top of a base `FemMesh`, manages hanging nodes, balances levels 2:1,
and produces a Galerkin-reduced system `Kc = TᵀKT` that you can hand
to the existing solvers. Stages 45 v1 through v4 cover uniform refine,
non-uniform refine via a criterion, DG-AMR face matching, coarsening
with Krylov restart.

## Data model

```rust
pub struct AmrMesh {
    pub base_mesh: FemMesh,
    pub refined_cells: Vec<RefinedCell>,
    pub refinement_levels: Vec<u8>,

    // hidden:
    cells: Vec<FemCell>,                // flat pool — leaves + refined
    vertices: Vec<Vec3>,                // flat pool — base + midpoints
    is_refined: Vec<bool>,
    is_removed: Vec<bool>,              // v4 coarsening
    midpoint_cache: HashMap<(u32, u32), u32>,
    hanging_nodes: Vec<HangingNode>,
}
```

The "flat pool" model is what makes AMR efficient:

- `cells[0..base.cell_count()]` are the base (level 0) cells. Indices
  beyond that are children added by refinement.
- `vertices[0..base.vert_count()]` are the base vertices. Indices
  beyond that are edge midpoints inserted by refinement.
- A cell is a **leaf** iff `!is_refined[i] && !is_removed[i]`.
- The `midpoint_cache` keys on the *sorted* edge endpoint pair so
  two neighbouring cells refining the same edge produce the same
  midpoint vertex — the mesh stays watertight on shared, fully-refined
  edges.

### Refinement records

```rust
pub struct RefinedCell {
    pub parent: u32,                   // flat cell index of parent
    pub children: RefinedChildren,     // 4 (Tri) or 8 (Tet) indices
    pub level: u8,                     // parent.level + 1
}

pub enum RefinedChildren {
    Tri([u32; 4]),
    Tet([u32; 8]),
}
```

Triangle 1→4 refinement is the standard "red" Bey split: insert each
edge midpoint, produce four congruent children (three corner-mass +
one centre-up). Tetrahedron 1→8 refinement uses a fixed canonical
ordering of the six edge midpoints that mirrors Bey's tet
red-refinement scheme.

## Refinement API

```rust
pub trait RefinementCriterion {
    fn should_refine(&self, cell: &FemCell, vertices: &[Vec3]) -> bool;
}

// Blanket impl for closures: any Fn(&FemCell, &[Vec3]) -> bool works.

pub fn near_point(p: Vec3, radius: Real) -> impl RefinementCriterion;

impl AmrMesh {
    pub fn new(base_mesh: FemMesh) -> Self;
    pub fn refine_uniform(&mut self);
    pub fn refine_where<C: RefinementCriterion>(&mut self, criterion: &C);
    pub fn coarsen_where<C: RefinementCriterion>(&mut self, criterion: &C);
    pub fn flatten(&self) -> FemMesh;
    pub fn flatten_with_remap(&self) -> (FemMesh, HashMap<...>);

    pub fn compute_hanging_nodes(&mut self);
    pub fn enforce_2to1_balance(&mut self);

    pub fn cell_count(&self) -> usize;
    pub fn vert_count(&self) -> usize;
    pub fn cells(&self) -> &[FemCell];
    pub fn hanging_nodes(&self) -> &[HangingNode];
}
```

`refine_uniform` is the v1 base operation — it refines every leaf
once. `refine_where` is the v2 selective version that takes a
criterion. `coarsen_where` (v4) removes children whose parents are no
longer flagged.

## Hanging nodes

When two adjacent leaves are at different refinement levels, the
fine side's edge has a midpoint vertex that the coarse side doesn't.
That midpoint is a **hanging node**:

```rust
pub struct HangingNode {
    pub dof: u32,         // AMR vertex id of the constrained midpoint
    pub parents: [u32; 2],// AMR vertex ids of the endpoints
}
```

The P1 constraint imposed is the simple linear interpolation

\\[
u_m = \tfrac{1}{2}(u_a + u_b).
\\]

`compute_hanging_nodes` walks every leaf-edge that introduces a
midpoint and records the constraint. It's called automatically after
every `refine_*` operation.

### ASCII diagram — single hanging node

```text
                                            Hanging node m
                                                  ↓
   Coarse leaf:                  ┌───────────┬───●───┬───────────┐
   one edge a-b                  │           │       │           │
                                 │           ╲       ╱           │
                                 │           │  ╲    ╱           │
                                 a ──────────┴─── m ──────────── b
                                            (Fine edge endpoints: a-m, m-b)

   Constraint: u_m = ½(u_a + u_b)
```

## 2:1 balance

The "two cells along a single edge differ in level by at most 1" rule.
Larger jumps would produce hanging-node configurations our P1 reducer
can't express (the midpoint of the fine-side edge would itself be a
hanging node of a further-refined neighbour, leading to recursive
constraints). `enforce_2to1_balance` walks every leaf, looks for
neighbours whose level differs by ≥ 2, and refines the coarser one
to bring the difference back to 1. This is called automatically
after every `refine_where`.

## Constraint matrix `T`

The Galerkin reduction \\( K_c = T^T K T, \, f_c = T^T f \\) is the
textbook way to handle hanging nodes:

- \\( T \\) is `(n_total × n_free)`: maps the free-DoF vector to the
  full AMR-DoF vector via `u_full = T · u_free`.
- For an unconstrained DoF \\( i \\), row \\( i \\) of \\( T \\) is the
  unit basis vector \\( e_i \\) — that DoF is its own free variable.
- For a hanging midpoint \\( m \\) with parents \\( (a, b) \\), row
  \\( m \\) of \\( T \\) has entries \\( T_{m,a} = T_{m,b} = 1/2 \\) —
  the midpoint is expressed as the average of its parents.

The result: a Galerkin-reduced system on the free DoFs only. The
reduced matrix \\( K_c \\) is SPD if \\( K \\) is SPD (Galerkin
projection preserves symmetric definiteness), so CG works on it
directly.

```rust
pub fn assemble_global_amr<F>(
    amr_mesh: &AmrMesh,
    space: FunctionSpace,
    local_fn: F,
    f_full: Option<&[Real]>,
) -> Result<(BsrMatrix, Vec<Real>, BsrMatrix)>;  // (K_c, f_c, T)
```

The caller is responsible for **applying Dirichlet BCs to the reduced
system** — the textbook ordering is constraint-reduce first,
BC-pin second. After solving \\( K_c u_{\text{free}} = f_c \\), the
full-DoF solution is recovered via the sparse matvec `u = T · u_free`.

Helper:

```rust
pub fn tt_times_vector(t: &BsrMatrix, v: &[Real]) -> Vec<Real>;
pub fn compute_ttkt(t: &BsrMatrix, k: &BsrMatrix) -> Result<BsrMatrix>;
```

`compute_ttkt` is useful when the caller has assembled `K` separately
(e.g. for a P2 problem) and wants to reduce it against a `T` built by
`AmrMesh::build_constraint_matrix`.

## DG-AMR face matching — Stage 45 v3

For DG fields, the constraint at a 2:1-balanced face split is that
the **numerical-flux integral** over the parent face (computed from
the coarse side's basis) equals the sum of flux integrals over the
child sub-faces (each from the refined side's basis).

```rust
pub struct DgAmrFaceMatch {
    pub parent_cell: u32,
    pub parent_face: Vec<u32>,         // 2 or 3 verts
    pub child_faces: Vec<Vec<u32>>,    // child sub-faces
}

pub fn compute_dg_amr_faces(amr_mesh: &AmrMesh) -> Vec<DgAmrFaceMatch>;
```

- Tri (2D): parent edge `[a, b]` ⇒ children `[a, m], [m, b]`.
- Tet (3D): parent triangle `[a, b, c]` ⇒ four sub-triangles from the
  red-split inserting midpoints on each edge.

The DG flux assembler then walks both interior faces (one cell on
each side, same level) and the AMR face matches (one cell on the
coarse side, several on the fine side) and adds the right per-face
block to the global system.

## Coarsening + Krylov restart — Stage 45 v4

```rust
pub fn coarsen_where<C: RefinementCriterion>(&mut self, criterion: &C);
pub fn transfer_state(&self, old: &AmrMesh, u_old: &[Real]) -> Vec<Real>;
```

`coarsen_where` marks children for removal (the `is_removed` flag) but
keeps them in the flat pool for index stability. Removed cells are
skipped by every leaf-walk (`flatten`, `compute_hanging_nodes`, ...);
the `refined_cells` list has the corresponding entries trimmed so
clients iterating it see only currently-active refinements.

`transfer_state` interpolates a solution from an old AmrMesh to a new
one — used by the **Krylov restart** workflow:

1. Run iterative solver on coarse mesh to partial convergence.
2. Refine where the residual indicator says.
3. Transfer current iterate to the new mesh via `transfer_state`.
4. Use it as the warm start for the new-mesh CG.

This is how the `AmrHeat` viewer scene (#100) gets to a sharp
near-source profile without paying for a global fine mesh.

## DG-AMR coarsening — Stage 45 v5

Stage 45 v5 adds two pieces needed by transient DG-AMR runs:

```rust
pub fn AmrMesh::coarsen_subtree(&mut self, parent_idx: usize) -> Result<()>;

pub fn project_dg_p1_tri_children_to_parent_l2(
    parent: &FemCell,
    children: &[FemCell],
    vertices: &[Vec3],
    child_values: &[[Real; 3]],
) -> Result<[Real; 3]>;
```

`coarsen_subtree` collapses a deep refinement tree from the leaves
upward before coarsening the requested parent, so callers do not have to
manually un-refine every descendant first.

`project_dg_p1_tri_children_to_parent_l2` is the DG coarsening transfer
operator for triangle P1 fields. It solves the parent mass projection

```text
∫_parent N_i u_parent dx = Σ_child ∫_child N_i u_child dx
```

using the degree-2-exact triangle quadrature rule, so linear DG fields
survive child-to-parent coarsening exactly.

## Cross-references

- [Mesh](./mesh.md) — `FemMesh` is what `AmrMesh` wraps.
- [Assembly](./assembly.md) — `assemble_global_amr` extends the
  standard scatter with the constraint reduction.
- [DG flux](./dg_flux.md) — DG-AMR face matching feeds the same
  per-face block routines.
- [Multigrid](./multigrid.md) — uses `AmrMesh::refine_uniform` to
  build the geometric multigrid hierarchy.
- [BC](./bc.md) — applied to the *reduced* system after `T^T K T`.
