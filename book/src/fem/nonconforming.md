# Non-conforming meshes — L² projection and contact

Two FEM meshes that share no DoFs need a transfer operator to move a
scalar field from one to the other. The textbook recipe is **L²
projection mediated by a cross-mesh mass matrix**, which lives in
`crates/molla/src/fem/nonconforming.rs`. Stages 47 v1–v4 cover scalar
projection, vector projection, surface-only (contact) variant, and a
penalty-contact force that integrates against it.

## The math

For source mesh \\( S \\) (carrying \\( u_S \\)) and target mesh
\\( T \\) (where we want \\( u_T \\)):

\\[
M_{ts}^{st} = \int_T \phi_t^T(\mathbf{x}) \cdot \phi_s^S(\mathbf{x}) \, d\mathbf{x}, \quad
M_{tt} = \int_T \phi_t^T \cdot \phi_t^T \, d\mathbf{x}, \quad
u_T = M_{tt}^{-1} \, M_{st} \, u_S.
\\]

- \\( M_{st} \\) is **rectangular** (`n_target × n_source`). It's the
  "cross-mesh mass" — couples a target scalar DoF to a source scalar
  DoF.
- \\( M_{tt} \\) is the **target mesh's own consistent mass matrix**
  (SPD, square, CG-solvable).
- The integral that defines \\( M_{st} \\) is computed on the
  **target** cells — that way \\( \phi_t^T \\) is a simple polynomial
  in each integration sub-domain, and we only need to evaluate
  \\( \phi_s^S \\) at each quadrature point via barycentric
  tet-location in the source mesh.

## Properties (P1 nodal, target-side quadrature)

- **Constant reproduction.** \\( u_S \equiv 1 \Rightarrow u_T \equiv 1 \\)
  exactly (modulo CG tolerance). The reason: \\( M_{st} \cdot \mathbf{1}_S \\)
  reproduces the target lumped row sums
  \\( \int \phi_t^T \cdot \sum_j \phi_j^S \, dx = \int \phi_t^T \, dx \\), which is
  also \\( M_{tt} \cdot \mathbf{1}_T \\).
- **Linear reproduction.** With degree ≥ 2 quadrature, the projection
  is exact on affine source fields up to floating-point round-off
  (P1 contains all affine functions exactly, and the L² projector
  onto a space that includes the source field is the identity).
- **Quadratic loss.** A quadratic source field is not in P1, so the
  projection introduces an \\( O(h^2) \\) error; halving \\( h \\)
  cuts the error by ~4× — the standard P1 best-approximation rate.

## `NonconformingField`

```rust
pub struct NonconformingField<'a> {
    pub source: &'a FemMesh,
    pub target: &'a FemMesh,
    coupling_matrix: BsrMatrix,   // M_st  (n_target × n_source)
    target_mass: BsrMatrix,       // M_tt  (n_target × n_target)
    domain: IntegrationDomain,    // Volume or Surface
}

impl<'a> NonconformingField<'a> {
    pub fn new(source: &'a FemMesh, target: &'a FemMesh) -> Result<Self>;
    pub fn new_surface(source: &'a FemMesh, target: &'a FemMesh) -> Result<Self>;
    pub fn project(&self, u_source: &[Real]) -> Result<Vec<Real>>;
    pub fn project_vector(&self, u_source: &[Real], dim: usize) -> Result<Vec<Real>>;
    pub fn is_surface(&self) -> bool;
}
```

### Volume vs surface integration

- **`new`** (v1) integrates \\( M_{st} \\) over the target tets.
  Quadrature: 5-point Keast (degree-3 exact in barycentric
  coordinates).
- **`new_surface`** (v3) integrates over the target's **outer
  surface** — the triangle facets returned by
  `geometry::boundary_facets`. Quadrature: 6-point degree-4 Dunavant
  on every facet. Interior target DoFs receive zero contribution.

Use volume for "field transfer between non-matching solid domains";
use surface for **contact** formulations where only the boundary
matters.

### Tet-location strategy (v1)

v1 ships a **brute-force linear scan** over `source.cells` per
quadrature point. The same approach as
[`apic_coupling::locate_particle_tet`](./apic.md) and roughly
\\( O(N_T \cdot N_S) \\) for the matrix build. A
`TetBvh` follow-up could drop this to
\\( O(N_T \cdot \log N_S) \\) without changing the public API — the
crate `crates/molla/src/geometry/tet_bvh.rs` is the candidate. AABB
filtering on the source side is already in place, which prunes most
non-overlapping pairs cheaply.

### `project` / `project_vector`

`project` runs

\\[
u_T = M_{tt}^{-1} (M_{st} \cdot u_S)
\\]

via one matvec of `M_{st}` followed by one CG solve against `M_{tt}`.
The mass matrix is well-conditioned (\\( \kappa \sim O(1) \\)
independent of mesh size) so 200 iterations is generous — defaults
are `DEFAULT_CG_TOL = 1e-12, DEFAULT_CG_MAX_ITERS = 1000`.

For **vector-valued fields** (Stage 47 v2):

```rust
let u_t = field.project_vector(&u_source, 3)?;  // dim = 3
```

The same scalar \\( M_{st} \\) and \\( M_{tt} \\) are reused once per
component. **Storage choice:** we keep the single scalar `BsrMatrix`
and loop over components rather than building a `B = 3` block
matrix. The math is identical (the cross-mesh mass operator is
**scalar** — for an isotropic vector field the components decouple),
and a `B = 3` block matrix would hold three identical scalar copies
on its block diagonal with zero off-diagonal blocks: 3× the storage
for the same numerics. The 3× CG solves still reuse the cached SPD
`M_tt` so the cost is 3 × the scalar path with zero re-assembly.

## `NonconformingPenaltyContact` — Stage 47 v2

```rust
pub struct NonconformingPenaltyContact<'a> {
    pub field: &'a NonconformingField<'a>,
    pub stiffness: Real,
}

impl<'a> NonconformingPenaltyContact<'a> {
    pub fn compute_penalty_forces(
        &self,
        u_target: &[Real],
        u_source: &[Real],
        dim: usize,
    ) -> Result<(Vec<Real>, Vec<Real>)>;
}
```

For a slave-master pair where the slave (target) sits on or just
beyond the master (source), the displacement gap is

\\[
\Delta u = u_T - M_{tt}^{-1} M_{st} u_S
\\]

— the difference between the target's own field and the source's
field projected onto the target. The contact spring force is then

\\[
\mathbf{f}_T = -k \cdot M_{tt} \cdot \Delta u, \quad \mathbf{f}_S = +k \cdot M_{st}^T \cdot \Delta u.
\\]

The target receives `f_T = − k · M_tt · Δu` (consistent-mass
weighting); the source receives the **transpose** of the projection
acting on the same `Δu`. The relation
\\( \mathbf{1}^T M_{tt} = \mathbf{1}^T M_{st} \\) (both reproduce the
target volume measure) makes the total force vanish exactly, so
**Newton's third law holds** at the discrete level.

## Coupled elasticity Newton-Raphson — Stage 47 v4

The final v4 stage wires the non-conforming penalty contact into a
two-mesh elasticity Newton-Raphson solve. Each iteration:

1. Assemble per-mesh elastic stiffness \\( K_T, K_S \\) and residuals.
2. Compute penalty forces via
   `NonconformingPenaltyContact::compute_penalty_forces` on the
   current `(u_T, u_S)`.
3. Build a coupled tangent that combines per-mesh \\( K \\) with
   penalty Jacobian \\( k \, M_{tt} \\) (target) and \\( k \, M_{st}^T \\)
   (source).
4. Solve the coupled increment \\( (du_T, du_S) \\) via CG.
5. Update.

The result: a Hertz-style contact problem where two non-matching
elastic bodies push against each other with the right force-distance
relationship.

## What's out of scope (v4)

- **Hard Signorini contact** — see [`signorini`](./signorini.md) for
  the Lagrange-multiplier version.
- **Friction** — see [`friction`](./friction.md). The current
  v4 ships the **normal** force only.
- **Pressure contact** — Hertz with a contact pressure unknown
  requires an additional surface mass operator on the slave side plus
  a Lagrange formulation.

## Cross-references

- [Mesh](./mesh.md) — `FemMesh` for both source and target.
- [Quadrature](./quadrature.md) — the 5-pt tet and 6-pt tri-3D rules
  used to build \\( M_{st} \\).
- [Elements](./elements.md#linear-isotropic-elasticity) — the
  per-mesh stiffness assembly in the Stage 47 v4 path.
- [Signorini](./signorini.md) — exact non-penetration via Lagrange
  multipliers.
- [Friction](./friction.md) — Coulomb tangential force.
