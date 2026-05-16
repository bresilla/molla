# Magnetostatics — H(curl) Nedelec edge elements

Stage 46 brought H(curl)-conforming **first-order Nedelec edge
elements** to molla, with a steady-state curl–curl solver for
magnetostatics. The element basis lives in
`crates/molla-fem/src/nedelec_element.rs`; the magnetostatics solver
is in `magneto.rs`.

## Why edge elements?

For PDEs in \\( H(\mathrm{curl}) = \{ v \in L^2 : \mathrm{curl}\, v \in L^2 \} \\)
the natural conforming basis has **tangential continuity** across
element interfaces — the tangential component matches on shared
faces, the normal component is free to jump. This is exactly what
magnetostatics (vector potential \\( \mathbf{A} \\)) and time-harmonic
electromagnetics need: the curl operator is well-defined globally,
but \\( \nabla \cdot \mathbf{A} \\) is not constrained and
\\( \mathbf{n} \cdot \mathbf{A} \\) has no physical meaning at material
interfaces.

**Nodal (H¹) elements over-constrain such problems** by enforcing
continuity of all components; they produce spurious "ghost solutions"
that pollute the spectrum. Edge elements eliminate these by
construction: the kernel of `curl` over the discrete H(curl) space is
exactly the discrete gradient of H¹ scalar fields (the discrete de
Rham complex).

## The first-order Nedelec basis

On a tetrahedron with barycentric coordinates
\\( \lambda_0, \lambda_1, \lambda_2, \lambda_3 \\) and constant
barycentric gradients \\( \nabla \lambda_k \\), the lowest-order
Nedelec basis (also called Whitney 1-forms) is one function per
**edge**:

\\[
\mathbf{N}_{(i, j)}(\mathbf{x}) = \lambda_i \nabla \lambda_j - \lambda_j \nabla \lambda_i
\\]

for edge \\( e = (i, j) \\). Properties:

- \\( \mathbf{N}_e \\) is **linear** in \\( \mathbf{x} \\) (since
  \\( \lambda_k \\) are linear and the gradients constant).
- The **tangential component** of \\( \mathbf{N}_e \\) along edge
  \\( (i, j) \\) is constant and integrates to 1 along that edge. Along
  every other edge it integrates to 0 — so the DoF value naturally
  represents the *tangential-line integral* of the field along the
  edge.
- The curl is **constant per element**:
  \\( \nabla \times \mathbf{N}_e = 2 \nabla \lambda_i \times \nabla \lambda_j \\).

## Edge orientation convention

Every unique mesh edge is stored canonically as `[min, max]` (see
[`compute_edges`](./mesh.md#edge-enumeration--compute_edges)). The
corresponding **global** Nedelec basis function is the one that
"points" from the lower-indexed vertex to the higher-indexed vertex.

When a cell's local edge \\( k \\) (in the `TET_EDGES` ordering) has
local endpoints \\( (a, b) \\) mapping to global vertices
\\( (v_a, v_b) \\), the local basis
\\( \mathbf{N}_{\mathrm{loc}} = \lambda_a \nabla \lambda_b - \lambda_b \nabla \lambda_a \\)
points from local \\( a \\) to local \\( b \\). To express it in the
global basis we apply an orientation sign:

\\[
\mathbf{N}_{\mathrm{glob}} = \mathrm{sign} \cdot \mathbf{N}_{\mathrm{loc}}, \quad
\mathrm{sign} = \begin{cases} +1 & v_a < v_b \\ -1 & v_a > v_b \end{cases}
\\]

Element matrices computed in the local-edge basis are converted to
the global-edge basis during assembly by multiplying entry
\\( (i, j) \\) by \\( \mathrm{sign}_i \cdot \mathrm{sign}_j \\) —
equivalently, absorbing the sign into the per-DoF scatter.

## `NedelecDofMap`

```rust
pub struct NedelecDofMap {
    pub edge_id: HashMap<[u32; 2], u32>,
    pub edge_count: usize,
}

impl NedelecDofMap {
    pub fn build(mesh: &FemMesh) -> Self;
    pub fn ndof(&self) -> usize;                       // edge_count
    pub fn edge_dof(&self, a: u32, b: u32) -> usize;   // [min, max] lookup
    pub fn edge_sign(cell: &FemCell, local_edge: usize) -> Real;
}
```

## Element matrices

### Mass — closed-form

```rust
pub fn compute_element_nedelec_mass(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec<Real>>;
```

The closed-form integral \\( \int_K \lambda_p \lambda_q \, dV = \frac{(1 + \delta_{pq})}{20} \cdot |K| \\)
combined with the constant gradients \\( \nabla \lambda_k \\) lets us
write the mass entries directly without a quadrature loop:

\\[
M_{e_1, e_2} = |K| \cdot \big[
c(j,l) \nabla \lambda_i \cdot \nabla \lambda_k
- c(j,k) \nabla \lambda_i \cdot \nabla \lambda_l
- c(i,l) \nabla \lambda_j \cdot \nabla \lambda_k
+ c(i,k) \nabla \lambda_j \cdot \nabla \lambda_l \big]
\\]

with \\( e_1 = (i, j), e_2 = (k, l) \\) and
\\( c(p, q) = (1 + \delta_{pq})/20 \\).

### Curl-curl stiffness

```rust
pub fn compute_element_curl_curl(cell, vertices, nu) -> Vec<Vec<Real>>;
```

Since the curl of a Whitney 1-form is **constant per element**:

\\[
\nabla \times \mathbf{N}_{(i, j)} = 2 \, \nabla \lambda_i \times \nabla \lambda_j,
\\]

the curl-curl integral collapses to

\\[
K_{e_1, e_2} = 4 \nu \cdot |K| \cdot (\nabla \lambda_i \times \nabla \lambda_j) \cdot (\nabla \lambda_k \times \nabla \lambda_l)
\\]

— one cross product per entry, no quadrature. `nu` is the **magnetic
reluctivity** \\( \nu = 1/\mu \\); for free space pass
\\( 1/\mu_0 \approx 7.958 \times 10^5 \\) in SI units, or just `1.0`
in non-dimensional units.

## Nedelec assembly

```rust
pub fn assemble_global_nedelec<F>(
    mesh: &FemMesh,
    map: &NedelecDofMap,
    element_visitor: F,
) -> Result<BsrMatrix>;
```

Walks every cell, calls the visitor for the local 6×6 matrix in
local-edge basis, then scatters entry \\( (i, j) \\) to global edge
DoFs `(edge_dof(...), edge_dof(...))` multiplied by
\\( \mathrm{sign}_i \cdot \mathrm{sign}_j \\). The `assemble_global`
entry point in [assembly](./assembly.md) auto-routes here when
`FunctionSpace::NedelecEdge { order }` is passed.

## Magnetostatics solver

```rust
pub fn solve_magnetostatics_steady(
    mesh: &FemMesh,
    nu_per_cell: &[Real],
    j_source: &[Vec3],
    bcs: &[BoundaryCondition],
    max_iters: usize,
    tol: Real,
) -> Result<MagnetostaticsResult>;

pub struct MagnetostaticsResult {
    pub a: Vec<Real>,     // vector potential, one DoF per edge
    pub residual: Real,
    pub iters: usize,
}
```

Solves the **steady-state curl–curl equation** for the magnetic vector
potential \\( \mathbf{A} \\):

\\[
\nabla \times (\nu \, \nabla \times \mathbf{A}) = \mathbf{J} \quad \text{in } \Omega,
\quad \mathbf{n} \times \mathbf{A} = 0 \quad \text{on } \partial \Omega,
\\]

The "perfect conductor" boundary condition
\\( \mathbf{n} \times \mathbf{A} = 0 \\) is the natural BC for the
vector potential — in Nedelec DoFs it's simply
\\( A_e = 0 \\) for every edge on the mesh boundary (handled via
strong Dirichlet).

### Gauge fixing

The continuous curl-curl operator has a non-trivial kernel: every
pure gradient \\( \mathbf{A} = \nabla \phi \\) is annihilated. On the
discrete level the kernel is exactly the discrete gradient of P1
nodal scalars (lowest-order de Rham complex).

For a magnetostatics problem with a divergence-free source
\\( \nabla \cdot \mathbf{J} = 0 \\) and fully-Dirichlet boundary on
\\( \partial \Omega \\), the gradient kernel is killed at the
boundary and what remains is gauge-equivalent. We **don't enforce
Coulomb gauge** \\( \nabla \cdot \mathbf{A} = 0 \\) explicitly — it's
not required to recover the physically meaningful
\\( \mathbf{B} = \nabla \times \mathbf{A} \\).

In practice a tiny mass regularisation \\( \varepsilon M \\) (with
\\( \varepsilon \ll \nu / L^2 \\)) is added to nudge CG towards a
unique solution; the bias on \\( \mathbf{B} \\) is \\( O(\varepsilon) \\)
and below the discretisation error.

## Helper utilities

```rust
pub fn compute_curl_per_cell(mesh, map, a) -> Vec<Vec3>;
pub fn project_field_on_edges<F>(mesh, map, field: F) -> Vec<Real>;
pub fn integrate_field_along_edges<F>(mesh, map, field: F) -> Vec<Real>;
```

`compute_curl_per_cell` recovers \\( \mathbf{B} = \nabla \times \mathbf{A} \\)
as one `Vec3` per cell — used by the
`Magnetostatics` viewer scene (#101) to render a B-field arrow at
every cell centroid.

`project_field_on_edges` projects a vector field given as a closure
\\( \mathbf{F}(\mathbf{x}) \\) onto the Nedelec DoFs (tangential-line
integrals along edges). Used for L²-projecting an analytic test
field.

`integrate_field_along_edges` integrates a vector field along each
mesh edge — useful for setting Dirichlet boundary values from a known
\\( \mathbf{A}_{\text{ref}} \\).

## Solenoid contract test

The Stage 46 test `magnetostatics_solenoid_axial_field` builds a
cylindrical current loop, solves the magnetostatics system, and
verifies that the recovered \\( \mathbf{B} \\) on the symmetry axis
matches the analytic solenoid formula
\\( B_z = \mu_0 \, I \, n \\) inside, decaying to zero outside, within
~5 % at the 16³ tet resolution.

## Cross-references

- [Eddy currents](./eddy.md) — the time-harmonic Maxwell extension
  uses the same Nedelec basis.
- [Mesh / function spaces](./mesh.md#function-spaces) — the
  `NedelecEdge { order: 1 }` variant.
- [BC](./bc.md) — strong Dirichlet works directly on edge DoFs (one
  per edge, no shared-DoF ambiguity), unlike the DG case.
- [Multigrid AMS](./multigrid.md) — the auxiliary-Maxwell solver
  preconditioner uses the discrete gradient \\( G \\) which lives in
  this same module's `build_discrete_gradient`.
