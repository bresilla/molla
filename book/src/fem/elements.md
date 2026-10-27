# Element kernels

The element kernels in `crates/molla-fem/src/element.rs` are the
per-cell building blocks of every assembly: they take a cell + the
vertex array (+ whatever material parameters the operator needs) and
return a small dense matrix in row-major
`Vec<Vec<Real>>` form. The [assembly walker](./assembly.md) then
scatters those matrices into a global `BsrMatrix`.

All kernels follow the same shape — `compute_element_*(cell, vertices, …)
-> Vec<Vec<Real>>` — and they all use the same two geometric helpers:

```rust
pub fn tri_area_2d(p: &[Vec3; 3]) -> Real;
pub fn tet_volume(p: &[Vec3; 4]) -> Real;          // absolute
pub fn tet_signed_volume(p: &[Vec3; 4]) -> Real;   // signed
```

plus two shape-gradient helpers used by every gradient-based operator:

```rust
pub(super) fn tri_shape_grads_2d(p: &[Vec3; 3]) -> [[Real; 2]; 3];
pub(super) fn tet_shape_grads(p: &[Vec3; 4]) -> [Vec3; 4];
```

The triangle gradient routine returns `[gx, gy]` pairs only — the
out-of-plane component is zero by construction for a planar
\\(z = 0\\) triangle. The tet routine returns full 3D vectors built
from the inverse-transpose of `D = [v₁−v₀ | v₂−v₀ | v₃−v₀]`, with the
standard \\( \nabla \phi_0 = -\sum_{i \ge 1} \nabla \phi_i \\)
identity that the `partition_of_zero` test verifies.

## Mass

```rust
pub fn compute_element_mass(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec<Real>>;
```

The consistent P1 mass matrix is closed-form in the cell measure:

\\[
M_{ij}^{\text{tri}} = \frac{|K|}{12} \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix},
\quad
M_{ij}^{\text{tet}} = \frac{|K|}{20} \begin{pmatrix} 2 & 1 & 1 & 1 \\ 1 & 2 & 1 & 1 \\ 1 & 1 & 2 & 1 \\ 1 & 1 & 1 & 2 \end{pmatrix}
\\]

The reference matrices `TRI_MASS_REF` and `TET_MASS_REF` are stored as
`const` arrays; the kernel just multiplies by `area / 12` or
`vol / 20`. Row sums equal the lumped-mass row sums and the total sum
is the cell measure — the
`mass_matrix_row_sum_equals_lumped_mass` test in
[assembly](./assembly.md) checks this on a 4×4 triangle grid.

## Laplacian stiffness

```rust
pub fn compute_element_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    coeff: Real,
) -> Vec<Vec<Real>>;
```

For P1 the shape-function gradients are piecewise constant, so

\\[
K_{ij} = \alpha \int_K \nabla \phi_i \cdot \nabla \phi_j \, dx
       = \alpha \cdot |K| \cdot \nabla \phi_i \cdot \nabla \phi_j
\\]

— one cross product (or 2D dot product) per `(i, j)` entry. No
quadrature loop, no allocator churn beyond the output matrix. The
identity \\( \sum_j \nabla \phi_j = 0 \\) implies
\\( K \cdot \mathbf{1} = 0 \\) per cell, captured by
`unit_tet_stiffness_kills_constant_field`.

The wave-equation operator
\\( c^2 \nabla^2 \\) reuses the same code:

```rust
pub fn compute_element_wave_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    c2: Real,
) -> Vec<Vec<Real>>;
```

is literally `compute_element_stiffness(cell, vertices, c2)` —
re-badged so callers reading the assembly visitor list can tell which
PDE is being assembled.

## Linear isotropic elasticity

```rust
pub fn compute_element_elasticity_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    lambda: Real,
    mu: Real,
) -> Vec<Vec<Real>>;
```

The full \\( \mathbf{B}^T \mathbf{D} \mathbf{B} \\) operator:

\\[
\mathbf{K}_e = \int_K \mathbf{B}^T \mathbf{D} \mathbf{B} \, dV
\\]

with the strain–displacement matrix `B` in Voigt notation and the
elasticity tensor

\\[
\mathbf{D}^{\text{3D}} =
\begin{pmatrix}
\lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
\lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{pmatrix}
\\]

For a tetrahedron \\( \mathbf{B} \\) is 6×12, so
\\( \mathbf{K}_e \\) is 12×12 — three components per vertex,
interleaved as `[ux₀, uy₀, uz₀, ux₁, ...]`. The implementation builds
`B` once, computes `D · B` (6×12), then performs the 12×12
contraction `Bᵀ · (D · B)` and scales by `vol`. Triangle elements
follow the same recipe in plane strain: \\( \mathbf{B} \\) is 3×6,
\\( \mathbf{D} \\) is the in-plane \\( (\lambda + 2\mu, \lambda, \mu) \\)
3×3 block, and the output is 6×6.

Lamé parameters in terms of Young's modulus and Poisson's ratio:

\\[
\lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)},
\quad
\mu = \frac{E}{2(1 + \nu)}.
\\]

## St Venant–Kirchhoff hyperelastic stress

```rust
pub fn compute_element_st_venant_kirchhoff_stress(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    lambda: Real,
    mu: Real,
) -> Vec<Real>;
```

The SVK model is the geometrically-nonlinear extension of linear
elasticity. Given the deformation gradient
\\( F = \partial x / \partial X = I + \sum_i u_i \otimes \nabla \phi_i \\),
the Green strain is

\\[
E = \tfrac{1}{2}(F^T F - I),
\\]

the second Piola–Kirchhoff stress is

\\[
S = \lambda \, \mathrm{tr}(E) \, I + 2\mu E,
\\]

and the first Piola–Kirchhoff stress is \\( P = F S \\). The nodal
force on local vertex \\( a \\) is

\\[
\mathbf{f}_a = |K| \cdot P \cdot \nabla \phi_a.
\\]

The implementation uses **signed** volume (`tet_signed_volume`) so a
flipped rest tet produces the correct internal-force direction — the
volume's sign and the shape-gradient signs cancel.

Triangle plane-strain SVK is *not* implemented (the 2D contract tests
all use linear elasticity); calling it panics with a clear message
that points the user to the tet path. This is a deliberate "fail
loud, don't fail silent with wrong-sized output" choice.

## Neo-Hookean and Mooney-Rivlin tet tangents

```rust
pub fn compute_element_neo_hookean_energy(... ) -> Real;
pub fn compute_element_neo_hookean_force(... ) -> Vec<Real>;
pub fn compute_element_neo_hookean_tangent(... ) -> Vec<Vec<Real>>;

pub fn compute_element_mooney_rivlin_energy(... ) -> Real;
pub fn compute_element_mooney_rivlin_force(... ) -> Vec<Real>;
pub fn compute_element_mooney_rivlin_tangent(... ) -> Vec<Vec<Real>>;
```

The hyperelastic tet path now includes compressible Neo-Hookean and
Mooney-Rivlin energies. Internal forces are central finite-difference
gradients of the closed-form element energy, and tangents are central
finite differences of those forces. This keeps the first implementation
auditable and gives Newton solves a consistent element tangent while a
future optimisation pass can swap in analytic fourth-order material
tangents.

The Neo-Hookean density is stress-free at identity and has the same
small-strain tangent as the linear isotropic element. The Mooney-Rivlin
variant uses two invariants plus a volumetric penalty and is likewise
stress-free at identity. Both paths currently support P1 tetrahedra.

## Convection + SUPG stabilisation

```rust
pub fn compute_element_convection(
    cell: &FemCell,
    vertices: &[Vec3],
    b: Vec3,
) -> Vec<Vec<Real>>;

pub fn compute_element_supg_convection(
    cell: &FemCell,
    vertices: &[Vec3],
    b: Vec3,
    alpha: Real,
    h: Real,
) -> Vec<Vec<Real>>;
```

The Galerkin convection block
\\( C_{ij} = \int_K N_i \, (\mathbf{b} \cdot \nabla N_j) \, dx \\)
with P1 elements has a rank-one structure: \\( \mathbf{b} \cdot \nabla N_j \\)
is a single number per column \\( j \\), and
\\( \int_K N_i \, dx = |K| / (n+1) \\) for every row — so every row of
\\( C \\) is the same column vector.

The operator is **non-symmetric** and goes oscillatory once the cell
Péclet number \\( |\mathbf{b}| h / (2\alpha) \\) exceeds 1. Streamline-
Upwind Petrov–Galerkin (SUPG) restores monotonicity by adding a
symmetric stabilisation block

\\[
S_{ij}^{\text{SUPG}} = \tau \int_K (\mathbf{b} \cdot \nabla N_i)(\mathbf{b} \cdot \nabla N_j) \, dx
\\]

with Hughes' textbook
\\( \tau = h / (2|\mathbf{b}|) \\). The kernel multiplies by an
optional `alpha` factor and returns a rank-one symmetric block of
\\( \tau |K| (\mathbf{b} \cdot \nabla N) (\mathbf{b} \cdot \nabla N)^T \\).
Pair `compute_element_convection` with `compute_element_supg_convection`
to recover a textbook Petrov–Galerkin convection–diffusion operator.

## Symmetric-Dirichlet distortion energy

```rust
pub fn compute_element_distortion_energy(
    cell: &FemCell,
    rest_vertices: &[Vec3],
    deformed_vertices: &[Vec3],
) -> (Real, Vec<Vec3>, Vec<Vec<Real>>);
```

Returns `(energy, gradient, hessian)`. The energy is

\\[
E = \tfrac{1}{2} \, \mathrm{tr}(F^T F + (F^T F)^{-1}) - \dim
\\]

where \\( F = D_{\text{def}} \cdot D_{\text{rest}}^{-1} \\). It is
zero on the identity map, grows monotonically away from it, and has a
barrier at every degenerate (zero-volume) configuration — the
canonical "distortion-aware" objective for parameterisation /
inverse-deformation tasks.

The v1 implementation uses **central finite differences** on the
analytic energy for both gradient and Hessian. This keeps the kernel
short and easy to audit; finite differences are essentially exact to
roundoff thanks to the smooth analytic energy. A follow-up stage can
swap in the analytic \\( \partial \Psi / \partial F \\) /
\\( \partial^2 \Psi / \partial F^2 \\) once the
\\( dF/dx \\) chain rule is wired through.

The Hessian is collected as a `(N · 3) × (N · 3)` dense matrix in
interleaved layout — `9×9` for a triangle, `12×12` for a tet. The
[`solve_distortion_minimization`](./solvers.md#newton-on-distortion-energy)
driver runs Newton-Raphson against this block.

## Visitor signature contract

Every kernel returns a row-major `Vec<Vec<Real>>` with row count and
column count both equal to `cell.vert_count() · dofs_per_vertex`:

- Scalar P1 mass / stiffness: 3×3 (Tri) or 4×4 (Tet).
- Elasticity 3D: 12×12.
- Plane-strain elasticity: 6×6.
- Convection / SUPG: same as scalar P1.
- Distortion Hessian: 9×9 / 12×12.

The choice of `Vec<Vec<…>>` over a packed `[Real; 144]` is deliberate:
the v1 visitor API needs to stay readable and copy-safe, and the
allocator overhead is dwarfed by the matvec cost downstream. The GPU
elasticity port in [`elasticity_gpu`](./elasticity_gpu.md) drops to a
packed `[f32; 144]` payload — but only inside the WGSL shader.

## Cross-references

- The [quadrature](./quadrature.md) chapter lists the Gauss rules
  these kernels invoke.
- The [assembly](./assembly.md) chapter shows how the local matrices
  get scattered.
- The [solvers](./solvers.md) chapter is where these kernels meet
  CG / PCG / Newton-Raphson.
- The [P2](./p2.md) and [DG](./dg.md) chapters add their own higher-
  order kernels that share the same `Vec<Vec<Real>>` shape.
