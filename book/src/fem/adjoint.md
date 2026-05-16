# Adjoint / differentiable FEM

`crates/molla-fem/src/adjoint.rs` implements reverse-mode differentiation
through the linear elasticity assembly + solve. Given a forward solve
\\( K(p) u = f(p) \\) with design parameters \\( p \\) (Young's modulus,
applied traction, ...) and a scalar loss \\( J(u, p) \\), the adjoint
method computes \\( \partial J / \partial p \\) with **one extra linear
solve** of the same matrix — independent of the dimensionality of
\\( p \\).

This is what gives molla a differentiable FEM, used by the Stage 38.2
fitting demo and any forward-design optimisation that wants to invert
"observed deformation" → "material property".

## The variational identity

Differentiate the equilibrium \\( K(p) u(p) = f(p) \\) with respect to
\\( p \\):

\\[
\frac{\partial K}{\partial p} u + K \frac{\partial u}{\partial p} = \frac{\partial f}{\partial p}.
\\]

So

\\[
\frac{\partial u}{\partial p} = K^{-1} \left( \frac{\partial f}{\partial p} - \frac{\partial K}{\partial p} u \right).
\\]

For a loss \\( J(u, p) \\):

\\[
\frac{\partial J}{\partial p} = \frac{\partial J}{\partial p}\bigg|_{\text{expl}}
+ \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial p}
\\]

Now define the adjoint variable \\( \lambda \\) by

\\[
K^T \lambda = \frac{\partial J}{\partial u}.
\\]

Then

\\[
\frac{\partial J}{\partial p} = \frac{\partial J}{\partial p}\bigg|_{\text{expl}}
+ \lambda^T \left( \frac{\partial f}{\partial p} - \frac{\partial K}{\partial p} u \right).
\\]

For **SPD** \\( K \\) (linear elasticity, Laplacian, ...), \\( K^T = K \\)
so the same matrix is used for the adjoint solve — no transpose, no
new factorisation.

## `solve_adjoint`

```rust
pub fn solve_adjoint(
    k: &BsrMatrix,
    dj_du: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
    dim: usize,
) -> Vec<Real>;

pub fn solve_adjoint_with(...) -> Vec<Real>;  // exposes tol / max_iters
```

Pass:

- `k` — the **un-reduced** stiffness matrix. The routine clones it,
  applies BCs internally (zero Dirichlet values + dropped
  NeumannTraction entries — see below), and runs `cg_solve` against
  the seed `∂J/∂u`.
- `dj_du` — \\( \partial J / \partial u \\), length `n_dof`.
- `bcs` — same BCs the forward solve used.
- `vertices` + `dim` — passed through to `apply_bcs` for any
  vector-valued constraints.

### BC handling

The adjoint inherits the same Dirichlet reduction as the forward solve
(zero out the row, set the diagonal to 1, RHS = 0 on constrained
rows). Two subtleties:

1. **`NeumannTraction` is dropped.** The adjoint RHS is `∂J/∂u`, not
   the physical load. Including `NeumannTraction` would double-count
   the load into the adjoint variable. The internal `pin_only`
   filter strips out `Neumann` and `NeumannTraction` entries before
   pinning rows.
2. **Dirichlet values are replaced with zero.** The forward solve
   pinned `u[r] = v` at Dirichlet rows; the adjoint must pin
   \\( \lambda[r] = 0 \\) (otherwise \\( \lambda \\) would carry
   spurious sensitivity at the prescribed DoFs). The Dirichlet
   *vertex* survives so the row pinning happens; only the *value* is
   forced to zero.

Defaults:

```rust
pub const DEFAULT_ADJOINT_TOL: Real = 1.0e-10;
pub const DEFAULT_ADJOINT_MAX_ITERS: usize = 50_000;
```

Tighter than the forward solve to keep `λ` clean — gradient noise
propagates if `λ` has stale CG iterates left in it.

## Gradient w.r.t. Young's modulus

```rust
pub fn compute_grad_youngs(
    mesh: &FemMesh,
    vertices: &[Vec3],
    u: &[Real],
    lambda: &[Real],
    poisson: Real,
    youngs: Real,
) -> Real;
```

The linear-elasticity stiffness matrix is **linear in \\( E \\)**:
both Lamé parameters

\\[
\lambda_l = \frac{E \nu}{(1+\nu)(1-2\nu)}, \quad
\mu = \frac{E}{2(1+\nu)}
\\]

are proportional to \\( E \\), so

\\[
\frac{\partial K}{\partial E} = \frac{K}{E}.
\\]

The variational identity collapses to

\\[
\frac{\partial J}{\partial E} = -\frac{1}{E} \, \lambda^T K u.
\\]

The implementation computes \\( \lambda^T K u \\) **element-by-element**
— assembles each local stiffness, gathers local \\( u \\) and
\\( \lambda \\) in interleaved layout, computes the local
\\( \lambda^T K_e u \\), sums. \\( O(n_{\text{cells}}) \\) total, no
global matvec, no global sparse matrix touched.

The routine assumes:

- \\( \dim = 2 \\) (tri plane-strain) or \\( \dim = 3 \\) (tet).
- The loss \\( J \\) has no explicit \\( E \\) dependence — compliance,
  tip deflection, target-displacement L2 norms all read `u` only.

## Gradient w.r.t. surface traction

```rust
pub fn compute_grad_load(
    vertices: &[Vec3],
    lambda: &[Real],
    traction_face: [u32; 3],
    dim: usize,
) -> [Real; 3];
```

For a uniform traction \\( t \\) on a triangular face, the consistent
P1 lumping puts \\( (area / 3) \cdot t_c \\) into the RHS at each
vertex's component DoF. So

\\[
\frac{\partial f}{\partial t_c} = \frac{area}{3} \, e_c \text{ (summed over face vertices)},
\\]

and the adjoint identity gives

\\[
\frac{\partial J}{\partial t_c} = \frac{area}{3} \sum_{v \in face} \lambda_{v \cdot \dim + c}.
\\]

Triangle face area is computed in 3D via
\\( \tfrac{1}{2} |\, (\mathbf{b} - \mathbf{a}) \times (\mathbf{c} - \mathbf{a}) \,| \\).
Components beyond `dim` (e.g. asking for `[Real; 3]` from a 2D
plane-strain run) come back as 0.

## Stage 38.2 — Young's modulus fitting in log space

The first end-to-end consumer of the adjoint is the Stage 38.2 demo
that fits Young's modulus \\( E \\) from observed deformation. The
recipe:

1. Start with an initial guess \\( E_0 \\).
2. Forward solve: \\( K(E) u = f \\).
3. Evaluate loss \\( J(u) = \tfrac{1}{2} \|u - u_{\text{obs}}\|^2 \\).
4. Adjoint solve: \\( K \lambda = u - u_{\text{obs}} \\).
5. Gradient: \\( \partial J / \partial E = -\lambda^T K u / E \\) via
   `compute_grad_youngs`.
6. Adam update in **log space**: \\( \log E \leftarrow \log E - \alpha \cdot g \cdot E \\)
   (chain rule from \\( \partial J / \partial \log E = E \, \partial J / \partial E \\)).

Log-space optimisation is the right answer because \\( E \\) ranges
across many orders of magnitude (rubber: 1e6 Pa; steel: 1e11 Pa);
linear-space Adam either overshoots soft regions or undershoots stiff
ones.

The demo converges in ~50 Adam steps and matches finite-difference
gradients to ~1e-3 relative.

## Caveats

- **Linear elasticity only.** SVK / plasticity adjoints (Stage 36.4.2)
  need to differentiate through the Newton iteration; deferred.
- **No Stokes / Navier-Stokes adjoint.** Stage 36.4.1 — pressure-
  velocity coupling makes the bookkeeping fiddly. Deferred.
- **CPU only.** The GPU port would mirror the forward GPU CG with the
  adjoint RHS as the seed.

## Cross-references

- [Solvers](./solvers.md) — the forward solve the adjoint mirrors.
- [BC](./bc.md) — the same reducer used; the adjoint drops
  `NeumannTraction` entries and zeroes Dirichlet values.
- [Element kernels](./elements.md#linear-isotropic-elasticity) — the
  per-cell elasticity stiffness that `compute_grad_youngs`
  re-assembles per element.
- [Autograd](../autograd/overview.md) (when documented) — the upper
  layer that wraps adjoint solves into a tape.
