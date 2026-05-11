# Eddy currents — time-harmonic Maxwell

Stage 48 added a **time-harmonic eddy-current** solver on top of the
H(curl) Nedelec basis from [magnetostatics](./magneto.md). It lives
in `crates/molla/src/fem/eddy_current.rs` (~1450 lines, including the
AMS preconditioner) and solves the complex curl–curl–mass system that
arises from the Fourier transform of Maxwell's equations.

## Problem statement

Solve the time-harmonic vector-potential equation

\\[
\nabla \times (\nu \, \nabla \times \mathbf{A}) + i \omega \sigma \mathbf{A} - \omega^2 \varepsilon \mathbf{A} = \mathbf{J} \quad \text{in } \Omega
\\]

with \\( \mathbf{n} \times \mathbf{A} = 0 \\) on \\( \partial \Omega \\),
where:

- \\( \mathbf{A} = \mathbf{A}_{\text{re}} + i \mathbf{A}_{\text{im}} \\) is the
  complex magnetic vector potential (one tangential DoF per mesh edge
  in each of the real and imaginary parts).
- \\( \nu = 1/\mu \\) is the magnetic reluctivity (real, per cell).
- \\( \sigma \\) is the electric conductivity (real, per cell — only the
  conductor cells have non-zero values).
- \\( \omega = 2\pi f \\) is the angular frequency (a single global scalar).
- \\( \mathbf{J} = \mathbf{J}_{\text{re}} + i \mathbf{J}_{\text{im}} \\) is the
  source current density (one complex `Vec3` per cell).
- \\( \varepsilon \\) is electric permittivity (typically negligible at
  the frequencies of interest; included for completeness).

The off-diagonal \\( i \omega \sigma \mathbf{A} \\) term couples real
and imaginary parts — this is the **skin-effect** mechanism that
causes high-frequency fields to decay exponentially into a conductor.

## Block-real reformulation

The complex unknown is split into
\\( (\mathbf{a}_{\text{re}}, \mathbf{a}_{\text{im}}) \in \mathbb{R}^{2N} \\)
(where \\( N \\) is the number of edges). Writing the curl–curl
block as \\( K_{\text{curl}} \\) and the Nedelec mass scaled by
conductivity as \\( M_\sigma \\), the discrete weak form becomes

\\[
\begin{cases}
K_{\text{curl}} \mathbf{a}_{\text{re}} - \omega M_\sigma \mathbf{a}_{\text{im}} = \mathbf{f}_{\text{re}} \\
\omega M_\sigma \mathbf{a}_{\text{re}} + K_{\text{curl}} \mathbf{a}_{\text{im}} = \mathbf{f}_{\text{im}}
\end{cases}
\\]

i.e. in block form:

\\[
\begin{pmatrix} K_{\text{main}} & K_{\text{off}} \\ -K_{\text{off}} & K_{\text{main}} \end{pmatrix}
\begin{pmatrix} \mathbf{a}_{\text{re}} \\ \mathbf{a}_{\text{im}} \end{pmatrix} =
\begin{pmatrix} \mathbf{f}_{\text{re}} \\ \mathbf{f}_{\text{im}} \end{pmatrix}
\\]

with \\( K_{\text{main}} = K_{\text{curl}} + \varepsilon M_N \\) (tiny mass
regularisation for the gradient gauge — same trick as
magnetostatics) and \\( K_{\text{off}} = -\omega M_\sigma \\).

The full \\( 2N \times 2N \\) system is **non-symmetric** when
\\( M_\sigma \ne 0 \\), so we use restartable **GMRES** instead of CG.

## Public API

```rust
pub struct EddyCurrentConfig {
    pub omega: Real,
    pub epsilon_regulariser: Real,
    pub max_iters: usize,
    pub tol: Real,
    pub gmres_restart: usize,
    pub use_ams: bool,
}

pub struct EddyCurrentResult {
    pub a_re: Vec<Real>,
    pub a_im: Vec<Real>,
    pub residual: Real,
    pub iters: usize,
}

pub fn assemble_eddy_system(
    mesh: &FemMesh,
    map: &NedelecDofMap,
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    omega: Real,
    epsilon: Real,
) -> Result<(BsrMatrix, BsrMatrix)>;       // (K_main, K_off)

pub fn solve_eddy_current_steady(
    mesh, nu, sigma, j_re, j_im, bcs, config
) -> Result<EddyCurrentResult>;

pub fn solve_eddy_current_steady_ams(...) -> Result<EddyCurrentResult>;
pub fn solve_eddy_current_steady_ams_mg(...) -> Result<EddyCurrentResult>;
```

## Solvers — three paths

### `solve_eddy_current_steady` (v1) — Jacobi-preconditioned GMRES

The simplest path: build the block-real `2N × 2N` system, extract its
diagonal as the Jacobi preconditioner, and run **restartable GMRES**.
This works on any system and is the right starting point for
correctness validation. For the v1 skin-depth contract test, it
converges in ~50 GMRES iterations.

### `solve_eddy_current_steady_ams` (v2) — Auxiliary-Maxwell Solver

For larger systems the Jacobi preconditioner stalls because the
curl-null space (gradients of P1 scalars) is left unaddressed. The
**Auxiliary Maxwell Solver** (Hiptmair–Xu 2007) decomposes the
preconditioning into:

1. A "scalar auxiliary" Poisson solve in the H¹ space (handled by
   the [multigrid hierarchy](./multigrid.md)) — corrects the gradient
   component.
2. A "vector auxiliary" Poisson solve in the P1 vector space —
   corrects the divergence-free component.
3. A Jacobi smoother on the edge DoFs themselves.

The combination is provably **mesh-independent** under the discrete
Helmholtz decomposition — iteration count stays bounded as `h → 0`.

`build_discrete_gradient` and `build_p1_vector_laplacian` are the
ingredients:

```rust
pub fn build_discrete_gradient(mesh: &FemMesh) -> Result<BsrMatrix>;
pub fn build_p1_vector_laplacian(mesh: &FemMesh, dim: usize) -> Result<BsrMatrix>;
```

The first returns the discrete gradient \\( G : \mathbb{R}^{n_{\text{verts}}} \to \mathbb{R}^{n_{\text{edges}}} \\)
that maps a nodal scalar to its tangential edge-integral (entries
\\( \pm 1 \\) only — orientation-dependent). The second assembles the
\\( \mu \int \nabla u : \nabla v \\) Laplacian on P1 vector space.

### `solve_eddy_current_steady_ams_mg` (v3) — AMS + Multigrid V-cycle

The v3 path replaces the auxiliary-block scalar solver with a
**geometric multigrid V-cycle** on a uniformly refined mesh
hierarchy. See [multigrid](./multigrid.md) for the V-cycle structure.
Empirically this cuts the iteration count on the larger eddy
fixtures (32³ tets) by 4–5× over the v2 fixed-iter CG.

## `AmsPreconditioner`

```rust
pub struct AmsPreconditioner {
    g: BsrMatrix,                 // discrete gradient
    k_aux_scalar: BsrMatrix,      // H¹ Laplacian
    k_aux_vector: BsrMatrix,      // P1 vector Laplacian
    diag_smoother: Vec<Real>,     // Jacobi smoother diagonal
    aux_solver: AuxSolver,        // CG or V-cycle
    mg_hierarchy: Option<MeshHierarchy>,
    mg_cycles_per_apply: usize,
}

impl AmsPreconditioner {
    pub fn build(mesh, map, k_main, n_aux_iters, ...) -> Result<Self>;
    pub fn apply(&self, r: &[Real]) -> Vec<Real>;
}
```

`apply(r)` computes \\( M^{-1} r \\) where \\( M^{-1} \\) is the AMS
preconditioner action — three sub-steps fused into one call. The
`mg_hierarchy` slot is `None` for the v2 path and `Some(...)` for v3.

## `EddyBlockOp` — block-real GMRES operator

```rust
pub struct EddyBlockOp<'a> {
    k_main: &'a BsrMatrix,
    k_off: &'a BsrMatrix,
    n: usize,
}
```

implements the `LinearOperator` trait expected by `gmres` — exposes
`mv(x, y)` over the block layout
`x = [a_re ; a_im]`, `y = [...]`. This is the way to avoid actually
assembling the full \\( 2N \times 2N \\) matrix (which would double
the memory cost) — instead we just dispatch two scalar matvecs per
block-matvec.

## Skin-depth contract test

The Stage 48 test `eddy_current_skin_depth_decay` solves the
eddy-current problem in a conducting half-space with a tangential
boundary current at the free surface. The analytic solution decays
exponentially with depth at the **skin depth**

\\[
\delta = \sqrt{\frac{2}{\omega \mu \sigma}}.
\\]

At the test frequency (\\( f = 60 \\) Hz, copper-like
\\( \sigma = 5.96 \times 10^7 \\) S/m, \\( \mu = \mu_0 \\)),
\\( \delta \approx 8.5 \\) mm. The solver matches the analytic decay
profile to ~3 % at 16³ tet resolution and within a few percent of the
phase shift between real and imaginary parts.

## `gmres_with_precond`

```rust
pub fn gmres_with_precond<P>(
    op: impl LinearOperator,
    pre: P,
    b: &[Real],
    x0: &[Real],
    config: IterConfig,
) -> CgResult
where
    P: Fn(&[Real]) -> Vec<Real>;
```

A generic restartable GMRES driver that takes any `LinearOperator` +
any preconditioner closure. Used internally by the eddy-current
solver but exposed publicly so callers with their own non-symmetric
systems can reuse it.

## Cross-references

- [Magnetostatics](./magneto.md) — the curl-curl machinery that this
  builds on (steady, real case).
- [Nedelec basis](./magneto.md#the-first-order-nedelec-basis) — the
  edge-element basis.
- [Multigrid](./multigrid.md) — the V-cycle inside the AMS v3 path.
- [Iterative solvers](../math/iterative.md) (when documented) — the
  `gmres` driver under the hood.
