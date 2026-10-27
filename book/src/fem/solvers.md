# Top-level solvers

`crates/molla-fem/src/solver.rs` strings the pieces together. Every
top-level solver in this chapter follows the same pattern:

1. Clone the assembled matrix.
2. Call `apply_bcs` / `apply_dirichlet` to produce a BC-reduced
   `(K_red, f_red)` pair.
3. Seed the initial guess with Dirichlet values so CG starts with zero
   residual on the pinned rows.
4. Hand `|x| matvec(&K_red, x)` to `cg_solve` (or `pcg_solve`).
5. Return the result.

Tolerances default to:

```rust
pub const DEFAULT_TOL: Real = 1.0e-8;
pub const DEFAULT_MAX_ITERS: usize = 2000;
```

— sufficient for the v1 contract tests at 16³ resolution.

## Steady-state Laplace / Poisson

```rust
pub fn solve_steady_state(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    max_iters: usize,
    tol: Real,
) -> Vec<Real>;

pub fn solve_steady_state_with_initial(...) -> CgResult;
pub fn solve_steady_state_pcg(...) -> CgResult;
```

Solves \\( K u = f \\) with Dirichlet BCs reduced symmetrically. The
PCG variant builds a `JacobiPreconditioner` from `K_red.diagonal()` —
typically cuts CG iters by 25–40 % on Laplacian-like SPD systems for
free.

There's also a vector-aware sibling that knows about elasticity:

```rust
pub fn solve_steady_state_vector(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
    dim: usize,
    max_iters: usize,
    tol: Real,
) -> Vec<Real>;
```

— same recipe but routes through `apply_bcs` so
`DirichletVector` / `NeumannTraction` are honoured. `dim` is the
per-vertex DoF count (3 for tets, 2 for plane-strain tris).

## Transient parabolic — backward Euler

```rust
pub fn solve_transient(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
) -> Vec<Real>;
```

The backward-Euler step solves

\\[
\left(\frac{M}{\Delta t} + K \right) u^{n+1} = f + \frac{M}{\Delta t} u^n.
\\]

The LHS is materialised via the private `add_csr_scaled` helper that
walks both triplet lists and reuses
`BsrMatrix::from_triplets` to dedupe; the RHS is built with a single
matvec of `M` against `u_old`. Dirichlet BCs are then applied in
tandem on `(A, b)` before CG.

`solve_transient_pcg` is the Jacobi-preconditioned variant. For repeated
steps with the same left-hand side, `solve_transient_pcg_restarted`
adds a recycled-Krylov initial guess:

```rust
pub struct KrylovRestartCache { /* bounded recent correction vectors */ }

pub fn solve_transient_pcg_restarted(
    m, k, f, u_old, dt, bcs, x0, cache, max_iters, tol,
) -> CgResult;
```

The cache stores recent PCG correction directions. Before each solve,
the restarted path computes a tiny minimum-residual projection in
`x0 + span(cache)` and uses that projected vector as the PCG initial
guess, then stores the new correction for the next step. This is the
CPU oracle for Krylov restart/recycling in transient FEM steppers.

## Wave equation — Newmark-β

```rust
pub fn solve_wave_transient(
    m: &BsrMatrix,
    k: &BsrMatrix,
    u_old: &[Real],
    v_old: &[Real],
    a_old: &[Real],
    dt: Real,
    beta: Real,
    gamma: Real,
    bcs: &[BoundaryCondition],
) -> WaveStepResult;
```

The classical Newmark-β implicit step for \\( M\ddot u + K u = 0 \\):

1. **Predictor:** \\( u_{\text{pred}} = u_{\text{old}} + \Delta t v_{\text{old}}
   + (\frac{1}{2} - \beta) \Delta t^2 a_{\text{old}} \\).
2. **Solve:** \\( \left(\frac{M}{\beta \Delta t^2} + K\right) u_{\text{new}}
   = \frac{M}{\beta \Delta t^2} u_{\text{pred}} \\).
3. **Corrector:**
   \\( a_{\text{new}} = (u_{\text{new}} - u_{\text{pred}}) / (\beta \Delta t^2) \\),
   \\( v_{\text{new}} = v_{\text{old}} + \Delta t [(1-\gamma) a_{\text{old}} + \gamma a_{\text{new}}] \\).

The defaults `β = 1/4, γ = 1/2` give the "constant-average-acceleration"
variant, which is unconditionally stable for linear undamped wave
problems. The constants are exposed as:

```rust
pub const NEWMARK_BETA_DEFAULT: Real = 0.25;
pub const NEWMARK_GAMMA_DEFAULT: Real = 0.5;
```

After the corrector, `v_new` and `a_new` are zeroed on every Dirichlet
DoF. Without that, a non-zero `v_old` or `a_old` on a pinned DoF
(e.g. seeded from a consistent initial-acceleration solve) gets
carried into `a_new` through `u_pred` and integrates to spurious
boundary motion across steps. For time-varying Dirichlet data the
caller is expected to overwrite the pinned `v_new` / `a_new` entries
after each call.

Returns a `WaveStepResult { u_new, v_new, a_new, residual, iters }`.

## Nonlinear static — Newton-Raphson

```rust
pub fn solve_nonlinear_static<KAsm, RAsm>(
    tangent_assembler: KAsm,
    residual_assembler: RAsm,
    f_ext: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
    dim: usize,
    max_newton_iters: usize,
    tol: Real,
) -> Vec<Real>
where
    KAsm: Fn(&[Real]) -> BsrMatrix,
    RAsm: Fn(&[Real]) -> Vec<Real>;
```

Solves \\( R(u) = K_{\text{int}}(u) - f_{\text{ext}} = 0 \\) by
classical Newton iteration:

1. Assemble the tangent `K(u)` via `tangent_assembler(u)`.
2. Compute the residual via `residual_assembler(u)`.
3. Build `rhs = f_ext - K_int(u)`.
4. Apply Dirichlet BCs with zero values (the absolute Dirichlet value
   is already baked into the seeded `u`; the Newton increment must
   satisfy `du = 0` on constrained rows).
5. Solve \\( K \, du = -R \\) via `cg_solve`.
6. Update \\( u \leftarrow u + du \\).
7. Terminate on \\( \|du\|_\infty < \text{tol} \\) or when the
   residual norm grows for two consecutive iterations (divergence
   bail-out — emits a `tracing::warn!`).

v1 has no line search; small problems only. The hyperelasticity
contract tests in `crates/molla-fem/tests/` use this driver against the
SVK kernel from [elements](./elements.md#st-venant-kirchhoff-hyperelastic-stress).

## Newton on distortion energy

```rust
pub fn solve_distortion_minimization(
    rest_mesh: &FemMesh,
    init_deformed: &[Vec3],
    max_iter: usize,
    tol: Real,
) -> Vec<Vec3>;
```

Newton-Raphson minimisation of the per-cell symmetric-Dirichlet
distortion energy (see
[`compute_element_distortion_energy`](./elements.md#symmetric-dirichlet-distortion-energy)).
Each iteration assembles the global gradient \\( g \\) and Hessian
\\( H \\), regularises the diagonal with \\( 10^{-8} I \\) to keep
\\( H \\) SPD on degenerate elements, then solves \\( H \, dx = -g \\)
via CG. Termination: \\( \|dx\|_\infty < \text{tol} \\).

This is the driver behind the `WarpFemDistortion` viewer scene and the
canonical "snap perturbed positions back to a low-distortion map" use
case.

## GPU steppers — Slice H

The GPU port keeps the *state* resident on the device across timesteps
so per-step host↔device traffic is zero. Two stepper objects expose
this:

### `WaveStepperGpu`

```rust
let mut stepper = WaveStepperGpu::new(
    engine, mesh, &m, &k,
    dt, beta, gamma, &bcs, &u, &v, &a)?;
for _ in 0..n_steps {
    let stats = stepper.step()?;       // GPU dispatches only
}
let (u, v, a) = stepper.read_state()?;
```

Internals:

- BC-reduced \\( A = \frac{M}{\beta \Delta t^2} + K \\) is uploaded once
  (`FemGpuData`); the per-step BC column-shift `bc_shift` is also
  pre-computed (so the GPU never re-runs `apply_bcs`).
- Persistent storage buffers hold `(u, v, a)` in f32.
- Scratch buffers (`u_pred`, `m_upred`, `b`) are allocated once.
- A Jacobi preconditioner `inv_diag = 1 / diag(A_red)` is extracted at
  construction; the per-step CG runs as **PCG** (~30 % iter
  reduction).
- Each `step()` dispatches: predictor → `M · u_pred` → RHS build →
  PCG solve → corrector → copy `u_new` into the resident `u` buffer.

### `TransientStepperGpu`

Same pattern for the backward-Euler transient step. Persistent
buffers: `u`, `f` (source term, defaults to zero — `write_f` to update),
and scratch (`m_uold`, `b`). The per-step source is
`write_f`-controlled so a time-varying forcing works without
rebuilding the stepper.

Both steppers expose `read_u()`, `read_state()`, `write_state()` /
`write_u()` for the viewer reset path. The contract is "render thread
reads back once per frame, simulation thread steps many times per
frame without ever touching the host".

## Russell direct solver

```rust
#[cfg(feature = "russell")]
pub fn solve_steady_state_direct(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
) -> Result<Vec<Real>>;
```

Routes through a **pure-Rust dense LU with partial pivoting**
(`DirectSolver::factorize` + `back_solve`). The upstream
`russell_sparse` crate was deemed unworkable in molla's nix devShell
(see the rationale in `russell.rs`), so this stage ships a fallback
that exposes the same `new` / `factorize` / `back_solve` / `solve` API
shape.

Use this when:

- CG stalls (indefinite or stiff systems).
- The problem is very small and one LU is cheaper than many CG iters.
- You're doing "factor once, back-solve many times" (e.g. adjoint
  sweeps over multiple loads).

A hard cap `MAX_DIRECT_DIM = 12_000` keeps the dense factor below
~1 GB; above that, callers should keep iterating with `cg_solve`.

## Cross-references

- [Assembly](./assembly.md) — builds the input matrices.
- [BC](./bc.md) — the reducer every solver invokes.
- [Adjoint](./adjoint.md) — the gradient solve reuses
  `solve_adjoint_with` which mirrors this driver shape.
- [Stokes](./stokes_ns.md) — Uzawa-CG uses the same `cg_solve` core
  on the Schur complement.
- [Signorini](./signorini.md) — projected Uzawa nests inner-CG calls
  inside the dual update loop.
- [Multigrid](./multigrid.md) — the V-cycle wraps a coarsest-level
  Cholesky solve plus per-level ω-Jacobi smoothing.
