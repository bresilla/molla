# Iterative solvers

`crates/molla/src/math/iterative.rs` ships **closure-driven**
Krylov-subspace solvers — CG, Preconditioned CG, and BiCGSTAB. They
complement the trait-driven solvers in
[`BsrMatrix`](./bsr_cpu.md#iterative-solvers) (`cg`, `bicgstab`,
`gmres`) with a cleaner API that returns a fresh result struct and
takes `F: Fn(&[Real]) -> Vec<Real>` instead of `&dyn LinearOperator`.
Both paths converge to the same iterates given the same matrix; the
choice is API ergonomics.

## Conventions

- All three solvers are `f64`.
- `op(v)` represents `A · v`; self-adjoint for `cg_solve` / `pcg_solve`,
  no symmetry constraint for `bicgstab_solve`.
- `tol` is the **relative tolerance** against `‖b‖`: convergence when
  `‖b − A·x‖ ≤ tol · max(‖b‖, ε)`.
- All three return after `max_iters` if they fail to converge — the
  caller inspects `CgResult::residual` and decides whether to relax
  `tol`, swap preconditioner, or fall back to a direct solve.

## `CgResult`

```rust
#[derive(Clone, Debug)]
pub struct CgResult {
    pub x: Vec<Real>,
    pub iters: usize,
    pub residual: Real,  // relative
}

impl CgResult {
    pub fn converged(&self, tol: Real) -> bool;
}
```

## CG and PCG

```rust
pub fn cg_solve<F>(op: F, b: &[Real], x0: &[Real], tol: Real, max_iters: usize) -> CgResult
where F: Fn(&[Real]) -> Vec<Real>;

pub fn pcg_solve<F, P>(
    op: F,
    precond: P,
    b: &[Real],
    x0: &[Real],
    tol: Real,
    max_iters: usize,
) -> CgResult
where F: Fn(&[Real]) -> Vec<Real>,
      P: Fn(&[Real]) -> Vec<Real>;
```

Plain CG with the standard scalar recurrence (`α = rr / pAp`, `β =
rr' / rr`). PCG adds one extra step per iteration: `z = M⁻¹·r`, and
the recurrence runs on `rz = ⟨r, z⟩` instead of `rr`. `M` must be SPD
for the convergence guarantee to hold.

Edge cases:

- **Zero RHS** ⇒ zero solution, exit immediately.
- **Degenerate search direction** (`|pAp| < 1e-30`) ⇒ exit gracefully
  with whatever residual we have. Don't fault.
- **Initial guess already converged** ⇒ exit with `iters = 0`.

## Jacobi preconditioner

```rust
#[derive(Clone, Debug)]
pub struct JacobiPreconditioner { /* inv_diag: Vec<Real> */ }

impl JacobiPreconditioner {
    pub fn new(diag: &[Real]) -> Self;
    pub fn apply(&self, r: &[Real]) -> Vec<Real>;
    pub fn inv_diag(&self) -> &[Real];
}
```

Cheapest possible preconditioner — one division per row per matvec.
`M⁻¹ = diag(1 / aᵢᵢ)`. Zero diagonals become zero (pass-through);
substituting `1` would scale non-zero RHS in a null-space row, which
is the wrong behaviour. Effective when the diagonal already dominates
(mass matrices, implicit-MPM grid systems, finite-difference
Laplacians with non-uniform coefficients) and harmless on
well-conditioned systems. For SPD matrices, `JacobiPreconditioner` is
itself SPD, preserving the symmetry guarantee that PCG requires.

Typical wiring:

```rust
use molla::math::iterative::{pcg_solve, JacobiPreconditioner};

let m = JacobiPreconditioner::new(&diag);
let result = pcg_solve(
    |x| matvec(x),
    |r| m.apply(r),
    &b,
    &x0,
    1e-8,
    200,
);
```

## When to pick which

- **`cg_solve` (unpreconditioned)** — small systems where you trust
  the conditioning, or as the baseline when benchmarking PCG. Test
  fixtures and the 1D Laplacian smoke tests run unpreconditioned.
- **`pcg_solve` with `JacobiPreconditioner`** — the default for any
  diagonally-dominant SPD system. Implicit-MPM grid solves, Stam-fluid
  pressure Poisson, FEM mass-matrix smoothers, parameter-ID normal
  equations — all use this.
- **`bicgstab_solve`** — non-symmetric `A`. The PDE-with-advection
  cases (Navier-Stokes, magnetostatics with anisotropic media). See
  the BiCGSTAB note below for the stability caveats.

## BiCGSTAB — stability caveats

```rust
pub fn bicgstab_solve<F>(
    op: F,
    b: &[Real],
    x0: &[Real],
    tol: Real,
    max_iters: usize,
) -> CgResult
where F: Fn(&[Real]) -> Vec<Real>;
```

BiCGSTAB stagnates when one of the orthogonality scalars (`ρ`, `ω`)
collapses near zero — inherent to the algorithm, not a bug. molla
guards with two heuristics drawn from Saad's *Iterative Methods for
Sparse Linear Systems* §7.4:

1. **Restart on `|ρ|` or `|ω|` collapse.** When either scalar falls
   below `1e-30`, re-seed the shadow residual from the current true
   residual and restart the recurrence (`r̂ = r`, `ρ = α = ω = 1`,
   `p = v = 0`). One fresh matvec, almost always recovers convergence
   on systems where plain BiCGSTAB would stall.

2. **Bail on `tt → 0`.** When `‖t‖² < 1e-30` — the preconditioned
   `s` lies in the kernel of `A` — accept the half-step `x ← x + α·p`
   and exit. Textbook safe-exit.

Even with these guards, BiCGSTAB remains less robust than GMRES for
pathological non-symmetric systems. Fall back to
[`crate::math::sparse::gmres`](./bsr_cpu.md#iterative-solvers) if you
hit a system where BiCGSTAB can't converge — GMRES has stronger
convergence guarantees at the cost of more memory (it stores `restart`
basis vectors).

## GPU port

There's no closure-driven GPU mirror. WGPU has no native sparse BLAS,
and the WGSL kernel for "user closure" would need to be hand-written
per consumer; instead each on-device solver wires its own
[`BsrAlgebraKernel`](./bsr_gpu.md) matvec into a custom CG loop.
Stage 27.1 in `PLAN.md` tracks the deferred port.
