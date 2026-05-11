# FEM adjoint

Stage 38.2 ships the classical adjoint method for the linear-
elasticity FEM solve: given a forward problem `K(p) · u = f(p)` and
a scalar loss `J(u, p)`, the adjoint computes `∂J/∂p` with **one
extra linear solve** of the same matrix — independent of the
dimensionality of `p`. The implementation reuses molla's existing
sparse-matrix infrastructure (BSR + CG) and exposes a single
`solve_adjoint(...)` entry-point.

Source: `crates/molla/src/fem/adjoint.rs`. Forward solver:
`crates/molla/src/fem/elasticity.rs`. Glue with optimisers:
`molla::math::optim::Adam`.

## Algorithm (textbook recap)

1. **Forward**: solve `u = K(p)⁻¹ · f(p)`.
2. **Adjoint**: solve `Kᵀ · λ = ∂J/∂u`. For SPD `K` (linear
   elasticity always is), `Kᵀ = K`, so the same matrix is reused.
3. **Gradient**:
   `∂J/∂p = ∂J/∂p|_explicit + λᵀ · (∂f/∂p − ∂K/∂p · u)`.

For linear elasticity the per-element stiffness scales **linearly**
with Young's modulus `E`:

- `μ = E / (2(1 + ν))`,  `λ_Lamé = E · ν / ((1 + ν)(1 − 2ν))`
- Both are proportional to `E`, so `∂K/∂E = K / E` globally.

The gradient collapses to

```text
∂J/∂E = −λᵀ · K · u / E
```

for load-independent loss functions, and to a slightly longer form
when `f` depends on `E` too.

## Traction-load gradient

For an applied traction `t` on a face with area `A`, P1 consistent
lumping gives `f_v += (A / 3) · t` per face vertex (component-by-
component):

```text
∂f/∂t_c = (A / 3) · e_c summed over the 3 face vertex DoFs
```

so

```text
∂J/∂t_c = (A / 3) · Σ_{v ∈ face} λ_{v · dim + c}
```

For each face component the gradient is a one-pass sum over the
face's vertex DoFs — `O(face_count)` total, no extra solves.

## Public API

```rust
use molla::fem::adjoint::{
    solve_adjoint, solve_adjoint_with,
    compute_grad_youngs, compute_grad_load,
};

// 1. Forward (existing FEM solver).
let u = solve_elasticity(&mesh, &bcs, e, nu, traction)?;

// 2. Adjoint: one CG solve of K · λ = ∂J/∂u.
let lambda = solve_adjoint(&mesh, &bcs, e, nu, &grad_j_wrt_u)?;

// 3. Compose ∂J/∂E.
let grad_e = compute_grad_youngs(&mesh, &bcs, e, nu, &u, &lambda);

// 4. Compose ∂J/∂t (per face component).
let grad_t = compute_grad_load(&mesh, &bcs, &lambda, /* face_ids = */ &faces);
```

`solve_adjoint_with(..., cg_solve)` lets the caller pass a custom CG
closure — handy when chaining with the same preconditioner used in
the forward to keep iteration counts identical.

## Adam in log space — fitting `E` from a target deflection

The Stage 38.2 demo recovers `E` from a target deflection measured
at the free end of a beam. Two practical tricks:

1. **Optimise `ξ = ln(E)` instead of `E` directly.** This guarantees
   `E > 0` at every step without a projection, and the gradient
   `dJ/dξ = dJ/dE · E` naturally rescales — Adam learns at a sensible
   step size across orders of magnitude.

2. **Use Adam, not vanilla SGD.** The `J(E)` landscape for an
   elasticity inverse problem is steep but smooth; Adam's momentum +
   per-parameter learning-rate adaptation converges in ~50 iters
   where SGD would need 500.

```rust
use molla::math::optim::Adam;

let mut opt = Adam::new(/* lr = */ 0.05);
let mut xi = (e_initial.ln(),);   // single-parameter Adam state
for _ in 0..n_iters {
    let e = xi.0.exp();
    let u = solve_elasticity(&mesh, &bcs, e, nu, traction)?;
    let j = loss(&u);
    let dj_du = grad_loss_u(&u);
    let lambda = solve_adjoint(&mesh, &bcs, e, nu, &dj_du)?;
    let grad_e = compute_grad_youngs(&mesh, &bcs, e, nu, &u, &lambda);
    let grad_xi = grad_e * e;
    opt.step(&mut xi.0, grad_xi);
}
```

This produces an `E` estimate within 1 % of ground truth on the demo
beam after ~50 Adam iterations.

## Boundary-condition handling

Dirichlet BCs are eliminated by **row-and-column elimination** in
both forward and adjoint solves, so the BC rows don't pollute either
linear system. The same elimination structure is reused (computed
once at mesh build), so the adjoint solve costs roughly the same as
the forward — a single matrix factorisation / preconditioner setup
amortises across both.

Neumann (traction) BCs enter through the right-hand side and are
differentiable per the formula above; the BC structure is otherwise
unchanged.

## Validation

The Stage 38.2 tests compare the analytic gradient against
central-difference FD at `ε = 1e−3 · E` and pass to within
`5e−3` relative error on the cantilever beam case (`hex` mesh of
~10k DoFs).

## Cross-references

- [`FEM / Adjoint / differentiable FEM`](../fem/adjoint.md) — the
  primary FEM-side chapter (same source file; this autograd-side
  chapter focuses on the autograd contract).
- [`FEM / GPU elasticity`](../fem/elasticity_gpu.md) — forward
  solver the adjoint reuses.
- [`Math / Optimizers`](../math/optim.md) — Adam implementation
  shared with the MPM / splat demos.
- [`MPM adjoint`](./mpm.md) — same Adam-in-log-space fitting
  pattern over a different forward.
