# MPM adjoint

Stage 38.1 added a fully differentiable CPU MPM forward + adjoint
chain. The forward (`mpm_cpu_diff`) runs in `f64` with a recorded
tape; the adjoint walks the tape in reverse, threading derivatives
through P2G + grid update + G2P + corotated SVK stress, and emits
gradients for Young's modulus, initial positions, and initial
velocities. With Adam on top, a 64-particle / 30-step demo fits
`E` from a target final-state centroid to FD parity `4.14e−6`.

Source: `crates/molla/src/autograd/mpm_adjoint.rs`. Forward + tape
in `crates/molla/src/compute/mpm_cpu_diff.rs`. Shared math in
`crates/molla/src/math/dense3.rs` (`adj_svd3`).

## Why CPU-only

Per molla's [GPU-first rule](../architecture/design_rules.md),
forward solvers run on the GPU and the differentiable CPU path is
the exception. MPM is doubly so: the forward already exists in
`shaders/mpm.wgsl` for production runs; the diff path is a separate
`f64` host implementation purpose-built for inverse-problem demos.
The two are kept in parity by the explicit-MPM regression tests but
their data structures don't intersect — the tape would not survive
a wgpu round-trip.

## Forward pipeline (single step)

```text
P      = 2μ(F − R) + λ tr(S − I) R       # corotated PK1 (SVD: F = U Σ Vᵀ; R = U Vᵀ)
stress = −dt · vol · P · Fᵀ              # Cauchy stress contribution
P2G:   for each (p, node):
         mom_grid[node]  += w · m · (v + C · (xn − xp))
                          + (4w/dx²) · stress · (xn − xp)
         mass_grid[node] += w · m
grid:  v_grid = mom_grid / mass_grid + g · dt
       floor: v_grid.y = max(v_grid.y, 0)  if node below floor
              v_grid.x *= damp, v_grid.z *= damp
G2P:   for each p:
         v_new   = Σ_node w · v_grid
         C_new   = Σ_node (4w/dx²) · v_grid · (xn − xp)ᵀ      # APIC
         pos_new = pos + v_new · dt
         F_new   = (I + dt · C_new) · F
```

The Lamé constants `(μ, λ)` enter only through `P`. Everything else
is geometric.

## Adjoint chain

```text
P     = 2μ(F − R) + λ tr(S − I) R
g_E   = g_μ · dμ/dE + g_λ · dλ/dE
        with  dμ/dE     = 1 / (2(1+ν))
              dλ/dE     = ν / ((1+ν)(1−2ν))
```

The PK1 derivative `corotated_pk1_adjoint(...)` returns `(adj_F,
adj_μ, adj_λ)` given the upstream `adj_P`. The `F` chain in turn
threads through the SVD adjoint `adj_svd3` (so `R` and `S` are
differentiated as smooth functions of `F`).

The B-spline weight `w(pos)` and the associated scatters
`mass_grid = Σ_p w · m_p` and `mom_grid` are chained back into
`grad_pos_t` so multi-particle / multi-step gradients are
accurate — verified against finite differences to within
`1e−2` relative error on the same 64-particle / 30-step / 16³-grid
demo.

```rust
pub struct StateAdjoint {
    pub grad_pos: Vec<Vec3>,
    pub grad_vel: Vec<Vec3>,
    pub grad_c: Vec<Mat3>,
    pub grad_f: Vec<Mat3>,
    pub grad_mu: Real,
    pub grad_lambda: Real,
}

impl StateAdjoint {
    pub fn grad_youngs(&self, poisson: Real) -> Real;  // composes dμ/dE + dλ/dE
}

pub fn backward(tape: &MpmTape, grad_pos_final: &[Vec3]) -> StateAdjoint;
```

## Tape layout + cost

For `T` steps × `N` particles × `G³` cells:

- Particle tape `≈ T · N · 200 B`.
- Grid tape `≈ T · G³ · 56 B`.
- One full backward pass costs roughly `3×` a forward pass (the
  SVD adjoint is the dominant cost, slightly more expensive than
  the forward SVD).

The 64-particle / 30-step / 16³-grid demo:

- Tape `≈ 30 · (64 · 200 + 4096 · 56) ≈ 7 MB`.
- Backward `≈ 100 ms` on a laptop CPU (debug build).

## End-to-end example — fitting Young's modulus

```rust
use molla::compute::mpm_cpu_diff::{
    mpm_cpu_diff, MpmCpuConfig, MpmTape,
};
use molla::autograd::mpm_adjoint::{backward, forward_backward_e_centroid_loss};

let mut tape = MpmTape::default();
let cfg = MpmCpuConfig { /* dt, dx, gravity, floor, damp, ... */ };
let mut state = /* initial particles */;

// One forward step records into `tape` and updates `state`:
mpm_cpu_diff(&cfg, &mut state, &mut tape);

// Compute J = ‖centroid(state_T) − target‖² and ∂J/∂E in one call.
let (loss, grad_e) = forward_backward_e_centroid_loss(
    &cfg, &state0, target_centroid, n_steps,
);

// Adam step on E (in log space — see fem.md for the same trick).
e -= lr * grad_e;
```

FD parity for the demo's gradient: `4.14e−6` relative error
(`grad_E_analytic` vs central difference at `ε = 1e−4`). The same
chain composes with [robust losses](./tape.md) — wrap the centroid
residual in `norm_huber` before seeding `∂J/∂(centroid)`.

## What's not differentiable

- **Floor friction damping (`v_grid.{x,z} *= damp`)** is
  differentiable, but the `max(v_grid.y, 0)` floor clamp is
  non-smooth at the contact surface. The current implementation
  keeps the `max` in both forward and reverse with the standard
  sub-gradient convention (`dy/dx = 0` when clamped, `1` when
  passing through) — fine for trajectories that don't sit exactly
  on the floor.
- **Implicit-Bingham + semi-implicit MPM variants** ship a separate
  forward in `shaders/implicit_mpm.wgsl`. Their adjoint is on the
  backlog; only the explicit forward in `mpm_cpu_diff` is
  differentiable in v1.

## Cross-references

- [`Solvers / MPM`](../solvers/mpm.md) — forward (GPU) and
  algorithm overview.
- [`Solvers / MPM / Explicit`](../solvers/mpm_explicit.md) — the
  forward variant that `mpm_cpu_diff` mirrors.
- [`Math / Small-matrix decompositions`](../math/dense3.md) —
  `adj_svd3` SVD adjoint used inside `corotated_pk1_adjoint`.
- [`FEM adjoint`](./fem.md) — same Adam-in-log-space `E` fitting
  trick on the FEM forward.
- [`Tape-based autograd`](./tape.md) — generic tape; MPM uses its
  own purpose-built `MpmTape` because the per-step data structure
  is dense + statically shaped.
