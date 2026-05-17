# Featherstone reverse adjoint

Stage 23 introduced reverse-mode automatic differentiation for the
Featherstone Articulated Body Algorithm (ABA) — i.e. closed-loop
trajectory-optimisation gradients through a rigid-body dynamics
rollout. The implementation is hybrid (analytic chain rule over the
integrator + per-step Jacobian) and ships in four iterative cuts
within Stage 23.1, plus a robust-loss extension at Stage 24b.

Source: `crates/molla/src/autograd/featherstone_adjoint.rs`. Shader
(GPU scaffold): `crates/molla-kernels/shaders/featherstone_adjoint_jacobians.wgsl`.

## What it solves

Given a serial revolute chain with `n` links, a joint torque
trajectory `τ_{0..T-1}`, and a scalar loss `L(q_T, qd_T)` (typically
end-effector tracking), the adjoint returns `∂L/∂τ_t` for every step
`t ∈ [0, T)`. The headline use cases:

- Inverse-dynamics trajectory optimisation (Carpentier & Mansard
  2018).
- Differentiable model-predictive control.
- Sim-to-real domain randomisation with learned controllers in the
  loop.

The three Featherstone sweeps map onto three `TapeOp` chunks:

| Sweep                  | TapeOp                                  |
|------------------------|------------------------------------------|
| Forward kinematics + v | `FeatherstoneFwdKinematics`             |
| Articulated inertia    | `FeatherstoneUpwardInertia`             |
| Forward acceleration   | `FeatherstoneDownwardAccel`             |

These ops are recorded for tape inspection / determinism checks but
the adjoint is **not** evaluated through the generic
`Tape::backward` dispatch — the three sweeps form an inseparable
unit (inertia + acceleration sweeps need the kinematic
intermediates) and run together inside
`FeatherstoneRevoluteChain::reverse`.

## Stage 23.1 cuts

### 23.1.1 — closed-form CPU, single-DoF

The first cut shipped:

- A hand-written analytic adjoint over the symplectic-Euler
  integrator that threads `∂L/∂q + ∂L/∂qd` through every step in
  closed form, accumulating `∂L/∂τ_t` at every step.
- A **per-step single-step Jacobian** computed via central
  differences on the full Featherstone forward at this step's
  `(q_t, qd_t)`. For serial revolute chains of size ≤ 7 the cost is
  `O(3n · per-step)` per step — negligible compared to trajectory
  length.
- Coverage: single pendulum, 2-link planar arm, 6-DoF Franka chain.

This is the standard "adjoint via gradient through inverse
dynamics" pattern from Carpentier & Mansard 2018 ("Analytical
Derivatives of Rigid Body Dynamics Algorithms"), applied at the
single-step granularity.

### 23.1.2 — multi-DoF coverage + GPU scaffold

Extended the per-link math to **stacked single-DoF representations**
of Ball / Free / Prismatic joints. Each multi-DoF user joint is
exposed as a single composite joint to user code while the adjoint
operates on the equivalent stacked single-DoF representation. This
keeps the closed-form Jacobian path uniform: no special-case kernels
per joint type, just per-DoF spatial Jacobian columns.

`AutogradError::UnsupportedJoint` surfaces if a composite kind
(`Ball / Free / Distance / D6`) reaches the per-link math directly —
the user is expected to decompose upstream. Helpers like
`FeatherstoneRevoluteChain::spherical_pendulum(...)` handle the
decomposition for the standard cases.

The GPU scaffold (`featherstone_adjoint_jacobians.wgsl`) is in
place but not yet wired through the live solver — the
per-step Jacobian on the device is the obvious next port and gives
the largest speedup for long chains.

### 23.1.3 — Distance + D6 adjoint kinematics

Added per-DoF adjoint contributions for `Distance` and `D6`
composite joints by exposing them as stacked single-DoF chains. The
trajectory-level adjoint is unchanged; only the per-DoF spatial
Jacobian columns extend.

### 23.1.4 — analytic ∂h/∂q (CPU)

Replaced the central-difference per-step Jacobian with an analytic
derivative of the full ABA inertia / bias sweep. About 5–7× faster
on the 6-DoF Franka case and removes the central-difference
quantisation that limited gradient checking to `1e-3` (the analytic
version passes `1e-5`).

### 24b — robust-loss extension

The six robust-loss `TapeOp` variants (`HuberLoss`, `TukeyLoss`,
`CauchyLoss`, etc.) compose with the Featherstone trajectory adjoint:
the EE-position error can be wrapped in `norm_huber(e, δ)` before
the seed `∂L/∂(EE)` flows into the Featherstone reverse, so
trajectory optimisation with a robust loss runs end-to-end through
the tape without bespoke seed logic.

## Public API

```rust
use molla::autograd::{
    record_featherstone_step, FeatherstoneRevoluteChain,
    FeatherstoneTrace, JointKind, JointProjection,
    JacobianScratch, JacobianStrategy, RevoluteLink,
};

// Build the chain (decomposed to single-DoF links).
let chain = FeatherstoneRevoluteChain::from_links(vec![
    RevoluteLink { /* ... */ },
    RevoluteLink { /* ... */ },
    /* ... */
]);

// Roll out forward with torques `tau`, populating a trace.
let mut trace = FeatherstoneTrace::default();
let (q_T, qd_T, ee_T) = chain.rollout(q0, qd0, &tau, dt, &mut trace);

// Seed the loss gradient (∂L/∂ee at the final step) and reverse.
let grad_ee_T = /* user-supplied seed */;
let grads = chain.reverse(
    &trace,
    grad_ee_T,
    JacobianStrategy::Analytic, // or CentralDifference
    &mut JacobianScratch::new(chain.n_dof(), trace.steps()),
);

// grads.tau[t] is ∂L/∂τ_t for each step.
```

## Implementation strategy summary

Featherstone's three sweeps are nontrivial to differentiate
line-by-line — the §10 derivation is a multi-week algebra exercise.
For Stage 23.1 v1 molla ships a hybrid adjoint that is correct,
verifiable, and matches the reverse-mode contract:

- **Per-step "single-step" Jacobian** — analytic (23.1.4) or
  central-difference (23.1.1 fallback).
- **Trajectory-level adjoint** — fully analytical: the chain rule
  threads `∂L/∂q + ∂L/∂qd` through the symplectic-Euler integrator
  in closed form, accumulating `∂L/∂τ` at every step.
- **Tape-level recording** — each sweep maps to its own `TapeOp`
  chunk so a future GPU port can swap the per-step Jacobian build
  for a hand-derived analytical sweep without touching the
  trajectory adjoint or the tape format.

## Determinism

Every operation is a fixed-shape arithmetic loop over the trace —
no `HashMap` iteration, no parallel reductions, no GPU readback
reorder. Identical inputs (`q0`, `qd0`, `tau`) yield bit-exact
identical gradients across runs, verified by the
`tape_replay_is_deterministic` test in
`tests/autograd_featherstone.rs`.

## Cross-references

- [`Tape-based autograd`](./tape.md) — generic `Tape` infrastructure
  that hosts the three sweep chunks.
- [`Sim / Featherstone ABA`](../sim/featherstone.md) — forward
  algorithm the adjoint differentiates.
- [`Math / Iterative solvers`](../math/iterative.md) — robust-loss
  functions used by Stage 24b.
- References: Featherstone 2008 (Rigid Body Dynamics §10),
  Carpentier & Mansard 2018 ("Analytical Derivatives of Rigid Body
  Dynamics Algorithms").
