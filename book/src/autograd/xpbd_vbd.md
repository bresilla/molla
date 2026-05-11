# XPBD / VBD adjoint

molla ships closed-form reverse-mode adjoints for the XPBD
distance-constraint solver and a partial GPU path for VBD cloth.
Both follow the *"frozen narrow-phase"* pattern Newton uses:
freeze the contact set at the start of the step, treat the per-
contact normals + bodies as constants for both forward and
reverse, and differentiate only the smooth constraint-projection
math.

Source: `crates/molla/src/autograd/xpbd_adjoint.rs` (XPBD), plus
the VBD adjoint scaffolding in the same module. Shader (partial
GPU XPBD adjoint): `crates/molla/shaders/xpbd_equality.wgsl` gives
the equality-constraint forward; the adjoint counterpart is
host-only in v1.

## Why "frozen narrow-phase"?

Contact generation is fundamentally non-smooth: the set of active
contacts changes discretely as bodies move. Differentiating through
contact onset would require subgradient calculus and Newton's
practice (mirrored here) is to:

1. Run the narrow phase once at the start of the step.
2. Treat the resulting `(body_a, body_b, normal, depth)` tuples as
   **constants** for both the forward and reverse pass.
3. Differentiate only the smooth constraint-projection equations
   that act on those frozen tuples.

This gives a well-defined `∂(post-step state) / ∂(pre-step state +
parameters)` jacobian for the duration of the step, at the cost
of zero gradient flow through contact-onset events. In practice
this is fine for trajectory optimisation over short horizons — the
discrete events do not contribute usable gradient anyway.

## XPBD distance-constraint adjoint

Forward (matches `solvers::xpbd::step_particles` minus the strain
limiter and ground plane — both non-smooth):

```text
1. Predict: p_pred = p + (v + g · dt) · dt
2. For each Gauss-Seidel iteration over the spring list:
     d   = p_b − p_a
     L   = ‖d‖
     n   = d / L
     C   = L − rest
     α   = compliance / dt²
     Δλ  = −(C + α · λ) / (w_a + w_b + α)
     λ  += Δλ
     p_a −= w_a · Δλ · n
     p_b += w_b · Δλ · n
3. Velocity recovery: v_post = (p_final − p_init) / dt
```

Reverse: walk the iterations in reverse, undoing each constraint
update by replaying its position effect on a saved per-iteration
`(p_a, p_b, λ)` snapshot. The derivation (Macklin et al., *XPBD:
Position-Based Simulation of Compliant Constrained Dynamics*,
2016 §4) shows each `Δλ` is a smooth function of the saved
positions + `λ`, so the chain rule applies cleanly.

### Saved adjoint state

```rust
pub struct DistanceConstraintAdjoint {
    pub p_a_pre: Vec3,
    pub p_b_pre: Vec3,
    pub lambda_pre: Real,  // cumulative λ before this iteration
}
```

One entry per `(iteration, spring)` pair. For a 100-iteration /
1000-spring chain that's 100k entries × ~64 bytes ≈ 6.4 MB, well
within host budget for the trajectory-optimisation use case.

### Public API

```rust
use molla::autograd::{
    xpbd_distance_step_with_grad,
    distance_chain_grad_initial_pos,
    distance_chain_grad_stiffness,
    DistanceConstraintAdjoint,
};

// Forward + trace recording.
let (p_out, lambda_out, adj_trace) = xpbd_distance_step_with_grad(
    &p_init, &v_init, &inv_mass,
    &spring_a, &spring_b, &rest_length, &compliance,
    gravity, dt, iterations,
);

// Gradient of (final chain length) wrt the initial positions.
let grad_p0 = distance_chain_grad_initial_pos(
    &p_init, &v_init, &inv_mass,
    &spring_a, &spring_b, &rest_length, &compliance,
    gravity, dt, iterations,
);

// Gradient wrt per-spring stiffness (the inverse-compliance handle).
let grad_k = distance_chain_grad_stiffness(/* same args */);
```

### Why hand-written instead of taped?

The XPBD inner loop is the hot path; recording each `Add` / `Sub` /
`Mul` / `Norm` through the generic [`Tape`](./tape.md) would mean
hundreds of thousands of heap allocations per step (one per op per
spring per iteration). Hand-written closed-form gradients run the
reverse pass over the same loop the forward path uses, without any
heap allocation past the `adj_trace` snapshot — orders of
magnitude faster, and matches Newton's `wp.func_grad` pattern of
shipping hand-derived gradients for hot solver paths.

The verification harness in `tests/autograd_xpbd.rs` checks the
analytical adjoint against central-difference finite differences
to within `1e-3` (standard `gradcheck` tolerance).

## VBD adjoint — partial GPU

The VBD (Vertex Block Descent) solver is XPBD's smoother sibling:
instead of Gauss-Seidel over springs, it does block-Jacobi
iterations over the local Hessian at each vertex. The adjoint
follows the same frozen-narrow-phase pattern but the per-iteration
update is denser (a 3×3 block solve per vertex instead of a 1×1 λ
update per spring).

molla ships VBD on the GPU via `shaders/vbd_cloth.wgsl` for the
forward pass; the adjoint is **scaffolded but not yet GPU-resident**.
The CPU adjoint follows the same shape as the XPBD one and is
useful for testing the parity contract before the GPU port lands.
This mirrors Newton's status: VBD's reverse-mode kernel is on
their backlog too, exactly because the per-iteration matrix shape
makes the closed-form adjoint denser.

## Limits + scope

- **Smooth constraints only.** The XPBD strain limiter (a hard
  bound on `‖d‖ / rest`) and the ground-plane projection are
  non-smooth — they're excluded from the differentiable forward in
  v1. The forward path stays bit-identical to the non-strain-limited
  XPBD; the simulation just doesn't allow runaway stretching.
- **No friction gradient.** Tangential friction couples to the
  contact set in a non-smooth way; v1 doesn't try to differentiate
  through it.
- **No mass differentiability.** `inv_mass` is treated as a
  constant. `inv_mass = 0` still pins the particle (its position
  survives the projection unchanged).

These restrictions disappear case-by-case as the soft-contact
pipeline lands smooth surrogates for friction and contact
inception.

## Cross-references

- [`Solvers / XPBD`](../solvers/xpbd.md) — forward solver.
- [`Solvers / VBD`](../solvers/vbd.md) — VBD forward.
- [`Tape-based autograd`](./tape.md) — generic tape for cases that
  don't justify hand-written adjoints.
- [`MPM adjoint`](./mpm.md) — sister hand-written adjoint for the
  diff-MPM forward.
- References: Macklin et al. 2016 ("XPBD: Position-Based
  Simulation of Compliant Constrained Dynamics").
