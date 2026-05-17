# Semi-implicit integrator

[`SemiImplicitSolver`](https://docs.rs/molla/latest/molla/solvers/struct.SemiImplicitSolver.html)
is molla's plain **symplectic Euler integrator** for free rigid
bodies. It's the simplest integrator molla ships — no constraints,
no contacts, just gravity + applied wrenches → new pose + velocity.

```rust,ignore
use molla::solvers::{SemiImplicitSolver, Solver};

let mut solver = SemiImplicitSolver::new_cpu();        // or new_gpu(&engine)
solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out)?;
```

The contact set is **ignored** — Semi-Implicit doesn't resolve any
constraints. For contact-rich rigid bodies use [XPBD](./xpbd.md); for
articulated systems use [Featherstone](../sim/featherstone.md).

## Algorithm

For each body `i`:

```text
applied_wrench = body_f[i] + control.body_external_f[i]   (world frame, at COM)
g = world_gravity[body_world[i]]

# Linear:
v_lin_new = v_lin + dt · (applied_force / mass + g)
p_new     = p_old + dt · v_lin_new                         (semi-implicit: uses new v)

# Angular:
ω_world          = body_qd.angular                          (world frame)
I_world          = R · I_body · Rᵀ                          (transform inertia to world)
τ_gyro           = ω_world × (I_world · ω_world)            (gyroscopic term)
ω_dot            = I_world⁻¹ · (applied_torque − τ_gyro)
ω_new            = ω_world + dt · ω_dot

# Quaternion update:
dq    = quat(0, ω_new) · 0.5 · dt
q_new = normalize(q_old + dq · q_old)
```

**Semi-implicit** = the position update uses the **new** velocity
(not the old). This makes the integrator a **symplectic** map on the
phase space — energy is bounded over arbitrarily long simulations
(though it may oscillate). The angular part uses the gyroscopic-
preserving form so a spinning rigid body precesses correctly.

Kinematic bodies (`BodyFlags::KINEMATIC` set) are passed through
unchanged — their pose / velocity are user-prescribed.

## Use cases

- **Free-floating debris** that doesn't need collisions (test
  fixtures, viewer demos, projectile motion).
- **Smoke-test integrator** against analytical trajectories (drop
  test, parabolic arc) — every parity test starts with a
  `SemiImplicitSolver` check.
- **Per-body authoritative integrator** when the user wants to apply
  ad-hoc wrenches each step (RL "push the cube" tasks).

It does **not** handle:

- Contacts (use XPBD).
- Joints (use Featherstone).
- Soft bodies (use VBD or XPBD particles).

## CPU implementation

The CPU path is a one-loop kernel — for each body apply gravity,
external wrench, integrate linear and angular state, normalise the
quaternion. ~100 lines of plain Rust; serves as the parity oracle for
the GPU kernel.

## GPU implementation

```rust,ignore
#[cfg(feature = "gpu")]
let solver = SemiImplicitSolver::new_gpu(&engine)?;
```

Backed by [`IntegrateBodiesKernel`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/integrate_bodies.wgsl)
— one workgroup invocation per body, identical maths. The kernel
binds:

- `Model.body_*`           (mass, inertia, com, flags, world)
- `Model.gravity`          (per-world)
- `State.body_q, body_qd`  (input)
- `Control.body_external_f` (external wrench; dummy 1-element buffer
  when the user-supplied `Control` is empty)
- `state_out.body_q, body_qd` (output)

The kernel matches the CPU oracle to ~1e-9 (f32) / ~1e-15 (f64)
roundoff per step under the parity tests. The `dummy_external_f`
buffer pattern is the same trick used for optional inputs throughout
the codebase: 1 element of zeros, always bound so the WGSL bind group
layout doesn't depend on whether the caller supplied a `Control`.

## Substepping

`SemiImplicitSolver` does **not** carry its own substep knob — the
caller is expected to chain multiple `step(dt/n)` calls externally if
finer resolution is needed. The integrator is `O(n)` in body count
and the GPU dispatch overhead amortises across 1000s of bodies, so
this is rarely a bottleneck.

## Differentiability

Stage 23 added a forward-mode autograd tape for rigid-body integration;
the reverse pass is handled by the generic
[`crate::autograd::tape`](../autograd/tape.md) walker. Since the
integrator is contact-free, the gradient is exact (no contact
discontinuities to worry about) — useful for short-horizon
trajectory optimisation against a free-flight rigid target.
