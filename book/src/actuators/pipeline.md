# Composable actuator pipeline

Stage 18.1 introduced molla's composable actuator pipeline — a thin
orchestrator that wires three pluggable stages in series:

```text
Delay  →  Controller  →  Clamping
```

It mirrors Newton's `Actuator` class
(`xtra/newton/_src/actuators/actuator.py`) and is the structure that
[`PD / PID`](./pd_pid.md), [`neural`](./neural.md), and
[`DC-motor clamping`](./dc_motor.md) controllers all plug into. The
orchestrator type is `molla::actuators::Actuator`.

Source: `crates/molla/src/actuators/actuator.rs`. Each stage lives
in its own file (`delay.rs`, `controller_*.rs`,
`clamping/*.rs`) and has its own WGSL kernel
(`shaders/actuator_*.wgsl`, `shaders/clamping_*.wgsl`).

## The three stages

### 1. Delay

Pushes the current command targets into a per-DOF circular buffer
and reads the per-DOF lagged sample. Use to emulate the
communication / actuation lag of real hardware. When constructed
with `delay_steps == 0` for every actuator, the kernel passes the
current target through unchanged (zero-overhead passthrough). See
[`Delay`](./pd_pid.md#delay) (used inside the pipeline) for
configuration knobs.

### 2. Controller

Consumes the (possibly lagged) targets and writes raw effort into
`Control.joint_force[dof]`. Concrete variants live behind the
`ControllerStage` enum so the orchestrator can hold a non-`Box`
controller without dynamic dispatch in the per-substep hot path:

```rust
pub enum ControllerStage {
    Pd(ControllerPipelinePd),
    Pid(ControllerPipelinePid),
    NeuralMlp(ControllerNeuralMlp),
    Lstm(ControllerNeuralLstm),
}
```

- **Pd / Pid** — closed-loop joint-space controllers; see
  [`PD / PID controllers`](./pd_pid.md).
- **NeuralMlp / Lstm** — learned policies whose output writes
  directly into `joint_force`; see
  [`Neural controllers`](./neural.md). The MLP / LSTM read their
  observation vector from a host-pushed buffer (`set_inputs(...)`)
  rather than from the delay-stage outputs, so the delay still
  ticks each step but produces nothing the policy consumes.

### 3. Clamping

Modifies `Control.joint_force[dof]` *in place* to enforce
actuator-physics limits. Variants:

```rust
pub enum ClampingStage {
    None,                              // no-op
    MaxEffort(ClampingMaxEffort),      // symmetric cap |τ| ≤ τ_max
    DcMotor(ClampingDcMotor),          // four-quadrant DC-motor envelope
    PositionBased(ClampingPositionBased), // shared (q → τ_max) lookup table
}
```

All clamps read the current effort plus whatever state they need
(joint position for `PositionBased`, joint velocity for `DcMotor`)
and write the clipped value back into the same `joint_force` slot.
See [`DC motor clamping`](./dc_motor.md) for the
torque-vs-speed envelope details.

## API

```rust
use molla::actuators::*;

let delay = Delay::new(
    &engine,
    &[DelayConfig { dof: 0, delay_steps: 4 },
      DelayConfig { dof: 1, delay_steps: 4 }],
    /* max_delay = */ 8,
)?;

let pd = ControllerPipelinePd::new(&engine, &model, &[
    ControllerPipelinePdConfig { dof: 0, kp: 50.0, kd: 5.0 },
    ControllerPipelinePdConfig { dof: 1, kp: 50.0, kd: 5.0 },
])?;

let clamp = ClampingMaxEffort::new(&engine, &[
    ClampingMaxEffortConfig { dof: 0, max_effort: 20.0 },
    ClampingMaxEffortConfig { dof: 1, max_effort: 20.0 },
])?;

let mut actuator = Actuator::new(
    delay,
    ControllerStage::Pd(pd),
    ClampingStage::MaxEffort(clamp),
);

// Per-substep:
actuator.step(&engine, &state, &mut control, dt)?;
```

`Actuator::step(engine, state, control, dt)` runs all three stages
in order on the GPU; `dt` is consumed only by the integral term of
the PID controller (the PD / MLP / LSTM variants ignore it).

The orchestrator does **not** verify that the per-stage DOF lists
match in arity or order — the caller is responsible for keeping the
configs aligned. The motivation is composition flexibility: a
neural policy that writes to 12 DOFs while a `ClampingMaxEffort`
caps only the 4 most powerful actuators is a perfectly reasonable
configuration, and forcing equal arity at the API layer would
preclude it.

## DOF aliasing

Each stage's per-DOF config list is independent, but inside one
stage every config slot must target a **unique** DOF. Two PD
actuators driving the same DOF would race in
`Control.joint_force[dof]`; the constructor in `ControllerPd::new`
explicitly rejects duplicates with a `Solver` error.

If two pipelines drive disjoint DOF sets and run in parallel, that's
fine — the caller orchestrates the order. If two pipelines drive
*overlapping* DOFs, the caller must zero `joint_force` between them
or accept that the second write wins.

## Equivalence with the standalone controllers

A pipeline configured as `(delay_steps = 0, ctrl = PD, clamp =
MaxEffort)` is bit-exact equivalent to the standalone
[`ControllerPd`](./pd_pid.md) with the same gains — same kernel,
same write-once-into-`joint_force` semantics. The pipeline form
gives you the option to swap in delay or different clamping without
touching the controller code, at the cost of one extra
passthrough kernel dispatch per step.

## Cross-references

- [`PD / PID controllers`](./pd_pid.md) — controller-stage details.
- [`Neural controllers`](./neural.md) — MLP / LSTM controllers
  inside `ControllerStage::NeuralMlp` / `Lstm`.
- [`DC motor clamping`](./dc_motor.md) — back-EMF clamp variant.
- [`Muscle (Hill-type + SEE)`](./muscle.md) — alternative actuator
  family that runs through `solvers::featherstone` rather than
  this pipeline (no pipeline integration yet — Stage 42 deliberate).
