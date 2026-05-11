# DC motor clamping

The `Clamping*` family enforces actuator-physics limits *after* the
controller has written raw effort into `Control.joint_force[dof]`.
Three variants ship today, all configurable via the
[composable pipeline](./pipeline.md)'s `ClampingStage`:

- **`ClampingMaxEffort`** — symmetric per-actuator cap `|τ| ≤ τ_max`.
- **`ClampingDcMotor`** — four-quadrant DC-motor torque-vs-speed
  envelope (back-EMF + current limit).
- **`ClampingPositionBased`** — shared `(position → τ_max)` lookup
  table, useful for joints whose available effort drops near hard
  limits.

Source: `crates/molla/src/actuators/clamping/`. Shaders:
`shaders/clamping_*.wgsl`. Every clamp reads its inputs, computes
the clipped effort, and writes it back into the same
`joint_force[dof]` slot — no auxiliary buffers, no host
round-trip.

## ClampingDcMotor — four-quadrant envelope

Models a brushed DC motor with current-limited torque ceiling.
Parameters mirror what's on a real motor datasheet:

```rust
pub struct ClampingDcMotorConfig {
    pub dof: u32,
    pub stall_torque: Real,       // τ_stall : peak torque at zero velocity
    pub no_load_speed: Real,      // ω_nl    : velocity at zero torque
    pub max_motor_effort: Real,   // τ_cap   : current-limited torque ceiling
}
```

The physical model is:

```text
back-EMF envelope:    τ_line(ω) = τ_stall · (1 − ω / ω_nl)
current cap:          τ_cap
applied:              τ_avail(ω) = min(τ_line(ω), τ_cap)
```

In the four-quadrant case (positive *and* negative ω, motoring *and*
regenerating), the envelope is mirrored across the origin to form a
trapezoid, and the kernel clamps the requested effort to lie inside
it.

### The corner velocity

The linear envelope `τ_stall · (1 − ω/ω_nl)` and the current cap
`τ_cap` intersect at a **corner velocity**

```text
ω_corner = ω_nl · (1 + τ_cap / τ_stall)
```

stored per-actuator in `corner_velocity` and recomputed every time
`set_params(...)` runs. Beyond `ω_corner` the linear envelope is
already below the cap and the cap doesn't bite — and worse, the
linear envelope would *flip sign* without the clamp, sending the
torque limit into the wrong quadrant. The kernel clamps `ω` to
`±ω_corner` before evaluating the line, which keeps the envelope
well-defined for arbitrarily high velocities.

If `τ_cap ≥ τ_stall` the corner is `ω_nl` itself (the cap is
inactive — the linear envelope is the binding constraint
everywhere).

### Validation

The constructor rejects:

- `no_load_speed <= 0` — would divide by zero in the envelope.
- `stall_torque < 0` — physically meaningless.
- `max_motor_effort < 0` — physically meaningless.

`set_params(...)` accepts in-place updates of all three parameters
plus the implicit `corner_velocity` recompute, making it cheap to
sweep motor sizes during tuning without rebuilding the GPU
resources.

### API

```rust
use molla::actuators::{ClampingDcMotor, ClampingDcMotorConfig};

let dc = ClampingDcMotor::new(&engine, &[
    ClampingDcMotorConfig {
        dof: 0,
        stall_torque: 12.0,     // N·m
        no_load_speed: 30.0,    // rad/s
        max_motor_effort: 8.0,  // N·m current-limited ceiling
    },
])?;

// Wire into the actuator pipeline:
let mut actuator = Actuator::new(delay, ctrl, ClampingStage::DcMotor(dc));
```

## ClampingMaxEffort — symmetric cap

The simplest clamp:

```text
τ_clipped = clamp(τ, −τ_max, +τ_max)   if τ_max > 0
τ_clipped = τ                          if τ_max == 0  (disabled)
```

```rust
pub struct ClampingMaxEffortConfig {
    pub dof: u32,
    pub max_effort: Real,  // 0 disables (passthrough)
}
```

Equivalent to the in-kernel symmetric cap on
[`ControllerPd`](./pd_pid.md), exposed as a standalone clamping
stage so the controller can be replaced (e.g. with a neural policy)
without losing the safety net. `set_max_effort(...)` updates the
caps in-place from host data.

## ClampingPositionBased — shared lookup table

All actuators in one stage share the same `(position → max_effort)`
table; the table is interpolated linearly per-DOF using the current
`state.joint_q[dof]`. Models joints whose available torque drops as
they approach their hard limits (e.g. real servos that current-limit
near end-stops).

```rust
let pb = ClampingPositionBased::new(
    &engine,
    /* dof_indices = */ &[0, 1, 2],
    /* positions   = */ &[-1.57, -0.5, 0.0, 0.5, 1.57],
    /* efforts     = */ &[ 0.0,   5.0, 20.0, 5.0, 0.0 ],
)?;
```

The constructor validates that `positions` is monotonically
non-decreasing (else the linear-interp dispatch's binary-search
behaviour is undefined) and that every `effort` is non-negative.

## Why a clamp instead of a tighter controller?

Real motors saturate. A PD controller doesn't know it's commanding
a torque the motor cannot deliver — feeding the raw effort into
the integrator would cause overshoot, oscillation, or integrator
wind-up. The clamping stage models the *actuator* (whose envelope
is physics, not control), and the controller is free to ignore it.
Pairing `ControllerPid` with `ClampingDcMotor` is the canonical
recipe for an RL environment that wants the policy to learn the
motor's natural envelope.

## Cross-references

- [`Composable actuator pipeline`](./pipeline.md) — defines
  `ClampingStage` and wires the clamp after the controller.
- [`PD / PID controllers`](./pd_pid.md) — typical upstream
  controller stages.
- [`Neural controllers`](./neural.md) — same upstream slot, learned
  weights instead of hand-tuned gains.
