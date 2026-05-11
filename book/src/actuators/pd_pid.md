# PD / PID controllers

The `ControllerPd` and `ControllerPid` types are molla's
canonical joint-space closed-loop controllers, mirroring Newton's
`ControllerPD` and `ControllerPID` minus the feed-forward /
const-effort plumbing. They come in two flavours:

- **Standalone** — `ControllerPd` and `ControllerPid` read directly
  from `Control.joint_target_*` and write to `Control.joint_force`,
  one kernel dispatch per step.
- **Pipeline** — `ControllerPipelinePd` and `ControllerPipelinePid`
  read from upstream `Delay` output buffers and otherwise behave
  identically; they're the controllers the
  [composable pipeline](./pipeline.md) plugs into.

Source: `crates/molla/src/actuators/controller_pd.rs`,
`controller_pid.rs`, `controller_pipeline_*.rs`. Shaders:
`shaders/actuator_pd.wgsl`, `shaders/actuator_pid.wgsl`,
`shaders/actuator_pipeline_*.wgsl`. Per molla's GPU-first rule, all
effort math runs in WGSL — the Rust orchestrators only own the
per-actuator parameter buffers and the dispatch wrapper.

## PD: effort law

```text
effort = kp · (target_pos − q) + kd · (target_vel − qd)
effort = clamp(effort, −max_effort, +max_effort)   if max_effort > 0
```

Stateless. The cap is symmetric and disabled when `max_effort == 0`.
For asymmetric caps or position-dependent caps, drop the in-kernel
cap (set `max_effort = 0` in the controller and bolt a
[`ClampingMaxEffort`](./pipeline.md#3-clamping) /
[`ClampingPositionBased`](./pipeline.md#3-clamping) on top via the
pipeline).

```rust
pub struct ControllerPdConfig {
    pub dof: u32,            // index into joint_q / joint_qd / joint_force
    pub kp: Real,            // N/m or N·m/rad
    pub kd: Real,            // N·s/m or N·m·s/rad
    pub max_effort: Real,    // 0 disables the cap
}
```

```rust
let pd = ControllerPd::new(&engine, &model, &[
    ControllerPdConfig { dof: 0, kp: 100.0, kd: 10.0, max_effort: 50.0 },
    ControllerPdConfig { dof: 1, kp: 100.0, kd: 10.0, max_effort: 50.0 },
])?;

// Per-step:
pd.step(&engine, &state, &mut control)?;
```

## PID: effort law + anti-windup

```text
err   = target_pos − q
err_d = target_vel − qd
i_t   = clamp(i_{t-1} + err · dt, −integral_max, +integral_max)
effort = kp · err + ki · i_t + kd · err_d
effort = clamp(effort, −max_effort, +max_effort)   if max_effort > 0
```

Stateful: each step updates the per-actuator integral accumulator
from the previous step's value. The kernel writes the new integral
into a ping-pong buffer that the orchestrator swaps after every
dispatch, avoiding a host round-trip.

```rust
pub struct ControllerPidConfig {
    pub dof: u32,
    pub kp: Real,
    pub ki: Real,
    pub kd: Real,
    pub integral_max: Real, // 0 disables the integral cap (anti-windup)
    pub max_effort: Real,   // 0 disables the effort cap
}
```

### Anti-windup

`integral_max` is the symmetric clamp on the integral accumulator
itself — applied before multiplication by `ki`. This is the simplest
form of anti-windup: it prevents the integrator from running away
during prolonged saturation (when the plant can't respond fast
enough to the requested effort). The alternative — back-calculation
or conditional integration — would buy slightly faster recovery from
saturation at the cost of extra plumbing; the symmetric clamp is
sufficient for the trajectory-tracking use cases that drive most
RL / teleop demos.

`ControllerPid::reset()` zeros both ping-pong sides of the integral
buffer — useful at episode boundaries.

```rust
let pid = ControllerPid::new(&engine, &model, &[
    ControllerPidConfig { dof: 0, kp: 100.0, ki: 5.0, kd: 10.0,
                          integral_max: 2.0, max_effort: 50.0 },
])?;

for _ in 0..n_steps {
    pid.step(&engine, &state, &mut control, dt)?;
    integrator.step(&engine, &model, &mut state, &control, dt)?;
}
```

## Single-DOF assumption

Both controllers read `state.joint_q[dof]` / `state.joint_qd[dof]`
and write `control.joint_force[dof]` directly — they assume `q` and
`qd` use the **same** index. This is true for revolute / prismatic
joints; **not** true for ball / free joints whose `q` carries
quaternion components while `qd` carries angular velocity (the two
arrays diverge in length).

Position-tracking PD on a quaternion isn't well-defined anyway, so
this is a soft restriction in practice — multi-DoF orientations
should be driven by an external orientation-error controller that
writes effort to the corresponding `qd` slots after computing the
torque from the quaternion error.

## Validation + DOF uniqueness

The constructors validate:

- `dof` is in range against `model.joint_dof_total`.
- No two entries share the same `dof` (would race in
  `joint_force[dof]`).

A failure surfaces as `Error::Solver` at construction; the
per-step kernel cannot fail because the validation already ran.

## Pipeline variants

`ControllerPipelinePd` and `ControllerPipelinePid` have the same
config types, the same effort math, and the same anti-windup
semantics. The only difference is the source of the target: they
read from upstream `Delay::out_pos()` / `out_vel()` buffers instead
of from `control.joint_target_*`. That makes them the natural drop-
in for the [composable pipeline](./pipeline.md); use the standalone
variants when there's no delay stage.

## Cross-references

- [`Composable actuator pipeline`](./pipeline.md) — the
  `Delay → Controller → Clamping` orchestrator that hosts the
  pipeline-flavoured variants.
- [`DC motor clamping`](./dc_motor.md) — replace the in-kernel
  symmetric cap with a four-quadrant DC-motor envelope.
- [`Neural controllers`](./neural.md) — MLP / LSTM controllers that
  share the per-step contract (write into `joint_force` via DOF
  indices, optional clamping bolt-on).
