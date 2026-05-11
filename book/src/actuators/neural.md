# Neural controllers

molla ships two on-device neural-controller flavours that drop a
pre-trained policy into the actuator pipeline without ever
round-tripping the network outputs through the host:

- **`ControllerNeuralMlp`** (Stage 18.1) — fixed-architecture
  3-layer feed-forward MLP with `tanh` non-linearities.
- **`ControllerNeuralLstm`** (Stage 31) — single-cell LSTM forward
  pass + linear output projection, hidden + cell state carried on
  the device across substeps.

Both write directly into `Control.joint_force[dof]` and plug into
the [composable actuator pipeline](./pipeline.md) via the
`ControllerStage::NeuralMlp` / `ControllerStage::Lstm` variants,
so downstream clamping (effort cap, DC-motor envelope, etc.) sees
the policy's output just like any other controller's.

Source: `crates/molla/src/actuators/controller_neural_mlp.rs`,
`controller_neural_lstm.rs`. Shaders:
`shaders/actuator_neural_mlp.wgsl` + `actuator_neural_mlp_scatter.wgsl`,
`shaders/actuator_neural_lstm.wgsl`.

## MLP — `ControllerNeuralMlp`

Architecture (hard-coded in WGSL):

```text
y = W3 · tanh(W2 · tanh(W1 · x + b1) + b2) + b3
```

Configurable sizes via `ControllerNeuralMlpConfig` (`input_dim`,
`hidden_dim ≤ 256`, `output_dim`). All three dims are bumped to a
shared `MAX_HIDDEN` constant in the WGSL kernel; recompilation isn't
needed to switch dimensions, only `hidden_dim ≤ 256` is enforced.

```rust
pub struct ControllerNeuralMlpConfig {
    pub input_dim: u32,
    pub hidden_dim: u32,   // <= 256
    pub output_dim: u32,
}

pub struct MlpWeights<'a> {
    pub w1: &'a [Real], pub b1: &'a [Real],
    pub w2: &'a [Real], pub b2: &'a [Real],
    pub w3: &'a [Real], pub b3: &'a [Real],
}
```

Row-major layout:

| Slab | Shape                       | Length             |
|------|-----------------------------|--------------------|
| w1   | `[hidden_dim × input_dim]`  | `H · I` elements   |
| b1   | `[hidden_dim]`              | `H` elements       |
| w2   | `[hidden_dim × hidden_dim]` | `H · H` elements   |
| b2   | `[hidden_dim]`              | `H` elements       |
| w3   | `[output_dim × hidden_dim]` | `O · H` elements   |
| b3   | `[output_dim]`              | `O` elements       |

`controller.expected_weight_dims()` returns the exact tuple so
exporters can size their tensors without re-deriving the math.

### Per-step contract

The MLP forward pass runs as a single WGSL dispatch; the outputs
are scattered into `Control.joint_force[dof_indices[i]]` by a
second dispatch (`actuator_neural_mlp_scatter.wgsl`). The
controller owns:

- The `inputs` buffer (`input_dim` floats). Pushed each step by the
  caller via `set_inputs(queue, values)` — typical sources are
  proprioceptive sensors (joint angles, IMU readings) and task
  observations.
- The `outputs` buffer (`output_dim` floats). Downloadable via
  `download_outputs(...)` for diagnostic UIs; otherwise stays on the
  device and feeds the scatter pass directly.
- The six weight slabs (immutable after `set_weights`).
- The `dof_indices` map (`output[i] → joint_force[dof_indices[i]]`).
  Duplicate DOFs are rejected at construction to avoid scatter
  races.

```rust
let cfg = ControllerNeuralMlpConfig {
    input_dim: 24, hidden_dim: 128, output_dim: 12,
};
let dof_indices: Vec<u32> = (0..12).collect();
let mut mlp = ControllerNeuralMlp::new(&engine, cfg, &dof_indices)?;
mlp.set_weights(&device, &queue, &MlpWeights { w1, b1, w2, b2, w3, b3 })?;

for _ in 0..n_steps {
    let obs = build_observation(&state, &sensors);
    mlp.set_inputs(&queue, &obs)?;
    actuator.step(&engine, &state, &mut control, dt)?;   // runs forward + scatter
}
```

## LSTM — `ControllerNeuralLstm`

Single-cell recurrence with a linear output projection. The hidden
state `h_t` and cell state `c_t` live on the device across substeps,
so the controller can express stateful policies without any host
round-trip per step.

```text
i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)
f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)
g_t = tanh(W_g · x_t + U_g · h_{t-1} + b_g)
o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
y_t = W_y · h_t + b_y
```

`LstmWeights` lays out all 14 slabs (per gate: `w_g` input-projection
`H × I`, `u_g` recurrent-projection `H × H`, `b_g` bias `H` × four
gates, plus the output linear `w_y` `O × H` and `b_y` `O`).
`controller.expected_weight_dims()` returns the same shape as a
`LstmWeightDims` struct so callers can iterate / zip with named
fields.

### Hidden-state reset

`reset_hidden_state(...)` zeros both `h` and `c` on the device.
Call at episode boundaries — without it the policy effectively
"remembers" context from the previous trajectory, which is exactly
the wrong inductive bias for an off-policy RL replay buffer.

```rust
let cfg = ControllerNeuralLstmConfig {
    input_dim: 24, hidden_dim: 64, output_dim: 12,
};
let mut lstm = ControllerNeuralLstm::new(&engine, cfg, &dof_indices)?;
lstm.set_weights(&device, &queue, &LstmWeights { ..weights })?;

for episode in 0..n_episodes {
    lstm.reset_hidden_state(&engine)?;
    for _ in 0..n_steps {
        let obs = build_observation(&state, &sensors);
        lstm.set_inputs(&queue, &obs)?;
        actuator.step(&engine, &state, &mut control, dt)?;
    }
}
```

## Weight loading

Both controllers expect row-major weight slabs and validate the
total length at `set_weights` time. The mismatch error includes both
the expected and actual lengths so an exporter mismatch shows up at
load rather than at the first frame of inference.

Reasonable PyTorch / JAX export conventions:

- PyTorch `nn.Linear(I, H)` ships `w` as `[H, I]` already — flatten
  with `w.detach().cpu().numpy().flatten()` and pass straight into
  the slab.
- JAX / Flax flax-linen uses `(I, H)` — transpose before flattening.
- Recurrent state should always be initialised by the controller
  itself (`new` zeros `h_state` and `c_state`); don't try to load it
  from a checkpoint.

## Single-DoF + uniqueness

Same restriction as PD / PID: each `dof_indices[i]` must be unique
inside one controller (else the scatter writes race). The constructor
checks both bounds (`dof_indices[i] < model.joint_dof_total`) and
uniqueness; a duplicate produces a `Solver` error.

## Cross-references

- [`Composable actuator pipeline`](./pipeline.md) — wires neural
  controllers into the `Delay → Controller → Clamping` chain.
- [`PD / PID controllers`](./pd_pid.md) — same per-step contract,
  hand-tuned gains instead of learned weights.
- [`Differentiable physics / Tape-based autograd`](../autograd/tape.md)
  — train policies inside molla using the autograd tape (vs. PyTorch
  + export route).
