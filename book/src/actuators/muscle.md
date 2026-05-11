# Muscle (Hill-type + SEE)

Stage 42 added a biomechanical Hill-type muscle actuator that
drives [`CABLE` joints](../sim/joints_cable.md) — the joint type
molla provides for 1-DoF linear-stretch DOFs like tendons or
muscle-tendon units (MTUs). Together they let molla simulate
musculoskeletal models in the OpenSim / MyoSuite tradition without
leaving the GPU-first pipeline.

The muscle ships in four progressive cuts (v1 → v4):

- **v1** — Single muscle per joint, instantaneous activation, rigid
  tendon. Hill three-element model only.
- **v2** — Adds Thelen-style series-elastic tendon (SEE), first-order
  excitation → activation filter, and per-joint co-contraction via a
  CSR (multiple muscles summed at the solver level).
- **v3** — GPU kernel (`shaders/muscle.wgsl`) for the per-step torque
  computation on host-allocated parallel buffers.
- **v4** — SEE Newton solve runs on the GPU with full host parity.

Source: `crates/molla/src/actuators/muscle.rs`. Shader:
`crates/molla/shaders/muscle.wgsl`. The host-side Hill model is
always compiled (no GPU feature gate) so unit tests and the CPU
solver can use it directly.

## Three-element Hill model

Total path length of the MTU splits between the contractile fibre
(at pennation angle α) and the series-elastic tendon:

```text
l_total = l_m · cos(α) + l_t
```

In cable coordinates, `q_l = 0` means the MTU is at rest (fibre at
`l_opt`, tendon at slack). The active fibre force comes from the
Hill model:

```text
f_fibre = a · f_max · f_L(l_m / l_opt) · f_V(v_m / v_max)
```

with the tendon-side balance

```text
f_tendon(ε) = f_fibre · cos(α),    ε = (l_t − l_t_slack) / l_t_slack
```

## Force–length curve

Gaussian centred at `l_norm = 1.0`:

```text
f_L(l_norm) = exp( − ((l_norm − 1) / σ)² ),   σ = 0.45
```

Peaks at 1.0, falls to 0 outside `[0.4, 1.6]`. Implementation in
`force_length(l_norm)`; the choice of a Gaussian (rather than the
piecewise-polynomial Thelen curve) is for cheap GPU evaluation and
analytical differentiability — within the operating range it's
within a few percent of Thelen.

## Force–velocity curve (Hill hyperbola)

```text
            ⎧ 0,                              v_norm ≤ −1
f_V(v_norm) ⎨ (1 + v_norm) / (1 − v_norm/A_F), −1 < v_norm < 0
            ⎩ F_V_max − (F_V_max − 1) · e^{−K_V · v_norm}, v_norm ≥ 0
```

with `A_F = 0.25`, `F_V_max = 1.5`, `K_V = 5`. Returns 1.0 at
`v_norm = 0` (isometric), 0 at `v_norm = −1` (full shortening), and
asymptotes to `≈ 1.5` for large positive `v_norm` (eccentric
lengthening). The exponential lengthening branch is a smooth
saturating ramp instead of Thelen's two-piece formulation —
same reasoning as the force-length curve.

## Tendon force–strain curve

Quadratic past slack so the slope vanishes at `ε = 0` (`C¹`
continuous); zero below slack (tendons don't push):

```text
f_t(ε) = f_max · k · ε²,    ε > 0
f_t(ε) = 0,                 ε ≤ 0
```

`k = (stiffness_ratio / 2)²`. With Thelen's default
`stiffness_ratio = 30`, `k = 225`, the tendon reaches `f_max` at
`ε ≈ 6.7 %` — mid-range for real tendons. Matching the linear
tendon's `E·A = stiffness_ratio · f_max` slope at the
`f_t = f_max` strain pins the constant.

## Activation lag

The Hill curves take the *filtered* activation `a`, not the raw
excitation `u`. molla models the lag with a first-order ODE:

```text
da/dt = (u − a) / τ_act
```

`update_muscle_activation(model, dt)` integrates one step with
**implicit Euler**:

```text
a_{n+1} = (a_n + (dt / τ) · u) / (1 + dt / τ)
```

Implicit Euler is unconditionally stable for any `dt > 0` — there is
no CFL condition. `τ_act` defaults to 20 ms (`DEFAULT_TAU_ACTIVATION`,
matching Thelen 2003). Muscles with `τ_act ≤ 0` get an instantaneous
`a := clamp(u, 0, 1)` (v1 behaviour, useful for unit tests).

Activations are clamped to `[0, 1]` after each update. Call this
**before** the dynamics solver every step:

```rust
use molla::actuators::update_muscle_activation;
// 1. user writes new neural drives:
model.muscle_excitation.host_mut()?[i] = 0.6;
// 2. low-pass filter into activation:
update_muscle_activation(&mut model, dt)?;
// 3. step the solver — Featherstone reads model.muscle_activation
solver.step(...)?;
```

## SEE Newton solve

When `l_tendon_slack > 0`, `Muscle::torque(q_l, qd_l)` runs a
damped Newton solver on the tendon strain `ε` to balance the
fibre–tendon force equation:

```text
f_fibre(l_m(ε), v_m(ε)) · cos(α) − f_t(ε) = 0
```

The seed `ε₀` picks between the geometric strain (when the MTU is
stretched past rest) and the analytic-root strain `√(a · cos α / k)`
(when the MTU is at rest but the fibre is active). Up to 8
iterations; damped step size `≤ 0.5 · |ε|` prevents overshoot near
`l_m = l_opt` where `df_L/dε` vanishes. Convergence criterion
`|residual| < 1e-4 · f_max`. Typical step count: 2–3 inside the
operating envelope.

The rigid-tendon fast path (when `l_tendon_slack == 0`) skips the
Newton solve entirely and matches the v1 behaviour bit-for-bit —
useful as a regression oracle.

## Co-contraction (CSR)

Multiple muscles per joint are summed at the solver level via a CSR
laid out as parallel buffers on `Model`:

- `model.joint_muscle_offsets[j..j+1]` — start / end indices into
  the per-joint slice.
- `model.joint_muscle_indices[k]` — flat list of muscle ids
  attached to each joint.
- Per-muscle data lives in parallel
  `model.muscle_config`, `model.muscle_activation`,
  `model.muscle_excitation`, `model.muscle_tau_activation` etc.

Antagonist pairs are modelled with a per-muscle `polarity ∈ {+1,
−1}` flag that flips the fibre kinematics so the Hill curves see
the muscle's own coordinate frame. The torque returned at the joint
level is the sum of `m.torque(q_l, qd_l)` across the muscles in the
CSR slice; flexor + extensor pairs on the same joint cancel
naturally.

## GPU kernel (v3, v4)

`shaders/muscle.wgsl` runs the per-step torque computation in
parallel across all muscles. The kernel reads the
`Model::muscle_*` parallel buffers and writes per-muscle torque
contributions into a `joint_force` scatter buffer; the
Featherstone solver's torque-injection pass then sums them into the
joint-space tau vector.

v4 ships the SEE Newton solve on the GPU too, with full host
parity (verified by parity tests in `tests/muscle_gpu.rs`). The same
damped Newton kernel runs across all muscles in lockstep — no
per-muscle control divergence inside the 8 iterations, because every
muscle that converges early simply repeats its converged solution
for the remaining iterations.

## Configuration

```rust
pub struct MuscleConfig {
    pub f_max: Real,            // peak isometric force (N)
    pub l_opt: Real,            // optimal fibre length (m)
    pub v_max: Real,            // peak shortening velocity (m/s)
    pub activation: Real,       // a ∈ [0, 1] (initial filtered state)
    pub excitation: Real,       // u ∈ [0, 1] (initial neural drive)
    pub tau_activation: Real,   // τ_act (s), default 0.020
    pub l_tendon_slack: Real,   // 0 ⇒ rigid tendon, > 0 enables SEE
    pub pennation_angle: Real,  // α₀ at l_opt (rad), default 0
    pub polarity: Real,         // +1 / −1 (antagonist flag)
}
```

`Default::default()` gives a "medium" mammalian limb muscle: 100 N
peak isometric, 5 cm optimal fibre length, 0.5 m/s peak contraction,
rigid tendon (v1 compatibility).

## Cross-references

- [`Sim / Joints / CABLE joint`](../sim/joints_cable.md) — the only
  joint type a muscle can attach to (1-DoF linear stretch).
- [`Sim / Featherstone ABA`](../sim/featherstone.md) — torque-
  injection consumer of the per-muscle force.
- [`Composable actuator pipeline`](./pipeline.md) — currently
  *separate* from the muscle path (Stage 42 deliberate); muscles
  feed Featherstone directly without going through the
  `Delay → Controller → Clamping` chain.
- References: Zajac 1989 ("Muscle and Tendon: Properties, Models,
  Scaling…"), Thelen 2003 ("Adjustment of Muscle Mechanics Model
  Parameters to Simulate Dynamic Contractions in Older Adults").
