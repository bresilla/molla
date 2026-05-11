# Semi-implicit MPM

The **production sweet spot** for granular flow in molla. Stage 24d.1
deliverable, building on the [Explicit MPM](./mpm_explicit.md) pipeline
with:

1. **Semi-implicit time step** — Jacobi-preconditioned grid relaxation
   that takes 4×–6× larger `dt` than CFL-bound explicit.
2. **Calibrated angle of repose** — `MpmConfig::sand_calibrated` tuned
   to 30° ± 5° in a column-collapse test.
3. **Snow rheology** (Stomakhin 2013) — multiplicative elastic-clamp
   plasticity with hardening on `log det F_p`.
4. **Two-way rigid coupling** — per-cell stress impulses aggregated
   into a 6-DoF wrench per registered rigid body.

```rust,ignore
use molla::compute::{MpmSimulation, MpmConfig, MpmRheology, MpmTimeStep};

let mut cfg = MpmConfig::sand_calibrated();
cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 3 };
cfg.rheology  = MpmRheology::Sand { friction_angle: 30f32.to_radians() };
// Or snow:
// cfg.rheology = MpmRheology::Snow {
//     theta_c: 2.5e-2, theta_s: 7.5e-3, hardening: 10.0,
// };
let mut sim = MpmSimulation::new(&engine, cfg)?;
sim.set_particles(queue, &positions)?;
for _ in 0..240 {
    sim.step(&engine, 1.0 / 240.0)?;            // 4× larger dt than explicit
}
let wrenches = sim.read_rigid_wrenches(&engine)?;
```

## Semi-implicit time step

The brief's scope-realism note ("if you hit a wall, ship semi-implicit")
guided the algorithmic choice: instead of running full matrix-free CG
over the grid Lagrangian (Klár-style implicit MPM, which is heavyweight),
ship a **Jacobi-preconditioned relaxation** with a per-cell scalar
stiffness estimate `k_diag`.

Each iteration applies

```text
v_new[i] = (m[i] · v_explicit[i] + dt² · k_diag[i] · v_avg[i]) /
           (m[i] + dt² · k_diag[i])
```

where `v_avg[i]` is the **6-neighbour grid average**:

```text
v_avg[i] = (v[i−x] + v[i+x] + v[i−y] + v[i+y] + v[i−z] + v[i+z]) / 6
```

`k_diag[i]` is a particle-scattered estimate of the stress Hessian:
each particle deposits `m_p · ‖∇w‖² · k_p` where `k_p` is the
per-particle bulk modulus. The end-of-relaxation velocity is the
fixed point of the linear system `(M + dt² · K_diag) · v_new = M · v_explicit`.

**Convergence**: weaker than Klár's per-iteration accuracy because
the off-diagonal couplings are ignored, but the **stability gain** —
4×–6× larger `dt` allowed without explicit-CFL blow-up — is the same.
Default `jacobi_iters = 3` is enough for stiff sand; 1 (the
absolute minimum) collapses to explicit; 10+ approaches the full
implicit linear-system solve.

For full-Hessian implicit (off-diagonal stencil), see
[Implicit MPM with Bingham](./mpm_implicit.md).

## Snow rheology

Stomakhin 2013 §6.2 multiplicative split: `F = F_e · F_p` with
**elastic singular values clamped** to `[1 − θ_c, 1 + θ_s]`:

```text
SVD: F = U · Σ · Vᵀ
For each σ_i in Σ:
    σ_clamped_i = clamp(σ_i, 1 − θ_c, 1 + θ_s)
Σ_clamped = diag(σ_clamped_i)
F_e_new = U · Σ_clamped · Vᵀ
F_p_new = V · Σ_clamped⁻¹ · Σ · Vᵀ · F_p_old
```

`log det F_p` rolls into a scalar `jp` that exp-hardens μ and λ:

```text
μ_eff = μ · exp(hardening · (1 − jp))
λ_eff = λ · exp(hardening · (1 − jp))
```

The clamping captures snow's brittle behaviour (cracking when
`σ < 1 − θ_c`) and dense packing (resistance when `σ > 1 + θ_s`).
There's **no friction angle** — snow is parameterised by the trio
`(θ_c, θ_s, hardening)`:

| param | semantics |
|-------|-----------|
| `θ_c` | compression threshold (typical 2.5e-2 for soft snow) |
| `θ_s` | stretch threshold (typical 7.5e-3) |
| `hardening` | exponential bulk-modulus growth (10.0 default) |

## Sand calibration (Stage 24d.1)

Stage 24d v1's pile test was qualitative — pour particles, get a
pile, eyeball the slope. Stage 24d.1 ships `MpmConfig::sand_calibrated`
parameters tuned so a `friction_angle = 30°` parameter produces a
**measured pile slope of 30° ± 5°** in a column-collapse test:

```rust,ignore
pub fn sand_calibrated() -> Self {
    Self {
        grid_size: 64,
        cell_size: 0.05,
        gravity_y: -9.81,
        youngs: 5.0e4,                  // ≥ 5e4 to hold pile shape
        poisson: 0.3,
        // sand: DP cone parameters
        cohesion: 0.0,
        boundary_friction: 0.3,
        time_step: MpmTimeStep::Implicit { jacobi_iters: 3 },
        ..Default::default()
    }
}
```

Calibration knobs that mattered most:

- **Particle volume / cell volume ≈ 1 / (4–8)** — under-resolved cells
  produce a soft pile that flattens too far.
- **Young's modulus ≥ 5e4 Pa** — the elastic stress needs to hold
  against gravity; 1 kPa is too soft.
- **dt small enough that the elastic step doesn't oscillate** — 1/480 s
  comfortable with `E = 5e4 Pa`.

The test (`tests/mpm_v3_gpu.rs::mpm_calibrated_aor_30deg`) is gated
to a tolerance of ±5° on the measured slope.

## Two-way rigid coupling

Stage 24d.1 added a **`mpm_rigid_force_integrate` kernel** that
aggregates per-cell stress impulses (`m_cell · w · (v_rigid − v_grid)`)
for cells inside or near each rigid sphere into a 6-DoF wrench per
rigid body:

```text
For each rigid body r:
    F_r = 0; τ_r = 0
    For each grid cell within rigid_r's sphere:
        impulse = m_cell · (v_rigid_at_cell − v_grid_cell)
        F_r += impulse / dt
        τ_r += (cell_centre − rigid_r.com) × (impulse / dt)
    register wrench (F_r, τ_r) on rigid_r
```

The host reads back via [`MpmSimulation::read_rigid_wrenches`](https://docs.rs/molla/latest/molla/compute/struct.MpmSimulation.html#method.read_rigid_wrenches):

```rust,ignore
let wrenches: Vec<SpatialVector> = sim.read_rigid_wrenches(&engine)?;
for (i, w) in wrenches.iter().enumerate() {
    rigid_state.body_f.host_mut()?[i] = w.clone();
}
```

The wrench can be applied to a [SemiImplicit](./semi_implicit.md) or
[XPBD](./xpbd.md) rigid-body step in the same simulation tick, giving
proper **two-way** coupling: the rigid body affects the particle flow,
and the particles push back on the rigid body. The Stage 24d.1 demo
shows a sphere bouncing in a sand pit with realistic crater formation
and bounce response.

The 1-cell `rigid_imprint` from Stage 24d v1 remains as a fallback for
purely kinematic colliders that don't need force feedback.

## Atomic-contention reduction (Stage 24d.3)

The P2G scatter is the bottleneck — every particle issues ≥ 5 global
storage atomics per cell, and with quadratic B-splines that's 5 × 27 =
135 atomics per particle per step. At 50k particles + 60 substeps
that's 405 M global atomics per second.

Stage 24d.3 added a **workgroup-local two-pass P2G** (`p2g_two_pass`):

```text
Pass 1 (within workgroup): each thread atomic-adds into a 4×4×4 = 64-slot
        workgroup-local scratch (var<workgroup>).
        Slot = lower 2 bits of cell index per axis (direct map).
Pass 2 (per workgroup):    each non-empty slot is committed once to
        global storage via a single global CAS.
```

For spatially-clustered particles (the common MPM granular case),
consecutive particles in a workgroup share slots and the workgroup-
local CAS goes through L1 / scratchpad memory (no L2 round-trip).
Result: **4.66× speedup** at 50k particles + 64³ grid (7.9 → 36.9 Hz),
hitting the ≥ 30 Hz Stage 24d.3 DoD.

Hash collisions (different cells mapping to the same slot) fall back
to the legacy global-atomic path; correctness is preserved either way.

## Persistent-buffer dispatch (Stage 24d.2)

`MpmSimulation::step_n(engine, dt, n_substeps)` batches **N substeps
into a single command encoder + single `queue.submit`**, sharing
pre-allocated uniform + readback staging buffers. The legacy
`MpmSimulation::step` is now a back-compat wrapper that calls
`step_n(_, _, 1)`. The viewer + perf tests use the batched path;
`tests/mpm_perf.rs::mpm_step_n_beats_individual_steps` proves the
**9.1× speedup at 60 substeps** (well above the 2× DoD).

The mechanism is simple: wgpu submit overhead is ~100 µs per call,
which dwarfs MPM substep time for small problem sizes. Batching
amortises that cost.

## Configuration recap

```rust,ignore
let cfg = MpmConfig {
    grid_size: 64,
    cell_size: 0.05,
    grid_origin: [-1.5, 0.0, -1.5],
    gravity_y: -9.81,
    youngs: 5.0e4,
    poisson: 0.3,
    dp_alpha: 0.367,                            // sin(30°) · √(2/3)
    cohesion: 0.0,
    boundary_friction: 0.3,
    rheology: MpmRheology::Sand { friction_angle: 30f32.to_radians() },
    time_step: MpmTimeStep::Implicit { jacobi_iters: 3 },
    snow_theta_c: 2.5e-2,                        // ignored for sand
    snow_theta_s: 7.5e-3,
    snow_hardening: 10.0,
};
```

When to choose semi-implicit over explicit:

- **`dt > 1/240 s`** — explicit CFL fails.
- **Snow rheology** — explicit doesn't ship Stomakhin.
- **Two-way rigid coupling** — the wrench aggregation kernel only
  fires in the semi-implicit path.

For high-yield-stress fluids (cement, toothpaste) with stiff plastic
yielding, switch to [Implicit MPM with Bingham](./mpm_implicit.md).
