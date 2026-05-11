# Explicit MPM

The simplest of molla's three [MPM](./mpm.md) flavours: a **symplectic
Euler** scatter–update–gather cycle with no implicit solve. Stage 24d
v1 deliverable.

```rust,ignore
use molla::compute::{MpmSimulation, MpmConfig, MpmRheology, MpmTimeStep};

let cfg = MpmConfig {
    rheology: MpmRheology::Sand { friction_angle: 30f32.to_radians() },
    time_step: MpmTimeStep::Explicit,                   // ← key
    ..MpmConfig::default()
};
let mut sim = MpmSimulation::new(&engine, cfg)?;
sim.set_particles(queue, &positions)?;
sim.step(&engine, 1.0 / 480.0)?;                        // small dt — CFL-bound
```

## Pipeline (one substep)

```text
1. reset_grid        — clear mass, momentum, k_diag, v_prev atomics
2. p2g               — particle → grid scatter (mass, momentum + APIC term)
3. rigid_imprint     — stamp rigid-body velocities into overlapped cells
4. grid_update       — momentum / mass → velocity, apply gravity + boundary
5. mpm_rigid_wrench  — aggregate two-way coupling wrenches (Stage 24d.1)
6. g2p               — grid → particle gather; APIC affine matrix C; F update
7. plasticity        — SVD-based return mapping (sand DP)
```

### P2G — scatter with elastic stress

For each particle, scatter into its 3³ stencil:

```text
For δ in stencil(3³ around base_cell):
    w = w_x · w_y · w_z                 (quadratic B-spline)
    # Mass
    atomic_add(g_mass[cell], w · m_p)
    # Momentum (PIC + APIC + elastic stress contribution)
    P_p = pk1_stress(F_p, μ, λ)         (co-rotated SVK)
    Δm = w · (m_p · v_p − dt · vol · P · Fᵀ · ∇w)
    atomic_add(g_momentum[cell], Δm)
```

The Piola-Kirchhoff stress `P` lets the grid carry the elastic
restoring force from per-particle strain back to the velocity field.
The `∇w` term is the gradient of the B-spline weight — this is the
"weak form" stress propagation.

`atomic_add` here is one of two backends:

- **Legacy `p2g`** — global storage atomics directly (≈ 5 CAS per
  cell per particle: 1 mass + 3 momentum + 1 `k_diag`).
- **Stage 24d.3 `p2g_two_pass`** — workgroup-local 4×4×4 = 64-slot
  scratch (`var<workgroup>` atomic CAS), committed once at the end
  per non-empty slot. For spatially-clustered particles this reduces
  global atomic traffic by ~5–10×; the slot is direct-mapped from
  the lower 2 bits of the cell index per axis.

Both backends produce identical output (parity-tested); the two-pass
path is the default on Stage 24d.3+ but can be toggled via
`set_p2g_mode` for testing.

### Grid update

```text
For each cell with mass > 0:
    v_cell = momentum / mass + g · dt
    # Boundary: smooth ground penalty
    if cell.y < floor_y:
        v_cell.y = max(v_cell.y, 0)            (no-penetration)
        v_cell *= (1 − floor_friction)         (Coulomb damping)
```

The ground boundary is the only one shipped — for general rigid
collision the **rigid imprint** + **two-way wrench** kernels (Stage
24d.1) take over.

### G2P + APIC update

```text
For each particle:
    v_new = 0; C_new = 0
    For δ in stencil:
        v_new += w · g_velocity[cell]
        C_new += w · g_velocity[cell] ⊗ (cell_centre − x_p)
    # APIC affine matrix (4 / cell_size² factor is the quadratic correction)
    C_p = (4 / cell_size²) · C_new
    # Position update
    x_p += v_new · dt
    # Deformation gradient update (standard APIC MPM)
    F_p = (I + dt · C_p) · F_p
```

### Plasticity — Drucker-Prager (sand)

Per particle, after the elastic F update:

1. **SVD**: `F = U · Σ · Vᵀ`.
2. **Trial stress** in principal coords: `τ_trial = log(Σ)` (Hencky strain).
3. **Project onto DP cone**: if `‖dev(τ_trial)‖ > α · trace(τ_trial)`,
   pull back along the cone normal.
4. **Reconstruct**: `Σ_new = exp(τ_proj)`; `F_e = U · Σ_new · Vᵀ`.

The friction angle `φ` parameterises `α = sin(φ) · √(2/3)`. Stage
24d.1's `MpmConfig::sand_calibrated` ships parameters tuned so that
`φ = 30°` produces a measured pile slope of 30° ± 5° in a
column-collapse test.

## Configuration

```rust,ignore
pub struct MpmConfig {
    pub grid_size:        u32,        // cubic edge length in cells
    pub grid_origin:      [f32; 3],   // lower-left corner
    pub cell_size:        f32,
    pub gravity_y:        f32,
    pub youngs:           f32,        // Young's modulus
    pub poisson:          f32,
    pub dp_alpha:         f32,        // DP friction parameter (sin φ · √(2/3))
    pub cohesion:         f32,
    pub boundary_friction: f32,
    pub rheology:         MpmRheology,
    pub time_step:        MpmTimeStep,
    /* … */
}

pub enum MpmTimeStep {
    /// Stage 24d v1 — symplectic Euler. Hard CFL cap on dt.
    Explicit,
    /// Stage 24d.1 — Jacobi-preconditioned semi-implicit.
    Implicit { jacobi_iters: usize },
}
```

The explicit path's `dt` is hard-bounded by the CFL condition
`dt < cell_size / v_sound` with `v_sound = √((λ + 2μ)/ρ)`. For typical
`E = 5e4 Pa`, `cell_size = 0.05 m`, density `1500 kg/m³`, this is
`dt < 2.5 × 10⁻³ s` — about 1/400 of a second. For larger `dt`,
switch to semi-implicit.

## Rigid imprint (one-way coupling, v1)

The explicit pipeline supports a **one-way** rigid coupling at v1: the
`rigid_imprint` kernel stamps `v_rigid_p` (interpolated rigid body
velocity at the cell) into any cell whose centre lies within a
registered rigid sphere. This works for static / kinematic colliders
but doesn't transmit force back to the rigid body.

For two-way coupling, switch to the semi-implicit flavour with the
`mpm_rigid_force_integrate` kernel — see [Semi-implicit](./mpm_semi_implicit.md#two-way-rigid-coupling).

## Performance — Stage 24d.3 numbers

Measured on the dev laptop (debug build, single-threaded `cargo test`):

- **10k particles + 32³ grid**, `step_n(100)`: ~ 45 Hz unchanged by
  24d.3. At this scale the wgpu submit cost dominates over per-particle
  atomics; the workgroup-local trick doesn't help.
- **50k particles + 64³ grid**:
  - Legacy single-pass `p2g`: **7.9 Hz** (per-step ≈ 127 ms).
  - Stage 24d.3 two-pass: **36.9 Hz** (per-step ≈ 27 ms).
  - Speedup **4.66×**, hitting the ≥ 30 Hz DoD.

The remaining gap to 60 Hz at 50k particles requires the implicit-CG
path (Stage 24d.4 follow-up).

## When to choose explicit

- Bulk granular flow at low stiffness (`E < 5 × 10⁴ Pa`).
- Smoke-test / parity test for the more advanced flavours.
- Differentiable workloads (Stage 38.1's CPU forward closely mirrors
  this pipeline, minus plasticity).

For everything else — sand + 2-way rigid, snow, stiff sand —
[Semi-implicit](./mpm_semi_implicit.md) is the production path.
