# MPM

molla ships **three flavours of Material Point Method**, each
specialised for a different regime:

| flavour | algorithm | rheology | timestep | rigid coupling | GPU | CPU oracle |
|---------|-----------|----------|----------|----------------|:---:|:----------:|
| [Explicit MPM](./mpm_explicit.md) | symplectic Euler P2G → grid update → G2P | sand (Drucker-Prager) | tiny (CFL-limited) | one-way (imprint) | ✓ | — |
| [Semi-implicit + snow](./mpm_semi_implicit.md) | Jacobi-preconditioned semi-implicit | sand DP, snow Stomakhin | 4×–6× larger dt | two-way wrench | ✓ | — |
| [Implicit Bingham](./mpm_implicit.md) | 27- / 125-point Newton + Bingham return mapping | Bingham (yield-stress fluid) | stiff-plastic safe | two-way (v3+) | ✓ (v2+) | ✓ (Stage 38.1 differentiable) |

The shared idea is the **Material Point Method** itself: a hybrid
Lagrangian–Eulerian scheme that scatters particle state onto a
background grid, runs the equations of motion on the grid (where they
are easy: regular topology, fast Poisson-like solves), then gathers
the updated state back to particles.

```text
            ┌────────────┐
            │ particles  │   positions, velocities, F, mass, …
            └─────┬──────┘
                  │ P2G — scatter via B-spline weights
                  ▼
       ┌─────────────────────┐
       │   background grid   │   m_i, mv_i, K_i, …
       │ (uniform 3D mesh)   │
       └────┬────────────────┘
            │ Grid update
            │   - momentum/mass → velocity
            │   - apply gravity, boundary
            │   - implicit grid solve (if stiff)
            │   - rigid imprint / 2-way coupling
            ▼
       ┌─────────────────────┐
       │  updated grid       │   v_i_new
       └────┬────────────────┘
            │ G2P — gather + advect particles
            ▼
            ┌────────────┐
            │ particles  │   new positions, velocities, F, plasticity
            └────────────┘
```

The three flavours differ in:

- **What gets scattered to the grid in P2G**: explicit ships only
  mass + momentum; semi-implicit adds a per-cell diagonal stiffness
  `k_diag`; implicit adds a full 27- or 125-point cell-coupled
  Hessian.
- **How the grid step is solved**: explicit just divides
  `momentum/mass`; semi-implicit runs N Jacobi sweeps with the
  per-cell `k_diag`; implicit runs colored Gauss-Seidel over the
  full stencil.
- **What return-mapping plasticity is applied at G2P**: sand uses
  Drucker-Prager; snow uses Stomakhin singular-value clamping;
  Bingham uses yield-stress projection on the deviatoric strain rate.

## Quadratic B-spline kernel

All three flavours use the same 3×3×3 quadratic B-spline scatter
kernel. Given a particle at world position `x_p` and a grid cell at
`x_i` (cell centre):

```text
ξ = (x_p − x_origin) / cell_size − 0.5
i0 = floor(ξ)                        (base cell)
For δ ∈ {0, 1, 2} per axis:
    w_axis[δ] = ¾ − (ξ − i0 − δ + 1)²       (quadratic B-spline)
w_total = w_x · w_y · w_z                   (separable weight)
```

Each particle scatters into the 3³ = 27 cells around `i0`. The grid
read-back in G2P uses the same 27 weights. Quadratic B-spline gives
`C²` continuity — smooth enough that the velocity / stress fields
on the grid don't introduce numerical noise into the particle update.

## APIC (Affine Particle-in-Cell)

The two more advanced flavours use **APIC** transfer: in addition to
the velocity, the particle carries an affine matrix `C_p` (3×3) such
that the velocity field around the particle is approximated as
`v(x) = v_p + C_p · (x − x_p)`. This:

- Conserves angular momentum exactly (FLIP / PIC don't).
- Damps PIC's grid-aliasing noise without the FLIP "ringing".
- Adds one matrix per particle to the P2G + G2P passes.

The trade-off is `C_p` storage and an extra 3×3 in the hot loop. molla
defaults to APIC for both the semi-implicit and implicit paths.

## Why three solvers?

|              | explicit      | semi-implicit | implicit Bingham |
|--------------|---------------|---------------|------------------|
| dt cap       | CFL `~ cell/v_sound` | 4–6× explicit | stiff-plastic safe |
| target use   | bulk granular flow at low stiffness | sand + snow + 2-way rigid | high-yield-stress fluids |
| GPU mature   | Stage 24d v1  | Stage 24d.1   | Stage 40 v2 / v3 |
| differentiable| —            | —             | Stage 38.1 (CPU only) |
| atomic perf  | Stage 24d.3 (4.66× speedup via workgroup-local two-pass P2G) | inherits | full stencil cost |
| host API     | `MpmSimulation::step` | `MpmSimulation::step` (Implicit time step) | `implicit_mpm_step` |

The explicit path is the simplest and most thoroughly profiled (Stage
24d.3 cut its per-step wall time by 4.66× via workgroup-local atomic
aggregation). The semi-implicit path is the production sweet spot for
sand + snow; the implicit Bingham path is the only one that handles
high-yield-stress slurries (cement, toothpaste, ketchup) without
exploding.

## Public API entry points

```rust,ignore
// Explicit + semi-implicit:
use molla::compute::{MpmSimulation, MpmConfig, MpmRheology, MpmTimeStep};

let mut cfg = MpmConfig::sand_calibrated();
cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 3 };
cfg.rheology  = MpmRheology::Sand { friction_angle: 30f32.to_radians() };
let mut sim = MpmSimulation::new(&engine, cfg)?;
sim.set_particles(queue, &positions)?;
for _ in 0..240 {
    sim.step_n(&engine, 1.0 / 240.0, 8)?;        // 8 substeps per outer
}
let wrenches = sim.read_rigid_wrenches(&engine)?;

// Implicit Bingham:
use molla::compute::implicit_mpm::{
    implicit_mpm_step, implicit_mpm_step_with_stencil,
    HessianStencil, MpmState, ImplicitMpmConfig, RigidSphere,
};

let mut state = MpmState::new(&positions, &velocities);
let report = implicit_mpm_step_with_stencil(
    &mut state, &cfg, &mut rigids,
    HessianStencil::Full27,
);
```

## Stage progression

The three solvers landed across many stages — the key milestones:

- **Stage 24d v1** — explicit MPM for sand (Drucker-Prager), GPU only.
- **Stage 24d.1** — semi-implicit time step + calibrated angle of
  repose + Stomakhin snow rheology + two-way rigid coupling.
- **Stage 24d.2** — persistent-buffer dispatch (re-use one wgpu
  bind group across substeps, eliminating per-step allocation).
- **Stage 24d.3** — atomic-contention reduction in P2G (workgroup-
  local two-pass scatter, 4.66× speedup at 50k particles + 64³ grid).
- **Stage 38.1** — CPU-only differentiable explicit MPM
  (`mpm_cpu_diff.rs`) for gradient-based parameter inference.
- **Stage 39A** — implicit MPM v1 with Bingham rheology + diagonal
  preconditioner.
- **Stage 40 v1** — full 27-point Newton Hessian.
- **Stage 40 v2** — GPU port of the colored-GS sweep.
- **Stage 40 v3** — GPU Bingham return mapping + two-way rigid.
- **Stage 40 v4** — 125-point (5×5×5) "strict-Newton" Hessian.

Per-flavour chapters dig into the algorithm details and the
configuration knobs.
