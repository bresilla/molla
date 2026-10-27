# Implicit MPM (Bingham)

The third [MPM](./mpm.md) flavour — molla's **implicit, full-Hessian
Newton step with Bingham yield-stress rheology**. Stage 39A landed
the v1 diagonal preconditioner; Stage 40 v1–v4 progressively built
out the full 27-point Newton Hessian, GPU port, and 125-point
stencil. Used for **high-yield-stress fluid flow** — cement, ketchup,
toothpaste, mud — where the material is rigid below a stress
threshold and Newtonian-viscous above it.

Reference: Yue (2015), Daviet (2016). The molla implementation
mirrors Newton's `implicit_mpm` solver structure (forward grid step →
assemble linear system → colored Gauss-Seidel solve → Bingham
projection).

```rust,ignore
use molla::compute::implicit_mpm::{
    implicit_mpm_step_with_stencil,
    HessianStencil, MpmState, ImplicitMpmConfig, RigidSphere,
};

let cfg = ImplicitMpmConfig {
    grid_size: 32,
    cell_size: 0.05,
    grid_origin: [-0.8, 0.0, -0.8],
    dt: 1.0 / 60.0,                                   // stiff-plastic safe
    gravity_y: -9.81,
    yield_stress: 100.0,                              // τ_y (Pa)
    viscosity: 0.5,                                   // μ_b (Pa·s)
    density: 1200.0,
    particle_volume: 5.0e-5,
    floor_y: 0.0,
    floor_friction: 0.4,
    max_gs_iters: 16,
    gs_tol: 1.0e-3,
    max_newton_iters: 1,
};

let mut state = MpmState::new(&positions, &velocities);
let mut rigids: Vec<RigidSphere> = vec![];
let report = implicit_mpm_step_with_stencil(
    &mut state, &cfg, &mut rigids,
    HessianStencil::Full27,
);
```

## The Bingham flow rule

The Bingham model is the canonical yield-stress fluid: rigid below
`τ_y`, viscous above it.

```text
‖τ_dev‖ ≤ τ_y       ⟹   rigid       (no flow)
‖τ_dev‖ >  τ_y      ⟹   τ_dev = (τ_y + μ_b · γ̇) · γ̇_hat
```

In words: the deviatoric stress equals the yield stress plus a
Newtonian-viscous response proportional to the **strain-rate**
magnitude. When the trial stress falls below `τ_y` the material
freezes into a rigid plug; above `τ_y` it flows.

## Algorithm (one substep)

```text
1. P2G — scatter (mass, momentum, K-stencil) onto the grid.
2. Predict velocity: v_pred = momentum/m + g · dt.
3. Implicit grid solve: (M + dt · K) · v_new = M · v_pred
   via colored Gauss-Seidel sweeps.
4. Bingham projection at each particle:
   - Compute D = sym(C) − (tr/3)·I from APIC affine matrix.
   - If 2μ_b · ‖D‖_F ≤ τ_y → set D = 0 (rigid plug).
   - Else → scale D by (1 − τ_y / (2μ_b · ‖D‖_F)).
5. G2P + advect: gather grid velocity → particle, recompute APIC C,
   advance position by v_new · dt.
6. Two-way rigid coupling (Stage 40 v3+): accumulate cell-by-cell
   impulse `m_cell · w · (v_rigid − v_grid)` per RigidSphere, update
   the sphere's linear velocity.
```

## v1 → v4 progression

The Stage 40 v1–v4 sequence built the implicit grid solve up from
"diagonal preconditioner" to "full strict-Newton Hessian":

### v1 — Stage 39A: diagonal Hessian

```rust,ignore
HessianStencil::Diagonal
```

```text
(M + dt · K_diag) · v_new = M · v_pred
```

Per-cell scalar `K_diag[i]` (no inter-cell coupling). Same algebra as
the [semi-implicit Jacobi flavour](./mpm_semi_implicit.md), just with
Bingham return mapping at G2P instead of Drucker-Prager.

Stability win over the explicit timestep: `dt` can be 10× larger
without explicit-CFL blow-up. Convergence per GS sweep: slow on
stiff yield-stress configurations because the diagonal preconditioner
under-counts the off-diagonal couplings.

### v2 — Stage 40 v1: full 27-point Newton Hessian

```rust,ignore
HessianStencil::Full27
```

```text
(M + dt · K) · v_new = M · v_pred
```

`K` is the **full 27-point velocity-domain Hessian**: each grid cell
is coupled to its 26 immediate neighbours (3³ stencil centred on the
cell) plus itself. Off-diagonal entries `K_ij` for `i ≠ j`
accumulate

```text
K_ij = Σ_p k_p · w_ip · w_jp
```

over every particle that scatters onto both cell `i` and cell `j`.
The diagonal slot reduces to v1's `K_diag`.

The implicit grid solve is **colored Gauss-Seidel** on this stencil.
The 27-point stencil produces an **8-coloring** on the parity of
`(ix mod 2, iy mod 2, iz mod 2)`; within a colour, no two cells share
a stencil entry, so the sweep is race-free in WGSL. Per-cell update:

```text
v_i ← (M_i · v_pred_i − dt · Σ_{j ≠ i} K_ij · v_j) / (M_i + dt · K_ii)
```

Convergence per sweep is ~5× tighter than the diagonal v1 on stiff
Bingham configurations.

### Stage 40 v2: GPU port of the colored-GS sweep

The host-side colored-GS sweep was ported to a WGSL kernel
([`implicit_mpm.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/implicit_mpm.wgsl)).
One workgroup per colour class, parallel within colour. The
`implicit_mpm_step_with_solver` hook exposes a closure
`(GridSystem, &mut [Vec3], HessianStencil) -> (iters, residual)`
where the GPU sweep plugs in:

```rust,ignore
let mut gpu_sweep = |sys: &GridSystem, v: &mut [Vec3], stencil: HessianStencil|
    -> (usize, f64) {
    gpu_pipeline.sweep(sys, v, stencil)
};
implicit_mpm_step_with_solver(&mut state, &cfg, &mut rigids,
    HessianStencil::Full27, Some(&mut gpu_sweep));
```

The host pre-builds the `GridSystem` (mass, momentum, K-stencil),
the GPU runs the GS sweeps to convergence, and the host runs G2P +
Bingham projection + advect.

### Stage 40 v3: GPU Bingham return mapping + two-way rigid

The Bingham per-particle projection was moved into the GPU kernel
(previously CPU-only). Two-way rigid coupling — sphere wrenches +
sphere velocity update — also moved onto the device.

### Stage 40 v4: 125-point (5×5×5) "strict-Newton" Hessian

```rust,ignore
HessianStencil::Full125
```

v1/v2/v3 shipped the 27-point Hessian (cells coupled to their 26
immediate neighbours). v4 extends the stencil to the **true Newton
reach** for the quadratic-B-spline kernel: two cells whose 3×3×3
particle support boxes overlap can share a particle whenever their
index offset is at most ±2 on every axis, so the strict Hessian is
**5×5×5 = 125-point**.

v4 keeps the 27-point path as the default (a 5× cheaper assembly) but
exposes `HessianStencil::Full125` for callers that need tighter
convergence per GS iteration on stiff Bingham configurations. The
125-point coloring uses parity on `(ix mod 5, iy mod 5, iz mod 5)`
which produces 125 colour classes; the per-class parallelism is the
same as the 27-point version, just with more classes per sweep.

## Configuration

```rust,ignore
pub struct ImplicitMpmConfig {
    pub grid_size: usize,
    pub cell_size: Real,
    pub grid_origin: [Real; 3],
    pub dt: Real,
    pub gravity_y: Real,
    pub yield_stress: Real,           // τ_y (Pa)
    pub viscosity: Real,              // μ_b (Pa·s)
    pub density: Real,
    pub particle_volume: Real,
    pub floor_y: Real,
    pub floor_friction: Real,
    pub max_gs_iters: usize,          // GS sweep cap (default 16)
    pub gs_tol: Real,                 // rel. ‖Δv‖ tolerance (default 1e-3)
    pub max_newton_iters: usize,      // outer Newton (v1: 1)
}

pub enum HessianStencil {
    Diagonal,                         // Stage 39A v1
    Full27,                           // Stage 40 default
    Full125,                          // Stage 40 v4
}
```

Per-substep [`StepReport`](https://docs.rs/molla/latest/molla/compute/implicit_mpm/struct.StepReport.html)
returns the actual `gs_iters` consumed and the final residual norm
so callers can verify convergence.

## When to choose implicit Bingham

|                                | use implicit Bingham? |
|--------------------------------|:---------------------:|
| Cement, mortar, mud, ketchup, toothpaste | ✓ |
| Sand at very high yield stress | ✓ (Bingham approximates DP) |
| Granular flow at low stiffness | use [Explicit](./mpm_explicit.md) or [Semi-implicit](./mpm_semi_implicit.md) |
| Snow                           | use [Semi-implicit](./mpm_semi_implicit.md) (Stomakhin) |
| Differentiable parameter inference | use [Stage 38.1 CPU explicit](./mpm.md) (Bingham diff TBD) |

The implicit path is **the most expensive** of the three by
assembly + solve cost, but it's the **only** one that doesn't blow up
on stiff plastic regimes. Stage 40 v3 hit the ~ 60 Hz target on
~ 5k particles + 32³ grid; bigger problems need the 125-point or a
geometric multigrid pre-conditioner (deferred).

## Out of scope for v1

- **Drucker-Prager / snow** — single rheology (Bingham). The
  semi-implicit flavour covers the other two.
- **Adjoint** — Stage 38.1 ships the differentiable explicit MPM on
  CPU; the implicit Bingham adjoint is a follow-up.
- **Multigrid** — current solver is colored Gauss-Seidel; geometric
  multigrid (V-cycle / W-cycle over the grid hierarchy) is the next
  performance lever.
