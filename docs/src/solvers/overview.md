# Solvers overview

molla ships **eight** time-integration solvers, all implementing the
common [`Solver`](https://docs.rs/molla/latest/molla/solvers/trait.Solver.html)
trait so callers can swap integrators without rewiring the host loop:

```rust,ignore
pub trait Solver: Send + Sync {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()>;
}
```

The choice of solver is driven by the **simulation regime** (rigid /
soft / granular / fluid) and the **stiffness / contact pattern**.
The matrix below summarises coverage; chapters below dig into each.

## Coverage matrix

| solver | rigid bodies | particles / cloth | tetrahedral | cable / rope | MPM material | contacts | joints | CPU | GPU | autograd |
|--------|:------------:|:-----------------:|:-----------:|:------------:|:------------:|:--------:|:------:|:---:|:---:|:--------:|
| [`SemiImplicitSolver`](./semi_implicit.md)   | ✓ free bodies | — | — | — | — | — | — | ✓ | ✓ | — |
| [`FeatherstoneSolver`](../sim/featherstone.md) | ✓ articulated | — | — | ✓ joint | — | — | ✓ all types | ✓ | ✓ | ✓ revolute chains |
| [`XpbdSolver`](./xpbd.md)         | ✓ contacts + equality | ✓ particles + distance | — | ✓ polyline | — | ✓ rigid + soft | ✓ via equality | ✓ | ✓ | ✓ (Stage 23.x) |
| [`VbdSolver`](./vbd.md)           | ✓ kinematic only | ✓ cloth | ✓ Neo-Hookean | ✓ stretch + bend | — | ✓ soft–soft + soft–rigid | — | ✓ | ✓ | ✓ |
| [`MpmSimulation`](./mpm.md) (explicit) | one-way coupling | — | — | — | ✓ sand / snow | particle-grid floor | — | — | ✓ | — |
| [`MpmSimulation`](./mpm_semi_implicit.md) (semi-implicit + two-way rigid) | ✓ two-way | — | — | — | ✓ sand / snow | particle-grid + spheres | — | — | ✓ | — |
| [`ImplicitMpm`](./mpm_implicit.md) (Bingham) | ✓ two-way (v3+) | — | — | — | ✓ Bingham | particle-grid + spheres | — | ✓ | ✓ (v2+) | Stage 38.1 (CPU) |
| [`MuJoCoSolver`](./mujoco.md)     | ✓ via MuJoCo | — | — | — | — | ✓ MuJoCo PADMM | ✓ all types | ✓ (CPU) | — | — |

Legend: ✓ = supported, — = not applicable / out of scope.

## Picking a solver

```text
                    ┌──── 1 body, free-floating ──────► SemiImplicit
                    │
                    ├──── articulated, no contact ─────► Featherstone
                    │
 rigid    ──────────┤──── stacked / contact ──────────► XPBD (Jacobi)
                    │
                    └──── benchmark vs reference ─────► MuJoCo
 ───────────────────┤
                    ├──── cloth / soft membrane ──────► XPBD or VBD
                    │       (stiffness > 1e6: VBD)
                    │
 soft     ──────────┤──── tetrahedral solids ─────────► VBD
                    │
                    └──── cable / rope / muscle ──────► VBD (polyline) or
                                                        Featherstone CABLE
 ───────────────────┤
                    ├──── sand / snow / debris ───────► MPM (explicit)
                    │     (need 2-way w/ rigids)        MPM (semi-implicit)
 granular ──────────┤
                    └──── high-yield-stress slurry ───► Implicit MPM (Bingham)
                          (cement, toothpaste, …)
```

## XPBD vs VBD — when to use which for soft bodies

Both XPBD and VBD solve the implicit-Euler soft-body step on the same
particle / spring graph (and the same per-vertex graph colouring).
Choose between them by the **stiffness regime**:

|                 | XPBD                          | VBD                                       |
|-----------------|-------------------------------|-------------------------------------------|
| Algorithm       | per-constraint position projection (Gauss-Seidel) | per-vertex 3×3 Newton block (per-vertex Gauss-Seidel) |
| Stiffness scale | non-physical (more iters = stiffer) | **physical** (true implicit-Euler step)   |
| Convergence with stiffness | slows linearly | independent (block Newton)                |
| Rigid coupling  | rigid-via-equality, particle plane only | AVBD soft / hard (Stage 20 + 33)          |
| Tet support     | particles + springs only      | Neo-Hookean + Mooney–Rivlin tets + plasticity |
| GPU port        | mature (Stage 4–14)           | mature (Stage 20.x + 33.x)                |
| Differentiable  | Stage 23.x                    | Stage 23.x                                |

Rule of thumb: cloth with spring stiffness `< 1 × 10⁶ N/m` → XPBD is
cheaper. Above that → VBD. Tet soft bodies → VBD. Granular → MPM.

## Sub-stepping pattern

Every solver supports an internal `with_substeps(n)` knob:

```rust,ignore
let solver = XpbdSolver::new_cpu().with_substeps(4);
```

This advances `n` internal substeps of `dt/n` per outer `step` call,
re-using the caller-supplied `contacts` set across all substeps. For
contact-heavy workloads the Müller (2020) "many small substeps × few
iterations" stability win typically needs `pipeline.collide` re-runs
between substeps, in which case the **external substep loop** is the
right pattern:

```rust,ignore
for _ in 0..n_sub {
    pipeline.collide(&mut contacts, &state, &model)?;
    solver.step(&model, &state, &control, &contacts, dt / n_sub as f64, &mut state_out)?;
    std::mem::swap(&mut state, &mut state_out);
}
```

The internal `with_substeps` is most useful for contact-free
workloads (Featherstone, particle-only XPBD, MPM).

## Convention recap

All eight solvers obey the same `(state_in, state_out)` ping-pong
contract:

1. The caller pre-allocates `state_out` (`model.state()`).
2. The solver reads `state_in`, the model, the per-step `control`, and
   the collision-pipeline `contacts`, and writes the next-step state
   into `state_out`.
3. The two state buffers must be **distinct** — solvers may hold
   borrows of both for the call duration.

State double-buffering is the standard host-side pattern; users
typically allocate two `State` instances and `std::mem::swap` between
calls.
