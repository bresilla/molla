# Viewer overview

`molla_viewer` is the Bevy 0.18 application that exercises every
solver, kernel, sensor, and FEM stepper molla ships. It is the
showroom and the integration test, all in one binary.

## Architecture in one diagram

```
┌──────────────────────────────────────────────────────────┐
│                molla_viewer (single Bevy app)            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Sim resource                                    │    │
│  │    - molla::Model      (current scene's model)   │    │
│  │    - molla::State      (positions / velocities)  │    │
│  │    - molla::Control    (joint targets, wrenches) │    │
│  │    - molla::Contacts   (collision output)        │    │
│  │    - Box<dyn Solver>   (active solver)           │    │
│  │    - compute::Engine   (wgpu device + queue)     │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  SceneStates SystemParam bundle                  │    │
│  │    - per-scene resources (~40 of them)           │    │
│  │  PaneSceneStates SystemParam bundle              │    │
│  │    - same resources, reborrowed for UI           │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Systems:                                                │
│    apply_scene_switch  - rebuild Sim on dropdown change  │
│    step_simulation     - dispatch to active Solver       │
│    update_<scene>      - per-scene custom update logic   │
│    sync_to_bevy        - copy molla state into bevy::Transform │
│    control_pane        - egui right-rail UI              │
│    diagnostic_pane     - bottom-left perf readout        │
│    handle_hotkeys      - keyboard shortcuts              │
└──────────────────────────────────────────────────────────┘
```

## How a scene works

Every scene in the viewer follows a four-part recipe:

1. **A variant in the `Scene` enum** (`Scene::Pile`, `Scene::Mpm`,
   `Scene::FemBeam`, ...). The enum currently has ~100 variants.
2. **A `build_scene` arm** in `Sim::build_scene(scene: Scene)` that
   constructs the `molla::Model` via `ModelBuilder` and (if the
   scene has its own particle / mesh / actuator state) initialises
   the corresponding per-scene resource.
3. **A label in `Scene::label()`** like `"01 Pile"` or
   `"47 FEM heat (Stage 36 v1)"`. The number prefix doubles as the
   dropdown ordering and the historical reference for the
   [scenes.md](./scenes.md) catalog.
4. **Optionally**: an `update_<scene>` system if the scene has
   custom step logic that needs to run outside the standard
   `step_simulation` dispatch. MPM, FEM, and the differentiable
   scenes all have their own update systems.

## The control flow

Each frame:

1. Bevy's runner calls `handle_hotkeys` — keyboard input may queue
   a scene switch into `PendingSceneSwitch`.
2. `apply_scene_switch` checks `PendingSceneSwitch`, and if non-empty,
   calls `Sim::build_scene(...)` to rebuild the model, picks the
   right solver, resets every per-scene resource through `SceneStates`,
   and clears the bevy world of the previous scene's gizmo entities.
3. `step_simulation` runs the active `Solver::step(...)` if the
   scene uses the standard pipeline. Scenes with their own update
   loop (`Scene::Mpm`, `Scene::FemHeat`, etc.) short-circuit and
   call their own `update_*` system instead.
4. `sync_to_bevy` copies `sim.state.body_q` into each tracked
   entity's `Transform`. Cloth / soft body / MPM scenes copy
   per-particle / per-vertex positions into a deforming Bevy
   `Mesh`.
5. `control_pane` renders the right-rail egui UI. Each scene's
   pane is one `match` arm reading + writing the corresponding
   `PaneSceneStates` field.
6. `diagnostic_pane` renders bottom-left perf data:
   - Active solver name + backend (CPU / GPU).
   - Frame time (ms).
   - Body count + contact count.
   - Optional Tracy frame markers when `--features profile`.

## The CPU↔GPU toggle

The diagnostic pane has a `[CPU] / [GPU]` radio button. Clicking
it sets `Sim::backend` and triggers an in-place solver rebuild:

```rust
match (current_solver_kind, backend) {
    (SolverKind::Xpbd, Backend::Cpu)  => XpbdSolver::new_cpu(),
    (SolverKind::Xpbd, Backend::Gpu)  => XpbdSolver::new_gpu(&engine)?,
    (SolverKind::Featherstone, Backend::Cpu) => FeatherstoneSolver::new_cpu(),
    (SolverKind::Featherstone, Backend::Gpu) => FeatherstoneSolver::new_gpu(&engine)?,
    (SolverKind::Vbd, Backend::Cpu)   => VbdSolver::new_cpu(),
    (SolverKind::Vbd, Backend::Gpu)   => VbdSolver::new_gpu(&engine)?,
    (SolverKind::MuJoCo, _)           => MuJoCoSolver::new(&model)?,
}
```

The simulation continues running while the toggle is flipped — no
window restart, no recompile. This is the load-bearing
demonstration of the [CPU↔GPU design rule](../architecture/design_rules.md).

## Visual conventions

- **Rigid bodies** render via Bevy `PbrBundle` with palette-coloured
  materials. A toggle key `T` switches to Muli-style activity
  shading: sleeping bodies turn grey.
- **Cloth / soft body / FEM mesh** renders as a single Bevy `Mesh`
  with per-frame vertex rewrites. Wireframe overlay is a toggle on
  most cloth / FEM panes.
- **Particles** (MPM, sparse-grid, autograd-particle scenes) render
  as gizmo points — Bevy's `Gizmos::sphere(...)` calls in the draw
  system.
- **Sensors** render as gizmo lines + arrows: lidar rays, IMU
  axis triads, contact normals, camera-ray bundles.
- **Articulated bodies** render with one bevy entity per body and
  one gizmo cylinder per joint axis.

## Hotkeys

| Key   | Action                                                         |
|-------|----------------------------------------------------------------|
| `Space` | Pause / resume                                                |
| `R`   | Reset current scene                                            |
| `T`   | Toggle Muli-style activity colouring                           |
| `Tab` | Toggle the control pane                                        |
| `Q`   | Switch to `Scene::Quadrotor`                                   |
| `V`   | Switch to `Scene::VbdCloth`                                    |
| `J`   | Switch to `Scene::MuJoCo`                                      |
| `G`   | Switch to `Scene::DiffArm`                                     |
| `H`   | Switch to `Scene::DiffArm6`                                    |
| `M`   | Switch to `Scene::MeshTools`                                   |
| `N`   | Switch to `Scene::NeuralPolicy`                                |
| `L`   | Switch to `Scene::NeuralLstm`                                  |
| `C`   | Switch to `Scene::Cable`                                       |
| `P`   | Switch to `Scene::Mpm`                                         |
| `Y`   | Switch to `Scene::Jelly`                                       |

(For the full list of scenes including ones without hotkeys, use
the dropdown in the top-left.)

## See also

- [`adapters/bevy.md`](../adapters/bevy.md) — Bevy plumbing
  patterns + the SystemParam ceiling.
- [`viewer/scenes.md`](./scenes.md) — full ~100-scene catalog.
- `crates/molla_viewer/src/main.rs` — the source.
