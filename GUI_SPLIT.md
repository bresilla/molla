# GUI-split map — extracting the viewer chrome to an external crate

Audience: the maintainer doing the split. Scope decision (2026-07-03, INSPECTION.md):
GUI chrome (egui panels, buttons, sliders, controls) moves OUT; scene-level
visualization (gizmos, meshes, particle rendering, physics wiring) stays IN.
This document maps exactly where that cut runs and which seams will fight back.
All `file:line` references are against commit `1f3e02b`.

## The headline (it's not what the audit assumed)

- The **20 `fn tire_*_panel` functions are already clean**: every one takes only
  `ui: &mut egui::Ui` and renders precomputed `molla_vehicle::*_report()` structs.
  They port to the external crate unchanged (given a `molla_vehicle` dependency
  or, better, a data-provider handoff — see §5).
- The real GUI mass is **103 pane/panel functions spread across the 131 example
  files** (101 of 131 files touch egui). Each example file mixes its physics
  `update_*` system (stays) with its pane function (leaves). The split slices
  through every scene file, not just `main.rs`.
- Everything funnels through **one chokepoint**: `control_pane`
  (`bin/molla_viewer/src/main.rs:5231`), a Bevy system holding `ResMut<Sim>`
  plus ~12 other resources, which dispatches to the per-scene panes. Fixing the
  chokepoint's contract fixes most of the split.

## 1. HARD seams — GPU/solver access from UI code (fix BEFORE extraction)

These panes hold `&mut Sim` and reach through it to device handles, solver
internals, or host-mutable state. Each needs its logic reduced to a typed
command before the GUI can leave the repo:

| Pane | Where | What it does from UI code |
|---|---|---|
| `control_pane` | `main.rs:5231` | Calls `sim.reset()` (5407) → rebuilds model/state/solver/GPU buffers; hands `&mut sim` to the panes below (wiring ~5545–5640) |
| `cloth_collar_pane` | `examples/44_cloth_collar.rs:130` | Clones `sim.engine.device()/queue()`, `particle_q.ensure_device()` + `.upload()` — **GPU upload inline in a paint closure** |
| `mpm_pane` | `examples/24_mpm.rs:135,155` | `MpmSimulation::new(&sim.engine, …)` — **creates a GPU sim from UI**; assigns `sim.mpm` |
| `cable_pane` | `examples/23_cable.rs:193,202,207` | Mutates `sim.state_a.body_q/particle_q.host_mut()`; mutably matches `sim.solver` |
| `vbd_cloth_pane` | `examples/16_vbd_cloth.rs:166,199` | Mutates solver internals / calls `sim.reset()` |
| `softbody_drop_pane` | `examples/43_softbody_drop.rs:130` | Same class |
| `fast_soft_impact_pane` | `examples/45_fast_soft_impact.rs:120,125` | Same class |
| `soft_brake_pane` | `examples/46_soft_brake.rs:121` | Writes `sim.model.soft_surfaces[bi].friction` inline |

The hardest three are `cloth_collar_pane`, `mpm_pane`, `cable_pane` — they are
the only places where `sim.engine`, `sim.solver`, and `host_mut()` appear in
GUI code.

## 2. MEDIUM seams — physics executed inside paint closures

Five tire panels run real CPU simulation per frame inside egui closures (the
compute lives in `molla_vehicle`, so nothing leaves the repo, but the
*invocation* should move off the paint path):

- `tire_thermal_transient_soil_panel` (`main.rs:6435`) — runs
  `tire_rig_thermal_wear_cycle(...)` (6544) + ~10 report calls per frame.
- `tire_cavity_bead_sidewall_panel` (7070) — constructs `TetraTireFem::new`
  (7158) and runs FEM inline.
- `tire_fem_terrain_contact_panel` (6666) — ~17 heavy report calls (6681–7039).
- `tire_resolution_convergence_panel` (7282), `tire_force_shape_envelope_panel`
  (6234) — 6 report calls each.

Extraction rule: these become "render a `Report` struct" panels; the reports are
computed sim-side (cached, recomputed on parameter change, not per paint).

## 3. CLEAN — ports unchanged

- All 10 display-only tire report panels (`tire_gpu_parity_matrix_panel` 5850 …
  `tire_runtime_production_gate_panel` 7467).
- `scene_description_pane` (5754) + `tire_acceptance_checks_panel` (6418) —
  plain data + pane-local flags.
- The flag-style resources are already the right pattern (see §4): panel writes
  a flag, a sim-side system consumes it.

## 4. Shared state that must be split or interfaced

| Struct | Where | Who touches it |
|---|---|---|
| `Sim` | `main.rs:1181` | THE boundary object: model, state_a/b, solver, pipeline, engine (device/queue), mpm, particles. Panels mutate it; scene systems own it. Must become an interface. |
| `WarpExampleStates` | `main.rs:7579` | ~55 fields, one per warp example; mutated by BOTH warp panes and warp scene systems. Split per-scene into (params: GUI-writable) + (state: sim-owned). |
| `PaneSceneStates` / `SceneStates` | `main.rs:1115` / `1043` | SystemParam bundles of per-scene `*ExampleState` resources; same both-sides mutation pattern. Same split rule. |
| `SimStats` | `main.rs:1021` | Already clean: sim writes, panel reads. Template for the snapshot side. |
| `SimRunState` (628), `ActivityView` (3282), `PendingSceneSwitch` (3275), `LidarConfig` (3321), `SceneDescriptionPaneState` (634) | plain flags/config | Already the clean command pattern. Template for the command side. |

## 5. The interface: what the external GUI crate needs

Everything the GUI **reads** becomes a plain snapshot; everything it **writes**
becomes a command. Concretely, a `ViewerApi` needs four surfaces:

1. **Telemetry snapshot** (plain data, no handles): scene id + labels,
   `body_count`, `adapter_label`, `physics_audit()`, `SimStats` timings, FPS.
   Covers every `control_pane` readout.
2. **Command channel**: `switch_scene(Scene)`, `set_paused(bool)`,
   `step_once()`, `reset()`, `set_activity_view(bool)` — replacing `pending.0`,
   `run.*`, `activity.*`, and the direct `sim.reset()` call.
3. **Typed per-scene setters** replacing the eight `&mut Sim` panes:
   `set_soft_surface_friction(..)`, `set_solver_param(..)`, `reset_mpm(..)`,
   `upload_particles(..)` — so `engine`/`solver`/`host_mut` never appear in GUI
   code.
4. **Report provider** for the tire panels: sim side computes/caches the
   `molla_vehicle` report structs; GUI renders them (also fixes §2).

With that in place: the tire + display panels port unchanged, `control_pane`
drops `ResMut<Sim>` for snapshot + command sink, and the per-scene pane bodies
move out while their `update_*` physics systems stay.

## 6. Suggested order of attack

1. Introduce the command channel + snapshot structs INSIDE the current binary
   (no crate split yet); migrate `control_pane` onto them.
2. Convert the three hardest panes (`cloth_collar`, `mpm`, `cable`) to typed
   commands; then the remaining five HARD panes.
3. Move the §2 physics invocations to sim-side cached reports.
4. Split `WarpExampleStates` + per-scene state structs into params/state pairs.
5. Only THEN lift the (now dependency-free) pane functions into the external
   crate — at that point it's a mechanical move.

Steps 1–4 keep the binary working at every commit; step 5 is the actual split
and is small once the seams are cut.
