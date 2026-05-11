# Bevy adapter

`molla_viewer` is a Bevy 0.18 application. It is the only crate in
the workspace that depends on Bevy — the engine crate has zero
Bevy footprint. This chapter documents the Bevy integration patterns
the viewer uses; they are the same patterns you would use to embed
molla in your own Bevy app.

## What the viewer pulls in

From `crates/molla_viewer/Cargo.toml`:

- `bevy = "0.18"` with a curated feature list — ECS, render, PBR,
  scene, picking, gizmos, asset, scene, X11, Wayland.
- `bevy_egui = "0.39"` for the immediate-mode UI.
- [`bevy_frost`](https://github.com/bresilla/bevy_frost) — egui-themed
  glass UI kit.
- [`bevy_glacial`](https://github.com/bresilla/bevy_glacial) — orbit
  camera + grid + gizmos plugin.
- `mimalloc` as the global allocator — egui's tessellator
  thrashes the system allocator and mimalloc cuts frame time from
  ~120 ms to ~13 ms in the heavy demos.

## The ECS model

molla state lives in Bevy **resources**, not components. Two
top-level singletons:

- **`Sim`** — owns the `molla::Model`, the current `State`, the
  `Control`, the `Contacts`, the active solver, and the
  `compute::Engine`. Created at startup; rebuilt every time the
  user picks a new scene from the dropdown.
- **`SimStats`** — wall-clock frame timing, solver telemetry,
  contact counts. Updated each frame; consumed by the diagnostic
  pane.

Per-scene runtime state (target sliders, accumulated time, MPM
particle counts, the IK solver instance, etc.) lives in **per-scene
resource types**:

```rust
#[derive(Resource, Default)] struct LidarState { ... }
#[derive(Resource, Default)] struct ImuExampleState { ... }
#[derive(Resource, Default)] struct ContactExampleState { ... }
// ... ~40 of these total
```

Each scene has at most two of these: one for the simulation
update logic, one for the right-rail UI pane.

## The SystemParam 16-tuple ceiling

Bevy's `SystemParam` derive macro has a **hard 16-element ceiling**
on tuple length. Both `apply_scene_switch` (the system that runs
on scene change to reset per-scene state) and `control_pane` (the
system that draws the right-rail UI) need access to ~40 resources
at once.

The viewer's solution is to bundle per-scene resources into named
`SystemParam` structs:

```rust
#[derive(bevy::ecs::system::SystemParam)]
struct SceneStates<'w> {
    lidar: ResMut<'w, LidarState>,
    imu: ResMut<'w, ImuExampleState>,
    contact: ResMut<'w, ContactExampleState>,
    arm: ResMut<'w, ArmExampleState>,
    actuator: ResMut<'w, ActuatorExampleState>,
    gpu_ik: ResMut<'w, GpuIkArmState>,
    camera: ResMut<'w, CameraExampleState>,
    stats: ResMut<'w, SimStats>,
    gpu_np: ResMut<'w, GpuNpExampleState>,
    gpu_bp: ResMut<'w, GpuBpExampleState>,
    quad: ResMut<'w, QuadrotorState>,
    mujoco: ResMut<'w, MuJoCoExampleState>,
    diff_arm: ResMut<'w, DiffArmExampleState>,
    diff_arm6: ResMut<'w, DiffArm6ExampleState>,
    mesh: ResMut<'w, MeshToolsExampleState>,
    neural: ResMut<'w, NeuralPolicyExampleState>,
    // ... ~30 more
}

fn apply_scene_switch(mut sim: ResMut<Sim>, mut scene_states: SceneStates) {
    // sim takes 1 slot, scene_states takes 1 slot — well under 16.
}
```

`PaneSceneStates` is a parallel bundle of the same resources but
declared independently so `control_pane` can borrow them
disjointly from `apply_scene_switch`.

The convention going forward: **whenever you add a new scene with
its own state struct, add the field to BOTH `SceneStates` and
`PaneSceneStates`**. This is the pattern documented in the
`PLAN.md` Stage 22 "Bevy plumbing constraint" note.

## The update / draw cycle

Each scene contributes one system that runs on `Update`. Conditional
dispatch via `current_scene == Scene::X`:

```rust
fn update_mpm_example(
    sim: Res<Sim>,
    mut mpm_state: ResMut<MpmExampleState>,
    pending: Res<PendingSceneSwitch>,
) {
    if pending.0 != Some(Scene::Mpm) { return; }
    // ... call mpm_state.simulation.step_n_with(...)
    // ... rewrite gizmo positions
}
```

The actual step happens through molla's `step_simulation` system,
which dispatches to the active `Solver` for most scenes. Scenes
that have their own substep loop (MPM, FEM steppers, neural
controllers) **short-circuit** `step_simulation` and run their own
update system.

## Egui control panes via bevy_frost

The right-rail UI is composed of one `control_pane` system that
calls `egui::Window::show()` with bevy_frost styling. Each scene
contributes a small block inside the match:

```rust
fn control_pane(/* ... */ mut pane_states: PaneSceneStates) {
    egui::SidePanel::right("rail").show(ctx, |ui| {
        match scene {
            Scene::Mpm => {
                ui.add(egui::Slider::new(&mut pane_states.mpm.friction_angle, 5.0..=45.0));
                ui.label(format!("Particles: {}", pane_states.mpm.particle_count));
            }
            Scene::Camera => {
                pane_states.camera.draw_thumbnails(ui);
            }
            // ... ~100 more arms
        }
    });
}
```

The "input firewall" — preventing egui mouse clicks from also
triggering camera orbit — is provided by bevy_glacial which reads
pane rects from the egui frame state and clips its own input.

## Hotkeys

Marquee scenes have single-letter hotkeys mapped via Bevy's
`Input<KeyCode>`:

| Key | Scene |
|-----|-------|
| `Q` | Quadrotor |
| `V` | VbdCloth |
| `J` | MuJoCo |
| `G` | DiffArm |
| `H` | DiffArm6 |
| `M` | MeshTools |
| `N` | NeuralPolicy |
| `L` | NeuralLstm |
| `C` | Cable |
| `P` | Mpm |
| `Y` | Jelly |

Toggle keys for cross-cutting UX:

- `T` — Muli-style activity visualisation (sleeping bodies render
  grey).
- `Space` — pause / resume.
- `R` — reset current scene.

The full keybinding table is in `crates/molla_viewer/src/main.rs`
in the `handle_hotkeys` system.

## Embedding molla in your own Bevy app

If you want a molla simulation inside an existing Bevy app rather
than the molla viewer:

1. Add `molla = { path = "...", default-features = true }` to your
   `Cargo.toml`. Default-on `gpu` ensures the wgpu paths compile.
2. Create the `Engine` at startup and store it in a resource:

```rust
#[derive(Resource)]
struct MollaEngine(molla::compute::Engine);

fn setup(mut commands: Commands) {
    commands.insert_resource(MollaEngine(
        molla::compute::Engine::new(molla::compute::Backend::Gpu).unwrap()
    ));
}
```

3. Build a `Model` via `ModelBuilder` (or load via URDF), store it
   in another resource.
4. Step in a `FixedUpdate` system:

```rust
fn step_physics(
    engine: Res<MollaEngine>,
    mut sim: ResMut<MyMollaSim>,
    time: Res<Time<Fixed>>,
) {
    sim.solver.step(&sim.model, &sim.state, &sim.control, &sim.contacts, time.delta_secs() as f64, &mut sim.next_state).unwrap();
    std::mem::swap(&mut sim.state, &mut sim.next_state);
}
```

5. Copy `sim.state.body_q` into your Bevy `Transform`s each frame.

That's the entire integration surface. molla doesn't care that it's
inside Bevy; it doesn't even know Bevy exists.

## See also

- [`viewer/overview.md`](../viewer/overview.md) — full viewer
  architecture.
- [`viewer/scenes.md`](../viewer/scenes.md) — the scene catalog.
- `crates/molla_viewer/src/main.rs` — the entire viewer in one
  ~24k-line file. Search by scene name to find the per-scene
  implementation.
