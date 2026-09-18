# Scout E — facade `molla`, `molla-sim`, `bin/molla_viewer`

Region: `crates/molla`, `crates/molla-sim`, `bin/molla_viewer` incl. every file under
`src/examples/`. Read-only. Anchors are `file:line` or the exact command.

## 1. Per-crate purpose

**crates/molla (facade, 5580 loc / 123 files).** A namespace, not an implementation. `src/lib.rs:18-33`
declares 11 modules and 4 wholesale re-exports; `src/prelude.rs`, `src/math/mod.rs:3`,
`src/storage.rs:3`, `src/optim.rs:3`, `src/vehicle.rs:4`, `src/fem/mod.rs:6` are one-line
`pub use <subcrate>::*`. The only substantive thing it owns is a wiring surface:
`src/kernels/mod.rs:118-200` is a ~400-identifier `pub use molla_kernels::{...}` wall that admits an
entire embedded tire simulation stack (FEM/DEM/Bekker/thermal/brush owners) in one statement. Source-only
line counts: kernels 330/29 files, geometry 218/26, actuators 109/13, sensors 91/6, solvers 89/11,
compute 78/11, autograd 77/7, math 15/2, fem 6/1.

**crates/molla-sim (6208 loc / 12 files).** A data contract, not a simulator. `src/lib.rs:1-11` says so
explicitly: Model/State/Control/Contacts, "the contract every solver consumes". `builder.rs` (3699) is
validation and construction and contains no stepping; the only real algorithms are `articulation.rs:31`
`eval_fk` (FK+CRBA) and `soft_surface.rs` edge/triangle derivation. It depends only on core, math, storage
(`crates/molla-sim/src/model.rs:3-5`) — no geometry, no solvers.

**bin/molla_viewer (136 src files; main.rs 11245 lines + 33950 lines of fragments).** A Bevy+egui app that
`include!`s 135 scene fragments into the bin crate root (`main.rs:170-7902`) and dispatches a 151-variant
`enum Scene` (`main.rs:1284-1895`) through six parallel match statements.

## 2. Real dependency edges vs Cargo claims

Live and thin (each proven by exactly one line): core (`lib.rs:21`), sim (`lib.rs:22`), math
(`math/mod.rs:3`), storage (`storage.rs:3`), optim (`optim.rs:3`), vehicle (`vehicle.rs:4`), fem
(`fem/mod.rs:6`), compute (`compute/mod.rs:41-45`, gpu-gated), kernels (`kernels/mod.rs:118-200`),
solvers (`solvers/mod.rs:32,37`), geometry (24 direct uses in `src/geometry/*`).

**molla-sim's consumers** (the important edge): geometry `src/pipeline.rs:8`, `broad_phase.rs:14`,
`narrow_phase.rs:9`, `shape_world.rs:5`, `bvh.rs:15`, `sdf_bake.rs:37`, `lbvh.rs:54` …; kernels
`integrate_bodies.rs:6`, `tiled_camera.rs:19`, `actuator_pd.rs:4`; actuators, solvers, sensors, fem.
23 files in geometry alone. That is a healthy contract edge.

**Facade consumers:** none but the viewer. `bin/molla_viewer/Cargo.toml:34-35` depends on `molla` and
`molla-vehicle`; every fragment imports through the facade (`main.rs:44-70`). So the facade has exactly
one in-repo client and its shape is never pressure-tested by a second one.

**Stale claims (7 deps, zero source tokens in src/tests/examples/benches each):** `glam`
(Cargo.toml:59), `tracing` (:60), `slotmap` (:61), `bytemuck` (:62), `rayon` (:66-69 — its own comment
claims decimate/SDF-bake use it; those files contain no `rayon`/`par_iter`), and dev-dep `rustfft`
(:76-78, "test-only" but absent from tests/). `criterion` is live (`benches/step_cpu.rs:9`). In the
viewer, `wgpu` (`Cargo.toml:36-41`) has no `wgpu::` use anywhere in src — GPU access is via
`molla::compute::Engine`.

## 3. Dead / dormant

Unused cargo deps above. Dormant work: `finalize` uploads model+state_a+state_b+control+contacts
(`main.rs:3814-3827`) before any scene is known to step, while ~110 of 151 scenes early-return from
`step_simulation` (`main.rs:4520-4670`); `_ => SolverKind::Xpbd` (`main.rs:2701`) builds a GPU XPBD solver
for host-driven scenes the comment itself calls unnecessary (2698-2700); three Warp panes are label-only
(`examples/59_warp_dem.rs:10`, `60_warp_fluid.rs:7`, `61_warp_sph.rs:7`); `BodyVisual::GaussianSplat` is
`#[allow(dead_code)]` with "stub" in its doc (`main.rs:1259-1271`); `PhysicsAuditStatus::Placeholder`
carries `#[allow(dead_code)]` and a test enforces it stays empty (`main.rs:9779-9792`).

## 4. Redundancy

Six-to-eight parallel matches over `Scene`, all verified exhaustive today: `build_scene_model:3566`,
`solver_kind:2645`, `label:2056`, `description:2214`, `physics_audit:2770`, pane title `5608`, pane body
`5724`, plus `Scene::ALL` `1898-2053` (already in a different order than the enum declaration). The
"which scenes are real physics" partition is written twice in opposite directions (151 builder arms vs
~110 skip arms). The Scene::label numbering duplicates itself: 150 numbered labels, 142 distinct numbers,
duplicates at 103/104/105/106/107/108/109/110, 117 unused — e.g. `103` is both `MpmMultiMaterial` and
`Vehicle`. Example filenames collide the same way (`103_mpm_multi_material.rs` vs `103_vehicle.rs`).
`finalize:3835-3885` repeats `tune_scene_solver:3990-4026` verbatim for four VBD scenes. 20 of 151 labels
reuse another scene's builder (`WarpSph`→`build_mpm_model`, `VehicleOffroad`/`VehicleSysid`→
`build_vehicle_model`, `TerrainLab`→`build_tire_dynamics_model`, `NeuralLstm`→`build_neural_policy_model`).

## 5. Connection health

Two real edges carry the region: subcrate→`molla-sim` (clean, wide) and `molla_viewer`→`molla` (single
consumer). The facade is a re-export trampoline with seven dead Cargo edges and a 400-name namespace, and
nothing in-tree validates its public surface. Missing lifecycle is the visible symptom: `Sim::reset`
(`main.rs:4031-4091`) restores only `body_q/body_qd/particle_q`; only 8 of 135 fragments define their own
`fn reset` (`109_warpfemconvdiffdg.rs:3486`, `112_warpfemstokestransfer.rs`, `121_tire_traction.rs`,
`47_fem_heat.rs`, `48_fem_beam.rs`, `50_fem_stokes.rs`, `96_dgburgersshock.rs`, `97_shallowwaterdambreak.rs`)
and the rest use ad-hoc `user_reset: bool` flags, so `R` means something different — or nothing — in most
scenes.

## 6. Removal candidates

- Six unused deps in `crates/molla` + `wgpu` in the viewer. Low risk (build-time only); `wgpu` medium
  because it may be pinning Bevy-molla feature unification.
- Duplicate label/file numbering 103-110. Low risk. Unblocks a readable dropdown.
- `Scene::WarpSph` as a separate variant (identical path to `WarpFluid`; real SPH is `SphDamBreak`).
- `PhysicsAuditStatus::Placeholder`. Low risk.
- The ~42 host-only scenes routed to `build_warp_empty_model` (`main.rs:3684-3720`), several with 7-11 line
  fragments. High risk: many are the only in-repo demonstrator of a `molla-compute-primitives` kernel, and
  the pane/gizmo code is most of the 33950 fragment lines. This is the owner's real decision.

## 7. Verdict

This region is a **demonstration shell around a healthy contract**. `molla-sim` is the good part: a small,
leaf-side data contract consumed by six sibling crates through one import line each. The facade is a
re-export corridor that has grown a flat 400-name namespace and seven dead Cargo edges, and the viewer is a
11245-line bin whose 135 `include!` fragments give 1194 definitions a single namespace, dispatched through
six to eight hand-maintained 151-arm matches. It should be: a facade that re-exports per family and keeps
the tire stack opt-in; a viewer whose scenes are `mod`ules behind one `SceneSpec` table (variant, label,
builder, solver, audit, pane, reset) so a scene is added in one place; and a solver readout derived from
the running solver instead of typed into the UI. The physics underneath is not the problem — the wiring
above it is.

### Obstacles to change, smallest fix

1. **One flat namespace.** 135 `include!`s, 1194 top-level defs (verified: `examples/131_ppf_gallery2.rs:5`
   calls `ppf_add_cloth` from `examples/126_ppf_gallery.rs:9` with no `use`). *Fix:* wrap the includes in
   `mod e01; …` and `use` them; one file per commit.
2. **Six parallel matches + `Scene::ALL`.** *Fix:* one `SceneSpec` table that `ALL`, `label`, `solver_kind`
   and `physics_audit` project from; migrate the two cheapest tables first to prove the shape.
3. **The UI lies about the backend.** `main.rs:5559` prints `"XPBD-GPU"` for all 151 scenes while
   `solver_kind` (`2645-2704`) and the CPU overrides (`3810-3885`) select Featherstone/VBD/Barrier/BarrierCpu/
   CPU-VBD. *Fix:* a `label()` on `SolverKind`/`SolverBackend`, used at 5559.
4. **No lifecycle contract.** *Fix:* a `reset` fn pointer in the same table; migrate two scenes.
5. **Names describe stages, not implementations** (150 labels / 142 numbers; `WarpSph`→MPM,
   `TerrainLab`→one FEM tire). *Fix:* render the builder name in the Audit pane next to the label.
6. **GPU required for CPU-only scenes.** `main.rs:3803` unconditionally `expect`s a GPU and uploads
   (3814-3827) before `build_solver`. *Fix:* skip the device path for CPU solver kinds / add a `--cpu` flag.
7. **Dead edges + the 400-name wall.** *Fix:* delete the six unused deps; split `kernels/mod.rs` per family.

## Build probe (verbatim)

```
$ cargo metadata --no-deps --format-version 1
exit 0  (17 packages: molla 10 targets, molla-sim 1, molla_viewer 1)

$ cargo check -p molla
warning: spurious network error (3 tries remaining): failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
warning: spurious network error (2 tries remaining): ... Read-only file system ...
warning: spurious network error (1 tries remaining): ... Read-only file system ...
error: failed to get `bevy_frost` as a dependency of package `molla_viewer v0.0.1 (/home/bresilla/data/code/OUSD/molla/bin/molla_viewer)`
Caused by: failed to load source for dependency `bevy_frost`
Caused by: Unable to update https://github.com/bresilla/bevy_frost?branch=develop#001fc84a
Caused by: failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
exit 101

$ CARGO_NET_OFFLINE=true cargo check -p molla        -> exit 101, identical (checkout absent, cache unwritable)
$ CARGO_HOME=/tmp/cargo-scout-e cargo check -p molla -> exit 101, 'No space left on device' (/tmp is a full tmpfs) + DNS failure
$ CARGO_HOME=/dev/shm/cargo-scout-e cargo check -p molla -> exit 101, 'failed to resolve address for github.com: Temporary failure in name resolution'
```

No compiler ever ran. Every compile-time claim here is text-derived; `cargo metadata --no-deps` is the only
green command. Note the hazard is structural: because `bin/molla_viewer` is a default workspace member,
`cargo check -p molla` still resolves `bevy_frost` — nobody can type-check the facade in a sandbox.

## Limits

Unread by rule: `plans/MAP.md`, `plans/STATUS.md`, `plans/_prior_execution_record.md`, `REPORT.md`,
`GUI_SPLIT.md` (incidentally seen to exist: `plans/` also holds `ROADMAP.md`, `TIRE.md`; `assets/scans/`
holds a sibling `D.json`/`D.md`). Not checked: Makefile targets named by `physics_audit`, the internals of
the sibling crates, and whether every facade re-export is reachable (needs a build).
