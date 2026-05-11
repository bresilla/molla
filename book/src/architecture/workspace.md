# Workspace layout

molla is a Cargo workspace with two member crates and one set of
read-only reference trees under `xtra/`. The full top-level layout:

```
molla/
├── Cargo.toml              workspace root, members = [molla, molla_viewer]
├── Cargo.lock
├── flake.nix               nix devShell — pulls wgpu + Vulkan loaders
├── flake.lock
├── .envrc                  direnv hook; exports MOLLA_NVIDIA_VERSION
├── Makefile                every `make` target — see docs/architecture/build.md
├── PLAN.md                 ~6000 lines — source of truth for scope + roadmap
├── GAP_v2.md               Newton/Warp gap analysis (refreshed periodically)
├── PROJECT                 name + version (read by Makefile)
├── LICENSE                 MIT
├── crates/
│   ├── molla/              the engine crate
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── prelude.rs
│   │   │   ├── core/       BodyId / JointId / Error
│   │   │   ├── math/       f64 host types + BSR sparse + dense3
│   │   │   ├── storage/    Buffer<T> + BufferUsage
│   │   │   ├── compute/    Engine, MPM, stam fluid, tile_*
│   │   │   ├── kernels/    58 Rust dispatchers (one per WGSL kernel)
│   │   │   ├── geometry/   shapes, hull, BVH, GJK, EPA, MPR, sparse_grid, splat
│   │   │   ├── sim/        Model, State, Control, Contacts, ModelBuilder
│   │   │   ├── solvers/    semi-implicit, xpbd, featherstone, vbd, mujoco
│   │   │   ├── fem/        the entire FEM stack (~30 modules)
│   │   │   ├── actuators/  PD, PID, DC motor, neural, delay, muscle
│   │   │   ├── sensors/    raycast, camera, IMU, contact, frame_transform
│   │   │   ├── autograd/   tape, featherstone reverse, raycast adjoint
│   │   │   ├── optim/      Adam, SGD, robust loss
│   │   │   ├── utils/      URDF importer
│   │   │   └── core/
│   │   ├── shaders/        79 WGSL kernels (one file per pipeline)
│   │   ├── tests/          154 integration tests (most #[ignore]'d for GPU)
│   │   └── benches/        43 criterion benches
│   └── molla_viewer/       the Bevy 0.18 viewer app
│       ├── Cargo.toml
│       └── src/main.rs     single-file viewer (~24k lines, ~100 scenes)
└── xtra/                   reference codebases — NOT compiled
    ├── newton/             Python Newton — trajectory oracle
    └── warp/               NVIDIA Warp — compute primitive reference
```

## What lives where

### `crates/molla` — the engine

This is the part you would embed in your application. Zero
dependencies on Bevy, USD, ROS, or any rendering / IO format.
~50k lines of Rust + ~30k lines of WGSL.

Module breakdown by line count (approximate, May 2026):

| Module     | LOC   | Purpose                                          |
|------------|-------|--------------------------------------------------|
| `fem`      | ~14k  | FEM framework: assembly, P1/P2, elasticity, Stokes, NS, DG, AMR, magnetostatics, eddy, multigrid, Signorini, Coulomb |
| `kernels`  | ~9k   | Rust dispatchers for the 58 WGSL kernels         |
| `geometry` | ~8k   | shapes, hull, BVH, narrow phase, sparse grid, splats |
| `sim`      | ~6k   | data model + `ModelBuilder`                      |
| `solvers`  | ~5k   | semi-implicit, XPBD, Featherstone, VBD, MuJoCo   |
| `compute`  | ~3k   | wgpu runtime + MPM + tile primitives             |
| `math`     | ~3k   | host math + BSR + iterative solvers + dense3     |
| `autograd` | ~2k   | tape + Featherstone reverse + raycast adjoint    |
| `actuators`| ~1.5k | PD / PID / DC / neural / muscle / delay          |
| `sensors`  | ~1.5k | raycast / camera / IMU / contact / frame xform   |
| `optim`    | ~0.5k | Adam / SGD / robust losses                       |
| `storage`  | ~0.5k | `Buffer<T>`                                      |

### `crates/molla_viewer` — the showroom

A Bevy 0.18 + bevy_egui + bevy_frost + bevy_glacial application
that loads any of ~100 scenes and renders them in real time. It is
deliberately a single file (`src/main.rs`, ~24k lines) so the
"add a scene" workflow is one PR that touches one file.

Why bevy_frost + bevy_glacial:

- **bevy_frost** is an egui-themed glass UI kit. It provides the
  right-rail control pane and the per-scene sliders.
- **bevy_glacial** is an orbit camera + grid + gizmos plugin. It
  saves the viewer from reimplementing scene-camera logic from
  scratch.

Both are pulled from `bresilla/*` repos because they are not yet on
crates.io. They are pure Bevy plugins — there is no molla code in
either.

### `xtra/` — read-only reference

The directory is **excluded** from the Cargo workspace
(`exclude = ["xtra"]`) so `cargo build` never compiles it. The trees
are:

- `xtra/newton/` — the Python Newton tree. `tests/` here are the
  trajectory-oracle source. Apache-2.0.
- `xtra/warp/` — the NVIDIA Warp tree. `warp/examples/*.py` are the
  reference renderings for the ~30 `Warp*` scenes in the viewer.
  Apache-2.0.

You can ignore `xtra/` entirely unless you are porting a new
algorithm — at which point you'll want both trees open in your
editor (see the [`reference_warp_newton.md`] note in user memory).

## Why not split `molla` into more crates?

The engine was scoped as a single crate from Stage 0 so it would have
exactly one `Cargo.toml` to keep in sync, exactly one feature flag
graph, and exactly one place to insert `#[cfg(feature = "gpu")]`
attributes. The downsides — slower clean builds, less granular
public API — are real, but compile times stay manageable
(incremental builds are O(seconds)) and the public API is enforced
by `prelude` + module-level `pub use` re-exports.

If/when a downstream user wants only the `math` layer (e.g. as a
no_std reusable crate) we will split it out; until that user
materialises, splitting just adds maintenance overhead.

## What was here before

Earlier in development (pre-Stage 8) the workspace contained a
sibling `crates/usd_molla/` that wrapped `openusd::Stage` ↔
`molla::Model`. It was removed in commit `8c81b13` — molla now
integrates with USD through the existing `bevy_openusd` adapter
ecosystem rather than maintaining a sibling crate. See
[`importers/usd.md`](./../importers/usd.md) for the full story.
