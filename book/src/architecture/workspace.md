# Workspace layout

molla is a Cargo workspace with reusable library crates under `crates/`, runnable
applications under `bin/`, and read-only reference trees under `xtra/`. The
full top-level layout:

```
molla/
├── Cargo.toml              workspace root
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
│   ├── molla-core/         IDs, Error/Result, Axis, tiny shared primitives
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-math/         Real, Vec3, Transform, spatial algebra, sparse math
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-storage/      Buffer<T>, BufferUsage, host/device conversion
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-compute/      Engine/Backend, wgpu device/queue runtime
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-compute-primitives/ tiled compute primitives
│   │   ├── Cargo.toml
│   │   ├── shaders/
│   │   └── src/
│   ├── molla-sim/          Model, State, Control, Contacts, ModelBuilder
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-optim/        host Adam/SGD optimizer state
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-actuators/    muscle, PD/PID, neural, delay, clamping
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-kernels/      first extracted WGSL dispatch wrappers
│   │   ├── Cargo.toml
│   │   ├── shaders/
│   │   └── src/
│   ├── molla-sensors/      CPU raycast adjoint/oracle
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-geometry/     first extracted geometry primitives
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-solvers/      Solver trait, semi-implicit, dynamics helpers
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-autograd/     first extracted CPU tape/adjoint helpers
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla-fem/          FEM module tree plus FEM-specific GPU kernels
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── molla/              public facade + still-mostly-monolithic engine crate
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── prelude.rs
│   │   │   ├── math/       facade re-export + compatibility GPU sparse wrapper
│   │   │   ├── storage.rs  facade re-export
│   │   │   ├── compute/    facade runtime re-export + MPM/tile/stam compat paths
│   │   │   ├── kernels/    58 Rust dispatchers (one per WGSL kernel)
│   │   │   ├── geometry/   shapes, hull, BVH, GJK, EPA, MPR, sparse_grid, splat
│   │   │   ├── solvers/    semi-implicit, xpbd, featherstone, vbd, MPM
│   │   │   ├── fem/        facade re-export for molla-fem
│   │   │   ├── actuators/  PD, PID, DC motor, neural, delay, muscle
│   │   │   ├── sensors/    raycast, camera, IMU, contact, frame_transform
│   │   │   ├── autograd/   tape, featherstone reverse, raycast adjoint
│   │   │   ├── optim.rs    facade re-export
│   │   │   ├── utils/      compatibility re-exports
│   │   │   └── core/
│   │   ├── shaders/        79 WGSL kernels (one file per pipeline)
│   │   ├── tests/          facade-level integration/compatibility tests
│   │   └── benches/        43 criterion benches
├── bin/
│   └── molla_viewer/       the Bevy 0.18 viewer app
│       ├── Cargo.toml
│       └── src/main.rs     single-file viewer (~24k lines, ~100 scenes)
└── xtra/                   reference codebases — NOT compiled
    ├── newton/             NVIDIA Newton — read-only algorithm reference
    └── warp/               NVIDIA Warp — compute primitive reference
```

## What lives where

### `crates/molla-core` — core handles and errors

Extracted foundation crate. It owns stable IDs (`BodyId`,
`JointId`, `ShapeId`, `ParticleId`, `WorldId`), the public `Error` /
`Result` types, `MAXVAL`, and `Axis`. The facade re-exports it as
`molla::core` so existing paths keep working.

### `crates/molla-math` — host math and numerics

Extracted foundation crate for `Real`, `Vec3`, `Quat`, `Mat3`,
`Transform`, spatial algebra, dense/sparse CPU numerics, iterative solvers,
and robust losses. The facade re-exports it as `molla::math`.

### `crates/molla-storage` — buffer abstraction

Extracted foundation crate for `Buffer<T>`, `BufferElement`, and
`BufferUsage`. It owns the host/device conversion boundary and is re-exported
as `molla::storage`.

### `crates/molla-compute` — runtime only

Extracted foundation crate for runtime selection and `wgpu` device/queue
ownership (`Engine`, `Backend`, `AdapterInfo`). Higher-level MPM and tile
functionality has moved out: MPM drivers live in `molla-solvers`, tiled compute
primitives live in `molla-compute-primitives`, and the 2-D Stam stable-fluid
solver lives in `molla-fem` because its GPU projection reuses FEM FFT-Poisson.

### `crates/molla-compute-primitives` — tiled compute primitives

Functional crate for reusable tiled compute code. It owns tiled matmul,
block-Cholesky, convolution/filtering, stream compaction, and MC-GP helpers plus
their WGSL files. The facade keeps `molla::compute::tile_*` compatibility paths.

### `crates/molla-sim` — raw simulation data model

Extracted functional crate for `Model`, `State`, `Control`, `Contacts`,
`ModelBuilder`, shape descriptors, raw `TriMesh`, and raw soft-surface model
descriptors. The facade re-exports it as `molla::sim`.

Raw soft-surface descriptors live here; FEM boundary-facet extraction lives in
`molla-fem`, and the old `molla::geometry::soft_surface::*` facade path
re-exports both sides for compatibility.

### `crates/molla-optim` — host optimizers

Extracted functional crate for host-side `Adam` and `SGD` optimizer state.
The facade re-exports it as `molla::optim`.

### `crates/molla-actuators` — actuator models

Functional crate for actuator code. It owns the host Hill-type muscle model and
the GPU-first PD/PID, neural MLP/LSTM, delay, clamping, and actuator-pipeline
orchestrators. `molla::actuators::*` remains a facade compatibility path while
the dispatch wrappers and WGSL files live under `molla-kernels`.

### `crates/molla-kernels` — extracted GPU dispatch slices

Started kernel crate. It currently owns dense3, robust-loss, optimizer
Adam/SGD, scan, Dahl friction, contact reduction,
GPU hydroelastic anchor reduction, sensor dispatch wrappers,
actuator/clamping/muscle dispatch wrappers, and
rigid-body solver dispatch wrappers for integrate-bodies, CRBA, RNEA,
Jacobian, XPBD, Featherstone, VBD/soft-contact, implicit-MPM, FFT, sparse-grid,
Gaussian splat raytrace, and Featherstone adjoint, plus their WGSL files. Radix
sort lives in `molla-compute-primitives`; geometry GJK/EPA/MPR, mesh-BVH
raycast, and LBVH live in `molla-geometry`, with compatibility re-exports from
`molla-kernels`. The facade keeps `molla::kernels::*` compatibility through
thin re-export modules only; implementation ownership lives in the kernel,
geometry, compute-primitive, or functionality crate.

### `crates/molla-sensors` — sensor helpers

Functional crate for sensor code. It owns the GPU-first raycast, tiled camera,
IMU, contact aggregation, and frame-transform orchestrators, plus the CPU
finite-difference raycast adjoint/oracle. `molla::sensors::*` remains a facade
compatibility path.

### `crates/molla-geometry` — geometry primitives

Started functional crate for geometry/collision code. It currently owns `Aabb`,
`SpatialHashGrid`, `MeshBvh`, tet AABB helpers, convex support functions,
convex hull authoring, world-frame shape-transform/AABB helpers, QEM mesh
decimation, dense mesh SDF baking, sparse volume/SDF storage, marching cubes,
brute-force/SAP/BVH broad phase, CPU narrow-phase dispatch, GJK/EPA/MPR convex
queries, Gaussian splat math, hydroelastic contact, soft BVH/VT CCD, and the CPU contact-reduction
primitive plus warm-start/matching extras; the broader collision pipeline
remains in the facade until the kernel and domain dependencies are untangled.

### `crates/molla-solvers` — solver and dynamics helpers

Started functional crate for solver code. It currently owns CPU-only CRBA
mass-matrix assembly, recursive Newton-Euler inverse dynamics, geometric
Jacobian helpers, inverse kinematics, VBD soft-contact helpers, the shared
`Solver` trait, the CPU/GPU semi-implicit rigid-body solver, CPU/GPU XPBD,
CPU/GPU VBD, CPU/GPU Featherstone, the implicit MPM CPU driver, and the
explicit GPU `MpmSimulation` driver. Featherstone's tape
helper API is gated behind `molla-solvers/autograd`; the public facade enables
that feature to preserve the compatibility API without making standalone
solver users depend on autograd unconditionally.

### `crates/molla-autograd` — CPU tape and adjoints

Started functional crate for autograd code. It currently owns the
engine-agnostic CPU tape, differentiable arm helper, Featherstone adjoint
data/trace helpers, XPBD distance-constraint adjoint helpers, and the
differentiable CPU MPM forward/adjoint pair. The facade still exposes
`molla::compute::mpm_cpu_diff::*` as a compatibility re-export.

### `crates/molla-fem` — finite-element methods

Extracted functional crate for FEM code. It compiles the FEM module tree
independently: mesh/function spaces, quadrature, assembly, CPU/GPU solvers,
DG/AMR/APIC coupling, Stokes/NS paths, magnetostatics, eddy currents,
Signorini, nonconforming contact, and FEM-specific GPU kernels. The facade
`molla::fem::*` module is a compatibility re-export of `molla-fem`; generic
GPU BSR sparse algebra lives in `molla-kernels` and is also preserved through
the old `molla::math` facade path.

### `crates/molla` — the facade and remaining engine

This is the part you would embed in your application. Zero
dependencies on Bevy, ROS, or any rendering / IO format.
~50k lines of Rust + ~30k lines of WGSL.

Module breakdown by line count (approximate, May 2026):

| Module     | LOC   | Purpose                                          |
|------------|-------|--------------------------------------------------|
| `fem`      | ~14k  | FEM framework: assembly, P1/P2, elasticity, Stokes, NS, DG, AMR, magnetostatics, eddy, multigrid, Signorini, Coulomb |
| `kernels`  | ~9k   | Rust dispatchers for the 58 WGSL kernels         |
| `geometry` | ~8k   | shapes, hull, BVH, narrow phase, sparse grid, splats |
| `solvers`  | ~5k   | semi-implicit, XPBD, Featherstone, VBD, MPM      |
| `compute`  | ~3k   | wgpu runtime + MPM + tile primitives             |
| `math`     | ~3k   | host math + BSR + iterative solvers + dense3     |
| `autograd` | ~2k   | tape + Featherstone reverse + raycast adjoint    |
| `actuators`| ~1.5k | PD / PID / DC / neural / muscle / delay          |
| `sensors`  | ~1.5k | raycast / camera / IMU / contact / frame xform   |
| `storage`  | ~0.5k | `Buffer<T>`                                      |

### `bin/molla_viewer` — the showroom

A Bevy 0.18 + bevy_egui + bevy_frost + bevy_glacial application
that loads any of ~100 scenes and renders them in real time. It is
kept under `bin/` because it is a runnable app, not a reusable
library crate. It is deliberately a single file (`src/main.rs`, ~24k lines) so the
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

- `xtra/newton/` — the NVIDIA Newton tree. It is a read-only algorithm
  reference and is not part of the Rust build graph. Apache-2.0.
- `xtra/warp/` — the NVIDIA Warp tree. `warp/examples/*.py` are the
  reference renderings for the ~30 `Warp*` scenes in the viewer.
  Apache-2.0.

You can ignore `xtra/` entirely unless you are porting a new
algorithm — at which point you'll want both trees open in your
editor (see the [`reference_warp_newton.md`] note in user memory).

## Modularization rule

`molla` is being split by functionality, not by CPU/GPU backend. Each
functional crate owns both its CPU oracle and its GPU implementation wherever a
GPU path is feasible. `crates/molla` stays as the public facade and
`bin/molla_viewer` stays as a runnable viewer only.
