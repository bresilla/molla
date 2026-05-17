# Crate boundaries

molla is being modularized by functionality. The current workspace has the
foundation crates `molla-core`, `molla-math`, `molla-storage`, runtime-only
`molla-compute`, tiled compute primitive crate `molla-compute-primitives`,
data-model crate `molla-sim`, host optimizer crate `molla-optim`, the first
`molla-actuators` slice, the first `molla-kernels` slices, the first
`molla-sensors` slice, the first `molla-geometry` slice, the first
`molla-solvers` slice, the first `molla-autograd` slice, and the first
`molla-fem` slice extracted; `molla` remains the public facade plus
remaining engine modules; `bin/molla_viewer` is the binary-only Bevy
application.

```
            ┌────────────────────┐
            │   molla_viewer     │   Bevy 0.18 application
            │  bin/molla_viewer  │
            └─────────┬──────────┘
                      │ depends on
                      ▼
            ┌────────────────────┐
            │       molla        │   public facade + engine modules
            └─────────┬──────────┘
                      │ re-exports
                      ▼
            ┌──────────────────────────────────────────┐
            │ molla-core · molla-math · molla-storage │
            │ molla-compute · molla-compute-primitives │
            │ molla-sim · molla-optim                 │
            │ molla-actuators                         │
            │ molla-kernels                           │
            │ molla-sensors                           │
            │ molla-geometry                          │
            │ molla-solvers                           │
            │ molla-autograd                          │
            │ molla-fem                               │
            └──────────────────────────────────────────┘
```

## The rules

1. **Library crates know nothing about Bevy, ROS, JSON, YAML, or external asset
   formats.** The main `molla` facade/engine only knows about simulation data,
   solvers, kernels, and physics modules it owns.
2. **`bin/molla_viewer` is allowed to depend on Bevy + bevy_egui +
   bevy_frost + bevy_glacial + mimalloc.** That is the full
   surface; the viewer binary package is the only place those dependencies
   live.
3. **No importer/exporter or external solver-wrapper crates.** Procedural model
   construction is the supported surface; physics modules own physics data and
   kernels only.
4. **Optional native dependencies are behind Cargo features.** See
   [features.md](./features.md). The features gate compilation of
   whole modules so a clean `cargo build` on a stock Rust toolchain
   succeeds without SuiteSparse / LAPACKE installed.
5. **There is no Python in the build graph.** The `xtra/` trees are read-only
   references and are *not* workspace members.

## Modularization direction

The accepted direction is functional crate boundaries, not CPU-vs-GPU splits.
`crates/` is for reusable libraries, `bin/` is for runnable applications, and
`crates/molla` remains the public facade that preserves `molla::prelude::*`.

Current extraction status:

- `molla-core` — done.
- `molla-math` — done.
- `molla-storage` — done.
- `molla-compute` — done for `Engine`/`Backend` runtime only.
- `molla-compute-primitives` — done for tiled matmul, block-Cholesky, convolution/filtering, stream compaction, and MC-GP helpers.
- `molla-sim` — done.
- `molla-optim` — done for host Adam/SGD.
- `molla-actuators` — done for host muscle, PD/PID, neural MLP/LSTM, delay, clamping, and actuator pipeline.
- `molla-kernels` — started for dense3, robust-loss, optim, scan, Dahl friction, contact reduction, GPU hydroelastic anchor reduction, sensor dispatch wrappers, actuator/clamping/muscle dispatch wrappers, rigid-body solver kernels (integrate bodies, CRBA, RNEA, Jacobian, XPBD, Featherstone), VBD/soft-contact kernels, implicit-MPM, FFT, sparse-grid kernels, Gaussian splat raytrace kernels, and Featherstone adjoint dispatchers. Radix sort now lives in `molla-compute-primitives`; geometry GJK/EPA/MPR, mesh-BVH raycast, and LBVH now live in `molla-geometry`.
- `molla-sensors` — done for raycast, tiled camera, IMU, contact aggregation, frame transforms, and CPU raycast adjoint/oracle.
- `molla-geometry` — started for AABB, spatial hash, broad-phase BVH/LBVH, mesh BVH, tet AABB helpers, convex support functions, convex hull authoring, world-frame shape helpers, QEM mesh decimation, dense mesh SDF baking, sparse volume/SDF storage, marching cubes, CPU/GPU narrow-phase dispatch, GJK/EPA/MPR convex queries, Gaussian splat math, hydroelastic contact, soft BVH/VT CCD, CPU contact reduction, and contact warm-start/matching extras.
- `molla-solvers` — started for the shared `Solver` trait, CPU/GPU semi-implicit rigid-body integration, CPU/GPU XPBD, CPU/GPU VBD, CPU/GPU Featherstone, implicit MPM CPU driver, explicit GPU MPM driver, VBD soft-contact helpers, inverse kinematics, CPU-only CRBA mass matrices, recursive Newton-Euler inverse dynamics, geometric Jacobians, and optional `autograd` integration for Featherstone tape helpers.
- `molla-autograd` — started for CPU tape, differentiable arm helper, Featherstone trace/adjoint helpers, XPBD distance adjoints, and differentiable CPU MPM forward/adjoint code.
- `molla-fem` — extracted for the FEM module tree, FEM-specific GPU kernels, the 2-D Stam stable-fluid solver, and facade-flipped behind `molla::fem::*`.

## Workspace exclusions

```toml
[workspace]
resolver = "2"
members = [
    "crates/molla-core",
    "crates/molla-math",
    "crates/molla-storage",
    "crates/molla-compute",
    "crates/molla-compute-primitives",
    "crates/molla-sim",
    "crates/molla-optim",
    "crates/molla-actuators",
    "crates/molla-kernels",
    "crates/molla-sensors",
    "crates/molla-geometry",
    "crates/molla-solvers",
    "crates/molla-autograd",
    "crates/molla-fem",
    "crates/molla",
    "bin/molla_viewer",
]
exclude = ["xtra"]
```

The `exclude = ["xtra"]` line is what stops `cargo build` from ever
descending into the Newton / Warp reference trees. They are
read-only references kept side-by-side in the repository for
convenience.

## Dependency rules for outside users

If you depend on `molla` from another crate:

- **Default features include `gpu`** — your build will pull `wgpu`
  + `pollster`. Use `default-features = false` to skip those if you
  only want the CPU paths.
- **`russell` is off by default** and reserves the direct-solver surface for
  optional sparse-linear-algebra experiments. The current implementation stays
  pure Rust in the nix dev shell.
- **The `prelude` is stable.** Anything outside `prelude` may move
  between minor releases until 0.1.0.

The crate is on `0.0.1` at present; no SemVer guarantees yet. The
PLAN's quality bar is that every shipped feature has a trajectory or
unit test, an example, a bench, and CPU/GPU parity — that is the
unofficial release criterion.
