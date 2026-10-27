# Architecture overview

`molla` is structured as a stack of layers, each of which depends only
on the ones below it. The same engine binary can run wholly on the
CPU, wholly on the GPU, or on a mix of both, and the layering makes
that toggle a single line of code instead of an architectural surgery.

## The layered design

```
                ┌─────────────────────────────────────────┐
                │  adapters    molla_viewer (Bevy ECS)    │
                ├─────────────────────────────────────────┤
                │  solvers     SemiImplicit · XPBD ·      │
                │              Featherstone · VBD · MPM · │
                │              FEM steppers               │
                ├─────────────────────────────────────────┤
                │  sim         Model · State · Control ·  │
                │              Contacts · ModelBuilder    │
                ├─────────────────────────────────────────┤
                │  geometry · sensors · actuators · fem · │
                │  autograd · optim                       │
                ├─────────────────────────────────────────┤
                │  compute     Engine · Backend selector  │
                │  kernels     58 WGSL kernels + Rust     │
                │              dispatchers + CPU oracles  │
                ├─────────────────────────────────────────┤
                │  storage     Buffer<T> · BufferUsage    │
                ├─────────────────────────────────────────┤
                │  math        f64 host (DVec3 / DMat3 /  │
                │              DQuat / Transform / BSR)   │
                ├─────────────────────────────────────────┤
                │  core        BodyId / JointId / Errors  │
                └─────────────────────────────────────────┘
```

Read it bottom-up — each layer is built from the ones below it.

### `core` — IDs and errors

[`core`](https://docs.rs/molla/latest/molla/core/index.html) defines
the typed handles (`BodyId`, `JointId`, `ParticleId`, `ShapeId`,
`WorldId`) and the engine-wide `Error` enum (`thiserror`-derived; no
`anyhow` inside the engine crate). Everything else uses these.

### `math` — host arithmetic

[`math`](https://docs.rs/molla/latest/molla/math/index.html) is the
host-side `f64` math layer. It wraps `glam::DVec3` / `DMat3` /
`DQuat` (renamed `Vec3` / `Mat3` / `Quat` for terseness), exposes
`Transform`, `SpatialVector` / `SpatialInertia` for Featherstone
spatial algebra, the BSR sparse matrix + iterative solvers (CG,
BiCGStab, GMRES, PCG), the small-matrix decompositions (SVD3, polar,
QR3, symmetric eig3), and the graph-coloring utilities.

### `storage` — the buffer abstraction

[`storage::Buffer<T>`](https://docs.rs/molla/latest/molla/storage/struct.Buffer.html)
is the single chokepoint where host `f64` becomes device `f32`. Every
piece of long-lived data in `Model`, `State`, `Control` is a `Buffer`.
A buffer holds either a host `Vec` (CPU mode) or a `wgpu::Buffer`
(GPU mode), tracked by `BufferUsage`. Conversions happen once on
upload / download — math code never sees mixed precision.

### `compute` — the wgpu runtime

[`compute::Engine`](https://docs.rs/molla/latest/molla/compute/struct.Engine.html)
owns the `wgpu::Device` + `Queue` and exposes the `Backend` enum:
`Backend::Cpu` skips device dispatch entirely; `Backend::Gpu` uses
the real adapter. The engine is constructed once (via `pollster` for
the sync API) and threaded into every solver / sensor that needs a
device.

### `kernels` and functionality-owned GPU dispatchers

GPU dispatchers are split by ownership instead of by CPU/GPU:

1. Shared dispatch wrappers and WGSL sources live in
   `crates/molla-kernels/`.
2. Functionality-specific dispatchers live next to their owner crate; for
   example GJK/EPA/MPR, LBVH, and mesh-BVH raycast live in
   `crates/molla-geometry/`, FEM GPU kernels live in `crates/molla-fem/`, and
   tiled/radix primitives live in `crates/molla-compute-primitives/`.
3. `crates/molla/src/kernels/` is now a compatibility surface for
   `molla::kernels::*`, not the source of ownership.

Each dispatcher still follows the same pattern: a WGSL source file, a Rust type
that owns the `wgpu::ComputePipeline` and bind-group layouts, and CPU/oracle
coverage in the owning crate's tests where applicable.

### `geometry`, `sensors`, `actuators`, `fem`, `autograd`, `optim`

These sit at the same horizontal level. They consume `kernels` +
`math` + `storage` and expose pluggable building blocks. The FEM
stack alone is ~30 kernels deep and is documented as its own
chapter family.

### `sim` — the data model

[`sim::Model`](https://docs.rs/molla/latest/molla/sim/struct.Model.html)
is the immutable simulation description. [`sim::State`] is the
time-varying positions / velocities / forces. [`sim::Control`] is
the user input (joint targets, applied wrenches).
[`sim::Contacts`] is the collision-pipeline output that solvers
consume each step. [`sim::ModelBuilder`] is the fluent authoring
API used by procedural scene/test builders.

### `solvers` — the step functions

Each solver implements the [`Solver`] trait: given `&Model`,
`&State`, `&Control`, `&Contacts`, `dt`, write the next `State`.
Solver kinds in `prelude`:

- `SemiImplicitSolver` — unconstrained Euler, the baseline (Stage 1).
- `XpbdSolver` — position-based constraints, particles + rigid + cloth.
- `FeatherstoneSolver` — reduced-coordinate articulated rigid (ABA).
- `VbdSolver` — Vertex Block Descent for cloth + cable + tets +
  jelly.
There is no FEM solver in this list — FEM steppers live in `fem` and
are wired into specific scenes via state structs.

### Adapters

The Bevy adapter is the `molla_viewer` crate. The engine roadmap is
physics-only: no importer/exporter, MJCF/USD/URDF/Gmsh path, or external FFI
solver wrapper is part of the supported architecture.

## The CPU↔GPU rule

This is the architecture's most load-bearing convention:

> Every solver, sensor, actuator, FEM stepper, and primitive kernel
> can step on either CPU or GPU, and the choice is a runtime
> `Backend` toggle — not a compile-time feature.

Concretely:

- The `gpu` Cargo feature gates the *availability* of the GPU path
  (it pulls in `wgpu` + `pollster`). With `gpu` off, the CPU paths
  still compile and run.
- With `gpu` on, every `XxxSolver::new_cpu()` / `new_gpu(&engine)`
  pair gives the user a runtime choice.
- Parity tests live with the crate that owns the functionality
  (`crates/molla-solvers/tests/*_gpu.rs`,
  `crates/molla-kernels/tests/*_gpu.rs`, etc.) unless they are
  facade/product compatibility coverage in `crates/molla/tests/`.
  They take the same input, run both backends, and assert that the
  trajectories agree to within the documented tolerance (1e-4 for
  f32-narrowing, 1e-3 for adjoint-vs-FD checks).

The viewer's UI has a `[CPU] / [GPU]` radio button — flipping it
rebuilds the active solver from scratch and continues simulating. No
restart, no recompile.

See [`design_rules.md`](./design_rules.md) for the full convention
list, including why we picked BSR `B=1` everywhere, why we use the
atomic-CAS pattern over `atomic<f32>`, and why we don't depend on
`cooperative_matrix`.

## Cross-references

- [Workspace layout](./workspace.md) — exact directory tree.
- [Crate boundaries](./crates.md) — dependency rules.
- [Feature flags](./features.md) — the Cargo feature matrix.
- [Build & test commands](./build.md) — every `make` target and
  `cargo` invocation.
- [Design rules](./design_rules.md) — the conventions.
