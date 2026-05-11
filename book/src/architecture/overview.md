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
                │  importers   URDF · MJCF · USD · Gmsh   │
                ├─────────────────────────────────────────┤
                │  solvers     SemiImplicit · XPBD ·      │
                │              Featherstone · VBD · MPM · │
                │              MuJoCo · FEM steppers      │
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

### `kernels` — compute dispatchers

Each kernel is a triple:

1. A WGSL source file at `crates/molla/shaders/<name>.wgsl`.
2. A Rust dispatcher at `crates/molla/src/kernels/<name>.rs` that
   owns the `wgpu::ComputePipeline`, the bind-group layouts, and a
   `dispatch_blocking` method that takes typed `Buffer`s.
3. A CPU reference implementation in the same Rust module (private
   `fn ..._cpu(...)`) that produces the same output and is used as
   the oracle in parity tests.

There are 58 such kernels at present, covering integration,
constraint projection, contact reduction, narrow-phase (GJK / EPA /
MPR), broad-phase (LBVH), sensors (raycast / camera / IMU / contact /
frame transform), actuators (PD / PID / DC motor / muscle / delay /
neural MLP / neural LSTM), FEM (assembly / CG / PCG / elasticity /
Stokes / Navier-Stokes / wave / DG SWE flux), sparse BSR
(matvec / axpy / transpose / SpGEMM), splats (forward + backward),
optimizers (Adam / SGD), and primitive utilities (scan / radix sort /
FFT / SVD3 / QR3 / eig3 / dense Cholesky / tile matmul / tile
convolution / tile stream compaction / tile MCGP).

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
API every importer ends up calling.

### `solvers` — the step functions

Each solver implements the [`Solver`] trait: given `&Model`,
`&State`, `&Control`, `&Contacts`, `dt`, write the next `State`.
Solver kinds in `prelude`:

- `SemiImplicitSolver` — unconstrained Euler, the baseline (Stage 1).
- `XpbdSolver` — position-based constraints, particles + rigid + cloth.
- `FeatherstoneSolver` — reduced-coordinate articulated rigid (ABA).
- `VbdSolver` — Vertex Block Descent for cloth + cable + tets +
  jelly.
- `MuJoCoSolver` — gated behind the `mujoco` Cargo feature; wraps
  the upstream MuJoCo C library.

There is no FEM solver in this list — FEM steppers live in `fem` and
are wired into specific scenes via state structs.

### Importers + adapters

[`utils::urdf`](./../importers/urdf.md) parses URDF into a
`ModelBuilder` via `urdf-rs`. `fem::FemMesh::from_gmsh*` parses MSH
files via `mshio`. The MJCF writer lives inside the
`solvers::mujoco` module (only used by the MuJoCo wrapper). USD
support is documented in [`importers/usd.md`](./../importers/usd.md) —
currently routed through the upstream `bevy_openusd` adapter.

The Bevy adapter is the `molla_viewer` crate.

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
- Parity tests (in `crates/molla/tests/*_gpu.rs`) take the same
  input, run both backends, and assert that the trajectories agree
  to within the documented tolerance (1e-4 for f32-narrowing,
  1e-3 for adjoint-vs-FD checks).

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
