# PLAN — `molla`: a pure-Rust, wgpu-accelerated physics engine

A from-scratch reimplementation of [Newton Physics](https://github.com/newton-physics/newton)
in pure Rust with `wgpu` for GPU compute, designed to run on **non-NVIDIA hardware**
(AMD, Intel Arc, Apple Silicon) as a first-class target. The Python Newton tree
at `./xtra/newton/` is the **reference** for behavior, data layout, and test
oracles — we read it, we do not port it line-for-line.

This document is the source of truth for scope, architecture, decisions, and
the staged delivery roadmap. It is intentionally long: skim the table of
contents, jump to the section you need.

---

## Table of contents

0.  Vision and what success looks like
1.  Goals and non-goals
2.  Locked decisions
3.  Open questions (still to decide)
4.  Reference: what Newton actually is
5.  Workspace layout and crate boundaries
6.  Tech stack
7.  Math layer
8.  Storage and buffer abstraction
9.  Compute layer (kernels)
10. Data model: `Model`, `State`, `Control`, `Contacts`
11. `ModelBuilder` and authoring API
12. Solver progression
13. Collision pipeline
14. Importers (URDF, MJCF, USD)
15. Adapters (USD, Bevy, viewer)
16. Differentiability strategy
17. Testing strategy and the "newton oracle"
18. Performance targets and benchmarking
19. CI, tooling, repo hygiene
20. Stage-by-stage roadmap
21. Risks and how we mitigate them
22. Decision log

---

## 0. Vision and what success looks like

**One sentence**: a Rust crate that loads a USD stage with `UsdPhysics` opinions,
runs the simulation on the user's GPU regardless of vendor, and produces
trajectories that match Newton's reference output within a documented
tolerance.

**Concrete first milestone (end of stage 2)**: 100 spheres falling onto a
plane, simulated on Vulkan/Metal/DX12 via `wgpu`, trajectories match Newton's
`semi_implicit` solver to within 1e-3 m at t=1.0s.

**Concrete medium milestone (end of stage 4)**: a 6-DOF robot arm (Franka
Panda) loaded from URDF, articulated rigid body sim with contact, runs on a
laptop integrated GPU at >= 100Hz with full CCD-disabled collision, matches
Newton's Featherstone solver within documented tolerance.

**Long-term shape**: `molla` becomes the engine that `usd_molla` (an
adapter sibling to `usd_rapier`) projects USD physics opinions onto, plugging
into `usd_bevy` for visualization. Robotics research workflows that today
require an NVIDIA box become possible on any modern laptop.

---

## 1. Goals and non-goals

### 1.1 Goals (what we will deliver)

- **Pure Rust.** No Python runtime, no Warp, no CUDA-only paths. `cargo build`
  on a fresh toolchain produces a working crate.
- **wgpu primary backend.** Vulkan / Metal / DX12 / WebGPU are all in scope
  for the GPU path. No vendor lock-in.
- **Mixed precision by design.** `f64` on the CPU side (state, builder,
  importer, USD I/O). `f32` in GPU kernels. Conversions at the buffer
  boundary, never at the math-op boundary.
- **Engine-neutral data model.** `Model` / `State` / `Control` / `Contacts`
  shape mirrors Newton's, because that shape is good and adapters depend on
  it.
- **One reference solver per major regime** before broadening:
  rigid-unconstrained (semi-implicit Euler), rigid-articulated (Featherstone),
  rigid-with-contacts (XPBD), particles/cloth (XPBD).
- **A USD adapter sibling crate** (`usd_molla`) that mirrors `usd_rapier`'s
  role: read marker components emitted by `usd_bevy`, produce a Newton
  `Model`, drive the sim, write back state.
- **Trajectory tests against Python Newton.** For every solver we ship, there
  is a reproducible test that runs the same scene through both engines and
  asserts numerical agreement within tolerance.
- **`no_std`-compatible math core.** The math crate compiles without `std` so
  it can be reused in shaders (via `cubecl` or codegen) and in embedded
  contexts.

### 1.2 Non-goals (what we will explicitly not deliver, at least not soon)

- **No 1:1 Newton port.** We are not reproducing all 8 solvers, hydroelastic
  contacts, MPM, or Style3D cloth in v1.
- **No MuJoCo XLA equivalent.** That is a multi-year effort and not needed
  for the molla use case.
- **No autodiff in v1.** Differentiability is a stage-9+ topic, not a
  blocking concern for stage 0–6.
- **No `f64` GPU path.** WGSL has no `f64`. We will not invest in a Vulkan-only
  `f64` extension path until and unless a workload actually requires it.
- **No Python bindings in v1.** A `pyo3` shim is a stage-10+ topic if at all.
  Newton's value to molla is its Rust nativeness.
- **No own renderer.** Visualization piggybacks on `usd_bevy` and `rerun`.
- **No general LCP / NLP solver.** We use position-based constraints (XPBD)
  and reduced-coordinate articulation (Featherstone). MuJoCo-style hard
  constraint handling is out of scope.
- **No URDF/MJCF beyond what robotics workflows need.** We will not implement
  every esoteric corner of those formats. USD is the primary import path.

### 1.3 Quality bars

Every shipped feature must have:

1. A trajectory or unit test against the Newton reference (where applicable).
2. A doctest or example demonstrating idiomatic use.
3. A bench in `criterion` measuring cost per step.
4. CPU-fallback parity: the same algorithm runs on CPU (single-threaded) for
   debugging, and produces bit-identical results modulo floating-point
   ordering nondeterminism on GPU.

---

## 2. Locked decisions

These are decided. Changes require an explicit revision to this section.

| # | Decision | Rationale |
|---|---|---|
| D1 | Pure Rust, no Python, no Warp dependency | Scope of the project |
| D2 | `wgpu` is the primary GPU backend | Cross-vendor portability |
| D3 | `f64` on CPU host-side, `f32` in GPU kernels | Industry-standard for GPU physics; WGSL has no `f64` |
| D4 | Crate name: `molla` (engine), `usd_molla` (USD adapter) | Mirrors existing `rapier3d` / `usd_rapier` naming |
| D5 | Crate location: `crates/molla/` and `crates/usd_molla/` inside the molla workspace | Co-located with `usd_schema`, `usd_rapier`, `usd_bevy` |
| D6 | License: Apache-2.0 (matches Newton, maximally compatible) | Newton itself is Apache-2.0; we want the option to upstream |
| D7 | `glam` for vector/quat/mat math (`glam::DVec3`, `glam::DQuat`, `glam::DMat3`) on host, `glam::Vec3` on device | Battle-tested, fast, has both `f32` and `f64` types |
| D8 | Edition 2024, MSRV = stable | Matches the rest of the molla workspace |
| D9 | The Python Newton tree (`./xtra/newton/`) is reference and oracle, not a port source | Apache-2.0 lets us read it; we reimplement clean-room from APIs and behavior |
| D10 | `Model` data layout follows Newton's grouping (per-particle, per-body, per-shape, per-joint, per-world) | The shape is good and downstream adapters depend on it |
| D11 | Errors are `thiserror`-typed; no `anyhow` in the engine crate (only in apps) | Library crate hygiene |
| D12 | Async only at the top-level `Engine::step` boundary, never in the math/kernel layer | Avoids coloring functions; matches `wgpu`'s natural seams |

## 3. Open questions

These need to be resolved during stage 0; each is tagged with the stage that
needs the answer.

| # | Question | Needed by | Default lean |
|---|---|---|---|
| Q1 | Direct WGSL or CubeCL for kernel authoring? | Stage 2 | Start with raw WGSL; revisit CubeCL once ergonomics hurt |
| Q2 | One queue for compute or compute + transfer queue split? | Stage 2 | Single queue until bandwidth measurements say otherwise |
| Q3 | How do we ship WGSL — embedded strings, build script, or `naga_oil`? | Stage 2 | `naga_oil` for module composition + `include_str!` for shaders |
| Q4 | CPU broad-phase: BVH, SAP, or grid? | Stage 4 | SAP first, swap to BVH if articulated worlds with disparate AABB sizes hurt |
| Q5 | Which articulation algorithm: Featherstone (ABA) or CRBA? | Stage 3 | Featherstone — cleaner GPU port, what Newton uses |
| Q6 | Differentiability strategy when we get there | Stage 9+ | Manual adjoints for XPBD; never attempt autodiff over articulation |
| Q7 | Do we ship a Python binding? | Stage 10+ | No, unless someone explicitly asks |
| Q8 | Workspace member or standalone repo? | Stage 0 | Workspace member, can split later when stable |
| Q9 | Async API at `Engine::step` boundary? | Stage 1 | Sync API; user pumps a `wgpu::Device` themselves |

---

## 4. Reference: what Newton actually is

A condensed map of the Python Newton tree at `xtra/newton/`. Use this as the
shopping list, and use it to *not* reimplement what we don't need.

### 4.1 Top-level layout

```
newton/_src/
├── core/           ~ 200 lines    Axis, MAXVAL, Devicelike
├── math/           ~ 400 lines    spatial.py — twists, wrenches, transforms
├── geometry/       ~ 25,000 lines KEY: collision, BVH, SDF, contact reduction
├── sim/            ~ 16,000 lines KEY: Model, State, Control, Contacts, ModelBuilder
├── solvers/        ~ 17,000 lines 8 solvers (XPBD, VBD, Featherstone, …)
├── actuators/      ~ 1,000 lines  PD, delays, controllers
├── sensors/        ~ 1,500 lines  IMU, raycast, tiled camera, contact, frame transform
├── usd/            ~ 1,000 lines  USD physics schema reader
├── utils/          ~ 5,000 lines  URDF/MJCF/USD import, mesh ops, heightfield
└── viewer/         ~ 2,000 lines  GL, Rerun, Viser, USD, file
```

### 4.2 Solver inventory

| Solver | Lines | Purpose | Port priority |
|---|---|---|---|
| `semi_implicit` | ~2,000 | Plain Euler, no constraints | Stage 1 (cheap baseline) |
| `xpbd` | ~3,400 | Position-based, particles + rigid + cloth | Stage 4–5 (the workhorse) |
| `featherstone` | ~2,000 | Articulated rigid (ABA) | Stage 3 |
| `vbd` | ~10,000 | Vertex Block Descent (cloth/soft) | Stage 7+ if needed |
| `mujoco` | ~2,000 | Newton's MuJoCo Warp port | Out of scope (would need MuJoCo) |
| `style3d` | ~2,000 | Production cloth | Out of scope |
| `implicit_mpm` | ~5,000 | Material Point Method | Out of scope |
| `kamino` | ~variable | Research solver | Out of scope |

### 4.3 What we will use as a reference oracle

For each solver we port, the test harness loads the same scene description in
both engines and compares trajectories. The reference Python `newton` runs in
a venv; the harness shells out to it. See section 17.

---

## 5. Workspace layout and crate boundaries

### 5.1 Crate structure

```
molla/
├── Cargo.toml                   workspace
├── crates/
│   ├── usd_schema/              (existing)  pure-Rust USD readers
│   ├── usd_rapier/              (existing)  Rapier adapter
│   ├── usd_bevy/                (existing)  Bevy plugin
│   ├── molla/               (NEW)       the engine
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── core/            types, ids, errors
│   │   │   ├── math/            DVec3, SpatialVector, Transform
│   │   │   ├── storage/         Buffer<T>, host/device mirrors
│   │   │   ├── compute/         wgpu pipeline / bind group plumbing
│   │   │   ├── kernels/         WGSL files + Rust dispatch wrappers
│   │   │   ├── geometry/        shapes, BVH, broad/narrow phase
│   │   │   ├── sim/             Model, State, Control, Contacts, ModelBuilder
│   │   │   ├── solvers/         semi_implicit, xpbd, featherstone
│   │   │   ├── actuators/       PD, target, force
│   │   │   ├── sensors/         imu, raycast, contact
│   │   │   └── prelude.rs
│   │   ├── shaders/             WGSL source (one file per kernel module)
│   │   ├── tests/               integration tests, oracle tests
│   │   └── benches/             criterion benches
│   └── usd_molla/              (NEW)       USD physics → molla Model
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── builder.rs       UsdPhysics markers → ModelBuilder
│           └── projection.rs    Newton State → USD time-sampled attrs
├── xtra/
│   └── newton/                  Python reference (read-only)
└── PLAN.md                      this file
```

### 5.2 Dependency rules

- `molla` has **zero** dependency on Bevy, Rapier, USD, or any I/O format.
  It only knows about its own `Model` / `State`.
- `usd_molla` depends on `molla` and `usd_schema`. No Bevy.
- `usd_bevy` already depends on Bevy. It can optionally depend on `usd_molla`
  behind a feature flag (`features = ["newton"]`) so users who want the
  Newton physics adapter pull it in explicitly.
- The viewer binary `molla` depends on `usd_bevy` only; the choice of
  `usd_rapier` vs `usd_molla` is a feature flag.

```
molla  ←  usd_molla  ←  usd_bevy  ←  molla (bin)
                                ↑
                            usd_rapier
                                ↑
                            usd_schema
```

### 5.3 Cargo.toml additions to workspace root

```toml
[workspace]
members = [
    "crates/usd_schema",
    "crates/usd_rapier",
    "crates/usd_bevy",
    "crates/molla",   # NEW
    "crates/usd_molla",  # NEW (added in stage 5+)
]
```

---

## 6. Tech stack

### 6.1 Core dependencies (engine crate)

| Crate | Version | Purpose | Notes |
|---|---|---|---|
| `wgpu` | `0.22+` (latest stable) | GPU compute | Track `bevy`'s pinned version where possible |
| `bytemuck` | `1` | `Pod`/`Zeroable` traits for buffer types | Required by `wgpu` |
| `glam` | `0.29+` | Vec/Mat/Quat math, both `f32` and `f64` | `features = ["bytemuck"]` |
| `thiserror` | `2` | Error types | Library hygiene |
| `tracing` | `0.1` | Structured logging | Matches molla |
| `slotmap` | `1` | Stable IDs for builder-time entities | For `BodyId`, `ShapeId`, etc. |
| `pollster` | `0.3` | Block on `wgpu` futures in sync API | Until/unless we go async |
| `naga_oil` | latest | WGSL module composition | For shader includes |

### 6.2 Test/bench dependencies

| Crate | Purpose |
|---|---|
| `criterion` | Benchmarks |
| `approx` | Numerical comparisons in tests |
| `pretty_assertions` | Diffs in failed asserts |
| `serde` + `serde_json` | Snapshot serialization for golden trajectory tests |

### 6.3 Versioning policy

- We pin `wgpu` to whatever Bevy 0.18 in molla pins, to avoid having two
  `wgpu` versions in the linked binary.
- We pin `glam` to whatever Bevy and Rapier transitively need, for the same
  reason.
- Everything else is allowed to float on minor versions.

---

## 7. Math layer

### 7.1 Host-side types

The host math layer uses **`f64`** uniformly. Builder, importer, USD I/O,
sensors, actuator math, and CPU fallback solvers all operate in `f64`.

Primitives (newtype wrappers around `glam` where helpful):

```rust
// crates/molla/src/math/mod.rs
pub type Real = f64;                          // host-side scalar
pub type Vec3 = glam::DVec3;                  // 3D vector
pub type Quat = glam::DQuat;                  // unit quaternion
pub type Mat3 = glam::DMat3;                  // 3x3 matrix

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialVector {
    pub linear: Vec3,
    pub angular: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialInertia {
    pub mass: Real,
    pub com: Vec3,
    pub inertia: Mat3,
}
```

### 7.2 Device-side types

The device side uses **`f32`** uniformly. Mirrors of the host types live in
WGSL and are reflected in Rust as `#[repr(C)]` POD structs:

```rust
// crates/molla/src/math/gpu.rs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransform {
    pub position: [f32; 3],
    pub _pad: f32,
    pub rotation: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpatialVector {
    pub linear: [f32; 3],
    pub _pad0: f32,
    pub angular: [f32; 3],
    pub _pad1: f32,
}
```

### 7.3 Conversion contract

Conversion happens **only at the buffer boundary**:

- `Buffer::upload(host_slice: &[T])` does the f64→f32 narrowing if `T` is a
  host type and the device storage is the corresponding GPU type.
- `Buffer::download(host_slice: &mut [T])` widens f32→f64 on read-back.
- No kernel, no math operation, ever sees a mixed-precision argument.

This keeps the precision boundary obvious in code review and easy to audit.

### 7.4 What we will and will not implement

| Thing | Yes/no | Where |
|---|---|---|
| `Vec3` add/sub/dot/cross/length/normalize | Yes | `glam` gives us this |
| `Quat` mul, conjugate, slerp, from_axis_angle | Yes | `glam` |
| `Transform` compose, inverse, transform_point | Yes | Newtype methods |
| `SpatialVector` ops (twist/wrench algebra) | Yes | Hand-written |
| 6x6 spatial inertia ops (Featherstone) | Yes | Hand-written, stage 3 |
| 4x4 homogeneous matrices | No | Use `Transform` instead |
| Arbitrary-precision math | No | f64 host / f32 device, period |

---

## 8. Storage and buffer abstraction

### 8.1 Goals of the abstraction

- One type, `Buffer<T>`, that owns either CPU or GPU memory or both.
- Zero-copy where the layout permits, explicit narrowing/widening where it
  doesn't.
- Same code path for tests (CPU mirror) and production (GPU + optional
  download).
- Sized at construction; no resizing during simulation. (Re-builds need a new
  `Model`.)

### 8.2 Sketch

```rust
// crates/molla/src/storage/mod.rs
pub struct Buffer<T: BufferElement> {
    /// Authoritative host copy in T (host type, typically f64).
    host: Box<[T]>,
    /// Device buffer holding T::Gpu (f32 mirror), if a GPU is attached.
    device: Option<wgpu::Buffer>,
    /// Whether host and device are in sync.
    state: SyncState,
}

pub trait BufferElement: Copy + 'static {
    type Gpu: bytemuck::Pod;
    fn to_gpu(&self) -> Self::Gpu;
    fn from_gpu(gpu: &Self::Gpu) -> Self;
}

impl BufferElement for Vec3 { type Gpu = [f32; 3]; … }
impl BufferElement for Transform { type Gpu = GpuTransform; … }
impl BufferElement for SpatialVector { type Gpu = GpuSpatialVector; … }

#[derive(Copy, Clone, Eq, PartialEq)]
enum SyncState { HostOnly, DeviceOnly, BothInSync, HostDirty, DeviceDirty }
```

### 8.3 Operations

```rust
impl<T: BufferElement> Buffer<T> {
    pub fn from_slice(host: &[T]) -> Self;
    pub fn zeros(len: usize) -> Self;

    pub fn host(&self) -> &[T];                       // requires HostOnly | BothInSync | HostDirty
    pub fn host_mut(&mut self) -> &mut [T];           // marks HostDirty
    pub fn device(&self) -> Option<&wgpu::Buffer>;    // requires DeviceOnly | BothInSync | DeviceDirty

    pub fn upload(&mut self, queue: &wgpu::Queue);    // host -> device
    pub fn download(&mut self, device: &wgpu::Device, queue: &wgpu::Queue);  // device -> host

    pub fn ensure_device(&mut self, device: &wgpu::Device);
    pub fn len(&self) -> usize;
}
```

### 8.4 Why a single owning type

Newton's Python uses `wp.array` everywhere. We need the same conceptual
ergonomics — one thing you pass around, kernels read it, you can `.numpy()`
to inspect — but bound to Rust's ownership. `Buffer<T>` is that.

---

## 9. Compute layer (kernels)

### 9.1 Kernel authoring strategy

**Stage 1**: WGSL source files in `crates/molla/shaders/`, one file per
kernel module, included via `include_str!`. Compose with `naga_oil` for
shared utility modules (math, atomics, etc.).

**Revisit at stage 6**: if WGSL ergonomics are killing us (lots of duplicated
math between f64-host and f32-device), we evaluate **CubeCL** as a kernel
DSL. CubeCL produces wgpu-compatible kernels from Rust source. We do not
adopt it preemptively — the migration risk is high and the Rust-DSL ecosystem
is still young.

### 9.2 Shader organization

```
shaders/
├── lib/
│   ├── math.wgsl                Vec3, Quat, Transform helpers (f32)
│   ├── spatial.wgsl             SpatialVector ops
│   ├── atomics.wgsl             atomic helpers (counter, append-buffer)
│   └── prng.wgsl                deterministic prng for stochastic kernels
├── integrate/
│   ├── particles.wgsl           semi-implicit Euler particles
│   └── bodies.wgsl              semi-implicit Euler rigid bodies
├── broad_phase/
│   ├── nxn.wgsl                 brute-force AABB pairs
│   └── sap.wgsl                 sweep-and-prune
├── narrow_phase/
│   ├── sphere_sphere.wgsl
│   ├── sphere_box.wgsl
│   ├── gjk.wgsl
│   └── mpr.wgsl
├── solver/
│   ├── xpbd_distance.wgsl
│   ├── xpbd_contact.wgsl
│   └── featherstone.wgsl
└── sensors/
    ├── raycast.wgsl
    └── imu.wgsl
```

Each shader is paired with a Rust dispatch wrapper:

```rust
// crates/molla/src/kernels/integrate/bodies.rs
pub struct IntegrateBodiesKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IntegrateBodiesKernel {
    pub fn new(device: &wgpu::Device) -> Self { … }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &State,
        gravity: Vec3,
        dt: Real,
    ) { … }
}
```

### 9.3 Bind group layout convention

For consistency and to make bind group caching easy:

| Group | Slot | Purpose |
|---|---|---|
| 0 | 0..15 | Read-only `Model` data (immutable across the step) |
| 1 | 0..15 | Read-write `State` data (modified by the kernel) |
| 2 | 0..3 | Constants / push-constant analog (small uniform buffer) |
| 3 | 0..3 | Scratch buffers (kernel-local intermediates) |

### 9.4 What WGSL does and does not give us

We need to be clear-eyed about WGSL's limits up front. These are the things
that will hurt:

- **No `f64`.** Everything device-side is `f32`. (Section 7.)
- **No pointers into storage buffers from arbitrary functions.** Functions
  that take `ptr<storage, T, read_write>` work but with restrictions.
- **Limited atomics.** `atomic<u32>`, `atomic<i32>` only. No `atomic<f32>`.
  Floating-point reductions need workarounds (compare-and-swap loops or
  pre-sorted scatter).
- **No subgroup/wave intrinsics in WGSL spec yet.** wgpu has experimental
  `subgroup` extensions; we will not depend on them in v1.
- **No printf/debugger.** Debug via writing to a debug buffer and reading
  back. The CPU-fallback path is our real debugger.

### 9.5 CPU fallback contract

Every GPU kernel has a CPU implementation in the same Rust module that
produces the same output (modulo float-op ordering). The `Engine` exposes a
`Backend::Cpu | Backend::Gpu` selector. Tests run both; CI compares.

---

## 10. Data model: `Model`, `State`, `Control`, `Contacts`

### 10.1 The four pillars (mirrors Newton)

- **`Model`**: static, immutable after build. Mass properties, geometry,
  joint topology, shape data, world groupings.
- **`State`**: time-varying. Positions, velocities, forces. Mutated by
  solver each step.
- **`Control`**: user inputs. Joint targets, applied wrenches.
- **`Contacts`**: produced by collision pipeline, consumed by solver. Reset
  each step.

### 10.2 `Model` — sketch

```rust
// crates/molla/src/sim/model.rs
pub struct Model {
    // ── Particles ────────────────────────────────────────────────
    pub particle_count: usize,
    pub particle_mass: Buffer<Real>,
    pub particle_inv_mass: Buffer<Real>,
    pub particle_radius: Buffer<Real>,
    pub particle_flags: Buffer<u32>,
    pub particle_world: Buffer<i32>,         // -1 = global

    // ── Rigid bodies ─────────────────────────────────────────────
    pub body_count: usize,
    pub body_mass: Buffer<Real>,
    pub body_inv_mass: Buffer<Real>,
    pub body_inertia: Buffer<Mat3>,
    pub body_inv_inertia: Buffer<Mat3>,
    pub body_com: Buffer<Vec3>,
    pub body_flags: Buffer<u32>,
    pub body_world: Buffer<i32>,

    // ── Shapes ──────────────────────────────────────────────────
    pub shape_count: usize,
    pub shape_body: Buffer<i32>,             // owning body, -1 if static
    pub shape_transform: Buffer<Transform>,  // local to body
    pub shape_geometry: Vec<ShapeGeometry>,  // tagged enum on host
    pub shape_flags: Buffer<u32>,
    pub shape_material: Buffer<MaterialIndex>,

    // ── Joints ──────────────────────────────────────────────────
    pub joint_count: usize,
    pub joint_type: Buffer<JointType>,
    pub joint_parent: Buffer<i32>,           // -1 = world
    pub joint_child: Buffer<i32>,
    pub joint_X_p: Buffer<Transform>,        // attachment in parent
    pub joint_X_c: Buffer<Transform>,        // attachment in child
    pub joint_axis: Buffer<Vec3>,
    pub joint_limits: Buffer<JointLimit>,
    pub joint_dof_offset: Buffer<i32>,       // CSR-style indexing
    pub joint_coord_offset: Buffer<i32>,

    // ── Worlds ──────────────────────────────────────────────────
    pub world_count: usize,
    pub gravity: Buffer<Vec3>,               // per-world

    // ── Materials ───────────────────────────────────────────────
    pub material_friction: Buffer<Real>,
    pub material_restitution: Buffer<Real>,
    pub material_density: Buffer<Real>,

    // ── Backend ─────────────────────────────────────────────────
    pub(crate) device: Option<Arc<wgpu::Device>>,
    pub(crate) queue: Option<Arc<wgpu::Queue>>,
}
```

### 10.3 `State` — sketch

```rust
// crates/molla/src/sim/state.rs
pub struct State {
    pub particle_q: Buffer<Vec3>,       // positions
    pub particle_qd: Buffer<Vec3>,      // velocities
    pub particle_f: Buffer<Vec3>,       // forces

    pub body_q: Buffer<Transform>,      // pose
    pub body_qd: Buffer<SpatialVector>, // twist
    pub body_f: Buffer<SpatialVector>,  // applied wrench

    pub joint_q: Buffer<Real>,          // generalized coords
    pub joint_qd: Buffer<Real>,         // generalized velocities

    // Optional / extended (allocated on demand)
    pub body_qdd: Option<Buffer<SpatialVector>>,
    pub body_q_prev: Option<Buffer<Transform>>,
}
```

### 10.4 `Control` — sketch

```rust
pub struct Control {
    pub joint_target: Buffer<Real>,      // for position/velocity-mode actuators
    pub joint_force: Buffer<Real>,       // direct torque/force
    pub body_external_f: Buffer<SpatialVector>,  // applied wrenches
}
```

### 10.5 `Contacts` — sketch

```rust
pub struct Contacts {
    pub count: AtomicCounter,            // GPU-resident counter
    pub capacity: usize,
    pub a_index: Buffer<u32>,            // body or shape index
    pub b_index: Buffer<u32>,
    pub point: Buffer<Vec3>,             // world-space contact point
    pub normal: Buffer<Vec3>,            // pointing from a to b
    pub depth: Buffer<Real>,
    pub friction: Buffer<Real>,
    pub restitution: Buffer<Real>,
}
```

### 10.6 World grouping

`particle_world` / `body_world` allow a single `Model` to hold many
independent physics worlds (RL training pattern). Index `-1` means "global"
(static obstacles, fixed bodies). Broad-phase respects this: pairs are only
generated within the same world or against `-1`.

---

## 11. `ModelBuilder` and authoring API

### 11.1 Public API surface

```rust
// crates/molla/src/sim/builder.rs
pub struct ModelBuilder {
    // Internal storage in Vec<…> (no Buffer<> until finalize)
    …
}

impl ModelBuilder {
    pub fn new() -> Self;

    // ── Bodies ────────────────────────────────────────────
    pub fn add_body(&mut self, params: BodyParams) -> BodyId;

    // ── Shapes (auto-attach to last body or by id) ────────
    pub fn add_shape_sphere(&mut self, body: BodyId, radius: Real, params: ShapeParams) -> ShapeId;
    pub fn add_shape_box(&mut self, body: BodyId, half_extents: Vec3, params: ShapeParams) -> ShapeId;
    pub fn add_shape_capsule(&mut self, body: BodyId, radius: Real, half_height: Real, params: ShapeParams) -> ShapeId;
    pub fn add_shape_mesh(&mut self, body: BodyId, mesh: Arc<Mesh>, params: ShapeParams) -> ShapeId;

    // ── Joints ────────────────────────────────────────────
    pub fn add_joint_revolute(&mut self, params: RevoluteParams) -> JointId;
    pub fn add_joint_prismatic(&mut self, params: PrismaticParams) -> JointId;
    pub fn add_joint_fixed(&mut self, parent: BodyId, child: BodyId, params: FixedParams) -> JointId;
    pub fn add_joint_free(&mut self, child: BodyId) -> JointId;

    // ── Particles ─────────────────────────────────────────
    pub fn add_particle(&mut self, params: ParticleParams) -> ParticleId;
    pub fn add_particle_grid(&mut self, …) -> Range<ParticleId>;

    // ── Worlds ───────────────────────────────────────────
    pub fn set_world_gravity(&mut self, world: WorldId, g: Vec3);

    // ── Importers ────────────────────────────────────────
    pub fn add_urdf(&mut self, path: &Path, opts: ImportOpts) -> Result<…>;
    pub fn add_usd(&mut self, stage: &usd_schema::Stage, opts: ImportOpts) -> Result<…>;
    pub fn add_mjcf(&mut self, path: &Path, opts: ImportOpts) -> Result<…>;

    // ── Finalize ─────────────────────────────────────────
    pub fn build(self) -> Result<Model>;
    pub fn build_with_device(self, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Model>;
}
```

### 11.2 Naming convention (matches Newton)

Newton uses prefix-first naming for autocompletability:
`add_shape_sphere` not `add_sphere_shape`. We follow the same rule. Spelled
out in `AGENTS.md` for newton; we'll do the same.

### 11.3 ID stability

Builder-time IDs (`BodyId`, `ShapeId`, …) are stable across the build. They
are simple `u32` newtypes with `slotmap` semantics. After `build()`, IDs are
indices into the `Model`'s arrays. The mapping is preserved.

### 11.4 World metadata

The builder tracks a current "world index". `builder.with_world(2, |b| {
b.add_body(…); … })` is the ergonomic way to bulk-add to a specific world.

---

## 12. Solver progression

### 12.1 Solver trait

```rust
// crates/molla/src/solvers/mod.rs
pub trait Solver: Send + Sync {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<(), SolverError>;
}
```

### 12.2 Roadmap

| Stage | Solver | Purpose | LOC budget |
|---|---|---|---|
| 1 | `SemiImplicitSolver` (rigid only, no contacts) | Baseline integrator | ~500 |
| 2 | Same, ported to GPU | Validate the compute layer | (kernel only) |
| 3 | `FeatherstoneSolver` | Articulated reduced-coordinate rigid | ~3,000 |
| 4 | `XpbdSolver` (rigid contacts) | Constraints, stable contacts | ~3,000 |
| 5 | `XpbdSolver` (particles, distance, bending) | Cloth and rope | ~2,000 |
| 7+ | `VbdSolver` | Soft bodies (if needed) | (large) |

### 12.3 Featherstone (stage 3, in detail)

Featherstone's Articulated Body Algorithm (ABA) computes joint accelerations
from joint forces in O(n) for an n-body chain. The implementation strategy:

1. **CPU first.** Implement and validate against Newton's reference on a
   pendulum and a 6-DOF arm. Use `glam::DMat3` for spatial inertia ops; we
   do not need a full 6×6 spatial inertia type — we exploit structure.
2. **GPU port.** ABA's outer loop is sequential along a chain; parallelism
   is across articulations (worlds). One thread per articulation, sequential
   inside. This trades arithmetic intensity for parallelism — appropriate
   for RL-style "many worlds" workloads, less ideal for one big robot.
3. **Hybrid.** For "one robot per world" with N>=64 worlds, GPU wins. For
   one robot, single-thread CPU may match or beat GPU due to dispatch
   overhead. Engine picks based on `world_count >= threshold` (configurable;
   default 16).

### 12.4 XPBD (stage 4–5, in detail)

XPBD (eXtended Position-Based Dynamics) runs:
1. Predict positions from velocities + external forces.
2. For each iteration (typically 5–20):
   - Project each constraint (distance, contact, joint).
3. Update velocities from position deltas.

This is **embarrassingly parallel within a constraint type**, with the caveat
that constraints sharing a body conflict. Newton solves this with **graph
coloring**: each color is a set of mutually non-conflicting constraints,
colors are processed sequentially, threads within a color in parallel.

Our port:
- CPU graph coloring at builder time. Colors are stored in `Model`.
- Each color is a separate compute dispatch on the GPU.
- Per-color kernel reads body positions, computes correction, writes
  position update. No atomics needed because of the coloring guarantee.

---

## 13. Collision pipeline

### 13.1 Stages of the pipeline

```
[shapes]              [bvh / sap]               [pair list]
   │                       │                         │
   ▼                       ▼                         ▼
[update AABBs] ─→ [broad phase] ─→ [filter / dedupe] ─→ [narrow phase] ─→ [contact reduction] ─→ [Contacts]
```

### 13.2 Stage 4 (first real collision) scope

- Shape types: sphere, box, capsule, plane (static).
- Broad phase: SAP (single-axis sort + sweep). Newton's `broad_phase_sap.py`
  is ~700 lines; ours will be similar.
- Narrow phase: sphere-sphere, sphere-box, sphere-plane, sphere-capsule,
  capsule-capsule, box-box (SAT). No GJK in stage 4.
- Filtering: same world or `-1` only; collision groups (bitmasks).
- Reduction: simple, no hydroelastic.

### 13.3 Stage 6 expansion

- GJK + EPA for general convex pairs.
- Mesh-shape via BVH.
- Optional MPR alternative.
- Convex hull preprocessing for meshes (using `qhull` or hand-rolled).

### 13.4 What we explicitly punt

- Hydroelastic contacts (Newton's `sdf_hydroelastic.py` is 1,800 lines).
- Multi-contact reduction (`contact_reduction_global.py` is 1,580 lines —
  worth studying but not porting in v1).
- SDF-based contact (`sdf_contact.py` is 1,500 lines).

These are real differentiators of Newton and we may want them later, but not
for the molla-shaped use case.

---

## 14. Importers (URDF, MJCF, USD)

### 14.1 USD (primary)

Path of least resistance: `usd_schema` already exists in this workspace and
parses USD physics opinions. `usd_molla::ModelBuilderExt::add_usd_stage`
walks a `usd_schema::Stage`, finds prims with `UsdPhysics` schemas, and
emits the corresponding `ModelBuilder::add_*` calls.

The `PLAN_USDPHYSICS.md`-shaped marker components from the previous plan
become the on-the-wire format: `usd_bevy` emits markers, `usd_molla`
consumes them. (The previous plan is not wasted work — it's stage 5 of this
plan, with `usd_molla` as the adapter.)

### 14.2 URDF

A focused subset: links, inertials, joints (revolute, prismatic, fixed,
continuous), visual + collision geometry. We use the `urdf-rs` crate or
hand-roll a parser (it's only ~300 lines for the subset we care about).

Stage 3 deliverable: load Franka Panda URDF, run Featherstone, compare
trajectory to Newton.

### 14.3 MJCF

Lower priority. Stage 8+ if there's demand. MJCF's quirks (default classes,
sensor stanzas, contact pair definitions) are nontrivial.

---

## 15. Adapters (USD, Bevy, viewer)

### 15.1 `usd_molla` crate

Public surface:

```rust
// crates/usd_molla/src/lib.rs
pub fn build_model_from_stage(
    stage: &usd_schema::Stage,
    opts: ImportOpts,
) -> Result<molla::Model>;

pub fn project_state_to_stage(
    stage: &mut usd_schema::Stage,
    model: &molla::Model,
    state: &molla::State,
    time: f64,
);
```

- Read path: `Stage` → `ModelBuilder` calls → `Model`.
- Write path: per-step, write `body_q` back as time-sampled USD attributes
  on the corresponding prims.

### 15.2 Bevy integration

The existing `usd_bevy` plugin emits marker components from `UsdPhysics`
schemas. We add an optional system set `NewtonPhysicsSystems` (feature-gated
on `newton`) that:

1. On `OnAdd<UsdPhysicsScene>`: spawn a `molla::Engine` resource.
2. On `OnAdd<UsdRigidBody>`, `UsdCollider`, etc.: build the `Model` lazily.
3. Each frame: copy controls from Bevy components → `Control`, step the
   engine, copy `body_q` → `Transform` components.

The Bevy world is never authoritative about physics state — `molla` is.
Bevy only mirrors for rendering.

### 15.3 Viewer

For development, we ship a minimal example using `rerun` to visualize the
state each step. No own viewer code; `rerun` handles the rendering.

---

## 16. Differentiability strategy

**Stage 9+. Not in scope for v1.** But we make design decisions now that don't
preclude it:

- All kernels are pure functions of input buffers → output buffers. No
  hidden state.
- The `Solver` trait already takes `state_in` and `state_out` separately.
- We avoid in-place mutation in solvers wherever possible.

When we get to it, the strategy is:

1. **Manual adjoints for XPBD only.** XPBD has well-known analytic adjoints.
2. **Treat the engine as a differentiable black box** at the level of one
   `step`. User code wraps `step` in their own AD framework (Burn / Candle)
   and we provide the VJP.
3. **No autodiff over articulation.** Featherstone's adjoint is published
   but painful; if a user needs differentiable articulation, they should
   stay on MuJoCo XLA / Brax.

---

## 17. Testing strategy and the "newton oracle"

### 17.1 Test taxonomy

| Type | Where | What |
|---|---|---|
| Unit | `src/**/*.rs` `#[cfg(test)]` | Individual functions |
| Integration | `tests/*.rs` | Multi-component scenarios |
| Oracle (vs Newton) | `tests/oracle/*.rs` | Trajectory comparison |
| GPU determinism | `tests/gpu_determinism.rs` | Same scene → same output across runs (modulo float ordering) |
| CPU/GPU parity | `tests/cpu_gpu_parity.rs` | Same scene → same output regardless of backend |
| Bench | `benches/*.rs` | criterion |

### 17.2 The newton oracle harness

```rust
// crates/molla/tests/oracle/harness.rs
pub struct NewtonOracle {
    venv_python: PathBuf,
    workdir: PathBuf,
}

impl NewtonOracle {
    pub fn new() -> Result<Self>;

    pub fn run_scene(
        &self,
        scene_yaml: &str,           // declarative scene description
        steps: usize,
        dt: f64,
    ) -> Result<TrajectoryRecording>;
}

pub struct TrajectoryRecording {
    pub body_q: Vec<Vec<Transform>>,    // [step][body]
    pub body_qd: Vec<Vec<SpatialVector>>,
    pub joint_q: Vec<Vec<f64>>,
    pub joint_qd: Vec<Vec<f64>>,
}
```

Scene description is a small declarative YAML (or RON) format we share
between the Rust test and a Python script that runs Newton:

```yaml
gravity: [0, -9.81, 0]
bodies:
  - id: a
    pose: [0, 5, 0, 0, 0, 0, 1]
    mass: 1.0
    inertia: identity
    shape: { sphere: { radius: 0.5 } }
  - id: ground
    static: true
    shape: { plane: {} }
solver: semi_implicit
steps: 100
dt: 0.01
```

### 17.3 Tolerance policy

For each solver and each test scene, we record a tolerance:

```rust
assert_trajectory_close(
    &ours,
    &reference,
    Tolerance {
        position_abs: 1e-3,        // meters
        rotation_abs: 1e-3,        // radians (geodesic)
        velocity_abs: 1e-2,        // m/s
        max_step_drift: 1e-4,      // accumulating per step
    },
);
```

Tolerances are per-test, version-controlled, and explicitly documented when
relaxed (with a reason).

### 17.4 What goes in CI

- Unit + integration on every push.
- Oracle tests on a label (`run-oracle`) — they need the Python venv.
- Benches manually triggered (track regressions over time, don't gate on them).

---

## 18. Performance targets and benchmarking

### 18.1 Targets

| Scene | Target on integrated GPU | Target on discrete GPU |
|---|---|---|
| 1k spheres on plane, semi-implicit | 1 kHz | 5 kHz |
| 10k particles cloth, XPBD 20 iter | 60 Hz | 240 Hz |
| 1 humanoid, Featherstone | 1 kHz | 1 kHz (CPU faster anyway) |
| 1024 humanoids, Featherstone (RL) | 100 Hz | 1 kHz |
| 10k rigid bodies, XPBD contacts | 60 Hz | 240 Hz |

### 18.2 Benchmark structure

`benches/` uses `criterion` and tracks:
- per-step wall time
- per-step GPU time (via `wgpu` query timestamps)
- frame-to-frame variance (jitter)

Benchmarks run on CI on a known machine for trend tracking, not as gates.

### 18.3 Profiling tools

- `tracy` integration via `tracing-tracy` for CPU profiling.
- `wgpu` GPU timestamp queries for kernel-level timing.
- `RenderDoc` / `Xcode GPU Frame Capture` / `radv-perftest` for vendor-specific
  shader analysis.

---

## 19. CI, tooling, repo hygiene

### 19.1 CI matrix

| Job | OS | GPU | What runs |
|---|---|---|---|
| `lint` | linux | none | `cargo fmt`, `cargo clippy --all-targets` |
| `test-cpu` | linux/mac/win | none (llvmpipe / WARP) | unit + integration with `Backend::Cpu` |
| `test-gpu-vk` | linux | software Vulkan (lavapipe) | `Backend::Gpu` against lavapipe |
| `test-gpu-mtl` | mac | Metal (CI's GPU) | `Backend::Gpu` |
| `test-gpu-dx12` | win | WARP | `Backend::Gpu` |
| `oracle` | linux | software Vulkan + Python venv | trajectory tests |
| `bench-track` | linux self-hosted | discrete GPU | criterion, post results to artifact |

### 19.2 Lints

- `#![deny(unsafe_op_in_unsafe_fn)]`, `#![warn(missing_docs)]` on public items
  in `lib.rs`.
- `clippy::pedantic` allowed-but-tuned: deny `clippy::correctness`, warn the
  rest, allow `clippy::module_name_repetitions` etc.
- `cargo deny` for license + duplicate version checks.

### 19.3 Commit conventions

Per workspace CLAUDE.md: conventional commits, title-only, no signature, max
50 chars. `feat(newton): add semi-implicit solver`.

### 19.4 Documentation

- `cargo doc` builds with no warnings for `molla`.
- `examples/` directory with runnable scenes.
- `mdbook` user guide deferred to stage 6+.

---

## 20. Stage-by-stage roadmap

Each stage is **merge-ready** before the next begins. Stages are not weeks —
they're scopes. Estimates are calendar weeks for one developer working
half-time.

### Stage 0 — Foundation ✅ COMPLETE (2026-05-08)

**Goal**: empty crate that compiles, with the architecture decisions encoded
in code and CI green.

Deliverables:
- `crates/molla/` scaffold with `Cargo.toml`, `lib.rs`, module skeletons
  matching section 5.1.
- `crates/molla/src/math/` complete: `Vec3`, `Quat`, `Transform`,
  `SpatialVector`, `SpatialInertia`. Unit tests.
- `crates/molla/src/storage/` complete: `Buffer<T>` with CPU-only mode
  working. GPU mode stubbed with `todo!()` panics.
- `crates/molla/src/compute/` skeleton: `Engine` struct that owns
  `wgpu::Instance`, `Adapter`, `Device`, `Queue`. `Engine::new_cpu()`,
  `Engine::new_gpu()`.
- `crates/molla/src/sim/` types defined as empty structs with the fields
  from section 10.
- CI: lint + cpu test on linux. Cargo doc builds.
- A "hello world" example that builds an `Engine` and prints the chosen
  adapter info.
- `oracle/` harness builds against a venv'd Newton.

Definition of done:
- `cargo test -p molla` passes on linux.
- `cargo doc -p molla` produces zero warnings.
- A `tests/oracle/smoke.rs` test successfully shells out to Python Newton
  and reads back a trivial trajectory (no comparison yet).

### Stage 1 — CPU semi-implicit baseline ✅ COMPLETE (2026-05-08)

**Goal**: drop spheres, integrate forward, match Newton's `semi_implicit`
solver to tolerance.

Deliverables:
- `ModelBuilder::add_body`, `add_shape_sphere`, `set_world_gravity`.
- `SemiImplicitSolver` (CPU). No collision yet — just gravity + integration.
- `Engine::step` end-to-end (CPU backend).
- Oracle test: `tests/oracle/free_fall.rs` — drop one sphere, run 100
  steps, assert trajectory matches Newton within 1e-6 m.
- Oracle test: `tests/oracle/free_fall_many.rs` — 100 spheres, same.

Definition of done:
- Single-sphere trajectory matches reference within 1e-6.
- 100-sphere case matches within 1e-5 (numerical reordering tolerance).
- `cargo bench --bench step_cpu` reports < 100µs per step for 100 spheres.

### Stage 2 — GPU semi-implicit (2–3 weeks)

**Goal**: same scene, GPU backend, on Vulkan/Metal/DX12.

Deliverables:
- `Buffer<T>::ensure_device`, `upload`, `download` working.
- `IntegrateBodiesKernel` in WGSL + Rust dispatch wrapper.
- `Engine::step` GPU path: upload → dispatch → (optional) download.
- CPU/GPU parity test: same scene on both backends, trajectories agree to
  1e-3 (f32→f64 narrowing tolerance).
- GPU oracle test on lavapipe in CI.

Definition of done:
- `Engine::new_gpu()` returns successfully on linux/lavapipe, mac/Metal,
  win/DX12 (covered by CI).
- 1k spheres at >= 1 kHz on a 2020-era discrete GPU.
- `cargo bench --bench step_gpu` runs.

### Stage 3 — Articulation (Featherstone) (4–6 weeks)

**Goal**: load Franka Panda URDF, simulate, match reference within tolerance.

Deliverables:
- Joint types: revolute, prismatic, fixed, free.
- `joint_q` / `joint_qd` plumbing.
- `FeatherstoneSolver` (CPU first).
- URDF importer (subset).
- `eval_fk`, `eval_ik` (Newton's `eval_*` analogs) — CPU only initially.
- Oracle test: pendulum, double pendulum, Franka Panda (all 100 steps,
  1ms dt).
- GPU port (parallelism across articulations).

Definition of done:
- Pendulum trajectory matches Newton's Featherstone within 1e-4 rad.
- Franka URDF loads, runs without contacts, matches reference.
- `cargo bench --bench feathestone_arm` reports per-step time budget met.

### Stage 4 — First collision pipeline (XPBD rigid contacts) (4–6 weeks)

**Goal**: spheres + boxes + planes collide and pile correctly.

Deliverables:
- `Model.shape_geometry` populated, AABB computation per shape.
- SAP broad-phase (CPU + GPU).
- Narrow-phase: sphere-sphere, sphere-box, sphere-plane, box-box (SAT).
- `Contacts` populated.
- `XpbdSolver` for contacts: position correction, friction, restitution.
- Graph-coloring builder pass.
- Oracle: stack of 10 boxes, 1s simulation, final positions match Newton's
  XPBD within 5mm.

Definition of done:
- 10-box stack settles, no jitter or interpenetration > 1mm.
- 1k spheres falling on a plane, 60Hz on integrated GPU.
- Friction parameter measurably affects sliding.

### Stage 5 — `usd_molla` crate (3–4 weeks)

**Goal**: load a USD file with `UsdPhysics` opinions, simulate, write back.

Deliverables:
- `crates/usd_molla/` scaffold.
- `build_model_from_stage` for `UsdRigidBody`, `UsdCollider`,
  `UsdPhysicsJoint`, `UsdMass`, `UsdPhysicsScene`, `UsdPhysicsMaterial`.
- `project_state_to_stage` for time-sampled writes.
- Integration test: round-trip a small USD scene through builder → step ×
  100 → write back → compare to a recorded golden.

Definition of done:
- A USD file authored in `molla` opens, runs, and the resulting USD has
  correct time samples.
- `usd_bevy` with `features = ["newton"]` swaps Rapier for newton in the
  viewer and the scene runs identically (modulo solver differences).

### Stage 6 — General convex contact (GJK + EPA) (3–5 weeks)

**Goal**: arbitrary convex shapes (cones, cylinders, convex hulls of meshes).

Deliverables:
- GJK in WGSL.
- EPA fallback for deep penetration.
- Convex hull builder for meshes (qhull port or `parry` reuse).
- Capsule, cone, cylinder primitives.

Definition of done:
- Cylinder-cylinder contact stable.
- Mesh (convex hull) on plane stable.

### Stage 7+ — XPBD particles & cloth, sensors, actuators

Deliverables grow from here based on what molla actually needs. Likely
order: cloth (XPBD particles + distance/bending), then IMU + raycast
sensors, then PD actuators.

### Stage 9+ — Differentiability

Manual adjoints for XPBD only. Out of scope until there's a concrete user.

### Stage 10+ — Python bindings

If at all. Newton's Python users probably stay on Python Newton; we serve
Rust users.

---

## 21. Risks and how we mitigate them

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | wgpu compute is too slow vs CUDA — discourages users | Medium | High | Bench early (stage 2), set realistic targets, communicate the tradeoff (portability over peak perf) |
| R2 | f32 precision insufficient for some workload | Low | Medium | Documented mitigations (local frames, renormalize quats); Vulkan-only `f64` extension path is feasible if it ever becomes critical |
| R3 | WGSL ergonomics make solvers painful to author | Medium | Medium | Stage 6 evaluation of CubeCL; module composition via `naga_oil` |
| R4 | Featherstone GPU parallelism not great for one-robot scenes | High | Low | Hybrid CPU/GPU dispatch based on `world_count`; this is well-understood |
| R5 | Oracle harness is flaky / Newton venv hard to set up in CI | Medium | Medium | Oracle is a CI label, not a gate; cache the venv; fall back to recorded golden trajectories |
| R6 | Collision pipeline is huge and a tarpit | High | High | Aggressive scope cuts (no hydroelastic, no SDF, no MPM). Reuse `parry` if it helps. |
| R7 | "Newton-equivalent" expectations from users we can't meet | Medium | Medium | README is explicit about scope: "a subset of Newton's rigid + cloth, on portable GPUs" |
| R8 | wgpu version churn breaks us against Bevy's pinned wgpu | High | Medium | Track Bevy's wgpu pin in workspace; document the pin in `Cargo.toml` |
| R9 | License/IP issue with reading Newton source | Low | High | Apache-2.0 license, attribution in README, reimplement clean from APIs and behavior; never copy substantial verbatim sections |
| R10 | Project loses momentum because the slope is too long | Real | Existential | Stages are sized so each is shippable on its own; molla can use `usd_rapier` while we're still at stage 3 |

---

## 22. Decision log

A running record of decisions made and reversed. Every entry: date, decision,
rationale, who decided.

| Date | Entry |
|---|---|
| 2026-05-08 | Project initiated. Locked decisions D1–D12 per section 2. |
| 2026-05-08 | Resolved precision question: `f64` host / `f32` device, no Vulkan-only `f64` GPU path in v1. WGSL has no `f64`; industry standard for GPU physics is `f32`. |
| 2026-05-08 | Crate names: `molla` (engine), `usd_molla` (USD adapter). Mirrors `rapier3d` / `usd_rapier` precedent. |
| 2026-05-08 | Workspace placement under `crates/` co-located with existing crates. Excluded `xtra/` reference Newton tree from workspace. |
| 2026-05-08 | Stage 0 complete. 19 tests pass, 0 doc warnings, 0 clippy warnings. wgpu pinned at 23 (later upgrade tracked separately). Switched test asserts from `approx` to `(a-b).length() < eps` because `glam::DVec3` doesn't impl `AbsDiffEq`; dropped `approx` from dev-deps. SI unit annotations use `(unit)` parens not `[unit]` because rustdoc treats brackets as broken intra-doc links. |
| 2026-05-08 | Stage 1 complete. 25 tests pass, 2 oracle tests ignored (require venv). `SemiImplicitSolver` (CPU) is a direct port of Newton's `integrate_rigid_body`. Free-fall trajectory matches the closed-form `g·dt²·n(n+1)/2` to within 1e-9. Bench: 4.92µs/step for 100 spheres (DoD target was <100µs; 20× under). `Engine::step` deferred to stage 2 — `Solver::step` is the canonical API and mirrors Newton; the Engine type is the GPU runtime, not a frontend wrapper. |

---

## Appendix A — How to read this document

- If you are a new contributor: start at section 4 (what Newton is), then 5
  (workspace layout), then 20 (roadmap).
- If you want to know what's next: section 20, find the stage marked active.
- If you want to know "why this and not that": section 2 (locked) and 21
  (risks).
- If you want to argue with a decision: open a PR that updates section 22
  with a new entry.

## Appendix B — Files in `xtra/newton/` worth re-reading

When working on each stage, re-read these specific reference files:

| Stage | Reference files in `xtra/newton/newton/_src/` |
|---|---|
| 1 | `solvers/semi_implicit/solver_semi_implicit.py`, `solvers/semi_implicit/kernels_body.py` |
| 2 | Same as 1 — ports the same kernels |
| 3 | `solvers/featherstone/solver_featherstone.py`, `solvers/featherstone/kernels.py`, `sim/articulation.py` |
| 4 | `solvers/xpbd/solver_xpbd.py`, `solvers/xpbd/kernels.py`, `geometry/broad_phase_sap.py`, `geometry/collision_primitive.py`, `sim/graph_coloring.py` |
| 5 | `usd/schemas.py`, `utils/import_usd.py` |
| 6 | `geometry/mpr.py`, `geometry/collision_convex.py` |

## Appendix C — Glossary

- **ABA** — Articulated Body Algorithm. Featherstone's O(n) forward dynamics.
- **CRBA** — Composite Rigid Body Algorithm. O(n²) inertia matrix; pairs with O(n³) solve.
- **XPBD** — eXtended Position-Based Dynamics. Position-correction iterative solver with stiffness as a Lagrange multiplier.
- **VBD** — Vertex Block Descent. Newton's specific cloth/soft-body solver.
- **Spatial vector** — 6D vector combining linear and angular components (twist or wrench).
- **Twist** — spatial velocity (linear + angular).
- **Wrench** — spatial force (force + torque).
- **MPR** — Minkowski Portal Refinement. Alternative to GJK for narrow phase.
- **GJK** — Gilbert-Johnson-Keerthi. Convex distance algorithm.
- **EPA** — Expanding Polytope Algorithm. GJK companion for penetration depth.
- **SAP** — Sweep And Prune. Broad-phase by axis-sorting AABBs.
- **BVH** — Bounding Volume Hierarchy. Tree-structured broad phase.
- **MJX** — MuJoCo XLA. Differentiable MuJoCo on GPU/TPU.
- **Brax** — Google's JAX-based differentiable rigid sim.
- **Warp** — NVIDIA's Python-to-CUDA JIT, what Newton uses.
- **CubeCL** — Burn project's Rust kernel DSL targeting wgpu/CUDA/HIP.
