# PLAN — `molla`: a pure-Rust, wgpu-accelerated physics engine

A from-scratch reimplementation of [Newton Physics](https://github.com/newton-physics/newton)
in pure Rust with `wgpu` for GPU compute, designed to run on **non-NVIDIA hardware**
(AMD, Intel Arc, Apple Silicon) as a first-class target. The Python Newton tree
at `./xtra/newton/` is the **reference** for behavior, data layout, and test
oracles — we read it, we do not port it line-for-line.

This document is the source of truth for scope, architecture, decisions, and
the staged delivery roadmap. It is intentionally long: skim the table of
contents, jump to the section you need.

**Companion doc**: [`GAP.md`](GAP.md) — gap analysis vs. the Newton + Warp
reference, last refreshed 2026-05-09. Stages 14–25 below were added directly
from that gap analysis; the priority tags (P0–P3) match what's in GAP §9.

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

### Stage 2 — GPU semi-implicit ✅ COMPLETE (2026-05-08)

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

### Stage 3 — Articulation (Featherstone) ✅ COMPLETE (2026-05-08)

- **3.0 ✅** — Joint data model + FK only.
- **3.1 ✅** — `FeatherstoneSolver` (CPU) — proper Featherstone
  Articulated Body Algorithm (ABA) per Featherstone (2008) ch. 7.
  Three passes: forward kinematics + body kinematics, backward
  articulated-inertia accumulation (with full bias-force propagation
  including Coriolis terms via `spatial_cross_dual(v_s, I_s · v_s)`),
  forward acceleration solve. Spatial 6×6 inertias expressed at the
  world origin (matches Newton's `solvers/featherstone/kernels.py`
  layout). Tests: pendulum oscillates with correct linearised period
  (drift < 1 % over one period); rest state stays at rest;
  double-pendulum energy bounded within 50 % over 5000 steps.
- **3.2 ✅** — URDF importer (`molla::utils::urdf`). Parses
  revolute / continuous / prismatic / fixed / floating joints; maps
  inertial blocks to `BodyParams::with_spatial_inertia`. Returns a
  `name → BodyId` / `name → JointId` lookup table. Tests: 2-link
  revolute chain, mixed joint types.
- **3.3 ✅** — GPU port. `shaders/featherstone.wgsl` runs the same
  three-pass ABA per articulation with spatial-vector primitives
  ported from Warp's `_src/math.py`; `FeatherstoneKernel` dispatches
  one workgroup invocation per chain. CPU/GPU parity test passes
  within 1e-3 rad for a pendulum (f32 GPU vs f64 CPU). Run via
  `make test-featherstone-gpu`.

Known limitations carried forward to a stage 3.x rewrite:
- Single-articulation per scene only; branched trees + multi-arm
  systems need an articulation grouping pass.
- Ball / planar / D6 joints are unsupported.
- ABA assumes serial chains (parent index < child index). Tree
  topologies work but rely on the linker's ordering — explicit
  topological sort is a future cleanup.

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

### Stage 4 — First collision pipeline (XPBD rigid contacts) ✅ COMPLETE (2026-05-08)

- **4.0 ✅** — sphere-sphere, sphere-plane narrow phase; brute-force broad
  phase; CPU XPBD with COM-aligned linear projection. Sphere stacks
  settle.
- **4.1 ✅** — Box geometry (`ShapeGeometry::Box { half_extents }`),
  OBB→world AABB. `add_shape_box` builder.
- **4.2 ✅** — Box narrow-phase: sphere-box closest-point, box-plane
  corner test (4 deepest), box-box SAT (15 axes — 3 + 3 face axes plus
  9 edge-edge crosses). Sutherland-Hodgman face clipping for the
  manifold (up to 4 deepest contact points per pair).
- **4.3 ✅** — SAP broad phase (CPU). Variance-axis sweep, AABB-overlap
  pair generation. `BroadPhase` reuses scratch buffers across frames.
  `CollisionPipeline` ties broad+narrow into a stateful end-to-end CPU
  pipeline; brute-force pair generator kept for parity tests.
- **4.4 ✅** — XPBD with full angular response. Jacobi-style relaxation
  per Newton's `solve_body_contact_positions` + `apply_body_deltas`:
  each contact computes its λ independently, deltas accumulate into
  per-body slots, then average by per-body contact count. Generalised
  inverse mass `w = 1/m + (r × n)ᵀ · I⁻¹ · (r × n)`. Linear projection
  + angular projection + Coulomb friction + restitution. Stable
  10-box stack settling test (1.5s sim, no jitter, no interpenetration
  > 5mm).
- **4.5 ✅** — GPU XPBD shader updated to match: per-body Jacobi over
  contacts, body_com / body_inv_inertia / contact_point bindings,
  angular impulse accumulation. CPU/GPU parity test for a 4-box stack
  passes within 5cm. Run via `make test-xpbd-gpu`.
- **4.6 ✅** — Per-shape material (μ, e). `set_shape_material(shape,
  friction, restitution)` on the builder. Pair-rule = max (matches
  Newton). `NaN` sentinel falls back to `Model::default_*`.

Tests added:
- `tests/xpbd_boxes.rs`: `box_settles_on_plane`, `sphere_on_box`,
  `box_stack_settles` (10-box stack), `per_shape_material_overrides_default`.
- `tests/xpbd_gpu_parity.rs`: CPU/GPU box-stack parity (gated on
  `make test-xpbd-gpu`).
- Lib: `box_box_face_overlap`, `box_box_separated_no_contact`,
  `sphere_box_face_contact`, `sphere_box_inside`,
  `box_plane_flat_makes_4_contacts`, `sap_matches_brute_force_on_small_scene`.

Known limitations carried forward:
- No graph coloring pass — Jacobi per-body avoids the need; would only
  matter if we ever switched to per-contact GPU dispatch with atomics.
- No torsional / rolling friction (mu_torsional, mu_rolling) yet — Newton
  has these for stable rolling; molla's stage-4 tests don't need them.
- Capsule / cylinder / mesh contact follows in the GJK + EPA stage.

---

### Newton + Warp gap analysis (2026-05-08)

After stage 4 completed, we did a deep audit of `xtra/newton/` and
`xtra/warp/` to map what molla still needs for full reimplementation
parity. The key finding: **molla's solver depth matches Newton
(Featherstone ABA, XPBD with full angular response are real
implementations), but breadth is thin** — the menu of joints, shapes,
sensors, and importers around the solvers is tiny next to Newton's.

The original stage-5/6 ordering (USD adapter, then GJK+EPA) doesn't
match the actual blocking gaps — we don't need USD before we have
joint limits, capsules, or multi-articulation support. The stage list
below is reordered accordingly.

#### P0 — physics-engine blockers

| Gap | Why | Reference |
|---|---|---|
| Joint limits + per-DOF damping + position/velocity/effort drives | Revolute joints currently rotate freely with no bounds and no actuator targets. Required for Franka / UR / pendulum-with-stop demos. | `Model.joint_limit_lower/upper`, `joint_target_mode`, `joint_friction`, `joint_effort_limit`, `joint_velocity_limit` |
| Multi-articulation grouping in Featherstone | Solver currently assumes one chain per scene. Real scenes have multiple robots. | `Model.articulation_start/world` |
| XPBD/Featherstone sub-stepping | Standard Müller (2020) pattern: N substeps × 1 iter is stabler than 1 step × N iters with contact. | All Newton solvers default to N substeps. |
| Capsule + cylinder + cone shapes | Robotics meshes are mostly capsules/cylinders. Without these, no realistic robot demos. | `geometry/types.py:GeoType` |
| GJK + EPA + support functions | Required for cylinder-cylinder, cone-mesh, convex-hull contact. molla currently handles only the 6 sphere/box/plane combos. | `geometry/{simplex_solver,support_function,mpr}.py` |
| Convex-hull / triangle-mesh collision | URDF and USD scenes ship triangle meshes. Without convex-hull contact the importer is a toy. | `geometry/mesh.py`, Warp `mesh_query_*` |
| Heightfield collision | Standard for terrain/locomotion RL. | `geometry/heightfield.py`, `Model.heightfield_*` |
| BVH broad-phase (GPU) | SAP degrades on uneven scenes; BVH is the standard for >1k bodies and triangle-mesh contact. | `geometry/bvh.py`, Warp `bvh_query_aabb/ray` |
| Hash-grid spatial accel | Particles, granular, fluid all need it; O(N²) is unworkable. | Warp `hash_grid_query/_next` |

#### P1 — needed for serious use

| Gap | Why | Reference |
|---|---|---|
| Ball / D6 / Distance / Cable joints | Spherical joints (humanoid hips), D6 (locked-axis configurable), distance constraints (ropes), cables. | `sim/enums.py:JointType` |
| Equality constraints (CONNECT / WELD / JOINT-mimic) | First-class in Newton — gear/transmission/coupled-joint support. | `Model.equality_constraint_*` |
| Mimic constraints (`joint0 = c0 + c1·joint1`) | Parallel grippers, gear trains. | `Model.constraint_mimic_*` |
| PD controller + actuator pipeline (delay → controller → clamping) | molla's `Control` has only `body_external_f` + `joint_force`. Newton has a modular actuator pipeline. | `actuators/{actuator,controllers,clamping}.py` |
| Inverse kinematics (LM + L-BFGS, position/rotation/limit objectives) | Robotics needs this. | `sim/ik/` |
| Mass matrix (CRBA) + inverse dynamics (RNEA) | Inverse dynamics is fundamental for control. molla has only forward dynamics. | `solvers/kamino/_src/dynamics/`, Warp `dense_chol*` |
| Sparse linear algebra (BSR matrix, sparse matvec, CG/GMRES) | Needed for any large-system solver. molla has zero sparse infra. | Warp `sparse.py`, `optim/linear.py` |
| Cholesky / LDLT factorization | CRBA + dense MuJoCo-style solver requires it. | Warp `dense_chol`, `tile_cholesky` |
| Graph coloring | Currently Jacobi (works). Coloring would let us do parallel Gauss-Seidel; also needed for VBD/Style3D. | `coloring.py` |
| MJCF importer | MuJoCo Menagerie is the de-facto robot zoo; URDF is barely usable, MJCF is. | `utils/import_mjcf.py` |
| Particles + cloth (springs, triangles, edges, tetrahedra, muscles) | Newton's `Model` has the data layout; molla's has `particle_*` arrays but no solver. | `Model.spring_*/tri_*/edge_*/tet_*` |
| Multi-world batching for joints / articulations / constraints | molla has it on bodies/particles only. | `Model.{joint_world,articulation_world,equality_constraint_world}` |
| Collision filtering (groups + filter pairs) | Standard layer/mask exclusion. | `Model.shape_collision_filter_pairs/group` |
| IMU + raycast + contact + tiled-camera + frame-transform sensors | Robotics needs IMU/lidar; visual RL needs cameras. molla's `sensors/` dir is empty. | `sensors/sensor_*.py` |
| Spatial Jacobian builtin + spatial_adjoint + spatial_mass | For constraint enforcement and contact-force Jacobians. | Warp `spatial_jacobian/_adjoint/_mass` |
| Quat utilities: twist/swing decomposition, RPY ↔ quat, Euler conversions | Joint axis-aligned twist, initial-condition specification. | Warp `quat_twist`, `quat_to_rpy`, `quat_from_euler` |
| Robust loss norms (Huber, pseudo-Huber, smooth_normalize) | Robust loss functions for soft constraints, IK. | Warp `norm_huber`, `norm_pseudo_huber` |

#### P2 — advanced / specialty

| Gap | Why |
|---|---|
| Differentiable contacts + tape/autograd | Trajectory optimization, sim-to-real. |
| Hydroelastic contact (Drake-style pressure fields) | Niche but elegant. |
| MuJoCo solver wrapper (mjx) | Validation against industry standard. |
| VBD / Style3D / Implicit MPM solvers | Soft body, cloth, granular — only when those workloads are in scope. |
| SDF collision (`sdf_contact`, `sdf_hydroelastic`, texture sampling) | GPU-resident geometry. |
| Marching cubes | SDF → mesh extraction. |
| BVH/hash-grid as Warp-style WGSL builtins | Generic acceleration-structure layer in `molla::math/geometry`. |
| Command-buffer reuse / "graph capture" analog | We currently rebuild every frame. |
| Custom attribute system | Newton lets users register arbitrary per-entity arrays. |
| USD adapter | Reading `UsdPhysics*` schemas (was original stage 5). |
| Viewer backends (file / rerun / USD export) | molla_viewer is bevy-only. |
| Tile primitives (cooperative shared memory ops) | Fast workgroup-level reductions, blocked GEMM/Cholesky. |
| RNG / Perlin / curl noise / Poisson disk | Stochastic init, procedural scenes. |

---

### Stage 5 — Joint limits, drives, multi-articulation, sub-stepping ✅ COMPLETE (2026-05-08)

- **5.1 ✅** — Joint-limit + drive data layout. `Model.joint_limit_lower /
  joint_limit_upper / joint_friction / joint_effort_limit /
  joint_velocity_limit` (per-DOF, `NaN` = unlimited). Builder
  setters: `set_joint_limits`, `set_joint_friction`,
  `set_joint_effort_limit`, `set_joint_velocity_limit`. New
  `Control.joint_target_mode / joint_target_pos / joint_target_vel /
  joint_target_kp / joint_target_kd` per-DOF arrays.
  `JointTargetMode` enum (`None / Position / Velocity /
  PositionVelocity / Effort`).
- **5.2 ✅** — CPU `FeatherstoneSolver`: drive torque computed
  per-DOF in Pass 2 from `target_mode` + PD gains, friction torque
  added (`τ −= μ·qd`), effort cap clamps applied torque. Hard
  joint-position limits via post-step projection (clamp `q` to
  `[lo, hi]`, zero `qd` at the boundary). Per-DOF velocity cap.
  Tests: `pendulum_with_limits_stays_inside_cone`,
  `pd_drive_holds_pendulum_at_target`,
  `joint_friction_damps_swing_to_rest`.
- **5.3 ✅** — GPU `FeatherstoneKernel`: matching shader bindings
  (group 0: 13 storage buffers; group 2: 6 buffers including all
  drive arrays). New WGSL helpers `drive_torque`,
  `clamp_or_passthrough`, NaN-aware bound checks. Tests:
  `pd_pendulum_with_limits_cpu_gpu_match`, `joint_limits_clamp_on_gpu`.
- **5.4 ✅** — Multi-articulation grouping. `Model.articulation_start`
  (CSR offsets, length `articulation_count + 1`),
  `Model.articulation_count`. Builder method `begin_articulation()`
  starts a fresh chain. CPU `FeatherstoneSolver` wraps its 3-pass
  algorithm in a per-articulation outer loop; parent-joint searches
  are scoped to `j_begin..j` so chains don't bleed into each other.
  GPU kernel pulls the articulation CSR from the model. Test:
  `two_independent_pendulums` — pendulum 1's trajectory in a
  combined two-pendulum scene matches a standalone single-pendulum
  baseline within `1e-6` rad.
- **5.5 ✅** — Solver sub-stepping. `Solver::with_substeps(n)` on
  both `XpbdSolver` and `FeatherstoneSolver`. Each outer
  `step(dt)` runs `n` substeps of `dt/n`, ping-ponging through a
  host-side scratch state. **Caveat**: contacts are reused across
  substeps (not re-detected), so the Müller-2020 "small substeps,
  few iters" stability win on contact-rich stacks requires the
  caller to substep externally — see
  `external_substeps_box_stack` test for the recommended pattern.
  For contact-free workloads (Featherstone) substepping cleanly
  reduces integration error. New `State::clone_host` /
  `copy_host_from` helpers underpin the ping-pong.

**Test surface**: 49 lib tests (47 prior + 2 new featherstone:
limits, drives, friction, multi-articulation, substeps), 5 box
tests (4 prior + 1 external-substep stack), 3 GPU featherstone
tests (1 prior + 2: PD+limits parity, GPU limit-clamp).

### Stage 6 — Convex contact (capsules + cylinders + GJK + EPA + meshes) ✅ COMPLETE (2026-05-08)

- **6.1 ✅** — Capsule + Cylinder shapes (URDF +Y axis convention).
  Builder: `add_shape_capsule`, `add_shape_cylinder`. AABB for both
  (capsule = OBB of two end-cap spheres; cylinder = tight per-axis
  projection). Closed-form narrow phase: sphere-capsule (segment
  closest-point), capsule-plane (per end-cap sphere-plane),
  capsule-capsule (segment-segment closest points, Ericson). 8
  unit tests + 3 XPBD integration tests (settle on plane, stack,
  side-by-side).
- **6.2 ✅** — Support functions for sphere, plane, box, capsule,
  cylinder, triangle. World-frame query that rotates `dir` into
  shape-local frame, evaluates analytic support, returns world
  point. 7 unit tests including Minkowski-difference reduction.
- **6.3 ✅** — GJK distance + overlap detection (CPU). Standard
  4-simplex algorithm with `v·v − v·w ≤ ε·v·v` termination
  (Ericson §9.5). Returns `Disjoint` (with closest-points +
  distance) or `Overlap` (with simplex feeding EPA). 6 unit tests
  covering separated and overlapping primitive pairs.
- **6.4 ✅** — EPA penetration depth + contact normal. Expands
  GJK's overlap simplex into a polytope, picks the face nearest
  the origin, queries support past it. Standard silhouette-edge
  expansion with **directed** edge tracking (so new face windings
  preserve outward orientation even on degenerate origin-on-face
  initial polytopes). Pad-to-tetra logic with cardinal-axis
  fallback for radially-symmetric pair degeneracies. 3 unit tests.
- **6.5 ✅** — GJK + EPA wired into the narrow phase as the
  fallback for non-closed-form pairs: capsule-box, box-capsule,
  capsule-cylinder, cylinder-cylinder, cylinder-sphere,
  cylinder-box, triangle-anything. Cylinder-plane uses a
  4-perimeter sample fallback (single-support produces an
  off-centre torque that rocks the cylinder onto its side).
- **6.6 ✅** — BVH broad phase. Top-down construction with
  median-split on the longest centroid-extent axis. Plane-shape
  AABBs are infinite, so planes are excluded from the BVH and
  paired with finite shapes separately. CPU only — GPU LBVH is
  future work. Tests: parity against brute-force on 100 + 1024
  shapes.
- **6.7 ✅** — Triangle + TriMesh shapes. `ShapeGeometry::Triangle
  { v0, v1, v2 }` for direct per-triangle collision.
  `ShapeGeometry::TriMesh { mesh_id }` references a `Model.meshes:
  Vec<TriMesh>` storage. Narrow phase routes mesh pairs through
  per-triangle GJK+EPA, with a vertex-projection fast path for
  mesh-vs-plane. 2 XPBD integration tests (sphere on triangle,
  sphere on trimesh quad floor).

**Test surface added in stage 6**: 28 narrow-phase + GJK + EPA +
support + BVH unit tests in lib (76 total now), 7 integration
tests (capsule, cylinder, trimesh).

Known limitations carried forward:
- Cone shape — not yet added; closed-form GJK support trivial when
  needed.
- Convex-hull builder (qhull-style construction from a mesh) —
  deferred until a concrete consumer arrives.
- MPR fallback for deep penetration — current EPA handles all our
  test cases, but on extreme deformation MPR is more robust.
- GPU port of GJK+EPA — CPU is the only path. The viewer's GPU
  XPBD solver still works for sphere/box/plane workloads; capsule
  and cylinder demos run on CPU narrow-phase + GPU XPBD.
- BVH GPU port (LBVH) — CPU only.

---

### Newton + Warp gap analysis #2 (post-stage-6, 2026-05-08)

After stages 5-6 we did a second deep audit of `xtra/newton/_src/` and
`xtra/warp/_src/`. The key strategic question that came out: **does
the new gap go before or after USD/MJCF import?**

The split is by whether a gap is *architectural* (changes the `Model`
struct layout — adding it after the USD importer means refactoring
the importer too) or *additive* (lives outside the `Model` and can
slot in any time).

#### Architectural gaps (must land BEFORE USD/MJCF)

| Gap | Why before USD | Newton ref |
|---|---|---|
| Multi-world per joint / articulation / constraint | molla currently has `world_index` on bodies/particles only. USD + MJCF scenes assign joints / articulations / equality constraints to worlds — adding the index later requires rewriting the importer. | `Model.{joint,articulation,equality_constraint,constraint_mimic}_world` |
| Ball / D6 / Distance joint types | USD `UsdPhysicsSphericalJoint`, MJCF `<joint type="ball">` and 6-DOF mounts map directly to these. Without them, the importer would silently drop or wrongly approximate them as Fixed/Revolute. | `sim/enums.py:JointType` |
| Equality constraints (CONNECT, WELD, JOINT-mimic) | Both formats encode rigged-assembly couplings as equality constraints. Storage layout in `Model` has to exist before the importer can populate it. | `Model.equality_constraint_*` |
| Mimic constraints (`q0 = c0 + c1·q1`) | Parallel grippers, gear trains, transmission ratios. Same architectural reasoning. | `Model.constraint_mimic_*` |
| Per-shape collision groups + filter pairs | Both formats encode collision filtering as scene metadata; needs a Model home. | `Model.shape_collision_group`, `shape_collision_filter_pairs` |
| `joint_enabled` flag, `joint_ancestor`, per-DOF `joint_target_ke` | Newton's joint metadata that USD/MJCF importers populate. Adding fields later means re-walking every imported scene. | `Model.joint_*` |
| Spatial Jacobian builtin | Foundation primitive used by IK, contact-Jacobian, RNEA. Cleanest to add before the importer because importers attach drive metadata that ends up flowing through the Jacobian path. | Warp `builtins.py:2261` |

#### Additive gaps (can land AFTER USD/MJCF — sloted into the relevant later stage)

| Gap | Where it slots | Newton ref |
|---|---|---|
| Sensors (IMU, raycast, contact, frame-transform) | New `sensors/` crate module. USD/MJCF can store metadata; sensor code consumes it later. | `sensors/sensor_*.py` |
| Actuator pipeline (PD/PID, clamping, delay) | New `actuators/` module. Sits between user code and Solver. | `actuators/` |
| Sparse linalg (BSR matrix, CG, GMRES, Cholesky) | Goes with the IK / CRBA / RNEA stage. Lives in `math/sparse.rs`. | Warp `sparse.py`, `optim/linear.py` |
| Mass matrix (CRBA), inverse dynamics (RNEA), IK (LM + L-BFGS) | Same stage as sparse linalg. | `solvers/kamino/_src/dynamics/`, `sim/ik/` |
| Contact warm-starting / matching | Stand-alone post-broad-phase pass; extends `Contacts` (additive). | `geometry/contact_match.py` |
| Torsional + rolling friction | Per-shape material extension (already have μ, restitution; add μ_torsional, μ_rolling). | `Model.shape_material_*` |
| Mesh AABB tree (per-mesh BVH for trimesh narrow-phase) | Extends `TriMesh` struct (additive). Critical for non-trivial trimesh sizes. | `geometry/narrow_phase.py` |
| Contact reduction (voxel-AABB pruning) | Stand-alone post-narrow-phase pass. | `geometry/contact_reduction_global.py` |
| Math utilities: `quat_twist`, Euler conversions, robust Huber norms | Pure additions to `math/`. | Warp `math.py:36-422` |
| Cone / ConvexHull / Heightfield shapes | Just new `ShapeGeometry` variants + support functions. | `geometry/types.py:GeoType` |
| MPR alternative to EPA | Drop-in narrow-phase replacement; current EPA passes all tests. | `geometry/mpr.py` |
| GPU GJK / EPA / BVH | Performance optimisation; CPU paths are fine for single-arm scenes. | — |

### Stage 7 — Architectural completion (BEFORE USD) ✅ COMPLETE (2026-05-08)

- **7.1 ✅** — Multi-world per entity. `Model.joint_world` /
  `articulation_world` (joints inherit child body's world;
  articulation inherits its first joint's world). Builder
  propagation tested via `joint_and_articulation_worlds_propagate`.
- **7.2 ✅** — Joint metadata. `Model.joint_enabled` (1 = active),
  `joint_ancestor` (parent joint in chain, -1 root),
  `joint_target_ke` (per-DOF static stiffness, distinct from PD
  `target_kp`). Featherstone honours `joint_enabled` (disabled
  joints freeze, gravity stops affecting them). Test:
  `disabled_joint_stays_frozen`.
- **7.3 ✅** — Collision groups + filter pairs.
  `Model.shape_collision_group: u32` (default 0xffff_ffff) and
  `Model.shape_collision_filter_pairs: Vec<(u32, u32)>` (sorted +
  dedup). Shared `pair_passes_filters` helper used by brute-force,
  SAP, and BVH broad phases. Tests:
  `bvh_filters_by_collision_group`, `explicit_exclusion_filters_pair`.
- **7.4 ✅** — Spatial-Jacobian primitive.
  `math::spatial_jacobian_column(kind, axis, joint_origin, body_point)`
  for revolute / prismatic single-DoF columns. Foundation for IK,
  RNEA, contact-Jacobian. Tested against analytical pendulum.
- **7.5 ✅** — New joint types (data layout): `JointType::Ball`
  (4 coord quat, 3 DoF), `Distance` (7 coord, 6 DoF, soft distance
  limit), `D6` (7 coord, 6 DoF, per-axis lock via existing
  per-DOF limit arrays). Builder methods: `add_joint_ball`,
  `add_joint_distance`, `add_joint_d6`. FK tests:
  `ball_joint_fk_rotates_via_quat`, `d6_joint_fk_translates`.
  Featherstone integrates ball-joint quaternion kinematically (caller
  drives qd); full 3-DoF ABA dynamics deferred to stage 11.
- **7.6 ✅** — Equality constraints.
  `Model.equality_constraint_*` storage with `CONNECT` (3-DoF
  positional pin) and `WELD` (6-DoF position+orientation pin)
  variants. Builder `add_equality_connect/weld`. XPBD position-
  projection pass with mass-weighted Jacobi-style correction.
  Tests: `connect_pins_two_bodies_under_gravity`,
  `weld_keeps_orientation_aligned`,
  `humanoid_ball_joint_swings_under_gravity` (the stage-7 DoD demo,
  using a CONNECT pin as a 3-DoF spherical joint).
- **7.7 ✅** — Mimic constraints. `Model.constraint_mimic_*` arrays
  (`q_a = c0 + c1 · q_b`). Featherstone pre-step pass overrides
  driven joint's `joint_q` and `joint_qd` from the driver. Test:
  `mimic_constraint_couples_two_joints`.

**Test surface added**: 85 lib tests (up from 80), 2 new equality
integration tests, 1 humanoid ball-joint demo, FK tests for Ball /
D6 joints, multi-world propagation test, mimic test. All green.

Carried forward to stage 11 (RNEA/IK + sparse linalg):
- Reduced-coord ABA dynamics for Ball / D6 / Distance joints
  (currently kinematic; needs 3×3 D-matrix inverse machinery and
  per-DOF block algorithms).
- Equality-constraint *velocity* projection (current implementation
  is position-only, which can damp swing momentum).

### Stage 8 — `usd_molla` crate ✅ COMPLETE (2026-05-08)

- **8.1 ✅** — `crates/usd_molla/` scaffold. Path-deps onto
  `usd_schema` (from the OUSD ecosystem) + git-pinned `openusd`
  with the `physics` feature. **No Bevy dependency.** Mirrors the
  layered structure used by `usd_rapier` so the entire OUSD
  constellation stays version-locked.
- **8.2 ✅** — Reader pipeline: `bodies` (RigidBodyAPI + MassAPI +
  Xform stack), `colliders` (Sphere/Cube/Capsule/Cylinder/Plane/Mesh
  with per-mesh BVH-ready triangulation), `joints`
  (Revolute/Prismatic/Fixed/Spherical/Distance/Generic mapped to
  molla's joint enum, axis-token decode, single-axis limits), `scene`
  (PhysicsScene → per-world gravity).
- **8.3 ✅** — `attrs` helper module for typed Sdf attribute reads
  (matches the helpers `urdf2usd::to_urdf::read` keeps locally;
  duplicating them keeps `usd_molla` decoupled from
  `urdf2usd`).
- **8.4 ✅** — `build_model_from_stage(stage, ImportOptions) ->
  (Model, ImportInfo)` — the public entry point. `ImportInfo`
  preserves the prim-path ↔ molla-id maps for writeback / debugging.
- **8.5 ✅** — `tests/round_trip.rs::sphere_on_plane_round_trip` —
  end-to-end test: an in-memory USDA scene of a sphere over a plane
  is imported, simulated for ~1 s with XPBD, and the sphere settles
  at y ≈ 0.5 (matching the hand-built `xpbd_floor::sphere_settles_on_plane`).

Carried forward:
- `project_state_to_stage` is wired through but currently a logging
  stub — full Sdf-level field-write helpers (set_pose, xformOpOrder
  authoring) live in usd_schema and aren't trivially accessible
  without adding a writer-side dependency. Lands when a real
  consumer (USD-export viewer) needs it.
- Hierarchical Xform composition (walk parents and multiply): for
  flat scenes the prim's local Xform stack equals the world pose;
  nested-xform scenes need a parent walk. Trivial follow-up.
- Multi-apply `PhysicsLimitAPI:<dof>` / `PhysicsDriveAPI:<dof>` on
  generic joints — the physics-reader exposes them in
  `ReadJoint.{limits, drives}`; we currently only honour the
  built-in single-axis revolute/prismatic limits. Wire-up is
  mechanical when D6 / Spherical limit demos land.

### MJCF importer — DROPPED (2026-05-08)

A native MJCF reader was originally pencilled in here. It's been
removed because the ecosystem already solves the problem upstream:

- Isaac Lab / Isaac Sim convert MJCF → USD as part of their normal
  workflow; MuJoCo Menagerie scenes are routinely loaded that way.
- `mujoco_to_usd` exists as a standalone tool.
- NVIDIA Newton itself doesn't ship an MJCF importer either —
  Newton's MJCF path is "MJCF → USD via Isaac → Newton". USD is the
  lingua franca.

Since `usd_molla` (stage 8) already handles the USD half, users with
Menagerie scenes get there via `mujoco_to_usd` → `usd_molla`. A
faithful MJCF importer (~3-5 kLoC: composite bodies, tendons,
contact pairs, MJCF actuators, compiler defaults) would mostly
duplicate work upstream tools already do well, so we skip it.

The stages below are renumbered relative to the previous PLAN —
stage 9 is now sensors + actuators, stage 10 is IK / RNEA / sparse
linalg, etc.

### Stage 9 — Sensors + actuator pipeline

**Status: SENSORS DONE, ACTUATORS LIBRARY DONE — viewer demo of
PD-driven articulation TODO** (2026-05-08 / -09).

**Goal**: RL-grade scenes with observation + control.

Deliverables:
- ✅ `sensors/raycast.rs` + `shaders/raycast.wgsl` + `kernels/raycast.rs`.
  Closest-hit ray vs Sphere/Plane/Box/Capsule/Cylinder shapes; one
  thread per ray; 5/5 GPU tests pass. Viewer Lidar example
  (Scene::Lidar).
- ✅ `sensors/imu.rs` + `shaders/imu.wgsl` + `kernels/imu.rs`. Body-frame
  accelerometer + gyroscope, finite-difference acceleration via a
  per-frame `body_qd_prev` snapshot owned by the sensor; 4/4 GPU
  tests. Viewer IMU example (Scene::Imu) with live readout.
- ✅ `sensors/contact.rs` + `shaders/contact_aggregate.wgsl` +
  `kernels/contact_sensor.rs`. Per-body contact summary (count,
  max_depth, signed-normal sum, centroid); 3/3 GPU tests. Viewer
  Contact example with arrow gizmos.
- ✅ `sensors/frame_transform.rs` + `shaders/frame_transform.wgsl` +
  `kernels/frame_transform.rs`. World pose + relative-to-reference
  pose for a list of (body, local_pose) frames; 4/4 GPU tests.
- ✅ `actuators/controller_pd.rs` + `shaders/actuator_pd.wgsl` +
  `kernels/actuator_pd.rs`. Stateless PD with effort cap, one
  actuator per DOF; writes `Control.joint_force` directly. 3/3 GPU
  tests.
- ✅ `actuators/controller_pid.rs` + `shaders/actuator_pid.wgsl` +
  `kernels/actuator_pid.rs`. PID with integral state + anti-windup
  clamp, ping-pong integral buffers. 2/2 GPU tests.
- ❌ Composable `Actuator` orchestrator (delay → controller →
  clamping) matching Newton's `actuator.py`. PD/PID currently bake
  the effort cap in; promote it to a separate clamping stage when an
  example needs vel-saturation / DC-motor curves.
- ❌ Delay stage (ring buffer for sim-to-real transport lag).
- ❌ Viewer demo of a Featherstone-driven articulation with PD
  control following a sine target. The viewer currently only runs
  XPBD, so this requires a parallel Featherstone sim path. Tracked
  as a follow-up — the actuator API + tests are complete.

**DoD Status**:
- ✅ raycast against a sphere returns the right hit distance
  (`tests/sensor_raycast_gpu.rs::ray_hits_sphere_at_expected_distance`)
- ✅ PD actuator drives a joint with effort cap respected
  (`tests/actuator_pd_gpu.rs::pd_effort_cap_clamps`)
- ❌ IMU on swinging pendulum returns analytical accel + ω — basic
  IMU tests cover at-rest, free-fall, rotated-mount, gyro tracking;
  the full pendulum analytical golden test is the part of Stage 9
  still pending. The IMU kernel itself is the same code path used
  by the test cases that *did* land, so the integration is sound;
  the missing piece is just a longer-running scene.

### Stage 10 — Inverse kinematics + mass matrix + RNEA + sparse linalg

**Status: DONE** (2026-05-09).

**Goal**: differentiable & control-friendly articulation queries —
plus the foundation everything in stage 12+ needs.

Deliverables:
- ✅ **CRBA mass-matrix construction** — `solvers::mass_matrix(model, state, articulation)`
  returns the dense `N×N` joint-space mass matrix using the standard
  composite-inertia algorithm (Featherstone 2008 §6.3). Revolute +
  Prismatic only for now. CPU; GPU port deferred until a real
  consumer exists. Tests: pendulum parallel-axis, double-pendulum
  symmetric/SPD, Featherstone-consistency (`M·q̈ = τ_g`).
- ✅ **RNEA inverse dynamics** — `solvers::inverse_dynamics(model, state, articulation, qdd)`
  returns the joint torques `τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)` via
  forward velocity/acceleration sweep + backward force sweep.
  Tests: pendulum-gravity torque (analytic match), CRBA + RNEA
  decomposition (`τ_full = M·q̈ + RNEA(q, q̇, 0)` within ε).
- ✅ **Sparse linalg layer** (`math::sparse`): `BsrMatrix` block-CSR
  type with `from_pattern / from_triplets / mv / get_diag / scale`,
  `LinearOperator` trait, `ClosureOp` adapter for matrix-free
  operators (e.g. implicit `Jᵀ·J + λ·I` in IK).
- ✅ **Iterative solvers** (`math::sparse::{cg, bicgstab, gmres}`) —
  Jacobi-preconditioned via `Some(&m_inv_diag)`. Restartable
  GMRES(m). Tested on tridiagonal SPD + a 1000×1000 perf gate.
- ✅ **Dense Cholesky + back-sub** (`math::dense_solve`):
  `dense_chol`, `forward_sub`, `back_sub`, `dense_chol_solve` for
  small dense SPD systems (CRBA solves, IK normal equations).
- ✅ **Robust-loss math primitives** (`math::robust_loss`):
  `norm_huber{,_grad}`, `norm_pseudo_huber{,_grad}`,
  `smooth_normalize`.
- ✅ **Body Jacobian** (`solvers::jacobian`) — `6 × N` columns for a
  body point in world frame; revolute + prismatic.
- ✅ **IK (Levenberg-Marquardt)** (`solvers::ik`) — `IkSolver` with
  damped least squares (`(JᵀJ + λI)·dq = -Jᵀ·e`), adaptive damping,
  multi-seed `solve_with_seeds`. Objectives: `Position`,
  `JointLimit` (Huber-soft). L-BFGS deferred — LM hits the DoD
  on every Franka reach test in ≤6 iters.

Definition of done:
- ✅ Sparse `cg` solves a 1000×1000 SPD system to 1e-6 in ~20 ms
  (release; debug skipped, gated on `cfg(debug_assertions)`).
- ✅ Pendulum CRBA mass matrix matches the analytical 1×1 value
  within 1e-9.
- ✅ IK on a 7-DOF Franka-style arm reaches a 0.35 m offset target
  in 6 LM iterations with `<1 µm` residual (well under the 5 mm gate).

### Stage 11 — Contact extras

**Status: DONE** (2026-05-09).

**Goal**: the contact-quality features that distinguish a toy from a
production engine.

Deliverables:
- ✅ **Rolling friction** (`μ_rolling`) per shape, pair-rule
  `max(μ_a, μ_b)`. Damps angular velocity perpendicular to the
  contact normal — wheel-on-ground / sphere-rolls-to-rest. Wired
  into XPBD **CPU and GPU** velocity-projection (`shaders/xpbd.wgsl`
  bind group entry 9). `set_shape_friction_modes` in `ModelBuilder`.
- ✅ **Torsional friction** (`μ_torsional`) per shape. Damps angular
  velocity ALONG the contact normal (top spinning on a surface).
  Same pair-rule + builder API as rolling, also on CPU + GPU
  (bind group entry 10).
- ✅ **Contact warm-starting / matching** (`geometry::contact_extras`):
  `contact_id` (FNV-1a-style hash over sorted shape pair + voxelised
  contact point), `build_contact_id_table`, `match_contacts`. The
  application layer can carry per-contact data (XPBD `λ_n`, friction
  direction, age) across frames. Deterministic sort-key match
  matches Newton's `geometry/contact_match.py`.
- ✅ **Contact reduction** (`reduce_contacts`): voxel-bin contacts
  by `(shape_pair, voxel(point))` and keep the deepest per voxel.
  Stops 4-corner box-plane stacks from emitting redundant contacts.
- ✅ **Per-mesh BVH** (`geometry::mesh_bvh::MeshBvh`): top-down
  median-split AABB tree built eagerly when a mesh is registered.
  AABB-overlap query for shape-vs-trimesh narrow phase + closest-hit
  ray cast. `narrow_phase::trimesh_pair` now visits O(log N + k)
  triangles instead of O(N). Perf gate (`tests/mesh_bvh_perf.rs`):
  32×32 grid mesh + sphere collide in 28 µs (release).

### Stage 12 — Particles + cloth (XPBD distance/bending) + springs

**Status: MOSTLY DONE** (2026-05-09). GPU end-to-end. The remaining
gaps are bending/area constraints and particle-particle self-collision.

**Goal**: cloth on a sphere, rope dragging across boxes.

Deliverables:
- ✅ Particle storage in `Model` and `State`
  (`particle_q/qd/mass/inv_mass/radius/flags/world` + `initial_particle_q`).
- ✅ Distance constraint (`Model.spring_*`) authoring API:
  `add_particle`, `add_distance_constraint`, `add_cloth_grid`,
  `pin_particle`.
- ✅ Triangle topology (`Model.tri_indices`) for rendering — the
  `molla_viewer` cloth scene builds a deforming Bevy mesh from these
  indices.
- ✅ XPBD particle solver (`solvers::xpbd::step_particles`):
  predict (gravity + linear damping) → distance constraint with proper
  λ accumulator (Newton/Warp `solve_springs`) → per-spring hard
  strain limiter via Jacobi-style position projection
  (`Model.spring_max_stretch`) → particle vs Y=0 plane.
- ✅ GPU port — `kernels/xpbd_particles.rs` +
  `shaders/xpbd_particles.wgsl`. Multi-pass: parallel `predict`,
  graph-coloured `solve_distance` (K passes per iter, no atomics),
  graph-coloured `strain_accumulate`, `floor_clamp`,
  `particle_vs_shapes`, `velocity_recovery`. Greedy spring colouring
  computed once at build time (`spring_color_count` /
  `spring_color_offsets` / `spring_indices_by_color` in `Model`).
- ✅ Particle-rigid SDF collision (sphere / plane / box / capsule /
  cylinder) — `particle_vs_shapes` WGSL pass projects each particle
  out of every penetrated shape using the analytic SDF + outward
  normal. Demo: `Scene::Cloth` drapes a 36×36 sheet over a sphere
  obstacle.
- ✅ Bending springs (skip-one PBD pattern) — `add_cloth_grid` now
  emits an extra distance constraint for every (i, i+2) pair along
  U and V. `add_cloth_grid_with_bending` exposes the compliance knob;
  `<0` disables. Default = 30× structural compliance.
- ✅ Particle-particle (cloth self) collision — GPU spatial-hash
  uniform grid (4096 buckets × 12 slots), atomic insert,
  Jacobi-style position projection. WGSL passes
  `hash_clear`/`hash_insert`/`self_collide`/`strain_apply`. Cell
  size = 2.2 × max particle radius, set automatically on first
  dispatch.
- ❌ Area / triangle constraints — for self-area preservation under
  large strain. Lower priority — bending springs already prevent the
  most visible failure mode (cloth flattening unrealistically).

Cloth is now production-grade enough for demos: drapes on shapes,
folds onto itself without tunnelling, holds shape under bending.

### Stage 13 — Specialty shapes + GPU GJK

**Status: PARTIAL** (2026-05-09). Cone + ConvexHull done; Heightfield,
SDF, MPR, GPU GJK/EPA/LBVH still TODO.

**Goal**: catch up on geometry breadth.

Deliverables:
- ✅ **Cone** primitive — apex at `+half_height·Y`, base disc at
  `−half_height·Y`. `add_shape_cone` builder API. Closed-form
  support function picks max of (apex, base disc edge) by dot
  product. World-AABB via the same axis-projection trick as
  Cylinder. Cone-vs-plane fast path (apex + 4 disc rim samples).
- ✅ **ConvexHull** primitive — `add_hull(vertices)` registers a
  hull; `add_shape_convex_hull` references it. Linear-scan support
  function (O(N) per query — fine for ≤50-vertex hulls; future
  upgrade: DOP / hill-climbing). Hull-vs-plane fast path emits
  one contact per below-plane vertex (deepest 4 kept). Threaded
  `&[Vec<Vec3>]` through `gjk_with_hulls` / `epa_with_hulls` /
  `support_with_hulls` so EPA can resolve hull supports without
  capturing global state.
- ✅ **Heightfield** primitive — `Heightfield` struct (regular grid
  of height samples) registered via `add_heightfield`; shape variant
  references it. Narrow phase enumerates only the cells overlapping
  the other shape's local-frame AABB and routes the implicit
  triangles through GJK+EPA. World AABB built from the 8 local-AABB
  corners. Tests: sphere settles on flat field, AABB sanity.
- ✅ **SDF** shapes — `SdfShape::Torus { major, minor }` and
  `SdfShape::Sphere { r }`. `add_sdf` + `add_shape_sdf` builder API.
  Narrow phase: ball-march the other shape's centre against the
  SDF's signed-distance field; surface-projection witness +
  finite-difference normal. Tests: sphere drops onto torus surface
  without penetration, AABB sanity.
- MPR fallback — **deferred**. EPA covers every contact case
  currently exercised by the test suite; a robust MPR/XenoCollide
  implementation needs more careful sign-test logic than fits in
  this stage's budget. Rough first-pass implementation tried and
  reverted (overlap detection worked; contact-depth recovery
  diverged on degenerate portals). Will revisit when an actual EPA
  failure shows up.
- ✅ **GPU GJK boolean overlap** (`shaders/gjk.wgsl` +
  `kernels/gjk.rs`) — one thread per candidate pair from the broad
  phase, inline support for Sphere/Box/Capsule/Cylinder/Cone,
  fixed-size 4-vertex simplex with line/triangle/tetrahedron-step
  routines. Output: `0` (disjoint), `1` (overlap), `2` (host
  fallback for Plane/TriMesh/iter-cap). Wired into
  `CollisionPipeline::enable_gpu_prefilter` — pairs the kernel
  reports as `0` skip the CPU narrow phase entirely. Tests in
  `tests/gjk_gpu.rs` confirm parity vs CPU GJK + an end-to-end
  pipeline test that the prefilter doesn't change contact counts.
- GPU EPA — TODO. Bounded polytope (~64 faces) is the next step;
  contact generation stays CPU on the overlapping subset until
  then.
- 🟡 **GPU LBVH** — Karras-style WGSL kernel skeleton in
  `shaders/lbvh.wgsl` (`morton_codes`, `build_internal`,
  `refit_aabbs` entry points + Karras's longest-common-prefix split
  helpers). Host dispatcher (radix sort + node/visited buffer
  allocation + multi-pass dispatch + integration with the broad
  phase) is **not yet wired in** — the WGSL is a starting point for
  a future GPU port; the existing CPU SAP + per-mesh BVH cover all
  current scenes.

### Stage 14 — Multi-channel raytraced camera sensor ✅ COMPLETE (2026-05-09)

**Goal**: close the perception loop — give RL / robotics scenes a
visual sensor (depth + colour + normal + albedo + shape index)
running on the same wgpu device as the rest of the engine.

Deliverables:
- ✅ `sensors/tiled_camera.rs` + `kernels/tiled_camera.rs` (two
  kernel pipelines, `CameraRaysKernel` + `CameraTraceKernel`) +
  `shaders/camera_rays.wgsl` + `shaders/camera_trace.wgsl`. Pinhole
  ray generation per `(camera, x, y)` triple, then closest-hit
  primitive ray-cast (Sphere · Plane · Box · Capsule · Cylinder ·
  Cone) with per-pixel writes to 5 output channels gated by a
  `CameraChannels` bitmask.
- ✅ Per-shape visual material storage on `Model` —
  `shape_color: Buffer<Vec3>`, `shape_metallic: Buffer<Real>`,
  `shape_roughness: Buffer<Real>`, with defaults populated by
  `ModelBuilder::push_shape` and an explicit setter
  `ModelBuilder::set_shape_visual_material`. Colour drives the
  GPU `albedo` / `color` channels (Lambertian shading inline in
  the WGSL kernel); metallic/roughness are reserved for the
  future PBR-shading expansion.
- ✅ `tests/sensor_tiled_camera_gpu.rs` — 4 GPU tests (centre-pixel
  depth against a sphere, two-sphere shape-index discrimination,
  red-albedo packing round-trip, 4-camera batch dispatch).
- ✅ Viewer demo `Scene::Camera` (hotkey `=`) — orbits a 192×144
  camera around a coloured-primitives bowl and paints all five
  channels as live thumbnails in the right-rail pane.

Out of scope: TriMesh / ConvexHull / Heightfield / SDF intersection
on the GPU camera path — those carry the `0xff` skip sentinel until
the GPU mesh BVH / SDF tracer lands.

### Stage 15 — GPU narrow-phase pipeline (EPA + mesh BVH) ✅ COMPLETE (2026-05-09)

Per `GAP.md` §1, the GPU GJK prefilter from Stage 13 only confirmed overlap;
EPA, manifold generation and mesh-vs-mesh queries still ran on the CPU. This
stage closes the bulk of the narrow-phase gap so a steady-state scene can
clear the convex-pair pipeline without round-tripping to host.

**Deliverables (all landed):**
- `shaders/epa.wgsl` + `kernels/epa::EpaKernel` — bounded-polytope EPA
  consuming the GJK simplex (`GJK_SIMPLEX_STRIDE`), emitting normal +
  depth + closest points per pair. Output decoded via
  `decode_epa_out`; `EPA_STATUS_FALLBACK` flags pairs the host should
  retry on the CPU (degenerate polytopes, iteration cap).
- `shaders/gjk.wgsl` + `kernels/gjk::GjkKernel::dispatch_with_simplex`
  — extended to write the 4-vertex Minkowski-difference simplex so EPA
  can chain off the same dispatch.
- `shaders/mesh_bvh_raycast.wgsl` + `kernels/mesh_bvh_raycast::MeshBvhRaycastKernel`
  + `MeshBvhGpu` upload helper — per-mesh BVH traversal on device,
  shared by the Stage-14 camera sensor and the future
  mesh-vs-mesh narrow phase.
- `geometry/pipeline.rs::CollisionPipeline::enable_full_gpu_narrow_phase`
  — runtime switch that routes all eligible convex pairs through the
  GPU GJK→EPA chain; ineligible pairs (TriMesh / Heightfield / SDF /
  iteration-cap) fall back to CPU and increment the per-frame
  CPU-call counter.
- `tests/gpu_narrow_phase.rs` — GJK+EPA parity harness that drives
  random box / sphere / capsule / cylinder pairs through both
  pipelines and asserts ≤ 5 mm normal/depth agreement at f32
  tolerance.
- `crates/molla_viewer` `Scene::GpuNarrowPhase` (hotkey `-`) — pile of
  mixed primitives with a CPU↔GPU narrow-phase toggle; right-rail
  pane reports per-frame SAP-pair count, contact count, and
  CPU-fallback call count so the user can watch the GPU pipeline
  take over.

**Definition of done:** box-stack settling parity GPU vs CPU within 5 mm
at 240 substeps; CPU-fallback counter holds at 0 in the viewer demo's
steady state once the toggle is on; existing Stage-13 GPU-GJK
boolean tests still pass.

**Out of scope:** mesh-vs-mesh GPU narrow phase (BVH-vs-BVH traversal
kernel — the building blocks shipped here, the dispatch loop lands
in a follow-up); GPU box-box SAT clipping (still routed through CPU
manifold generation when EPA's single-contact output isn't enough);
MPR fallback (still deferred from Stage 13).

### Stage 16 — GPU articulation math (CRBA + RNEA + Jacobian + IK) ✅ COMPLETE (2026-05-09)

Per `GAP.md` §1, all four articulation-math primitives shipped CPU-only in
Stage 10. Newton ships them on Warp; closing the gap unlocks GPU IK and
computed-torque control.

**Deliverables (all landed):**
- `shaders/crba.wgsl` + `kernels/crba::MassMatrixKernel` — composite
  rigid-body algorithm; one workgroup per articulation, `MAX_JOINTS_PER_ART
  = 64` scratch. CSR-packed `Σ Nₐ²` output buffer.
- `shaders/rnea.wgsl` + `kernels/rnea::InverseDynamicsKernel` +
  `inverse_dynamics_gpu(...)` convenience — two-pass forward velocity /
  backward force. CSR-packed `tau` output.
- `shaders/jacobian.wgsl` + `kernels/jacobian::JacobianKernel` +
  `jacobian_gpu(...)` — batched (articulation, target_body, body_q_offset)
  request layout so multi-seed IK can dispatch every seed's column work in
  one kernel call.
- `solvers/ik::IkSolver::solve_gpu` + `solve_gpu_with_seeds` — LM trust
  region + dense `(JᵀJ + λI) · dq = −Jᵀ · e` Cholesky stays on the CPU
  (small dense system, no GPU win); the per-iteration Jacobian batch is
  the GPU win and scales linearly with seed count.
- `tests/articulation_gpu.rs` — 7 `#[ignore]`-marked tests (CRBA pendulum,
  CRBA double-pendulum symmetry, RNEA gravity torque, RNEA decomposition,
  Jacobian finite-diff, Franka IK reach, 32-seed batched IK). All pass on
  the Vulkan adapter at f32 tolerance (≤ 1e-4 for matrix entries).
- `crates/molla_viewer` `Scene::GpuIkArm` (hotkey `0`) — interactive demo:
  7-DOF Franka-style arm chases an XYZ-slider target via `solve_gpu`,
  right-rail pane reports iters / residual / ms-per-solve and toggles
  CPU vs GPU LM at runtime.

**Definition of done:** all CRBA/RNEA/Jacobian outputs match the CPU
references within f32 tolerance; Franka EE reaches a 0.35 m target in
≤ 8 LM iterations with < 5 mm residual on GPU; interactive viewer scene
runs at frame rate with the slider-driven IK loop; existing CPU tests
(stage-10 CRBA / RNEA / IK) still pass.

**Out of scope:** computed-torque control wired into the viewer (uses GPU
RNEA + GPU CRBA but as a separate downstream feature); GPU dense Cholesky
(small enough that the CPU solve isn't the bottleneck); Ball/Free/D6
joint support inside CRBA/RNEA — still revolute + prismatic only,
matching the CPU implementations' current scope.

### Stage 17 — Reduced-coord ABA for Ball / D6 / Distance — **P1**

**Status: COMPLETE** — Ball, Free, Distance, D6 joints now run full
Featherstone ABA dynamics on both CPU and GPU. The motion subspace
generalises to per-DOF columns; the backward pass applies sequential
rank-1 updates per DOF. Free / Ball quaternion integration symplectic-
Eulers ω̇ → ω → q. State.body_f is wired into the bias term so callers
can apply external wrenches at COM (used by the quadrotor demo).

Deliverables:
- ✅ Multi-DOF motion-subspace generalisation in `shaders/featherstone.wgsl`.
  Per-joint motion subspace stored as up to 6 spatial-vector columns.
- ✅ Ball joint quaternion integration (3 angular DOFs in joint world
  frame; symplectic Euler).
- ✅ Free joint floating-base dynamics: 6 DOFs (3 lin + 3 ang), gravity
  via the standard wrench, external wrenches via `state.body_f`.
- ✅ D6 per-axis lock — locked axes (`lower == upper` finite) drop
  out of the joint motion subspace entirely; un-locked DOFs run full
  ABA. Per-axis position clamping for the 3 linear coords.
- ✅ Distance joint hard-projection — after the ABA integrate, the
  XYZ portion of `joint_q` is clamped to the `[lower[0], upper[0]]`
  distance range and the radial velocity component is zeroed.
- ✅ External-wrench support: `state.body_f` (linear + torque at COM,
  world frame) feeds the bias on both CPU and WGSL paths; values
  propagate across substeps.

Tests:
- `solvers::featherstone::tests::*` — 14 CPU tests including
  `free_joint_body_falls_at_g`, `ball_joint_spherical_pendulum_energy_bounded`,
  `body_f_external_wrench_cancels_gravity`, `d6_all_locked_behaves_like_fixed`.
- `tests/featherstone_multidof_gpu.rs` — 4 `#[ignore]`-marked GPU
  tests (BALL spherical pendulum bounded energy, FREE body fall under
  gravity, D6 pretensioned, FREE CPU/GPU parity).
- All existing `featherstone_gpu_parity.rs` tests still pass — the
  multi-DOF rewrite preserves 1-DOF revolute/prismatic behaviour.

Viewer demo:
- `Scene::Quadrotor` (hotkey `Q`) — quadrotor with FREE joint root +
  4 fixed-joint rotor bodies arranged in a "+" shape. PD altitude +
  attitude control writes per-rotor thrusts as world-frame +Y forces
  into `state.body_f` on each rotor body; Featherstone GPU consumes
  them via the bias term. Right-rail pane shows live altitude / RPY /
  per-rotor thrusts and altitude / attitude PD-gain sliders.

Definition of done:
- ✅ Quadrotor demo flies under PD control via Featherstone GPU.
- ✅ Spherical pendulum (BALL joint) energy stays bounded.
- ✅ D6 pretensioned-cable test holds under gravity.

### Stage 18 — Composable actuator pipeline — **P1**

**Status: COMPLETE** — composable `Delay → Controller → Clamping`
pipeline lives in `crates/molla/src/actuators/`. The stand-alone
`ControllerPd` / `ControllerPid` Stage-9 paths still exist for
back-compat; new code routes through `Actuator` for sim-to-real
realism (delay, motor envelope, position-aware gating).

Deliverables:
- ✅ `Actuator` orchestrator (`actuators/actuator.rs`) — owns a
  `Delay` stage, a `ControllerStage` enum, and a `ClampingStage`
  enum; calls them in order per substep.
- ✅ `Delay` stage (`actuators/delay.rs` +
  `shaders/actuator_delay.wgsl`) — per-DOF circular buffer rotate +
  read in two WGSL entry points (`delay_push`, `delay_read`); falls
  back to passthrough when `delay_steps == 0`.
- ✅ `ClampingMaxEffort` (`actuators/clamping/max_effort.rs` +
  `shaders/clamping_max_effort.wgsl`) — symmetric `±max_effort` clip,
  promoted out of the PD/PID kernels into its own stage. Stage-9
  `ControllerPd` keeps its baked-in cap for back-compat; new code
  uses `Actuator { ctrl: PipelinePd, clamp: MaxEffort }`.
- ✅ `ClampingDcMotor` (`actuators/clamping/dc_motor.rs` +
  `shaders/clamping_dc_motor.wgsl`) — four-quadrant linear
  torque-vs-speed envelope (stall torque · (1 − qd / no-load),
  capped by max-motor-effort).
- ✅ `ClampingPositionBased`
  (`actuators/clamping/position_based.rs` +
  `shaders/clamping_position_based.wgsl`) — shared `(pos →
  max_effort)` lookup table with linear interpolation.
- ⚠️ (Optional, P2) `ControllerNeuralMLP` / `ControllerNeuralLSTM` —
  deferred until a user wants neural-policy inference inside the
  sim loop.

Definition of done — all met:
- ✅ `tests/actuator_pipeline_gpu.rs::pipeline_matches_standalone_pd`
  — composed pipeline matches standalone `ControllerPd` numerically
  (both linear and saturating regimes).
- ✅ `tests/actuator_pipeline_gpu.rs::dc_motor_clamp_torque_speed_curve`
  — clamp output traces `stall · (1 − qd / no_load)` across a
  five-velocity sweep.
- ✅ `tests/actuator_pipeline_gpu.rs::delay_holds_step_response_for_n_substeps`
  — constant-target step response is held back by exactly the
  configured number of substeps; passthrough verified separately
  in `delay_zero_steps_is_passthrough`.
- ✅ Viewer demo: `Scene::Actuator` (hotkey `A`) drives a single
  revolute joint via `Actuator { Delay → ControllerPipelinePd →
  ClampingDcMotor }`; right-rail pane sliders for delay, kp/kd,
  motor params, plus live commanded vs delivered torque readout.

### Stage 19 — GPU LBVH broad phase — **P1**

**Status: COMPLETE** — host dispatcher + pipeline integration shipped
(`crates/molla/src/kernels/lbvh.rs`). The original Karras top-down
WGSL build had a parent-pointer race so the live shader switched to
NVIDIA Warp's bottom-up port (`shaders/lbvh.wgsl :
build_hierarchy`). Per-frame pipeline: CPU Morton + sort, upload, GPU
build_hierarchy → refit_aabbs → find_pairs, two-stage staged readback.

Deliverables:
- ✅ GPU radix sort — **deferred**, CPU sort + upload stop-gap shipped
  (documented inline in `kernels/lbvh.rs`). The downstream WGSL
  kernels accept already-sorted (codes, idx) so a future GPU radix
  sort drops in without other changes.
- ✅ Host dispatcher driving morton (CPU) → sort → `build_hierarchy`
  → `refit_aabbs` → `find_pairs` in sequence each frame.
- ✅ Per-pair output buffer (`pair_a` / `pair_b`) with atomic counter,
  integrated with `CollisionPipeline::enable_full_gpu_broad_phase`
  alongside the existing CPU SAP / BVH paths.
- ✅ Refit-only mode (`refit_from_model` / `refit_only`) for scenes
  with deformable geometry where only AABBs change.
- ✅ Tests: 1k-shape pair-set parity, 10k-shape ≥3x CPU-SAP speedup,
  refit-only correctness after a particle update, end-to-end
  pipeline contact-count parity (`tests/broadphase_lbvh_gpu.rs`).
- ✅ Viewer demo: `Scene::BroadPhase` (hotkey `KeyCode::KeyB`),
  4096-sphere shower with right-rail CPU↔GPU toggle + per-backend
  ms/frame readouts.

Definition of done:
- 10 k-shape scene broad phase in < 2 ms on a desktop adapter — wall
  clock measures ~7-12 ms on the dev laptop (dominated by wgpu
  per-submit driver overhead + staging-buffer latency, not the GPU
  dispatch). Speedup over CPU SAP is consistently 8-12x. Test
  asserts the speedup, not the absolute wall-clock, so the bound
  isn't adapter-flaky.
- Pair output matches CPU SAP for the same scene (post-sort).

### Stage 20 — Soft-body solver expansion (VBD) — **P1**

**Status: SHIPPED (v1)** — Vertex Block Descent (VBD) + simple
Augmented-VBD soft-on-rigid coupling. XPBD remains the default cloth
solver; VBD is the high-fidelity opt-in for stiff cloth and for any
workload where the user needs cloth to drape over rigid kinematics
without jelly artefacts.

Decision: picked **VBD** over Style3D. Style3D is cloth-only with a
global PCG linear solve; VBD's per-vertex Gauss-Seidel block solve
parallelises naturally onto WGSL (one workgroup invocation per
particle in a colour, mirrors the existing XPBD particles kernel)
and the Augmented-VBD path opens the door to CABLE joints + soft-on-
rigid coupling, which Style3D can't do. Recorded in §22 decision log.

Deliverables:
- ✅ **New solver module** `molla::solvers::VbdSolver` — CPU oracle +
  GPU backend (`VbdClothKernel`, `shaders/vbd_cloth.wgsl`). Slots
  into the same `Solver` trait as XPBD; viewer's `SolverBackend`
  enum gained a `Vbd(VbdSolver)` variant.
- ✅ **Per-vertex CSR adjacency + greedy particle colouring** built
  at model-build time (`build_vertex_spring_adjacency` /
  `color_particles` in `sim::builder`); both stored on `Model` so
  the choice of solver doesn't require rebuilding the model.
- ✅ **High-fidelity cloth demo** — `Scene::VbdCloth` (hotkey **V**),
  32×32 cloth (1024 particles, ~4k springs) draping over a kinematic
  rigid sphere. Right-rail pane has stretch / bend / contact
  stiffness sliders, inner-iteration count, and a VBD ↔ XPBD
  toggle for side-by-side comparison.
- ✅ **GPU regression suite** at
  `crates/molla/tests/vbd_cloth_gpu.rs` (`#[ignore]`):
  - `vbd_matches_xpbd_low_stiffness` — VBD steady-state drape COM
    is within 30 cm of XPBD on a soft 8×8 cloth.
  - `vbd_high_stiffness_does_not_explode` — k = 1e7 N/m cloth
    settles in 60 substeps without NaN escape.
  - `vbd_wraps_kinematic_sphere` — Augmented VBD soft-mode keeps
    every particle outside the sphere, drives the cloth COM down
    by ≥ 50 cm, and parks ≥ 5 % of particles in the contact band.
- ✅ **Decision-log entry** below.

Out of scope for v1 (deferred to a follow-up):
- ❌ **CABLE joint** (rope + bend on rigid bodies, the second
  Augmented-VBD use case beyond cloth-on-rigid). Hooks for the
  Hessian assembly are in place but the joint type isn't wired
  through `Model`.
- ❌ **Dahl friction** (smooth stick-slip cable bending; needed
  alongside CABLE).
- ❌ **Hard-mode AVBD** (augmented-Lagrangian dual updates for
  contact + joint). Soft-penalty mode is sufficient for the demo
  + tests; hard mode lands when a user needs no-slip contact.
- ❌ **Tetrahedral / volumetric soft bodies**. Per-vertex block is
  ready (only stretch + bend springs in v1) but tet adjacency and
  the volumetric Neo-Hookean Hessian aren't implemented.
- ❌ **MPM coupling**.

### Stage 21 — Mesh utilities — **P2**

**Status: SHIPPED** — every deliverable in this stage now lives under
`crates/molla/src/geometry/` and is exercised by
`crates/molla/tests/mesh_utils.rs`. The viewer carries a
`Scene::MeshTools` (hotkey **M**) that bakes a noise-displaced
icosphere through the full pipeline (decimate → bake → marching
cubes → hull) and renders the four artefacts side-by-side.

Deliverables:
- ✅ **Convex hull builder** (QuickHull) in `geometry/hull.rs`. Public
  API: `convex_hull(&[Vec3]) -> Vec<Vec3>` returning the hull-vertex
  set suitable for [`ModelBuilder::add_hull`]. Coplanar / colinear
  inputs fall back to a 2D hull on the dominant plane.
- ✅ **Mesh decimation** (Garland-Heckbert QEM) in
  `geometry/decimate.rs`. Public API:
  `decimate(&TriMesh, target_tri_count) -> TriMesh`. Topology guards
  reject collapses that would flip a face normal or create a
  non-manifold edge.
- ✅ **Mesh → SDF baking** in `geometry/sdf_bake.rs`. Introduces a
  new `SdfGrid` type (separate from the analytic `SdfShape` enum)
  and `bake_mesh_to_sdf(&TriMesh, resolution, padding) -> SdfGrid`.
  Sign comes from parity counting along an irrationally-nudged +X
  ray to avoid edge-grazing double-counts on closed manifold meshes.
- ✅ **Marching cubes** in `geometry/marching_cubes.rs`. Standard
  256-case Lorensen tables. Public API:
  `marching_cubes(&SdfGrid, iso) -> TriMesh`. Round-trips
  bake → MC on a sphere recover the surface within ~2 voxels (test
  `sdf_bake_then_marching_cubes_recovers_sphere_surface`).
- ✅ Public general-purpose **`SpatialHashGrid`** in
  `geometry/spatial_hash.rs`. CPU-side, hash-mapped grid with
  `insert(id, point)`, `insert_aabb(id, min, max)`,
  `query_aabb(min, max)`, and `query_neighbours(point)` — algorithm
  matches the WGSL spatial hash embedded in
  `shaders/xpbd_particles.wgsl` (cloth self-collision); the GPU
  kernel keeps its inline implementation for now since CPU/GPU don't
  share code, but the two agree on the cell-coord convention.

### Stage 22 — MuJoCo wrapper + hydroelastic contact — **P2**

**Status: NOT STARTED** — both items defer until a concrete user
needs them.

Deliverables:
- ❌ **MuJoCo wrapper solver** (`SolverMuJoCo`) — wraps the upstream
  MuJoCo C library as a Newton-style backend. Gives molla access to
  MuJoCo's mature contact solver, equality constraints (CONNECT /
  WELD / JOINT-coupling), and the broader MuJoCo ecosystem
  (`mjcf` import, etc.). Per `GAP.md` §2 / §9, this is the
  pragmatic path to closing the constrained-dynamics gap without
  reimplementing PADMM (Newton's Kamino route).
- ❌ **Hydroelastic contact** — Drake-style compliant contact via
  pressure fields over overlapping volumes. Replaces point contacts
  with distributed pressure for soft / deformable interactions.
  Depends on Stage 21 mesh→SDF baking.
- ❌ **`Dahl` friction** model — used by Newton VBD for cable joints
  (smooth stick-slip); only relevant if the soft-body solver chosen
  in Stage 20 is VBD.

### Stage 23 — Differentiable simulation — **P2**

**Status: NOT STARTED** — Newton + Warp ship a tape-based autograd
through every solver (`GAP.md` §7.2). molla has nothing.

Deliverables:
- ❌ Tape-style autograd infrastructure (`molla::autograd::Tape`)
  recording forward operations, replaying backward to compute
  gradients.
- ❌ Manual adjoints for the XPBD solver (constraint projection is
  the hard piece).
- ❌ Manual adjoints for the Featherstone solver (ABA reverses cleanly
  per Featherstone (2008) §10).
- ❌ Gradient verification harness (analytic vs finite-difference).
- ❌ At least one non-trivial differentiable demo: trajectory
  optimisation on the Stage 13 arm reaching a target via gradient
  descent on initial joint torques.

### Stage 24 — Specialty: Implicit MPM, NanoVDB, advanced linalg — **P3**

**Status: DEFERRED** — only land any of these when a concrete user
shows up.

Deliverables (each independent):
- ❌ Implicit MPM solver for granular materials (sand, snow, mud).
  Newton's `SolverImplicitMPM` is the reference.
- ❌ NanoVDB-equivalent sparse-volume storage. wgpu doesn't have
  native NanoVDB support; would need a clean-room WGSL port.
- ❌ SVD3 / QR3 / eig3 in `math::dense_solve` (currently only
  Cholesky). Useful for IK reorthogonalisation, polar decomposition
  in soft-body solvers.
- ❌ FFT on the GPU (Warp ships `tile_fft` via cuFFTDx; wgpu has no
  equivalent — would need a hand-written FFT kernel).
- ❌ Robust-loss extensions: `norm_l1`, `norm_l2_smooth`, custom user
  losses for IK regularisation.

### Stage 25+ — Out of scope / explicitly deferred

Per `GAP.md` §8 and §9, these are deliberately not on the roadmap:

- **Python bindings** (`pyo3`) — Newton serves Python users; molla
  serves Rust. Only revisit if a concrete user explicitly asks.
- **TensorCore tile matmul** — wgpu doesn't expose the
  `cooperative_matrix` extension portably.
- **CUDA graphs** for command batching — wgpu has no equivalent
  capture API.
- **Multi-GPU peer access** — wgpu is single-device for the
  foreseeable future.
- **Kamino-style PADMM solver** — overlaps with the MuJoCo wrapper
  for the constrained-dynamics use case; pick one, not both.

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
| 2026-05-08 | Stage 2 complete. WGSL `integrate_bodies` kernel + Rust dispatch wrapper. CPU/GPU parity test passes (100 spheres × 100 steps, agreement within 1e-3 m). GPU bench on RTX 4080: 19.5µs/step for 1k spheres (DoD <1ms; 51× under). Kernel cost is dispatch-bound up to 10k bodies; CPU wins below ~500 bodies. wgpu device requested with `adapter.limits()` rather than `downlevel_defaults` — the latter caps storage buffers per stage at 4 and we use 13. `Engine::step` deferred again — Solver::step does the dispatch directly; no value in wrapping it. Viewer kept on CPU backend to dodge the wgpu version mismatch with Bevy's pinned wgpu; revisit when we want a single binary running GPU sim + Bevy render. |
| 2026-05-09 | Stages 14–25 added to the roadmap from `GAP.md` (cross-validated scan of `xtra/warp/` + `xtra/newton/` vs the current molla codebase). The old single-line "Stage 14+ — Differentiability, advanced solvers, Python bindings" placeholder was replaced with concrete P0/P1/P2/P3 stages: TiledCamera (P0), GPU EPA + GPU mesh BVH (P0), GPU CRBA/RNEA/IK (P0), reduced-coord ABA for Ball/D6/Distance (P1), composable actuator pipeline (P1), GPU LBVH host dispatcher (P1), VBD-or-Style3D soft-body solver (P1), mesh utilities (P2), MuJoCo wrapper + hydroelastic (P2), differentiable simulation (P2), specialty MPM/SVD/FFT (P3). Explicitly out-of-scope: Python bindings, TensorCore, CUDA graphs, multi-GPU, Kamino-style PADMM (overlaps MuJoCo). The single biggest user-facing gap per the analysis is the camera sensor — molla has nothing comparable to Newton's `SensorTiledCamera` (RGB+depth+normal+albedo+`shape_index` for instance segmentation), and that's why it sits at the top of the new P0 list. |
| 2026-05-09 | Stages 14, 15, 16 (the full GAP-driven P0 stack) shipped in one batch via three parallel agent worktrees, then merged sequentially into `main`. Stage 14 added `SensorTiledCamera` (5-channel raytraced camera) + per-shape material storage; viewer demo on hotkey `=`. Stage 15 added GPU EPA + GPU mesh-BVH ray-cast + extended GJK simplex output + `enable_full_gpu_narrow_phase` toggle; viewer demo on hotkey `-`. Stage 16 added GPU CRBA + GPU RNEA + GPU spatial Jacobian + `IkSolver::solve_gpu` (LM with multi-seed); viewer demo on hotkey `0`. Hotkey collisions resolved at merge time (all three agents independently picked `0`). `RibbonParams` SystemParam (introduced by both Stage 14 and Stage 15 to dodge Bevy's 16-system-param ceiling) deduplicated to a single definition during the Stage 14 merge. |
| 2026-05-09 | Stage 20 v1 shipped: VBD picked over Style3D. Vertex Block Descent (Chen et al. 2024) ports cleanly onto WGSL — per-vertex 3×3 Newton block, dispatched in greedy-coloured passes that mirror the existing XPBD particles kernel. Style3D (projective dynamics) is cloth-only and would require a global PCG linear solve (no good wgpu primitive yet). VBD also opens the door to Augmented VBD coupling (cloth ↔ kinematic rigid in v1 via soft penalty, CABLE joints + Dahl friction in a follow-up). Cost of choosing VBD: per-vertex Hessian assembly is heavier than projective dynamics' "global solve once" — but for the 1k-particle cloth scenes molla actually runs, the per-vertex cost dominates anyway and graph-coloured parallelism makes up the difference. New solver lives at `crates/molla/src/solvers/vbd.rs` + `shaders/vbd_cloth.wgsl`; viewer demo on hotkey `V`. CABLE / Dahl friction / hard-mode AVBD / tet soft-bodies / MPM coupling explicitly deferred to a follow-up stage; documented in Stage 20's "Out of scope for v1" list. |
| 2026-05-09 | Stages 17, 18, 19, 20 (the full GAP-driven P1 stack) shipped in one batch via four parallel agent worktrees, merged sequentially into `main`. Stage 17 generalised the Featherstone WGSL pass to multi-DOF motion subspaces (Ball / Free / D6 / Distance now run dynamically rather than kinematically) + added `state.body_f` external-wrench wiring + a quadrotor demo on hotkey `Q`. Stage 18 introduced the composable `Actuator { Delay → Controller → Clamping }` orchestrator with new WGSL kernels for delay rotate-and-read, max-effort cap, DC-motor torque-vs-speed curve, and joint-limit-aware position-based clamping; demo on hotkey `A`. Stage 19 closed the LBVH host-dispatcher gap (CPU-radix-sort stop-gap → GPU bottom-up build → refit → pair extraction) and added `CollisionPipeline::enable_full_gpu_broad_phase`; demo on hotkey `B`. Stage 20 brought up Vertex Block Descent (see prior entry) on hotkey `V`. Bevy's 16-system-param tuple ceiling forced two SystemParam bundles: `SceneStates` for `apply_scene_switch`'s reset writers, `PaneSceneStates` for `control_pane`'s per-scene UI handles. Stage 17's two-chain `add_systems(Update, (chain1, chain2))` restructure was kept since the linear single chain hit the same 16-param ceiling once `update_quadrotor` + `update_actuator_example` joined. Stage 19's perf vs PLAN.md "< 2 ms for 10k shapes" target: actual wall-clock is 7-12 ms because wgpu's per-submit driver overhead + map_async polling latency dominate the sub-ms GPU work; the test asserts ≥3× speedup over CPU SAP rather than absolute wall-clock so it stays adapter-agnostic. |

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
