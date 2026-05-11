# Introduction

`molla` is a **pure-Rust, wgpu-accelerated physics engine**. It is a
from-scratch reimplementation of the algorithms that ship in
[NVIDIA Newton](https://github.com/newton-physics/newton) and
[NVIDIA Warp](https://github.com/NVIDIA/warp), expressed in idiomatic
Rust with WGSL kernels so that the same binary runs on Vulkan, Metal,
DirectX 12, and (one day) WebGPU. There is no CUDA in the build graph,
no Python runtime in the hot loop, and no vendor lock-in: a laptop with
an AMD APU, an Intel Arc card, an Apple M-series chip, or an NVIDIA
discrete GPU are all first-class targets.

This book is the long-form companion to the source. It covers what the
engine looks like, what the code is doing, and — just as importantly —
what is *not* in scope and why. If you only have five minutes, the
short version is below; if you have an afternoon, jump straight to the
[Architecture overview](./architecture/overview.md) or to the [Scene
catalog](./viewer/scenes.md) for ~100 self-contained Bevy demos that
exercise every solver and kernel.

## What molla is, in one paragraph

A workspace of two crates: `molla` (the engine — math, storage, GPU
runtime, kernels, simulation core, solvers, FEM, sensors, actuators,
autograd) and `molla_viewer` (a Bevy 0.18 application that loads any of
~100 scenes and renders them in real time, with sliders, hotkeys, and
live CPU↔GPU backend toggles). The engine compiles standalone, has no
Bevy dependency, and is the part you embed in your robot, your VFX
plugin, or your headless training loop. The viewer is the showroom.

## Design ethos

molla is opinionated. The choices that shaped the codebase are:

1. **GPU-first, but always CPU-checkable.** Every kernel has a CPU
   implementation that produces the same output (modulo float-op
   ordering). The two paths run side-by-side in `#[ignore]`-gated
   parity tests. The CPU path is not a fallback — it is the oracle.
2. **Mixed precision by design.** Host-side state is `f64`. GPU
   kernels are `f32` because WGSL has no `f64`. Conversions happen
   exactly once, at the `Buffer<T>` boundary in `storage`. No code in
   the math layer ever sees a precision-mixing site.
3. **Data layout mirrors Newton.** `Model` (immutable), `State`
   (positions / velocities / forces), `Control` (joint targets,
   wrenches), `Contacts` (collision output) are the four pillars.
   The shape is good, downstream adapters depend on it, and copying
   it makes Newton's test fixtures usable verbatim as oracles.
4. **One reference algorithm per regime, then expand.** Semi-implicit
   Euler for unconstrained rigid, Featherstone ABA for articulated,
   XPBD for constrained / particles / cloth, VBD for soft bodies, MPM
   for granular and snow, and a full FEM stack (heat / elasticity /
   Stokes / Navier-Stokes / Maxwell). Each was bootstrapped, tested,
   and benched before the next one was started.
5. **Switchable CPU↔GPU at all times.** The `Engine` exposes a
   `Backend::Cpu | Backend::Gpu` toggle. Solvers, sensors, actuators,
   sparse linear algebra, FEM steppers, and the viewer all honor it.
   You can flip the toggle at runtime in the viewer's UI and watch
   the wall-clock change without rebuilding.

## Newton / Warp lineage

molla's reference is the Python tree at `xtra/newton/` and
`xtra/warp/`, both Apache-2.0. We **read** them — we do not port them
line for line. Algorithms are reimplemented clean-room from the
documented behaviour and the test fixtures; Newton remains the
trajectory oracle through release-mode integration tests that run the
same scene in both engines and assert numerical agreement within a
documented tolerance.

When the same primitive exists in Warp (raycast, FFT, marching cubes,
sparse-grid, tile primitives, gaussian splats, autograd tape), molla
ships the same primitive with the same name where it makes Rust
ergonomics work. ~30 scenes in the viewer are direct ports of
`warp/examples/*.py` so behaviour can be eyeballed against the
upstream renderer.

## Why Rust + wgpu?

The original brief — "GPU physics that runs on something other than
NVIDIA" — has only one good answer in 2026: WebGPU as the lingua
franca. `wgpu` is the mature Rust implementation of that spec.
Choosing it gets us Vulkan (Linux desktop + Steam Deck), Metal (Apple
laptops + iPad), DirectX 12 (Windows), and WebGPU (browser) from a
single backend with no platform-specific code in the engine. The cost
is the WGSL feature set: no `f64`, no `atomic<f32>`, no subgroup
intrinsics in the stable spec, no `cooperative_matrix`. The engine is
designed around those constraints — atomic-CAS loops on `u32` reinterpreted
as `f32`, host-resident `f64` state, persistent device buffers — and
it pays back the constraint in portability.

Rust as the host language gives us deterministic memory layout via
`bytemuck`, lifetime-safe buffer ownership through the `Buffer<T>`
abstraction, and the same `glam` types on host (DVec3 / DMat3 /
DQuat) and (via codegen) the device side. The codebase has zero
`unsafe` blocks outside of `Pod`/`Zeroable` derives and the MuJoCo
FFI shim.

## Scope: everything below Stage 51

The roadmap is split into numbered stages, each a self-contained
deliverable with tests + benches + a viewer scene (see
[Stage history](./roadmap/stages.md)). At the time of writing, stages
0 through 51 are complete:

- **0–6**: scaffolding, math, storage, GPU runtime, semi-implicit
  Euler, XPBD rigid + capsule + cylinder + GJK + EPA + BVH +
  trimesh.
- **7–13**: joint types (revolute / prismatic / fixed / free / ball
  / D6 / distance / CABLE), equality constraints (CONNECT / weld /
  joint mimic), articulation (Featherstone ABA + CRBA + RNEA),
  specialty shapes (cone, convex hull, heightfield, SDF).
- **14–19**: GPU sensors (raycast / camera / IMU / contact / frame
  transform), GPU IK, GPU narrow-phase, multi-DOF Featherstone, GPU
  broadphase (LBVH).
- **20–24**: VBD soft bodies (cloth / cable / tets / jelly), mesh
  utilities (decimate / SDF bake / marching cubes), MuJoCo wrapper,
  differentiable Featherstone, raycast adjoint, MPM (explicit +
  semi-implicit + implicit Bingham + 27-stencil), sparse-grid
  (NanoVDB-equivalent).
- **25–35**: hardening, multi-schema USD readers, joint-mimic
  equality, neural LSTM controllers, soft self-collision, soft VT
  contact GPU, cloth-collar self-collision, fast-impact CCD.
- **36–46**: FEM framework v1 (P1 nodal scalar), elasticity (linear
  + St.-Venant-Kirchhoff), Stokes (Taylor-Hood + Uzawa-CG),
  adjoint FEM, GPU FEM, Gmsh import, P2 elements, APIC FEM-MPM
  hybrid, Navier-Stokes (IMEX), DG (P0 / P1 / P2 + Barth-Jespersen
  + HLL/HLLC/Roe SWE flux), magnetostatics (H(curl) Nedelec),
  eddy currents (time-harmonic Maxwell + AMG + AMS), Gaussian
  splat renderer (forward + reverse + SH band 3), AMR, distortion
  energy.
- **47–51**: nonconforming cross-mesh L² projection, surface-only
  cross-mesh quadrature, penalty + Signorini hard contact, Coulomb
  friction, active-set update loop.

Each stage has a test, a bench (where applicable), and a viewer
scene. The stage system is for historical / changelog purposes only —
you don't need to know which stage shipped which kernel to use the
engine. The `prelude` re-exports the most common types and you write
`use molla::prelude::*` and go.

## Explicit out-of-scope

These were considered and deliberately rejected, mostly because they
are either NVIDIA-only or non-portable:

- **Python bindings.** Newton serves Python users; molla serves Rust.
  Will revisit only if a concrete user files an issue with their
  requirements.
- **Style3D cloth.** Newton's production cloth solver — multi-year
  effort, not needed for molla's audience (robotics + games + VFX
  research).
- **Kamino PADMM.** Overlaps with the MuJoCo wrapper for the
  constrained-dynamics use case; we ship the wrapper instead.
- **TensorCore tile matmul / `cooperative_matrix`.** wgpu does not
  expose the extension portably across the four backends. molla's
  tile-matmul kernel is a plain WGSL implementation.
- **CUDA Graphs.** wgpu has no capture / replay API. The molla
  equivalent is persistent device buffers + single-encoder dispatch
  batching (see Stage 24d.2 for the MPM case study).
- **Multi-GPU peer access.** wgpu is single-device. The `Engine`
  type owns one `wgpu::Device` and one `wgpu::Queue`.
- **General LCP / NLP solver.** Position-based constraints (XPBD)
  and reduced-coordinate articulation (Featherstone) cover the
  workhorse cases. The MuJoCo wrapper (Stage 22) is there when you
  need a real LCP / pyramid friction model.
- **Own renderer.** Visualization piggybacks on Bevy (in the viewer
  crate) and the Gaussian-splat differentiable renderer for the
  diff-render scenes. The engine has zero renderer dependencies.

## Audience

This book is written for:

- **Rust users** who want a physics engine that doesn't drag a CUDA
  toolchain into their build.
- **Robotics researchers** who want differentiable trajectories on
  the laptop they already own.
- **Graphics + simulation engineers** who want a clean reference
  reimplementation of Newton / Warp's algorithms with readable,
  test-covered Rust.
- **Anyone porting from Python Newton** — the data model is
  intentionally familiar and the test fixtures are reused.

What this book is *not*: an introduction to physics simulation in
general. We assume you know what a quaternion is, what XPBD does in
broad strokes, and why a semi-implicit Euler is different from RK4.
Newton's [docs](https://newton-physics.github.io/newton/) and the
seminal papers each chapter links to are the right starting points
for the underlying numerics.

## How this book is organized

- **Foundations** — what's in `crates/molla`, how the workspace is
  laid out, what each Cargo feature gates, and the design rules every
  contributor needs to follow.
- **Math & Numerics** — the host `f64` math layer, the BSR sparse
  algebra, iterative solvers, optimizers, graph coloring, radix sort.
- **Compute & GPU** — the `wgpu` runtime, the `Buffer<T>` /
  `Storage` abstraction, the kernel infrastructure, the tile-compute
  primitives, the persistent-buffer pattern.
- **Geometry & Collision** — shape primitives, hull builder, BVH,
  spatial hash, broad-phase, narrow-phase (GJK / EPA / MPR),
  hydroelastic, sparse grids.
- **Simulation Core** — `Model` / `State` / `Control` / `Contacts`,
  `ModelBuilder`, joints, articulation, CRBA / RNEA / Jacobian, IK.
- **Solvers** — semi-implicit, XPBD, VBD, MPM (three flavors),
  MuJoCo wrapper.
- **Finite-Element Method** — the FEM v1 framework and everything
  built on top (P2, DG, Stokes, Navier-Stokes, APIC, adjoint, AMR,
  magnetostatics, eddy currents, multigrid, nonconforming meshes,
  Signorini contact, Coulomb friction).
- **Sensors** — raycaster, tiled camera, IMU, contact, frame
  transform.
- **Actuators & Control** — composable pipeline, PD / PID, neural
  controllers (MLP + LSTM), DC motor clamping, Hill-type muscle.
- **Differentiable Physics** — tape-based autograd, Featherstone
  reverse adjoint, XPBD / VBD / MPM / FEM / raycast / splat
  adjoints.
- **Importers** — URDF, MJCF writer, USD bridge, Gmsh.
- **Viewer** — the Bevy adapter, the scene catalog.
- **Testing & Performance** — strategy, parity tests, FD gradient
  checks, criterion benches, GPU port status matrix.
- **Roadmap** — chronological stage history, open follow-ups, the
  out-of-scope list.

Start where you need. The [Scene catalog](./viewer/scenes.md) is
the most fun way in if you just want to see what the engine does.
The [Architecture overview](./architecture/overview.md) is the
right place if you're embedding the engine in something else and
want to know what each layer is doing.
