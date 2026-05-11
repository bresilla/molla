# MuJoCo wrapper

[`MuJoCoSolver`](https://docs.rs/molla/latest/molla/solvers/struct.MuJoCoSolver.html)
wraps the upstream **MuJoCo** C library as a Newton-style backend
([`Solver`](./overview.md) impl). Stage 22 deliverable. Gated by the
`mujoco` Cargo feature.

## Why a CPU wrapper inside a GPU-first engine?

molla's house rule is **GPU-first**; this is the documented
exception. MuJoCo is the industry-reference contact + constraint
solver (Todorov 2014 PADMM, frictional contact via convex projection,
mature equality constraints, the entire [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
of robots). Re-deriving PADMM in WGSL is a multi-month effort;
wrapping the C library is a one-week effort that unlocks the entire
ecosystem — `mjcf`, MuJoCo Menagerie, validated against published
benchmarks. See `PLAN.md` Stage 22 and `GAP.md` §2 / §9 for the full
rationale.

## How it works

```text
Model (molla)
     │ MJCF writer  (build_mjcf)
     ▼
<mujoco>...</mujoco>  XML
     │ mujoco_rust::Model::from_xml_str
     ▼
mjModel (compiled MuJoCo)
     │ mjData allocation
     ▼
mjData (sim state)
     │ MuJoCoSolver::step:
     │   1. Write molla state_in → mjData qpos/qvel
     │   2. mj_step × substeps
     │   3. Read mjData qpos/qvel → state_out
     │   4. Refresh body_q from MuJoCo body world transforms
     ▼
state_out (molla)
```

1. [`MuJoCoSolver::new`](https://docs.rs/molla/latest/molla/solvers/struct.MuJoCoSolver.html#method.new)
   takes a [`Model`](../sim/model.md) and emits an MJCF XML document
   describing it (bodies, joints, shapes, equality constraints,
   gravity).
2. `mujoco_rust::Model::from_xml_str` parses + compiles the MJCF into
   MuJoCo's internal `mjModel`.
3. [`MuJoCoSolver::step`] writes molla's `state_in` into MuJoCo's
   `mjData.qpos` / `qvel`, calls `mj_step` (`mujoco_rust::Simulation::step`)
   `substeps` times, then reads `qpos` / `qvel` back into `state_out`
   along with refreshed body world poses.

```rust,ignore
use molla::solvers::{MuJoCoSolver, Solver};

let mut solver = MuJoCoSolver::new(&model)?;
solver.substeps = 4;                     // mj_step calls per outer step

solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out)?;

println!("MJCF source:\n{}", solver.mjcf_source());        // for debugging
```

## What's supported in v1 (Stage 22)

- **Joints**: Revolute, Prismatic, Ball (rotation only), Fixed
  (rendered as no joint at all — child becomes part of parent), Free
  (6-DoF root joint). D6 / Distance round-trip to `Free`; per-axis
  locks aren't translated yet.
- **Shapes**: Sphere, Box, Capsule, Cylinder, Plane. Triangle meshes
  / convex hulls / heightfields / SDFs handled in Stage 22.1.
- **Equality constraints**: `EqualityType::Connect` → MJCF
  `<connect>`, `EqualityType::Weld` → MJCF `<weld>`. The first 3 / 6
  algebraic constraints flow through MuJoCo's hard solver.
- **Gravity**: world 0's gravity vector becomes the MJCF `<option>`
  gravity attribute. Multi-world models collapse to the world-0
  gravity (MuJoCo doesn't have a multi-world concept; the solver
  emits a `tracing::warn`).

## What Stage 22.1 added on top of v1

- **TriMesh / ConvexHull / Heightfield assets** — emitted as
  `<asset><mesh.../></asset>` + `<geom mesh="..."/>` (and `<hfield/>`
  for heightfields) so URDF / USD imports with mesh-collision
  round-trip into MuJoCo natively.
- **SDF shapes** — converted to a baked mesh via
  [`crate::geometry::marching_cubes`](../geometry/sdf.md) at
  translation time (MuJoCo has no native SDF primitive). Lossy:
  surface fidelity is bounded by the SDF grid resolution, and an
  analytic-SDF variant (e.g. `SdfShape::Torus`) gets discretised into
  the triangulated approximation. A `tracing::info` records the
  conversion.
- **Per-DOF joint limits, friction, armature** — emitted via
  `<joint range="..." damping="..." armature="..."/>`.
- **FIXED joints** — emitted as MJCF `<equality><weld/>` constraints
  between the parent body and the FIXED-joint child (instead of a
  loose freejoint that could drift). The child body is still emitted
  with full inertia so the constraint solver can hold it rigidly.

## What's NOT supported in v1

Documented as gaps so the user knows what to expect; promote to ✓ in
a follow-up as the gap is closed:

- **Mimic constraints** — MJCF supports them via `<equality
  type="joint">` + per-joint coupling polynomials, but molla's mimic
  API is minimal (`q_a = c0 + c1·q_b`) and the round-trip isn't wired
  yet. (The MJCF→molla direction works via the `usd_molla` importer's
  MJCF compatibility shim.)
- **Per-shape friction / restitution overrides** — MJCF supports
  per-geom `friction="..." solref/solimp"..."` attributes; currently
  we use MuJoCo's defaults.
- **Multi-world dispatch** — MuJoCo is single-world. Multi-world
  models collapse onto world 0 with a warn.
- **Particles / springs / cloth** — out of scope for the rigid-body
  wrapper. Use [XPBD](./xpbd.md) or [VBD](./vbd.md) for those.

## Feature gating

```toml
# Cargo.toml
[features]
mujoco = ["mujoco-rust"]
```

When the `mujoco` Cargo feature is **off** (the default), the module
exposes `MuJoCoSolver` as a **stub** that errors at runtime with
`Error::Build`. The type, MJCF builder, and `Solver` impl all compile
so callers can write `#[cfg]`-free user code — the `step` call just
returns an error if you try to use it without the feature.

When the `mujoco` feature is **on**, the module pulls in `mujoco_rust`
and provides a working solver. The `mujoco_rust` crate's build script
searches for `libmujoco` at `~/.local/mujoco` or in standard system
search paths; missing the library produces a clear linker error at
compile time. **molla doesn't try to install MuJoCo for you** — see
<https://mujoco.org/> for download links.

```bash
# typical workflow
$ mkdir -p ~/.local/mujoco && tar xzf mujoco-3.x-linux-x86_64.tar.gz \
        -C ~/.local/mujoco --strip-components=1
$ cargo test --features mujoco
```

## MJCF round-trip — debugging

The MJCF source is kept on the host so you can dump it:

```rust,ignore
let solver = MuJoCoSolver::new(&model)?;
println!("{}", solver.mjcf_source());
```

This is mostly useful when MuJoCo's compiler rejects the XML (the
rejection message includes the offending token; cross-reference it
against the generated MJCF). The `tests/mujoco_mjcf_roundtrip.rs`
suite parses each generated MJCF through MuJoCo's compiler as a
fitness check.

## Parity testing

The Stage 22 deliverable includes parity tests against the
[Featherstone solver](../sim/featherstone.md) on:

- Single-pendulum drop test
- Double pendulum (chained revolutes)
- 6-link humanoid leg with Ball joints at hip + knee
- 4-bar linkage with `<equality><joint/>` mimic

The tests check that MuJoCo and Featherstone agree on positions and
velocities to ~ 1e-3 over a 1-second sim — both solvers integrate the
same equations of motion, so they should differ only by their
integrators' truncation error.

## When to choose MuJoCo

- **Reference comparison** — validating molla's Featherstone /
  XPBD against the de-facto reference engine.
- **MuJoCo Menagerie scenes** — running published robotics scenes
  without re-authoring them in molla's `ModelBuilder`.
- **Contact-heavy stacking** — MuJoCo's PADMM is more mature than
  molla's XPBD per-body Jacobi for tall stacks of frictional rigid
  bodies (Stage 22.x doesn't aim to beat MuJoCo on this).

When **not** to choose MuJoCo:

- GPU dispatch (no GPU port).
- Soft bodies, particles, cloth, MPM (use the dedicated solvers).
- Multi-world RL (use Featherstone or XPBD on multi-world Models).
- Differentiable physics (use the molla autograd tape).
