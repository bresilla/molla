# Crate boundaries

molla currently ships two member crates and they have a strict
linear dependency graph.

```
            ┌────────────────────┐
            │   molla_viewer     │   Bevy 0.18 application
            └─────────┬──────────┘
                      │ depends on
                      ▼
            ┌────────────────────┐
            │       molla        │   the engine — no Bevy / USD
            └────────────────────┘
```

## The rules

1. **`molla` knows nothing about Bevy, USD, ROS, JSON, or YAML.** It
   only knows about `Model` / `State` / `Control` / `Contacts` and
   the WGSL kernels it owns.
2. **`molla_viewer` is allowed to depend on Bevy + bevy_egui +
   bevy_frost + bevy_glacial + mimalloc.** That is the full
   surface; the viewer crate is the only place those dependencies
   live.
3. **Importers live inside `molla`.** URDF (`urdf-rs`) lives at
   `molla::utils::urdf`. Gmsh (`mshio`) lives at
   `molla::fem::FemMesh::from_gmsh_*`. The MJCF writer lives at
   `molla::solvers::mujoco::MjcfWriter` because only the MuJoCo
   wrapper consumes it. They are *not* a separate crate — that
   would add a workspace member without changing the dep graph
   shape.
4. **Optional native dependencies are behind Cargo features.** See
   [features.md](./features.md). The features gate compilation of
   whole modules so a clean `cargo build` on a stock Rust toolchain
   succeeds without MuJoCo / SuiteSparse / LAPACKE installed.
5. **There is no Python in the build graph.** Newton's Python tree
   is at `xtra/newton/` and is *not* a workspace member. The
   `test-oracle` Makefile target shells out to a separately-managed
   venv.

## Why so few crates?

molla started with the option to split into many small crates (a
`molla-math`, `molla-storage`, `molla-fem`, `molla-solvers`, etc.)
and explicitly rejected it. The reasons:

- **One feature graph** is easier to reason about than ten. The
  `gpu` feature gates every WGSL kernel; if it were spread across
  five crates the gate logic would need to be replicated and kept
  consistent.
- **One `Cargo.toml`** keeps version pins consistent (especially
  `wgpu` + `glam` + `bytemuck`, which are tightly coupled).
- **Incremental compile times are good.** Rust's incremental rebuild
  on a single library crate is fast enough that the wallclock
  argument for splitting doesn't materialise.
- **Public API is enforced by `prelude` + module-level `pub use`.**
  Even though everything is one crate, the surface a user touches
  is `molla::prelude::*` and not the 200+ items underneath. If a
  type is not in `prelude`, the docs note that it is an
  implementation detail.

When a future user explicitly wants only the `math` layer (no GPU,
no `wgpu`, possibly `no_std`), we will split `crates/molla_math/`
out as a sibling. Until then, splitting is overhead without
benefit.

## What used to be a separate crate

- **`crates/usd_molla/`** — existed before Stage 8 as the
  UsdPhysics↔molla bridge. It read `openusd::Stage`, walked the
  `UsdPhysics` opinions (`RigidBodyAPI` + `MassAPI` +
  `CollisionAPI`, `RevoluteJoint`, `SphericalJoint`, etc.) and
  produced a `molla::Model`. It also implemented `project_state_to_stage`
  for writing simulation results back as time samples on the same
  prims. It was removed in commit `8c81b13` — molla now integrates
  with USD via the upstream `bevy_openusd` ecosystem rather than
  maintaining a sibling crate. See
  [`importers/usd.md`](./../importers/usd.md).

## Workspace exclusions

```toml
[workspace]
resolver = "2"
members = ["crates/molla", "crates/molla_viewer"]
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
- **`mujoco` and `russell` are off by default** and pull in system
  C dependencies (`libmujoco` for `mujoco`; nothing native at the
  moment for `russell`, but the feature flag is reserved for the
  upstream SuiteSparse-based solver).
- **The `prelude` is stable.** Anything outside `prelude` may move
  between minor releases until 0.1.0.

The crate is on `0.0.1` at present; no SemVer guarantees yet. The
PLAN's quality bar is that every shipped feature has a trajectory or
unit test, an example, a bench, and CPU/GPU parity — that is the
unofficial release criterion.
