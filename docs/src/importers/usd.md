# USD importer

molla's relationship to USD has evolved across the roadmap. The
current state and the history both matter, so this chapter covers
both.

## TL;DR

- **At present** there is **no** in-tree USD importer crate. USD
  scenes reach molla through the upstream
  [`bevy_openusd`](https://github.com/bresilla/bevy_openusd)
  adapter, which emits marker components from `UsdPhysics` opinions
  and a downstream Bevy plugin walks them into a `molla::Model` via
  `ModelBuilder`.
- **Previously** (Stages 5–8, Stage 29) molla shipped a sibling
  crate `crates/usd_molla/` that handled the bridge directly. It
  was removed in commit `8c81b13` (May 2026) to consolidate around
  the bevy_openusd path.
- **Long term** the brief is unchanged: USD physics is the
  primary import path. A USD scene with `UsdPhysics` opinions
  becomes a `molla::Model` with no manual `ModelBuilder` calls; a
  molla simulation result writes back as time samples on the
  authored prims.

## The mapping that was, and will be again

The mapping below is the historical Stage 8 mapping from the
removed `usd_molla` crate. It is reproduced here because it is the
contract the next iteration will re-implement:

| UsdPhysics                       | `molla::Model`                                    |
|----------------------------------|---------------------------------------------------|
| `UsdPhysicsScene`                | per-world gravity entry                           |
| `RigidBodyAPI` (+ `MassAPI`)     | `BodyParams` (mass / inertia / com / kinematic)   |
| `CollisionAPI` on a Sphere/Cube/…| `ShapeGeometry::{Sphere, Box, Capsule, …}`        |
| `PhysicsMaterial`                | per-shape friction / restitution                  |
| `RevoluteJoint`                  | `JointType::Revolute`                             |
| `PrismaticJoint`                 | `JointType::Prismatic`                            |
| `FixedJoint`                     | `JointType::Fixed`                                |
| `SphericalJoint`                 | `JointType::Ball`                                 |
| `DistanceJoint`                  | `JointType::Distance`                             |
| `Joint` (generic 6-DOF)          | `JointType::D6`                                   |
| `CollisionGroup`                 | `Model.shape_collision_group` bitmask             |
| `PhysicsLimitAPI:<dof>`          | `Model.joint_limit_lower / upper`                 |
| `PhysicsDriveAPI:<dof>`          | `Control.joint_target_*` defaults                 |
| `JOINT-mimic` (custom relation)  | `Model.constraint_mimic_*`                        |

The intent was an end-to-end round-trip: load a `.usda` stage,
simulate, and write the per-frame body poses and velocities back
as time samples on `xformOp:translate`, `xformOp:orient`,
`physics:linearVelocity`, `physics:angularVelocity`. Stage 8
shipped the round-trip; Stage 29 added schema resolvers for the
PhysX and MJC schemas so a stage authored against either could
also be loaded.

## The bevy_openusd path

The current path is:

1. Author the scene in any DCC tool that emits USD with
   `UsdPhysics` opinions (Houdini, Blender via the official USD
   plugin, Omniverse, or by hand in USDA).
2. Load the stage in `bevy_openusd` — it walks the prim tree and
   emits marker components on Bevy entities.
3. A user-side adapter system reads those marker components and
   builds a `molla::Model` via `ModelBuilder`. The molla side of
   this adapter is what the deleted `usd_molla` crate used to do
   inline.
4. Step the simulation; copy poses back to the entities; the
   `bevy_openusd` exporter (when called) writes the stage back as
   time samples.

This split — molla holds the engine, bevy_openusd holds the USD
parsing, the user (or a future molla extension) writes the glue —
keeps the engine crate free of any USD dependency.

## Why the change

The removal commit (`8c81b13`) notes "will use bevy_openusd
later". The motivation:

- **`openusd-rs` and `bevy_openusd` matured.** When the original
  `usd_molla` shipped, the Rust USD ecosystem was patchy. By
  Stage 29 enough of the bevy_openusd story had landed that the
  USD plumbing made more sense to consolidate there.
- **`usd_molla` overlapped with `usd_rapier` / `usd_bevy`.** The
  three sibling crates (`usd_molla`, `usd_rapier`, `usd_bevy`)
  duplicated about 80 % of the prim-walking logic — only the
  engine-side endpoint differed.
- **Reduce maintenance surface.** Deleting 2400 lines that
  duplicated upstream functionality is a net win.

## Status today

- Recent commits referencing `usd_molla` are historical (Stages
  5, 7, 8, 29 work, all merged before the May 2026 removal).
- The `docs/src/SUMMARY.md` chapter on USD points here for
  context; the live importer story is "go through bevy_openusd".
- A future molla extension crate (working name TBD) may take the
  role the deleted `usd_molla` had — but only when a concrete user
  files an issue with their requirements.

## See also

- `PLAN.md` Sections 14 and 15 — the original USD adapter design.
- Commit `8c81b13` ("chore: remove usd_molla crate (will use
  bevy_openusd later)") — the deletion + rationale.
- Commit `152bc6f` ("feat(usd_molla): end-to-end UsdPhysics
  round-trip — stage 8 complete") — the original Stage 8 work.
- Commit `43b5524` ("feat(usd_molla): multi-schema USD readers
  (PhysX + MJC)") — Stage 29 multi-schema work.
- [bevy_openusd](https://github.com/bresilla/bevy_openusd) — the
  upstream Bevy USD adapter.
