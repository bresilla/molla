# MJCF writer

molla ships a **writer** for MJCF (MuJoCo XML Configuration Format),
not a reader. MJCF in molla is an internal serialization the
MuJoCo wrapper solver (Stage 22) emits to bridge a `molla::Model`
into the upstream `libmujoco`. It is not intended as a general
authoring format — users author in URDF or USD, and (if they need
MuJoCo semantics) molla writes the corresponding MJCF on the fly.

## Where it lives

`crates/molla/src/solvers/mujoco.rs` — the `MjcfWriter` struct is
private to the module and is invoked from `MuJoCoSolver::new(...)`
just before handing the document off to `mujoco-rust`.

```rust
let mjcf: String = MjcfWriter::new(model, &cached_body_q, &body_to_mj).write()?;
```

The output is a UTF-8 string that MuJoCo's compiler reads via
`mjModel = mj_loadXMLString(mjcf.as_ptr(), ...)`. It is intentionally
not pretty-printed — MuJoCo doesn't care, and the writer is trivial.

## What it emits

- `<option timestep="0.001" gravity="gx gy gz" integrator="implicitfast"/>` —
  picks `implicitfast` because it matches MuJoCo's default for
  articulated mechanisms (reasonable energy / stability balance).
- `<default>` — a single tiny block making every geom visible
  (`rgba="0.7 0.7 0.7 1"`).
- `<asset>` — one `<mesh/>` per `ShapeGeometry::TriMesh` / `Hull`
  with vertices + indices flattened to space-separated floats; one
  `<hfield/>` per `Heightfield`. Empty when the model has no
  mesh-bearing shapes.
- `<worldbody>` — the full body tree.
  - Each body becomes a `<body name=...>` with `pos` + `quat`
    from `body_q`.
  - Each shape on a body becomes a `<geom type=... size=...>`
    inside.
  - Joints between parent and child bodies become `<joint type=...>`
    elements inside the child.
- `<equality>` — `<weld>` for each fixed joint (so MuJoCo enforces
  rigid attachment); `<connect>` for `EqualityType::Connect`;
  per-pair distance + joint-mimic equality.

The full mapping table:

| molla `ShapeGeometry`         | MJCF `<geom>` |
|-------------------------------|---------------|
| `Sphere { radius }`           | `type="sphere"` |
| `Box { half_extents }`        | `type="box"` |
| `Capsule { radius, half_height }` | `type="capsule"` |
| `Cylinder { radius, half_height}` | `type="cylinder"` |
| `Cone { radius, half_height } `   | approximated as capsule (MuJoCo has no cone) |
| `Hull { ... }`                | `type="mesh"` referencing `<asset><mesh/>` |
| `TriMesh { ... }`             | `type="mesh"` |
| `SdfShape { ... }`            | discarded — MuJoCo has no SDF primitive |
| `Heightfield { ... }`         | `type="hfield"` referencing `<asset><hfield/>` |

| molla `JointType`             | MJCF `<joint>` |
|-------------------------------|----------------|
| `Revolute`                    | `type="hinge"` + `axis` |
| `Prismatic`                   | `type="slide"` + `axis` |
| `Fixed`                       | not a joint; emitted as `<weld>` equality |
| `Free`                        | `type="free"` |
| `Ball`                        | `type="ball"` |
| `D6`                          | composite of three slides + three hinges |
| `Distance`                    | not a joint; emitted as a `<distance>` equality |
| `CABLE`                       | discarded — MuJoCo has no native CABLE; see solvers/mujoco.md |

## Multi-world collapse

The writer **collapses any multi-world model onto world 0**. If
`model.world_count > 1` it logs:

```text
WARN MuJoCo solver: collapsing multi-world model onto world 0;
     multi-env dispatch isn't supported in v1
```

This is a Stage 22 limitation — MuJoCo's API is single-environment;
batched / multi-env stepping would need either independent
`mjData` instances or the upstream MuJoCo MPC / MJX path (out of
scope for the wrapper).

## Why a writer instead of a reader

molla's import path is **URDF → ModelBuilder** for ROS workflows
and **USD → bevy_openusd → Model** for everything else. MJCF as a
reader was rejected at Stage 22 because:

- MJCF is MuJoCo's authoring format. USD is the format the
  graphics + simulation ecosystem is converging on. Adding an MJCF
  reader would duplicate the importer maintenance burden without
  serving a use case URDF or USD can't cover.
- `mujoco-rust` already loads MJCF internally. The wrapper solver
  has a real MuJoCo compiler available; the molla side only needs
  to **emit** MJCF, not parse it.

The decision is documented in `PLAN.md` Stage 22 and in commit
`5bf55df` ("drop MJCF importer — go through usd_molla instead").

## Limitations

- **No actuators in the emitted MJCF.** molla's actuator pipeline
  (PD / PID / DC motor / muscle / neural) runs host-side and writes
  forces into `state.body_f` before each MuJoCo step. MuJoCo's
  `<actuator>` block is not used.
- **No tendons, no soft constraints.** MuJoCo's soft-constraint
  family is rich; molla doesn't use it. Constraint equalities
  (`<weld>`, `<connect>`, `<distance>`, `<joint>`) are the only
  ones emitted.
- **Materials are minimal.** Friction / restitution per shape are
  not yet pushed into `<geom solimp>` / `<geom friction>`; the
  wrapper uses MuJoCo's default contact response.
- **Sensors are not emitted.** molla's sensor pipeline runs
  host-side over the post-step state; MuJoCo's `<sensor>` block is
  not used.

## See also

- [`solvers/mujoco.md`](../solvers/mujoco.md) — the wrapper solver
  that consumes the MJCF output.
- [`architecture/features.md`](../architecture/features.md) — the
  `mujoco` Cargo feature flag.
- `PLAN.md` Stage 22 and 22.1 — full design notes including the
  hydroelastic contact integration.
