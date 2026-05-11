# Data model

The simulation core is built around four host-allocated, GPU-mirrored
structures that every solver consumes via the
[`Solver::step`](../solvers/overview.md) contract:

```text
                ┌─────────────┐
                │   Model     │   immutable: bodies, shapes, joints,
                │  (Buffer<T>)│   particles, springs, equality
                └─────┬───────┘
                      │
  ┌──────────┐ step  ┌─▼──────────┐  ┌─────────┐  ┌──────────┐
  │  State   │──────▶│   Solver   │◀─│ Control │  │ Contacts │
  │ (q, qd, f│       │  (XPBD/Aba │  │ (drives,│  │ (broad+  │
  │  body, …)│◀──────│   VBD/MPM) │  │ wrenches│  │  narrow) │
  └──────────┘       └────────────┘  └─────────┘  └──────────┘
```

All four types are defined in [`crates/molla/src/sim/`][sim] and re-exported
from `molla::sim`.

[sim]: https://github.com/molla/molla/tree/main/crates/molla/src/sim

## `Model` — static, immutable scene description

[`Model`](https://docs.rs/molla/latest/molla/sim/struct.Model.html) is built
once via a [`ModelBuilder`](./builder.md) and then **never mutated** by
solvers. Every field is a `Buffer<T>` — a host `Vec<T>` paired with a
lazily-allocated GPU mirror (`wgpu::Buffer`). The host slab is always
authoritative; `Model::upload_to_device(device, queue)` ensures the GPU
mirror matches.

### Bodies, particles, shapes

```rust,ignore
pub struct Model {
    // Particles
    pub particle_count: usize,
    pub particle_mass:     Buffer<Real>,
    pub particle_inv_mass: Buffer<Real>,
    pub particle_radius:   Buffer<Real>,
    pub particle_flags:    Buffer<u32>,
    pub particle_world:    Buffer<i32>,
    pub initial_particle_q: Buffer<Vec3>,

    // Rigid bodies
    pub body_count: usize,
    pub body_mass:        Buffer<Real>,
    pub body_inv_mass:    Buffer<Real>,
    pub body_inertia:     Buffer<Mat3>,
    pub body_inv_inertia: Buffer<Mat3>,
    pub body_com:         Buffer<Vec3>,
    pub body_flags:       Buffer<u32>,
    pub body_world:       Buffer<i32>,
    pub initial_body_q:   Buffer<Transform>,

    // Collision shapes
    pub shape_count: usize,
    pub shape_body:        Buffer<i32>,   // owning body, -1 = world
    pub shape_geom_kind:   Buffer<u32>,   // GPU discriminant
    pub shape_geom_data:   Buffer<Vec3>,  // packed scalar params
    pub shape_transform:   Buffer<Transform>,
    pub shape_friction:    Buffer<Real>,
    pub shape_restitution: Buffer<Real>,
    // (…rolling/torsional friction, collision groups, PBR colours)
    pub meshes:        Vec<TriMesh>,       // host-side
    pub hulls:         Vec<Vec<Vec3>>,
    pub heightfields:  Vec<Heightfield>,
    pub sdfs:          Vec<SdfShape>,
    // …
}
```

Every per-entity property is a **flat parallel array**: a body `i` is
the i-th entry of `body_mass`, `body_inertia`, `body_com`, … and so on.
This Newton/Warp-style "Struct of Arrays" layout maps directly onto
WGSL storage buffers — kernels index by thread id.

The GPU shape representation flattens the [`ShapeGeometry`](./builder.md#shapegeometry)
enum into `shape_geom_kind` (u32 discriminant: 0=Sphere, 1=Plane,
2=Box, 3=Capsule, 4=Cylinder, 5=Cone, 6=Ellipsoid, 0xff=host-only) and
`shape_geom_data` (`Vec3` of packed scalars). Mesh / hull / heightfield
/ SDF backing storage remains host-side via the `meshes` / `hulls` /
`heightfields` / `sdfs` vectors, indexed by a `mesh_id` / `hull_id` /
… `u32` stored in `shape_geom_data`.

### Constraints

```rust,ignore
// Distance constraints (cloth edges, ropes, springs)
pub spring_count: usize,
pub spring_a:              Buffer<u32>,
pub spring_b:              Buffer<u32>,
pub spring_rest_length:    Buffer<Real>,
pub spring_compliance:     Buffer<Real>,   // XPBD α = 1/k
pub spring_max_stretch:    Buffer<Real>,
pub spring_color_count:    usize,
pub spring_color_offsets:  Buffer<u32>,    // CSR for parallel solve
pub spring_indices_by_color: Buffer<u32>,

// Equality constraints (CONNECT / WELD / JointMimic)
pub equality_count: usize,
pub equality_type:     Buffer<u32>,
pub equality_body_a:   Buffer<i32>,
pub equality_body_b:   Buffer<i32>,
pub equality_anchor_a: Buffer<Vec3>,
pub equality_anchor_b: Buffer<Vec3>,
// …

// Mimic constraints: q_a = c0 + c1·q_b
pub mimic_count: usize,
pub mimic_joint_a: Buffer<i32>,
pub mimic_joint_b: Buffer<i32>,
pub mimic_c0:      Buffer<Real>,
pub mimic_c1:      Buffer<Real>,
```

Spring constraints carry a **graph colouring** (greedy, computed at
build time) so the XPBD GPU kernel can solve a colour class in
parallel without write-races. The same colouring is exposed at the
**vertex** level (`vertex_adjacent_springs_*` + `particle_indices_by_color`)
for VBD's per-vertex Gauss-Seidel sweeps. See [XPBD](../solvers/xpbd.md)
and [VBD](../solvers/vbd.md) for how the CSR layout is consumed.

### Joints & articulations

```rust,ignore
pub joint_count: usize,
pub joint_type:   Buffer<u32>,    // JointType discriminant
pub joint_parent: Buffer<i32>,
pub joint_child:  Buffer<i32>,
pub joint_x_p:    Buffer<Transform>,
pub joint_x_c:    Buffer<Transform>,
pub joint_axis:   Buffer<Vec3>,
pub joint_coord_offset: Buffer<u32>,  // CSR into joint_q
pub joint_dof_offset:   Buffer<u32>,  // CSR into joint_qd
pub joint_coord_total: usize,
pub joint_dof_total:   usize,

pub joint_limit_lower:    Buffer<Real>,
pub joint_limit_upper:    Buffer<Real>,
pub joint_friction:       Buffer<Real>,
pub joint_effort_limit:   Buffer<Real>,
pub joint_velocity_limit: Buffer<Real>,
pub joint_target_ke:      Buffer<Real>,  // joint-drive stiffness

// Articulations (kinematic trees) — Featherstone dispatches one
// workgroup per articulation.
pub articulation_count: usize,
pub articulation_start: Buffer<u32>,  // CSR over joints
```

Per-DoF arrays (limits, friction, effort cap, drive stiffness) are
indexed by `joint_dof_offset[j] + k` for joint `j`'s k-th DoF. See
[Joints](./joints.md) for the DoF accounting.

### Tetrahedra, cables, muscles, soft surfaces

Tetrahedral soft bodies (`tet_count`, `tet_indices`, `tet_dm_inv`, Lamé
constants, plasticity state), cables (`cable_count`, `cable_particles`,
bending stiffness CSR), muscles (`muscle_count`, Hill parameters, CSR
`joint_muscle_*`), and cross-mesh soft surfaces (`soft_surfaces`,
`soft_surface_tris`, EE/VT adjacency) all live on `Model` with the same
"flat SoA" discipline.

### Multi-world support

Every per-entity slab carries a `*_world: Buffer<i32>` companion: bodies,
particles, joints, articulations, equality constraints, and mimics
each remember which world they belong to. `Model.world_count` and
`Model.gravity: Buffer<Vec3>` (length `world_count`) parameterise per-world
gravity. Solvers honour world ownership by routing each body/particle
into its world's gravity slot during integration.

This lets a single `Model` describe N parallel RL environments that
the GPU dispatches as one workgroup grid — `world_count` is the
outermost dimension. Newton's "worlds" concept maps directly.

### Global solver knobs

```rust,ignore
pub angular_damping: Real,        // applied per step
pub linear_damping:  Real,
pub particle_velocity_max: Real,  // stability cap
pub contact_margin: Real,         // broad-phase AABB slack
pub default_friction:    Real,
pub default_restitution: Real,
```

These are scalars (not buffers) — every solver reads them once per
step. Per-shape overrides (`shape_friction`, etc.) take precedence
where set (sentinel `NaN` = "use default").

## `State` — time-varying simulation data

```rust,ignore
pub struct State {
    pub particle_q:  Buffer<Vec3>,
    pub particle_qd: Buffer<Vec3>,
    pub particle_f:  Buffer<Vec3>,

    pub body_q:  Buffer<Transform>,      // pose (body-frame origin in world)
    pub body_qd: Buffer<SpatialVector>,  // (v_lin, ω) at COM, world frame
    pub body_f:  Buffer<SpatialVector>,  // applied wrench at COM, world

    pub joint_q:  Buffer<Real>,
    pub joint_qd: Buffer<Real>,

    // VBD plasticity (Stage 20.3/20.4)
    pub tet_f_plastic:      Buffer<Mat3>,
    pub tet_plastic_strain: Buffer<Real>,
    pub tet_back_stress:    Buffer<Mat3>,
}
```

`State` holds **everything the solver mutates**. The integrator
contract is _double-buffered_:

```rust,ignore
solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out);
```

Solvers may ping-pong `state_in` and `state_out` internally across
substeps (see [Featherstone](./featherstone.md#substepping) and
[XPBD](../solvers/xpbd.md#substepping)), but to the caller it's a
simple A→B copy.

Allocate via `model.state()`. The factory copies `initial_body_q`,
`initial_particle_q`, `initial_joint_q`, and (for tet meshes)
`initial F_p` into the working state; velocities + forces start at
zero. `State::clone_host` makes a host-side replica (used by
substepping wrappers) and `State::copy_host_from` does in-place
copies between same-shape states.

`body_q.translation` is the **body-frame origin in world**, not the
COM (the COM offset is `Model.body_com`). The angular component of
`body_qd` is the body angular velocity in **world frame**, not
body frame — same convention as Newton / Warp.

## `Control` — per-step user inputs

```rust,ignore
pub struct Control {
    pub joint_target_mode: Buffer<u32>,            // per-DoF
    pub joint_target_pos:  Buffer<Real>,           // (rad / m)
    pub joint_target_vel:  Buffer<Real>,
    pub joint_target_kp:   Buffer<Real>,           // PD gain
    pub joint_target_kd:   Buffer<Real>,
    pub joint_force:       Buffer<Real>,           // feed-forward
    pub body_external_f:   Buffer<SpatialVector>,  // per-body wrench
}
```

`joint_target_mode` selects the per-DoF drive policy from
[`JointTargetMode`](./joints.md#targets-and-drives) (`None`, `Position`,
`Velocity`, `PositionVelocity`, `Effort`). The Featherstone backward
pass folds drive torque + feed-forward `joint_force` + joint friction
into each DoF's `u_k`, then clamps by `joint_effort_limit`. See
[Featherstone](./featherstone.md#joint-drives--limits--friction).

Per-body external wrenches (`body_external_f`) are added at the COM in
world frame — typical use is RL-style pushers, hand-of-god dragging,
or aerodynamic forces from a wing model.

Allocate via `model.control()`; everything starts zero (drive mode =
`None` = passive).

## `Contacts` — collision pipeline output

```rust,ignore
pub struct Contacts {
    pub count: u32,
    pub capacity: usize,
    pub a_index: Buffer<u32>,        // first body / shape index
    pub b_index: Buffer<u32>,
    pub point:   Buffer<Vec3>,       // world-space
    pub normal:  Buffer<Vec3>,       // a → b
    pub depth:   Buffer<Real>,
    pub friction:           Buffer<Real>,
    pub rolling_friction:   Buffer<Real>,
    pub torsional_friction: Buffer<Real>,
    pub restitution:        Buffer<Real>,

    // CSR per-body indexing (built after each broad+narrow pass)
    pub body_contact_offsets: Buffer<u32>,    // length body_count + 1
    pub body_contact_indices: Buffer<u32>,    // length 2·count
}
```

`count` is the number of valid entries; the per-contact arrays are
sized to `capacity` and only `0..count` is meaningful.

The CSR `body_contact_offsets` + `body_contact_indices` is the
per-body index built so the XPBD GPU kernel can dispatch one thread
per body, processing only its own contacts (no cross-body write
hazards). Every contact shows up under both endpoints, with the
**sign bit of the index** marking which side the body is on:

- top bit clear → body is the `a` side
- top bit set (`0x8000_0000`) → body is the `b` side, real index =
  `value & 0x7fff_ffff`

Decode via `Contacts::decode_index(packed)` → `(idx, is_b_side)`.
Static-world peers (`STATIC_BODY`) are deliberately excluded — they
have no body slot to update. This per-body Jacobi pattern is detailed
in [XPBD](../solvers/xpbd.md#per-body-jacobi-projection).

Pair friction / restitution follow Newton's `max(μ_a, μ_b)` rule:
the collision pipeline writes the per-contact value using
`shape_*_friction` / `shape_restitution` with the `NaN` sentinel
falling back to `Model.default_friction` / `Model.default_restitution`.

## Buffer ownership recap

| field | owner | mutated by |
|-------|-------|------------|
| `Model.*` | built once | nothing (immutable at sim time) |
| `State.*` | per-sim | solver |
| `Control.*` | per-step | user code (between steps) |
| `Contacts.*` | per-step | collision pipeline (`pipeline.collide`) |

All four upload to GPU via their respective `upload_to_device(device,
queue)` helpers. The model uploads once before the first step; the
state uploads on initial sim startup and is read back via
`download_from_device` for the viewer / sensors; the control + contacts
re-upload every step (the contact set turns over completely).
