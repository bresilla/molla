# ModelBuilder

`ModelBuilder` is the **ergonomic authoring surface** for [`Model`](./model.md).
It accumulates bodies, shapes, joints, particles, springs, cables,
tetrahedra, equality constraints, mimics, and muscles in host-side
`Vec<T>` buffers, then `build()` consumes the builder and emits a
fully-baked `Model` with all CSR offsets, graph colourings, and BVHs
computed.

```rust,ignore
use molla::sim::{ModelBuilder, BodyParams};

let mut b = ModelBuilder::new();
let body = b.add_body(BodyParams::new().with_position([0.0, 1.0, 0.0].into()));
b.add_shape_sphere(body, Default::default(), 0.5)?;
let model = b.build()?;
let mut state = model.state()?;
```

## Bodies

```rust,ignore
pub fn add_body(&mut self, params: BodyParams) -> BodyId;
```

[`BodyParams`](https://docs.rs/molla/latest/molla/sim/struct.BodyParams.html)
is a builder-style struct:

```rust,ignore
pub struct BodyParams {
    pub mass: Real,           // kg; 0 = static / kinematic
    pub inertia: Mat3,        // at COM, body frame (kg·m²)
    pub com: Vec3,            // COM in body frame
    pub pose: Transform,      // initial world pose
    pub flags: u32,           // BodyFlags::DYNAMIC | KINEMATIC
    pub world: WorldId,
}

impl BodyParams {
    pub fn new() -> Self;                        // 1 kg cube-ish defaults
    pub fn with_mass(self, mass: Real) -> Self;
    pub fn with_inertia(self, inertia: Mat3) -> Self;
    pub fn with_spatial_inertia(self, si: SpatialInertia) -> Self;
    pub fn with_com(self, com: Vec3) -> Self;
    pub fn with_pose(self, pose: Transform) -> Self;
    pub fn with_position(self, p: Vec3) -> Self;
    pub fn with_world(self, world: WorldId) -> Self;
    pub fn with_flags(self, flags: u32) -> Self;
}
```

`with_spatial_inertia` is the recipe for shape-derived inertia: any
`ShapeGeometry` computes a [`SpatialInertia`](../math/types.md) given
density via `crate::math::inertia::*`.

## Shapes

Each `add_shape_*` returns a `ShapeId` and is bound to a body (or
`BodyId::NONE` for static-world shapes). The signatures share a
common shape transform + per-shape material setup pattern.

```rust,ignore
pub fn add_shape_sphere(&mut self, body: BodyId, x_b: Transform, radius: Real) -> Result<ShapeId>;
pub fn add_shape_box(&mut self, body, x_b, half_extents: Vec3) -> Result<ShapeId>;
pub fn add_shape_capsule(&mut self, body, x_b, half_height, radius) -> Result<ShapeId>;
pub fn add_shape_cylinder(&mut self, body, x_b, half_height, radius) -> Result<ShapeId>;
pub fn add_shape_cone(&mut self, body, x_b, half_height, radius) -> Result<ShapeId>;
pub fn add_shape_ellipsoid(&mut self, body, x_b, half_axes: Vec3) -> Result<ShapeId>;
pub fn add_shape_plane(&mut self, body, x_b, normal: Vec3) -> Result<ShapeId>;
pub fn add_shape_triangle(&mut self, body, x_b, v0, v1, v2: Vec3) -> Result<ShapeId>;

pub fn add_mesh(&mut self, mesh: TriMesh) -> u32;               // returns mesh_id
pub fn add_shape_trimesh(&mut self, body, x_b, mesh_id: u32) -> Result<ShapeId>;
pub fn add_hull(&mut self, vertices: Vec<Vec3>) -> u32;
pub fn add_shape_convex_hull(&mut self, body, x_b, hull_id: u32) -> Result<ShapeId>;
pub fn add_heightfield(&mut self, hf: Heightfield) -> u32;
pub fn add_shape_heightfield(&mut self, body, x_b, field_id: u32) -> Result<ShapeId>;
pub fn add_sdf(&mut self, sdf: SdfShape) -> u32;
pub fn add_shape_sdf(&mut self, body, x_b, sdf_id: u32) -> Result<ShapeId>;
```

### ShapeGeometry

The full geometry tag enum (host-side, mirrored at build time into the
GPU-friendly `shape_geom_kind` / `shape_geom_data` parallel arrays):

```rust,ignore
pub enum ShapeGeometry {
    Sphere      { radius: Real },
    Plane       { normal: Vec3 },                          // half-space
    Box         { half_extents: Vec3 },
    Capsule     { half_height: Real, radius: Real },       // local +Y axis
    Cylinder    { half_height: Real, radius: Real },
    Cone        { half_height: Real, radius: Real },       // apex at +Y
    Ellipsoid   { half_axes: Vec3 },
    Triangle    { v0: Vec3, v1: Vec3, v2: Vec3 },
    TriMesh     { mesh_id: u32 },
    ConvexHull  { hull_id: u32 },
    Heightfield { field_id: u32 },
    Sdf         { sdf_id: u32 },
}
```

Per-shape material:

```rust,ignore
pub fn set_shape_material(
    &mut self, shape: ShapeId,
    friction: Real, restitution: Real,
) -> Result<()>;

pub fn set_shape_friction_modes(
    &mut self, shape: ShapeId,
    rolling: Real, torsional: Real,
) -> Result<()>;

pub fn set_shape_visual_material(
    &mut self, shape: ShapeId,
    color: Vec3, metallic: Real, roughness: Real,
);

pub fn set_shape_collision_group(&mut self, shape: ShapeId, group: u32);
pub fn add_shape_collision_exclusion(&mut self, a: ShapeId, b: ShapeId);
```

Friction / restitution sentinels: `NaN` means "fall back to the model
default". Collision groups are 32-bit bitmasks (`a.group & b.group != 0`
to interact); explicit exclusions override.

## Joints

See [Joints](./joints.md) for the DoF accounting and the per-joint-type
chapters ([basic](./joints_basic.md), [multi-DoF](./joints_multi_dof.md),
[cable](./joints_cable.md), [mimic](./joints_mimic.md)) for the
semantics.

```rust,ignore
pub fn begin_articulation(&mut self);    // close current chain, start new

pub fn add_joint_revolute(&mut self, parent: BodyId, child: BodyId,
    x_p: Transform, x_c: Transform, axis: Vec3) -> Result<JointId>;
pub fn add_joint_prismatic(&mut self, parent, child, x_p, x_c, axis) -> Result<JointId>;
pub fn add_joint_fixed(&mut self, parent, child, x_p, x_c) -> Result<JointId>;
pub fn add_joint_ball(&mut self, parent, child, x_p, x_c) -> Result<JointId>;
pub fn add_joint_free(&mut self, parent, child, x_p, x_c) -> Result<JointId>;
pub fn add_joint_distance(&mut self, parent, child, x_p, x_c) -> Result<JointId>;
pub fn add_joint_d6(&mut self, parent, child, x_p, x_c) -> Result<JointId>;
pub fn add_joint_cable(&mut self, parent, child, x_p, x_c,
    stiffness_linear, stiffness_bend,
    damping_linear,   damping_bend) -> Result<JointId>;

// Per-DOF settings
pub fn set_joint_limits(&mut self, joint: JointId,
    lower: &[Real], upper: &[Real]) -> Result<()>;
pub fn set_joint_friction(&mut self, joint, friction: &[Real]) -> Result<()>;
pub fn set_joint_effort_limit(&mut self, joint, effort: &[Real]) -> Result<()>;
pub fn set_joint_velocity_limit(&mut self, joint, vel: &[Real]) -> Result<()>;
pub fn set_joint_target_ke(&mut self, joint, ke: &[Real]) -> Result<()>;
pub fn set_joint_enabled(&mut self, joint, enabled: bool) -> Result<()>;
```

`begin_articulation` is what splits joints into Featherstone's "one
workgroup per articulation" partitioning. The first `add_joint_*` opens
articulation 0; further joints append until `begin_articulation` closes
the chain.

## Particles, springs, cables

```rust,ignore
pub fn add_particle(&mut self, position: Vec3, mass: Real, radius: Real,
    world: WorldId) -> u32;
pub fn pin_particle(&mut self, i: u32);              // mass → 0 = kinematic

pub fn add_distance_constraint(&mut self, a: u32, b: u32,
    rest_length: Real, compliance: Real, max_stretch: Real) -> usize;

// CABLE polyline = chain of distance constraints + bookkeeping
pub fn add_cable(&mut self, particles: &[u32], stiffness: Real,
    max_stretch: Real) -> usize;
pub fn add_cable_polyline(&mut self, start: Vec3, end: Vec3, n: u32,
    particle_mass: Real, particle_radius: Real,
    stiffness: Real, max_stretch: Real,
    world: WorldId) -> (u32, u32, usize);
pub fn set_cable_bending_stiffness(&mut self, cable_id: usize, k_b: Real);
```

`compliance` is XPBD's `α = 1/stiffness` in `m/N`; `max_stretch` is a
hard cap on stretch ratio applied as a non-linear projection after the
soft XPBD pass.

## Cloth + tet meshes

```rust,ignore
pub fn add_cloth_grid(&mut self, ...) -> ClothGridResult;
pub fn add_cloth_grid_with_bending(&mut self, ...) -> ClothGridResult;
pub fn add_triangle(&mut self, v0: u32, v1: u32, v2: u32) -> usize;

pub fn add_tet_mesh(&mut self, particles: &[u32], tets: &[[u32; 4]],
    material: MaterialNeoHookean) -> Result<TetMeshId>;

pub fn add_tet_mesh_per_material(&mut self, particles, tets,
    materials: &[MaterialNeoHookean],
    material_idx: &[u32]) -> Result<TetMeshId>;

pub fn set_tet_mesh_self_collision(&mut self, id: TetMeshId, enable: bool);
pub fn set_tet_mesh_friction(&mut self, mesh: TetMeshId, mu: Real);
pub fn set_tet_mesh_restitution(&mut self, mesh: TetMeshId, e: Real);
```

`MaterialNeoHookean` carries Young's modulus, Poisson ratio, and the
plasticity knobs (yield stress, hardening, kinematic hardening — see
[VBD plasticity](../solvers/vbd.md#plasticity)).

## Equality, mimic, muscle

```rust,ignore
// Equality (CONNECT = 3-constraint pin, WELD = 6-constraint rigid weld)
pub fn add_equality_connect(&mut self, body_a, body_b, anchor_a, anchor_b,
    world: WorldId) -> usize;
pub fn add_equality_weld(&mut self, body_a, body_b, anchor_a, anchor_b,
    relative_rotation: Quat, world: WorldId) -> usize;

// Joint mimic via 4th-order polynomial (Stage 32)
pub fn add_equality_joint_mimic(&mut self, driver: JointId, driven: JointId,
    poly: [Real; 5], world: WorldId) -> usize;

// Linear mimic (q_a = c0 + c1·q_b) — simpler API
pub fn add_mimic_constraint(&mut self, joint_a: JointId, joint_b: JointId,
    c0: Real, c1: Real, world: WorldId) -> usize;

// Hill-type muscle attached to a CABLE joint's linear DoF (Stage 42)
pub fn add_muscle(&mut self, joint: JointId,
    config: actuators::MuscleConfig) -> Result<usize>;
```

## World + global knobs

```rust,ignore
pub fn set_world_gravity(&mut self, world: WorldId, gravity: Vec3);
pub fn set_angular_damping(&mut self, damping: Real);
pub fn set_linear_damping(&mut self, damping: Real);
pub fn set_default_friction(&mut self, friction: Real) -> Result<()>;
pub fn set_default_restitution(&mut self, restitution: Real) -> Result<()>;
```

## `build()` — finalise the model

```rust,ignore
pub fn build(self) -> Result<Model>;
```

`build` performs the heavy lifting that turns the host-side scratch
into a sim-ready, GPU-uploadable `Model`:

1. Closes the open articulation (`begin_articulation` semantics).
2. Computes per-body `inv_mass` and `inv_inertia` (with the
   kinematic-body sentinel `0`).
3. Builds the spring **graph colouring** — greedy by particle adjacency,
   producing the `spring_color_*` CSR consumed by the XPBD GPU kernel.
4. Builds the **VBD per-vertex colouring** + `vertex_adjacent_springs`
   CSR for VBD's Jacobi sweeps.
5. Builds per-vertex tet adjacency (`vertex_adjacent_tets`) +
   per-vertex cable-bending CSR.
6. Computes per-tet `Dm⁻¹` and `rest_volume` (pre-inverted at build
   so the VBD hot loop is branch-free).
7. Computes joint coord / DoF CSR offsets (`joint_coord_offset`,
   `joint_dof_offset`).
8. Registers soft surfaces (tet boundary tris + cloth tessellation
   tris) into the cross-mesh VT contact pipeline.
9. Builds per-trimesh BVHs (`TriMesh::build_bvh`).
10. Mirrors the host-side enum data into the GPU-friendly parallel
    arrays (`shape_geom_kind` / `shape_geom_data`).

After `build`, `Model::upload_to_device(device, queue)` is the **one
call** that mirrors all the buffers into wgpu storage, ready for
GPU-backed solvers and kernels.
