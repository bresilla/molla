# URDF importer

molla parses URDF (Unified Robot Description Format) into a
`ModelBuilder` via the third-party [`urdf-rs`](https://crates.io/crates/urdf-rs)
crate. The importer lives at `crates/molla/src/utils/urdf.rs`. Two
entry points:

```rust
use molla::prelude::*;
use molla::utils::urdf::{load_urdf, parse_urdf, UrdfBodies};

// From file:
let mut b = ModelBuilder::new();
let names = load_urdf(Path::new("franka.urdf"), &mut b)?;

// From a string:
let names = parse_urdf(include_str!("franka.urdf"), &mut b)?;

let model = b.build()?;
```

`UrdfBodies` is a pair of `HashMap<String, BodyId>` /
`HashMap<String, JointId>` so the caller can look up bodies and
joints by their URDF name string after import.

## What is supported

### Links → bodies

Each `<link>` becomes one `BodyParams` with:

- **Mass** from `<inertial><mass>`. Cast `f64 → f64`.
- **Center of mass** from the position of `<inertial><origin>` —
  the `<rpy>` rotation is currently ignored (the molla
  `SpatialInertia` puts COM in the link's local frame as a
  position vector, not an oriented frame).
- **Inertia tensor** from `<inertial><inertia>` — the upper
  triangular `(ixx, ixy, ixz, iyy, iyz, izz)` mapped into a
  symmetric `Mat3`. URDF's inertia is given in the link's local
  frame, which is the same convention molla uses.

Visual and collision blocks are **parsed but ignored**. Wiring them
into the narrow-phase pipeline is a follow-up — for now, importers
typically add collision shapes via `ModelBuilder::add_shape_*` after
import for the bodies they care about. See the IK and Featherstone
viewer scenes for the pattern.

### Joints

| URDF joint type   | molla mapping                  |
|-------------------|--------------------------------|
| `revolute`        | `add_joint_revolute`           |
| `continuous`      | `add_joint_revolute` (no limits) |
| `prismatic`       | `add_joint_prismatic`          |
| `fixed`           | `add_joint_fixed`              |
| `floating`        | `add_joint_free`               |
| `planar`          | **error** — no molla equivalent |
| `spherical`       | **error** — use `ModelBuilder::add_joint_ball` directly |

`continuous` is mapped to `revolute` with no joint limits applied
(the URDF spec says continuous joints have no upper / lower
bound).

### Joint frames

- `joint/origin` becomes the joint's `x_p` (joint frame in
  parent body).
- The child anchor `x_c` is identity — URDF anchors the child
  link's origin at the joint origin by spec.
- `joint/axis` is consumed as the rotation / translation axis
  for revolute / prismatic joints. For ball / free / fixed it is
  ignored.
- RPY (roll-pitch-yaw) is converted to a quaternion via
  `glam::DQuat::from_euler(EulerRot::XYZ, r, p, y)`.

## What is not supported

- **Visual / collision geometry.** URDF visual + collision blocks
  are parsed (the upstream crate handles them) but molla doesn't
  wire them into shapes. Add shapes manually with
  `ModelBuilder::add_shape_box / sphere / capsule / cylinder / cone /
  hull / trimesh` after import.
- **Joint limits beyond hard upper / lower.** Effort and velocity
  limits from `<limit effort=... velocity=...>` are not yet
  consumed; URDF's `<safety_controller>` is parsed but ignored.
- **Transmissions / mimic joints from URDF.** URDF has a
  `<mimic joint="..." multiplier=... offset=.../>` syntax that
  molla supports natively via `EqualityType::JointMimic`, but the
  importer doesn't currently wire it. Add the mimic constraint
  manually with `ModelBuilder::add_equality_joint_mimic`.
- **Materials, sensors, gazebo extensions.** Out of scope.
- **Planar joints.** No molla equivalent — they don't fit the
  reduced-coordinate Featherstone model cleanly. The importer
  returns `Error::Build` if one is encountered.
- **Spherical (ball) joints via URDF.** Newton's URDF parser
  accepts `<joint type="spherical">` and routes it to a 3-DOF
  joint. molla supports ball joints natively (`add_joint_ball`)
  but does not yet route URDF spherical through. The importer
  returns a clear error directing the user to `add_joint_ball`.

## Worked example — Franka Panda

The `tests/ik_franka.rs` and `Scene::GpuIkArm` viewer demo load a
7-DOF Franka from URDF and add capsule colliders for narrow-phase
manually:

```rust
let mut b = ModelBuilder::new();
let names = load_urdf(Path::new("franka.urdf"), &mut b)?;

// Add capsule colliders for the visible links (not in URDF, added by hand).
for link_name in &["link1", "link2", "link3", "link4", "link5", "link6", "link7"] {
    let body = names.links[link_name];
    b.add_shape_capsule(body, 0.05, 0.20, Transform::IDENTITY)?;
}

let model = b.build()?;
```

## When to use URDF

URDF is fine for:

- ROS-native robot descriptions.
- Quick prototypes where you want the kinematic tree plus mass
  properties.
- Importing a model that already exists in the ROS / Gazebo
  ecosystem.

URDF is **not** fine for:

- Scenes with complex contact geometry (use MJCF or USD).
- Closed kinematic loops (URDF only supports trees; molla supports
  closures via `EqualityType::JointMimic` — see the `FourBar` scene).
- Cloth, particles, MPM, or any non-articulated content (URDF has
  no schema for it).

For non-articulated scenes the molla way is to build the `Model`
directly via `ModelBuilder` (see the `Pile`, `Cloth`, `Mpm`
scenes for examples), or to author in USD with `UsdPhysics`
opinions.

## See also

- [`importers/usd.md`](./usd.md) — USD physics, the long-term import
  path molla cares about.
- [`importers/mjcf.md`](./mjcf.md) — MJCF *writer* (not reader)
  for the MuJoCo wrapper.
- [`sim/builder.md`](../sim/builder.md) — `ModelBuilder` reference.
