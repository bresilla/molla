# Revolute / Prismatic / Fixed / Free

The four "primary" joint types are the workhorses of every articulated
scene. Together they cover ~95 % of robotics use cases.

## Revolute

```rust,ignore
JointType::Revolute = 1   // (1 coord, 1 dof)
b.add_joint_revolute(parent, child, x_p, x_c, axis: Vec3) -> Result<JointId>;
```

1-DoF rotation about a **fixed joint-local axis**. `axis` is in the
joint's local frame (`x_p`) and is normalised at build time — `Error::Build`
fires on a near-zero or non-finite axis.

Joint-local transform:

```text
J(q) = R(axis, q)   (pure rotation)
X_w_c = X_w_p · X_p · J(q) · X_c⁻¹
```

The motion subspace `S` is a single spatial-vector column:

```text
S = (0, 0, 0; axis_x, axis_y, axis_z)   — axis is the angular row
```

Featherstone reads `S` from `joint_axis`, transforms it into the
joint-world frame, and assembles the per-joint dynamics from there.

**Typical use**: hinges, motor joints, finger knuckles. The standard
"manipulator" robot is N stacked revolutes.

```rust,ignore
let mut b = ModelBuilder::new();
let link = b.add_body(BodyParams::new());
b.add_joint_revolute(
    BodyId::NONE, link,
    Transform::IDENTITY,                                       // x_p: world frame
    Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),     // x_c: top of link
    Vec3::Z,                                                   // rotation axis
)?;
```

At `q = 0`, the pendulum hangs along `-Y`. At `q = π/2`, the
right-hand rule curls `−Y → +X`, so the link points along `+X`.

## Prismatic

```rust,ignore
JointType::Prismatic = 0   // (1 coord, 1 dof)
b.add_joint_prismatic(parent, child, x_p, x_c, axis: Vec3) -> Result<JointId>;
```

1-DoF translation along a joint-local axis. Same axis-validation as
revolute. Joint-local transform:

```text
J(q) = T(axis · q)         (pure translation)
S    = (axis_x, axis_y, axis_z; 0, 0, 0)   — axis is the linear row
```

**Typical use**: linear actuators, gripper finger sliders, rail joints.

## Fixed

```rust,ignore
JointType::Fixed = 3   // (0 coords, 0 dofs)
b.add_joint_fixed(parent, child, x_p, x_c) -> Result<JointId>;
```

Locks all 6 relative DoFs. Writes `J(q) = Identity` and contributes
**no entries** to `joint_q` / `joint_qd`. The Featherstone solver
passes the joint through with zero acceleration.

Use Fixed when the child should rigidly track the parent and you want
the solver to **eliminate the DoF entirely** rather than enforce a
high-stiffness constraint each step. Two common patterns:

- **End-effector mount**: bolt a sensor / camera shape to a rigid
  parent. The mount doesn't move relative to the host, so a Fixed
  joint is faster than two bodies + a WELD equality.
- **Sub-body aggregation**: procedural builders that split one rigid assembly
  into multiple component bodies use Fixed joints to glue them back together
  while preserving the visual hierarchy.

## Free

```rust,ignore
JointType::Free = 4   // (7 coords, 6 dofs)
b.add_joint_free(parent, child, x_p, x_c) -> Result<JointId>;
```

Unconstrained 6-DoF "root joint" between a body and the world (or
between two bodies, if you want a loose floater attached to a parent).
Adds 7 coords `[tx, ty, tz, qx, qy, qz, qw]` and 6 DoFs `[v_x, v_y,
v_z, ω_x, ω_y, ω_z]`.

```text
J(q) = T(t) · R(quat)
S    = (I_3, 0; 0, I_3)     — full 6-DoF motion subspace
```

Free joints are how molla represents a floating-base robot (a
humanoid's pelvis, a quadrotor's body). The Featherstone ABA solves
the full 6×6 articulated inertia at the root. A floating-base scene
is just `Free` for the pelvis followed by Revolute / Ball joints for
each limb.

**Free vs `add_body` + Semi-Implicit**: a free-floating rigid body
that you'd otherwise integrate with [`SemiImplicitSolver`](../solvers/semi_implicit.md)
becomes a `Free`-rooted articulation when you want to add downstream
joints. The `Free` joint adds 6 DoFs to `joint_qd`; the body's
`body_qd` follows automatically. The two paths agree to roundoff for
single-body scenes.

**Quaternion housekeeping**: the 7th coord `qw` is the scalar; the
Featherstone integrator updates the quaternion via the small-rotation
recipe `q_new = normalize(q + 0.5·dt · (0, ω) · q)`, and re-normalises
each step.

## Comparison

|                         | Revolute | Prismatic | Fixed | Free |
|-------------------------|---------:|----------:|------:|-----:|
| coord count             |    1     |    1      |   0   |  7   |
| DoF count               |    1     |    1      |   0   |  6   |
| GPU support             |    ✓     |    ✓      |   ✓   |  ✓   |
| Joint drive (`Control`) |    ✓     |    ✓      |   —   |  ✓   |
| Limits via `set_joint_limits` | ✓ |    ✓      |   —   |  per-axis |
| Procedural builder      |    ✓     |    ✓      |   ✓   |  ✓ (root) |

See [Multi-DoF joints](./joints_multi_dof.md) for `Ball`, `Distance`, and
`D6`, and [Cable joint](./joints_cable.md) for the rope/tendon variant.
