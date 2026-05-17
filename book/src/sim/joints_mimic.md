# Joint mimic

A mimic constraint couples two joint coordinates so the **driven joint**
follows a function of the **driver joint**. molla ships two flavours:

- **Linear mimic** — `q_a = c0 + c1·q_b`. Stored on `Model.mimic_*`.
  Simple two-parameter API for the common case (gripper finger pair,
  geared joint, scissor mechanism).
- **Polynomial mimic** (Stage 32) — `q_a = poly[0] + poly[1]·q_b +
  poly[2]·q_b² + … + poly[4]·q_b⁴`. Encoded as `EqualityType::JointMimic`.
  Covers Newton's `JOINT` equality type and the closed-loop kinematics
  it enables.

Both implementations run as **post-step projective updates**: after the
Featherstone (or XPBD-rigid) integrator finishes a step, the mimic
pass overwrites `joint_q[driven]` to satisfy the coupling. The driver's
velocity propagates by a Jacobian product the next step.

## Linear mimic API

```rust,ignore
pub fn add_mimic_constraint(
    &mut self,
    joint_a: JointId,   // the driven joint (q_a is overwritten)
    joint_b: JointId,   // the driver joint (q_b is read)
    c0: Real,           // constant offset
    c1: Real,           // linear coefficient
    world: WorldId,
) -> usize;
```

Storage on `Model`:

```rust,ignore
pub mimic_count: usize,
pub mimic_joint_a: Buffer<i32>,
pub mimic_joint_b: Buffer<i32>,
pub mimic_c0:      Buffer<Real>,
pub mimic_c1:      Buffer<Real>,
pub mimic_enabled: Buffer<u32>,
pub mimic_constraint_world: Buffer<i32>,
```

**Use cases**:

- **Gripper finger pair** (Robotiq 2-finger): one motor drives both
  fingers symmetrically. `q_finger_b = q_finger_a` → `c0 = 0, c1 = 1`.
- **Antagonist coupling**: `q_b = −q_a` for opposing axes.
- **Constant offset stage**: `q_b = q_a + π/4` for a fixed-phase
  scissor cut.

## Polynomial joint-mimic (Stage 32)

The quartic mimic is the polynomial extension that closes **kinematic
loops** — the canonical example being a 4-bar linkage, where the
"return" link's angle is a non-linear function of the input link's
angle.

```rust,ignore
pub fn add_equality_joint_mimic(
    &mut self,
    driver: JointId,
    driven: JointId,
    poly: [Real; 5],    // [a0, a1, a2, a3, a4]
    world: WorldId,
) -> usize;
```

The polynomial:

```text
q_driven = poly[0] + poly[1]·q_driver + poly[2]·q_driver²
                   + poly[3]·q_driver³ + poly[4]·q_driver⁴
```

Storage piggy-backs on the equality-constraint arrays via the
`EqualityType::JointMimic` variant:

```rust,ignore
pub equality_count: usize,
pub equality_type: Buffer<u32>,                         // == 2 (JointMimic)
pub equality_mimic_driver_joint: Buffer<i32>,           // (-1 if not mimic)
pub equality_mimic_driven_joint: Buffer<i32>,
pub equality_mimic_poly:        Buffer<Real>,           // length 5 · equality_count
```

The `equality_type` slot discriminates which entries are "real"
positional equalities vs joint-mimic entries; the post-step pass
reads `equality_mimic_*` only when `equality_type == JointMimic`.

### Why 4th-order?

Newton's `JOINT` equality defines mimic via a polynomial up to `q⁴`.
molla keeps the same native polynomial form because it is compact,
GPU-friendly, and covers common gear/linkage relationships. Anything beyond
quartic typically signals a kinematic problem better expressed as two
independent joints + a position equality.

### Examples

| Use case | poly coefficients |
|----------|-------------------|
| Linear mimic with bias `c0 + c1·q` | `[c0, c1, 0, 0, 0]` |
| Pure linear `q_b = q_a` | `[0, 1, 0, 0, 0]` |
| Negated `q_b = -q_a + π` | `[π, -1, 0, 0, 0]` |
| 4-bar linkage (quadratic approx) | `[0, 1.0, -0.05, 0, 0]` |
| Gear ratio 2:1 | `[0, 2.0, 0, 0, 0]` |

For the 4-bar specifically, the exact transmission is transcendental;
the quartic least-squares fit over the working range is usually
accurate to <1 % positional error and Featherstone's stiff regime
absorbs the residual.

### Stage 32 algorithm

The mimic projection runs **after** the Featherstone (or XPBD-rigid)
substep:

```text
for each mimic equality e:
    if not equality_enabled[e]: continue
    if equality_type[e] != JointMimic: continue
    j_driver = equality_mimic_driver_joint[e]
    j_driven = equality_mimic_driven_joint[e]
    q_drv = joint_q[coord_offset[j_driver]]
    q_target = poly[0] + poly[1]·q_drv + poly[2]·q_drv²
             + poly[3]·q_drv³ + poly[4]·q_drv⁴
    joint_q[coord_offset[j_driven]] = q_target
    # Velocity is the polynomial derivative · q̇_driver
    dpoly = poly[1] + 2·poly[2]·q_drv + 3·poly[3]·q_drv²
          + 4·poly[4]·q_drv³
    joint_qd[dof_offset[j_driven]] = dpoly · joint_qd[dof_offset[j_driver]]
```

The velocity propagation matches the chain rule so kinetic energy
inside the kinematic loop is conserved to first order.

## When to choose mimic vs gear / equality

|                       | Linear mimic | Polynomial mimic | Gear (effort) | Equality (CONNECT) |
|-----------------------|:------------:|:----------------:|:-------------:|:------------------:|
| Closed kinematic loop | ✓            | ✓                |              | ✓ (positional)     |
| Non-linear relation   |              | ✓                |               |                    |
| Native serialization  | ✓            | ✓                |               | ✓                  |
| Hard constraint       | ✓ (projection) | ✓ (projection) |               | XPBD-soft          |
| Driver chooses force  |              |                  | ✓             |                    |

For a 4-bar linkage, `add_equality_joint_mimic` is the right answer.
For a Robotiq 2-finger gripper, `add_mimic_constraint` is enough.
For an under-actuated tendon-driven hand, use a [CABLE joint](./joints_cable.md)
or model the tendon explicitly as particles.
