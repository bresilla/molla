# Joints

A joint constrains the relative motion between two rigid bodies. molla
inherits Newton's joint taxonomy (`xtra/newton/_src/sim/enums.py`) and
adds a single new entry — the Newton-style `CABLE` joint for stretchy
ropes / muscle tendons (Stage 39E).

## Joint-type taxonomy

| `JointType` | u32 | coords | DoFs | semantics |
|-------------|----:|-------:|-----:|-----------|
| `Prismatic` |   0 |      1 |    1 | translation along a fixed axis |
| `Revolute`  |   1 |      1 |    1 | rotation about a fixed axis |
| `Ball`      |   2 |      4 |    3 | spherical (3 angular DoF, quaternion coords) |
| `Fixed`     |   3 |      0 |    0 | locks all 6 DoF |
| `Free`      |   4 |      7 |    6 | unconstrained 6-DoF (xyz + quat) |
| `Distance`  |   5 |      7 |    6 | 6-DoF with anchor distance limit |
| `D6`        |   6 |      7 |    6 | 6-DoF with per-axis lock / limit |
| `Cable`     |   7 |      2 |    2 | linear stretch + isotropic bend |

`coord_count()` is the number of entries the joint writes into
`State.joint_q`; `dof_count()` is the matching length in `State.joint_qd`.
They differ for joints whose orientation is parameterised by a quaternion
(`Ball`, `Free`, `Distance`, `D6`) — `joint_qd` carries angular velocity
in 3 components, not 4.

```rust,ignore
pub enum JointType { Prismatic, Revolute, Ball, Fixed, Free, Distance, D6, Cable }

impl JointType {
    pub fn coord_dof(self) -> (usize, usize);  // (coord_count, dof_count)
    pub fn coord_count(self) -> usize;
    pub fn dof_count(self)   -> usize;
}
```

The full taxonomy is laid out across four chapters:

- [Revolute / Prismatic / Fixed / Free](./joints_basic.md)
- [Ball / D6 / Distance](./joints_multi_dof.md)
- [CABLE joint](./joints_cable.md)
- [Joint mimic](./joints_mimic.md)

## How DoFs map to `joint_q` / `joint_qd`

`Model` stores **two CSR offset arrays** that index into the flat
`State.joint_q` / `State.joint_qd` slabs:

```rust,ignore
pub joint_coord_offset: Buffer<u32>,    // length joint_count + 1
pub joint_dof_offset:   Buffer<u32>,    // length joint_count + 1
pub joint_coord_total:  usize;
pub joint_dof_total:    usize;
```

Joint `j` owns coords `joint_q[coord_offset[j] .. coord_offset[j+1]]`
and velocities `joint_qd[dof_offset[j] .. dof_offset[j+1]]`. The
totals are the last entries of the offset arrays.

For a model that mixes joint types — say `[Free, Revolute, Ball]` —
the layout is:

```text
joint   type        coords (size)        DoFs (size)
─────── ─────────── ──────────────────── ─────────────────
0       Free        [0..7]   (7)         [0..6]   (6)
1       Revolute    [7..8]   (1)         [6..7]   (1)
2       Ball        [8..12]  (4)         [7..10]  (3)
─────── ─────────── ──────────────────── ─────────────────
total                        12                   10
```

Solvers read `(coord_offset, dof_offset)` once per joint and use the
local sub-slice. The asymmetry (coords ≠ DoFs) only matters for the
integrator and for differentiable-physics tape recording.

## Targets and drives

Per-DoF drive policy lives in [`Control`](./model.md#control--per-step-user-inputs)
via the `JointTargetMode` enum:

```rust,ignore
pub enum JointTargetMode {
    None             = 0,    // passive — no drive torque
    Position         = 1,    // τ = kp·(q_target − q) + kd·(0 − qd)
    Velocity         = 2,    // τ = kd·(qd_target − qd)
    PositionVelocity = 3,    // PD around both
    Effort           = 4,    // τ = q_target  (direct torque)
}
```

The Featherstone backward pass [`featherstone.md`](./featherstone.md)
folds the drive torque into the per-DoF `u_k` along with `joint_force`
(feed-forward), `joint_friction` (Coulomb damping), and any muscle
contribution; the result is then clamped by `joint_effort_limit`.

## Limits, friction, effort, velocity

```rust,ignore
// On the Model, each is length joint_dof_total. NaN = unlimited.
pub joint_limit_lower:    Buffer<Real>,
pub joint_limit_upper:    Buffer<Real>,
pub joint_friction:       Buffer<Real>,   // Coulomb damping
pub joint_effort_limit:   Buffer<Real>,   // torque clamp
pub joint_velocity_limit: Buffer<Real>,   // post-step qd clamp
pub joint_target_ke:      Buffer<Real>,   // static joint-drive stiffness
```

D6 joints encode **per-axis locks** by setting `lower == upper == 0`
for the locked axes (no extra "locked" flag needed); the motion
subspace builder drops those DoFs and the integrator skips them.

## Articulations

A connected group of joints (one kinematic tree) is an **articulation**.
The builder opens an articulation at the first `add_joint_*` after
`begin_articulation()`; further joints append until the next
`begin_articulation()`. The CSR

```rust,ignore
pub articulation_count: usize,
pub articulation_start: Buffer<u32>,    // length articulation_count + 1
```

records joint-index ranges. The Featherstone solver dispatches one
workgroup per articulation, so two independent robots in the same
`Model` run in parallel on GPU.

## Equality constraints vs joints

A FIXED joint locks two bodies into one — it adds 0 DoFs to the
simulation. An `EqualityType::Weld` does the same thing but as a
post-step **projective constraint** rather than a structural part of
the kinematic tree. The two are interchangeable in steady-state but
differ during failure modes: a FIXED joint is rock-solid; a WELD has
some compliance and shows up as 6 algebraic constraints in the XPBD
equality solve.

Choose `Fixed` for permanent welds; choose `Weld` for closed loops
(impossible to express as a tree) and for runtime-toggleable rigid
links. See [Joints — mimic](./joints_mimic.md) for the polynomial
`JointMimic` equality that closes 4-bar linkages.
