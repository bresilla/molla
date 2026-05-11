# Ball / D6 / Distance

The three multi-DoF joint types are full reduced-coordinate citizens
of [Featherstone ABA](./featherstone.md): the same forward dynamics
that drives a chain of revolutes handles a 3-DoF ball or a 6-DoF D6,
with the motion subspace `S` growing to 3 or 6 columns and the per-
joint `D = Sᵀ I^A S` matrix becoming 3×3 or 6×6.

Stage 17 introduced multi-DoF Featherstone; Stage 17.1 added the
rigorous dense Cholesky `D`-solve for non-diagonal inertia (Free joints
with COM offsets) and the velocity-level Lagrange multiplier for the
distance constraint. Both are toggleable per-solver:

```rust,ignore
let solver = FeatherstoneSolver::new_cpu()
    .with_rigorous_d_solve(true)         // dense Cholesky on D
    .with_rigorous_distance(true);       // multiplier (vs hard clamp)
```

## Ball joint

```rust,ignore
JointType::Ball = 2   // (4 coords, 3 dofs)
b.add_joint_ball(parent, child, x_p, x_c) -> Result<JointId>;
```

3-DoF spherical joint — pure rotation, **no translation**. Coords are
a unit quaternion `(qx, qy, qz, qw)`; DoFs are body-frame angular
velocity `(ω_x, ω_y, ω_z)`. Joint-local transform:

```text
J(q) = R(quat(qx, qy, qz, qw))
```

Motion subspace (3 columns):

```text
        ┌ 0  0  0 ┐                ex, ey, ez = the world-axis frame
S_ang = │ ex ey ez│                  expressed in joint-world (one
        └         ┘                  column per angular DoF)
```

Featherstone integrates the quaternion via the same small-rotation
update as Free joints. Per-step quaternion re-normalisation keeps
drift bounded.

**Limits**: per-DoF lower / upper bounds in `joint_limit_lower / upper`
mean **per-axis angle limits** in the joint-local frame, applied as
projective clamps after the integration step. Set them via
`set_joint_limits(joint, [lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z])`.

**Use case**: humanoid shoulders, hips, wrists; closed-chain robotic
fingers. The Stage-17 humanoid demo uses Ball joints throughout the
spine and limbs and a single `Free` joint at the pelvis.

```rust,ignore
let humerus  = b.add_body(...);
let shoulder = b.add_joint_ball(torso, humerus, x_p, x_c)?;
b.set_joint_limits(shoulder,
    &[-PI/2.0, -PI/3.0, -PI/4.0],
    &[ PI/2.0,  PI/3.0,  PI/4.0])?;
```

## Distance joint

```rust,ignore
JointType::Distance = 5   // (7 coords, 6 dofs)
b.add_joint_distance(parent, child, x_p, x_c) -> Result<JointId>;
```

A 6-DoF "soft" pin between two anchor points with a `[min_distance,
max_distance]` constraint. Storage matches `Free` (7 coords, 6 dofs)
so the kinematics is identical; the **limit** is what makes it a
distance joint.

Limits live in the first per-DoF slot pair:

```rust,ignore
b.set_joint_limits(j, &[min_dist, ...], &[max_dist, ...])?;
//                     │                  │
//                     └──────────────────┴── first slot = anchor distance
```

The other 5 slots are normally unlimited (`NaN`).

The Stage-17.1 polish replaced the original "hard-clamp" projection
(`p ← p · L_target / ‖p‖` plus radial-velocity zeroing) with a
**velocity-level Lagrange multiplier** that satisfies the constraint
exactly to within solver tolerance. The hard clamp remains as a
fallback when the multiplier is numerically degenerate (collinear
anchors). The behaviour is toggleable via `with_rigorous_distance(bool)`
for A/B parity testing.

**Use case**: rope-like rigid links (a ball on a string with a
rigid-body ball), suspension bridge segments. For elastic ropes use
[CABLE joints](./joints_cable.md) or distance constraints between
particles.

## D6 joint

```rust,ignore
JointType::D6 = 6   // (7 coords, 6 dofs)
b.add_joint_d6(parent, child, x_p, x_c) -> Result<JointId>;
```

The "Swiss army knife" 6-DoF joint with **per-axis lock / limit**.
Coord layout matches `Free` / `Distance` (`[tx, ty, tz, qx, qy, qz, qw]`).
A locked DoF is encoded by setting `lower == upper == 0` in the
per-DoF limit arrays — no separate "lock" flag is needed.

```text
                  lower  upper       state
                  ─────  ─────       ─────
locked       :     0      0          DoF disabled (S column = 0)
unlimited    :    NaN    NaN         free motion
soft-limited :   -π/4    π/4         clamp + spring restoring force
```

The motion-subspace builder reads the limit pair and produces an
effective DoF count `≤ 6`. Locked DoFs are not integrated and don't
contribute to the per-joint `D` matrix.

D6 covers every joint pattern you'd otherwise need a fixed library
for:

|         | locked axes | unlocked axes | equivalent |
|---------|-------------|---------------|------------|
| Revolute (Z) | tx, ty, tz, ωx, ωy | ωz | `add_joint_revolute` |
| Prismatic (X) | ty, tz, ωx, ωy, ωz | tx | `add_joint_prismatic` |
| Universal | tx, ty, tz, ωz | ωx, ωy | (no dedicated type) |
| Cylindrical | tx, ty, ωy, ωz | tz, ωx | (no dedicated type) |
| Planar | tz, ωx, ωy | tx, ty, ωz | (no dedicated type) |

The Stage-17 multi-DoF Featherstone runs the same Cholesky on the
effective `D` matrix regardless of how many axes are locked. The
dedicated Revolute / Prismatic types remain in the API for clarity
(and because URDF/MJCF reference them directly), but mechanically D6
covers them.

## Reduced-coordinate dynamics — what changed at Stage 17

In Stage 17 the Featherstone backward pass was generalised from a
1-DoF rank-1 deflation to a **per-DoF column rank-1 deflation**:

```text
For each DoF column S_k of the joint:
    U_k  = I^A · S_k
    D_k  = S_kᵀ · U_k                  (scalar)
    I^A := I^A − U_k U_kᵀ / D_k       (deflation)
```

After processing all DoFs of a joint, `I^A` is propagated to the
parent. The forward pass then solves per-DoF
`q̈_k = (u_k − U_kᵀ · a_p) / D_k`.

This sequential rank-1 path is exact when `D = Sᵀ I^A S` is **diagonal**,
which is the case for Ball, Revolute, decoupled Free (no COM offset),
and D6 with axis-aligned locks. For asymmetric Free joints — where
the COM offset couples linear and angular DoFs — the off-diagonal
entries of `D` matter; the Stage 17.1 `with_rigorous_d_solve(true)`
path builds the full `D` matrix and solves `D · q̈ = u_τ` via dense
Cholesky.

The two paths agree to roundoff on diagonal-`D` joints (Ball,
Revolute, Prismatic, Fixed, Distance with aligned anchors); they
differ only when `D` is non-diagonal.
