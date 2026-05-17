# Articulation (Featherstone ABA)

[`FeatherstoneSolver`](https://docs.rs/molla/latest/molla/solvers/struct.FeatherstoneSolver.html)
is molla's full **Articulated Body Algorithm** (ABA) per Featherstone
(2008) ch. 7 — the reduced-coordinate forward dynamics solver of
choice for serial chains and trees of revolute / prismatic / ball /
free / D6 / distance / cable joints.

It is a direct port of the structure in Newton's
`solvers/featherstone/kernels.py`, with spatial-vector primitives
borrowed from Warp's `_src/math.py` / `builtins.py`. The CPU path is
the reference oracle; the GPU path (WGSL kernel
[`featherstone.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/featherstone.wgsl))
implements the identical three-pass algorithm one workgroup per
articulation.

```rust,ignore
use molla::solvers::{FeatherstoneSolver, Solver};

let mut solver = FeatherstoneSolver::new_cpu()        // or new_gpu(engine, &model)
    .with_substeps(4)
    .with_damping(0.05)
    .with_rigorous_d_solve(true)                       // dense Chol on D
    .with_rigorous_distance(true);                     // multiplier on Distance

solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out)?;
```

## The three passes

Each substep walks the articulation three times. Notation: `S_ik` =
the `k`-th column of joint `i`'s motion subspace (in world frame),
`v_i`, `a_i` = body spatial velocity / acceleration, `I^A_i` =
articulated body inertia, `p^A_i` = articulated bias force, `c_i` =
Coriolis term, `f_grav_i` = gravity bias.

### Pass 1 — forward kinematics + body kinematics (root → leaf)

For each joint in declaration order:

```text
1. Compute child world pose from parent + joint's local transform.
2. Build the world-frame motion subspace S_i (1, 2, 3, or 6 columns).
3. Body spatial velocity:        v_i = v_p + Σ_k S_ik · q̇_ik
4. Coriolis term:                c_i = v_i × (Σ_k S_ik · q̇_ik)
5. Body spatial inertia (world): I_i (transform body-frame Mat3 + mass)
6. Bias force:                   p^A_i = v_i ×* (I_i · v_i) − f_grav_i
```

The `×` and `×*` are Featherstone's spatial cross-product operators
(see [Spatial algebra](../math/spatial.md)).

### Pass 2 — articulated inertia accumulation (leaf → root)

For each joint, processed in reverse declaration order:

```text
For each DoF column S_k of joint i:
    U_k  = I^A_i · S_k
    D_k  = S_kᵀ · U_k
    u_k  = τ_drive_k + τ_ff_k + τ_friction_k + τ_cable_k − d · q̇_k
                                                          [− muscle_k]
    u_k -= S_kᵀ · p^A_i

    # Sequential rank-1 deflation (default path):
    I^A_i := I^A_i − U_k U_kᵀ / D_k
    p^A_i := p^A_i + U_k · u_k / D_k

# Propagate to parent in parent frame:
I^A_p += transform(I^A_i)
p^A_p += transform(p^A_i)
```

The **sequential rank-1 deflation** is exact when the per-joint matrix
`D = Sᵀ I^A S` is diagonal, which covers Revolute, Prismatic, Ball,
decoupled Free (no COM offset), Distance, and D6 with axis-aligned
locks. For asymmetric Free joints where the COM offset cross-couples
linear and angular DoFs the off-diagonal entries of `D` matter — the
Stage 17.1 polish path (`with_rigorous_d_solve(true)`) builds the
full `D` matrix and `D · q̈ = u_τ` is solved via dense Cholesky.

### Pass 3 — forward acceleration solve (root → leaf)

```text
For each joint i:
    For each DoF column S_k:
        q̈_k = (u_k − U_kᵀ · a_p) / D_k
    a_i = a_p + Σ_k S_k · q̈_k + c_i
```

### Pass 4 — symplectic integration

```text
q̇_new = q̇_old + dt · q̈
q_new  = q_old + dt · q̇_new        (or quaternion update for Ball/Free)
```

Symplectic (not just plain forward) Euler — the velocity is updated
**before** the position, so energy drift is bounded for a wide range
of `dt`.

### Pass 5 — refresh body poses

After integration, a final FK pass writes `state_out.body_q` from the
new `joint_q` so downstream consumers (sensors, contact pipeline,
viewer) see consistent body world transforms.

## Multi-DoF joint support (Stage 17)

```text
joint type   coords  dofs    motion subspace S
─────────── ──────  ────    ───────────────────────────────
Revolute      1      1       (0; axis_w)
Prismatic     1      1       (axis_w; 0)
Fixed         0      0       — (skipped)
Ball          4      3       (0; ex_w, ey_w, ez_w)
Free          7      6       (I_3, 0; 0, I_3)
Distance      7      6       same as Free (limit applied post-step)
D6            7      6       same as Free, locked columns zeroed
Cable         2      2       (z_w; 0), (0; x_w)
```

The Stage 17 work generalised the backward pass from a 1-DoF rank-1
update to a per-DoF column rank-1 deflation — the same algorithm
sequence, just looped over `dof_count(joint)` columns. Stage 17.1
added the rigorous dense-`D` solve for the asymmetric Free case
(`with_rigorous_d_solve`) and the velocity-level Lagrange multiplier
for the Distance constraint (`with_rigorous_distance`).

## Joint drives, limits, friction

Each per-DoF `u_k` (the right-hand side of the backward pass) is the
sum of:

```text
u_k = τ_drive_k + τ_ff_k + τ_friction_k + τ_cable_k + τ_muscle_k
                                                  − d · q̇_k
```

- `τ_drive_k` — from `Control.joint_target_mode`:
  - `Position`        → `kp·(q_target − q) + kd·(0 − qd)`
  - `Velocity`        → `kd·(qd_target − qd)`
  - `PositionVelocity` → `kp·(q_target − q) + kd·(qd_target − qd)`
  - `Effort`          → `q_target`  (interpreted as direct torque)
- `τ_ff_k` — `Control.joint_force[k]` (feed-forward).
- `τ_friction_k = −Model.joint_friction[k] · q̇_k` — Coulomb damping.
- `τ_cable_k = −k_l q_l − c_l q̇_l` / `−k_b q_b − c_b q̇_b` for cable joints.
- `τ_muscle_k` — Hill-type muscle force on a CABLE linear DoF
  (Stage 42; pre-computed by the muscle kernel into a per-DoF accumulator).
- `d · q̇_k` — uniform `solver.joint_damping`.

After `u_k` is assembled, **effort clamping** applies:

```text
u_k := clamp(u_k, −effort_limit_k, +effort_limit_k)    (NaN limit = unlimited)
```

After integration, **velocity clamping** applies:

```text
qd_k := clamp(qd_k, −velocity_limit_k, +velocity_limit_k)
```

Joint **position limits** are not enforced inside Featherstone; the
caller is expected to either:

1. Use [XPBD](../solvers/xpbd.md) for the rigid-body solve and let it
   project joint limits, or
2. Use the post-step `joint_limit_lower / upper` clamp pass in the
   viewer / wrapper layer, or
3. Rely on a high `joint_target_ke` joint-drive stiffness to act as
   a soft limit barrier.

## Substepping

```rust,ignore
let solver = FeatherstoneSolver::new_cpu().with_substeps(4);
```

Each outer `step(dt)` advances `n` inner substeps of `dt/n`, with the
contact set (rare in Featherstone — typically empty) reused. The
implementation ping-pongs `state_out` with a host-side scratch
`state` to avoid intermediate allocations.

For most articulation workloads `substeps = 1` is fine — Featherstone
is **second-order accurate** and stable at large `dt`. Increase
substepping when:

- A joint hits its effort limit and the limit clamp introduces
  high-frequency content.
- A Distance-joint constraint produces stiff oscillation at the
  multiplier-update frequency.
- Coupling with an XPBD contact pass that runs at higher rate.

## GPU port

The WGSL kernel `featherstone.wgsl` implements all three passes inline
per workgroup (one workgroup = one articulation). Spatial-vector and
spatial-matrix operations live as inline helpers using `vec4<f32>` and
`mat3x3<f32>`. The kernel reads:

- `Model.joint_*`, `Model.body_*`, `Model.gravity`, `Model.articulation_start`
- `State.joint_q`, `State.joint_qd`, `State.body_q`, `State.body_qd`
- `Control.joint_target_mode/pos/vel/kp/kd/force`, `Control.body_external_f`
- Optional muscle torque buffer (Stage 42 GPU port — see
  [muscle](../actuators/muscle.md))

and writes the next-step `joint_q`, `joint_qd`, `body_q`, `body_qd`
into `state_out`. The CPU and GPU paths agree to roundoff on every
joint type — verified by the parity tests in
`tests/featherstone_gpu_parity*.rs`.

The Stage 42 v3 GPU path also runs a **muscle kernel pre-pass**
(`MuscleKernel::dispatch_torque_sum`) that fills a per-DoF muscle
torque buffer from current activations; Featherstone reads from it in
pass 2.

## Differentiable Featherstone

Stage 23.1 added a **forward-mode tape recorder** that captures each
step's `(q, q̇, τ)` inputs:

```rust,ignore
solver.step_with_tape(&model, &state_in, &control, &contacts, dt, &mut state_out,
    &mut tape, &chain, q_id, qd_id, tau_id)?;
```

The reverse pass walks the tape backwards via
`FeatherstoneRevoluteChain::reverse` to compute `dJ/dq`, `dJ/dq̇`,
`dJ/dτ`. v1 supports serial revolute chains only — multi-DoF joints
are gated behind the differentiability follow-up. See
[Featherstone adjoint](../autograd/featherstone.md) for the full
adjoint chapter.

## Forward kinematics shortcut

When you only need world poses (not forward dynamics) — IK targets,
contact-pipeline setup, sensor placement — call `crate::sim::eval_fk`
directly:

```rust,ignore
use molla::sim::eval_fk;
eval_fk(&model, &mut state)?;       // writes body_q from joint_q
```

`eval_fk` is the pose-only kinematics subset of Featherstone's pass 1
— no velocities, no inertia, no dynamics. Multi-branch trees work as
long as joints are listed in topological order (parent-before-child),
which is what `ModelBuilder` produces by construction.
