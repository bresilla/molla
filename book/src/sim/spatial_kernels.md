# CRBA / RNEA / Jacobian

Three standalone spatial-algebra primitives complement
[Featherstone](./featherstone.md) for analytical control + planning
work. They share the same world-origin spatial-vector convention and
the same joint / articulation indexing, so the three compose
trivially.

| primitive | output | algorithm | CPU | GPU |
|-----------|--------|-----------|----:|----:|
| `crba::mass_matrix` | `M(q)`, `N×N` joint-space mass matrix | composite-rigid-body, Featherstone (2008) §6.3 | ✓ | Stage 16 |
| `rnea::inverse_dynamics` | `τ` for given `(q, q̇, q̈)` | recursive Newton-Euler, Featherstone §5 | ✓ | Stage 16 |
| `jacobian::jacobian` | `J(q)`, `6 × N` body-point Jacobian | per-DoF spatial column | ✓ | (CPU-only) |

## CRBA — joint-space mass matrix `M(q)`

```rust,ignore
pub fn mass_matrix(model: &Model, state: &State, articulation: usize)
    -> Result<Vec<Real>>;
```

Returns a flat `N²` row-major array (`M[i,j] = out[i*N + j]`) where
`N` is the articulation's total joint DoFs (counted via
`joint_dof_offset`). For the implicit "single articulation = all
joints" case, pass `articulation = 0` regardless of `Model.articulation_count`.

`M(q)` is symmetric positive definite and is the matrix that satisfies

```text
τ = M(q) · q̈ + C(q, q̇) + G(q)
```

where `C` is the Coriolis / centrifugal vector and `G` is gravity. The
two share a closed-form decomposition: `RNEA(q, q̇, q̈=0)` returns
`C + G`, so `M · q̈ = τ − RNEA(q, q̇, 0)` is the inverse-dynamics
recipe for forward dynamics — useful as a sanity check vs Featherstone.

### Algorithm

Three passes over the articulation:

1. **FK + per-body world-frame spatial inertia.** Same as Featherstone
   pass 1, minus the dynamics terms.
2. **Leaf-to-root composite inertia.** `I^c_i = I_i + Σ I^c_j` over
   child joints `j`.
3. **Per-joint column build.** For joint `i`:
   - `F = I^c_i · S_i`     (spatial force = composite inertia · subspace)
   - `M[i, i] = S_iᵀ · F`
   - Walk parent chain: `M[i, p] = M[p, i] = S_pᵀ · F`

Single-articulation; multi-DoF joints (Ball / Free / D6) are TODO for
CRBA in v1 — the per-DoF axis split mirrors RNEA's and lands together
with the multi-DoF RNEA port.

## RNEA — recursive Newton-Euler inverse dynamics

```rust,ignore
pub fn inverse_dynamics(model: &Model, state: &State,
    articulation: usize, qdd: &[Real]) -> Result<Vec<Real>>;
```

Given `(q, q̇, q̈)` returns the joint forces `τ` that would produce
the requested accelerations. Two passes per articulation:

1. **Forward** (root → leaf):

   ```text
   v_i = v_p + S_i · q̇_i
   a_i = a_p + S_i · q̈_i + v_i × (S_i · q̇_i)
   f_i = I_i · a_i + v_i ×* (I_i · v_i)
   ```

2. **Backward** (leaf → root):

   ```text
   τ_i = S_iᵀ · f_i
   f_p += f_i           (transferred via parent-child cross terms)
   ```

Gravity is folded into the base acceleration: `a_root = −g`, so every
gravity-affected body sees an effective acceleration of `−g`
relative to free-fall. Matches Featherstone §5 and Newton's
`eval_rigid_id` kernel.

Single-articulation, single-DoF joints (Revolute / Prismatic / Fixed)
in v1. Multi-DoF joints return `Err(Build)` so callers fall back to
Featherstone for forward dynamics or do a DoF expansion themselves.

## Jacobian — `6 × N` body-point Jacobian

```rust,ignore
pub fn jacobian(model: &Model, state: &State,
    articulation: usize, target_body: usize, point_local: Vec3)
    -> Result<Vec<Real>>;
```

For a serial chain:

```text
v_target = J(q) · q̇
```

where `J` is `6 × N` and `v_target = (v_linear; ω)` is the spatial
velocity at `point_local` (in the target body's frame), expressed in
world axes. Layout is row-major: row `r ∈ [0, 5]`, column `k ∈ [0, N)`
gives the spatial-velocity contribution from joint DoF `k`. Columns
for non-ancestor joints are zero.

This is the primary input to [IK](./ik.md), operational-space control,
and impedance regulation. The implementation walks the
parent-of-target chain to mark ancestor joints, then builds each
ancestor's spatial column via `math::spatial_jacobian_column`:

```text
For each ancestor joint k:
    parent_w = body_q[parent_of(k)]
    joint_w  = parent_w · joint_x_p[k]
    axis_w   = joint_w · joint_axis[k]
    r_world  = target_world − joint_w.origin
    col[k]   = spatial_jacobian_column(jt, axis_w, r_world)
```

For revolute: `(axis_w × r_world; axis_w)`.
For prismatic: `(axis_w; 0)`.
Multi-DoF joints (Ball, D6, Free) write 3 / 6 columns following the
joint's motion-subspace layout.

The Jacobian stays on **CPU** in v1; no consumer needs the GPU port
yet (IK runs on CPU, operational-space control hasn't been profiled
under GPU load).

## CPU + GPU kernel split (Stage 16)

Stage 16 split CRBA + RNEA into matched CPU oracles and GPU kernels:

| kernel | shader | purpose |
|--------|--------|---------|
| `CrbaKernel` | [`crba.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/crba.wgsl) | per-articulation `M(q)` build |
| `RneaKernel` | [`rnea.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/rnea.wgsl) | per-articulation `τ(q, q̇, q̈)` |
| `JacobianKernel` | [`jacobian.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/jacobian.wgsl) | per-(articulation, target) `J(q)` (planned) |

The kernel layout mirrors Featherstone's: one workgroup per
articulation, parallel passes over joints within the articulation
where the algorithm allows (composite-inertia and force-propagation
passes are serial by data dependency, but the per-joint column build
in CRBA is embarrassingly parallel).

Parity tests in `tests/spatial_kernels_*.rs` exercise CPU and GPU
paths against analytically derived `M`, `τ`, `J` for a hand-rolled
double-pendulum + a 7-DoF KUKA-style arm.

## Composition example — forward dynamics from `M` and `τ_bias`

```rust,ignore
use molla::solvers::{mass_matrix, inverse_dynamics};
use molla::math::dense_solve::dense_chol_solve;

let q  = state.joint_q.host()?;
let qd = state.joint_qd.host()?;

// Coriolis + gravity bias: τ_bias = RNEA(q, q̇, q̈ = 0)
let qdd_zero = vec![0.0; n_dof];
let tau_bias = inverse_dynamics(&model, &state, 0, &qdd_zero)?;

// Mass matrix
let m = mass_matrix(&model, &state, 0)?;

// τ_applied - τ_bias = M · q̈   →   q̈ = M⁻¹ · (τ_applied − τ_bias)
let mut rhs: Vec<f64> = tau_applied.iter().zip(&tau_bias).map(|(a, b)| a - b).collect();
let qdd = dense_chol_solve(&mut m.clone(), &rhs, n_dof)?;
```

This is the analytical recipe Featherstone implements directly (and
more efficiently — Featherstone's three passes are `O(n)` while
`M` is `O(n²)` to assemble and `O(n³)` to invert). The composition
is useful when you need the explicit `M` for IK damping, projection,
or impedance control.
