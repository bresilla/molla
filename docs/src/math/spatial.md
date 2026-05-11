# Spatial algebra

`crates/molla/src/math/spatial.rs` ships molla's 6-component spatial
algebra primitives — the workhorses of Featherstone's articulated-body
algorithms (ABA, CRBA, RNEA). All types use the `[linear; angular]`
ordering, matching Warp's `wp.spatial_vector` (`spatial_top` = linear).
Mixing conventions across modules is a silent-correctness hazard, so
the layout is documented at every boundary and `repr(C)` is enforced
so the GPU mirrors line up.

## `SpatialVector` — twists and wrenches

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SpatialVector {
    pub linear: Vec3,
    pub angular: Vec3,
}
```

The same struct represents motion vectors (twists: linear + angular
velocity) and force vectors (wrenches: force + torque) — context
distinguishes them, not type. `SpatialVector::ZERO`, `::new(linear,
angular)`, `::linear(v)`, `::angular(w)` cover construction; `+`,
`-`, scalar `*`, and `Neg` are implemented componentwise.

## `SpatialInertia` — rigid-body inertia

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialInertia {
    pub mass: Real,      // kg
    pub com: Vec3,       // body-frame COM, m
    pub inertia: Mat3,   // at COM, body frame, kg·m²
}
```

This is the compact form used when the inertia is still attached to a
single body — sphere/cuboid constructors are provided, plus
`inv_mass()` / `inv_inertia()` that return `0` and `Mat3::ZERO` when
`mass == 0` (the static / kinematic sentinel).

```rust
SpatialInertia::sphere(mass, radius);
SpatialInertia::cuboid(mass, full_extents);  // (sx, sy, sz)
```

`SpatialInertia::ZERO` is the const "no body" placeholder.

## `SpatialMatrix6` — articulated inertia

```rust
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialMatrix6 {
    pub m: [[Real; 6]; 6],  // row-major
}
```

Featherstone's ABA accumulates child contributions through the
backward pass, and the simple `(mass, com, inertia)` representation
isn't closed under that update — the articulated inertia outgrows it.
`SpatialMatrix6` carries the full 6×6 in row-major layout (matches
Warp's `wp.spatial_matrix`).

Methods:

```rust
SpatialMatrix6::ZERO;
SpatialMatrix6::identity();
fn scale(self, s: Real) -> Self;
fn add(&self, o: &Self) -> Self;
fn sub(&self, o: &Self) -> Self;
fn mul_sv(&self, v: SpatialVector) -> SpatialVector;
fn outer(u: SpatialVector, v: SpatialVector) -> Self;
fn from_body(mass: Real, com: Vec3, inertia_world: Mat3) -> Self;
```

`mul_sv(v)` is `M · v` with `v = [linear; angular]` laid out as
`[v0..v2, v3..v5]`. `outer(u, v)` is `u · vᵀ`, used in ABA's rank-1
update

```text
I^A := I^A − (I^A · S) · (I^A · S)ᵀ / (Sᵀ · I^A · S)
```

`from_body(mass, com, inertia_world)` builds the spatial inertia of a
rigid body about an arbitrary reference point, per Featherstone (2008)
eq. (2.63):

```text
        [ m·I₃                     m·skew(c)         ]
I_s =   [ m·skew(c)ᵀ   I_C − m·skew(c)·skew(c)       ]
```

with `[v; w]` ordering. The skew matrix `skew(c)` is built inline
(`skew3` private helper); since `skew` is antisymmetric the bottom-left
block ends up `-m·skew(c)` rather than the transpose, and the code
exploits that.

## Spatial cross products

```rust
pub fn spatial_cross(a: SpatialVector, b: SpatialVector) -> SpatialVector;
pub fn spatial_cross_dual(a: SpatialVector, b: SpatialVector) -> SpatialVector;
```

Two operators, matching Newton's `spatial_cross` / `spatial_cross_dual`
WGSL builtins (`xtra/warp/_src/builtins.py`):

- `spatial_cross(a, b)` — motion-vector cross product:

  ```text
  linear  = w_a × v_b + v_a × w_b
  angular = w_a × w_b
  ```

  Drives the Coriolis terms in ABA's forward sweep.

- `spatial_cross_dual(a, b)` — force-vector cross product (the dual
  operator):

  ```text
  linear  = w_a × v_b
  angular = w_a × w_b + v_a × v_b
  ```

  Drives the Coriolis bias in ABA's wrench accumulation.

## `spatial_jacobian_column` — single-DoF Jacobian columns

```rust
pub enum JacobianKind { Revolute, Prismatic }

pub fn spatial_jacobian_column(
    kind: JacobianKind,
    joint_axis_world: Vec3,
    joint_origin_world: Vec3,
    body_point_world: Vec3,
) -> SpatialVector;
```

Builds one column of the spatial Jacobian for a single-DoF joint's
effect on a downstream body's reference point. All inputs are in the
**world frame** — callers (CRBA, IK, sensor frame transforms) are
responsible for transforming joint axes / origins to world before
calling.

Revolute joints produce `(axis × (body_point − joint_origin), axis)`;
prismatic joints produce `(axis, 0)`. Multi-DoF joints (Ball, D6,
Distance, CABLE) decompose into multiple single-DoF columns by
sweeping through their axes and calling this helper per axis.

The function mirrors Newton's `spatial_jacobian` builtin
(`xtra/warp/_src/builtins.py:2261`) and is the host fallback for the
GPU Jacobian kernel — both paths must produce identical column vectors
for parity tests to pass.

Cross-references:

- [Articulation (Featherstone ABA)](./../sim/featherstone.md) and
  [CRBA / RNEA / Jacobian](./../sim/spatial_kernels.md) consume these
  primitives directly.
- [Vectors, matrices, transforms](./types.md) covers the underlying
  `Vec3`/`Mat3`/`Quat` aliases.
