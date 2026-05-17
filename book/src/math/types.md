# Vectors, matrices, transforms

`crates/molla/src/math/mod.rs` is the type-aliases hub for the host
side. Everything physics-facing on the CPU runs in `f64`; the matching
`f32` mirrors live in `crates/molla/src/math/gpu.rs` and are produced
at the buffer boundary by [`BufferElement::to_gpu`](./../compute/storage.md).

## The `Real` scalar and glam aliases

```rust
/// Host-side scalar. The crate is `f64` everywhere on the host —
/// see the precision policy in `PLAN.md` §7.
pub type Real = f64;

pub type Vec3 = glam::DVec3;
pub type Quat = glam::DQuat;
pub type Mat3 = glam::DMat3;

pub const GRAVITY: Real = 9.806_65;
```

molla never owns its own `Vec3`/`Mat3`/`Quat` types — every algebraic
primitive is a `glam::D*` alias. The crate gets glam's vectorised SIMD
paths, swizzles, and `From`/`Into` for free, and the cost is a single
re-export wall in `math/mod.rs`. Storage layout is glam's: `Mat3` holds
three columns left-to-right (`m.col(0)`, `m.col(1)`, `m.col(2)`); a
`Quat` is `(x, y, z, w)`. Both invariants leak into the WGSL mirrors,
so do not change them without coordinating with [`GpuMat3`](./../compute/storage.md).

## `Transform` — rigid pose

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}
```

`crates/molla/src/math/transform.rs` defines a rigid transform as
`(position, rotation)` — first apply `rotation` to the body-local
frame, then translate by `position`. The layout matches Warp's
`wp.transform` byte-for-byte, so a `Transform` (host `f64`, padded to
32 bytes through [`GpuTransform`](./../compute/storage.md)) round-trips
cleanly into WGSL.

The key entry points:

```rust
impl Transform {
    pub const IDENTITY: Self;
    pub const fn new(position: Vec3, rotation: Quat) -> Self;
    pub const fn from_translation(t: Vec3) -> Self;
    pub const fn from_rotation(r: Quat) -> Self;
    pub fn compose(&self, other: &Self) -> Self;
    pub fn inverse(&self) -> Self;
    pub fn transform_point(&self, p: Vec3) -> Vec3;
    pub fn transform_vector(&self, v: Vec3) -> Vec3;
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3;
    pub fn rotation_matrix(&self) -> Mat3;
    pub fn renormalize(&mut self);
    pub fn to_gpu(&self) -> GpuTransform;
}
```

`compose(self, other)` reads as "first apply `other`, then `self`" —
the convention that lets you write
`world = parent_world.compose(&child_local)` without juggling sides.
After composition we always `normalize()` the quaternion; quaternion
drift is the dominant numerical hygiene problem in long-running
articulated integration, and one normalise per compose is cheap.
Integrators that step quaternions explicitly should additionally call
[`Transform::renormalize`] every step (see `solvers::semi_implicit`).

`Transform::IDENTITY` is `const` so it can be used as a default
field value without `Default` blanket trickery — every model-builder
field that holds an unset pose initialises to `IDENTITY`.

## Glam vs hand-rolled — why not a custom Mat3?

molla's predecessor stacks (Newton / Warp) carry their own
`wp.transform`/`wp.spatial_*` types. Rust has
glam, glam has been the de-facto game-engine pick for half a decade,
and `glam::DMat3` already inlines into the same SIMD that a custom
type would synthesise. The single concession is that **glam stores
columns**, which means anything that reads "row-major" in a textbook
or a Warp shader needs translating: `m.row(i)` returns a `Vec3` of
that row, and `mset(&mut m, i, j, v)` (in `dense3.rs`) sets row `i`
column `j`.

Cross-references:

- [Spatial algebra](./spatial.md) builds 6-component types
  (`SpatialVector`, `SpatialInertia`, `SpatialMatrix6`) on top of the
  3-component aliases here.
- [Small-matrix decompositions](./dense3.md) ships SVD/QR/eig for the
  3×3 case used in IK regularisation and per-vertex polar decomposition.
- [Storage abstractions](./../compute/storage.md) shows the host→`f32`
  conversion path that bridges these types to WGSL.
