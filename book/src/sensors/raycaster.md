# Raycaster

`Raycaster` is molla's batched closest-hit ray sensor. It mirrors
Newton's `SensorRayCaster` one-for-one and is the primitive that
LIDAR-style point-cloud sensors, proximity probes, and pixel
visibility queries all build on top of.

The implementation lives in `crates/molla/src/sensors/raycast.rs` with
the heavy math (per-primitive ray intersection + outward-normal
computation) in `crates/molla-kernels/shaders/raycast.wgsl`. Following molla's
GPU-first rule, the Rust orchestrator owns the per-batch ray / hit
buffers and the dispatch wrapper; the CPU oracle path is only used
inside parity tests.

## What it computes

For each ray `(origin, direction)` and a `max_distance` budget, the
kernel returns the closest hit:

```rust
pub struct RayHit {
    pub shape_id: Option<u32>,   // index into Model::shape_geometry
    pub body_id: BodyId,          // owner body, BodyId::NONE for static
    pub t: Real,                  // hit distance along the ray (f64::INFINITY = miss)
    pub point: Vec3,              // world-space hit point
    pub normal: Vec3,             // world-space outward surface normal
}
```

The direction is **not** required to be unit-length. `t` is measured
in whatever units `direction` carries — pass a unit direction to read
distance in metres, or a non-unit direction to read the parametric `t`
that lands the hit at `origin + t · direction`.

## Supported primitives

The GPU kernel walks every shape on every ray (no BVH culling in v1;
that lands when the GPU mesh BVH integration ships, tracked in the
module-level doc comment). It handles five primitives:

| Kind     | Math                                              |
| -------- | ------------------------------------------------- |
| Sphere   | Closed-form quadratic                             |
| Box      | Slab intersection in local frame, 6 faces         |
| Capsule  | Cylinder + 2 hemisphere caps in local frame       |
| Cylinder | Infinite cylinder + 2 disc caps                   |
| Plane    | Single half-space (`+Y` outward normal in local)  |

Each primitive transforms the ray into the shape's local frame using
the inverse of the body world pose × shape local pose, intersects in
local coordinates, then transforms the closest hit point + normal
back into world space.

Triangle / TriMesh shapes carry a `0xff` "skip" sentinel in
`Model::shape_geom_kind` so the GPU walks past them without aborting.
The mesh BVH already ships (Stage 15, see
[`geometry/bvh`](../geometry/bvh.md)) but the integration with
`raycast.wgsl` is still TODO. The
[`raycast_adjoint`](../autograd/raycast.md) CPU module provides a
finite-difference fallback for inverse-rendering experiments that
need triangles in the loop.

## API

```rust
use molla::compute::Engine;
use molla::sensors::Raycaster;
use molla::math::Vec3;

let engine = Engine::new_gpu()?;
let mut sensor = Raycaster::new(&engine, /* max_rays = */ 64)?;

let origins    = vec![Vec3::new(0.0, 5.0, 0.0); 1];
let directions = vec![Vec3::new(0.0, -1.0, 0.0); 1];
let hits = sensor.cast(&engine, &model, &state, &origins, &directions, 100.0)?;

if let Some(shape) = hits[0].shape_id {
    println!("hit shape {shape} at t={:.3}, n={:?}",
             hits[0].t, hits[0].normal);
}
```

`Raycaster::new(engine, max_rays)` allocates the per-batch GPU
buffers up-front and is cheap thereafter: every `cast` call reuses
the same device storage, so spinning up a new sensor per frame is
unnecessary (and wasteful). The `max_rays` parameter is the
capacity ceiling; calls with fewer rays simply dispatch a smaller
threadgroup count.

The orchestrator validates input slices on every call:

- `origins.len() == directions.len()`.
- `origins.len() <= max_rays` (set at construction).

Failures surface as `Error::Solver`.

## Ray bundle generation

For LIDAR-style scans, the caller usually generates a fan of rays
around a single origin. The convenience pattern is to precompute the
direction list once (the rays sweep a polar grid in body frame) and
multiply by the body pose each frame, then call `cast` once per
sensor sweep. The
[`TiledCamera`](./tiled_camera.md) chapter covers the more
specialised case of pinhole-camera bundles where direction generation
itself runs as a GPU kernel.

## State requirements

`Model` and `State` must have their device mirrors uploaded before
`cast` runs — call `Model::upload_to_device(engine)` once after
building, and ensure `State::body_q` is current on the device (the
solvers already keep it there). The kernel reads `state.body_q`
through its device buffer and accepts the shapes via the static
`Model::shape_*` slabs.

## Cross-references

- [`Sensors / TiledCamera`](./tiled_camera.md) — the multi-channel
  pinhole-camera sensor built on the same intersection routines but
  with on-device ray generation + multi-camera batching.
- [`Sensors / Raycast adjoint`](../autograd/raycast.md) — Stage 37.7
  CPU finite-difference adjoint used for inverse-rendering demos
  while the analytic ∂t/∂param adjoint is on the backlog.
- [`Geometry / Shape primitives`](../geometry/shapes.md) — defines
  `Model::shape_geom_kind` + the per-primitive parameter layout the
  WGSL kernel reads from.
