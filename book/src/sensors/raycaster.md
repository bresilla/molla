# Raycaster

`Raycaster` is Molla's batched closest-hit ray sensor, shared by its
LiDAR and camera tracing paths.

`encode_scene` supports [direct leased outputs](device_results.md) and mixed
producer batches over owned scenes. It uploads checked host rays into leased
input channels and computes primitive/moving-mesh hits directly into the leased
result channels. Generated-ray public composition remains pending; existing
`dispatch`, `dispatch_scene`, `cast` and `cast_scene` remain available.

The implementation lives in `crates/molla-sensors/src/raycast.rs` with
the heavy math (per-primitive ray intersection + outward-normal
computation) in `crates/molla-kernels/shaders/raycast.wgsl`. Following molla's
GPU-first rule, the Rust orchestrator owns the per-batch ray / hit
buffers and the dispatch wrapper. Intersections and nearest-hit mesh reduction
execute on GPU. `dispatch` returns device channels without waiting or collecting;
`cast` collects the four active hit channels together after the last mesh pass.

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

The direction is **not** required to be unit-length, but must be finite and
nonzero. The public API normalizes it before both primitive and mesh queries.
`t` is always in meters, and the point is `origin + t * normalize(direction)`.
Origins and positive maximum range must be finite when converted to GPU f32.
Invalid rays return an error for the batch, rather than an ordinary miss.

## Supported primitives

The primitive GPU kernel walks every shape for each ray. Triangle meshes use
local-space GPU BVHs with rigid instance poses; there is not yet a top-level
scene acceleration structure.

| Kind     | Math                                              |
| -------- | ------------------------------------------------- |
| Sphere   | Closed-form quadratic                             |
| Box      | Slab intersection in local frame, 6 faces         |
| Capsule  | Cylinder + 2 hemisphere caps in local frame       |
| Cylinder | Infinite cylinder + 2 disc caps                   |
| Cone     | Finite side surface and circular base            |
| Ellipsoid | Quadratic in scaled local coordinates           |
| Plane    | Single half-space (`+Y` outward normal in local)  |

Each primitive transforms the ray into the shape's local frame using
the inverse of the body world pose × shape local pose, intersects in
local coordinates, then transforms the closest hit point + normal
back into world space.

| Other geometry | Current public raycaster support |
|---|---|
| Static/body-attached TriMesh | Local GPU BVH, live instance pose and GPU nearest-hit merge |
| Standalone Triangle, Heightfield | Converted by `SensorScene`; skipped by raw-model queries |
| ConvexHull, RoundCylinder, SDF | Rejected by default in `SensorScene`; skipped by raw-model queries |

Use [owned sensor scenes](./scene.md) for shared immutable geometry, revisions,
terrain conversion and explicit unsupported-shape diagnostics. Raw-model queries
still skip unsupported geometry; do not treat those misses as free space. Their
mesh cache compares source geometry once per unique mesh per query and rebuilds
after edits/replacement. Pose changes reuse local BVHs. `clear_mesh_cache()`
explicitly releases the legacy cache; owned scenes use `replace_model` instead.
The separate `raycast_adjoint` helper uses CPU finite differences over
spheres/planes; it is not a triangle-query fallback.

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
- Finite GPU-representable origins/range and finite nonzero directions.
- GPU buffer/workgroup capacity limits before construction allocates ray arrays.

Failures surface as `Error::Solver`.

### Device results and optional collection

```rust
use molla::sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};

let mut collector = ReadbackRing::new(&engine, ReadbackConfig {
    slots: 2,
    bytes_per_slot: 64 * 8,
    max_total_bytes: 2 * 64 * 8,
})?;
let hits = sensor.dispatch(&engine, &model, &state, &origins, &directions, 100.0)?;
let ticket = collector.try_enqueue(&[
    hits.distance.readback_source(0..hits.ray_count)?,
    hits.shape.readback_source(0..hits.ray_count)?,
], ReadbackStamp::default())?;
// Poll an accepted ticket later; None from try_enqueue means backpressure.
```

`RayDeviceHits` borrows distance, shape, point and normal `DeviceBuffer`s plus
the active `ray_count`. GPU consumers can bind their `device_buffer()` directly.
Only the active prefix is valid for the current batch. Device miss values are
distance -1, shape -1, point zero and normal +Y; the synchronous wrapper converts
them to `RayHit { shape_id: None, t: INFINITY, .. }`.

Submit downstream commands or enqueue a snapshot on the same engine **before**
the next sensor dispatch. A recorded but unsubmitted command buffer does not
reserve a previous output version. Already enqueued readback copies retain their
own snapshot when the sensor outputs are reused. A different engine is rejected;
the caller remains responsible for uploading raw Model/State buffers on that engine.

Exact equal-distance hits retain the analytical result before meshes. Within
each category, the first shape index wins. Mesh normals oppose the incident ray;
primitive normals retain their outward geometric convention.

`readback_stats()` counts the sensor's internal synchronous collection, not an
external collector. Device-only dispatch leaves all internal counters zero.
`cast` lazily allocates one staging slot of `40 * capacity` bytes, transfers only
`40 * ray_count` active bytes, and performs one grouped collection regardless of
mesh count. The grouped allocation must fit the device's maximum buffer size.
An empty batch does not dispatch or collect. A collection wait is bounded to
30 seconds and returns an error on timeout or failure.

Direct `Raycaster::dispatch` normalizes/uploads CPU ray slices. For generated
terrestrial/aerial rays use [device-resident LiDAR batches](./lidar_batch.md),
which retains rays and sub-ray reductions on GPU. `Raycaster` now encodes
primitive queries (or miss initialization), each mesh traversal and its nearest-hit
merge into one command buffer and submits once per nonempty query. Separate
uniform buffers for each mesh/merge preserve instance IDs and parameters within
that shared submission.

`query_stats()` reports encoded queries, direct query submissions and cumulative
primitive/mesh/merge binding preparations. A primitive or mesh preparation creates
one uniform buffer and four bind groups; a merge preparation creates one uniform
and one bind group. These counters exclude initial construction, geometry uploads,
command encoder/buffer creation and wgpu's internal upload staging.

The sensor retains two input-buffer variants for the primitive pass and two per
mesh instance, supporting alternating physics state buffers without rebinding
after warm-up. Pose values, active ray counts and ranges may change without new
bindings. Replaced buffer identities rebuild affected bindings; removed mesh
instances are pruned after a nonempty query. More than two rotating input-buffer
variants can cause bounded cache churn. Legacy mesh-content comparisons/CPU ray
uploads still have their documented costs.

The low-level kernel compatibility dispatch methods still allocate temporary
bindings. Full shared execution across different sensors, leased outputs for
unsubmitted consumers and top-level scene acceleration remain unfinished. The
prepared kernel encode APIs require distinct parameter sets for commands with
different arguments, and submission before rewriting a parameter set; they are
not independently versioned output leases.

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
`cast` runs — call `Model::upload_to_device(device, queue)` after
building, and ensure `State::body_q` is current on the device (the
solvers already keep it there). The kernel reads `state.body_q`
through its device buffer and accepts the shapes via the static
`Model::shape_*` slabs.

All-static primitive scenes are supported without allocating a physics body.
The sensor supplies an internal identity pose for the unused body binding.

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
