# Raycaster

`Raycaster` is Molla's batched closest-hit ray sensor, shared by its
LiDAR and camera tracing paths.

`encode_scene` supports [direct leased outputs](device_results.md) and mixed
producer batches over owned scenes. It uploads checked host rays into leased
input channels and computes primitive/moving-mesh hits directly into the leased
result channels. `LidarBatch` and `TiledCamera` use generated rays in the same
device-only command batches; existing
`dispatch`, `dispatch_scene`, `cast` and `cast_scene` remain available.

The implementation lives in `crates/molla-sensors/src/raycast.rs` with
the shared primitive intersections in `crates/molla-kernels/shaders/ray_shapes.wgsl`
and owned-scene traversal in `scene_trace.wgsl`. Following molla's
GPU-first rule, the Rust orchestrator owns the per-batch ray / hit
buffers and the dispatch wrapper. Intersections and nearest-hit mesh reduction
execute on GPU. `dispatch` returns device channels without waiting or collecting;
`cast` collects the four active hit channels together after the query completes.

## What it computes

For each ray `(origin, direction)` and a `max_distance` budget, the
kernel returns the closest hit:

```rust
pub struct RayHit {
    pub shape_id: Option<u32>,   // index into Model::shape_geometry
    pub body_id: BodyId,          // owner body, BodyId::NONE for static
    pub t: Real,                  // hit distance along the ray (f64::INFINITY = miss)
    pub point: Vec3,              // world-space hit point
    pub normal: Vec3,             // outward primitive normal; ray-opposing mesh normal
}
```

The direction is **not** required to be unit-length, but must be finite and
nonzero. The public API normalizes it before both primitive and mesh queries.
`t` is always in meters, and the point is `origin + t * normalize(direction)`.
Origins and positive maximum range must be finite when converted to GPU f32.
Invalid rays return an error for the batch, rather than an ordinary miss.

## Supported primitives

Owned-scene queries traverse a refitted finite-instance hierarchy and shared local
mesh BVHs, with infinite planes handled separately. Raw-model queries retain the
linear primitive loop and per-instance mesh passes. See [scene acceleration](scene.md#shared-accelerated-queries).

| Kind     | Math                                              |
| -------- | ------------------------------------------------- |
| Sphere   | Closed-form quadratic                             |
| Box      | Slab intersection in local frame, 6 faces         |
| Capsule  | Cylinder + 2 hemisphere caps in local frame       |
| Cylinder | Infinite cylinder + 2 disc caps                   |
| Cone     | Finite side surface and circular base            |
| Ellipsoid | Quadratic in scaled local coordinates           |
| Plane    | Infinite boundary plane with its configured local normal |

Each primitive transforms the ray into the shape's local frame using
the inverse of the body world pose × shape local pose, intersects in
local coordinates, then transforms the closest hit point + normal
back into world space.

| Other geometry | Current public raycaster support |
|---|---|
| Static/body-attached TriMesh | Local GPU BVH, live instance pose and GPU nearest-hit merge |
| Standalone Triangle, Heightfield | Converted by `SensorScene`; skipped by raw-model queries |
| ConvexHull | Triangulated by `SensorScene`; skipped by raw-model queries |
| RoundCylinder | Dilated cylinder approximation in `SensorScene`; skipped by raw-model queries |
| SDF | Rejected by default in `SensorScene`; skipped by raw-model queries |

Use [owned sensor scenes](./scene.md) for shared immutable geometry, revisions,
terrain conversion and explicit unsupported-shape diagnostics. Raw-model queries
still skip unsupported geometry; do not treat those misses as free space. Their
mesh cache compares source geometry once per unique mesh per query and rebuilds
after edits/replacement. Pose changes reuse local BVHs. `clear_mesh_cache()`
explicitly releases the legacy cache; owned scenes use `replace_model` instead.
The separate `raycast_adjoint` helper uses CPU finite differences over
spheres/planes; it is not a triangle-query fallback.
Its finite nonzero ray directions are also normalized for metric distances.
Unlike the GPU sensor's `Result` API, invalid CPU adjoint rays return infinity.

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
threadgroup count. A zero requested capacity preserves the legacy one-slot minimum.

### Ray-array storage budgets

`new` uses `RaycasterLimits::default()`: a **1 GiB combined host/device ray-array
payload budget**. `RayStorageUsage::for_capacity(max_rays)` reports that payload
without creating an engine, allocating arrays or accessing a GPU. `new_with_limits`
accepts an explicit override, but GPU binding-size/buffer-size/workgroup limits
still apply independently. Validation precedes ray-array allocation and pipeline
creation; an oversized request returns an error rather than allocating or chunking.

```rust
use molla::sensors::{Raycaster, RaycasterLimits, RayStorageUsage};

let storage = RayStorageUsage::for_capacity(1_000_000)?;
let sensor = Raycaster::new_with_limits(&engine, storage.capacity, RaycasterLimits {
    max_ray_storage_bytes: storage.total_bytes,
})?;
```

The estimate includes six ray/hit channel arrays, three mesh-hit scratch arrays,
their host mirrors, two retained packed upload arrays and the static body-pose
fallback. GPU payload is `96 * capacity + 32` bytes; host payload uses the actual
Rust element sizes. The result reports host, device and combined bytes separately.
Unit/GPU tests compare this estimate with constructed array lengths and GPU
buffer sizes, including the one-slot minimum.

This is **not a process-wide memory or peak-allocation cap**. It excludes mesh/BVH
caches, owned-scene refit storage, model/state storage, pipeline/binding metadata, allocator/driver overhead,
temporary upload packing/staging, optional readback, leased output pools, caller
ray vectors and returned host hit vectors. Passing validation does not guarantee
that the machine has enough free memory. Budget those other resources separately.

Legacy `Lidar` and `AerialLidar` expose matching `new_with_limits` constructors.
`LidarBatch::new_with_ray_limits` and `TiledCamera::new_with_ray_limits` override
their shared raycaster budget; descriptors, pulse outputs, camera poses/images
and aperture storage are additional. Their ordinary constructors use the same
1 GiB default. Heterogeneous `CameraBatch` uses that default per homogeneous
group, separately from its aggregate pixel and leased-output budgets.

The default can now reject a large capacity that previously passed individual
buffer checks. This is an explicit capacity error, not silent truncation or a
change to valid scan results. Callers needing more ray storage must opt into a
larger budget and still satisfy device limits.

The orchestrator validates input slices on every call:

- `origins.len() == directions.len()`.
- `origins.len() <= max_rays` (set at construction).
- Finite GPU-representable origins/range and finite nonzero directions.
- Aggregate ray-array budget and GPU buffer/workgroup capacity limits before construction allocates ray arrays.

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
which retains rays and sub-ray reductions on GPU. Raw-model `Raycaster` queries encode
primitive queries (or miss initialization), each mesh traversal and its nearest-hit
merge into one command buffer and submits once per nonempty query. Separate
uniform buffers for each mesh/merge preserve instance IDs and parameters within
that shared submission. Owned scenes instead encode dependency-ordered refit levels
followed by one shared hierarchy traversal, retaining the same submission boundary.

`query_stats()` reports encoded queries, direct query submissions and cumulative
legacy primitive/mesh/merge binding preparations. A primitive or mesh preparation creates
one uniform buffer and four bind groups; a merge preparation creates one uniform
and one bind group. These counters exclude initial construction, geometry uploads,
command encoder/buffer creation and wgpu's internal upload staging.

The raw-model path retains two input-buffer variants for the primitive pass and two per
mesh instance, supporting alternating physics state buffers without rebinding
after warm-up. Pose values, active ray counts and ranges may change without new
bindings. Replaced buffer identities rebuild affected bindings; removed mesh
instances are pruned after a nonempty query. More than two rotating input-buffer
variants can cause bounded cache churn. Legacy mesh-content comparisons/CPU ray
uploads still have their documented costs.

Owned-scene queries instead report `scene_builds`, `scene_rebuilds`, `scene_quality_checks`, `scene_preparations`
and `refit_preparations`. They keep up to 128 trace-binding variants and two body-pose
refit variants. Trace parameters are immutable; a new count/range or buffer set
prepares a new variant, while changed pose values reuse bindings. These counters
do not prove zero allocations inside wgpu or end-to-end speedups.

The low-level kernel compatibility dispatch methods still allocate temporary
bindings. Their queue-written prepared parameters require submission before
rewriting a parameter set. The shared owned-scene path uses immutable parameters
and [leased output frames](device_results.md); low-level bindings alone are not
independently versioned output leases. [Periodic GPU rebuilding](scene.md#periodic-gpu-spatial-rebuilding)
uses a bounded per-producer policy without pose readback. Its counter/cadence track
encoded queries, including cancelled batches; empty queries and backpressure do not
advance it. Optional [GPU quality-triggered rebuilding](scene.md#gpu-quality-triggered-rebuilding)
measures normalized hierarchy area and conditionally reorders without host readback.
Its encoded-check counter is separate from executed GPU decisions. Full lifecycle/
allocation and performance acceptance remain unfinished.

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
