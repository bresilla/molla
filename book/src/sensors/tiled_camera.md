# TiledCamera

`TiledCamera` is Molla's batched, multi-channel pinhole/thin-lens camera
sensor. A single `dispatch()` submits the passes for every camera in the batch
and writes per-pixel `(depth, color, normal, albedo, shape_index)`
into device buffers of shape `(num_cameras, height, width)`.

Owned-scene invalid poses produce depth `-2` and shape ID `u32::MAX - 1`, not
ordinary misses. Enable depth or shape indices when checking validity: zero
color/albedo/normal alone cannot distinguish invalid geometry from a miss.
Aperture accumulation preserves the center depth/shape sentinel. Dispatch and
readback success do not assert valid scene poses; see
[query validity](contracts.md#invalid-owned-scene-results).

It's the workhorse for pixel-observation RL environments, where you
typically have one camera per parallel world and want all of them
rendered in lock-step so the policy sees a tensor of shape
`(num_envs, H, W, channels)` each step.

Source: `crates/molla-sensors/src/tiled_camera.rs`. Shaders:
`crates/molla-kernels/shaders/camera_rays.wgsl` (ray generation), shared
raycast/mesh-query kernels, `camera_shade.wgsl` (channel shading), and
`crates/molla-kernels/shaders/camera_accumulate.wgsl` (aperture accumulation/resolve).

## Pipeline

Each aperture sample runs these device stages:

1. **Ray generation** (`CameraRaysKernel`). Per-pixel `(camera, x, y)`
   triples are turned into world-space `(ray_origin, ray_direction)`
   pairs using the per-camera world pose + shared intrinsics
   (vertical FOV, aspect ratio). One thread per pixel.

2. **Closest-hit query** (`Raycaster`). Shared primitive queries, mesh BVH
   traversal and GPU nearest-hit merging select visibility. Rigid moving mesh
   instances consume live GPU body poses without rebuilding vertices/BVHs.

   Owned-scene bounds refit on the center sample of every frame. Later aperture
   samples reuse those bounds within the same encoding call. Scheduled periodic
   or quality-triggered spatial rebuilding retains its post-reorder refit and
   query-count cadence. Separate frames always refresh, including after canceled
   batches; no bounds-readiness state persists across frame encodings.

3. **Shading** (`CameraShadeKernel`). Closest-hit distance/normal/shape ID and
   per-shape colors produce the selected [`CameraChannels`](#channels). Hits
   at or beyond `max_distance` are misses. Miss depth is -1, ID is `u32::MAX`,
   normal is zero and packed color/albedo are zero. Analytic normals are
   outward; triangle normals oppose the incident ray. Exact distance ties
   retain analytic hits first, then the earliest mesh instance.

The center sample writes selected channels. Additional aperture samples write
only color, accumulating unpacked RGBA8 byte values in GPU floating-point
storage. One final resolve divides by the sample count, rounds halfway values
up and packs RGBA8. This preserves the previous packed-color average: it is
not linear-light averaging. Depth, normals, albedo and IDs retain the center
sample without another trace or host restoration.

All ray/query/shade/accumulate/resolve passes share one submission per nonempty frame.
There are no intermediate downloads or host image arrays. Camera poses and
parameters are uploaded; aperture offsets are still generated on the CPU.
Without COLOR, only the center sample runs and no accumulation resources are
created. A NONE channel mask submits no work.

## Configuration

```rust
pub struct TiledCameraConfig {
    pub width: u32,
    pub height: u32,
    pub fov_y_radians: f32,
    pub channels: CameraChannels,
    pub ambient: Vec3,           // ambient RGB added to every shaded pixel
    pub light_dir: Vec3,         // world-space directional light vector
    pub light_color: Vec3,       // directional-light colour (× n·-L)
    pub max_distance: f32,
    pub near: f32,               // informational, not enforced in v1
    pub optics: PhysicalOptics,
}
```

`Default` produces a 128×96 pinhole at 60° vertical FOV with a
warm-white directional light and a soft blue ambient — useful for a
viewer thumbnail without wiring custom lighting.

Width / height / camera count are fixed at construction time because
the output buffers are sized then. Everything else can be flipped at
any frame via `set_config()` — including the channels mask, so you
can render only depth on most frames and turn on the full RGBA +
normal stack only for the frames you log.

### Fixed-array storage budgets

Construction checks a default 1 GiB combined host/device camera-array budget
before allocating buffers. `CameraStorageUsage::for_dimensions(width, height,
num_cameras)` estimates the shared ray arrays, all five image channels, camera
poses and pose-packing scratch, the empty color fallback, and reserved aperture
sums/uniform. All channels count even with `CameraChannels::NONE`; aperture
storage counts even in pinhole mode. Changing channels or optics therefore does
not bypass the fixed-array limit. `storage_usage()` returns this estimate, not
current driver memory usage.

`TiledCamera::new_with_limits` accepts `TiledCameraLimits` with
`max_storage_bytes` and a separate `ray_limits` sub-budget. Both default to
1 GiB and both must pass, along with device binding and dispatch limits.
`new_with_ray_limits` overrides only the ray sub-budget, retaining the default
combined camera cap. For example, four 1280×720 images exceed the default
combined cap even though their ray arrays fit the ray-only budget. An explicit
larger cap permits such configurations only when device limits also allow them.

This is an array-payload budget, not a process peak-memory cap. Scene/query
resources, other uniforms/bindings, driver/upload staging, temporary packing,
caller-owned data and leased output/readback pools are excluded. Aperture sums
are still allocated lazily; their reserved GPU bytes are reported separately.
See also [ray-array budgets](raycaster.md#ray-array-storage-budgets) and
`crates/molla-sensors/src/camera_storage.rs`.

`PhysicalOptics` supplies focal length/sensor width (both positive to override
FOV), aperture diameter, focal distance and aperture sample count. Zero aperture
is pinhole. Sample counts 0/1 select the center; counts through 1024 are accepted,
and larger counts are rejected before allocation/dispatch. Config values must
be finite; distances/optics are nonnegative, max distance is positive, and both
configured/effective FOV must be strictly between zero and pi. Invalid updates
leave the previous configuration unchanged. Camera poses must be finite rigid
transforms with unit quaternions. Near remains informational, not clipping.

## Channels

`CameraChannels` is a bitmask of which output buffers to write:

```rust
impl CameraChannels {
    pub const COLOR: Self        = Self(1 << 0);
    pub const DEPTH: Self        = Self(1 << 1);
    pub const NORMAL: Self       = Self(1 << 2);
    pub const ALBEDO: Self       = Self(1 << 3);
    pub const SHAPE_INDEX: Self  = Self(1 << 4);
    pub const ALL: Self          = ...; // all five OR'd together
    pub const NONE: Self         = Self(0);
}
```

The semantics:

- **COLOR** — Lambert-shaded RGB packed into a `u32` (8 bits per
  channel). Cheap, suitable for visualisation thumbnails.
- **DEPTH** — `Real` (`f64`) metric distance along the normalized pixel ray,
  not camera-axis Z depth. `-1.0`
  marks a miss. This is the channel the depth-loss / pose-recovery
  loops consume.
- **NORMAL** — world-space outward normal per pixel as `Vec3`. Zero
  vector on miss. Useful for inverse-rendering or as a feature
  channel for the policy.
- **ALBEDO** — unshaded per-shape colour (no lighting). Useful for
  rendering pipelines that want to apply post-process lighting
  themselves.
- **SHAPE_INDEX** — `Model::shape_geometry` index of the hit shape
  (`u32::MAX` for miss). This is the instance-segmentation channel:
  every pixel tells you exactly which body-owned shape lit it up.

Bits are combined with `|`:

```rust
let cfg = TiledCameraConfig {
    channels: CameraChannels::DEPTH | CameraChannels::SHAPE_INDEX,
    ..Default::default()
};
```

Channels not in the mask are skipped at write time inside the kernel
(the backing buffers are always allocated, so the mask can be
flipped per-frame without reallocation).

## API

### Direct leased outputs

`output_pool(engine, config)` creates bounded slots for
`encode_scene(batch, engine, scene, state, pool, stamp)`. It returns `None`
on backpressure without pose uploads, binding preparation or GPU commands;
otherwise `EncodedCamera` contains a receipt, dimensions, camera count, channel
mask and actual scene stamp. The caller's frame stamp remains separate metadata.

Pool channels are **depth `Real`, color `u32`, normal `Vec3`, albedo `u32`,
shape ID `u32`**, in that order. This order is not the numeric bit order of
`CameraChannels` (COLOR is bit zero). Enabled channels have active range
`0..width*height*camera_count`; disabled channels are empty and their storage
is cleared, not inherited from previous frames. A NONE mask reserves a fresh
empty frame with no pose upload or kernels. Legacy dispatch still preserves
disabled-channel buffers and their last-write stamps.

Each slot reserves `32 * width * height * camera_count` bytes, separately
budgeted from camera scratch and legacy outputs. Ray generation and scene
queries use retained private scratch; shading writes directly into leased
channels. Aperture accumulation shares retained sums/uniforms and caches
output bindings, then resolves directly into the leased color channel. Center
depth/normal/albedo/shape semantics and existing packed-byte color averaging
are unchanged. No intermediate readback or legacy-output snapshot copy occurs.

Multiple independent `TiledCamera` producers can share one `SensorCommandBatch`
with LiDAR and processing stages. Feed pending color channel 1 to
`ImageProcessor::encode_pending` and shape channel 4 to
`AnnotationProcessor::encode_pending`. Homogeneous camera images occupy
contiguous `width*height` slices. The heterogeneous `CameraBatch` manager
encodes all of its groups into the same command batch and supplies per-image
pending jobs; see [heterogeneous camera batches](#heterogeneous-camera-batches).

Producer, scene and body-state borrows last until submission/drop. Active
body-bearing queries require current canonical-engine GPU poses. Cancellation,
poison, reset/history and device-loss rules follow the shared batch contract;
legacy channel stamps are not updated by a direct encode. Submitted frames
remain valid after camera/pool drop while their device remains healthy.

Direct shading and aperture bindings cache up to 128 variants each. Camera ray
bindings and aperture sums remain shared; changing samples/focus/range does not
rebuild fixed-slot bindings. `encoded_frames` counts successful encodes,
including cancelled work; legacy `submissions` is unchanged. Pass/preparation
counters include both paths. `input_upload_bytes` records the 32-byte GPU pose
per camera, excluding uniforms. Pose conversion uses caller-retained packed
storage through `Buffer::upload_with_scratch`, sized once to the camera count;
neither direct nor legacy camera dispatch allocates a conversion vector each
frame. These counters still do not prove whole-pipeline zero allocation:
command encoders, submission bookkeeping and driver staging allocate.

### Legacy dispatch and collection

```rust
use molla::compute::Engine;
use molla::sensors::{TiledCamera, TiledCameraConfig, CameraChannels};
use molla::math::Vec3;

let engine = Engine::new_gpu()?;
let cfg = TiledCameraConfig {
    width: 256,
    height: 192,
    channels: CameraChannels::COLOR | CameraChannels::DEPTH,
    ..Default::default()
};
let mut cam = TiledCamera::new(&engine, cfg, /* num_cameras = */ 16)?;

// Aim the cameras. The convenience helper builds a lookAt pose.
for i in 0..16 {
    let eye = Vec3::new(0.0, 1.5, 4.0);
    let tgt = Vec3::ZERO;
    cam.set_camera_look_at(i, eye, tgt, Vec3::Y)?;
}

cam.dispatch(&engine, &model, &state)?;
let depth = cam.download_depth(&engine)?;
let color = cam.download_color(&engine)?;
```

Per-camera poses use molla's `Transform` (`position + rotation`),
where rotation maps the canonical camera axes (`+X` right, `+Y` up,
`-Z` forward) into world. `set_camera_look_at(idx, eye, target,
up_hint)` builds the same pose without the caller having to compose
a quaternion by hand — it's the same matrix as the classical OpenGL
`lookAt`, just expressed via a rotation quaternion.

Output buffers are laid out image-major, row-major within an image.
The linear pixel index for `(cam, x, y)` is exposed via:

```rust
let idx = cam.pixel_index(cam_id, x, y);
let d   = depth[idx];
```

The `download_*` helpers wrap `Buffer::download` + `Buffer::host`
into a single call that returns a borrow on the per-pixel slice —
useful when the policy code wants to view the depth tensor without
copying.

For grouped asynchronous collection or copied snapshots, use
`camera.readback_source(channel, pixel_range)`: choose exactly one channel per
source and enqueue several sources in a [ReadbackRing](readback.md) or
[DeviceSnapshotPool](snapshot.md). These copies leave the camera host mirrors
stale. The dispatch path also leaves written outputs device-authoritative;
call a collection API before reading host pixels. Borrowed outputs are overwritten
on redispatch: enqueue consumers/copies first, or hold a copied snapshot lease.
The legacy `download_*` methods still perform synchronous per-channel collection.

### Resource reuse and counters

`dispatch_stats()` records submissions, ray/visibility-query/accumulation/resolve
passes and resource preparations. `trace_passes` counts logical queries;
`trace_preparations` now counts the separate hit-shading bindings. Query resource
preparations are reported by `query_stats()`; its submissions stay zero because
queries share the camera's encoder. Ray bindings retain 1024 aligned parameter
slots; shading retains two bound-buffer variants with center/color parameter
slots. Shared queries independently cache two physics buffer variants. This
supports physics ping-pong and changing channel masks without rebuilding bindings.
Replacing a bound buffer prepares a new set. Aperture sums are lazy and retained
across frames; replacing the color buffer recreates only its bind group,
retaining scratch, parameters and pipelines. Inactive channels retain their previous contents.

Each ray or shading preparation creates one uniform buffer and one bind group.
Shared primitive/mesh query preparation counts follow the [Raycaster](raycaster.md)
contract. Accumulator initialization
creates a 16-byte-per-pixel sum buffer, one uniform, one bind group and two pipelines.
`accumulator_buffer_allocations` stays at two through color-buffer rebinding.
The counters exclude initial camera buffers/pipelines, command encoders, driver
upload staging and explicit readback. This is retained application-resource reuse,
not a claim that the entire dispatch is allocation-free. More than two rotating
trace input sets can cause cache churn.

Model/state and public replacement output buffers must have current GPU storage
on the camera's engine. Engine arguments are identity-checked; raw `Buffer`
provenance remains a caller precondition. Host-dirty inputs/output replacements
must be uploaded before dispatch. A changed output length is rejected; reconstruct
the camera to resize. Pixel capacity also obeys the shared query's 64-thread
one-dimensional dispatch limit; oversized batches are rejected before allocation,
not silently truncated. Low-level `CameraRaysKernel` / `CameraShadeKernel` prepared
encoding is also available: differing pending parameters require distinct slots,
and callers own buffer authority, valid indices/ranges and submission lifetime.
The legacy primitive-only `CameraTraceKernel` remains a low-level convenience API;
`TiledCamera` no longer uses it for visibility.

## Owned scenes and result revisions

```rust
let mut scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default())?;
camera.dispatch_scene(&engine, &scene, &state)?;
let id_scene = camera.channel_scene_stamp(CameraChannels::SHAPE_INDEX)?;
```

Owned scenes snapshot both geometry and per-shape color. Multiple cameras and
LiDAR/ray sensors share the same mesh BVHs; `mesh_build_count()` remains zero
for owned-scene queries. The ordinary `dispatch(model, state)` also uses shared
queries and supports raw TriMesh models, with legacy geometry comparisons and
per-camera BVH caching. Use `dispatch_scene` for Triangle/Heightfield conversion,
unsupported-shape diagnostics and revision identity.

After scene replacement, each written channel receives the new scene stamp;
disabled channels keep their earlier stamp and contents. Retain channel stamps
with pending readback/snapshot metadata before redispatch. Do not resolve an old
ID against a replacement scene. Stamps identify scene revisions, not sample times:
partial channel updates are not automatically a coherent multi-channel sample.
External writes to public output buffers are outside this metadata tracking.

## Heterogeneous camera batches

`CameraBatch` groups `CameraSpec` entries with identical complete
`TiledCameraConfig` values. Nonadjacent entries can share one `TiledCamera`;
different resolutions, intrinsics, lighting, range, masks or optics form separate
groups. Every entry keeps its own unique `u64` sensor ID and pose. Layouts remain
in input order, with a group-local camera index and pixel range. Output slices
contain exactly one image, not the whole group.

### Direct grouped encoding

`CameraBatch::output_pool(engine, config)` creates a `CameraBatchOutputPool`
with one independent leased pool per homogeneous group. `config.slots` applies
to every group; `max_total_bytes` covers **all groups together**:
`32 * aggregate_pixels * slots`. This excludes original camera buffers, ray
scratch, aperture storage and driver allocations. Empty managers need no output
storage and accept a zero-byte budget, but still require 1–64 slots.

```rust
use molla_math::Real;
use molla_sensors::SensorCommandBatch;

let mut outputs = cameras.output_pool(&engine, output_config)?;
let mut commands = SensorCommandBatch::new(&engine)?;
let Some(frame) = cameras.encode_scene(
    &mut commands, &engine, &scene, &state, &mut outputs, Some(simulation_time),
)? else {
    return Ok(()); // Output backpressure; nothing from this manager was encoded.
};
let front_job = frame.image_job(front_sensor_id, &operations)?;
let processed = processor.encode_pending(
    &mut commands, &[front_job], &mut processed_outputs, processing_stamp,
)?;
let submitted = commands.submit()?;
let (layout, group) = frame.image(front_sensor_id).unwrap();
let camera_frame = submitted.frame(&group.frame)?;
let depth = camera_frame.channel::<Real>(0)?;
let source = depth.readback_source(layout.pixels.clone())?;
```

All groups and downstream stages share **one submission**, without copying the
legacy outputs or downloading intermediate data. Each group consumes one of the
command batch's 64 output entries; downstream outputs need additional entries.
The manager checks the entire group's output capacity, scene/state provenance,
pool compatibility and slot availability before any group encodes. If a later
group is full, `None` leaves earlier groups unreserved and unmodified. Preflight
errors preserve any unrelated work already encoded in the command batch. A later
encoding failure poisons the whole batch, preventing partial submission.
Already queued pose uploads can still flush later; cancellation does not undo
queue writes or publish their results.

`EncodedCameraBatch` shares immutable image layouts through `Arc<[CameraImageLayout]>`
and stores up to 64 group receipts inline in an `ArrayVec`. It does not allocate
new layout/group vectors per encode. Layouts remain alive across reconfiguration
while old results or pools retain them. `image(id)` selects
the group and group-relative image range. Channels have the same typed order as
`EncodedCamera`. `image_job(id, operations)` selects color channel 1;
`annotation_job(id)` selects shape channel 4 and supplies the image dimensions.
Both reject disabled channels and unknown IDs. Their metadata preserves the full
sensor ID, shared sample ID, actual scene geometry revision and simulation time.
Group frames use sensor ID 0 because a group may contain multiple sensors; use
`readback_stamp(id)` for an individual image, and retain the batch stamp and full
group scene stamp beside asynchronous results. A scene geometry revision alone
does not identify the scene or its topology.

Successful encodes consume a sample ID, even if later cancelled; preflight
rejections/backpressure do not. `stats().encoded_frames` counts these successful
manager encodes, not GPU completion. Legacy `frames`/`submissions` are unchanged;
use the returned `SensorSubmission` for shared submission counters. Empty managers
return empty receipts with a new sample ID, but record no GPU work.

Producer, scene and state borrows last through submission/drop. Submitted frames
own their slots, remain readable after manager/pool drop and survive reset or
reconfiguration. The pool accepts only its original manager and exact image/group
layouts. Reconfiguring intrinsics/poses while preserving those layouts can reuse
it; changes to IDs, image order, dimensions, masks or group mapping require a new
pool. Pool reset does not reclaim retained frames. `available_slots()` polls
without waiting and returns the minimum available count over all groups; encoding
itself does not poll or wait. Known device loss rejects even empty manager pools.

### Legacy group submissions

```rust
use molla_sensors::{CameraBatch, CameraBatchLimits, CameraSpec, CameraChannels};

let specs = [
    CameraSpec { sensor_id: 10, config: front_config, pose: front_pose },
    CameraSpec { sensor_id: 20, config: rear_config, pose: rear_pose },
];
let mut cameras = CameraBatch::new(&engine, &specs, CameraBatchLimits::default())?;
let frame = cameras.dispatch_scene(&engine, &scene, &state)?;
let receipt = frame.stamp().clone();
let front = frame.image(10).expect("configured front camera");
let scene_stamp = front.scene_stamp(CameraChannels::DEPTH)?;
let stamp = front.readback_stamp(CameraChannels::DEPTH, Some(simulation_time))?;
let source = front.readback_source(CameraChannels::DEPTH)?;
let pending = ring.try_enqueue(&[source], stamp)?;
// Retain receipt and scene_stamp beside the ticket; None means ring backpressure.
```

Legacy dispatch is device-only and submits once per active group, not once per image
and not once for the entire heterogeneous batch. Each group retains its resources
and shares owned-scene BVHs. `NONE` masks contribute zero submissions; an empty
batch returns an empty frame with a new sample ID and no GPU work. Disabled
channels cannot be exposed as fresh data. `stats()` reports aggregate successful
frames, actual group submissions, failed frames and reconfigurations;
`group_stats()` and `query_stats()` expose the underlying resource/work counters.

`set_camera_pose(id, pose)` and `set_camera_look_at(id, ...)` affect only that
camera. `reconfigure(engine, specs)` supports changed IDs, order and settings.
It validates the entire plan before GPU allocation and builds replacement groups
before swapping. Invalid input leaves the previous layouts and receipts intact.
Reconfiguration rebuilds groups; it is not a warmed per-frame operation.
Old and new storage coexist during the transaction.

Legacy frames/images borrow the batch, preventing mutable redispatch while those views
are still used. Enqueue readback or capture a `DeviceSnapshotPool` copy before
releasing the borrow. A raw GPU consumer must be submitted before mutable reuse,
or consume a copied snapshot with its lease retained through GPU completion.
These are not direct leased producer buffers. Queued copies survive reconfiguration
and reset. `reset()` invalidates receipts without clearing outputs, changing
layouts or restarting sample IDs. `is_current(receipt)` checks batch ownership
and generation, not whether the receipt is the most recent sample: historical
copies in the same generation remain valid. Neither stamps nor borrows create
an interpolated simulation state; the caller supplies a coherent state across
all group submissions and a finite, nonnegative simulation timestamp.

A legacy group failure can occur after earlier groups submitted. No frame is returned;
all prior-generation receipts are invalidated and the attempted sample ID is
consumed. Cross-group execution is not atomic. Keep both the batch receipt and
full per-channel `SceneStamp` with external tickets: `ReadbackStamp` alone does
not contain batch ownership/generation or the owned scene's identity.

Default limits are 64 cameras, 16 groups and 4,000,000 aggregate pixels.
Configured limits allow at most 1,024 cameras and 64 groups. Checked dimensions,
per-group storage/dispatch capacity, poses, configurations and unique IDs are
validated before allocating camera storage. Aggregate pixels include every image,
even those with no enabled channels. In addition, the fixed-array budget above
defaults to 1 GiB **summed across all groups**, checked before constructing any
group's GPU buffers. `new_with_storage_limits` accepts an explicit
`TiledCameraLimits`; `storage_bytes()` reports the summed estimate, and
`reconfigure` preserves that cap. An empty batch has zero camera-array payload.
The cap does not include driver allocation or the transient old-plus-new
reconfiguration peak. Readback/snapshot pools have separate byte budgets.
Oversized groups are rejected, not automatically split.

## Tiled multi-camera × multi-world layout

In Newton's RL recipe, each parallel world owns its own camera and
all of them render together. molla handles that the same way: the
caller picks `num_cameras = num_worlds` at construction time, then
either calls `set_camera_pose(world_i, world_i_camera_pose)` each
frame, or — more often — pre-bakes the per-world camera offset into
the pose buffer once and updates only the world-root body each step.
The resulting depth / colour buffers are tensors of shape
`(num_worlds, H, W)` ready to feed straight into the policy.

## Scope (Stage 14)

Sphere, Plane, Box, Capsule, Cylinder, Cone, Ellipsoid and rigid TriMesh instances
use shared GPU visibility. Owned scenes additionally convert Triangle and
Heightfield geometry. Unsupported owned geometry is rejected by default or
explicitly reported with ReportAndSkip; legacy raw-model omissions remain those
of the Raycaster. Camera color/albedo ranges feed [GPU image processing](image_processing.md)
without host collection. Owned scenes provide top-level acceleration/refits and
stable application handles; see [scene snapshots](scene.md). Direct tiled and heterogeneous camera outputs support leased
ownership and pending image/annotation consumers; whole-pipeline allocation and
performance gates remain open.

Verification: `make test-gpu P=molla-sensors T=sensor_camera_device_gpu A='--test-threads=1'`
checks host-average parity, center geometry, masks, buffer reuse/replacement,
GPU-authoritative state, engine errors and the 1024-sample boundary. Run the
existing `sensor_tiled_camera_gpu` and `tiled_camera_optics_gpu` suites as well.
`sensor_camera_scene_gpu` checks moving/static meshes, primitive occlusion,
independent triangle/terrain references, revisioned materials, pending reads,
partial channels, strict max range, informational near and empty scenes.
`sensor_camera_batch_gpu` compares all enabled channels of heterogeneous groups
against independent cameras, including pose changes, masks, aperture settings,
range, reordered/resized reconfiguration, queued copies and engine identity.
`sensor_camera_manager_direct_gpu` checks one-submission camera/image/annotation
composition, exact legacy parity, full-request backpressure, cancellation,
reconfiguration, aggregate output budgets, empty/lost-device behavior, 1,024
camera IDs and the 64-group command limit. Its 24-frame/four-slot fixture retains
60 pool buffers across three groups, one ray preparation per group, four shading
preparations per group and four aperture preparations plus two aperture buffers
for the aperture-enabled group. Independently computed center distances and IDs
remain correct in historical images while camera poses change. These counters
are not a whole-pipeline allocation or performance measurement.
GPU-feature unit tests validate every grouping field, pre-allocation budgets,
partial-frame failure and identity exhaustion. `make test-sensors` includes these
host and explicitly ignored GPU lanes.

`bench_sensor_camera` includes `sensor_camera_batch_gpu` cases for 1, 4 and 16
cameras, homogeneous and heterogeneous layouts. `legacy_completed` submits each
active group separately; `direct_completed` encodes all groups once into leased
outputs. Both use the same owned scene, independent poses and configurations,
and wait for completed GPU work on every iteration. Heterogeneous groups use
four resolutions and one four-sample aperture configuration. One output slot per
group is warmed before direct measurements. The direct measurement includes
batch/result metadata and submission overhead; neither path downloads outputs.
These are completed-work comparisons, not queue-enqueue-only timings. Run:

```sh
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera -- sensor_camera_batch_gpu'
```

Record adapter, driver, competing GPU activity and thermal state before accepting
timings. `--test` exercises benchmark cases once and is only a smoke check, not
a performance result or an allocation measurement.

## Cross-references

- [`Sensors / Raycaster`](./raycaster.md) — same intersection logic,
  but caller-supplied rays instead of on-device generation.
- [`Geometry / Gaussian splats`](../geometry/gaussian_splat.md) — the
  3DGS family of differentiable renderers covered in
  [`Differentiable Physics / Gaussian splat renderer`](../autograd/splat.md);
  unrelated raster path but the rays it consumes match
  `camera_rays.wgsl`'s output format.
- [`Architecture / Design rules`](../architecture/design_rules.md)
  for the GPU-first contract these kernels follow.
