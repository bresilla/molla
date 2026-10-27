# Direct leased sensor outputs

`SensorCommandBatch` separates producer encoding from submission. Frame-transform,
contact, IMU, owned-scene ray queries, batched LiDAR, tiled/heterogeneous cameras,
image processing, annotations, cloud construction/filtering and cloud grids
implement this path. This is distinct from `DeviceCommandBatch`,
which protects consumers of copied snapshots.

## Capture and compose

Create a pool once with `sensor.output_pool(&engine, config)`. It allocates one
separate storage buffer per channel per slot, including an internal reference
element for frame-transform sensors that have one. The pool accepts 1–64 slots and enforces `max_total_bytes`
before GPU allocation. Channel capacities and host element types are fixed.

```rust
use molla::compute::Engine;
use molla::sensors::{FrameTransformSensor, SensorCommandBatch, SensorFrame, SensorFramePool};
use molla::sensors::readback::ReadbackStamp;
use molla::sim::State;

fn capture(
    engine: &Engine, state: &State, sensor: &mut FrameTransformSensor,
    pool: &mut SensorFramePool, stamp: ReadbackStamp,
) -> molla::core::Result<Option<SensorFrame>> {
    let mut batch = SensorCommandBatch::new(engine)?;
    let Some(receipt) = sensor.encode(&mut batch, engine, state, pool, stamp)? else {
        return Ok(None);
    };
    let submitted = batch.submit()?;
    Ok(Some(submitted.frame(&receipt)?.clone()))
}
```

Encode additional **distinct** frame-transform, contact, IMU or ray-query producers into the same batch
before calling `submit`. All their clears and kernels execute in one queue
submission. A batch holds at most 64 outputs. An empty batch submits nothing
and returns `index: None`. A receipt is only an identity token: it has no
CPU-readable channels before submission and cannot resolve in a different batch.
Earlier outputs can feed later GPU work through the scoped pending-input APIs below.

The producer is mutably borrowed and its `State`, `Model` and/or `Contacts`
inputs are immutably borrowed until submission/drop. Safe caller code cannot
redispatch/reset that producer or mutate those inputs while commands remain
unsubmitted. Compile-fail tests cover these cases. This does not prevent external mutation through cloned raw
wgpu handles or independently submitted commands; those remain caller-owned
synchronization responsibilities.

## Input ownership

Initialize the state with the engine's **canonical Arc**, retaining its identity:

```rust
state.upload_to_device(engine.device().unwrap(), engine.queue().unwrap())?;
```

`Model::upload_to_device` and `Contacts::upload_to_device` likewise preserve
the canonical Arc when passed directly. IMU input poses, velocities, COMs,
world IDs and gravity, and nonempty contact input arrays/CSR, are all checked
before output reservation. Empty contacts use no input buffers.

`Buffer::ensure_device` records that Arc only when allocating a new mirror.
`Buffer::device_buffer_for` checks Arc identity plus the existing dirty-state
contract. Cloning the Arc is supported; wrapping a cloned raw device in a new
Arc is not the same ownership identity. Raw `&wgpu::Device` allocation remains
supported for legacy callers, but its provenance is unverified and the new
producer API rejects it. Re-ensuring an already allocated raw mirror cannot
retroactively certify ownership; initialize a fresh state with the canonical
Arc instead. Legacy frame `dispatch`/`update` still require the caller to supply
state from the correct device.

The cross-instance regression demonstrated why raw `wgpu::Device` equality is
not sufficient: this version's native backend compares local device IDs,
which may coincide across independently created instances. No native handle
casts, global registry or backend-specific identity assumptions are used.

## Channels and collection

| Producer | Channels in order | Active range |
|---|---|---|
| Frame transform | `Vec3`, `Quat`, `Vec3`, `Quat`: world position/rotation, relative position/rotation | Targets excluding the internal reference |
| IMU | `Vec3`, `Vec3`: accelerometer, gyroscope | All mounts; empty when priming |
| Contact | `u32`, `Real`, `Vec3`, `Vec3`: count, maximum depth, signed normal sum, centroid | All bodies, including zero-contact bodies |
| Owned-scene ray query | `Vec3`, `Vec3`, `Real`, `i32`, `Vec3`, `Vec3`: origin, unit direction, distance, shape, point, normal | `0..ray_count` |
| Batched LiDAR | Six ray channels as above, then `i32`, `i32`, `u32`: first ray, last ray, hit count | Ray count for channels 0–5; pulse count for channels 6–8 |
| Tiled camera | `Real`, `u32`, `Vec3`, `u32`, `u32`: depth, color, normal, albedo, shape ID | All pixels for enabled channels; empty for disabled channels |
| Image processing | `u32`, `Vec3`: packed pixels, per-stage gains | Aggregate pixels and operation slots; image layouts select subranges |
| Annotations | `u32`, `u32`, `DeviceBoundingBox`, `u32`: shapes, classes, boxes, counts | Aggregate image pixels, image × active-label box slots and four counters per image |
| Cloud grid | `CloudPoint`, `u32`, `CloudGridCell`, `u32`: copied points, indices, cells, diagnostics | Selected point count, selected cell count and six diagnostics |
| Cloud processing | `CloudPoint`, `u32`: compacted points, count/summary words | Input-count upper bound with zero tail; one total count plus five words per job |

Channels 0–3 are world position (`Vec3`), world rotation (`Quat`), relative
position (`Vec3`) and relative rotation (`Quat`). Their active ranges exclude
the internal reference element. `binding()` starts at byte zero and covers
the full-capacity buffer; GPU consumers must respect `active_range()`.
This avoids treating a 16-byte reference offset as a storage-aligned binding.
`readback_source(0..N)` instead uses indices **relative to the active range**;
it can feed the existing `ReadbackRing` or device processing adapters without
a snapshot copy. Only requested elements are collected.

General `SensorFramePool` channels may have zero logical capacity; these have
one-element backing allocations, empty active ranges and zero-byte readback sources.
Zero-width element types, invalid ranges, excess device capacities and budget
overflow are rejected. Frame-transform and IMU producers require nonempty target/mount lists.

## IMU history transactions

`ImuSensor::encode(batch, engine, model, state, dt, pool, stamp)` returns
`Result<Option<EncodedImuSample>>`. `None` means pool backpressure, with no
sample or history update. `Some(sample)` contains an opaque `frame` receipt and
a `priming` flag. Priming reserves a normal pool slot, encodes a velocity-history
copy, and publishes two empty active channels on submission; it does not invent
zero accelerometer readings. Later samples encode evaluation followed by the
history copy in the same ordered command stream.

The IMU stages its model identity while encoding. A bounded list of at most 64
borrowed submission actions commits the CPU primed/model/dirty state only after
queue submission. Actions require no per-action heap allocation. Dropping the
batch, a poisoned batch or a rejected submission cancels staged CPU changes;
unsubmitted history-copy commands are discarded with the encoder. A failed
producer validation before any commands is recoverable and does not poison
already valid work in the batch.

Pass elapsed `dt` since the **last submitted sample**, not since a cancelled or
backpressured attempt. The API does not infer time from the caller-provided
stamp. `reset()` cannot run while that IMU is borrowed by a pending batch. Once
released, it clears the filter so the next submitted sample primes again.
Pool reset alone changes frame generation, not the IMU filter. Legacy
`dispatch`/`update` share the same history, so alternating direct and legacy
calls preserves finite differences. Same-sized model replacement still needs
an explicit IMU reset; body remapping/count changes require recreation.

Contact frames have no temporal commit action. With `contacts.count == 0`,
encoding clears all four output channels without an aggregation dispatch.
A zero-body sensor publishes empty active channels with minimal backing
allocations. Contact readings remain geometric, not force measurements.

## Owned-scene ray queries

`Raycaster::encode_scene(batch, engine, scene, state, origins, directions,
max_distance, pool, stamp)` queries a `SensorScene` directly into leased slots.
It returns `None` on backpressure or an `EncodedRayQuery` containing a frame
receipt, active ray count and the actual `SceneStamp`. The caller's
`ReadbackStamp` remains separate metadata; do not substitute its arbitrary
`scene_revision` for the query's authoritative scene stamp when resolving IDs.

Primitive intersections, per-instance local BVH queries and closest-hit merges
run in the same command encoder, interleaved with other producers if requested.
Rigid meshes use the current GPU body poses; no vertex/BVH rebuild is needed
for pose changes. Converted heightfields and standalone triangles use the
owned scene's mesh path. Exact distance ties retain analytical hits before
meshes, then the earliest mesh instance. Hits at `max_distance` are misses.
Miss channels contain distance/shape `-1`, point zero and normal +Y.
Invalid included scene poses instead write distance/shape `-2`, point/normal zero
to every active ray. Channel availability is not numerical validity; see
[invalid owned-scene results](contracts.md#invalid-owned-scene-results).

This entry point accepts **CPU ray arrays**. It validates finite origins,
finite nonzero directions, matching counts, capacity and a positive finite f32
range. Directions are normalized robustly in f64 and packed into retained f32
upload arrays. Exactly `32 * ray_count` payload bytes are queued into the leased
origin/direction buffers; hit channels are produced directly by GPU kernels,
without copying legacy hit outputs. `input_upload_bytes` counts these payloads,
not uniform parameter writes. Scene/producer/state borrows last through
submit/drop. Body states referenced by nonempty queries require canonical
device ownership; static scenes can use an empty state.

Input uploads are queue writes and therefore run before encoded commands.
The private batch initialization mask excludes the two uploaded ray channels
from encoder clears, while the four hit channels are cleared normally.
Otherwise a clear would erase uploaded rays before tracing. Dropping a batch
discards query commands, not already queued upload writes; those may flush on
a later submission. Slot leases and ordered initialization make subsequent
reuse safe, and no cancelled result is published. Empty queries reserve a
frame with six empty active channels, without input uploads or kernel dispatches.

Direct queries retain up to 128 primitive variants, 128 query/merge pairs per
active mesh instance and 64 miss initializers. A changed scene ID/revision
clears direct geometry-dependent bindings. Old submitted frames remain
readable after replacement, but `resolve_body` on the replacement scene rejects
their old scene stamp. Retain `scene.identities()` together with the query's
actual `SceneStamp` to resolve historical dense shape/body indices into stable
handles; the immutable table survives replacement and scene destruction without
retaining GPU geometry. Explicit replacement remaps preserve selected handles,
while removed handles cannot alias new objects. See [scene identities](scene.md#stable-body-and-shape-handles).
Frames do not attach those CPU tables automatically. Existing raw-model
dispatch/cast paths remain unchanged.
`LidarBatch::encode_scene` shares the owned-query implementation with generated
GPU rays, followed by per-scan clipping and pulse selection; see
[LiDAR batches](lidar_batch.md). `TiledCamera::encode_scene` composes generated-ray
queries and aperture shading into leased camera outputs. `CameraBatch::encode_scene`
preflights every heterogeneous group before encoding any of them, then appends
all groups to the same submission. See [cameras](tiled_camera.md#heterogeneous-camera-batches)
for aggregate output budgets, per-image helpers and sample-identity rules.

## Lifetimes, reset and consumers

- A pending batch occupies its output slots. Dropping it discards commands and
  releases slots without publishing frames. A producer encoding error/panic
  after reservation poisons the batch; it cannot submit partial work.
- Submitted CPU frame clones and readback source leases prevent reuse. GPU
  completion callbacks retain only occupancy tokens, not device-owning frames.
- `available_slots()` polls without waiting. An exhausted pool returns `None`
  from producer encoding without recording commands or preparing bindings.
- `reset()` advances generation without reclaiming pending/retained slots.
  A batch encoded before reset can still publish a historical frame afterwards;
  `is_current()` returns false. Historical results remain readable while healthy.
- Frames remain usable after pool drop. Known device loss rejects encoding,
  submission, bindings and collection sources; reset cannot revive resources.

For raw GPU consumers of submitted direct frames, use
`SensorCommandBatch::encode_with_frames(&[&frame], |encoder| ...)` and submit
that batch. It retains up to 64 input references through GPU completion, sharing
that budget with direct image/annotation/cloud/grid processing's external device sources, even
if the caller drops all frame handles before submission. The callback must
only read those input bindings and use resources from the same device. Failed
or panicking consumer callbacks poison the batch. No CPU collection is required.
Arbitrary raw shader work in `encode_with_frames` is not included in producer kernel counters.

`encode_auxiliary(|encoder| ...)` encodes commands without sensor-frame inputs,
such as timestamp markers and query resolves. Use resources from this batch's device
and encode through the provided encoder; do not issue queue writes or submissions.
The hook does not retain sensor-frame leases: frame consumers still need
`encode_with_frames` or `encode_with_pending`. At most 128 auxiliary callbacks
are accepted per batch. Successful callbacks count in `auxiliary_encodes`, not
producer kernel dispatch counts. Callback error/panic poisons the batch and prevents
submission. Auxiliary-only batches submit normally; truly empty batches do not.

## Pre-submission GPU chains

`encode_with_pending(&[&receipt, ...], |encoder, inputs| ...)` appends GPU work
after producers already encoded in the same batch. `PendingSensorInputs` offers
`stamp(input)` and `channel::<T>(input, channel)`. Channel element types are checked
against the pool's host type, so `Real` selects the f32 GPU representation of the
host f64 type; asking for a different host type is rejected even if its width matches.

`PendingSensorChannel` exposes length, `active_range()`, a checked relative
`slice(range)` and a full-capacity read-only `binding()`. It has no host access,
readback source, submitted frame conversion or independently retained lease.
Views borrow only the callback scope; a compile-fail test prevents returning one
through a captured variable. Raw wgpu handles can still be cloned: callers must
not use that escape hatch to mutate inputs, collect unsubmitted data or launch
work outside this encoder. Bounds checking of views does not constrain arbitrary
shader indexing; the shader must honor the selected range.

`encode_from_pending` additionally reserves a new output slot, clears all of its
channels, then invokes a custom producer callback with the input views and output
buffers. Its receipt can immediately feed another pending stage. Pool layout,
budget and range validation are the same as for built-in direct producers.
For example, this custom stage selects the first world position:

```rust
use molla::math::Vec3;
use molla::sensors::{EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool};
use molla::sensors::readback::ReadbackStamp;

fn first_position(
    batch: &mut SensorCommandBatch<'_>, positions: &EncodedSensorFrame,
    output: &mut SensorFramePool, stamp: ReadbackStamp,
) -> molla::core::Result<Option<EncodedSensorFrame>> {
    batch.encode_from_pending(
        &[positions], output, &[SensorChannelSpec::of::<Vec3>(1)], &[0..1], stamp,
        |encoder, inputs, outputs| {
            let positions = inputs.channel::<Vec3>(0, 0)?;
            let first = positions.slice(0..1)?;
            encoder.copy_buffer_to_buffer(
                first.binding()?.buffer, first.active_range().start as u64 * 16,
                &outputs[0], 0, 16,
            );
            Ok(0)
        },
    )
}
```

Both callbacks return their kernel-dispatch count (`0` for copies only).
`kernel_dispatches` includes these caller-reported counts and rejects addition
overflow; `downstream_encodes` counts successful callbacks. Raw shader behavior
and truthful counts are caller responsibilities, not instrumentation inferred
from wgpu. Output buffers must be initialized through the supplied encoder,
not queue writes: queued writes execute before encoder clears. Custom uniforms
must use distinct storage for simultaneously pending parameter values, and
custom resources must not be mutated before submission.

Input lists must contain 1–64 receipts, all belonging to the same batch; duplicate
inputs are allowed. Lookup uses a fixed stack array of temporary Arc clones and
does not allocate an input Vec. Inputs are already held by the batch and do not
consume its separate submitted-frame input budget. Chains share the overall
64-output limit. An empty active channel is valid; an empty input list is not.

Invalid receipts, layouts and ranges are rejected before the callback and leave
earlier commands usable. Output backpressure returns `None` without callback,
clears or new output. Once a callback begins, its error or panic poisons the whole
batch, including earlier producers; it cannot submit partial work. Dropping the
batch releases all pending slots, even while receipt tokens survive. Resetting
a pool does not invalidate already encoded dependencies; their eventual frames
are historical. Known device loss rejects further encoding and submission.

The GPU fixture composes frame-transform and ray-query producers, a custom shader
reading `Vec3`/`Quat`/`Real`/`i32`, a range-copy stage and a final consumer in one
submission, then checks the outputs against independent arithmetic. It covers
nonzero storage ranges and 65 elements across a workgroup boundary. These APIs
are the custom-composition foundation. `ImageProcessor::encode_pending`,
`AnnotationProcessor::encode_pending`, `CloudProcessor::encode_pending` and `CloudGrid::encode_pending` use them for
built-in chains; see [GPU image processing](image_processing.md),
[annotations](annotation.md), [cloud processing](cloud_processing.md) and [cloud grids](cloud_grid.md).
`PendingCloudInput::from_lidar` connects a direct LiDAR scan to cloud processing.
Tiled-camera pending color and shape channels feed image and annotation jobs.
`EncodedCameraBatch::image_job` and `annotation_job` select heterogeneous image
slices with their own IDs and timestamps. Each homogeneous group consumes one
output entry, so a full 64-group manager cannot add a downstream output in that
same command batch. Exceeding the shared limit rejects the manager before
reserving any of its outputs, without poisoning previously encoded valid work.

## Resource evidence and remaining work

Direct production writes leased buffers without the old snapshot D2D copy.
Pool allocation counters remain fixed. Each auxiliary sensor retains a separate
bounded cache of 128 direct binding variants (up to 64 slots × two pose buffers)
and two legacy variants; more distinct layouts/inputs evict least-recent entries.
Bindings include one retained uniform buffer per variant. Warm tests cycle
four slots and two pose buffers for 32 submitted frames with only eight binding
preparations, 16 pool output buffers and zero sensor readback bytes.
Retained bindings also retain their referenced GPU buffers until eviction or
sensor drop; dropping a pool alone does not evict the sensor's binding cache.
The corresponding IMU/contact warm test submits 24 paired measurements after
priming with eight preparations per producer, eight IMU buffers and 16 contact
buffers. Neither sensor performs host collection or waits. A mixed 70-mount
test composes two IMUs with different timesteps, contacts and frame transforms
in one submission, matching legacy results and an independent central-mount
acceleration/gyro check.
The direct ray warm fixture covers four slots and two body-pose buffers over
24 variable-count queries with two shared-mesh instances. It stabilizes at
eight primitive preparations, 12 mesh preparations and 12 merge preparations,
with 24 pool buffers and no sensor collection or BVH rebuilds. A mixed-query
fixture composes two raycasters and frame transforms in one submission;
terrain/triangle tests use independent plane equations for distances/normals.

Batch frame/input/source-lease/temporal-commit collections now use inline
capacity-64 storage, rather than allocating four vectors when constructing each
batch. Completion tracking stores up to 128 occupancy tokens inline (64 outputs
plus 64 inputs). The wgpu callback still owns an allocated closure, now with a
larger fixed payload. This trades larger batch/callback objects for fewer heap
allocations; it is not by itself a latency improvement. Bounds, cancellation,
temporal rollback and completion ownership remain unchanged.

`SensorSubmission::into_frames_iter()` consumes frames in encoding order without
allocating a result vector. The existing `into_frames()` convenience API still
returns a `Vec` and now performs that explicit allocation during conversion.
Neither transfers or downloads GPU channel contents.

### Calling-thread allocation audit

`sensor_allocation_gpu` supplies a scoped, thread-local Rust allocator probe.
It records allocations, reallocations and requested bytes separately for batch
creation, camera encoding, image processing, annotation processing, submission,
completion and result release. It counts Rust dependency allocations on the
calling thread, including wgpu; it does not count native driver allocation,
GPU memory, other threads, peak live memory or selected-host-readback costs.
Requested bytes are cumulative allocation/reallocation requests, not retained
bytes. Counter self-tests cover zeroed allocation, reallocation and unwind reset.

The debug-profile fixture uses four homogeneous 9×7 cameras, one sphere, identity
image processing and one annotation label. It warms eight frames, then reports
min/max over sixteen completed frames with one output slot per producer and no
host image readback. On RTX 4080/Vulkan, before/after the inline-storage/shared-layout/
retained-pose changes:

| Phase | Allocations before → after | Requested bytes before → after |
|---|---|---|
| Batch creation | 9 → 5 | 4,614 → 1,030 |
| Camera encode | 26 → 23 | 5,524 → 5,156 |
| Image encode | 2 → 2 | 4,696 → 4,696 |
| Annotation encode | 18 → 18 | 11,266 → 11,266 |
| Submission | 98 → 97 | 16,345 → 18,337 |
| Completion/release | 0 → 0 | 0 → 0 |
| Total | **153 → 145** | **42,445 → 40,485** |

Reallocations remain 21 per frame in this fixture. These counts were constant
across all sixteen samples in the recorded runs; they are not portable ABI or
driver guarantees. The final counts also reproduced in the release-profile
audit; the before/after comparison above used debug builds. Submission's larger callback payload increases its allocated
bytes even as total allocated bytes decrease. Logs and validation are recorded
in `SENSORS_PROGRESS.md`.

```sh
make test-gpu P=molla-sensors T=sensor_allocation_gpu A='--test-threads=1 --nocapture'
```

The suite additionally tests 64 outputs plus 64 retained inputs together, inline
result iteration with zero measured allocations, completion lease release, and
retained upload scratch conversion/authority for scalar/vector/transform values.
The audit reports current costs; it is deliberately **not** a passing claim for
zero heap allocation per frame (see the attribution below).

### Warm-frame resource retention and attribution

`sensor_resource_retention_gpu` (feature `resource-counters`, which enables
wgpu's internal object counters) drains the device and snapshots live buffers,
buffer memory, textures, texture views, bind groups and layouts, compute
pipelines, pipeline layouts, shader modules, query sets, samplers and command
encoders after four warm frames, then asserts all of them unchanged after 32
more. Three chains are covered, each in one submission per frame:

| Chain | Live buffers / bind groups (cumulative) |
|---|---|
| Four 9×7 cameras → image processing → annotation | 83 / 11 |
| 16-ray `Raycaster` + two 8×32 `LidarBatch` scans → `CloudProcessor` → `CloudGrid` | 167 / 32 |
| IMU + contacts + frame transforms over eight mounts | 230 / 42 |

A second test installs a backtrace-attributing global allocator and assigns
each warm-frame allocation to the first non-plumbing crate on its stack
(generic `<T as trait_crate::Trait>` frames count as the trait's crate):

| Chain | Allocations per warm frame | Molla-owned before → after | wgpu-owned |
|---|---|---|---|
| Camera/image/annotation | 188 → 182 | 8 → 2 | 180 |
| Ray/LiDAR/cloud/grid | 299 → 294 | 7 → 2 | 292 |
| IMU/contact/frame | 113 → 112 | 1 → 0 | 112 |

The reduction came from a process-unique `BatchId` in place of a per-batch
`Arc` owner token and from retained `PlanScratch` planning storage in the
annotation, image and cloud processors. The remaining Molla allocations are the
per-frame layout and scan metadata returned to the caller inside receipts; the
rest is wgpu command submission. Counts are exact and stable across frames on
the recorded RTX 4080/Vulkan runs, not portable guarantees.

```sh
make test-sensor-resources A=--nocapture
```

## Lifecycle audit

Every producer and pool was checked against seven lifecycle properties:

- **P1:** bounded in-flight work with explicit backpressure;
- **P2:** no reuse before GPU completion;
- **P3:** no reuse before CPU release;
- **P4:** foreign engine or device rejection;
- **P5:** reset/generation invalidation;
- **P6:** cancellation/poison;
- **P7:** known device loss.

Producers that write into a `SensorFramePool` inherit P2/P3 from that pool.

| Producer / pool | Lifecycle tests |
|---|---|
| `SensorFramePool`, `SensorCommandBatch` | `device_results` unit tests (P1–P7), `released_frames_wait_for_gpu_completion_and_poll_failure_is_terminal` (strict P2, terminal poll failure) |
| `ReadbackRing` | `sensor_readback_gpu`, `cancelled_slots_are_reclaimed_only_by_poll_releases_after_mapping`, `sensor_device_loss_gpu` |
| `DeviceSnapshotPool` | `sensor_snapshot_gpu`, `released_snapshots_wait_for_copy_completion_and_poll_failure_is_terminal` |
| IMU / contact / frame transforms | `sensor_aux_direct_gpu` including `contact_backpressure_foreign_pools_and_device_loss_are_explicit`, `sensor_direct_outputs_gpu` |
| `Raycaster`, `LidarBatch` | `sensor_ray_direct_gpu`, `sensor_scene_rebuild_gpu`, `sensor_lidar_direct_gpu` (foreign pools and batches) |
| `TiledCamera`, `CameraBatch` | `sensor_camera_direct_gpu`, `sensor_camera_manager_direct_gpu` (foreign pools/batches, device loss with groups pending) |
| Image / annotation / cloud / grid | `sensor_image_direct_gpu`, `sensor_annotation_direct_gpu`, `sensor_cloud_direct_gpu` (reset invalidates held frames), `sensor_grid_direct_gpu` |

The audit found one defect. After a terminal poll failure, `DeviceSnapshotPool`
still accepted `reset()`, and `is_current()` could still return true, because
both checked only device loss. Both now share the pool's terminal-state check,
matching `SensorFramePool`. Strict P2 is tested without polling: wgpu runs
completion callbacks only inside a later poll or submit, so a submitted but
unobserved slot must read as occupied. Gearbox integration is not part of this change.

## Processor construction budgets

`ImageProcessor`, `AnnotationProcessor`, `CloudProcessor` and `CloudGrid` retain
host mirrors as well as device arrays. Device-only dispatch does not mean these
host mirrors are absent. Their existing `max_buffer_bytes` configs cap device
buffer payloads; `new` additionally applies a default 1 GiB combined construction
payload cap. `new_with_storage_limits` overrides that cap explicitly.

`ProcessorStorageUsage::for_image`, `for_annotation`, `for_cloud` and `for_grid`
validate configurations/device limits and estimate payloads without allocating
the arrays. `storage_usage()` reports the same estimate on a constructed processor.
The estimate sums retained host arrays, device arrays including kernel scratch/
uniforms, and the largest sequential upload-packing array. This conservatively
counts all retained arrays alongside that temporary packing; it is not a measured
process peak. Driver staging, allocator overhead, pipelines/bindings, per-frame
metadata, caller-owned inputs and output/readback pools are excluded.

```rust
use molla_sensors::{
    CloudProcessor, CloudProcessorConfig, ProcessorStorageLimits, ProcessorStorageUsage,
};

let config = CloudProcessorConfig::default();
let usage = ProcessorStorageUsage::for_cloud(config, &engine.device().unwrap().limits())?;
let processor = CloudProcessor::new_with_storage_limits(
    &engine,
    config,
    ProcessorStorageLimits { max_construction_bytes: usage.construction_bytes },
)?;
assert_eq!(processor.storage_usage(), usage);
```

Both device and combined payload checks run before host arrays and GPU buffers
are allocated. Host arrays and upload conversion storage reserve fallibly; dirty
host/device authority assertions are unchanged. Four GPU fixtures compare the
retained host arrays and largest packing payload with actual mirrored buffers.
Kernel-only device sizes are additionally source-audited and covered by the
existing device-budget formulas; this does not measure allocator or driver peaks.

The calling-thread preflight regression uses 65,537 pixels/points and a cap that
fits device buffers but not host mirrors/packing. Each rejected constructor
requested 116 bytes for its error message. Temporarily moving image preflight
after allocation makes the unchanged regression fail at 1,509,413 requested bytes.
These checks are separate from warm-frame zero-allocation and performance gates.
