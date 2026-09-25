# Device-resident LiDAR batches

`LidarBatch` generates terrestrial grids and aerial Halton-distributed pulses
on GPU, queries the scene, and selects nearest/farthest hits among each pulse's
divergent sub-rays. No CPU ray vectors, intermediate downloads, or host hit
reductions occur in this API. The legacy `Lidar::cast` and `AerialLidar::cast`
retain their CPU layouts and synchronous return types.

Owned-scene queries preserve invalid-pose distance/shape `-2` through range
clipping. Invalid pulses have first/last `-2` and zero hit count, distinct from
ordinary all-miss pulses (`-1`). Check distance or index, not hit count alone.
See [query validity](contracts.md#invalid-owned-scene-results); raw-model calls
still require valid finite rigid poses from the caller.

## Storage budgets

Construction checks a default 1 GiB combined retained host/device array budget
before allocating any arrays. `LidarBatchStorage::for_capacity(max_scans, max_rays)`
estimates the shared ray arrays, three return-index/count arrays on both host and
device, device descriptors, and reserved host descriptor/range capacity. Return
arrays are sized to `max_rays`, even when several sub-rays share each pulse.
Unused scan capacity still counts; `storage_usage()` reports the fixed estimate.

`LidarBatch::new_with_limits` accepts `LidarBatchLimits` with
`max_storage_bytes` and `ray_limits`. The shared ray-array sub-budget is checked
independently and also defaults to 1 GiB. Existing `new_with_ray_limits` overrides
only that sub-budget; the default combined cap still applies. Individual device
buffer, binding and dispatch limits remain enforced even with larger byte caps.
Scan capacity is 1–1024 and ray capacity is 1–`i32::MAX`, further restricted by
the device and storage limits.

These budgets exclude scene/query acceleration, uniforms/bindings, driver/upload
staging, per-result copied scan metadata, output leases and readback pools. They
are fixed array-payload limits, not process peak-memory limits. See
[ray-array budgets](raycaster.md#ray-array-storage-budgets) for the shared component.

### Legacy CPU layouts

`Lidar`/`AerialLidar` retain the ray-only constructor budget through their existing
`new_with_limits`. Their synchronous `cast` additionally checks a default 1 GiB
per-call CPU layout budget before expanding scan metadata into vectors.
`cast_with_layout_limits(engine, model, state, scan, LidarLayoutLimits)` overrides
that per-call cap without changing retained ray capacity or its constructor cap.

`scan.layout_storage_bytes()` validates metadata and estimates the layout payload
without expanding it. For terrestrial scans it is
`2 * ray_count * size_of::<Vec3>()`; aerial scans additionally retain two vectors
per pulse in the returned `AerialScan`, so it is
`2 * (ray_count + pulse_count) * size_of::<Vec3>()`. Both paths use checked counts,
reject requests beyond the sensor's ray capacity, and reserve vector storage
fallibly before filling it. Allocator-reported reservation errors return `Err`;
this does not guarantee recovery from operating-system memory exhaustion.
Terrestrial metadata rejects non-finite origins and angular intervals whose
endpoint difference overflows, even when both endpoints individually are finite.

The CPU layout limit excludes collected `RayHit` records, readback staging,
retained ray arrays and scene storage. It does not apply to device-generated
`LidarBatch` layouts, which allocate no CPU ray vectors. Sampling order, cone
sampling and Halton sequences are unchanged. `sensor_capacity_gpu` tests exact
and under-budget calls against normal synchronous results; pure layout tests
reject extreme counts before expansion. `make test-sensors` also runs the actual
batched-array payload comparison and shared-ray storage tests.

## Independent scans

```rust
use molla::sensors::{LidarBatch, LidarPattern, LidarScan, ScanMetadata};
use molla::math::Vec3;

let mut pipeline = LidarBatch::new(&engine, 16, 100_000)?;
let front = ScanMetadata::new(Vec3::new(0.0, 3.0, 1.0), 64, 128);
let rear = ScanMetadata::new(Vec3::new(0.0, 4.0, -1.0), 32, 64);
let requests = [
    LidarScan { sensor_id: 10, pattern: LidarPattern::Terrestrial(front) },
    LidarScan { sensor_id: 20, pattern: LidarPattern::Terrestrial(rear) },
];
let batch = pipeline.dispatch(&engine, &model, &state, &requests)?;
```

Each descriptor carries its own origin/frame, grid or footprint, divergence,
sub-ray count, maximum distance and caller ID. Terrestrial and aerial scans can
share a batch. Metadata is world-space: update each descriptor from its mount's
current pose before dispatch. The API does not read mount transforms or generate
simulation timestamps itself.

The query uses the largest batch range, then a GPU pass clips every sub-ray to
its own scan's range. A distant scan cannot extend a nearby scan's returns.
`sensor_id` is preserved verbatim; duplicate IDs are not deduplicated. The
descriptor ordinal and `LidarScanRange` distinguish individual requests.

## Layout and return selection

`batch.scans` holds the original descriptors and their global ray/pulse offsets
and counts. Scans occupy contiguous slices in input order. Terrestrial pulses
use `row * n_phi + col`, aerial pulses use Halton index `pulse + 1`. Sub-rays are
contiguous within each pulse, with the central ray first. Zero `subray_count`
means one ray, matching the legacy APIs.

`batch.hits` contains GPU origins/directions and distance/shape/point/normal
channels. Only `0..batch.hits.ray_count` is active. Directions are normalized;
distance is in meters. Misses use distance -1, shape -1, point zero, normal +Y.
Each scan retains the raycaster's supported geometry and shape identity rules.

`first_ray` and `last_ray` contain global ray indices, not copied hit records;
index the hit channels to consume them. Both are -1 for an all-miss pulse.
`hit_count` counts valid sub-rays per pulse. Only `0..batch.pulse_count` is active.
Nearest-distance ties select the first sub-ray; farthest ties select the last,
matching the legacy `min_by`/`max_by` selectors. This is **not** multiple physical
intersections along a single beam, material transmission, or intensity modeling.

## GPU consumption and collection

### Direct leased output and shared submissions

`LidarBatch::output_pool(engine, config)` allocates bounded reusable results for
`encode_scene(batch, engine, scene, state, scans, pool, stamp)`. The latter
returns `None` on backpressure, without descriptor uploads or GPU work, or an
`EncodedLidarBatch` containing an opaque receipt, ray/pulse counts, owned scan
ranges and the actual `SceneStamp`. The caller's frame stamp is separate metadata.

All generation, queries, nearest-hit merges, per-scan clipping and first/last
selection write directly into the leased slot. No legacy output copy, CPU ray
array or intermediate readback is involved. Each nonempty scan list uploads
128 bytes per descriptor when it produces rays; uniforms are excluded from
`SensorCommandBatch::input_upload_bytes`.

| Channel | Type | Active range |
|---|---|---|
| 0, 1 | `Vec3` origin, unit direction | `0..ray_count` |
| 2, 3 | `Real` distance, `i32` shape | `0..ray_count` |
| 4, 5 | `Vec3` point, normal | `0..ray_count` |
| 6, 7, 8 | `i32` first, `i32` last, `u32` hit count | `0..pulse_count` |

A slot reserves `84 * max_rays` bytes, including worst-case pulse capacity;
pool slots and their total byte budget are separate from legacy/scratch storage.
Up to 128 generation/selection output-binding variants are cached, reusing the
original descriptors, uniform and pipelines. Query caches retain their existing
per-output/per-pose limits. Cached groups retain buffers until eviction/drop.

Multiple independent `LidarBatch` producers and downstream cloud processors can
share one `SensorCommandBatch::submit`. `PendingCloudInput::from_lidar(&encoded,
scan_index)` selects the five typed ray channels and constructs the raster for a
`PendingCloudJob`; the job stamp must use that scan's sensor ID. All divergent
sub-rays are included, not only first/last selections. The normal cloud filters
discard misses and preserve full-width scan/sample IDs.

The LiDAR producer, scene and body state remain borrowed through submit/drop.
Nonempty queries of body-bearing scenes require canonical engine-owned current
GPU body poses. Empty input reserves nine empty channels and clears the slot
without upload or kernels. Cancellation publishes no result; reset does not
reclaim retained frames, and scene replacement does not alter older results.
Known device loss rejects pending work. The usual pending-receipt/custom-stage
rules apply; raw cloned GPU handles remain caller synchronization responsibilities.

`encoded_batches` counts successful encodes, including cancelled ones;
`batches` and `submissions` retain their legacy meaning. Descriptor upload bytes
include both APIs. Fixed binding/pool counters do not prove zero allocations:
owned result metadata, command encoders, staging and driver work remain outside
those counters.

### Legacy device views

Bind each typed channel's `device_buffer()` to a downstream compute pass, or
enqueue chosen active ranges in a shared `ReadbackRing`. For example, collecting
only first/last/count requires `12 * pulse_count` bytes; collecting all hit
channels requires `40 * ray_count` bytes. Origins and directions are an optional
additional `32 * ray_count` bytes. Round staging budgets up to multiples of eight.

```rust
let ticket = collector.try_enqueue(&[
    batch.first_ray.readback_source(0..batch.pulse_count)?,
    batch.last_ray.readback_source(0..batch.pulse_count)?,
    batch.hit_count.readback_source(0..batch.pulse_count)?,
], stamp)?;
```

`None` means collector backpressure, not a dropped accepted request. Use caller
simulation time and scene revision in `stamp`; the pipeline does not fabricate
them. Submit downstream work or enqueue snapshots on the same engine **before**
the next dispatch. Enqueued copies survive output reuse. Merely recording an
unsubmitted GPU command buffer does not reserve the old batch. See
[asynchronous readback](./readback.md) for cancellation and failure behavior.

`stats()` reports successful nonempty batches, queue submissions and descriptor upload bytes
(`128 * scan_count` per batch). `readback_stats()` stays zero: collection is
external. Empty batches neither dispatch nor upload descriptors. The descriptor
vectors, GPU layouts, binding groups, uniforms and selection buffers are retained.
Generation, the complete primitive/mesh query, nearest-hit merges and pulse
selection now share **one queue submission per nonempty batch**, rather than
one submission per pipeline stage/mesh. Empty batches submit nothing.

`query_stats()` exposes retained query preparations; its direct-query submission
count stays zero because `LidarBatch` submits the combined command buffer.
Two input-buffer variants are cached per primitive pass/mesh instance, so a
fixed scene with ping-pong physics states stops allocating query uniforms and
bind groups after warm-up. Changing range, active ray count or scan metadata
does not rebuild those bindings. See [Raycaster](raycaster.md) for counter scope,
buffer replacement and cache limits; command encoders and internal upload staging
are not eliminated by this reuse.

## Bounds and current limitations

- Capacity is fixed at construction: 1–1024 scan descriptors, positive ray
  capacity within device storage/dispatch limits, at most 1024 sub-rays per pulse.
  Overflow/excess requests return errors before ray generation; there is no
  silent truncation or automatic chunking.
- Frame axes must be finite, nonzero and nonparallel. Origins, range, extents
  and angular bounds must remain finite in f32; ranges must remain positive.
  GPU layout uses f32, not bit-identical f64 trigonometry. Numeric tolerances
  depend on scene scale and distance from visibility boundaries.
- `dispatch_scene` accepts an [owned sensor scene](./scene.md): shared local
  BVHs, moving rigid meshes, triangle/terrain conversion and revision stamps.
  Replace that snapshot explicitly on geometry changes. The raw-model path
  compares mesh data to detect edits but still skips unsupported geometry;
  do not interpret those misses as free space. Top-level acceleration remains
  unfinished in both paths.
- Beam exit diameter remains metadata only. Noise, motion distortion, physical
  multiple returns and force sensing are separate deferred capabilities.
- A single largest-range query can do unnecessary work for shorter scans.
  Top-level acceleration, camera participation in the shared batch and full
  allocation/performance acceptance remain incomplete.

## Verification

```sh
make test-gpu P=molla-sensors T=lidar_pipeline_gpu A='--test-threads=1'
make test-gpu P=molla-sensors T=sensor_lidar_direct_gpu A='--test-threads=1'
make test-sensors
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_lidar'
```

The benchmark compares legacy CPU-layout host scans, GPU-layout host scans, and
GPU-layout scans with explicit device completion at 1k/10k/100k/1M rays on a
static plane. It includes warm resource reuse, not cold construction, and is
not a pure GPU timestamp measurement or a mesh-rich simulator benchmark.
