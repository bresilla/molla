# GPU point-cloud processing

`CloudProcessor` constructs clouds directly from device LiDAR hits or consumes
packed `CloudPoint` ranges. It applies a rigid transform, filters points and
compacts survivors in stable input order, without host point arrays, count
downloads or blocking waits inside `process`.

## Device pipeline

```rust
use molla_sensors::{CloudFilter, CloudInput, CloudJob, CloudProcessor, CloudProcessorConfig};

let mut processor = CloudProcessor::new(&engine, CloudProcessorConfig::default())?;
let scans = lidar.dispatch_scene(&engine, &scene, &state, &scan_configs)?;
let jobs = [CloudJob {
    input: CloudInput::from_lidar(&scans, 0)?,
    transform: Transform::IDENTITY,
    filter: CloudFilter {
        distance: Some((Vec3::ZERO, 30.0)),
        ..Default::default()
    },
    stamp: ReadbackStamp {
        sensor_id: scan_configs[0].sensor_id,
        sample_id,
        scene_revision: scene.stamp().geometry_revision,
        simulation_time: Some(simulation_time),
    },
}];
let cloud = processor.process(&jobs)?;
let pending_count = ring.try_enqueue(&[cloud.count_source()?], jobs[0].stamp)?;
```

The count stays on device unless explicitly collected. The checked source API's
name does not perform a download. After optional count collection, request only
`cloud.points_source(0..count)` for export. `summary_source(job_index)` returns
five `u32` values decoded by `CloudSummary::from_words`: compacted start, survivor
count, misses, invalid points and filtered points. Those categories partition the
job's original input count. Full scene identity remains caller-owned; retain the
original `SceneStamp` when resolving shape IDs later.

[Selected point-cloud export](./cloud_export.md) provides CPU-only full-record
binary PLY and legacy XYZ/RGB projection writers for explicitly collected
ranges. Full-record export preserves u64 IDs without a legacy scan-ID cast.

For another GPU consumer, capture output/count sources in `DeviceSnapshotPool`
and retain/track its leases. Another `CloudProcessor` can consume a packed output
source directly. Without host count collection, use the active input extent:
the compacted prefix is valid and its remaining tail is zeroed with the valid
flag unset. A subsequent processor skips this tail as misses. This is not yet
an indirect-dispatch path restricted to the GPU count; it scans the supplied
upper bound. No stale previous-frame points appear in that extent.

[`CloudGrid`](./cloud_grid.md) consumes the same packed ranges for device-only
spatial membership, counts, centroids and extrema. Its per-cell membership order
is unspecified; it does not reorder the packed input records themselves.

## Construction and identity

`CloudInput::from_lidar` selects one scan's point, normal, actual direction,
distance and signed shape-ID channels. The method supports terrestrial and aerial
patterns and all divergent sub-rays, not just first/last selection. It verifies
the selected range against active hits; processing checks channel types, lengths,
raster products, device identity and the job's sensor ID before submitting.

## Shared submissions and direct outputs

`output_pool(engine, config)` creates fixed-budget leased results for
`encode(batch, jobs, pool, stamp)` and `encode_pending(batch, jobs, pool, stamp)`.
Both return `None` on pool backpressure, before any copies, clears, uniform
writes or binding preparation. Otherwise `EncodedCloud` contains an opaque
frame receipt, input layouts and the aggregate input-count upper bound.

The two output channels are:

| Channel | Type | Active range |
|---|---|---|
| 0 | `CloudPoint` | `0..input_count`; compacted prefix followed by zero records |
| 1 | `u32` | `0..1 + 5 * job_count`; total count, then five summary words per job |

Per-slot storage is `112 * max_points + 4 * (1 + 5 * max_jobs)` bytes,
separately budgeted from processor scratch. The three passes write compacted
points and counts directly into the slot, not into legacy outputs followed by
a snapshot copy. Inputs still copy into retained working-point/raw-ray scratch.
Up to 128 output binding variants are cached; scratch, descriptors, uniforms
and pipelines are shared with the legacy path. Cached groups retain their
buffers until eviction or processor drop.

`PendingCloudJob` accepts packed `PendingCloudInput::Points` or five typed
`PendingCloudInput::Rays` sources. Each `PendingCloudSource` names an earlier
receipt, channel and element range **relative to that channel's active range**.
Ray sources must have equal lengths and match their raster. The five channel
types/order are `Vec3` point, `Vec3` normal, `Vec3` direction, `Real` distance,
and `i32` shape. For `EncodedRayQuery` these are channels `[4, 5, 1, 2, 3]`.
Different sources can reference different earlier frames in the same batch.

`PendingCloudInput::from_lidar(&encoded_batch, scan_index)` constructs these
views and the raster directly from `LidarBatch::encode_scene` output. It retains
the scan's independent full-width sensor ID, ray offset and terrestrial/aerial
layout; use the same sensor ID in the corresponding `PendingCloudJob` stamp.

Pending dependencies are deduplicated by receipt identity, including cloned
receipts. This preserves up to 1024 logical jobs without spending an input
reference per job. There are still at most 64 distinct dependencies and 64
total output frames per command batch. External `CloudJob` inputs instead
share the batch's 64-external-reference budget: a packed source counts once,
and a ray job counts five times, including zero-length source references.
Legacy `process` retains its existing configured job limit.

The processor and external input jobs remain borrowed until submit/drop.
Independent processors can encode different parameters into one batch; a
single processor cannot overwrite pending scratch or uniform data. Submitted
source leases remain occupied through GPU completion. A preflight error leaves
earlier valid work usable; errors after encoding starts poison the batch.
Cancellation publishes nothing, and reset does not reclaim retained frames.

Empty jobs still have zero summaries. An entirely empty input publishes fresh
zero counts with no kernels. A nonempty all-rejected input executes three
passes and publishes zero points with rejection counts. Chaining the entire
upper-bound range into another processor or grid treats the zero tail as misses;
it does **not** indirectly dispatch only the survivor count. Full-width IDs,
reserved attributes and unused NaN payloads are preserved in surviving packed
records. Ray construction sets IDs from the job stamp, as in `process`.

`encoded_frames`, copied bytes and kernel counters report encoded work,
including subsequently cancelled batches. `frames`/`submissions` continue to
count legacy `process` calls. These counters do not measure GPU completion or
prove the whole pipeline allocation-free; per-call metadata and driver work
remain outside the fixed-buffer counters.

The 112-byte `CloudPoint` format is identical on host and GPU:

| Field | Contents |
|---|---|
| `position` | XYZ and intensity |
| `normal` | XYZ and original ray distance |
| `direction` | Actual ray direction XYZ and optional scalar |
| `ids` | Full 64-bit sensor ID and sample/scan-sequence ID, each split into two u32 words |
| `grid` | Row, column, sub-ray and original source index |
| `extra` | Shape ID, packed RGBA8, flags and reserved word |
| `angles` | Source theta/phi and two reserved floats |

Flags are valid=1, color-present=2, scalar-present=4 and angles-present=8.
`sensor_id()` and `sample_id()` recover the full integer identities without float
conversion. Constructed LiDAR records use the job's `ReadbackStamp.sample_id` as
scan-sequence identity; packed input records preserve their existing IDs even
when multiple source scans are combined in one job. Job stamps/layouts describe
the submitted group and do not retag packed points.

Terrestrial rows/columns/sub-rays map back to the original grid; theta/phi describe
the source cell, while `direction` is the actual divergent sub-ray direction.
Aerial records use row zero, column=pulse and the original sub-ray index; they do
not fabricate terrestrial angles. The source index includes the original LiDAR
batch ray offset. Rigid transforms affect position, normal and direction only;
the acquisition distance, angles and source metadata remain unchanged.

LiDAR intensity defaults to 1 and color is absent. These are placeholders, not
measured reflectivity or a new material/intensity model. Packed points can supply
color, intensity and one designated scalar. The CPU `HitPoint` API's arbitrary
HashMap of named scalars is not mirrored as a GPU hash table: select the scalar
channel explicitly. CPU cloud utilities and filesystem writers remain unchanged.

## Filters and invalid data

All enabled filters are combined with AND after transformation:

- Distance: inclusive radius around a finite origin; zero radius is allowed.
- XYZ bounds: inclusive box, with either inside or outside retention.
- Intensity: inclusive minimum.
- Optional scalar: `<`, `<=`, `>`, `>=`, or absolute-difference `< 1e-6` equality.
  Points without the selected scalar are filtered out.

Input valid-flag absence (or a negative LiDAR shape ID) counts as a miss. Nonfinite
position/normal/direction/intensity/range, negative range, nonfinite present scalar
or angles, and nonfinite transformed coordinates count as invalid. Otherwise,
failing the selected predicates counts as filtered. These diagnostics are not
silently conflated. GPU computation is f32; input and transform values must be
representable and rotation must be normalized. Threshold behavior near f32
rounding boundaries can differ from a f64 CPU oracle.

Distance testing scales the displacement before taking its norm rather than
overflowing a squared norm. Angular interpolation computes its normalized fraction
before multiplication; large finite angular endpoints need not produce intermediate
overflow. Invalid job configurations fail as a whole before encoding: reversed
bounds, negative/nonfinite radii, nonfinite thresholds/times, mismatched ray
channels, invalid rotations and capacity overflow do not submit partial work.

## Ordering, ownership and resources

One submission copies inputs, clears active output/counts and runs three passes:

1. Parallel construction/transform/filter and a per-256-point exclusive rank.
2. Parallel prefix scan of block totals using contiguous thread-owned chunks.
3. Stable scatter and per-job range summaries.

Survivors retain job order and original order within each job. Atomics count
diagnostic categories only; they do not decide scatter order. Empty jobs can
appear anywhere. An entirely empty batch still submits a count clear, returning
fresh zero rather than the previous frame's count, with no compute passes.

Frames borrow the processor. Queue copies or capture snapshots before mutable
reuse; raw consumers must submit before reuse or retain copied snapshot leases
through completion. Snapshot input occupancy tokens remain held until the cloud
submission completes. Output views are not direct leased producer slots, and
there is no public high-level encode-only producer composition API yet.

Defaults are 65,536 aggregate input points, 64 jobs and 64 MiB of application
buffer memory. The maximum job count is 1,024; point capacity also obeys device
storage/dispatch limits. Checked budgets include eight retained buffers:

- Raw ray channels: 56 bytes per point.
- Working and compacted records: 112 bytes per point each.
- Local ranks: 8 bytes per point.
- Block totals/offsets: 8 bytes per 256-point block.
- Descriptors: 160 bytes per job.
- Counts: `4 * (1 + 5 * max_jobs)` bytes.
- Parameters: 16 bytes.

The total is `288*N + 8*ceil(N/256) + 180*J + 20` bytes, excluding host mirrors,
bounded per-frame metadata/leases, pipelines and driver staging. All buffers,
the bind group and three pipelines are retained across frames. Input copies and
output-tail clearing are real bandwidth costs; no allocation-free or speedup
claim follows from the retained-resource counters.

## Verification and benchmarks

```bash
make test-gpu P=molla-sensors T=sensor_cloud_gpu A='--test-threads=1'
make test-gpu P=molla-sensors T=sensor_cloud_direct_gpu A='--test-threads=1'
make test-sensors
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_cloud -- --test'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_cloud'
```

Tests compare transforms/filters/order against a CPU oracle, preserve full-width
IDs and attributes, verify terrestrial/aerial LiDAR grids, handle all-rejected
and empty batches, exercise pending snapshots/readback reuse and invalid data,
check scan boundaries including 65,537 points, and verify the first/boundary/last
compacted records of a 1,000,003-point batch. Host tests validate wire sizes,
summary decoding and exact byte/device budgets.

The benchmark covers transform-only, approximately half-kept and all-rejected
workloads at 1,024 / 65,536 / 1,000,003 points. Separate CPU, completed-GPU and
selected-prefix GPU-to-host lanes expose collection overhead. CPU timings use
packed records and f64 transforms without a GPU round trip; GPU timings include
input copies, validation, clearing and all compaction passes. Initial upload is
outside timed iterations. The export lane first collects the GPU count, then
copies only the surviving prefix. Smoke success is not a statistical performance
measurement or parity claim, and this matrix does not cover every filter mix.

Spatial binning/reductions, annotations and export adapters have separate APIs
and tests. Shared lifecycle/performance gates and viewer integration remain;
renderer-specific/Gearbox integration is not part of this API.
