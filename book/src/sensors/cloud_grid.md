# GPU spatial bins and reductions

`CloudGrid` consumes a checked device range of packed `CloudPoint` records. It
groups points into a regular 3D grid and computes membership, counts, centroids,
mean intensity, position bounds and intensity extrema without host collection.

```rust
use molla_sensors::{CloudGrid, CloudGridConfig, CloudGridSpec};

let mut grid = CloudGrid::new(&engine, CloudGridConfig::default())?;
let cells = grid.process(
    cloud.points_source(0..cloud.input_count())?,
    CloudGridSpec {
        minimum: [-10.0, -10.0, -2.0],
        maximum: [10.0, 10.0, 4.0],
        dimensions: [40, 40, 12],
    },
    stamp,
)?;
let pending = ring.try_enqueue(
    &[cells.cells_source(0..cells.cell_count())?, cells.diagnostics_source()?],
    cells.stamp,
)?;
```

`process` submits one command buffer and never waits or downloads. Its five
compute passes classify, prefix cell counts, prefix block counts, scatter input
indices and reduce cells. Empty input submits only fresh summary/diagnostic
clears. `ReadbackSource` creation itself does not collect data. The example
requests only summaries and diagnostics, not the cloud or membership array.

## Membership and identity

- Bounds are half-open: minimum included, maximum excluded. Dimensions are
  strictly positive. Cells flatten as `(z * ny + y) * nx + x`.
- `CloudGridCell` is a 64-byte record. `header` contains index-array start,
  member count, numeric flags and a reserved zero. `mean`, `minimum`, `maximum`
  each contain XYZ plus intensity. Empty-cell vectors are zero.
- The membership array contains indices relative to the selected input range,
  not the original buffer offset. `points_source` exposes the matching packed
  input copy, including unchanged full-width sensor/sample IDs and attributes.
- Only the accepted prefix of `indices_source` is valid. Its remaining active
  input extent is `u32::MAX`, never stale earlier-frame membership. Per-cell
  index order is unspecified because scatter uses atomic cursors. This does
  not change the stable ordering provided by upstream `CloudProcessor`.
- There is **no per-cell point cap**. The global membership allocation holds
  the entire configured input capacity. A million accepted points in one cell
  keep a million indices; they are not truncated to a fixed bucket size.
- Legacy `process` frames borrow the grid, so outputs cannot be overwritten through safe API
  reuse while a frame is borrowed. Enqueued readback copies retain their
  version. Use copied `DeviceSnapshotPool` channels for independently retained
  GPU versions. Snapshot inputs are held through submission completion even
  after public snapshot drop or pool reset.

## Validity, bounds and numerical behavior

`CloudGridDiagnostics::from_words` decodes six `u32` values: accepted, misses,
invalid, outside, index overflow and numeric-overflow cells. The first four
partition the input count. Classification precedence is:

1. Unset valid flag: miss.
2. Nonfinite position XYZ or intensity: invalid.
3. Position outside half-open bounds: outside.
4. Otherwise: accepted.

Normals, ray directions, angles and optional scalar fields do not participate
in these reductions and are preserved without validation. Use `CloudProcessor`
first when validation of those attributes is required. Its unused compacted
tail has validity unset and can be passed directly as an upper bound; grid
processing counts that tail as misses. This is not indirect dispatch.

The grid API rejects reversed or degenerate bounds, dimension overflow,
nonfinite spans and cell widths smaller than the smallest positive normal
f32. Unlike the legacy CPU convenience `bin_into_grid`, it does not reorder
bounds or replace zero divisions with one. Classification uses f32 arithmetic;
points extremely close to an internal cell plane may differ from an f64 CPU
classification. Upper-bound exclusion is checked before the final cell clamp.

Means use parallel f32 reductions. Atomic membership order can change their
rounding, so they are not bitwise deterministic. Very large finite values can
overflow during reduction even when their exact mean is representable. A
nonfinite result sets cell flag bit 0, zeros its entire mean and increments
`numeric_overflow_cells`; its finite extrema and membership remain available.
Do not interpret a flagged zero mean as a physical measurement. No universal
accuracy claim is made for extreme cancellation or large dynamic ranges.

`index_overflow` reports a defensive scatter-capacity failure. It must remain
zero for validated input; a nonzero value means consumers must reject the
frame rather than silently use incomplete membership. Capacity errors are
normally rejected before submission, not clamped. Foreign devices, wrong
record types and invalid/negative simulation timestamps are also rejected.

## Direct leased output and pending inputs

`CloudGrid::output_pool(&engine, config)` creates a fixed-budget pool with four
channels in this order:

| Channel | Type | Active range |
|---|---|---|
| 0 | `CloudPoint` | Selected input records, copied byte-for-byte |
| 1 | `u32` | Input-count extent; accepted prefix followed by `u32::MAX` |
| 2 | `CloudGridCell` | Product of the selected grid dimensions |
| 3 | `u32` | Six diagnostics |

`encode(batch, source, spec, pool, stamp)` accepts an existing checked device
source and appends the copy and five grid passes to a `SensorCommandBatch`.
`encode_pending(batch, input, spec, pool, stamp)` accepts `PendingCloudGridInput`
with an earlier receipt, a `CloudPoint` channel index, and a point range relative
to that channel's active extent. No intermediate submission or readback is needed.
For example, another grid can re-bin a selected range of an earlier grid's
unchanged packed input:

```rust
let input = PendingCloudGridInput {
    frame: &earlier.frame,
    channel: 0,
    points: selected_points,
};
let result = grid.encode_pending(&mut batch, input, spec, &mut pool, stamp)?;
let submitted = batch.submit()?;
if let Some(result) = result {
    let frame = submitted.frame(&result.frame)?;
    let summaries = frame.channel::<CloudGridCell>(2)?;
    let diagnostics = frame.channel::<u32>(3)?;
    // Queue only the requested cell summaries and diagnostics.
}
```

Both APIs return `Result<Option<EncodedCloudGrid>>`. The result records its
opaque frame receipt, `spec`, `input_count` and `cell_count`; readable frame
channels become available after submission. `None` means pool backpressure,
with no clears, copies, parameter writes or binding preparation. The processor
and external input remain borrowed until submission/drop. Snapshot/direct-frame
source occupancy is also retained until GPU completion after submission. A
device-source grid consumes one of the shared 64 external-input references;
pending inputs are already owned by the batch and do not consume that budget.

All outputs are cleared before encoding. Empty input still reserves a frame,
exposes the requested zeroed cell summaries and six zero diagnostics, and has
empty point/index ranges. It dispatches no kernels. Nonempty input executes all
five passes, including when every point is rejected. Membership indices remain
relative to the selected range even when the source frame's active range starts
at a nonzero buffer index. Packed attributes, full-width IDs, reserved bits and
nonparticipating NaN payloads are preserved in the copied point channel.

Whole-request validation precedes reservation: canonical device/type, receipt,
range, point capacity, bounds/dimensions and simulation time. Invalid requests
leave earlier batch work usable. Encoding errors poison the enclosing batch;
dropping it publishes nothing. Mutable grid borrowing prevents shared uniforms
or scratch from being reused while commands are pending. Cancelled uniform writes
may flush later, but subsequent ordered parameters define later work. Pool reset
preserves historical frames and does not reclaim pending slots. Known device loss
rejects encoding and readable channels. Raw cloned wgpu handles remain the caller's
synchronization responsibility.

Slot storage is `116 * max_points + 64 * max_cells + 24` bytes, separately budgeted
from the grid's original eight buffers. Grid membership scratch, prefix scratch,
block offsets, uniforms and pipelines are shared across direct output variants.
A bounded 128-entry binding cache retains output buffers until eviction or grid
drop, even after their pool is dropped; per-pool budgets do not bound aggregate
memory across arbitrarily many replaced pools.

`encoded_frames` counts successful direct encodes, including subsequently dropped
batches. `frames` and `submissions` remain legacy `process` counters. Copy and
kernel counters include direct encoded work rather than necessarily completed GPU
work; the command batch reports the shared submission. No point payload uploads
or host reductions occur inside these direct APIs.

The direct tests compare typed multi-grid chains to independent CPU membership
and reduction calculations, including nonzero input offsets and 66,049 cells
requiring two-dimensional reduction dispatch. Empty/rejected input, four identical
maximum-f32 intensities, cancellation, historical frames, foreign/invalid inputs
and device loss are covered. The extreme-value fixture remains finite because
the shader divides each contribution by count before summing; the existing
overflow-reporting contract still applies when the normalized sum overflows.

The four-slot warm fixture varies counts and grid dimensions for 24 frames with
four direct binding preparations, 16 fixed pool buffers and the original eight
grid buffers. This is resource-reuse evidence, not whole-pipeline zero allocation
or performance acceptance. Direct cloud construction/filtering and generated
LiDAR/camera producers remain separate unfinished S02 work.

## Memory and measurement

`CloudGridConfig` defaults to 65,536 points, 65,536 cells and 64 MiB. Eight
retained GPU buffers use exactly

`120 * max_points + 76 * max_cells + 8 * ceil(max_cells / 256) + 104` bytes.

This includes the packed input copy, memberships, output indices, cell counts,
prefixes/cursors, block prefixes, cell summaries, diagnostics and uniforms.
It excludes host mirrors, pipeline/driver memory and staging allocations.
Application and device limits are checked before allocation. Statistics expose
retained bytes/buffers/binding preparations, submissions, copied bytes and
compute dispatches; they are not proof of allocation-free driver execution.

```sh
make test-gpu P=molla-sensors T=sensor_grid_gpu A='--test-threads=1'
make test-gpu P=molla-sensors T=sensor_grid_direct_gpu A='--test-threads=1'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_grid'
```

The matrix covers 1,024 / 65,536 / 1,000,003 points in one dense cell or 1,024
distributed cells. It compares CPU membership/reductions, completed GPU work
and GPU plus selected-summary collection. The CPU lane specializes to finite,
in-bounds benchmark points, retains per-cell vectors and uses sequential f32
reduction; it is not a timing of all GPU API validation or identical reduction
order. GPU timings include the input copy and all five passes. Initial upload
and processor creation are outside timed iterations. Dense cells currently use
one strided reduction workgroup per cell; dense-cell performance and crossover
must be measured, not inferred from correctness or smoke-test success.

The regression suite covers empty/all-rejected inputs, 255/256/257 boundaries,
65,537-point prefixing, 66,049 cells (multiple block chunks and 2D dispatch),
a million members in one cell, extreme values, selected source offsets,
queued copies and cloud → snapshot → grid without intermediate host arrays.
