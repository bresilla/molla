# Selected point-cloud export

`cloud_export` serializes explicitly collected `CloudPoint` slices on the CPU.
It never calls a GPU API, downloads extra channels, opens a file or flushes a
caller-owned writer. Use any `std::io::Write`, including a buffered file or an
in-memory byte vector. Existing `pointcloud` exports of legacy CPU `HitPoint`
and `Scan` values remain available.

## Collect only the selected scan or job

```rust
use molla_sensors::{CloudPoint, CloudSummary};
use molla_sensors::cloud_export::write_cloud_ply;

let ticket = ring.try_enqueue(&[cloud.summary_source(job_index)?], stamp)?;
// Explicitly poll/wait for that ticket; then decode its five-word summary.
let summary = CloudSummary::from_words(&summary_read.channel::<u32>(0)?)?;
let begin = summary.start as usize;
let end = begin.checked_add(summary.count as usize).ok_or("range overflow")?;
let ticket = ring.try_enqueue(&[cloud.points_source(begin..end)?], stamp)?;
// Explicitly poll/wait for the selected point copy.
let points = points_read.channel::<CloudPoint>(0)?;
write_cloud_ply(&points, &mut output)?;
```

The frame must remain borrowed during this two-step selection, or capture its
point/count sources in a leased snapshot before allowing the producer to run
again. Do not apply an earlier frame's summary to a later frame's points.
Retain the readback stamp and scene identity with the export. Per-point sensor
and sample IDs are stored in the full record, but simulation time, scene
revision, extrinsics and arbitrary user labels are not automatically embedded
in the file. They remain caller-owned metadata.

No host point array is needed between LiDAR generation, visibility and cloud
processing. Optional export is the explicit collection boundary. The GPU
integration test selects a one-point job from a 259-point input using a staging
slot of only 112 bytes: neither the other job nor the unused tail fits in that
slot. A second test exports separate all-miss/hit scans from a GPU LiDAR chain,
including an empty cloud, without collecting input ray/hit arrays.

## Full-record binary PLY v1

`write_cloud_ply` writes an explicit binary-little-endian PLY vertex schema with
the comment `molla packed cloud v1`. Each vertex occupies 112 bytes. It retains
the exact bits of every `CloudPoint` field, including full-width IDs, validity
and optional-attribute flags, acquisition metadata and reserved words.

Properties, in order:

| Type | Properties |
|---|---|
| float | x, y, z, intensity |
| float | nx, ny, nz, range |
| float | dx, dy, dz, scalar |
| uint | sensor_id_low, sensor_id_high, sample_id_low, sample_id_high |
| uint | row, column, subray, source_index, shape_id |
| uchar | red, green, blue, alpha |
| uint | flags, reserved0 |
| float | theta, phi, reserved1, reserved2 |

The two halves reconstruct each u64 ID as `low | (high << 32)`. They are not
converted to floating point or narrowed to legacy `HitPoint::scan_id`. RGBA
bytes retain the original packed value; flags still determine whether color,
scalar or angles are present. Generic viewers may ignore custom properties or
flags, so their display is not proof of full-record retention. This schema uses
the scalar types and binary-endian format supported by [PCL's PLY
implementation](https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/io/include/pcl/io/ply/ply.h);
it has not been visually imported into every external application.

Serialization performs no filtering or numerical normalization. If explicitly
given missed/invalid records, it preserves them too, including negative zero,
NaN payloads and infinities. That is archival fidelity, not a claim that those
records are valid measurements. Normal exports should select the compacted
valid prefix. Reserved data is preserved rather than interpreted.

`read_cloud_ply` is a **strict reader for this exact schema**, not a general
PLY importer. It accepts LF or CRLF headers, rejects reordered/unknown
properties, other versions/formats and truncated records, and stops immediately
after the declared vertex payload. Subsequent bytes can belong to another
frame; callers needing strict file EOF must check it themselves.

`CloudReadLimits` bounds point count and header-plus-payload byte size before
allocating the result. Defaults are 1,000,000 records and 128 MiB. Header lines
are capped at 512 bytes, and the fixed schema bounds the number of lines.
Malformed counts and allocation-size overflow return errors. The limit does
not account for a caller's existing input file buffer or other application
memory. Header/stream errors return no partial point vector. Writers can leave
partial bytes on I/O failure; atomic replacement/durability is caller-owned.

## Legacy XYZ/RGB projection

`write_cloud_xyz_rgb` writes the existing six-column text convention directly
from packed records, without first allocating `HitPoint` objects or scalar
maps. It validates all selected records' validity flag and finite XYZ before
writing. Missing color becomes white; packed color becomes normalized RGB.
Intensity, normals, direction, alpha, IDs and scan metadata are omitted.
Coordinates use six decimal places and RGB uses four: this is intentionally
lossy, including for tiny coordinates. Use full-record PLY for exact retention.

The existing `pointcloud::read_xyz_rgb` reads this projection. Existing legacy
ASCII PLY and PTX writers are unchanged. A compacted/filtered cloud is not
silently reconstructed into a PTX scan with invented missing grid cells or
scanner metadata; use the original `Scan` API when that format is required.

## Verification and timing

```sh
make test APP_TARGET='-p molla-sensors --features gpu --lib cloud_export'
make test-gpu P=molla-sensors T=cloud_export_gpu A='--test-threads=1'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_cloud_export'
```

Tests cover bit-exact all-field round trips, explicit little-endian offsets and
RGBA bytes, empty records, CRLF, negative zero/NaN payload/infinity retention,
every truncated prefix of a fixture, malformed schemas/counts, bounded header
reads, exact budgets and injected I/O errors. GPU tests additionally verify
selected-job metadata and re-upload of decoded points into the GPU pipeline.
Legacy XYZ/RGB, ASCII PLY and PTX tests run in the full sensor lane.

The CPU benchmark covers 1,024 / 65,536 / 1,000,000 already collected points:
binary PLY write, XYZ/RGB write and binary PLY read. Writes retain an output
vector; reads allocate the bounded result. No GPU transfer or filesystem
latency is included. The separate `bench_sensor_cloud` measures completed GPU
processing and selected-prefix collection. Smoke checks establish executable
paths, not throughput, crossover or end-to-end performance acceptance.
