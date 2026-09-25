# Camera and processing-chain latency

`bench_camera_latency` extends the [per-frame measurement method](latency_benchmarks.md)
to camera batches and same-submission image/annotation processing. It reuses the
tested allocation, quantile and GPU timestamp helpers. It does not capture a
Bevy/Gearbox render window or benchmark renderer materials/shadows.

## Run

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  LATENCY_BENCH=bench_camera_latency \
  A='--camera-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/camera-matrix.tsv'
```

Add `--no-timestamps` for the instrumentation-off control. `--filter SUBSTRING`
selects named cases, rejecting an unmatched filter. Without `--camera-matrix`,
six quick cases cover a single 640×480 pinhole camera and the largest four-camera
1280×720 aperture/processing chain. Output paths must be new. Do not run other
tests or benchmarks concurrently with these measurements.

## Defined 56-case coverage matrix

- Every combination of 640×480 / 1280×720, 1 / 4 cameras, pinhole / four-sample
  aperture, raw camera / camera-processing chain, device completion / final-host
  completion: 32 cases.
- Four-camera heterogeneous groups with alternating vertical FOVs 1.05 / 0.9
  radians, both resolutions/optics/pipelines/consumption modes: 16 cases.
- Four 640×480 cameras observing a device-updated mesh, both optics/pipelines/
  consumption modes: eight cases.

All cameras have distinct mounted positions and IDs. Homogeneous batches form one
group; heterogeneous batches form two. The fixture contains a body-attached wavy
32,768-triangle mesh, 100 colored spheres and an infinite background plane. Moving
cases upload a fresh translation/rotation before each query, without physics
stepping. These are synthetic geometry workloads, not photorealistic vehicle scenes.

All five camera channels remain enabled in both pipelines. Aperture cases use a
0.04-m lens, 5-m focal distance and four color samples, with center-sample geometry.
The chain adds exposure, GPU gray-world reduction, Reinhard tone mapping and gamma,
plus parallel shape-to-class masks and per-instance boxes. Inputs pass through
pending GPU receipts in one submission; there is no intermediate host image or
per-aperture color download.

## Explicit capacity limits

The benchmark uses a **2 GiB combined camera-array budget** and 2 GiB shared-ray
sub-budget, without changing library defaults or adapter limits. Four 1280×720
cameras reserve about 1.342 billion bytes of combined host/device camera arrays,
above the default 1 GiB cap. The override permits this required matrix case on the
named 16-GiB adapter rather than silently skipping it. This is an array-payload
budget, not a process-memory cap.

Camera, image and annotation output pools each have one slot and a 256 MiB cap.
Image and annotation producer arrays each have a 128 MiB cap. Readback uses one
slot sized to the selected final outputs. Adapter/capacity errors terminate the
run rather than masquerading as successful or omitted cases. Only one workload is
live at a time. Logs report camera-array, producer-array, camera-output, staging
and scene-atlas payloads separately; processing output slots, driver allocations
and hierarchy/preparation resources are additional memory.

## Timing/consumption boundaries

Setup, first use, warmup and measured samples are separated, with the same
nearest-rank p50/p95 and descriptive bootstrap interval as the LiDAR harness.
CPU encoding includes the optional body-pose upload, camera encoding and processing
job construction/encoding. GPU timestamps cover the entire command stream; result
collection and diagnostic timestamp download remain separate. `pose_upload_bytes`
is producer-reported input payload plus explicit body updates; uniform traffic,
initial scene upload and one-time label tables are excluded.

Raw-camera host cases collect depth, color, normal, albedo and shape for all groups
in **one grouped readback**: 32 bytes/pixel. Host checks decode depth and shape;
they do not construct CPU normal/color arrays. Chain host cases instead collect
processed RGBA8, shape/class masks, all reserved box slots and per-image counts:
`12 * pixels + cameras * (102 * 32 + 16)` bytes. Counts must label every pixel,
with no misses/unknown IDs, and the decoded classes must be in the label table.
Thus raw and chain host cases intentionally collect **different outputs**; a lower
chain host latency is not evidence that adding GPU processing is free.

Device-only cases wait for completion without downloading sensor results. The
timestamped run then makes its separate 16-byte diagnostic transfer outside the
measured interval, as documented for LiDAR. CPU allocation counters include
processing-job vectors, submission and selected collection; requested bytes are
not live/peak memory. `dispatches` reports encoded sensor/processing kernels, not
CPU submissions or timer commands. Processing `copied_bytes` statistics describe
device-to-device input copies, not host readbacks.

All cases assert one retained scene/refit preparation per camera group. Standalone
producer submission counters stay zero; the shared batch submits once, with a
separate result-copy submission only in host mode. API lifecycle tests remain
necessary: benchmark completion alone does not prove reset/cancellation/device-loss
handling, zero allocations, rendered fidelity or a stable realtime rate.

## Evidence status

On 2026-09-25, both timestamp-enabled and instrumentation-off runs completed all
**56 cases × 3 repeats × 64 measured frames**: 10,752 measurements per run, plus
1,344 warmups and 168 first-use frames. All host validation and transfer/submission
invariants passed. Before/after crate fingerprints matched:
`850f99b93f8a2277d8195553953b616f3ffd2c2d45a0d034f54492106c55b4f8`.

Hardware/environment: RTX 4080/Vulkan, NVIDIA 595.84, Rust/Cargo 1.94.1, Linux
7.0.0-31-generic, release profile, HEAD `367d92c` with WIP. Raw TSV, full logs, board
monitoring, metadata and a source-input snapshot are archived with checksums in
`benchmarks/sensors/2026-09-25-camera-latency/`. Smoke runs are not mixed into these
measurements. No production sensor implementation changed during this baseline.

### Subsequent frame-local refit comparison

After the full baseline, redundant hierarchy refits between aperture samples were
removed. Every frame still refits its center sample; scheduled rebuilds still
refit after reordering. A matched four-camera 640x480 moving-mesh aperture case
completed three repeats of 256 measured frames per version, with timestamps off
and 64 warmups per repeat. Device-completion p95 ranges were **6.425–6.586 ms before**
and **5.839–5.930 ms after**. CPU encode p50 ranges were 124.436–126.014 µs and
95.505–107.013 µs. Ordinary frames encoded 49 → 25 dispatches and made 760 → 424
calling-thread allocations; rebuild frames retained their additional 22 dispatches
and 310 allocations. Result readback remained zero, with one producer submission.

Evidence and source snapshots are in
`benchmarks/sensors/2026-09-25-aperture-refit/`. This is a sequential single-workload
observation on the same RTX 4080 environment, without controlled clocks or a
board-clock trace. It supports lower measured overhead here, not an all-workload
speedup, zero allocations or resolution of the full-matrix variance below.

### Provisional completion observations

Ranges below span the three **instrumentation-off per-repeat p95** values, in ms.
They are measured completion boundaries, not promised realtime rates:

| Workload | Device completion | Final-host completion |
|---|---:|---:|
| 1×640×480 pinhole, raw camera | 0.462–0.622 | 1.804–1.810 |
| 1×640×480 pinhole, image/annotation chain | 0.886–1.408 | 1.335–1.927 |
| 4×1280×720 pinhole, raw camera | 3.266–3.451 | 47.220–50.069 |
| 4×1280×720 pinhole, image/annotation chain | 6.048–6.101 | 20.417–23.935 |
| 4×1280×720 four-sample aperture, raw camera | 13.136–13.773 | 55.781–60.861 |
| 4×1280×720 four-sample aperture, image/annotation chain | 15.432–16.805 | 31.395–36.334 |
| 4×640×480 four-sample aperture, moving mesh, raw camera | 6.281–7.322 | 19.895–20.405 |
| 4×640×480 four-sample aperture, moving mesh, chain | 7.988–9.411 | 12.802–13.867 |

These rows use homogeneous intrinsics. The archive also contains every heterogeneous
case. The largest raw-camera host frame copies **117,964,800 bytes**; its chain
counterpart copies **44,249,920 bytes**. Smaller final payload explains why host
chain completion can be lower despite additional GPU work; this is not a matched
output comparison or proof of an optimization speedup. Device-only samples have
zero sensor-result downloads and one submission; host samples use two submissions.

Calling-thread allocations remain nonzero. The largest homogeneous aperture case
used 756–1,066 allocations device-only and 794–1,104 with raw host collection;
the image/annotation chain used 1,247–1,557 / 1,286–1,596. Periodic scene maintenance
contributes to those ranges. Buffer preparation statistics stayed retained: each
group built one scene/refit binding set, while aperture queries still performed
scene refits per sample. The subsequent frame-local reuse change described in
[TiledCamera](tiled_camera.md#pipeline) is not included in this archived baseline.

Whole-process high-water RSS reached about 1,551 MiB timestamped and 1,514 MiB in
the control. Whole-board memory peaked at 3,811 MiB including the approximately
2,682 MiB desktop baseline. Neither measure is an isolated per-sensor VRAM peak.

### Acceptance remains open

**26/56 timestamped and 29/56 instrumentation-off cases exceeded 10% p95 repeat
spread.** Graphics clocks ranged 210–2,805 MHz during board monitoring. No desktop
processes were closed and no GPU clock/power configuration was changed. Removing
timestamps therefore did not establish universal stationarity; do not waive the
10% investigation rule or turn these observations into hard regression budgets.

The [cloud/grid-chain matrix](cloud_latency_benchmarks.md) is now recorded separately.
Remaining performance work includes controlled variance investigation,
CPU/GPU physics combined load, matched legacy comparisons and
complete transfer/peak-memory accounting. Rendered/Gearbox acceptance and other
SENSORS.md numerical/resource gates remain separate from these measurements.
