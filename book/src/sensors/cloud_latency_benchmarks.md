# LiDAR, cloud compaction and spatial-grid latency

`bench_sensor_latency --cloud-matrix` measures a complete device-resident chain:
LiDAR generation and owned-scene tracing → rigid cloud transformation, filtering
and stable compaction → grid membership and cell reductions. All three producers
encode into one `SensorCommandBatch`; pending results stay on the GPU.

## Reproduce

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--cloud-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/cloud-matrix.tsv'

NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--cloud-matrix --no-timestamps --samples 64 --warmup 8 --repeats 3 --output /tmp/cloud-control.tsv'
```

`--cloud-matrix` and `--lidar-matrix` are mutually exclusive. Output files must
not exist. `--filter` selects case names and fails when unmatched. Two samples,
one warmup and one repeat are smoke coverage, not performance evidence.

## Defined matrix

Each lane runs at nominal **1k, 10k, 100k and 1M total rays**, with keep-all,
spatial-crop and reject-all filters, in both device-only and final-host modes:
**96 configurations**. Actual ray counts round up to rectangular scans and are
recorded in the TSV. This is a representative matrix, not every combination.

| Geometry | Independent mounts | Divergent sub-rays | Lifecycle |
|---|---:|---:|---|
| One detailed mesh | 1 | 1 | Warm fixed scene |
| Ten shared-mesh instances | 4 | 4 | GPU-only prescribed body translations |
| 100 shared-mesh instances | 16 | 1 | GPU-only prescribed body translations |
| 129×129 heightfield | 4 | 1 | Warm fixed scene |

The meshes and terrain contain 32,768 triangles; mesh instances share one atlas.
An infinite plane fills misses below the scene. Scans point downward and retain
distinct sensor IDs. This is prescribed motion, **not physics stepping**.
Default periodic scene rebuilding remains enabled.

Each mount becomes one cloud job. All divergent sub-rays produce candidate cloud
points; this is not first/last-return selection or a new physical echo model.
The cloud applies a 0.15-radian Y rotation and translation `(0.25, 0.5, 0.75)`.
Crop retains transformed points inside `(0.25, -16, -128)` to `(128, 128, 128)`;
it is a workload-dependent fraction, not a promised 50% crop. Reject-all requires
intensity ≥2, while these ray points have intensity 1.

The grid covers `[-128,128) × [-16,16) × [-128,128)` with `32×4×32` cells. Its
input is the cloud's full reserved output range; zeroed compacted tails are
counted as misses without reading a count back to the CPU.

## Consumption and validation

- **Device-only:** one producer submission and one completion wait; no sensor
  result readback. Cloud/grid standalone-submission counters must remain zero.
- **Final host:** one grouped readback copies reserved cloud points, per-job
  summaries/count, grid indices, all 4,096 cell records and grid diagnostics.
  Total is `116*rays + 4*(1 + 5*mounts + 6) + 64*4096` bytes, even when all
  points are rejected. No preliminary count readback or second prefix download.
  There are two measured submissions: production and final collection.
  A blocking-wait count of zero means `ReadbackRing::wait` found the result ready
  in its initial poll; it is not a missing completion check or readback.
- Host checks require correct keep/reject/crop counts, valid per-mount summary
  accounting, zero invalid/missed input rays, grid accepted count equal to kept
  cloud points, no grid overflow/outside errors and matching cell-member totals.
  Kept points must have finite positions, valid ranges, IDs and original indices.
  Decoding and these checks are inside CPU-visible completion time.

Host and device timings measure different consumption boundaries. Rejecting more
points does **not** reduce this fixed readback payload, although validation work
over the kept prefix decreases. This is not a selected-prefix export benchmark.

## Budgets and measurement boundaries

The original LiDAR array budgets remain. Cloud retained-buffer budget is 512 MiB;
grid retained-buffer budget is 256 MiB. Each producer has one output slot capped
at 256 MiB. The one-slot readback ring is sized to the exact final payload with
alignment padding. Constructor/device limits still apply; rejection aborts rather
than silently omitting a large configuration. Only one configuration is live.

Setup logs report sensor array usage, geometry-atlas bytes, cloud/grid retained
buffer bytes and allocations. Their `copied_bytes` counts are **GPU-to-GPU**
staging copies, not host transfers. The host-upload column still counts only
LiDAR descriptors and prescribed body poses; cloud/grid uniforms and transient
driver upload work are excluded. These narrower counters are not complete
transfer/peak-memory accounting. RSS/HWM is process-wide, not sensor-only VRAM.

Timing, allocator scope, quantiles and bootstrap assumptions follow the
[LiDAR measurement contract](latency_benchmarks.md#timing-and-accounting-boundaries).
GPU timestamps span all three producers. Their separate 16-byte diagnostic
readback is outside measured completion and result-transfer counters. The TSV
also reports encoded kernel-dispatch counts. First use, warmup and measured
samples are separate; no outliers are removed.

## Acceptance status

Both full runs completed on 2026-09-25: **96 cases × 3 repeats × 64 measurements**,
or 18,432 measured frames each, plus 2,304 warmups and 288 first frames. All
producer and final-output checks passed. Submission/transfer invariants passed;
13 timestamped and 15 control host frames completed before blocking was needed.
The source fingerprint remained unchanged before/after both runs:
`6310448c09baf979c52ce54ec50936fa4c7a4de5187e54f8e24ac54228db09bb`.

Environment: RTX 4080/Vulkan, NVIDIA 595.84, Rust/Cargo 1.94.1, Linux
7.0.0-31-generic, release profile, HEAD `367d92c` with WIP. Raw measurements,
metadata, board monitoring and matching source inputs are archived with checksums
in `benchmarks/sensors/2026-09-25-cloud-latency/`. Smoke data are separate.

### Provisional observations

Values below are instrumentation-off **p95 ranges across three repeats**, in ms:

| Workload | Device completion | Final-host completion |
|---|---:|---:|
| 1k rays, one mesh, keep all | 0.222–0.379 | 0.287–0.465 |
| 100k rays, 10 moving instances, four mounts/four sub-rays, keep all | 0.554–3.954 | 2.012–2.397 |
| Same moving workload, crop | 0.547–1.058 | 2.061–2.692 |
| Same moving workload, reject all | 0.463–0.903 | 1.742–2.604 |
| 1M rays, terrain, keep all | 4.043–7.714 | 46.646–51.055 |
| 1M rays, terrain, crop | 6.297–6.746 | 43.776–46.657 |
| 1M rays, terrain, reject all | 2.507–5.123 | 39.483–41.016 |

The terrain million-ray cases actually contain 1,000,064 rays and always collect
116,269,676 final bytes in host mode. Their control measured frames make 243
calling-thread allocations device-only and 282 with host collection; timestamps
add three allocations. These are not zero-allocation pipelines.

**81/96 timestamped and 83/96 control cases exceeded 10% p95 repeat spread.**
The 100k moving keep-all device case varied over sixfold. Those results require
investigation, not a waived performance gate or a speedup claim. Whole-board
graphics clocks ranged 210–2,790 MHz; board memory peaked at 4,067 MiB including
the approximately 2,682 MiB desktop baseline. Process high-water RSS was
1,221,828 KiB timestamped and 1,221,916 KiB control (about 1,193 MiB). Neither is
an isolated per-sensor VRAM peak. No user windows were closed and no GPU
clock/power settings changed; measurements alone do not identify the variance cause.

This extends measured cloud-chain coverage, not full SENSORS.md acceptance.
Controlled variance investigation, matched legacy comparisons, CPU/GPU physics
combined load and complete peak-memory/transfer accounting remain open.
