# Physics and sensor combined-load measurements

`bench_sensor_physics_latency` steps Featherstone physics, then encodes LiDAR,
optional cloud/grid processing and optional cameras against the resulting poses.
CPU physics uploads final poses; GPU physics keeps them device-authoritative.
The benchmark lives in the Molla facade crate so the sensor library does not gain
a solver dependency.

## Reproduce

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  SENSOR_BENCH_PKG=molla LATENCY_BENCH=bench_sensor_physics_latency \
  A='--physics-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/physics-matrix.tsv'

NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  SENSOR_BENCH_PKG=molla LATENCY_BENCH=bench_sensor_physics_latency \
  A='--physics-matrix --no-timestamps --samples 64 --warmup 8 --repeats 3 --output /tmp/physics-control.tsv'
```

Files must not already exist. `--filter` selects case names and rejects an empty
selection. Without `--physics-matrix`, the quick set has eight cases: both physics
backends, one articulation, one step, the first two loads and both consumption modes.
`NVIDIA_SMI` overrides only the metadata command; it does not select Vulkan libraries.
The recorded machine required its matching 595.84 NVML library for telemetry after
the host `nvidia-smi` library changed to 595.91 while the kernel remained 595.84.

## Defined matrix

The complete matrix is **48 cases**: CPU/GPU physics × 1/32 articulations × 1/10
physics steps per sample × three sensor loads × device/host consumption.

Each articulation has four revolute sphere links, a fixed root anchor at Y=8,
radius 0.22 m, initial joint coordinates 0.35 rad, initial velocities 0.15 rad/s
and damping 0.1. Steps use 0.001 s and explicit ping-pong states. The floor is
query geometry only. There are **no contacts, actuators, muscles or vehicle/soil
dynamics**; these measurements are not Gearbox or full-physics parity results.

| Load | Actual LiDAR rays | Independent mounts | Cameras | Optical samples |
|---|---:|---:|---|---:|
| `lidar1k` | 1,024 | 1 | None | — |
| `cloud100k_camera640` | 100,096 | 4 | 1 × 640×480 | Pinhole |
| `cloud1m_camera4x720_aperture4` | 1,000,448 | 16 | 4 × 1280×720 | 4 per pixel |

Scans point downward with separate IDs/origins and 32 rows. The larger loads add
the same keep-all rigid cloud transform and 4,096-cell grid as the
[cloud benchmark](cloud_latency_benchmarks.md). All sensor/processing stages share
one command batch and use live final physics poses without an intermediate pose
download. Cameras produce all five channels; the aperture load uses a 0.04 m lens
diameter and an 8 m focal distance. Default scene-rebuild policies remain enabled.

## Measurement boundaries

- `physics_cpu_elapsed_us` measures CPU time spent stepping and, for CPU physics,
  uploading final poses. For GPU physics it measures **enqueue work, not GPU
  dynamics completion**; it must not be read as a CPU/GPU physics speedup.
- `sensor_encode_us` covers sensor batch construction/encoding after stepping.
  `submit_elapsed_us` starts before physics and ends after sensor submission.
- `complete_us` ends after completion, host validation when requested, bookkeeping
  and frame release. Host decoding and checks of every cloud point, grid totals,
  depth and shape IDs are inside this boundary. This is not transfer-only latency
  or an isolated timestamp for when host bytes first become available.
- Optional `gpu_queue_span_us` spans a separate start-marker submission before
  physics through the sensor batch's end marker. It includes queue idle gaps while
  the CPU works and multiple submissions; it is **not summed kernel execution**.
  The 16-byte timestamp readback is outside frame completion/allocation accounting.
- Calling-thread allocations/reallocations/requested bytes cover the measured
  frame, including host validation. They exclude other threads and driver memory.
  Setup, first-use, warmup and measured frames are reported separately.

Device-only mode waits for completion without downloading sensor results. Host
mode performs one final grouped download: seven LiDAR channels for the small
load, or cloud points/summaries plus grid indices/cells/diagnostics and all five
camera channels for the mixed loads. Payload is `52*rays` bytes for the small
load, otherwise `CloudChain::host_bytes(rays, mounts) + 32*pixels`. Reserved cloud
ranges are transferred, not a count-selected prefix. A zero blocking-wait count
means the ring's initial poll found the result ready.

The current contactless GPU Featherstone path has **five source-audited queue
submissions per step**: four state copies and one dynamics submission. Its
48-byte-per-step uniform and these submission counts are labelled as source
counts in the TSV, not runtime instrumentation. CPU final-pose upload is 32 bytes
per body. Sensor input counters cover LiDAR descriptors and camera poses, not all
uniforms, immutable uploads, staging or driver traffic. Sensor and readback
submissions are runtime counters. Timer collection is outside those counters.

Each producer has one output slot capped at 256 MiB; the final ring has one slot
sized to its payload. Camera retained-array budget is 2 GiB; cloud/grid budgets
are 512/256 MiB. These are separate payload limits, not peak process/VRAM caps.
Only one configuration is live. After measurement, the GPU case downloads final
poses once to check finite state and motion; this is not an intermediate transfer.

## Verified status, 2026-09-25

The initial combined run exposed a batched owned-scene traversal timeout. The
same scan completed as scalar queries, but large batches also failed with
CPU-generated rays. Threaded TLAS/BLAS traversal replaces private traversal
stacks while preserving acceleration, geometry, record sizes and right-first
ordering. Focused regressions now check live bounds, scalar/batched equivalence
and independent sphere/ground hits after GPU physics.

After that change, the eight-case timestamped quick run and all 48 no-timestamp
matrix smoke cases completed with transfer/output checks. The full sensor lane
passed 371 executions across 105 suites. Additional release smoke matrices passed
120 LiDAR, 96 cloud/grid and 56 camera/image/annotation cases, including detailed
meshes. Both headless viewer lanes passed 17 tests. Evidence is archived under
`benchmarks/sensors/2026-09-25-threaded-traversal/`.

The smoke matrix has only one measured sample per case; it establishes execution
coverage, **not latency distributions or accepted performance budgets**.

### Repeated combined-load measurements

The subsequent timestamped and no-timestamp control matrices each completed
48 cases × three repeats × 64 samples: **18,432 measured frames** in total, plus
2,304 warmups and 288 first frames. All transfer/output checks passed; no outliers
were removed. An independent GNU Awk checker verifies complete phase coverage,
dimensions, transfer/submission accounting and exact agreement with the reported
nearest-rank p50/p95 values. Source fingerprint stayed
`195c6a3339fa39a5bd61820edf6669047dc864cdc614f88285a7fdca63f8f9bd`.

These are **provisional, not accepted budgets**: p95 repeat spread exceeds 10%
in 27/48 timestamped and 35/48 control cases. Selected no-timestamp completion
ranges below are min/max across the three repeat quantiles, not confidence bounds.
All rows use 32 articulations and ten physics steps; host rows include validation.

| Producer / load | Device p50, ms | Device p95, ms | Host p95, ms |
|---|---:|---:|---:|
| CPU / 1k LiDAR | 0.709–0.787 | 0.761–0.893 | 0.767–0.917 |
| GPU / 1k LiDAR | 1.927–5.474 | 2.667–7.061 | 2.681–8.175 |
| CPU / 100k cloud + 640 camera | 1.183–1.849 | 1.217–1.942 | 4.582–6.465 |
| GPU / 100k cloud + 640 camera | 2.353–6.217 | 3.020–10.776 | 5.759–19.613 |
| CPU / 1M cloud + four 720 cameras, aperture 4 | 11.787–11.849 | 12.288–19.323 | 107.128–113.652 |
| GPU / 1M cloud + four 720 cameras, aperture 4 | 11.468–11.623 | 14.258–14.568 | 112.265–117.478 |

A follow-up on the 32-articulation, ten-step, 100k-cloud/640-camera device workload
uses five repeats, 128 warmups and 512 samples for each physics producer: another
5,120 measured frames. GPU p95 ranges 3.101–3.296 ms (6.27% spread); CPU p95 ranges
1.218–2.271 ms (86.46%). Longer warmup therefore does not establish general
stability. One-second whole-board telemetry observes changing clocks but cannot
attribute individual latency spikes to them. No clock/power settings were changed.

The full matrices reached process VmHWM of 2,853,880/2,877,088 KiB; this is host
memory, not peak VRAM. Calling-thread allocations remain nonzero. These runs were
sequential, not an isolated-host experiment: desktop activity, lightweight analysis
and a book build overlapped part of the timestamped run. Five old telemetry polling
processes were later discovered and stopped by verified PID/stdout identity, along
with the new monitor that survived its shell cleanup. Earlier statements that all
monitoring had stopped were incorrect; their timing impact remains unmeasured.

Raw data, analysis, environment limitations and cleanup evidence are in
`benchmarks/sensors/2026-09-25-physics-latency/`. Matched legacy comparisons,
controlled performance acceptance and broader SENSORS.md resource/visual/integration
gates remain open. Earlier LiDAR/camera/cloud measurements used a different
traversal source and must not be presented as measurements of this implementation.
