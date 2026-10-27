# Per-frame sensor latency measurements

`bench_sensor_latency` records individual completed frames rather than treating
Criterion's confidence interval on a mean as a p95 latency. It currently implements
the **LiDAR portion** of the SENSORS.md performance matrix. The camera/image/
annotation counterpart is [documented separately](camera_latency_benchmarks.md).
The same harness has a [cloud/grid-chain matrix](cloud_latency_benchmarks.md)
selected with `--cloud-matrix`. Physics-combined and like-for-like legacy
comparison lanes remain separate work.

## Reproduce

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--samples 64 --warmup 8 --repeats 3 --output /tmp/lidar-quick.tsv'

NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--lidar-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/lidar-matrix.tsv'
```

Output paths must not already exist. `--filter SUBSTRING` selects named cases;
an unmatched filter fails. `--no-timestamps` provides an instrumentation-off
control; absent hardware timestamp support produces `NA`, not CPU time relabelled
as GPU time. Defaults are a ten-case quick subset, eight warmups and 64 samples
per case, repeated three times. Use two samples/one repeat only for smoke checks.

`sensor-bench-info` records compiler/Cargo versions, OS, revision, dirty-entry count,
a sorted fingerprint of crate Rust/WGSL/manifests, root manifest/lock/environment
hashes and available NVIDIA board metadata. The benchmark records its actual
adapter/backend and release/debug profile. Keep metadata, raw TSV and stderr case
descriptions together; a dirty checkout's commit ID alone is not reproducible.

## LiDAR workload coverage

The named 120-case matrix runs each row below with both final-host and device-only
consumption. It is a defined coverage matrix, **not every Cartesian combination**.

| Lane | Nominal total rays | Geometry | Mounts / sub-rays |
|---|---|---|---|
| Warm | 1k, 10k, 100k, 1M | 100, 1,000, 10,000 spheres; 1, 10, 100 detailed mesh instances; heightfield | 1 / 1 |
| GPU pose updates | Same four sizes | 10 detailed mesh instances | 1, 4, 16 / 1, 4 |
| Topology replacement | Same four sizes | 1,000 spheres or one detailed mesh | 4 / 1 |

Meshes contain 32,768 non-coplanar triangles shared by all instances. Terrain is a
129×129 heightfield converted to 32,768 triangles. An infinite plane at Y=-2 fills
the background. Rays form a 32-row downward angular grid; counts round up to full
rows/pulses/mounts and the exact count is recorded. Multiple mesh mounts target
distributed tiles; a one-mount view targets the first tile, not every object.
These are synthetic fixtures, not complex vehicle assets or worst-case overlapping
geometry. Four-sub-ray cases use 0.003-radian divergence; mounts retain distinct IDs
and poses. Each moving frame uploads fresh body translations; this is prescribed
pose motion, **not physics stepping**. Replacement alternates model topology by
adding/removing a distant sphere, recompiling the scene and rebuilding its query
acceleration in the measured interval. Warm/pose cases assert one scene/binding
preparation and all cases assert zero producer-internal host readbacks.

The default retained-array budgets and one-slot, 256 MiB output pool apply. A
capacity/adapter rejection terminates visibly rather than silently omitting cases.
Only one case is live at a time. The largest nominal case is one million total
rays, not one million per mount. The benchmark samples serialized completion, not
maximum queued/concurrent throughput or a realtime deadline guarantee.

## Timing and accounting boundaries

- `setup_ms`: fixture/model construction, owned-scene compilation, retained sensor
  buffers, output/readback pools and timer creation, followed by an upload flush
  and device wait. Scene-query pipelines prepared lazily are in the first frame.
  Setup uses an existing device; it is not fresh-process/driver-cache cold start.
- `first`, `warmup`, `sample`: separate TSV phases. First-use shader/BVH/binding
  construction is not mixed into steady-state quantiles. Replacement intentionally
  repeats scene preparation in every sample. Default periodic rebuild policy stays
  enabled and its work contributes to measured latency.
- `encode_us`: CPU wall time from pose update/topology replacement through command
  encoding, including descriptor uploads and any required preparation.
- `submit_elapsed_us`: cumulative wall time from the same start through submit,
  not submission duration alone. Subtract encode time for that interval.
- `complete_us`: same start through GPU completion, selected final-output readback
  for host cases, distance/ID decoding checks and receipt/result release. Device
  cases perform a completion wait with no sensor-result download. Host cases copy
  distance, shape, point, normal and first/last/count; they do not materialize CPU
  point/normal vectors or a legacy `Vec<RayHit>`.
- `gpu_us`: timestamps around the sensor command stream (including output clears,
  generation, refit/traversal and reduction), excluding queue-write uploads,
  topology compilation, CPU encode/wait, result downloads and timer resolution.
  Timestamp resolution/copy still adds work to the completion interval. A separate
  16-byte diagnostic download, one submission and one wait happen **after** each
  measured interval; they are excluded from result-transfer/submission columns.
- Upload accounting is explicitly **descriptor plus live-pose payload**, not all
  queue traffic. Uniforms, initial buffers and replacement scene uploads are not
  included. Scene atlas bytes are reported separately, not substituted for total
  transfer. Result-download and readback submission/wait counters are actual deltas.
- Allocation counts/bytes cover successful Rust allocations/reallocations on the
  calling thread during a frame, including runtime submission/collection cleanup.
  They exclude other threads, driver allocations, sample logging, timing collection
  and statistics. Requested allocation bytes are not live or peak memory.
- Retained sensor host/device arrays, scene atlas, output pool and staging payloads
  are reported separately. `/proc/self/status` RSS/high-water values describe the
  **whole benchmark process**; high-water does not reset between cases. They omit
  device-local VRAM and are not per-sensor peak-memory measurements. Optional
  `nvidia-smi` monitoring measures the entire board, including desktop/other users.

Quantiles use nearest rank on per-frame samples. Each repeat reports p50/p95,
mean-derived completed ray throughput, and a deterministic 1,000-resample percentile
bootstrap interval for p95 (2.5–97.5%). That interval assumes exchangeable samples;
thermal/frequency changes, scheduling and temporal correlation can invalidate that
assumption. Compare independent repeats and instrumentation-off controls, retain
outliers, and investigate greater-than-10% p95 changes. Do not turn a favorable
single run or allocation counter into a speedup or completed S08 acceptance.

## Recorded baseline

On 2026-09-25, both timestamp-enabled and instrumentation-off matrices completed
all **120 cases × 3 repeats × 64 measured frames** (23,040 measurements each), plus
8 warmups and one first frame per case/repeat. Hardware was RTX 4080/Vulkan,
NVIDIA 595.84, Rust/Cargo 1.94.1, Linux 7.0.0-31-generic, release profile. HEAD was
`367d92c` with WIP; the before/after crate fingerprint was
`6d46d0ed83a52eaca8ef8389c3545c65ba4c73127f5ec6590ced11e167d66125`.

Raw compressed TSV, full logs, source metadata and board monitoring are preserved
in `benchmarks/sensors/2026-09-25-lidar-latency/` at the repository root, with
checksums. No production sensor implementation was optimized during these runs.

The following **provisional observations** use instrumentation-off completion
latency. Each range spans the three per-repeat p95 values, in milliseconds:

| Workload | Actual rays | Device completion p95 | Final-host completion p95 |
|---|---:|---:|---:|
| 100 spheres, one mount, warm | 1,024 | 0.138–0.217 | 0.164–0.244 |
| 1,000 spheres, one mount, warm | 10,016 | 0.168–0.173 | 0.247–0.265 |
| 10 moving detailed meshes, 16 mounts, four sub-rays | 100,352 | 0.222–0.720 | 0.921–5.169 |
| Terrain, one mount, warm | 1,000,000 | 1.508–1.814 | 18.634–19.597 |
| 10,000 spheres, one mount, warm | 1,000,000 | 1.571–1.768 | 18.735–19.819 |
| Detailed mesh topology replacement, four mounts | 10,112 | 8.214–9.916 | 8.255–10.034 |

One-million-ray single-sub-ray host frames copy **52,000,000 bytes** of selected
outputs. Their additional completion cost includes copy/map/host inspection; the
measurements do not isolate PCIe transfer from CPU copying. Moving 100,352-ray
four-sub-ray frames upload 2,368 descriptor/pose bytes and collect 4,315,136 bytes
only in host mode. Device-only samples had zero sensor-result downloads and one
submission; host samples had one result-copy submission in addition. Instrumentation
downloads are accounted separately as described above.

Calling-thread allocation counts were nonzero: terrain one-million-ray warm frames
used 105 allocations device-only and 144 with host collection; the moving-mesh case
used 165/204. The 100-sphere case varied from 203 to 513 device-only allocations
because periodic maintenance also runs in measured frames. These are workload-
specific observations, not comparisons with an earlier optimized implementation.
Whole-process high-water RSS was about 469 MiB timestamped and 455 MiB without
timestamps. Board memory peaked at 3,300 MiB including the approximately 2,682 MiB
desktop baseline; that difference is not an isolated sensor VRAM peak.

### Variance investigation remains open

**104/120 timestamped cases and 93/120 instrumentation-off cases exceeded 10% p95
spread between repeats of unchanged code.** Removing timestamp overhead did not
establish stationarity. Monitored graphics clocks ranged 210–2,790 MHz. No desktop
processes were closed and no GPU clock/power settings were changed.

Focused instrumentation-off probes used 256 measured frames/repeat after 5,000
warmups for the 1k-ray case and 128 warmups for the million-ray terrain case.
Small-case p95 still varied 0.191–0.306 ms device-only and 0.163–0.359 ms host.
Terrain device p95 varied 0.785–1.657 ms, while host p95 was 19.399–19.522 ms.
Longer warmup therefore did **not** resolve all variation. Graphics/memory clocks
also varied during these probes; that is not proof that frequency changes alone
caused every latency excursion. Investigate power-state/scheduling/maintenance
effects under a controlled environment before setting hard regression budgets.

These are reproducible workload/data artifacts, **not stable performance acceptance,
a promised rate, or a speedup over Rapier/legacy sensors**. No noisy case is waived.
Remaining matrix work includes cameras/optics, processing chains, CPU/GPU physics
combined load, matched legacy baselines, total transfer and peak-memory accounting,
and controlled repeatability for the greater-than-10% rule.

### Locked-clock rerun (2026-09-25)

All three instrumentation-off matrices were rerun as two independent sessions
with graphics clocks locked at 2505 MHz and memory at 10501 MHz. The lock was
reset afterwards. Raw data, clock logs, board monitoring and the analysis are
in `benchmarks/sensors/2026-09-25-controlled/` (see its `README.md`).

Cases whose median (p50) or p95 varied by more than 10% across three repeats:

| Matrix | Unlocked p50 / p95 | Locked session a | Locked session b |
|---|---|---|---|
| LiDAR (120) | 88 / 93 | 7 / 46 | 40 / 60 |
| Camera (56) | 10 / 29 | 12 / 23 | 15 / 26 |
| Cloud (96) | 84 / 83 | 63 / 77 | 35 / 50 |

Locking the clocks removes the frequency swings behind the unlocked spread.
What remains has three identified causes:

- **Interference windows.** Runs of consecutive cases were 1.3–2.3× slower
  while the clock stayed at 2505 MHz. This is outside load: the desktop was a
  GNOME Remote Desktop session with a GPU encoder.
- **Too few warm-ups.** Cloud terrain and static-mesh cases are slower on their
  first repeat; eight warm-up frames are not enough for them.
- **Tail sensitivity.** A 64-sample p95 moves with single outliers.

Without interference, 113 of 120 LiDAR medians repeat within 10%. Hard p95
budgets are not set: that would need an otherwise idle GPU and a longer cloud
warm-up. Every device-only case downloads zero result bytes per frame. Host
cases download only their selected outputs, up to 118 MB per frame for four
720p aperture cameras.
