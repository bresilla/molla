# Live primitive benchmark

Run through the repository Nix environment and Makefile:

```sh
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
nix develop --impure -c make bench-primitives
nix develop --impure -c make bench-primitives A='--primitive mixed --counts 1 --backend both'
nix develop --impure -c make bench-primitives PRIMITIVE_BENCH_PROFILE=dev A='--backend cpu --counts 1 --warmup 2 --steps 4'
```

The default release matrix runs distance, cone, D6 drive, reaction, break,
friction and mixed scenes at 1, 100, 1,000 and 10,000 instances. Each case uses
20 warm-up frames and 1,000 measured frames at 1/240 s with four substeps.
`--help` lists overrides. Debug/small-step runs are smoke tests, not the required
performance matrix. Do not overlap this command with other GPU tests or benchmarks.

`cargo test --all-targets --all-features` runs the custom harness in smoke mode:
all seven fixtures on both CPU and hardware GPU, one instance each, one warm-up
frame and two measured frames. `cargo bench` supplies `--bench` and retains the
full performance defaults above. `--smoke` or `--test` explicitly selects smoke
defaults even under `cargo bench`; explicit size/frame overrides still apply.
Direct executable invocation without `--bench` also defaults to smoke mode.

`make test-primitives-benchmark-smoke` runs the hardware-backed smoke matrix
through the repository GPU wrapper. `RELEASE=1` selects release compilation.
Smoke runs exercise creation, stepping, readout and break-event count checks;
they are **not** the P6 numerical or performance matrix. Schema 6 labels
`run_mode=smoke` or `run_mode=benchmark` in parent and worker records. A benchmark
mode label does not certify the full matrix when its size/frame settings are
overridden.

Each backend/case/count runs in a fresh child process, sequentially. Failed
construction, capacity errors, numerical faults and process termination produce
a nonzero aggregate exit status while the remaining cases continue. A process
signal is reported as such; it is not automatically diagnosed as OOM. CPU runs
are reference timings, not fallback execution for a failed GPU case. GPU cases
require a hardware adapter and encoder timestamp support.

Each worker has a wall-clock budget of 3,600 seconds, including setup and warm-up;
override it with `--case-timeout-seconds N` (positive integer). A timed-out child
is terminated and reaped, emits `case_process_timeout`, and stops the matrix with
a nonzero status. Later cases are unrun, not passing or silently executed on CPU.
Unlike an ordinary numerical/capacity error, a timeout can leave the driver in an
unknown state, so inspect device/kernel diagnostics before further GPU work.
A timeout is not automatically OOM or proof of a particular shader defect.
Setup/warm-up completion records distinguish construction from stepping. Optional
`--trace-frames` emits each frame's phase/index before stepping for fault isolation;
its output overhead contaminates measured wall/timeline intervals, so traced
runs are diagnostic only, not performance acceptance.

Fixtures use the public `RigidWorld` runtime and independent articulations.
Initial authoring uses `RigidSceneBuilder`: the complete description compiles once,
not once per inserted body/collider/joint or axis edit. Build validates geometry
and topology before exposing a runtime scene. Existing live edits retain their
transactional compilation semantics. Setup timing includes staged authoring and
final compilation; this construction change makes older setup timings historical.
Mixed instances contain six bodies: a distance pendulum, loaded cone, D6 drive
with friction contact, fixed reaction body, breakable support and free friction
body. Only one support breaks in the mixed world; the break-only case breaks one
joint per instance. Break thresholds are armed halfway through measured frames,
so topology recompilation and continuation are included. Reaction readout is
enabled for every case. Measured break counts must match the fixture contract.

Every warm-up/measured frame must advance exactly once and quarantine no body.
Runtime quarantine can accompany a successful step, so a successful return alone
does not produce `case_ok`. Final body poses, velocities, forces and joint
coordinates/rates must be finite, including the last output before another
quarantine check could run. Final finite-state validation runs after measured
wall/device intervals; the constant-time per-frame guards are inside them.
Warm-up plus measured frame-count overflow is rejected before worker creation.
These guards do not replace the separate physical residual/trajectory gates.

Output is tab-separated tagged records (`benchmark_schema`, `adapter`,
`case_process_start`, `case_start`, `case_ok`, `case_process_error`). Successful
records report fixture size, settings, setup time, measured wall time, maximum
frame time, device timeline span, encoded dispatch count, runtime state-transfer diagnostics, observed
device allocation maxima, model compilations and break counts. Adapter records
include driver information. Geometry uses separated 16 m cells and fixed contact
capacity: `max(8, 2 * instances)` pairs, eight contact rows per island.

## Measurement scope and remaining limits

- Schema 2 and later report Molla-encoded dispatch commands from the instrumented live
  runtime, not workgroup invocations, completed shader work or hidden backend
  commands. Counting is thread-local and enabled by `resource-counters`; normal
  builds compile recording calls away. Recompilation inside measured frames is
  included, while setup and warm-up are excluded. Nested scopes are rejected.
- Device timestamps bracket the complete measured loop; **queue idle and host
  orchestration gaps are included**, not just kernel execution. Timer readback
  is outside the wall-time interval and separate from runtime readback bytes.
- `state_upload_bytes` and `state_readout_bytes` are runtime frame diagnostics.
  `all_upload_bytes`, `all_readback_bytes` and their sum `all_transfer_bytes`
  additionally cover instrumented orchestrator queue writes, initial buffer
  contents and mapped view footprints during the measured loop, including
  model/control/identity/parameter transfers and break-time recompilation.
  These are logical byte counts, not measured PCIe traffic. Timestamp-harness
  transfers and uninstrumented driver-internal work are excluded.
- Host counters use the benchmark's process-wide Rust global allocator. They
  record successful allocations/reallocations and requested layout bytes inside
  each measured frame, plus peak live bytes/blocks and increases above that
  frame's initial live state. Background Rust threads are included; driver
  allocations bypassing this allocator are not. Setup and warm-up are excluded.
- `boundary_*` device counters are sampled after frames. `observed_device_*`
  also samples instrumented buffer creation events in the orchestrator during
  measured frames. Both report HAL device totals, including allocations already
  live at the beginning, not only allocations owned by the fixture. A missing
  positive native-allocation counter is `None`, never evidence of zero use.
- `device_transient_peak_allocations=unavailable`: observation points do not
  establish the true high-water mark of all transient native/driver allocations.
  Timestamp buffers are allocated before device observation begins; their
  instrumentation is excluded, but they remain part of the live HAL totals.
  Exact native transient peaks and the full measured matrix remain open work.

Do not rebuild this benchmark executable while its parent driver is running;
workers execute that same binary. Keep smoke runs separate from uncontended
performance measurements.

This harness is not numerical acceptance evidence by itself. Use the primitive
CPU/GPU integration gates for residuals, momentum, equilibrium and trajectory
validation. A failed or unrun matrix is never a passing performance gate.

## Current reverse-lookup-build release measurements — 2026-10-01

Full distance1000/10000 CPU workers complete original20 warm-up/1000 measured
frames, dt1/240, four substeps and all finite/frame/quarantine guards. These are
two CPU records; with the matched GPU10000 worker below, the current post-lookup
build has3/56 records, not a full matrix.

| Instances | Setup, s | Measured wall, s | Maximum frame, s |
|---:|---:|---:|---:|
| 1000 | .007296650 | 25.222094146 | .026505202 |
| 10000 | .044310393 | 288.381595548 | .299031301 |

Evidence: `primitives-dense-body-lookup-current-release-cpu-scale.log` in the same
evidence directory. The10000 CPU row is measured stepping, not a construction
test. It records2,539,340,000 Rust allocations/388,907,656,000 requested bytes
over1000 frames, with105,527,475 B observed Rust live peak. No fixed speedup
claim follows from the reverse-index cache; remaining allocation churn is visible.
Earlier GPU timings below belong to another binary, not a matched comparison.

The matched current RTX4080/Vulkan/NVIDIA595.91.07 distance10000 worker passes
the same full frames/guards. GPU setup114.705031063 s, measured wall54.002751644 s,
device timeline54.002081792 s and maximum frame.072876891 s. It encodes49000
dispatches, transfers13520672000 logical bytes and observes144 device buffers/
51045776 B. Native/transient peaks remain unavailable, and these are not
exclusive-machine timings. Log: `primitives-dense-body-lookup-ten-thousand-current-release-gpu.log`.
Current selected-source/binary hashes: `primitives-dense-body-lookup-current-release-sources.sha256`.

## Historical staged-construction release measurements — 2026-10-01

All seven primitives complete CPU measurements at1 and100 instances using
20 warm-up/1000 measured frames, dt1/240 and four substeps. With the hardware
distance10000 record below, these are15 of the56 required current-build records,
not a completed matrix. The subsequent dense body reverse-lookup repair changes
runtime performance; these15 rows remain historical for their recorded binary.
All frame/finiteness
guards pass; break-only yields1/100 events, mixed yields exactly one event.

| Primitive | CPU setup / measured wall at1, s | CPU setup / measured wall at100, s |
|---|---:|---:|
| Distance | .000112128 / .036345637 | .000562372 / 2.420972477 |
| Cone | .000109749 / .031565053 | .000672679 / 2.096999654 |
| D6 | .000068644 / .024830433 | .000706006 / 1.539233249 |
| Reaction | .000102497 / .012459001 | .000537113 / .588444634 |
| Break | .000107412 / .016517483 | .000543689 / .783824138 |
| Friction | .000096402 / .034294635 | .000674589 / 2.265660374 |
| Mixed | .000104467 / .117804467 | .003912287 / 16.139558184 |

Mixed at100 instances contains600 independent articulations. Current data and
allocation details are in `primitives-initial-builder-current-release-cpu-matrix.log`
under `benchmarks/primitives/impact-history-gpu-2026-09-29/`; selected-source and
binary hashes are in `primitives-initial-builder-current-release-sources.sha256`.
The CPU fixture contract builds10,000 distance articulations in.2225 s in debug;
that is construction evidence, not a measured ten-thousand-step performance row.
Desktop and unrelated CPU activity remain; these are not exclusive-machine timings.

### Historical hardware ten-thousand distance record

RTX4080/Vulkan/NVIDIA595.91.07 completes the same20 warm-up/1000 measured
frames at10,000 independent distance articulations. Setup120.916713553 s;
measured wall55.796242803 s; device timeline55.795588096 s; maximum frame
.218352099 s. It encodes49000 dispatches, transfers13520672000 logical bytes
and observes144 buffers totaling51045776 B. Native/transient allocation peaks
remain unavailable. Frame/final finite-state guards pass, with no quarantine or
unexpected break events. The current CPU10000 measured reference is not collected.
Full record is `primitives-initial-builder-ten-thousand-current-release-gpu.log`
under the same evidence directory, using the same source/binary fingerprint.
No shape/contact workload or complete matrix is implied by this distance fixture.

## Historical partial release measurements — 2026-10-01

Distance at 1 and 1,000 instances completes on CPU and actual RTX 4080/Vulkan
using all original 20 warm-up/1,000 measured frames, dt=1/240 and four substeps.
Native workers used explicit 180-second deadlines; the full matrix defaults
remain unchanged. These four records belong to the earlier source build, not the
current final-source matrix or full acceptance.

| Instances | CPU setup / measured wall, s | GPU setup / measured wall, s | GPU dispatches | GPU all transfers, bytes |
|---:|---:|---:|---:|---:|
| 1 | .000166342 / .036764728 | 77.422233343 / 11.312587452 | 49000 | 2024000 |
| 1000 | 6.829185953 / 25.829522681 | 78.526783778 / 15.763829907 | 49000 | 1352672000 |

The thousand-instance fixture uses 1,000 bodies/independent articulations and
the unchanged 2,000-pair/eight-contact-width reservation. The old numerical
failure is preserved; this current full worker passes, including frame/final
state guards. Complete schema-6 allocation/transfer/timestamp/adapter data is in
`benchmarks/primitives/impact-history-gpu-2026-09-29/primitives-distance-{one,thousand}-current-release-benchmark.log`.
Selected-source and binary hashes are in `primitives-distance-one-current-release-sources.sha256`.

Measurements ran after own tests/release compilation ended, but the desktop and
remote sessions were active (baseline GPU 0%, 10469/16376 MiB used), and unrelated
CPU workspace activity was present. No exclusive-machine or fixed-speedup claim
is made. Native transient allocation peaks remain unavailable; construction,
host allocation churn and logical transfer costs are separate from GPU kernels.

### Source-version note, 2026-10-01

The partial-angular-lock feedback repair and initial construction batching change
the shader/runtime build after the four
distance measurements recorded above. Their manifests and results remain valid
historical evidence for those exact builds, not a completed performance matrix
for the final current engine binary.
