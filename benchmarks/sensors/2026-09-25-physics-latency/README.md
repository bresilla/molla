# Combined physics/sensor latency — 2026-09-25

These are provisional current-source measurements, **not accepted performance
budgets, a speedup comparison, or complete SENSORS.md acceptance**.

## Workloads and results

`matrix` enables GPU queue timestamps; `control` disables them. Each contains
48 cases × three repeats × 64 measured frames: 9,216 samples, 1,152 warmups and
144 first frames. Together they contain **18,432 measured frames**. All runs
completed successfully; no outliers or failed cases were omitted.

The matrix crosses CPU/GPU Featherstone, 1/32 four-link articulations, 1/10 steps,
three LiDAR/cloud/camera loads and device/host consumption. This is contactless
articulated physics, not a vehicle/soil or full-engine parity benchmark.

- P95 repeat spread exceeds 10% in **27/48 timestamped** and **35/48 control** cases.
- Calling-thread allocation counts remain nonzero. Stable retained buffer counts
  do not establish an allocation-free frame.
- Process VmHWM maxima: 2,853,880 KiB timestamped; 2,877,088 KiB control. These are
  host process high-water marks, not per-case attribution or peak VRAM.
- No full-run continuous GPU telemetry was captured. Start/end board metadata
  must not be presented as peak GPU memory evidence.

`longwarm` investigates the device-only 100,096-ray cloud/grid plus 640×480 camera
load with 32 articulations and ten physics steps. Both CPU/GPU producers run five
repeats, 128 warmup frames and 512 measured frames each: **5,120 additional samples**.
CPU p95 ranges 1.218–2.271 ms (86.46% spread); GPU p95 ranges 3.101–3.296 ms
(6.27% spread). The narrower GPU result does not waive the full-matrix variability
or prove a particular cause. Host consumption and the other loads were not rerun
with this longer warmup. These numbers are combined completion, not pure physics.

## Artifacts and validation

- `{matrix,control,longwarm}.{log,tsv}.gz`: complete output and per-frame records.
- `*-summary.tsv`: independently recomputed nearest-rank p50/p95 and medians of
  encoding, physics CPU elapsed, queue spans and calling-thread allocations.
- `{matrix,control}-ranges.tsv`: min/max repeat quantiles and p95 spread.
- `*-analysis.log`: strict dimensions, transfer/submission accounting, frame
  uniqueness, complete phase coverage and repeat-spread checks.
- `{matrix,control}-crosscheck.log`: all 144 recomputed summary groups match the
  benchmark's reported p50/p95 exactly. Bootstrap intervals are in the raw logs.
- `analyze.awk`: GNU Awk analysis; `analyzer-self-check.log` records constant-value
  quantile checks and rejection of malformed transfers, missing rows and unexpected
  control timestamps. These synthetic records are not performance measurements.
- `longwarm-gpu.csv.gz`: one-second whole-board samples, including desktop usage
  and an idle tail after the benchmark. Clocks vary within the observed interval;
  there is no per-frame causal attribution or fixed-clock experiment.
- `source-inputs.tar.gz`, `source-after.log`: immutable code/build inputs and final
  source/environment metadata. Dependencies, toolchains and full assets are not
  vendored. See the sibling threaded-traversal archive for regression evidence.
- `SHA256SUMS`: artifact checksums.

The before/after crate Rust/WGSL/manifests fingerprint is
`195c6a3339fa39a5bd61820edf6669047dc864cdc614f88285a7fdca63f8f9bd`.
Hardware: RTX 4080 Vulkan, NVIDIA kernel/runtime 595.84, Rust/Cargo 1.94.1, optimized
bench profile. No source, driver, GPU clock or power settings changed during runs.

## Reproduce and analyze

See `book/src/sensors/physics_latency_benchmarks.md` for Make commands, payload
budgets and timing boundaries. Host completion includes decoding and validation;
GPU queue spans include CPU-induced idle gaps. Timer diagnostic readback is outside
the recorded frame transfer/allocation accounting. Physics submission/uniform
counts are source-audited, not runtime counters. First use means fresh workload
resources within the process, not a cleared driver cache or cold machine.

```sh
gzip -cd matrix.tsv.gz > /tmp/physics-matrix-analysis.tsv
gawk -v timed=1 -f analyze.awk /tmp/physics-matrix-analysis.tsv
gzip -cd control.tsv.gz > /tmp/physics-control-analysis.tsv
gawk -v timed=0 -f analyze.awk /tmp/physics-control-analysis.tsv
gzip -cd longwarm.tsv.gz > /tmp/physics-longwarm-analysis.tsv
gawk -v timed=0 -v repeats=5 -v frames=512 -v warmups=128 -v case_count=2 \
  -v case_filter=a32_steps10_cloud100k_camera640_device \
  -f analyze.awk /tmp/physics-longwarm-analysis.tsv
```

Long-warmup Make arguments:

```text
SENSOR_BENCH_PKG=molla LATENCY_BENCH=bench_sensor_physics_latency
A='--physics-matrix --filter a32_steps10_cloud100k_camera640_device --samples 512 --warmup 128 --repeats 5 --no-timestamps --output NEW.tsv'
```

## Environment limitations and telemetry cleanup

Only one benchmark ran at a time; no viewer or competing GPU tests were launched.
This was not an isolated host: the desktop remained active, lightweight archive/
analysis work and a book build overlapped part of the timestamped run, and five
older `nvidia-smi` polling processes were discovered after the long-warmup run.
Their stdout paths identified them as earlier sensor benchmark monitors. The new
monitor also survived its initial shell cleanup. All six were then terminated by
exact PID after verifying their executable name and stdout path; see
`telemetry-cleanup.log`. No user window or unrelated process was closed.

Earlier claims that all sensor benchmark monitors had exited were incorrect.
The old monitors' effect on these timings was not measured. Raw results are
retained, but neither exclusivity nor a variance cause is claimed. Remaining work
includes controlled variance investigation, matched legacy baselines, complete
resource/visual acceptance and outstanding Gearbox integration gates.
