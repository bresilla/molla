# Cloud/grid-chain latency evidence — 2026-09-25

Method, reproduction commands, workload boundaries and provisional results:
`book/src/sensors/cloud_latency_benchmarks.md`.

- `matrix.{log,tsv}.gz`: timestamp-enabled full 96-case matrix.
- `control.{log,tsv}.gz`: identical cases with timestamps disabled.
- `*-gpu.csv.gz`: 1-second whole-board monitoring, including desktop usage.
- `*-after.log`: final source/environment metadata for each run.
- `smoke.{log,tsv}.gz`: separate 96-case correctness/capacity smoke run.
- `source-inputs.tar.gz`: crate Rust/WGSL/manifests and root Cargo/Make/Nix inputs.
  Third-party dependencies/toolchain binaries are not included.
- `SHA256SUMS`: artifact checksums.
- Gate logs: original ten-case LiDAR quick smoke, 365 passing sensor test
  executions, all-target GPU check, scoped format check and final book build.

Each complete run has 96 cases, 288 case/repeat groups, 18,432 measured frames,
2,304 warmups and 288 first frames. No outliers were removed. Before/after crate
fingerprints match `6310448c09baf979c52ce54ec50936fa4c7a4de5187e54f8e24ac54228db09bb`.
All benchmark and monitoring processes exited before archival and regression work.

All producer, grouped-transfer and host output checks passed. Device-only frames
have one submission/one completion wait/no result bytes; host frames have two
submissions and positive final-result bytes. There are 13 timestamped and 15
control host frames whose blocking-wait counter is zero because the result was
already ready at the initial poll. Timer diagnostics are outside those counters.

P95 spread exceeds 10% between repeats for 81/96 timestamped and 83/96 control
cases. These are provisional workload measurements, **not accepted performance
budgets, a speedup comparison or full SENSORS.md completion**. Movement is prescribed
GPU pose upload, not physics. Host collection copies full reserved output ranges,
not only the kept prefix. Array counters exclude driver allocation/peak VRAM and
uniform uploads; processing copy counters describe GPU-to-GPU transfers.
