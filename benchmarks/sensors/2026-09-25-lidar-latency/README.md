# LiDAR latency evidence — 2026-09-25

These are provisional observations, not accepted regression budgets. Full
methodology and selected results: `book/src/sensors/latency_benchmarks.md`.

## Artifacts

- `matrix.{tsv,log}.gz`: timestamp-enabled 120-case matrix, 3 repeats, 64 measured
  frames, 8 warmups and one first frame per case/repeat.
- `control.{tsv,log}.gz`: identical matrix with timestamps disabled.
- `matrix-gpu.csv.gz`, `control-gpu.csv.gz`: whole-board 1-second monitoring.
- `*-after.log`: final source/environment metadata; crate fingerprint matches
  the respective initial log. Markdown/artifact files are outside that fingerprint.
- `steady-small.{tsv,log}.gz`: two 1k-ray cases, 5,000 warmups, 256 samples, 3 repeats.
- `steady-large.{tsv,log}.gz`: two million-ray terrain cases, 128 warmups, 256 samples,
  3 repeats. Both focused probes disable timestamps.
- `steady-gpu.csv.gz`: 200-ms board monitoring across the focused probes, including
  graphics and memory clocks.
- `SHA256SUMS`: checksums of all captured data/metadata artifacts.
- `source-inputs.tar.gz`: crate Rust/WGSL/manifests and root build/environment
  manifests captured before subsequent benchmark changes. This retains the source
  inputs covered by the original fingerprint, not third-party toolchain binaries.

Each full matrix has exactly 360 case/repeat groups, 23,040 measured rows, 2,880
warmup rows and 360 first-frame rows. Focused probes each have 1,536 measured rows.
Release smoke runs are excluded from this evidence archive. All benchmark and
monitor processes finished before archival; no rows/outliers were removed.

## Reproduction

Use repository-root Makefile tasks inside its Nix environment:

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--lidar-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/new-matrix.tsv'
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--lidar-matrix --no-timestamps --samples 64 --warmup 8 --repeats 3 --output /tmp/new-control.tsv'
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--lidar-matrix --no-timestamps --samples 256 --warmup 5000 --repeats 3 --filter prim100_r1000_m1 --output /tmp/new-small.tsv'
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  A='--lidar-matrix --no-timestamps --samples 256 --warmup 128 --repeats 3 --filter terrain_r1000000_m1 --output /tmp/new-large.tsv'
```

Paths must be new. Capture stdout/stderr as well as TSV. Record board monitoring
separately and do not run other benchmarks/tests concurrently. Metadata is in the
logs, including adapter, toolchain, revision, dirty state and source fingerprints.

## Interpretation

- 104/120 timestamped and 93/120 uninstrumented cases showed greater-than-10% p95
  repeat spread. Longer resource warmup did not remove all variation. The shared
  desktop and dynamic GPU power/frequency environment were preserved, not made
  exclusive. Acceptance/repeatability investigation remains open.
- Scene replacement costs are included; replacement upload totals are not counted
  by the explicitly named descriptor/pose upload column. GPU time excludes CPU
  preparation, uploads, result collection and timer resolve/readback.
- Device-only completion includes a wait, but no sensor-result download. Timer
  diagnostics add a separate 16-byte download/submission/wait outside its interval.
- Allocation data is calling-thread Rust allocation traffic, not process-wide or
  peak memory. Process RSS high-water and whole-board VRAM are different measures.
- These fixtures are the defined LiDAR subset, not the entire SENSORS.md matrix.
