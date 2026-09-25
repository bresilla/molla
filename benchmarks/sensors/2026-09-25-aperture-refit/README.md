# Frame-local aperture refit evidence — 2026-09-25

One matched workload: four homogeneous 640x480 cameras, four aperture samples,
moving 32,768-triangle mesh, 100 spheres and background plane, device-only raw
camera completion. No image/annotation processing or result readback. Each run
contains three repeats, each with one first frame, 64 warmups and 256 measurements.
Timestamp instrumentation is off. Both runs completed without overlapping tests.

## Reproduce

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  LATENCY_BENCH=bench_camera_latency \
  A='--camera-matrix --no-timestamps --samples 256 --warmup 64 --repeats 3 --filter 640x480_c4_aperture4_homogeneous_moving_camera_device --output /tmp/new-aperture-refit.tsv'
```

RTX 4080/Vulkan, NVIDIA 595.84, Rust/Cargo 1.94.1, Linux 7.0.0-31-generic,
release profile, HEAD 367d92c with WIP. No desktop windows were closed or GPU
clock/power settings changed. No board-clock trace was captured for these runs.
They are sequential observations, not an interleaved controlled experiment.

## Observations

| Metric | Before | After |
|---|---:|---:|
| Completion p95, repeat range (ms) | 6.425–6.586 | 5.839–5.930 |
| CPU encode p50, repeat range (µs) | 124.436–126.014 | 95.505–107.013 |
| Ordinary frame dispatches | 49 | 25 |
| Periodic-rebuild frame dispatches | 71 | 47 |
| Ordinary calling-thread allocations | 760 | 424 |
| Periodic-rebuild calling-thread allocations | 1,070 | 734 |

All 768 measured frames per version have the same pose-upload bytes (160), result
download bytes (0), producer submissions (1) and waits (1). Each version schedules
21 rebuilds per repeat including first/warmup frames. Only redundant refits between
aperture samples were removed; post-rebuild refits and query cadence remain.

These measurements support lower overhead on this workload, not a general sensor
speedup, zero allocations or full SENSORS.md performance acceptance. Repeated p95
ranges are below 10% spread here; earlier full-matrix variance remains unresolved.

## Artifacts

- `before/after.{log,tsv}.gz`: all raw first/warmup/sample records, metadata,
  per-repeat p50/p95/bootstrap intervals, resource counters and source fingerprints.
- Before crate fingerprint:
  `850f99b93f8a2277d8195553953b616f3ffd2c2d45a0d034f54492106c55b4f8`.
  Matching source snapshot is `../2026-09-25-camera-latency/source-inputs.tar.gz`.
- After crate fingerprint:
  `28b40af18f3b2b92fa5c1e79776a4a233b44b1c8e97c8135407d37b9182f90df`.
  `after-source-inputs.tar.gz` contains crate Rust/WGSL/manifests and root
  build/environment manifests. Third-party toolchain binaries are not included.
- Regression logs include the expected red dispatch-budget failure, six passing
  focused GPU tests, 365 full sensor test executions, 17 headless viewer tests
  with timing off and another 17 with timing on, all-target GPU check, scoped
  format check and final documentation build.
- `smoke.{log,tsv}.gz`: all 56 post-change camera configurations with two measured
  frames each; correctness/capacity smoke evidence, not performance statistics.
- `final-info.log.gz`: post-measurement source/environment metadata; fingerprint
  matches the after run.
- `SHA256SUMS` records data/source/log checksums.

The new cancellation/motion regression compares all five channels with raw queries
under disabled, periodic and quality-triggered rebuilding. Existing tests cover
channel masks, invalid-pose recovery, moving triangles, terrain/scene replacement,
heterogeneous groups and held output history. Headless success is not visual proof.
