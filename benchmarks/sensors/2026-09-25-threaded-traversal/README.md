# Threaded scene traversal evidence — 2026-09-25

The shared owned-scene ray traversal now uses threaded, stackless TLAS and BLAS
walks. Full acceleration, supported geometry and right-first tie order remain.
Escape links reuse existing GPU record words; public CPU node heights are unchanged.
No linear fallback or omitted mesh path remains.

The original 1,024-ray batch timed out even though scalar queries and every
contiguous power-of-two partition through 512 rays completed. The new traversal
passes those cases and independent sphere/ground references after GPU physics.
This demonstrates a working fix, not a proven driver/compiler root cause.

## Source provenance

The crate Rust/WGSL/manifests fingerprint for final gates and smoke runs is
`195c6a3339fa39a5bd61820edf6669047dc864cdc614f88285a7fdca63f8f9bd`.

- `source-inputs.tar.gz`: crate/viewer Rust, WGSL and manifests plus root Cargo,
  Make and Nix inputs. This is an input snapshot, not a vendored checkout,
  dependency archive, toolchain or full asset bundle.
- `source-after.log`: hardware, compiler, build-input hashes and source fingerprint.
- `before/`: the three old traversal/packing files extracted from the checksum-
  verified prior cloud-latency archive. Reproduction requires restoring all three
  in a disposable copy; they must not replace the active checkout's files.
- `prior-archive-check.log.gz`: checksum verification before that extraction.
- `diagnostics/`: raw logs of discarded isolation experiments, including expected
  assertion failures and timeouts. Those experiments are not production code.

## Correctness and execution evidence

- `final-tests.log.gz`: 371 passing test executions across 105 suites.
- `final-release-tests.log.gz`: four optimized physics/query regression tests.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17 headless viewer tests each, with and
  without sensor timing. These are not rendered visual acceptance.
- `final-{fmt,check,cpu-check}.log.gz`, `docs.log.gz`: scoped formatting, GPU and
  CPU-only all-target checks, and book build.
- `physics-quick.{log,tsv}.gz`: eight cases, two samples, one warmup, timestamps on.
- `physics-smoke.{log,tsv}.gz`: all 48 combined-load cases, one sample, no timestamps.
- `{lidar,cloud,camera}-smoke.{log,tsv}.gz`: 120, 96 and 56 additional cases,
  respectively, each with one sample and no timestamps.
- `*-check.log`: smoke coverage/accounting checks.
- `initial-regression-layout-assertion.log.gz`: the initial regression failure
  against the old private GPU height word. The unit test was updated to verify
  escape links, complete traversal and preserved public CPU heights instead.
  `regression-pass.log.gz` and the final suite include the corrected invariant test.
- `SHA256SUMS`: checksums for the archived artifacts.

These smoke runs establish correctness/capacity/execution coverage, not latency
distributions, performance budgets, a speedup, or full SENSORS.md completion.
Repeated current-source physics measurements are recorded separately.

## Measurement limits and remaining acceptance

The combined fixture is contactless articulated physics, not a vehicle or soil
benchmark. Host completion includes decoding and validation. Its GPU queue timer
includes CPU-induced idle gaps and multiple submissions; it is not kernel-only
time. Physics submission/uniform counts are source-audited, while sensor/readback
counts are runtime counters. See `book/src/sensors/physics_latency_benchmarks.md`.

Repeated controlled measurements, matched legacy baselines, broader numerical and
resource-lifecycle acceptance, hardware visual checks and remaining Gearbox
integration gates are still required. Earlier LiDAR/cloud/camera performance
archives describe different traversal source and are not current-source results.
