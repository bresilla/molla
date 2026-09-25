# Camera/processing latency evidence — 2026-09-25

Methodology and selected observations:
`book/src/sensors/camera_latency_benchmarks.md`.
These data are provisional, not accepted regression budgets or a speedup claim.

## Captured artifacts

- `matrix.{tsv,log}.gz`: timestamp-enabled 56-case camera/processing matrix.
- `control.{tsv,log}.gz`: the same cases with timestamps disabled.
- `*-gpu.csv.gz`: whole-board 1-second monitoring during each complete run.
- `*-after.log`: final environment/source metadata. Crate fingerprints match the
  initial logs; source was unchanged during both runs.
- `source-inputs.tar.gz`: crate Rust/WGSL/manifests and root build/environment
  manifests corresponding to the fingerprint, not third-party toolchain binaries.
- `SHA256SUMS`: checksums of every data/metadata/source artifact.

Each run has 56 cases x 3 repeats x 64 measurements = **10,752 measured rows**,
1,344 warmups and 168 first frames. Every case/repeat completed; no outliers were
removed. Smoke tests are excluded. All producer/host checks passed, as did grouped
transfer/submission invariants: device=0 result bytes/1 submission, host=positive
result bytes/2 submissions. Timestamp diagnostic transfers are additional and
outside measured completion intervals. All benchmark/monitor processes exited
before archival.

## Reproduce

From the repository root, with new output paths:

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  LATENCY_BENCH=bench_camera_latency \
  A='--camera-matrix --samples 64 --warmup 8 --repeats 3 --output /tmp/new-camera-matrix.tsv'
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors-latency \
  LATENCY_BENCH=bench_camera_latency \
  A='--camera-matrix --no-timestamps --samples 64 --warmup 8 --repeats 3 --output /tmp/new-camera-control.tsv'
```

Capture stdout/stderr as well as TSV, and board monitoring separately. Do not run
other benchmarks/tests concurrently. Library defaults remain unchanged: this
harness explicitly permits 2 GiB of combined camera arrays and retains all adapter
checks for the required four-camera 1280x720 cases.

## Interpretation limits

- 26/56 timestamped and 29/56 control cases exceeded 10% p95 spread between
  unchanged-code repeats. Shared desktop/frequency effects remain under investigation.
- The synthetic scene contains one 32,768-triangle mesh, 100 colored spheres and
  a background plane. Moving cases upload poses, not physics-generated states.
- Raw-camera collection copies five channels; chain collection copies processed
  RGBA8, shape/class masks, box slots and counts. Different final payloads prevent
  treating their host latency difference as a matched-output speedup.
- Timestamp diagnostics, uniform/initial uploads and driver allocations are not
  silently included in the narrower payload/allocation counters. See the book for
  every boundary. Process RSS and whole-board memory are not per-sensor peak VRAM.
- Camera/chain performance coverage does not complete all of SENSORS.md.
