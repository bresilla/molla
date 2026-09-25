# Long-ray sensor intersections — 2026-09-25

This increment conditions distant finite-primitive rays before rotation and local
intersection, retains the local surface point for normals, uses stable curved
surface discriminants, removes the positive ellipsoid miss sentinel, and recognizes
finite plane intersections below the former angular cutoff. Raw/owned primitive
queries and the low-level camera now compile the same intersection functions.

Power-of-two direction scaling preserves represented f32 component ratios. Both
primitive and legacy/owned mesh traversal explicitly distinguish ray parameters
from metric distances. Legacy mesh merging reconstructs points with the parameter.
No new GPU passes, buffers, bindings or output layouts were introduced. There are
no physics-solver, Gearbox, machine asset, tyre or soil changes in this increment.

## Reproduced failures

- `before.log.gz`: all five original regressions fail, including sphere distance
  999.2929077 versus 999.3000000 m, an ellipsoid miss becoming a NaN-normal hit at
  max range 1e31, and an incorrectly missed plane at slope 1e-10.
- `silhouette-before.log.gz`: an adjacent-f32 unit-sphere silhouette fails its
  independent normal reference even at 4 m. `stable-discriminants.log.gz` passes
  six focused regressions after perpendicular/cross-product discriminants.
- `oblique.log.gz`: an oblique 1,000,000 m ray returns 999999.4375 instead of
  999999.3001015253 m. `scaled.log.gz` passes seven tests after ratio-preserving
  direction scaling, including the completed low-level camera change.
- `rotated-before.log.gz`: rotating a sphere changes an independently known hit:
  999.867980957 versus 999.868304391 m. Rebasing before rotation fixes this;
  `rotated-after.log.gz` passes eight focused regressions.
- `full-tests-before-mesh.log.gz`: the first complete-suite attempt catches an
  exact raw/owned camera depth mismatch because legacy mesh traversal still
  returns ray parameters. `mesh-metric.log.gz` passes the unchanged camera suite
  after metric conversion; no parity assertion was relaxed.

`before/` contains the five initial shader/wrapper sources. `before-rotation/`
captures the intermediate four shaders before world-relative rebasing.
`before-mesh/` preserves the legacy mesh and merge sources before metric conversion.
Restore snapshots only in a disposable reproduction checkout, not over active work.

## Numerical coverage and verification

`sensor_long_range_gpu` has nine GPU regressions covering six finite primitives,
raw/owned/low-level camera paths, ranges through 1,000,000 m, oblique sphere chord
references, sphere rotation invariance, adjacent-f32 silhouettes, grazing planes,
ellipsoid misses, and low-level mesh direction scaling/clipping/slot reuse.
References use represented f32 inputs and independent f64 geometry. Tolerances
were not relaxed to make the fixes pass; see `contracts.md` for their exact scope.

- `full-tests.log.gz`: 385 passing test executions, 110 reported suite results
  including zero-test feature lanes; all nine new ignored GPU tests actually run.
- `release-tests.log.gz`: nine new GPU tests pass in optimized release.
- `geometry.log.gz`: the geometry-owned mesh GPU regression passes.
- `check.log.gz`, `cpu-check.log.gz`: all-target sensor/facade GPU and CPU checks.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17 headless tests pass in each mode.
  No GUI was launched; this is not hardware visual acceptance.
- `fmt-check.log.gz`, `docs.log.gz`: scoped formatting and book validation.
- `source-inputs.tar.gz`: crate/viewer Rust/WGSL/manifests and root Cargo/Make/Nix
  inputs, not vendored dependencies, toolchains or machine assets.
- `source.log`: compiler, source and hardware metadata. The RTX 4080 Vulkan run
  uses NVIDIA 595.84 through the repository Nix environment. No driver/clock/power
  settings changed and no telemetry polling monitor was launched.
- `{lidar,cloud,camera,physics}-smoke.{log,tsv}.gz`: all 120, 96, 56 and 48 release
  cases pass respectively, with one first frame and one sample per case. These
  320 workload checks are not latency distributions or performance acceptance.
- `check-smoke.awk`, `smoke-coverage.log`: independently checked counts, unique
  case/phase pairs and numeric fields. Case sets also exactly match the preceding
  query-precision archive. `physics-smoke-accounting.log` additionally validates
  dimensions/transfers/submissions through that matrix's existing GNU Awk checker
  (`../2026-09-25-physics-latency/analyze.awk`, one repeat/sample, no warmup).
  Zero spread in this one-sample output does not establish timing stability.
- `fingerprint-after.log`: source unchanged after all gates and smoke runs.
- `SHA256SUMS`: archive checksums.

Frozen crate source/manifests fingerprint:
`dc6d01a037f3f1ab21f86bf57e46f097c2950774e9acdea1f6a034484f76e24d`.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu \
  P=molla-sensors T=sensor_long_range_gpu RELEASE=1 A='--test-threads=1'
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu \
  P=molla-geometry T=gpu_narrow_phase A='gpu_mesh_bvh_raycast --test-threads=1'
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors \
  SENSOR_BENCH=bench_sensor_latency \
  A='--lidar-matrix --samples 1 --warmup 0 --repeats 1 --no-timestamps --output /tmp/new-lidar-smoke.tsv'
```

Use `--cloud-matrix` for cloud workloads; cameras use `bench_camera_latency` with
`--camera-matrix`. Combined physics uses `SENSOR_BENCH_PKG=molla`,
`SENSOR_BENCH=bench_sensor_physics_latency` and `--physics-matrix`. Each output path
must be new; the benchmark rejects overwriting existing data.

Full SENSORS.md completion remains unproven. General far-mesh/grazing/degenerate
coverage, capacity/lifecycle/allocation audit, controlled matched performance,
hardware visual acceptance and remaining Gearbox camera/transfer integration are
still open. Earlier latency archives measure different shader arithmetic.
