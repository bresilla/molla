# Large-coordinate sensor query precision — 2026-09-25

Primitive ray queries now subtract the instance position before rotating the
origin into shape-local space. The prior ordering rotated two large world-space
positions before cancelling their translation, losing nearby-offset precision.
This changes only raw raycast, owned scene-query and low-level camera shader
arithmetic. Mesh traversal already used subtraction before rotation.

There are no new passes, buffers, bindings, output formats or physics changes.

## Independent regression evidence

- `before.log.gz`: the first independent surface fixture fails on the raw path.
- `before-both.log.gz`: both raw and owned queries fail the prescribed normal
  tolerance at an 8,192 m offset, with the same incorrect normal. Comparing the
  two GPU paths alone therefore did not detect the error.
- `after.log.gz`: both paths pass after the arithmetic fix, without relaxed budgets.
- `expanded2.log.gz`: four GPU fixtures pass, adding cameras and moving meshes.
- `camera-mutation.log.gz`: restoring only the old low-level camera expression
  causes the new plane reference to fail at -32,768 m: 3.25 m versus the independent
  3.25047755 m depth. The final source restores the corrected expression.
- `expanded.log.gz`: a fixture-only compile error from an incorrect BodyId integer
  cast, corrected before the expanded successful run; not a production failure.

The new `sensor_large_coordinates_gpu` suite covers:

1. Independent f64 surface points/normals for sphere, plane, box, capsule, cylinder,
   cone, ellipsoid and triangle mesh, both raw and owned paths, six signed offsets
   through 1,000,000 m, direction scales 1e-250 through 1e250, inside/backside rays
   and live device-authoritative poses.
2. Independent plane depth/normal/identity for raw/owned tiled cameras and the
   separate low-level camera kernel at those offsets.
3. Independent brute-force triangle intersections for 66 shared-mesh instances,
   four offsets, pose/instance-order changes, periodic and quality-triggered
   rebuild policies, exact misses/IDs/owners and no mesh reconstruction.
4. A non-GPU oracle/ULP-budget self-check, explicitly included in `make test-sensors`.

The prescribed local-distance error is <2e-4 m, normal-vector error <4e-4 and
unit-normal error <2e-6 for the surface fixture. World points allow two f32 ULPs
at the largest absolute target coordinate plus 2e-4 m. Input origins/translations
are explicitly quantized before constructing the f64 reference; quaternions use
the normalized represented rotation. These tests separate intersection arithmetic
from input representation. They do not promise submillimeter world positions at
1,000 km or establish accuracy for long local rays, grazing/degenerate surfaces,
or arbitrary composed-transform rounding. See `contracts.md` in this archive.

## Verification and provenance

- `full-tests.log.gz`: 376 passing test executions, 108 reported suite results
  including zero-test feature lanes. All four new ignored GPU tests actually ran;
  the independent host oracle check ran separately.
- `release-tests.log.gz`: all four new GPU fixtures pass in optimized release.
- `check.log.gz`, `cpu-check.log.gz`: all-target GPU and CPU-only facade/sensor checks.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17/17 headless tests each. Not visual proof.
- `fmt-check.log.gz`, `docs.log.gz`: scoped formatting and book validation.
- `{lidar,cloud,camera,physics}-smoke.{log,tsv}.gz`: 120, 96, 56 and 48 release
  cases respectively, each with one first frame and one measured frame. All
  execution/output checks passed; these are smoke results, not distributions.
- `smoke-coverage.log`: exact case/phase counts with no missing or duplicate rows.
  `physics-smoke-accounting.log` additionally verifies all 25 physics-matrix fields
  using the preceding physics archive's GNU Awk checker with one sample/no warmup.
- `source-inputs.tar.gz`: crate/viewer Rust/WGSL/manifests and root Cargo/Make/Nix
  inputs; not vendored dependencies, toolchains or a full asset checkout.
- `source-after.log`: source, compiler and hardware metadata.
- `before/`: the three original shaders; verified byte-identical to the preceding
  physics-latency source archive. Restore only in a disposable reproduction copy.
- `SHA256SUMS`: artifact checksums.

Final crate Rust/WGSL/manifests fingerprint:
`2a8eb7c38708e78f42643fcf66c9436f564839c0eecc9d99677a6cf86c0ad3b3`.

Use the existing Nix environment and Makefile, for example:

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu \
  P=molla-sensors T=sensor_large_coordinates_gpu RELEASE=1 A='--test-threads=1'
```

Performance acceptance remains open. Earlier repeated latency archives describe
the preceding shader arithmetic, not this fix; smoke timings are not new latency
distributions or proof of no regression. Full numerical/resource, hardware visual,
matched performance and outstanding Gearbox integration gates remain required.
