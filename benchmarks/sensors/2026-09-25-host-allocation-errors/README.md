# Fallible sensor constructor arrays — 2026-09-25

Fixed host arrays in ray, IMU, contact, frame-transform, batched-LiDAR and tiled-camera
construction now reserve with `try_reserve_exact` and propagate `Error::Storage`.
This includes mount arrays, ray upload scratch, camera pose packing and LiDAR scan
descriptors/ranges. The four processors already reserved fallibly; their helper
is now shared through the private `host_arrays` module.

Successful payload values, capacities, output layouts and budget definitions are
unchanged. No shader, solver, Gearbox, machine asset, tyre or soil edits are part
of this increment. No GUI or telemetry polling monitor was launched.

## Failure evidence

The integration test's thread-local allocator rejects one exact payload request,
then restores allocation immediately. An unwind guard clears any unconsumed request.
This is injected failure, not exhaustion of actual system RAM. Core dumps were
disabled for the intentional pre-fix abort.

`before.log.gz` records SIGABRT from the first contact array's 16,396-byte request.
`after.log.gz` and `release.log.gz` record typed errors and successful retries for
ten families: contact, IMU, frames, rays, camera poses, LiDAR descriptors, images,
annotations, clouds and cloud grids. The final fixture extends the original six
constructor checks with the four already-fallible processors. It injects one
selected reservation per family, not every possible allocator/driver failure.

Two host helper tests check values/exact capacity, empty inputs and overflow before
element initialization. The main regression suite retains independent successful
numerical, mounted-frame, pipeline, lifecycle and resource checks.

This is not global out-of-memory recovery or a whole-process memory cap. Scene/model
copies, some metadata, error-message allocation, driver staging and GPU exhaustion
remain separate. There is no zero-allocation or latency speedup claim.

## Verification and provenance

- `full-tests.log.gz`: 407 passing executions across 112 reported suite results,
  including empty feature lanes. No tolerance or budget threshold was relaxed.
- `release.log.gz`: optimized injected failures and successful retries passed.
- `check.log.gz`, `cpu-check.log.gz`: facade/sensors all-target GPU/CPU checks passed.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17/17 headless tests passed in each mode;
  these do not establish hardware visual acceptance.
- `fmt-check.log.gz`, `docs.log.gz`: scoped format and book build passed.
- All 320 release smoke cases passed with unchanged case sets: 120 LiDAR, 96 cloud,
  56 camera and 48 physics. `smoke-coverage.log` validates first/sample coverage;
  the physics accounting files independently validate dimensions/transfers/submissions.
  One first frame plus one measured sample is not performance or stability evidence;
  the physics report's zero repeat spread is vacuous for this one-repeat smoke run.
- `fingerprint-after.log` matches the frozen source below. Protected engine/runtime
  documentation hashes match; no benchmark or telemetry polling process remains.
- `SHA256SUMS` covers all archived files except itself.
- `before/`: original touched source files plus the test file before the new
  fixture; not the whole checkout. To reproduce the old abort in a disposable copy,
  restore the original production sources while retaining the new test fixture.
- `source-inputs.tar.gz`: current crate/viewer Rust/WGSL/manifests and root Cargo,
  Make and Nix inputs; excludes dependencies, toolchains and machine assets.
- `source.log`: compiler, hardware and source provenance. RTX 4080 Vulkan with
  NVIDIA 595.84 via the repository Nix environment; no driver/clock/power changes.

Frozen source/manifests fingerprint:
`d331cf4ced82b95f49a56830fac02c8d87c7cef8ce39c930c08a36bac5a9bff6`.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu RELEASE=1 \
  P=molla-sensors T=sensor_allocation_gpu \
  A='constructor_host_allocation_failure --test-threads=1 --nocapture'
```

Full S01/S02 memory/lifecycle acceptance, numerical/tie coverage, controlled matched
performance, hardware visual checks and remaining Gearbox camera/moving-part/transfer
integration remain open.
