# Processor construction payload budgets — 2026-09-25

Image, annotation, cloud and cloud-grid processors now preflight retained host
arrays, device arrays and the largest sequential upload-packing payload together.
`ProcessorStorageLimits` defaults to 1 GiB; `new_with_storage_limits` overrides it.
Existing per-processor device-buffer caps still apply independently.

`ProcessorStorageUsage` exposes allocation-free configuration/device validation
and byte estimates. Host output arrays and `Buffer::upload_with_scratch` reserve
fallibly. No dirty-state assertion, GPU output format, kernel, physics solver or
Gearbox/asset code was changed. The shared upload helper changes allocation-failure
handling while preserving successful uploads and retained scratch reuse.

These are logical payload budgets, not process or VRAM peaks. They exclude driver
staging, allocator overhead, pipeline/binding and frame metadata, caller-owned
inputs, scene construction and separately budgeted output/readback pools.

## Evidence

- `host.log.gz`: three pure tests cover byte formulas, exact/under-budget limits,
  invalid/extreme configurations, invalid uniform alignment and fallible host
  reservation overflow. A two-million-point cloud fits a 1 GiB device-only budget
  but exceeds the combined construction budget; no large arrays are allocated.
- `gpu.log.gz`, `release-gpu.log.gz`: four GPU constructor fixtures pass in debug
  and release, comparing retained host arrays and largest packing payload to the
  actual mirrored buffers. Kernel-only device arrays are covered by source-audited
  device-budget formulas, not a claim of whole-driver allocation instrumentation.
- `preflight.log.gz`, `release-preflight.log.gz`: all four 65,537-pixel/point
  rejected constructors request only 116 calling-thread bytes for the error.
- `preflight-mutation.log.gz`: moving image preflight after construction makes
  the same test fail with 3,214 allocations, 699 reallocations and 1,509,413 requested
  bytes. The mutation was reverted before all final gates. This checks rejection
  ordering, not merely that an error is eventually returned.
- `full-tests.log.gz`: 393 passing test executions across 110 reported suite
  results including zero-test lanes; all five added ignored GPU tests run.
- `storage-cpu.log.gz`, `storage-gpu.log.gz`: three storage unit tests pass in each
  feature mode; full sensors additionally exercise GPU upload scratch/authority.
- `check.log.gz`, `cpu-check.log.gz`: GPU and CPU all-target sensor/facade checks.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17 headless tests pass in each mode;
  not rendered visual acceptance. No GUI was opened or closed.
- `fmt-check.log.gz`: scoped formatting validation.
- `docs.log.gz`: book build validation.
- `{lidar,cloud,camera,physics}-smoke.{log,tsv}.gz`: 120, 96, 56 and 48 release
  workload cases pass, with one first frame and one sample per case. Exact case
  sets match the preceding long-ray archive. `smoke-coverage.log` checks unique
  case/phase coverage and numeric fields using `../2026-09-25-long-rays/check-smoke.awk`;
  `physics-smoke-accounting.log` additionally validates all physics matrix dimensions,
  transfers and submissions with `../2026-09-25-physics-latency/analyze.awk`.
  These are execution checks, not distributions or timing-stability acceptance.
- `source-inputs.tar.gz`: crate/viewer Rust/WGSL/manifests and root Cargo/Make/Nix
  inputs; not vendored dependencies, toolchains or machine assets.
- `before/`: five original files from the preceding long-ray source archive.
- `source.log`: source/compiler/hardware metadata; RTX 4080 Vulkan using the pinned
  NVIDIA 595.84 Nix environment. No driver, clock or power settings were changed.
- `fingerprint-after.log`: source remained frozen through the final checks.
- `SHA256SUMS`: checksums of all archived artifacts.

Frozen crate Rust/WGSL/manifests fingerprint:
`6eaaa2b5cc18784a38e31fd3d546fa058d93716573af3a2560c07ef547320c89`.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test \
  CARGO='nixVulkan cargo' \
  APP_TARGET='-p molla-sensors --features gpu --release --lib construction_budget_matches_host_arrays_and_upload_payload -- --ignored --test-threads=1'
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu \
  P=molla-sensors T=sensor_allocation_gpu RELEASE=1 \
  A='processor_construction_preflight --test-threads=1 --nocapture'
```

The full S01 constructor/transient/legacy audit, S02 warm allocation/lifecycle gates,
controlled performance, hardware visual and remaining Gearbox integration gates
remain open. Neither these byte caps nor the preflight measurements prove zero
warm-frame allocations, a speedup or full SENSORS.md completion.
