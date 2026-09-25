# Exact CPU mesh-BVH capacity — 2026-09-25

`MeshBvh::build` now reserves the exact median-tree node count, rather than twice
the triangle count. `MeshBvh::planned_node_count` supports arbitrary leaf sizes and
checked arithmetic; the GPU raycast preflight delegates to the same calculation.
Partitioning, triangle order, node layouts, traversal and GPU atlas sizes are unchanged.

## Measured requested storage

The regression builds 65,537 triangles with eight triangles per leaf. Its input
arrays are created before the calling-thread allocation measurement begins.

| Metric | Before | After |
|---|---:|---:|
| Actual nodes | 16,385 | 16,385 |
| Reserved node capacity | 131,074 | 16,385 |
| Node stride | 72 bytes | 72 bytes |
| Requested construction bytes | 14,418,140 | 6,160,532 |
| Allocations / reallocations | 4 / 0 | 4 / 0 |

`before.log.gz` fails the unchanged capacity assertion; `after.log.gz` and
`release.log.gz` pass. Requested bytes include centroid, triangle-bounds, triangle
order and node arrays, not input geometry. This is reserved host payload, not
resident-process/VRAM peaks, zero allocations or a latency speedup.

## Verification and provenance

- Nine leaf sizes (including zero's existing clamp to one), uneven tree levels and
  arithmetic limits are checked against actual builds. GPU preflight keeps its
  independent tree-count checks. The Make sensor gate includes CPU mesh-BVH tests.
- `geometry-cpu.log.gz`, `geometry-gpu.log.gz`: 121 and 125 geometry library tests
  pass respectively. The GPU-feature library lane is not itself GPU execution.
- `mesh-gpu.log.gz`: the geometry-owned mesh GPU regression actually runs and passes.
- `full-tests.log.gz`: 404 passing sensor-lane executions across 112 reported suite
  results, including zero-test feature lanes and the added CPU geometry lane.
- `check.log.gz`, `cpu-check.log.gz`: all-target GPU/CPU facade and sensor checks.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17 headless tests pass in each mode.
  No GUI was launched; hardware visual acceptance remains separate.
- `fmt-check.log.gz`: scoped formatting validation.
- All 320 release smoke cases passed with identical case sets to scene-preflight:
  120 LiDAR, 96 cloud, 56 camera and 48 physics. `smoke-coverage.log` checks both
  phases; `physics-smoke-accounting.log` checks dimensions/transfers/submissions.
  One first frame and one sample per case do not establish timing stability.
- `docs.log.gz`: book build passed. `fingerprint-after.log` matches the frozen
  source below. Protected engine/runtime documentation hashes remained unchanged;
  final process inspection found no remaining benchmark or telemetry polling job.
- `SHA256SUMS` covers every archived file other than itself.
- `before/`: the two implementation files before this change. Restore only in a
  disposable copy when reproducing the failing allocation fixture.
- `source-inputs.tar.gz`: crate/viewer Rust/WGSL/manifests and root Cargo/Make/Nix
  inputs, not dependencies, toolchains or machine assets.
- `source.log`: source/compiler/hardware metadata. RTX 4080 Vulkan, NVIDIA 595.84
  through the repository Nix environment; no driver, clock or power changes.

Frozen crate source/manifests fingerprint:
`bbe4d893a8696352079739cb5f80c4d7941eea05fa07a42d5f53b7a7f0e27bc9`.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test \
  APP_TARGET='-p molla-sensors --features gpu --release --test sensor_allocation_gpu mesh_bvh_construction -- --nocapture'
```

Full S01/S02 construction/resource/lifecycle acceptance, numerical/tie coverage,
controlled matched performance, hardware visual acceptance and remaining Gearbox
camera/moving-part/transfer integration are not closed by this allocation change.
