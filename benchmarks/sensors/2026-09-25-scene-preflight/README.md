# Early scene query-budget rejection — 2026-09-25

Shape/plane records and unique raw-mesh, triangle and heightfield packed payloads
are now checked before scene identity/model copies, terrain conversion or GPU
uploads. Exact raycast BVH node counts follow the existing median split and shared
leaf-size constant without allocating tree arrays. Converted geometry, including
convex-hull output, is checked again before upload; built trees must match the plan.

This preserves packed-budget semantics and shared-instance storage. Convex-hull
workspace, model/identity host storage, refits, pipelines and driver staging remain
outside that budget. No tree topology algorithm, shader, physics solver, Gearbox,
asset, tyre or soil changes were made. The Make sensor gate now includes geometry
validation tests, including the new node-count oracle.

## Reproduced rejection costs

`before.log.gz` fails the new calling-thread allocation regression with a 64 KiB
query budget on three valid source models. `after.log.gz` and `release.log.gz` pass
without relaxing its allocation threshold:

| Fixture | Before requested bytes | After requested bytes |
|---|---:|---:|
| 65,537 sphere instances | 21,897,134 | 37 |
| 65,537 triangles in one mesh | 795,054 | 89 |
| 257×257 heightfield | 3,168,418 | 89 |

Source-model construction is outside these measured regions. These are requested
calling-thread allocation bytes on rejected operations, not process/VRAM peaks or
successful-query speedups. Early errors still allocate error/metadata storage.

## Verification

- `node-count.log.gz`: planned counts match actual median trees at empty, leaf,
  split and uneven-level boundaries through 4,097 triangles; extreme count checks
  do not build giant trees. Runtime atlas construction also checks the count.
- The existing atlas GPU fixture additionally accepts an exact shared-mesh budget
  and rejects one byte under. All scene replacement/identity/reference tests pass.
- `full-tests.log.gz`: 397 passing executions, 111 reported suite results including
  empty feature lanes. The new allocation and geometry validation tests run.
- `release.log.gz`: the new rejected-scene allocation fixture passes in release.
- `check.log.gz`, `cpu-check.log.gz`: all-target GPU/CPU facade and sensor checks.
- `viewer.log.gz`, `viewer-timed.log.gz`: 17 headless tests pass in each mode;
  no GUI was launched, so this is not hardware visual acceptance.
- `fmt-check.log.gz`: scoped formatting check.
- `docs.log.gz`: book build validation.
- `{lidar,cloud,camera,physics}-smoke.{log,tsv}.gz`: all 120, 96, 56 and 48 release
  cases pass, with one first frame and one sample per case. Exact case sets match
  the preceding processor-storage archive. `smoke-coverage.log` uses the long-ray
  archive's `check-smoke.awk`; `physics-smoke-accounting.log` validates dimensions,
  transfers and submissions with the physics-latency archive's `analyze.awk`.
  These 320 execution checks are not timing distributions or performance acceptance.
- `before/`: three implementation files copied before this increment.
- `source-inputs.tar.gz`: crate/viewer Rust/WGSL/manifests and root Cargo/Make/Nix
  inputs; not dependencies, toolchains or machine assets.
- `source.log`: compiler/hardware metadata. RTX 4080 Vulkan, NVIDIA 595.84 through
  the repository Nix environment; no driver, clock or power changes.
- `fingerprint-after.log`: source remained frozen through final gates and smoke runs.
- `SHA256SUMS`: artifact integrity checksums.

Frozen crate source/manifests fingerprint:
`98335bf138e317427e78893e2a58a6053d01b0ba658f96edebe4d0bf04d50c1f`.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make test-sensors
NVIDIA_VERSION=595.84 nix develop --impure -c make test-gpu \
  P=molla-sensors T=sensor_allocation_gpu RELEASE=1 \
  A='scene_query_budget_precedes --test-threads=1 --nocapture'
```

Remaining S01/S02 construction/resource/lifecycle gates, numerical reference/tie
coverage, controlled matched performance, hardware visual acceptance and outstanding
Gearbox camera/moving-part/transfer integration are not closed by this increment.
