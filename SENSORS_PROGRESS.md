# Sensor implementation progress

## Goal and scope

Implement `SENSORS.md` core phases S00–S08. S09 remains separately gated on Gearbox approval; S10 remains optional/deferred by the plan. This is not a completion report.

Started 2026-09-21 from `d2369a9` plus pre-existing working-tree changes. `SENSROS.md` contains a user-requested full snapshot; `SENSORS.md` remains the execution plan. Preserve unrelated viewer, tyre, and FULL_COMPACT work. No Gearbox source or assets changed.

## Current phase status

| Phase | Status | Evidence / remaining work |
|---|---|---|
| S00 | In progress | Existing host/GPU and completed-camera baselines captured; independent triangle oracle and mount tests pass. Full performance budgets, cold-build measurements and broader capability contract remain. |
| S01 | In progress | IMU device history, metric ray inputs, cone hits, checked capacities and cache revision invalidation implemented and tested. Broader lifecycle/geometry checks remain. |
| S02 | In progress | Grouped readback, retained bindings, snapshots and device-loss boundaries are verified. IMU/contact/frame, owned-scene host rays, GPU-generated batched LiDAR, tiled cameras, images, annotations, cloud construction/filtering and cloud grids compose into direct leased slots; IMU history commits at submission and rolls back on cancellation. Submitted-frame/device-source consumers retain leases; scoped typed pending inputs support custom and built-in image/annotation/cloud/grid chains, including LiDAR-to-cloud, camera-to-processing and 1024 logical annotation/cloud jobs through deduplicated dependencies. Heterogeneous CameraBatch migration plus full allocation/lifecycle gates remain. |
| S03 | In progress | GPU primitive/static-mesh merging, mixed terrestrial/aerial ray layouts, independent scan descriptors and first/last sub-ray reduction implemented. Expanded pipeline verification and performance evidence recorded below; shared-query resource/lifecycle gates remain. |
| S04 | In progress | Owned revisioned geometry, shared local mesh BVHs, moving instances and terrain conversion implemented. Top-level acceleration, stable application handles and pose-refit lifecycle remain. |
| S05 | Core verified | Independent simulation-time rates, pause/reset/replacement contracts, device-only auxiliary outputs and grouped wrappers pass. Caller snapshot ownership and explicit sensor reset/recreation are documented; S02 resource and S08 viewer gates remain. |
| S06 | Core verified | Shared visibility, GPU aperture resolve, center geometry, channel revisions, heterogeneous groups and device-only camera/image chains pass. Shared S02/S04 and S08 gates remain. |
| S07 | Core verified | GPU images/clouds/spatial reductions, annotations and selected CPU export pass, including full-record packed-cloud round trips. Shared S02/S04 lifecycle and S08 full performance/visual gates remain. |
| S08 | Not complete | Viewer integration, visual acceptance and full performance evidence remain. |
| S09 | Not authorized | Gearbox untouched. |
| S10 | Deferred | No new promises of physical multi-return, force sensing, or GPU analytic adjoints. |

## Implemented changes

- `Makefile`: retained existing launch fix; added `BENCH_ARGS`, explicit-file formatting and a sensor host/GPU regression target.
- `crates/molla-storage/src/buffer.rs`: `device_buffer_current()` rejects unallocated/stale GPU mirrors while accepting synchronized or device-authoritative buffers. Existing ownership assertions remain unchanged.
- `crates/molla-kernels/src/imu.rs`: separate `encode` entry point; existing `dispatch` wraps it. IMU evaluation and velocity-history copy now share ordered commands at the sensor level.
- `crates/molla-sensors/src/imu.rs`: device-to-device previous velocity copy, valid f32 timestep checks, shape checks, same-sized model replacement detection, explicit reset/recreate behavior. Priming still returns an empty vector in the synchronous API. S02 adds device-only dispatch and grouped optional collection, described below.
- `crates/molla-kernels/shaders/imu.wgsl`: uses validated positive normal f32 timestep instead of silently disabling differentiation below 1e-9 seconds.
- `crates/molla-sensors/src/raycast.rs`: normalize nonzero finite directions robustly before either query path; both hits and reconstructed mesh points use meters. Reject invalid origins/ranges/capacities before work. Supply a real identity dummy pose buffer for all-static primitive scenes.
- `crates/molla-kernels/shaders/raycast.wgsl`: cone side/base intersections, linear quadratic-degeneracy case and finite apex normal; initialize all miss fields for zero direction.
- `crates/molla-sensors/src/lidar.rs`: checked grid/pulse/sub-ray products used by validation/casting. Legacy count getters saturate on invalid metadata rather than panic/wrap; callers must validate before allocation.
- `crates/molla-sensors/src/tiled_camera.rs`: checked camera/pixel products, u32 indexing, binding sizes and workgroup limits before pixel-array allocation.
- Sensor benches now fail when no GPU is available and print adapter identity. Camera cases separately measure completed device dispatch and completed host-channel downloads; neither is presented as pure GPU timestamp time.
- `book/src/sensors/{raycaster,imu}.md`: corrected units, current geometry limitations, source paths, device-history and reset contracts. Unsupported geometry still silently skips in the current query implementation; the documentation explicitly warns against interpreting those misses as free space.

## Baseline environment

- GPU confirmed by the GPU integration regression: NVIDIA GeForce RTX 4080, Vulkan, NVIDIA driver; `is_software: false`.
- Installed NVIDIA kernel driver: 595.84.
- Cargo/Rust environment: repository Nix development tools, `rustc 1.96.1 (31fca3adb 2026-06-26)`.
- GPU test wrapper: `nixVulkan`; tests use the debug profile unless explicitly noted.
- Initial sandbox run could not find an adapter. Rerunning with GPU access passed; the sandbox failure is not counted as a sensor regression or a successful GPU test.

## Verification evidence, 2026-09-21

### Pre-change baseline

```sh
make test CARGO='nixVulkan cargo' APP_TARGET='-p molla-sensors --features gpu --tests -- --ignored --test-threads=1'
make test APP_TARGET='-p molla-sensors --no-default-features'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_raycast -- --save-baseline sensors-s00'
```

- GPU suites: **35 passed**, no failures; 56 host unit tests and 3 CPU adjoint integration tests were filtered out of that explicit ignored-test lane.
- No-default-features host run: **3 unit + 3 adjoint integration tests passed**; GPU-gated files correctly ran zero tests.
- Raw logs: `/tmp/molla-sensors-baseline-gpu.log`, `/tmp/molla-sensors-baseline-host.log`, `/tmp/molla-sensors-baseline-bench.log`.
- Criterion raycast release baseline: 1,000 rays / 100 sphere-loop shapes: **[3.7356, 3.8606, 4.0876] ms**; 10,000 / 1,000: **[5.3464, 7.1687, 8.0528] ms**. Each scene also contains its ground plane. These are Criterion reported estimate intervals, not p50/p95. The larger case has substantial variance; repeat before judging regressions. Both include host input generation outside the timed loop and ray upload/query/readback inside it.

### Red-to-green correctness regressions

- Real GPU body integration originally failed IMU priming with `Storage("host copy is stale; call download() first")`; invalid `dt=0` was also accepted. After the fix the IMU suite passed six tests, then seven after adding 65 mounts and model-replacement/reset coverage.
- Mixed scaled rays originally missed the mesh due to inconsistent distance/range units. After the fix, scale factors 0.25, 1, 4, 1e-30 and 1e30 preserve nearest shape, metric distance, hit point and max-range clipping.
- Original LiDAR grid/aerial validation accepted overflowed budgets; both new host-only regression functions failed before the checks and pass afterward.
- Original cone query returned no hit. Side, base, inside, apex, generator-parallel and miss cases now pass; normals are finite/unit length with known side/base expectations.
- All-static primitive queries originally failed due to an absent body-pose GPU buffer; the new static-only test passes without creating a fake physics body.
- Independent f64 brute-force triangle oracle: 100 rays against a rotated/translated mesh, including both hits and misses; exact hit/miss agreement and absolute distance/point/normal errors below **1e-4** in a scene under 20 m.
- IMU reference tolerances: specific force error below **1e-3 m/s²**, gyro below **1e-4 rad/s** for multi-mount tests; free-fall device state remains host-stale throughout sampling, including reset.
- Raw red logs: `/tmp/molla-sensors-s01-red.log`, `/tmp/molla-sensors-overflow-red.log`, `/tmp/molla-sensors-cone-red.log`.
- Raw green logs: `/tmp/molla-sensors-imu-green.log`, `/tmp/molla-sensors-rays-green.log`, `/tmp/molla-sensors-cone-green.log`, `/tmp/molla-sensors-s01-extra.log`.
- Host tests with GPU code enabled: **59 passed**, including overflow and camera budget tests, without requiring an adapter (`/tmp/molla-sensors-host-green.log`).
- First full GPU re-run after fixes: **40 passed**, before the final two oracle/mount additions (`/tmp/molla-sensors-s01-gpu-all.log`).

### Final verification for this correctness increment

- `make test-sensors`: exit 0; **42 GPU integration tests**, **59 GPU-feature host unit tests**, and **3 no-default-features unit + 3 adjoint integration tests** passed. Raw log: `/tmp/molla-sensors-s01-verified.log`.
- `make test APP_TARGET='-p molla-storage --features gpu --lib'`: 3 host buffer tests passed (`/tmp/molla-sensors-storage-tests.log`). These are not GPU transfer tests; the real transfer/state-authority path is exercised by the IMU GPU regressions above.
- `make check APP_TARGET='-p molla-sensors -p molla-storage -p molla-kernels --no-default-features'`: passed (`/tmp/molla-sensors-feature-check.log`).
- `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'`: passed, including sensor benches/tests (`/tmp/molla-sensors-target-check.log`).
- `make check APP_TARGET='-p molla_viewer'`: passed in 47.13 seconds (`/tmp/molla-sensors-viewer-check.log`); this is compile checking, not a viewer launch or visual acceptance.
- `make fmt-files FMT_ARGS=--check FILES='<all eleven changed Rust files>'` and `git diff --check`: passed. Existing unrelated viewer/workspace files were not reformatted.

### Camera baseline with explicit completion

Command: `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera -- --save-baseline sensors-s00-completed'`.
Release profile; one camera; 36 spheres plus a ground plane; all five channels enabled. Core camera rendering/readback algorithms were unchanged; constructor now includes size checks. Raw log: `/tmp/molla-sensors-baseline-camera.log`.

| Resolution | Dispatch + GPU completion, no readback | Dispatch + five host channels |
|---|---|---|
| 256×256 | [362.14, 366.97, 372.12] µs | [11.182, 11.520, 12.003] ms |
| 512×512 | [526.03, 537.04, 555.31] µs | [12.715, 12.797, 12.879] ms |
| 1280×720 | [1.0842, 1.1486, 1.2231] ms | [38.134, 39.854, 41.137] ms |

Intervals are Criterion estimates, not p50/p95 or pure GPU execution time. The original dispatch-only benchmark had no explicit completion wait; these renamed benchmark IDs must not be compared to old enqueue-only timing as a performance regression. These measurements include warm resource reuse, not first-use shader compilation or BVH construction. They support prioritizing readback/host processing, but do not establish a future speedup or the full acceptance matrix.

## Remaining risks / next actions

1. The current correctness increment passes the checks above. Continue the remaining phase requirements; no visual acceptance has been performed in this implementation session.
2. Define remaining S00 capability/error contracts and numerical budgets, including near-equal hits, cold BVH setup and performance variance. Existing timings do not prove the expanded matrix or production throughput.
3. Finish S01 cache/model revision rules without introducing an unbounded per-frame CPU mesh hash/rebuild path. Model fields are publicly mutable; a reliable revision contract requires deliberate scene ownership, not raw pointer identity.
4. Extend the S02 IMU/readback work below to remaining wrappers and shared encoding, then S03 GPU ray layout/reduction and S04 instance acceleration. Current raycaster still downloads primitive/per-mesh outputs and CPU-merges them. Moving meshes remain unsupported.
5. Complete S05–S08 exactly as specified; do not mark the entire plan done based on these initial correctness fixes.

No implementation commits, pushes, GUI launches, window closes, or changes outside Molla have been made for this sensor work.

## S02 increment: bounded asynchronous readback and IMU integration

This increment overlaps the unresolved S01 mesh-cache revision work only for independent IMU/collection infrastructure. It does not mark S01 or S02 complete and does not bypass the remaining geometry requirements.

### Architecture and changes

- `crates/molla-sensors/src/readback.rs`: `DeviceBuffer<T>` retains the originating engine's `Arc<Device>` and queue; there is no public method to replace its inner buffer with an arbitrary foreign buffer. Typed borrowed sources validate active ranges and GPU authority.
- `ReadbackRing` owns 1–64 preallocated staging slots with checked per-slot/total GPU budgets. Each request can group up to 64 mixed channels in a single copy submission. Mapping is asynchronous; `poll` uses `PollType::Poll`, while `wait` is explicitly finite and synchronous.
- Request tickets identify their ring, slot and generation. Metadata includes sensor/sample/revision identity and optional simulation time. Stamps are supplied by callers, not generated by a scheduler.
- Cancellation invalidates tickets immediately but retains staging until GPU mapping completes; `poll_releases` reclaims completed cancelled slots without allocating replacements. This keeps transient GPU memory bounded under repeated cancellation. Reset invalidates outstanding tickets but does not promise immediate physical slot availability.
- Mapping failures make the ring terminal; all tickets are invalidated and further enqueue attempts return an error. A failed ring must be recreated after device/context recovery. The implementation does not unmap resources whose map state is unknown after failure.
- Collected frame bytes are independent of staging reuse, and typed iteration decodes GPU representations without intermediate arrays. Source host mirrors remain stale after collection.
- `ImuSensor::dispatch` submits evaluation/history only and returns borrowed device outputs; it neither allocates staging nor downloads results. GPU consumers must be submitted to the same engine queue before another sample reuses those outputs.
- Existing `ImuSensor::update` remains compatible and uses one lazily allocated staging slot for both channels, instead of separate per-channel staging allocations/waits. Its synchronous wait limit is 30 seconds; the device-only path has no blocking wait.
- `crates/molla-sensors/Cargo.toml`/`Cargo.lock`: enable the already-present bytemuck dependency for typed mixed-channel decoding under the GPU feature. No dependency version was upgraded.
- Added `book/src/sensors/readback.md`, linked from the book summary, and updated the IMU API documentation.

### Discoveries and regression fixes

1. A real cross-device test disproved using wgpu `Device::eq` for provenance across independently constructed instances. In wgpu 27, native `CoreDevice` equality compares its instance-local ID; separate engines can compare equal. A proposed raw `Buffer` owner check using that equality was removed before integration. Engine `Arc` identity now protects the sensor-owned buffers/rings and rejects the foreign-engine IMU call. No broad Engine or storage ownership rewrite was made.
2. Fault injection exposed a cleanup panic from unmapping an empty/unmapped slot, including a destructor panic. Cleanup was redesigned: cancelled transfers retire only after completion; empty transfers never unmap; terminal mapping failures invalidate the ring without trying to unmap unknown state. The explicit failure/timeout regression passes after this change.

### Current targeted evidence

- `sensor_readback_gpu`: mixed partial ranges, type mismatch errors, slot exhaustion, out-of-order collection, 20 warmed reuse iterations, cancellation/reset, empty data, invalid budgets/ranges, cross-device rejection and pending-drop safety. The warmed test records two staging allocations, 22 submissions and exactly 22×52 copied bytes, with zero blocking waits.
- `sensor_device_pipeline_gpu`: IMU output consumed by a compute shader without host download, two queued sample snapshots remain distinct after output reuse, matching tags, and foreign-engine rejection. Eight later synchronous samples reuse one staging allocation and perform eight grouped 32-byte transfers.
- `sensor_imu_gpu`: seven existing/correctness regressions passed after the collection migration (`/tmp/molla-sensors-s02-imu.log`).
- GPU unit fault injection: `readback_mapping_failure_is_terminal_and_pending_wait_is_retryable` passed (`/tmp/molla-sensors-s02-fault.log`). This proves injected mapping-error handling and retryable pending completion, not physical GPU device-loss recovery.
- Targeted logs: `/tmp/molla-sensors-s02-readback.log`, `/tmp/molla-sensors-s02-pipeline.log`.

### Final S02-increment verification, 2026-09-21

- `make test-sensors`: passed **49 GPU integration tests + 1 GPU fault-injection unit test**, **59 GPU-feature host unit tests**, and **3 no-default-features unit + 3 CPU adjoint integration tests**. Log: `/tmp/molla-sensors-s02-all.log`.
- `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'`: passed (`/tmp/molla-sensors-s02-check.log`).
- `make check APP_TARGET='-p molla_viewer'`: passed in 14.29 seconds (`/tmp/molla-sensors-s02-viewer.log`). No viewer was launched and no visual acceptance is claimed.
- Explicit-file rustfmt check through Makefile and `git diff --check`: passed.
- `make docs`: exited 0 and generated HTML, but logged a pre-existing book configuration/tool incompatibility (`output.html.copy-fonts` unknown). This is **not a clean documentation gate**. `book/book.toml` was left unchanged. Generated untracked output was moved to `/tmp/molla-sensors-s02-book-20260921`; log: `/tmp/molla-sensors-s02-docs.log`. Resolve the book/tool compatibility during S08 instead of treating exit status alone as sufficient evidence.

### Limitations at the end of the S02 increment

- Public device-only migration currently covers IMU, not LiDAR/contact/transforms/cameras. The shared ring is not evidence those sensors have been ported.
- IMU kernel bind groups/uniform resources still allocate per dispatch; full warmed-pipeline zero-resource-allocation gates are not achieved.
- Typed sources enforce engine identity for the new sensor-owned storage. Existing raw Model/State inputs remain caller-owned and must belong to the engine; this is not a cross-device audit of every pre-existing kernel.
- Raw wgpu handle users must submit consumers before buffer reuse. Arbitrarily retaining an unsubmitted command buffer does not acquire a sensor-output lease.
- Scene revisions/timestamps are metadata only until the scheduler/scene adapter is implemented. Resetting an IMU does not automatically cancel requests in an external ring; callers must reset the ring for their scene transition.
- CPU channel descriptors and collected bytes still allocate. Physical device-loss recovery, cold/large-workload performance, and allocation counters across every sensor remain unverified.

## S03 increment — GPU closest-hit reduction and device raycast outputs

### Goal and implementation

Remove raycaster intermediate downloads/CPU mesh-hit merging while retaining the synchronous API. This advances S02/S03; it is not full LiDAR pipeline completion.

- `Raycaster::dispatch` validates and uploads CPU rays, runs analytical/static-mesh queries and ordered GPU closest-hit reduction, and returns borrowed `RayDeviceHits`. No internal staging allocation, host download or GPU wait occurs on this path.
- `crates/molla-kernels/src/raycast_merge.rs` and `shaders/raycast_merge.wgsl` hold persistent merge bindings/uniform storage. One invocation handles each ray; mesh passes submit sequentially so scratch/output updates cannot race. The same kernel initializes mesh-only/empty scenes to miss values.
- `Raycaster::cast` wraps dispatch and uses one reusable grouped staging slot. It transfers exactly 40 bytes per active ray across distance, shape, point and normal, independent of mesh count; no host hit-candidate merge remains.
- `RayDeviceHits` exposes active count and typed device channels. Device misses use distance/shape -1, point zero, normal +Y; legacy host misses remain distance infinity/shape None. Raw output ownership has the same submitted-consumer requirement as the IMU path.
- Exact ties preserve existing precedence: analytical shapes before meshes; ascending shape index within each category. Mesh normals oppose the incident ray. Tests exercise both analytical/mesh ties and duplicate meshes.
- The originating engine is checked with retained `Arc` identity. Analytical model mirrors must be device-current; raw Model/State ownership still remains the caller's responsibility.
- Documentation: `book/src/sensors/raycaster.md` includes device consumption, selective collection, byte budgets, tie/miss semantics and explicit remaining limitations. `SENSROS.md` is a full requested snapshot, not the authoritative status tracker.

### Regression evidence

- `sensor_raycast_device_gpu`: device query → compute consumer → grouped readback, 65 rays across workgroups, mixed sphere/two-mesh occlusion, clipped ranges, out-of-order snapshots after sensor reuse, static-mesh-only initialization, empty scenes/batches, exact ties, foreign-engine rejection and stale model-mirror rejection.
- A warmed synchronous sequence with 65, 1, 0 and 3 active rays in a 128-ray sensor uses exactly one staging allocation, three copy submissions and `(65 + 1 + 3) * 40` bytes. The preceding device-only samples leave all internal collection counters zero.
- A 17-mesh separated-instance regression checks that every merge retains the correct shape identity across reused uniform/scratch buffers, and that three scans perform only three host copy submissions.
- Existing independent triangle oracle, scaled-direction, cone, LiDAR grid/aerial/sub-ray, and IMU regressions remain required; see the final command evidence below.

### Like-for-like warm release measurements

RTX 4080 / Vulkan, original primitive-only synchronous benchmark, unchanged rays/scenes. Repeated command: `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_raycast -- --baseline sensors-s00'` (before adding the separate mesh group). Future reruns should add the `sensor_raycast_gpu/` filter when comparing this baseline.

| Workload | Original S00 interval | New repeated interval |
|---|---|---|
| 1,000 rays / 100 spheres + plane | [3.7356, 3.8606, 4.0876] ms | [109.93, 112.94, 116.01] µs |
| 10,000 rays / 1,000 spheres + plane | [5.3464, 7.1687, 8.0528] ms | [700.96, 787.09, 864.51] µs |

These are Criterion estimate intervals, not p50/p95 or pure GPU time. Both paths include upload, query and host-visible completion; input ray generation is outside timing. The new path groups four copies and uses bounded pooled collection, so these improvements do not measure acceleration or GPU ray generation. They are not whole-simulator speedups. Large-case variance remains substantial.

The first run overlapped this session's regression checks; `/tmp/molla-sensors-s03-bench.log` is diagnostic only. The repeat above ran without this session's GPU tests (`/tmp/molla-sensors-s03-bench-isolated.log`); it was not an exclusive-machine reservation. Full p95 budgets/matrix remain pending.

### Mesh scaling measurements (diagnostic, not an acceptance budget)

Added `sensor_raycast_mesh_gpu` to the raycast benchmark: 1,000 rays and 1/16/64 static instances, two triangles per instance, warm caches. It measures both dispatch plus explicit device completion and full host collection; neither is pure GPU timestamp time. Command: `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_raycast -- sensor_raycast_mesh_gpu --save-baseline sensors-s03-merge'`.

| Instances | Device-completed interval | Host-completed interval |
|---|---|---|
| 1 | [0.6360, 1.6463, 2.3363] ms | [0.2020, 0.2170, 0.2313] ms |
| 16 | [1.1950, 1.2571, 1.3356] ms | [1.2543, 1.2738, 1.2880] ms |
| 64 | [4.7203, 4.8101, 4.9179] ms | [4.7972, 4.8953, 5.0506] ms |

The anomalously slow/variable one-mesh device case was repeated rather than accepted as representative. The repeat was also inconsistent: one-mesh device [4.5521, 4.9138, 5.0950] ms / host [1.7596, 2.3435, 2.7494] ms; 16-mesh device [2.4892, 3.1311, 3.6309] ms / host [0.6803, 0.6869, 0.6978] ms. These short-run wall-clock figures do **not** establish stable mesh latency or prove device-only dispatch slower than collection. Diagnose the wait/scheduling variance and repeat with longer controlled measurements before S00/S08 acceptance. No pre-change mesh baseline exists, so no mesh speedup is claimed.

Logs: `/tmp/molla-sensors-s03-mesh-bench.log`, `/tmp/molla-sensors-s03-one-mesh-repeat.log`. The repeat filter `sensor_raycast_mesh_gpu/.*/1` intentionally also matches 16. Mesh count still increases per-mesh query/merge submission work; shared acceleration and encoding remain necessary regardless of host-copy elimination.

### Final increment verification, 2026-09-21

- `make test-sensors`: passed **53 GPU integration tests + 1 GPU fault-injection unit test**, **59 GPU-feature host unit tests**, **3 no-default-features unit + 3 CPU adjoint integration tests**. Log: `/tmp/molla-sensors-s03-final-tests.log`.
- `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'`: passed, including the expanded mesh benchmark (`/tmp/molla-sensors-s03-final-check.log`).
- `make check APP_TARGET='-p molla-kernels -p molla-sensors --no-default-features'`: passed (`/tmp/molla-sensors-s03-feature-check.log`).
- `make check APP_TARGET='-p molla_viewer'`: passed (`/tmp/molla-sensors-s03-viewer.log`); no visual acceptance was attempted.
- Changed Rust files passed `make fmt-files FMT_ARGS=--check FILES='...'`; `git diff --check` passed. Source inspection confirms no `.download()` or host running-best merge in `raycast.rs`.
- All this increment's build/test/benchmark processes reached terminal status. No commits or pushes were made.

### Remaining work and discoveries

1. GPU merge removes host synchronization but retains two queue submissions per static mesh and per-query resource construction in the existing primitive/mesh wrappers. The merge wrapper itself retains its binding group/uniform; this does not establish the full allocation-free gate.
2. Queue writes to a reused uniform are safe here because every merge is submitted before the next write. A future shared-encoder migration needs distinct/dynamic uniform ranges or equivalent ordered copies, not several pre-submit writes to one uniform.
3. Ray layout/normalization still happens on CPU; LiDAR first/last sub-ray aggregation remains host-side. Add GPU terrestrial/aerial layouts and scan reduction next, preserving flattened ordering and independent mounts.
4. Mesh geometry is still world-space, static-only and manually invalidated. Scene revisioning, moving instances, shared top-level acceleration and explicit unsupported-geometry reporting remain S01/S04 requirements.
5. Camera/contact/transform migration, scheduling, processing, full performance acceptance and visual demos remain unfinished. No Gearbox edits, asset edits, viewer launch or window close occurred in this increment.

## S03 increment — device-generated multi-scan LiDAR

### Architecture and scope

- Added `LidarBatch` in `crates/molla-sensors/src/lidar_batch.rs`, exported with `LidarScan`, `LidarPattern`, `LidarScanRange` and `LidarDeviceBatch`. A batch can mix terrestrial/aerial descriptors while keeping each scanner's origin/frame, footprint/grid, range, divergence, sub-ray count and ID independent.
- Only 128-byte descriptors are uploaded per scan. GPU threads generate terrestrial angular grids or aerial Halton origins/directions, then normalized Fibonacci-cone sub-rays. No host ray vectors are built in the new API. Legacy CPU layouts remain independent reference/convenience paths.
- Descriptor binary search maps flat GPU ray/pulse indices to scan ranges. Row/column/pulse/sub-ray ordering is preserved; CPU metadata back-pointers are retained per descriptor, not per ray.
- The existing raycaster accepts internally generated device rays and performs GPU nearest-hit merging. A final GPU pass clips hits to each descriptor's own range and produces first/last ray indices plus valid sub-ray counts. It queries with the largest batch range, then clips on device; shorter-range scans never inherit another scanner's range.
- Added retained generation/reduction pipelines and bindings in `crates/molla-kernels/src/lidar_scan.rs` / `shaders/lidar_scan.wgsl`. Descriptor vectors and output buffers are preallocated. The older primitive/mesh query wrappers still allocate per dispatch; this is not the complete S02 allocation-free gate.
- `RayDeviceHits` now also exposes typed origins/directions. GPU generation marks those host mirrors stale. There is no dirty-state bypass and no conversion of generated rays back through host storage.
- First-return ties keep the first sub-ray; last-return ties keep the last, matching Rust's legacy `min_by` / `max_by` behavior. These are selections across divergent sub-rays, not physical multi-return along one ray.
- Fixed explicit budgets: 1–1024 scan descriptors, 1–1024 effective sub-rays per pulse, ray capacity constrained by device storage/workgroup limits. Requests beyond capacity return errors before generation, without silent truncation or implicit chunking. Empty batches enqueue no GPU work.
- GPU consumers or external `ReadbackRing` snapshots may select final channels. Submit consumers/copies before the next dispatch; no scheduler/timestamp/scene-ownership guarantees were added beyond existing caller-managed metadata.
- Added book chapter `book/src/sensors/lidar_batch.md` and completed-layout benchmark `bench_sensor_lidar`, covering 1k/10k/100k/1M rays, legacy host completion, GPU-layout host completion, and device-only completion.

### Discoveries and fixes

1. The high-sub-ray oracle exposed accumulated azimuth/trigonometry error: direct f32 `i * golden_angle` failed the predeclared `3e-5` direction tolerance at sub-ray 770 of 1024. Red log: `/tmp/molla-sensors-s03-layout-all.log`. The shader now reduces the azimuth using split f32 constants and fused multiply-add residuals before trigonometry. The same 1024-ray oracle passes without loosening tolerance (`/tmp/molla-sensors-s03-layout-green.log`). This was verified on the current Vulkan GPU, not every backend's fma implementation.
2. Legacy `ScanMetadata::validate` documented nonparallel frame axes but checked only their magnitudes. Parallel axes could silently select an arbitrary fallback frame. Validation now rejects parallel/antiparallel normalized axes; new host regressions cover X, Y and oblique directions. GPU metadata additionally rejects non-finite f32 conversions, range underflow and sub-ray budget excess.

### Targeted regression coverage

- `lidar_pipeline_gpu` compares every generated ray, hit ID, distance, point and normal against legacy CPU layouts/queries in a mixed plane/mesh/sphere scene. It exercises terrestrial skew-frame grids, aerial Halton scans, divergence, different per-scan ranges, zero-divergence ties and per-pulse reductions.
- Reference tolerances: `2e-5` for ordinary generated origins/directions, `1e-3` for hit distance/point/normal in the fixture; `3e-5` for directions across all 1024 sub-rays at divergence 0.1 rad. Hit/miss and shape IDs match exactly away from fixture visibility boundaries.
- A 70-mount case spans workgroups, preserves 70 independent IDs/settings, covers effective K=1/2, and checks out-of-order snapshots after buffer reuse. All-miss, empty, fewer-scan reuse, wrong-device and capacity errors are exercised.
- The new API's internal readback counters remain zero, while a mixed four-scan dispatch uploads exactly 512 descriptor bytes. External final collection transfers only requested active channels.
- Named scripts and final test/benchmark results are recorded below when terminal; these targeted regressions alone are not whole-plan completion.

### Completed-layout benchmark evidence

Command: `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_lidar -- --save-baseline sensors-s03-layout'`. RTX 4080 / Vulkan, release profile, one terrestrial scan on one static plane, K=1. The two host modes return the same `Vec<RayHit>` representation and collect all four hit channels; the device-completed mode explicitly waits without collection. The GPU-layout path additionally performs sub-ray selection even though these timings do not collect selection channels.

| Rays | Legacy CPU layout + host hits | GPU layout + host hits | GPU layout + device completion |
|---|---|---|---|
| 1,000 | [257.14, 263.51, 277.24] µs | [298.70, 316.29, 329.68] µs | [177.93, 183.04, 190.19] µs |
| 10,000 | [851.91, 914.67, 988.20] µs | [366.18, 376.85, 393.47] µs | [191.58, 232.65, 275.65] µs |
| 100,000 | [7.9842, 8.2097, 8.5119] ms | [1.5578, 1.5989, 1.6456] ms | [194.22, 206.44, 224.31] µs |
| 1,000,000 | [77.177, 78.384, 79.443] ms | [35.029, 35.692, 36.892] ms | [403.45, 416.89, 439.11] µs |

Log: `/tmp/molla-sensors-s03-layout-bench.log`. These are Criterion estimate intervals from a first completed run, not p50/p95, cold-start costs or pure GPU timestamps. The 1M CPU case automatically lengthened collection beyond its 3-second target to obtain ten samples. No safety limit was bypassed to run 1M rays.

**Small-case regression is not waived:** the new optional GPU-layout host path is about 20% slower at 1k rays in this run. Legacy synchronous APIs were not switched to it. Separate generation/query/reduction submissions are still present; shared encoding/resource reuse and controlled repeats are needed before performance acceptance. Large-case improvements do not prove mesh-rich, many-mount or whole-simulator performance, and this table is not the full S00/S08 matrix.

### Final layout-increment verification, 2026-09-21

- `make test-sensors`: passed **57 GPU integration tests + 1 GPU fault-injection unit test**, **61 GPU-feature host unit tests**, and **3 no-default-features unit + 3 CPU adjoint integration tests** (`/tmp/molla-sensors-s03-layout-verified.log`).
- The four `lidar_pipeline_gpu` tests include a 1M-ray dispatch with explicit f64 plane-reference probes at the first/last rays, workgroup edges and 65,535/65,536 indices. First/last pulse outputs at those indices match exactly; 128 bytes of sampled output are collected, while the scan descriptor upload remains 128 bytes. This is boundary coverage, not a full million-hit CPU oracle comparison.
- `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'`: passed without warnings, including the new benchmark (`/tmp/molla-sensors-s03-layout-verified-check.log`).
- `make check APP_TARGET='-p molla-kernels -p molla-sensors --no-default-features'`: passed (`/tmp/molla-sensors-s03-layout-features.log`).
- `make check APP_TARGET='-p molla_viewer'`: passed (`/tmp/molla-sensors-s03-layout-viewer.log`). No visual acceptance or viewer launch performed.
- Explicit-file formatting check through Makefile and `git diff --check`: passed. No unrelated files formatted, commits made, pushes performed, or Gearbox/assets edited. All test/benchmark handles from this increment are terminal.

### Next work

Continue S01/S02/S04: revisioned scene ownership, moving meshes/shared acceleration, reusable query bindings and composable encoding. Then complete remaining sensors, scheduling, camera/processing migration, visual demos and full performance acceptance. S03 core behavior is verified for the documented supported scene types; the sensor plan as a whole remains incomplete. Keep S09 separately unauthorized and S10 deferred.

## S04 scene foundation and verification recovery, 2026-09-21

- `scene.rs` owns shape-only geometry snapshots, compiled local mesh BVHs shared across instances/query sensors, explicit unsupported-geometry policy, vertex/triangle budgets and revision-scoped shape resolution. Triangle and heightfield conversion are supported; ConvexHull, RoundCylinder and SDF reject by default or are explicitly reported/skipped. Replacement is atomic on validation failure.
- Mesh traversal composes live device body poses with shape-local transforms, traverses local geometry and rotates normals back to world. Legacy raw-model queries compare mesh contents once per unique mesh per query and rebuild on edits; owned scene queries avoid that comparison. Pose-only body motion does not rebuild mesh BVHs.
- Mesh upload validates indices, complete leaf/triangle coverage, finite containing bounds, acyclic connected trees and the 64-level traversal bound. Geometry host tests passed 123 tests after the final validation changes (`/tmp/molla-sensors-s04-geometry-host-final.log`).
- The initial full sensor suite passed 61 GPU integration tests plus one fault-injection unit test, with host/feature lanes also passing (`/tmp/molla-sensors-s04-all.log`). A later scene rerun passed three tests but failed creating a second GPU device: the GPU was nearly full and the kernel logged NVIDIA out-of-memory errors. This was not counted as a passing run.
- Stopped only the owned benchmark process; no user applications were closed. `/tmp/molla-sensors-s04-bench.log` and partial `sensors-s04-scene` Criterion data are contaminated/aborted and must not be used as performance evidence.
- After GPU utilization returned to zero with available memory, the exact scene suite rerun passed all four tests, including model replacement and foreign-engine checks (`/tmp/molla-sensors-s04-scene-recheck.log`); geometry narrow-phase GPU tests passed three tests (`/tmp/molla-sensors-s04-geometry-recheck.log`).
- Final all-target sensor/geometry GPU-feature checks and no-default-feature checks passed (`/tmp/molla-sensors-s04-final-check.log`, `/tmp/molla-sensors-s04-features.log`). Viewer check passed without launch in the earlier increment.
- S04 remains incomplete: no top-level scene acceleration/refit, stable remapping handles, independent shape-local pose updates, shared query encoder/resource retention or clean many-instance benchmark acceptance yet. Body pose freshness is separate from geometry revision and still requires caller sample identity.

## S05 contact and frame device paths, 2026-09-21

### Goal and instructions

Continue `SENSORS.md` implementation, migrating the remaining auxiliary sensor outputs onto S02's device buffers and bounded optional collection. Preserve existing synchronous entry points and unrelated work; use Makefile gates. The previous documentation turn updated the requested snapshot, not implementation completion. No Gearbox/assets, application windows, commits or pushes were touched.

### Accomplished

- `ContactSensor::dispatch` now returns borrowed typed count/depth/normal/centroid channels with no host collection. `FrameTransformSensor::dispatch` similarly exposes world/relative positions and rotations, plus the active target range excluding the internal reference slot. Device-only calls allocate no internal staging buffers.
- Existing `update` methods wrap dispatch plus one grouped bounded readback, retaining one staging slot across updates. Contacts collect 40 bytes per body; frames collect 64 bytes per requested target and omit the reference slot. Both expose internal readback counters, while caller-owned rings can collect selected channels independently.
- Constructors reject body/target capacities exceeding indexing, binding or dispatch limits before sensor buffer allocation. Frame mounts validate finite f32 positions and unit rotations. Dispatch rejects foreign engines, stale input mirrors, incompatible body counts and inconsistent contact-array lengths. Inputs remain raw simulation buffers: valid CSR contents and same-engine provenance are caller preconditions, not validated by downloading them.
- `Contacts::clear()` now produces zero readings even when its old CSR remains allocated. Empty-body contact batches produce empty outputs without collection. This closes a stale-result path in the legacy wrapper.
- Contact docs/shader comments now correctly describe a-side negative, b-side positive normals and explicitly reject interpreting unweighted normal sums or penetration depths as measured force. Frame docs correct the former cross-invocation reference dependency description: each invocation computes its own reference pose.

### Verification

- `make test-gpu P=molla-sensors T=sensor_aux_device_gpu A='--test-threads=1'`: **3 passed**, including 70 bodies/targets spanning workgroups, GPU-authoritative inputs with stale host mirrors, multiple pending samples collected in reverse order, a downstream GPU buffer copy, per-target rigid-transform CPU oracles, repeated staging reuse, zero-contact clearing, capacity/invalid-input/foreign-engine rejection. `/tmp/molla-sensors-s05-aux.log`.
- `make test-sensors`: **64 GPU integration tests + 1 GPU fault-injection unit test**, **61 GPU-feature host unit tests**, **3 no-default-feature unit + 3 CPU adjoint integration tests** passed. Existing contact and frame golden tests are included. `/tmp/molla-sensors-s05-all.log`.
- `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'`, the no-default-features sensor check, and `make check APP_TARGET='-p molla_viewer'` passed. Logs: `/tmp/molla-sensors-s05-final-check.log`, `/tmp/molla-sensors-s05-features.log`, `/tmp/molla-sensors-s05-viewer.log`.
- Explicit-file formatting checks and `git diff --check` passed. No viewer launched, no new timing claims, no benchmarks counted from the GPU-memory-contaminated earlier run.

### Discoveries and remaining work

1. A zero active contact count is authoritative even if old CSR entries remain: zero the output channels rather than tracing old adjacency. GPU-authoritative nonempty inputs must stay resident, without `host()` validation/download.
2. Frame output buffers include the reference slot, but public target collection must exclude it. The device API therefore exposes an explicit range, and counters verify only requested targets are transferred.
3. S05 is not complete: simulation-time scheduling, sample timestamps, pause/reset semantics and stable body remapping remain. Same-sized replacement is not automatically detectable from raw State/Contacts; recreate these fixed-index sensors after remapping.
4. S02 remains incomplete: kernel uniforms/bind groups still allocate per dispatch, encoder sharing is pending, and borrowed results do not lease output storage for unsubmitted command buffers. These limitations are documented, not waived.
5. Next: implement independent simulation-time scheduling and its lifecycle tests; continue persistent/shared query resources, top-level acceleration, camera/processing migrations, visual demos and controlled performance acceptance. S09 remains unauthorized and S10 deferred.

### Relevant files

- `crates/molla-sensors/src/{contact,frame_transform}.rs` — device outputs, validation and grouped sync collection.
- `crates/molla-sensors/src/lib.rs` — public device-reading exports.
- `crates/molla-sensors/tests/sensor_aux_device_gpu.rs` — residency, reuse, multi-target and validation regressions.
- `crates/molla-kernels/shaders/contact_aggregate.wgsl` — corrected geometric-output comments only.
- `book/src/sensors/{contact_sensor,frame_transform}.md` — current APIs, ranges, ownership and remaining limitations.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — checked S05 device-output tasks and evidence; full plan still active.

## S05 simulation-time scheduler, 2026-09-21

### Goal and constraints

Continue the full `SENSORS.md` goal. The previous goal turn was progress: auxiliary device paths and their verified regressions changed authoritative code. This increment implements the independent-rate scheduling requirement, not a replacement for the remaining acceleration/camera/processing work. No Gearbox, physics/tyre/soil behavior, authored assets, or user windows changed. Makefile gates and patch-only edits were used; no commits or pushes.

### Decisions and discoveries

- Use integer `Duration` timestamps/periods with nanosecond resolution. This keeps cadence exact at large epochs without accumulated floating-point deadline drift. External physics clocks must convert cumulative timestamps consistently; the scheduler does not redefine physics timesteps.
- A call consumes one coherent current snapshot and emits at most one request per configured sensor. Late arrivals report skipped deadlines rather than inventing historical states. At most 4096 schedules are supported, with retained output storage and no unbounded catch-up queue.
- Requests are issued in configuration order, with independent IDs, periods and phases. Phase zero starts at the epoch. Pause suppresses delivery without shifting cadence; reset/replacement increments generation and resets elapsed intervals while preserving monotonic sample IDs and paused state.
- Issuing a request is not an acknowledgement of successful dispatch. `elapsed` is since the prior request; missed dispatch/history updates require caller bookkeeping or IMU reset/re-priming. Readback backpressure does not undo successful IMU history. These distinctions are explicit in the public contract.
- Pending tickets retain their `ScheduledSample`. `is_current` checks scheduler identity/generation/scene revision; `matches_readback` checks collected metadata. Both are required, including reset to the same scene revision. Exact integer timestamps remain in the retained sample while compatible readback stamps use f64 seconds.
- The scheduler is host control logic available without GPU features. `molla-core` is now an unconditional, default-features-disabled dependency for the shared Result/Error convention; no GPU dependency is enabled in the host-only lane.

### Accomplished

- Added `SensorSchedule`, `ScheduledSample` and `SensorScheduler` public APIs in `scheduler.rs`, including atomic invalid-input/overflow handling, backward-time rejection, schedule replacement and sample metadata helpers.
- Added seven non-ignored host tests covering phases/rates, late current-state sampling, pause/reset/replacement/foreign scheduler rejection, render-rate independence, empty/invalid capacities, transactional failure handling, large epochs, retained allocation and sample/generation exhaustion.
- Added `sensor_scheduler_gpu.rs`: real GPU integration with independently scheduled 100 Hz IMU, 200 Hz frames and 50 Hz contacts. Different render cadences produce matching timestamps/readings; 16 pending readings collected in reverse order preserve identity, while all sensor-internal readback counters remain zero. Another test resets with the same scene revision, re-primes IMU history and rejects an obsolete pending generation while retaining correct old/new snapshot values.
- Added book guidance for CPU coherent snapshots, GPU queue ordering, independent mount batches, elapsed-time failure handling, ring backpressure and explicit reset/recreation responsibilities. The scheduler does not acquire producer locks, copy physics snapshots, reset sensors or cancel GPU work implicitly.

### Verification

- `make test-sensors`: **66 GPU integration tests + 1 GPU fault-injection unit test**, **68 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. `/tmp/molla-sensors-s05-scheduler-all.log`.
- Targeted scheduler GPU lane: **2 passed** (`/tmp/molla-sensors-s05-scheduler-gpu.log`). Initial host and all-target checks also passed (`/tmp/molla-sensors-s05-scheduler-host.log`, `/tmp/molla-sensors-s05-scheduler-check.log`).
- Final `make check APP_TARGET='-p molla-sensors --features gpu --all-targets'` and `make check APP_TARGET='-p molla_viewer'` passed (`/tmp/molla-sensors-s05-scheduler-final-check.log`, `/tmp/molla-sensors-s05-scheduler-viewer.log`). Explicit-file formatting and `git diff --check` passed. All execution handles from this increment are terminal.
- No viewer launch/visual acceptance, no timing benchmark, and no whole-plan completion claim. Existing IMU multi-mount/lever-arm/angular-acceleration, GPU-step and reset regressions remained green.

### Next steps

S05's core API/testing contract is verified; this does not waive S02's missing retained dispatch resources, shared command encoding and output leases. Continue those shared execution foundations plus S04 top-level acceleration/stable handles, then S06 cameras, S07 processing and S08 examples/visual/performance acceptance. S00/S01 broader matrices remain open; S09 remains unauthorized and S10 deferred. Wire scheduling into viewer examples in S08 rather than implying the new API already changes application behavior.

### Relevant files

- `crates/molla-sensors/src/scheduler.rs` — bounded integer-time cadence, generations, metadata and host tests.
- `crates/molla-sensors/src/lib.rs` / `Cargo.toml` — public ungated scheduler exports and CPU error dependency.
- `crates/molla-sensors/tests/sensor_scheduler_gpu.rs` — scheduled real-GPU physics and pending-generation regressions.
- `book/src/sensors/scheduler.md` / `book/src/SUMMARY.md` — API, failure/lifecycle and snapshot contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — S05 core gate evidence and remaining full-plan scope.

## S02 retained queries and single-submission LiDAR, 2026-09-21

### Goal and constraints

Continue the full sensor plan by removing repeated query binding creation and per-stage/mesh queue submissions. The previous turn was progress: verified scheduling code and tests were added. All edits remain in sensor/shared-query code, associated documentation and tests; no Gearbox, unrelated physics/assets, application windows, commits or pushes were touched.

### Implemented architecture

- Primitive raycast and mesh-BVH kernels expose retained binding sets plus encode-only prepared methods. Existing low-level convenience dispatch methods remain compatible and still allocate temporary bindings. Public `Raycaster` uses the prepared path.
- Each query caches two primitive input-buffer variants and two variants per mesh instance, matching common physics ping-pong states. Cached identity includes geometry/model/pose/ray/output buffers. Pose contents, range and active ray count can change without rebinding; actual buffer replacement prepares new bindings. Per-instance cache entries absent from the next nonempty query are pruned.
- Mesh merges have separate retained parameter buffers per instance. Encoding many meshes with repeated writes to one shared uniform before submission would otherwise make earlier commands see later IDs/parameters. Distinct binding sets preserve ordered nearest-hit ties and shape identity.
- `Raycaster` encodes primitive traversal or miss initialization, all nonempty mesh traversals and their merges into one command buffer, then submits once. `LidarBatch` additionally encodes GPU ray generation and pulse reduction into that same command buffer: **one submission per nonempty LiDAR batch**, none for empty batches. This replaces the prior stage/mesh-dependent submission count; it is structural evidence, not measured latency.
- `RayQueryStats` reports encoded queries, direct query submissions and primitive/mesh/merge preparations. `LidarBatchStats` adds submission counts; nested query submissions remain zero. A primitive/mesh preparation creates one uniform and four bind groups; a merge preparation creates one uniform and one group. Counters exclude construction, geometry uploads, command buffers, host temporary arrays and wgpu's internal upload staging.

### Verification and discoveries

- `sensor_raycast_device_gpu` now has five GPU tests. The added regression alternates two physics state buffers, varies ray count/range, replaces owned mesh geometry, and verifies correct hit/miss/shape/distance outputs while preparation counts stop changing after each warm-up. Sixteen queries produce sixteen submissions, four primitive preparations, six mesh preparations and two merge preparations, with zero sensor-internal collection.
- Existing 17-mesh shape-identity/occlusion, tie ordering, moving-instance triangle oracles and queued snapshot-reuse tests still pass, exercising distinct uniforms inside combined command buffers.
- LiDAR regressions assert one submission for the million-ray batch, no nested query submission, and unchanged binding preparations across independent multi-mount range/count changes. Empty/invalid requests do not add successful submissions.
- `make test-sensors`: **67 GPU integration tests + 1 GPU fault-injection unit test**, **68 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. Final log: `/tmp/molla-sensors-s02-shared-final-all.log`; targeted query-cache log: `/tmp/molla-sensors-s02-shared-cache.log`.
- Geometry GPU narrow-phase/raycast suite: **3 passed** (`/tmp/molla-sensors-s02-shared-geometry.log`). Geometry no-default-feature host suite: **120 passed** (`/tmp/molla-sensors-s02-shared-geometry-host.log`).
- Geometry GPU-feature host suite also passed **123 tests**, including BVH validation boundary tests (`/tmp/molla-sensors-s02-shared-geometry-host-gpu.log`). All execution handles from this increment are terminal.
- Sensor/geometry GPU-feature all-target check, kernel/geometry/sensor no-default-feature check and viewer check passed (`/tmp/molla-sensors-s02-shared-final-check.log`, `/tmp/molla-sensors-s02-shared-features.log`, `/tmp/molla-sensors-s02-shared-viewer.log`). Explicit-file formatting and `git diff --check` passed. Formatting the changed geometry export file also sorted its adjacent mesh module declarations without changing behavior.
- GPU utilization reached 100% with roughly 14 GiB used by concurrent applications. No latency benchmark was attempted under that load. The prior 1k LiDAR regression is **not** declared fixed by structural improvements alone; controlled repeats remain required.

### Remaining work

This is not the complete S02 gate. Prepared low-level encoding requires submission before changing a reused parameter buffer; higher-level cross-sensor encoding and leased output versions are still missing. IMU/contact/frame/camera kernels need retained bindings too. More than two rotating query input-buffer sets cause bounded cache churn, and retained per-instance resources still scale with instance count. S04 top-level acceleration/refits/stable handles, S06 cameras, S07 processing and S08 visual/performance acceptance remain; S09 is unauthorized and S10 deferred.

### Relevant files

- `crates/molla-kernels/src/{raycast,raycast_merge,lidar_scan}.rs` — prepared bindings and shared-encoder stages.
- `crates/molla-geometry/src/{mesh_bvh_raycast_gpu,lib}.rs` — prepared mesh instance query/export.
- `crates/molla-sensors/src/{raycast,lidar_batch,lib}.rs` — bounded caches, one submission and instrumentation.
- `crates/molla-sensors/tests/{sensor_raycast_device_gpu,lidar_pipeline_gpu}.rs` — warm reuse, replacement and submission regressions.
- `book/src/sensors/{raycaster,lidar_batch,scene}.md` — current execution path and explicit remaining limitations.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — progress evidence without whole-plan completion.

## S02 copied device snapshots and tracked consumer lifetime, 2026-09-21

### Goal and constraints

Continue the full `SENSORS.md` goal by protecting deferred GPU consumers from producer output reuse. The preceding goal turn was progress: the query/LiDAR execution path and verified cache/submission regressions changed authoritative source. This increment preserves existing borrowed APIs and unrelated work. Makefile tasks were used throughout; no Gearbox, soil/tyre/vehicle behavior, assets, application windows, commits or pushes changed.

### Architecture and discoveries

- `DeviceSnapshotPool` owns 1–64 fixed-budget device buffers. It copies selected active typed channels into storage-aligned ranges without host download/wait. Each accepted immutable `DeviceSnapshot` retains its version; full occupancy returns explicit backpressure, not implicit eviction or replacement allocation.
- Reuse requires both CPU snapshot owners and tracked GPU occupancy tokens to be released. Captures, forwarding copies, readbacks and `DeviceCommandBatch` consumers retain tokens through queue completion. Unsubmitted batches retain their snapshot owners; dropping them discards commands and releases ownership. Failed/panicking encoding closures poison submission rather than running partial commands.
- Completion callbacks retain **lightweight tokens, not SnapshotSlot/device/buffer owners**. A GPU unit regression proves slot owners can be dropped before polling completion, avoiding callback ownership of the GPU resource graph. Native submitted commands retain the actual resources they use.
- Snapshot channels carry checked element types, active ranges, storage binding offsets and readback descriptors. ReadbackRing now propagates optional source occupancy tokens through copy completion, including cancelled/reset readback requests. Ordinary unleased sources preserve prior behavior.
- Pools reset generation without stealing occupied slots. Capture IDs remain monotonic; old snapshots remain readable but `is_current` rejects obsolete/foreign generations. Pool drop does not invalidate outstanding CPU leases. Capacity growth is explicit new-pool creation; retained old versions still count toward application memory.
- This is a copy-based lifetime option, not zero-copy direct sensor slots. Nonempty capture adds a device copy and queue submission. Existing borrowed output views are not made persistent automatically. Raw wgpu handles outside managed batches require the caller to retain a snapshot through completion and must not mutate snapshot storage.

### Accomplished

- Added `DeviceSnapshotPool`, configuration/statistics, typed `SnapshotChannel`, cloneable `DeviceSnapshot` and completion-tracked `DeviceCommandBatch` APIs under GPU feature gating.
- Added checked budgets, channel counts/types/ranges, device provenance, timestamp validation, empty capture behavior, bounded nonblocking release polling, terminal observed poll errors, reset semantics and poisoned batch handling.
- Added five GPU integration tests: mixed channel packing/readback and allocation reuse; clone/reset/backpressure behavior; a real shader consuming an old raycast result after producer reuse and delayed submission; cancellation/drop/error/panic paths; empty/invalid/foreign-device rejection; snapshot forwarding and multi-pool consumers surviving pool-owner drop. Forwarding and consumer scenarios share one of these five tests where appropriate.
- Added host alignment validation and an ignored GPU callback-owner regression. Added book guidance for lifecycle, CPU/GPU ownership boundaries, explicit copy cost, metadata retention and remaining direct-slot work.

### Verification

- `make test-sensors`: **72 GPU integration tests + 2 GPU unit tests**, **69 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. The GPU unit cases are existing readback fault injection and the new callback-owner lifetime regression. `/tmp/molla-sensors-s02-leases-final-all.log`.
- The final snapshot suite rerun passed **5 tests**, including the added caught-panic poison assertion (`/tmp/molla-sensors-s02-leases-verified-gpu.log`). All existing sensor/readback numerical and authority regressions remained green.
- GPU-feature all-target sensor check passed (`/tmp/molla-sensors-s02-leases-verified-check.log`), and viewer check passed (`/tmp/molla-sensors-s02-leases-viewer.log`). The host-only feature lane ran in `make test-sensors` and the new module is GPU-gated.
- Explicit-file formatting and `git diff --check` passed. All execution handles from this increment are terminal. No viewer launch/visual acceptance, performance benchmark or physical device-loss recovery test was performed.

### Remaining work

S02 remains incomplete: producer-side cross-sensor encoding, direct dispatch into leased output slots and retained resources for IMU/contact/frame/camera kernels still need work. The pool bounds accepted result versions, not all application-owned raw command buffers or process-wide GPU memory. Command encoders/callback bookkeeping are not claimed allocation-free. Real device-loss/recovery remains unproven; observed poll failure is terminal and requires a new engine/pool.

Continue shared execution/resource work, then S04 top-level acceleration/refits/stable handles, S06 camera migration, S07 GPU processing and S08 integration/visual/performance acceptance. S00/S01 broader validation matrices remain open; S09 stays unauthorized and S10 deferred.

### Relevant files

- `crates/molla-sensors/src/snapshot.rs` — fixed-budget copied versions, typed bindings and tracked consumer commands.
- `crates/molla-sensors/src/readback.rs` — typed channel metadata sharing and optional GPU source-lease propagation.
- `crates/molla-sensors/src/lib.rs` — GPU-gated public exports.
- `crates/molla-sensors/tests/sensor_snapshot_gpu.rs` — version lifetime, consumers, reset/drop/failure and bounds regressions.
- `book/src/sensors/{snapshot,readback}.md` / `book/src/SUMMARY.md` — ownership protocol, examples, limits and navigation.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — partial S02 scope and verified evidence without full-plan completion.

## S02 retained auxiliary bindings and shared low-level encoding, 2026-09-21

### Goal and instructions

Continue `SENSORS.md`, not merely describe its existing snapshot. The previous goal turn was no progress toward implementation: it only confirmed `SENSROS.md`. Current sources confirmed IMU/contact/frame dispatch still recreated GPU uniform buffers and bind groups, so this independent S02 work proceeded. Makefile tasks and patch tools were used. Unrelated viewer/tyre/FULL_COMPACT work, Gearbox, assets and application windows were left untouched; no commits or pushes.

### Architecture and discoveries

- `ImuBindings`, `ContactAggregateBindings` and `FrameTransformBindings` retain cloned buffer identities, storage bind groups and a 16-byte parameter buffer. A private kernel helper shares resource construction/encoding; existing convenience kernel APIs still prepare temporary resources and preserve submission/dirty-state behavior.
- Sensor wrappers retain the two most recently used bound-buffer variants, with a preallocated two-entry cache. Matching every bound buffer supports physics ping-pong, detects replaced pose/contact/model storage and reuses resources when only contents or `dt` change. A third rotating variant can cause bounded cache churn; this is not an unbounded identity registry.
- Prepared kernel encoding records into a caller-owned encoder without submission. Queue-written parameters must not be rewritten until their earlier commands have been submitted. Different pending parameter sets need distinct bindings. Low-level callers own length/index/CSR validity, buffer authority and IMU history sequencing.
- IMU evaluation still precedes the history copy in one submission; priming only copies history. Reset invalidates history while retaining reusable bindings and lifetime counters. Contact clearing does not prepare bindings, and zero-body batches do not dispatch.
- `SensorDispatchStats` counts wrapper submissions, evaluations, history copies, output clears and binding preparations. Each preparation creates one uniform buffer plus three bind groups (four for IMU). Counters exclude constructor resources, command encoders, driver upload staging and readback; they are not a claim that all warm execution is allocation-free.

### Accomplished

- Migrated IMU/contact/frame sensor wrappers to the prepared paths without changing synchronous return types, units, authority validation or grouped readback.
- Added a host LRU/cache-storage regression and two GPU regressions. The first checks 70 bodies/mounts over 16 samples, alternating two input-buffer sets, varying `dt`, replacing pose/depth buffers midstream, collecting in reverse pairs and verifying all numerical outputs. Each sensor needs two initial preparations and one replacement preparation; warm in-place changes do not prepare again. Reset, clear and invalid-time counter behavior are checked.
- The second records two independent IMU parameter sets plus contact/frame queries into one encoder/submission. It verifies different IMU `dt` values yield different expected accelerations, zero-count encoding does not overwrite pending uniforms, invalid time returns an error, and recorded work survives dropping prepared bindings/pipelines before submission.
- Updated sensor book chapters and S02 status without checking off the full resource/lifecycle gate.

### Verification

- `make test-sensors`: **74 GPU integration + 2 GPU unit tests**, **70 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. `/tmp/molla-sensors-s02-aux-retained-all.log`.
- The auxiliary suite now has five GPU tests, including all existing authority, foreign-engine, snapshot, numerical and grouped-collection cases. An earlier four-test run also passed: `/tmp/molla-sensors-s02-aux-retained-gpu.log`.
- GPU-feature all-target sensor check, no-default-feature kernel/sensor check and viewer check passed: `/tmp/molla-sensors-s02-aux-retained-final-check.log`, `/tmp/molla-sensors-s02-aux-retained-host-check.log`, `/tmp/molla-sensors-s02-aux-retained-viewer.log`.
- `make test APP_TARGET='-p molla-kernels --no-default-features --lib'` passed its one host oracle test: `/tmp/molla-sensors-s02-aux-retained-kernels-host.log`. All jobs from this increment are terminal.
- Changed-file formatting and `git diff --check` passed. No viewer was launched and no performance benchmark was run. GPU observation was 14,311 / 16,376 MiB and 100% utilization; correctness results are not clean latency/throughput evidence.

### Next steps and remaining requirements

S02 is still incomplete: camera retained resources, public lifecycle-safe cross-sensor producer encoding, direct leased output slots, real device-loss/recovery and full allocation/performance gates remain. Low-level kernel composition is not the higher-level producer API. Existing copied snapshot leases remain optional; ordinary borrowed outputs are still overwritten by redispatch.

Continue S02 lifecycle work and S04 top-level acceleration/refits/stable handles, then S06 camera migration, S07 GPU processing and S08 demonstrations/visual/performance acceptance. S00/S01 broader matrices remain open. S09 is unauthorized and S10 deferred. The full goal remains active.

### Relevant files

- `crates/molla-kernels/src/{sensor_bindings,imu,contact_sensor,frame_transform,lib}.rs` — retained binding types, shared encode helper, compatibility dispatch and exports.
- `crates/molla-sensors/src/{dispatch,imu,contact,frame_transform,lib}.rs` — bounded LRU, preparation/submission counters and wrapper routing.
- `crates/molla-sensors/tests/sensor_aux_device_gpu.rs` — ping-pong/replacement reuse and mixed low-level command composition.
- `book/src/sensors/{imu,contact_sensor,frame_transform}.md` — resource reuse and parameter/lifetime contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — current partial-gate status and evidence.

## S02/S06 retained cameras and device-only aperture resolve, 2026-09-21

### Goal and instructions

Continue the full sensor implementation goal. The preceding goal turn was progress: it migrated and verified IMU/contact/frame resource reuse. Current camera code still downloaded each aperture sample, averaged CPU color arrays, re-traced the center and restored color through another download/upload. This independent S06/S02 work removes that path without claiming shared camera scene queries are finished. All validation used Makefile tasks and all edits used patch tools. No Gearbox, assets, soil/tyre/vehicle behavior, unrelated viewer work, windows, commits or pushes changed.

### Architecture and discoveries

- Camera ray and trace kernels now expose retained prepared bindings and parameter structs. Dynamic uniform offsets separate 1024 aperture parameter slots and two trace masks (center and color-only) in one pending command stream. Distinct pending parameters must not alias a rewritten slot. Existing convenience kernel dispatches remain available and use offset zero.
- `TiledCamera` retains ray bindings and a two-entry LRU over all trace input/output buffer identities. Physics ping-pong reuses resources; replaced model/state/output buffers invalidate the relevant bindings. The shared cache now supports fallible preparation without evicting a valid entry on failure.
- A GPU accumulator stores the sum of unpacked RGBA8 byte values in float4 storage, clears once, accumulates each sample and resolves once. Float integer sums are exact through the 1024-sample cap. Resolve uses the CPU-computed f32 reciprocal and positive half-up rounding, preserving the legacy packed-byte average rather than silently changing color space.
- Only the center trace writes geometric channels. Later traces write COLOR only. With COLOR disabled, only the center runs and no accumulation is needed; NONE submits nothing. A nonempty camera frame is one submission regardless of sample count. There are no host image arrays or downloads inside dispatch.
- Aperture scratch and pipelines are lazy and retained. Replacing public color storage rebinds only the accumulator group, preserving the sum buffer, uniform and both pipelines. `CameraDispatchStats` counts passes, submissions, preparations and accumulator buffers; these are application-resource counters, not driver/command-encoder allocation or timing measurements.
- Camera dimensions, model/output sizes, finite optical parameters, channel masks, sample limits, camera poses, input/output GPU authority and engine arguments are checked. Raw public Buffer provenance remains the caller's responsibility. A changed output length requires reconstruction. Near remains informational; this increment does not add clipping or change primitive visibility semantics.
- `readback_source(channel, range)` exposes checked single-channel ranges to existing grouped readback and copied snapshot APIs. Written camera host mirrors stay stale; legacy `download_*` still synchronously collect one channel. This is not direct leased producer storage or a new public cross-sensor producer scheduler.

### Accomplished and verification

- Implemented GPU packed-color accumulation/resolve and retained camera bindings, routing each camera frame through one command encoder/submission.
- Added three GPU tests covering two-camera odd-resolution CPU-average parity at K=2/3/7/16, exact center geometry, every individual channel mask, NONE, GPU-authoritative physics state, two-buffer reuse, color-output replacement, bad config/ranges, foreign engines, synthetic rounding at K=1/2/3/7/1024 and a complete K=1024 camera dispatch. Numerical tolerances were not relaxed; packed colors and center geometry match their legacy kernel references exactly in these fixtures.
- Warm camera frames retain one ray preparation and two trace preparations for the ping-pong test. Output replacement adds one trace/accumulator binding preparation while accumulator buffer allocations stay at two. CPU config validation and fallible-cache preservation are also covered by host tests.
- Final `make test-sensors`: **77 GPU integration + 2 GPU unit tests**, **71 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. Log: `/tmp/molla-sensors-s06-camera-verified-all.log`.
- Targeted camera tests and the existing optics suite passed during development: `/tmp/molla-sensors-s06-camera-final-gpu.log`, `/tmp/molla-sensors-s06-optics.log`. The final full lane includes the later scratch-preserving rebind change.
- GPU-feature all-target sensor check and viewer check passed: `/tmp/molla-sensors-s06-camera-verified-check.log`, `/tmp/molla-sensors-s06-camera-verified-viewer.log`. No-default-feature kernel/sensor check passed: `/tmp/molla-sensors-s06-camera-host-check.log`. Changed-file formatting and `git diff --check` passed.
- No benchmark or viewer/visual acceptance was performed. GPU observation was 14,054 / 16,376 MiB and 99% utilization. Single-submission/no-intermediate-readback evidence does not establish measured speedup or resolve prior performance regressions. All jobs from this increment are terminal.

### Next steps and remaining requirements

S06 still needs shared scene/BVH visibility for actual meshes and terrain, heterogeneous camera settings/grouping, broader clipping/occlusion cases and camera-to-GPU-processing chains. The camera currently intersects only its supported primitives, including ellipsoids; meshes/terrain remain invisible. Aperture offsets and per-sample parameter uploads still originate on the CPU.

S02 still needs lifecycle-safe public cross-sensor producer composition, direct dispatch into leased slots, real device-loss/recovery and full warm allocation/performance acceptance. S04 top-level acceleration/refits/stable handles, S07 processing and S08 demonstrations/visual/performance gates remain. S00/S01 broader contracts/matrices are not complete. S09 stays unauthorized and S10 deferred. The original full goal remains active.

### Relevant files

- `crates/molla-kernels/src/{tiled_camera,camera_prepared,camera_accumulate,lib}.rs` — dynamic parameter slots, prepared APIs, retained aperture scratch/pipelines and exports.
- `crates/molla-kernels/shaders/camera_accumulate.wgsl` — packed-byte floating accumulation and half-up resolve.
- `crates/molla-sensors/src/{tiled_camera,dispatch,lib}.rs` — single-frame command stream, fallible LRU, validation, grouped sources and statistics.
- `crates/molla-sensors/tests/sensor_camera_device_gpu.rs` — numerical CPU references, rounding, channel isolation, resource reuse and failure cases.
- `book/src/sensors/tiled_camera.md` — current optics/ownership/capacity/resource contracts and explicit visibility limitations.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified accumulation/center-geometry items and remaining full-plan gates.

## S06 shared camera visibility and revisioned materials, 2026-09-21

### Goal and instructions

Continue the full `SENSORS.md` objective. The previous goal turn was progress: retained camera resources and GPU aperture resolve landed with passing tests. Current camera visibility still bypassed shared scene queries and could not see meshes/terrain. This increment routes camera visibility through those existing foundations. Makefile tasks and patch edits were used; no Gearbox, soil/tyre/vehicle code, assets, unrelated viewer/FULL_COMPACT work, application windows, commits or pushes changed.

### Architecture and discoveries

- `TiledCamera` now generates rays directly into an internal `Raycaster`, encodes shared primitive/BVH/nearest-hit passes, then shades query results with `CameraShadeKernel`. It does not copy a second ray array or download/merge hits on the host. Queries, shading and aperture accumulation still share one camera submission. The low-level legacy `CameraTraceKernel` remains available for compatibility and independent primitive-reference fixtures, but is no longer used by TiledCamera.
- Ordinary `dispatch(model, state)` supports raw TriMesh models through the legacy exact-geometry cache; `dispatch_scene(scene, state)` shares owned BVHs and additionally sees compiled Triangle/Heightfield geometry. Raw-model omissions remain documented. Owned geometry is not rebuilt as GPU body poses move, and owned-scene queries do not populate per-camera legacy BVH caches.
- `SensorScene` now snapshots finite per-shape RGB colors along with geometry. Source color-array length/host authority are checked before compilation; replacement is transactional. Color-only replacement currently rebuilds geometry and advances the same revision. Independent material-only updates remain an optimization opportunity, not an implemented fast path.
- `channel_scene_stamp(channel)` tracks the last **sensor write** to each channel. Partial-mask dispatches preserve old stamps for unwritten channels, avoiding relabeling old IDs as a new scene. Raw writes have no owned scene stamp. Stamps are not sample timestamps, and direct user mutation of public output buffers is outside their tracking.
- Hit shading preserves packed RGBA8/albedo, zero miss normals/colors, -1 miss depth, `u32::MAX` miss IDs, strict `t < max_distance` and informational near. Analytic normals remain outward; mesh normals oppose the incident ray. The shared query retains analytic-first/earliest-mesh ties. A persistent dummy color binding supports genuinely empty scenes without requiring fabricated shape data.
- Shading now has its own two-variant resource cache. `trace_preparations` counts these shading preparations; `query_stats()` separately reports primitive/mesh/merge preparations and zero internal submissions. This distinction replaces the old all-in-one camera trace binding count. Pixel capacity also obeys the shared one-dimensional query dispatch limit; oversized batches are rejected before allocating pixel storage.
- The first terrain oracle fixture placed a center column exactly on a non-coplanar grid edge. GPU and f64 reference selected different valid adjacent-face normals (same depth/shape). The dense comparison now uses an off-edge camera position with unchanged 5e-4 tolerances, and a separate explicit crease sample checks depth/ID and either of the two mathematical face normals. This was a fixture ambiguity, not a reason to loosen all numerical checks.

### Accomplished and verification

- Added shared query-hit shading and migrated both camera dispatch paths. Scene colors, per-channel revision metadata, empty-scene handling, capacity checks and docs were updated.
- Added `sensor_camera_scene_gpu` with three GPU tests: two-camera moving/static mesh instances plus sphere/plane occlusion against independent f64 triangle/analytic references; actual heightfield/triangle rendering including an explicit crease case; strict max range/informational near, mesh aperture rendering, scene/material replacement, partial channels, reversed pending reads, invalid replacement rollback and empty scenes.
- The moving-instance fixture shares one BVH between static/dynamic instances. Over six frames, two physics buffers require two primitive and three mesh binding preparations, with one shading preparation and zero camera-owned mesh builds. Switching to raw model dispatch builds one legacy BVH and matches all five owned-scene output channels exactly.
- Existing primitive camera/optics CPU-reference tests still pass without numerical tolerance changes. Their resource assertions now distinguish shared query preparations from shading preparations.
- `make test-sensors`: **80 GPU integration + 2 GPU unit tests**, **71 GPU-feature host unit tests**, **10 no-default-feature unit + 3 CPU adjoint integration tests** passed. `/tmp/molla-sensors-s06-shared-camera-final-all.log`.
- The final expanded scene suite, including crease and raw/owned all-channel parity assertions, passed **3 tests**: `/tmp/molla-sensors-s06-camera-scene-final.log`. Earlier terrain failure logs are not acceptance evidence; their fixture issue is described above.
- GPU-feature all-target sensor check, no-default-feature kernel/sensor check and viewer check passed: `/tmp/molla-sensors-s06-shared-camera-check.log`, `/tmp/molla-sensors-s06-shared-camera-host.log`, `/tmp/molla-sensors-s06-shared-camera-viewer.log`. Changed-file formatting and `git diff --check` passed. All jobs from this increment are terminal.
- No viewer launch/visual acceptance or performance benchmark was performed. GPU observation was 14,054 / 16,376 MiB and 100% utilization. Shared-query camera buffers add storage and shading adds a GPU stage compared with primitive-only tracing; no end-to-end speedup or memory-budget acceptance is claimed.

### Next steps and remaining requirements

S06 still needs heterogeneous camera parameters/grouping and camera-to-GPU-processing chains. S02 public lifecycle-safe producer composition/direct leased output slots and device-loss/full allocation gates remain; S04 still lacks top-level acceleration, independent refits/local-pose updates and stable application handles. S07 processing and S08 demonstrations/visual/performance acceptance are unfinished. S00/S01 broader contracts and matrices remain open. S09 is unauthorized; S10 deferred. The full goal remains active.

### Relevant files

- `crates/molla-kernels/shaders/camera_shade.wgsl` — shared-hit channel conversion, miss handling and Lambertian color.
- `crates/molla-kernels/src/{camera_prepared,tiled_camera,lib}.rs` — retained shading kernel/bindings and exports.
- `crates/molla-sensors/src/{tiled_camera,scene}.rs` — shared camera dispatch, geometry/color snapshots, per-channel stamps and empty-scene support.
- `crates/molla-sensors/tests/{sensor_camera_scene_gpu,sensor_camera_device_gpu}.rs` — independent visibility/material/revision fixtures and updated resource assertions.
- `book/src/sensors/{tiled_camera,scene}.md` — dispatch capabilities, result identity, resource semantics and remaining limits.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified shared-visibility item and open full-plan gates.

## S06 heterogeneous camera groups and sensor facade, 2026-09-21

### Goal

Continue implementing the full `SENSORS.md` goal. The preceding response restated the plan location without advancing implementation (no progress). Revalidated the unfinished `camera_batch.rs` against the current checkout: its earlier all-target check was terminal and successful, but it lacked tests, documentation and facade exports. This increment completes and verifies that camera grouping requirement, not the full sensor plan.

### Instructions

Makefile tasks and patch tools only. Preserve unrelated WIP, Gearbox, soil/tyres/vehicles, assets and user application windows. No commits or pushes. Engram context/save/session-summary tools were searched again and remain unavailable; this file records the session evidence instead, without claiming an external memory save.

### Discoveries

- `CameraBatch` groups by exact full `TiledCameraConfig` equality, not merely image dimensions. Independent IDs, poses and input-order layouts map to group-local camera indices/ranges. Different masks, lighting, range, intrinsics and physical optics cannot silently share settings. Nonadjacent compatible members share one retained `TiledCamera`.
- Dispatch stays device-only and makes one submission per active group, not one per camera or one per heterogeneous frame. Each group shares owned-scene mesh BVHs. Empty batches and NONE-mask groups submit no work. Disabled channels are rejected by `CameraImage`, including their metadata methods.
- Entire grouping/validation precedes GPU allocation: unique IDs, finite/normalized poses, complete config, checked dimensions, aggregate pixels and per-group storage/dispatch limits. Default limits are 64 cameras, 16 groups and 4 million aggregate pixels; maxima are 1,024 cameras and 64 groups. This is an image-capacity bound, not a full GPU-byte budget. Transactional reconfiguration temporarily retains both old and new storage.
- Frame views borrow the batch; callers enqueue readback or copied snapshots before mutable reuse. Queued copies remain intact across reconfiguration/reset. Receipts distinguish batch owner, generation and monotonically increasing samples. `is_current` checks owner/generation rather than latest sample, so historical copies in a generation remain valid. Readback metadata alone lacks full scene identity and batch generation; retain those beside tickets.
- Cross-group execution is not atomic. A later group can fail after earlier submissions; no frame is returned, old receipts are invalidated and the attempted sample ID is consumed. Sample/generation exhaustion fails before dispatch. Failed configuration validation leaves the previous batch intact; explicit reconfiguration rebuilds groups and is not a warmed frame operation.
- The top-level `molla::sensors` facade had not exposed the new scene, LiDAR-batch, scheduler, readback and snapshot APIs. Added those alongside camera batches/device readings/statistics while preserving legacy exports. Scheduler exports remain available without the GPU feature. The facade's old static-only/query-path description was replaced with current behavior.

### Accomplished

- Added four host grouping tests: nonadjacent membership/IDs/input order/ranges; each full configuration field; invalid config/pose and explicit budgets; aggregate group storage/dispatch capacity rather than individual-image-only validation.
- Added three `sensor_camera_batch_gpu` tests. Four independently configured cameras form three groups and match independently dispatched single-camera references exactly for every enabled output channel over three frames. Pose and look-at changes affect only the selected sensor. Warm ray/shading preparations remain at one per group; only the aperture group allocates two retained accumulator buffers; internal query submissions remain zero.
- GPU lifecycle tests cover resized/reordered/changed-config reconfiguration, copied snapshots and pending readback collected in reverse order, stale and foreign-owner receipts, raw versus owned scene stamps, NONE/partial masks, empty groups, unknown IDs, invalid timestamps/poses, rollback and foreign engines. An additional ignored GPU unit test injects failure after the first group submits and checks statistics, receipt invalidation, sample gaps and identity exhaustion.
- Added facade API/type-identity regressions in `crates/molla/tests/sensor_api.rs` for CPU-only and GPU-enabled configurations. `make test-sensors` now runs those feature lanes as well as the existing sensor regressions.
- Final `make test-sensors` passed **83 GPU integration + 3 GPU unit tests**, **75 GPU-feature host unit tests**, **10 no-default sensor unit + 3 CPU adjoint integration tests**, and **3 facade test executions** (1 CPU-only, 2 GPU-feature): **177 passing executions**, zero failures. Evidence: `/tmp/molla-sensors-s06-batch-final-all.log`; the first sensor-only run also passed at `/tmp/molla-sensors-s06-batch-all.log`.
- Sensor GPU all-target check passed (`/tmp/molla-sensors-s06-batch-check2.log`). Kernel/sensor/facade no-default check and viewer check passed (`/tmp/molla-sensors-s06-batch-host-check.log`, `/tmp/molla-sensors-s06-batch-viewer-check.log`). Explicit-file formatting and `git diff --check` passed. All owned check/test jobs are terminal.
- Updated camera ownership/grouping documentation and marked only S06's heterogeneous grouping item complete. No visual acceptance or benchmark was performed. RTX 4080 observation was 15,078 / 16,376 MiB and 100% utilization; no uncontended performance claim is made.

### Next Steps

S06 still requires device-only camera-to-processing chains; implement S07 image processing and its CPU-reference/numerical/consumer-lifecycle tests next. S02 still needs public lifecycle-safe producer composition, direct leased output slots, physical device-loss recovery and full allocation gates. S04 still lacks top-level acceleration, independent refits/local-pose updates and stable application handles. S00/S01 broader contracts/performance matrices and S08 viewer/visual/performance acceptance remain open. S09 is unauthorized and S10 deferred. The full goal remains active.

### Relevant Files

- `crates/molla-sensors/src/camera_batch.rs` — configuration groups, checked layout planning, independent identities/poses, receipts, reconfiguration and unit tests.
- `crates/molla-sensors/src/{tiled_camera,lib}.rs` — shared validation, comparable configs and public batch exports.
- `crates/molla-sensors/tests/sensor_camera_batch_gpu.rs` — independent-camera parity, isolation, resource reuse and lifecycle tests.
- `crates/molla/src/sensors/mod.rs` and `crates/molla/tests/sensor_api.rs` — complete current device-sensor facade and feature-boundary tests.
- `Makefile` — sensor lane now includes both facade feature tests, preserving earlier unrelated launch changes.
- `book/src/sensors/tiled_camera.md` — batching example, grouping/ownership/limits/failure contracts and verification commands.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified heterogeneous grouping status and full-plan remaining work.

## S07 GPU image operations and S06 camera-processing chains, 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The preceding goal turn was progress: heterogeneous camera groups and facade wiring were implemented and verified. Current code still performed all image post-processing on host arrays. This increment ports the image-operation package and verifies the previously missing S06 camera-to-device-processing chain, without claiming the cloud/annotation or performance packages are complete.

### Instructions

Use Makefile tasks and patch tools; preserve unrelated WIP, Gearbox, authored assets, soil/tyres/vehicles and application windows. No commits/pushes. Answered the user's progress question during implementation with the working paths and concrete remaining work, not a speculative percentage. Engram tools were searched again and remain unavailable; session evidence is recorded here instead.

### Discoveries

- `ImageProcessor` consumes checked `ReadbackSource<u32>`-equivalent ranges without downloading. Each job owns an independent operation list and caller-supplied sensor/sample/revision/time stamp. Jobs are validated together before encoding, copied GPU-to-GPU into retained working storage, then processed in one submission. Input pixels are not mutated; empty/no-op handling and capacity bounds are explicit.
- Implemented exposure, gamma, linear/sRGB encoding/decoding, Reinhard and white-point tone mapping, brightness/contrast, saturation, gray-world and white-patch white balance. Each stage quantizes RGBA8 just like the CPU convenience path; alpha is copied bit-exactly, including transparent pixels. This preserves packed-image semantics rather than pretending clipped camera color is HDR. CPU functions remain available, with an added reverse sRGB convenience function.
- White balance performs a 256-thread partial reduction, a second parallel reduction over partials and a pixel application pass. Statistics and gains are per image/per stage, not pooled across cameras. Empty/black/zero-channel behavior matches the CPU formulas. Gains are device-resident Vec3 channels; only stages that wrote gains can expose them.
- Gamma endpoints are explicit in WGSL. With extreme finite gamma, its reciprocal can become subnormal/flush to zero on the GPU; direct pow at black can otherwise lose the CPU's black endpoint. Tests include gamma `f32::MAX`, zero gamma, exposure `f32::MAX` and zero/huge white points. Nonfinite parameters and overflow-capable brightness/contrast combinations are rejected before submission.
- Resources are retained: four application buffers, one bind group and three pipelines. The exact configured byte budget covers packed pixels, reduction partials, stage gains and aligned uniform slots; it excludes CPU metadata, pipelines, driver memory and upload staging. Each pending stage uses its own uniform slot; up to 64 images × 64 operations are supported. Metadata/lease vectors remain bounded per-frame host allocations, not an allocation-free claim.
- Processed views borrow their producer. Queued readback and copied snapshots survive processor reuse; snapshot input occupancy tokens remain retained until GPU completion, without capturing device-owning snapshot objects in completion callbacks. Direct leased producer slots and public cross-sensor encode-only composition are still separate S02 requirements.
- Numerical comparisons use the existing CPU formulas, with one RGB-code tolerance for individual operations and two for the tested seven-stage chain; alpha/no-op copies are exact. GPU f32/tree sums are not represented as bit-identical CPU f64 reductions, and arbitrary long chains can amplify quantization differences.

### Accomplished

- Added the retained `ImageProcessKernel` and WGSL apply/reduce/finalize stages, the validated image processor API, typed borrowed outputs and per-stage gain access. Exported them through both `molla-sensors` and the `molla::sensors` facade and extended facade regressions.
- Added three host tests for exact aligned byte budgets, device/capacity/stage limits and CPU-compatible scalar validation. Added five `sensor_processing_gpu` tests covering all operation types against CPU references; range/alpha/stamp preservation; independent per-image reductions; empty/black/zero-channel inputs; 255/256/257 and 65,537-pixel reduction boundaries; invalid jobs/foreign devices; the final uniform slot at 4,096-slot capacity; reversed queued reads and output reuse.
- The device chain fixture executes heterogeneous camera groups → image pipelines → copied snapshot → second image pipelines, without host intermediate pixels. Dropping the snapshot and resetting its pool after consumer submission preserves results and permits eventual reclamation. This completes S06's camera-processing-chain test requirement; broader S02/S04/S08 gates remain.
- Final `make test-sensors` passed **88 GPU integration + 3 GPU unit tests**, **78 GPU-feature host unit tests**, **10 no-default sensor unit + 3 CPU adjoint integration tests**, and **3 facade test executions**: **185 passing executions**, zero failures. `/tmp/molla-sensors-s07-images-final-all.log`. The initial full run had 184 executions before the final-slot/queued-reuse test was added; `/tmp/molla-sensors-s07-images-all.log` is that earlier evidence, not the final count.
- Final focused image GPU suite passed all **5 tests**: `/tmp/molla-sensors-s07-images-gpu-final.log`. GPU-feature all-target sensor check, no-default kernel/sensor/facade check and viewer check passed: `/tmp/molla-sensors-s07-images-final-check.log`, `/tmp/molla-sensors-s07-images-host-check.log`, `/tmp/molla-sensors-s07-images-viewer-check.log`. Explicit formatting and `git diff --check` passed.
- Added `bench_sensor_images`: ten operations × three sizes (1,024 / 65,536 / 1,048,576 pixels) × CPU / completed-GPU / GPU-plus-host-readback lanes. `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_images -- --test'` passed all **90 smoke cases**, including the million-pixel reduction paths. `/tmp/molla-sensors-s07-images-bench-smoke.log`. This is execution validation, not statistical performance evidence.
- Added the image-processing book chapter, corrected the stale CPU-only/no-bottleneck claim in `camera_post`, linked camera device consumers and recorded only verified checklist progress. All owned check/test/benchmark jobs are terminal. No viewer was launched. GPU observation remains RTX 4080, 15,078 / 16,376 MiB and 100% utilization; no uncontended speedup or performance-budget acceptance is claimed.

### Next Steps

Continue S07 GPU point-cloud construction/transforms/filtering/stable compaction with scan/sensor identity, then bounded spatial binning and annotation reductions/CPU export parity. `pointcloud.rs` remains the CPU reference. S02 lifecycle-safe producer composition/direct leased outputs/device-loss/allocation gates and S04 top-level acceleration/refits/stable handles remain. S00/S01 broader contracts/performance matrices and S08 viewer/visual/performance acceptance are unfinished. S09 remains unauthorized; S10 remains deferred. Full goal stays active.

### Relevant Files

- `crates/molla-kernels/src/image_process.rs` and `shaders/image_process.wgsl` — retained image passes, aligned stage slots and white-balance reductions.
- `crates/molla-sensors/src/image_processing.rs` — budgets, jobs/operations, validation, device orchestration, borrowed results and host tests.
- `crates/molla-sensors/src/camera_post.rs` — CPU reference contract and explicit sRGB decoding.
- `crates/molla-sensors/tests/sensor_processing_gpu.rs` — scalar/reduction/chain/ownership/boundary fixtures.
- `crates/molla-sensors/benches/bench_sensor_images.rs` and `Cargo.toml` — three-mode per-operation benchmark matrix.
- `crates/molla-kernels/src/lib.rs`, `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — exports and feature-boundary coverage.
- `book/src/sensors/image_processing.md`, `tiled_camera.md` and `book/src/SUMMARY.md` — API, ownership, numerical, memory and benchmark contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified image/S06 chain work and complete remaining scope.

## S07 cloud work in progress — renderer question interruption, 2026-09-21

The two preceding status replies added no implementation progress. Revalidated the CPU cloud APIs and existing compaction primitive: that primitive is blocking and limited to 1,024 values, so it does not satisfy the planned device-resident large-cloud path. Added **unverified** `cloud_process.wgsl` and `cloud_process.rs` plus kernel exports. They define retained transform/filter, parallel block-prefix and stable scatter passes with preserved integer IDs and per-job diagnostics. No high-level cloud wrapper or cloud tests exist yet; no build/test has been run for this new kernel. Do not count cloud processing as implemented or verified.

Current WIP wire format is a 112-byte point (position/intensity, normal/range, direction/scalar, 64-bit sensor/sample IDs as u32 pairs, row/column/subray/source index, shape/color/flags, angles). The planned wrapper must validate all jobs/budgets before allocation/dispatch, supply packed points or copied LiDAR hit channels, retain snapshot input leases, and expose GPU counts plus stable compacted ranges. Shader descriptors are 160 bytes; counts use one total plus five words per job (start/count/misses/invalid/filtered). Compile and exercise these contracts before relying on them. Spatial binning and annotations remain separate unfinished work.

Answered the renderer-boundary question from current camera shaders and Engine APIs: Molla's camera independently traces sensor scene geometry with basic per-shape shading; it does not capture another renderer's framebuffer/materials. `Engine` currently creates its own GPU context and has no public external-device constructor. Gearbox integration/shared-device texture interop remain unimplemented and separately gated. No Gearbox edits or new performance claims.

## S07 stable device point-cloud pipeline, 2026-09-21

### Goal

Continue the full `SENSORS.md` goal. The preceding goal turn made partial progress by adding an unverified cloud kernel before answering the renderer question. This turn revalidated and completed that kernel's high-level API, numerical/lifecycle fixtures, facade wiring and documentation. This supersedes the preceding WIP note's missing-wrapper/tests status, not the broader remaining requirements.

### Instructions

Makefile tasks and patch edits only. Preserve unrelated WIP, Gearbox, authored assets, soil/tyres/vehicles and application windows. No commits/pushes. Engram tools were searched and remain unavailable; this file records session evidence without claiming an external memory save.

### Discoveries

- The existing compute-primitive compactor is blocking, allocates temporary resources and handles at most 1,024 scalar values. The new cloud kernel instead retains fixed-budget resources and performs point preparation/filtering plus local ranks, a parallel block-prefix pass, then stable scatter in three dispatches. It does not use atomic append order or a host prefix sum. Diagnostic atomics do not determine point ordering.
- `CloudProcessor` accepts packed `CloudPoint` sources or five device LiDAR channels. `CloudInput::from_lidar` supports terrestrial and aerial scans, selecting all divergent sub-rays and preserving actual directions. Per-job sensor identity/raster dimensions/type/range/device checks happen before encoding. Packed input records are not retagged by aggregate job stamps.
- The 112-byte record retains position/intensity, normal/acquisition distance, actual direction/optional scalar, full-width sensor and sample/scan-sequence identities split into u32 pairs, row/column/sub-ray/source index, shape/color/flags and source angular metadata. Terrestrial cell angles are retained; aerial points do not fabricate them. Default LiDAR intensity=1 and absent color are placeholders, not a physical-return model. One explicitly selected scalar channel is supported; arbitrary CPU scalar maps remain CPU data.
- Rigid transforms precede AND-combined inclusive distance, inside/outside XYZ, intensity and scalar filters. All comparison variants preserve CPU predicate semantics within f32 precision. Missing returns, invalid attributes/transformed coordinates and predicate rejection are counted separately per job. Global compaction preserves job order and point order; each job's start/count stays on the GPU.
- Active output tails are zeroed, so stale points from earlier frames cannot masquerade as current survivors. Device-only consumers can process the active upper bound and use flags/counts; this still scans the supplied upper bound and is not GPU-indirect dispatch. Optional export first collects the count, then requests only the compacted prefix. Empty batches submit a count clear and zero compute passes.
- Frames borrow their processor. Queued copies survive reuse, and copied snapshot input occupancy tokens remain held through GPU completion even after public snapshot drop/pool reset. As with images, this is not public cross-sensor producer encoding or direct leased output slots.
- The eight-buffer budget is `288*N + 8*ceil(N/256) + 180*J + 20` bytes: raw inputs, working/output records, ranks, block offsets, descriptors, counts and parameters. Defaults are 65,536 points / 64 jobs / 64 MiB, with at most 1,024 jobs and additional device limits. Host mirrors, bounded metadata vectors, driver staging and pipelines are not included. No allocation-free or bandwidth-speedup claim is made.
- Angular interpolation was reordered to multiply by the normalized grid fraction instead of multiplying a large angular span by the integer index first. The latter could overflow despite finite endpoints and a finite final value. A GPU regression now verifies a `3e38` endpoint without false invalid points; a separate transformed-position overflow case verifies explicit invalid diagnostics.

### Accomplished

- Completed cloud kernel/API and typed range/count/summary access, with facade exports and feature-boundary tests. Added explicit bytemuck derive support for the public wire record.
- Seven `sensor_cloud_gpu` tests verify CPU-oracle transforms/filters/stable ordering; empty and all-rejected batches; 255/256/257 and 65,537-point boundaries; malformed attributes, all scalar comparisons and inclusive bounds/zero radius; full-width IDs and terrestrial/aerial grid metadata; pending snapshot/readback reuse; invalid descriptions/foreign engines; large finite angular interpolation and transform overflow. The 1,000,003-point test checks total/diagnostics and exact first/boundary/final survivor records across all block-prefix chunks.
- Two host tests check shader wire sizes, full-width ID recovery, summary decoding, exact byte budgets and device/application rejection before allocation.
- Final `make test-sensors` passed **95 GPU integration + 3 GPU unit tests**, **80 GPU-feature host unit tests**, **10 no-default sensor unit + 3 CPU adjoint integration tests**, and **3 facade test executions**: **194 passing executions**, zero failures. `/tmp/molla-sensors-s07-cloud-final-all.log`. The preceding 193-execution full run is `/tmp/molla-sensors-s07-cloud-all.log`; it predates the angular-overflow regression and is not the final count.
- Final expanded cloud suite passed all **7 tests** (`/tmp/molla-sensors-s07-cloud-edge-gpu.log`); the final full lane repeated them. GPU all-target check, no-default kernel/sensor/facade check and viewer check passed (`/tmp/molla-sensors-s07-cloud-final-check.log`, `/tmp/molla-sensors-s07-cloud-host-check.log`, `/tmp/molla-sensors-s07-cloud-viewer-check.log`). Explicit-file formatting and `git diff --check` passed.
- Added `bench_sensor_cloud`: transform-only, roughly half-kept and all-rejected workloads at 1,024 / 65,536 / 1,000,003 points, with CPU / completed GPU / count-then-selected-prefix collection lanes. Its **27 release-profile smoke cases passed**: `/tmp/molla-sensors-s07-cloud-bench-smoke.log`. Smoke validation is not a statistical benchmark or parity claim, and the matrix does not cover every filter combination.
- Added the cloud book chapter and marked the S07 construction/transforms/filters/compaction item verified, without marking spatial binning, annotations or export integration complete. All owned jobs are terminal. No viewer was launched. Latest GPU observation: RTX 4080, 14,558 / 16,376 MiB, 100% utilization; uncontended performance remains unverified.

### Next Steps

Implement S07 bounded spatial binning/reductions with explicit overflow reporting, then GPU instance/class annotations and selected CPU export adapters/round trips. S02 lifecycle-safe producer composition/direct leased slots/device-loss/allocation gates, S04 top-level acceleration/refits/stable handles, S00/S01 broader contracts/performance matrices and S08 viewer/visual/performance acceptance remain. S09 stays unauthorized and S10 deferred. The full goal is active, not complete.

### Relevant Files

- `crates/molla-kernels/src/cloud_process.rs` / `shaders/cloud_process.wgsl` — retained preparation, prefix and stable scatter passes.
- `crates/molla-sensors/src/cloud_processing.rs` — wire records, LiDAR inputs, validation/budgets, GPU orchestration, count/range APIs and host tests.
- `crates/molla-sensors/tests/sensor_cloud_gpu.rs` — CPU references, IDs/grid metadata, prefix/lifecycle/invalid/numeric boundary fixtures.
- `crates/molla-sensors/benches/bench_sensor_cloud.rs` / `Cargo.toml` — cloud benchmark matrix and typed-record dependency feature.
- Kernel/sensor `lib.rs`, `crates/molla/src/sensors/mod.rs` and `crates/molla/tests/sensor_api.rs` — public exports and feature coverage.
- `book/src/sensors/cloud_processing.md` / `book/src/SUMMARY.md` — API, ordering, ownership, numerical, memory and benchmark contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified cloud item and full remaining scope.

## S07 spatial bins/reductions — verified increment, 2026-09-21

### Goal

Continue the complete `SENSORS.md` objective with bounded device-resident spatial membership and reductions. The preceding renderer explanation was a read-only clarification, not an implementation increment. Re-read the unverified grid kernel and authoritative plan before continuing; this section supersedes its unverified status.

### Instructions

Used Makefile entry points for formatting, checks, tests and benchmark smoke; patches for edits. Gearbox, soil/tyres/vehicles, authored assets and user preview windows remain untouched. No commit or push. Engram tool discovery again returned no context/save/session-summary tools; this file records the handoff rather than claiming an unavailable memory save.

### Discoveries and decisions

- Added globally bounded CSR membership: cell counts/prefixes plus an index allocation sized for the whole input capacity. There is no fixed bucket cap, so a dense cell cannot silently truncate points. Input-range-local indices address a retained packed point copy with original IDs/metadata intact.
- Five retained passes classify, prefix cells, prefix block totals, atomically scatter membership and reduce cells. Each cell has count/start, mean XYZ/intensity, minimum/maximum XYZ/intensity and a numeric flag. Cell-local index order is unspecified; floating-point reductions are not bitwise deterministic. Upstream cloud compaction remains stable.
- Bounds are half-open and widths must be positive normal finite f32. Unlike legacy CPU convenience binning, the checked API rejects reversed/degenerate bounds and zero dimensions. Dimension products, 2D dispatch extent, application byte budgets and device limits are checked before allocating. Near internal cell planes, f32 classification can differ from an f64 oracle.
- Six diagnostics distinguish accepted, missing, invalid position/intensity, out-of-bounds, defensive index overflow and numeric-overflow cells. Nonfinite reductions flag the cell and zero its mean while preserving membership/extrema. The extreme-value test accepts either a valid near-maximum finite mean or correctly reported overflow, since compiler/hardware rounding can differ. No extreme-cancellation accuracy promise is made.
- Whole source validation precedes submission. Only GPU copies/compute occur in `process`; no implicit wait/readback. Selected sources permit GPU chaining or optional collection. Snapshot occupancy is retained through queue completion without capturing owning GPU references. Empty input explicitly clears active summaries/diagnostics; all-rejected input clears membership tails to `u32::MAX`.
- Eight retained GPU buffers use `120*N + 76*C + 8*ceil(C/256) + 104` bytes, excluding host mirrors, driver staging and pipeline memory. Default limits are 65,536 points / 65,536 cells / 64 MiB. Statistics do not imply driver allocation-free execution. Dense reduction currently uses one strided workgroup per cell; performance remains to be measured.

### Accomplished

- Completed `CloudGrid`/configuration/spec/frame/cell/diagnostics/stats, low-level kernel and shader, checked typed range access, facade exports and feature API coverage. Added the grid book chapter, linked it from cloud processing/SUMMARY, and marked only the S07 spatial-binning item verified.
- Five real-GPU tests passed: independent CPU membership/mean/extrema references with 0/1/255/256/257/65,537 points; 66,049 cells crossing prefix chunks and the 2D reduction dispatch boundary; a 1,000,003-point single cell retaining every member; malformed/foreign/over-capacity requests without submissions; extreme finite intensity and all-rejected resets; packed cloud → copied snapshot → grid with queued readbacks surviving reuse/reset/drop and selected source offsets. `/tmp/molla-grid-gpu2.log`.
- Two host tests verify wire sizes, exact buffer budgets/device rejection, dimension overflow, degenerate/nonfinite/subnormal-width grids and diagnostic decoding.
- `make test-sensors` passed **201 executions**: **100 GPU integration + 3 GPU unit**, **82 GPU-feature host unit**, **10 no-default sensor unit + 3 CPU adjoint integration**, and **3 facade test executions**, zero failures. `/tmp/molla-grid-all.log`.
- Added `bench_sensor_grid`, with CPU / completed GPU / GPU plus selected summaries at 1,024 / 65,536 / 1,000,003 points, for one dense or 1,024 distributed cells. All **18 release-profile smoke cases passed** (`/tmp/molla-grid-bench-smoke.log`). This is not a statistical performance result or parity claim. The specialized CPU benchmark assumes finite in-bounds points and sequential reductions; GPU work includes validation, input copies and five passes.
- GPU all-target check, no-default kernel/sensor/facade check, and viewer compile check passed: `/tmp/molla-grid-all-target-check.log`, `/tmp/molla-grid-host-check.log`, `/tmp/molla-grid-viewer-check.log`. Explicit-file formatting and `git diff --check` passed (`/tmp/molla-grid-fmt-check.log`). No viewer launched. Latest GPU observation during work was RTX 4080, 15,098 / 16,376 MiB, 82% utilization. All owned jobs are terminal.

### Next Steps

Implement S07 GPU class/instance annotations and bounds with explicit misses/camera isolation, then selected export adapters and round trips. Remaining full-goal requirements include S02 public lifecycle-safe producer composition/direct leased output/device-loss/allocation gates; S04 top-level acceleration/refits/stable handles; S00/S01 broader contracts/performance matrices; S08 viewer integration, visual acceptance and full performance evidence. S09 is still unauthorized; S10 deferred. The goal is active, not complete.

### Relevant Files

- `crates/molla-kernels/src/cloud_grid.rs` / `shaders/cloud_grid.wgsl` — bounded membership/prefix/scatter/reduction passes and validation.
- `crates/molla-sensors/src/cloud_grid.rs` — public API, budgets, GPU orchestration, typed outputs, diagnostics, ownership and host tests.
- `crates/molla-sensors/tests/sensor_grid_gpu.rs` — independent references, million-point dense membership, errors, numeric/lifecycle/device-chain regressions.
- `crates/molla-sensors/benches/bench_sensor_grid.rs` / `Cargo.toml` — completed-work grid benchmark matrix.
- Kernel/sensor `lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — module/facade exports and feature coverage.
- `book/src/sensors/cloud_grid.md`, `cloud_processing.md`, `book/src/SUMMARY.md` — API/precision/ownership/measurement contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified increment and complete remaining scope.

## S07 GPU annotations and selected annotation export — verified increment, 2026-09-21

### Goal

Continue `SENSORS.md` with GPU class/shape-instance annotation and bounding boxes, preserving independent cameras and selecting only requested export channels. Previous goal turn was concrete spatial-bin implementation/verification progress; current source and the CPU `Dataset` contract were re-read before starting this increment.

### Instructions

All formatting, compilation, tests and benchmark smoke used Makefile entry points. Edits used patches. No Gearbox changes, user-window closes, unrelated vehicle/soil/asset edits, commits or pushes. Engram discovery again returned no save/context/session-summary tools, so this file contains the execution handoff rather than a claimed persistent-memory tool save.

### Discoveries and decisions

- Existing CPU annotation defines one instance per shape, not a body/machine grouping. A labelled class 0 still emits a box; unlabelled shapes and missed pixels map to class 0 but do not emit boxes. Preserved those distinctions, full u32 shape/class IDs and u64 camera/sample stamps. Raw shape masks retain unlabelled IDs and use `u32::MAX` only for misses.
- A sorted sparse label table avoids allocating by the largest shape ID. Checked replacement rejects duplicates/reserved miss IDs before writes, advances a checked revision on success, and retains old mappings on failure. `Dataset::shape_labels` provides a sorted CPU-to-GPU label adapter without silently discarding invalid authored IDs.
- Retained passes initialize image×label boxes, binary-search independent image/label layouts while mapping pixels and atomically reducing integer bounds/counts, then compact visible boxes in stable shape-ID order per image. Bounding-box capacity covers every label for every image. No host image arrays or readbacks occur inside `process`.
- Output boxes use a checked visible prefix; its unused active label extent is zeroed. Empty images, an empty batch and an empty label table have explicit fresh-output behavior. Frames borrow the processor; queued copies preserve historical label/shape versions, and copied snapshot occupancy is pinned through completion. The label revision must be retained beside generic readback stamps; it is not a stable cross-scene shape identity.
- Eight GPU buffers use `8*P + 64*I*L + 8*L + 32*I + 16` bytes. Defaults are 4,000,000 total pixels / 16 images / 1,024 labels / 64 MiB; image×label products and device limits are checked before allocation. Per-call host metadata, host mirrors and driver/pipeline memory are outside this GPU-buffer budget. Dense-instance atomic contention and binary lookup costs remain performance work, not solved by test success.
- Real GPU shader validation caught two WGSL reserved identifiers (`class`, then `target`) that Rust compilation alone did not detect. Renamed both and reran successfully. Red logs: `/tmp/molla-annotation-gpu.log`, `/tmp/molla-annotation-gpu2.log`; initial four-test green: `/tmp/molla-annotation-gpu3.log`.
- Selected CPU JSON/PGM writers now work without constructing an incomplete legacy `AnnotationFrame`. Existing frame writers delegate to them. Invalid mask sizes/box bounds fail before output, and writer errors propagate. Legacy PGM intentionally remains lossy 8-bit (class clamp; instance miss=0 and shape+1 clamp), while typed device/readback outputs and bbox JSON preserve full-width IDs.

### Accomplished

- Added `AnnotationProcessor`, typed configuration/jobs/layouts/frame/counts/stats, `ShapeLabel`, `DeviceBoundingBox`, camera adapter, label revisioning and checked range APIs. Added kernel/shader, sensor/facade exports and feature API coverage.
- Five `sensor_annotation_gpu` tests verify independent CPU class masks/boxes/counts with sparse near-u32-max IDs, class 0, unlabelled/missed/empty images and selected source offsets; stable per-camera compaction across 65,537 labels; an exact 1,000,003-pixel dense instance; transactional invalid label/input rejection; and queued relabelling plus copied-snapshot reset/drop/reuse with empty labels/batches.
- Expanded `dataset_export_gpu` to three tests. The new real heterogeneous camera → GPU annotation chain performs no input image download, collects only box counts/prefixes for JSON, separately selects masks for parsed PGM round trips, verifies all-miss isolation and preserves a near-u64-max camera ID. `/tmp/molla-annotation-export-gpu.log` and the final full lane.
- Added host wire/budget/count validation and selected-writer malformed-input/I/O-error tests. `Dataset::shape_labels` is exercised by the final camera/export fixture.
- Final `make test-sensors`: **209 passing executions**, zero failures: **106 GPU integration + 3 GPU unit**, **84 GPU-feature host unit**, **10 no-default sensor unit + 3 CPU adjoint integration**, and **3 facade executions**. `/tmp/molla-annotation-final-all.log`. Earlier `/tmp/molla-annotation-all.log` also passed 209 but precedes the final Dataset label adapter.
- `bench_sensor_annotations` covers 1,024 / 65,536 / 1,000,000 total pixels, one/four cameras and one dense/257 distributed instances, with CPU / completed GPU / selected-box-collection lanes. All **36 release-profile smoke cases passed** (`/tmp/molla-annotation-bench-smoke.log`). These are functional smoke cases, not statistical speedup/parity acceptance. CPU `Dataset::annotate` allocates outputs; GPU timing includes copies and three passes, not initialization/label uploads.
- GPU all-target, no-default kernel/sensor/facade and viewer checks passed: `/tmp/molla-annotation-final-check.log`, `/tmp/molla-annotation-host-check.log`, `/tmp/molla-annotation-viewer-check.log`. Explicit-file formatting and `git diff --check` passed (`/tmp/molla-annotation-fmt-check.log`). Book chapter and plan status updated. No viewer launched. Latest GPU observation during work: RTX 4080, 14,658 / 16,376 MiB, 100% utilization. All owned jobs are terminal.

### Next Steps

Finish S07 selected packed-cloud export adapters and round trips, with explicit handling of full-width IDs and legacy export-schema limitations. Then remaining shared/full-goal requirements: S02 public producer composition/direct leased slots, device-loss and allocation gates; S04 top-level acceleration/refits/stable handles; S00/S01 broader contracts/performance matrices; S08 viewer integration, visual acceptance and reproducible full performance evidence. S09 Gearbox remains unauthorized; S10 deferred. Full goal stays active, not complete.

### Relevant Files

- `crates/molla-kernels/src/annotation.rs` / `shaders/annotation.wgsl` — retained label/class/bbox/compaction pipeline.
- `crates/molla-sensors/src/annotation.rs` — checked GPU API, budgets, camera bridge, ownership and host tests.
- `crates/molla-sensors/src/dataset.rs` — CPU label adapter and independently selected checked JSON/PGM writers.
- `crates/molla-sensors/tests/sensor_annotation_gpu.rs` / `dataset_export_gpu.rs` — CPU parity, large IDs/counts, camera isolation, lifecycle and selected-export regressions.
- `crates/molla-sensors/benches/bench_sensor_annotations.rs` / `Cargo.toml` — annotation measurement matrix.
- Kernel/sensor `lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — exports and feature coverage.
- `book/src/sensors/annotation.md` / `book/src/SUMMARY.md` — API, ordering, identity, ownership, export and measurement contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified items and complete remaining scope.

## S07 selected packed-cloud export — verified increment, 2026-09-21

### Goal

Continue the full `SENSORS.md` objective by finishing selected packed-cloud CPU serialization/import and GPU export round trips. The preceding goal turn made concrete annotation/export progress. Re-read the existing cloud records, legacy point writers and current plan before implementing this increment.

### Instructions

Used Makefile entry points for formatting, tests, checks and benchmark smoke, and patches for edits. No Gearbox, unrelated vehicle/soil/assets or user-window changes; no commit/push. Engram discovery still has no save/context/session-summary tools. This file records the execution handoff without claiming an unavailable memory tool save.

### Discoveries and decisions

- Legacy `HitPoint` carries only a u32 scan ID, and XYZ/RGB omits acquisition identity/metadata entirely. Added a direct full-record binary-little-endian PLY v1 schema rather than silently casting u64 sensor/sample IDs through that legacy representation. IDs use low/high uint properties; all 112 packed bytes map to typed properties, with RGBA exposed as four uchar values. Every bit, flag, optional attribute and reserved word is retained; this is record fidelity, not validation of measurements.
- The strict matching reader accepts only that versioned schema, supports LF/CRLF and stops after the declared payload. It rejects malformed/reordered schemas, invalid counts and truncation. Explicit point/file-byte limits precede result allocation, header lines are capped at 512 bytes, and allocation/size failures return errors. Defaults: one million records / 128 MiB. Subsequent stream bytes remain caller-owned; no claim of a general PLY importer or strict EOF check.
- New direct packed XYZ/RGB writing avoids allocating legacy `HitPoint`/scalar maps. It rejects invalid flags or nonfinite XYZ before emitting bytes, but remains an explicitly lossy six/four-decimal projection with missing-color white. Original legacy ASCII PLY and original-Scan PTX APIs remain unchanged; compacted records are not turned into invented PTX scan metadata.
- All new APIs accept CPU slices/readers/writers, with no device access or implicit collection. Output I/O failures can leave partial bytes; atomic replacement, flushing and filesystem durability remain caller responsibilities. Complete point IDs are in the file; scene revisions, time, extrinsics and arbitrary application metadata remain beside it, not magically reconstructed from points.
- Verified PLY scalar aliases and binary-endian support against primary PCL source (`https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/io/include/pcl/io/ply/ply.h`), linked in the book. A separate attempted format-page fetch was blocked by that site's crawler policy; no workaround or download was attempted. No third-party format implementation was copied.

### Accomplished

- Added `cloud_export::{write_cloud_ply, read_cloud_ply, write_cloud_xyz_rgb}`, `CloudReadLimits`, facade exports and API feature coverage. Added book documentation describing selected collection, exact schema, bounds, identity, lossy legacy formats and ownership.
- Five host tests verify bit-exact all-field round trips including negative zero, NaN payloads, infinities and reserved fields; independent little-endian property/ID/RGBA offsets; empty files; CRLF and small-buffer reading; stream-boundary preservation; every truncated prefix of a valid fixture; malformed schema/counts and bounded header reads; exact file/point budgets; XYZ/RGB compatibility; and injected header/mid-record I/O failures. Final expanded run: `/tmp/molla-cloud-export-final-host.log`.
- Two real-GPU tests pass (`/tmp/molla-cloud-export-gpu.log`). One selects a single surviving point in the second job of a 259-point input using a **112-byte** readback slot, round-trips all fields and re-uploads the decoded record through the GPU cloud processor. The other runs LiDAR → cloud → selected all-miss/hit scan export without host ray/hit intermediates, preserving u64 IDs, scan/sub-ray/grid/source metadata, angles and ranges.
- `make test-sensors` passed **216 executions**, zero failures: **108 GPU integration + 3 GPU unit**, **89 GPU-feature host unit**, **10 no-default sensor unit + 3 CPU adjoint integration**, and **3 facade executions**. `/tmp/molla-cloud-export-all.log`. This full lane precedes the final extra mid-record I/O assertions; the final five-test host run above verifies those added branches. No production code changed after the full lane.
- Added CPU `bench_sensor_cloud_export`: binary PLY write/read and XYZ/RGB write at 1,024 / 65,536 / 1,000,000 already-collected records. All **9 release-profile smoke cases passed** (`/tmp/molla-cloud-export-bench-smoke.log`). It excludes GPU transfer and filesystem latency, retains write buffers and allocates bounded read results; no statistical throughput/parity claim.
- GPU all-target, no-default kernel/sensor/facade and viewer checks passed: `/tmp/molla-cloud-export-check.log`, `/tmp/molla-cloud-export-host-check.log`, `/tmp/molla-cloud-export-viewer-check.log`. Explicit-file formatting and `git diff --check` passed (`/tmp/molla-cloud-export-fmt-check.log`). No viewer launched; all owned jobs are terminal.
- S07's implementation/correctness items are now checked, and its phase is **core verified**, not full-goal completion. Shared lifecycle/scene and S08 visual/performance gates remain unproven.

### Next Steps

Return to the shared requirements: S02 lifecycle-safe public producer composition/direct leased output slots and complete device-loss/allocation gates; S04 top-level acceleration/refits/stable application handles; remaining S00/S01 sizing/geometry/lifecycle contracts and baseline/performance matrices; then S08 viewer integration, visual inspection and reproducible completed-work performance acceptance. S09 Gearbox remains unauthorized and S10 optional/deferred. The full goal stays active, not complete.

### Relevant Files

- `crates/molla-sensors/src/cloud_export.rs` — CPU full-record PLY, strict bounded import, direct legacy XYZ/RGB and host regressions.
- `crates/molla-sensors/tests/cloud_export_gpu.rs` — selected job/scan GPU export, metadata and re-upload round trips.
- `crates/molla-sensors/benches/bench_sensor_cloud_export.rs` / `Cargo.toml` — CPU serialization benchmark matrix.
- Sensor `lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports and feature checks.
- `book/src/sensors/cloud_export.md`, `cloud_processing.md`, `book/src/SUMMARY.md` — schema, identity, selection, limits and measurement contracts.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — core S07 evidence and complete remaining scope.

## S02 observed device-loss handling — verified increment, 2026-09-21

### Goal

Resume shared S02 lifecycle requirements after concrete S07 export progress. Inspected current readback, snapshot, command-batch and engine code before changing it. Added explicit observed-device-loss handling rather than treating a successful device poll as proof that GPU storage is usable.

### Instructions

Makefile entry points for all formatting/build checks/tests; patches for edits. Destruction fixtures use only fresh isolated **logical test devices**, not physical GPU resets or user applications. No Gearbox/vehicle/soil/authored-asset changes, viewer closes, commits or pushes. Engram tools remain absent from discovery; this file records the handoff without claiming a memory-tool save.

### Discoveries and decisions

- Local wgpu-core 27.0.3 `device/resource.rs` shows destruction can deliver a loss callback after queue drain while polling itself succeeds. Pools previously detected poll/map errors, but zero-byte operations and snapshots could still appear usable when no such error surfaced. Added one engine-owned loss callback and a shared `DeviceHealth` signal.
- `DeviceHealth` stores the first reason/message permanently; the healthy check is atomic and does not wait for GPU work or acquire the diagnostic mutex. The callback/signal owns no GPU references. Raw device/queue accessors retain their old semantics. Engine `destroy_device` marks loss immediately then idempotently destroys only its logical device; CPU engines reject that operation.
- Backend notification may require polling. Pools check before/after nonblocking polls; a raw `device.destroy()` call is not synchronously observable until its callback arrives. The engine owns the single callback; replacing it externally defeats this tracking. No guarantee is made against loss racing arbitrary raw submissions or a driver hang before notification.
- Readback failures are terminal, including zero-byte pending requests; reset cannot revive the device. Snapshot capture/binding/source/reset and deferred consumer encoding/submission reject known loss. Exact device identity prevents rebinding old sources to a fresh engine. Existing map/callback lifetime handling is retained without premature unmapping.
- Added early known-loss guards to sensor constructors/dispatch and scene operations, camera legacy downloads/readback sources, and image/cloud/grid/annotation processing. Synchronized host mirrors and collected frames remain readable. Full physics-kernel/renderer fault handling, physical driver recovery and automatic GPU-state restoration are not claimed.
- Recovery is explicit construction of a new engine and its resources, with application generation reset/history re-priming. Logical-device destruction/callback and successful fresh-engine readback are verified; physical GPU removal/driver faults are not induced.

### Accomplished

- Added engine `DeviceHealth`/`DeviceLoss`, access/check/shutdown methods and facade exports; two host loss tests plus the existing CPU-engine test. `make test-sensors` now includes all three compute host tests.
- Three new GPU fixtures pass in `sensor_device_loss_gpu`: outstanding/cancelled readbacks and unsubmitted leased commands fail after explicit logical destruction; the real backend callback rejects even an empty request while previously collected host data survives; all guarded sensor/processing families reject known-loss constructors/dispatch before submissions. Fixtures also verify fresh-engine reconstruction, old-source rejection and diagnostics not retaining device owners. `/tmp/molla-device-loss-gpu.log` and the final full lane.
- The first expanded full lane exposed an accidental duplicate test-module name in the engine; renamed the added module to `loss_tests`, preserving the original tests. Red log `/tmp/molla-device-loss-all.log`; fixed final lane below. No test was excluded to obtain green.
- `make test-sensors` passed **222 executions**, zero failures: **111 GPU integration + 3 GPU unit**, **89 GPU-feature sensor host unit**, **3 compute host unit**, **10 no-default sensor unit + 3 CPU adjoint integration**, and **3 facade executions**. `/tmp/molla-device-loss-all2.log`.
- Shared-runtime regression lanes passed: **3 storage host tests** (`/tmp/molla-device-loss-storage.log`), **5 geometry LBVH/broad-phase tests** (`/tmp/molla-device-loss-geometry.log`) and **1 solver CPU/GPU trajectory parity test** (`/tmp/molla-device-loss-solvers.log`). These are scoped shared-runtime checks, not all workspace/device-failure acceptance or a new sensor performance report.
- GPU all-target, no-default compute/kernel/sensor/facade and viewer checks passed: `/tmp/molla-device-loss-final-check.log`, `/tmp/molla-device-loss-host-check.log`, `/tmp/molla-device-loss-viewer-check.log`. Explicit-file formatting and `git diff --check` passed (`/tmp/molla-device-loss-fmt-check.log`). No viewer launched; all owned jobs terminal.
- Added device-lifecycle documentation, updated snapshot/readback failure contracts, corrected stale IMU-only integration text, and checked only the S02 grouped-readback/loss/error-handling item. The full S02 gate remains open.

### Next Steps

Implement lifecycle-safe public producer composition and direct sensor dispatch into leased output slots; verify no unsubmitted/uncollected version can be reused and measure warmed fixed-capacity allocations. S04 top-level acceleration/refits/stable handles, remaining S00/S01 sizing/geometry/lifecycle audit and S08 viewer/visual/full-performance acceptance remain. S09 Gearbox is still unauthorized; S10 deferred. The complete objective remains active, not complete.

### Relevant Files

- `crates/molla-compute/src/engine.rs` / `lib.rs`, `crates/molla/src/compute/mod.rs` — sticky observed-loss signal, logical shutdown and public exports.
- `crates/molla-sensors/src/readback.rs` / `snapshot.rs` — loss guards on buffers, rings, snapshot views/pools and deferred batches.
- Sensor `imu.rs`, `contact.rs`, `frame_transform.rs`, `raycast.rs`, `lidar_batch.rs`, `tiled_camera.rs`, `camera_batch.rs`, `scene.rs` — guarded construction/dispatch/legacy collection.
- Sensor `image_processing.rs`, `cloud_processing.rs`, `cloud_grid.rs`, `annotation.rs` — known-loss rejection before GPU allocation/encoding/writes.
- `crates/molla-sensors/tests/sensor_device_loss_gpu.rs`, `crates/molla/tests/sensor_api.rs`, `Makefile` — destruction/callback/reconstruction/producer fixtures, facade types and compute host gate.
- `book/src/sensors/device_lifecycle.md`, `readback.md`, `snapshot.md`, `book/src/SUMMARY.md` — shutdown, observation, ownership and recovery limits.
- `SENSORS.md` / `SENSORS_PROGRESS.md` — verified lifecycle item and complete remaining scope.

## S02 direct producer outputs and canonical input ownership — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The preceding camera/renderer explanation was a read-only clarification, not implementation progress. Revalidated the plan and current frame-transform, snapshot, readback and storage code, then implemented public producer composition and direct leased output foundations rather than another copied-snapshot adapter.

### Instructions

Makefile entry points for formatting/builds/tests; patches for edits. Preserve unrelated work, Gearbox, authored assets, vehicle/soil/tyre behavior and application windows. No commits or pushes. Engram tools were searched and remain unavailable; this file records the handoff without claiming a memory-tool save.

### Discoveries

- The first cross-device regression failed because raw `wgpu::Device` equality accepted devices from separate instances. Inspection of the installed wgpu 27 native backend showed equality delegates to a local DeviceId without the instance/context identity. Canonical `Arc<Device>` identity, not raw Device equality, is now required by the new producer path.
- `Buffer::ensure_device` can preserve the canonical Arc when supplied at allocation. A sealed `BufferDevice` argument supports both Arc and legacy raw-device calls; existing raw mirrors are never retroactively certified. `State::upload_to_device` preserves the argument type. Legacy callers remain supported, but deferred producer encoding rejects unverified raw mirrors. Clone the engine Arc rather than wrapping a cloned raw device in a new Arc.
- Active frame-transform ranges start at element one when a reference exists; that byte offset is not generally storage-binding aligned. Direct bindings therefore expose full-capacity buffers at offset zero plus an explicit active range. Selected readback indices are relative to the active range.
- Pool buffers and binding allocations can remain constant while encoder, callback and metadata allocations still occur. The warmed resource counters are not proof of the entire S02 zero-allocation/performance gate. Binding caches also retain their referenced GPU buffers until eviction/sensor drop, even after pool drop.

### Accomplished

- Added `SensorFramePool`, typed channel specs/views, submitted immutable `SensorFrame`, opaque `EncodedSensorFrame` receipts, `SensorCommandBatch` and `SensorSubmission`. Pools enforce 1–64 slots/channels, checked element widths/capacities/device limits and a total output-buffer budget. Zero-capacity channels have one-element backing allocations but zero active/readback length.
- Frame-transform `encode` writes directly into four leased channel buffers without a snapshot D2D copy, submission, host collection or blocking wait. Multiple distinct producers compose into one submission. Producer/state borrows last through submit/drop; two compile-fail regressions cover redispatch and state mutation before submission.
- Pending receipts cannot expose channels or resolve against another batch. Dropped/failed/panicking batches cannot publish partial outputs. Empty batches submit nothing. Reset advances generation without reclaiming pending results; historical frames remain readable while healthy but are not current. CPU clones, pending consumers and readback leases prevent reuse; completion callbacks retain occupancy tokens only.
- Added `encode_with_frames` for checked raw read-only GPU consumers of already submitted frames, retaining up to 64 input references through completion. Device-only doubling of a direct frame position passes after caller frame release and delayed consumer submission. Raw callbacks remain responsible for their own resource correctness and must not mutate input bindings.
- Frame-transform direct bindings use a separate bounded 128-variant cache; legacy bindings retain two variants. Four slots × two state-pose buffers across 32 frames stabilize at eight preparations and 16 pool buffers, with zero sensor readback bytes. Batch counters distinguish encoded work from submission; legacy sensor submission counters are not incremented by unsubmitted encoding.
- Added canonical storage provenance checks without relaxing dirty-state checks, public facade exports/type checks, a direct-output book chapter and explicit remaining-plan status. Legacy `dispatch`/`update` behavior is retained; they still require caller-correct raw input device provenance.

### Verification

- **232 passing executions** in final `make test-sensors`, `/tmp/molla-direct-final-all2.log`: 117 GPU integration tests + 5 GPU unit tests, 91 GPU-feature sensor host unit tests, 3 compute host tests, 10 no-default sensor unit tests + 3 CPU-adjoint integrations, and 3 facade executions. No failures or skipped tests in the explicitly requested GPU lane.
- New direct-output suite: six real-GPU tests cover two-producer/65-target legacy parity on all four channels, full-width stamps, reference ranges, deferred GPU consumers, dropped/reset/pending versions, backpressure, pool/sensor drop, warm resource reuse, invalid/foreign/unverified inputs and known device loss. Two GPU unit tests cover empty channels, injected errors/panics and 64-output batch bounds; two new host unit tests cover channel sizing and fixed-cache reuse.
- Two producer/input borrow compile-fail doctests passed: `/tmp/molla-direct-final-doc.log`. One pre-existing ignored dataset doctest remains ignored; no claim it ran.
- Shared storage host tests: **3 passed**, `/tmp/molla-direct-storage.log`. Solver CPU/GPU parity: **1 passed**, `/tmp/molla-direct-solvers.log`. Geometry LBVH GPU regressions: **5 passed**, `/tmp/molla-direct-geometry.log`.
- GPU sensor all-target check passed, `/tmp/molla-direct-final-check2.log`; no-default compute/kernel/sensor/facade check passed, `/tmp/molla-direct-host-check.log`; viewer check passed, `/tmp/molla-direct-viewer-check.log`; **workspace/all-target `make check-all` passed**, `/tmp/molla-direct-workspace-check.log`.
- Explicit-file final formatting and `git diff --check` passed, `/tmp/molla-direct-final-fmt.log`. No viewer launch/visual acceptance or end-to-end performance claim. All owned test/check sessions are terminal.
- Intermediate red evidence retained: `/tmp/molla-direct-gpu.log` used the wrong readback stats field name in the new test; corrected to `copied_bytes`. `/tmp/molla-direct-gpu2.log` exposed the genuine raw-device equality problem described above. Subsequent canonical-provenance and full-suite runs passed. A later formatting-only check required wrapping the zero-capacity Vec3 size assertion; final formatting passed.

### Next Steps

Migrate remaining sensor producers (IMU/contact, ray queries/LiDAR/camera and processing) to composable direct leased outputs, including transactional history updates for stateful IMU; add typed pre-submission downstream composition and prove the full warmed allocation/performance gates. S04 top-level acceleration/refits/stable handles, remaining S00/S01 broader sizing/geometry/lifecycle contracts and baseline matrices, and S08 viewer integration/visual/full-performance evidence remain. S09 Gearbox is still unauthorized; S10 deferred. The complete goal remains active, not complete.

### Relevant Files

- `crates/molla-sensors/src/device_results.rs` — typed fixed-budget output pools, frame/channel leases, receipt validation, producer/consumer batches and unit regressions.
- `crates/molla-sensors/src/frame_transform.rs` / `dispatch.rs` — direct producer path, pool layout helper, borrowed lifetimes and configurable bounded binding cache.
- `crates/molla-storage/src/buffer.rs` / `lib.rs`, `crates/molla-sim/src/state.rs` — sealed raw/canonical device references, allocation provenance and checked current buffer access.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports and facade checks.
- `crates/molla-sensors/tests/sensor_direct_outputs_gpu.rs` — direct production, lifetime, numerical, ownership, consumer and resource-reuse fixtures.
- `book/src/sensors/device_results.md`, `frame_transform.md`, `book/src/SUMMARY.md`, `SENSORS.md` / `SENSORS_PROGRESS.md` — API contracts, ownership caveats, evidence and remaining complete scope.

## S02 direct IMU/contact production and transactional history — verified 2026-09-21

### Goal

Continue the complete `SENSORS.md` objective. The preceding goal turn was progress: it implemented and verified frame-transform direct slots, public producer batches and canonical input ownership. This increment migrates the other auxiliary producers and implements submission-time IMU history transactions rather than treating stateless frame composition as completion of S02.

### Instructions

Used Makefile tasks for formatting/checks/tests and patches for edits. Preserved unrelated WIP, Gearbox, vehicle/soil/tyre behavior, authored assets and user application windows. No commits or pushes. Engram tool discovery still exposes no context/save/session-summary tools; this file is the structured handoff, not a claimed memory-tool save.

### Discoveries

- IMU history is temporal state, not just an output buffer. Advancing `primed`, model identity or history dirty state at encode time makes cancelled batches corrupt later samples. The new path stages identity, encodes evaluation then the history copy, and commits CPU state only after queue submission. Cancellation drops unsubmitted commands and clears only the pending identity.
- Caller `dt` must measure time since the last submitted sample, not a cancelled or backpressured attempt. Pool generation reset does not reset an IMU filter. Legacy and direct APIs can share ordered GPU history without a readback or blocking wait.
- Submission actions can be bounded borrowed trait objects stored in a preallocated 64-entry vector, with Drop cancellation and no per-action Box/heap allocation. Borrowing the producer also prevents reset/redispatch while its history update is pending.
- Generalizing the old two-entry binding cache exposed an actual LRU bug: swapping a cache hit with the last entry changes the relative ages of other entries for capacities above two. A four-entry regression failed with `[1, 4, 3, 2]` instead of `[1, 3, 4, 2]`. Rotating the hit-to-end range preserves order without reallocating; regression and full lanes pass after the fix.

### Accomplished

- Added `ImuSensor::output_pool`/`encode`, `ContactSensor::output_pool`/`encode`, and public `EncodedImuSample`. All three auxiliary sensor kinds can now compose in one `SensorCommandBatch` and write direct leased slots rather than copying legacy output buffers.
- IMU returns `None` on backpressure without sample/history work. An encoded priming sample has `priming: true` and two empty active channels, but reserves a pool slot and encodes the initial velocity copy. Measured samples contain all mounts. No invented zero accelerometer readings are published for priming.
- Added bounded submission actions with commit/cancel behavior to the command batch, and encoded `history_copies` counters. Successful submission commits primed/model/dirty state; dropped, poisoned or known-device-loss submissions cancel staged changes. Recoverable validation errors before commands do not poison earlier valid work. Actions remain private to producer implementations.
- Contact direct outputs contain count, maximum depth, signed normal sum and centroid. Empty contacts clear all active body outputs without aggregation; zero-body sensors publish empty channels. Nonempty arrays and CSR must be current and owned by the canonical engine device. Contact sensing remains geometric, not force sensing.
- Preserved canonical device ownership through `Model::upload_to_device` and `Contacts::upload_to_device`, using the sealed raw/Arc device argument introduced in the previous increment. Raw-device compatibility remains; unverified mirrors are not retroactively certified and are rejected by new deferred producers. Added model/state dimension checks before IMU reservation/history work.
- Each migrated producer retains a separate 128-entry direct binding cache plus its existing two-entry legacy cache. The paired warm test cycles four slots and two inputs for 24 measurements after priming: eight preparations per producer, eight IMU output buffers and 16 contact output buffers, zero sensor readback bytes/waits. Binding caches may retain dropped-pool buffers until eviction/sensor drop; no entire-pipeline allocation/performance claim.
- Added facade exposure/checks and documented priming, cancellation, mixed submission, canonical input ownership, timestep responsibility and unchanged legacy paths. Fixed the larger-cache eviction ordering noted above.

### Verification

- **240 passing executions** in final `make test-sensors`, `/tmp/molla-aux-direct-final-all2.log`: 123 GPU integration tests + 6 GPU unit tests, 92 GPU-feature sensor host unit tests, 3 compute host tests, 10 no-default sensor unit tests + 3 CPU-adjoint integrations, and 3 facade executions. Every explicitly requested ignored GPU test ran; no failures.
- Six new auxiliary direct-GPU integration tests cover: mixed two-IMU/contact/frame production at 70 mounts with independent IMU timesteps and legacy parity; independent central-mount acceleration/gyro values; priming cancellation from a different same-sized model; dropped/error/panicking measured batches; direct/legacy history continuity; backpressure; reset/replacement and retained versions; empty/zero-body contacts; foreign, unverified, dirty and malformed input rejection; known loss; and warmed multi-slot/two-input resource reuse.
- New GPU unit regression covers all 64 submission actions, fixed-vector reuse, exactly-once commit, and cancellation on drop, subsequent producer error and device loss. The new host cache-order regression is red/green in `/tmp/molla-aux-direct-lru-red.log` and `/tmp/molla-aux-direct-lru-green.log`, and green in the final full suite.
- **Five compile-fail doctests passed** (three new), `/tmp/molla-aux-direct-final-doc.log`: frame producer/state mutation, IMU reset/model mutation, and contact mutation are rejected while the batch is pending. The pre-existing ignored dataset doctest was not run.
- GPU sensor all-target check passed, `/tmp/molla-aux-direct-final-check.log`; no-default compute/kernel/sensor/facade check passed, `/tmp/molla-aux-direct-host-check.log`; **workspace/all-target `make check-all` including viewer passed**, `/tmp/molla-aux-direct-final-workspace-check.log`.
- Shared solver CPU/GPU parity: **1 passed**, `/tmp/molla-aux-direct-solvers.log`; geometry LBVH GPU regressions: **5 passed**, `/tmp/molla-aux-direct-geometry.log`. Explicit-file formatting and `git diff --check` passed, `/tmp/molla-aux-direct-fmt.log`.
- No new benchmarks or visual acceptance claims. All owned check/test sessions are terminal. Initial direct tests and full lanes were green; the intentional new cache-order regression exposed the separate LRU defect, which is fixed and reverified.

### Next Steps

Migrate ray-query/LiDAR/camera and image/cloud/grid/annotation producers to public composable direct-output APIs; implement typed pre-submission downstream composition rather than requiring separately submitted consumer batches; verify full warmed allocations and completed-work performance. Preserve temporal transaction guarantees when adding more stateful producers. S04 top-level acceleration/refits/stable handles, remaining S00/S01 sizing/geometry/lifecycle and baseline matrices, and S08 viewer/visual/full-performance acceptance remain. S09 Gearbox remains unauthorized and S10 deferred. The complete objective remains active, not complete.

### Relevant Files

- `crates/molla-sensors/src/device_results.rs` — bounded borrowed commit/cancel actions, history-copy counters and transaction regression.
- `crates/molla-sensors/src/imu.rs` / `contact.rs` — direct auxiliary producers, channel/pool contracts, IMU staged history and compile-fail contracts.
- `crates/molla-sensors/src/dispatch.rs` — correct larger-cache LRU ordering and allocation-preserving regression.
- `crates/molla-sim/src/model.rs` / `contacts.rs` — canonical device ownership preserved through uploads, with legacy raw-device support.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — encoded IMU sample exports/type check.
- `crates/molla-sensors/tests/sensor_aux_direct_gpu.rs` — mixed production, temporal cancellation, lifecycle, input validation and warm resource fixtures.
- `book/src/sensors/device_results.md`, `imu.md`, `contact_sensor.md`, `SENSORS.md` / `SENSORS_PROGRESS.md` — new API/history contracts, validation evidence and remaining full scope.

## S02 owned-scene direct rays and scoped pending GPU chains — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The preceding implementation turn made progress on owned-scene direct ray queries; its terminal test/check handles were revalidated here. The intervening camera explanation made no source changes. This continuation completes that evidence record and adds typed access to pending outputs and custom downstream production in the same submission. Neither this increment nor the full objective is reduced to passing this new fixture.

### Instructions

Used Makefile tasks for formatting, checks, doctests and GPU tests, and apply_patch for edits. Preserved unrelated WIP, Gearbox, machine assets, tyres/soil and user windows. No commits or pushes. Engram discovery still returned no context/save/session-summary tools, so this structured project handoff is not a claimed memory-tool save.

### Discoveries

- Queue writes execute before encoder commands. Direct host-ray uploads must be excluded from encoder clears or their origins/directions are erased before tracing. The private initialization mask skips these two channels and clears the four hit channels. Cancelled batches discard tracing commands, not already queued uploads; ordered slot initialization protects later reuse.
- Pending consumers can resolve earlier receipt identities without publishing a CPU-readable frame. Fixed stack storage holds temporary Arc clones for up to 64 inputs; scoped typed channel views expose checked ranges and bindings but no readback/conversion path. Raw wgpu cloning remains a trusted-caller escape hatch, not something Rust view lifetimes can prevent.
- Custom callbacks introduce caller-reported dispatch counts. Their addition is now checked; overflow poisons the batch instead of wrapping or partially publishing outputs. Shader read-only access, range compliance, resource provenance and truthful counters remain explicit raw-callback responsibilities.

### Accomplished

- `Raycaster::output_pool` and `encode_scene` write six direct channels for owned scenes: origin, normalized direction, distance, shape ID, point and normal. Primitive, moving/shared-mesh, converted terrain and triangle paths share one encoder with auxiliary producers. Separate `SceneStamp` preserves authoritative geometry identity independently of caller metadata.
- Host rays are finite-validated and robustly normalized before queued packing. Active input upload bytes are counted; empty queries reserve empty frames without kernels/uploads, and saturated pools do no query work. Direct binding variants are cached across slot and pose-buffer reuse. Legacy raw-model and internal generated-ray APIs remain intact.
- Added `PendingSensorInputs` and `PendingSensorChannel<T>`, with typed channel lookup, metadata, length, checked relative slices and full-capacity bindings. Nonzero active offsets remain explicit, not misrepresented as storage-aligned buffer offsets. A compile-fail doctest rejects a view escaping its callback.
- Added `SensorCommandBatch::encode_with_pending` for custom consumers and `encode_from_pending` for custom producers writing newly leased output slots. These support mixed-input, multi-stage dependencies before the single submission. Custom outputs can immediately become inputs to another stage. Output channels are cleared before callbacks; writes must use the supplied encoder, not queue writes.
- Receipt identity, input count, pool ownership/layout/ranges and known loss are validated before work. Backpressure skips the callback and clears. Callback error/panic or dispatch-count overflow poisons the complete batch. Dropped chains release slots while surviving opaque receipts retain no output leases. Chains share the existing 64-output bound; each callback accepts 1–64 pending references, including duplicates.
- Exported pending view types through the facade; documented raw-resource responsibilities and the distinction between custom composition support and still-unmigrated built-in processing wrappers. No renderer/Gearbox integration was added.

### Verification

- Recovered ray increment: **246 passing executions** in `/tmp/molla-ray-direct-final-all.log`; the exact owned session returned exit 0. GPU/no-default/workspace checks and six compile-fail doctests also returned exit 0: `/tmp/molla-ray-direct-final-check.log`, `/tmp/molla-ray-direct-host-check.log`, `/tmp/molla-ray-direct-workspace-check.log`, `/tmp/molla-ray-direct-doc.log`.
- Ray integration suite adds six GPU fixtures: mixed moving primitive/mesh legacy parity and independent sensor ranges; mesh/analytic ties, strict maximum range, empty queries/scenes; cancellation/backpressure/replacement and old leases; four-slot/two-pose warm binding reuse; malformed/foreign/unverified/device-loss rejection; and terrain/triangle intersections checked against independent plane equations. No new performance benchmark claim.
- Final combined `make test-sensors`: **250 passing executions**, `/tmp/molla-pending-all.log`: 131 GPU integrations + 8 GPU unit tests, 92 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests + 3 CPU-adjoint integrations, and 3 facade executions. The explicitly requested ignored GPU lane ran all its tests without failure.
- Two new integration tests exercise a real 65-element mixed frame/ray shader followed by another leased stage and final consumer in one submission, checking independent numerical values, full-width stamps, nonzero ranges, historical frames and post-owner-drop collection. Validation coverage includes foreign/abandoned receipts, invalid types/ranges/counts, backpressure, cancellation, errors/panics and known loss.
- Two new GPU unit tests cover empty channels, foreign pools, wrong layouts, custom-output callback error/panic, dispatch-count overflow in both new APIs, the 64-stage/output bound, and complete slot release after dropped chains.
- **Seven compile-fail doctests passed**, `/tmp/molla-pending-final-doc.log`. The pre-existing ignored dataset doctest remains ignored. GPU all-target check, no-default check and workspace/all-target check including viewer passed: `/tmp/molla-pending-final-check.log`, `/tmp/molla-pending-host-check.log`, `/tmp/molla-pending-workspace-check.log`.
- Explicit-file formatting including recovered ray changes passed, `/tmp/molla-pending-fmt.log`; `git diff --check` passed. All owned sessions are terminal. Initial pending integration compilation used an extra plane-builder argument (`/tmp/molla-pending-gpu.log`); corrected to the actual three-argument API, then both targeted and full suites passed.
- No viewer launch, visual acceptance, whole-pipeline allocation-free claim or performance-completion claim.

### Next Steps

Migrate generated-ray/LiDAR/camera and built-in image/cloud/grid/annotation producers to the direct-output and pending-input APIs, preserving cancellation and bounded resource semantics. Prove the full warm allocation/completed-work performance gates rather than inferring them from fixed output counters. S04 top-level acceleration/refits/stable handles, remaining S00/S01 broad sizing/geometry/contracts/baselines, and S08 viewer/visual/performance acceptance remain. S09 Gearbox remains unauthorized, S10 deferred. The complete goal remains active and uncompleted.

### Relevant Files

- `crates/molla-sensors/src/device_results.rs` — scoped typed pending views, custom consumer/output APIs, checked dispatch accounting, private initialization mask and GPU unit regressions.
- `crates/molla-sensors/src/raycast.rs`, `crates/molla-kernels/src/raycast_merge.rs`, `crates/molla-sensors/src/dispatch.rs` — direct owned-scene queries, explicit merge bindings and retained cache handling.
- `crates/molla-sensors/tests/sensor_ray_direct_gpu.rs` — six direct ray fixtures; `sensor_pending_pipeline_gpu.rs` — real mixed GPU chain and lifecycle fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — direct-ray and pending-view public exports/type checks.
- `book/src/sensors/device_results.md`, `raycaster.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — API/range/lifetime contracts, verification and remaining full scope.

## S02 built-in image processing on direct and pending outputs — verified 2026-09-21

### Goal

Continue the complete `SENSORS.md` objective. The previous goal turn was progress: scoped pending inputs and custom GPU chains were implemented and verified. This increment migrates an actual built-in processing wrapper onto that foundation instead of treating the custom callback API as completion of S02.

### Instructions

Used Makefile formatting/check/test targets and apply_patch. Preserved unrelated worktree edits, Gearbox, assets, tyre/soil behavior and user application windows; no commits or pushes. Memory registry lookup found no relevant sensor-composition entries; Engram tool discovery again exposed no context/save/session-summary tools. This structured project log is the available handoff, not a claimed memory-tool save.

### Discoveries

- Direct image processors can reuse their existing reduction scratch, dynamic uniform-stage bank and compute pipelines if each processor remains mutably borrowed until submission/drop. Output-slot-specific bind groups need no additional GPU buffers. Separate processors can then have independent pending parameters in one batch without overwriting each other.
- Retaining source buffers only until submission is insufficient for pooled snapshots/direct frames. The batch now holds their occupancy tokens through GPU completion, with device-source jobs and submitted-frame consumers sharing the same 64-reference budget. Pending inputs already owned by the batch do not consume that external-input budget.
- Existing exposure semantics are a linear multiplier, not stops; empty gray-world gains are one while empty white-patch gains are one million. Two initial test expectations were wrong on those points. The CPU and shader implementations agreed; only tests/docs were corrected, not the established formulas.

### Accomplished

- Added `ImageProcessor::output_pool`, `encode` for existing `ImageJob` device ranges, and `encode_pending` for new `PendingImageJob` receipt/channel/relative-range inputs. Both produce `EncodedProcessedImages` with an opaque frame receipt and independent per-image layouts/stamps. Device-only and device-authoritative inputs require no host access or intermediate submission.
- Output channels contain packed u32 pixels and Vec3 per-stage gains. Inputs copy directly into the leased working pixels and all adjustments run in place; no legacy output snapshot copy is involved. White-balance gains occupy their stage slots; other active gain entries are cleared to zero. Empty direct batches reserve empty frames and encode clears, explicitly distinct from legacy `process`'s empty no-submission behavior.
- Refactored `ImageProcessKernel` to prepare output-specific bindings and encode through them while sharing original scratch/uniform/pipeline resources. The legacy API remains on its original binding. A bounded 128-entry direct binding cache retains slot variants; cached GPU buffers persist until eviction/processor drop even if a pool is dropped.
- Added bounded device-source lease retention to `SensorCommandBatch`, with shared input-count validation and completion callbacks holding occupancy tokens rather than device-owning objects. Drop, poison and known device loss release unsubmitted source leases. Per-frame metadata allocations and whole-pipeline allocation/performance gates are still not claimed complete.
- Whole-list preflight rejects bad types/devices/receipts/ranges, duplicate IDs, excessive images/pixels/stages, nonfinite operation values and invalid times without reserving outputs or poisoning earlier valid work. Backpressure performs no copies, parameter writes or binding preparation. Image processor/source borrows protect against pending reuse and input mutation.
- Added public exports, documented pool budgets, ranges, empty behavior, source ownership and encoded-versus-submitted counters, and updated remaining plan scope. No Gearbox or camera-renderer integration was implemented.

### Verification

- Final `make test-sensors`: **257 passing executions**, `/tmp/molla-image-direct-final-all2.log`: 137 GPU integrations + 9 GPU unit tests, 92 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests + 3 CPU-adjoint integrations, and 3 facade executions. Every explicitly requested ignored GPU test ran without failure.
- Six new integration tests in `sensor_image_direct_gpu`: all ten operation types and multi-stage image chains with exact legacy GPU pixel/gain parity; cancelled uniform writes/backpressure/reset/historical output survival; four-slot warm reuse with varying counts/exposure and snapshot inputs; preflight/device-loss rejection; shared 64-external-input bounds while pending chains remain usable; and relative image ranges over a custom frame with nonzero active offsets. Independent fixtures check doubled packed colors, per-image gray-world output and empty gains. Source host access is explicitly rejected after marking the source device-authoritative.
- The warm fixture submits 24 frames over four slots, stabilizing at four direct bind-group preparations plus the initial legacy group, eight fixed pool buffers and the original four processor buffers. A subsequent no-operation copy adds no binding preparation. These are resource-count observations, not whole-pipeline allocation-free or speed acceptance.
- A new GPU unit test verifies all 64 retained source leases fit preallocated vector storage and are released on successful completion, drop, later callback failure and known loss. Existing producer, consumer and snapshot lifecycle tests also pass with the shared input budget.
- **Ten compile-fail doctests passed**, `/tmp/molla-image-direct-final-doc.log`; three new tests protect both direct/pending processor reuse and external source mutation. The existing dataset doctest remains explicitly ignored.
- GPU sensor all-target and no-default compute/kernel/sensor/facade checks passed: `/tmp/molla-image-direct-final-check.log`, `/tmp/molla-image-direct-host-check.log`. Final workspace/all-target check including viewer passed: `/tmp/molla-image-direct-final-workspace-check.log`. Explicit-file formatting and `git diff --check` passed, `/tmp/molla-image-direct-final-fmt.log`.
- Initial incorrect expectations are retained in `/tmp/molla-image-direct-gpu.log` (exposure multiplier) and `/tmp/molla-image-direct-gpu2.log` (empty white-patch gains); corrected targeted and full runs are green. All owned sessions are terminal. No new benchmark, viewer launch or visual acceptance claim.

### Next Steps

Migrate generated-ray/LiDAR/camera and cloud/grid/annotation wrappers to direct outputs and pending-input composition. Verify full warm allocation/completed-work performance rather than equating fixed pool counters with performance completion. S04 top-level acceleration/refits/stable handles, remaining S00/S01 broad contracts/baselines, and S08 viewer/visual/performance work remain. S09 Gearbox is unauthorized; S10 deferred. The original goal remains active, not complete.

### Relevant Files

- `crates/molla-sensors/src/image_processing.rs` — direct/pending jobs, validated layout planning, image pool contract, encoded results and retained direct bindings.
- `crates/molla-kernels/src/image_process.rs`, `crates/molla-kernels/src/lib.rs` — reusable explicit output bindings and shared image dispatch implementation.
- `crates/molla-sensors/src/device_results.rs` — source occupancy retention, shared external-input bounds, preflight pending-range helper and lease regression.
- `crates/molla-sensors/tests/sensor_image_direct_gpu.rs` — numerical, multistage, lifecycle, offset, capacity and warm-resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public result/job exports and facade type checks.
- `book/src/sensors/image_processing.md`, `device_results.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current APIs, caveats, evidence and remaining full scope.

## S02 direct annotations and deduplicated pending image dependencies — verified 2026-09-21

### Goal

Continue the original `SENSORS.md` objective. The preceding turn was progress: built-in image processing gained verified direct/pending APIs. This increment applies those lifecycle and composition contracts to class/instance masks, bounding boxes and annotation counts. It does not claim the remaining sensing, acceleration, allocation/performance or viewer gates are complete.

### Instructions

Used Makefile tasks and apply_patch, preserved unrelated worktree changes and all Gearbox/assets/tyre/soil boundaries, and did not launch or close user applications. No commits/pushes. Memory lookup found no relevant sensor-composition registry entries, and Engram discovery still returned no context/save/session-summary tools. This structured project log is the available handoff, not a claimed memory-tool save.

### Discoveries

- Logical image count and dependency count are different bounds. Annotation supports 1024 images; repeating a pending receipt per image must not exhaust the command batch's 64-dependency limit. The new pending path validates every range but deduplicates receipts by owning batch identity and output index, including cloned receipts.
- Arc value equality is not receipt identity when the owner value is `()`. A dedicated host regression checks owner-pointer identity and distinct output indices rather than permitting foreign batch receipts to alias by equal unit values.
- A label revision belongs to its processor, not a globally unique label table or scene. Encoded results record the revision used, and historical output leases preserve old data after relabelling; callers still need their own producer/scene identity when interpreting metadata.

### Accomplished

- Added `AnnotationProcessor::output_pool`, `encode` for existing device-source `AnnotationJob` lists, and `encode_pending` for typed receipt/channel/range `PendingAnnotationJob` lists. `EncodedAnnotations` contains the opaque result receipt, per-image layouts and captured label revision.
- The four direct channels are u32 shapes, u32 classes, packed `DeviceBoundingBox` records and u32 counters. Active box extents use the label count at encoding, not a later replacement table. Shape inputs copy straight into the leased working channel, and existing kernels write other outputs there without a legacy-output snapshot copy.
- Reused the existing label/descriptor/scratch/parameter buffers and pipelines through explicit `AnnotationBindings`, with a bounded 128-entry direct binding cache. Mutable processor borrowing protects descriptors and labels until submit/drop; source jobs and occupancy tokens use the shared batch lifetime/completion machinery.
- Pending jobs retain configured capacity through 1024 logical images using deduplicated dependencies. External device-source jobs explicitly share the remaining 64-input budget with other producers/consumers; excess external job lists are rejected before work, not truncated or secretly split. Legacy `process` retains its configured capacity. Large composed pipelines can select many image ranges from an aggregate pending channel.
- Whole-list preflight validates devices/types, receipts, checked dimensions and total pixels, selected range lengths, duplicate IDs and times. Bad jobs leave earlier commands usable. Backpressure does no copies/descriptor writes/preparation. Empty direct batches reserve cleared frames with empty active ranges; empty-label images still report misses/unlabelled pixels. No hidden host access, internal submission or readback was added.
- Exported the new job/result types through the facade and documented old/new API distinctions, processor-local revisions, pool/cache memory caveats, empty behavior and encoded-versus-submitted statistics. Updated current plan/progress rows without closing S02 prematurely.

### Verification

- **264 passing executions** in final `make test-sensors`, `/tmp/molla-annotation-direct-all.log`: 143 GPU integrations + 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests + 3 CPU-adjoint integrations, and 3 facade executions. All explicitly requested ignored GPU tests ran without failure.
- Six new GPU integrations cover independent label-table chains in one submission against CPU masks/bounds/counts; 1024 per-image jobs from cloned receipts over a nonzero source active offset; cancellation of differing descriptor layouts, relabelling, reset/backpressure and historical versions; four-slot warm reuse with changing labels/counts and snapshot inputs; malformed/foreign/type/capacity/known-loss rejection before work; and explicit external-input bounds with empty-image/label/batch handling.
- The warm fixture submits 24 frames across four output slots and six label tables with four direct binding preparations plus the legacy group, 16 fixed pool buffers and the original eight processor buffers. A final snapshot-input frame verifies source occupancy release after completion. These are resource-reuse observations, not a speed or whole-pipeline allocation-free claim.
- One new host test verifies receipt deduplication's batch identity/index semantics. **13 compile-fail doctests passed**, `/tmp/molla-annotation-direct-doc.log`; three new cases reject pending relabelling, processor reuse and external source mutation. The pre-existing dataset doctest remains ignored.
- GPU sensor all-target, no-default compute/kernel/sensor/facade and workspace/all-target checks including viewer passed: `/tmp/molla-annotation-direct-final-check.log`, `/tmp/molla-annotation-direct-host-check.log`, `/tmp/molla-annotation-direct-workspace-check.log`. Explicit-file format check passed, `/tmp/molla-annotation-direct-fmt.log`; `git diff --check` passed.
- Initial new test compilation passed a usize image index to the CPU oracle's u32 API (`/tmp/molla-annotation-direct-gpu.log`); corrected the bounded test index, then all six targeted GPU tests and the full suite passed. No runtime red failure remained. All owned sessions are terminal; no new benchmark or visual acceptance claim.

### Next Steps

Migrate generated-ray/LiDAR/camera and cloud/grid producers to direct output and pending composition. Prove full warmed allocation and completed-work performance gates. S04 top-level acceleration/refits/stable handles, remaining S00/S01 broad contract/baseline coverage, and S08 viewer/visual/performance acceptance remain. S09 Gearbox is unauthorized; S10 deferred. The full goal stays active and incomplete.

### Relevant Files

- `crates/molla-sensors/src/annotation.rs` — direct annotation pools, device/pending encodes, preflight planning and encoded revision/layout results.
- `crates/molla-kernels/src/annotation.rs`, `crates/molla-kernels/src/lib.rs` — explicit reusable annotation bindings and shared dispatch implementation.
- `crates/molla-sensors/src/device_results.rs` — private receipt identity comparison and host regression.
- `crates/molla-sensors/tests/sensor_annotation_direct_gpu.rs` — six direct/pending numerical, lifecycle, capacity and warm-resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports and facade checks.
- `book/src/sensors/annotation.md`, `device_results.md`, `image_processing.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — API/dependency/identity contracts, current evidence and full remaining scope.

## S02 direct cloud grids and typed pending spatial reductions — verified 2026-09-21

### Goal

Continue the complete `SENSORS.md` objective. The previous goal turn was progress: annotation masks/boxes/counts gained verified direct and pending composition. This increment migrates cloud-grid spatial membership and per-cell reductions without treating grid support as completion of cloud construction/filtering or the full S02/S08 gates.

### Instructions

Used Makefile targets for formatting, checks and tests, and apply_patch for edits. Preserved unrelated WIP, Gearbox, machine assets, tyre/soil code and all user application windows. No commits/pushes. No relevant memory-registry hits were found; Engram discovery again exposed no context/save/session-summary tools. This structured project handoff is not a claimed memory-tool save.

### Discoveries

- A pending grid must preserve two offsets separately: the selected range inside a source channel and the channel's active start inside its buffer. The direct copy uses their combined byte offset; membership indices remain local to the copied output, not the original source indices. Tests cover both offsets with packed 112-byte records.
- The existing reduction divides contributions by member count before summing. Four identical maximum-f32 intensities therefore remain finite; a new test initially assumed a naive overflowing sum. The established shader formula was retained and the test now checks the independent finite result. Other normalized reductions can still trigger the existing overflow flag contract.
- Grid output bindings can vary independently from shared membership/prefix scratch, block offsets and uniforms. Borrowing each grid until submit/drop prevents pending parameter reuse, while distinct grids can compose different dimensions and bounds in one submission.

### Accomplished

- Added `CloudGrid::output_pool`, `encode` for a checked external device range, and `encode_pending` for `PendingCloudGridInput`. `EncodedCloudGrid` records the opaque frame, specification, input count and cell count. Channels contain copied `CloudPoint` records, membership indices, `CloudGridCell` summaries and six diagnostics.
- Inputs copy straight into the leased working point channel; the five existing passes write indices/cells/diagnostics there, without copying legacy outputs or collecting data on the host. Whole-request device/type/range/capacity/spec/time checks run before reservation. Backpressure does no clears, copies, uniform writes or binding preparation.
- Refactored `CloudGridKernel` to support explicit `CloudGridBindings`, retaining original scratch/uniform/pipeline resources and the legacy group. A 128-entry output-binding cache reuses slots across changing counts/specs. Processor buffers remain eight; per-pool output budgets are separate and cached groups retain their buffers until eviction/grid drop.
- Empty inputs reserve fresh zeroed summaries/diagnostics with empty point/index ranges and no kernels. Nonempty all-rejected inputs still execute all five passes and produce sentinel index tails. Packed records preserve all bytes, including full-width IDs, reserved data and NaN payloads in nonparticipating attributes.
- External device inputs use the existing shared 64-reference budget and completion-held occupancy tokens; pending inputs already belong to the batch. Source/grid borrowing protects unsubmitted work. Cancellation, reset/historical frames and downstream-poison behavior follow the shared batch contract. Legacy `process` keeps its submission behavior and now shares source/spec validation helpers.
- Exported the new input/result types through the facade, documented channel/range/budget/statistics semantics, and updated the live plan without closing broader performance or lifecycle gates.

### Verification

- Final `make test-sensors`: **269 passing executions**, `/tmp/molla-grid-direct-all.log`: 148 GPU integrations + 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests + 3 CPU-adjoint integrations, and 3 facade executions. All explicitly requested ignored GPU tests ran without failure.
- Five new GPU integrations cover: typed multi-grid chains with shifted active ranges and 66,049 cells requiring two-dimensional dispatch; empty/rejected/extreme finite reductions; four-slot warm reuse across specs/counts and snapshot sources; cancellation/reset/backpressure/history and poisoned downstream work; and malformed/foreign/type/capacity/time/known-loss rejection before reservation. CPU membership is independent, cell-local indices are sorted for comparison, and means use the previously recorded 2e-4 relative/absolute-floor tolerance rather than claiming deterministic f32 sums.
- The warm fixture submits 24 frames across four slots with four direct preparations plus the initial legacy group, 16 fixed pool buffers and eight original grid buffers. Snapshot reset/drop and subsequent completion release are verified. No whole-pipeline allocation-free or speed claim is inferred from these counters.
- **15 compile-fail doctests passed**, `/tmp/molla-grid-direct-doc.log`; two new tests reject input mutation and pending grid reuse. The reuse test constructs a second input descriptor, so it tests grid borrowing rather than a moved-value error. The pre-existing dataset doctest remains ignored.
- GPU sensor all-target, no-default compute/kernel/sensor/facade and workspace/all-target checks including viewer passed: `/tmp/molla-grid-direct-final-check.log`, `/tmp/molla-grid-direct-host-check.log`, `/tmp/molla-grid-direct-workspace-check.log`. Explicit-file formatting and `git diff --check` passed, `/tmp/molla-grid-direct-fmt.log`.
- Initial grid tests incorrectly expected overflow for four maximum-f32 intensities (`/tmp/molla-grid-direct-gpu.log`, repeated before correction in `gpu2.log`). Source inspection showed normalized contributions; corrected targeted tests and the full suite pass. No shader change or silent tolerance relaxation was made. All owned sessions are terminal; no new benchmark, viewer launch or visual acceptance claim.

### Next Steps

Migrate cloud construction/filtering and generated-ray/LiDAR/camera producers to direct outputs and pending composition. Prove full warmed allocation and completed-work performance gates. S04 top-level acceleration/refits/stable handles, remaining S00/S01 broad contracts/baselines and S08 viewer/visual/performance acceptance remain. S09 Gearbox stays unauthorized; S10 deferred. The original goal remains active, not complete.

### Relevant Files

- `crates/molla-sensors/src/cloud_grid.rs` — direct/pending APIs, pool/result contracts, common validation and retained direct bindings.
- `crates/molla-kernels/src/cloud_grid.rs`, `crates/molla-kernels/src/lib.rs` — explicit output bindings over shared grid scratch and pipelines.
- `crates/molla-sensors/tests/sensor_grid_direct_gpu.rs` — five numerical/composition/lifecycle/budget/warm-resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — input/result exports and facade type checks.
- `book/src/sensors/cloud_grid.md`, `device_results.md`, `image_processing.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current API semantics, numerical observations, evidence and remaining full scope.

## S02 direct cloud construction/filtering and pending compaction — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The previous implementation turn was progress, interrupted during regression verification; its exact live handles were resumed rather than restarting unknown work. This increment migrates cloud construction, transforms, filtering and stable compaction into the shared direct-output pipeline. It does not close S02 or the overall goal.

### Instructions

Use Makefile validation and apply_patch edits. Preserve unrelated WIP, assets and Gearbox; no commits or viewer/window operations. The user's renderer question was answered from current source: Molla's camera traces its own sensor scene with simple colour shading, not Gearbox's framebuffer/materials. Renderer RGB integration and shared-device interoperability remain separate work. Engram tool discovery still returns no memory tools; this file remains the project handoff.

### Discoveries

- A cloud job has either one packed-record source or five differently typed ray sources. External reference budgets count every source, not jobs; 60 empty packed jobs plus one ray job exceed the shared 64-reference bound, while 59 plus one fit.
- Pending jobs can share receipts, including clones, without consuming one dependency per logical job. Deduplicated dependencies preserve the configured 1024-job limit. Empty jobs require valid zero summaries even when the job count exceeds the actual point count.
- Pending ranges must include the source channel's active offset as well as the selected relative offset. The typed copy helper uses each element's GPU stride, including 4-byte GPU `Real` values rather than the host width.
- Cloud counts remain on device. The published point range is an input-count upper bound; zero tails are rejected as misses by downstream processors/grids. This is not survivor-count indirect dispatch.

### Accomplished

- Added `CloudProcessor::output_pool`, external `encode` and typed `encode_pending`, with `EncodedCloud`, `PendingCloudSource`, `PendingCloudInput` and `PendingCloudJob`. Results contain compacted `CloudPoint` records and `u32` total/per-job count words; layouts and input-count metadata accompany the receipt.
- The existing three passes write directly into leased output slots. Input copies still populate shared working-point/raw-ray scratch; no legacy output copy or intermediate collection is added. Per-slot budget is `112*N + 4*(1 + 5*J)` bytes, separate from original processor storage.
- `CloudProcessKernel` now supports retained explicit output bindings over its original ranks, block scratch and uniform. The processor caches 128 variants. Borrowing the processor and external inputs through submission prevents safe API reuse of pending scratch/parameters; independent processors compose freely.
- Shared metadata validation covers legacy/direct/pending paths, including raster identity/products/source-index overflow, transforms/filters, point/job budgets and finite nonnegative timestamps. External validation checks canonical device and typed source widths; pending views enforce active ranges and receipt ownership before reservation. Legacy planning streams checked metadata rather than adding an extra metadata vector.
- Empty input clears fresh counts without kernels; nonempty rejected input still dispatches three passes. Completion-held source leases, cancellation, poison, generation reset, backpressure and retained historical output semantics use the shared batch contract. Encoded counters explicitly include cancelled work; legacy frame/submission counters keep their original meaning.
- Exported all new types through the facade, documented direct budgets/channel order/upper-bound tails, corrected stale cloud documentation and updated the live plan without marking broader gates complete.

### Verification

- Final `make test-sensors`: **275 passing executions**, `/tmp/molla-cloud-direct-final-all.log`: 154 GPU integrations, 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests, 3 CPU-adjoint integrations and 3 facade executions. Explicit ignored GPU tests ran. The preceding full run also passed, `/tmp/molla-cloud-direct-all.log`.
- Six new GPU tests cover mixed external packed/ray input with offsets and full-width metadata; ray-query → cloud construction; cloud → cloud → grid in one submission; 1024 logical pending jobs with empty jobs, cloned receipts and shifted active ranges; four-slot reuse across 24 frames with changing counts/filters/transforms and snapshot sources; cancellation/reset/history/backpressure/poison; malformed/foreign/lost-device rejection before reservation; and external per-channel budgets with fresh empty summaries. Packed survivor checks are byte-exact, including reserved NaN payloads. Warm counters remain four direct binding preparations plus the initial legacy group, eight pool buffers and eight original processor buffers.
- **18 compile-fail doctests passed**, `/tmp/molla-cloud-direct-final-doc.log`, including three new cloud producer/input-borrow tests. The pre-existing dataset doctest remains ignored.
- GPU sensor all-target check passed (`/tmp/molla-cloud-direct-final-check.log`); final no-default compute/kernel/sensor/facade check passed (`/tmp/molla-cloud-direct-final-host-check.log`); final workspace/all-target check including viewer passed (`/tmp/molla-cloud-direct-final-workspace-check.log`). Explicit-file formatting and `git diff --check` passed (`/tmp/molla-cloud-direct-fmt.log`).
- The initial sandbox GPU run failed before sensor execution because no Vulkan adapter was accessible (`/tmp/molla-cloud-direct-gpu.log`). Approved outside-sandbox execution passed four initial and then six expanded GPU tests (`/tmp/molla-cloud-direct-gpu6.log`), followed by both full-suite passes. No fake adapter or silent test skipping was used. All owned validation sessions are terminal. No benchmark, viewer launch or visual acceptance claim is made.

### Next Steps

Migrate generated-ray/LiDAR/camera producers into direct leased outputs and pending composition. `Raycaster::encode_owned_scene` already traces arbitrary output buffers internally; generated producers currently use the separate `encode_generated`/legacy scratch path. LiDAR layout/reduction bindings and camera output/accumulation bindings need retained per-slot variants plus matching validation/lifecycle tests. Complete the full warmed allocation and completed-work performance gates, S04 top-level acceleration/refits/stable handles, remaining S00/S01 contracts/baselines, and S08 viewer/visual/performance acceptance. S09 Gearbox remains unauthorized; S10 deferred. Keep the original goal active.

### Relevant Files

- `crates/molla-sensors/src/cloud_processing.rs` — direct/pending APIs, shared metadata validation, budgets, lifetimes and retained bindings.
- `crates/molla-kernels/src/cloud_process.rs`, `crates/molla-kernels/src/lib.rs` — explicit cloud output bindings with unchanged kernels and shared scratch.
- `crates/molla-sensors/tests/sensor_cloud_direct_gpu.rs` — six numerical/composition/lifecycle/resource regression fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports and facade identity checks.
- `book/src/sensors/cloud_processing.md`, `book/src/sensors/device_results.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — API contracts, verification evidence and remaining full scope.

## S02 direct batched LiDAR and pending cloud construction — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The previous goal turn made progress by completing the cloud direct-output migration. This increment connects GPU-generated terrestrial/aerial LiDAR to leased outputs and downstream pending cloud jobs. The original goal remains active; camera migration and broader acceptance gates are not complete.

### Instructions

Use Makefile validation and apply_patch edits; preserve unrelated WIP, authored assets, Gearbox and user windows. No subagents, commits, viewer launches or renderer changes. Engram tool discovery still returns no memory tools; persist the project handoff here.

### Discoveries

- LiDAR already had a complete internally encoded generation/query/reduction pipeline. Direct composition requires rebinding all nine output channels together: reduction mutates the hit channels while clipping each scan to its independent range, then writes pulse selections.
- A direct LiDAR slot needs 84 bytes per maximum ray: 72 bytes of ray channels and 12 bytes of worst-case pulse outputs. Actual pulse ranges may be much shorter than ray ranges. No CPU ray arrays are needed; only 128-byte descriptors are uploaded.
- The owned-scene ray helper can trace generated leased origin/direction buffers directly. It is now crate-visible and reused by the LiDAR wrapper after device/state/capacity preflight, preserving existing primitive/mesh ordering and revision-sensitive caches.
- Nearest/farthest indices remain global ray indices, with first-nearest/last-farthest tie order. They select among divergent sub-rays, not physical multi-intersection returns. Scan IDs remain independent full-width values; identical low words do not merge scans.

### Accomplished

- Added `LidarBatch::output_pool`, `encode_scene` and `EncodedLidarBatch`. The result owns scan-range metadata and the actual scene stamp separately from the caller frame stamp. Generation, scene tracing, per-scan range clipping and pulse selection write directly into leased buffers and compose with other producers in one submission.
- Added `LidarScanBindings` and a 128-entry output-binding cache over the original descriptor/uniform/pipeline resources. Shared legacy/direct planning retains descriptor vectors and validates configured scan/ray limits without silently truncating. Legacy `dispatch`/`dispatch_scene` behavior is preserved.
- Added `PendingCloudInput::from_lidar` to select five typed pending ray channels and construct terrestrial/aerial raster metadata without host collection. Checked bounds and cloud preflight preserve per-scan source offsets, full-width IDs and sample stamps.
- Producer, scene and body-state borrowing prevents pending parameter/state/geometry mutation. Canonical state provenance, finite timestamps and metadata validate before reservation. Backpressure performs no uploads or GPU work; empty input produces fresh empty channels with no kernels. Cancellation, poison, reset/history, scene replacement and device loss follow the shared batch contract.
- `encoded_batches` counts successful encodes, including cancelled work; legacy batches/submissions retain their meaning. Descriptor byte counters include both paths. Public exports, facade type checks, plan rows and docs describe budgets, channel ordering and remaining limits.

### Verification

- `make test-sensors`: **280 passing executions**, `/tmp/molla-lidar-direct-all.log`: 159 GPU integrations, 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests, 3 CPU-adjoint integrations and 3 facade executions. All explicitly requested ignored GPU tests ran.
- Five new GPU tests (`/tmp/molla-lidar-direct-gpu5.log`) verify independent terrestrial/aerial batches plus pending cloud construction in one submission, all-nine-channel equality with legacy device batches, analytic plane/mesh hit checks, moving body poses, retained output versions, cancellation/backpressure/reset/poison and known loss, geometry replacement and empty-scene misses, 1024 independent scan descriptors, 1024 tied sub-rays and full-width scan identity. Cloud records preserve source offsets and acquisition metadata.
- The mixed fixture submits two LiDAR batches plus cloud processing once with 13 kernel dispatches, 20 output clears and exactly four 128-byte descriptors uploaded. Sensor readback counters remain zero; test collection happens explicitly afterwards.
- The 24-frame warm fixture cycles four output slots and two pose buffers with changing counts, ranges and body poses. It stabilizes at four direct LiDAR binding preparations plus the initial legacy group, eight primitive and eight mesh query preparations, and 36 fixed pool buffers. These counters do not prove whole-pipeline zero allocation or a speedup.
- **21 compile-fail doctests passed**, `/tmp/molla-lidar-direct-final-doc.log`, including three new cases for LiDAR reuse, state mutation and scene replacement while pending. The pre-existing dataset doctest remains ignored. The state test uses an actual supported indexed mutation, not an unavailable method.
- Workspace/all-target check including viewer passed (`/tmp/molla-lidar-direct-workspace-check.log`); no-default compute/kernel/sensor/facade check passed (`/tmp/molla-lidar-direct-host-check.log`); explicit-file formatting and `git diff --check` passed (`/tmp/molla-lidar-direct-fmt.log`). An initial new-test compile found a locally shadowed fixture helper; the declaration order was corrected before GPU execution. All owned processes are terminal. No new benchmark or visual acceptance evidence is claimed.

### Next Steps

Migrate `TiledCamera` and heterogeneous `CameraBatch` into direct leased outputs and pending composition, retaining aperture accumulation/resolve and center geometry semantics. Prove whole-pipeline warmed allocation and completed-work performance gates. S04 top-level acceleration/refits/stable handles, remaining S00/S01 contracts/baselines and S08 viewer/visual/performance acceptance remain. S09 Gearbox stays separately unauthorized; S10 deferred.

### Relevant Files

- `crates/molla-sensors/src/lidar_batch.rs` — direct pool/result API, retained output bindings, shared planning and lifetime checks.
- `crates/molla-kernels/src/lidar_scan.rs`, `crates/molla-kernels/src/lib.rs` — prepared generation/reduction bindings over unchanged shaders and shared uniform.
- `crates/molla-sensors/src/raycast.rs` — crate-visible owned-scene query helper used by both host rays and generated LiDAR.
- `crates/molla-sensors/src/cloud_processing.rs` — pending LiDAR-to-cloud adapter.
- `crates/molla-sensors/tests/sensor_lidar_direct_gpu.rs` — five numerical/composition/lifecycle/warm-resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — result exports and facade identity coverage.
- `book/src/sensors/lidar_batch.md`, `device_results.md`, `cloud_processing.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current API contracts, evidence and remaining full scope.

## S02 direct tiled-camera images and aperture resolve — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. The previous goal turn made progress by migrating batched LiDAR. This increment migrates homogeneous `TiledCamera` production and pending image/annotation consumers. Automatic heterogeneous `CameraBatch` manager composition is still outstanding; the original goal remains active.

### Instructions

Use Makefile validation and apply_patch edits. Preserve unrelated WIP, assets, Gearbox and user windows. No subagents, commits, renderer integration or viewer launch. Engram memory-tool discovery still returns an empty list; retain the structured project handoff here.

### Discoveries

- Camera channel buffer order is depth, color, normal, albedo, shape ID, while `CameraChannels` bit order starts with color then depth. Active-range selection must map explicit channel constants, not assume bit position equals output index.
- Direct camera images need only leased shading outputs; generated rays and hit queries remain private retained scratch. Aperture samples may overwrite that scratch repeatedly while center-sample geometry remains in output channels. Extra samples shade color only, then accumulated color resolves into the leased slot.
- Camera pose uploads currently pack a new host vector in `Buffer::upload`; retained GPU/binding counts alone are not an allocation-free pipeline. `input_upload_bytes` records the actual 32-byte GPU pose per camera, excluding uniform writes.
- Analytic center depths after f32 camera rotation/mesh traversal are not necessarily bit-exact integers: the new fixture measured 5.000000476837158 instead of 5.0. Independent geometric checks use 2e-5 absolute depth/normal tolerance; legacy/direct channel equality, IDs and miss sentinels remain exact.

### Accomplished

- Added `TiledCamera::output_pool`, `encode_scene` and `EncodedCamera`, with five typed leased channels, dimensions/count/mask and actual scene stamp. Selected channels expose all pixels; unselected channels expose empty ranges and are cleared. A NONE mask produces an empty frame without uploads/kernels. Legacy disabled-channel/stamp semantics remain unchanged.
- Shading writes directly into pool outputs. Added prepared aperture bindings over original sums/uniform/pipelines, with 128-entry shading and accumulation caches; no per-slot aperture-sum buffer is allocated. Multiple distinct tiled cameras compose with image and annotation processors without intermediate host collection or snapshot copies.
- Kept pinhole/thin-lens ray generation, sample offsets, center geometry, packed-byte aperture averaging and strict maximum range semantics. Generated query helpers now report actual encoded dispatch counts, including initial miss/primitive work and mesh query/merge pairs, without changing their algorithms.
- Producer/scene/state borrowing prevents pending mutation. Active body-bearing queries validate canonical current GPU poses before reservation. Timestamps, pool layout/provenance, backpressure, cancellation, poison, reset/history and known loss obey the shared result contract. Direct writes do not mark legacy output mirrors dirty or overwrite their last-write stamps.
- Exports/facade checks, channel/budget/statistics docs and live plan rows now distinguish tiled-camera support from the still-unmigrated heterogeneous manager. A direct slot costs 32 bytes per capacity pixel, separate from original camera storage and aperture scratch.

### Verification

- `make test-sensors`: **285 passing executions**, `/tmp/molla-camera-direct-all.log`: 164 GPU integrations, 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests, 3 CPU-adjoint integrations and 3 facade executions. Explicit ignored GPU tests ran, including the corrected fifth direct-camera fixture.
- Five new GPU fixtures passed (`/tmp/molla-camera-direct-gpu5-final.log`): independent dimensions/camera counts/optics composed with image/annotation consumers; exact all-channel parity with legacy plus center geometry parity with pinhole; all 32 masks and explicit inactive ranges; 24-frame/four-slot aperture reuse across focus/sample/range changes with retained historical images; and cancellation/invalid inputs/empty scenes/poison/loss. Moving Triangle and converted Heightfield tests retain old outputs across pose and scene changes and enforce strict distance clipping.
- The warm fixture retains 20 pool buffers, one ray preparation, four direct shading preparations, four direct accumulator preparations and two aperture scratch/uniform buffers. The mesh fixture records 25 passes for four aperture samples (ray, miss initialization, mesh query/merge, shading, accumulate per sample, then resolve). These are resource/dispatch checks, not performance or zero-allocation proof.
- **24 compile-fail doctests passed**, `/tmp/molla-camera-direct-final-doc.log`, including camera reuse, state mutation and scene replacement while pending. The pre-existing dataset doctest remains ignored.
- Workspace/all-target check including viewer passed (`/tmp/molla-camera-direct-workspace-check.log`); no-default compute/kernel/sensor/facade check passed (`/tmp/molla-camera-direct-host-check.log`); explicit-file formatting and `git diff --check` passed (`/tmp/molla-camera-direct-fmt.log`). The first test compilation corrected a misspelled existing image-config field. The initial geometric fixture failed an inappropriate exact f32-depth assertion (`/tmp/molla-camera-direct-gpu5.log`); corrected independent tolerances and exact miss/ID checks pass. No shader change or hidden test skipping was used.
- All owned processes are terminal. No benchmark, viewer launch, renderer image comparison or visual acceptance is claimed.

### Next Steps

Migrate heterogeneous `CameraBatch` into the shared command batch with explicit multi-group output ownership and whole-request validation/backpressure, preserving independent IDs/configs/layouts and reset/reconfigure behavior. Distinct `TiledCamera` instances already compose manually; do not present that as completion of the manager API. Then prove whole-pipeline warmed allocation and completed-work performance gates, S04 top-level acceleration/refits/stable handles, remaining S00/S01 contracts/baselines, and S08 viewer/visual/performance acceptance. S09 Gearbox remains separately unauthorized; S10 deferred.

### Relevant Files

- `crates/molla-sensors/src/tiled_camera.rs` — direct pool/result APIs, masks, validation, borrowing, retained shading/accumulation variants and counters.
- `crates/molla-kernels/src/camera_accumulate.rs`, `crates/molla-kernels/src/lib.rs` — prepared aperture output bindings over shared scratch and unchanged shaders.
- `crates/molla-sensors/src/raycast.rs` — generated/internal query functions return actual encoded dispatch counts.
- `crates/molla-sensors/tests/sensor_camera_direct_gpu.rs` — five composition/optics/mask/geometry/lifecycle/resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — exports and facade type checks.
- `book/src/sensors/tiled_camera.md`, `device_results.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — API contracts, verification and remaining full scope.

## S02 direct heterogeneous camera manager — verified 2026-09-21

### Goal

Continue the full `SENSORS.md` objective. Revalidated the status-only preceding turn against the current worktree and successful all-target compilation, then ran and expanded the new GPU tests. Heterogeneous manager migration is now verified; the original goal remains active because full allocation/performance, acceleration and viewer gates are not complete.

### Instructions

Use Makefile validation and apply_patch edits. Preserve unrelated WIP, assets, Gearbox and user windows. No subagents, commits, viewer launch or renderer integration. Engram tool discovery still returns no save/context/session-summary tools; this structured project handoff records the continuation.

### Discoveries

- Manager output budgeting must sum all groups, not independently permit each group the full caller budget. Direct output storage costs 32 bytes per image pixel per slot, including disabled channels. Camera scratch and legacy buffers remain separately allocated.
- A later full group must reject the whole manager before earlier groups reserve slots or upload poses. The manager also preflights the shared 64-output command limit; 64 groups leave no room for another downstream output in that command batch.
- Per-image IDs cannot be represented by one group frame stamp. Group frames use sensor ID 0, while image/annotation helpers supply each full-width sensor ID, shared sample ID, actual geometry revision and simulation time. Full scene/batch stamps remain necessary for provenance.
- An empty manager pool still needs explicit device-health ownership: zero groups must not make available/reset silently succeed after known device loss.
- Successful encodes consume sample IDs even if cancelled; rejected preflight/backpressure does not. Layout-identical reconfiguration can reuse an existing manager-owned pool, whereas changed IDs/order/masks/dimensions/group mapping cannot.

### Accomplished

- Added `CameraBatchOutputPool`, `EncodedCameraBatch`, `CameraBatch::output_pool` and `encode_scene`. All groups append direct leased camera outputs to one command batch, with whole-request validation/backpressure and poisoning after an encoding failure. Earlier unrelated commands remain valid when manager preflight rejects a request.
- Added per-image lookup, readback metadata, pending color-processing and annotation job helpers. Nonadjacent grouped images keep correct slices and independent IDs/settings; helpers reject unknown IDs and disabled channels. Legacy group submission APIs remain unchanged.
- Added health/layout/availability/output-capacity preflight helpers to the shared output machinery and factored tiled-camera scene validation for reuse. Pending manager/scene/state borrows prevent mutation until submit/drop. Submitted images outlive managers/pools and remain readable through reset/reconfiguration while healthy.
- Exposed manager output types through the sensors crate and public Molla facade. Added aggregate pool/group statistics and separate successful-encode accounting without treating it as completion or altering legacy submission counters.
- Updated camera/device-result documentation and the live plan. Only the composable producer-API checklist item is now marked complete; full S02 resource/lifecycle/performance gates remain open.
- Added completed-work camera-manager benchmarks for 1/4/16 cameras, homogeneous and four-configuration heterogeneous groups. Paired legacy/direct cases use the same owned scene/poses and include completion waits every iteration; direct slots are warmed first. Mixed configurations include four resolutions and one four-sample aperture group. No runtime integration outside sensors was changed.

### Verification

- `make test-sensors`: **290 passing executions**, `/tmp/molla-camera-manager-all-final.log`: 169 GPU integrations, 9 GPU unit tests, 93 GPU-feature sensor host tests, 3 compute host tests, 10 no-default sensor tests, 3 CPU-adjoint integrations and 3 facade executions. Explicit ignored GPU lanes ran. This is a sum of executions across feature lanes, not a unique-test count.
- Five new GPU fixtures passed, `/tmp/molla-camera-manager-gpu5-final.log`: heterogeneous cameras plus image/annotation consumers in one submission with exact all-channel legacy parity; later-group saturation/cancellation/reconfiguration/poison; 1,024 independent camera IDs and 64 groups; aggregate budgets/foreign state/pools/empty managers/loss; and warmed historical images under changing poses. The existing GPU unit fixture additionally tests direct sample-ID exhaustion before reservation.
- The warm manager fixture submits 24 frames through four slots per group. Three groups retain 60 pool buffers, one ray preparation per group, four shade preparations per group, and four aperture preparations plus two aperture buffers for the aperture group. Each frame records 19 kernel dispatches, 15 output clears, 128 pose-upload bytes and one submission. Independent sphere/plane center distances use 2e-5 tolerance and exact IDs. Saturation after reset does not change group encoding counters. These are bounded-resource/correctness checks, not proof of zero host allocation or a speedup.
- **27 compile-fail doctests passed**, `/tmp/molla-camera-manager-doc.log`, including manager reset, body-state mutation and scene replacement while pending. The pre-existing dataset doctest remains ignored.
- Workspace/all-target check including viewer passed (`/tmp/molla-camera-manager-workspace-check-final.log`); no-default compute/kernel/sensor/facade check passed (`/tmp/molla-camera-manager-host-check-final.log`); updated benchmark/all-target check passed (`/tmp/molla-camera-manager-bench-check.log`); explicit-file formatting and `git diff --check` passed (`/tmp/molla-camera-manager-fmt-final.log`).
- `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera -- --test'` passed all **18 smoke cases** on NVIDIA RTX 4080/Vulkan, `/tmp/molla-camera-manager-bench-smoke.log`: six existing camera cases and twelve new manager comparisons. Smoke execution is not an accepted timing result.
- Two local test-compilation mistakes were corrected before final validation: the new warm fixture used `Transform.p` instead of `position`; the internal exhaustion fixture assumed a nonexistent `SensorFramePoolConfig::default`, replaced with an explicit bounded configuration. Failed logs remain `/tmp/molla-camera-manager-gpu5.log` and `/tmp/molla-camera-manager-all.log`; the corrected full lane passes.
- All owned test/check/benchmark processes are terminal. No viewer launch, visual acceptance, allocation-free result, renderer matching or production performance claim is made.

### Next Steps

All planned core sensor/processing producers now have direct leased-output composition, including heterogeneous camera management. Continue the full S02 lifecycle audit and warmed whole-pipeline allocation/completed-work performance requirements; camera pose packing, command/result metadata and driver allocations are still present. Complete S04 top-level acceleration/refits/stable application handles, remaining S00/S01 contracts/capacity/baseline coverage and S08 example migration/visual/performance acceptance. S09 Gearbox remains separately unauthorized; S10 deferred. Do not mark the full goal complete from producer API or regression completion alone.

### Relevant Files

- `crates/molla-sensors/src/camera_batch.rs` — direct heterogeneous manager, pooled outputs, per-image pending adapters and lifetime/identity checks.
- `crates/molla-sensors/src/device_results.rs`, `tiled_camera.rs` — shared capacity/layout/availability preflight and camera validation.
- `crates/molla-sensors/tests/sensor_camera_manager_direct_gpu.rs` — five GPU composition, identity, lifecycle, budget and warm-resource fixtures.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports/facade checks.
- `crates/molla-sensors/benches/bench_sensor_camera.rs` — paired completed-work manager benchmarks and retained existing camera cases.
- `book/src/sensors/tiled_camera.md`, `device_results.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — API semantics, benchmark guidance, current evidence and remaining full scope.

## S02 warmed allocation audit and retained metadata — verified 2026-09-21

### Goal

Continue the complete `SENSORS.md` objective after the verified heterogeneous-manager increment. The preceding goal turn made progress. This increment measures allocations instead of inferring them from stable GPU buffer counters, then removes identified temporary storage. The goal remains active; this is not completion of the zero-allocation or performance gates.

### Instructions

Use Makefile tasks and apply_patch; preserve all unrelated WIP, user windows, Gearbox and assets. No subagents, commits or viewer launch. Engram save/context/session-summary tool discovery remains empty; keep the project handoff current. Memory-folder guidance about GPU-first reusable libraries and the separate viewer remains consistent with the current plan.

### Discoveries

- Stable pool/binding counters hid substantial calling-thread Rust allocation. In a warmed four-camera → image → annotation chain, submission alone made 98 allocations and 16 reallocations before these changes. Counts include wgpu Rust bookkeeping, not native driver allocations or other threads.
- Each `SensorCommandBatch::new` allocated four capacity-64 vectors. Submission allocated another occupancy-token vector, in addition to wgpu's callback allocation. Inline collections eliminate those vectors but increase the fixed object/callback payload; fewer allocations must not be reported as a measured latency improvement.
- Heterogeneous camera encodes allocated a group vector, copied image-layout storage and repacked poses into a fresh vector. Layouts are immutable per configuration and can be shared across old/new results. Packed pose capacity is fixed at camera construction.
- Global allocator instrumentation needs thread-local, nonallocating counters and unwind-safe reset. Requested bytes are cumulative requests, including full reallocation sizes, not peak or retained bytes. The report deliberately does not assert the still-unmet whole-pipeline zero-allocation gate.

### Accomplished

- Added `sensor_allocation_gpu.rs` with a scoped calling-thread allocator audit. It records creation, camera/image/annotation encoding, submission, completion and release over eight warm-up plus sixteen measured completed frames. The fixed fixture uses four homogeneous 9×7 cameras, one sphere, identity image processing, one annotation label and one output slot per producer; no image readback occurs in the measured path.
- Changed private command/submission frame, input, source-lease and temporal-commit lists to capacity-64 `ArrayVec`; completion tokens use capacity 128 for 64 outputs plus 64 inputs. Existing bounds/poison/drop/IMU rollback behavior is retained. Added optional GPU-only `arrayvec` dependency using the already locked 0.7.6 release; no version bump or unrelated dependency change.
- Added `SensorSubmission::into_frames_iter()` for allocation-free ownership transfer in encoding order. Legacy `into_frames()` still returns a `Vec` and now explicitly allocates on that convenience conversion rather than inheriting batch vector storage.
- `EncodedCameraBatch.groups` now stores its bounded receipts inline; `images` uses `Arc<[CameraImageLayout]>` shared with manager/pool configuration. Existing legacy camera views/convenience signatures remain unchanged. Historical layouts survive replacement; per-image IDs/ranges and pool compatibility are unchanged.
- Added `Buffer::upload_with_scratch`, preserving the original device-dirty authority assertion and conversion/synchronization behavior. Legacy `upload` delegates with temporary scratch. Both direct and legacy camera paths retain packed pose scratch initialized once at camera construction. Queue staging still allocates separately; no broad storage caching or physics behavior was changed.
- Added explicit tests for all 64 outputs plus all 64 retained inputs, completion release after dropping source frames, zero-allocation result iteration, scalar/vector/transform upload conversion, retained scratch capacity and missing/empty mirrors. Debug upload-authority panic checks are caught intentionally; the existing assertion was not removed or weakened.
- `make test-sensors` now also runs the allocator counter's host self-test; its explicit GPU lane runs the three new GPU tests. Documentation/live plan records measurement scope, storage/callback tradeoffs and unmet gates.

### Verification

- Baseline `/tmp/molla-allocation-before.log`: 153 allocations, 21 reallocations, 42,445 requested bytes per warmed frame. Final debug `/tmp/molla-allocation-checks-final.log`: **145 allocations, 21 reallocations, 40,485 requested bytes**. Every phase had identical min/max across sixteen measured samples. Eight allocation calls and 1,960 requested bytes were removed; this is not a timing or peak-memory claim.
- Phase allocation counts before → after: creation 9 → 5; camera 26 → 23; image 2 → 2; annotation 18 → 18; submission 98 → 97; completion/release 0 → 0. Submission requested bytes increased from 16,345 to 18,337 because of the larger inline callback payload, while total requested bytes decreased. Intermediate attribution logs: `/tmp/molla-allocation-inline.log`, `-layouts.log`, `-packed.log`.
- Three GPU allocation/lifecycle/upload fixtures passed in debug and release (`/tmp/molla-allocation-checks-final.log`, `/tmp/molla-allocation-release.log`). The final counts reproduced exactly in release; no pre-change release allocation measurement was taken. Environment: NVIDIA RTX 4080, Vulkan, driver 595.84; rustc 1.96.1 (31fca3adb 2026-06-26); HEAD d2369a9 plus the existing/current worktree.
- `make test-sensors`: **294 passing executions**, `/tmp/molla-allocation-all.log`: previous 290 plus the counter host self-test and three explicit GPU tests. Numerical camera/processing/IMU/ray results, maximum group counts, pending lifetimes and lifecycle regressions all remain green.
- **27 compile-fail doctests passed**, `/tmp/molla-allocation-doc.log`; the pre-existing dataset doctest remains ignored. Storage host tests passed both no-default and GPU-feature configurations, three each (`/tmp/molla-allocation-storage-host.log`, `-storage-feature.log`).
- Workspace/all-target check including viewer passed (`/tmp/molla-allocation-workspace.log`); no-default compute/kernel/sensor/facade check passed (`/tmp/molla-allocation-host-check.log`); explicit formatting and `git diff --check` passed (`/tmp/molla-allocation-fmt.log`). All 18 camera benchmark smoke cases passed again after the storage/API changes (`/tmp/molla-allocation-bench-smoke.log`); no measured speedup is claimed.
- The first full-capacity test compile rejected dropping a reference vector and its borrowed owner together in one tuple. Separate ordered drops corrected the fixture; final debug/release suites pass. The expected caught authority-check panic text in `--nocapture` logs is successful negative coverage, not a hidden failure.
- All checks/tests/benchmarks started by this increment are terminal. An independently running `cargo check --all-targets` was observed at final process inspection and left untouched. No viewer/visual acceptance or Gearbox integration was performed.

### Next Steps

Continue full S02 allocation/resource/lifecycle and measured completed-work performance acceptance. The audit still reports 145 allocations and 21 reallocations per warmed fixture frame; producer processing metadata, batch owner identity, wgpu encoder/submission bookkeeping and staging remain. Investigate measured dominant phases rather than claiming the allocation gate from fixed buffers or this reduction. Complete S04 top-level acceleration/refits/stable application handles, remaining S00/S01 contract/capacity/baseline coverage and S08 example migration/visual/performance gates. S09 Gearbox remains separately unauthorized; S10 deferred. The full goal must not be marked complete.

### Relevant Files

- `crates/molla-sensors/tests/sensor_allocation_gpu.rs` — calling-thread allocation audit/self-test and full-capacity/upload GPU regressions.
- `crates/molla-sensors/src/device_results.rs` — inline bounded metadata/completion tokens and owned frame iterator.
- `crates/molla-sensors/src/camera_batch.rs` — shared immutable layouts and inline group receipts.
- `crates/molla-sensors/src/tiled_camera.rs` — retained packed camera pose storage.
- `crates/molla-storage/src/buffer.rs` — optional caller-retained upload scratch with unchanged authority checks.
- `crates/molla-sensors/Cargo.toml`, `Cargo.lock`, `Makefile` — GPU-only arrayvec dependency and allocation self-test lane; previous unrelated edits preserved.
- `book/src/sensors/device_results.md`, `tiled_camera.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — measured allocation scope, tradeoffs, evidence and remaining full objective.
