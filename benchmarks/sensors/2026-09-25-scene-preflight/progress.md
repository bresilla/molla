# Sensor implementation progress

## Goal and scope

Implement `SENSORS.md` core phases S00–S08 and the separately authorized S09 integration. S10 remains optional/deferred by the plan. This is not a completion report.

Started 2026-09-21 from `d2369a9` plus pre-existing working-tree changes. `SENSROS.md` contains a user-requested full snapshot; `SENSORS.md` remains the execution plan. Preserve unrelated viewer, tyre, and FULL_COMPACT work. Gearbox was untouched during the initial core work; the later authorized S09 changes are recorded below. Assets remain untouched.

## Current phase status

| Phase | Status | Evidence / remaining work |
|---|---|---|
| S00 | In progress | Existing host/GPU and completed-camera baselines captured; independent triangle oracle and mount tests pass. Full performance budgets, cold-build measurements and broader capability contract remain. |
| S01 | In progress | IMU device history, metric ray inputs, cone hits, checked capacities and cache revision invalidation implemented and tested. Broader lifecycle/geometry checks remain. |
| S02 | In progress | Grouped readback, retained bindings, snapshots and device-loss boundaries are verified. IMU/contact/frame, owned-scene host rays, GPU-generated batched LiDAR, tiled/heterogeneous cameras, images, annotations, cloud construction/filtering and cloud grids compose into direct leased slots; IMU history commits at submission and rolls back on cancellation. Submitted-frame/device-source consumers retain leases; scoped typed pending inputs support custom and built-in image/annotation/cloud/grid chains, including LiDAR-to-cloud, camera-to-processing and 1024 logical annotation/cloud jobs through deduplicated dependencies. Full allocation/lifecycle/performance acceptance remains. |
| S03 | Core verified | GPU primitive/static-mesh merging, mixed terrestrial/aerial ray layouts, independent scan descriptors and first/last sub-ray reduction implemented. Expanded pipeline verification and performance evidence recorded below; shared-query resource/lifecycle/performance gates remain. |
| S04 | In progress | Owned revisioned geometry, stable handles, moving instances and terrain conversion are implemented. Shared threaded TLAS/BLAS traversal, GPU pose refits, periodic and quality-triggered rebuilds are verified by focused regressions. Full numerical/resource/performance acceptance remains. |
| S05 | Core verified | Independent simulation-time rates, pause/reset/replacement contracts, device-only auxiliary outputs and grouped wrappers pass. Caller snapshot ownership and explicit sensor reset/recreation are documented; S02 resource and S08 viewer gates remain. |
| S06 | Core verified | Shared visibility, GPU aperture resolve, center geometry, channel revisions, heterogeneous groups and device-only camera/image chains pass. Shared S02/S04 and S08 gates remain. |
| S07 | Core verified | GPU images/clouds/spatial reductions, annotations and selected CPU export pass, including full-record packed-cloud round trips. Shared S02/S04 lifecycle and S08 full performance/visual gates remain. |
| S08 | Not complete | Seven asynchronous viewer migrations and scene 144 have headless coverage. Full visual acceptance and reproducible performance acceptance remain. |
| S09 | Integrated, verified 2026-09-25 | Authorized by the user. Gearbox samples `role = "sensor"` links with Molla IMU/LiDAR on Bevy's render device and streams them per link; headless GPU fixture and live tractor drive, pause/play and despawn verified. Transfer-overhead measurement and camera links remain. |
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

## S01 ray storage budgets and cloud numeric regression — verified 2026-09-24

### Goal

Continue the complete `SENSORS.md` objective. The preceding status-only turn made no implementation progress; this continuation revalidated its pending changes, completed full regression/release verification, and reconciled the S01 regression checklist against actual fixtures. The full goal remains active.

### Instructions

Use Makefile tasks and patch edits. Preserve Gearbox, assets, physics, unrelated WIP and user windows. No subagents, commits or viewer launch. Engram context/save/session-summary tools remain unavailable; this project handoff records the work instead. Historical allocation counts are not current performance evidence after dependency changes.

### Discoveries

- Resumed checkout HEAD is `367d92c`. Earlier sensor work is committed; physics/sensor GPU dependencies now use wgpu 29.0.4. The workspace lockfile also retains wgpu 27 for other consumers. The new wgpu binding-size limit is u64, not u32. No dependencies were changed in this increment.
- Per-buffer validation does not bound the combined payload of ten raycaster buffers, their host mirrors and retained upload vectors. Fixed ray storage can be estimated before engine creation or allocation, but transient staging, geometry, outputs and camera/LiDAR-specific arrays require separate accounting.
- The existing cloud transform-overflow regression failed on this checkout: a position at f32::MAX translated by f32::MAX was accepted. Post-arithmetic nonfinite checks were insufficient for this GPU execution. This reproduces on the upgraded checkout; it does not establish the dependency upgrade as the cause.
- Scaling every rotation changed ordinary-path rounding. Dividing by a magnitude near f32::MAX was also unsuitable for the extreme fixtures. The final implementation preserves ordinary rotation arithmetic and scales extreme vectors by exact powers of two. WGSL hex float constants work with Naga 29; bitcast constant expressions were rejected by its shader parser.

### Accomplished

- Added public `RayStorageUsage::for_capacity` and `RaycasterLimits`, exported through the Molla facade. Raycaster construction validates device binding/buffer/dispatch limits and a default 1 GiB combined host/device fixed-array payload budget before resource allocation. Explicit budget overrides are available in raycaster, legacy terrestrial/aerial LiDAR, batched LiDAR and tiled-camera constructors. Existing zero-capacity raycaster/legacy-LiDAR behavior still clamps to one slot.
- The estimate counts six ray/hit channels, three mesh scratch channels, their host mirrors, two retained packed upload vectors and one fallback body pose. GPU payload is `96 * capacity + 32`; host bytes use actual Rust element sizes. Mesh caches, pipelines/bindings, transient staging/packing, readback, output pools, camera images/poses/aperture buffers and legacy result/layout vectors are explicitly excluded. This is not a whole-sensor or process peak-memory cap.
- Added three pure sizing/boundary tests, one ignored GPU payload-accounting unit test, and one GPU constructor/legacy-query integration fixture. Coverage includes exact versus one-byte-under budgets, device limits, extreme capacities without giant allocations, representative scan/image capacities, actual GPU buffer sizes, one-slot compatibility and independent plane-hit distances.
- Cloud transforms now guard addition before overflow and scale large rotations before restoring finite results. Large intermediate rotated positions can be cancelled by translation if the final position is representable. Normal/direction overflow counts as invalid. Distance subtraction overflow counts as filtered, not invalid input. Ordinary/identity paths preserve their prior arithmetic and record behavior.
- Added an extreme-finite cloud fixture covering signed overflow/cancellation, identity and rotated f32::MAX vectors, translation cancellation of a large rotated position, finite-radius inclusion/exclusion and normal/direction overflow. Existing cloud tests were retained unchanged.
- Updated raycaster, LiDAR, camera and cloud documentation. Marked only the named S01 regression-coverage item complete after inspecting its fixtures and rerunning all sensor lanes; aggregate/transient constructor auditing and the broader S01 gate remain open.

### Verification

Commands used the repository Makefile via `NVIDIA_VERSION=595.84 nix develop --impure -c make ...`. Installed driver version was rechecked in `/proc/driver/nvidia/version`; benchmark adapter reports NVIDIA RTX 4080/Vulkan. Host plain Cargo selected an obsolete compiler in an earlier attempt; repository Nix validation was used instead. No environment files were modified.

- `make test-sensors`: **300 passing executions**, `/tmp/molla-ray-budget-regression.log`. This sums feature-lane executions, not unique tests: previous 294 plus three host budget tests, one GPU budget-accounting unit test, one capacity GPU integration and one cloud GPU integration. Explicit ignored GPU lanes ran, including downstream cloud consumers/export fixtures.
- `make test-gpu P=molla-sensors T=sensor_cloud_gpu RELEASE=1`: **8/8 passed**, `/tmp/molla-ray-budget-cloud-release.log`. Includes the pre-existing failure and new extreme/cancellation/distance cases under optimized compilation.
- `make test APP_TARGET='-p molla-sensors --features gpu --doc'`: **27 compile-fail doctests passed**, `/tmp/molla-ray-budget-doc-final.log`; the pre-existing dataset doctest remains ignored.
- Workspace/all-target check including viewer passed, `/tmp/molla-ray-budget-workspace-final.log`; unrelated existing solver test dead-code warnings remain. No-default compute/kernel/sensor/facade check passed, `/tmp/molla-ray-budget-host-check-final.log`.
- Explicit-file `make fmt-files FMT_ARGS=--check` passed, `/tmp/molla-ray-budget-fmt.log`. `git diff --check` passed.
- Camera/cloud benchmark smoke passed **45 cases** (18 camera, 27 cloud), `/tmp/molla-ray-budget-bench-smoke.log`, using `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera --bench bench_sensor_cloud -- --test'`. This is execution coverage, not a measured speedup or completed performance gate.
- Earlier failed logs are retained: `/tmp/molla-ray-budget-all.log` exposed the transform-overflow regression; `/tmp/molla-ray-budget-all-final.log` stopped at the intermediate WGSL constant-parser error despite its name. Neither is final evidence. The successful full rerun is `molla-ray-budget-regression.log`.
- All owned validation processes are terminal. No GUI/visual acceptance, Gearbox integration, current allocation-count comparison or runtime speed claim is made.

### Next Steps

Continue the real S00–S08 scope: finish whole-sensor/transient capacity accounting and contract/tolerance inventory; S02 warmed allocation/lifecycle and completed-work performance gates; S04 top-level acceleration, refit/rebuild rules and stable handles; S08 migration of viewer examples 05/06/07/12/105/106/108, multi-sensor moving-mesh demonstration and visual/performance acceptance. Existing S03/S05/S06/S07 core APIs do not close those shared gates. Gearbox S09 remains separately unauthorized; S10 advanced physical sensing stays deferred. Do not mark the full goal complete from this regression increment.

### Relevant Files

- `crates/molla-sensors/src/raycast.rs` — pure storage accounting, default/override budget validation, host and GPU accounting tests.
- `crates/molla-sensors/src/lidar.rs`, `lidar_batch.rs`, `tiled_camera.rs` — shared ray-budget constructor overrides.
- `crates/molla-sensors/tests/sensor_capacity_gpu.rs` — budget propagation, boundary rejection and valid legacy queries.
- `crates/molla-kernels/shaders/cloud_process.wgsl` — checked transforms and overflow-safe distance filtering.
- `crates/molla-sensors/tests/sensor_cloud_gpu.rs` — extreme-finite transform/filter regression.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public type exports and facade coverage.
- `book/src/sensors/{raycaster,lidar_batch,tiled_camera,cloud_processing}.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current semantics, completed evidence and remaining scope.

## S04 stable identities across scene replacement — verified 2026-09-24

### Goal

Continue the full `SENSORS.md` objective after the previous turn's verified ray-budget/cloud-fix increment. That turn made progress. This increment implements the missing stable body/shape identity portion of S04; top-level acceleration/refits and the full objective remain incomplete. HEAD remains `367d92c`; the preceding uncommitted ray-budget/cloud changes were preserved.

### Instructions

Use Makefile tasks and patch edits. Do not edit Gearbox, assets, physics or unrelated WIP, launch/close windows, spawn agents or commit. Engram context/save/session-summary tools remain unavailable; this structured project handoff records the session. No new memory-folder files were written.

### Discoveries

- Dense shape/body indices cannot establish object continuity between models: geometry, names, attachment and ordering can all change. Preservation needs an explicit mapping from each replacement slot to a currently live opaque handle, not an index/geometry heuristic.
- A revision stamp prevents interpreting old indices against new geometry, but does not by itself retain historical object identity. Immutable CPU identity snapshots can outlive geometry, GPU resources and the scene without retaining those resources.
- A shape's stable identity and body attachment are independent. Explicitly preserved shapes may be reparented; old snapshots must retain the old attachment. Four-byte dense owner indices inside each immutable table avoid duplicating full optional body handles per shape.
- Readback slot capacity rounds to eight bytes. The initial new three-i32 fixture supplied a 12-byte total budget and correctly failed construction; it now budgets 16 bytes. Readback validation was not weakened.

### Accomplished

- Added opaque, typed `SensorBodyHandle` and `SensorShapeHandle`, plus `SceneIdentitySnapshot` and `SensorSceneRemap`, exported through `scene`, the sensors crate and Molla facade. Scene-scoped monotonically allocated published identities never recycle. Exhaustion returns an error rather than wrapping.
- `SensorScene::identities()` cheaply clones an immutable Arc-backed CPU table. Stamped dense-index lookups check exact scene/revision and bounds; inverse handle lookups resolve indices in that particular snapshot. Static/world body ownership is `None`. Wrong-scene/retired handles cannot alias current objects, and historical snapshots still identify deleted objects.
- Added transactional `replace_model_with_remap`: slices follow the replacement model's complete dense body/shape ordering; `Some` preserves a currently live identity, `None` creates one, omission retires an old identity. Duplicate/foreign/retired handles or wrong lengths reject before GPU geometry compilation. Geometry failures leave current geometry, identities, revision and published identity sequence unchanged.
- Plain `replace_model` retains its geometry/revision behavior but intentionally assigns all-new handles rather than guessing identity continuity. Replacements preserve the scene namespace without consuming a new scene ID. Scene body count now shares shape-count GPU-capacity preflight before identity-table allocation.
- Query/GPU channel formats remain unchanged: outputs still contain dense indices. Callers retain the matching identity snapshot and actual SceneStamp beside delayed results; tables are not implicitly attached to every frame. Applications own the association to persistent machine/asset IDs and must supply body state in the replacement model's ordering.
- Added four pure host identity tests and two GPU scene fixtures: body/shape reordering, explicit replacement/deletion with reused slots, signed owner validation, empty replacement, static ownership, reparented historical ownership, counter exhaustion, stale/foreign/duplicate inputs, failed geometry transactions and old readback identity after replacement/drop. The public facade fixture covers all four new types.
- Added a calling-thread allocator fixture: 1,024 iterations clone current and historical snapshots and perform successful forward/inverse/owner lookups with zero allocations/reallocations. Scene construction/replacement and error-string creation are outside this measurement. This is not the whole-pipeline allocation or performance gate.
- Updated scene/device-result documentation and marked only the S04 moving-mesh/stable-handle checklist item complete. The S04 status row explicitly keeps acceleration/refits pending.

### Verification

Commands used `NVIDIA_VERSION=595.84 nix develop --impure -c make ...`, the repository GPU wrapper and the same RTX 4080/Vulkan environment. No dependencies or environment files changed.

- Final `make test-sensors`: **307 passing executions**, `/tmp/molla-scene-handles-verified.log`. This adds four host identity tests, two GPU scene fixtures and one GPU allocation fixture to the prior 300. The count sums feature-lane executions; explicit ignored GPU lanes ran.
- `make test-gpu P=molla-sensors T=sensor_scene_gpu RELEASE=1`: **6/6 passed**, `/tmp/molla-scene-handles-release-final.log`. Existing moving-instance/terrain/cache regressions and new identity/readback/transaction fixtures pass together.
- New identity allocation fixture passed in debug and release; final release evidence `/tmp/molla-scene-handles-allocation-release-final.log`. The final full suite includes its debug run. No calling-thread allocations/reallocations were observed in successful snapshot cloning/lookup; native driver activity and full camera/sensor submission are not measured by that claim.
- Workspace/all-target check including viewer passed, `/tmp/molla-scene-handles-check-final.log`; existing unrelated solver test dead-code warnings remain. No-default compute/kernel/sensor/facade check passed, `/tmp/molla-scene-handles-host-check.log`, and the final full sensor suite repeated no-default/facade test lanes.
- **27 compile-fail doctests passed**, `/tmp/molla-scene-handles-doc-final.log`; the existing dataset doctest remains ignored. Explicit-file formatting passed, `/tmp/molla-scene-handles-fmt-verified.log`; `git diff --check` passed.
- Initial `/tmp/molla-scene-handles-gpu.log` records the corrected 12-byte readback-fixture budget error. The final suite uses the compact owner-index table and the allocation fixture; intermediate `-all.log`/`-all-final.log` runs are not the final source evidence.
- All owned validation processes are terminal. No viewer visual acceptance, new timing benchmark, zero-allocation whole-pipeline result or Gearbox integration is claimed.

### Next Steps

Continue S04 top-level acceleration over finite primitives/mesh instances, separate infinite planes, and explicit refit/rebuild/revision rules with many-instance completed-work benchmarks. Stable handles are now available for that work; do not replace them with dense-index identity heuristics. Remaining S00/S01 contract/tolerance/whole-sensor capacity audits, S02 whole-pipeline allocation/lifecycle/performance gates, and S08 viewer migration/multi-sensor demo/visual acceptance remain required. S09 Gearbox is still separately unauthorized and S10 deferred. The full goal remains active.

### Relevant Files

- `crates/molla-sensors/src/scene_identity.rs` — opaque handles, immutable bidirectional identity/owner tables, remap validation and four host tests.
- `crates/molla-sensors/src/scene.rs` — snapshot access, transactional remapped compilation/replacement and body capacity preflight.
- `crates/molla-sensors/tests/sensor_scene_gpu.rs` — two GPU remap/readback/deletion/transaction regressions.
- `crates/molla-sensors/tests/sensor_allocation_gpu.rs` — current/historical identity clone/lookup allocation fixture.
- `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — public exports and facade checks.
- `book/src/sensors/scene.md`, `device_results.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — identity contracts, usage, verified scope and remaining requirements.

## S04 instance hierarchy and GPU refit foundation — verified 2026-09-25

### Goal

Continue the complete `SENSORS.md` objective after the previous turn's stable-handle implementation, which made progress. Implement the hierarchy/bounds/refit core needed for accelerated scene queries. This is an intermediate foundation: existing ray/LiDAR/camera paths still use their previous primitive/per-instance traversal and have not been accelerated by this increment. The full goal remains active.

### Instructions

Use Makefile tasks and patch edits; preserve earlier uncommitted work, Gearbox, assets and physics. No agents, commits or GUI actions. Engram context/save/session-summary tools remain unavailable; this project handoff records the work. No memory-folder updates were made.

### Discoveries

- Refit levels need dependency-ordered compute passes. Mutable queue-written level uniforms would expose later values to earlier encoded dispatches; immutable per-level uniforms avoid that hazard without CPU waits or inter-workgroup synchronization.
- World boxes need conservative f32 rounding margins, not merely nearest-rounded min/max values. Invalid live poses must produce explicit status instead of silently excluding geometry. The core writes full finite-f32 bounds for invalid/unsafe leaves and keeps planes separate.
- A hierarchy's internal indices depend on its partition seeds, even when the scene revision is unchanged. The borrowed view includes its topology; callers retaining debug bounds across reconstruction must retain that topology too. Stable object identity remains the scene's shape/body handles, not internal node indices.
- Construction benchmarks must explicitly submit pending queue writes before waiting. Device polling alone does not flush initialization uploads. The constructor benchmark submits an empty command sequence before its completion wait.
- The current Nix shell reports Rust **1.94.1 (e408947bf 2026-03-25)**, not the historical 1.96.1 label in earlier sections. Current component measurements use the observed version. HEAD remains `367d92c`; driver is 595.84, CPU i7-13700KF, GPU RTX 4080/Vulkan.
- Formatting a Rust crate root follows child modules and can reorder existing imports/modules. The initial explicit-file formatter also changed four unrelated kernel files; their formatting-only changes were reverted with patches and verified to have zero diff. Final checks use `skip_children=true` and exclude the kernel crate root with its existing unsorted declarations; its only retained diff is the four module/export lines, reviewed directly.

### Accomplished

- Added `SceneAcceleration` and its config/storage/view/stats/node types, exported through the sensors crate and Molla facade. Construction builds deterministic balanced median-split topology over supported finite primitives and compiled mesh instances. Planes are separate; skipped unsupported geometry and empty meshes have no finite leaves. Each unique mesh's local bounds are computed once per construction.
- Optional caller-provided CPU body poses seed spatial partitioning only. Without them, shape-local centers seed the tree. Runtime refits always read authoritative live GPU body poses, never host pose arrays or an implicit download. Large motion does not trigger automatic topology rebuilding yet.
- Added `SceneBoundsKernel` and WGSL leaf/parent passes. Leaves compose rigid instance poses, generate outward-rounded padded bounds/world poses and status; parents union children in contiguous increasing-height ranges. Fixed buffers and immutable level uniforms are reused. Two body-pose binding variants support ping-pong state. One explicit dispatch call submits once, with no collection/wait in that path.
- Status 0 means valid; 1 marks invalid/nonfinite pose data; 2 marks conservative unsafe magnitude above 2^120, not a proof of actual arithmetic overflow. Invalid/unsafe outputs use full finite-f32 boxes and an identity pose. Consumers must inspect status. Valid subsequent refits clear old flags. Bounds padding uses 32 f32 epsilons times a magnitude bound plus one.
- Added pure GPU-payload sizing with a default one-million-leaf/256 MiB cap and independent binding/buffer/dispatch checks before GPU allocation. The estimate includes instances, topology, bounds, poses, flags, fallback pose and level uniforms, including empty dummy buffers. Host/staging/pipeline/readback/model storage is excluded. A GPU unit fixture compares allocated buffer sizes plus fixed uniform payload against the estimate.
- Scene ID/revision changes reject refitting and require constructing a new hierarchy; pose-only changes reuse topology and existing mesh geometry. Foreign devices/scenes, body-count mismatch and unuploaded/host-dirty pose data reject before encoding. Empty/plane-only cases have zero active nodes and no compute dispatches.
- Added three host tests for tree coverage/dependency order, budget boundaries and primitive/mesh local bounds; three GPU integrations cover 129 mixed instances, eight changing-pose frames across two device-authoritative buffers, independent f64 corner enclosure, exact parent unions, invalid-pose recovery, stale scenes and empty/plane-only budgets.
- Added `bench_scene_refit` with completed construction and warmed completed-refit cases at 1,000 and 10,000 finite instances, using spheres, one repeatedly instanced mesh and an infinite plane. Updated scene documentation and the existing S04 unchecked acceleration item; it remains unchecked because query traversal/unified mesh dispatch/quality policy are not integrated.

### Verification

- Final `make test-sensors`: **314 passing executions**, `/tmp/molla-scene-refit-all-final.log`. This is the previous 307 plus three host tests, one ignored GPU buffer-accounting unit test and three GPU refit integrations. Counts sum feature-lane executions; explicit ignored GPU lanes ran.
- `make test-gpu P=molla-sensors T=sensor_scene_refit_gpu RELEASE=1`: **3/3 passed**, `/tmp/molla-scene-refit-release-final.log`. Debug tests also pass in the full lane.
- Workspace/all-target check including the new benchmark and viewer passed, `/tmp/molla-scene-refit-workspace.log`. Existing unrelated solver test dead-code warnings remain. No-default compute/kernel/sensor/facade check passed, `/tmp/molla-scene-refit-host-check.log`.
- **27 compile-fail doctests passed**, `/tmp/molla-scene-refit-doc.log`; the existing dataset doctest stays ignored. Scoped formatting passed, `/tmp/molla-scene-refit-fmt-verified.log`; `git diff --check` passed. The earlier `-fmt-scoped.log`/`-fmt-scoped-final.log` checks failed on existing crate-root ordering and alternate import wrapping; neither is final formatting evidence. Whole-workspace formatting is not claimed.
- Actual release benchmark command: `NVIDIA_VERSION=595.84 nix develop --impure -c make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_scene_refit'`, `/tmp/molla-scene-refit-bench.log`. Criterion: ten samples, one-second warmup, three-second measurement target per case; no other owned GPU test ran concurrently. Construction excludes model/scene/BLAS creation and includes queue-flushed initialization plus completion. Refit includes completion waits but no pose upload, traversal or readback.
- 1,000 instances: construction **956.57 µs [949.62, 969.21]**, warm refit **127.70 µs [108.57, 161.96]**. 10,000: construction **5.4830 ms [5.4219, 5.5151]**, warm refit **133.07 µs [130.95, 137.80]**. Brackets are Criterion's reported intervals. The 1,000-construction case had one outlier in ten measurements; no precision/scaling claim is made from the wide small-refit interval.
- These are component timings, not a ray-query speedup, cold driver compilation, zero-allocation whole-pipeline result or S04/S08 acceptance. All owned validation/benchmark processes are terminal; no viewer or Gearbox work occurred.

### Next Steps

Wire shared ray/LiDAR/camera visibility to actual top-level traversal over this refitted hierarchy, with unified mesh/BLAS dispatch and separate infinite-plane handling. Preserve existing metric distances, primitive-before-mesh/equal-distance tie rules, scene/result identities and direct leased-output composition. Add full build/refit/traversal/readback comparisons against current linear/per-instance paths and define quality/rebuild policy before closing S04. The present `SceneAcceleration::dispatch` is a standalone borrowed-output diagnostic/refit API; shared-command query integration is still pending. Remaining S00/S01 contracts/capacity, S02 lifecycle/allocation/performance and S08 viewer/visual acceptance remain required. S09 Gearbox is unauthorized; S10 deferred.

### Relevant Files

- `crates/molla-sensors/src/scene_acceleration.rs` — fixed topology, bounds/config/budgets, refit storage, borrowed views and host/GPU accounting tests.
- `crates/molla-kernels/src/scene_bounds.rs`, `shaders/scene_bounds.wgsl` — immutable level parameters, GPU pose/bounds/status pass and bottom-up unions.
- `crates/molla-sensors/tests/sensor_scene_refit_gpu.rs` — independent corner/parent references, ping-pong reuse, invalid poses and revision/empty cases.
- `crates/molla-sensors/benches/bench_scene_refit.rs`, `crates/molla-sensors/Cargo.toml` — reproducible completed-work component benchmark.
- `crates/molla-kernels/src/lib.rs`, `crates/molla-sensors/src/lib.rs`, `crates/molla/src/sensors/mod.rs`, `crates/molla/tests/sensor_api.rs` — kernel/public exports and facade coverage.
- `book/src/sensors/scene.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current contracts, measured component costs and remaining full query integration.

## S04 unified owned-scene traversal — verified 2026-09-25

### Goal

Continue the complete `SENSORS.md` goal by connecting the previous hierarchy/refit foundation to actual ray, LiDAR and camera queries. The preceding status-only turn was no implementation progress; this continuation rechecked the working tree, tested the unfinished helper extraction and implemented the query path. The full goal remains active; S00/S01/S02/S04/S08 still have open gates.

### Instructions

Use Makefile tasks and patch edits. Preserve earlier uncommitted work, physics, assets, Gearbox and user windows. No agents, commits, dependency changes or GUI actions. Engram context/save/session-summary tools were re-discovered unavailable; this project handoff records the session. No memory-folder writes occurred.

### Discoveries

- Standalone bounds refits did not accelerate any query. Removing per-instance CPU encoding requires an addressable shared mesh atlas and a traversal kernel that resolves both finite primitive leaves and local mesh BVHs.
- The two legacy paths compose quaternions and transform origins with slightly different floating-point expressions. Shared primitive helpers preserve primitive arithmetic; mesh candidates retain vector quaternion multiplication and rotate-after-subtraction. Exact cross-shape ties require explicit primitive-first/dense-index ordering because hierarchy visit order is not shape order.
- Queue-written query uniforms are unsafe when the same buffer set is encoded with different parameters before submission. Unified bindings retain immutable count/range parameters and include them in the bounded cache key. Range changes can prepare new variants; pose values alone do not.
- Bodyless/plane-only scene replacements historically accept an unused old State. Initially passing those extra poses as hierarchy seeds caused a real LiDAR regression. Seeds now require matching length; bodyless refits use their internal identity buffer and ignore unused State poses. Nonzero-body count and device-authority checks remain enforced.
- Refit/traversal changes dispatch counts, not just speed: one finite primitive needs an extra refit pass compared with a linear loop, while many mesh instances lose their query/merge pair per instance. Existing dispatch/preparation tests were updated to exact new counts; numerical and ownership checks were not relaxed.
- Warm-query improvement is not startup improvement. This fixture's 1,000-instance construct-and-collect path regresses from about 98 ms raw to 209 ms hierarchical. Pipeline construction and small-scene/refit overhead remain performance work.

### Accomplished

- Extracted unchanged primitive intersection helpers into `ray_shapes.wgsl`, shared by the raw primitive kernel and the new unified trace shader. The original seven primitive GPU regressions passed before further integration.
- Added `MeshBvhGpu::build_validated` and `validate` so scene compilation can build/check CPU trees without first allocating per-mesh GPU buffers. Existing topology, bounds, index and 64-level stack checks remain in force.
- Replaced SensorScene's individual GPU mesh buffer sets with two immutable buffers: 96-byte shape records plus a dense plane list, and a packed mesh atlas containing unique local BVH nodes, triangle order, indices and padded vertices. Instance records reference atlas offsets. CPU geometry and existing model metadata remain available; no duplicate per-mesh GPU BVHs are retained by owned scenes.
- Added a 256 MiB default packed-query GPU payload cap, independent device/binding/u32-address checks, `from_model_with_query_budget`, and `query_device_bytes`. Replacement retains the configured budget and remains transactional. The cap is explicitly not whole-process/peak memory; host trees, model buffers, refit/ray storage, pipelines, staging and result pools are excluded.
- Added `SceneTraceKernel` with one closest-hit dispatch after dependency-ordered GPU refit levels. It traverses finite instance bounds, then shared local mesh BVHs, and separately intersects infinite planes. Twelve storage bindings are checked before pipeline creation. Finite and local stacks are 32/64 entries respectively under the enforced hierarchy/tree limits. Slabs explicitly handle zero direction components; a bounded range margin avoids overflowing a finite maximum range.
- Owned-scene `Raycaster` direct/borrowed paths now use unified traversal, automatically covering GPU-generated LiDAR and camera visibility/optics. Raw-model queries remain the independent linear/per-instance compatibility/reference path. Scene identity/revision changes lazily reconstruct the active hierarchy; body pose changes refit without mesh/topology rebuilding. No extra submission, pose download or host wait was added to encoding.
- Preserved strict positive/range hit rules, primitive-before-mesh and lowest-dense-index cross-shape ties, original local mesh triangle visit order, world-space normals/points, dense IDs, scene stamps and retained output leases. Queries use actual live poses, not the diagnostic refit's identity fallback. Invalid GPU poses remain outside the documented finite-rigid input contract; there is no newly claimed frame error channel.
- Added cumulative hierarchy-build, unified-binding and refit-binding counters. One active hierarchy and 128 immutable trace variants are retained per Raycaster, with two pose-buffer refit variants. Count/range changes are distinct variants; unchanged warmed variants reuse resources. Removed obsolete direct per-instance binding caches and their unused clear helper.
- Added four GPU integration fixtures: 129 moving mesh instances sharing two meshes over three GPU-authoritative pose frames against raw and independent f64 triangle references; all six finite primitive types at rotated/negative/large coordinates with inside and axis-parallel rays; exact primitive/mesh/plane ties, strict range and empty/replaced scenes; exact/under-budget atlas construction and transactional budget-preserving replacement. Added one host budget/address test and one GPU payload test proving actual buffer sizes and one mesh payload across 65 instances.
- Added `bench_scene_queries` for 1,024 rays over 1,000/10,000 mixed finite instances plus a plane, measuring construction+collection, warmed completed query and warmed collection for both paths. Updated plan, scene/raycaster documentation, counters and explicit remaining limitations.

### Verification

All build/test/benchmark commands use `NVIDIA_VERSION=595.84 nix develop --impure -c make ...` and the repository GPU wrapper. Final hardware/toolchain recheck: RTX 4080/Vulkan, NVIDIA 595.84, i7-13700KF/24 logical CPUs, Rust 1.94.1 (e408947bf 2026-03-25), wgpu 29.0.4, HEAD `367d92c` plus the uncommitted worktree.

- Final `make test-sensors`: **320 passing executions**, `/tmp/molla-scene-trace-all-final.log`. This is the previous 314 plus one host budget test, one ignored GPU payload unit test and four GPU integrations; feature-lane executions are summed and the ignored GPU lane ran.
- Release GPU scene/new traversal/camera scene/direct ray/direct LiDAR/direct camera/heterogeneous camera manager suites: **34 passed**, `/tmp/molla-scene-trace-release.log`. New traversal's four fixtures pass in both debug and release.
- Shared geometry validation unit tests: **2 passed**, `/tmp/molla-scene-trace-geometry.log`; `make test-gpu P=molla-geometry T=gpu_narrow_phase A='--test-threads=1'`: **3 passed**, `/tmp/molla-scene-trace-geometry-gpu.log`, including mesh raycast compatibility.
- Workspace/all-target check including viewer and new benchmark passed, `/tmp/molla-scene-trace-workspace.log`; existing unrelated solver test dead-code warning remains. No-default compute/kernel/sensor/facade check passed, `/tmp/molla-scene-trace-host.log`.
- **27 compile-fail doctests passed**, `/tmp/molla-scene-trace-doc.log`; existing dataset doctest remains ignored. Scoped formatting of all changed/untracked Rust files passed with `skip_children=true` and the existing unsorted kernel root excluded, `/tmp/molla-scene-trace-fmt-verified.log`. Its eight added module/export lines were inspected separately. `git diff --check` passed.
- Initial broad GPU audit `/tmp/molla-scene-trace-gpu-audit.log` exposed obsolete counter assertions and the bodyless-seed regression. The second audit left only two counter assertions: camera manager's former primitive preparation and LiDAR's newly immutable range variant. Final full/release runs pass after exact counter updates; no numerical tests were removed or relaxed.
- Final completed-work benchmark: `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_scene_queries'`, `/tmp/molla-scene-trace-bench-final.log`. Ten samples, one-second warmup, three-second target (extended for slow cases). No other owned build/test/benchmark ran concurrently. Construction includes fresh sensor/required scene/geometry/hierarchy/pipelines and readback, but not initial model construction/upload; driver caches are warm. Warm queries include ray normalization/upload, refit/traversal and completion; warm collection includes grouped result download and CPU hit construction. Body poses are fixed and already uploaded. Full intervals and scope are in `book/src/sensors/scene.md`.

| Instances | Path | Construct + collect | Warm completed | Warm collect |
|---|---|---|---|---|
| 1,000 | Raw linear/per-instance | 97.853 ms | 22.724 ms | 24.661 ms |
| 1,000 | Hierarchy | 208.71 ms | 0.28782 ms | 0.34534 ms |
| 10,000 | Raw linear/per-instance | 603.03 ms | 385.65 ms | 412.13 ms |
| 10,000 | Hierarchy | 230.77 ms | 1.8290 ms | 3.2458 ms |

The preliminary `/tmp/molla-scene-trace-bench.log` preceded the overflow-safe range margin. Its 10,000-instance hierarchy timings were 1.6995/2.7273 ms completed/collected; the final run is slower and its readback interval is wide with a high-severity outlier. Do not use the earlier faster numbers as final-source evidence or infer a cause from this uncontrolled run-to-run difference. These are spatially separated synthetic queries, not a realistic machine workload, first-process cold start, GPU timestamp-only duration, physics/Rapier comparison, zero-allocation result or complete performance gate. All owned processes are terminal; no viewer or Gearbox work occurred.

### Next Steps

Continue full S04/S02 acceptance: define and implement bounded hierarchy-quality/rebuild controls for large motion/GPU-only initial layouts; measure degraded/overlapping/primitive-only and realistic mesh scenes; reduce new-pipeline startup and small-scene refit overhead; audit full lifecycle/allocation costs. The new shared query path is integrated, not just the earlier diagnostic core. Remaining S00/S01 contract/tolerance/whole-sensor capacity audits and S08 viewer migration, multi-sensor demo and visual acceptance remain required. S09 Gearbox is separately unauthorized; S10 deferred. Do not mark the complete goal achieved from these query benchmarks/tests.

### Relevant Files

- `crates/molla-sensors/src/scene_geometry.rs`, `scene.rs` — immutable shared atlas, exact payload budget and transactional scene compilation.
- `crates/molla-sensors/src/scene_trace.rs`, `scene_acceleration.rs`, `raycast.rs`, `lidar_batch.rs` — shared-command refit/traversal integration, bounded retained resources and query counters.
- `crates/molla-kernels/shaders/ray_shapes.wgsl`, `raycast.wgsl`, `scene_trace.wgsl`; `src/raycast.rs`, `src/scene_trace.rs`, `src/lib.rs` — shared numerical helpers and unified closest-hit compute path.
- `crates/molla-geometry/src/mesh_bvh_raycast_gpu.rs` — allocation-independent validated BVH construction with unchanged raw-query compatibility.
- `crates/molla-sensors/tests/sensor_scene_trace_gpu.rs` — independent many-instance/oracle, primitive/range/tie and budget regression coverage; existing direct/scene fixtures retain numerical checks with new exact work counters.
- `crates/molla-sensors/benches/bench_scene_queries.rs`, `Cargo.toml` — completed construction/query/readback comparisons.
- `book/src/sensors/scene.md`, `raycaster.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — contracts, final measurements, limitations and continuation state.

### Concurrent external edit and compatibility recheck

The final status check found a new, externally authored change in `crates/molla-compute/src/engine.rs`: `Backend::External`, `Engine::from_device`, public adapter-info conversion/device-loss recording and a unit test. This file was clean at the start of this continuation, was not edited by this work and was left untouched. Its modification time (00:47:52 local) is after the full sensor/release validation and after the final benchmark binary was built. The 320/34 counts and query timings above describe that earlier engine source, not runtime verification of the new external-device API.

A subsequent workspace/all-target compatibility check passed with the external edit, `/tmp/molla-scene-trace-external-check.log`; all four compute GPU-feature unit tests passed, `/tmp/molla-scene-trace-external-unit.log`. The engine file hash was unchanged across that recheck (`/tmp/molla-scene-trace-external-engine.sha256`). This confirms compilation and those unit tests only, not renderer adoption or external-device GPU execution. Preserve this external WIP in the next turn; do not attribute it to the traversal implementation. All owned processes, including this recheck, are terminal.

## S04 periodic GPU rebuilding — recovery and status 2026-09-25

### Goal

Continue the full `SENSORS.md` objective. This status turn recovered the preceding implementation handoff and checked current plan rows and verification logs; it did not run additional builds or tests. Engram session-summary/context tools remain unavailable.

### Instructions

Use Makefile tasks and patch edits, preserve external engine work, and leave Gearbox, assets and user windows untouched. Do not describe partial sensor verification as completion of the full plan.

### Discoveries

- Retained radix-sort bindings allow sorting inside shared command batches; the existing primitive dispatch helper submits independently. The new kernel reuses its WGSL without changing compute-primitives.
- Copying reordered instances back into the retained buffer avoids CPU ping-pong ownership assumptions after cancelled command batches. Reordering is private to query acceleration; standalone diagnostic CPU topology stays unchanged.
- Periodic rebuilding is a heuristic, not a quality-triggered policy. Minimum radix scratch capacity also matters for small-scene buffer limits.

### Accomplished

- Added bounded periodic GPU Morton/radix leaf rebuilding for shared ray, LiDAR and camera queries: default every 64 encoded queries for at least 64 leaves, with a separate 128 MiB payload cap. Includes transactional policy changes, group preflight and encoded/cancelled-query counters.
- Added host budget checks, a GPU hierarchy-quality unit fixture, and three integration fixtures covering mixed-shape ties, cancellation/leases, and ray/LiDAR/camera budget rejection.
- Latest full suite logged 326 passing executions in `/tmp/molla-scene-rebuild-all-verified.log`; the three new integration fixtures passed release in `/tmp/molla-scene-rebuild-release-verified.log`. Workspace/no-default checks and 27 compile-fail doctests passed in the preceding implementation continuation.
- Completed 18 benchmark cases in `/tmp/molla-scene-rebuild-bench.log`. With 1,024 rays and scrambled GPU-only initial poses, 10,000 instances measured 3.3121 ms fixed versus 0.70164 ms periodic; 1,000 instances regressed from 0.33328 ms to 0.42276 ms. These are synthetic completed-query measurements, not physics or Gearbox results.
- Default hierarchy construction measured 236.83/261.48 ms for 1,000/10,000 instances. The 10,000-instance warmed collection measured 7.9753 ms versus the earlier 3.2458 ms; causes are unproven and run-to-run baselines changed. Startup/readback regressions must not be hidden.

### Next Steps

- The last test-only extension alternates two GPU pose buffers in the cancellation/lease fixture and expects two refit preparations. It was added after the above runs: format and rerun that fixture in debug/release before claiming verification of the latest test source.
- Finish detailed benchmark documentation and S04 checklist reconciliation; preserve historical measurements as historical. Recheck formatting/diff and the external engine hash.
- Implement/evaluate quality-triggered rebuild decisions, investigate small-scene/startup/readback costs, and complete full lifecycle/capacity/performance audits. S00/S01/S02/S04 remain open; S08 viewer migration, multi-sensor demo and visual acceptance remain unfinished. Gearbox integration is separately unauthorized.

### Relevant Files

- `crates/molla-sensors/src/scene_rebuild.rs`, `scene_trace.rs`, `scene_acceleration.rs`, `scene_geometry.rs` — policy, accounting, retained hierarchy and rebuilding integration.
- `crates/molla-kernels/src/scene_rebuild.rs`, `shaders/scene_rebuild.wgsl` — Morton keys, retained radix passes and leaf permutation.
- `crates/molla-sensors/src/raycast.rs`, `lidar_batch.rs`, `tiled_camera.rs`, `camera_batch.rs` — public policies, counters and transactional preflight.
- `crates/molla-sensors/tests/sensor_scene_rebuild_gpu.rs`, `benches/bench_scene_queries.rs` — regression fixtures and completed-work measurements.
- `book/src/sensors/scene.md`, `raycaster.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — contracts and remaining acceptance gates.

## S04 policy controls and S08 benchmark entrypoint — verified 2026-09-25

### Goal

Continue the complete sensor plan. The preceding status-only turn was classified as no implementation progress. This continuation verified the outstanding alternating-pose-buffer test, expanded performance controls, investigated the collection regression and added the requested sensor-specific benchmark target. The full goal remains active.

### Instructions

Makefile tasks and patch edits only; preserve external engine changes and user processes/windows. No subagents, commits, Gearbox edits or GUI actions. Engram save/session-summary tools remain unavailable; this project handoff is the session record, not an Engram save.

### Discoveries

- The old 10,000-instance 7.9753 ms collection result did not reproduce in the first controlled run: default periodic collection measured 0.80190 ms without a query/readback implementation change. This is evidence against assuming a deterministic code regression, not proof of a fix or its historical cause.
- External GPU activity can begin mid-run. During the repeat, `nvidia-smi pmon` identified an independently launched `gearbox` process (PID 3412950) at 99% SM utilization while the small fixed case rose from 0.19919 to 3.8331 ms. The saved follow-up snapshot `/tmp/molla-scene-policy-contention.log` shows both GPU processes; activity varies by sample. Process filtering for `gearbox-sim`/`gearbox-cli` alone missed the actual executable name `gearbox`.
- Only the owned benchmark PID 3409111 was sent SIGINT; its tool session terminated with exit 2. Gearbox remained running and untouched. Do not use the interrupted repeat as uncontended timing evidence, and do not attribute older runs to this later observed contention without evidence.

### Accomplished

- Verified the latest cancellation/lease fixture with alternating same-device GPU-authoritative pose buffers; it retains two refit bindings and preserves old results through later queries and policy changes.
- Added fixed-order and initial-only (`interval = u32::MAX`) warm controls to `bench_scene_queries`, alongside default periodic ordering. All use the same seeded model/rays; initial IDs/distances have analytical assertions, and scene/preparation/rebuild counters are checked. The full executable now offers 26 cases; this continuation measured only the twelve filtered policy controls, not all 26.
- Added `make bench-sensors`, selecting `SENSOR_BENCH` (default `bench_scene_queries`) and forwarding Criterion options via `A`, always with the GPU feature and `RUN_WITH`. Existing generic benchmark behavior is unchanged. Updated help and plan commands.
- Documented the preceding 18-case rebuilding measurements, including small-scene/construction/readback regressions, and the new twelve-case controls with intervals and limitations. Reconciled the detailed S04 checklist with periodic rebuilding, without closing quality/performance gates. Corrected stale query-counter documentation.

### Verification

- `make test-gpu P=molla-sensors T=sensor_scene_rebuild_gpu`: **3 passed**, `/tmp/molla-scene-rebuild-pingpong-debug.log`.
- Same with `RELEASE=1`: **3 passed**, `/tmp/molla-scene-rebuild-pingpong-release.log`.
- Current `make test-sensors`: **326 passing executions**, `/tmp/molla-scene-rebuild-pingpong-full.log`, including the final test extension. Subsequent library source change is documentation only.
- `make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_scene_queries -- scene_queries_1024_rays/hierarchy.*warm'`: **12 timing cases completed**, `/tmp/molla-scene-policy-control.log`. Same RTX 4080/Vulkan, NVIDIA 595.84, Rust 1.94.1, HEAD `367d92c` plus WIP; ten samples, one-second warmup, three-second target. No owned test/build ran concurrently. External Gearbox compilation had ended before this run; later Gearbox GPU activity contaminated the repeat, not proven to affect this first run.
- Default periodic/fixed/initial-only completed-query point estimates were 0.19998/0.19919/0.20559 ms at 1,000 instances and 0.74172/0.78361/0.84012 ms at 10,000. Corresponding collection estimates were 0.26896/0.22196/0.22832 and 0.80190/0.94713/0.96337 ms. These sequential Criterion comparisons do not prove a universally best policy and are not per-frame p50/p95 measurements.
- `make bench-sensors A='scene_queries_1024_rays/hierarchy.*warm'` launched the correct executable/filter, then was deliberately interrupted for observed external contention; `/tmp/molla-scene-policy-control-repeat.log`. Dry-run default and alternate LiDAR smoke commands also matched the intended recipes. No successful complete repeat is claimed.
- GPU/all-target sensors check passed: `/tmp/molla-scene-policy-check.log`. Scoped formatting check passed for changed/untracked Rust, excluding the pre-existing unsorted kernel root and external engine file: `/tmp/molla-scene-policy-fmt.log`. `git diff --check` passed; the external engine hash still matches `/tmp/molla-scene-trace-external-engine.sha256`.

### Next Steps

All owned tool sessions are terminal. Do not rerun uncontended timing while the external GPU application is busy or close it without user instruction. Other implementation work is not blocked. Add per-process activity capture to future performance evidence; obtain stable repeated controls before choosing policy defaults or attributing regressions. Quality-triggered rebuild decisions, realistic/moving/overlapping geometry and full percentile/resource/combined-load performance acceptance remain open. S00/S01/S02 audits and S08 viewer migrations, multi-sensor demo and visual acceptance are still required. S09 remains unauthorized, S10 deferred; do not mark the full goal complete.

### Relevant Files

- `crates/molla-sensors/tests/sensor_scene_rebuild_gpu.rs` — final alternating-pose lifecycle fixture, now verified.
- `crates/molla-sensors/benches/bench_scene_queries.rs` — same-scene policy controls and correctness/resource assertions.
- `Makefile` — scoped sensor benchmark target and help.
- `crates/molla-sensors/src/raycast.rs` — corrected counter documentation only this continuation.
- `book/src/sensors/scene.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — complete measured results, explicit contamination and remaining gates.

### Key Learnings

1. Mid-run external GPU contention can overwhelm policy differences; check per-process activity rather than assuming an idle initial sample guarantees isolation.
2. Passing correctness/resource counters and a faster single run do not establish repeatable latency or resolve the full sensor performance gate.

## S08 asynchronous IMU/contact previews — implementation 2026-09-25

### Goal

Continue the full `SENSORS.md` objective through viewer integration while leaving performance timing separate. The previous goal turn was progress: it added benchmark controls and established external contention. This continuation migrates examples 06/07 and adds shared preview infrastructure; it does not replace the required seven-example migration, moving-mesh multi-sensor demonstration or visual acceptance with these two examples.

### Instructions

Makefile commands, patch edits, no agents or commits, and no user-window actions. No Gearbox, assets, soil/tyre implementation or physics-library edits. Engram context/save/session-summary tools remain unavailable, so this project handoff records the session. Base memory was read only for the full-scope/no-unproven-performance constraint; stale implementation facts were superseded by current source.

### Discoveries

- IMU and contact examples used synchronous convenience collection on every render update, including paused states. The IMU used a nominal frame interval rather than elapsed simulation time after skipped measurements.
- Viewer `Sim::reset` blindly called `State::download_from_device`. A deliberately host-dirty velocity in the new fault/retry fixture reproduced the protected storage assertion. Reset now uses the existing authority-aware readback helper for body/particle pose and velocity buffers on both states; storage assertions and physics kernels were not changed.
- The model's exact default gravity is 9.80665, not the UI-rounded 9.81. The initial analytical test failed on that fixture assumption; it now uses the configured constant with the same 1e-4 tolerance. Actual readout was approximately (11.999999, 9.806650, 0).
- An asynchronous pipeline can still copy unused data. Closing the IMU scene pane now stops new display readbacks while GPU sampling/history continues; one latest leased frame stays within the three-slot pool. Reopening while paused collects it without redispatching the IMU. Contact still needs all four channels for viewport gizmos even with its pane closed.
- An external change in `book/src/compute/runtime.md` documents renderer adoption and claims Gearbox use. It predates this continuation and was preserved, not authored or runtime-verified here. The external engine hash remains unchanged.

### Accomplished

- Added `sensor_preview.rs`: checked three-slot output/staging budgets, bounded FIFO tickets, nonblocking latest-frame collection, simulation stamps, duplicate-state suppression, deferred-state accounting, CPU submission timing and explicit unmeasured GPU duration.
- Added a viewer clock advanced once per completed ordinary rigid substep. Reset and scene rebuild advance its revision. It deliberately does not claim to clock host-driven scenes that bypass the ordinary stepper; those must be wired during the remaining migrations.
- Examples 06/07 now encode leased outputs via `SensorCommandBatch` and optionally enqueue grouped display readback rather than calling synchronous `update`. IMU priming has no display copy; later samples copy 32 bytes when needed. The three-body contact display copies 120 bytes and waits for a post-reset physics step rather than reusing old collision data.
- The next IMU finite difference spans simulation time since its last successful submission, including backpressure gaps. Errors remain visible and stop repeated failing work until reset/switch retries. Panes show cadence, sample age, submitted/collected and pending/deferred counts. Contact arrows/text explicitly describe geometry, not net force.
- Added five headless viewer tests covering clock/budget arithmetic, bounded readback/latest draining, GPU-authoritative IMU velocity, analytical acceleration, saturation and skipped intervals, pause/single-step, pending-result reset, producer fault/retry, pane visibility and paused reopen, scene switching, CPU-reference contact aggregation/empty clearing, and mixed-authority body/particle reset.
- Added `make test-viewer-sensors` (explicit ignored GPU tests plus pure checks), and `DOCS_DIR` override for non-destructive documentation builds. Added a viewer integration book chapter/navigation entry; removed stale claims that IMU/contact leased batching is pending. S08 now reads IN PROGRESS with five example migrations and visual/demo/performance gates still open.

### Verification

- `make test-viewer-sensors`: **5 passed**, including final closed-pane/paused-reopen behavior; `/tmp/molla-viewer-sensors-visible.log`. Earlier target run also passed before that extension (`/tmp/molla-viewer-sensors-target.log`). Two tests are pure, three explicitly require GPU.
- Existing `viewer_readback_tests` via `make test CARGO='nixVulkan cargo' APP_TARGET='-p molla_viewer --bin molla_viewer -- viewer_readback_tests --ignored --test-threads=1'`: **2 passed**, including PPF/barrier/VBD/Arm/Pile scene switching; `/tmp/molla-viewer-readback-regression.log`.
- Initial viewer/all-target check passed, `/tmp/molla-viewer-aux-check.log`. Current targeted tests compile the final implementation. Initial failures are retained in `/tmp/molla-viewer-aux-test.log` and `-recheck.log` (gravity reference), and `-current.log` (real reset assertion); `-reset.log` passes after the authority fix. No tolerance/assertion was weakened.
- Scoped formatting was applied only to the new modules and two changed examples. Main's small integration diff was inspected manually; its large pre-existing formatting was not rewritten. `git diff --check` passes. External engine and compute-doc hashes match `/tmp/molla-scene-trace-external-engine.sha256` and `/tmp/molla-viewer-external-runtime.sha256`.
- `make docs DOCS_DIR=/tmp/molla-sensors-viewer-book` exited 0 and emitted the new chapter HTML, but mdbook reports an ERROR for the pre-existing `output.html.copy-fonts` option in `book/book.toml`. Log: `/tmp/molla-viewer-sensors-docs.log`. This is **not a clean documentation-build pass**; the config/tool-version mismatch remains unresolved. No generated repository documentation was overwritten.
- No viewer window was launched or closed. Headless tests do not establish visual acceptance or a measured speedup. The library's earlier 326-execution result was not rerun or re-labelled as new evidence this turn.

### Next Steps and live process

The broad ordinary viewer test command is **still running**, not failed or blocked: tool session **78160**, test process PID **3487625**, `/tmp/molla-viewer-host-final.log`. It was last verified live at approximately five minutes, using one CPU core, progressing through existing `tire_rig_replay_panel_*` tests. **Poll the same session; do not restart or kill it because observation expires.** This binary predates only the final IMU visibility extension, which is separately covered by the passing five-test current lane. Its shell chain will subsequently run current viewer/all-target check (`/tmp/molla-viewer-aux-check-final.log`) and scoped formatter check (`/tmp/molla-viewer-sensors-fmt.log`). All other owned tool sessions have completed. No aggregate pass count is claimed for this broad run yet.

After collecting that evidence, continue examples 05/12/105/106/108 and the deterministic moving-mesh multi-sensor demo, then inspect real rendered outputs and interaction behavior. Resolve the mdbook compatibility error without silently changing intended theme behavior. Preserve S00/S01/S02/S04 resource/quality/performance gates; stable repeated timing requires external GPU activity monitoring. S09 remains outside this agent's authorization; external work is not evidence of this goal's integration acceptance. Keep the full goal active.

### Relevant Files

- `bin/molla_viewer/src/sensor_preview.rs` — bounded asynchronous preview, clock, pane metrics and helper tests.
- `bin/molla_viewer/src/examples/06_imu.rs`, `07_contact.rs` — direct sensor submission, selected display collection, state/fault/reset handling and UI readouts.
- `bin/molla_viewer/src/viewer_sensor_tests.rs` — headless ECS/GPU integration and authority-aware reset fixtures.
- `bin/molla_viewer/src/main.rs` — module wiring, simulation clock/revision updates and reset ownership fix.
- `Makefile` — viewer sensor lane and documentation output override.
- `book/src/sensors/viewer.md`, `imu.md`, `contact_sensor.md`, `book/src/SUMMARY.md`, `SENSORS.md` — integration contracts and explicit unfinished acceptance.

### Key Learnings

1. Pause and backpressure need simulation-time bookkeeping: skipped IMU samples must increase the next finite-difference interval, not replay an unavailable state.
2. Reset must respect Buffer authority just as stepping does; blindly downloading a host-dirty buffer loses data or trips the intentional assertion.
3. Hidden-pane device sampling can retain a latest leased frame without host copies, allowing paused reopen without changing sensor history.

## S08 GPU LiDAR previews and documentation repair — verified 2026-09-25

### Goal

Continue the complete sensor plan after the preceding IMU/contact implementation. The prior turn was progress, and its live broad viewer suite was polled through the same session rather than restarted. This continuation migrates examples 105/106; 05/12/108 and the multi-sensor/visual/performance work remain required.

### Instructions

Use Makefile tasks and patch edits. Preserve the external engine/compute-doc work, all Gearbox/asset/physics implementation, and user windows. No agents, commits or GUI actions. Engram tools remain unavailable; this project handoff is the persistent session record. Memory-folder contents were not changed.

### Discoveries

- 105 previously constructed host scan rays and synchronously collected every render update; 106 used a synchronous one-shot aerial scan and did not refresh on same-scene reset. Both scenes bypass the ordinary physics solver, so the IMU/contact rigid-step clock needed explicit sensor-only advancement on run/single-step.
- Selecting only display channels reduces transfer payload without changing the full device pipeline: 105 needs distance/shape/point (24 bytes/ray), while 106 needs origin/shape/point (36 bytes/ray). Normals, directions and pulse reductions remain on device. Both still need collection with a closed pane because viewport gizmos use those channels.
- Installed mdbook is 0.5.3. Its official 0.5 migration notes confirm `copy-fonts` was removed and default fonts are copied automatically unless overridden: https://github.com/rust-lang/mdBook/blob/master/CHANGELOG.md#05-migration-guide . Removing that obsolete key exposed a second real error: configured `theme/molla.css` did not exist or appear in tracked files. The dangling CSS reference was removed, not replaced with an invented stylesheet. Configured navy/ayu themes remain; generated HTML and emitted font files confirm those assets/settings are present.

### Accomplished

- Added shared `lidar_preview.rs`: checked/64-rounded capacity, one retained owned `SensorScene`, GPU `LidarBatch`, three-slot leased outputs and selected asynchronous readback. The full output payload is 84 bytes per capacity ray per slot; staging includes only selected display channels. Library capacity/device checks remain in force.
- Migrated 105 with its existing 24×64 grid, three sub-rays, 0.006 rad divergence and 25 m range. It submits once per new simulation frame, polls while paused, uses live GPU poses and retains its hierarchy. No host ray vectors or synchronous scan collection remain in the viewer path.
- Migrated 106 preserving one-shot behavior while adding same-scene reset refresh and continued paused polling. GPU Halton/nadir layout matches the prior scan. Origin display selects one origin per pulse; hit points preserve valid-ray ordering.
- Static sensor-only scenes now advance the explicit viewer clock on run/single-step without invoking a physics solver. Their update systems run after `step_simulation`, and gizmos after the corresponding update. Other host-driven scenes are not silently assigned time semantics.
- Added right-pane settings/status for 105/106, including fixed configuration, hit counts, sample age, bounded pending/deferred counts and truthful CPU/unmeasured-GPU timing. Corrected old descriptions that claimed nonexistent editable scan-grid/density controls. Settings remain fixed in these two demos; interactive configuration belongs to later work if required.
- Added a comprehensive LiDAR headless GPU fixture: legacy layout/query parity, live device-authoritative body poses, one retained scene build, exact selected bytes/descriptors/output allocation counts, pause/single-step, pending reset, switching and aerial one-shot/re-entry semantics. Existing IMU/contact/visibility tests still pass.
- Repaired book configuration and rebuilt outside the repository via `DOCS_DIR`. Updated S08 to four migrated examples, leaving all remaining gates explicit, and documented the LiDAR paths and budgets.

### Verification

- Previous broad viewer process/session **78160** finished its ordinary test command: **85 passed, 5 ignored**, 785.71 seconds, `/tmp/molla-viewer-host-final.log`. This is the earlier viewer snapshot, not new coverage of the LiDAR migration. The subsequent chained check encountered the intentionally unfinished intermediate edit (main referenced the two pane functions before their definitions were patched), and the chain exited 2. No broad test failed; final current-source checks below supersede that transient compile failure. Its queued formatter did not run.
- Current `make test-viewer-sensors`: **6 passed**, `/tmp/molla-viewer-lidar-tests.log` (two pure tests, four explicitly ignored GPU tests actually executed). Tests compare cloud points/distances within 1e-4 and retain the previous IMU/contact assertions. No tolerance was weakened.
- Final `make check APP_TARGET='-p molla_viewer --all-targets'` passed without compiler warnings: `/tmp/molla-viewer-lidar-check-final.log`. Scoped formatter check passed for both new modules, shared test file and four migrated examples: `/tmp/molla-viewer-lidar-fmt-final.log`. Main's small integration diff was inspected manually; pre-existing unrelated formatting was not rewritten. `git diff --check` passed.
- Final `make docs DOCS_DIR=/tmp/molla-sensors-viewer-book` passed **without the prior configuration/rendering errors**: `/tmp/molla-viewer-docs-final.log`. The new chapter HTML exists, navy/ayu settings are emitted and default font files are present. This validates documentation generation/assets, not visual inspection of the book or application.
- External engine and compute-document hashes still match `/tmp/molla-scene-trace-external-engine.sha256` and `/tmp/molla-viewer-external-runtime.sha256`. No dependency, physics-library, Gearbox or asset changes were made. All owned tool sessions are now terminal; no viewer window was launched or closed.

### Next Steps

Continue viewer 05 GPU layout/shared scene migration, 12 selected camera collection, and 108 camera-to-annotation device chaining. Then implement the deterministic moving-mesh multi-sensor demonstration and inspect actual rendering, pause/reset/switch and pane behavior. Keep S00/S01/S02/S04 allocation, contract, quality-triggered rebuild and full performance gates open; these headless results are not speed measurements or visual acceptance. The prior live session is closed: do not try to restart it or claim it remains running. S09/S10 remain separately gated/deferred, and external Gearbox work is not this goal's integration evidence.

### Relevant Files

- `bin/molla_viewer/src/lidar_preview.rs` — shared GPU-layout/query and selected-readback owner.
- `bin/molla_viewer/src/examples/105_lidar_v2.rs`, `106_aerial_lidar_pass.rs` — migrated pipelines, revision handling and panes.
- `bin/molla_viewer/src/main.rs` — sensor-only clock advancement, update ordering, pane resources and accurate descriptions.
- `bin/molla_viewer/src/sensor_preview.rs` — per-preview cadence display, including one-shot mode.
- `bin/molla_viewer/src/viewer_sensor_tests.rs` — LiDAR parity/lifecycle/device-authority fixture.
- `book/book.toml` — removed obsolete option and absent CSS reference; retained theme settings.
- `book/src/sensors/viewer.md`, `SENSORS.md`, `SENSORS_PROGRESS.md` — current migration status, evidence and remaining work.

### Key Learnings

1. Host-driven sensor demos need explicit simulation-time advancement; inheriting the rigid-step clock without wiring their skipped step branch would freeze sampling.
2. GPU LiDAR can retain its complete device outputs while copying only the channels the display actually consumes.
3. A successful mdbook exit with logged configuration errors is not a clean build; fix the configuration and inspect emitted assets before claiming success.

## S09 Gearbox sensor links — verified 2026-09-25

### Goal

The user authorized S09 ("gearbox now runs on this physics, so there is no gate anymore"). Integrate Molla's GPU IMU and LiDAR into Gearbox on authored sensor links, streaming per link beside the existing controller telemetry, and validate on a real machine scene. Molla-side S00–S08 work continues separately; this increment does not close those gates.

### Discoveries

- Gearbox (`d93e6a8`, clean) pins Molla at git `ef25106`, 66 commits behind this checkout, with no `molla-sensors` dependency. Bevy 0.19.1 and Molla share wgpu 29.0.4, so the render device can be adopted directly; Bevy installs its own device-lost callback, so an adopted engine must not overwrite it.
- Gearbox machine colliders are convex decompositions (`ConvexHull`) and tyres are `RoundCylinder`s; both were unsupported by `SensorScene`, which would have left the LiDAR seeing only terrain and primitives.
- The machine agent's shared-memory config allowed 8 publishers (7 already taken) and 16 KiB payloads; the second sensor publisher silently failed to register and a 360×16 sweep (~92 KB) could never be sent.
- Mount poses resolved from f32 `GlobalTransform`s jitter; an exact signature comparison rebuilt the rig every frame until the machine settled.
- Entering the Gearbox Nix shell without `NVIDIA_VERSION` bakes the Mesa ICD into `nixVulkan`; the first live launch rendered on `llvmpipe` at 35 s per frame.

### Accomplished

- Molla: `Engine::from_device(device, queue, adapter_info)` with `Backend::External` and public `AdapterInfo::from_wgpu`/`DeviceHealth::record`; owned scenes triangulate convex hulls (degenerate hulls stay unsupported) and approximate round cylinders by their dilated cylinder, with `sensor_scene_gpu` extended for both; book and capability tables updated.
- Gearbox: `gearbox:sensor:*` parsing on `LinkSpec` (kind inferred from `imu*`/`lidar*`/`laser*` names, explicit `kind` otherwise; out-of-range values are validation errors); `PhysicsBackend::with_molla_scene`/`molla_body_handle`; `PhysicsWorld::simulated_seconds`; `bin/gearbox/src/sensors.rs` with an ECS resolver and an ECS-free `Rig` (`build`/`sample`/`collect`); `gearbox-api` `LidarScan` wire type (chunked, JSON codec), per-link `publish_sensor_*` returning delivery, `MAX_PAYLOAD_BYTES`/`LIDAR_CHUNK_PULSES`, 24 publishers; CLI topic listing; spec §7.6; `assets/world/sensor_yard.usda`; `docs/MOLLA_BACKEND.md` section.
- Frames: IMU readings are in the link frame with orientation remapped like `odom` (`q_ros = P · q_sim`, `P` = the `(sim_z, sim_x, sim_y)` permutation); LiDAR sweeps look along +X with +Z zenith and azimuth towards +Y.

### Verification

- Molla `make test-gpu P=molla-sensors T=sensor_scene_gpu`: 6/6 with hull and round-cylinder rays (`/tmp/claude-1000/…/scratchpad/logs/molla-scene-round.log`); `molla-compute` unit tests 4/4. Full `make test-sensors`: **326 passing executions**, 0 failures (`…/scratchpad/logs/molla-test-sensors.log`), summing feature-lane executions including the explicit ignored GPU lanes.
- Gearbox `nixVulkan cargo test -p gearbox-sim --bin gearbox imu_and_lidar -- --ignored` (RTX 4080): IMU 9.81 m/s² along link +Z, zero gyro, remapped orientation; LiDAR ranges `[inf, inf, inf, 6.364, 4.5, 6.364, inf, inf]`, three hits, one 8-pulse chunk. Unit tests: link parsing, frame conventions, wire/JSON round trip.
- Live release instance on the RTX 4080 with `sensor_yard.usda`: IMU parked `az 9.80` with meadow tilt; driving `--forward 2 --turn 0.3` gave `gyro.vz 0.25–0.38 rad/s`, `ax 1.0–1.4`, `ay 0.3–1.0 m/s²`; a full LiDAR sweep arrived as seven chunks of one `sample`, upper rows missing and lower rows hitting terrain and the tractor body (0.48–59.9 m). Pause froze both streams (same `sample` for 4 s), play resumed them, `clear machines` dropped the rig cleanly. Frame rate 8–12 fps with and without the machine; physics 0.15–0.44 ms/step. Screenshot `/tmp/gearbox-sensor-yard.png`, sweep `/tmp/gearbox-sensor-lidar-sweep.json`.
- Gearbox binary suites `cargo test -p gearbox-sim --bin gearbox` pass 158/158 (81 ignored) with `GEARBOX_PHYSICS=molla` and with `rapier` (`/tmp/claude-1000/…/scratchpad/logs/gearbox-bin-{molla,rapier}.log`), so the backend trait additions and the Molla checkout switch (66 commits ahead of the previous `ef25106` pin) did not regress the existing physics lanes.
- Not measured: sensor transfer overhead in isolation, moving parts of a second machine, cameras.

## S08 configurable GPU LiDAR preview — verified 2026-09-25

### Goal

Continue the full SENSORS.md objective. The preceding status-only turn was no progress;
this continuation implements example 05 rather than repeating its proposed work.
S08 now has five of seven viewer migrations, not completed acceptance.

### Instructions

Use Makefile tasks and patch edits, preserve unrelated work and user windows, and do
not infer performance from correctness tests. No agents, commits or GUI actions.
Engram context/summary tools were unavailable in the tool registry; this project
handoff records the continuation instead. Memory-folder contents were not modified.

### Discoveries

- The GPU spherical descriptor accepts negative theta. Forward -Z, up +Y and
  theta = pitch - pi/2 reproduce the old +Z-forward / +X-positive-yaw convention
  while preserving ascending pitch rows and yaw columns, including pole crossing.
- Paused configuration changes require identities separate from simulation ticks.
  Retrying a request after a full ring must not advance simulation time, whereas
  ordinary deferred simulation states should not be replayed.
- Asynchronous ray display must retain sampled configuration: using current range
  or origin would misdraw a completed older scan while new settings are pending.
- The live shared progress file now contains separately authored S09 authorization
  and Gearbox integration evidence. Earlier handoffs calling S09 unauthorized are
  stale. This continuation neither changed nor independently revalidated Gearbox.

### Accomplished

- Replaced 05's host ray-array construction and synchronous Raycaster collection
  with GPU LidarBatch layout, a retained SensorScene and selected asynchronous
  readback. Preserved the model, controls, row ordering and singleton-axis centers.
- Added ray-fan projection to the shared helper: origin/direction/shape/point only,
  52 device bytes per ray. Full outputs remain leased on GPU; display queues and
  configuration snapshots are bounded to three in-flight requests.
- Auto-spin now follows completed simulation ticks and freezes while paused.
  Manual paused edits use monotonically increasing request IDs at unchanged
  simulation time. Same-request updates do not resubmit; the newest unsent request
  retries after backpressure clears.
- Bound displayed rays to their sampled origin/range. Capacity grows transactionally
  in checked 64-ray increments, retains the owned scene, discards prior pending
  readbacks/metadata and keeps the previous valid displayed fan until replacement.
  Shrinking reuses capacity. Pool/readback counters restart on growth.
- Added pane status/errors and reset/revision invalidation. Removed the now-unused
  hit_to_world helper located in 19_diff_arm6.rs and its RayHit import; no arm
  simulation behavior changed.
- Added layout validation and headless GPU parity/lifecycle tests, including
  pending growth to the full 512 × 64 UI resolution. Updated plan and viewer docs.

### Verification

Commands used NVIDIA_VERSION=595.84 nix develop --impure -c make with the native
nixVulkan GPU wrapper.

- Final make test-viewer-sensors: **8 passed**, zero failures, 4.75 seconds;
  /tmp/molla-viewer-lidar05-tests-final.log. Three pure tests and five explicit GPU
  tests ran; earlier IMU/contact and 105/106 fixtures remain passing.
- New coverage independently reconstructs old yaw/pitch directions and compares
  ordering and hit results; checks singleton axes, wrapped angles, pole crossing,
  paused edits/deduplication, sampled display configuration, simulation-time spin,
  growth/shrink, pending resize/reset, scene switching/re-entry, exact selected
  transfer bytes and zero preview blocking waits. The shared ring test explicitly
  verifies a denied request can retry with the same ID after draining.
- Final make check APP_TARGET='-p molla_viewer --all-targets' passed without compiler
  warnings: /tmp/molla-viewer-lidar05-check-final.log. Scoped make fmt-files --check
  and git diff --check passed; /tmp/molla-viewer-lidar05-fmt.log. Main and the small
  orphan-helper deletion were inspected without reformatting legacy files.
- make docs DOCS_DIR=/tmp/molla-sensors-viewer-book passed cleanly:
  /tmp/molla-viewer-lidar05-docs.log.
- External engine/compute-doc hashes still match their existing /tmp manifests.
  No library physics, Gearbox, assets or user windows were changed this turn.
  All owned tool sessions (61622, 31663, 41660, 43212) finished; none is live.
- These are functional/headless checks, not rendered acceptance or timed sensor
  performance evidence. The full sensor-library matrix was not rerun this turn.

### Next Steps

Migrate 12 Camera (currently synchronous TiledCamera dispatch/collection into five
Bevy images) and 108 Annotation (currently camera collection then host annotation)
without losing optics/settings or display behavior. Then add the deterministic
moving-mesh multi-sensor demo and inspect actual rendering/pane/pause/reset/switch
behavior. Keep S00/S01/S02/S04 contracts, allocations, quality-triggered rebuilds
and full benchmark gates open. S09 remaining measurements/camera links and S10
deferred scope retain their current separately recorded status. Goal stays active.

### Relevant Files

- bin/molla_viewer/src/examples/05_lidar.rs — configurable GPU preview and sampled display.
- bin/molla_viewer/src/lidar_preview.rs — projection, request submission and capacity growth.
- bin/molla_viewer/src/sensor_preview.rs — retryable bounded requests and ring test.
- bin/molla_viewer/src/viewer_sensor_tests.rs — layout/resize/pause/reset parity fixtures.
- bin/molla_viewer/src/main.rs — config equality, state reset and import cleanup.
- bin/molla_viewer/src/examples/19_diff_arm6.rs — removed orphaned LiDAR hit conversion only.
- book/src/sensors/viewer.md, SENSORS.md — current integration contracts and open gates.

### Key Learnings

1. Interactive sensor request identity must be independent of simulation time.
2. Async visualization needs the configuration sampled with each completed frame.
3. A negative-theta basis can preserve LiDAR pitch/yaw ordering without host rays.

## S08 asynchronous camera and GPU photographic processing — verified 2026-09-25

### Goal

Continue the complete SENSORS.md objective after the verified example-05 migration.
The previous turn was implementation progress. This turn migrates example 12,
bringing code migrations to six of seven; broader plan acceptance stays open.

### Instructions

Use native Makefile tasks through the Nix environment and built-in patch edits.
Preserve unrelated WIP, external engine/runtime documentation, physics, Gearbox,
assets and user windows. No agents, commits or GUI actions. Engram tools remained
unavailable in the registry; this project summary persists the work instead.
No memory-folder files were written.

### Discoveries

- The previous camera update rendered and synchronously downloaded all five channels
  every wall-clock frame, then cloned color again for CPU photographic processing.
  Its orbit advanced even while simulation was paused and set_config errors were ignored.
- All five channels are used by the open pane. Whole-pane visibility can suppress
  new readbacks without suppressing GPU sensing; paused reopen needs a retained
  latest device frame plus its sampled camera pose.
- Bevy thumbnail images need not be recreated when the Molla scene/device changes.
  Reusing five assets/texture registrations avoids repeated thumbnail allocation;
  a validity marker prevents displaying previous-scene images after reset.
- TiledCamera does not expose readback_stats. An initial test compile failed on
  that nonexistent method; the test now uses real dispatch/query counters and
  preview-ring transfer/wait counters. No production API was invented to hide it.

### Accomplished

- Added camera_preview.rs with owned SensorScene, TiledCamera, GPU ImageProcessor,
  three-slot camera/processed outputs, three-slot readback, bounded sampled-pose
  metadata and separately identified configuration requests.
- Camera and exposure/Reinhard/linear-to-sRGB processing share one encoded batch.
  Processing preserves RGBA8 quantization/order. Neutral processing uses the raw
  leased color channel directly with no redundant processing copy.
- Preserved 1280 × 720, five thumbnails and all FOV/physical-optics/post controls.
  Orbit now follows simulation time; paused edits refresh at the same timestamp,
  duplicate paused updates do not redispatch, and bounded requests can retry.
- Closing the pane stops new readbacks, not GPU sampling. One latest device image
  is retained; paused reopen collects it without another camera render. Open-pane
  transfer is 32 bytes/pixel across depth/processed-color/normal/albedo/shape ID.
  No additional raw-color readback accompanies processed color.
- Added errors/status and revision/reset handling. Thumbnail assets and texture IDs
  are reused across reset/switch/re-entry, and the displayed eye belongs to the
  collected frame. CPU code still performs thumbnail color conversion and computes
  the finite positive depth range; this is not renderer texture interop or PBR.
- Added two GPU fixtures: small five-channel/optics/photographic parity against the
  previous camera+CPU post path, and full-resolution ECS image/lifecycle acceptance.
  Updated SENSORS.md and the viewer integration chapter without closing S08.

### Verification

Commands used NVIDIA_VERSION=595.84 nix develop --impure -c make with nixVulkan.

- Final make test-viewer-sensors: **10 passed**, zero failures, 11.56 seconds:
  /tmp/molla-viewer-camera-tests-final.log. Three pure tests and seven explicit GPU
  tests executed. All prior IMU/contact/LiDAR/reset fixtures remain passing.
- Camera parity includes aperture sampling and post combinations; RGB tolerance
  is one 8-bit level, depth/normal 1e-5, albedo/shape IDs exact. Checks include one
  retained query-scene build, output allocation counters, exact readback bytes,
  zero preview blocking waits, no duplicate requests and hidden sampling/reopen.
- Full-resolution ECS fixture checks five populated 720p images, five stable assets,
  unchanged handles across reset/re-entry, simulation-time orbit, paused settings,
  pane closure/reopen without resampling, pending reset, invalid-FOV error/retry
  and scene switching. These are headless checks, not rendered visual acceptance.
- make check APP_TARGET='-p molla_viewer --all-targets' passed without compiler
  warnings: /tmp/molla-viewer-camera-check-final.log.
- Scoped make fmt-files --check and git diff --check passed:
  /tmp/molla-viewer-camera-fmt.log. Main's small integration edits were not used
  as a reason to reformat the large legacy file.
- Final make docs DOCS_DIR=/tmp/molla-sensors-viewer-book passed cleanly:
  /tmp/molla-viewer-camera-docs-final.log.
- External engine and compute-doc hashes still match their existing manifests.
  No library source, physics, Gearbox, assets or user windows changed this turn.
  All owned sessions (18917, 1041, 82770, 97239, 50466, 91509) are terminal.
  Session 1041 was the initial compile failure, superseded by passing final checks.
- No timed performance claim, broad ordinary-viewer rerun, or full sensor-library
  matrix rerun is claimed for this turn.

### Next Steps

Migrate 108 Synthetic Annotation: currently it synchronously collects camera shape
indices then calls Dataset::annotate on the host. Use the existing camera-to-pending
AnnotationProcessor chain, selected bounded box readback, revision/reset behavior
and explicit simulation clock for this static host-driven scene. Preserve its labels
and display semantics and add parity/lifecycle tests.

Then implement the deterministic moving-mesh multi-sensor demo and inspect actual
rendering/panes/pause/reset/switch behavior. Keep S00/S01/S02/S04 contract, resource,
allocation, quality-triggered rebuild and full performance gates open. S09 remaining
measurements/camera links and S10 deferred status are unchanged. Goal stays active.

### Relevant Files

- bin/molla_viewer/src/camera_preview.rs — owned GPU camera/post chain and bounded display.
- bin/molla_viewer/src/examples/12_camera.rs — simulation-time configuration, pane gating,
  asynchronous thumbnails, persistent assets and error handling.
- bin/molla_viewer/src/main.rs — module registration and camera reset integration.
- bin/molla_viewer/src/viewer_sensor_tests.rs — small parity and full-resolution ECS fixtures.
- book/src/sensors/viewer.md, SENSORS.md — migrated camera contract and open acceptance.

### Key Learnings

1. A hidden sensor pane can stop host collection while retaining the latest GPU frame.
2. Camera photographic processing can chain from pending outputs without host copies.
3. Thumbnail assets can survive scene resets while their displayed-frame validity resets.

## S08 GPU annotation preview — verified 2026-09-25

### Goal

Continue the complete SENSORS.md objective after the verified camera migration.
The preceding turn was implementation progress. This turn migrates example 108,
completing the seven named code migrations, not S08 or the full objective.

### Instructions

Use Makefile tasks through the native Nix environment and patch edits. Preserve
unrelated WIP, external engine/runtime documentation, physics, Gearbox, assets and
user windows. No agents, commits or GUI actions. Engram tools remain unavailable
in the registry; this project handoff records the work. No memory-folder writes.

### Discoveries

- 108 previously synchronously downloaded a 192 × 192 shape-index image, then
  generated annotation masks/boxes on the CPU. Its viewport cubes were approximate
  world-space body markers, not projected pixel boxes, despite the old description.
- The camera pool allocates all five output-channel capacities even when only shape
  index is active. Selecting shape-only dispatch/readback does not remove those
  allocations; that limitation is now explicit in the viewer documentation.
- Annotation output compacts visible boxes first and zeroes unused active label
  slots. Reading all three fixed label records costs 96 bytes and requires no
  separate count readback; zero-pixel records can be filtered on display.

### Accomplished

- Added annotation_preview.rs: one retained SensorScene, shape-index-only camera,
  AnnotationProcessor with the original three shape/class labels, bounded camera/
  annotation output pools and a three-slot 96-byte-per-frame readback ring.
- Camera output feeds PendingAnnotationJob in the same command batch, without
  camera-image host collection or CPU Dataset processing. Shape/class masks and
  counts stay on GPU; only compacted bounding-box records are collected.
- Preserved camera pose/FOV/range, labels and approximate detection-marker behavior.
  Added a pane with actual inclusive pixel bounds, class/shape IDs, pixel counts,
  sample age/counters and visible errors. Corrected the description/audit text to
  distinguish those pixel boxes from the world-space viewport markers.
- Added sensor-only simulation-clock advancement and explicit update/draw ordering
  for this static scene. It samples at entry/reset even when paused, then only on
  new simulation ticks; paused updates continue polling but do not redispatch.
  Reset/switch clears revision-bound pending results, detections and failures.
- Added exact CPU-reference parity, empty-result, GPU-authoritative pose and lifecycle
  coverage. Marked only S08's seven-example code-migration checklist item complete.
  Combined demo, rendered acceptance, documentation/performance gates remain open.

### Verification

Commands used NVIDIA_VERSION=595.84 nix develop --impure -c make with nixVulkan.

- Final make test-viewer-sensors: **11 passed**, zero failures, 12.08 seconds:
  /tmp/molla-viewer-annotation-tests-final.log. Three pure tests and eight explicit
  GPU tests executed; all preceding camera/LiDAR/IMU/contact/reset fixtures pass.
- New fixture compares every inclusive bound, shape/class ID and pixel count
  exactly with the former all-channel camera plus Dataset::annotate path.
  It verifies three initial detections, moved GPU-authoritative bodies producing
  an empty result, retained scene build, one label upload, two direct camera/
  annotation encodes with zero convenience submissions, exactly 96 bytes per
  display sample, no preview blocking waits, pause/single-step, pending reset,
  missing-shape-channel failure/retry and scene switch/re-entry.
- Final make check APP_TARGET='-p molla_viewer --all-targets' passed without compiler
  warnings: /tmp/molla-viewer-annotation-check-final.log.
- Scoped make fmt-files --check passed: /tmp/molla-viewer-annotation-fmt.log.
  git diff --check passed; main's targeted integration changes were not used to
  reformat unrelated legacy code.
- make docs DOCS_DIR=/tmp/molla-sensors-viewer-book passed cleanly:
  /tmp/molla-viewer-annotation-docs.log.
- External engine and compute-doc hashes still match their existing manifests.
  No library source, physics, Gearbox, assets or user windows changed this turn.
  All owned sessions (97707, 23915, 36765) are terminal; none is live.
- Transfer payload reduction is measured by byte counters, not evidence of a wall-
  time speedup. No rendered acceptance, full library matrix rerun or ordinary
  broad-viewer suite rerun is claimed for this increment.

### Next Steps

Implement the deterministic moving-mesh multi-sensor example required by S08:
shared scene, independent sensor rates/settings, sample age, ray counts,
dropped/deferred samples and separate CPU/GPU timing. Inspect current timing/device
feature APIs before choosing GPU timing; existing panes honestly say unmeasured.
Then inspect actual rendering and interactions for all named examples and the demo.

Keep S00/S01/S02/S04 contracts, allocation/resource checks, quality-triggered rebuild
policy and full benchmark matrix open. S09 remaining integration measurements/
camera links and S10 deferred status are unchanged. Do not mistake all seven
code migrations for completion of these broader gates. Goal stays active.

### Relevant Files

- bin/molla_viewer/src/annotation_preview.rs — device-only camera/annotation chain.
- bin/molla_viewer/src/examples/108_synthetic_annotation.rs — async detections,
  revision/error handling, pixel-bound pane and accurate world-marker semantics.
- bin/molla_viewer/src/main.rs — module/pane registration, system ordering, static
  sensor clock, description/audit updates and removed unused Dataset import.
- bin/molla_viewer/src/viewer_sensor_tests.rs — exact annotation parity/lifecycle fixture.
- book/src/sensors/viewer.md, SENSORS.md — migration completion and open acceptance.

### Key Learnings

1. Compacted box records avoid downloading the shape-index image for annotations.
2. A shape-only camera dispatch still uses full-channel pool capacities currently.
3. World-space detection markers are not the same as projected pixel bounding boxes.

## S08 multi-sensor timing foundation — decision 2026-09-25

### Goal

Build the required combined moving-mesh demo, including honest separate CPU/GPU
timing, without altering the externally authored Engine implementation.

### Decision

Add a viewer-owned, opt-in timestamp-capable device constructor using the existing
Engine::from_device API. Default startup continues using Engine::new_gpu; enable
profiling with MOLLA_SENSOR_TIMING=1 so scene switches retain the profiled device.
Forward this owned device's loss callback into its adopted engine's health signal.
Add a documented auxiliary command hook to SensorCommandBatch for in-batch GPU
timestamp markers and resolves; frame consumers must still use the existing
lease-retaining APIs. Use bounded asynchronous timestamp readback, not queue
completion latency or CPU submission time mislabeled as GPU execution time.

Engram tools are unavailable; this project decision record is the persistence
fallback. The full multi-sensor demo and remaining acceptance gates stay open.

### Timing implementation discoveries

- The first timing fixture reported zero because SensorCommandBatch's empty-frame
  optimization discarded auxiliary-only command buffers. Added a bounded
  auxiliary_encodes counter (128 callbacks maximum) and submit such batches while
  retaining the true empty-batch fast path. A nontrivial compute fixture now reaches
  a strictly positive timestamp assertion; final aggregate verification follows.
- Tiny readback fixtures still require eight-byte-aligned allocation budgets:
  four bytes of requested payload need an eight-byte staging slot. Fixed the
  fixture budgets rather than weakening the ring's budget validation.

## S08 optional GPU timing foundation — verified 2026-09-25

### Goal

Continue the full sensor plan toward the combined moving-mesh demo. The preceding
annotation turn was progress. This increment implements its real GPU timing
prerequisite and integrates it into all seven existing previews; the new combined
scene itself is not implemented or checked complete.

### Instructions

Use Makefile tasks/native Nix and patch edits; preserve external engine/runtime
work, physics, Gearbox, assets and windows. No agents, commits or GUI actions.
Engram tools remain unavailable; this project handoff is the persistence fallback.
Memory-folder contents were not modified.

### Discoveries

- Default Engine devices request no optional wgpu features, so existing devices
  cannot acquire timestamps retroactively on scene switch. Opt-in device creation
  belongs at viewer startup; ordinary startup remains unchanged.
- The new auxiliary-only fixture initially collected zero timestamp words because
  the batch's old empty-frame fast path skipped submission. It was not evidence
  of instantaneous GPU work. Tracking auxiliary callbacks fixes that path while
  preserving true empty-batch no-op behavior.
- Readback staging is eight-byte aligned even for a four-byte requested payload.
  Early scalar fixtures used too-small budgets; these were corrected, with storage
  validation retained. The final timing fixture executes observable compute work
  and verifies both a positive device interval and the computed output.

### Accomplished

- Added SensorCommandBatch::encode_auxiliary for timestamp markers/resolves and
  other non-frame commands, capped at 128 successful callbacks. It checks device
  health/poisoning, records auxiliary_encodes separately from kernels, and poisons
  on callback error/panic. Auxiliary-only batches now submit; empty batches do not.
  Frame consumers still require the existing lease-retaining APIs; this hook does
  not retain frame leases or permit queue writes/submissions in the callback.
- Added sensor_timing.rs. MOLLA_SENSOR_TIMING=1 creates a viewer-owned device with
  TIMESTAMP_QUERY and TIMESTAMP_QUERY_INSIDE_ENCODERS if both are supported, then
  adopts it using Engine::from_device and forwards device loss. Default startup
  uses Engine::new_gpu unchanged. The externally modified engine.rs was untouched.
- Added bounded SensorGpuTimer: two queries, 16-byte resolve/transfer buffers, three
  16-byte asynchronous staging slots. It converts device ticks with timestamp
  period, rejects reversed intervals, filters/invalidate stale revisions, tracks
  timing deferrals and never waits in the preview path.
- Instrumented all seven migrated previews via SensorPreview begin_gpu/end_gpu/
  collect_gpu. Full timing queues omit timing, not the sensor sample. Camera and
  IMU do not enqueue new timing measurements with their panes closed.
- Panes distinguish CPU submission from GPU batch interval, show timed sample
  identity/age and timing deferrals, and retain an explicit unmeasured label when
  optional features are absent. The GPU interval excludes uploads queued before
  the batch, pre-marker queue waiting, query resolve/copy and display readback.
- Added direct pollster 0.4 dependency to the viewer, reusing the already-locked
  package; Cargo.lock only adds it to the viewer dependency list.
- Added auxiliary transaction/poison/empty/capacity/device-loss tests and a real
  timestamp fixture with nontrivial compute output, three-slot saturation, exact
  bytes/allocations, zero timer blocking waits, foreign-engine rejection, revision
  invalidation and default/opt-in device-feature selection.

### Verification

Commands used NVIDIA_VERSION=595.84 nix develop --impure -c make and nixVulkan.

- Dedicated final GPU timing fixture passed:
  /tmp/molla-sensor-timing-fixture-final.log.
- Default make test-viewer-sensors: **12 passed**, 10.20 seconds:
  /tmp/molla-sensor-timing-viewer-final.log.
- MOLLA_SENSOR_TIMING=1 make test-viewer-sensors: **12 passed**, 9.68 seconds:
  /tmp/molla-sensor-timing-profiled-viewer-final.log. These functional suite wall
  times are not a controlled comparison or performance claim.
- Full make test-sensors: **327 passing executions**, zero failures across 98
  result lines: /tmp/molla-sensor-timing-full.log. This counts repeated feature
  lanes/doctests, not 327 distinct GPU tests.
- After adding the explicit callback-panic assertion, the focused final
  make test-gpu P=molla-sensors T=sensor_device_pipeline_gpu A='--test-threads=1'
  passed **3/3**: /tmp/molla-sensor-timing-auxiliary-final.log. Production code was
  unchanged after the full matrix; this strengthens the new fixture's coverage.
- Final viewer/all-target check passed without compiler warnings:
  /tmp/molla-sensor-timing-check-final.log. Scoped formatter check and git diff
  --check passed: /tmp/molla-sensor-timing-fmt.log.
- Book build passed cleanly outside the repo:
  /tmp/molla-sensor-timing-docs.log.
- External engine/runtime-doc hashes still match existing /tmp manifests.
  No physics/Gearbox/assets/windows changed. All owned sessions (23872, 84069,
  56688, 65136, 50079, 30086, 77593, 77544, 74783) are terminal. Early failing
  timing runs are superseded by the verified final fixture and both viewer lanes.
- No rendered pane acceptance, exclusive-device benchmark, broad ordinary viewer
  suite rerun or full-goal completion is claimed.

### Next Steps

Implement the still-missing deterministic moving-mesh multi-sensor scene using
one SensorScene and the existing SensorScheduler for independent rates. Add actual
per-sensor settings, samples/ages/rays/drop counters, previews and the now-available
CPU/GPU timing. Scene enum/ALL/labels/descriptions/build_scene_model/physics audit,
resource/system/pane wiring need explicit integration. Scene labels already reach
at least 140; inspect current numbering before allocating the new example.

Sim.rebuild_scene reuses its Engine. Launch with MOLLA_SENSOR_TIMING=1 before
switching to the combined scene when GPU timings are required; do not replace
the external Engine implementation or create a fresh physics device on switches.
BodyVisual currently has primitive shapes rather than a generic mesh variant;
a matching rendered box with a real triangulated-box collider is one straightforward
moving-mesh fixture, but the final scene must actually use body-attached TriMesh
queries and observable occlusion, not merely rename a primitive demonstration.

Then inspect actual rendering/interactions and pursue the remaining S00/S01/S02/
S04 contract, resource/allocation, quality-triggered rebuild and full benchmark
gates. S09 measurements/camera links and S10 deferred status are unchanged.
The full goal stays active.

### Relevant Files

- crates/molla-sensors/src/device_results.rs — bounded auxiliary encoder hook,
  statistics and auxiliary-only submission.
- crates/molla-sensors/tests/sensor_device_pipeline_gpu.rs — new transaction fixture.
- bin/molla_viewer/src/sensor_timing.rs — optional device factory and GPU timer/test.
- bin/molla_viewer/src/sensor_preview.rs — timing lifecycle and pane metrics.
- bin/molla_viewer/src/lidar_preview.rs, camera_preview.rs, annotation_preview.rs,
  examples/06_imu.rs, examples/07_contact.rs — in-batch marker integration.
- bin/molla_viewer/src/main.rs — viewer factory selection/module wiring.
- bin/molla_viewer/Cargo.toml, Cargo.lock — direct existing pollster dependency.
- book/src/sensors/viewer.md, device_results.md, SENSORS.md — timing/API contracts.

### Key Learnings

1. Auxiliary command buffers must count as real work even without output frames.
2. GPU timing needs device features selected before creating the viewer's engine.
3. Timestamp collection can be bounded independently without dropping sensor work.

## S08 combined sensor lab — design 2026-09-25

Implement scene 144 (current highest label is 143) with a genuinely body-attached
triangle-mesh box moving deterministically in front of a stationary sphere.
LiDAR and camera share one SensorScene; an IMU is mounted at the moving body's
origin. Motion and velocity are prescribed from simulation time, not a contact
solver. Independent SensorScheduler cadences default to 10/5/30 Hz and expose
deadline skips separately from output/readback backpressure. Keep bounded fixed
capacities, interactive rates/scan/FOV controls, asynchronous displays and the
optional per-producer timestamp measurements. This is a calibration/sensing
fixture, not a claim of dynamic-mesh deformation or Featherstone dynamics.

### Headless verification — 2026-09-25

- Implemented scene 144 and `sensor_lab.rs`: shared retained SensorScene, moving
  body-attached TriMesh, independent 10/5/30 Hz LiDAR/camera/IMU schedules,
  interactive rates/scan/FOV settings, three-slot outputs/readbacks and optional
  per-producer timestamp measurements. Camera/IMU retain device output while
  hidden; LiDAR points remain visible and collected. Main viewer scene catalog,
  physics audit, pane, update ordering and sensor-only simulation clock include it.
- Added a comprehensive headless fixture. Initial failure exposed an incorrect
  test oracle of 9.81 m/s² instead of the model's 9.80665; the first correction
  incorrectly subtracted the gravity Buffer itself and failed compilation. The
  final oracle reads `model.gravity.host().unwrap()[0]`. No implementation
  tolerance was widened. Final fixture passed, including the previously untested
  camera parity, hidden collection, paused reconfiguration, saturation, pending
  reset/error recovery and scene switch branches.
- `make test-viewer-sensors`: 13/13 passed in default mode and 13/13 with
  MOLLA_SENSOR_TIMING=1. Logs: /tmp/molla-sensor-lab-viewer.log and
  /tmp/molla-sensor-lab-profiled.log. Scene model/description/audit catalog checks
  passed 3/3: /tmp/molla-sensor-lab-catalog.log. Targeted fixture passed 1/1:
  /tmp/molla-sensor-lab-test-3.log. Check, scoped formatting, diff whitespace and
  book build passed: /tmp/molla-sensor-lab-{check-final,fmt,docs}.log.
- Updated SENSORS.md's combined-example implementation checkbox and viewer book
  contracts. Visual acceptance and full performance gates remain open. No sensor
  library production code changed in this scene implementation; the previous
  full library matrix remains separate evidence, not a new run this turn.
- External engine/runtime-doc hashes still match. No physics/Gearbox/assets or
  user windows changed. Engram tools remain unavailable in this tool session.
- For visual inspection, started an owned isolated Xvfb display :91 (session
  99701) and a viewer `make run` build (29448), with a disposable XDG_CONFIG_HOME
  /tmp/molla-sensor-visual-config. These are not user preview windows. The first
  launch exited before building because overriding XDG_CONFIG_HOME outside Nix
  hid Nix's feature config; the corrected command applies it inside nix develop.
  No global Nix/config change was made. xdotool/ImageMagick are available in the
  pinned flake's Nix packages; tools log: /tmp/molla-sensor-visual-tools.log.

### Key Learnings

1. Analytic IMU references must use the model's gravity, not rounded 9.81.
2. Reconfiguring at a previously sampled tick must defer the IMU to avoid dt=0.
3. Isolated viewer config must be applied inside Nix to retain its feature setup.

### Rendered inspection findings — 2026-09-25

The NVIDIA/Xvfb viewer exited with a segmentation fault during default Pile startup
before sensor scene selection (/tmp/molla-sensor-visual.log, session 29448 terminal).
No crash root cause is claimed. Software Vulkan (Mesa llvmpipe 26.1.3, forced ICD
inside the existing nixVulkan wrapper) renders on the same isolated display. This
is visual inspection only, not NVIDIA runtime/performance acceptance.

Inspected /tmp/molla-sensor-start.png and /tmp/molla-sensor-lab-render-1.png after
selecting scene 144 through the actual menu. Mesh/sphere/LiDAR gizmos and timing
metrics render, but the right pane exceeded the 1000-pixel window and hid the
camera/IMU. The left pane also retained Pile physics stats despite no solver
running in scene 144. Added a viewport-bounded scroll area for the eight migrated
sensor panes and suppressed stale physics/contact/XPBD labels in the four
sensor-only scenes. Other scene panes/physics remain untouched. A pure egui
layout regression checks long content at 600/1000-pixel viewport heights.
The test viewer remains on owned display :91, not on the user's desktop.

## S08 combined demo and first rendered inspection — session summary 2026-09-25

### Goal

Continue the full SENSORS.md objective. Implement and validate the combined moving
mesh demo, then inspect actual rendered output rather than equating headless
checks with visual acceptance. The preceding status-only turn made no progress;
this turn revalidated the compile failure and took the available corrective action.

### Instructions

Use Makefile tasks through the native Nix environment; preserve external WIP and
all user windows. No physics/Gearbox/assets changes, commit or push. The full goal
remains active; no performance speedup, zero-allocation or full completion claim.

### Discoveries

- The new analytic IMU reference must read the model's actual gravity buffer.
- Actual rendering exposed pane overflow and stale previous-scene physics stats,
  neither covered by the existing headless producer/lifecycle tests.
- Viewport-bounded ScrollArea makes camera/IMU reachable without changing the
  external Frost pane library. Scene-local scroll IDs prevent cross-scene offsets.
- NVIDIA/Xvfb startup segfaulted before selecting a sensor scene; llvmpipe/Xvfb
  rendered successfully. This comparison does not prove the crash's root cause
  or NVIDIA presentation correctness. A repeated paused viewport capture also
  differs at pixel level despite unchanged visible pose; no bit-identical
  full-renderer assertion is made (headless sensor outputs remain tested).
- The existing full-width Play button leaves the adjacent Step control invisible
  in the inspected layout. Single-step is covered headlessly but its actual UI
  accessibility remains to inspect/fix; no visual single-step acceptance claimed.

### Accomplished

- Scene 144 combined demo and all headless lifecycle assertions pass. Sensor
  library code was not changed in this turn. SENSORS.md/book describe exact
  motion, scheduling, pool/transfer budgets, priming and deferred request behavior.
- Added bounded scrolling for the eight sensor panes, a pure egui 600/1000-height
  layout regression, and a shared `Scene::sensor_only` predicate for the static
  step branch and truthful main-pane telemetry. Non-sensor panes are unchanged.
- Final default/profiled `make test-viewer-sensors` both pass **14/14**:
  /tmp/molla-sensor-lab-viewer-final.log and
  /tmp/molla-sensor-lab-profiled-final.log. Final all-target viewer check passed:
  /tmp/molla-sensor-lab-check-final.log. Scene catalog checks earlier passed 3/3.
- Visually inspected scene 144 with Mesa llvmpipe 26.1.3 Vulkan, an owned Xvfb :91
  and disposable /tmp/molla-sensor-visual-config. Actual menu selection rendered
  moving mesh/sphere occlusion, colored hit points, camera and IMU data/timing.
  Verified reachable bottom controls after scrolling, pause, keyboard R reset,
  paused column/FOV edits changing points/image, hide/reopen and 1000×650 layout.
  Screenshots inspected: /tmp/molla-sensor-lab-scroll-{top,bottom}.png,
  /tmp/molla-sensor-lab-paused-camera{,-2}.png,
  /tmp/molla-sensor-lab-reset-camera.png,
  /tmp/molla-sensor-lab-edited-settings.png,
  /tmp/molla-sensor-lab-camera-zoom.png,
  /tmp/molla-sensor-lab-hidden.png and /tmp/molla-sensor-lab-small.png.
- The corrected binary's only later source change before the final headless lane
  was indentation of the telemetry branch. No user desktop windows were opened,
  interacted with or closed. Owned test viewers exited normally via Escape
  (81108, 9546); owned Xvfb 99701 was terminated after viewer exit. All owned
  build/test/action sessions are terminal, including 25231. External engine and
  runtime-doc hashes still match their prior manifests. No background task remains.

### Next Steps

1. Inspect all seven migrated examples' rendering and settings/reset/pause/switch
   interactions; finish scene 144 rate/step/switch interaction acceptance and
   check hardware presentation independently of the failed NVIDIA/Xvfb setup.
   Repair the actual Step-button accessibility without changing physics behavior.
2. Complete the remaining S00/S01/S02 contracts, resource/allocation audits and S04
   quality-triggered hierarchy rebuild requirements; do not substitute counters
   or this demo for the stated acceptance gates.
3. Finish documentation audit and full reproducible benchmark matrix. Software
   renderer FPS and displayed single-sample GPU intervals are not speedup evidence.
   S09 remaining measurements/camera links and S10 deferral are unchanged.

### Relevant Files

- bin/molla_viewer/src/sensor_lab.rs — three scheduled producers sharing one scene.
- bin/molla_viewer/src/examples/144_sensor_lab.rs — model/motion, UI and camera asset.
- bin/molla_viewer/src/viewer_sensor_tests.rs — combined GPU lifecycle/analytic test.
- bin/molla_viewer/src/sensor_preview.rs — bounded pane scroll and layout test.
- bin/molla_viewer/src/main.rs — scene wiring, sensor-only predicate and telemetry.
- bin/molla_viewer/src/examples/12_camera.rs — reusable sized image allocation.
- SENSORS.md, book/src/sensors/viewer.md — implementation/acceptance status.

### Key Learnings

1. Headless sensor correctness does not reveal inaccessible rendered controls.
2. Sensor-only demos must not show stale physics timings as current work.
3. Software Vulkan can provide isolated visual evidence, not hardware acceptance.

## S08 playback control repair — 2026-09-25

The prior goal turn was progress (combined demo, new layout tests and actual
rendered findings). Revalidated the live main-pane code: Frost's button consumes
the available width, so placing Play/Step/Reset in an unconstrained horizontal
row hides the later actions. Replaced that row with three equal-width columns in
`viewer_controls.rs`, preserving the existing handlers and making Step visibly
disabled while running. No solver or sensor sampling behavior changed.

The new pure egui test covers 240/320-pixel panels, both pause states, nonoverlap,
in-bounds button rectangles, individual click delivery and disabled Step clicks.
`make test ... sensor_playback_controls` passes 1/1:
/tmp/molla-sensor-controls-test.log. Engram memory tools remain unavailable; this
project handoff records the fix. Full viewer regression and rendered acceptance
are the next validation steps, not inferred from this layout test.

Rendered playback checks confirm Step and Reset are reachable. Scene 05 advances
to time 0.017 and submitted/collected 2/2 after one Step; scene 06 advances from
priming 1/0 to a measured 2/1 at 0.017. Scene 07 reset clears the contact readout
until a post-reset step. Isolated software captures live at
/tmp/molla-acceptance-{lidar-step-final,imu-priming,imu-step,contact-reset}.png.

The same inspection found stale ordinary-physics EMA/contact counters after a
paused reset/switch. Added `SimStats::begin_revision` before the paused stepper
return; only a changed sensor-clock revision clears counters, not ordinary pause.
No solver work or inputs changed. The pure revision regression and full viewer
lane now pass 16/16 with timing off and on:
/tmp/molla-sensor-controls-{viewer,profiled}-final.log. Final build/render recheck
is in progress. The repeated NVIDIA/Xvfb startup with WGPU_BACKEND=vulkan also
segfaulted before scene selection; software Vulkan remains the inspection backend.

### Sensor-pane resize hit region

Actual checkbox clicks near the camera pane's left edge were intercepted by
Frost's resize strip (visibly highlighted during the attempted click); clicking
the label worked. Source confirms a 10-pixel interior resize hit region. Added
12-pixel content padding inside the sensor scroll area, without modifying Frost
or nonsensor panes. A pure egui regression renders the real resizable Pane2 and
clicks the checkbox's box (four pixels from its left), not its distant label.
It passes 1/1: /tmp/molla-sensor-resize-checkbox.log. Full regression and rendered
checkbox recheck follow; this is a UI hit-region fix, not a sensor algorithm change.

## S08 rendered acceptance continuation — 2026-09-25

The intervening user-status turn was no progress (status only). Revalidated the
running viewer on owned Xvfb :91, the final 17/17 default/profiled logs and current
source before continuing. Engram tools remain unavailable. This project record
is the handoff; no persistent memory folder was edited.

The final-binary captures confirm ordinary counters clear after paused reset
(`/tmp/molla-acceptance-final-reset-counters.png`), checkbox-square clicks now
toggle camera sRGB and physical optics
(`/tmp/molla-acceptance-final-camera-{gamma-state,checkbox-physical}.png`), and
scene 144 Step advances time without sampling before each sensor's deadline.
Rendered rate edits set LiDAR/camera/IMU to 52/30/6 Hz independently; play/pause
changed motion/readings and reset restored initial geometry. Evidence:
`/tmp/molla-acceptance-final-lab-{lidar-rate,camera-rate-edited,imu-rate-edited,independent-rates,reset}.png`.
The earlier `final-lab-camera-rate.png` did not actually change its value; use
`final-lab-camera-rate-edited.png` as the successful interaction evidence.

Updated `book/src/sensors/viewer.md` and S08 with the basic inspections of all
seven migrated examples, explicit remaining gates and scene 105's first-step
sampling rule. Final all-target viewer check, scoped formatter check and mdBook
build passed: `/tmp/molla-sensor-ui-{check,fmt,docs}-complete.log`.
Software viewer session 98637 exited 0 via Escape on the owned display; user
desktop windows were not touched. External engine/runtime-doc hashes still match.

### Contact glyph repair

Source and a settled rendered scene confirmed the contact preview only drew
short depth-tested line segments; its pane called them arrows and they were
hidden inside bodies. Added a dedicated ContactNormalGizmos group with depth bias
-1 (overlay), actual arrowheads, and explicit pane wording that these geometry
markers show through bodies and are not force measurements. Normal direction,
centroid, depth scaling and sampled data are unchanged; no contact/solver code
changed. Removed unrelated stale camera/diff-arm comments from the contact file.
Full viewer regression and a rendered check of this repair are pending below.

## Session handoff — sensor UI acceptance and contact overlays, 2026-09-25

### Goal

Implement the full SENSORS.md objective. Goal remains active; this session advances
S08 and does not complete S00–S08 or substitute viewer success for performance.

### Instructions

- Use repository Makefile tasks through the Nix environment, apply_patch edits,
  no Python, no commits/push, no physics/Gearbox/asset changes.
- Preserve unrelated WIP and user preview windows. All visual interaction here
  used owned Xvfb :91 and disposable config, not the user's desktop.

### Discoveries

- Scene 105 intentionally has no scan at paused tick zero after entry/reset;
  Play/Step is required. Scene 106 is one-shot at entry/reset, while 108 samples
  at entry/reset and on subsequent ticks. The book now distinguishes these.
- Checkbox-square hits work after the 12-pixel sensor-pane inset. A labelled
  checkbox-only test would have missed the original resize-strip interception.
- Short contact normal lines were occluded inside bodies. A dedicated overlay
  group exposes them without changing ordinary gizmo depth behavior or readings.
- Scene 144 Step is not synonymous with all sensors sampling: each independent
  simulation-time deadline still applies, and IMU history must be primed.

### Accomplished

- Rechecked final-binary reset counters, real checkbox-square clicks, scene 144
  step/independent rate edits/play/pause/reset and switching into camera/contact.
  The earlier double-click numeric edit failed; slider clicks successfully set
  52/30/6 Hz. Only the successful captures are used as rate-edit evidence.
- Repaired the contact glyphs with actual arrowheads and a scene-specific
  depth-bias overlay group. The pane explains through-body geometry markers and
  explicitly disclaims force measurement. No sensor/physics library changes.
- Inspected `/tmp/molla-acceptance-overlay-contact-{settled,near,closer}.png`:
  three yellow normals now visible at default/closer scales. Inspected
  `overlay-contact-reset.png` and `overlay-contact-first-step.png`: reset removes
  prior markers/readings and clears physics counters; first step reports three
  sampled bodies, zero contacts, time 0.017 and no old arrows.
- Final default/profiled viewer sensor tests pass 17/17 each:
  `/tmp/molla-contact-overlay-{tests,profiled}.log`. Final build, all-target viewer
  check, scoped rustfmt check and mdBook build passed:
  `/tmp/molla-contact-overlay-{build,check,fmt,docs}.log`. `git diff --check` passes.
  The production sensor-library matrix was not rerun (library code unchanged).
- Updated S08 and the book's per-example inspection table, controls, sampling
  rules, contact overlay semantics and remaining acceptance. Overall S08 visual
  gate remains unchecked: full settings/layout and hardware checks are not proven.
- Software viewer 98637 exited 0; new overlay viewer 68328 exited 0 via Escape.
  Owned Xvfb PID 3651395/session 94618 was then terminated (143 as expected).
  Sessions 8201/88918/43689 and all UI action sessions are terminal. Process check
  finds no owned viewer/Xvfb remaining. No user windows were touched.
- External engine/runtime-doc hash manifests still match. Unrelated WIP,
  including examples/19_diff_arm6.rs, remains preserved. No commits were made.

### Next Steps

1. Return to substantive remaining core gates: S00 capability/tolerance/baseline
   audit, S01 capacity/transient-allocation audit, S02 whole producer lifecycle
   and resource/allocation verification, and S04 quality-triggered rebuilding.
2. Complete reproducible benchmarks including regressions, construction/refit,
   upload/readback and warm fixed-capacity allocations. No speedup or zero-host-
   allocation claim follows from these viewer checks or GPU buffer counters.
3. Finish remaining S08 settings/layout and hardware presentation checks and the
   broader documentation audit. NVIDIA/Xvfb startup crash remains unclassified;
   software Vulkan acceptance is not hardware presentation/performance evidence.
   S09 transfer/camera work and S10 deferrals remain separately labelled.

### Relevant Files

- bin/molla_viewer/src/examples/07_contact.rs — normal-overlay group/arrows and
  accurate pane wording; removed stale unrelated comments.
- bin/molla_viewer/src/main.rs — registers only the contact overlay depth bias;
  prior playback/revision-counter fixes retained.
- bin/molla_viewer/src/viewer_controls.rs — prior equal-column playback controls
  and click/revision-counter regressions, unchanged this continuation.
- bin/molla_viewer/src/sensor_preview.rs — prior inset/scroll/regressions retained.
- book/src/sensors/viewer.md — inspected behavior, lifecycle and remaining gates.
- SENSORS.md — precise S08 status without closing incomplete acceptance.
- SENSORS_PROGRESS.md — evidence and full-scope next steps.

### Key Learnings

1. Contact centroids and signed normal sums can produce glyphs inside a body;
   explicitly labelled overlays reveal the measurement without changing it.
2. Rendered controls require actual pointer-hit testing, not only value setters.
3. Software-rendered acceptance, headless numerical tests and hardware benchmarks
   are separate evidence and must remain separately reported.

## S04 quality-triggered rebuilding — 2026-09-25

Previous goal turn made progress: contact overlay repair, rendered control checks,
regression runs and reconciled S08 evidence. Revalidated source and policy state;
the query hierarchy previously only supported a CPU-counted periodic cadence.
Engram tools remain unavailable; decisions are recorded here, not in memory files.

Added an optional `quality_ratio_percent` policy (None preserves periodic behavior).
At each eligible query, GPU reductions measure summed internal-node area normalized
by root area; the threshold compares with the best observed cost since rebuilding.
The first submitted check initializes the tree/baseline. Subsequent deterioration
above the threshold enables Morton/radix/reorder/commit work using GPU-written
indirect arguments; skipped checks cannot copy stale alternate instances. Baseline
updates after a final refit, including improvement-only updates on skipped rebuilds.
All buffers/bindings are retained and budgeted. No query-path CPU quality readback,
wait, host sort or per-check resource construction was introduced.

Quality mode remains opt-in until performance evidence supports a default change.
The current implementation still encodes indirect passes and a second refit on
every quality check, even when no rebuild runs. This is a cost heuristic, not a
guarantee that Morton reordering improves every distribution. Encoded quality
checks are counted separately from unconditional encoded rebuilds; actual quality
decisions remain GPU diagnostic state. Cancelled encoding is not GPU execution.

Initial 513-leaf GPU test passed: deterministic independent area oracle, skipped
stable/translated poses, scrambled topology repair, coincident geometry, shape-ID
permutation and bounds coverage, cancelled command buffer and GPU-only pose input.
The kernel's GPU allocation payload matches the estimator. Initial lifecycle/
budget/independent-camera integration tests passed 3/3; mixed mesh/primitive quality
parity was then added and awaits the final full run. Logs:
`/tmp/molla-scene-quality-{gpu,integration,check}.log`.

Final full `make test-sensors` completed with **329 passing test executions** and
zero failures (`/tmp/molla-scene-quality-full.log`); optional ignored GPU tests
were explicitly exercised by the last lane. A subsequent test-only extension
checks recovery from coincident geometry: improved bounds lower the baseline,
then a newly scrambled pose layout triggers another repair. That focused GPU
test also passes (`/tmp/molla-scene-quality-recovery.log`). The default and
timestamp-enabled viewer lanes pass 17/17 each
(`/tmp/molla-scene-quality-viewer{,-profiled}.log`). No GUI was launched this turn.

The conditional instance commit uses its own bind group: an indirect argument
buffer must not also be bound as writable storage in the same dispatch. Quality
decision and capture passes are separate from indirect sort/commit passes. The
actual GPU diagnostic counter verifies skipped and executed rebuilds, rather than
inferring decisions from host encode counts. `for_quality_leaves` is checked
against sums of allocated buffer sizes, not just a repeated estimator formula.

Added same-fixture benchmark variants `quality_64` and `quality_every_query` next
to fixed, periodic-64 and every-query reordering. Focused measurements and final
source/doc checks are in progress; no speedup conclusion has been drawn.

## Session handoff — GPU quality policy, 2026-09-25

### Goal

Implement full SENSORS.md. S04 now has optional GPU quality decisions, while the
complete S00/S01/S02/S04/S08 audits and performance gates remain open. Goal active.

### Instructions

- Makefile tasks through Nix; apply_patch edits only. No commits/push requested.
- No soil/tyre/physics solver/Gearbox/asset changes; preserve unrelated WIP.
- Keep encoded counters, actual GPU work, numerical correctness and measured
  performance distinct. No zero-allocation or general speedup claim.

### Discoveries

- Reusing old scatter output after a skipped conditional reorder would overwrite
  live instances. The quality path instead gates a compute commit with the same
  GPU decision as the sort; periodic mode retains its original device copy.
- Indirect arguments cannot be writable storage in that same dispatch's bind
  group. Quality metric/decision and conditional commit use separate bindings.
- A baseline left at a costly overlapping layout would hide later degradation.
  Cheaper refits lower the baseline; the recovery fixture verifies another
  scramble triggers rebuilding afterward.
- Skipping radix work does not remove indirect-command, quality-reduction or
  second-refit overhead. Checking every query is not automatically efficient.

### Accomplished

- Added `SceneRebuildPolicy::quality_ratio_percent`, default None, validated >100;
  `SceneRebuildStorage::for_quality_leaves` covers all fixed added GPU buffers.
- Added GPU normalized internal-area reductions, best-since-rebuild baseline,
  indirect dispatch selection, safe conditional instance commit and post-refit
  baseline capture. Existing periodic behavior/default remains unchanged.
- Added distinct `RayQueryStats::scene_quality_checks`; actual conditional counts
  stay in the GPU diagnostic state. Query encoding introduces no quality readback,
  CPU decision wait or per-check buffer/bind-group construction.
- The 513-leaf fixture checks actual GPU check/rebuild counts, independent CPU
  area reference (relative tolerance 1e-5), stable/translated skips, scrambled
  repair, coincident bounds, recovery, cancellation, leaf identity and bounds.
  Expanded integration cases run both periodic and quality policies for primitive/
  mesh ties, cancelled batches, alternate GPU-only poses, retained leases,
  backpressure, budget failures and independent heterogeneous camera groups.
- `make test-sensors`: 329 passing executions, zero failures. Then expanded the
  recovery test (test-only change) and reran it successfully. Viewer sensor tests
  pass 17/17 both normally and with MOLLA_SENSOR_TIMING=1. Exact logs are listed
  above; only later production changes were API doc-comment wording.
- Final sensor all-target check, viewer all-target check, scoped formatter check,
  mdBook build and whitespace check passed:
  `/tmp/molla-scene-quality-{check-final,viewer-check,fmt-final,docs-final}.log`.
  Rust version verified: 1.94.1 (e408947bf 2026-03-25).
- Ran the focused GPU-pose benchmark twice on RTX 4080/NVIDIA 595.84/Vulkan,
  optimized profile, HEAD 367d92c plus worktree. All ten cases complete in each
  run; full results and intervals are in book/src/sensors/scene.md. Logs:
  `/tmp/molla-scene-quality-bench{,-repeat}.log`, with matching per-process GPU
  monitor logs. Existing desktop/remote-display processes were not stopped; the
  repeat sampled two 3% SM readings from gnome-remote-de, so exclusive use is not
  claimed. First run's ancillary source check finished during benchmark build;
  the repeat ran without another build/check job.
- Quality-64 at 10k instances was 0.40075 then 0.46467 ms; periodic-64 was 0.45472
  then 0.40760 ms, with overlapping intervals. Quality-every-query rose from
  0.59984 to 1.0110 ms without a code change. These are scoped overhead/variability
  observations, not a reproducible general win. Default stays periodic.
- All owned test/check/benchmark sessions are terminal. Owned pmon PIDs 3722146
  and 3725075 were terminated after the respective runs; session 39580 exited 0.
  Final process inspection finds no benchmark, pmon, viewer or owned Xvfb left.
  No GUI was launched. External engine/runtime-doc hashes match prior manifests.
  No commits, pushes or memory-folder writes occurred; Engram tools unavailable.

### Next Steps

1. Complete S00 geometry/coordinate/miss/tolerance contract inventory and S01
   constructor/transient-allocation audit; use current source/tests, not old flags.
2. Finish S02 lifecycle/resource/allocation scope; 329 tests do not establish
   warmed zero-allocation behavior or finish every producer's acceptance.
3. Finish S04 transform/reference audit and full performance matrix, especially
   continuously changing layouts, construction/readback, overlapping geometry,
   realistic mesh complexity and varied ray counts. Optimize measured overhead
   rather than enabling the new quality policy by default without evidence.
4. Complete S08 documentation/settings/layout/hardware presentation acceptance.
   The NVIDIA/Xvfb startup issue is still unclassified. S09 remaining integration
   and S10 explicitly deferred capabilities retain their separate status.

### Relevant Files

- crates/molla-kernels/shaders/scene_quality.wgsl — GPU cost/decision/baseline.
- crates/molla-kernels/shaders/scene_rebuild_commit.wgsl — conditional instance copy.
- crates/molla-kernels/src/scene_quality.rs — retained quality resources/pipelines.
- crates/molla-kernels/src/scene_rebuild.rs — periodic/indirect paths and diagnostics.
- crates/molla-sensors/src/scene_rebuild.rs — policy validation/storage estimates.
- crates/molla-sensors/src/scene_trace.rs — ordered refit/check/reorder/refit/capture.
- crates/molla-sensors/src/scene_acceleration.rs — kernel preparation and GPU oracle.
- crates/molla-sensors/src/raycast.rs — separate maintenance counters.
- crates/molla-sensors/tests/sensor_scene_rebuild_gpu.rs — dual-policy regressions.
- crates/molla-sensors/benches/bench_scene_queries.rs — quality policy controls.
- book/src/sensors/{scene,raycaster}.md — contracts, budgets, limitations and data.
- SENSORS.md, SENSORS_PROGRESS.md — accurate remaining gates and evidence.

### Key Learnings

1. GPU-gated maintenance needs a gated commit as well as a gated computation.
2. A quality baseline should recover after easier layouts, not stay permanently
   inflated by a previously overlapping scene.
3. Quality decisions without CPU readback are feasible, but their command/refit
   overhead must be measured; fewer sorts alone do not prove faster queries.

## S01 camera fixed-array budget and mount validation — 2026-09-25

The previous goal turn made progress (GPU quality policy, regressions, repeated
benchmarks). Current-source capacity audit found that the camera's 1 GiB ray-array
budget excluded five retained image channels, camera poses/packing and lazy
aperture scratch. Heterogeneous groups had pixel caps but no summed fixed-array
byte cap; a later group could fail after earlier groups allocated GPU resources.

Added `TiledCameraLimits`/`CameraStorageUsage` with a default combined 1 GiB
host/device fixed-array cap and separately enforced ray sub-budget. The estimate
always reserves aperture sums/uniform even if currently inactive, so enabling
optics or channels cannot bypass the constructor budget. New explicit constructors
allow overrides without changing existing method signatures. `CameraBatch` checks
the sum across all planned groups before constructing any group's GPU arrays, and
reconfiguration preserves the custom cap transactionally. These limits explicitly
exclude driver/upload staging, bindings/parameters, scene/query resources, output
leases/readback pools and caller-owned data; they are not a process peak-memory cap.

The same audit found IMU mount poses were not validated, unlike camera/frame
mounts. Added finite/f32-representable position and unit-quaternion validation
before configuration buffers. Frame mounts now validate via a chained iterator,
eliminating the temporary concatenated target array before validation. New GPU
tests cover invalid/valid mounts, exact/under camera budgets, optics/channel changes
and heterogeneous transactional replacement. The first test compile used a
nonexistent look-at helper; corrected it to the existing set_camera_look_at method.
The camera-only full regression finished successfully: 335 test executions in
98 successful suites (`/tmp/molla-camera-storage-full.log`). The later auxiliary
budget regression below includes these changes. Engram tools remain unavailable;
no memory files edited.

## S01 auxiliary array budgets and camera documentation — 2026-09-25

### Goal

Continue the full SENSORS.md implementation, closing constructor-capacity gaps
without treating individual passing suites as completion of S01 or the plan.
The preceding status-only turn made no implementation progress; this turn
rechecked its 335-execution log and advanced the current-source capacity audit.

### Instructions

- Use repository Makefile tasks through `NVIDIA_VERSION=595.84 nix develop --impure`.
- Preserve unrelated engine/runtime/example changes, Gearbox and assets.
- No GUI windows, commits, pushes or memory-folder writes requested or performed.

### Discoveries

- IMU/contact/frame constructors had per-buffer/dispatch checks but no combined
  retained host/device array budget. IMU history scales with all model bodies,
  not mount count; frame reference slots consume full configuration/output arrays.
- Empty contact buffers allocate neither host payload nor device mirrors; tests
  measure zero rather than inventing GPU padding allocations.
- The new test initially used u32 for wgpu 29's storage-binding limit, which is
  u64. Corrected that test type before the successful full regression.
- Batched LiDAR still budgets shared ray arrays separately from its three
  per-pulse buffers and descriptor/range arrays; this is a concrete remaining
  combined-budget audit item, not a completed whole-sensor cap.
- Image, annotation, cloud and grid processing each have existing GPU byte-budget
  helpers; their host/transient scope still needs a complete audit. Merely locating
  those helpers does not establish full constructor acceptance.

### Accomplished

- Added `SensorArrayLimits` (default 1 GiB per sensor) and pure
  `SensorArrayStorage::{for_imu,for_contact,for_frames}` estimators. Constructors
  check totals and GPU limits before allocation; `new_with_limits` exposes
  overrides without changing existing `new` signatures. `storage_usage()` reports
  host/device/total/largest-device-buffer payloads.
- Count validation bounds the closed set of array layouts to u32 before byte
  arithmetic. Frame input-body pose capacity remains device-checked but excluded
  from owned payload. Invalid/empty mount/target behavior remains explicit.
- Added two pure boundary/accounting tests, three GPU tests measuring actual
  retained arrays (including empty contacts and both reference modes), and an
  integration test for exact/under-budget constructors and empty zero-cap contacts.
  Updated the public facade type-identity test.
- Documented combined camera budgets, reserved aperture storage, separate ray
  sub-limits and heterogeneous group sums. Documented auxiliary sensor estimates,
  overrides and exclusions. Corrected IMU gravity prose to match shader subtraction.
- `make test-sensors` passed **341 executions across 98 successful suites**,
  terminal exit 0; `/tmp/molla-aux-storage-full.log`. This includes CPU-only/facade
  lanes and explicitly executed ignored GPU tests, not just host discovery.
- Scoped formatting and mdBook build passed (exit 0):
  `/tmp/molla-aux-storage-fmt.log`, `/tmp/molla-aux-storage-docs.log`;
  book output `/tmp/molla-aux-storage-book`.
- Viewer headless sensor regressions passed **17/17** with timing disabled and
  **17/17** with `MOLLA_SENSOR_TIMING=1`: `/tmp/molla-aux-storage-viewer.log` and
  `/tmp/molla-aux-storage-viewer-profiled.log`. This is not visual/hardware
  presentation acceptance or a performance measurement.
- Sensor GPU-feature and viewer all-target checks passed:
  `/tmp/molla-aux-storage-check.log`, `/tmp/molla-aux-storage-viewer-check.log`.
  Session 71217 completed with exit 0; all test/build sessions from this turn
  are terminal. No GUI or benchmark processes were launched.
- `git diff --check` passed; engine/runtime SHA-256 manifests still match their
  preserved external edits. Engram discovery returned no memory tools; this
  structured handoff is in the project progress file, not an Engram save.

### Next Steps

1. Finish remaining S01 constructor/legacy/transient audit, starting with the
   batched-LiDAR pulse/descriptor arrays and CPU scan metadata expansion paths.
2. Complete S00 geometry/frame/miss/tolerance inventory and reproducible baseline
   coverage; S02 allocation/lifecycle evidence still does not prove zero allocations.
3. Finish S04 transform/reference and realistic moving-scene performance gates;
   S08 full settings/layout/hardware presentation and benchmark acceptance remain.
4. S09 remaining camera/transfer integration stays separate; S10 remains deferred.

### Relevant Files

- crates/molla-sensors/src/sensor_storage.rs — auxiliary estimators and budgets.
- crates/molla-sensors/src/{imu,contact,frame_transform}.rs — preflight and payload tests.
- crates/molla-sensors/tests/sensor_capacity_gpu.rs — exact/under-budget regressions.
- crates/molla-sensors/src/lib.rs, crates/molla/src/sensors/mod.rs — GPU-gated exports.
- crates/molla/tests/sensor_api.rs — facade type identity.
- book/src/sensors/{imu,contact_sensor,frame_transform,tiled_camera}.md — contracts.
- SENSORS.md, SENSORS_PROGRESS.md — scope, remaining acceptance and evidence.

### Key Learnings

1. Mount count alone cannot bound IMU storage: velocity history covers all bodies.
2. Array-payload budgets exclude process peaks, driver staging and separately
   leased output/readback resources; their API/docs must preserve that distinction.

## S01 batched LiDAR storage and legacy layout validation — 2026-09-25

### Goal

Continue the full SENSORS.md objective. The preceding auxiliary-budget turn made
verified progress (341 sensor executions plus viewer regressions). This turn
closed the identified batched-LiDAR combined-budget gap and audited CPU expansion.

### Instructions

- All build/test/format/doc commands use repository Makefile tasks through Nix.
- Preserve external engine/runtime changes, Gearbox/assets and user windows.
- No commits, pushes, GUI launches or memory-folder writes.

### Discoveries

- Batched LiDAR allocates three return arrays at ray capacity, not actual pulse
  count, plus device descriptors and reserved host descriptor/range capacity.
  Budgeting only shared ray arrays omitted these retained allocations.
- Legacy terrestrial layout used a saturating ray-count getter; aerial layout
  multiplied pulse/sub-ray counts locally. Public casts had validated counts
  first, but the private expansion paths lacked independent byte caps and used
  infallible Vec reservations. Both paths now consume checked counts and reserve
  fallibly after capacity/byte preflight.
- `ScanMetadata::validate` accepted non-finite origins. A new host regression
  reproduced this before the fix. It also accepted individually finite angular
  endpoints with an overflowing difference, which feeds non-finite interpolation.
  Validation now rejects both cases before layout expansion.
- The first new legacy GPU fixture omitted required raw-model/state uploads and
  failed the existing dirty-state guard. Fixed the fixture uploads, not the guard
  or production ownership contract; focused capacity tests then passed 6/6.

### Accomplished

- Added `LidarBatchLimits` (default 1 GiB combined payload, independent ray
  sub-budget) and `LidarBatchStorage::for_capacity`. `new_with_limits` preflights
  all retained arrays; `new_with_ray_limits` keeps its signature and overrides
  only the ray sub-budget. `storage_usage()` reports the fixed estimate.
- Added `LidarLayoutLimits` (default 1 GiB per-call CPU layout), metadata
  `layout_storage_bytes()` estimates, and legacy `cast_with_layout_limits` methods.
  Terrestrial payload includes two per-ray vectors; aerial also includes two
  per-pulse vectors returned with the scan. Ray capacities remain independent.
- Reservations return allocator errors through Result. Layout caps exclude
  collected hits, staging, retained rays and scenes; batch caps exclude scene,
  uniforms/bindings, driver staging, per-result metadata copies and output pools.
  Neither promises recovery from OS memory exhaustion or caps process peak RAM.
- Added pure aggregate/exact/under/device-limit/extreme-count tests, actual GPU
  retained-payload accounting, exact-capacity batch reuse after invalid requests,
  CPU layout payload/order checks, and legacy GPU equivalence fixtures.
- Metadata validation red/green evidence:
  `/tmp/molla-lidar-metadata-red.log` (expected failing assertion),
  `/tmp/molla-lidar-metadata-green.log` (1/1 pass after production validation fix).
- Final `make test-sensors` passed **350 executions across 98 successful suites**:
  `/tmp/molla-lidar-storage-full-validated.log`. CPU-only/facade lanes and the
  explicitly ignored GPU suites ran. The earlier fixture failure is preserved
  in `/tmp/molla-lidar-storage-full.log`; the intermediate pre-metadata-fix pass
  is `/tmp/molla-lidar-storage-full-final.log`, not the current final evidence.
- Viewer headless sensor tests passed **17/17** with timing disabled and **17/17**
  with `MOLLA_SENSOR_TIMING=1`: `/tmp/molla-lidar-storage-viewer.log` and
  `/tmp/molla-lidar-storage-viewer-profiled.log`. Sensor GPU-feature and viewer
  all-target checks passed in `/tmp/molla-lidar-storage-check.log` and
  `/tmp/molla-lidar-storage-viewer-check.log`. Session 93527 exited 0.
- Scoped formatting and documentation build passed, session 58400 exit 0:
  `/tmp/molla-lidar-storage-fmt-final.log`, `/tmp/molla-lidar-storage-docs-final.log`;
  book output `/tmp/molla-lidar-storage-book`. Whitespace check and external
  engine/runtime SHA-256 manifests passed. All owned sessions are terminal;
  no viewer GUI, Xvfb or benchmark was launched.
- Engram tool discovery returned no memory tools; this is a structured project
  handoff, not an Engram save. Full goal remains active with the gates below open.
- Updated GPU-gated exports/facade identity checks and LiDAR storage documentation.

### Next Steps

1. Finish the S01 capacity audit across processing constructor host payloads,
   raw geometry/cache preparation and synchronous collection. Existing image,
   annotation, cloud and grid `checked_budget` helpers run before their main
   arrays; their documented caps are GPU bytes, not combined host/process peaks.
2. Complete S00 geometry/frame/miss/tolerance inventory and baseline matrix.
3. Finish S02 resource/lifecycle/allocation evidence, including direct-result
   per-sample metadata allocations. Stable array counts still do not prove zero
   allocations or a speedup.
4. Finish S04 transform/reference/performance and S08 settings/layout/hardware
   presentation gates. S09 remaining camera/transfer integration is separate;
   S10 remains explicitly deferred. Full goal stays active.

### Relevant Files

- crates/molla-sensors/src/lidar_storage.rs — combined/legacy budget contracts.
- crates/molla-sensors/src/lidar_batch.rs — constructor preflight and payload oracle.
- crates/molla-sensors/src/lidar.rs — checked/fallible layouts and metadata validation.
- crates/molla-sensors/tests/sensor_capacity_gpu.rs — budget/parity GPU fixtures.
- crates/molla-sensors/src/lib.rs, crates/molla/src/sensors/mod.rs — public exports.

## 2026-09-25 — Verified processor construction budget increment

### Goal

Continue full SENSORS.md implementation; close the four-processor construction
payload gap without claiming completion of S01/S02 or the overall plan.

### Discoveries

- Retained host mirrors and upload conversion can exceed the device-only budget.
  Device-resident dispatch does not imply absence of host mirror allocations.
- The preflight allocation test detects ordering: its temporary late-check mutation
  fails at 1,509,413 requested bytes versus 116 bytes for each correct rejection.
  The mutation was reverted before final validation; debug and release agree.

### Accomplished

- Full sensors pass 393 test executions, 110 reported suite results including empty
  feature lanes. Three new pure tests, four GPU mirror/packing fixtures and one
  allocation preflight regression are included. Release passes all five new GPU tests.
- Storage unit tests pass 3/3 in CPU and GPU modes. GPU/CPU all-target sensor/facade
  checks, scoped formatting, book build and diff whitespace checks pass.
- Both headless viewer modes pass 17/17. No GUI was opened or closed.
- Release smoke matrices pass all 320 cases (120 LiDAR, 96 cloud/grid, 56 camera,
  48 physics/sensor). Independent coverage/case-set and physics accounting checks
  pass. One first frame plus one sample per case is not performance acceptance.
- Source stayed at `6eaaa2b5cc18784a38e31fd3d546fa058d93716573af3a2560c07ef547320c89`.
  Protected engine/runtime-doc hashes remain unchanged. No physics solver, Gearbox,
  asset or shader edits. Shared storage changed only fallible upload reservation.
- Sessions 10009, 73932, 62714, 6921, 48789, 47932 and 46437 are terminal/collected.
  No benchmark or telemetry polling process remained at final inspection.
- Evidence and source snapshots: `benchmarks/sensors/2026-09-25-processor-storage/`.

### Next Steps

1. Audit remaining scene construction/packing and legacy constructor metadata;
   this budget excludes those paths and driver/binding/per-frame allocations.
2. Finish S02 lifecycle/warm allocation and S04 reference/tie gates from their
   actual requirements; do not infer completion from this constructor-only work.
3. Establish controlled matched performance, hardware visual/settings acceptance,
   and remaining authorized Gearbox camera/moving-part/transfer integration. S10
   remains deferred; the goal remains active.

### Relevant Files

- processor_storage.rs plus four processor wrappers — estimates, caps and arrays.
- sensor_allocation_gpu.rs — rejection ordering and measured error-path requests.
- molla-storage/src/buffer.rs — fallible retained upload scratch reservation.
- sensor_api.rs and facade exports — public API exposure checks.
- SENSORS.md and book/src/sensors/device_results.md — status and budget boundaries.
- benchmarks/sensors/2026-09-25-processor-storage/ — reproducible evidence.

### Key Learnings

1. GPU-only processing still needs constructor accounting for retained CPU mirrors.
2. An allocation check must precede construction, not just return an error afterward.
- crates/molla/tests/sensor_api.rs — facade identities.
- book/src/sensors/lidar_batch.md — storage scopes, limits and exclusions.
- SENSORS.md, SENSORS_PROGRESS.md — current status and evidence.

### Key Learnings

1. Batched LiDAR return arrays consume ray-capacity storage even with fewer pulses.
2. Finite angular endpoints do not guarantee a finite interpolation span.
3. CPU layout budgets and retained GPU/host array budgets cover different lifetimes.

## S00/S04 geometry contracts and independent intersection fixes — 2026-09-25

### Goal

Continue the full SENSORS.md objective by auditing geometry/frame semantics with
independent references, rather than adding only source-path parity tests. The
preceding LiDAR-budget turn made verified progress (350 sensor executions).

### Instructions

- Use Makefile tasks through the repository Nix environment; edit with apply_patch.
- Preserve unrelated engine/runtime work, Gearbox/assets and existing windows.
- No commits/pushes or memory-folder writes; Engram tools remain unavailable.

### Discoveries

- Capsule queries intersected both entire cap spheres. An inside ray could hit an
  inward-facing sphere portion hidden within the capsule cylinder. The new rigid
  analytic-surface fixture failed at distance 0.21035 m instead of the outer
  surface near 0.95079 m. Raw/owned parity had not exposed this shared-kernel bug.
- CPU `RayScene`/`DiffRay` still returned direction-parametric t, despite docs
  claiming parity with the metric GPU raycaster. Very small finite directions
  were classified as misses by its unnormalized quadratic threshold. The new
  scaled-ray/gradient regression reproduced this before the fix.
- LiDAR module docs reversed azimuth handedness (`forward × up`); actual CPU/GPU
  layout uses `up × forward`. Default equatorial order is +Z,+X,-Z,-X.
- Camera depth is radial metric distance, not axis-Z depth. Plane normals are
  configured local normals, not always +Y. Mesh normals oppose the incident ray;
  primitive normals remain outward for exiting rays.
- Live invalid GPU body poses are diagnosed by refit status, but those flags
  still are not attached to query results. The rigid-pose precondition remains;
  this turn does not claim that S04 invalid-live-pose reporting is complete.

### Accomplished

- Fixed capsule cap tests in shared `ray_shapes.wgsl` and exported legacy
  `camera_trace.wgsl`: evaluate both sphere roots and accept only exposed
  hemisphere hits. No physics-solver algorithms changed.
- Added independent f64 known-surface references for all seven analytic primitive
  kinds and a nonuniformly scaled/reflected baked mesh under body/local rotations
  and GPU-only pose changes. Both raw/owned queries must match metric distances,
  points, owners and oriented normals, including inside and backside rays.
- Added axial/oblique capsule regressions, zero-height sphere limit, shared/legacy
  camera capsule depths, arbitrary plane normals, radial camera depths, strict far
  range, informational near, and independently rotated CPU/GPU LiDAR axes.
- Normalized finite nonzero CPU adjoint directions using max-component scaling
  before sphere/plane intersection. Invalid rays retain infinity in its existing
  non-Result API; parameter gradients now use metric depth independently of ray
  magnitude. Existing optimization convergence tests still pass.
- Added `book/src/sensors/contracts.md`: complete per-family geometry inventory,
  explicit unsupported/raw omissions, units/frames/misses/ties, renderer boundary,
  fixture tolerances and remaining limitations. Marked only the S00 geometry
  inventory checkbox complete; broad numerical/baseline/S04 gates remain open.
- Corrected stale camera/refit and frame-output documentation, LiDAR handedness,
  plane-normal and camera-depth descriptions.
- Red evidence: `/tmp/molla-sensor-contracts-focused.log` (capsule),
  `/tmp/molla-sensor-contracts-adjoint-red.log` (CPU scaled rays).
  Focused green: contracts 5/5, camera suite 4/4, CPU adjoint suite 5/5 in
  `/tmp/molla-sensor-contracts-{fixed,camera,adjoint-green}.log`.
- Full `make test-sensors` passed **358 executions across 100 successful suites**:
  `/tmp/molla-sensor-contracts-full.log`, session 34341 exit 0. It includes
  CPU-only/facade lanes and explicitly executed ignored GPU tests.
- Viewer headless regressions passed **17/17** timing off and **17/17** with
  `MOLLA_SENSOR_TIMING=1`; sensor GPU/all-target and viewer all-target checks,
  scoped formatting and initial documentation build passed, session 41795 exit 0.
  Logs: `/tmp/molla-sensor-contracts-{viewer,viewer-profiled,check,viewer-check,fmt,docs}.log`.
  Final documentation-only rebuild passed after stale-prose cleanup:
  `/tmp/molla-sensor-contracts-docs-final.log`, session 18234 exit 0. All owned
  build/test sessions from this turn are terminal.
- `git diff --check` and external engine/runtime SHA-256 manifests passed. No GUI,
  Xvfb or benchmark processes were launched. These are correctness/headless checks,
  not performance or visual acceptance. Engram discovery returned no memory tools;
  this structured handoff is stored in the project file, not Engram.

### Next Steps

1. Complete S00 reproducible baseline/numerical matrix and remaining S01 capacity
   audit; current reference budgets are fixture-specific, not universal guarantees.
2. Finish S04 transform/error reporting, especially attaching invalid live-pose
   diagnostics to query validity rather than relying only on a caller precondition.
   Independent large-coordinate/near-tie/grazing references remain broader work.
3. Complete S02 lifecycle/allocation gates and realistic S04/S08 performance matrix;
   no speedup or zero-allocation conclusion follows from the new correctness tests.
4. Finish S08 full settings/layout/hardware presentation and S09 remaining camera/
   transfer integration. S10 stays deferred. Full goal remains active.

### Relevant Files

- crates/molla-kernels/shaders/{ray_shapes,camera_trace}.wgsl — exposed capsule roots.
- crates/molla-sensors/src/raycast_adjoint.rs — metric normalization and invalid rays.
- crates/molla-sensors/src/lidar.rs — corrected coordinate-convention docs.
- crates/molla-sensors/tests/sensor_contracts_gpu.rs — independent GPU contracts.
- crates/molla-sensors/tests/sensor_camera_device_gpu.rs — both camera capsule paths.
- crates/molla-sensors/tests/raycast_adjoint.rs — CPU scaled-ray/gradient regressions.
- book/src/sensors/contracts.md, book/src/SUMMARY.md — sensor contract inventory.
- book/src/sensors/{raycaster,tiled_camera,frame_transform}.md — corrected semantics.
- SENSORS.md, SENSORS_PROGRESS.md — precise gate status and evidence.

### Key Learnings

1. Shared-kernel parity can preserve the same geometric bug; independent surface
   references exposed hidden capsule-cap intersections.
2. Direction normalization is part of the distance and gradient contract, not
   merely an optimization.
3. Sensor camera depth is radial distance and need not match renderer Z buffers.

## 2026-09-25 — Owned-scene invalid-pose query reporting

### Goal

Continue the full SENSORS.md objective by attaching live GPU pose/bounds failures
to owned ray, LiDAR and camera results. The immediately preceding status-only turn
was no progress; this turn added failing regressions, production fixes and passing
focused evidence. Full S00–S09 acceptance is not inferred from this subtask.

### Instructions

- Use Makefile tasks through the repository Nix environment and apply_patch edits.
- Preserve unrelated engine/runtime work, Gearbox/assets and user windows.
- No commits, pushes or memory-folder writes. Engram tool discovery returned no
  tools; this project handoff is not an Engram save.

### Discoveries

- Per-leaf refit diagnostics did not prevent invalid GPU poses from looking like
  ordinary misses or valid hits on other objects. Both new initial regressions
  failed in `/tmp/molla-query-validity-red.log` before the fix.
- LiDAR reduction would erase an invalid sentinel as an ordinary miss; camera
  shading did the same. Those consumers require explicit preservation, not only
  a traversal change.
- Lower-bound GPU vec4 padding can carry aggregated flags through existing refit
  passes without adding channels, GPU payload arrays or dispatches. Infinite
  planes require separate pose checks because they have no finite hierarchy leaf.

### Accomplished

- Invalid finite-leaf flags propagate to the root, including the refit after
  periodic/quality-triggered spatial reordering. Traversal rejects the entire
  active owned-scene query when included geometry is invalid; unrelated invalid
  bodies without included geometry do not poison valid GPU queries.
- Ray/LiDAR distance and signed shape index are -2, point/normal zero. LiDAR
  first/last become -2 and count zero without losing the flag during per-scan
  range clipping. Camera depth is -2, shape ID u32::MAX-1 and other channels zero;
  center geometry retains the flag through aperture accumulation. Ordinary misses
  remain unchanged. Public constants are exported by both sensor crates/facades.
- Synchronous `Raycaster::cast_scene` returns an error for invalid included poses
  or unsafe bounds instead of constructing misleading RayHit misses.
- Documented active-channel versus numerical validity, color-only ambiguity,
  source validity retention through cloud/annotation processing, GPU 2e-5 unit
  quaternion tolerance, conservative 2^120 guard, recovery and raw-model scope.
- Focused `sensor_query_validity_gpu` passes 4/4 in
  `/tmp/molla-query-validity-focused.log`: NaN/Inf/unsafe positions, bad quaternions,
  primitive/mesh/plane-only and mixed scenes, 66-leaf hierarchy/rebuilds, standalone
  and direct LiDAR/aperture camera, delayed historical frames, recovery, empty
  queries and scene replacement. Expanded test helper initially failed to compile
  because ModelBuilder is not Clone; rebuilt the second fixture explicitly.
- Full `make test-sensors` passed **362 executions across 102 successful suites**
  in `/tmp/molla-query-validity-full.log`; session 70575 exited 0. This includes
  CPU-only/facade lanes and explicit ignored GPU tests, not merely compilation.
- Viewer headless tests passed **17/17** normally and **17/17** with
  `MOLLA_SENSOR_TIMING=1`. Sensor GPU/all-target and viewer all-target checks,
  scoped format check, documentation build, `git diff --check` and both external
  engine/runtime hash manifests passed; session 61013 exited 0. Logs:
  `/tmp/molla-query-validity-{viewer,viewer-profiled,check,viewer-check,fmt-check,docs}.log`.
  Built book: `/tmp/molla-query-validity-book`.
- All owned build/test sessions from this turn are terminal. No GUI, Xvfb or
  benchmark was launched; no performance or hardware visual acceptance is claimed.

### Next Steps

1. Continue S00 numerical/baseline matrix, S01 remaining capacity audit, S02
   lifecycle/allocation gates and realistic S04/S08 performance acceptance.
2. Finish hardware/settings visual coverage and remaining S09 camera/transfer
   integration. Raw-model compatibility calls still require valid rigid poses;
   non-query sensors do not acquire the owned-scene sentinel contract. S10 deferred.
3. Full SENSORS.md objective remains active. The invalid-pose reporting subtask is
   verified; remaining S04 reference/numerical/transform gates are not checked off.

### Relevant Files

- crates/molla-kernels/shaders/{scene_bounds,scene_trace}.wgsl — aggregated validity.
- crates/molla-kernels/shaders/{lidar_scan,camera_shade}.wgsl — preserve invalid flags.
- crates/molla-sensors/src/{scene,raycast,lib,lidar_batch,scene_acceleration}.rs — API/error contracts.
- crates/molla/src/sensors/mod.rs, crates/molla/tests/sensor_api.rs — facade constants.
- crates/molla-sensors/tests/sensor_query_validity_gpu.rs — live-pose regressions.
- book/src/sensors/{contracts,scene,device_results,lidar_batch,tiled_camera}.md — validity semantics.
- SENSORS.md, SENSORS_PROGRESS.md — remaining full-scope gates and evidence.

### Key Learnings

1. Invalid geometry must invalidate closest-hit/free-space claims even when the
   particular ray points elsewhere; per-object diagnostic flags alone are insufficient.
2. A successful GPU submission or active result channel is not a numerical-validity
   guarantee. Reduction/shading stages must preserve invalid-result distinctions.

## 2026-09-25 — Per-frame LiDAR performance matrix

### Goal

Continue full SENSORS.md performance acceptance by measuring per-frame latency,
not treating Criterion mean confidence intervals as p95. The preceding goal turn
was progress: owned-scene validity fixes and 362 passing regression executions.

### Instructions

- Use repository Makefile/Nix tasks, apply_patch edits, no commits or pushes.
- Preserve external engine/runtime changes, Gearbox/assets and user windows.
- No memory-folder writes; Engram tool discovery still returned no tools. This
  structured project handoff is not an Engram save.

### Discoveries

- Existing Criterion cases do not supply raw per-frame p95 data across the required
  detailed-mesh, moving-pose and topology-replacement workloads.
- Resource warmup is not thermal/frequency stabilization. In the completed initial
  timestamp-enabled run, 104/120 named cases had greater-than-10% p95 spread across
  three repeats. Graphics clocks ranged from 210 to 2790 MHz during the monitored
  run; this is correlation, not a proven explanation for individual spikes.
- Device completion, result transfer/decode and scene replacement are distinct
  measured costs. GPU timestamps alone exclude descriptor uploads, CPU scene
  compilation, final readback and CPU processing.

### Accomplished

- Added `bench_sensor_latency` with per-frame TSV output, separate setup/first/
  warmup/sample phases, CPU encode/submit/completion intervals, optional GPU
  timestamps, actual selected-output transfer/submission/wait counters, calling-
  thread Rust allocation counts and explicit partial memory/transfer accounting.
- Added a defined 120-case LiDAR coverage matrix: nominal 1k/10k/100k/1M total rays;
  100/1k/10k spheres; 1/10/100 shared detailed-mesh instances (32,768 triangles);
  129x129 terrain; 1/4/16 mounts; 1/4 divergent sub-rays; live per-frame GPU poses;
  actual alternating topology replacement; device-only and selected final-host
  consumption. Counts round to full rectangular grids; actual counts are logged.
- Added deterministic nearest-rank p50/p95 and bootstrap p95 interval helpers,
  tested allocator unwind recovery and quantile arithmetic (2/2 host tests).
- Added `make bench-sensors-latency` and `sensor-bench-info` for source/toolchain/
  environment fingerprints. Added dev-only pollster dependency, already in lockfile
  for other crates; preserved existing viewer dependency change.
- Initial and corrected-grid release smoke runs passed all ten quick cases.
  Corrected smoke: `/tmp/molla-latency-smoke-v2.{log,tsv}`. Initial smoke used a
  single-elevation fan; it is not the measured rectangular-grid baseline.
- Timestamp-enabled matrix completed: 360 case-repeats, 23,040 measured frames,
  2,880 warmups and 360 first frames. Session 21482 exited 0 and stopped its own
  board monitor. Artifacts `/tmp/molla-latency-matrix.{log,tsv}`,
  `/tmp/molla-latency-matrix-gpu.csv`, `/tmp/molla-latency-matrix-after.log`.
  Crate fingerprint stayed `6d46d0ed83a52eaca8ef8389c3545c65ba4c73127f5ec6590ced11e167d66125`.
- Recorded hardware: RTX 4080/Vulkan, NVIDIA 595.84, Rust/Cargo 1.94.1, release,
  Linux 7.0.0-31-generic, HEAD 367d92c with WIP. No clock/power settings changed.
- Instrumentation-off control also completed all 120 cases x 3 repeats x 64
  measured frames; session 95964 exited 0 and stopped its own monitor. Logs/data:
  `/tmp/molla-latency-control*`. 93/120 cases still exceeded 10% repeat p95 spread;
  removing timestamps did not establish stationarity. Crate fingerprint stayed
  identical to the timestamp-enabled run. All 23,040 control samples passed
  transfer/submission invariants: device=0 result bytes/1 submission, host=positive
  result bytes/2 submissions. Diagnostic timestamp transfers are separate.
- Longer-warmup instrumentation-off probes completed (session 28946 exit 0):
  256 samples x 3 repeats x 2 consumption modes, with 5,000 warmups for 1k rays
  and 128 warmups for million-ray terrain. Each probe has 1,536 measured frames.
  Small-case p95 ranges remained 0.191–0.306 ms device / 0.163–0.359 ms host;
  terrain remained 0.785–1.657 ms device / 19.399–19.522 ms host. Warmup alone did
  not remove all variation. Graphics and memory clocks varied; individual causes
  remain unproven. No clock changes or user-process closures were made.
- Preserved every raw sample/outlier, log, board monitor and final metadata in
  `benchmarks/sensors/2026-09-25-lidar-latency/` (1.4 MiB compressed), with verified
  SHA256SUMS and reproduction README. Published provisional p95 ranges and exact
  transfer/allocation exclusions in the book. Million-ray single-sub-ray host
  frames collect 52 MB; control terrain p95 was 1.508–1.814 ms device versus
  18.634–19.597 ms host. These are completion boundaries, not isolated PCIe timing
  or a before/after optimization speedup. Shared desktop peak VRAM was 3,300 MiB;
  whole-process RSS high-water about 469 MiB timestamped / 455 MiB uninstrumented.
- `make test-sensors` passed **364 executions across 104 successful suites**:
  `/tmp/molla-latency-full-tests.log`. GPU/all-target sensor check and scoped
  format check passed. Initial and final results-page book builds passed. Combined
  gate session 38452 and final docs/metadata session 68079 exited 0; logs
  `/tmp/molla-latency-{all-check,fmt-check,docs}.log`. No production sensor or
  viewer source changed this turn, so no new visual acceptance is claimed.
- Final docs: `/tmp/molla-latency-docs-final.log`; final source/toolchain snapshot:
  `/tmp/molla-latency-final-info.log`, same crate fingerprint as both matrix runs.
- External engine/runtime SHA-256 manifests and `git diff --check` passed. All
  benchmark/test/monitor sessions listed above are terminal; there is no live GPU
  measurement to restart or wait on.
- Added methodology/limits in `book/src/sensors/latency_benchmarks.md`. This is
  LiDAR coverage only, not full S00/S08 camera/processing/physics acceptance.

### Next Steps

1. Continue missing camera/processing/physics/legacy comparison lanes, peak-memory/
   full transfer accounting, remaining numerical/capacity/lifecycle gates, hardware
   visual coverage and S09 camera/transfer integration. Full goal remains active.
2. Continue controlled power-state/scheduling/maintenance investigation alongside
   the missing matrix coverage. Both timestamp-off controls and longer warmup
   failed to establish universal <=10% repeatability; no waiver or hard budgets yet.

### Relevant Files

- crates/molla-sensors/benches/bench_sensor_latency.rs — raw per-frame LiDAR matrix.
- crates/molla-sensors/benches/support/{metrics,timing}.rs — allocation/quantile/timestamp support.
- crates/molla-sensors/tests/benchmark_metrics.rs — measurement helper regression tests.
- crates/molla-sensors/Cargo.toml, Cargo.lock — benchmark registration and dev dependency.
- Makefile — benchmark/metadata entrypoints.
- book/src/sensors/latency_benchmarks.md, book/src/SUMMARY.md — measurement boundaries.
- benchmarks/sensors/2026-09-25-lidar-latency/ — raw evidence archive and checksums.
- SENSORS.md, SENSORS_PROGRESS.md — partial matrix coverage and evidence.

### Key Learnings

1. Confidence intervals on an averaged benchmark are not per-frame p95 latency;
   raw samples and separate cold/warm/transfer boundaries are necessary.
2. Eight resource-warmup frames did not yield stationary latency on this shared
   desktop GPU. Repeat spread must be investigated before fixing acceptance budgets.

## 2026-09-25 — Camera and same-submission processing latency

### Goal

Continue full SENSORS.md with the missing camera/optics and camera-to-processing
performance matrix. The previous goal turn made progress: LiDAR raw latency
evidence, control/probe runs and 364 passing regression executions. Performance
acceptance remains distinct from executable workload coverage.

### Instructions

- Use Makefile/Nix tasks and apply_patch edits. No commits/pushes, no user-window
  closures or GPU clock/power configuration changes.
- Preserve unrelated engine/runtime work, Gearbox/assets and memory-folder state.
- Engram discovery still returned no tools; this is a project-file handoff.

### Discoveries

- Four 1280x720 cameras reserve 1,341,850,224 bytes of combined camera host/device
  arrays. That exceeds the library's default 1 GiB aggregate cap; the benchmark
  explicitly opts into 2 GiB without changing defaults or disabling device checks.
- Processing input `copied_bytes` counters describe GPU-to-GPU copies, not host
  transfers. Raw final-camera channels and processed dataset outputs have different
  readback payloads and cannot be compared as identical consumption workloads.
- Prior LiDAR evidence had fingerprints but no source-input snapshot. Captured its
  exact crate Rust/WGSL/manifests and root build/environment manifests before any
  camera benchmark changes in `source-inputs.tar.gz` (3.3 MiB), updating SHA256SUMS.

### Accomplished

- Added `bench_camera_latency` using the tested shared allocation/quantile/timestamp
  helpers. Makefile `LATENCY_BENCH` selects it through the existing metadata runner;
  default remains the LiDAR harness. No production sensor implementation changed.
- Defined 56 cases covering both required resolutions (640x480, 1280x720), 1/4
  cameras, pinhole/four-sample optics, homogeneous/two-FOV heterogeneous grouping,
  a moving body-attached detailed mesh, raw/processed output and device/host modes.
- The processing chain encodes exposure, gray-world reduction, Reinhard, gamma and
  parallel class/instance masks/boxes in the same submission as all cameras. Host
  collection is one grouped copy; no per-group or per-aperture host collection.
- Fixture has 100 colored spheres, a 32,768-triangle wavy mesh and a background
  plane. Moving samples upload current device poses; they do not run physics.
  Host checks validate positive finite depths/known IDs or complete label counts.
- All six quick and all 56 full-matrix smoke cases passed in release mode, including
  the largest aperture/chain case. Logs `/tmp/molla-camera-latency-smoke*` and
  `/tmp/molla-camera-latency-matrix-smoke*`. GPU/bench check passed in
  `/tmp/molla-camera-latency-check.log`. Formatting applied through Makefile.
- Both timestamped and instrumentation-off matrices completed: each 56 cases x
  3 repeats x 64 measured frames, totaling 21,504 measurements. All benchmark and
  monitor processes exited. Before/after crate fingerprints matched:
  `850f99b93f8a2277d8195553953b616f3ffd2c2d45a0d034f54492106c55b4f8`.
- Archived raw measurements, logs, monitoring, source inputs and checked SHA256SUMS
  in `benchmarks/sensors/2026-09-25-camera-latency/`. All submission/readback
  invariants passed. P95 repeat spread exceeded 10% in 26/56 timestamped and 29/56
  control cases; this is provisional evidence, not performance acceptance.
- Added `book/src/sensors/camera_latency_benchmarks.md` with matrix, memory overrides,
  output accounting, selected control p95 ranges and interpretation limits.
- Session 33912 completed regression/check/format/docs gates: 364 passing executions
  across 104 suites. Logs `/tmp/molla-camera-latency-{full-tests,all-check,fmt-check,docs}.log`.
  Documentation results were subsequently expanded; rebuild final docs next.

### Next Steps

1. Rebuild final documentation after adding the recorded results.
2. Continue cloud/physics/legacy comparison lanes, total transfer/peak-memory and
   controlled repeatability investigation. Remaining numerical/capacity/lifecycle,
   visual and S09 camera/transfer gates stay open; full goal remains active.

### Relevant Files

- crates/molla-sensors/benches/bench_camera_latency.rs — camera/processing latency matrix.
- crates/molla-sensors/benches/support/{metrics,timing}.rs — reused measurement helpers.
- crates/molla-sensors/Cargo.toml, Makefile — benchmark registration/selection.
- book/src/sensors/camera_latency_benchmarks.md, book/src/SUMMARY.md — methodology.
- book/src/sensors/latency_benchmarks.md — camera companion link.
- benchmarks/sensors/2026-09-25-lidar-latency/ — preserved source-input snapshot.
- SENSORS.md, SENSORS_PROGRESS.md — exact remaining performance gates.

### Key Learnings

1. Required large camera cases can exceed a safe default aggregate budget; explicit
   bounded benchmark overrides preserve validation without silently omitting cases.
2. Processing-chain host timing includes a different final-output payload than raw
   camera channels; lower collection time alone does not prove processing is free.

## 2026-09-25 — Frame-local aperture hierarchy reuse

### Goal

Continue full SENSORS.md by removing redundant camera hierarchy work, preserving
moving geometry and query correctness. The preceding status turn collected the
terminal regression process; it introduced no implementation change.

### Instructions

Use Makefile/Nix and apply_patch; preserve unrelated WIP, assets and user windows.
No commits, GPU settings changes or memory-folder updates. Engram tools unavailable.

### Discoveries

- Owned-scene cameras refitted the same hierarchy for every aperture sample.
  Body poses cannot change between samples within one camera encoding call.
- Cross-frame caching would require submission/cancellation and pose-version
  tracking. The implemented scope is only samples after zero in one call.

### Accomplished

- Collected a pre-change 3 x 256-frame control measurement of four moving 640x480
  aperture cameras; `/tmp/molla-aperture-refit-before.{log,tsv}`, terminal success.
- Added a private sample-index path: sample zero always refreshes bounds; later
  samples reuse them. New scene traces force refresh. All standalone ray/LiDAR
  calls still refresh; periodic/quality policy counters and post-rebuild refits
  remain unchanged. No cross-frame or cancellation-persistent readiness flag.
- Added a cancellation/GPU-only-motion regression comparing all camera channels
  against raw queries with rebuild disabled, periodic and quality-triggered.
- Updated exact dispatch budgets. Red test failed as expected: five-sample frame
  used 26 dispatches rather than the new target of 22.
- Focused GPU suite passed 6/6, including raw-query channel parity across all three
  maintenance policies, GPU-only poses and cancellation. Full `make test-sensors`
  passed 365 executions across 104 suites, and the default headless viewer passed
  17/17. GPU all-target check, scoped formatting and external engine/runtime
  hashes passed. Logs `/tmp/molla-aperture-refit-{direct,full-tests,viewer,check,fmt-check}.log`.
- Matched post-change control completed 3 x 256 measurements. P95 completion was
  6.425–6.586 ms before and 5.839–5.930 ms after; CPU encode p50 was
  124.436–126.014 µs and 95.505–107.013 µs. Ordinary frame dispatches decreased
  49 to 25 and calling-thread allocations 760 to 424. Periodic rebuild overhead
  remains 22 dispatches/310 allocations, with unchanged query cadence. All 1,536
  measured before/after frames retain 160 pose-upload bytes, no result download,
  one producer submission and one wait. Sequential single-workload observations
  are not general speedup or full-matrix acceptance evidence; no board-clock trace.
- All 56 camera smoke cases passed after the optimization, including host outputs,
  heterogeneous groups, aperture processing chains and largest budget overrides.
  Logs `/tmp/molla-aperture-refit-smoke.{log,tsv}`. No benchmark process remains live.
- Archived before/after data, regression logs and matching post-change source
  inputs in `benchmarks/sensors/2026-09-25-aperture-refit/`. Source fingerprint
  stayed `28b40af18f3b2b92fa5c1e79776a4a233b44b1c8e97c8135407d37b9182f90df`
  through the post-change measurement, full smoke and final metadata capture.
- Updated camera pipeline/measurement documentation and reconciled S00's stale
  camera-matrix status without checking off incomplete full-performance gates.
- `MOLLA_SENSOR_TIMING=1 make test-viewer-sensors` passed 17/17 and the final book
  build passed in `/tmp/molla-aperture-refit-{viewer-timed,docs}.log`. Session 2634
  exited successfully. All owned execution handles from this turn are terminal.

### Next Steps

1. Continue cloud-processing and CPU/GPU physics combined-load performance lanes,
   matched legacy comparisons and transfer/peak-memory accounting. The full
   camera/LiDAR repeat-variance issue remains open despite this stable subset.
2. Complete remaining S00-S09 numerical/lifecycle/visual/integration gates. The
   broader goal is active; this optimization does not complete SENSORS.md.

### Relevant Files

- crates/molla-sensors/src/{raycast,scene_trace,tiled_camera}.rs — frame-local refit reuse.
- crates/molla-sensors/tests/sensor_camera{_direct,_manager_direct}_gpu.rs — acceptance.
- book/src/sensors/{tiled_camera,camera_latency_benchmarks}.md — behavior and measurements.
- benchmarks/sensors/2026-09-25-aperture-refit/ — reproducible raw evidence and sources.
- SENSORS.md — reconciled baseline coverage; incomplete gates remain unchecked.
- SENSORS_PROGRESS.md — verified state and remaining full-scope work.

### Key Learnings

1. Aperture samples share body poses, but separate frames cannot assume bounds are current.
2. Scheduled spatial reordering still needs its post-reorder refit even within one frame.
3. This detailed-mesh fixture saves 24 dispatches and 336 calling-thread allocations
   per four-sample frame; scene size and aperture count determine the savings.

## 2026-09-25 — LiDAR cloud/grid-chain latency matrix

### Goal

Continue full SENSORS.md with end-to-end cloud-processing performance evidence.
The previous goal turn made verified progress: frame-local aperture refit reuse,
matched measurements, regression/feature/viewer/documentation gates and archives.
CPU/GPU physics combined-load work remains part of the full objective.

### Instructions

Use Makefile/Nix and apply_patch. Preserve unrelated WIP, engine/runtime files,
Gearbox/assets, user windows and GPU configuration. No commits or memory-folder
updates. Engram discovery returned no tools; this is the project handoff.

### Discoveries

- Pending LiDAR receipts can feed per-mount cloud conversion and a spatial grid
  in one producer submission with no intermediate host readback.
- A full reserved-range cloud/grid collection is one final readback, but retains
  its full payload even for reject-all filtering. Kept-prefix host validation
  work varies; this must not be confused with selected-prefix export throughput.
- Processing `copied_bytes` measures GPU staging copies. The benchmark's narrower
  upload counter does not include cloud/grid uniforms or driver upload staging.
- `ReadbackRing::wait` polls first and increments `blocking_waits` only if not yet
  ready. Zero host blocking waits with two submissions and positive result bytes
  is legitimate completed collection, not an invariant violation.

### Accomplished

- Extended `bench_sensor_latency` with mutually exclusive `--cloud-matrix`, leaving
  existing LiDAR case names/defaults intact and adding a kernel-dispatch TSV column.
- Added a reusable benchmark cloud-chain helper: per-mount rigid transform,
  keep/crop/reject filtering, stable compaction and a 4,096-cell grid. Three direct
  outputs share one command batch; host output is one grouped final collection.
- Defined 96 cases spanning nominal 1k/10k/100k/1M rays, 1/4/16 mounts, 1/4 divergent
  sub-rays, 1/10/100 mesh instances or terrain, fixed/GPU-prescribed motion, three
  filters and device/host consumption. No physics stepping is claimed.
- All 96 release smoke configurations passed, including largest buffers and
  per-mount metadata, compaction counts, grid diagnostics/member totals and final
  result validation. Source check and scoped formatting passed; logs
  `/tmp/molla-cloud-latency-{check,fmt,smoke}.log` and `-smoke.tsv`.
- Added book methodology and linked it from the sensor latency documentation.
  No production sensor implementation or dependency manifests changed this turn.
- Both repeated runs completed (session 50386 exited successfully): each 96 cases
  x 3 repeats x 64 measured frames, totaling 36,864 measurements, with 2,304 warmups
  and 288 first frames per run. All producer/output/transfer checks passed. There
  were 13 timestamped and 15 control host frames already ready before blocking.
  No benchmark/monitor remains live. Logs
  `/tmp/molla-cloud-latency-{matrix,control}.{log,tsv}`, `-gpu.csv`, `-after.log`.
  Before/after crate fingerprints matched in both runs:
  `6310448c09baf979c52ce54ec50936fa4c7a4de5187e54f8e24ac54228db09bb`.
- Archived raw data, monitoring, logs and source inputs with checked SHA256SUMS in
  `benchmarks/sensors/2026-09-25-cloud-latency/`. Book results report all limitations.
- P95 repeat spread exceeds 10% for 81/96 timestamped and 83/96 control cases;
  the moving 100k keep-all device case varied over sixfold. Performance acceptance
  remains open, not waived. Control million-ray terrain keep-all p95 is
  4.043–7.714 ms device / 46.646–51.055 ms host, with 116,269,676 result bytes;
  these are different consumption boundaries, not a matched speedup comparison.
- Whole-board clocks ranged 210–2,790 MHz and memory peaked at 4,067 MiB including
  desktop usage; process HWM was 1,221,828/1,221,916 KiB. No isolated peak-VRAM claim.
- Original ten-case LiDAR quick smoke passed after harness extension. Full sensor
  regression passed 365 executions/104 suites, all-target GPU check and scoped
  formatting passed, and final book build passed. Logs
  `/tmp/molla-cloud-latency-{lidar-smoke,full-tests,all-check,fmt-check,docs}.log`.
  Sessions 54958/38364 exited successfully; all owned execution handles are terminal.
- Reconciled S00/S08 and camera documentation with the measured cloud matrix;
  no incomplete acceptance checkbox was marked complete. Whitespace and external
  engine/runtime hashes passed. Current source still matches the archived fingerprint.

### Next Steps

1. Implement CPU/GPU physics combined-load measurement. The facade crate already
   depends on solvers/sensors and GPU pollster, so a facade benchmark can use both
   without introducing solver dependencies into the sensor library. Existing
   Featherstone CPU/GPU benchmarks demonstrate uploaded model/state/control with
   ping-pong state and contactless articulated chains; combined coverage must label
   its exact solver/contact/mount/timestamp boundaries rather than imply Gearbox parity.
2. Continue matched legacy comparisons and repeat-variance investigation, complete
   transfer/peak-memory accounting and remaining numerical/lifecycle/visual/S09
   integration acceptance. Do not mark the broader goal complete from this matrix.

### Relevant Files

- crates/molla-sensors/benches/bench_sensor_latency.rs — LiDAR and cloud matrix orchestration.
- crates/molla-sensors/benches/support/cloud_chain.rs — same-submission cloud/grid workload.
- book/src/sensors/cloud_latency_benchmarks.md — method, budgets and measurement boundaries.
- book/src/sensors/latency_benchmarks.md, book/src/SUMMARY.md — navigation.
- benchmarks/sensors/2026-09-25-cloud-latency/ — raw evidence and matching source snapshot.
- SENSORS.md — measured matrix coverage, remaining gates still unchecked.
- SENSORS_PROGRESS.md — authoritative progress and remaining work.

### Key Learnings

1. Device-resident compacted tails can feed grid processing without a count download.
2. Fixed reserved-range collection does not become smaller when a filter rejects points.
3. A successful synchronous collection can report zero blocking waits if its initial
   poll finds the mapping ready; completion/transfer counters must be interpreted together.

## 2026-09-25 — Physics and sensor combined-load measurements

### Goal

Continue full SENSORS.md with CPU/GPU physics combined-load coverage. The previous
goal turn made verified progress: 96 cloud/grid cases, repeated measurements,
archives and regression gates; repeatability acceptance remains unresolved.

### Instructions

Use Makefile/Nix and apply_patch. Preserve unrelated WIP, engine/runtime changes,
Gearbox/assets and user windows. No commits, GPU configuration or memory-folder
updates. Engram discovery returned no tools; this is the project-file handoff.

### Discoveries

- Current GPU Featherstone performs four distinct state-copy submissions followed
  by one dynamics submission per step. This is five, not one. The new benchmark
  labels that count and its 48-byte uniform as source-audited, not runtime counters.
- Explicit ping-pong steps avoid `with_substeps`' host scratch path. GPU body poses
  remain device-authoritative for subsequent sensor queries; CPU stepping uploads
  only final body poses through retained conversion scratch.
- A GPU marker before physics plus a final sensor marker measures queue elapsed
  span, including gaps while the CPU works; it is not summed GPU kernel execution.
- Host NVML now reports 595.91 while `/proc/driver/nvidia/version` remains 595.84.
  The initial quick launcher failed in metadata collection before compiling/running
  the benchmark. A command-local `LD_LIBRARY_PATH` pointing at existing
  `/nix/store/5awz9kh3mbyk28j24i4p8n00954y0knk-nvidia-x11-595.84-nixGL/lib`
  restores nvidia-smi telemetry. No driver/package/system configuration changed.

### Accomplished

- Added a facade-crate benchmark to use existing solver/sensor dependencies without
  adding solver dependencies to the sensor library. Makefile `SENSOR_BENCH_PKG`
  chooses the benchmark package; existing sensor targets/defaults are preserved.
- Added a shared four-link articulated-sphere fixture with gravity, damping and
  nonzero initial angles/velocities. Floor is query geometry only; no contacts,
  collisions, actuators, muscles or vehicle dynamics are implied.
- Added same-frame CPU/GPU physics-to-query regression for 1/4 articulations,
  alternating 1/10 explicit steps and 24 samples, checking analytical sphere
  distance/point against CPU-stepped poses. GPU poses stay host-inaccessible.
  Focused GPU test passed 1/1: `/tmp/molla-physics-sensors-test.log` (session 5026).
- Added that regression to `make test-sensors` and drafted a 48-case benchmark:
  CPU/GPU, 1/32 articulations, 1/10 steps, three LiDAR/cloud/camera loads, device/host.
  Initial benchmark check passed (session 5049). No production physics/sensor code changed.
- Added configurable Makefile `NVIDIA_SMI` metadata command and explicit kernel
  version/tool-command logging, allowing matching telemetry libraries per invocation.
  NVIDIA Vulkan execution remains through existing pinned nixVulkan 595.84.
- Quick release smoke exposed a real unresolved composition failure: CPU cases
  complete, but the first GPU-physics/LiDAR frame times out on GPU completion.
  Disabling timestamps still fails. Standalone same-frame ray-query coupling passes
  in debug and release; a new external-device LiDAR regression reproduces the hang.
  An explicit diagnostic wait proves physics completes before the LiDAR timeout,
  so the failure is not simply slow dynamics or missing a physics completion wait.
  Failed runs are terminal; logs `/tmp/molla-physics-latency-{quick-fixed,gpu-control-smoke}.log`
  and `/tmp/molla-physics-sensors-{release-test,external-test,separated-test}.log`.
- The skip-physics/device-dirty pose test also timed out (71628, terminal), proving
  live dynamics are not required to reproduce it. Raw-model LiDAR completed four
  GPU-stepped frames (83796, terminal success); owned-scene standalone LiDAR still
  timed out (39060, terminal). Logs `unseeded-test`, `raw-test`, `owned-test` under
  `/tmp/molla-physics-sensors-*.log`. The issue is narrower than general GPU dynamics.
- Temporary TLAS and mesh traversal visit bounds did not fix the timeout and were
  reverted exactly. Do not treat them as production changes or completed fixes.
- Diagnostic switches in the new external-device test currently include
  MOLLA_DIAG_WAIT_PHYSICS, MOLLA_DIAG_SKIP_PHYSICS and MOLLA_DIAG_RAW (`scene` selects
  owned standalone instead of raw). There is temporarily a MOLLA_DIAG_SKIP_TRACE
  guard in `src/scene_trace.rs` to isolate traversal versus other passes. All these
  temporary guards must be removed before acceptance. The no-traversal test
  completed in 3.24 seconds with the expected invalid-output assertion, not a
  timeout (`/tmp/molla-physics-sensors-no-trace-test.log`). Skipping only shared
  traversal removes the hang; this isolates the failing dispatch but does not
  establish its root cause or constitute a fix.

### Next Steps

1. Diagnose and fix the shared-traversal timeout, remove temporary diagnostic
   switches, then run quick/full release smoke and correct accounting failures.
2. Record repeated timestamped/control matrices with source snapshots and isolated
   measurement processes; then regression/feature/format/docs checks.
3. Preserve the complete remaining S00-S09 contract, lifecycle, performance, visual
   and integration gates. Contactless articulated loads are not Gearbox performance parity.

### Relevant Files

- crates/molla/benches/bench_sensor_physics_latency.rs — combined-load benchmark.
- crates/molla/benches/support/sensor_physics.rs — shared articulated dynamics fixture.
- crates/molla/tests/sensor_physics_gpu.rs — same-frame live-pose coupling regression.
- crates/molla/Cargo.toml, Makefile — benchmark registration/package and regression target.
- SENSORS_PROGRESS.md — verified progress and remaining scope.

### Key Learnings

1. Current GPU Featherstone's five submissions per step must be included in combined-load accounting.
2. Queue timestamps spanning CPU and GPU work are elapsed queue spans, not pure kernel timings.

## 2026-09-25 — Batched traversal isolation and stackless prototype

### Goal

Resolve the combined-load failure without dropping accelerated or mesh queries.
The previous implementation turn made progress by adding the combined benchmark
and reproducer; the intervening status reply did not advance implementation.

### Discoveries

- The unseeded hierarchy's live bounds enclose CPU-reference sphere positions,
  its status flags are valid, and independently inspected GPU topology/shape
  records and query parameters match their expected values.
- All 1024 scan directions complete as scalar host-ray queries. All contiguous
  power-of-two partitions through 512 rays complete, including the second half.
  The full batched host scan, 1023 rays with spare capacity, and a uniform plane
  ray repeated 1024 times reproduce the timeout. GPU LiDAR generation is not
  required. This is not established as a physics-engine failure.
- The failing process maps matching NVIDIA 595.84 compiler/Vulkan libraries.
  The separate host NVML mismatch does not establish a runtime compiler mismatch.
- A diagnostic that removes both private traversal stacks and scans primitive
  leaves linearly completes four physics/LiDAR frames and matches independent
  shape IDs and sphere/ground distances (1 mm tolerance). This is diagnostic
  evidence, not an acceptable production fallback or a proven driver root cause.

### Decision

Prototype threaded, stackless TLAS and BLAS traversal. Keep full acceleration,
all geometry kinds, refits/rebuilds and metric/tie behavior. Pack escape links
into existing GPU record words so binding counts and retained payload sizes do
not increase. Preserve the public CPU topology and its height/child contract.
Do not retain the linear diagnostic or disable safety/authority checks.

### Next Steps

Implement and verify threaded traversal, then rerun independent geometry,
rebuild/lifecycle, camera/LiDAR and combined-load checks. The complete SENSORS.md
objective remains active; no performance or completion acceptance is claimed.

### Verified increment

The accelerated threaded TLAS/BLAS implementation passes all four focused GPU
physics/query tests, including the formerly hanging 1024-ray scans and four
physics/LiDAR frames checked against independent sphere/ground references.
Log: `/tmp/molla-physics-sensors-threaded.log` (4 passed, 16.11 seconds).
No linear fallback or mesh omission remains. Full geometry/rebuild regression
and performance acceptance are still pending. A driver/compiler cause is not
proven by this successful implementation change.

## 2026-09-25 — Verified status handoff

### Goal

Complete SENSORS.md without narrowing its acceptance requirements.

### Instructions

Use Makefile targets for validation, preserve unrelated work, and distinguish
correctness smoke results from performance acceptance.

### Discoveries

- Accelerated stackless TLAS/BLAS traversal resolves the reproduced batched-query
  timeout; the underlying driver/compiler cause remains unproved.
- Public CPU topology is unchanged. GPU escape links reuse existing record words
  and retain right-first traversal order. Temporary diagnostic switches are removed.

### Accomplished

- Final sensor suite: 371 passing test executions across 105 suites; optimized
  physics/query regression: 4 passing tests.
- Headless viewer checks: 17 passed normally and 17 with sensor timing enabled.
- GPU and CPU-only checks, scoped formatting, and book build passed.
- Combined physics/sensor smoke: 8 quick cases and all 48 full-matrix cases passed.
- Additional release smoke matrices completed: 120 LiDAR, 96 cloud, 56 camera cases.
- Final sequential validation session 60925 exited successfully and was collected.
- Validated source fingerprint:
  `195c6a3339fa39a5bd61820edf6669047dc864cdc614f88285a7fdca63f8f9bd`.

### Next Steps

- Finish the threaded-traversal evidence archive README, final log copies and checksums.
- Run repeated controlled performance measurements on the updated traversal.
- Complete remaining contract/resource, hardware visual, and Gearbox integration
  acceptance gates. Smoke timings do not establish performance parity or speedups.
- SENSORS.md remains incomplete; earlier timeout/prototype notes above are historical.

### Relevant Files

- crates/molla-sensors/src/scene_threading.rs — escape-link construction.
- crates/molla-kernels/shaders/scene_trace.wgsl — accelerated stackless queries.
- crates/molla/tests/sensor_physics_gpu.rs — independent geometry and batched regressions.
- book/src/sensors/physics_latency_benchmarks.md — measurement methods and limitations.
- benchmarks/sensors/2026-09-25-threaded-traversal/ — evidence archive in progress.

## 2026-09-25 — Repeated combined-load evidence and variance investigation

### Goal

Implement the complete SENSORS.md objective. The preceding status-only reply did
not advance implementation. This turn revalidated the source, collected terminal
session 60925 and moved to the next available performance-verification work.

### Instructions

Keep Makefile-based validation, preserve unrelated WIP and user windows, and do
not equate measured smoke execution or stable buffers with full acceptance.

### Discoveries

- Repeated current-source physics/sensor workloads now complete without the old
  batched traversal timeout. Full-matrix variability remains substantial: p95
  repeat spread exceeds 10% in 27/48 timestamped and 35/48 control cases.
- A longer-warmup two-case investigation gives GPU p95 spread 6.27%, but CPU
  spread 86.46%. This does not establish general stability or a variance cause.
- Five older sensor-benchmark nvidia-smi polling processes remained alive despite
  earlier reports that monitoring had stopped. Their stdout paths identified them
  as owned benchmark telemetry. The newly launched monitor also survived its shell
  cleanup. All six were terminated using exact PID and verified executable/stdout
  identity; a subsequent process inspection confirmed none remained. Their effect
  on measured variability is unknown. Earlier all-monitors-exited claims are corrected.
- Whole-board telemetry observes varying clocks, but one-second samples have no
  per-frame causal attribution. No driver, GPU clock or power settings changed.

### Accomplished

- Completed the threaded-traversal archive: README, final regression/check/doc logs,
  120/96/56 additional smoke outputs, input snapshots and checked artifact hashes.
- Sequential full timestamp/control session 23665 completed successfully: each
  48 cases × 3 repeats × 64 samples, totaling 18,432 measured frames, 2,304 warmups
  and 288 first frames. All output/transfer assertions passed.
- Follow-up session 60578 completed successfully: CPU/GPU, 32 articulations, ten
  steps, 100k cloud/grid plus 640 camera, device consumption, 5 repeats × 512
  samples with 128 warmups. Another 5,120 measured frames; no omitted outliers.
- Added independently checked GNU Awk analysis of raw frames: dimensions, unique
  frame/phase coverage, payloads, submissions, waits and nearest-rank quantiles.
  Both complete matrices exactly match all 144 benchmark p50/p95 summaries.
  Synthetic constant inputs verify quantiles; malformed transfers, missing frames
  and unexpected control timestamps are rejected.
- Code and build-input fingerprints stayed unchanged throughout: crate fingerprint
  `195c6a3339fa39a5bd61820edf6669047dc864cdc614f88285a7fdca63f8f9bd`.
  No Rust/WGSL, solver, assets or Gearbox source was changed in this turn.
- Documented provisional results and all measurement limits in the book and new
  `benchmarks/sensors/2026-09-25-physics-latency/` archive. Full runs were not an
  isolated-host experiment: desktop/old telemetry remained active, and lightweight
  archival work plus one book build overlapped part of the timestamped run.
- Final Makefile book build and `git diff --check` passed. Artifact checksums
  verify 71 threaded-traversal files and 27 physics-latency files. Protected engine
  and runtime-document hashes remain unchanged; no owned benchmark/telemetry
  processes remain active. Sessions 23665, 60578 and 64256 are terminal/collected.

### Next Steps

1. Investigate variability with verified process isolation and phase-correlated
   telemetry/profiling; do not silently replace or waive the full-matrix results.
2. Establish matched legacy/current-source comparisons and complete transfer/peak
   memory accounting. Earlier standalone latency archives predate threaded traversal.
3. Continue remaining S00/S01/S02/S04 numerical, capacity, resource and lifecycle
   gates; passing the present regression lane does not close their full scope.
4. Complete hardware visual acceptance and outstanding Gearbox camera links,
   moving-part visibility and transfer-overhead validation. S10 stays deferred.

### Relevant Files

- SENSORS.md — updated measurement evidence without closing incomplete gates.
- book/src/sensors/physics_latency_benchmarks.md — methods, results and limitations.
- benchmarks/sensors/2026-09-25-threaded-traversal/ — correctness/source archive.
- benchmarks/sensors/2026-09-25-physics-latency/ — raw measurements, independent
  analysis, monitored long-warmup investigation and telemetry cleanup record.

### Key Learnings

1. Longer warmup improved repeat consistency for this GPU workload, not all producers.
2. Verify the actual telemetry process exits; shell completion alone did not prove it.

## 2026-09-25 — Large-coordinate query precision fix in progress

- The preceding goal turn made progress through verified repeated measurements
  and archival. Source/process revalidation found no active benchmark or monitor.
- Independent f64 surface references with explicitly quantized input origins and
  body translations expose a shared raw/owned normal error at an 8,192 m offset.
  Both paths fail the predeclared 4e-4 normal budget with identical wrong output;
  raw/owned parity alone did not detect this.
- Primitive ray paths rotated large world coordinates separately before cancelling
  the inverse translation. Raw, owned and low-level camera shaders now subtract
  the instance position first, then rotate the nearby offset. Physics/common
  transform APIs, record layouts and buffer authority checks remain untouched.
- The first two independent large-coordinate regressions now pass through signed
  1,000,000 m offsets, including all analytic primitives, planes and mesh surfaces,
  scaled directions, inside/backside rays and live device-only poses. Distance
  budget stays 2e-4 m, normal 4e-4, world point two f32 ULPs plus 2e-4 m.
- Camera and many-instance triangle/rebuild coverage and the full regression gates
  are being extended; no broad acceptance or performance equivalence is claimed.
- Before/failing logs: `/tmp/molla-query-precision-before{,-both}.log`; initial
  passing log `/tmp/molla-query-precision-after.log`. Original shader snapshots:
  `benchmarks/sensors/2026-09-25-query-precision/before/`.

## 2026-09-25 — Verified large-coordinate precision increment

### Goal

Advance S00/S04 independent geometric acceptance while keeping the complete
SENSORS.md objective active. Previous goal work made concrete progress through
repeated measurements; this increment addresses a correctness gap instead of
repeating noisy performance runs without a new diagnostic.

### Instructions

Use Makefile tasks and patch edits, preserve unrelated source/assets and user
windows, keep buffer-authority checks, and do not overstate scoped evidence.

### Discoveries

- Raw and owned primitive queries shared inverse-translation cancellation: both
  returned the same wrong sphere normal at an 8,192 m offset. Independent f64
  reference fixtures detected what raw/owned parity missed.
- Subtracting instance position before rotation fixes the nearby-offset arithmetic.
  The standalone low-level camera had the same issue; restoring only its old
  expression reproduces a 3.25 versus 3.25047755 m plane-depth discrepancy at
  -32,768 m. The mutation was removed and the corrected source revalidated.
- Reference input origins/translations are explicitly rounded to f32 first. The
  tight local-distance tolerance does not imply submillimeter absolute world
  positions at 1,000 km or undo composed-transform/input rounding.

### Accomplished

- Fixed only sensor primitive local-ray arithmetic in three WGSL files. No
  physics, common transform API, output format, buffer/binding layout or asset changes.
- Added four independent GPU fixtures: raw/owned analytic and mesh surfaces,
  raw/owned/low-level camera plane depth, and 66 moving shared-mesh instances with
  brute-force triangle references under periodic and quality-triggered rebuilding.
  Includes exact hit/miss/shape/body identities, scaled directions, inside/backside
  rays and device-only live poses through signed 1,000,000 m offsets.
- Added a CPU oracle/ULP-budget self-check and its explicit Makefile regression lane.
  Prescribed distance 2e-4 m, normal 4e-4, surface unit-normal 2e-6; world points use
  two f32 ULPs at target magnitude plus 2e-4 m. No budget was relaxed after failure.
- Full `make test-sensors`: 376 passing executions, 108 reported suite results
  including zero-test feature lanes. All four new GPU fixtures also pass release.
- GPU/CPU-only all-target checks passed. Headless viewer lanes passed 17/17 with
  timing off and on. Scoped format checking and the Makefile book build passed.
- Release smoke matrices passed 120 LiDAR, 96 cloud/grid, 56 camera/processing and
  48 combined physics/sensor cases. Coverage checks found no missing/duplicate
  rows; physics transfer/submission/dimension accounting checks passed. These are
  one-sample execution checks, not latency distributions or performance acceptance.
- Source remained frozen through final gates and smoke runs at fingerprint
  `2a8eb7c38708e78f42643fcf66c9436f564839c0eecc9d99677a6cf86c0ad3b3`.
  Protected engine/runtime-document hashes remain unchanged. Sessions 14735,
  99626 and 48341 are terminal/collected; no owned benchmark/monitor remains active.
- Preserved before/failing/mutation/pass logs, complete source/build inputs and
  per-frame smoke data in `benchmarks/sensors/2026-09-25-query-precision/`.
  The book explicitly labels earlier performance archives as preceding this fix.

### Next Steps

1. Add independent long-local-ray and grazing/degenerate-surface fixtures; current
   large-world tests have nearby local rays below 10 m, not long-range accuracy proof.
2. Continue remaining numerical/tie, constructor/transient-capacity and full
   resource-lifecycle/allocation gates without weakening their requirements.
3. Establish controlled matched performance evidence for the updated arithmetic,
   including the earlier repeat variability and complete transfer/peak-memory limits.
4. Finish hardware visual acceptance and separately authorized Gearbox camera links,
   moving-part visibility and transfer overhead. S10 stays deferred.

### Relevant Files

- crates/molla-kernels/shaders/{raycast,scene_trace,camera_trace}.wgsl — stable
  primitive world-to-local origin arithmetic.
- crates/molla-sensors/tests/sensor_large_coordinates_gpu.rs — four GPU regressions
  and the independent CPU oracle/budget self-check.
- Makefile — executes the new host reference test in the complete sensor lane.
- book/src/sensors/contracts.md — representation boundaries, tolerances and coverage.
- book/src/sensors/physics_latency_benchmarks.md — source-version caveat.
- SENSORS.md — updated evidence and concrete remaining numerical scope.
- benchmarks/sensors/2026-09-25-query-precision/ — reproducible evidence archive.

### Key Learnings

1. Agreement between raw and accelerated queries does not establish geometric correctness.
2. Subtract nearby world positions before rotating them; input quantization remains a separate limit.

## 2026-09-25 — Long-ray arithmetic and shared camera intersections in progress

- The preceding goal turn made progress with the verified large-coordinate fix.
  New independent long-ray tests fail all three primitive paths at 1,000 m:
  sphere depth 999.2929077 versus analytic 999.3000000 m.
- Two additional regressions reproduce an ellipsoid miss becoming a false hit
  with NaN normals when max range exceeds its old positive 1e30 sentinel, and a
  finite grazing plane intersection incorrectly omitted by the 1e-9 angular cutoff.
- Finite primitive queries now conditionally advance distant rays to just before
  conservative local bounds, evaluate the short intersection, and retain its local
  surface point for normals instead of reconstructing it from the rounded full range.
  The ellipsoid miss sentinel is negative; plane parallel rejection uses exact zero.
- The low-level camera kernel now compiles the same ray_shapes WGSL as raw and
  owned rays. Removed its duplicate primitive/normal/transform implementations;
  public bindings and output channels are unchanged. This prevents further drift
  between the three intersection paths without changing physics or rendering APIs.
- All five initial GPU regressions pass without relaxed tolerances. Exact adjacent-
  f32 silhouette checks and complete regressions/performance validation are pending.
  Before/after logs: `/tmp/molla-long-rays-{before,after}.log`; source snapshots:
  `benchmarks/sensors/2026-09-25-long-rays/before/`.
- Adjacent-f32 unit-sphere silhouette rays exposed discriminant cancellation even
  at 4 m. Curved-shape discriminants now use perpendicular/cross-product terms;
  sphere/capsule/ellipsoid roots share one routine. All six focused regressions
  pass, including exact tangent/inside/outside classification without tolerance
  relaxation. Oblique long-ray chord references and full gates follow.

## 2026-09-25 — Oblique and rotated long-ray correction

### Goal

Continue the full SENSORS.md objective, specifically independent numerical
acceptance across raw rays, owned scene queries and low-level cameras.
The preceding status-only turn made no implementation progress; this turn resumed
the unfinished camera direction update and verified the next corrections.

### Discoveries

- Re-normalizing represented f32 directions can change their component ratios.
  At a 1,000,000 m origin this failed the independent oblique-sphere reference:
  999999.4375 m versus 999999.3001015253 m. Power-of-two direction scaling now
  preserves those ratios; traversal uses parameter units and explicitly converts
  accepted distances back to meters. The raw/owned/low-level camera paths pass
  all seven existing focused regressions after the camera update.
- A new rotation-invariant sphere fixture failed at a 1,000 m origin even after
  local-ray rebasing: rotating the distant origin first loses precision before
  the rebase. Finite primitive queries now first advance in world-relative space
  to a conservative rotation-independent bound, then rotate into local space.
  The same helper serves all three primitive paths. No bindings, buffers, passes,
  public APIs or physics code changed in this correction.

### Accomplished

- Eight focused GPU regressions pass, with independent f64 chord/surface
  references, represented f32 inputs, exact IDs/misses and unchanged tolerances.
- Preserved the oblique failure, direction-scaling pass and rotation failure,
  plus the intermediate shaders, under
  `benchmarks/sensors/2026-09-25-long-rays/`.
- The full sensor lane is now running; this is not yet full-suite or performance
  acceptance. Protected engine/runtime-document hashes still match.

### Next Steps

- Collect the full suite, release, feature and viewer gates; investigate failures.
- Continue far-mesh, general grazing/degenerate and tie/resource acceptance.
- Re-measure performance on this arithmetic; older latency archives are not
  current-source evidence. Preserve all outstanding S00–S09 gates and deferred S10.

### Relevant Files

- crates/molla-kernels/shaders/ray_shapes.wgsl — shared ray conditioning and roots.
- crates/molla-kernels/shaders/{raycast,scene_trace,camera_trace}.wgsl — query callers.
- crates/molla-kernels/src/tiled_camera.rs — shared camera shader compilation.
- crates/molla-sensors/tests/sensor_long_range_gpu.rs — eight independent regressions.

### Key Learnings

1. Direction normalization is not geometrically neutral after f32 rounding.
2. Long-ray conditioning must precede rotation, not only follow it.

### Full-suite feedback and metric mesh correction

- The first full-suite attempt stopped at an existing exact raw/owned camera depth
  comparison. The owned scene used metric mesh distances after direction scaling,
  but legacy per-mesh traversal still returned the unscaled parameter. No parity
  assertion or tolerance was relaxed.
- Legacy mesh traversal now uses power-of-two-scaled directions and metric range/
  output conversion; its merge reconstructs world points with the corresponding
  parameter. The previously failing three-test camera suite passes unchanged.
- Added a ninth independent focused regression for the low-level mesh kernel:
  directions scaled 1e-30/1/1e30, exact zero, metric clipping, initialized misses
  and hit/miss/hit buffer reuse. All nine focused tests pass.
- Restarted the complete sensor lane, followed sequentially by release regressions,
  the geometry-owned mesh GPU test, GPU/CPU feature checks, both headless viewer
  modes, scoped formatting and book build. These final results are pending.
- Additional files: geometry mesh raycast shader and metric API documentation,
  plus kernels raycast_merge shader/wrapper. No collision solver was changed.

## 2026-09-25 — Verified long-ray and metric mesh increment

### Goal

Implement the full SENSORS.md scope. This increment completes the current long-ray
correction and its regression/workload checks, not the entire sensor plan.

### Discoveries

- Local conditioning after rotation is insufficient for distant rays: the rotation
  can already have lost the transverse offset. Conservative world-relative ray
  advancement before rotation preserves the tested geometry.
- Metric conversion must cover legacy mesh traversal and merging as well as the
  shared scene; the unchanged exact camera-parity regression caught the omission.
- Full-range f32 distance and a nearby surface point require separate arithmetic.
  Finite primitive hits retain their conditioned local point for normal evaluation.

### Accomplished

- Full `make test-sensors` passes 385 executions across 110 reported suite results,
  including zero-test feature lanes. All nine new GPU regressions actually run.
- The nine new GPU regressions pass in release; the geometry-owned mesh GPU test
  passes separately. GPU and CPU all-target facade/sensor checks pass.
- Headless viewer tests pass 17/17 normally and 17/17 with sensor timing enabled.
  Scoped formatting, book build and `git diff --check` pass. No GUI was launched.
- All release smoke matrices pass: 120 LiDAR, 96 cloud/grid, 56 camera/processing,
  48 combined physics/sensor cases. Exact case sets match the preceding archive;
  independent count/phase/numeric checks and physics dimension/transfer/submission
  accounting pass. Each case has one first frame and one sample; not performance
  distributions, regression-budget acceptance or proof of stable timing.
- Source stayed frozen at
  `dc6d01a037f3f1ab21f86bf57e46f097c2950774e9acdea1f6a034484f76e24d`
  through final gates and smoke runs. Protected engine/runtime-document hashes
  remain unchanged. No physics solver, Gearbox, asset, tyre or soil edits here.
- Sessions 30751, 33585, 21385, 45337, 84772, 48183, 8199, 49127 and 17304 are
  terminal/collected. No benchmark or telemetry polling process remained in the
  final process inspection; no polling monitor was started this turn.
- Preserved failing/intermediate/final logs, snapshots, complete build/source
  inputs and smoke rows in `benchmarks/sensors/2026-09-25-long-rays/`.

### Next Steps

1. Continue the explicitly outstanding S01 constructor/transient/legacy metadata
   and S02 allocation/lifecycle acceptance audits; passing retained-buffer counters
   does not establish zero allocations or whole-process memory budgets.
2. Reconcile S04 numerical/tie requirements with the independent fixtures; general
   far-mesh/grazing/degenerate cases are not proved by these finite-primitive tests.
3. Establish controlled matched performance acceptance on the current shader
   arithmetic, including the prior repeat variability, transfers and peak memory.
4. Finish hardware visual/settings acceptance and separately authorized Gearbox
   camera links, moving-part coverage and transfer overhead. S10 stays deferred.

### Relevant Files

- crates/molla-kernels/shaders/ray_shapes.wgsl — common primitive roots, bounds and
  world/local ray conditioning, preserving surface points and direction ratios.
- crates/molla-kernels/shaders/{raycast,scene_trace,camera_trace,raycast_merge}.wgsl
  — metric query/traversal/merge callers.
- crates/molla-kernels/src/{tiled_camera,raycast_merge}.rs — shared shader assembly.
- crates/molla-geometry/shaders/mesh_bvh_raycast.wgsl — metric legacy mesh traversal.
- crates/molla-geometry/src/mesh_bvh_raycast_gpu.rs — metric API documentation.
- crates/molla-sensors/tests/sensor_long_range_gpu.rs — nine GPU regressions.
- SENSORS.md, book/src/sensors/contracts.md — status, budgets and limitations.
- book/src/sensors/physics_latency_benchmarks.md — older-source measurement caveat.
- benchmarks/sensors/2026-09-25-long-rays/ — reproducible verification archive.

### Key Learnings

1. Preserve represented ray direction ratios while converting parameters to meters.
2. Passing focused primitive tests does not establish mixed mesh/camera parity.

## 2026-09-25 — Processor construction payload audit in progress

### Goal

Continue the full SENSORS.md objective through the remaining S01 capacity and
constructor-allocation audit. The preceding long-ray turn made verified progress;
its shader/source archive remains an immutable earlier checkpoint.

### Discoveries

- Image, annotation, cloud and cloud-grid processors checked device buffer budgets
  before construction but did not cap their combined host mirrors and temporary
  upload conversions. A two-million-point cloud fits a 1 GiB device budget while
  its combined requested construction payload exceeds 1 GiB.
- Buffer upload packing used infallible vector growth. Its scratch now reserves
  the required capacity fallibly before conversion; existing dirty-state assertions,
  warm scratch retention and queue-staging behavior are unchanged.

### Accomplished

- Added shared ProcessorStorageUsage estimates and ProcessorStorageLimits with a
  default 1 GiB combined construction payload cap, independently preserving each
  processor's existing device-buffer budget. Explicit-limit constructors and usage
  accessors are re-exported by the facade. Checks precede host arrays and GPU work.
- Construction host arrays reserve fallibly. Upload scratch uses exact requested
  capacity reservation before conversion. These remain logical payload budgets,
  excluding allocator overhead, driver staging, pipelines/bindings, frame metadata
  and separately budgeted output/readback pools—not process or VRAM peak limits.
- Three pure estimate/limit/invalid-input tests pass. Four GPU mirror/packing checks
  and a calling-thread allocation preflight regression are added; GPU gates pending.

### Next Steps

- Collect focused GPU checks, verify facade exposure, then full sensors/storage,
  feature and viewer regressions. Reconcile results with S01 rather than claiming
  completion from this four-processor increment alone.

### Relevant Files

- crates/molla-sensors/src/processor_storage.rs — estimates, limits, fallible arrays.
- crates/molla-sensors/src/{image_processing,annotation,cloud_processing,cloud_grid}.rs
  — constructor preflight, explicit limits, usage and mirrored-array fixtures.
- crates/molla-sensors/tests/sensor_allocation_gpu.rs — measured rejected constructors.
- crates/molla-storage/src/buffer.rs — fallible upload scratch reservation.
- crates/molla-sensors/src/lib.rs, crates/molla/src/sensors/mod.rs — public exports.

## 2026-09-25 — Early scene query-budget rejection

The preceding processor-budget increment was verified (393 sensor test executions,
all 320 smoke workloads; archive `2026-09-25-processor-storage`). Its verified
summary appears earlier in this file; the former trailing in-progress entry is
historical, not the current verification state. The last goal turn made progress.

- A new allocation regression reproduces late scene query-budget checks: rejected
  requests allocated 21,897,134 bytes for 65,537 spheres, 795,054 bytes for a
  65,537-triangle mesh and 3,168,418 bytes for a 257×257 heightfield.
- Shape/plane and unique mesh/triangle/heightfield packed payloads are now checked
  before identity tables, model copies, terrain conversion or GPU uploads. Exact
  raycast BVH node counts are computed without building its arrays; the planner
  shares the builder's leaf-size constant and has an independent tree-count test.
- Converted geometry is checked again before uploads, including convex-hull output.
  Actual tree node counts must match preflight before atlas packing/upload.
- The unchanged rejection regression now measures 37 bytes for primitives and
  89 bytes for mesh/terrain errors. Exact-budget shared-mesh acceptance and
  one-byte-under rejection were added to the existing atlas GPU fixture.
- Convex-hull construction workspace, identity/model host storage, refit resources
  and driver staging remain outside the packed query budget. This is not full
  constructor-memory acceptance or a warmed-query speedup.
- Full sensors, release preflight, feature and headless viewer gates are running.
  Sources: scene.rs, scene_geometry.rs, mesh_bvh_raycast_gpu.rs,
  sensor_allocation_gpu.rs and Makefile (geometry validation added to sensor gate).

## 2026-09-25 — Verified early scene query-budget rejection

### Goal

Continue full SENSORS.md implementation through scene-construction capacity checks.
This increment is verified; the full goal remains active, not complete or blocked.

### Discoveries

- Median-split BVH node counts depend only on triangle count and leaf size. Exact
  planning rejects insufficient atlas budgets without building large tree arrays.
- Packed-query budgets do not cover convex-hull workspace or identity/model host
  storage. Those remain separate construction-memory work, not implicitly capped.

### Accomplished

- The new before/after allocation regression passes in debug and release: rejected
  primitive/mesh/terrain scenes request 37/89/89 bytes instead of
  21,897,134/795,054/3,168,418 bytes. No tolerance or allocation threshold was relaxed.
- Exact-budget shared-mesh acceptance, one-byte-under rejection and node-count
  references pass. The Make sensor lane now includes geometry validation tests.
- Full sensors pass 397 test executions across 111 reported suites (including
  empty feature lanes). GPU/CPU all-target checks, 17/17 headless viewer tests in
  both modes, scoped formatting, book build and diff whitespace checks pass.
- All 320 release workload smoke cases pass, with exact case-set/phase coverage
  and physics dimensions/transfers/submissions validated independently. One first
  frame plus one sample per case is not performance acceptance or repeatability.
- Source remained `98335bf138e317427e78893e2a58a6053d01b0ba658f96edebe4d0bf04d50c1f`.
  Protected engine/runtime-doc hashes match. No shader, physics solver, Gearbox,
  asset, tyre or soil changes. No GUI or telemetry monitor was launched.
- Sessions 53627, 22525, 73442, 29586, 4693 and 63197 are terminal/collected;
  final process inspection found no remaining benchmark or telemetry polling job.
- Evidence: `benchmarks/sensors/2026-09-25-scene-preflight/`.

### Next Steps

1. Complete remaining constructor/scene host-workspace accounting and S02 warm
   allocation/lifecycle gates; retain the full S01–S09 scope and deferred S10.
2. Reconcile numerical/tie acceptance with independent fixtures and establish
   controlled matched performance, not one-sample smoke comparisons.
3. Complete hardware visual/settings checks and remaining authorized Gearbox
   camera/moving-part/transfer integration.

### Relevant Files

- crates/molla-sensors/src/scene.rs — budget preflight before model copies/uploads.
- crates/molla-sensors/src/scene_geometry.rs — shared-geometry and tree-size planning.
- crates/molla-geometry/src/mesh_bvh_raycast_gpu.rs — exact node-count helper/test.
- crates/molla-sensors/tests/sensor_allocation_gpu.rs — rejected-scene allocations.
- Makefile — geometry validation in the complete sensor gate.
- SENSORS.md, book/src/sensors/scene.md — current evidence and budget boundaries.
- benchmarks/sensors/2026-09-25-scene-preflight/ — source and verification archive.

### Key Learnings

1. Checking a scene budget after copying geometry still permits large failed allocations.
2. Shared mesh instances must be counted once in exact preflight payloads.
