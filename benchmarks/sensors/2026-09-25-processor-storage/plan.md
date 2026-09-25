# GPU Sensors Implementation Plan

Date: 2026-09-21. Planned against commit `d2369a9` plus the current working tree.

**Status: IMPLEMENTATION STARTED. Audit findings below describe the pre-implementation baseline, not current guarantees. See `SENSORS_PROGRESS.md` for execution evidence.**

This is the single handoff document for the sensor audit, originally named `SENSROS.md`. Checkboxes describe future work unless explicitly marked complete with verification evidence.

## 1. Objective and boundaries

Make Molla's existing GPU sensors correct, reusable, efficiently batched, and suitable for mesh-rich machine scenes. Prioritize parallel LiDAR, then shared camera tracing and the remaining sensor processing. Preserve CPU Featherstone physics as a supported producer of sensor state.

The problem is not an absence of GPU kernels: intersections, IMU, contact aggregation, transforms, and camera tracing already execute on the GPU. The missing pieces include device-resident orchestration, complete scene queries, lifecycle handling, and measured end-to-end performance.

### Scope

- `crates/molla-sensors/`: APIs, scheduling, processing, tests, benchmarks.
- Sensor-specific kernels in `crates/molla-kernels/src/` and `shaders/`.
- `crates/molla-geometry/`: mesh queries and scene acceleration needed by sensors.
- Narrow additions to `crates/molla-storage/` for reusable asynchronous/range readback and buffer ownership, only where needed.
- Narrow `crates/molla-sim/` snapshot/revision adapters if existing accessors are insufficient.
- `crates/molla/` public exports and feature wiring where required.
- Sensor examples in `bin/molla_viewer/`, sensor documentation in `book/src/sensors/`, and focused Makefile verification targets.
- Gearbox integration is specified below but requires a separate, explicit execution approval before editing that repository.

### Hard boundaries

- Do not change soil, tyres, vehicle dynamics, Rapier, authored USD assets, or unrelated rendering.
- Do not move Featherstone physics to GPU as a prerequisite.
- Do not modify or complete `FULL_COMPACT.md` as part of this plan.
- Preserve existing uncommitted work. At planning time, `Makefile`, viewer `main.rs`, `viewer_readback_tests.rs`, `FULL_COMPACT_PROGRESS.md`, `TYRE_CONTACT_VALIDATION.md`, and `FULL_COMPACT.md` already had changes/untracked content.
- Keep legacy synchronous sensor APIs working while adding asynchronous/device-resident alternatives. Do not silently change output units, frames, or return semantics.
- Never disable buffer-authority checks to suppress synchronization errors.
- Use Makefile tasks for builds, checks, tests, and benchmarks. Use patch tools, not Python, for edits.
- Comments describe behavior only; keep code comments within the user's 20% limit. If commits are requested, use unsigned, title-only Conventional Commits of at most 50 characters, without attribution trailers. Do not push automatically.

## 2. Current state and evidence

This section records the initial audit before implementation, not the current
feature status. Section 4 and `SENSORS_PROGRESS.md` track implementation and verified
gates. Paths below are repository-relative; historical line numbers may drift.

| Area | Existing implementation | Limitation / evidence |
|---|---|---|
| LiDAR | Terrestrial grids, aerial scans, divergent sub-rays | CPU ray layout in `crates/molla-sensors/src/lidar.rs:424-455,905-947`; first/last returns select among sub-rays, not multiple intersections on one ray (`:292-305`) |
| Ray intersections | GPU invocation per ray, workgroups of 64 | Primitive kernel loops over all shapes in `crates/molla-kernels/shaders/raycast.wgsl:308-363` |
| Static meshes | GPU traversal of per-mesh BVHs | Separate dispatch and three blocking downloads per mesh; CPU nearest-hit merge in `crates/molla-sensors/src/raycast.rs:352-465`; moving meshes skipped |
| Readback | Persistent sensor output buffers | Primitive raycast downloads four buffers; `crates/molla-storage/src/buffer.rs:311-338` allocates staging and blocks separately per download |
| IMU | GPU specific force and gyro with mounting offsets | `crates/molla-sensors/src/imu.rs:173,196` reads host velocity history even if GPU is authoritative; outputs are downloaded immediately |
| Contacts | GPU per-body contact aggregation | Four immediate downloads in `crates/molla-sensors/src/contact.rs:131-134`; count/depth/normal/centroid, not force/impulse measurements |
| Transforms | GPU-batched world/relative poses | Four immediate downloads in `crates/molla-sensors/src/frame_transform.rs:169-172` |
| Cameras | GPU camera-by-pixel batching, five channels | Primitive-only linear scene traversal in `crates/molla-kernels/shaders/camera_trace.wgsl:399-423`; shared intrinsics/resolution per batch |
| Camera optics | GPU thin-lens rays | CPU accumulation and repeated color readback in `crates/molla-sensors/src/tiled_camera.rs:443-570` |
| Processing | Image post, point-cloud transforms/filtering, annotations | Host implementations in `camera_post.rs`, `pointcloud.rs`, `dataset.rs` under `crates/molla-sensors/src/` |
| Gradients | CPU sphere/plane differentiable-query helper | `raycast_adjoint.rs` uses finite differences; not an existing GPU analytic adjoint |
| Verification | GPU fixtures and Criterion benchmarks | GPU tests are ignored unless explicitly selected; current raycast benchmark covers only 1,000/10,000 rays against 100/1,000 primitive shapes |

The sensors crate defaults to no GPU feature; `--features gpu` enables its GPU code. The Molla facade enables GPU by default. GPU modules can still contain CPU work: feature gating is not proof of device residency.

### Correctness anchors

```rust
// crates/molla-sensors/src/imu.rs, ImuSensor::update
let host_now = state.body_qd.host()?.to_vec();
```

`Buffer::host()` rejects device-authoritative data. The IMU history path must work after a real GPU solver step, not only fixtures that upload host-edited velocities.

```wgsl
// crates/molla-kernels/shaders/raycast.wgsl
let rd = rd_raw / rd_len;
```

The primitive kernel normalizes rays, but `crates/molla-geometry/shaders/mesh_bvh_raycast.wgsl:79-80` uses the raw direction. Host merging compares their hit distances directly. Non-unit directions therefore do not currently have a consistent mixed-geometry distance contract.

Other confirmed gaps: unchecked scan-size multiplication in `lidar.rs`; a cone kind without a matching primitive LiDAR intersection branch; caller-managed mesh cache invalidation. Review fixed traversal-stack capacity during acceleration work; overflow reachability was not established by this audit.

## 3. Target architecture

```text
CPU physics snapshot OR GPU physics state
                 |
       revisioned sensor scene + poses
                 |
   simulation-time scheduler / sensor batches
                 |
    ray generation / shared scene queries
                 |
  hit reduction / sensor outputs / processing
                 |
       device-resident result handles
           /                   \
 next GPU consumer       optional async readback
                                |
                       viewer / telemetry / export
```

1. **Scene ownership:** static mesh geometry uploaded once; instance transforms updated independently. Track topology/geometry revisions separately from pose revisions. Resolve stable application handles to dense GPU indices per snapshot.
2. **Execution:** separate command encoding from queue submission and collection. Compatible sensor work shares an encoder/submission where dependencies permit; no hidden downloads inside device-resident APIs.
3. **Results:** typed device result views carry active count, layout, sensor identity, timestamp, scene revision, and validity. Their lifetime must prevent reuse while GPU consumers or readback still reference them.
4. **Collection:** bounded reusable staging ring, requested channels/ranges only, explicit ready/pending/error states. No unbounded queues, indefinite waits in the nonblocking path, or silently dropped requested results.
5. **Compatibility:** existing host-returning calls wrap the new path and explicitly wait. CPU-only crate feature configurations must remain buildable; this does not promise CPU fallbacks for every GPU sensor.
6. **Scheduling:** use simulation time, not render-frame count; support distinct sensor rates, pause, reset, and scene replacement. Preserve per-sensor configuration rather than collapsing settings across mounts.
7. **Frames and units:** document meters, seconds, radians, coordinate conventions, local mounts, timestamps, specific force versus acceleration, and miss/invalid-ray semantics.

## 4. Execution order and status

Effort is relative scope, not a calendar promise. Update status with evidence, not merely code completion.

| ID | Work package | Priority | Effort / risk | Depends on | Status |
|---|---|---|---|---|---|
| S00 | Baselines and contracts | P0 | M / low | — | IN PROGRESS |
| S01 | Correctness fixes | P0 | M / medium | S00 | IN PROGRESS |
| S02 | Device-resident API and asynchronous collection | P1 | L / high | S01 | IN PROGRESS (all sensor/processing producers and heterogeneous camera manager have direct outputs; full lifecycle/allocation/performance gates pending) |
| S03 | GPU LiDAR generation and hit reduction | P1 | L / medium | S02 | CORE PIPELINE VERIFIED (shared-query lifecycle/resources and performance acceptance remain) |
| S04 | Shared accelerated scene and moving meshes | P1 | L / high | S02, S03 | IN PROGRESS (owned geometry, stable handles, GPU refits, unified traversal, bounded periodic and optional GPU-quality-triggered spatial rebuilding integrated; complete reference/transform audit and full performance acceptance pending) |
| S05 | IMU, contacts, transforms and scheduling | P1 | M / medium | S02 | CORE VERIFIED (device outputs, independent rates and reset contracts; S02 resource and S08 viewer gates remain) |
| S06 | Shared camera tracing and GPU optics | P1 | L / high | S04 | CORE VERIFIED (shared visibility, GPU optics, heterogeneous groups and device-only image-processing chains; shared S02/S04 and S08 gates remain) |
| S07 | GPU image/cloud processing and annotations | P2 | L / medium | S03, S06 | CORE VERIFIED (images, clouds, spatial bins, annotations and selected export; full performance acceptance and shared lifecycle gates pending) |
| S08 | Viewer, documentation and performance acceptance | P1 | M / medium | S03–S07 | IN PROGRESS (seven asynchronous preview migrations and combined scene 144 pass headless lifecycle tests; visual and performance acceptance pending) |
| S09 | Gearbox integration | Gated | L / high | S08, explicit approval | AUTHORIZED 2026-09-25 (user: Gearbox runs on this physics); IMU/LiDAR sensor links integrated and verified live; transfer-overhead measurement and camera links remain |
| S10 | Advanced sensing extensions | Optional | L / high | Core acceptance, separate decisions | DEFERRED |

### S00 — Establish reproducible baselines and contracts

- [x] Read current APIs/tests; inventory supported geometry for each sensor. `book/src/sensors/contracts.md` distinguishes owned/raw/low-level camera/CPU-adjoint geometry, unsupported conversions/omissions, and non-query sensors. Raw omissions remain explicitly documented; owned scenes reject unsupported kinds by default.
- [x] Recheck working-tree drift and preserve unrelated changes before editing. See `SENSORS_PROGRESS.md`.
- [x] Run the existing sensor GPU suites through the commands below. Record failures before modifications, adapter/backend, driver, compiler, build profile, scene sizes, and revision. Baseline: 35 GPU tests passed; evidence in `SENSORS_PROGRESS.md`.
- [ ] Capture existing end-to-end benchmarks, including uploads/readback. Distinguish warm reuse from first-use BVH/pipeline creation. `bench_sensor_latency` and `bench_camera_latency` record raw per-frame CPU completion/encoding, GPU timestamps, transfers and calling-thread allocations for the defined 120-case LiDAR, 56-case camera/image/annotation and 96-case LiDAR/cloud/grid matrices, separating setup/first/warmup/samples and pose/topology updates. `bench_sensor_physics_latency` adds 48 contactless CPU/GPU physics-plus-sensor cases: 18,432 timestamped/control samples and a 5,120-sample longer-warmup investigation on the threaded traversal. Full-matrix repeat variability, matched legacy baselines and controlled performance acceptance remain; see the book's latency measurement pages for accounting and environment limits.
- [x] Add deterministic mixed primitive/static-mesh reference fixtures, sensor mount/coordinate fixtures, and a small CPU brute-force triangle oracle independent of the GPU traversal. Scaled rays, transformed-mesh oracle and 65-mount IMU regression pass.
- [ ] Record numerical tolerances before refactoring: hit/miss and IDs exact away from ties; distance/position/normal tolerances tied to f32 scale; deterministic tie policy for equal-distance hits.

Current frame/miss/tie rules and fixture-specific numerical budgets are recorded
in `book/src/sensors/contracts.md`. Independent rigid-primitive/baked-mesh,
camera radial-depth and LiDAR handedness fixtures supplement the triangle oracles.
They exposed and fixed hidden capsule-cap intersections and the CPU adjoint's
nonmetric scaled-ray distances. Full numerical matrix/baseline acceptance remains
open; source-path parity is not an independent geometric reference.

**Gate:** existing suite results and baseline measurements recorded; new reference fixtures pass; failures are explicitly resolved or scoped, not silently excluded.

### S01 — Fix correctness before optimizing

Files: `crates/molla-sensors/src/{imu,lidar,raycast}.rs`, related raycast/IMU kernels, and corresponding integration tests.

- [x] Make IMU previous-velocity history a device-to-device snapshot. Evaluate against the previous sample, then advance history in the same ordered command stream. Define first-call, reset, body-count change, scene replacement, and invalid/nonpositive `dt` behavior. GPU-integrated, reset and multi-mount regressions pass.
- [x] Unify public ray distance as meters for all supported nonzero directions. Normalize consistently for primitive and mesh traversal and point reconstruction. Define finite/zero-direction handling without allowing stale output slots. Scaled mixed-geometry and invalid-input regressions pass.
- [ ] Replace unchecked scan/ray/pixel sizing with checked arithmetic. Validate capacity, binding sizes, dispatch limits, and allocation budgets before allocation or dispatch; chunk workloads when appropriate. Scan metadata and GPU constructor dimensions already have overflow/device checks. Shared ray construction now additionally checks a default 1 GiB aggregate host/device ray-array budget before allocation, with pure `RayStorageUsage` estimation and explicit `RaycasterLimits` overrides through ray/LiDAR/batched-LiDAR/tiled-camera constructors. Camera construction separately checks a default 1 GiB combined fixed-array budget (rays, all channels, poses/packing and reserved aperture scratch), with `CameraStorageUsage` estimates and `TiledCameraLimits` overrides. Heterogeneous camera groups check the summed payload before any group allocation and preserve custom limits on transactional reconfiguration. Boundary tests cover exact/under-budget, device-binding/dispatch limits, huge metadata and real allocated payload parity. IMU/frame mounts reject non-finite, f32-overflowing or non-unit poses before mount-buffer allocation. These are not whole-sensor/process peak-memory caps; remaining constructor/transient-allocation and legacy metadata audits stay open.
- [x] Implement and test cone intersection parity or reject unsupported cone queries explicitly until implementation lands. Completion of supported-primitive parity requires actual cone support, not a silent miss. Side/base/inside/apex/generator-parallel and miss references pass.
- [x] Owned scene revisions and exact raw-model mesh comparisons prevent stale cache reuse after geometry/model replacement. Transactional replacement and pending old snapshots are covered in `sensor_scene_gpu`; see progress for GPU-environment recheck status.
- [x] Add regressions: scaled rays against mixed geometry and range clipping (`raycast_trimesh_gpu`); inside-cone, cone hit/miss and degenerate input rays (`sensor_raycast_gpu`); terrestrial/aerial overflow metadata (host tests in `lidar.rs`); device-stepped IMU/reset and 65 mount lever arms (`sensor_imu_gpu`). These fixtures and the added ray-budget/cloud-overflow checks pass in the 300-execution full sensor regression on 2026-09-24. The broader allocation-budget audit above remains open.

**Gate:** `sensor_imu_gpu`, `sensor_raycast_gpu`, `raycast_trimesh_gpu`, scan-grid and aerial suites pass, including new regressions. CPU metadata-validation tests reject extreme dimensions without huge allocations. Never relax `Buffer` dirty-state assertions.

IMU/contact/frame constructors also enforce default 1 GiB combined retained
host/device array budgets through `SensorArrayLimits`, with explicit overrides
and pure `SensorArrayStorage` estimates. IMU includes full-body velocity history;
frames include the optional reference slot; empty contacts have zero array payload.
These remain array budgets, excluding caller-owned inputs, bindings, transient
packing/staging and separately budgeted collection/output pools. The remaining
constructor/transient-allocation and legacy metadata audit is still open.

Batched LiDAR now checks a combined retained-array cap through `LidarBatchLimits`
and `LidarBatchStorage`, including ray storage, all three return arrays and reserved
descriptor/range capacity before allocation. Legacy terrestrial/aerial `cast`
checks a separate per-call CPU layout cap (`LidarLayoutLimits`, default 1 GiB);
metadata exposes allocation-free byte estimates and layout vectors reserve
fallibly after count/capacity validation. Collected hits, staging and caller-owned
data are outside that layout cap. Full capacity/legacy audit remains open.

Image, annotation, cloud and cloud-grid processors now separately preflight a
default 1 GiB combined construction payload cap through `ProcessorStorageLimits`.
`ProcessorStorageUsage` reports retained host/device arrays plus the largest
sequential upload conversion. Existing device-only config budgets still apply;
explicit-limit constructors preserve caller control. Host arrays and upload
packing reserve fallibly. Pure boundary tests and GPU mirror/packing checks pass;
rejected constructors request only 116 calling-thread bytes for the error message
in the measured fixture, and deliberately moving preflight after image allocation
makes that regression fail at 1,509,413 requested bytes. These are not whole-process
or VRAM peaks: allocator overhead, driver staging, pipeline/binding and per-frame
metadata, scene construction, and output/readback pools remain separate. Full S01
constructor/transient/legacy audit and S02 allocation/lifecycle gates stay open.

### S02 — Separate encoding, submission and readback

Files: sensor wrappers, sensor kernel dispatch wrappers, narrowly scoped storage helpers; proposed tests `sensor_device_pipeline_gpu.rs` and `sensor_readback_gpu.rs`.

- [x] Add composable encode/device-result APIs without breaking existing convenience methods. Match existing `Result`/`Error` conventions, e.g. `Raycaster::cast`. Public `SensorCommandBatch` composes IMU/contact/frame producers, owned-scene host-ray queries, GPU-generated batched LiDAR, tiled/heterogeneous cameras, images, annotations, cloud construction/filtering and cloud grids into direct leased slots, and retains submitted-frame GPU consumers. Scoped typed pending inputs support custom stages and built-in image/annotation/cloud/grid chains in the same submission, including LiDAR-to-cloud and camera-to-image/annotation processing. Annotation/cloud dependencies are deduplicated so 1024 logical jobs can share pending outputs. IMU history commits at submission and rolls back on cancellation; producer/input lifetimes and opaque receipts are borrow-checked. Heterogeneous `CameraBatch` validates all groups/budgets/availability before encoding, preserves per-image IDs/slices and composes with image/annotation stages once; numerical/limit/lifecycle regressions pass in `sensor_camera_manager_direct_gpu`. Full S02 allocation/performance gates remain separate.
- [ ] Retain output, uniform, bind-group and staging resources across steady-state frames where layouts/capacities permit. Rebuild only on actual capacity/layout change. Query, IMU/contact/frame, batched LiDAR, camera, image, annotation, cloud and grid paths now retain bindings/uniforms; numerical/replacement and warm preparation-count regressions cover these paths. Batch/consumer/commit metadata and completion tokens now use bounded inline storage; camera layouts are shared and pose packing retains scratch. The calling-thread audit in `sensor_allocation_gpu` measured 153 → 145 allocations per warm camera/image/annotation frame, not zero; wgpu submission and remaining producer metadata/staging costs are still present. Full allocation/performance and producer-lifecycle gates remain.
- [x] Provide grouped asynchronous readback of requested active ranges/channels. Bounded rings expose backpressure/cancellation, terminal mapping/poll errors and the engine's sticky device-loss signal. Real logical-device destruction/callback tests cover outstanding/empty requests, snapshots, unsubmitted consumers and explicit fresh-engine reconstruction. Reset does not revive failed resources; physical driver-fault/hang recovery is not claimed. See `sensor_readback_gpu`, `sensor_device_loss_gpu` and device-lifecycle documentation.
- [ ] Bound in-flight batches and prevent output/staging reuse until GPU completion and CPU release. Reject cross-device handles. Direct auxiliary, owned-scene ray, batched LiDAR, tiled/heterogeneous camera, image, annotation, cloud and grid slots join copied snapshots and completion-tracked consumers: 1–64 fixed-budget slots, bounded batch references/temporal commits, generation reset, CPU/GPU leases and canonical Arc device provenance. External device-source image/annotation/cloud/grid inputs share the 64-input budget with submitted-frame consumers and retain source occupancy through completion. Heterogeneous camera pools enforce an aggregate byte budget, full-request backpressure, 64 shared output entries, historical images across reset/drop/reconfigure, cancellation/poison and known device loss; broader lifecycle audit and complete gates remain.
- [ ] Test device-only chaining, no host access to device-authoritative data, zero-sized batches, ring saturation, scene reset with pending results, capacity growth, and explicit legacy synchronous calls.

**Gate:** new suites pass; instrumentation asserts zero host downloads/waits in the encode-only path and no resource allocations in a warmed fixed-capacity pipeline. Host collection transfers only requested active data. The synchronous API matches its reference outputs.

### S03 — Make LiDAR device-resident end to end

Files: `lidar.rs`, `raycast.rs`, raycast kernels; add ray-layout and hit-reduction kernels and `lidar_pipeline_gpu.rs`.

- [x] Generate terrestrial grids, aerial origins, deterministic cone/sub-ray patterns on GPU. `LidarBatch` preserves row/column/pulse/sub-ray order and descriptor ranges; CPU-oracle regressions pass.
- [x] Batch multiple mounts with separate extrinsics, scan settings, ranges and identifiers. Mixed-pattern and 70-mount tests preserve independent metadata/settings; oversized batches are rejected before generation.
- [x] Merge primitive/static-mesh candidates on GPU with ordered per-mesh passes. Device-query, mixed-occlusion, tie-ordering and snapshot-reuse regressions pass; moving meshes remain S04. See `SENSORS_PROGRESS.md`.
- [x] Compute first/last ray indices and valid-sub-ray counts per pulse on GPU, preserving tie order. These are not penetration/transmission returns along one laser path.
- [x] Expose device channels and optional grouped active-range collection, independent of mesh count. Fixed-capacity batching preserves IDs/order; excess requests are explicitly rejected rather than silently truncated or chunked.
- [x] `lidar_pipeline_gpu` covers old/new parity, empty/miss scans, 70 mounts, K=0/1/2/3/4/7/1024, capacity/error handling, snapshots, and primitive/mesh occlusion. See `SENSORS_PROGRESS.md` for limits and evidence.

**Gate:** `lidar_pipeline_gpu` plus existing LiDAR suites pass. Counters show no per-mesh host readback or host hit merging, and no host ray-vector construction in the new GPU-layout path. Final scan matches CPU references within recorded tolerances.

### S04 — Share acceleration and support moving machine geometry

Files: `crates/molla-geometry/`, raycast kernels/wrappers, a proposed `crates/molla-sensors/src/scene.rs`, tests `sensor_scene_gpu.rs`.

Owned TLAS/BLAS queries now use threaded escape links instead of per-ray private
stacks. Existing record widths and right-first traversal order are preserved;
CPU topology still exposes heights. This resolves the focused batched-physics
query timeout regression; the full sensor lane now passes 393 executions, while broader
numerical/resource/performance acceptance remains open.

- [x] Store reusable mesh BVHs in local coordinates, shared across instances and query sensors by `SensorScene`. Live body-pose changes update queries without vertex/BVH rebuilds; top-level bounds/refits remain below.
- [ ] Add top-level acceleration over supported finite primitives and mesh instances; handle infinite planes separately. Owned-scene ray/LiDAR/camera paths now use `SceneAcceleration` refits followed by one unified primitive/shared-BLAS traversal dispatch, with separate infinite-plane iteration. Static meshes occupy one packed immutable atlas shared by all instances/sensors. Pose-only updates reuse topology; scene identity/revision changes rebuild the query hierarchy. Bounded periodic GPU Morton/radix rebuilding and optional quality-triggered decisions are integrated, with independent camera-group policies and preflight budgets. Quality checks reduce normalized internal-node area on GPU and gate sorting/reordering/commit using indirect dispatch, with no decision readback; stable layouts skip reordering and scrambled GPU-only poses trigger repair. Encoded checks and actual GPU decisions are distinguished. Refit/query fixtures cover independent references, ties, cancellation, alternating pose buffers and retained output ownership. Full performance-matrix acceptance remains pending; both periodic and quality-check overhead must be measured, not assumed faster. See `SENSORS_PROGRESS.md` and the scene documentation for verification and benchmark evidence.
- [ ] Transform rays/normals correctly through instances and preserve metric distances. Define supported scaling explicitly; reject unsupported singular/nonuniform transforms rather than returning plausible wrong hits. Rigid-transform and baked mesh-scaling contracts have independent references. The large-coordinate audit found and fixed inverse-translation cancellation shared by raw/owned primitive queries and the low-level camera: subtract instance position before rotation. Independent surface/triangle/plane fixtures now cover signed offsets through 1,000,000 m with explicitly quantized inputs and fixed local-distance/normal plus ULP world-point budgets. Owned-scene GPU invalid-pose/bounds flags propagate through refits/rebuilds to exact invalid ray/LiDAR/camera sentinels, and synchronous `cast_scene` errors; valid refits recover without changing historical frames. Raw-model APIs retain their valid-pose precondition. Nine focused long-ray regressions now pass for axis/oblique/rotated primitives, exact sphere silhouettes, finite grazing planes, ellipsoid misses and low-level mesh direction scaling. Raw/owned/low-level camera primitives share conditioned intersections; legacy mesh queries now convert parameters to metric distances too. The full sensor lane passes 393 test executions; general long-mesh/grazing/degenerate-surface acceptance remains open; see `sensor_long_range_gpu`, `sensor_large_coordinates_gpu`, `sensor_query_validity_gpu` and `book/src/sensors/contracts.md`.
- [x] Support body-attached triangle meshes. Opaque `SensorBodyHandle`/`SensorShapeHandle` identities survive explicit `replace_model_with_remap` dense remapping; omitted objects retire and new slots never recycle their identity. Retained `SceneIdentitySnapshot` tables resolve historical stamped results after replacement/drop. Four host identity tests and six `sensor_scene_gpu` fixtures cover remap/delete/replacement failures, historical readback, moving body-attached geometry and independent hit references. Plain `replace_model` assigns new identities rather than guessing object continuity. Acceleration/refits and broader S04 acceptance remain open.
- [x] `SensorScene` triangulates heightfields with the simulation grid diagonal, explicit vertex/triangle budgets and replacement-based updates. No soil simulation added.
- [x] Owned-scene capability matrix includes Triangle/Heightfield/ConvexHull conversion, a dilated-cylinder approximation of RoundCylinder, and default rejection or explicit diagnostics for degenerate hulls and SDF. Legacy raw-model omissions remain documented.
- [x] Mesh upload validates tree topology/coverage/bounds and retains the depth-64 compatibility limit for legacy stack-based queries. Owned queries use threaded traversal. Host boundary tests accept depth 64 and reject 65; geometry GPU regression passes.
- [ ] Test moving meshes, repeated instances, occlusion, shape IDs, pose-only updates, geometry edits, removals, negative/large coordinates, and near-equal hit ties against independent references.

**Gate:** `sensor_scene_gpu` and all raycast suites pass; changing a pose changes hits without rebuilding mesh geometry. Warm fixed-topology frames contain no per-instance BVH reconstruction. Many-instance benchmarks include acceleration build/refit costs, not just traversal.

### S05 — Finish other sensor pipelines and scheduling

Files: `imu.rs`, `contact.rs`, `frame_transform.rs`; proposed scheduler module and `sensor_scheduler_gpu.rs`.

- [x] Expose device-resident IMU/contact/transform outputs through S02; keep synchronous wrappers for callers that need them. Device-only and grouped synchronous collection regressions pass; see `SENSORS_PROGRESS.md`.
- [x] Schedule independent sensor rates from simulation timestamps. Specify first sample, bounded catch-up, skipped samples, pause, reset and scene replacement. Do not fabricate intermediate-time state when no interpolation/history exists. Integer-time scheduler and `sensor_scheduler_gpu` pass; caller dispatch/collection failures are separate from deadline skips.
- [x] Batch independent mount configurations and relative-frame targets. Define snapshot consistency when CPU physics is the producer. 65-mount IMU and 70-target frame tests pass; coherent CPU upload and ordered GPU snapshot contracts are documented in `book/src/sensors/scheduler.md`.
- [x] Preserve IMU gravity/specific-force, angular acceleration and lever-arm semantics, with multi-mount and GPU-stepped tests. Existing seven IMU regressions and independently scheduled free-fall sampling pass.
- [x] Describe contact output accurately as geometry/aggregation. Force sensing is optional S10 and must not infer forces from penetration alone. Contact API/book specify unweighted normals and penetration, not measured force.

**Gate:** IMU/contact/frame suites and `sensor_scheduler_gpu` pass; multi-rate timestamps are deterministic under different render frame rates; no implicit collection occurs in the device-only path.

### S06 — Reuse scene queries for cameras and move optics accumulation to GPU

Files: `tiled_camera.rs`, camera kernels, shared query code; existing camera/optics tests plus `sensor_camera_scene_gpu.rs`.

- [x] Route camera visibility through S04 while preserving five output channels, near/max-distance semantics, normals and shape IDs. Raw cameras use shared primitive/TriMesh queries; owned `dispatch_scene` also renders Triangle/Heightfield geometry. Independent moving-mesh/terrain, occlusion, channel/revision, strict range and empty-scene fixtures pass; see `sensor_camera_scene_gpu` and progress evidence.
- [x] Accumulate aperture color samples in GPU floating-point storage and resolve once. Packed-byte averaging/rounding matches the CPU reference; no intermediate readback. Color space remains unchanged. See `sensor_camera_device_gpu` and progress evidence.
- [x] Produce geometric channels from the center sample without repeated CPU readback/restoration. Non-center passes write COLOR only; masks without COLOR run only the center and skip accumulation. Numerical/channel-isolation regressions pass.
- [x] Preserve homogeneous batching, then group heterogeneous cameras by compatible layout or add per-camera intrinsics. `CameraBatch` groups by full configuration with independent IDs/poses/ranges, validated limits and transactional reconfiguration. Host grouping/budget tests and all-channel GPU parity against independent cameras pass; see `sensor_camera_batch_gpu` and progress evidence.
- [x] Test channel masks, one/many cameras, focal behavior, occlusion, batching isolation, and device-only camera-to-processing chains. Existing camera/optics/scene/batch suites pass; `sensor_processing_gpu` covers heterogeneous camera → processing → copied snapshot → processing without intermediate host collection. `sensor_camera_manager_direct_gpu` additionally covers heterogeneous camera → image/annotation stages in one submission without intermediate snapshot copies, including independent image IDs/ranges and warmed slot reuse.

**Gate:** camera, optics and `sensor_camera_scene_gpu` suites pass. No per-aperture host downloads occur. Primitive results remain within baseline tolerances; mesh depth/IDs match independent fixtures.

### S07 — GPU post-processing, clouds and annotations

Files: `camera_post.rs`, `pointcloud.rs`, `dataset.rs`, new processing kernels; tests `sensor_processing_gpu.rs`.

- [x] Port image adjustments to composable GPU passes: exposure, gamma/sRGB conversion, tone mapping, saturation, and white-balance reductions. `ImageProcessor` runs independent bounded packed-image pipelines in one submission, with per-stage device gains and retained resources. CPU APIs remain references; scalar/reduction/chain/lifecycle tests pass. Operation benchmarks are provided; uncontended performance acceptance remains open.
- [x] Add cloud construction, transforms, filters and compaction on device. `CloudProcessor` consumes terrestrial/aerial hits or packed records, preserves full-width sensor/sample identity and grid/source metadata, and runs stable global prefix compaction. CPU-oracle, lifecycle, invalid-input and million-point boundary tests pass. One explicitly selected scalar is supported; arbitrary CPU scalar maps remain host convenience data. See `sensor_cloud_gpu` and progress evidence.
- [x] Add spatial binning and reductions with bounded memory and overflow reporting. `CloudGrid` uses globally bounded CSR membership without a per-cell cap, retained prefix/scatter/reduction passes, cell bounds/means and explicit rejection/overflow diagnostics. Independent CPU membership/reduction, million-point dense-cell, large-cell-grid and device-chain/lifecycle tests pass. Cell-local order and f32 reduction rounding are not deterministic; performance acceptance remains open.
- [x] Reduce shape IDs into class/instance annotations and bounding boxes on GPU, with explicit miss handling and camera isolation. `AnnotationProcessor` maps sparse full-width labels, reduces independent image bounds/counts and stably compacts visible shape instances. Class 0, misses, unlabelled pixels, heterogeneous/empty images, 65,537 labels, million-pixel dense instances and queued relabelling/snapshot lifecycles match CPU references.
- [x] Keep file serialization and filesystem I/O on CPU; only download selected export data. Annotation boxes/class/shape masks feed checked CPU JSON/PGM helpers; packed clouds feed CPU full-record binary PLY or legacy XYZ/RGB writers. Selected-job/scan GPU round trips preserve full-width point IDs/metadata without collecting unrelated ranges. Bounded strict PLY import and error cases pass. Legacy 8-bit PGM and XYZ/RGB are explicitly lossy; original Scan-based PTX remains a CPU API.
- [x] Test empty/all-rejected inputs, invalid points, ordering, ID retention, numeric edge cases, image masks, and multi-camera annotations against CPU references. Covered by `sensor_cloud_gpu`, `sensor_grid_gpu`, `sensor_processing_gpu`, `sensor_annotation_gpu`, `dataset_export_gpu` and `cloud_export_gpu`; full performance/shared lifecycle gates remain separate.

**Gate:** `sensor_processing_gpu`, `dataset_export_gpu` and existing export round-trip tests pass. Device-to-device chains require no host intermediate arrays. Benchmark each operation; retain simple CPU paths where small workloads benefit.

### S08 — Demonstrations, documentation and performance acceptance

- [x] Update existing viewer examples 05 LiDAR, 06 IMU, 07 Contact, 12 Camera, 105 LiDAR v2, 106 Aerial LiDAR, and 108 Synthetic Annotation to exercise the new APIs without forcing unused host collection. All seven now encode leased outputs and asynchronously collect displayed channels through bounded rings, with simulation-time deduplication, reset epochs, backpressure and visible sample age/errors. 05/105/106 use GPU-generated layouts and shared scene queries; selected readback matches legacy results, and live GPU pose updates reuse the hierarchy. 05 preserves configurable scan ordering, paused edits with distinct request IDs, sampled display settings and simulation-time auto-spin. 12 preserves five channels/optics and chains photographic processing on GPU, stops new readbacks with its pane closed, reuses thumbnail assets and follows simulation-time orbit with paused settings requests. 108 chains camera shape IDs into GPU annotations, collecting only three compacted box slots; exact CPU annotation parity and empty/reset/pause/switch fixtures pass. This checks the code-migration deliverable, not the separate visual, combined-demo or performance gates below. See `book/src/sensors/viewer.md`.
- [x] Add a deterministic moving-mesh multi-sensor example. Scene 144 uses one SensorScene for independently scheduled LiDAR/camera queries against a prescribed moving body-attached triangle mesh; an IMU follows the same body's analytic velocity. Controls expose rates, LiDAR columns/FOV/range and camera FOV. The pane shows per-sensor sample age/rate, ray counts, skipped deadlines, backpressure, CPU submission and optional GPU timing. The headless fixture verifies mesh/sphere occlusion, camera parity, analytic IMU, cadence, selected transfers, pause/settings changes, saturation, pending reset and switch/re-entry. On 2026-09-25 all 13 viewer sensor tests passed with timing off and on; all three scene-catalog checks passed. Rendered acceptance remains separate below.

  Timing prerequisite implemented: `MOLLA_SENSOR_TIMING=1` requests optional
  timestamp features in the viewer-owned device. All seven existing previews and
  the three producers in scene 144 can
  report in-batch GPU intervals through bounded nonblocking collection, with timed
  sample identity/age and timing deferrals separate from CPU submission. Default
  devices still report unmeasured GPU duration. This does not complete visual
  or performance acceptance; see `book/src/sensors/viewer.md`.

- [ ] Inspect rendered outputs and scene-switch/reset/pause behavior. Basic rendering now covers all seven migrated examples and scene 144 on isolated llvmpipe/Xvfb: LiDAR fan/clouds, IMU priming/readouts, contact readouts/normal overlays, all five camera channels/optics, aerial one-shot behavior and annotation bounds/markers. Inspection exposed and fixed off-screen panes, inaccessible Step/Reset buttons, stale paused reset/switch counters, resize strips intercepting sensor controls and occluded contact-normal segments. Contact arrows now use an explicitly labelled overlay group and were checked at default/closer scales and across reset. Final viewer lane passes 17/17 with timing off and on. Complete settings/layout coverage and hardware presentation remain open; NVIDIA/Xvfb crashed at default startup before scene selection. See `book/src/sensors/viewer.md` and `SENSORS_PROGRESS.md` for scoped evidence. A successful build or headless fixture is not visual acceptance. Do not close user preview windows without instruction.
- [ ] Update `book/src/sensors/` with capability matrix, frame/unit conventions, sync/async examples, feature requirements, resource ownership, unsupported geometry and true return semantics. Correct stale optics descriptions and test commands.
- [ ] Record full benchmark results using the matrix below, including regressions and remaining limitations. Add focused Makefile sensor regression/benchmark targets rather than relying on default tests that skip GPU suites. Per-frame harnesses cover the defined 120-case LiDAR, 56-case camera/image/annotation and 96-case LiDAR/cloud/grid subsets with explicit first-use/warm/transfer/optics/grouping/filter boundaries and capacity overrides. All three matrices completed three timestamped and three instrumentation-off repeats on the earlier traversal. The current threaded traversal additionally passes all three smoke matrices and has repeated 48-case contactless physics/sensor measurements. Raw samples, metadata/source snapshots and provisional p95 ranges are archived. Matched legacy/current-source comparisons, transfer/peak-memory accounting and repeatability acceptance remain open; see the book's latency measurement pages.

**Gate:** all required tests pass, demos have visual acceptance, performance report contains reproducible commands and hardware details, and no unsupported capability is described as complete.

### S09 — Gearbox integration, separately approved

Authorized by the user on 2026-09-25 ("gearbox now runs on this physics"). Integration lives in Gearbox at `bin/gearbox/src/sensors.rs` with `specs/CONTROLLER_SPEC.md` §7.6 and `docs/MOLLA_BACKEND.md` ("Simulated sensors on sensor links"); Molla gained `Engine::from_device` and owned-scene ConvexHull/RoundCylinder support for it. `SENSORS_PROGRESS.md` records the evidence.

- [x] Re-audit live Gearbox after approval: clean worktree at `d93e6a8`, no Makefile (`.make.lua` recipes under the Nix shell), Molla pinned by git `rev` and now patched to this checkout. No user WIP was present in Gearbox.
- [x] Stable mounts and a revisioned scene snapshot: `role = "sensor"` links parse `gearbox:sensor:*` into `SensorSpec`; a per-machine `Rig` compiles `SensorScene` once per Molla scene revision and copies body poses/twists per sample under the physics lock, with GPU results collected on later frames. No GPU wait holds the mutex.
- [x] Device sharing: Bevy 0.19.1 and Molla both use wgpu 29.0.4; the sensors adopt the render device through `Engine::from_device` (no renderer or physics dependency upgrade).
- [x] Telemetry: `/machines/<id>/sensors/<link>` streams `datapod.imu.v1` (link frame, orientation in the `odom` world remap) and chunked `gearbox.lidar_scan.v1` (link-frame points, `NaN`/infinite misses, `pulse_offset`/`pulse_count` per 900-pulse chunk). Contacts are not published by the sensor path; the controller telemetry is unchanged.
- [x] Per-link rates and settings are independent; CPU Featherstone, tyres, soil and rendering are untouched; rigs rebuild on revision/mount change and drop with their machine, discarding pending readbacks.
- [ ] Validation on the tractor scene: terrain, own-body and hull colliders visible; two sensors; pause/play and despawn verified live; the headless GPU fixture and the live drive match commanded motion. Moving parts of other machines and a separate transfer-overhead measurement remain; frame rate was 8–12 fps with and without the machine (render-bound), physics 0.15–0.44 ms/step.

**Gate:** integration tests and visual acceptance pass in Gearbox; telemetry reflects actual sensor readings; no unrelated assets or systems change. Remaining for closure: transfer-overhead measurement, moving-part visibility across machines, and camera links.

### S10 — Explicitly deferred advanced capabilities

These require independent requirements/physics decisions; they are not hidden prerequisites for GPU pipeline completion.

- **True multiple returns:** multiple intersections per laser path, material transmission/attenuation, return ordering/limits and energy thresholds. Do not relabel divergent-sub-ray first/last aggregation as this feature.
- **Physical fidelity:** emitter exit-aperture sampling (currently metadata only), reflectance/intensity models, sensor noise/bias/dropout and within-scan motion distortion. Specify units, deterministic seeds and calibration targets before implementation.
- **Force sensing:** solver impulse/force provenance, time integration and reference-frame conventions; contact depth/normal alone is insufficient.
- **Sensor gradients:** review `raycast_adjoint.rs` separately. Decide whether finite differences suffice or a GPU analytic/differentiable-query design is justified; define visibility-discontinuity behavior before promising gradients.

**Gate before starting any extension:** approved semantics, reference fixtures, error budget, performance budget and bounded implementation scope.

## 5. Verification commands

Commands below are for implementation time, not evidence of tests already run. Use the existing Nix development environment and GPU wrapper. If outside that shell, prefix the command with `nix develop --impure -c`; set the installed NVIDIA driver version only if the local shell requires it, never copy a stale version blindly.

```sh
# Reconcile both committed and uncommitted drift before editing.
git status --short
git diff --stat d2369a9..HEAD -- crates/molla-sensors crates/molla-kernels crates/molla-geometry crates/molla-storage crates/molla-sim bin/molla_viewer
git diff -- crates/molla-sensors crates/molla-kernels crates/molla-geometry crates/molla-storage crates/molla-sim bin/molla_viewer Makefile

# Feature boundaries and host tests.
make check APP_TARGET='-p molla-sensors --no-default-features'
make check APP_TARGET='-p molla-sensors --features gpu --all-targets'
make test APP_TARGET='-p molla-sensors --no-default-features'

# Existing GPU suites: each must actually execute its ignored GPU tests.
for suite in sensor_raycast_gpu raycast_trimesh_gpu lidar_scan_grid_gpu \
  lidar_multi_return_gpu aerial_lidar_gpu sensor_imu_gpu sensor_contact_gpu \
  sensor_frame_transform_gpu sensor_tiled_camera_gpu tiled_camera_optics_gpu \
  dataset_export_gpu; do
  make test-gpu P=molla-sensors T="$suite" A='--test-threads=1' || exit 1
done

# Separate CPU finite-difference helper regression.
make test APP_TARGET='-p molla-sensors --test raycast_adjoint'

# Viewer integration after source changes.
make check APP_TARGET='-p molla_viewer'
make test-viewer-sensors
make build
make run
```

**Benchmark command prerequisite (implemented):** S00 added `BENCH_ARGS` to the `bench` recipe (`$(RUN_WITH) $(CARGO) bench $(BENCH_ARGS)`), preserving its default behavior. Use these scoped commands:

```sh
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_raycast'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera'
make bench-sensors # Shared scene-query matrix; release profile and GPU wrapper.
make bench-sensors SENSOR_BENCH=bench_sensor_lidar
make bench-sensors A='scene_queries_1024_rays/hierarchy.*warm'
```

`test-sensors` runs the host/feature and explicitly ignored GPU regression lanes.
`bench-sensors` selects one sensor benchmark executable (default
`bench_scene_queries`); `A` forwards Criterion filters/options. `A=--test` is a
smoke run, not timing evidence. Neither target implies full performance-matrix or
viewer acceptance.

For each new suite specified in S02–S07, add an integration test file under `crates/molla-sensors/tests/` and run `make test-gpu P=molla-sensors T=<suite_name> A='--test-threads=1'`. Follow existing `sensor_raycast_gpu.rs` GPU locking/setup conventions. Keep pure validation tests non-ignored so the ordinary host lane executes them.

Expected result: exit 0, named tests actually run and pass, no validation errors, panics or unsupported-device skips disguised as success. Record test counts. Tests depending on optional GPU features/capabilities must identify that requirement explicitly. Run geometry/storage/solver regression lanes as well whenever their shared infrastructure changes; select affected existing suites after inspecting their Makefile/test definitions.

## 6. Performance matrix and acceptance rules

Measure both device execution and CPU-visible completion. Report p50/p95 latency, scan/frame throughput, CPU encode/layout time, GPU time where timestamp queries are supported, uploads/downloads, submissions, blocking waits, allocations and peak memory. Do not call CPU wall-clock submission time GPU time.

| Dimension | Required cases |
|---|---|
| Ray count | 1k, 10k, 100k, 1M; large cases chunked or explicitly rejected by documented budgets |
| Sensors | 1, 4, 16 independent mounts |
| Geometry | Primitives, one detailed mesh, many instances, moving meshes, terrain representation |
| Primitive count | Existing 100/1,000 baseline plus an accelerated larger-scene case |
| Mesh instances | 1, 10, 100; report triangles and BVH memory |
| Returns | Single ray and multiple divergent sub-rays per pulse |
| Cameras | 1 and 4 cameras, 640×480 and 1280×720, pinhole and multisample optics |
| Lifecycle | Cold build, warm fixed scene, pose updates, topology replacement |
| Consumption | Device-only, host final outputs, cloud/image processing chains |
| Combined load | CPU physics + sensors; GPU physics + sensors where supported |

Use representative subsets for quick gates and the full supported matrix for release evidence; do not silently omit expensive cases. Bound memory rather than executing unsafe allocations.

Structural acceptance is mandatory: zero intermediate host readbacks in device-only paths; no per-mesh host merge; no per-aperture color download; retained warmed resources; bounded in-flight memory; correct moving-scene invalidation.

Performance acceptance: establish and record budgets from S00 on named hardware before optimization results are assessed. Compare repeated release-profile runs, including confidence/variance. Flag any greater-than-10% p95 end-to-end regression on an unchanged baseline workload for investigation; do not automatically waive it. Expanded geometry correctness can be reported separately from like-for-like performance. Do not promise a specific real-time rate or speedup without measurements.

## 7. Completion, stop conditions and maintenance

- [ ] All core packages S00–S08 meet their gates; status rows link to test/benchmark evidence.
- [ ] Public APIs document device ownership, timestamps, units, limitations and compatibility wrappers.
- [ ] Feature boundaries and affected shared-crate regressions pass.
- [ ] No unsupported geometry, unavailable force output or approximate return model is silently advertised as complete.
- [ ] No source/asset outside the approved scope is modified; unrelated WIP is preserved.
- [ ] S09 and S10 remain separately labeled if not authorized or completed.

Stop the affected work package and report if source drift invalidates its assumptions; a dependency requires out-of-scope changes; sensor semantics cannot be preserved; the required GPU lane cannot execute; or passing tests would require disabling buffer/device validation. Do not replace missing measurements with claims. Preserve remaining independent work and report the exact blocked requirement.

Future changes to scene topology, body indexing, buffer dirty-state ownership, camera channel layouts, ray ordering, or telemetry frames must extend the corresponding regression fixtures. Keep CPU references independent of optimized GPU traversal and reductions so tests can detect shared-kernel mistakes.
