# GPU Sensors Implementation Plan

Date: 2026-09-21. Planned against commit `d2369a9` plus the current working tree.

**Status: IMPLEMENTATION STARTED. Audit findings below describe the pre-implementation baseline, not current guarantees. See `SENSORS_PROGRESS.md` for execution evidence.**

This is the full sensor plan at the requested filename, `SENSROS.md`, not just a link. It snapshots the existing `SENSORS.md` execution plan on 2026-09-21. Continue recording implementation status in `SENSORS.md` and evidence in `SENSORS_PROGRESS.md`; do not treat this snapshot as a second independent implementation backlog. Checkboxes describe future work unless explicitly marked complete with verification evidence.

This documentation update does not execute the plan, change sensor code, or run builds/tests.

## Reading order and sensor coverage

Start with correctness and shared device-result ownership (S00–S02), then finish parallel LiDAR and reusable scene queries (S03–S04). Migrate the remaining sensors onto those foundations before viewer and performance acceptance. GPU work must preserve each sensor's independent configuration and sample identity.

| Sensor / operation | Planned GPU work | Package |
|---|---|---|
| Terrestrial and aerial LiDAR | Ray generation, independent multi-mount batches, intersections, nearest-hit merge and sub-ray return selection | S03 |
| Moving machine meshes and terrain visibility | Shared local mesh BVHs, instance poses, scene acceleration and revision-safe results | S04 |
| IMU | Device-resident velocity history, mounting offsets and optional collection | S01, S05 |
| Contact sensing | Batched contact geometry aggregation without mandatory downloads; no invented force readings | S05 |
| Frame transforms | Batched world/relative poses with independent targets | S05 |
| Cameras | Shared scene visibility, independent camera settings and GPU aperture accumulation | S06 |
| Images and point clouds | GPU post-processing, transforms, filtering, compaction and reductions | S07 |
| Synthetic annotations | GPU class/instance reductions and bounding boxes; CPU file serialization | S07 |
| All sensor types | Simulation-time scheduling, bounded asynchronous readback, lifecycle tests and measured performance | S02, S05, S08 |
| Advanced fidelity and gradients | Separately specified true multiple returns, noise, intensity, force provenance and differentiable queries | S10, deferred |

### Existing work must be reconciled before implementation

The current checkout already contains `lidar_batch.rs`, `readback.rs`, `scene.rs`, and corresponding GPU tests. Do not implement duplicate versions from this snapshot's unchecked tasks. Read `SENSORS.md` for current execution status and `SENSORS_PROGRESS.md` for recorded evidence, then verify the live code. File presence is not proof of runtime correctness or performance acceptance. The audit table and checklist below retain the original planning baseline, not a fresh claim that those gaps still exist.

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

## 2. Baseline audit and evidence

Paths below are relative to this repository. Line numbers are audit anchors and may drift; verify the named symbols before implementing.

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
| S02 | Device-resident API and asynchronous collection | P1 | L / high | S01 | IN PROGRESS (IMU + shared readback; remaining sensors pending) |
| S03 | GPU LiDAR generation and hit reduction | P1 | L / medium | S02 | TODO |
| S04 | Shared accelerated scene and moving meshes | P1 | L / high | S02, S03 | TODO |
| S05 | IMU, contacts, transforms and scheduling | P1 | M / medium | S02 | TODO |
| S06 | Shared camera tracing and GPU optics | P1 | L / high | S04 | TODO |
| S07 | GPU image/cloud processing and annotations | P2 | L / medium | S03, S06 | TODO |
| S08 | Viewer, documentation and performance acceptance | P1 | M / medium | S03–S07 | TODO |
| S09 | Gearbox integration | Gated | L / high | S08, explicit approval | NOT AUTHORIZED |
| S10 | Advanced sensing extensions | Optional | L / high | Core acceptance, separate decisions | DEFERRED |

### S00 — Establish reproducible baselines and contracts

- [ ] Read current APIs/tests; inventory supported geometry for each sensor. Record unsupported types explicitly rather than treating them as ordinary misses.
- [x] Recheck working-tree drift and preserve unrelated changes before editing. See `SENSORS_PROGRESS.md`.
- [x] Run the existing sensor GPU suites through the commands below. Record failures before modifications, adapter/backend, driver, compiler, build profile, scene sizes, and revision. Baseline: 35 GPU tests passed; evidence in `SENSORS_PROGRESS.md`.
- [ ] Capture existing end-to-end benchmarks, including uploads/readback. Distinguish warm reuse from first-use BVH/pipeline creation.
- [x] Add deterministic mixed primitive/static-mesh reference fixtures, sensor mount/coordinate fixtures, and a small CPU brute-force triangle oracle independent of the GPU traversal. Scaled rays, transformed-mesh oracle and 65-mount IMU regression pass.
- [ ] Record numerical tolerances before refactoring: hit/miss and IDs exact away from ties; distance/position/normal tolerances tied to f32 scale; deterministic tie policy for equal-distance hits.

**Gate:** existing suite results and baseline measurements recorded; new reference fixtures pass; failures are explicitly resolved or scoped, not silently excluded.

### S01 — Fix correctness before optimizing

Files: `crates/molla-sensors/src/{imu,lidar,raycast}.rs`, related raycast/IMU kernels, and corresponding integration tests.

- [x] Make IMU previous-velocity history a device-to-device snapshot. Evaluate against the previous sample, then advance history in the same ordered command stream. Define first-call, reset, body-count change, scene replacement, and invalid/nonpositive `dt` behavior. GPU-integrated, reset and multi-mount regressions pass.
- [x] Unify public ray distance as meters for all supported nonzero directions. Normalize consistently for primitive and mesh traversal and point reconstruction. Define finite/zero-direction handling without allowing stale output slots. Scaled mixed-geometry and invalid-input regressions pass.
- [ ] Replace unchecked scan/ray/pixel sizing with checked arithmetic. Validate capacity, binding sizes, dispatch limits, and allocation budgets before allocation or dispatch; chunk workloads when appropriate.
- [x] Implement and test cone intersection parity or reject unsupported cone queries explicitly until implementation lands. Completion of supported-primitive parity requires actual cone support, not a silent miss. Side/base/inside/apex/generator-parallel and miss references pass.
- [ ] Validate mesh-cache revision changes and prevent stale results after model replacement or geometry changes.
- [ ] Add regressions: scaled rays against mixed geometry; max range; inside-shape hits; degenerate rays; overflow metadata; GPU-stepped IMU including reset and mount lever arm; cone hit/miss.

**Gate:** `sensor_imu_gpu`, `sensor_raycast_gpu`, `raycast_trimesh_gpu`, scan-grid and aerial suites pass, including new regressions. CPU metadata-validation tests reject extreme dimensions without huge allocations. Never relax `Buffer` dirty-state assertions.

### S02 — Separate encoding, submission and readback

Files: sensor wrappers, sensor kernel dispatch wrappers, narrowly scoped storage helpers; proposed tests `sensor_device_pipeline_gpu.rs` and `sensor_readback_gpu.rs`.

- [ ] Add composable encode/device-result APIs without breaking existing convenience methods. Match existing `Result`/`Error` conventions, e.g. `Raycaster::cast`.
- [ ] Retain output, uniform, bind-group and staging resources across steady-state frames where layouts/capacities permit. Rebuild only on actual capacity/layout change.
- [ ] Provide grouped asynchronous readback of requested active ranges/channels. Expose backpressure, cancellation, device loss and mapping failures; define shutdown/reset handling for outstanding requests.
- [ ] Bound in-flight batches and prevent output/staging reuse until GPU completion and CPU release. Reject cross-device handles.
- [ ] Test device-only chaining, no host access to device-authoritative data, zero-sized batches, ring saturation, scene reset with pending results, capacity growth, and explicit legacy synchronous calls.

**Gate:** new suites pass; instrumentation asserts zero host downloads/waits in the encode-only path and no resource allocations in a warmed fixed-capacity pipeline. Host collection transfers only requested active data. The synchronous API matches its reference outputs.

### S03 — Make LiDAR device-resident end to end

Files: `lidar.rs`, `raycast.rs`, raycast kernels; add ray-layout and hit-reduction kernels and `lidar_pipeline_gpu.rs`.

- [ ] Generate terrestrial grids, aerial origins, deterministic cone/sub-ray patterns on GPU. Preserve logical row/column/pulse/sub-ray ordering and metadata back-pointers.
- [ ] Batch multiple mounts with separate extrinsics, scan settings, ranges and identifiers. Use descriptors/offsets or compatible groups; never combine sensors' independent settings.
- [ ] Merge primitive/mesh candidates on GPU. Serialize reductions safely or use a proven reduction design; avoid unsynchronized writes to shared nearest-hit records.
- [ ] Compute current first/last/sub-ray aggregates on GPU. Keep their semantics explicit: they are not penetration/transmission returns along one laser path.
- [ ] Support downstream device consumption or one logical final host collection, independent of mesh count. Chunking must preserve sensor IDs and scan ordering.
- [ ] Test old/new parity, empty/miss scans, several mounts beyond one workgroup, varying sub-ray counts, bounded capacity, and primitive/mesh occlusion.

**Gate:** `lidar_pipeline_gpu` plus existing LiDAR suites pass. Counters show no per-mesh host readback or host hit merging, and no host ray-vector construction in the new GPU-layout path. Final scan matches CPU references within recorded tolerances.

### S04 — Share acceleration and support moving machine geometry

Files: `crates/molla-geometry/`, raycast kernels/wrappers, a proposed `crates/molla-sensors/src/scene.rs`, tests `sensor_scene_gpu.rs`.

- [ ] Store reusable mesh BVHs in local coordinates, indexed separately from instances. Moving an instance updates its transform/bounds, not every vertex or its entire mesh BVH.
- [ ] Add top-level acceleration over supported finite primitives and mesh instances; handle infinite planes separately. Define refit versus rebuild triggers and geometry/topology revision invalidation.
- [ ] Transform rays/normals correctly through instances and preserve metric distances. Define supported scaling explicitly; reject unsupported singular/nonuniform transforms rather than returning plausible wrong hits.
- [ ] Support body-attached triangle meshes. Stable body/shape handles must survive dense-index remapping, deletion and replacement without aliasing old results.
- [ ] Establish an explicit terrain path: direct heightfield queries or a revisioned triangulated sensor representation. Document conversion cost and updates; do not add soil simulation here.
- [ ] Define a geometry capability matrix for Triangle, ConvexHull, Heightfield and SDF. Add tested conversions/support where feasible; return explicit unsupported diagnostics for remaining gaps rather than claiming completeness.
- [ ] Bound traversal stack use and expose overflow safely; prove/test the bound or use a traversal strategy that cannot silently omit children.
- [ ] Test moving meshes, repeated instances, occlusion, shape IDs, pose-only updates, geometry edits, removals, negative/large coordinates, and near-equal hit ties against independent references.

**Gate:** `sensor_scene_gpu` and all raycast suites pass; changing a pose changes hits without rebuilding mesh geometry. Warm fixed-topology frames contain no per-instance BVH reconstruction. Many-instance benchmarks include acceleration build/refit costs, not just traversal.

### S05 — Finish other sensor pipelines and scheduling

Files: `imu.rs`, `contact.rs`, `frame_transform.rs`; proposed scheduler module and `sensor_scheduler_gpu.rs`.

- [ ] Expose device-resident IMU/contact/transform outputs through S02; keep synchronous wrappers for callers that need them.
- [ ] Schedule independent sensor rates from simulation timestamps. Specify first sample, bounded catch-up, skipped samples, pause, reset and scene replacement. Do not fabricate intermediate-time state when no interpolation/history exists.
- [ ] Batch independent mount configurations and relative-frame targets. Define snapshot consistency when CPU physics is the producer.
- [ ] Preserve IMU gravity/specific-force, angular acceleration and lever-arm semantics, with multi-mount and GPU-stepped tests.
- [ ] Describe contact output accurately as geometry/aggregation. Force sensing is optional S10 and must not infer forces from penetration alone.

**Gate:** IMU/contact/frame suites and `sensor_scheduler_gpu` pass; multi-rate timestamps are deterministic under different render frame rates; no implicit collection occurs in the device-only path.

### S06 — Reuse scene queries for cameras and move optics accumulation to GPU

Files: `tiled_camera.rs`, camera kernels, shared query code; existing camera/optics tests plus `sensor_camera_scene_gpu.rs`.

- [ ] Route camera visibility through S04 while preserving five output channels, near/max-distance semantics, normals and shape IDs. Test actual meshes and terrain representation, not only primitives.
- [ ] Accumulate aperture color samples in GPU floating-point storage and resolve once. Preserve existing packed-color averaging semantics initially; changing color space requires a separate documented decision and reference updates.
- [ ] Produce geometric channels from the center sample without repeated CPU readback/restoration. Skip color accumulation when COLOR is disabled.
- [ ] Preserve homogeneous batching, then group heterogeneous cameras by compatible layout or add per-camera intrinsics. Keep each camera's pose, parameters and output slice independent.
- [ ] Test channel masks, one/many cameras, focal behavior, occlusion, batching isolation, and device-only camera-to-processing chains.

**Gate:** camera, optics and `sensor_camera_scene_gpu` suites pass. No per-aperture host downloads occur. Primitive results remain within baseline tolerances; mesh depth/IDs match independent fixtures.

### S07 — GPU post-processing, clouds and annotations

Files: `camera_post.rs`, `pointcloud.rs`, `dataset.rs`, new processing kernels; tests `sensor_processing_gpu.rs`.

- [ ] Port image adjustments to composable GPU passes: exposure, gamma/sRGB conversion, tone mapping, saturation, and white-balance reductions. Preserve CPU functions as reference/convenience APIs.
- [ ] Add cloud construction, transforms, filters and compaction on device. Preserve scan/sensor IDs and define ordering; use prefix-sum compaction where stable order is required.
- [ ] Add spatial binning and reductions with bounded memory and overflow reporting. Do not silently lose points in full cells.
- [ ] Reduce shape IDs into class/instance annotations and bounding boxes on GPU, with explicit miss handling and camera isolation.
- [ ] Keep file serialization and filesystem I/O on CPU; only download selected export data.
- [ ] Test empty/all-rejected inputs, invalid points, ordering, ID retention, numeric edge cases, image masks, and multi-camera annotations against CPU references.

**Gate:** `sensor_processing_gpu`, `dataset_export_gpu` and existing export round-trip tests pass. Device-to-device chains require no host intermediate arrays. Benchmark each operation; retain simple CPU paths where small workloads benefit.

### S08 — Demonstrations, documentation and performance acceptance

- [ ] Update existing viewer examples 05 LiDAR, 06 IMU, 07 Contact, 12 Camera, 105 LiDAR v2, 106 Aerial LiDAR, and 108 Synthetic Annotation to exercise the new APIs without forcing unused host collection.
- [ ] Add a deterministic moving-mesh multi-sensor example. Show per-sensor settings, sample age/rate, ray counts, dropped/deferred samples, and CPU/GPU timing separately.
- [ ] Inspect rendered outputs and scene-switch/reset/pause behavior. A successful build or headless fixture is not visual acceptance. Do not close user preview windows without instruction.
- [ ] Update `book/src/sensors/` with capability matrix, frame/unit conventions, sync/async examples, feature requirements, resource ownership, unsupported geometry and true return semantics. Correct stale optics descriptions and test commands.
- [ ] Record full benchmark results using the matrix below, including regressions and remaining limitations. Add focused Makefile sensor regression/benchmark targets rather than relying on default tests that skip GPU suites.

**Gate:** all required tests pass, demos have visual acceptance, performance report contains reproducible commands and hardware details, and no unsupported capability is described as complete.

### S09 — Gearbox integration, separately approved

Current integration evidence: `/home/bresilla/data/code/robolibs/gearbox/bin/gearbox/src/physics/molla/mod.rs` wraps CPU `RigidWorld`; `bin/gearbox/src/controller.rs:5089-5103` publishes a controller-derived IMU. No Molla sensor API integration was found during this audit. `crates/molla-sim/src/runtime.rs` already exposes scene `model()`/`state()` accessors; reuse those where suitable rather than assuming a new physics architecture is necessary.

- [ ] Re-audit live Gearbox and its Makefile after explicit approval; record drift and user WIP before edits. No Gearbox execution commands are prescribed here because its current integration/build requirements need that review.
- [ ] Define stable sensor mounts/configuration and a revisioned scene snapshot adapter. Upload static geometry once and batched changed poses at sensor sample times; do not hold the physics mutex during GPU waits.
- [ ] Verify GPU device/wgpu-version compatibility before selecting direct resource sharing. Use an explicit boundary if renderer and sensors cannot share resources; do not silently upgrade the renderer or physics dependencies.
- [ ] Connect actual Molla IMU readings and LiDAR outputs to existing telemetry consumers with documented frame/time conversion. Define contacts unavailable from the producer explicitly rather than inventing readings.
- [ ] Preserve independent per-machine/per-sensor settings, CPU Featherstone behavior, and existing soil/tyre/rendering work. Mount/unmount and machine removal invalidate pending sensor results safely.
- [ ] Validate on an actual machine scene, including moving parts, terrain visibility, multiple sensors, pause/reset and spawn/despawn. Measure physics-plus-sensors frame cost and transfer overhead.

**Gate:** separately approved integration tests and visual acceptance pass in Gearbox; telemetry reflects actual sensor readings; no unrelated assets or systems change. S08 can complete while S09 remains NOT AUTHORIZED.

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
make build
make run
```

**Benchmark command prerequisite (implemented):** S00 added `BENCH_ARGS` to the `bench` recipe (`$(RUN_WITH) $(CARGO) bench $(BENCH_ARGS)`), preserving its default behavior. Use these scoped commands:

```sh
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_raycast'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_camera'
```

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
