# Sensor geometry snapshots

`SensorScene` owns a compiled copy of sensor geometry, per-shape colors and shared local-space mesh
BVHs. It is independent of the mutable physics model: editing that source model
does not silently alter an existing snapshot. GPU body poses remain live inputs
from the caller's `State`; geometry is not rebuilt when those poses move.

```rust
use molla::sensors::{Raycaster, SensorScene, SensorSceneOptions};

let mut scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default())?;
let mut raycaster = Raycaster::new(&engine, 100_000)?;
let hits = raycaster.dispatch_scene(
    &engine, &scene, &state, &origins, &directions, 100.0,
)?;
let stamp = hits.scene_stamp.expect("owned scene query");
```

`Raycaster::cast_scene` is the synchronous equivalent. GPU-generated LiDAR uses
`LidarBatch::dispatch_scene`, whose `hits.scene_stamp` carries the same identity.
`TiledCamera::dispatch_scene` uses the same visibility queries and BVHs; its
`channel_scene_stamp(channel)` identifies the last sensor write to each channel.
Multiple raycasters, LiDAR batches and cameras can share one `&SensorScene`. Compilation
builds one BVH per referenced mesh identity and packs all mesh data into one immutable
GPU atlas. Instances reference offsets; geometry is not duplicated per instance or sensor.

## Coordinates and live poses

Each instance uses `body world pose * shape local pose`; static instances use
only the shape pose. Mesh vertices/BVH bounds stay in local coordinates. GPU
queries transform rays into that frame and normals back to world space, keeping
distance in meters. Triangle normals oppose the incident ray.

Transforms are rigid: translation and unit quaternion, with no scale component.
Compilation rejects non-finite/non-unit shape transforms and invalid primitive
dimensions. Live `State` body poses must also be finite rigid poses on the same
engine and have the expected body count. Bodyless scenes ignore unused state poses.
Their device authority is checked;
they are not downloaded for validation. Invalid GPU-produced pose values are
not yet reported through a dedicated device error channel.

An uploaded pose update changes hits without rebuilding mesh geometry. A changed
shape-local pose or geometry in an owned snapshot currently requires explicit
replacement; independent local-pose updates/refits remain future work.
Per-shape colors are immutable too: changing them requires explicit replacement,
which currently recompiles geometry and advances the same scene revision. Colors
must have one finite f32-representable entry per shape; material updates are not
silently read from the mutable source model. Independent material-only updates
without geometry rebuilding remain an optimization opportunity.

## Revisions and result identity

```rust
let previous = scene.stamp();
scene.replace_model(&engine, &edited_model)?;
assert_eq!(scene.stamp().scene_id, previous.scene_id);
assert_ne!(scene.stamp().geometry_revision, previous.geometry_revision);
```

Replacement validates/builds/uploads a candidate before replacing the old
snapshot. A failed replacement leaves the current geometry and revision intact.
Source host geometry and colors must be current; the compiler does not secretly download
device-authoritative model metadata.

Shape indices are scoped to `SceneStamp { scene_id, geometry_revision }`.
`scene.resolve_body(stamp, shape_index)` rejects a foreign/old revision or an
out-of-range index instead of mapping an old result to a newly numbered shape.
Replacement invalidates **all** old revision-scoped indices, including shapes
whose geometry did not change. Stable handles are separate from those indices.

An already enqueued readback retains its old snapshot data. Store the scene
identity/revision alongside its `ReadbackStamp`; do not resolve it against a
new revision. Geometry revisions do not identify simulation pose time. The
caller still supplies sample IDs, simulation time and pose synchronization.

### Stable body and shape handles

`scene.identities()` returns a cheap `Arc`-backed `SceneIdentitySnapshot`. It owns
only CPU identity tables, not GPU resources or geometry. Capture it alongside a
sample's `SceneStamp` before encoding/dispatch; a clone does not rebuild or copy
the tables. It remains usable after scene replacement or destruction.

- `shape_handle(stamp, index)` maps a non-miss dense shape index to an opaque
  `SensorShapeHandle`; `shape_body(stamp, index)` gives its historical owner.
- `body_handle(stamp, BodyId)` maps a dense body index to an opaque
  `SensorBodyHandle`. Static/world ownership is `None`, not an ordinary handle.
- These lookups reject mismatched scene/revision stamps and out-of-range indices.
  Filter negative/unsigned-max miss IDs before resolving a shape.
- `shape_index(handle)` and `body_index(handle)` locate that identity in the
  snapshot on which they are called. Resolving against the current snapshot
  gives current dense indices; resolving against a retained snapshot gives the
  old ones. A foreign or absent handle errors rather than aliasing a reused slot.

`replace_model_with_remap` preserves identities explicitly. Its two slices follow
the **replacement** model's dense body and shape order and must cover both arrays
exactly. `Some(current_handle)` preserves that object; `None` allocates a new
identity. Old handles not retained by the mapping are retired. There is no
geometric/name/index heuristic for deciding whether an object survived.

```rust
use molla::core::BodyId;
use molla::sensors::SensorSceneRemap;

// Both models have two bodies and two shapes, reordered in the replacement.
let old = scene.identities();
let a = old.body_handle(old.stamp(), BodyId(0))?.unwrap();
let b = old.body_handle(old.stamp(), BodyId(1))?.unwrap();
let x = old.shape_handle(old.stamp(), 0)?;
let y = old.shape_handle(old.stamp(), 1)?;
scene.replace_model_with_remap(&engine, &replacement, SensorSceneRemap {
    bodies: &[Some(b), Some(a)],
    shapes: &[Some(y), Some(x)],
})?;
assert_eq!(scene.identities().body_index(a)?, BodyId(1));
assert_eq!(scene.identities().shape_index(x)?, 1);
assert_eq!(old.shape_index(x)?, 0);
```

The replacement model defines geometry and body attachments. A preserved shape
may explicitly move to another preserved/new body or become static; historical
snapshots retain its old owner. Geometry edits can retain a handle if the caller
declares that it is the same application object. Duplicate, foreign, retired or
wrong-length remaps fail transactionally before GPU geometry compilation. Later
geometry validation failures also preserve the current scene and identity table.

Plain `replace_model` deliberately assigns **all-new handles**, even for an
identical model. Published identity serials are never recycled within a scene, including
after an empty replacement; exhaustion returns an error. Deleted handles cannot
be resurrected by a later remap, but retained historical snapshots still identify
their old results. Handles are process-local opaque Rust values, not serialized
asset IDs. The application owns any association to its machine/part identifiers.

All query channels still contain dense indices, so ray, LiDAR and camera formats
and GPU kernels are unchanged. Store the matching snapshot/stamp with delayed
outputs; the APIs do not automatically attach identity tables to every frame or
download GPU shape indices. Live body poses must match the replacement model's
body ordering. These handles do not add top-level acceleration or pose refitting.
Tables allocate at scene compilation/replacement and remain retained as long as
their last snapshot clone lives; account for historical metadata separately from
ray/output memory budgets.

## Geometry capabilities

| Geometry | Owned scene support |
|---|---|
| Sphere, plane, box, capsule, cylinder, cone, ellipsoid | Analytic GPU intersections |
| TriMesh, static or body-attached | Shared local BVH, rigid GPU instance transforms |
| Standalone Triangle | One-triangle local mesh conversion |
| Heightfield | Full local triangulation at snapshot compilation |
| ConvexHull | Triangulated once at snapshot compilation; degenerate (fewer than four non-coplanar points) hulls are unsupported |
| RoundCylinder | Approximated by its dilated cylinder (`half_height + border_radius`, `radius + border_radius`); the rounded rim is not modelled |
| SDF | Unsupported; rejected by default |

Heightfields preserve the simulation grid's X-max/Z-min to X-min/Z-max diagonal,
origin and separate X/Z cell sizes. The one-time conversion creates `nx * nz`
vertices and `2 * (nx - 1) * (nz - 1)` triangles. Edits need an explicit scene
replacement. This is a sensor terrain representation, not soil simulation or
incremental deformable-terrain coupling.

`SensorSceneOptions` defaults to a total budget of one million vertices and one
million triangles across unique referenced meshes/conversions. GPU buffer limits
are also checked. Exceeding limits errors before conversion allocation; there is
no automatic truncation. Repeated instances share the converted mesh.

To inspect a partially supported scene, explicitly choose
`UnsupportedGeometryPolicy::ReportAndSkip`, then inspect
`scene.unsupported_shapes()` for original shape indices and unsupported kinds.
Those omitted shapes cannot occlude rays. A miss in such a scene is not proof of
free space. `Reject` is the default so omission is not silently enabled.

## Legacy model queries

Direct `Raycaster::cast`/`dispatch` and `LidarBatch::dispatch` remain compatible.
Their raw-model path now supports rigid moving TriMesh shapes too. It caches
local geometry per mesh ID and compares exact source vertices/indices once per
unique mesh per query. Edits and replacement models rebuild changed geometry;
removed entries are released on the next nonempty query. Pose-only updates reuse
the mesh BVH. `clear_mesh_cache()` explicitly releases this compatibility cache.

Exact comparisons cost CPU work proportional to mesh data and retain a CPU copy.
Use owned scenes for a warm immutable geometry path without these comparisons.
Raw-model queries still skip Triangle/Heightfield/ConvexHull/RoundCylinder/SDF;
owned compilation provides conversion or explicit capability diagnostics.

## Traversal and remaining acceleration work

Mesh upload rejects invalid indices, non-finite bounds/vertices, cycles,
disconnected trees, duplicate/omitted triangle slots and bounds excluding their
children/vertices. Depth must be at most 64. Depth-first binary traversal then
needs at most 64 pending nodes, matching the raw compatibility shader's stack. Unit tests accept a
depth-64 tree and reject depth 65; the median-split builder produces much
shallower trees for supported buffer sizes. Raw GPU node buffers must not be
mutated into an invalid tree after validation.

### Shared accelerated queries

Owned-scene ray, LiDAR and camera queries now refit a finite-instance hierarchy and
run one closest-hit traversal dispatch. Infinite planes are intersected separately
inside that dispatch. A candidate mesh is traversed through its local BVH in the
shared atlas; no per-mesh query/merge dispatch or host geometry loop is encoded.
The raw-model compatibility path still uses linear primitive queries and per-instance
mesh dispatches, and remains an independent comparison path.

Exact-distance ties prefer primitives over meshes, then the smallest dense shape
index within that group. Equal mesh-triangle distances preserve local BVH traversal
order, with the right child visited first. Hits must be strictly inside the positive
maximum range. Owned instance and mesh traversal is stackless: immutable escape
links skip rejected subtrees and advance after leaves. The GPU instance record's
fourth word stores its escape link; mesh node word seven stores a local escape
index. CPU topology retains its original heights and children. Refits and leaf
reordering preserve these links. GPU record sizes and binding counts are unchanged;
construction temporarily packs a separate 16-byte-per-node instance array.
The existing mesh validation depth limit remains 64. Axis-parallel slab checks explicitly
handle zero directions. Bounds and traversal range margins are conservative; actual
surface intersections still use the same primitive routines as the raw path.

Each raycaster retains one active scene hierarchy and up to 128 query-binding
variants. Switching scene identity/revision replaces that hierarchy; returning to
an earlier scene rebuilds it. Pose-only updates refit without topology rebuilding.
Initial current host poses seed partitioning when available; GPU-authoritative poses
are never downloaded. Owned queries can subsequently reorder leaves on GPU as
described below. Correctness does not depend on partition quality. Small scenes may
cost more dispatches than the linear path.

The immutable shape/mesh atlas has a default 256 MiB GPU-payload budget. Override it
with `SensorScene::from_model_with_query_budget`; replacements retain that limit.

Shape/plane records and unique raw-mesh, triangle and heightfield payloads are
preflighted before identity tables, model copies, terrain conversion or uploads.
Raycast BVH node counts are planned exactly from the median split/leaf size, without
building centroid, bounds, triangle-order or node arrays. Shared mesh/field instances
are counted once. The resulting tree must match its planned node count before packing.
Converted geometry is checked again before upload, including convex-hull output;
convex-hull workspace itself is outside this packed-query budget.

An allocation regression rejects a 64 KiB query budget for 65,537 primitive
instances, a 65,537-triangle mesh and a 257×257 heightfield. Before preflight these
errors requested 21,897,134 / 795,054 / 3,168,418 calling-thread bytes; afterward
they request 37 / 89 / 89 bytes for error/metadata handling. This is not a process
or VRAM peak measurement, and does not establish successful-construction or warm
query performance. Exact-budget shared geometry remains accepted; one byte under
is rejected. Model/identity host storage, hull workspace, refits and driver staging
need their separate construction/resource acceptance.

CPU mesh BVHs now reserve the planned node count rather than two nodes per
triangle. The common median-tree count supports every leaf size, including zero's
existing clamp to one; GPU raycast preflight delegates to it. Partitioning, triangle
order, node records and query traversal are unchanged. On a 65,537-triangle,
eight-triangles-per-leaf fixture, node capacity falls from 131,074 to the 16,385
nodes actually used. Calling-thread requested construction bytes fall from
14,418,140 to 6,160,532, with four allocations and no reallocations in either case.
This measures reserved host payload, not resident-memory peaks or a latency
speedup. GPU atlas bytes and warmed query resources are unchanged.
`query_device_bytes()` reports the two allocated buffers, including empty padding.
The cap excludes host geometry/trees, model metadata, ray/refit storage, staging and
pipeline/binding objects. Buffer-size, binding-size and u32-word-address limits also
apply. This is not a process-memory cap. Unified tracing requires twelve storage
bindings per compute stage; unsupported devices reject pipeline construction.

Query parameters are immutable per binding variant, keyed by input/output buffers,
ray count and range. Encoding multiple queries cannot overwrite an earlier query's
uniform parameters. Refit levels and traversal are encoded into the existing sensor
command batch, without extra submission, pose readback or host waits.

### Periodic GPU spatial rebuilding

Owned-scene queries default to `SceneRebuildPolicy { interval: 64, min_leaves: 64,
max_device_bytes: 128 << 20, quality_ratio_percent: None }`. Eligible scenes reorder on the first nonempty encoded
query and every 64 encoded queries thereafter. Smaller scenes retain their CPU-built
topology. An interval of zero disables reordering; `min_leaves` must be at least two.
Each camera aperture sample is a separate query, not a simulation frame.

After a normal refit, a GPU pass derives 30-bit Morton keys from live world-space
leaf bounds. The existing stable four-pass radix shader sorts those keys and leaf
indices. A scatter pass places instance records into the tree's depth-first leaf
slots and updates GPU leaf-to-shape IDs. The instances copy back into their retained
buffer, then another refit updates all bounds before traversal. Tree edges and
escape links remain fixed, the tree stays balanced, mesh geometry is untouched,
and dense/stable object IDs do not change.
All parameters and sort bindings are retained and immutable. Rebuild encoding makes
no queue submission, host pose/bounds download or CPU sort.

A scheduled reorder adds 14 compute dispatches, one 80-byte-per-leaf device copy,
and another refit (one dispatch per tree level). This is a bounded periodic heuristic,
**not** a measured-quality trigger or a guarantee that every reorder improves an
already good tree. Overlapping/degenerate scenes can gain nothing. Morton ordering
may also be worse than the initial median tree for some distributions. The optional
quality trigger below avoids some unnecessary reorders; candidate-tree accept/reject
and workload-dependent cadence remain performance work.

```rust
use molla::sensors::{SceneRebuildPolicy, SceneRebuildStorage};

let extra = SceneRebuildStorage::for_leaves(10_000)?;
raycaster.set_scene_rebuild_policy(SceneRebuildPolicy {
    interval: 32,
    min_leaves: 64,
    max_device_bytes: extra.device_bytes,
    ..Default::default()
})?;
```

`Raycaster`, `LidarBatch` and `TiledCamera` expose the same policy getter/setter;
`CameraBatch` takes a group index so heterogeneous groups retain independent policies.
Changing a policy discards the cached hierarchy and applies it on the next query.
Invalid changes to an existing cached scene leave that scene/policy intact. New-scene
rebuild-budget checks run before input upload, output reservation or query preparation;
a later camera group's failing budget is checked before earlier groups encode.
An unchanged policy keeps warmed resources.

`SceneRebuildStorage` estimates additional fixed GPU payload only. For `L >= 2`,
with `C = max(L, 256)`, bytes are `92*L + 8*C + 1024*ceil(C/256) + 80`.
The estimate includes alternate instances, leaf destination indices, key/value pairs,
radix scratch/bins and immutable parameters. A GPU fixture compares it with allocated
buffer sizes. Host construction data, pipelines/bindings, driver overhead, existing
hierarchy/scene/ray buffers and in-flight older resources are excluded. Workgroup and
per-buffer device limits apply independently; this is not a peak-memory guarantee.

`query_stats().scene_rebuilds` counts **encoded**, not necessarily executed, reorders.
Cancelled batches may advance the cadence but cannot partially mutate GPU topology;
later queries remain correct and already leased results remain unchanged. Backpressure
and empty queries do not advance it. The standalone `SceneAcceleration` diagnostic API
does not reorder: its borrowed CPU topology remains valid. Only the private query-owned
hierarchy uses GPU leaf permutations, which are not exposed through that diagnostic view.

### GPU quality-triggered rebuilding

Set `quality_ratio_percent: Some(125)` to check for a cost increase above 25 percent
at each eligible query instead of rebuilding unconditionally. The value must exceed
100. `interval` still controls the check cadence, `min_leaves` still gates small
scenes, and zero interval disables both modes. The default remains periodic until
the performance matrix supports changing it.

```rust
let extra = SceneRebuildStorage::for_quality_leaves(10_000)?;
raycaster.set_scene_rebuild_policy(SceneRebuildPolicy {
    interval: 32,
    quality_ratio_percent: Some(125),
    max_device_bytes: extra.device_bytes,
    ..Default::default()
})?;
```

The GPU sums internal-node AABB surface areas divided by the root's surface area.
Root-scaled extents keep intermediate products bounded. The first executed check
reorders and records a baseline after refitting. Later checks rebuild when cost
exceeds the configured multiple of the baseline. A rebuild resets the baseline to
its new cost; a cheaper refit lowers the baseline without rebuilding. This prevents
a previously overlapping layout from permanently suppressing later deterioration
checks. Cost is clamped to at least one for degenerate extents.

GPU-written indirect arguments enable or suppress the Morton, radix, scatter and
instance-commit kernels. A skipped rebuild never copies old scatter scratch back
into current instances. No quality data or poses are downloaded to decide, and no
query-time CPU sort, blocking wait or quality-resource construction is required.
The final refit and baseline reduction run in the same command stream before ray
traversal. Cancelling an unsubmitted batch does not update GPU quality state.

Additional fixed payload above periodic storage is
`84 + 4*ceil((L-1)/256)` bytes for `L >= 2`: partial sums, 32-byte diagnostic state,
36-byte indirect arguments and 16-byte parameters. Disabled/small scenes allocate
none of this. Preflight includes these bytes before uploads or output reservations.
The independent raw-kernel fixture compares the estimator to actual buffer sizes.

Each scheduled check encodes 19 compute dispatch commands plus a second refit,
even if 15 indirect commands select zero workgroups. This command/refit/reduction
overhead is real; quality mode is not automatically faster. The metric is a
hierarchy-cost heuristic, not measured ray latency or a guarantee that a new Morton
ordering improves the candidate. Highly overlapping geometry can remain expensive.

`query_stats().scene_quality_checks` counts encoded checks, including cancelled
batches. `scene_rebuilds` continues to count unconditional periodic encodes only;
neither counter claims how many conditional rebuilds executed. Low-level kernel
diagnostic state retains baseline/current cost, wrapping u32 executed check/rebuild
counts and the last decision; query APIs do not read it back automatically.

GPU regressions cover stable/translated layouts skipping work, scrambled-pose
repair, coincident geometry and recovery, independent CPU area references, leaf-ID
and bounds preservation, cancellation, budget rejection, historical leases and
independent heterogeneous camera policies. Full lifecycle/performance acceptance
remains separate from these focused tests.

### GPU instance-bounds and refit core

`SceneAcceleration` builds a balanced instance hierarchy once and refits its bounds
from live GPU body poses. The owned-scene query path uses this core internally;
its standalone API also exposes diagnostic bounds and status. The component timings
below measure only the refit core, not sensor query performance.

```rust
use molla::sensors::SceneAcceleration;

let mut hierarchy = SceneAcceleration::new(&engine, &scene, None, Default::default())?;
let bounds = hierarchy.dispatch(&engine, &scene, &state)?;
// Select active bounds/status ranges for a GPU consumer or grouped readback.
let lower = bounds.lower.readback_source(0..bounds.node_count)?;
```

Construction computes local bounding boxes for finite supported primitives and
compiled triangle meshes (including converted triangles/heightfields). Repeated
instances reuse each mesh's local bounding box. Empty triangle meshes and explicitly
unsupported/skipped shapes have no leaves. Infinite planes are listed separately
by dense shape index and must be intersected separately; they cannot be bounded by
a finite box.

Leaves retain scene dense shape order; internal nodes occupy contiguous increasing-
height ranges with children preceding parents. Splitting uses the longest centroid
axis, deterministic index tie-breaking and balanced median partitions. An optional
slice of CPU body poses seeds spatial partitioning; it must contain one valid rigid
pose per body. Seeds affect topology only. Without seeds, shape-local centers are
used, so initial partition quality may be poor when bodies are far apart. No GPU
pose is downloaded for construction or refitting.

Each dispatch reads authoritative `State::body_q`, composes body/shape transforms,
updates world leaf bounds/poses and then unions internal bounds level by level.
Separate compute passes order dependent levels; no inter-workgroup atomic barrier
or host wait is used. Level parameters are immutable buffers: there are no per-level
queue writes that could overwrite earlier encoded parameters. Two body-buffer
binding variants are cached for ping-pong physics states. A dispatch submits once;
selected collection/readback is optional and separate. Even an empty hierarchy
submits an empty command buffer but executes no compute dispatches.

Local bounds are rounded outward to f32; world bounds include a scale-aware rounding
margin of `32 * f32::EPSILON * (magnitude + 1)`, where magnitude bounds the local-box,
shape-translation and body-translation terms. The padding makes bounds conservative,
not exact geometry tests. The GPU fixtures compare every leaf against independent
f64 transformed corners and every parent against the exact union of its children.

Per-leaf status is `0` for valid bounds, `1` for invalid/nonfinite pose data or a
non-unit live quaternion, and `2` for arithmetic inputs exceeding the conservative
`2^120` magnitude guard. Status 2 does not assert that every such transform would
actually overflow. Invalid/unsafe leaves receive full finite-f32 bounds and an
identity pose; consumers **must inspect status**, not interpret that fallback pose
as valid geometry or drop those leaves as ordinary misses. A later valid refit
clears the previous status.

Owned queries propagate invalid-leaf flags through the unused lower-bound vec4
padding to the root, then reject the whole active query on GPU before traversal.
Infinite planes are validated separately, including plane-only scenes. Invalid
results carry distance/shape `-2`; synchronous `Raycaster::cast_scene` returns an
error. LiDAR reduction and camera shading preserve distinct invalid sentinels;
see [the query validity contract](contracts.md#invalid-owned-scene-results).
Valid queries intersect the original live poses, never the diagnostic identity
fallback. Raw-model compatibility queries retain their valid-pose precondition.

The returned view includes its topology, plane list, actual scene stamp and active
node/leaf counts. Empty hierarchies retain dummy buffers but have zero active counts
and no root. Enqueue consumers/readback before the next refit; these are reusable
borrowed outputs, not leased historical frames. If retaining a bounds readback across
hierarchy destruction/reconstruction, retain its CPU topology too: alternative seeds
can produce different internal node indices for the same scene revision.

Pose-only changes refit without rebuilding topology or mesh geometry. Any scene
identity/revision change rejects dispatch, requiring a new hierarchy; this includes
local-pose, geometry, ownership, addition/removal and material-triggered replacements.
Reconstruction with fresh optional seeds can improve partition quality after large
motion. This standalone diagnostic API retains fixed leaf assignments; periodic GPU
query rebuilding and optional GPU quality decisions are described above.
Stale/foreign scenes, foreign devices, mismatched nonzero body counts and
unuploaded/host-dirty required body poses reject before encoding or submission.
Bodyless scenes use an internal identity buffer and ignore unused caller poses.

`SceneAccelerationStorage::for_leaves` estimates fixed GPU buffer payload without
allocation. `SceneAccelerationConfig` defaults to one million leaves and 256 MiB;
buffer binding, buffer-size and dispatch limits apply independently. The budget
counts packed instances, topology, bounds, world poses, flags, fallback body pose
and immutable per-level uniforms, including empty dummy allocations. It excludes
CPU tables, construction scratch, staging, pipelines/bindings, readback and model/
scene geometry. It is not a whole-process or peak-memory limit. Boundary and GPU
buffer-size fixtures verify this accounting.

The component benchmark separates completed construction (including CPU topology,
GPU buffer initialization and an explicit queue flush/wait) from warmed completed
refitting. Its 1,000/10,000-instance fixtures mix one shared mesh with spheres and
one infinite plane. Model/scene/BLAS creation is outside those timings; refits do
not include pose uploads, ray traversal or readback. No linear-query comparison
or application-level speedup follows from these component measurements.

```sh
make test-gpu P=molla-sensors T=sensor_scene_refit_gpu
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_scene_refit'
```

Component measurement on 2026-09-25, HEAD `367d92c` plus this worktree: RTX 4080 /
Vulkan, NVIDIA 595.84, i7-13700KF (24 logical CPUs), Rust 1.94.1, release profile,
wgpu 29.0.4. Criterion used 10 samples, one-second warmup and three-second target
measurement per case. No other agent-owned GPU test ran concurrently. Results
below are Criterion point estimates with its reported intervals, not a query
speedup or first-process cold-shader measurement. Log: `/tmp/molla-scene-refit-bench.log`.

| Finite instances | Completed construction | Warm completed refit |
|---|---|---|
| 1,000 | 956.57 µs [949.62, 969.21] | 127.70 µs [108.57, 161.96] |
| 10,000 | 5.4830 ms [5.4219, 5.5151] | 133.07 µs [130.95, 137.80] |

The 1,000-instance construction run reported one outlier among ten measurements;
the wide small-refit interval should not be interpreted as precise scaling.
GPU completion waits are included. Completed query comparisons are reported below;
GPU timestamp-only timing and the remaining performance matrix stay open.

### Completed query comparison

`bench_scene_queries` compares the unchanged raw-model linear/per-instance path
with owned-scene hierarchy queries. Each scene has 1,000 or 10,000 finite instances
(25% spheres, 75% instances of one triangle mesh) and an infinite plane. There are
1,024 downward rays distributed over those instances. The fixed body pose is already
uploaded; every timed query includes host ray normalization/upload. Hierarchy queries
include all refit levels and traversal. This is a spatially separated synthetic
fixture, not a complex machine scene, degraded-topology case or physics benchmark.
Both paths use `Engine::new_gpu`; adopted/shared renderer devices are not measured.

`construct_and_collect` constructs a fresh Raycaster, builds required geometry/
hierarchy/pipelines and collects all four result channels. Model construction and
initial model/state uploads are outside timing. `warm_completed` reuses resources
and waits for GPU completion without readback. `warm_collect` instead performs
grouped readback and assembles CPU hits. Timings are not additive components and
do not isolate GPU kernel duration. Driver shader caches are warm; construction
is not first-process cold-start timing.

The pre-periodic-rebuild 2026-09-25 run uses the hardware/toolchain above and 10 Criterion samples,
one-second warmup, three-second measurement target. Slow cases extend measurement
to fit ten samples. No owned build/test/benchmark ran concurrently. Log:
`/tmp/molla-scene-trace-bench-final.log`. Values are milliseconds, point estimate
with reported interval:

| Instances | Path | Construct and collect | Warm completed | Warm collect |
|---|---|---|---|---|
| 1,000 | Raw linear/per-instance | 97.853 [96.897, 98.495] | 22.724 [22.511, 22.967] | 24.661 [23.880, 25.169] |
| 1,000 | Hierarchy | 208.71 [208.53, 208.92] | 0.28782 [0.28244, 0.29263] | 0.34534 [0.33887, 0.35152] |
| 10,000 | Raw linear/per-instance | 603.03 [585.21, 635.61] | 385.65 [384.13, 387.21] | 412.13 [396.92, 430.27] |
| 10,000 | Hierarchy | 230.77 [229.52, 232.23] | 1.8290 [1.8153, 1.8468] | 3.2458 [2.9454, 3.6394] |

Warm queries improve strongly in this fixture, but 1,000-instance construction
regresses from roughly 98 ms to 209 ms. Small-scene extra refit dispatches, shader/
pipeline startup and poor spatial partitions remain optimization targets. The final
10,000-instance hierarchy timings were slower than the preliminary run (1.6995 ms
completed; 2.7273 ms collected); the final collection interval is wide and includes
a high-severity outlier. The preliminary shader preceded the overflow-safe traversal
range margin. No causal attribution or application-wide speedup follows from that
run-to-run difference. Full performance-matrix and viewer acceptance remain open.

### Periodic rebuild measurements

The subsequent 2026-09-25 run enables the default periodic policy in the same
fixture. Hardware, sample counts and timing boundaries are unchanged; no other
owned build/test/benchmark ran concurrently. Log:
`/tmp/molla-scene-rebuild-bench.log`. Values are milliseconds with Criterion's
reported intervals:

| Instances | Path | Construct and collect | Warm completed | Warm collect |
|---|---|---|---|---|
| 1,000 | Raw linear/per-instance | 102.07 [97.410, 104.34] | 34.878 [33.573, 36.485] | 32.951 [27.652, 37.323] |
| 1,000 | Hierarchy, periodic 64 | 236.83 [236.30, 237.52] | 0.26217 [0.25654, 0.26881] | 0.36631 [0.35458, 0.38339] |
| 10,000 | Raw linear/per-instance | 616.14 [591.53, 656.64] | 447.69 [439.98, 456.20] | 462.55 [456.59, 468.83] |
| 10,000 | Hierarchy, periodic 64 | 261.48 [260.47, 262.50] | 1.7409 [1.7109, 1.7749] | 7.9753 [7.1372, 8.9357] |

Construction increased versus the preceding run, and the 10,000-instance warm
collection regressed from 3.2458 to 7.9753 ms. The unchanged raw-query baseline also
became slower. These separate runs do not establish whether sorting, changed
spatial ordering, readback or machine/driver state caused the collection regression.
Fixed, periodic and first-query-only policies need a same-run comparison; do not
substitute completed-query timings for synchronous collection cost.

The additional `scene_rebuild_gpu_poses_1024_rays` fixture has one body per finite
instance, the same sphere/shared-triangle mix, and a plane. Body positions are
written directly to GPU storage using permutation `(i * 997) % instance_count`;
host poses remain inaccessible. Actual poses stay fixed during timing. The initial
cast, pipeline construction and pose upload are outside timing. Each timed query
uploads 1,024 host rays, refits, traverses and waits for GPU completion, with no
result download. Periodic sorting is included when due. All policies retain one
scene build and one trace/refit binding preparation; rebuilding does not trigger
new preparation on each sample.

| Instances | Fixed local-seed hierarchy | Periodic every 64 queries | Every query |
|---|---|---|---|
| 1,000 | 0.33328 [0.30867, 0.35729] | 0.42276 [0.39272, 0.48537] | 1.1661 [1.0891, 1.3412] |
| 10,000 | 3.3121 [3.1754, 3.3891] | 0.70164 [0.65429, 0.77697] | 1.3688 [1.2898, 1.4243] |

Periodic rebuilding helps the larger scrambled layout but regresses the smaller
one. Rebuilding every query is slower than periodic reuse in both cases. This
does not measure a continuously deforming/moving layout, overlapping geometry,
real machine meshes, GPU timestamp-only duration, physics or renderer integration.
The optional quality-triggered policy is now implemented; the full performance
acceptance matrix remains unfinished.

### Quality-policy measurements

Two 2026-09-25 runs compare the same GPU-pose fixture above with optional quality
threshold 125 and intervals 64 or 1. Hardware: RTX 4080, NVIDIA 595.84, Vulkan;
Rust 1.94.1, optimized bench profile, HEAD `367d92c` plus the sensor worktree.
Each case uses ten Criterion samples, one-second warmup and three-second target
measurement. Cases run sequentially, not randomized. Logs are
`/tmp/molla-scene-quality-bench{,-repeat}.log`; per-process GPU activity was sampled
throughout into the corresponding `*-processes.log` files.

```sh
NVIDIA_VERSION=595.84 nix develop --impure -c make bench-sensors A=scene_rebuild_gpu_poses_1024_rays
```

Values are milliseconds with Criterion's reported confidence intervals, not
per-frame p50/p95. Initial construction, pose upload and first result collection
are excluded; ray upload, refit, scheduled maintenance, traversal and completion
wait are included. Poses remain fixed during timing, so these measurements exercise
quality-check/skip overhead after the first rebuild, not sustained moving geometry.

| Instances | Policy | First run | Repeat |
|---|---|---|---|
| 1,000 | Fixed local seed | 0.28618 [0.27341, 0.31367] | 0.29085 [0.26707, 0.31930] |
| 1,000 | Periodic 64 | 0.27290 [0.23752, 0.34768] | 0.42343 [0.40354, 0.44408] |
| 1,000 | Periodic every query | 0.66883 [0.59979, 0.71686] | 0.60818 [0.60138, 0.61621] |
| 1,000 | Quality 64 | 0.24071 [0.22299, 0.25756] | 0.23849 [0.22412, 0.25710] |
| 1,000 | Quality every query | 0.68784 [0.60532, 0.85385] | 0.75139 [0.69092, 0.87601] |
| 10,000 | Fixed local seed | 2.9115 [2.9011, 2.9261] | 2.9271 [2.9139, 2.9422] |
| 10,000 | Periodic 64 | 0.45472 [0.42500, 0.48905] | 0.40760 [0.34529, 0.52871] |
| 10,000 | Periodic every query | 1.0297 [0.96224, 1.1601] | 1.0435 [0.99943, 1.0952] |
| 10,000 | Quality 64 | 0.40075 [0.35488, 0.49134] | 0.46467 [0.44192, 0.50052] |
| 10,000 | Quality every query | 0.59984 [0.57346, 0.64075] | 1.0110 [0.96748, 1.0818] |

Quality checks every query cost substantially more than checking every 64 queries
in this fixture, despite skipping sorting after initialization. Quality-64 does
not show a consistent large-scene advantage over periodic-64, and the repeat has
material variation without a code change. Small-scene quality-64 results are
encouraging but are not evidence for a universal default change. The periodic
default therefore remains unchanged.

The desktop/remote-display processes remained running. The repeat includes two
sampled 3% SM-utilization readings for `gnome-remote-de`; one-second process
sampling does not prove exclusive GPU use or explain the observed variation.
No external process was stopped. Both raw runs, including regressions, are retained.
Continuously moving/overlapping layouts, construction, result collection, real
machine meshes, other ray counts and the full S08 performance matrix remain open.

### Same-scene policy control and contention check

`bench_scene_queries` also compares the CPU-seeded fixture with rebuilding disabled
(`hierarchy_fixed`), the default interval 64 (`hierarchy`), and interval `u32::MAX`
(`hierarchy_once`, effectively only the initial warmup query in these runs).
Each policy uses the same model, rays and query implementation. Initial hits are
checked against analytical shape IDs/distances. Counters assert one scene build,
one trace/refit binding preparation and the expected encoded rebuild cadence.
Both completed queries and synchronous collection are measured; construction is
excluded from these controls.

The first 2026-09-25 control run completed all twelve cases on the hardware above,
with the same Criterion settings. Log: `/tmp/molla-scene-policy-control.log`.
Values are milliseconds with reported intervals:

| Instances | Policy | Warm completed | Warm collect |
|---|---|---|---|
| 1,000 | Periodic 64 | 0.19998 [0.19713, 0.20469] | 0.26896 [0.23426, 0.28971] |
| 1,000 | Fixed | 0.19919 [0.18996, 0.21359] | 0.22196 [0.21764, 0.22655] |
| 1,000 | Initial only | 0.20559 [0.20222, 0.21514] | 0.22832 [0.22532, 0.23369] |
| 10,000 | Periodic 64 | 0.74172 [0.69992, 0.80424] | 0.80190 [0.79085, 0.83097] |
| 10,000 | Fixed | 0.78361 [0.63778, 0.91389] | 0.94713 [0.94122, 0.95496] |
| 10,000 | Initial only | 0.84012 [0.83086, 0.84534] | 0.96337 [0.90701, 1.0588] |

The previous 7.9753 ms collection result did not reproduce in this run, without a
query/readback implementation change. This is not proof of a fix or proof that
periodic rebuilding always wins. The cases run sequentially, not interleaved;
machine/driver variability and the small-scene collection overhead remain relevant.
Criterion intervals are not per-frame p50/p95 latency measurements.

The repeat via the new `bench-sensors` target was interrupted deliberately after
external GPU contention was observed. An independently launched `gearbox` process
registered 99% SM utilization in `nvidia-smi pmon` while the 1,000-instance fixed
case increased from 0.19919 to 3.8331 ms. A subsequent saved snapshot also shows
both GPU processes (`/tmp/molla-scene-policy-contention.log`). That partial run,
`/tmp/molla-scene-policy-control-repeat.log`, is not uncontended performance evidence.
Only the owned benchmark was stopped; the external application was untouched.
This establishes contamination of that repeat, not the cause of older measurements.
Check per-process GPU activity during future timing runs, not just process names
or total utilization before starting. Repeatability and full acceptance remain open.

```sh
make bench-sensors A='scene_queries_1024_rays/hierarchy.*warm'
make bench-sensors SENSOR_BENCH=bench_sensor_lidar A=--test
```

The second command is a smoke-test example, not a timing measurement. The target
always selects the GPU feature and release benchmark profile and uses `RUN_WITH`.

```sh
make test-gpu P=molla-sensors T=sensor_scene_trace_gpu
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_scene_queries'
```

```sh
make test-gpu P=molla-sensors T=sensor_scene_gpu A='--test-threads=1'
make test APP_TARGET='-p molla-geometry --features gpu --lib'
make test-gpu P=molla-geometry T=gpu_narrow_phase A='--test-threads=1'
```
