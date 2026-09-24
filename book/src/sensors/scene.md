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
builds/uploads one BVH per referenced mesh identity, not per instance or sensor.

## Coordinates and live poses

Each instance uses `body world pose * shape local pose`; static instances use
only the shape pose. Mesh vertices/BVH bounds stay in local coordinates. GPU
queries transform rays into that frame and normals back to world space, keeping
distance in meters. Triangle normals oppose the incident ray.

Transforms are rigid: translation and unit quaternion, with no scale component.
Compilation rejects non-finite/non-unit shape transforms and invalid primitive
dimensions. Live `State` body poses must also be finite rigid poses on the same
engine and have the expected body count. Their device authority is checked;
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
whose geometry did not change. Stable application handles preserved across dense
remapping are not implemented yet.

An already enqueued readback retains its old snapshot data. Store the scene
identity/revision alongside its `ReadbackStamp`; do not resolve it against a
new revision. Geometry revisions do not identify simulation pose time. The
caller still supplies sample IDs, simulation time and pose synchronization.

## Geometry capabilities

| Geometry | Owned scene support |
|---|---|
| Sphere, plane, box, capsule, cylinder, cone, ellipsoid | Analytic GPU intersections |
| TriMesh, static or body-attached | Shared local BVH, rigid GPU instance transforms |
| Standalone Triangle | One-triangle local mesh conversion |
| Heightfield | Full local triangulation at snapshot compilation |
| ConvexHull, RoundCylinder, SDF | Unsupported; rejected by default |

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
needs at most 64 pending nodes, matching the shader stack. Unit tests accept a
depth-64 tree and reject depth 65; the median-split builder produces much
shallower trees for supported buffer sizes. Raw GPU node buffers must not be
mutated into an invalid tree after validation.

There is **no top-level scene acceleration yet**. Queries still loop over all
primitives and encode a traversal/reduction pair per mesh instance, but now submit
the complete query once. Retained bindings support two physics-buffer variants
without new query uniforms/bind groups after warm-up. Scene ownership and shared
local BVHs do not remove per-instance traversal/encoding overhead. Top-level
acceleration/refits, stable application handles, cross-sensor execution and
controlled performance acceptance remain in `SENSORS.md`.

```sh
make test-gpu P=molla-sensors T=sensor_scene_gpu A='--test-threads=1'
make test APP_TARGET='-p molla-geometry --features gpu --lib'
make test-gpu P=molla-geometry T=gpu_narrow_phase A='--test-threads=1'
```
