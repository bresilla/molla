# Sensor geometry, frames and numerical contracts

This inventory describes the current sensor APIs, not every geometry type supported
by the physics solver. GPU sensors compute in f32 even when their Rust `Real`/`Vec3`
outputs are f64. Downloading does not restore precision lost on the device.

## Geometry routing

| API family | Geometry source and capabilities |
|---|---|
| Owned-scene calls on `Raycaster`, `LidarBatch`, `TiledCamera` and `CameraBatch` | Shared `SensorScene` visibility: analytic sphere, plane, box, capsule, cylinder, cone and ellipsoid; rigid static/body-attached triangle meshes; conversions and exclusions below |
| Raw-model `Raycaster`, `LidarBatch`, `TiledCamera`, `CameraBatch`; legacy `Lidar`/`AerialLidar` | Same seven analytic primitives and static/body-attached TriMesh; skips standalone Triangle, Heightfield, ConvexHull, RoundCylinder and SDF |
| Exported low-level `CameraTraceKernel` | Seven analytic primitives only; no mesh pass or owned-scene conversion. Public `TiledCamera` uses the shared query path instead |
| CPU `RayScene` / `DiffRay` adjoint helper | Spheres with parameterized centers, fixed Y and Z planes. No mesh traversal or generic SDF query; finite-difference gradients, not a CPU fallback for GPU sensor APIs |
| IMU | Body poses, sampled linear/angular velocities, COM, gravity and local mount poses; does not intersect shapes |
| Contact sensor | Existing solver contact arrays and body-contact CSR; does not run collision detection or expand supported collision geometry |
| Frame-transform sensor | Body poses and local target/reference poses; no surface query |
| Image/cloud processing, grids, annotation and export | Consume existing pixels, points, IDs and labels; do not introduce new scene visibility or geometry support |

Owned compilation converts standalone triangles to one-triangle meshes,
heightfields to triangles with the simulation grid diagonal, and nondegenerate
convex hulls to triangulated surfaces. RoundCylinder is a **dilated-cylinder
approximation**, not an exact rounded rim. SDF and degenerate convex hulls are
unsupported. Default compilation rejects unsupported shapes. Opt-in
`ReportAndSkip` records omissions; their absence from a hit is not evidence of
free space. See [geometry snapshots](scene.md#geometry-capabilities).

Source inventory: `scene.rs`, `raycast.rs`, `lidar_batch.rs`, `tiled_camera.rs`,
`camera_batch.rs`, `raycast_adjoint.rs`, `imu.rs`, `contact.rs`, `frame_transform.rs`
under `crates/molla-sensors/src`, and the corresponding kernel shaders. Shape
variants are defined in `crates/molla-sim/src/builder.rs`.

## Constructor host-array failures

Ray, IMU, contact, frame-transform, batched-LiDAR, tiled-camera, image, annotation,
cloud and cloud-grid constructors reserve their fixed host arrays fallibly after
capacity/budget checks. A failed reservation returns `Error::Storage`; successful
array contents and output layouts are unchanged. Camera pose packing and LiDAR
descriptor/range arrays use the same reservation path. CPU helper tests cover
capacity overflow before initialization; the GPU allocation-injection regression
checks one failed payload reservation and a subsequent successful construction for
each of these ten families.

This is not a general process-out-of-memory recovery guarantee. Scene/model copies,
some metadata allocations, error-message allocation, driver staging and GPU resource
exhaustion have separate behavior. The named payload budgets are not process-memory
caps. See the sensor-specific pages for exactly which arrays each budget counts.

## Frames, units and validity

- Positions, distances and geometric dimensions are meters; simulation times are
  seconds; angular settings and angular velocity use radians and radians/second.
- An instance pose is **body-world pose × shape-local pose**. Static shapes use
  only their local pose. Rays enter in world coordinates; hit points and normals
  return in world coordinates.
- `Transform` contains translation and quaternion, not scale/shear. Owned
  compilation rejects non-finite/f32-overflowing translations and non-unit
  quaternions (squared-length tolerance `1e-6`). A quaternion is not a scale carrier.
  Bake mesh scale into vertices or use primitive dimensions. The independent mesh
  fixture includes nonuniform baked scaling with a reflected axis; query transforms
  themselves remain rigid. There is no affine-matrix import in these sensor APIs.
- Live GPU body poses must remain finite and rigid. Owned-scene queries flag every
  active ray/pixel when any included instance has an invalid pose or unsafe bounds;
  synchronous `cast_scene` returns an error. Raw-model and low-level kernel calls
  still require valid poses from the caller. Host compilation checks do not
  validate future GPU poses. See [invalid query results](#invalid-owned-scene-results).
- Finite nonzero public ray directions are normalized before intersection. Hit
  distance is meters, independent of direction magnitude. GPU host-ray calls
  reject invalid rays for the batch. The CPU adjoint helper instead returns
  infinity for invalid rays, retaining its non-`Result` interface.
- Primitive normals are outward (plane normals follow the configured local
  normal), including for rays exiting a solid. Mesh triangle normals face against
  the incident ray. These are different conventions, not interchangeable backface
  tests. Normals are unit vectors away from singular surface points.
- Hit range is strictly `0 < distance < max_distance`. Device ray/LiDAR misses
  carry distance `-1`, signed shape ID `-1`, zero point and normal `+Y`. Collected
  `RayHit` misses use infinity, `None`, `BodyId::NONE`, zero point and normal `+Y`.
- Camera miss depth is `-1`, shape ID `u32::MAX`, normal zero, color/albedo zero.
  Disabled direct-output channels have empty active ranges; legacy disabled
  channels retain old storage. Neither case is a new valid observation.
- Dense IDs belong to a scene/revision. Sample time, scene identity, channel
  validity and output leases must travel with delayed data; see
  [device outputs](device_results.md) and [scheduling](scheduler.md).

### Invalid owned-scene results

`SensorScene` queries fail closed for the entire scene, not just rays aimed at the
invalid object. This avoids claiming free space or a closest hit while an object's
location is unknown. The existing refit propagates finite-instance invalid flags
to the hierarchy root; traversal validates infinite planes separately. No new GPU
dispatch, output channel, payload allocation or synchronous readback is added.

| Output | Invalid scene | Ordinary miss |
|---|---|---|
| Ray/LiDAR distance | `INVALID_SCENE_DISTANCE` (`-2`) | `-1` |
| Ray/LiDAR signed shape index | `INVALID_SCENE_INDEX` (`-2`) | `-1` |
| Ray/LiDAR point and normal | both zero | zero point, +Y normal |
| LiDAR first/last sub-ray indices | `INVALID_SCENE_INDEX` (`-2`) | `-1` |
| LiDAR hit count | zero | zero |
| Camera radial depth | `INVALID_SCENE_DISTANCE` (`-2`) | `-1` |
| Camera unsigned shape index | `INVALID_SCENE_SHAPE_ID` (`u32::MAX - 1`) | `u32::MAX` |
| Camera color/albedo/normal | zero | zero |

Sentinels are exact and public from `molla_sensors` and `molla::sensors`. Range
clipping and LiDAR return reduction do not erase them. Camera aperture passes
preserve center-sample depth/identity. Enable **DEPTH or SHAPE_INDEX** to observe
validity; color alone cannot distinguish an invalid scene from an empty image.
An active `SensorFrame` channel means the GPU wrote that range, not that its
geometry was valid. Dispatch/submit/readback success is not a numerical-validity
check. Consumers must inspect the sentinel rather than treat every negative
distance as an ordinary miss. Point filters/annotation reductions can discard
invalid samples, so retain source distance/identity when validity matters.

GPU pose checks reject non-finite translation/quaternion values, quaternion
squared-length error above `2e-5`, and source translations/bounds above the
conservative `2^120` magnitude guard. This guard is not an accuracy guarantee at
large coordinates. Shape-local compilation uses its stricter host tolerance.
The next valid refit clears the flag; retained result slots remain historical.
Only included geometry is diagnosed: empty active queries carry no validity
observation, and `ReportAndSkip` omissions remain omissions. Non-query sensors
and legacy raw-model APIs do not acquire this owned-scene validity contract.

`sensor_query_validity_gpu` covers NaN/infinite/unsafe positions, invalid
quaternions, primitives/meshes/plane-only scenes, root aggregation across 66
leaves, periodic/quality-triggered rebuilds, direct/standalone LiDAR and aperture
cameras, per-scan clipping, delayed historical frames and recovery.

### LiDAR axes

Terrestrial theta is zenith from the corrected local up axis. Phi zero follows
forward; phi pi/2 follows **up × forward**. With default forward `+Z` and up `+Y`,
an equatorial scan visits `+Z, +X, -Z, -X`. The up vector is orthogonalized against
forward. Axis ranges include both endpoints, so `[0, 2*pi]` duplicates the seam;
a one-element axis samples its lower endpoint. Layout is theta-row/phi-column,
then contiguous sub-rays per pulse. Aerial scans use world-Y-down nadir directions
and Halton-distributed origins across the footprint's upper X/Z face.

First/last returns select nearest/farthest hits **among divergent sub-rays**, not
multiple penetrations along one physical beam. Each scan retains its own range,
layout, pose and identifier.

### Camera axes and renderer boundary

Camera local axes are `+X` right, `+Y` up, looking along `-Z`. Pixels sample their
centers; row zero is the image top. Layout is camera-major, then row, then column.
Depth is radial distance along the normalized pixel ray, **not optical-axis Z**.
`near` remains informational; `max_distance` clips radially. Aperture samples
average color while geometry/IDs retain the center sample.

This sensor traces the Molla geometry snapshot; it does not capture a window or
automatically use whichever renderer an application selects. Colors come from
the sensor's shape colors and directional/ambient shading, not an arbitrary
renderer’s textures, shadows or vegetation. A Gearbox or other renderer adapter
must supply coherent poses/scene data and consume the sensor image through the
device/readback interfaces. Camera links and transfer costs remain S09 work;
the shared-camera pipeline is not proof of completed renderer integration.

### Other sensors

IMU outputs sensor-local specific force and angular velocity. Its acceleration is
`a_COM - gravity + alpha × r + omega × (omega × r)`, rotated world-to-sensor, with
the lever arm measured from COM. The first sample primes device velocity history;
it is not a zero-acceleration measurement. Invalid/nonpositive sample `dt` errors.

Contact depth and centroid are geometric meters/world coordinates; signed normal
sums are unweighted and are not forces or impulses. Frame outputs compose body
and local poses; relative pose is `inverse(reference_world) × target_world`.
An optional reference consumes an internal slot excluded from returned target
ranges. Neither sensor independently interpolates simulation state.

## Numerical acceptance fixtures

The following are explicit fixture budgets, not universal accuracy guarantees.
For f32, absolute position uncertainty grows roughly with
`f32::EPSILON * coordinate_magnitude`; transformed/intersection operations compound
that error. Normal sensitivity also depends on local feature size and proximity
to corners/tangencies. Do not accept a large-coordinate oracle with an unchanged
small-coordinate absolute tolerance, or silently relax a failing fixture.

Long-ray fixtures prescribe metric distance error below two f32 ULPs at the
distance magnitude plus 2e-4 m. Nearby finite-surface points and normals retain
4e-4 absolute/vector budgets even when the ray starts 1,000,000 m away; normals
must remain finite and unit within 2e-6. Axis-aligned analytic references use
the represented primitive dimensions, including inside/outside silhouette rays.
Grazing plane fixtures use represented slopes through 1e-15 and distinguish finite
intersections from exactly parallel misses. These are fixture budgets, not a
uniform guarantee for every possible direction or distance. Oblique sphere tests
use independent f64 closest-approach/chord references from represented f32 rays;
another fixture checks that rotating a sphere does not change its geometry.
Adjacent-f32 unit-sphere silhouette rays require exact inside/tangent/outside
classification and normal error below 1e-4.

Raw and owned primitive queries and the low-level camera now share intersection
code. They power-of-two scale directions without changing represented component
ratios, track ray parameters separately from metric distances, and condition distant
finite-shape rays near conservative bounds before world-to-local rotation. Local
surface points are retained for normal reconstruction rather than reconstructed
from the rounded full hit distance. Planes are unbounded and skip this rebasing.
These fixtures do not establish long-range triangle-mesh or universal near-tangent
accuracy; those remain separate acceptance work.

Legacy per-mesh queries use the same power-of-two direction convention and return
metric distances; the merge stage reconstructs points using the corresponding ray
parameter. The low-level mesh fixture exercises direction scales 1e-30, 1 and 1e30,
zero directions, range clipping and hit/miss/hit slot reuse. Existing camera tests
still require exact raw/owned depth, normal, color, albedo and identity parity.

The large-coordinate fixture separates input representation from intersection
arithmetic: its ray origins and instance translations are explicitly rounded to
f32 before constructing independent f64 surface references. It spans signed world
offsets through 1,000,000 m, with local ray distances below 10 m. Its prescribed
distance budget is 2e-4 m, normal-vector budget 4e-4, and world-point budget two
f32 ULPs at the largest absolute target coordinate plus 2e-4 m. These are not
submillimeter world-position guarantees for unquantized inputs at that scale.
Primitive queries subtract the instance position before rotating the ray origin
into local space. This preserves nearby offsets instead of cancelling two rotated
large world positions. Raw, owned and low-level camera paths use this ordering;
mesh traversal already used it. This does not recover precision lost when inputs
or composed instance translations are rounded to f32.

| Fixture | Independent reference / required comparison |
|---|---|
| `sensor_contracts_gpu`: rigid primitives and baked mesh | Known f64 surface points/normals, composed body/local rotations, live device-only pose updates, inside/outside rays and direction scales `1e-250..1e250`. Coordinates below 20 m, primitive radii at least 0.4 m. Distance/point error `<2e-4` m; normal vector error `<4e-4`; unit-normal error `<2e-6`; IDs/owners exact |
| `sensor_contracts_gpu`: capsule boundaries | Exposed hemispheres, axial/oblique inside/outside rays and zero-height sphere limit. Distance/point/normal errors `<2e-5`; IDs exact |
| `sensor_contracts_gpu`: camera plane | Analytic radial depth on a 9×9 pinhole image, error `<2e-5` m; arbitrary local plane normal, informational near, exact miss sentinels at strict far range |
| `sensor_contracts_gpu`: LiDAR axes | Fixed cardinal directions and independently rotated axes; CPU error `<1e-12`, GPU error `<2e-6` |
| `sensor_scene_gpu`, `raycast_trimesh_gpu` | Independent f64 triangle intersections, moving/rotated meshes, normal/point and metric-distance checks; typical moderate-coordinate limits `1e-4` m |
| `sensor_scene_trace_gpu` | Independent many-instance triangle distances `<5e-5` m; separate raw/owned comparisons at large coordinates. Raw/owned parity alone is not an independent geometric oracle |
| `sensor_large_coordinates_gpu`: analytic surfaces | Independent f64 sphere/plane/box/capsule/cylinder/cone/ellipsoid/mesh surface points and normals for raw and owned paths; six signed offsets through 1,000,000 m, scaled directions `1e-250..1e250`, inside/backside rays and live device-only poses. Distance `<2e-4` m, normal-vector error `<4e-4`, unit-normal error `<2e-6`; world points use the ULP budget above; IDs/owners exact |
| `sensor_large_coordinates_gpu`: camera plane | Independent f64 plane depth for raw/owned cameras and the low-level camera kernel at six offsets; depth `<2e-4` m and normal error `<4e-4`, exact shape ID |
| `sensor_large_coordinates_gpu`: moving meshes | 66 shared-mesh instances, independent brute-force f64 triangle intersections at four offsets, changing pose/instance ordering, periodic and quality-triggered rebuild policies; same distance/normal/point budgets, exact hits/misses/IDs/owners and no mesh reconstruction |
| `raycast_adjoint` | CPU metric sphere/plane distances within `1e-12`, scaled-direction gradient within `1e-8` of analytic derivative, invalid-ray infinity and existing optimization convergence |

Exact-distance ties prefer analytic primitives over meshes, then the smallest
dense shape index within the group. Equal triangles within one mesh follow local
BVH traversal order. First-sub-ray/last-sub-ray selection uses first/last ties
respectively. ID equality is required away from ties; tiny f64 separations that
round together in f32 cannot promise the same winner as an f64 oracle.

These fixtures do not complete the full S00/S04 numerical/geometry matrix,
grazing/degenerate-surface analysis or performance
acceptance. Run `make test-sensors` for host, feature and explicit GPU lanes;
`make test-gpu P=molla-sensors T=sensor_contracts_gpu A='--test-threads=1'`
isolates the GPU contract fixtures.
