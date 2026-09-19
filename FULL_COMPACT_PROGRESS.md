# FULL_COMPACT implementation checkpoint

The objective is the entire `FULL_COMPACT.md`, including the Gearbox backend
and its live acceptance scenes. This is a checkpoint, not a completion report.

## Current implementation

- `molla_sim::runtime::RigidScene` owns a rigid-scene description and dense
  `Model`/`State`/`Control` snapshot. Body, collider and joint handles are opaque
  monotonic `u64` values, distinct from the low-level dense IDs. Successful
  mutation changes the revision; rejected mutation leaves the scene intact.
- Body/collider/joint insertion and removal rebuild the snapshot and copy
  surviving body poses, velocities, forces and controls, plus joint coordinates,
  velocities and motor controls. Removing a body cascades to its colliders and
  incident joints, not its surviving child bodies. Handles are not recycled.
- Directed joint forests are compiled in deterministic parent-first order.
  Attaching a root to another articulation merges them; detaching creates a new
  free root from the body's current pose and COM velocity. Cycles and multiple
  parents are rejected. Reverse-edge re-rooting is not implemented yet.
- Fixed and kinematic root kinds retain physical mass/inertia while exposing
  zero inverse mass/inertia. Dynamic roots get free joints automatically;
  immovable roots get fixed world anchors. Dynamic/fixed root transitions keep
  the current pose; fixing zeroes velocity. Changing a non-root to fixed is
  currently rejected and still needs a re-rooting implementation.
- Runtime collider replacement supports ball, cuboid, Y capsule, endpoint
  capsule, Y cylinder, round cylinder, convex hull, triangle mesh and native
  rectangular heightfield, with local pose, friction, restitution and tags.
  Indexed-mesh convex decomposition now lowers to multiple convex pieces
  owned by a single stable collider handle.
- `molla_geometry::decomposition::convex_decomposition` runs VHACD offline
  during authoring and reconstructs hulls from the indexed surface. It uses
  pinned `parry3d-f64 = 0.26.1` with deterministic, non-parallel features for
  this geometry operation only; it does not introduce Rapier or replace
  Molla collision/Featherstone stepping. The same Parry version is already in
  the clean Gearbox physics worktree's lockfile. New dependencies were resolved
  from the local Cargo cache; existing dependency versions were not refreshed.
- Decomposition removes unused input vertices, normalizes its working frame,
  validates authoring limits/indices/solid geometry, reconstructs original
  units and sorts output pieces/points. Runtime convex hulls/pieces reject
  non-finite, empty and coplanar clouds. Algorithm failure returns an error,
  not a fallback hull which silently fills concavities.
- `RigidScene::collider_indices` exposes every dense piece and the inverse
  map resolves every piece to its stable owner. `collider_index` remains for
  single-piece colliders only. Materials, groups, flags, pose edits, removal
  and replacement apply to all pieces. AABBs union piece bounds; ray queries
  choose the nearest piece; contact readout aggregates pieces under the one
  logical collider pair. Compound shape views report `Other`.
- Endpoint capsules retain the authored collider frame and lower to a posed
  Y capsule; coincident endpoints lower to a sphere. Collider descriptor,
  local/world pose reads and local/world pose writes are available without
  stepping. Geometry-frame, world and arbitrary-pose AABBs use the same shape
  bounds as broad phase, including the capsule's canonical-frame offset.
- Collider descriptors now carry validated density or explicit total mass,
  defaulting to density 1. Mass-only edits retain prepared query geometry.
  `molla_geometry::mass_properties` computes collider-frame COM/full inertia
  and combines enabled colliders (including sensors) in the parent body frame.
  Primitive/hull/indexed-mesh integrals use the existing pinned Parry geometry
  dependency; compounds sum pieces and explicit mass scales the whole collider,
  not each piece. Mesh calculations normalize coordinates and exclude unused
  vertices. Rounded cylinders use their inner cylinder's mass, matching Parry;
  heightfields have zero volume (an explicit mass can still supply point mass).
- `molla_sim::MassProperties` validates finite nonnegative mass and symmetric,
  physically admissible inertia; it exposes principal moments/axes, transformed
  properties, parallel-axis combination and a scale-relative pseudoinverse.
  Runtime dynamic inverse tensors remain finite for singular principal axes;
  fixed/kinematic inverse tensors remain zero. This does not yet establish
  Featherstone/Rapier torque-response parity for singular-inertia bodies.
- Body descriptors now persist optional additional mass. Creation with an
  additional component uses it as the initial total before colliders exist;
  legacy descriptors without it retain their explicitly authored totals.
  `RigidWorld::set_body_additional_mass` replaces the stored component and
  recomputes atomically. `recompute_body_mass(body)` reuses the stored component,
  adding enabled collider contributions without accumulating previous totals.
  Recompute remains explicit after shape/density/pose edits; the Gearbox adapter
  must invoke it at the appropriate mutation boundaries.
- Runtime total-mass edits rebase COM velocity and both applied-wrench buffers
  to preserve the body-origin velocity field and world force application
  points. Joint coordinates/rates/controls remain unchanged. Invalid mass,
  inertia or rebasing overflow rejects the entire edit transactionally.
- Runtime body reads expose pose, COM velocity, world COM, velocity at a point,
  descriptor/tag and collider handles. Pose edits teleport a body's subtree;
  pose/velocity edits are projected onto any incoming joint's freedoms and
  reconstruct descendant kinematics immediately. Free roots and prescribed
  fixed/kinematic roots retain their requested poses, including across rebuilds.
  Collider queries see teleports without stepping and retain prepared geometry.
- World COM wrenches, forces at points and force reset are runtime operations.
  Articulated impulses use CRBA and point Jacobians to update generalized rates,
  then reconstruct all affected COM velocities. They use the whole articulation,
  not independent-body inverse masses, and do not integrate poses or persist as
  forces. Point impulses use the current world COM. Fixed/kinematic bodies ignore
  impulses. State changes reject non-finite input/output atomically.
- CRBA and body Jacobians now use the translated angular pivot for free/D6/
  distance joints and omit disabled/immovable motion columns. CRBA references
  positions to the articulation root; Jacobians reference the target point.
  This avoids subtracting large origin-based inertia terms for translated rigs.
  Singular impulse systems explicitly constrain zero-inertia angular directions;
  a generalized-coordinate pseudoinverse alone was incorrect with an offset COM.
  Continuous-force Featherstone singular-inertia parity still needs verification.
- `step_with_forces` / `step_with_collision_and_forces` evaluate transient body
  force elements on every fresh substep state. The scene argument supplies stable
  handles/descriptors; the separate state argument supplies current kinematics.
  Transient wrenches are restored to the persistent input after each substep,
  preventing accumulation. Callback failure leaves `RigidWorld` state unchanged.
  Per-body linear/angular damping uses this force path with bounded substep
  coefficients. Tyre/soil adapters still need to be connected to the hook.
- `molla_geometry::collider_query` supplies primitive/convex/other shape views.
  Convex views have compact hull vertices and sorted true edges, excluding
  coplanar triangulation diagonals. `RigidWorld` exposes the shape-view and
  AABB queries by stable collider handle.
- Round cylinders are exact cylinder-plus-sphere support maps on CPU GJK/EPA,
  with expanded analytic bounds and multi-point plane contacts. The stored
  height/radius describe the inner cylinder, matching Gearbox's actual Rapier
  constructor (not its contradictory "shrunk" trait comment). Parry documents
  this as sphere dilation:
  https://docs.rs/parry3d-f64/latest/parry3d_f64/shape/struct.SharedShape.html#method.round_cylinder
  GPU pipelines retain the CPU fallback rather than dropping these shapes.
  Direct GPU primitive kernels still do not implement round cylinders.
- Heightfield bounds now include `origin.y`, matching the existing triangle
  and hull-contact paths; the omission could cull elevated terrain contacts.
- `Heightfield::from_scaled(rows, cols, heights, scale)` constructs a centered
  row-major grid with independent X/Z spacing and baked Y-scaled heights.
  The low-level struct now requires `cell_size_z` alongside X's `cell_size`;
  all workspace literals were migrated. Dimensions, scales, samples and
  resulting coordinates are validated before attachment. Zero/negative Y
  scale is supported; horizontal scale must remain positive.
- Heightfield cell triangles, height/normal sampling, hull contacts and CPU
  ray queries share the anti-diagonal used by Parry's default heightfield
  (verified in local `parry3d-f64-0.26.1/src/shape/heightfield3.rs`). The old
  opposite-diagonal triangle path and bilinear hull path disagreed on curved
  cells. Rotated terrain contact culling now transforms all eight query-box
  corners; hull penetration and midpoint use the facet-normal projection.
- `molla_geometry::raycast_cpu` and `RigidWorld::cast_ray[_filtered]` return
  stable collider/body handles, metric distance, point and normal without a
  GPU or stepping. Directions are normalized, matching Gearbox's Rapier
  implementation. Enabled sensors participate; disabled colliders do not.
  Solid primitives/hulls return zero from inside, limits are inclusive, and
  filtered queries select colliders before geometry work. Body pose changes
  are read directly, including local shape offsets and nonzero COM.
- Terrain rays traverse only crossed X/Z cells with a bounded DDA; vertical
  rays avoid scanning the height array or constructing mesh acceleration.
  Other primitives use bounded signed-distance minimization/root refinement
  (spheres have a direct analytic path), hulls use facet clipping, and meshes
  use the CPU BVH. `CpuRaycaster` now lazily retains local bounds, mesh BVHs
  and hull facet planes; `RigidWorld` owns a mutex-protected persistent cache.
  Geometry preparation happens only after filtering and bounds intersection.
  The free ray functions remain one-shot conveniences.
- Opaque reference-counted scene/collider stamps distinguish fresh worlds
  even when their numeric revisions and handles coincide. Collider replacement
  and deletion invalidate the corresponding prepared data; pose, body and joint
  edits preserve unrelated geometry preparation. Contact readout also uses a
  scene stamp, preventing stale pairs after replacing `RigidWorld::scene` with
  a different scene at the same numeric revision. Shared query reads retain
  the `Send + Sync` contract required by Gearbox.
- CPU mesh BVH rays now include origin/maximum-distance hits and handle
  parallel slab-boundary rays without `0 * infinity` NaNs. The triangle
  intersection is shared with the terrain traversal.
- CPU Featherstone now publishes world COM/angular velocities. Its moving
  free/D6/distance angular subspace includes the translated joint origin and
  the associated velocity bias. The acceleration solve includes the bias in
  the projected right-hand side. Fixed links propagate mass and applied forces
  to their parent. Regression tests cover world-translation invariance and
  an analytically checked fixed-link load.
- Fixed-body masks are consumed by CPU Featherstone, semi-implicit and XPBD,
  and the GPU semi-implicit/XPBD shaders. The GPU Featherstone implementation
  has not yet received the CPU dynamics/readback fixes; this remains explicit
  follow-up work, not verified parity.
- CPU Featherstone consumes penalty contact wrenches before ABA, with per-shape
  normal stiffness, compression damping, tangential gain and velocity smoothing.
  The spring/damper and Coulomb structure follows Newton's local
  `eval_body_contact` reference. Tangent normalization uses a Huber-regularized
  speed (not Warp's dimensionful Huber loss), preserving a strict Coulomb bound.
- `molla_solvers::rigid_world::RigidWorld` integrates the live scene with fresh
  collision detection and body-pair filtering every substep (eight by default).
  Failed/non-finite steps do not commit output state. The low-level solver's
  ordinary `step` still consumes its supplied contact set; callers needing
  refreshed contacts must use `step_with_collision` or `RigidWorld`.
- Contact buffers retain both collider and body indices. Force readout retains
  every substep sample, including impacts which end before the final substep,
  with normal force, integrated normal impulse, sample duration, friction,
  world point/normal, penetration and acted flag. `RigidWorld::contact_pairs`
  groups samples by stable collider handles. Sum impulses over the outer step;
  do not sum instantaneous forces from successive substeps.
- Average/Min/Multiply/Max friction rules, 32-bit two-way memberships/filters,
  collider enabled flags and sensor flags are runtime-editable. Legacy builders
  retain Max as the default; new runtime collider descriptors use Average so
  a tyre's explicit Min rule can take priority. Sensors enter world readout
  without force; other collision pipelines exclude sensor overlaps unless
  explicitly configured to include them.
- Voxel contact reduction preserves distinct collider pairs and remaps its
  body-contact CSR after pruning. This keeps material identity and force
  attribution intact when multiple colliders share the same bodies.

## Requirement ledger

| Plan section | Status / remaining evidence |
| --- | --- |
| 0: solver and force architecture | Retained Featherstone for rigid runtime scenes; no XPBD replacement. Penalty contacts enter the working copy of `state.body_f` before ABA without accumulating in persistent applied forces. Tyre/soil force elements remain. |
| 1.1: penalty contacts | CPU spring/compression-damper and bounded smoothed friction implemented; world collision is refreshed every substep. Tractor-scale stability remains unverified. |
| 1.2: live mutation | Foundation implemented and tested; re-rooting and complete mutable descriptors remain. |
| 1.3: contact force readout | Implemented for CPU Featherstone, including all substep samples and stable pair grouping. Gearbox consumption remains pending. |
| 1.4: fixed bodies | Root transitions tested; non-root freezing/re-rooting and complete GPU Featherstone support remain. |
| 2: bodies | Body reads, subtree teleports, reduced-coordinate velocity edits, no-step collider queries, per-body damping, persistent additional mass, explicit recomputation, force/reset and articulated impulse APIs implemented. Enabled/quarantine, sleep/wake, CCD option, non-root fixed transitions and Gearbox accessor wiring remain. Continuous-force singular-inertia parity needs verification. |
| 3: colliders | All requested shape kinds have CPU authoring/query paths, including indexed-mesh convex decomposition with one stable owner for multiple pieces. Views, pose access, AABBs, materials, friction rules, two-way masks, enabled/sensor flags and density/total-mass editing implemented. Explicit body mass recomputation implemented. Gearbox consumption still requires integration tests. |
| 4: joints | Stable-ID insert/remove and directed-tree merge/split implemented for revolute/prismatic/fixed/ball/free/D6. Explicit locked-axis API, limits/motors/force caps, acceleration-based gains, mutable frames, softness, collision enable, enabled flag and full body-pair queries remain. |
| 5: world | Gravity access, CPU world step with configurable substeps, per-step body-pair callback and handle-aware CPU raycasts implemented. Solver-setting mapping and Gearbox wiring remain. Existing `molla-sensors::Raycaster` stays GPU-only; CPU queries live in geometry. |
| 6.1-6.3: tyres | Per-substep transient force-element hook implemented. Wheel force elements, ground masking, scene terrain adapter with per-cell friction and wheel output readback remain. |
| 6.4 / build order 7 | Sleeping/islands, deformable terrain and two-way MPM coupling remain; existing separate modules are not evidence of runtime integration. |
| 7: stability/performance | No tractor-scale acceptance evidence yet. Must measure 26-body/high-mass-ratio tractor and 2-10 t trailers at 120 Hz, re-detect contacts per substep, deterministic replay, <1 ms CPU step. |
| 8.5: Gearbox implementation | Pending `MollaBackend`, backend selection arm and `GEARBOX_PHYSICS=molla`. |
| 9: 43 binary tests on both backends | Pending; includes joint motors (constraint/reduced), heightfield ray and steering limits. |
| 9: tractor driving | Pending spawn, forward 2/turn 0.4, heading comparison and visible tracks. |
| 9: hitch/tow/detach | Pending rear_drawbar teleport attach, turn, detach, finite state and no quarantine. |
| 9: Kubota | Pending all five controllers, including hitch lift and PTO motion. |
| 9: slope | Pending 10 degree parking test lasting 60 s with no creep. |

## Verification and workspace notes

- Initial worktree had only untracked `FULL_COMPACT.md`; the specification is
  left unmodified and untracked.
- Actual backend trait is in
  `/home/bresilla/data/code/robolibs/gearbox-physics/bin/gearbox/src/physics/backend.rs`
  on `feat/physics-backend`, still clean at `8358c2a` at this checkpoint. The
  sibling `gearbox` worktree is changing externally: it is now at `f637c64`
  with dirty `controller.rs`, `globe.rs`, and `physics/writeback.rs` under
  `bin/gearbox/src`. Earlier terrain edits are not its current baseline.
  Do not switch or overwrite that tree; re-inspect before integration work.
- The Makefile refers to a missing `PROJECT`. Commands use
  `PROJECT_NAME=molla PROJECT_VERSION=0` overrides; no environment or Makefile
  files were changed. The Makefile still prints its missing-file sed warnings.
- Passed `make test PROJECT_NAME=molla PROJECT_VERSION=0 APP_PKG=molla-sim`:
  16 unit tests, 6 runtime integration tests, 1 doctest.
- Passed CPU solver unit tests (36), Featherstone articulation (4), polish (4),
  existing origin-invariance (1), and new runtime-scene tests (5) through
  `make test ... APP_TARGET='-p molla-solvers --lib --test runtime_scene
  --test featherstone_articulation_cpu --test featherstone_polish_cpu
  --test featherstone_origin_invariance'`.
- New GPU fixed-flag regression runs through
  `make test-gpu PROJECT_NAME=molla PROJECT_VERSION=0 P=molla-solvers
  T=runtime_scene A=--nocapture`: passed, exercising semi-implicit and XPBD
  on a live GPU. The contact CSR must be initialized by `CollisionPipeline`
  before GPU XPBD dispatch, including scenes with zero contacts.
- Strict scoped Clippy is run through a command-line Make target using
  `cargo clippy -p molla-sim -p molla-solvers --all-targets
  --no-default-features -- -D warnings`: passed. `git diff --check` also passed.

### Contact integration verification

- `make test ... APP_TARGET='-p molla-geometry --lib --test contact_materials'`:
  113 unit tests and 5 new integration tests passed (all coefficient priorities,
  masks using bit 31, callback-before-narrow-phase filtering, sensor opt-in and
  collider identity/CSR preservation during reduction).
- `make test ... APP_TARGET='-p molla-solvers --test featherstone_contacts
  --test runtime_scene'`: 10 contact tests and 5 runtime tests passed. Covers
  analytic normal force/COM torque, equal/opposite reactions, friction bounds,
  finite zero-slip behavior, disabled-joint velocity, transient impacts,
  no force accumulation, material/sensor/mask edits, replay determinism and
  rejected timesteps. A 1 kg sphere settles with 9.81 N support at 120 Hz;
  this is not evidence of the tractor's mass-ratio or slope acceptance.
- Existing solver unit (36), articulation (4), polish (4), origin-invariance
  (1), and sim unit/integration/doctest (23) lanes also passed after contact
  integration. The final solver rerun passed all 60 selected tests after the
  input-validation and disabled-joint velocity changes.
- `make test-gpu ... P=molla-geometry T=contact_materials A=--nocapture`:
  one live GPU GJK/EPA test passed, retaining Min friction and callback filtering
  with zero CPU narrow-phase fallbacks. The GPU fixture uses offset cuboids;
  a sphere-pair fixture exercised the existing EPA-to-CPU fallback instead.
- Strict Clippy for sim, geometry and solvers with all targets and no default
  features passed. An existing unconditional GPU-only test import in
  `narrow_phase_backend.rs` needed its missing feature guard. Scoped Clippy with
  all targets and all features also passed. No workspace-wide or Gearbox
  acceptance claim follows from these scoped gates.

### Shape/query integration verification

- Eight CPU tests in `molla-geometry/tests/runtime_shapes.rs` cover endpoint
  and collapsed capsules, body/local/canonical frame composition, COM offsets,
  local/world/arbitrary-pose bounds, invalid edits/queries, convex true edges,
  rounded-cylinder support, symmetric plane contacts and rounded-corner depth.
  Terrain bounds now have an explicit nonzero vertical-origin regression.
- The existing CPU MPR algorithm returned 1.30 m rather than the analytic
  0.05858 m depth for the rounded-corner fixture when experimentally enabled.
  Round cylinders therefore remain ineligible for MPR, using CPU GJK/EPA for
  every pipeline selection. The regression verifies both default and MPR
  selection produce the correct fallback result. General MPR repair is not
  claimed by this slice.
- The live `make test-gpu ... P=molla-geometry T=runtime_shapes` regression
  passed: a rounded-cylinder contact retains both collider IDs and records
  exactly one CPU fallback in a full GPU narrow-phase pipeline.
- CPU regression lanes passed: sim 23 tests/doctests; geometry 113 unit,
  5 material and 8 shape tests; solvers 36 unit, 24 selected Featherstone/runtime
  and 11 heightfield/SDF tests. Total: 220 CPU tests/doctests, plus the live GPU
  fallback test. This does not establish tractor/terrain acceptance.
- Workspace `make check-all` and strict scoped Clippy passed. No external
  dependencies, Makefile/environment changes or Gearbox edits were needed.

### Scaled terrain / CPU ray verification

- Twelve `cpu_scene_queries` tests cover rectangular row-major grids and Y
  scaling, validation/atomic rejection, nonplanar cell diagonals, hull-normal
  penetration, rotated terrain contact culling, metric and transformed rays,
  inside/zero/max-distance hits, tangencies and tangent intervals, enabled
  sensor/filter behavior, removals, pose edits without stepping, convex hulls
  and mesh slab boundaries. Two hundred oblique terrain rays match an
  independently traversed equivalent triangle-mesh BVH.
- A large-grid test fires 512 vertical rays against 131,841 height samples;
  one debug run took 467.432 microseconds for the ray loop. This is a bounded
  query-path observation, not a release benchmark or tractor-step acceptance.
- Regression lanes passed: sim 23 tests/doctests; geometry 113 unit, 5 material,
  8 previous shape and 12 new terrain/query tests; solvers 71 selected tests.
  Total: 232 CPU tests/doctests. The rounded-cylinder GPU-fallback regression,
  workspace `make check-all` and strict scoped Clippy also passed.
- That terrain/ray slice added no dependencies or Gearbox checkout edits.
  Persistent mesh/hull query preparation was subsequently implemented below.

### Convex decomposition / persistent query verification

- Six `convex_decomposition` tests cover the concavity of a closed indexed
  L-shaped mesh (including a cavity contact miss, ray miss/hit and correct
  bounds), deterministic replay, unused input vertices, translated small
  meshes, authoring rejection, multiple dense shapes under one handle, atomic
  mutation and per-piece material/filter propagation.
- Cache preparation counts verify reuse over repeated rays, body motion,
  collider pose edits and body/joint topology edits. Replacements, removals
  and different scenes with equal numeric revisions cannot reuse stale data;
  filtered/bounds-missed geometry is not prepared.
- Four `compound_world` tests verify actual Featherstone force readout groups
  both convex pieces into one collider pair, sensor/disabled changes cover
  every piece, replacing a same-revision world clears contact readout, and
  concurrent shared-world rays preserve `Send + Sync` and correct hits.
  A panicking user query predicate does not permanently poison later queries.
- Regression lanes include sim 23 tests/doctests, geometry 144 tests and
  solvers 75 selected tests (242 CPU tests/doctests total). Tractor-scale,
  mass-ratio and full Gearbox acceptance are still unproven.

### Collider mass / body recomputation verification

- Seven sim `runtime_mass` tests cover parallel-axis combination, rotated full
  tensors, scale-relative principal decomposition, singular inverse tensors,
  fixed mass retention, invalid/overflowing edit atomicity, COM velocity/wrench
  rebasing, and free/articulated FK consistency after COM changes.
- Seven geometry `collider_mass` tests cover analytic primitive inertias,
  endpoint/collapsed capsules, rounded-cylinder reference semantics, compound
  mass assignment, translated small hulls, closed indexed meshes with unused
  remote vertices, sensor/disabled filtering, repeated recomputation and
  zero-volume terrain. The world `runtime_mass` test verifies changed totals
  actually change Featherstone's force/acceleration response on successive
  steps, and mass-only changes preserve the geometry-cache identity.
- Selected regression lanes passed: sim 30 tests/doctests, geometry 151 tests,
  solvers 76 tests, totaling 257 CPU tests/doctests. Workspace `make check-all`
  and strict scoped Clippy with all features and with no default features pass.
  One initial solver command used shortened/nonexistent target names; the
  corrected repo-native target names were then run successfully.
- No dependency, environment, Makefile, Gearbox or specification edits. Engram
  tools remain unavailable in this session; this tracked checkpoint records
  the work rather than pretending a persistent-memory tool call succeeded.
- Relevant new files: `crates/molla-sim/src/mass.rs`,
  `crates/molla-geometry/src/mass_properties.rs`, sim/solvers
  `tests/runtime_mass.rs`, and geometry `tests/collider_mass.rs`.

### Body mutation / impulse / force-element verification

- Twelve `body_mutation` tests cover full rotated tensors and translated roots,
  offset-COM impulses/forces, fixed-link total mass, anchored revolute impulses,
  rotated singular inertia with offset COM, persistent additional-mass replacement,
  fresh per-substep force callbacks, nonaccumulation, damping, callback failure,
  invalid-edit atomicity, subtree teleport/ray queries, generalized-rate velocity
  edits, and prescribed fixed/kinematic motion.
- The singular-inertia oracle initially failed when COM was offset: a bare mass
  pseudoinverse mixed the zero-inertia direction into permitted angular response.
  Projecting into the permitted angular subspace before solving fixes the oracle,
  including a later linear impulse with no unintended angular velocity.
- Regression lanes passed: sim 30 tests/doctests, geometry 151 tests, and solvers
  102 tests (283 total). The solver selection now includes CRBA, RNEA, multi-DOF
  oracle and Franka IK/Jacobian regressions. Workspace `make check-all`, strict
  scoped Clippy with all features and no default features, and diff checks pass.
- New files: `crates/molla-solvers/src/rigid_impulse.rs` and
  `crates/molla-solvers/tests/body_mutation.rs`. Runtime body descriptors, geometry
  mass helpers, CRBA/Jacobian helpers and the Featherstone/world force hooks were
  updated. No Gearbox, dependency, Makefile, environment or specification edits.
- The original objective remains incomplete. No tractor-scale performance,
  driving, hitch, Kubota-controller, slope, or full Gearbox acceptance is claimed.

## Next implementation steps

1. Finish fixed-body/re-rooting and runtime descriptor semantics; retain the
   now-tested contact/readout/filtering path while extending the live world.
2. Implement body enabled/quarantine, sleep/wake and CCD surface; verify
   continuous-force singular-inertia dynamics and benchmark queries with tyres.
3. Finish joint editing, locking, motors and softness, including capture frames.
4. Implement the Gearbox backend in its existing physics worktree.
5. Connect tyre/terrain outputs, then sleeping/deformation/MPM as specified.
6. Run and record every original acceptance scene, performance and deterministic
   replay gate. Do not mark the objective complete from library tests alone.
