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
- Undirected joint forests are compiled in deterministic parent-first order,
  choosing an immovable/world-anchored root when present. Attaching at either
  endpoint can merge articulations and reverse the required branch; detaching
  seeds newly free roots from current pose and COM velocity. Authored endpoint,
  frame, coordinate, rate, limit and motor signs remain unchanged. Ordinary tree
  joint insertion rejects cycles; explicit `insert_loop_joint` descriptors are
  now supported separately as CPU compliant force elements, without merging
  or projecting their endpoint trees. This is not reduced-coordinate loop closure.
- Fixed and kinematic root kinds retain physical mass/inertia while exposing
  zero inverse mass/inertia. Dynamic roots get free joints automatically;
  immovable roots get fixed world anchors. Dynamic/fixed transitions keep the
  current pose; fixing zeroes velocity. Pinning a former non-root re-roots the
  articulation instead of freezing or discarding its neighboring bodies.
- Body enable/disable is now separate from body kind. Disabled bodies lower
  to immovable, noncolliding placeholders while retaining physical mass, pose,
  pending velocity and authored collider flags. Incident joints are suspended,
  not deleted; healthy neighboring bodies become independent roots and continue
  simulating. Re-enable restores the authored connections with coordinates
  reseeded from current kinematics. Pose edits traverse active connections only.
- `RigidWorld` quarantines invalid input before collision detection, including
  invalid poses, non-finite body velocities/wrenches, bad generalized rates/
  controls and degenerate joint quaternions. The invalid bodies are disabled,
  bad wrenches/velocities are zeroed, and bad poses use the latest committed
  valid pose rather than teleporting back to spawn. Healthy bodies still step.
  `quarantined_bodies` exposes handles isolated before the latest valid-dt step
  attempt. Invalid dt does not quarantine; quarantine precedes the subsequent
  atomic time advance, so a later solver/callback failure does not undo isolation.
- `RigidScene::commit_step` validates rigid state dimensions/finite values and
  retains last-valid poses without changing scene identity. Manual body edits
  additionally change the scene stamp. Disabled-body CCD requests are stored
  alongside ordinary requests; CCD remains an explicitly permitted no-op.
- Runtime `JointSettings` separates enabled and connected-body collision flags
  from topology. Joints default to enabled with connected contacts disabled,
  matching Gearbox's descriptor. Collision exclusions cover every shape piece
  and remain authored even while the joint is disabled, matching Rapier's
  narrow-phase behavior. Removing the joint removes its exclusions.
- Joint local frames are readable/writable while active or suspended. Dormant
  joint/free-root motor controls survive body toggles, joint toggles and unrelated
  rebuilds; non-finite control values are sanitized during quarantine. Logical
  joint handles remain readable while `joint_index` returns `None` for suspended
  joints. Authored cycles and invalid axes/frames remain rejected even when the
  affected bodies/joints are disabled. Multiple authored incoming edges are
  accepted when their undirected graph is a tree.
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

## Initial integration requirement ledger

This table records the initial adapter checkpoint, not current test counts or
acceptance results. Later checkpoints below supersede its pending flags where
they provide explicit evidence. In particular, real imported Kubota driving,
pressure sweeps and CPU replay now have coverage; the full slope, towing,
lifecycle, fleet and under-1-ms acceptance set is still incomplete.

| Plan section | Status / remaining evidence |
| --- | --- |
| 0: solver and force architecture | Retained Featherstone for rigid runtime scenes; no XPBD replacement. Penalty contacts enter the working copy of `state.body_f` before ABA without accumulating in persistent applied forces. Tyre/soil force elements remain. |
| 1.1: penalty contacts | CPU spring/compression-damper and bounded smoothed friction implemented; world collision is refreshed every substep. Tractor-scale stability remains unverified. |
| 1.2: live mutation | Stable mutation, branch merge/split and re-rooting implemented and tested. Combined collider-plus-mass transactions need hardening at the adapter boundary. |
| 1.3: contact force readout | CPU Featherstone samples and stable pair grouping feed Gearbox contact manifolds; adapter impulse/material/filter tests pass. Tractor traction/stamping acceptance remains. |
| 1.4: fixed bodies | Root and non-root fixed transitions/re-rooting tested, including Gearbox accessors. Complete GPU Featherstone support remains. |
| 2: bodies | Reads, teleports, velocity edits, damping, mass, force/reset, impulses, enable/quarantine, CCD storage and Gearbox accessors implemented. Sleep/wake still no-op. Continuous-force singular-inertia parity needs verification. |
| 3: colliders | Requested shape kinds, queries, materials, masks and runtime edits implemented. Gearbox accessor/query/mass tests pass; compound edit failure atomicity and live scene coverage remain. |
| 4: joints | Stable insert/remove, arbitrary-tree merge/split/re-rooting, frames, enabled state, collision flags, locks, limits, motors/caps, acceleration gains, pair queries and CPU softness implemented. Gearbox hitch-capture loop tested. Soft angular coordinates use the D6 chart; multi-turn soft revolute position and GPU parity remain. |
| 5: world | Gravity, configurable CPU substeps, quarantine, pair filtering and CPU raycasts wired through Gearbox. Internal iterations map to substeps; direct ABA has no outer iterative-solver equivalent. CPU queries remain in geometry. |
| 6.1-6.3: tyres | Articulated CPU wheel forces, ground masking, static terrain sampling and per-cell friction API implemented. Gearbox scene registration, signed motors, traction and stamping consume tyre readout. Real field-profile friction authoring and live vehicle acceptance remain. |
| 6.4 / build order 7 | Sleeping/islands, deformable terrain and two-way MPM coupling remain; existing separate modules are not evidence of runtime integration. |
| 7: stability/performance | No tractor-scale acceptance evidence yet. Must measure 26-body/high-mass-ratio tractor and 2-10 t trailers at 120 Hz, re-detect contacts per substep, deterministic replay, <1 ms CPU step. |
| 8.5: Gearbox implementation | Experimental `MollaBackend`, compliant hitches and articulated tyres wired in `gearbox-physics`; runnable native build passes. Sleeping, failure-path hardening and live acceptance remain. |
| 9: 43 binary tests on both backends | Original 43 plus fourteen new checks pass (57 tests) with each explicit backend environment. Reduced-joint expectation is backend-specific; all original motor target/cap assertions remain. |
| 9: tractor driving | Pending spawn, forward 2/turn 0.4, heading comparison and visible tracks. |
| 9: hitch/tow/detach | Pending rear_drawbar teleport attach, turn, detach, finite state and no quarantine. |
| 9: Kubota | Pending all five controllers, including hitch lift and PTO motion. |
| 9: slope | Pending 10 degree parking test lasting 60 s with no creep. |

## Verification and workspace notes

- Initial worktree had only untracked `FULL_COMPACT.md`; the specification is
  left unmodified and untracked.
- Actual backend trait is in
  `/home/bresilla/data/code/robolibs/gearbox-physics/bin/gearbox/src/physics/backend.rs`
  on `feat/physics-backend`; integration began from clean `8358c2a`. The
  sibling `gearbox` worktree changes externally: an earlier snapshot was `f637c64`
  with dirty `controller.rs`, `globe.rs`, and `physics/writeback.rs` under
  `bin/gearbox/src`. Earlier terrain edits are not its current baseline.
  Do not switch or overwrite that tree; re-inspect before touching it.
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

### Enable / quarantine / joint lifecycle verification

- Thirteen `body_enable` tests verify disabled mass/motion retention, collision
  and ray exclusion without changing authored flags, middle-body graph splitting,
  healthy-neighbor motion, dormant control preservation, invalid-input quarantine
  and readout, corrupt motor/zero-quaternion handling, inactive-graph validation,
  disabled pose/velocity/CCD edits, explicit joint toggles, all-collider contact
  flags, active/suspended frame edits, last-committed-pose recovery and atomic
  state/dt validation. Tests exercise actual Featherstone steps and CPU queries.
- Selected regression lanes passed: sim 30 tests/doctests, geometry 151 tests,
  and solvers 115 tests (296 total). Workspace `make check-all` and strict scoped
  Clippy with all features and no default features pass. These remain library
  gates, not Gearbox/live-tractor acceptance.
- Main files: `crates/molla-sim/src/runtime.rs` (enabled state, dormant controls,
  last-valid poses, joint settings/frames and transactional commits),
  `crates/molla-geometry/src/raycast_cpu.rs` (effective collider enablement),
  `crates/molla-solvers/src/rigid_world.rs` (quarantine ordering/readout), and new
  `crates/molla-solvers/tests/body_enable.rs`.
- No Gearbox, dependency, environment, Makefile or specification edits. Engram
  tools remain unavailable; progress is recorded here. Full objective remains
  active, including all original named integration and acceptance gates.

### Axis limits / locks / motor verification

- Added canonical `joint_control` types and runtime axis-addressed edits.
  Revolute/prismatic freedoms are addressed as AngX/LinX irrespective of the
  authored axis. D6 lock masks, per-axis limits, authored motors and logical
  between-body queries survive suspensions and transactional rebuilds. Motor
  updates write live controls without rebuilding or invalidating scene stamps.
- CPU Featherstone caps motor effort separately from feed-forward joint force
  and external body wrenches. Acceleration gains use the reciprocal diagonal
  of the inverse articulation mass matrix, including floating-parent mobility.
  This currently performs a CRBA/factorization for each driven articulation
  each substep; tractor-scale performance has not been measured.
- Ball/D6 position drives now read their actual angular/linear coordinates.
  Ball limits and per-axis speed caps are enforced; D6 and Ball share XYZ
  angular projection. Equality-locked scalar/Ball/D6 freedoms contribute no
  motion-subspace column or published body twist. Multi-axis angular behavior
  retains the existing XYZ-coordinate convention, not a new swing/twist model.
- Twelve new `joint_control` tests exercise real CPU motion: semantic axes,
  separate force caps, 1000:1 inertia scaling, floating-parent reduced mass,
  invalid-edit atomicity, dormant authoring/raw controls, live locks/releases,
  scalar equality limits, and nonzero Ball/D6 position targets and saturation.
  Selected gates pass: sim 30, geometry 151, solvers 127 tests (308 total).
  Workspace check-all and scoped strict Clippy in both feature configurations
  pass. No GPU execution or Gearbox acceptance is claimed: acceleration models
  and motor-only caps explicitly reject GPU Featherstone rather than silently
  running unsupported semantics.
- New files: `crates/molla-sim/src/joint_control.rs` and
  `crates/molla-solvers/tests/joint_control.rs`. Runtime descriptions, controls,
  velocity evaluation and CPU Featherstone were extended. No specification,
  dependency, environment, Makefile or Gearbox edits. Engram tools remain
  unavailable; this checkpoint records the work. The full goal is still active.

### Tree re-rooting / fixed transitions verification

- Extracted runtime forest compilation into `runtime/tree.rs`. Components now
  choose a deterministic root, prioritizing prescribed immovable/world-anchored
  bodies. Joining two existing children merges trees without a second rigid
  solver or a fake detached constraint. Closed loops and multiple independent
  anchors still fail atomically; those require a loop solver outside this spec.
- Added a joint traversal-direction flag. Reversed traversal swaps compiled
  endpoints/frames, inverts FK and negates the motion subspace, but does not
  negate or rewrite authored joint coordinates, rates, limits, drives or queries.
  This handles Ball/D6 maps as well as scalar joints; simply swapping frames
  and negating a scalar axis would not preserve multi-axis motor semantics.
- `Model::joint_motion_frame` supplies the shared authored basis/moving pivot
  to velocity reconstruction, Featherstone, CRBA, Jacobians and RNEA. RNEA now
  includes the translated Free/D6 pivot and its velocity-bias term. Teleports
  follow compiled solver descendants, including reversed branches.
- Ten new `runtime_tree` tests cover child-to-fixed/dynamic transitions,
  non-root world pins, branch merges/detaches, signed prismatic limits/impulses,
  reversed Ball acceleration motors, freely moving reversed D6 bodies,
  kinematic-root teleports, frame edits, enable cycles and atomic loop rejection.
  The multi-axis oracle checks all Jacobian columns against pose finite
  differences, full mass matrices against kinetic energy, inverse-dynamics bias
  against differentiated physical COM/angular velocities with rotated full
  inertia/offset COM, and resulting Featherstone generalized-rate response.
- Selected regression gates pass: sim 30, geometry 151, solvers 137 tests
  (318 total), workspace `make check-all`, strict scoped Clippy with all
  features and no default features, and diff checks. These are CPU/library
  gates, not the original full Gearbox/tractor acceptance requirements.
- GPU Featherstone explicitly rejects reversed joints until its parity work
  is implemented. No new GPU, Gearbox, tractor/performance, or live acceptance
  claim is made. No dependencies, environment, Makefile, specification or
  Gearbox files changed. Engram tools remain unavailable; this ledger records
  the design and verification checkpoint.

### Gearbox backend integration checkpoint

- Gearbox source checkpoint: unsigned commit `6fd40ee`
  (`feat(physics): add molla backend adapter`) on `feat/physics-backend`.
- Working in the existing clean `feat/physics-backend` worktree at
  `/home/bresilla/data/code/robolibs/gearbox-physics`, not the independently
  changing `/home/bresilla/data/code/robolibs/gearbox` checkout. Added
  `physics/molla/{mod,body,collider,joint,convert,tests}.rs`, trait accessors,
  backend-name selection and local CPU-only Molla dependencies.
- Adapter accessors share an `Arc<Mutex<RigidWorld>>`; they do not copy live
  physics state into stale trait views. Stable handles map directly to Gearbox
  IDs, mass-affecting collider operations recompute parent totals, and queries,
  impulses, motors, contacts and per-step pair exclusions call Molla's runtime.
  All joints truthfully report reduced coordinates, regardless of Rapier's
  preferred constraint/reduced insertion hint. The existing constraint-motor
  test now checks this backend-specific truth while keeping its motor assertions.
- Integration discovered a Cargo feature-unification incompatibility: Molla's
  deterministic Parry collections require Rapier's `enhanced-determinism`
  feature too, or Rapier compiles the wrong HashSet `drain` calls. Enabled the
  matching Rapier feature; no dependency versions were updated. Host Rust
  1.94 is too old for this Gearbox checkout; use its Nix Rust 1.96 environment.
- `nix develop --impure -c oslo make check` compiles the binary but the all-target
  gate fails in unchanged `bin/gearbox/tests/oxbo_transforms.rs`, which names
  removed `openusd::Stage`/`usd_schema` APIs. Scoped binary tests passed with
  explicit `GEARBOX_PHYSICS=rapier` and `GEARBOX_PHYSICS=molla`: 51/51 each,
  including the original 43 plus seven adapter checks and one selection check.
  No live-scene acceptance is claimed. Strict Clippy including workspace
  dependencies is blocked by pre-existing warnings in `gearbox-api`,
  `gearbox-fields` and vendored clouds; those files were not changed.
  The `--no-deps` binary Clippy lane also fails on seven existing issues outside
  the Molla adapter. No lint suppressions were added and no green Clippy claim
  is made for Gearbox.
- Internal generalized-state quarantine now reaches Gearbox's entity report:
  a default backend readout exposes internally isolated body IDs, Molla maps
  its stable handles to those IDs, and `PhysicsWorld` collects their entities
  after the backend step. An actual-corruption test proves this path rather
  than leaving the UI/reporting layer falsely unaware of disabled bodies.
- This remains explicitly experimental. Softness requests warn and retain rigid
  constraints; sleep/wake are no-ops with never-sleeping readout; tyre/terrain
  integration is still absent. Collider replacement plus recomputation needs a
  combined transaction before final failure-path acceptance. Gearbox's
  `docs/MOLLA_BACKEND.md` records these gaps rather than hiding them.
- No environment, Makefile, Oslo recipes, vendor or specification edits. Engram
  tools remain unavailable. Original full-goal and live acceptance scope remain
  unchanged.

### CPU joint-compliance force foundation

- Added validated `JointSoftness` (natural frequency in Hz, damping ratio),
  per-DOF `JointCompliance` storage and an f32 device mirror. Every authored
  DOF gets a disabled record. `ModelBuilder::set_joint_compliance` replaces
  D6 hard limits with spring intervals; setting hard limits removes those
  springs. Invalid authoring is rejected before any bounds are changed.
- CPU Featherstone evaluates spring forces into transient body wrenches before
  ABA, sharing the acceleration-motor CRBA effective-inertia calculation:
  `1 / (M^-1)[i,i]`. Equal/opposite forces and torques use the shared moving
  joint pivot, including offset COMs, rotated frames and reversed traversal.
  Persistent caller wrenches do not accumulate the spring contribution.
- Equality intervals act as springs, bounded intervals act only outside their
  bounds and cannot pull toward an escaped side, and unbounded axes stay free.
  Zero natural frequency releases the constraint without damping. The force
  uses scalar backward-Euler regularization, not a second rigid-body solver.
  This regularization has timestep-dependent static compliance: for an isolated
  critically damped axis, the stiffness is divided by `(1 + omega*dt)^2`.
  It is not a fully coupled implicit spring solve. Angular displacement uses
  the existing D6 XYZ chart; multi-turn revolute phase remains a runtime concern.
- Eight integration tests and one wrench-level oracle cover analytic one-step
  response, 8/30 Hz capture settings, isolated 1000 Hz stability, static load,
  zero frequency, unilateral bounds, invalid edits, input preservation and
  1000:1 floating pairs in both traversal directions. The offset/rotated-frame
  oracle verifies net world force/torque and generalized virtual work.
- Selected CPU regressions pass: sim 30, geometry 151, solvers 146 (327 total).
  New integration tests also pass without default features. Workspace
  `make check-all`, strict scoped Clippy with all/no-default features and diff
  checks pass. Gearbox binary tests were rerun in its Nix environment with
  explicit Molla and Rapier selection: 51/51 each. Its existing all-target
  and lint limitations above remain unchanged; no live acceptance was run.
- This is the low-level force foundation only. `RigidScene` does not yet
  author compliance, and Gearbox's softness setter still warns and retains
  rigid behavior. GPU Featherstone rejects enabled compliance explicitly.
  No Gearbox, dependency, environment, Makefile or specification files changed.
  Engram tools remain unavailable; this checkpoint preserves the work.
- Next: lower soft runtime joints to D6 while retaining logical kind/handles;
  align scalar axes with both local frames, migrate semantic motor slots across
  hard/soft layouts, and reseed soft coordinates from live body kinematics.
  Capture-frame edits must change the spring rest frame without teleporting
  bodies. Only then wire Gearbox's 8-to-30 Hz hitch capture and test its motion.

### Runtime softness and Gearbox capture integration

- `RigidScene::joint_softness` / `set_joint_softness` now author validated
  optional compliance. Enabling or removing softness rebuilds the layout;
  frequency/damping changes on an already-soft joint update its live records
  without changing scene identity. Suspended joints retain their settings.
- Soft Fixed/Ball/Revolute/Prismatic/D6 joints lower to full D6 freedoms.
  Implicit and explicit locks become spring equality intervals, while authored
  limits become unilateral spring intervals. Scalar joints align both compiled
  local frames to their authored axis before any tree-traversal reversal.
  Logical descriptions, endpoints, frames, axes and handles remain unchanged.
- Soft coordinates/rates are reseeded from current body poses and COM
  velocities on every structural edit. Changing a capture frame changes the
  spring rest frame without teleporting bodies. Switching back to hard joints
  reseeds their natural coordinates; normal hard enforcement resumes on step.
- Drive snapshots now carry semantic-axis layouts. Authored motors and raw
  controls migrate between scalar/Ball and D6 slots through softness toggles,
  suspension and re-rooting; raw root-body drives retain their prior behavior.
- Ten runtime tests cover all logical kinds, arbitrary and reversed scalar
  axes, pose/velocity preservation, capture motion, live gain changes, dormant
  controls, authored motor migration, spring limits/locks, invalid edit
  atomicity, suspension and cascading body removal.
- Gearbox commit `1120a23` (`feat(physics): enable molla hitch compliance`)
  connects `JointMut::set_softness` to the runtime and removes the metadata-only
  warning/no-op. Two new binary tests prove compliant motion after frame edits
  and exercise the real `PhysicsWorld::capture_hitch` fixed-step capture loop,
  including its 8-to-30 Hz gain ramp and finite, enabled body state.
- Gates pass: sim 30, geometry 151, solvers 156 selected CPU tests (337 total),
  workspace `make check-all`, strict scoped all-feature/no-default Clippy,
  diff checks, and Gearbox's 53 binary tests with each explicit backend.
  Existing Gearbox all-target/lint limitations are unchanged. These synthetic
  capture tests do not prove the requested live rear_drawbar tow/turn/detach,
  tractor drive/tracks, Kubota, slope, determinism or performance acceptance.
- Compliance retains the documented scalar-regularization timestep dependence
  and D6 XYZ angular chart. Multi-turn soft revolute position targets and GPU
  support remain unfinished. No dependencies, environment, recipes, vendors
  or specification edits. Engram tools remain unavailable; this checkpoint
  records the design, verification and remaining scope.

### Articulated tyre-force and scene-terrain integration

- Added `molla_solvers::wheel_forces::WheelForces`, owned by `RigidWorld`.
  Wheel descriptors reference stable wheel, knuckle/heading and spin-joint
  handles, local hub/heading, radius, stiffness/damping, authored spin sign
  and an `Arc<dyn TireModel>`. Every CPU substep builds `WheelContact` from
  the current solver state and calls `wheel_ground_wrench`. Ground force,
  contact-point moment and aligning moment enter the wheel's COM wrench.
  Drive torque remains the revolute motor; Featherstone integrates both
  chassis and wheel spin. The separate `Vehicle::step` is not used here.
- Registered static, nonsensor heightfields/trimeshes supply top-surface
  terrain samples through the existing cached CPU raycaster. Collider-local
  rectangular `FrictionGrid` cells can override the collider coefficient.
  Missing/out-of-footprint or disabled ground supplies no tyre force. Dynamic
  terrain registration is rejected explicitly; deformable/MPM coupling remains.
  Ground bounds and dense wheel-ground exclusions are cached by scene stamp
  and rebuilt after configuration/topology changes, not rescanned per substep.
- Supplemental shape-level pipeline exclusions suppress only registered
  wheel shapes against registered ground. Obstacle contacts, including other
  parentless colliders, remain active. Removing a wheel registration restores
  ordinary ground contact. The per-step body-pair callback also filters tyre
  terrain queries. No authored collision groups or scene descriptions are
  rewritten to reserve a global collision bit.
- `WheelSample` publishes per-wheel/ground `TireOutput`, normal, friction and
  substep duration after a successful whole step. Forces integrate as
  `force*dt`; failed steps publish no partial readout. Body removal prunes
  stale registrations, scene edits invalidate readout, and invalid wheel/grid
  authoring or non-finite tyre output cannot commit a partial physics step.
- Corrected `wheel_ground_wrench`: it now uses the Terrain signed-distance
  contract for slope penetration, `friction_at` for local coefficients and
  surface-relative hub velocity. Previously it ignored those existing APIs,
  so terrain friction maps and moving surfaces could not affect this path.
- Eight runtime tests prove whole-articulation normal response, actual motor
  traction, analytic reaction torque reducing wheel spin, per-cell friction,
  pair filtering, obstacle-preserving ground masks, mesh terrain, cache/mask
  rebuilding after dense shape reorder, lifecycle/readout and atomic failure.
  Two helper regressions cover slope distance and moving/friction-map terrain.
- Gates: 345 selected sim/geometry/solver CPU tests pass, plus 190 tyre/real-tyre
  and two vehicle unit tests (537 selected tests total). Workspace
  `make check-all`, scoped all-feature/no-default Clippy and diff checks pass.
  Gearbox still passes 53 binary tests with each explicit backend; commit
  `e544b3d` records only the new local `molla-vehicle` lockfile dependency.
  No external dependency versions changed.
- This does not yet register Gearbox's scene wheels/terrain or replace its
  traction and stamping consumers. No tractor drive/tracks, slope, real towing,
  GPU tyre-force, determinism or <1 ms acceptance claim is made. The public
  Terrain query remains a top-surface/2.5D model, not arbitrary under-bridge
  contact. No USD/import logic, viewer code, environment, recipes, vendor or
  specification files changed. Engram tools remain unavailable.

### Gearbox articulated tyre consumers and scene registration

- Gearbox source checkpoint: unsigned commit `a0ac5d1`
  (`feat(physics): connect articulated tyre forces`).
- Added backend-neutral wheel descriptors, friction grids and last-step tyre
  outputs. Molla registers `WheelForces`; Rapier retains its existing rigid
  contact path. Readout integrates substep normal/friction impulses over the
  actual completed step duration, with impulse-weighted point, normal and slip.
  Changing future solver dt does not rescale an already completed readout.
- Gearbox's controller preparation now resolves wheel bodies, revolute spin
  joints, collider hub/radius and supported mass. Registration retries until
  the scene is ready and refreshes when geometry, joint or mass inputs change.
  Rolling direction follows the bearing/knuckle, not the spinning wheel.
  Authored axle direction determines both the tyre spin sign and controller
  motor-target sign. Removal clears the adapter's wheel metadata.
- Meadow heightfields, loaded USD terrain meshes and the flat slab register
  as tyre ground. Core registration now accepts explicitly selected static
  cuboids as well as heightfields/trimeshes, using the same top-surface query.
  Safety floors and obstacles are not automatically promoted to tyre terrain.
- Traction budgets use mean tyre normal/friction loads. Registered wheel
  stamps use tyre contact state/point rather than approximate terrain height;
  link values expose tyre slip ratio, slip angle and normal force. Unregistered
  wheels and Rapier retain the old contact/stamping path.
- Initial tyre tuning uses nominal per-wheel supported mass, 4% radial
  deflection at nominal load, damping ratio 0.7 and load-scaled smooth slip
  gains. This is not calibrated tractor acceptance. The current Gearbox
  field-profile source exposes visual ground materials, not a physics friction
  grid; scene builders currently retain scalar material friction. The new
  backend grid API is tested, but actual per-field/cell authoring remains.
- Four new Gearbox checks cover load/grid readout without penalty manifolds,
  opposite axle signs and forward motion, failed registration/removal, and a
  headless ECS scene executing registration, traction and elevated tyre-point
  stamping. Disabling terrain clears its stamp. With Molla selected, the ECS
  test proves the tyre consumer path; Rapier checks its unchanged fallback.
- Gearbox passes 57/57 binary tests with each explicit backend. Native
  `nix develop --impure -c oslo make build` now also passes, producing the
  runnable `target/debug/gearbox`; no GUI/live scene was launched this turn.
  Its scoped strict Clippy still reports the same seven pre-existing errors
  outside the new wheel code. Existing all-target test limitations remain.
- Molla's eight articulated-wheel tests, workspace `make check-all`, scoped
  all-feature/no-default solver Clippy and diff checks pass. Direct local
  `molla-vehicle` wiring changes no external dependency versions. No environment,
  recipes, vendors or specification changes. Engram tools remain unavailable.
- Next acceptance work should run the real binary and named tractor/hitch
  scenes, not infer drive/steer/tracks or <1 ms performance from headless tests.
  Sleeping, deformation/MPM, soft angular phase, GPU parity and failure-path
  hardening also remain within the unchanged full objective.

### Live Kubota discovery and compliant loop closures

- Native Gearbox launch works through `nix develop --impure -c oslo make run
  '--args=--name molla-loop-test --ephemeral'`. Oslo requires the equals form
  for this string parameter. Direct unwrapped launch failed GPU surface
  creation; no environment files were changed. `gearbox launch --help` starts
  the simulator with a scene path rather than displaying CLI help.
- The first actual Kubota load panicked in the adapter: its rear lift rods,
  stabilizers and front/rear cylinders close articulation loops. The asset
  marks those edges with `physics:excludeFromArticulation`; rejecting every
  cycle was insufficient for the named acceptance asset.
- `RigidScene::insert_loop_joint` now stores a validated compliant joint with
  stable handles, frames, limits, motors and collision exclusions, but omits
  it from the reduced tree. Insertion does not project or teleport either
  endpoint. Disable/remove/cascade and authoring survive tree rebuilds.
  Ordinary unmarked tree cycles remain rejected; loop softness is mandatory.
- CPU `RigidWorld` evaluates these force elements each fresh substep. Anchor
  translation and principal rotation-log rows have equal/opposite COM
  wrenches; rotating-frame translation and SO(3) derivative terms preserve
  virtual work and net world momentum. The coupled `J M^-1 J^T` response uses
  the existing articulated CRBA/Jacobian path, including across separate trees.
  A backward-Euler spring/damper solve produces impulses converted to body
  forces; a bounded active-set solve handles unilateral limits and motor caps.
  Featherstone remains the sole rigid integrator. Cholesky is reused across
  row right-hand sides; singular mass matrices use the existing pseudoinverse.
- Gearbox propagates the authored closure flag separately from `reduced` and
  inserts force joints with initial 30 Hz / critical damping when unspecified.
  Rapier's existing constraint path is unchanged. Five core regressions cover
  driven four-bar closure, a 1000:1 pair, motor caps/world anchors, analytic
  row derivatives/momentum and coupled bound complementarity. An adapter
  regression covers closure joint access, driving and independent removal.
- Live Kubota now loads: 26 bodies, 4917 kg, four tyres and all five controllers.
  The actual `--forward 2 --turn 0.4 --for 4s` command drives and steers;
  captured heading changed from about 0 to +0.365 rad, not the specified
  Rapier reference +0.60 rad. Wheel stamping logs and rendered tractor
  screenshots exist in `/tmp/molla-kubota-*.png`; this is not full parity.
  Current debug-build timing is about 5 ms/physics step, not the <1 ms target.
  Paused telemetry showed zero tyre loads, requiring further lifecycle/order
  investigation. Wire machine coordinates are Z-up (x=sim_z, y=sim_x,
  z=sim_y), so wire y changes are not vertical sinking.
- Live controller commands and TF sampling also show both PTO bodies rotating
  at commanded engagement, and both hitch bodies changing pose. Raising then
  lowering changed rear rockshaft orientation by approximately 0.966 rad and
  front lower-hitch orientation by 0.395 rad; chassis heading stayed near
  0.368 rad. These are actual TF poses, not just echoed command properties.
  No step rejection, panic or quarantine appears in the captured drive log.
  Exact PTO speed/load and hitch-load acceptance still require measurement.
- Gates pass: all 30 sim tests, all 354 solver tests (one explicit long MPM
  stress test ignored), workspace `make check-all`, strict sim/solver Clippy
  with all features and without default features, and Gearbox's 58 binary
  tests under each selected backend. Logs: `/tmp/molla-loop-*.log` and
  `/tmp/gearbox-loop-{tests,rapier,drive}.log`. Existing Gearbox all-target
  and broader Clippy limitations remain unchanged.
- Remaining limitations include principal-log angular branch/multi-turn
  semantics, finite loop compliance, omitted free-acceleration/bias prediction
  in the implicit force solve, GPU loop parity, actual hitch/PTO/towing/slope
  acceptance and performance. No external dependencies, asset, environment,
  recipes, vendors or original specification were changed. Engram tools are
  still unavailable; the full objective remains active.

### Idempotent material sync and live tyre readout

- Traced the zero-load telemetry to Gearbox's `apply_physics_materials`, which
  reapplies friction/restitution every Update. Molla replaced the whole
  collider even for identical values, invalidating the scene stamp before
  traction and wheel stamping consumed the previous step's samples.
- Molla adapter setters now skip unchanged friction, restitution and friction
  combine rules. A regression proves repeated material sync retains scene
  identity and tyre load, while a real material change invalidates samples and
  the next step restores them. No weakening of stale-readout checks was needed.
- A rebuilt native viewer confirms nonzero Kubota wheel telemetry: approximately
  15.3, 11.1, 13.4 and 8.6 kN after a turn, totaling 48.3 kN versus 48.2 kN
  static weight. The 4 s forward-2/turn-0.4 command reaches roughly 1.6 m/s and
  +0.64 rad in its final streamed sample. Later paused readout is +0.86 rad;
  CLI/network delay means these samples are not an exact fixed-step heading
  parity measurement. Logs and JSON: `/tmp/gearbox-material-live.log`,
  `/tmp/molla-material-{before,after}.json`.
- Gearbox binary tests pass 59/59 with both explicit backends. A real Krampe
  trailer load reports 15 bodies, 6729 kg and five tyres. The exact paused
  rear_drawbar teleport attach succeeds, but play then launches the rig and
  stepping rejects a non-finite mass system. This is failed towing acceptance,
  not a completed tow. Relevant paths: `attach.rs` teleports bodies one-by-one,
  including already-transformed descendants under Molla; resume synchronization
  writes stale ECS transforms back after a paused backend teleport. Both need
  regression-driven investigation. The failed scene is paused, with the
  original successful Kubota viewer in its separate instance unchanged.

### Straight-line tractor drive stability

- Scope: fix the reported straight-drive failure in Molla and validate through
  the isolated `gearbox-physics` worktree. No original Gearbox checkout edits,
  merges, branch changes or pushes. The broader FULL_COMPACT goal stays paused.
- Reproduced with the real Kubota at forward 2, turn 0: only about 0.35 m/s
  after five wall-clock seconds, with front/rear slip opposing each other.
  A 4917 kg/four-wheel fixture with 130 kg wheels, 15.6 kg m2 spin inertia,
  mixed radii and Meadow friction 1.4 reproduced the cancellation. Friction
  0.5 did not reproduce it. Tyre limiting alone and motor stabilization alone
  both failed the high-grip regression; the combined change passes.
- `FeatherstoneSolver::with_stable_motor_feedback` applies per-DOF
  effective-inertia backward-Euler feedback, followed by the authored motor
  effort cap. `RigidWorld` enables it. Low-level solvers retain their explicit
  default; GPU use of the new option returns an error instead of silently
  changing semantics. This is a diagonal motor approximation, not a coupled
  implicit motor/contact solve including external-acceleration prediction.
- `wheel_forces.rs` bounds each tangential tyre wrench by its current-state
  maximum-dissipation impulse using `w^T J M^-1 J^T w`. Mass factorizations are
  cached only within a substep. Normal support is unchanged and Featherstone
  remains the only rigid/spin integrator. This is a per-wheel bound, not a
  global passivity proof for simultaneous wheels, motors and terrain forces.
- Rebuilt live Kubota reaches approximately 2.01 m/s with zero steering and
  negligible heading change. Reverse reaches about 2.02 m/s backwards; a zero
  command stops it. Turning still reaches about 0.4 rad/s during the turn.
  Viewer was relaunched as `molla-straight-fixed`; inspected screenshot:
  `/tmp/molla-straight-fixed-final.png`. CLI timing and user interaction are
  not deterministic fixed-step acceptance evidence.
- CPU regressions include straight drive in both directions at 4/8/16
  substeps, analytic slip stopping without reversal or energy gain, stiff
  force/acceleration motors, preserved momentum, and explicit-mode retention.
  Broad solver run: 358 passed, one existing ignored MPM stress gate. After
  final option/guard changes, solver unit/control/compliance/loop/wheel lanes
  pass (39/14/8/3/11 tests). Workspace check and strict all-feature/no-default
  solver Clippy pass. Isolated Gearbox binary tests pass 59/59 with either
  backend selection. Logs: `/tmp/molla-straight-*.log` and
  `/tmp/gearbox-straight-*.log`; live JSON `/tmp/molla-straight-fixed-*.json`.
- GPU stabilization parity, global coupled contact stability, towing and
  performance remain unverified/incomplete; this fixes the reported drive
  regression, not the whole integration plan.

### Pressure-dependent articulated tyre foundation

- The user expanded `FULL_COMPACT.md` with required pressure-dependent tyre
  integration, controls, physical/visual deformation, traction, resistance,
  persistence and acceptance. The full goal is active again. The specification
  remains an untracked user file; it has not been staged with implementation.
- `pressure_tire.rs` connects the existing `real_tire::evaluate_brush_tire`
  contact-cell model to the actual Featherstone wheel force path. Pressure is
  explicit gauge Pa. Geometry and current load determine deflection, loaded
  radius, cell support and tread area; longitudinal/lateral response and
  available grip come from the existing pressure/load-sensitive brush model.
  The old normal spring is replaced, not added to the pressure contact force.
- Contact evaluation uses a local planar frame, including on slopes. Slip
  uses loaded radius; damping scales the supported contact load and its
  friction budget. Rolling losses are a separate resisting axle moment with
  surface and configured carcass-hysteresis contributions, not a tangential
  ground force which could incorrectly spin the wheel. The existing
  articulated tangential impulse bound also applies to that moment.
- `WheelForces::configure_pressure_tire` enables the path per wheel;
  `set_pressures` atomically validates grouped targets. Applied pressure ramps
  by an authored Pa/s rate during simulation; setting a target while paused
  does not advance the pressure clock. `pressure_tire` reads both values.
  `WheelStep` stages pressure and telemetry alongside the candidate rigid
  state; failed substeps commit neither. Re-registering an unchanged wheel
  or making unrelated scene edits preserves pressure/targets. Removed wheel
  bodies/joints prune state; incompatible geometry/bounds edits are rejected.
- `WheelSample::pressure` exposes pressure, deflection, loaded radius, tread
  area, patch extents, normal stiffness, grip budget, cell loads/forces and
  the applied rolling moment. Tangential cell forces and the rolling moment
  are adjusted together with the applied wrench's numerical impulse bound.
- Verification: 42 solver unit, 15 wheel, 14 joint-control, eight compliance
  and three loop tests pass. New cases cover matched-load pressure sweeps,
  slope-frame covariance, loaded-radius zero slip, resisting-moment sign,
  exact single normal support, atomic invalid edits, bounded pressure ramps,
  unrelated rebuilds/removal, injected fourth-substep rollback, and driving
  both directions while inflating. Workspace check and strict solver Clippy
  pass with all features and without defaults. Isolated Gearbox's existing
  binary suite remains 59/59. Logs: `/tmp/molla-pressure-*.log` and
  `/tmp/gearbox-pressure-core-tests.log`.
- This is opt-in core integration, NOT completed live Gearbox tyre acceptance.
  No Gearbox source has changed for this checkpoint. Remaining: authored
  width/pressure/presets, backend controls/readout, UI/CLI and saved state,
  visible deformation and stamping, per-cell scene terrain/material queries,
  camber and nonplanar footprint integration, real-machine calibration,
  pressure-extreme slope/towing tests, timing and GPU parity. Current tread
  area is the sum of sampled curved-cell areas, not projected footprint area;
  the compact brush is not a full FEM carcass. Do not call either calibrated
  production tyre realism from these internal-consistency tests.

### Paused pressure edits preserve contact telemetry

- Target-only pressure edits no longer invalidate the previous successful
  wheel samples. While paused, actual pressure, normal support and deflection
  remain unchanged; `pressure_tire()` immediately exposes the accepted new
  target. Historical samples retain the target used by their own substep.
- Verification: all 16 wheel integration tests pass, including the new paused
  telemetry regression; workspace all-target check and strict solver Clippy
  with all features and without defaults pass. Logs:
  `/tmp/molla-pressure-paused-final.log`, `/tmp/molla-pressure-check-final.log`,
  `/tmp/molla-pressure-lint-final.log`.

### Isolated Gearbox pressure integration

- All Gearbox changes remain in `robolibs/gearbox-physics`, branch
  `feat/physics-backend`. The original `robolibs/gearbox` checkout was not
  edited, merged, switched or pushed. No assets, Mara dependency source or
  environment/recipe files were changed.
- Discovered wheels now pass measured width/radius and optional authored
  `gearbox:value:tyre_*` properties into the pressure model. Reference values
  are explicitly uncalibrated: 1.8 gauge bar, 0.5–4.0 bar range and 0.2 bar/s
  transition rate. They are not real-machine inflation recommendations.
- Backend grouped target edits, per-wheel CLI controls and actual/target
  readback are wired. The Machine pane's separate Tyres tab has all-wheel and
  individual sliders. Rejected requests republish accepted targets. Drive and
  traction calculations use loaded radius and pressure brush grip; field
  stamping consumes physical patch length/width with legacy fallback.
- Two UI integration traps were found: collapsed nested controller sections
  omit response entries, so pressure controls need their own tab response
  namespace; Mara's typed state map is separate from egui's direct persisted
  store, so synchronizing the latter does not update displayed sliders.
  New regression tests cover response routing and the actual Mara memory API.
- Verification: isolated Gearbox binary suites pass 62/62 with Molla and
  Rapier, fields library passes 8/8, and native build passes. Logs:
  `/tmp/gearbox-pressure-controls-{molla,rapier}-tests.log`,
  `/tmp/gearbox-pressure-fields-final.log` and
  `/tmp/gearbox-pressure-controls-build.log`.
- Gearbox's strict no-dependency Clippy lane still fails on eight pre-existing
  findings outside the new tyre code; no lint suppressions or unrelated fixes
  were added. See `/tmp/gearbox-pressure-controls-clippy.log`.
- Live pressure tyres drive the Kubota straight at approximately 2.01 m/s
  for a 2 m/s zero-turn command. Settled fronts at 1.0 bar deflect about
  67 mm versus 53 mm at 1.8 bar; rears at 2.4 bar deflect about 47 mm. Total
  support is about 48.2 kN. Earlier live JSON is in
  `/tmp/molla-pressure-{before-edit,paused-edit,after-edit}.json`.
- A captured UI all-wheel edit changed all accepted targets to 2.58427 bar
  while paused actual pressure stayed 1.8 bar and contact deflection stayed
  unchanged: `/tmp/molla-pressure-controls-all.{json,png}`. The subsequent
  per-wheel mixed view exposed stale UI memory; the typed-memory fix is
  unit-tested, but final live CLI-to-slider verification was interrupted when
  the owned test instance disappeared. Do not report that last check as passed.
- Still incomplete: per-axle UI grouping, save/reload state, visible tyre
  deformation, visual pressure-sweep track comparison, per-cell field friction,
  nonplanar/camber response, calibrated presets and reference data, pressure
  extreme slope/towing checks, deterministic replay, timing and GPU parity.
  Original towing/resume failure and <1 ms acceptance remain unresolved.
- Relevant isolated files: `physics/backend.rs`, `physics/molla/{mod,wheel,tests}.rs`,
  `controller.rs`, `controller/wheel_forces.rs`, `services.rs`,
  `host/panes/{machine,mod}.rs`, `gearbox-fields/src/{contacts,render}.rs` and
  `docs/MOLLA_BACKEND.md`. The full objective remains active.

### Atomic articulated teleports and paused publication

- `RigidScene::set_body_poses` validates complete batches before committing.
  Explicit world-pose targets are applied together; unlisted descendants
  follow their parent's transform. Joint freedoms still project targets.
  Generalized rates and revolute winding persist unless velocity reset is
  requested, which clears only affected subtrees. Duplicate/missing handles
  and invalid poses leave state, revision and scene stamp unchanged.
- New `runtime_pose_batch.rs` tests cover both input orders, descendant motion,
  fixed roots, winding/rates, unrelated moving bodies and atomic rejection.
  The runtime, mass and batch suites pass 17 tests total. Seven solver suites
  pass 66 tests, including pressure wheels, runtime trees, softness, joint
  control/compliance and loops. Workspace all-target check passes.
- Independent `molla-sim --all-features` linting exposed a pre-existing GPU
  feature dependency omission: `molla-storage/gpu` used `Error::Gpu` without
  enabling `molla-core/gpu`. The storage feature now propagates that dependency.
  Strict simulation Clippy passes both all-feature and no-default lanes.
- Evidence: `/tmp/molla-pose-batch-{regression,solvers,check,lint}.log`.

### Live towing: teleport corruption and fictitious axle tyre

- Isolated Gearbox attachment/terrain-lift teleports now use the atomic core
  batch API. Paused backend changes publish through writeback. Resume compares
  current transforms with the last publication, applying only actual external
  edits and preserving untouched velocities. A two-backend ECS regression
  covers paused publication, ancestor gizmo edits, same-frame teleport/resume
  and unrelated moving bodies.
- The repaired batch-only build attached, resumed and detached the real
  Kubota/Krampe without rejected steps, but towing stalled at about 0.06 m/s.
  Live readout showed a fictitious front-axle tyre carrying 42 kN and all four
  actual trailer wheels unloaded. Authored wheel links were already known,
  but the fallback larger-collider heuristic added the axle too. Wheel pairs
  with an already known wheel endpoint now skip that inference.
- The final build's eight-second 2 m/s, 0.4 rad/s command reached about
  2.05 m/s and 0.44 rad/s, with the trailer following the turn and all four
  real trailer tyres supporting load. Screenshot/JSON:
  `/tmp/molla-tow-wheels-driven*`; log:
  `/tmp/gearbox-tow-wheels-live.log`. No rejected-step/quarantine messages
  appear. Debug two-machine timing is approximately 7.55 ms/step, not <1 ms.
- The final-build detach recheck timed out on the control plane; authoritative
  instance listing later showed no running instances. It is not passed.
  Earlier batch-only detach evidence remains in `/tmp/molla-tow-batch-detached*`.
  Rapier live parity, repeated detach, pressure extremes and replay remain.
- Both isolated backend binary suites pass 64/64, including the axle-selection
  regression; native build passes. Logs:
  `/tmp/gearbox-tow-batch-{molla,rapier}-tests.log` and
  `/tmp/gearbox-tow-batch-build.log`. Only the isolated Gearbox checkout changed.
- The user explicitly wants tyre pressure directly in the sidebar and liked
  those controls. Next UI change: move pressure pods into the Machine sidebar
  before collapsible controller sections, retaining stable response indices.

### Sidebar placement and missing visible deformation

- At the user's explicit request, pressure controls now appear directly in
  the isolated Machine sidebar before variants/controllers, not a separate
  tab. Top-level pod offsets stay independent of collapsed controller blocks.
  Screenshot inspected: `/tmp/molla-sidebar-pressure.png`. Binary tests pass
  64/64 and native build passes; `/tmp/gearbox-sidebar-pressure-{tests,build}.log`.
- The user then reported that tyres do not visibly change with pressure.
  This is a real missing integration: the physical brush supports pressure-
  dependent deflection, but the imported rendered meshes remain rigid. The
  subsequent sidebar interaction recheck could not run because its owned
  instance had disappeared; do not report it as passed.
- Next work is visible contact flattening/sidewall deformation driven by the
  same loaded radius and deflection, without scaling entire rigid wheel/rim
  assemblies or changing shared imported meshes across instances. Kubota
  separates `radial_tyre`, `swept_traction_tread`, `mould_line` meshes from
  hubs and rims, allowing targeted rubber geometry updates. Assets remain
  unchanged; implement in Molla's reusable tyre layer and isolated viewer.

### Visible pressure tyre integration (2026-09-19)

#### Goal and instructions

- Address the user's report that the tyres remain visually rigid when pressure
  changes. Original `robolibs/gearbox`, asset files and Mara remain untouched;
  viewer changes are confined to `gearbox-physics`.

#### Discoveries

- Imported rubber is separable from rigid hubs/rims. Material-subset parents
  can contain empty, main-world-only meshes: uploading those as render meshes
  causes repeated Bevy slab allocation errors. The deformation path now skips
  empty/non-renderable geometry and retains its subset children.
- Authored `Mesh::final_aabb` overrides calculated bounds; private deforming
  meshes clear it and update their bounds before visibility checks.
- This is a CPU compact contact-envelope approximation, not solved FEM carcass
  displacement or a calibrated sidewall model. Lower-sidewall bulge derives
  from actual loaded-radius compression, not an independent pressure animation.

#### Accomplished

- `molla-vehicle::real_tire::LoadedTireEnvelope` maps rubber onto the loaded
  plane and bulges the lower sidewall while preserving the bead and upper tyre.
  Molla wheel descriptors now expose read-only geometry; the isolated backend
  publishes current hub, heading, unloaded radius and tread width.
- The isolated viewer clones each rubber mesh, always deforms reference
  positions, preserves rigid rims/hubs and shared source assets, recalculates
  normals/tangents and bounds, restores unloaded geometry, and releases its
  records when bodies/meshes disappear. Paused external body transforms carry
  the contact frame with them instead of deforming against stale world poses.
- Molla envelope-filtered tests: 10 pass; wheel-force regressions: 16 pass;
  strict library Clippy passes. Isolated binary suites: 67 pass per backend;
  native build passes. Logs: `/tmp/molla-tyre-envelope-{tests,wheel-tests,lint}.log`
  and `/tmp/gearbox-tyre-mesh-{tests,rapier-tests,build}.log`.
- Live Kubota low/high pressure screenshots inspected at the same close camera:
  `/tmp/molla-tyre-checked-low-settled.png` and
  `/tmp/molla-tyre-checked-high.png`. Matching JSON shows 0.5 versus 4 bar applied;
  front compression changes from 83.7 to 35.2 mm, rear 79.0 to 36.5 mm.
  The sidebar's applied values/targets reflect CLI edits at both endpoints.
- The corrected live run no longer repeatedly reports slab allocation errors;
  one startup allocation warning remains. Debug frame rate is about 6 FPS,
  physics about 5.5 ms/step, so performance acceptance remains open. The owned
  live instance later disappeared; the separate `probe` instance was not touched.

#### Next steps and relevant files

- General authored rubber tagging, calibrated carcass deformation, nonplanar/
  camber agreement, GPU deformation and CPU/GPU pressure parity remain open.
  Name-based selection covers the tested Kubota rubber and Krampe BKT naming;
  it is not universal imported-asset support. Krampe deformation needs live proof.
- `crates/molla-vehicle/src/real_tire/envelope.rs`: reusable geometric envelope.
- `crates/molla-solvers/src/wheel_forces.rs`: read-only wheel descriptor.
- Isolated `bin/gearbox/src/viewer/tyre_mesh.rs`: deformation/lifecycle tests.
- Isolated `bin/gearbox/src/physics/{backend.rs,molla/wheel.rs}`: geometry readout.
- Engram session-summary/save tools are unavailable in this tool set; this
  progress entry records the session findings and handoff instead.

### Axle-level pressure controls (2026-09-19)

#### Goal and instructions

- Continue §6.1 axle-level pressure adjustment through the sidebar and CLI.
  Only the isolated `gearbox-physics` worktree changed; the original Gearbox,
  assets and environment files were not edited.

#### Discoveries

- The supported scene reset currently replays CLI spawn manifests. There is
  no live machine/scene snapshot export carrying tyre state; editor menu
  persistence is unrelated. Save/reload pressure acceptance remains open.
- Live remote driving returned approximately zero speed during this check.
  The later screenshot shows `tractor · viewer` ownership; target values also
  changed outside the scripted values during the running check. Do not count
  this run as controlled pressure adjustment while driving. Next audit should
  check local `UiDrive`/MachinePanel ownership before diagnosing solver motion.

#### Accomplished

- Added optional authored numeric `gearbox:value:tyre_axle` ids (1–65535).
  Fallback discovery clusters wheel-body longitudinal positions within 0.25 m,
  front to rear. Cached membership prevents renumbering under steering/motion;
  metadata changes refresh grouping. Invalid/ambiguous metadata is rejected.
- Machine sidebar now includes axle sliders between all-wheel and individual
  controls. Mixed targets are read-only until edited; each group uses the
  intersection of supported member ranges and rejects an invalid edit whole.
- `machine tyre-pressure BAR --axle N|--wheel LINK --machine ID` sends one
  grouped request; omitting selectors targets all tyres. Shared API helpers
  validate telemetry, scope and every range. The CLI waits for accepted target
  telemetry and distinguishes queued requests from confirmed edits.
- Live Kubota: inferred front wheels are axle 1, rears axle 2. CLI accepted
  front 1.0 / rear 2.4 bar targets while all applied pressures remained 1.8
  paused. An all-wheel 4.1 bar request exited 2 and left all targets unchanged.
  A sidebar axle-1 click accepted 1.494318 bar on both fronts, leaving rears
  at 2.4. Screenshots inspected: `/tmp/molla-axle-{paused,ui-paused}.png`.
  Matching JSON, invalid-edit and running-attempt logs use `/tmp/molla-axle-*`.
  The owned `molla-axles` instance was stopped after testing; `gearbox` was left alone.
- Binary suites pass 68 tests per backend; shared API selector tests pass 2;
  full CLI fake-host suite passes 12. Native build passes. Logs:
  `/tmp/gearbox-axle-{tests,rapier-tests,api-tests,cli-full,build}.log`.

#### Next steps and relevant files

- Implement actual supported snapshot save/reload, including applied pressure
  and a distinct target during transitions, without editing source assets.
- Repeat pressure controls under unambiguous remote/local drive ownership and
  verify unrelated mutation, attach/detach and save/reload lifecycle gates.
- Isolated `controller/wheel_forces.rs`: axle discovery/cache and grouped edits.
- Isolated `host/panes/machine.rs`: axle sliders and atomic UI range checks.
- Isolated `crates/gearbox-api/src/tyres.rs`: shared scope/range/telemetry validation.
- Isolated `crates/gearbox-cli/src/cli/machine.rs`: confirmed grouped CLI edits.
- Isolated `specs/TYRE_PRESSURE.md`: controls, inferred/authored axle contract,
  simulation defaults and explicit persistence/model limitations.
- Engram save/session-summary tools are still unavailable; this entry retains
  the findings and session handoff. The full objective remains incomplete.

### Pressure snapshot work interrupted for visual feedback (2026-09-19)

- Uncommitted isolated Gearbox work adds `machine save` configuration export
  and `spawn from` restoration of applied/target tyre pressure and variants.
  Source assets and original Gearbox remain untouched. This is configuration
  persistence, not a solver-state checkpoint; live restoration is unverified.
- Snapshot validation is all-wheel atomic, restored loads remain paused, and
  reset/delete clear stale pressure values. Binary tests passed 69 and CLI
  snapshot tests passed 2 before the final cleanup patch. The final native
  build passed (`/tmp/gearbox-pressure-save-build.log`).
- Pending files: isolated `bin/gearbox/src/load.rs`, API `tyres.rs`, CLI
  `machine.rs` and `spawn.rs`. Finish failure-event handling, live save/reload,
  documentation and regression validation before committing that feature.
- User reports no visible tyre deformation. Prioritize checking the actual
  running instance and visible meshes over further persistence work. Initial
  process inspection found only original Gearbox running; the isolated tyre
  build was not open. Engram summary/context tools remain unavailable.

### Recheck reported invisible tyre deformation (2026-09-19)

#### Goal and instructions

- Investigate the user's repeated report that pressure does not visibly
  inflate/deflate machine tyres. Leave original Gearbox and source assets
  untouched; keep all viewer changes in the isolated physics worktree.

#### Discoveries

- Initially only the original Gearbox instance was running. The isolated
  tyre build had been stopped after the prior checks. Launched a separate
  `molla-tyres` instance; the original remained independently running.
- Applied pressure advances with simulation time, not slider movement or
  wall time. Paused target edits cannot animate deformation. Low debug FPS
  also slows the inflation ramp relative to wall time.
- Kubota rubber does deform in the isolated build: at 0.5 bar front/rear
  deflections were 83.7/79.0 mm; at 4.0 bar they were 35.2/36.5 mm.
  Inspected rendered low/high screenshots, including the changing lower
  sidewall/contact shape. This is not acceptance for arbitrary asset names.

#### Accomplished

- Isolated commit `2e518c8` identifies instance and actual physics backend
  in the machine sidebar; shows paused/adjusting/at-target status, applied
  group pressure, and per-wheel deflection in mm. No solver changes here.
- Native Molla binary tests pass 70; native build passes. Logs:
  `/tmp/gearbox-tyre-status-{tests,build}.log`.
- Evidence: `/tmp/molla-tyres-low-detail.png`,
  `/tmp/molla-tyres-high-detail.png` (mid-inflation),
  `/tmp/molla-tyres-high-settled.{png,json}` and
  `/tmp/molla-tyres-low.json`. The camera changed before the final high
  screenshot, so the final pair is not a strict same-camera image diff.
- Reopened only the owned isolated instance on the updated build with a
  Kubota and low-pressure target. Original Gearbox was not edited/stopped.

#### Next steps

- User's exact affected machine and paused/playing state remain unconfirmed;
  a clarification question was sent. Do not claim all machines fixed.
- Resume the uncommitted snapshot work described above independently.
  Full-goal physics, traction, GPU and performance gates remain open.

#### Relevant files

- Isolated `bin/gearbox/src/host/panes/machine.rs`: live pressure status,
  instance/backend identity, deflection readout and status regression test.
- Isolated `specs/TYRE_PRESSURE.md`: explicit isolated launch and clock semantics.
- Engram save/summary tools are unavailable; this entry preserves the handoff.

### Pressure configuration round-trip and rejection cleanup (2026-09-19)

#### Goal and instructions

- Continue §6.1 machine pressure persistence in isolated Gearbox. The previous
  visibility turn was progress. Original Gearbox and source assets are off limits.
- The user repeated that the tractor I spawned still shows no visible tyre
  change. Persistence work was set aside again for a bare-ground close-up.

#### Discoveries

- A pressure configuration can restore exact applied and target pressures
  while keeping the scene paused. Reusing an ephemeral machine id can wait
  for an old topic ownership lease; `spawn from` now has a separate 90-second
  default `--wait-timeout` instead of relying on the short RPC timeout.
- Rejecting an already projected invalid snapshot in Update raced queued
  physics marker insertion and caused an entity-despawn command panic. Root
  deletion now runs in Last, after earlier projection commands, and explicitly
  removes bodies/colliders. A regression test covers command ordering/cleanup.
- When the user repeated the visibility problem, both isolated instances
  were gone and only original Gearbox was running. No cause was established;
  do not infer that the user deliberately closed them. Test windows previously
  had the same generic title and no always-visible instance/backend badge.

#### Accomplished

- Added versioned TOML machine configuration save, no-clobber publication,
  source/variant/root placement references, distinct applied/target pressures,
  all-member source-bound validation, paused restoration, readback confirmation
  and rejection events. Delete/reset clear obsolete cached pressure values.
- Live disposable fixture `/tmp/gearbox-pressure-roundtrip.LGK4rq`: exported
  Kubota applied 1.8 with front target 1.0 / rear target 2.4, refused overwrite,
  unloaded/reloaded the same id, and compared all eight pressure fields exactly.
  Restored scene stayed paused. After the cleanup fix, invalid 4.1-bar restore
  rejected cleanly and the existing tractor's pressure fields were unchanged.
- Resumed and paused during transition, exporting front applied 1.46 and rear
  applied 2.14 bar with distinct targets. This second snapshot has NOT yet been
  round-tripped: the instance disappeared before the next command. Unrelated
  spawn/despawn and resumed-transition acceptance remain unverified this turn.
- Final snapshot suites: 71 Molla binary tests, 71 Rapier binary tests, 2 CLI
  serialization/publication tests, 14 full CLI fake-host tests; native build
  passes. Logs `/tmp/gearbox-snapshot-*.log`.

#### Next steps and relevant files

- Finish exact mid-transition reload/resume and unrelated topology checks;
  this is machine configuration persistence, not a full solver checkpoint.
- Isolated `load.rs`: atomic restore validation, Last-stage failed-load cleanup.
- Isolated API `tyres.rs`, CLI `spawn.rs`/`machine.rs`, CLI integration tests,
  and `specs/TYRE_PRESSURE.md`: pressure configuration format and command path.
- Isolated `host/mod.rs`: current visible-feedback work adds an always-visible
  instance/backend/clock badge and distinct window title. Bare-ground instance
  `molla-tyres` is being checked with 0.5-to-4 bar transitions.
- Engram save/session-summary tools remain unavailable; this records findings
  and the handoff without editing the persistent-memory folder.

### User-requested visual retest (2026-09-19)

- User explicitly confirmed that rubber still looks unchanged in the labelled
  Molla window. Treat visual acceptance as unresolved, despite pressure
  telemetry and earlier screenshot interpretations. Do not claim it fixed.
- User requested another launch and will judge the rubber change themselves.
  Reopening only `molla-tyres` on bare ground with the real Kubota and sidebar;
  no scripted pressure sweep this time. Original Gearbox stays untouched.
- Completed isolated commits: `9e0fe7c` pressure configuration persistence and
  `a14f41d` visible instance/backend/clock identity. Final binary tests pass 71.
- Earlier bare-ground captures `/tmp/molla-flat-{low,high}-settled.png` show
  about 0.55 vs 4 bar. The low target was changed during the check; do not
  label that image as exactly 0.5 bar. User visual feedback overrides any
  assumption that the effect is clear enough.
- Next: obtain the user's retest feedback before further visual changes;
  keep the full objective active. Engram tools remain unavailable, so this
  tracked handoff records the session outcome.

### Rubber/ground mismatch investigation (2026-09-19)

#### Goal and instructions

- User still saw unchanged rubber, then suggested inflation grows through
  the ground rather than lifting the tractor. Investigate rendered vertices
  and ground support directly, without touching original Gearbox/assets.

#### Discoveries

- Opt-in vertex tracing proved real rubber meshes are being modified, but
  the old envelope mostly clamped bottom vertices and bulged sideways. Lower
  sidewall vertices above the support plane had no vertical compression.
- Renderer used positive support force as the condition for deformation.
  Damping can make support force zero while the tyre still intersects terrain;
  restoring the undeformed mesh then loses the ground constraint. Imported
  tread lugs can also extend beyond the nominal collider radius. Before the
  first support sample, traced front tread vertices reached about -0.042 m
  against the flat y=0 ground. Settled positive-force samples were near y=0.
- Loaded-radius readout used the last pre-integration sample with a final
  integrated hub pose. Those two frames need a consistent ground reference.

#### Accomplished

- Molla `WheelSample` now retains the actual terrain ray-hit point separately
  from tyre force application position; backend pressure output keeps that
  ground plane even with zero normal force. Loaded radius is evaluated from
  the final hub pose and terrain plane.
- Render deformation uses this geometry rather than the support-force flag,
  constrains rubber/tread vertices above its plane, and includes ground
  distance in the update cache. Collider helper meshes are excluded.
- Reusable contact envelope now distributes vertical tread displacement
  smoothly through the lower sidewall, keeping bead and upper rubber fixed.
- Regression tests: inflation raises a vertically supported chassis without
  teleporting, deflation returns it, final support plane stays grounded;
  zero-force rubber/lugs remain above flat and tilted planes; lower-sidewall
  compression is continuous and ground-safe. Binary suite passes 73, Molla
  wheel integration 16, envelope tests 4; native build passes.

#### Next steps and relevant files

- Live updated Kubota validation is in progress in `molla-rubber`. Do not
  claim user-visible acceptance before checking the updated output/feedback.
- Molla `real_tire/envelope.rs`, solver `wheel_forces.rs`; isolated Gearbox
  `physics/backend.rs`, `physics/molla/{wheel,tests}.rs`, `viewer/tyre_mesh.rs`.
- Opt-in `GEARBOX_TYRE_MESH_TRACE=1` reports actual mesh vertex displacement,
  world-space bottom and sampled terrain height. Logs `/tmp/molla-rubber-*`.
- Engram save/session-summary tools remain unavailable; this tracked entry
  preserves the diagnosis, changes and remaining acceptance work.

### Live pressure response confirmed (2026-09-19)

#### Goal

- Check whether sidebar pressure edits reach the rubber, then distinguish
  slow inflation from a frozen shape.

#### Instructions

- User now confirms the visible pressure change works after waiting. Their
  next concern is whether the fully inflated tyre should appear rounder.
- Original Gearbox and source assets remain untouched.

#### Discoveries

- The default pressure rate is 0.2 bar per simulated second. A real sidebar
  click changed the target from 0.5265 to 3.9735 bar; applied pressure reached
  it in about 25 wall seconds in this debug run. Front deflection decreased
  from 82.5 to 35.3 mm, with loaded radius increasing from 0.5614 to 0.6086 m.
- Imported tread extends beyond nominal tyre geometry. Its additional ground
  clipping may contribute to the still-flattened high-pressure appearance;
  this is a hypothesis to investigate, not a calibrated-shape diagnosis.

#### Accomplished

- Verified actual UI edits and time-series pressure/deflection readback;
  inspected the low-pressure rendered image. User accepted live response.
- Molla vehicle library: 232 tests pass; workspace check-all passes.
- Isolated Gearbox binary suites: 73 tests pass with each backend selection.

#### Next Steps

- Investigate high-pressure roundness against mesh radius, load and carcass
  properties. Do not remove all load-dependent flattening or exaggerate mesh
  changes independently of physics. Inflation rate has not been changed.
- Ground/envelope changes remain uncommitted; full compatibility goal remains
  incomplete. Engram tools remain unavailable.

#### Relevant Files

- `crates/molla-vehicle/src/real_tire/envelope.rs` — rubber envelope.
- `crates/molla-solvers/src/wheel_forces.rs` — terrain support sample.
- Isolated Gearbox `bin/gearbox/src/viewer/tyre_mesh.rs` — mesh deformation.
- `/tmp/response-{low,high}.png`, `/tmp/molla-pressure-response.log` — live
  evidence; high-pressure screenshot saved but not inspected in this check.

### Reverted roundness experiment (2026-09-19)

#### Goal

- User requested rounder fully inflated rubber, then rejected the resulting
  version: "you fucked up completely". Exact visible failure is not yet known.

#### Instructions

- Undo only the latest roundness experiment; preserve the pressure response
  the user previously confirmed and the earlier ground-contact fixes.
- Do not touch original Gearbox. Ask what went wrong before another attempt.

#### Discoveries

- Inferring the physical support radius from undeformed rubber raised the
  Kubota front radius from about 0.644 to 0.691 m and rear to 0.888 m. This
  changed physical support geometry, not merely rubber appearance.
- Tests passed (74 per backend selection), but that did not establish visual
  acceptance. The user rejected the live experiment before visual validation
  was complete. Do not treat its radius inference as an accepted solution.

#### Accomplished

- Reversed only the experiment in isolated Gearbox controller/wheel_forces.rs
  and restored renderer-local mesh naming helpers. Earlier envelope and
  ground-plane changes remain intact. Original Gearbox is untouched.
- Saved the reversed patch at `/tmp/molla-roundness-rollback.patch` for audit.
- Native rollback build passed; no replacement window was launched.

#### Next Steps

- Wait for the user's description of the failure.
- High-pressure roundness remains unresolved. No new tuning or relaunch until
  the failure is understood. Engram tools remain unavailable.

#### Relevant Files

- Isolated Gearbox `bin/gearbox/src/controller/wheel_forces.rs` — back to its
  pre-experiment state; collider-derived physical radius restored.
- Isolated Gearbox `bin/gearbox/src/viewer/tyre_mesh.rs` — previous working
  deformation retained, shared experiment helpers removed.
- `/tmp/gearbox-round-rollback-build.log` — rollback build result.

### Restored-version motion check (2026-09-19)

#### Goal

- Investigate the user's report that it no longer moves. Asked whether this
  means tractor driving or rubber inflation; clarification remains pending.

#### Instructions

- Leave roundness tuning reverted. Test only the isolated restored build.

#### Discoveries

- Fresh restored Kubota responds to a remote straight-drive command. A 1 m/s
  target ramped through 0.10, 0.33, 0.59 and 0.87 m/s. Before/after state moved
  from x=0.0126 to x=0.7936 m. This does not verify keyboard/gamepad ownership
  or prove the cause of the user's reported failure.
- The debug window ran near 2.2 FPS during this check; simulation advancement
  is slower than wall time. Local keyboard ownership can override bus input
  even with no movement key held; do not conflate that with solver failure.

#### Accomplished

- Relaunched restored `molla-rubber`, spawned the Kubota, tested remote drive,
  sent a stop, and inspected its rendered screenshot. No new source edits.
- Original Gearbox instance remains untouched.

#### Next Steps

- Obtain precise user clarification, then reproduce the relevant control
  path. Keyboard/gamepad and pressure deformation still need this retest.
- Do not resume high-pressure roundness changes yet. Engram unavailable.

#### Relevant Files

- `/tmp/molla-restored-drive.log`, `/tmp/molla-restored-drive.png` — live run.
- `/tmp/restored-{before,after}-drive.json` — movement readback.
- Isolated Gearbox `viewer/drive/keyboard.rs`, `controller.rs` — read-only
  inspection of local control ownership; no changes made.

### Rubber-only clarification and response retest (2026-09-20)

#### Goal and instructions

- User clarified that "does not move" means rubber does not visibly change
  with pressure, not that the tractor cannot drive. Stop drive debugging.
- Keep physical support-radius experiment reverted; no roundness changes.

#### Discoveries and decision

- Restored code accepts pressure targets and moves actual rubber vertices.
  In the instrumented run, front hub fell from 0.5902 to 0.5864 m over about
  7 wall seconds; maximum rubber displacement increased from 58.5 to 62.3 mm.
- Window ran around 2.1 FPS. With twelve 120 Hz steps per frame, pressure
  advancement at the old 0.2 bar/s becomes about 0.04 bar per wall second.
  Full-range inflation can therefore take over a minute. This is observed
  behavior in the retest, not proof of the user's exact previous state.
- Mesh update timing: about 90 ms total, 80 ms deformation, 5 ms normals,
  effectively no tangent cost. Do not replace tangent generation speculatively.
- Increase isolated backend's uncalibrated default inflation rate to 1 bar
  per simulated second, retaining authored rate overrides and successful-step
  advancement. Do not alter radii, shape mapping, pressure limits or loads.

#### Accomplished

- Added full backend-to-private-mesh inflation/deflation regression, checking
  both directions, unchanged source asset, cache refresh and ground contact.
- Added opt-in mesh timing alongside existing vertex diagnostics.

#### Next Steps

- Finish tests/build and check low/high rendered states of the faster-rate
  version; user-visible acceptance is still pending. Engram unavailable.

#### User acceptance update

- User confirmed the faster-rate version works ("nice it works") and again
  requested a rounder high-pressure tyre with a smaller ground contact patch.
- Both backend-selected binary suites passed 74 tests, including the new
  pressure-to-private-mesh regression; native build passed. Preserve this
  pressure response while addressing high-pressure shape.

#### Relevant Files

- Isolated Gearbox `physics/backend.rs`, `physics/molla/tests.rs`,
  `viewer/tyre_mesh.rs`, `specs/TYRE_PRESSURE.md`.
- `/tmp/molla-rubber-timing.log`, `/tmp/rubber-retest-{low,high}.png`.

### Roundness with accepted pressure response (2026-09-20)

#### Goal and instructions

- Preserve the now user-accepted inflation response and make maximum-pressure
  tyres rounder with less rubber flattened on the ground.
- Accepted baseline committed locally: Molla `9051d20`, isolated Gearbox
  `356760d`. Original Gearbox remains untouched.

#### Decision and discoveries

- Revisit reference-radius alignment while retaining the accepted 1 bar/s
  pressure rate. The earlier rejected experiment preceded the user's
  clarification that slow rubber response, not driving, was the failure.
- Collider-only front radius is about 0.644 m; imported tread reaches about
  0.691 m. At high pressure the hub is around 0.609 m, so rendering must flatten
  about 82 mm of outer tread despite solver deflection being only 35 mm.
- Use undeformed rubber's axle-plane radial extent at first registration.
  Keep it cached through live deformation; exclude rigid/rim/helper meshes.
  No pressure-dependent mesh scaling or radius changes. Missing render meshes
  retain the collider fallback; universal/headless metadata remains a limit.

#### Accomplished and next steps

- Restored and documented the bounded geometry correction; retained faster
  inflation and the new backend-to-rendered-mesh bidirectional regression.
- Build/tests and low/high visual comparison are pending. Do not claim this
  shape accepted until visual inspection and user feedback. Engram unavailable.

#### Validation update

- Native build passed; both backend-selected binary suites passed 75 tests.
- Real sidebar transition from 0.5 to 3.9735 bar completed in about 21 wall
  seconds in the 2 FPS debug window. Front hub rose from 0.6087 to 0.6554 m;
  deflection fell from 82.6 to 35.8 mm. Faster rate remains 1 bar/s simulated.
- Inspected `/tmp/round-responsive-low.png` and `-high.png`: high-pressure
  outer tread is rounder, with less flattened lower tread than the previous
  collider-radius-only rendering. Compared with its prior 0.6086 m high-
  pressure hub, corrected support removes about 47 mm of excess tread crush.
- This is geometry consistency, not measured real-tyre calibration. Rubber
  still flattens under load. User acceptance of the revised roundness remains
  pending. Leave the isolated viewer open at maximum pressure for inspection.

#### Relevant Files

- Isolated Gearbox `controller/wheel_forces.rs`, `viewer/tyre_mesh.rs`,
  `specs/TYRE_PRESSURE.md`; `/tmp/gearbox-round-responsive-*` validation logs.

### Machine-wide tyre compatibility contract (2026-09-20)

#### Goal

- Check whether the machine spec requires tyre support for every compatible
  machine rather than relying on recognised model/mesh names.

#### Instructions

- User requires tyre support to be part of machine compatibility. Apply
  pneumatic requirements to all pneumatic wheels, including passive trailer
  and implement wheels; explicitly classify solid/track/no-wheel cases.
- Keep original Gearbox untouched; documentation changes stay isolated.

#### Discoveries

- `FULL_COMPACT.md` §6.1 already requires pressure-dependent mechanics and
  matching visuals, but allows documented defaults for missing metadata.
- Existing controller spec had wheel joints/colliders, while the tyre spec
  documented Molla controls and name-based recognition. Neither established
  explicit per-asset rubber bindings as a full-compatibility requirement.
- Current loader does not enforce explicit contact type, rubber/rigid mesh
  bindings or authored reference geometry; their schema still needs design
  and implementation. Runtime success is not universal compatibility proof.

#### Accomplished

- Added controller-spec §5.1 and a linked tyre asset contract covering wheel
  identity/joints, explicit contact classification and mesh bindings, geometry,
  pressures, material response, axle membership and validation/acceptance.
- Listed currently consumed numeric attributes separately from missing schema.
  Marked the requirement as REQUIRED CONTRACT with enforcement incomplete;
  retained MUST's existing meaning of currently enforced runtime checks.
- Documented legacy fallback/degraded reporting, backend capability reporting,
  and rename-independent discovery as required acceptance gates.
- Documentation-only change, checked with `git diff --check`; no runtime or
  asset edits and no builds/tests needed. Engram tools unavailable.

#### Next Steps

- Define/implement explicit schema bindings and validator; migrate assets,
  then verify each machine's pressure/mesh/ground-contact lifecycle. The spec
  update does not make existing assets or backends compliant automatically.

#### Relevant Files

- Isolated Gearbox `specs/CONTROLLER_SPEC.md` — §5.1 and requirement status.
- Isolated Gearbox `specs/TYRE_PRESSURE.md` — required asset contract, current
  fields, enforcement gaps and acceptance gates.
- Molla `FULL_COMPACT.md` — consulted only, remains untracked and unchanged.

### Performance parity work (2026-09-20, in progress)

#### Goal and instructions

- User requires Molla performance comparable to original Gearbox/Rapier;
  optimise actual frame latency/responsiveness, not only solver timings.
- Original checkout and its active instance must remain untouched. Benchmark
  a pinned copy of its existing executable in a separate named instance.

#### Initial discoveries

- Current Kubota scene runs around 2.1 FPS. Per-frame rubber processing costs
  roughly 85–95 ms (75–85 ms deformation); physics is about 1.4 ms per step.
  Those do not explain the entire roughly 450 ms frame, so capture a system
  trace before attributing all cost to the solver.
- Kernel perf sampling is blocked (`perf_event_paranoid=4`); do not change
  system permissions. Use the existing `oslo make build-profile` Chrome-trace
  lane instead. Engram tools remain unavailable.

#### Accomplished / next steps

- Pinned a read-only copy of the original Gearbox executable in the isolated
  worktree's `target/perf-artifacts/gearbox-original`; SHA256
  `f311de5b3e1e5bca81185ade3134cd67064592f78f0306c82550f5012d273959`.
  Original checkout and its independently running instance remain untouched.
- At 1280x800, flat ground and one Kubota, CPU rubber rendering measured about
  8 FPS versus about 40 FPS for the original executable. A Chrome system trace
  measured 88.64 ms/frame in tyre processing plus 19.96 ms in mesh allocation.
- Added a viewer-only WGSL contact-envelope path in isolated Gearbox. Meshes
  remain resident, with 320-byte per-mesh storage buffers carrying current and
  previous deformation frames. Persistent buffers avoid rebuilding full PBR
  materials on pressure/pose changes. CPU fallback remains available through
  `GEARBOX_TYRE_RENDER=cpu`; no physics rate or tyre feature was disabled.
- Per-frame ancestor lookup memoization avoids classifying unrelated meshes;
  wheel/link lookup now resolves the machine's body paths once per frame rather
  than repeatedly scanning every USD prim for each wheel/link combination.
- Intermediate GPU trace: tyre update 1.23 ms/frame versus 88.64 ms before;
  full PBR material recreation still cost 2.21 + 1.39 ms/frame. This motivated
  the persistent-buffer refinement. Its unchanged-view rolling FPS samples were
  41.59 and 42.48 versus about 40–44 for original Gearbox. Exclude the later
  69 FPS reading after camera/window interaction. This is a single-scene
  checkpoint, not a general parity result.
- 78 binary tests pass, including bounds, source mesh preservation, pressure
  changes in both directions, material modification/reassignment/restoration,
  stable buffer/material handles and previous-frame tracking. A Vulkan test
  executed 124,416 production WGSL deformation cases; maximum CPU/GPU position
  difference was 0.0000001562 m. Final targeted tests passed with both Molla and
  Rapier selection (78 each), the Vulkan test passed again, and native
  `oslo make build` passed. Low/high-pressure screenshots were inspected on the
  final storage-buffer path: front deflection 82.61 -> 35.70 mm and rear
  78.54 -> 37.17 mm for 0.5 -> 4 bar. Driving reached the requested 2 m/s.
  Close-up pressure/driving runs were approximately 35–38 FPS.
- Native `oslo make test` is blocked compiling existing
  `bin/gearbox/tests/oxbo_transforms.rs`: removed `openusd::Stage` and
  `usd_schema` APIs. Do not report the full workspace suite as passing.
- Isolated Gearbox commit `3aca1a4` (`perf(tyres): keep deformation on the GPU`)
  contains the implementation, regression tests and detailed reproduction/
  caveats in `specs/TYRE_PERFORMANCE.md`. It is unsigned and not pushed/merged.
  The `molla-fast` instance was left running for user inspection; latest native
  binary also includes the small CPU reference-position memory saving.
- Next: controlled multi-machine/terrain benchmarks and solver profiling.
  Molla physics remains approximately 1.5–1.6 ms/step versus 0.35–0.46 ms for
  original Rapier. Background user work and original-renderer changes prevent
  laboratory-clean comparisons. General performance parity remains open.

#### Relevant Files

- Isolated Gearbox `viewer/tyre_mesh.rs`, `viewer/tyre_gpu.rs`,
  `viewer/tyre_gpu.wgsl`, `viewer/tyre_deform.wgsl`, `viewer/tyre_gpu_test.rs`,
  `controller/wheel_forces.rs`, `specs/TYRE_PRESSURE.md`,
  `specs/TYRE_PERFORMANCE.md`.
- `/tmp/molla-before-trace.json`, `/tmp/molla-gpu-final-trace.json`,
  `/tmp/perf-rapier-matched.log`, `/tmp/perf-molla-buffers.log`,
  `/tmp/gearbox-gpu-parity.log`, `/tmp/tyre-buffer-tests.log`,
  `/tmp/gearbox-perf-workspace-tests.log`, `/tmp/molla-fast.log`,
  `/tmp/tyre-buffer-low.png`, `/tmp/tyre-buffer-high.png`,
  `/tmp/tyre-final-molla-tests.log`, `/tmp/tyre-final-rapier-tests.log`,
  `/tmp/tyre-final-gpu-parity.log`, `/tmp/tyre-final-native-build.log`.

### Singular articulation response performance (2026-09-20)

#### Goal

- Continue the full `FULL_COMPACT.md` objective, specifically CPU tyre/rigid
  performance. The preceding goal turn made concrete progress with GPU rubber
  rendering; general performance and all acceptance scenes remain incomplete.

#### Instructions

- Preserve the original Gearbox checkout and active instance. No original
  Gearbox edits, launches, stops or merges were made during this continuation.
- Keep solver accuracy, eight substeps and 120 Hz unchanged. The synthetic
  benchmark below is not proof of the imported Kubota performance gate.

#### Discoveries

- `ArticulatedResponse` already cached full-rank Cholesky factors per substep,
  but its singular/ill-conditioned fallback reran normalization and spectral
  decomposition for each wheel's wrench. Locked D6 coordinates exercise this
  path. All wheel responses in one `apply` share the same kinematic state.
- The checked-in Makefile expects an absent `PROJECT` file. Native test recipes
  run with command-line `PROJECT_NAME=molla PROJECT_VERSION=0.0.1` overrides;
  no environment, recipe or project metadata file was edited. A harmless
  missing-PROJECT `sed` diagnostic still appears during shell setup.
- Engram tools remain unavailable; this tracked checkpoint is the handoff.

#### Accomplished

- Extracted a reusable normalized `MassFactor` for Cholesky/spectral solves.
  Singular tyre response now factors once per articulation per substep and
  reuses the result across wheel wrenches. The existing full-rank fast path,
  eigenvalue cutoff, nullspace behavior and indefinite-mass rejection remain.
  The cache is still scoped to one `WheelForces::apply`, not persisted across
  pose/topology changes.
- Added analytic full/rank-deficient factor tests over scales 1e-100 through
  1e100, repeated RHS solves, nullspace threshold, malformed/nonfinite inputs,
  indefinite systems and zero-dimensional systems.
- Added a 4,910 kg synthetic 26-body fixture with four pressure tyres, a
  partially locked D6 accessory and eight substeps. It checks force support,
  no quarantines, no accumulated transient forces, actual driving velocity
  and exact two-run replay. An ignored release timing test runs idle/driving
  workloads after settling. It does not use real imported mesh colliders.
- Before: median idle 0.36505 ms/step; driving 0.43387 ms/step. Final repeat:
  idle 0.30681 ms/step; driving 0.35890 ms/step (approximately 16% and 17%
  reductions). Pressure geometry, force models and solver settings were not
  simplified. Five 240-step batches were measured in each phase.
- Full CPU solver lib/integration run with `--no-fail-fast --test-threads=1`:
  369 passed, 1 failed, 2 ignored across 87 test binaries. The failure is
  `mesh_bvh_perf::bvh_accelerates_sphere_on_grid_mesh`: 256.5 microseconds versus
  its 200 microsecond release budget; an earlier run measured 238.6. This test
  directly calls collision generation, not the changed mass solve. The full
  suite must not be reported green, and the threshold was not relaxed.
- Isolated Gearbox binary regression tests pass for both Molla and Rapier
  selection (78 each, one GPU test ignored per run). Its running viewer was
  not rebuilt/restarted to claim a new live-frame or real-Kubota timing.

#### Next Steps

- Investigate the repeatedly failing mesh BVH collision performance gate;
  inspect geometry preparation and candidate traversal before changing code.
- Benchmark the real imported machine and fleet/terrain scenes after rebuilding
  the isolated Gearbox binary. Synthetic under-1-ms timings do not satisfy
  the real-machine acceptance requirement.
- Continue the remaining full-goal lifecycle, sleeping, pressure, terrain,
  hitch/towing, slope and deterministic replay acceptance work.

#### Relevant Files

- `crates/molla-solvers/src/rigid_impulse.rs` — reusable factorization and tests.
- `crates/molla-solvers/src/articulated_response.rs` — per-substep factor reuse.
- `crates/molla-solvers/tests/rigid_performance.rs` — synthetic CPU workload.
- `/tmp/molla-rigid-perf-before.log`, `/tmp/molla-rigid-perf-after.log`,
  `/tmp/molla-rigid-perf-final.log`, `/tmp/molla-mass-factor-full-tests.log`,
  `/tmp/gearbox-mass-factor-molla-tests.log`,
  `/tmp/gearbox-mass-factor-rapier-tests.log` — validation evidence.
- Reproduce the timing test through the native recipe:
  `nix develop --impure -c make test PROJECT_NAME=molla PROJECT_VERSION=0.0.1
  APP_TARGET='-p molla-solvers --release --test rigid_performance --
  --include-ignored --nocapture --test-threads=1'` (join onto one command line).

### Triangle-mesh collision cache checkpoint (2026-09-20)

#### Goal

- Continue `FULL_COMPACT.md`, prioritizing performance against Gearbox/Rapier.

#### Instructions

- Keep original Gearbox untouched; use only the isolated physics worktree.
- Preserve tyre behavior and existing performance thresholds.

#### Discoveries

- Stateful collision rebuilt raw triangle-mesh BVHs for each pair/query.
- Raw mesh data is publicly mutable; pointer identity cannot safely detect
  same-length vertex/index edits or replacement models.

#### Accomplished

- Added geometry-owned pipeline BVH reuse with exact source-data comparison
  once per queried mesh per collision call, shared across shape pairs.
- Covered positive contacts, both mesh pair orientations, CPU GJK/MPR,
  pose/material changes, same-length vertex/index edits, resizing, model
  replacement, empty scenes, exclusions, and GPU-selected CPU mesh fallbacks.
- Initial release grid benchmark improved from 238–256 to 31.8 microseconds
  per call. The 200-microsecond budget remains unchanged. Added an assertion
  that 1,000 repeated collision calls build the mesh BVH exactly once.
  Standalone release repeat: 28.9 microseconds/call, one BVH build.
- Native release geometry/solver suite: 557 passed, 0 failed, 2 ignored across
  111 binaries. Native GPU mesh-fallback test: 1 passed, covering MPR, full
  narrow phase and GJK prefilter selection. Isolated Gearbox Molla binary
  tests: 78 passed, 1 ignored. Logs are in `/tmp/molla-bvh-cache-full-tests.log`,
  `/tmp/molla-bvh-cache-gpu-tests.log`, and
  `/tmp/gearbox-bvh-cache-molla-tests.log`.
- Isolated Gearbox Rapier selection also passes 78 binary tests, one ignored:
  `/tmp/gearbox-bvh-cache-rapier-tests.log`.

#### Next Steps

- Commit the reviewed cache changes and rebuild isolated Gearbox for live
  real-machine measurements.
- Source comparison and broad-phase mesh-bound calculation still scan raw
  geometry. This is not a claim of constant-time terrain handling.
- Rebuild and benchmark real-machine/fleet scenes; synthetic improvements
  and flat-scene FPS do not establish general Rapier performance parity.
- Continue all outstanding full-plan acceptance requirements.

#### Relevant Files

- `crates/molla-geometry/src/mesh_cache.rs` — persistent source/BVH cache.
- `crates/molla-geometry/src/narrow_phase.rs`, `src/pipeline.rs` — cached dispatch.
- `crates/molla-geometry/tests/collision_mesh_cache.rs` — lifecycle/parity tests.
- `crates/molla-solvers/tests/mesh_bvh_perf.rs` — existing timing gate/build count.

### Loop/motor factor and profiling checkpoint (2026-09-20)

#### Goal

- Continue the full plan with measured real-machine physics optimization.

#### Instructions

- Original Gearbox remains untouched. Do not confuse synthetic speedups or
  traced timings with real-machine Rapier performance parity.

#### Discoveries

- Rebuilt isolated Kubota still measures about 1.5 ms/step after prior cache
  work. Straight driving reaches 2 m/s with all tyres at 1.8 bar, roughly
  53.5–54.3 mm deflection. No new live-machine gain was demonstrated.
- Warmed stage trace: collision 19 us/substep, external forces 148 us
  (including wheels 66 us), Featherstone 67 us (drive inertias 36 us).
  Timings are inclusive; eight substeps. Tracing raises full-step cost to
  about 2 ms, so compare against untraced measurements separately.
- Detailed Kubota loop trace shows 39 DOFs, 22 rows, successful Cholesky:
  response assembly about 62 us, including 29 us for mass RHS solves/products.
  The singular fallback cache does not accelerate this full-rank workload.

#### Accomplished

- Reused singular decompositions across loop-closure RHS vectors and motor
  inertia queries within one articulation/substep; no cross-state cache.
- Added a 26-body pressure-tyre fixture with a compliant closure and active
  D6 motor, checking support, motion, closure convergence and exact replay.
  Release timing median improved 0.855414 to 0.527412 ms/step (38%).
- Solver release suite: 371 passed, 0 failed, 3 ignored across 87 binaries.
  Final loop/pressure tests after stronger replay assertions: 5 passed,
  2 ignored. Isolated Molla binary tests: 78 passed, 1 ignored.
- Added disabled-by-default debug stage spans and isolated Gearbox's
  `GEARBOX_TRACE_FILTER` override. Native profile build passed; focused
  traces avoid recording unrelated render events.

#### Next Steps

- Optimize the measured full-rank loop response, not only the singular
  synthetic fixture. Keep reference algebra and live/CPU acceptance evidence.
- Continue fleet/sleeping, lifecycle, terrain, slope, towing and the rest
  of the original plan. Neither this checkpoint nor library tests complete it.

#### Relevant Files

- `crates/molla-solvers/src/{loop_forces,featherstone,crba,rigid_world,wheel_forces}.rs`
  — factor reuse and stage spans.
- `crates/molla-solvers/tests/rigid_performance.rs` — loop/motor workload.
- Isolated `bin/gearbox/src/main.rs`, `specs/TYRE_PERFORMANCE.md` — trace filter
  and measured outcomes, including the lack of proven live-machine improvement.
- `/tmp/molla-loop-perf-{before,after}.log`,
  `/tmp/molla-loop-cache-{full,final}-tests.log`,
  `/tmp/molla-{physics,loop}-stages.json`,
  `/tmp/molla-{physics,loop}-stages-summary.txt`, `/tmp/molla-loop-live.log`,
  `/tmp/molla-loop-state.json` — temporary evidence (not tracked).

### Full-rank loop response checkpoint (2026-09-20)

#### Goal

- Optimize the actual Kubota loop workload identified by the stage trace.

#### Discoveries

- The 39-DOF, 22-row loop response used both triangular solves per RHS,
  followed by all pair products. For Cholesky `M = scale * L * L^T`, the
  equivalent response is the Gram matrix of `L^-1 * J^T`, divided by scale.
- New extreme-scale reference tests exposed an existing small-inertia failure:
  unnormalized Cholesky could succeed but forward substitution rejected its
  tiny diagonal. Normalized factorization handles the tested 1e-60 scale.
- Background release builds and the user's independently changing original
  viewer make these live samples unsuitable for claiming controlled parity.

#### Accomplished

- Full-rank response uses one forward solve per row and one triangular Gram
  product, mirrored symmetrically. Singular matrices retain the cached
  generalized-inverse path. Solver settings and physical models are unchanged.
- Tests compare accumulated response with independent inverse products for
  full/singular rank, zero projected rows and scales 1e-60, 1 and 1e60.
- Added an ignored paired 39-DOF/22-row benchmark alternating old/new order.
  Gram batches: 15.292–21.525 us; direct inverse-product batches:
  25.535–43.504 us. Every paired Gram batch was faster (38–59% less time).
  This is response-assembly cost, not a whole-step speedup claim.
- Native release solver suite: 372 passed, 0 failed, 4 ignored, 87 binaries.
  Isolated Molla binary tests: 78 passed, 1 ignored. Profile build passes.
- Live untraced windows remained above 1 ms. The final driving command reached
  2 m/s, then slowed to zero around x=8.82 m; a viewer selection event occurred
  during the run, and the instance closed before final telemetry collection.
  Do not classify this as accepted continuous driving or assume its cause.
  Asked the user asynchronously whether they interacted with/closed that window.
- A render slab-allocator use-after-free diagnostic appears before machine load
  in both pre-Gram and post-Gram logs. No shared renderer files were changed.

#### Next Steps

- Resolve the final live stop/closure: inspect command ownership/UI override
  and reproduce without interactive inputs, preferably in an imported-machine
  headless benchmark. `machine move` keeps publishing commands throughout the
  requested duration; its final "drove" acknowledgement does not prove motion.
- Record controlled real-machine timing; concurrent user builds affected all
  sampled stages. Do not claim the real machine meets 1 ms or Rapier parity.
- Continue measured full-machine/fleet work and remaining full-plan acceptance.

#### Relevant Files

- `crates/molla-solvers/src/loop_forces.rs` — normalized Gram response,
  scale/reference tests and paired benchmark.
- `/tmp/molla-gram-before.log` — old timing and reproduced tiny-scale failure.
- `/tmp/molla-gram-after.log`, `/tmp/molla-gram-paired.log` — correctness/timing.
- `/tmp/molla-gram-full-tests.log`, `/tmp/gearbox-gram-molla-tests.log`,
  `/tmp/gearbox-gram-build.log` — regression/build evidence.
- `/tmp/molla-gram-stages.json`, `/tmp/molla-gram-stages-summary.txt`,
  `/tmp/molla-gram-live.log`, `/tmp/molla-gram-drive.log` — live evidence.

### Selected-machine command ownership checkpoint (2026-09-20)

#### Goal

- Resolve the live straight-drive stop before accepting physics optimizations.

#### Instructions

- Original Gearbox remains untouched; all controller changes are isolated.
- Keep UI/gamepad control available, but selection alone must not own driving.

#### Discoveries

- Isolated keyboard routing unconditionally sent zero `UiDrive` commands for
  a focused selected machine, even with no WASD key held. `apply_ui_drive`
  applies those after bus input, overriding a continuing CLI drive command.
- Two headless Bevy regression tests reproduced this before the fix: incoming
  `(2, 0.4)` commands became `(0, 0)` on idle selection and after key release.
- A separate mid-drive tyre edit was refused because its CLI process tried
  to claim the session already held by `machine move`. `--take` would interrupt
  that driving session and is not a valid concurrent-edit acceptance solution.

#### Accomplished

- Keyboard ownership now starts only with a held driving key. Key release,
  focus loss and leaving a keyboard-owned selection send one stop then yield.
  A gamepad takeover clears keyboard ownership; idle selection does not
  overwrite slider or remote commands. Opposing held keys still explicitly
  command zero. Six headless tests cover these cases.
- Both isolated backend selections pass 84 binary tests, one ignored each.
  Native `oslo make build` passes. Before-fix tests failed as expected.
- Live selected Kubota completed a 30-second straight CLI drive, with steady
  2 m/s samples throughout, no reported physics quarantine/nonfinite error,
  and four applied pressures at 1.8 bar. API selection readback and an inspected
  screenshot confirm the machine was selected and playing. Physics timings
  still approximately 1.43–1.51 ms/step; under-1-ms acceptance remains open.
- A selected-machine low-pressure turn also maintained approximately 2.02 m/s
  and 0.25 rad/s yaw. All tyres were at 0.5 bar before that run. The attempted
  4-bar mid-drive command was refused, so no transition acceptance is claimed.
- Stopped the isolated test instance by its exact name after collection.

#### Next Steps

- Support an explicit borrowed active drive session for tyre commands without
  stealing/releasing it; verify same-identity ownership, failures, and live
  pressure readback while `machine move` continues. Current CLI entry is
  `crates/gearbox-cli/src/cli/machine.rs::set_tyre_pressure`.
- Inspect the existing protocol rather than assuming session authentication:
  `gearbox-api/src/machine.rs` checks command session IDs; holder strings alone
  must not be described as transport-authenticated per-command ownership.
- Build a controlled imported-machine benchmark/headless acceptance path.
  Physics reading now uses `usd_bevy::route::physics` components and Gearbox's
  `physics/attach.rs`; the old `physics/reader/read.rs` no longer exists.
- Continue performance, fleet/sleeping, lifecycle and all remaining plan gates.

#### Relevant Files

- Isolated `bin/gearbox/src/viewer/drive/keyboard.rs` — routing fix and tests.
- `/tmp/gearbox-keyboard-before.log`, `/tmp/gearbox-keyboard-{molla,rapier}-tests.log`,
  `/tmp/gearbox-keyboard-build.log` — regression/build evidence.
- `/tmp/molla-control-check.log`, `/tmp/molla-control-drive.log`,
  `/tmp/molla-control-selected-state.json`, `/tmp/molla-control-selected.png` —
  selected live driving evidence.
- `/tmp/molla-control-low-state.json`, `/tmp/molla-control-pressure-drive.log`,
  `/tmp/molla-control-transition-start.json` — 0.5-bar turning evidence.
  Files named `/tmp/molla-control-high-state.json` and `-high.png` were captured
  after the refused high-pressure request and do not prove 4-bar operation.

## Checkpoint: concurrent driving pressure edits

### Goal

Continue the entire `FULL_COMPACT.md`; verify its CLI mid-drive pressure gate
without interrupting driving or touching the original Gearbox checkout.

### Instructions

- Changes remain in Molla and the isolated `gearbox-physics` worktree only.
- Use native make recipes, patch tools and unsigned title-only commits.

### Discoveries

- The explicit pressure command can borrow a same-identity active drive lease;
  it must not claim/release that lease on success, refusal or transport error.
  Holder comparison is CLI-side validation, not new server authentication.
- A completed CLI drive-duration message alone is insufficient evidence;
  fresh telemetry throughout the edit sequence confirms continuing motion.
- The user confirmed interacting with the older `molla-gram-perf` window.
  Its stop/closure is not evidence of an unattended crash. This confirmation
  does not identify the cause of closure; the keyboard override was separately
  reproduced by regression tests and corrected before the latest live run.

### Accomplished

- Isolated commit `dbc255c` adds `machine tyre-pressure --reuse-session`,
  rejects inactive/zero/different-holder sessions and conflicts with `--take`.
  Owned pressure sessions still release on command failure. Four new unit tests
  cover this lifecycle; six CLI unit and fourteen integration tests pass.
  Native `oslo make build` passes.
- Selected real Kubota completed a 90-second drive at forward 2 m/s and yaw
  0.12 rad/s. A mid-drive 0.5-to-4-bar edit reached all four applied pressures
  while fresh speed readback remained 2.004 m/s. Session id/holder remained
  unchanged. A front-axle edit changed only its two targets; a subsequent
  rear-left edit changed only that wheel. All final applied pressures reached
  their targets while driving on the same session.
- Invalid 99-bar input returned exit 2 without changing targets/session.
  After driving released normally, `--reuse-session` refused the idle session
  with exit 6. Stopped only the named `molla-pressure-reuse` instance.
- Inspected screenshot confirms the selected Molla/PLAYING machine, not a
  close-up tyre-shape comparison. Step samples 1.61–1.92 ms under concurrent
  user workloads are not a controlled performance comparison. Existing
  pre-load renderer/Vulkan diagnostics remain; no physics quarantine observed.

### Next Steps

- Redundant backward solves have now been removed from tyre quadratic responses
  and motor inverse diagonals. The release CPU solver suite passes 374 tests,
  zero failures, five ignored across 87 binaries. The isolated viewer rebuild
  and 84 binary tests pass, one ignored. See the live qualification below.
- Continue whole imported-machine profiling and all remaining performance,
  lifecycle, calibration, slope, fleet and deterministic replay gates.

### Relevant Files

- Isolated `crates/gearbox-cli/src/cli/machine.rs` — pressure session lifecycle.
- Isolated `specs/TYRE_PRESSURE.md`, `specs/TYRE_PERFORMANCE.md` — command and evidence.
- `/tmp/gearbox-pressure-session-{tests,build}.log` — build/test results.
- `/tmp/molla-reuse-{before,after,high,axle,invalid-state,wheel,final-state}.json`
  and `/tmp/molla-reuse-drive.log` — live telemetry and completed drive.
- `/tmp/molla-pressure-reuse.log`, `/tmp/molla-reuse-scoped.png` — viewer evidence.

## Checkpoint: quadratic mass responses

- `MassFactor::quadratic` computes `rhsᵀ M⁺ rhs` without reconstructing the full
  inverse response: one forward solve and squared norm for Cholesky, or squared
  retained eigenbasis projections for singular mass matrices. Scaling the RHS
  by the square root of the mass scale before squaring avoids intermediate
  overflow/underflow. Wheel response caches now use this shared factor directly.
- Motor effective inertia uses the same Cholesky norm identity for requested
  inverse diagonal entries; singular fallback semantics remain unchanged.
- Analytic full/singular-rank tests, nullspace cutoff, invalid inputs, zero RHS,
  mass/RHS scales from 1e-300 to 1e300, and a multi-joint motor-inertia comparison
  against full inverse columns pass. The motor comparison includes a massless
  tip and scales 1e-60/1/1e60. Existing physics/replay tests remain green.
- Alternating paired 39-DOF cached quadratic microbenchmarks measured full-rank
  0.326–0.358 us versus 0.791–1.053 us for full solve plus dot product; singular
  0.400–0.402 us versus 0.992–1.201 us. These isolate response evaluation and do
  not measure factorization, whole physics steps or imported-machine parity.
- Evidence: `/tmp/molla-quadratic-{unit-tests,full-tests,timings}.log`.
- Files: `crates/molla-solvers/src/{rigid_impulse,articulated_response,featherstone}.rs`.

### Live verification and remaining work

- Rebuilt selected Kubota completed a 60-second 2 m/s, 0.12 rad/s drive.
  Fresh 4-bar telemetry after the 0.5-to-4-bar transition showed 2.004 m/s,
  front deflection 34.60/35.66 mm and rear 36.94/38.29 mm. Inspected the
  high-pressure screenshot; it confirms the selected playing machine but is
  not a close-up shape comparison. Drive log ended with normal release.
- A return-to-0.5-bar request was accepted, but fresh readback then failed at
  `/gearbox/info` resolution, including a longer-timeout retry. The same viewer
  process and physics logs were confirmed still live. This is not evidence of
  a physics crash or proof of completed deflation. The return-low JSON is empty.
  Stopped only `molla-quadratic-perf` by its exact name after verification.
- Warm live physics samples ranged 1.40–2.01 ms under changing user workloads.
  The under-1-ms whole-machine gate remains unmet; microbenchmarks are not parity.
- Next: controlled imported-machine/headless profiling, remaining CPU mass/Jacobian
  and loop work, and the full plan's fleet, lifecycle, GPU and tyre acceptance
  gates. Investigate the CLI reachability failure separately if reproduced;
  do not change shared peerbus/agentio or original Gearbox as a shortcut.
- Additional evidence: `/tmp/gearbox-quadratic-{tests,build}.log`,
  `/tmp/molla-quadratic-{live,drive,stop}.log`, `/tmp/molla-quadratic-{low,high}.json`,
  `/tmp/molla-quadratic-high.png`.

## Checkpoint: real imported-machine headless acceptance

### Goal

Continue all of `FULL_COMPACT.md`, with a repeatable imported-machine physics
benchmark and replay gate rather than relying on window activity or synthetic
solver microbenchmarks.

### Instructions

- Original Gearbox, machine assets, shared USD/renderer dependencies and build
  recipes remain untouched. Changes are in the isolated physics checkout.
- No subagents. Use patch tools and unsigned title-only Conventional Commits.

### Discoveries

- The real asset's authored id is `robot`; the live loader changes it to the
  filename-based `kubota_tractor`. Headless physics can retain the authored id.
- The projected Kubota fixture has 26 bodies, 31 joints, 18 colliders including
  flat ground, four pressure tyres and 4916.010223 kg in this setup. It uses
  native USD projection, adapters, inertia guard and drive/wheel systems, not
  a reconstructed toy tractor. Imported rubber radii match the live values.
- Exact replay initially differed by 5.36e-14 after 60 steps. Sorting only the
  harness's initial placement order still differed by 2.68e-13. Production
  `prepare_machine_physics` collected body handles from a randomized HashMap;
  sorting this list stabilizes mass recomputation and tyre mass accumulation.
- Focused headless tracing confirms the remaining work is spread across
  collision, wheels, loop response and Featherstone; mass matrices are still
  rebuilt in wheels, loops and drive inertia within each substep.

### Accomplished

- Isolated commit `6dc1369` adds external-asset ignored benchmark/replay tests,
  a profile-only trace test, and stable production body preparation order.
- All six 0.5/1.8/4-bar parked/driving phases passed applied-pressure, support,
  imported-radius, finite-body, no-quarantine and speed checks. Increasing
  pressure decreases mean loaded deflection. Driving remains about 2.004 m/s.
- 600-sample dev/test-profile physics-step medians were 1.049–1.054 ms parked
  and 1.064–1.107 ms driving. Dependencies are optimized but the adapter/test
  harness is not a release build. This still does not meet the under-1-ms gate.
- Two independent imported scenes now match exactly at 30 checkpoints over
  1,800 steps with pressure, drive and stop edits. Snapshots include body
  poses/velocities and tyre pressure/load/grip/slip/deflection/footprint output.
  Separate after-fix runs passed; this does not establish every lifecycle replay.
- Both backend selections pass 84 binary tests, three ignored. Explicit external
  benchmark/replay run passes two tests, and the profile-only trace test passes.
- The trace records only 120 already-warmed driving steps (4.9 MB). Inclusive
  substep means: collision 19.91 us, wheels 44.70 us, loops 51.60 us, Featherstone
  44.68 us including drive inertia 20.83 us. Each of three mass-matrix builds is
  about 5–6 us. Traced step mean 1.401 ms is not an untraced timing comparison.

### Next Steps

- The first release imported benchmark/replay build completed after 7m20s;
  session `9005` is terminal. `/tmp/gearbox-imported-release.log` records both
  pressure tests passing, with medians 0.835–0.853 ms but p95 up to 1.199 ms.
  The later shared-response checkpoint below records paired follow-up results.
- Consider substep-scoped mass/factor reuse across wheel, loop and motor response
  only where state/control semantics permit it; preserve reference correctness,
  singular matrices, replay and convergence. Measure on the imported harness.
- Remaining full-plan gates include performance tails/fleet, sleep/islands,
  runtime lifecycle, GPU semantics/parity, slope/hitch and calibrated tyre/soil
  response. No completion or Rapier parity claim is warranted.

### Relevant Files

- Isolated `bin/gearbox/src/controller/benchmark.rs` — real-asset benchmark,
  bitwise replay and short solver trace with fixed 120-Hz stepping.
- Isolated `bin/gearbox/src/physics/benchmark.rs` — native projection/conversion
  fixture; controller.rs sorts prepared body handles.
- Isolated `specs/TYRE_PERFORMANCE.md` — measurements, limitations and commands.
- `/tmp/gearbox-imported-{final,replay-before,replay-sorted-fixture,replay-sorted-runtime,replay-telemetry,unit-tests,rapier-tests,trace}.log`.
- `/tmp/molla-imported-solver.json`, `/tmp/molla-imported-solver-summary.txt`,
  `/tmp/molla-imported-trace-summary.mjs` — trace and streaming summarizer.
- Asset SHA-256: `8aa28ea5bb56c0af11ac233d884661d1d775a2a7ab2d6c860dd2af3effb77718`.

## Checkpoint: share external-force articulation responses

### Goal

Improve full imported-machine physics performance without reducing substeps,
tyre fidelity, loop constraints or acceptance scope.

### Instructions

- Preserve the original Gearbox checkout, external assets, shared dependencies
  and existing build/environment recipes. No user processes were stopped.
- Engram tools remain unavailable; this checkpoint is the persistent handoff.

### Discoveries

- Wheel and loop forces query the same immutable model/state during one external
  force callback. Their mass factors and COM Jacobians can be shared safely
  within that callback; the next substep must get a fresh cache.
- Motor inertia is evaluated after Featherstone's mimic pre-pass and FK. It
  cannot blindly reuse the earlier state's factor. This change leaves its
  mass evaluation separate rather than introducing a stale-state shortcut.
- Concurrent user Magi builds affected the first post-change timing run. Only
  the subsequent alternating baseline/new runs below are used for speedup.

### Accomplished

- `ArticulatedResponse` now borrows one model/state and owns per-articulation
  factors plus per-body COM Jacobians. `RigidWorld` creates it inside each
  substep's external-force callback and shares it between wheels and loops.
  No cache survives state integration or a scene edit.
- Moved the existing full-rank Gram and singular fallback computation into
  `MassFactor::add_gram`; scaling, rank threshold and accumulation order remain.
  Tests cover shared wheel/loop factor reuse across two articulations, Jacobian
  reuse, fresh rotated-state response, zero matrices and invalid Gram dimensions.
  Existing scaled full/singular inverse-product comparisons remain green.
- Release CPU solver suite passes 376 tests, zero failures, five ignored across
  87 binaries. Imported release benchmark and exact replay both pass. Logged
  before/after pressure/drive checkpoints are text-identical (including reported
  poses, speed, load and deflection); this is sampled comparison, not every-step
  baseline/new trajectory equality.
- Preserved old/new release test executables and ran before/after/after/before
  sequentially on CPU 2 with no overlapping assistant build/test workloads.
  Across all pressures, old medians were 0.8335–0.8523 ms, new medians
  0.7586–0.7814 ms; averaging the two medians per condition gives about 8–9%
  reduction. New p95 values were 0.7786–0.8668 ms. Occasional maxima still reached
  1.201 ms on this shared host; no hard real-time or Rapier-parity claim follows.
- The post-change short trace passed and confirms 1,920 mass-matrix evaluations
  over 120 steps, versus 2,880 before: wheel and motor builds remain, loop builds
  are eliminated. Other user builds disturbed stage timings, so use the trace
  for attribution/counts, not a before/after timing ratio.
- Isolated binary tests pass 84 tests, three ignored. Native build passes.
  Final native rebuild after removing an unused import also passes; session
  `54906` is terminal. Log: `/tmp/gearbox-shared-response-build-final.log`.

### Next Steps

- No build/test process remains running for this checkpoint. Preserve the paired
  binaries/logs for further comparisons; rebuild normally for any new source.
- Continue whole-machine and fleet profiling, sleep/islands, lifecycle,
  slope/towing/PTO acceptance and the remaining tyre/terrain/GPU requirements.
  Headless release median/p95 samples do not replace the entire full-plan audit.

### Relevant Files

- `crates/molla-solvers/src/articulated_response.rs` — state-bound factor/Jacobian cache.
- `crates/molla-solvers/src/{wheel_forces,loop_forces,rigid_world}.rs` — callback-scoped reuse.
- `crates/molla-solvers/src/rigid_impulse.rs` — reusable Gram response and validation.
- `/tmp/molla-shared-response-{tests,unit,full-final}.log` — solver validation.
- `/tmp/gearbox-shared-response-{release,paired,trace,unit,build}.log` — imported and adapter evidence.
- `/tmp/gearbox-imported-{before,after}-cache` — immutable baseline/new release test executables.
- `/tmp/molla-shared-response-solver.json` — post-change short trace when complete.
- `/tmp/molla-shared-response-solver-summary.txt` — streamed post-change stage counts.

## Imported Kubota slope acceptance checkpoint

### Goal

Continue the complete `FULL_COMPACT.md` objective with its real-machine,
60-second, 10-degree slope-parking gate at both supported pressure extremes.
The preceding acknowledgement-only turn made no implementation progress;
this turn resumed with a new reproducible acceptance test and failure evidence.

### Instructions

- Original Gearbox and shared dependencies remain untouched. Only the isolated
  Gearbox-physics test fixture/documentation and this progress file changed.
- The user confirmed interacting with `molla-gram-perf`; that is not evidence
  that they closed it or that its stop/closure was spontaneous.
- Engram tools are unavailable in the current tool registry. This tracked
  checkpoint is the handoff fallback, not a claim that memory calls succeeded.

### Discoveries

- The imported Kubota fails unattended slope parking on current Molla
  `b6d8f27`. After 10 simulated seconds of settling, all four combinations
  creep about 8–9 m during the next 60 seconds. Two complete runs reproduced
  identical reported drift/speed values. This is actual test evidence, not a
  window-interaction inference.
- The test checks both +/-10-degree slopes and 0.5/4.0 bar. Maximum tangent
  drift / maximum speed were: -10/.5: 7.998699257 m / 0.133314298 m/s;
  -10/4: 9.031795585 / 0.150532806; +10/.5: 7.973623087 / 0.133080483;
  +10/4: 9.042705112 / 0.149730977.
- Every step remained finite/enabled with no quarantine. Per-second checks
  passed for all four tyre contacts, applied pressure, actual rotated ground
  normals, and normal load within 500 N of `mass*g*cos(10 degrees)`.
- Final tyre slip magnitudes were approximately 0.19–0.22 with nonzero wheel
  rotation. The fixture records these diagnostics, but they do not alone
  isolate how much drift comes from each force/drive mechanism.
- Source inspection shows the pressure path evaluates a steady brush law:
  tread displacement comes from instantaneous regularized slip, not retained
  contact shear. At zero longitudinal slip its longitudinal support vanishes.
  `wheel_forces::apply` also dissipatively limits tangent wrench using current
  velocity. A static shear implementation cannot blindly keep that limiter:
  elastic recovery can do positive work using previously stored energy.
- Parking currently uses zero-velocity motors, not a retained wheel-angle
  brake. Featherstone's scalar feedback regularization scales damping by
  `1 + kd*dt/effective_inertia` without external-acceleration prediction. These
  are code-level mechanisms to investigate, not yet isolated causal fixes.

### Accomplished

- Added `Fixture::incline`, retaining the ground collider handle and rotating
  all imported poses plus the slab by the same transform before settling.
  Poses are collected before writes, and body order is deterministic.
- Added the explicitly invoked external-asset slope gate. It measures maximum
  drift/speed across all 7,200 steps, requiring <10 mm tangent drift and
  <1 mm/s chassis speed, and reports all four scenarios before asserting.
  These tolerances were fixed before the first measured run and not relaxed.
- The gate fails as recorded; its ignore attribute is only for the external
  asset requirement. No production physics/gains were altered or bodies frozen
  to manufacture a pass. No Rapier or rendered slope acceptance is claimed.
- Release binary regression suite passes 84 tests, with four external-asset
  tests ignored. Test sessions 21490, 35115 and 10127 are terminal; there is no
  outstanding assistant test/build/viewer process for this checkpoint.

### Next Steps

- Fix physical standstill tyre shear and parking-brake behavior, with isolated
  force/drive regressions before rerunning the full imported slope gate.
  Preserve finite-state, pressure/support checks and existing drive/replay
  behavior; remeasure whole-machine performance after any solver change.
- Review persistent contact-state lifecycle (airborne, teleport, pressure,
  topology/ground changes) and elastic energy when adding static shear.
- Continue every remaining full-plan requirement; this failing gate is not
  slope completion and the goal remains active.

### Relevant Files

- `crates/molla-solvers/src/{pressure_tire,wheel_forces,featherstone}.rs` —
  inspected pressure/force/motor paths; no production edits this checkpoint.
- `crates/molla-vehicle/src/real_tire/rig/runners.rs` — steady brush law.
- `/home/bresilla/data/code/robolibs/gearbox-physics/bin/gearbox/src/controller/benchmark.rs`
  — imported slope fixture, continuous drift/speed and support checks.
- `/home/bresilla/data/code/robolibs/gearbox-physics/docs/MOLLA_BACKEND.md`
  — reproducible command, thresholds and explicitly failing results.
- `/tmp/gearbox-slope-{initial,diagnostic}.log` — two failed acceptance runs.
- `/tmp/gearbox-slope-unit.log` — passing ordinary release binary suite.

## Persistent tread-shear constitutive checkpoint

### Goal

Continue the full plan by implementing the retained contact shear needed by
the reproduced slope-parking failure. The previous goal turn made progress:
it committed the imported-machine gate and measured its failure. This turn
implements and tests the constitutive component before changing live forces.

### Instructions

- Original Gearbox and the isolated Gearbox integration are unchanged here.
- Keep reusable tyre behavior in `molla-vehicle`, with no viewer dependencies
  or second wheel/chassis integrator. Engram tools remain unavailable.

### Discoveries

- Elastic tread contact can legitimately do positive instantaneous work while
  unloading. A velocity-only dissipative wrench limiter would erase this
  response. Its admissibility must include the change in stored elastic energy.
- The new constitutive update satisfies `force dot slip_displacement +
  change_in_elastic_energy <= 0` for fixed parameters. This local kinematic
  work/energy bound is not proof of stability after coupling to Featherstone.
- Changing tread stiffness externally changes stored energy even without slip;
  inflation/load/geometry transitions need explicit handling at integration.

### Accomplished

- Added public `TreadBristle`, `TreadBristleParams` and `TreadBristleOutput` in
  `molla-vehicle::real_tire`. State stores two tangent-frame elastic displacements;
  inputs are ground-relative tread velocity, dt, integrated shear stiffness,
  one combined friction budget, and loaded-material renewal rate.
- The update uses backward-Euler renewal and circular elastic/plastic return.
  Zero renewal retains static shear. Friction saturation projects stored shear
  as well as force, preventing windup. Zero budget/stiffness clears history.
  Invalid/non-finite updates reject atomically. Runtime work is allocation-free.
- Tests cover 60 simulated seconds of retained standstill force, combined-slip
  saturation, plastic reversal, recoverable elastic work, analytic renewal,
  tangent-frame covariance, deterministic work/energy bounds, reduced budgets,
  input rejection/reset, and a driven mass that settles below the friction
  limit but continues moving when overloaded.
- This is a tested constitutive primitive, not live tyre adoption or a slope
  fix. The existing pressure brush, wheel force limiter, parking controller,
  imported slope results and measured performance are unchanged.
- Final native Make release gate passed all 241 vehicle library tests, zero
  failures or ignored tests, including nine new bristle tests. Command:
  `nix develop --impure -c make test PROJECT_NAME=molla PROJECT_VERSION=0.0.1
  APP_TARGET='-p molla-vehicle --release --lib --tests --no-fail-fast -- --test-threads=1'`.
  Final session 54564 is terminal; no assistant process remains running.

### Next Steps

- Integrate retained shear into pressure contact with consistent per-cell load,
  force/moment readout and surface budgets. Preserve current pressure/load
  response rather than replacing it with a universal grip multiplier.
- Keep history transactional across solver steps; preserve it on unrelated
  edits and clear/transform it appropriately on separation, removal, teleport,
  contact/frame changes and pressure-driven patch changes.
- Couple elastic recovery through a stiffness/timestep-aware force path, not
  the current purely dissipative tangent limiter. Validate full dynamics and
  timestep convergence; constitutive passivity alone is insufficient.
- Implement/validate bounded parking hold, rerun the imported 60-second slope
  gate at both pressure extremes, then drive/replay and whole-machine timing.
- The complete `FULL_COMPACT.md` objective remains active and incomplete.

### Relevant Files

- `crates/molla-vehicle/src/real_tire/bristle.rs` — persistent shear and tests.
- `crates/molla-vehicle/src/real_tire/mod.rs` — public constitutive API exports.
- `/tmp/molla-bristle-unit.log` — initial eight focused tests passed.
- `/tmp/molla-bristle-full.log` — initial full vehicle suite, 240 passed.
- `/tmp/molla-bristle-full-final.log` — final suite after budget/atomicity tests.

## Live persistent pressure-shear checkpoint

### Goal

Continue the full objective by wiring the preceding tested bristle primitive
into the actual pressure-tyre/Featherstone path. The preceding goal turn was
progress (committed constitutive mechanics and tests); this turn adds live
force integration, lifecycle regressions and imported-machine evidence.

### Instructions

- Original Gearbox, assets and shared dependencies remain untouched. No
  machine is frozen or given a second wheel/chassis integrator.
- Engram tools remain unavailable. This tracked checkpoint is the handoff.

### Discoveries

- Retained shear materially reduces the reproduced slope failure but does
  not fix parking: current zero-velocity wheel motors still allow rotation.
  Final +/-10-degree tests at 0.5/4.0 bar creep 3.8369–4.4852 m in 60 seconds,
  versus 7.9736–9.0427 m before this change. The unchanged acceptance gate
  still FAILS; no tolerance, brake gain or motor was changed.
- Final +10-degree/4-bar wheel slip magnitudes are about 0.00068–0.00213,
  rather than about 0.215–0.220 previously. Nonzero wheel rotation remains;
  this directs the next investigation to bounded parking hold rather than
  increasing terrain friction or freezing the chassis.
- The compiled scene stamp survives normal simulation/motor edits. Checking
  committed wheel/bearing/ground poses on scene edits distinguishes actual
  teleports from unrelated rebuilds without clearing every tyre's shear.
- Removing tyre-ground registration restores ordinary collider contact. The
  contact-loss regression explicitly zeros its subsequent test input velocity
  before checking for stale shear; otherwise new collider-induced motion can
  legitimately generate a new tyre force. Production bodies are not frozen.

### Accomplished

- Added `TireBrushState` and a transactional planar persistent-brush evaluator.
  It shares normal/patch geometry, pressure/load-dependent shear stiffness,
  surface budgets and cell readout with the steady brush. Per-segment history
  renews with tread transport, clears for absent cells/changed geometry, and
  projects into changed tangent frames without increasing shear magnitude.
  Existing material damping and structural bias remain in the cell force.
- Pressure tyre evaluation transforms retained shear between world contact
  frames and maps the resulting forces/moments back to the same Featherstone
  body wrench. Normal support and rolling-resistance moment remain single
  force contributions; actual slip reporting is not replaced by elastic strain.
- `WheelStep` owns cloned transient history and publishes only after a
  successful rigid step. Unrelated rebuild/re-registration preserves history;
  wheel/pressure removal, ground removal/re-registration, disabled/missing
  contact and detected teleport clear it. Contact-ground handle changes reset
  that wheel. Frame changes project the cell state.
- Pressure contacts no longer pass through the purely dissipative tangent
  limiter, which would erase elastic recovery at standstill. Legacy tyres
  retain it. This is an explicit constitutive force integration, not a fully
  coupled implicit elastic solve; timestep/convergence coverage must continue.
- Added tests for 60 seconds of held zero-speed force, world-frame projection,
  airborne reset, non-expansive/atomic frame transforms, live shear persistence
  across unrelated rebuild and failed-step rollback, teleport and ground loss.
- Native Make release suite passed 621 tests (379 solver + 242 vehicle), five
  ignored across 88 binaries: `/tmp/molla-shear-full.log`. Synthetic pressure
  replay and loop/timestep convergence gates passed. Final focused solver
  library/wheel tests and isolated release binary tests also pass (84 binary
  tests, four external-asset tests ignored).
- Real imported pressure benchmark and exact replay pass. Driving speed stays
  about 2.004 m/s across 0.5/1.8/4 bar, with four-wheel support and decreasing
  deflection under inflation. The combined imported command reports two
  passes and one failure because slope parking is still failing.
- This run's complete imported-machine physics medians were 0.7423–0.7587 ms;
  p95 0.8522–1.1040 ms and maxima up to 2.0531 ms on the shared host. No other
  assistant build/test overlapped those samples. These are CPU step timings,
  not rendered frame times, paired speedup evidence, hard real-time guarantees
  or Rapier parity. No live viewer was launched for this checkpoint.

### Next Steps

- Implement physically bounded parking hold and retest both slope directions
  and pressure extremes; preserve drive/pressure replay and measured speed.
- Continue energy/timestep stress tests and calibration. The primitive's
  local kinematic passivity proof does not establish passivity of its complete
  damped/material-biased, load-scaled articulated coupling. There is no GPU
  dynamic-shear parity claim.
- Audit same-handle ground geometry edits beyond pose changes and explicit
  re-registration, pressure-driven energy changes, and lifecycle combinations
  not yet covered. Preserve state through unrelated edits as required.
- Continue every remaining `FULL_COMPACT.md` requirement. Goal stays active.

### Relevant Files

- `crates/molla-vehicle/src/real_tire/{bristle,mod}.rs` — cell history/frame API.
- `crates/molla-vehicle/src/real_tire/rig/runners.rs` — shared persistent evaluator.
- `crates/molla-solvers/src/{pressure_tire,wheel_forces}.rs` — live shear forces,
  world contact frames, transactional history and mutation checks.
- `crates/molla-solvers/tests/wheel_forces.rs` — lifecycle/rollback regressions.
- `/tmp/molla-shear-{wheel,pressure,lifecycle,reset,full,final-focused}.log` —
  local validation; earlier reset/lifecycle logs include the test-input issue.
- `/tmp/gearbox-shear-imported-final.log` — final benchmark/replay and failing
  slope gate. `/tmp/gearbox-slope-shear.log` is an earlier pre-damping snapshot.
- `/tmp/gearbox-shear-unit.log` — passing ordinary isolated binary suite.
- All assistant sessions for this checkpoint are terminal (including 88576,
  43170, 88091 and 86281); no assistant viewer/build/test is left running.

## Imported Kubota parking-hold checkpoint

### Goal

Continue the complete plan by fixing the remaining rolling/parking failure
after retained tyre shear. Previous goal turn was progress: live shear landed
and reduced slip, but the unchanged 60-second slope gate still failed.

### Instructions

- All runtime changes are confined to isolated Gearbox-physics; original
  Gearbox, shared dependencies and assets remain untouched. Molla's solver
  source is unchanged this turn. No body freezing or direct velocity override.
- Engram tools remain unavailable; this tracked checkpoint is the handoff.

### Discoveries

- A bounded captured-angle motor hold stops the residual rolling that the
  velocity-only parking motor allowed. Combined with Molla's retained tyre
  shear, all four real Kubota slope/pressure combinations now pass the exact
  existing gate, with no tolerance, ground friction or solver-step changes.
- Capturing the actual motor coordinate avoids winding a multi-turn wheel
  toward an arbitrary zero or a principal quaternion angle. Molla hard
  revolute/prismatic coordinates can be read directly; unsupported coordinates
  must report `None`, not a fabricated zero. Soft angular coordinates and
  non-Molla coordinate support remain outside this new read capability.
- This is a torque-limited compliant angular hold with bounded reference error,
  not a calibrated dry-friction brake model. Broader braking/pressure/terrain
  calibration and lifecycle combinations remain full-plan work.

### Accomplished

- Isolated commit `176e945 fix(physics): hold parked wheel angles` adds the
  optional `Joint::motor_position` read capability and Molla's unwrapped hard
  joint implementation. Rapier returns the default `None` and retains its
  original velocity-brake behavior.
- New `controller/parking.rs` captures wheel angle on parking engagement,
  writes force-based position feedback using the existing torque budget, and
  drops the reference when driving resumes. Normal velocity motor application
  clears position stiffness before the parking override. Removed/unsupported
  joints, nonpositive/nonfinite budgets and unavailable coordinates lose holds.
- Reference stiffness is torque/0.01 rad and damping torque/(0.25 rad/s).
  Target error is bounded to 0.25 rad under sustained slip; jumps over pi
  between observations rebase the reference. Gains are uncalibrated reference
  controller parameters, not measured tractor brake stiffness.
- Three focused tests cover multi-turn capture, load resistance, drive release
  and re-engagement, zero budget, joint removal, overload slip with bounded
  stored error, and Rapier fallback. Overload still accelerates the wheel;
  it is not secretly locked. Wrong-axis coordinate queries return `None`.
- Two independent imported slope runs passed after 10 s settling and 60 s
  measurement at 120 Hz. Max drift / max speed:
  - -10 degrees, 0.5 bar: 0.000000393 m / 0.000002449 m/s.
  - -10 degrees, 4.0 bar: 0.000000489 m / 0.000002610 m/s.
  - +10 degrees, 0.5 bar: 0.000000207 m / 0.000001517 m/s.
  - +10 degrees, 4.0 bar: 0.000000250 m / 0.000001653 m/s.
  The original <10 mm drift / <1 mm/s gates, all-body finite/enabled and no
  quarantine checks, applied pressure, ground normals and support loads pass.
- The final imported release command passes all three tests: pressure/driving
  benchmark, exact deterministic replay and slope parking. Ordinary release
  binary suites pass 87 tests each with explicit Molla and Rapier selection;
  four external-asset tests remain ignored only in the ordinary suites.
- Native `GEARBOX_PHYSICS=molla nix develop --impure -c oslo make build`
  passes. The debug viewer is rebuilt, not launched. Existing Bevy deprecated
  API warning remains; no complete workspace/release acceptance is claimed.
- Final CPU-step medians were 0.7320–0.7448 ms, p95 0.7643–1.2219 ms, maxima
  up to 2.3572 ms on this shared host. No assistant build/test overlapped the
  imported samples. These are headless step timings excluding controllers,
  not rendered FPS, hard real-time bounds or matched Rapier performance.

### Next Steps

- Continue full-plan acceptance: exact requested straight/0.4-turn track run,
  five-controller hitch/PTO actuation, real drawbar capture/tow/detach, pressure
  lifecycle/save-reload and terrain/calibration, sleeping/fleet and GPU work.
- Add parking lifecycle/reconfiguration and further load/terrain coverage as
  those scenes are exercised. Current slope evidence is the real Kubota on a
  firm planar slab; it does not prove all machines, Rapier slope parity,
  deformable soil or a physically calibrated dry-friction brake.
- All assistant jobs from this checkpoint are terminal, including final
  imported session 91686 and regression/native-build session 80552. The full
  goal remains active and incomplete; no blocker or pause is requested.

### Relevant Files

- `/home/bresilla/data/code/robolibs/gearbox-physics/bin/gearbox/src/controller/parking.rs`
  — captured-angle hold and three focused tests.
- Isolated `controller.rs` — per-runtime cache and Ackermann parking override.
- Isolated `physics/{backend.rs,molla/joint.rs}` — optional motor-coordinate
  read, with Molla hard-joint support; `physics/mod.rs` exports Rapier for tests.
- Isolated `docs/MOLLA_BACKEND.md` — historical failures and current passing
  gate, controller limits, commands and measurement scope.
- `/tmp/gearbox-parking-{slope,imported,unit}.log` — imported and focused evidence.
- `/tmp/gearbox-parking-{molla,rapier}-regression.log` — 87 passed each.
- `/tmp/gearbox-parking-build.log` — successful native viewer build.

## Five-controller imported Kubota checkpoint

### Goal

Continue the full plan with actual hitch and PTO actuation on the imported
Kubota. The previous goal turn made progress by fixing parking and passing
the slope gate; this turn strengthens the fixture and controller acceptance.

### Instructions

- Only isolated Gearbox-physics test wiring/docs and this handoff changed.
  Original Gearbox, assets, shared dependencies and production physics stay
  untouched. No viewer was launched. Engram tools remain unavailable.

### Discoveries

- Earlier imported headless fixtures stepped the drive controllers but not
  the four service controllers. The fixture now runs production
  `apply_service_controllers` before each physics step, with its actual
  resources. All existing imported gates were rerun with this stronger setup.
  Earlier timings are not a five-active-controller baseline.
- The front hitch's authored range is negative (-0.395016 rad); the rear's
  range is +0.961676 rad. Normalized commands must preserve those authored
  signs rather than assuming positive hinge rotation means raising.
- Ordinary ignored-test counts include a Vulkan parity test. Correction to
  earlier summaries: the previous four ignored tests were three imported-asset
  gates plus Vulkan; the current five are four imported gates plus Vulkan.
- Source-level follow-up for towing: `MasterState::pto_rad_s` selects the first
  engaged value from a HashMap. Named front/rear service propagation deserves
  separate inspection when testing attachments; direct PTO actuation below
  does not validate that selection or master/slave service exchange.

### Accomplished

- Isolated commit `274e37d test(physics): exercise all Kubota controllers`
  adds a test-only service schedule and the real hitch/PTO acceptance gate.
  Every imported fixture now executes the drive and service controllers.
- New gate asserts all five discovered controllers, resolves the four actual
  target joints and moved links, then uses production-consumed LinkValues.
  It checks actual unwrapped joint coordinates, not just target readback.
- Commands 0.75 then 0.15 move and return both hitches, with raised-state
  authored-limit checks and <=0.05 rad tracking error. Raised front/rear
  coordinates are -0.295727/+0.686723 rad; returned -0.058691/+0.106347 rad.
- Front/rear PTOs engage at their authored 1000/540 RPM. Measured settled
  one-second rates are 104.404530/56.378446 rad/s (within 0.5 rad/s of targets).
  After disengagement and settling, every step of a one-second observation
  stays below 0.01 rad/s; observed peaks were 1.3861e-8 and zero rad/s.
  Body finiteness/enabled state and no-quarantine checks run every step.
- All four imported release gates pass together: hitch/PTO, pressure/driving,
  exact replay, and 60-second slope hold at both pressure extremes/directions.
  The final actuator test also passes its strengthened per-step stop check.
- Corrected-fixture CPU-step medians were 0.7472–0.7992 ms, p95
  0.7945–1.4431 ms, maxima up to 1.8752 ms on the shared host. No assistant
  workload overlapped timing samples. Controller execution remains outside
  the timer; these are not full-frame, paired speedup or Rapier-parity results.
- Ordinary release binary suites pass 87 tests each with explicit Molla and
  Rapier selection, five ignored as clarified above. All jobs are terminal:
  initial 26722, full gates 88204, final actuation 44188, regressions 94134.
  This is test-only wiring; no production viewer rebuild was needed this turn.

### Next Steps

- Continue exact straight/0.4-turn track acceptance and real drawbar attach,
  tow/turn/detach. Include production attached-service propagation, rather than
  assuming the current isolated service schedule covers that exchange.
- Add Rapier actuator/slope scene evidence separately; the current imported
  fixture explicitly uses Molla. LinkValues-based actuation does not prove
  CLI/network delivery, ownership, visual output or full scheduling parity.
- Continue every remaining lifecycle, pressure calibration/terrain,
  sleep/fleet, performance and GPU requirement. Full goal remains active.

### Relevant Files

- Isolated `bin/gearbox/src/controller/benchmark.rs` — service-aware fixture,
  joint-motion tracking, authored RPM and per-step stop assertions.
- Isolated `bin/gearbox/src/services.rs` — cfg(test) production-system schedule.
- Isolated `docs/MOLLA_BACKEND.md` — corrected scope, measurements and commands.
- `/tmp/gearbox-services-imported.log` — initial actuation probe.
- `/tmp/gearbox-five-controller-gates.log` — four imported gates passing.
- `/tmp/gearbox-five-controller-final.log` — strengthened actuator gate passing.
- `/tmp/gearbox-five-controller-{molla,rapier}-regression.log` — 87 passed each.

## 2026-09-20 — same-scene turns and live field tracks

### Goal

Continue the entire `FULL_COMPACT.md` objective, specifically the straight/
0.4-rad/s turn and visible track acceptance. Full completion remains unproven.
The preceding acknowledgement-only turn was no progress; this turn recovered
the authoritative terminal test log and completed the next safe validation.

### Instructions

- Preserve the original Gearbox checkout and shared renderer/assets; only the
  isolated integration worktree and this Molla checkpoint were changed.
- The user interacted with the old `molla-gram-perf` test. Do not classify its
  stopping/closure as an unattended crash or use it as controlled timing data.
- Engram tools remain unavailable in the tool registry. This tracked checkpoint
  is the session-summary fallback, not a claim that memory-tool calls succeeded.

### Discoveries

- Current same-asset Rapier turns about 1.495 rad in four simulated seconds
  at 2 m/s and +0.4 rad/s, not the historical +0.60 rad stated in the plan.
  Molla gives 1.461–1.470 rad across 0.5/1.8/4 bar. This discrepancy is reported,
  not concealed by changing the plan or tuning to its historical number.
- Imported masses differ by about 2.561 kg (0.052%): 4916.010223 kg Molla vs
  4913.449268 kg Rapier. The fixture now asserts backend-specific snapshots;
  the underlying difference still needs an audit, not a mass-equivalence claim.
- `GEARBOX_TERRAIN=flat` has no field cover/stamping. The initial flat live
  probe drove correctly but showed no tracks. Meadow enables the existing
  grass/harvested-wheat renderer, which visibly stamps tracks on both backends.
- CLI move summaries can lag the final pose. Fresh state subscriptions give
  better end-state evidence; wall-clock live commands are not deterministic
  four-simulated-second comparisons. Startup still reports Vulkan loader and
  Bevy slab allocator diagnostics on both backends.

### Accomplished

- Isolated commit `ba98498 test(physics): verify Kubota turns and tracks`
  adds the real dual-backend gate and production track-input scheduling.
- Twelve fresh scenes cover straight and both ±0.4 turns: Rapier plus Molla
  at three pressures. All five Kubota controllers run. Each step checks finite,
  enabled bodies/no quarantine; each sampled tick has four valid track inputs.
  Molla heading and final speed must match the same-scene Rapier reference
  within 0.1 rad and 0.1 m/s. Straight drift, movement and speed gates also pass.
- Mean straight footprint lengths decrease 0.60443 → 0.49442 → 0.40880 m
  across 0.5/1.8/4 bar. This is regression evidence, not physical calibration.
- Strengthened imported gate passes. Ordinary release suites pass 87 tests
  each on explicit Molla/Rapier, with six ignored: five imported plus Vulkan.
- Live CLI turn and straight commands work on both backends; screenshots were
  visually inspected and show tracks across grass and harvested ground.
  Molla applied/read back 0.5 bar before the turn and 4 bar before the straight
  run on all four wheels. No quarantine/non-finite log diagnostics were found.
  Different trajectories/cameras do not prove calibrated footprint-size parity.
- All assistant-owned viewers were explicitly stopped: `molla-track-gate`,
  `molla-field-track-gate`, `rapier-field-track-gate`. Their process handles
  63793, 6996, 49226 are terminal. Test handles 78953/2619 are also terminal.
  No production physics/render code changed this turn; no viewer rebuild needed.

### Next Steps

- Real rear-drawbar teleport attach, towing through a turn and detach on both
  backends, repeating Molla pressure extremes and checking service propagation.
- Resolve the mass difference; keep historical vs current turn measurements
  explicit. Extend pressure visual/lifecycle/UI and save/reload acceptance.
- Continue every remaining runtime/failure, sleep/island/fleet, calibrated
  grip/braking/coast-down, authored friction, Bekker/MPM, GPU and performance
  requirement. These live debug runs do not establish Rapier performance parity.

### Relevant Files

- Isolated `bin/gearbox/src/controller/benchmark.rs` — backend-selectable fixture,
  real track recording, straight/turn/pressure regression gate.
- Isolated `docs/MOLLA_BACKEND.md` — measurements, reproduction and limitations.
- `/tmp/gearbox-drive-track-final.log` — passing strengthened imported test.
- `/tmp/gearbox-drive-track-{molla,rapier}-regression.log` — 87 passed each.
- `/tmp/molla-track-{before,after}.png` — flat-ground visual probe.
- `/tmp/molla-field-{before,low-after,high-after}.png` — visible Molla tracks.
- `/tmp/rapier-field-{before,after,straight}.png` — visible Rapier tracks.
- `/tmp/{molla-field,rapier-field}-track-gate.log` — live viewer logs;
  `/tmp/molla-field-{low,high}-state.json` — applied pressure readback.

## 2026-09-20 — towing exposed uncoupled Krampe tipping

### Goal

Continue full-plan rear-drawbar attach/tow/detach acceptance, including Molla
pressure extremes. Previous goal turn was progress (dual-backend turn/track
gate plus inspected live screenshots). This turn also made progress, but
uncovered a failing stability requirement; the full objective stays active.

### Instructions

- Original Gearbox, assets, peerbus and agentio were not edited. Only the
  isolated integration test/docs and this checkpoint changed.
- Do not treat host API timeouts as a physics crash: the specific viewer was
  confirmed live by new physics logs and successful screenshots.
- Engram tools remain unavailable; this is the session-summary fallback.

### Discoveries

- The real Krampe tips nose-up under Molla even before being hitched. Live
  high-pressure pitch was -0.50662 rad after spawn, -0.55680 after detach and
  -0.45518 later. Rapier's detached trailer was nearly level (+0.00199 rad).
- New headless reproduction removes tractor, hitch/teleport, rendering and
  network entirely. After 15 simulated seconds, Molla pitches -0.32082,
  -0.38409 and -0.42598 rad at 0.5/1.8/4 bar. Front tyres lose ground contact.
  Final speeds are 0.01368/0.05080/0.10532 m/s. Rapier parks nose-down at
  +0.10997 rad, speed 0.01861 m/s. Root cause is not established.
- Diagnostic trace showed negative pitch before ordinary collider contacts:
  low-pressure Molla at step 120 had pitch -0.06539 with no penalty contacts.
  By step 240 chassis collider 19 contacted ground 48. At final low-pressure
  state rear wheel normal readouts were about 46.7 kN each, fronts zero.
  Investigate load/moment dynamics, not just stand-enable or detach code.
- Imported Krampe has 15 bodies, 14 joints, four tyres, 19 colliders including
  the floor. Body-mass totals are Molla 6728.10030 vs Rapier 6728.86735 kg.
  Probe per-body data localizes this difference to the four rounded wheel
  colliders (140.796294 vs 140.988058 kg per wheel); Molla's rounded-cylinder
  mass calculation explicitly uses the inner cylinder. This does not explain
  the large tipping discrepancy by itself.

### Accomplished

- Isolated commit `36fefd6 test(physics): reproduce parked trailer tipping`
  generalizes the imported fixture to the real Krampe and adds a reproducible
  external-asset support gate. It remains failing; its bounds were not removed
  to produce a green result. |pitch| <0.2 permits the reference parked angle,
  while rejecting the Molla gross tip. Every step also checks finite/enabled
  bodies and absence of quarantine.
- Live rear_drawbar/front_coupler teleport attach, 8 s turn and detach commands
  executed on Molla and Rapier, with inspected screenshots showing towing.
  No rejected-step/quarantine/non-finite diagnostics were found in these logs.
  Physical detach acceptance is NOT passed because of the nose-up fault.
- Low run: tractor at 0.5 bar, trailer at defaults; tractor pressures survived
  unrelated trailer spawn, attach and towing. Post-detach host discovery failed
  despite a live viewer; do not claim verified low-run detached API readback.
- High run: all eight tyres at 4 bar; detach returned an empty attachment list
  and the trailer retained 4 bar on all four tyres. Native standalone telemetry
  before attach independently confirms the tip is not exclusive to detach.
- Ordinary release suites still pass 87 each, now seven ignored (six imported
  assets plus Vulkan). Kubota straight/turn/track gate passes again after the
  fixture refactor. The Krampe acceptance gate fails as recorded above.
- All jobs are terminal, including final gate/regressions 73605 and Kubota
  recheck 25870. Owned viewers `molla-tow-gate` (21775), `rapier-tow-gate`
  (19807), `molla-tow-high-gate` (23897) were explicitly stopped and polled
  terminal. No original/user viewer was stopped. No production code changed.

### Next Steps

1. Fix uncoupled Krampe support dynamics using the new gate. Compare total
   force/moment and articulated response, including the bogie-rooted tree,
   versus actual mass/COM and Rapier. Do not guess or freeze the trailer.
2. Repeat stable detach, both-machine low/high pressure towing, deterministic
   replay and master/slave service propagation after fixing the underlying bug.
3. Continue all remaining full-plan lifecycle, sleep/island/fleet, calibration,
   terrain/Bekker/MPM, GPU and performance work. No parity claim from these runs.

### Relevant Files

- Isolated `bin/gearbox/src/controller/benchmark.rs` — new failing
  `imported_krampe_parking_support`; use `GEARBOX_BENCH_TRAILER` with real USDZ.
- Isolated `docs/MOLLA_BACKEND.md` — reproduction command and honest acceptance.
- `/tmp/gearbox-krampe-parking-final.log` — all four support scenarios, failure.
- `/tmp/gearbox-krampe-parking-probe.log` — earlier per-body/contact diagnostics.
- `/tmp/gearbox-krampe-{molla,rapier}-regression.log` — ordinary suites pass.
- `/tmp/gearbox-krampe-kubota-regression.log` — imported Kubota gate passes.
- `/tmp/{molla-tow-gate,molla-tow-high-gate,rapier-tow-gate}.log` — live logs.
- `/tmp/{molla-tow-low,molla-tow-high,rapier-tow}-{driven,detached}.png` —
  inspected visual evidence; corresponding JSON captures are in `/tmp`.

## 2026-09-20 — scalar joint-stop reaction fix

### Goal

Continue the full plan by fixing the reproduced Krampe support failure. The
preceding turn was progress (a failing imported regression and live evidence).
The complete goal remains active; trailer parking acceptance is still failing.

### Instructions

- Preserve original Gearbox and shared assets/network libraries. All production
  changes here are reusable Molla physics; isolated Gearbox only gets test/docs.
- Engram tools are still unavailable. This checkpoint records the completed
  bug fix and session handoff without claiming a successful memory-tool call.

### Discoveries

- Scalar joint limits only clamped coordinates/velocities after Featherstone
  integration; they did not transmit stop reaction to the other bodies.
  Two 2 kg free bodies on a lower-stopped slider receiving -8 N for 0.1 s
  incorrectly kept total momentum zero instead of -0.8 kg m/s. This is a
  dynamics bug independent of tyre tuning, hitching, rendering or networking.
- Molla with ordinary collider support also had incorrect Krampe motion;
  disabling pressure tyres was not a repair. Temporary diagnostic switches
  were removed after use, leaving the production fixture path unchanged.
- An initial projected Gauss-Seidel reaction correction passed equal-mass
  tests but failed a 1000:1:1000 chain. It was replaced, not retained, by a
  coupled nonnegative active-set solve with complementarity validation.

### Accomplished

- Molla commit `1a1cf89 fix(solver): transmit scalar joint stop reactions`
  computes unilateral lower/upper-stop reactions through the articulation's
  mass matrix, distributes them over all generalized accelerations, and then
  uses the existing Featherstone integration. No second rigid integrator,
  trailer freeze, asset/mass edits or disabled ground contacts were introduced.
- Inactive stops return without mass factorization. Active-set solves handle
  coupled stops and release; non-finite/nonconvergent/complementarity failures
  return solver errors rather than accepting invalid impulse vectors.
- Tests cover lower/upper prismatic and revolute stops, exact load/momentum
  transfer, stop release, multi-stop impact energy dissipation, translated
  origins and 1000:1:1000 masses, plus active-set removal of a redundant active
  stop. All 382 CPU solver tests pass, five ignored, across 88 test binaries.
- All five imported Kubota gates pass with the final active-set solver:
  hitch/PTO, pressure benchmark, exact replay, 60 s slope hold and dual-backend
  straight/turn/tracks. Krampe's gross nose-up tipping is gone at all pressures:
  final headless pitch +0.13173/+0.13663/+0.13286 rad at 0.5/1.8/4 bar.
- The unchanged Krampe gate STILL FAILS at 0.5 bar: speed 0.07754 m/s vs
  required <0.05, mainly vertical motion (+0.07430 m/s) with angular X
  -0.03485 rad/s. Reference/high speeds are 0.04966/0.00296 m/s. Do not loosen
  this bound or call the whole trailer bug/acceptance finished.
- Native isolated `oslo make build` passes. A fresh owned standalone Krampe
  viewer accepted 0.5 bar and streamed live state for 20 s. Final pitch was
  +0.11513 rad, pressure 0.5 on the front-left readout; rear-view screenshot
  inspected. This is not matched side-view deformation or tow/detach proof.
  No rejected-step/quarantine/non-finite diagnostics appeared. The viewer
  `molla-limit-fixed` (34159) was explicitly stopped and polled terminal.
- Imported timing medians in this mixed functional run were 1.018–1.056 ms,
  above target. Build/test/viewer work overlapped portions of the run, so this
  is not a controlled speed comparison. The added mass/stop solve needs a
  dedicated profile and reuse with the existing effective-drive factorization.

### Next Steps

1. Resolve residual low-pressure Krampe vertical rocking without hiding it by
   relaxing the acceptance limit. Inspect stop/contact/suspension dynamics and
   damping using the exact existing 15 s regression.
2. Profile/cache the added response solve, then repeat full-machine under-1-ms
   timing without overlapping work and run live tow/turn/detach again.
3. Extend coupled reaction semantics to Ball/D6 angular/linear limits and GPU
   paths; this fix covers CPU scalar revolute/prismatic stops only. Existing
   post-projection velocity-cap and general mutation semantics need further audit.
4. Continue every remaining full-plan lifecycle, terrain/soil, calibration,
   sleep/islands/fleet and acceptance requirement. Full completion is unproven.

### Relevant Files

- `crates/molla-solvers/src/joint_limits.rs` — scalar reaction active-set solver.
- `crates/molla-solvers/src/featherstone.rs` — pre-integration reaction phase.
- `crates/molla-solvers/tests/runtime_limit_reactions.rs` — momentum/release,
  coupled impact and high-mass-ratio regressions.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — velocity-vector diagnostics.
- `/tmp/molla-limit-reaction-before.log` — failing minimal momentum reproduction.
- `/tmp/molla-limit-ratio.log` — rejected first PGS implementation's failure.
- `/tmp/molla-limit-active-full.log` — final 382 CPU tests passing.
- `/tmp/gearbox-limit-active-imported.log` — five Kubota pass, Krampe speed fails.
- `/tmp/gearbox-limit-active-build.log` — native debug build passed.
- `/tmp/molla-limit-fixed{.log,.png,-state.jsonl}` — live standalone validation.

## Shared drive/stop mass cache checkpoint (2026-09-20)

### Goal

Continue the entire `FULL_COMPACT.md`; reduce duplicated CPU work introduced
by scalar stop reactions without changing the validated mechanics.

### Instructions

- Original Gearbox remains untouched. This turn changes only Molla; integration
  validation uses the existing isolated `gearbox-physics` checkout.
- User confirmed interacting with the earlier `molla-gram-perf` viewer. That
  run is not unattended stability/performance evidence, and the confirmation
  does not establish who or what closed the window.

### Discoveries

- Drive inertia and scalar limit reactions use the same post-mimic/FK state.
  The state remains immutable through every articulation's reaction phase;
  new coordinates are written only after all these phases finish.
- Motor inertia eliminates near-zero diagonal axes and has a different
  singularity policy from impulse response. Sharing those reduced factors
  indiscriminately changes dynamics. Only complete, well-conditioned Cholesky
  factors may be directly retained; other cases reuse the full matrix and
  lazily construct the original impulse pseudoinverse.
- User-owned original Gearbox processes were present on the host and were
  not disturbed. Repeated headless comparisons remove this agent's overlapping
  work, but do not constitute a dedicated/isolated-host latency guarantee.

### Accomplished

- Commit `02054f0`: immutable-state-scoped lazy `JointMass` cache shares mass
  matrices between drive inertia and scalar stops. Full-rank Cholesky retention
  preserves the impulse solver's exact normalized pivot cutoff. No across-step
  cache, changed gains, new integrator, or relaxed acceptance threshold.
- Tests verify matrix/factor reuse, independent articulations, cache refresh
  after removal, retained versus fresh responses across scales, singular
  fallback, and invalid/cutoff rejection. Drive-inertia tests compare complete
  shared response columns exactly with independent impulse solves.
- Final CPU solver lane: **384 passed, 0 failed, 5 ignored**, 88 test binaries.
  Explicit Molla and Rapier isolated binary lanes each: **87 passed, 7 ignored**.
- Sequential retained release binaries, A/B/B/A, real Kubota at 120 Hz with
  pressure tyres, parked/driving at 0.5/1.8/4 bar, 600 measured steps per case:
  average of 12 medians per version **1.020805 -> 0.957713 ms**, a **6.18%**
  reduction. Before median range 1.011678–1.027910 ms; after
  0.945239–0.977099 ms. All 12 reported physical-state samples match exactly
  across all four runs. This samples output state; it is not a full-state
  before/after replay comparison.
- After-run p95 reached 1.818128 ms and max 4.560390 ms. Under-1-ms tail latency,
  fleet performance and Rapier speed parity are **not achieved/proven**.
- Krampe low-pressure failure reproduces unchanged: 0.077537 m/s after 15 s,
  mostly vertical (+0.074302 m/s). This optimization is not its stability fix.
- Imported functional suite finishes **5 passed, 1 failed**: all five Kubota
  gates pass (controllers, benchmark mechanics, replay, slope parking and
  dual-backend drive/turn/track inputs); only Krampe fails. All launched test
  handles are terminal. No viewer was launched or stopped in this turn.

### Next Steps

1. Resolve the unchanged low-pressure trailer rocking using measured
   stop/contact/tyre dynamics; retain the original speed acceptance bound.
2. Profile remaining CPU work and compare against Rapier under matched load;
   no claim of full performance acceptance from sub-ms medians.
3. Repeat stable live tow/turn/detach and continue the remaining full-plan
   lifecycle, calibration, generic-joint/GPU, terrain and sleep/islands gates.

### Relevant Files

- `crates/molla-solvers/src/joint_mass.rs` — lazy immutable-state mass cache.
- `crates/molla-solvers/src/featherstone.rs` — shared drive/stop cache scope.
- `crates/molla-solvers/src/joint_limits.rs` — cached stop response lookup.
- `crates/molla-solvers/src/rigid_impulse.rs` — checked Cholesky retention.
- `/tmp/molla-shared-mass-final-full.log` — final CPU solver results.
- `/tmp/molla-shared-mass-{molla,rapier}-binary.log` — isolated binary results.
- `/tmp/molla-shared-mass-abba-{1-before,2-after,3-after,4-before}.log` —
  paired timing and physical-state samples; no concurrent agent build/test/viewer.
- `/tmp/gearbox-{before,after}-shared-mass` — retained release test executables,
  respectively baseline `aa75128` and the cache implementation `02054f0`.
- `/tmp/molla-shared-mass-imported.log` — imported functional validation;
  overlapped CPU tests/builds, so its timings are not performance evidence.

## Convex support continuity checkpoint (2026-09-20)

### Goal

Continue the entire `FULL_COMPACT.md`, resolving the remaining Krampe support
failure before repeating live towing acceptance. Previous goal turn made
verified performance progress; this turn makes collision/stability progress.

### Instructions

- Only Molla and the isolated `gearbox-physics` worktree changed. Original
  Gearbox and shared assets remain untouched. All owned test viewers were
  explicitly stopped; no user-owned process was stopped.
- Engram tools remain unavailable; this tracked checkpoint records the session.

### Discoveries

- Low-pressure Krampe rocking was traced to intermittent missing convex-hull
  versus ground contacts, not tyre damping. The chassis drawbar penetrated
  the slab by about 0.13 m while its contact force switched between zero and
  roughly 13 kN. A small geometry regression reproduced a GJK false disjoint
  result (reported gap 2.06 m) at true penetration 0.121835 m.
- The distance-query GJK reused a boolean tetrahedron reducer selecting the
  first exterior face. That can increase the closest distance, triggering
  the distance algorithm's non-improvement termination. The reducer now
  evaluates exterior faces with explicit opposite-vertex orientation and
  retains the closest feature.
- Diagnostic per-contact `normal_impulse / dt` is that sample's contribution
  to the full-step mean force, not its instantaneous substep force. The
  optional fixture trace labels it `step_mean_force` and includes body/shape
  identity, kinetic/gravitational energy, contact points and wheel support.

### Accomplished

- Molla commit `c95b4c3` fixes CPU GJK tetrahedron distance reduction. Tests
  cover all 24 vertex orders, 100 near-identical wide-slab poses including
  reversed pairs and separation, plus 1,000 deterministic rotated/translated
  hull-box pairs in both orders against independent box SAT.
- Geometry CPU lane: **190 passed**, 24 test binaries. CPU solver lane:
  **384 passed, 5 ignored**, 88 test binaries. Both explicit-backend isolated
  binary lanes: **87 passed, 7 ignored** each. Native `oslo make build` passes.
- All **six imported-machine gates pass**, including unchanged Krampe speed
  threshold. After 15 s at 0.5/1.8/4 bar, Krampe speeds are
  **0.0000342/0.0000548/0.0000145 m/s**, versus 0.077537/0.049655/0.002957
  before the fix. No damping/stiffness tuning, asset edits or relaxed bounds.
- Fresh live owned instances `molla-gjk-tow-low`, `molla-gjk-tow-high` and
  `rapier-gjk-tow` completed attach with teleport, `--forward 2 --turn 0.4
  --for 8s`, detach and empty attachment-list readback. Fresh post-detach
  trailer telemetry has no `attached_to`; final trailer pitch was
  -0.000223/+0.003044/+0.013473 rad respectively. Four trailer pressure
  readouts retain 0.5/4 bar; the high-run tractor's four readouts retain 4 bar.
- All three detached screenshots inspected. These are rear views, not
  calibrated side-view tyre-deformation or all-wheel-contact proof. No rejected
  step, quarantine or non-finite diagnostics appeared in these viewer logs;
  existing renderer startup diagnostics are not claimed fixed.
- Sequential A/B/B/A release-binary Kubota comparison after stopping owned
  viewers: average of 12 medians **0.956771 -> 0.958640 ms** (about +0.2%,
  small compared with run variation). Reported physical-state samples match
  across all four runs. New medians range 0.943500–0.970420 ms; p95 reaches
  1.739881 ms and max 4.301520 ms. No under-1-ms tail or Rapier-speed-parity claim.
- All launched build/test/CLI/viewer handles are terminal. Core full-plan
  completion remains unproven; CPU collision correctness is not GPU parity.

### Next Steps

1. Continue CPU profiling and matched Rapier/fleet performance gates; current
   per-machine medians do not satisfy the full performance requirement.
2. Audit box-box mid-plane contact witnesses in `emit_face_manifold`: it
   currently subtracts positive penetration along the reference normal.
   This was noticed statically; add a witness-position regression first.
3. Continue full pressure lifecycle/save-reload/UI acceptance, calibration,
   generic-joint/GPU semantics, per-cell terrain and sleep/islands/soil work.
   Towing deterministic replay and calibrated pressure visual proof remain.

### Relevant Files

- `crates/molla-geometry/src/gjk.rs` — closest exterior tetrahedron feature.
- `crates/molla-geometry/src/narrow_phase.rs` — independent contact regressions.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — optional
  `GEARBOX_BENCH_TRACE_SUPPORT=1` trace for the Krampe fixture.
- `/tmp/molla-hull-support-before.log`, `/tmp/molla-support-trace-before.log`
  — failing minimal contact case and original support discontinuity.
- `/tmp/molla-gjk-expanded.log`, `/tmp/molla-gjk-full-solvers.log`,
  `/tmp/molla-gjk-all-imported.log`, `/tmp/molla-gjk-{molla,rapier}-binary.log`
  — final regression results.
- `/tmp/molla-support-fixed-gjk.log` — fixed headless support trace.
- `/tmp/molla-gjk-{low,high}-*-state.json`, `/tmp/rapier-gjk-*-state.json`
  — live attached/detached/settled state; `*-detached.png` screenshots inspected.
- `/tmp/molla-gjk-tow-{low,high}.log`, `/tmp/rapier-gjk-tow.log` — viewer logs.
- `/tmp/molla-gjk-high-cli.log`, `/tmp/rapier-gjk-cli.log` — scripted command
  evidence; low-run command evidence is in this turn's tool transcript.
- `/tmp/molla-gjk-abba-{1-before,2-after,3-after,4-before}.log` — sequential
  timing comparison against the previous shared-mass-cache implementation.

## Mid-plane contact and paired-performance checkpoint (2026-09-20)

### Goal

Continue the full plan: finish the contact-witness regression identified in
the previous turn and measure the actual Rapier performance gap. The previous
goal turn made verified collision and live-acceptance progress.

### Instructions

- Original Gearbox remains untouched; only Molla and isolated test/docs code
  changed. No viewer was launched this turn. No production solver setting was
  changed. Engram tools remain unavailable; this checkpoint persists the work.

### Discoveries

- Box face clipping projected penetrating incident points in the wrong
  direction when computing their mid-plane contact. For faces at 0.9 and
  1.0 m, it returned 0.85 rather than 0.95 m. Both the normal path and fallback
  had the sign error. Commit `9d91efd` corrects both without changing depths.
- The real paired CPU comparison is about **4.1x**, not parity: default
  Molla medians 0.941989–0.948653 ms versus Rapier 0.225601–0.232383 ms.
  All 26 bodies were awake in both backends, including parked samples.
- Existing `imported_kubota_solver_trace` already supports headless profiling;
  use the isolated binary test with `--features gearbox-sim/profile`,
  `GEARBOX_BENCH_ASSET=.../kubota_tractor.usdz` and `GEARBOX_TRACE=...json`.
  Host `perf_event_paranoid=4`; no kernel configuration was changed.
- Profile of 120 driving steps: 960 substeps, 39 reduced DOFs and 22 loop rows;
  960/960 loop response samples use Cholesky, not the spectral fallback.
  Instrumented inclusive cost per frame: Featherstone 0.524 ms, loop forces
  0.360 ms, tyre forces 0.157 ms, collision 0.076 ms. Whole rigid step 1.168 ms.
  These include tracing overhead and are not uninstrumented latency evidence.
- Simply reducing substeps does not preserve current behavior. Two substeps
  cost about 0.25 ms but fail trailer support, hitch tracking, slope parking
  and turning (3 imported tests pass, 4 fail). Low/reference-pressure trailer
  speed reaches 0.245/0.275 m/s; slope drift reaches 0.543 m in 60 s.
- Four substeps cost about 0.48 ms but still fail slope parking and turning
  (5 pass, 2 fail): downhill low-pressure speed peaks at 0.002712 m/s against
  the unchanged 0.001 limit. Both configurations are rejected for production.
  Default eight-substep behavior remains in place. Lower-step tests expose
  integration/gain sensitivity, not a validated shortcut to Rapier parity.

### Accomplished

- Contact witness regression covers all axes, both signs, reversed pairs,
  global rotations/translations and tilted box/slab reference faces. It first
  failed at the measured 0.85 m midpoint, then passed with the sign correction.
- Combined CPU geometry/solver lane: **576 passed, 5 ignored**, across 112
  test binaries (192 geometry, 384 solver tests).
- Added ignored `imported_kubota_backend_timing`: fresh fixtures in R/M/M/R
  order, identical 120 Hz settings, parked and driving, 600 warmup steps and
  1,200 timed steps per case. Reports median/p95/p99/max, sleeping count,
  realized speed and complete solver settings. This measures performance;
  passing the test's mechanical checks does not certify timing acceptance.
- Added test-only `GEARBOX_BENCH_MOLLA_INTERNAL_ITERATIONS` override (1–64)
  for reproducible substep experiments. It affects only Molla fixtures and
  logs the effective settings; production defaults and Rapier are unchanged.
- Final default-settings rerun: **all seven imported tests pass**. Explicit
  Molla and Rapier ordinary binary lanes each pass **87 tests, eight ignored**.
  All launched build/test handles are terminal. Full-plan performance and
  other outstanding acceptance requirements remain unproven.

### Next Steps

1. Optimize measured articulated/loop/tyre work at validated settings, using
   the paired benchmark. Do not silently adopt the failed 2/4-substep settings.
2. Profile finer ABA and Jacobian/response costs; test any kinematics/mass reuse
   against reversed joints, mimic updates and immutable-state boundaries.
3. Continue full-plan pressure lifecycle/UI/save-reload, calibration and
   timestep convergence, sleep/islands/fleet, generic/GPU joints and soil gates.

### Relevant Files

- `crates/molla-geometry/src/narrow_phase.rs` — corrected mid-plane witnesses.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — paired timing and
  explicit test-only iteration override; existing optional solver profiler.
- `/tmp/molla-box-witness-before.log`, `/tmp/molla-box-witness-full.log` —
  failing reproduction and final combined CPU regression lane.
- `/tmp/molla-paired-backend-baseline.log` — default uninstrumented timings.
- `/tmp/molla-current-cpu-trace.json`, `/tmp/molla-current-cpu-profile.tsv` —
  instrumented spans and per-frame inclusive/exclusive summary.
- `/tmp/molla-two-substep-experiment.log`, `/tmp/molla-four-substep-experiment.log`
  — rejected lower-fidelity experiments; unchanged failure thresholds.
- `/tmp/molla-midpoint-default-imported.log` — final default-settings rerun.
- `/tmp/molla-midpoint-{molla,rapier}-binary.log` — ordinary regression lanes.

## Rejected kinematics cache and profiling checkpoint (2026-09-20)

### Goal

Continue `FULL_COMPACT.md` performance work without reducing the validated
eight-substep fidelity or changing the original Gearbox checkout.

### Instructions

- The user interacted with `molla-gram-perf`; that run is not an unattended
  failure reproduction. Its window closure remains unexplained.
- Keep reusable calculations in Molla. Only the isolated Gearbox worktree is
  used for the imported-machine tests; no original Gearbox changes or merges.

### Discoveries

- Body COM Jacobians already had a per-state cache, but every first body query
  repeated its ancestors' frame composition, normalized axes and lock checks.
- `Model::joint_ancestor` is documented as informational. The candidate used
  it only after checking the articulation bounds and matching parent body;
  missing/stale hints fall back to the same child search as the public oracle.
- Joint-origin columns can be shared without a world-origin moment-arm
  subtraction: each body query first subtracts its target from the cached
  joint position, then applies the angular cross product and traversal sign.
- A separate unrelated host build overlapped the first imported regression
  run. That run's timings are not used as optimization evidence. No unrelated
  process was stopped or modified.

### Accomplished

- Experimented with lazy articulation-local joint-motion caches owned by the existing
  immutable `ArticulatedResponse` state scope. Storage scales with requested
  articulations' joints, not fleet body count times articulation count.
- Kept the public uncached Jacobian algorithm as an independent comparison.
  Four new tests cover all eight joint kinds, branching and multiple trees,
  reversed traversal, locked/disabled/fixed joints, COM and body-origin points,
  billion-meter translations, absent/stale ancestor hints, empty/implicit
  articulations, and runtime reroot/split/removal/pose-change reconstruction.
- Candidate CPU solver lane: **388 passed, five ignored**, across 88 test
  binaries. The initial cache variant passed all seven imported-machine tests;
  the final variant passed four paired timing tests and the independent oracle
  tests. These counts describe the experiment, not additional shipped tests.
- Completed uninstrumented A/B/B/A with retained before/after binaries after
  the concurrent builds and owned regression processes finished. Each run uses
  the identical Rapier/Molla/Molla/Rapier fixture sequence at default settings.
  Run mean Molla medians (ms): A1 **0.9498975**, B1 **0.94454075**,
  B2 **0.9456455**, A2 **0.94497875**. Aggregate A **0.947438125** versus B
  **0.945093125**, only **0.2475%** lower. A2 itself beats B2; this is not a
  convincing performance gain. No Rapier-parity or tail-latency claim follows.
- Rejected and removed the cache and its experimental tests. Saved the full
  candidate patch in `/tmp/molla-kinematics-rejected.patch` for reproducibility.
- Added finer profiling spans for response Jacobian construction, mass
  factorization, wrench projection, and Gram whitening/dot products. Solver
  settings, mathematical operations and acceptance thresholds are unchanged.
- Finer 120-step profile passed the real Kubota mechanics checks. Inclusive
  timings (ms/frame, tracing enabled): wrench projection **0.16105**, including
  Jacobians **0.06117** (10 newly cached bodies/substep); mass factorization
  **0.05545**, whitening **0.06158**, Gram products **0.01891**. Projection
  excluding its Jacobian children is **0.09988**. The total traced step is
  **1.24663 ms**; extra span overhead means these are diagnostic attribution,
  not uninstrumented latency or a performance acceptance result.

### Next Steps

1. Use the finer profile to select the next optimization, then repeat the
   uninstrumented paired timing and unchanged mechanical acceptance gates.
   Inspect dense wrench projection's indexed row loads and bounds checks
   before expanding another state/kinematics cache.
2. Continue profiling at unchanged settings; Rapier timing parity remains
   unproven. Finish the remaining full-plan sleep/islands/fleet, generic/GPU
   joints, pressure lifecycle/calibration, and terrain/soil acceptance work.

### Relevant Files

- `crates/molla-solvers/src/articulated_response.rs` — Jacobian/factor profiling.
- `crates/molla-solvers/src/loop_forces.rs` — wrench projection profiling.
- `crates/molla-solvers/src/rigid_impulse.rs` — whitening/Gram profiling.
- `/tmp/molla-kinematics-rejected.patch` — rejected implementation and four tests.
- `/tmp/molla-kinematics-full-final.log` — candidate CPU solver regression lane.
- `/tmp/molla-kinematics-imported.log` — first imported regression run.
- `/tmp/gearbox-before-kinematics-cache`, `/tmp/gearbox-after-kinematics-final`
  — retained uninstrumented binaries for the same-host comparison.
- `/tmp/molla-kinematics-abba-{1,2,3,4}.log` — paired timing sequence.
- `/tmp/molla-response-detail-trace.json` — finer response profile.
- `/tmp/molla-response-detail-profile.tsv` — per-frame span totals.
- `/tmp/molla-response-detail-profile.log` — passing profile fixture output.

## Contiguous wrench projection checkpoint (2026-09-20)

### Goal

Reduce the measured loop-response projection cost while preserving the full
`FULL_COMPACT.md` objective and the validated integration settings.

### Discoveries

- Splitting a row-major Jacobian into six contiguous row slices before the
  wrench dot products gives a small consistent full-machine timing reduction.
  This does not require another kinematics cache, change multiplication/addition
  order, or change tyre/contact/drive parameters.
- Both A/B/B/A candidate runs have lower mean medians than both baselines.
  The Rapier control mean is effectively unchanged: 0.22512525 ms before versus
  0.225219375 ms after. The result is still about **4.12 times Rapier**, not
  performance parity. Some p95 samples and tails remain above 1 ms.

### Accomplished

- Replaced repeated flattened-row indexing in loop wrench projection with
  six row slices and an output-slice loop. Added an exact-arithmetic comparison
  covering 0/1/6/39/78 DOFs, zero/linear/angular/mixed wrenches, and accumulation
  into nonzero prior values.
- CPU solver regression lane: **385 passed, five ignored**, 88 test binaries.
- Uninstrumented A/B/B/A at unchanged settings: baseline mean of eight Molla
  medians **0.940904625 ms**, candidate **0.928933125 ms**, a **1.2723%** reduction.
  Both states (parked/driving) are sampled twice in each executable run, with
  Rapier/Molla/Molla/Rapier fixture order. No owned build/test ran concurrently
  with these timings. All four paired timing tests passed; the logged pressure,
  load, deflection and position readouts match exactly before and after.
- Final imported suite: **seven passed**, covering pressure sweeps, exact replay,
  Krampe parking, Kubota controllers, slopes, straight/turning/tracks and paired
  timing. Explicit Molla and Rapier ordinary binary lanes each pass **87 tests,
  eight ignored**. No regression thresholds were changed.
- Committed the projection change as `ce107f4`. All owned build/test handles are terminal;
  no viewers were launched. This checkpoint does not establish live UI,
  rendered tyre/track, GPU, fleet, soil or full-plan performance acceptance.

### Next Steps

1. Profile the larger ABA, drive-inertia and wheel-force costs. Keep unchanged
   thresholds and eight substeps until an alternative passes all stability gates.
2. Continue all remaining full-plan lifecycle/calibration, sleeping/islands/fleet,
   generic/GPU joint, rendered footprint and terrain/soil acceptance work.

### Relevant Files

- `crates/molla-solvers/src/loop_forces.rs` — contiguous projection and regression.
- `/tmp/molla-projection-full.log` — CPU solver validation.
- `/tmp/gearbox-before-projection`, `/tmp/gearbox-after-projection` — retained
  uninstrumented before/after binaries.
- `/tmp/molla-projection-abba-{1,2,3,4}.log` — same-host timing evidence.
- `/tmp/molla-projection-imported.log` — final imported-machine regression.
- `/tmp/molla-projection-{molla,rapier}-binary.log` — ordinary backend regressions.

## Drive inverse-diagonal suffix checkpoint (2026-09-20)

### Goal

Continue the full replacement plan's performance work at unchanged integration
settings, targeting effective drive-inertia calculations.

### Instructions

- Preserve original Gearbox and user-owned processes. A user-owned original
  Gearbox viewer was running during this checkpoint and was left untouched.
- Keep the numerical rank policy, motor gains, eight-substep integration and
  mechanical acceptance thresholds unchanged.

### Discoveries

- `effective_drive_inertias` solved a full triangular system for each unit
  right-hand side, allocating both the unit vector and solve result each time.
  For finite lower-triangular L, `L^-1 e_i` is identically zero before row i.
- Computing only that suffix preserves the packed solver's arithmetic order
  and `1e-30` pivot cutoff. One articulation-scoped workspace can serve every
  requested motor without clearing/reading its unused prefix.
- The shortcut checks factor finiteness once; a non-finite successful factor
  retains the old full-substitution path. Singular factors retain the existing
  spectral fallback. No mass-matrix rank thresholds or cache lifetimes change.
- A singular regression initially normalized its rank-one block by two, which
  produced a tiny positive floating-point Cholesky pivot rather than the
  intended failure. The corrected fixture uses exactly representable unit
  pivots and asserts that factorization actually fails before checking fallback.
- Separately found a fleet-scaling issue to address next: `cpu_step` allocates
  world-sized `q_dd` and `a_body` arrays inside its articulation loop. Their
  zero-initialization scales with articulation count times world size. The
  consumers currently index disjoint articulation ranges, including scalar
  stop reactions, so substep-scoped scratch is a candidate; it still needs
  independent-island equivalence tests and a fleet benchmark before adoption.

### Accomplished

- Implemented a finite-factor unit-column suffix solve and reusable workspace
  in `effective_drive_inertias`; existing fallback paths remain intact.
- Added exact comparisons against the full substitution for every column of
  1/2/6/39/78-dimensional factors, including zero/tiny pivots and a NaN-filled
  scratch buffer. Added an explicitly singular free-body motor regression.
- A/B/B/A run mean Molla medians (ms): A1 **0.94242025**, B1 **0.93356725**,
  B2 **0.93580975**, A2 **0.941804**. Aggregate A **0.942112125**, B
  **0.9346885**, about **0.79%** lower. Rapier controls were **0.226453** and
  **0.22590025** respectively. This is a small observed improvement with an
  active background viewer, not a quiet-host latency or Rapier-parity result.
- All four paired timing fixtures passed, at unchanged settings. Logged
  pressure/load/deflection/position readouts match before and after.
- Final corrected CPU lane: **387 passed, five ignored**, 88 test binaries.
  Final imported suite: **seven passed**, with the original slope/controller/
  pressure/replay/turning/trailer thresholds. Explicit Molla and Rapier ordinary
  binary lanes each pass **87 tests, eight ignored**.
- All owned build/test handles are terminal. No viewers were launched or
  stopped; the user-owned original Gearbox viewer remains outside this work.
  Full-plan performance, fleet, GPU and visual acceptance remain incomplete.

### Next Steps

1. Validate substep-scoped acceleration scratch against independent islands,
   then measure many-articulation scaling and real-machine regressions.
2. Continue the original full-plan stability, timestep convergence, lifecycle,
   calibration, sleeping, generic/GPU joint, terrain/soil and visual gates.

### Relevant Files

- `crates/molla-solvers/src/featherstone.rs` — suffix solve, workspace and tests.
- `/tmp/molla-drive-suffix-full.log` — initial full lane with fixture failure.
- `/tmp/molla-drive-suffix-singular.log` — corrected singular fixture passes.
- `/tmp/molla-drive-suffix-full-final.log` — final CPU lane.
- `/tmp/gearbox-{before,after}-drive-suffix` — retained timing binaries.
- `/tmp/molla-drive-suffix-abba-{1,2,3,4}.log` — timing evidence.
- `/tmp/molla-drive-suffix-imported.log` — final real-machine regression.
- `/tmp/molla-drive-suffix-{molla,rapier}-binary.log` — ordinary backend lanes.

## Articulation acceleration scratch checkpoint (2026-09-20)

### Goal

Remove the identified many-articulation allocation cost without changing the
full replacement plan, solver equations or numerical acceptance thresholds.

### Instructions

- Keep original Gearbox and unrelated viewers untouched.
- Distinguish solver-only scaling measurements from real imported-machine,
  rendered fleet, sleeping/islands and Rapier-parity acceptance.

### Discoveries

- `cpu_step` allocated `joint_dof_total` acceleration scalars and `joint_count`
  spatial accelerations for every articulation. Both consumers use disjoint
  joint/DOF ranges; parent acceleration lookup stays inside the current tree,
  and scalar-stop reactions only update that articulation's DOF slice.
- Hoisting these buffers into the substep scope reduces their allocation count
  from two per articulation to two per substep. It removes this particular
  articulation-count-times-world-size initialization cost, not every possible
  non-linear cost in the solver or runtime scene.
- Follow-up source audit: CRBA already sizes its scratch to the selected
  articulation. However, loop closure projection allocates one row for every
  world-wide closure row inside each articulation, even when most projections
  are zero. `MassFactor::add_gram` whitens and multiplies every such row, then
  `bounded_solve` factors the full global row system. Independent machines can
  therefore incur avoidable superlinear closure work. Partitioning must preserve
  rows that couple multiple articulations, bounded-solve behavior and rank policy;
  no partitioning change is implemented or validated at this checkpoint.

### Accomplished

- Moved the two acceleration scratch allocations outside the articulation loop.
  No mathematical operations, parameters, substeps or thresholds changed.
- Added eight-articulation versus eight separate-simulation comparison over
  120 steps and both rigorous/sequential multi-DOF paths. Joint coordinates,
  joint velocities, body poses and body velocities match exactly every step.
  Includes fixed roots, disabled and locked hinges, distinct external wrenches,
  feedback motors and both limit directions; explicit assertions verify stops
  were reached and unlocked dynamic bodies moved.
- Added an ignored CPU scaling fixture with 1/32/128/512 two-body trees. It
  times one solver substep from a fixed input state, with 20 warmups and 80
  samples, no contacts/rendering. This is not an imported tractor fleet.
- Four A/B/B/A cycles (eight runs of each retained binary) give these mean
  medians in ms: 32 articulations **0.099050 -> 0.086140375** (13.03% lower),
  128 **0.39276575 -> 0.35325925** (10.06%), 512 **2.222096625 -> 1.446364875**
  (34.91%). One articulation is effectively unchanged at **0.002414375 ->
  0.002426125** ms. The same independent-island correctness test passes before
  and after the production change. No owned build/test overlapped these timings;
  an unrelated original Gearbox viewer remained running.
- CPU solver lane: **388 passed, six ignored**, across 89 test binaries.
- Real Kubota A/B/B/A: mean Molla medians **0.931703375 -> 0.918061375 ms**,
  while Rapier controls also improve **0.23251375 -> 0.229295125 ms**. Relative
  to Rapier, the single-machine result is effectively unchanged at about **4x**
  its step time; do not attribute the common timing shift to this allocation fix.
  All four paired tests pass and their logged mechanical states match exactly.
- Final imported-machine suite: **seven passed** at unchanged thresholds.
  Explicit Molla and Rapier ordinary binary lanes each pass **87 tests, eight
  ignored**. All owned build/test handles are terminal; no viewers were launched
  or stopped. The original full-plan completion criteria remain unmet.

### Next Steps

1. Add multi-articulation loop-closure correctness/scaling fixtures, then reduce
   work on unrelated zero rows and investigate connected closure components.
   Preserve actual cross-articulation couplings; do not claim full islands or
   sleeping from this narrower constraint optimization. Continue profiling
   remaining single-machine ABA/drive/tyre costs.
2. Continue full-plan sleep/wake/islands, imported fleet, lifecycle/calibration,
   generic/GPU joint, terrain/soil and rendered acceptance. No full-plan or
   Rapier-comparable performance claim follows from the synthetic scaling result.

### Relevant Files

- `crates/molla-solvers/src/featherstone.rs` — substep-scoped acceleration buffers.
- `crates/molla-solvers/tests/featherstone_islands.rs` — exact independent-tree
  comparisons and ignored scaling fixture.
- `/tmp/molla-islands-{before,after}` — retained solver-test binaries.
- `/tmp/molla-islands-abba-{1..16}.log` — repeated scaling measurements.
- `/tmp/molla-islands-full.log` — CPU solver regression output.
- `/tmp/gearbox-after-drive-suffix`, `/tmp/gearbox-after-island-scratch` —
  retained real-machine timing binaries.
- `/tmp/molla-islands-machine-abba-{1,2,3,4}.log` — real Kubota paired timing.
- `/tmp/molla-islands-imported.log` — final imported-machine checks.
- `/tmp/molla-islands-{molla,rapier}-binary.log` — ordinary backend checks.

## Rejected compact closure response checkpoint (2026-09-20)

### Goal

Continue `FULL_COMPACT.md` by reducing unrelated closure-row work, without
changing the global bounded constraint solve or numerical rank thresholds.

### Instructions

- The user confirmed interacting with the old `molla-gram-perf` window. Its
  stopping is not unattended-failure evidence; the later closure is unexplained.
- Keep original Gearbox untouched. Only Molla and the isolated physics worktree
  are in scope; no viewer was launched or stopped in this checkpoint.
- Engram tools remain unavailable. This tracked checkpoint preserves the handoff.

### Discoveries

- Each articulation previously whitened every global closure row, including
  rows whose endpoints belong exclusively to other articulations. Topological
  row selection removes this work without a floating-point sparsity threshold.
- The first compact implementation reduced the 32-group synthetic response
  calculation from **4.657699 to 0.132739 ms** (97.15%), but the entire closure
  force pass only fell from **33.8073275 to 28.935823 ms** (14.41%). The unchanged
  global bounded solve dominates. This is not whole-physics or imported-fleet
  timing. One group regressed about 5%, approximately 0.00037 ms per force pass.
- The initial real Kubota A/B/B/A regressed **0.902220375 to 0.924508125 ms**
  (2.47%); Rapier controls stayed **0.225081625 to 0.225192 ms**. A fast path
  limited to models with one dynamic articulation still regressed 1.585%.
  Neither result supports a single-tractor performance improvement.
- The final candidate retained dense projection whenever all closure rows
  participated in the current articulation. Its real Kubota A/B/B/A still
  regressed **0.908614750 to 0.927983750 ms** (2.132%), while Rapier controls
  moved **0.225140500 to 0.227534750 ms** (1.063% slower). Relative to Rapier,
  this did not recover the original single-machine performance either.
- Rejected all three production variants rather than claiming the synthetic
  scaling win as a real-machine improvement. The original solver source is
  restored exactly to HEAD. The experimental patch includes the reusable test
  fixture, so future work can reproduce this attempt without reimplementing it.

### Accomplished

- Extracted closure response construction and added exact comparisons against
  an independent dense reference. Cases include independent groups, bridging
  constraints, fixed/world endpoints, unrelated dynamic bodies, zero DOFs,
  singular inertia, and mass/inertia scales from 1e-9 through 1e9.
- Added a release-only ignored synthetic scaling fixture for 1/4/16/32 groups,
  separately measuring response construction and the whole closure force pass.
- Initial candidate: CPU **390 passed, seven ignored** across 89 binaries;
  imported machines **seven passed**. Single-dynamic-articulation revision:
  CPU **390 passed, seven ignored**; ordinary Molla/Rapier lanes **87 passed,
  eight ignored each**. Logged real-machine states match before/after exactly.
- Final dense-participation candidate: CPU **390 passed, seven ignored**, and
  ordinary Molla/Rapier lanes **87 passed, eight ignored each**. All four paired
  timing tests passed; logged mechanical states matched exactly. No owned
  build/test overlapped any A/B/B/A timing; unrelated user processes were left
  alone. These are not quiet-host latency acceptance results.
- Removed the rejected production change and its test-only module from the
  checkout, saving both in `/tmp/molla-compact-closures-rejected.patch`.
  The saved patch passes `git apply --check`. Restored-source CPU checks pass
  **388 tests, six ignored** across 89 binaries; ordinary Molla/Rapier lanes
  each pass **87 tests, eight ignored**; the final imported-machine suite
  passes **all seven tests**. All owned build/test handles are terminal.
  No runtime improvement ships from this checkpoint, and the full objective
  remains unfinished. Single-machine CPU step time is still about 4x Rapier.

### Next Steps

1. Rework row selection/allocation and evaluate response extraction/code
   generation before retrying compact response work;
   preserve the dense single-machine path and require paired timing evidence.
2. Investigate disconnected components of the global bounded solve. Per-block
   normalization/rank cutoffs can change tiny or singular block behavior; retain
   global numerical semantics and test actual cross-articulation couplings.
3. Continue original full-plan imported-fleet, sleeping, lifecycle, tyre/terrain
   calibration, generic/GPU joint and rendered acceptance work.

### Relevant Files

- `crates/molla-solvers/src/loop_forces.rs` — restored unchanged to HEAD.
- `/tmp/molla-compact-closures-rejected.patch` — rejected production candidate,
  exact dense-reference comparisons and ignored scaling fixture.
- `/tmp/molla-closures-abba-{1..4}.log` — initial synthetic scaling measurements.
- `/tmp/molla-closures-machine-abba-{1..4}.log` — initial Kubota regression.
- `/tmp/molla-closures-fastpath-abba-{1..4}.log` — one-articulation fast path.
- `/tmp/molla-closures-densepath-*` — revised dense-participation validation.
- `/tmp/molla-closures-restored-*` — restored-source validation.

## Contiguous Cholesky rows checkpoint (2026-09-20)

### Goal

Reduce measured real-machine CPU costs while continuing the full replacement
plan, preserving substeps, physics equations and numerical rank policies.

### Instructions

- Keep original Gearbox and user processes untouched; integration validation
  runs only in the isolated physics worktree. No owned viewers were launched.
- Engram tools remain unavailable; this checkpoint is the session handoff.

### Discoveries

- The previous turn was progress: rejected compact closure variants supplied
  real-machine evidence against shipping their synthetic scaling improvements.
- A fresh Kubota trace confirms a 39-DOF response factorization eight times per
  outer step, alongside the drive-inertia factorization. The shared Cholesky
  inner loop repeatedly indexes flat storage. Disjoint row slices expose bounds
  and aliasing to the compiler without changing scalar arithmetic order.
- The new tests compare every output bit against the original scalar algorithm,
  including untouched upper triangles and partially updated matrices on failure.
  Existing non-finite behavior is preserved, not silently redefined.

### Accomplished

- Replaced flat inner-loop indexing in `dense_chol` with disjoint pivot/trailing
  row slices. Pivot validation, update order, square roots, divisions and error
  behavior remain unchanged. No unsafe code, SIMD reduction/reassociation, solver
  settings, tolerances or fallback/rank thresholds were introduced or changed.
- Added deterministic SPD fixtures at dimensions 0/1/2/3/6/26/39/78/128, eight
  seeds and scales 1e-18/1/1e18. Failure tests cover each pivot, zero/rank-one
  matrices, invalid length, signed zero, tiny values and non-finite diagonals.
- Initial factor-only A/B/B/A mean medians: 26 DOFs **2.6685 -> 1.485 us**,
  39 **8.5045 -> 4.3385 us**, 78 **65.512 -> 34.1615 us**, and 156
  **267.516 -> 145.266 us**. These are microbenchmarks, not whole-machine claims;
  the 78-DOF baseline varied noticeably between runs.
- Combined math/geometry/solver CPU lane: **681 passed, seven ignored** across
  118 test binaries. Ordinary Molla binary lane: **87 passed, eight ignored**.
- Three real-Kubota A/B/B/A cycles (12 runs, six per retained binary) give mean
  Molla medians **0.912905708 -> 0.886234125 ms**, a **2.922%** reduction.
  Individual cycle reductions are 3.097%, 2.969%, and 2.701%. Rapier controls
  shift **0.229482667 -> 0.227412667 ms** (0.902%); their cycle shifts vary
  from 3.785% faster to 2.092% slower. Keep that host noise visible: this is a
  modest measured gain, not a quiet-host latency guarantee or parity claim.
  The new Molla/Rapier ratio is still approximately **3.9x**.
- All 12 paired tests pass; every logged pressure/load/deflection/pose state
  matches the baseline exactly. No owned build/test overlapped these paired
  timings. User processes, including an unrelated Gearbox viewer, remained alone.
- Final imported-machine suite: **all seven tests pass** at unchanged
  thresholds, including pressure extremes/replay, straight driving/turning,
  slope parking, trailer support and hitch/PTO controllers. Ordinary Rapier
  binary lane: **87 passed, eight ignored**. Formatting checks pass.
- Candidate profiling passes. The response-factor span decreases from
  **0.057751692 to 0.044637458 ms/frame**, and exclusive drive-inertia time
  from **0.092871325 to 0.08104555 ms/frame**. Unchanged phases are slower in
  the second trace and total instrumented time increases; these separate traces
  localize work, not prove latency improvement. The uninstrumented repeated
  A/B/B/A measurements above are the performance evidence.
- Final focused bitwise-reference rerun passes **two tests, one ignored**.
  All owned build/test handles are terminal. Full-plan completion and
  Rapier-comparable performance remain unproven.

### Next Steps

1. Continue examining triangular solves and ABA allocation/lookup costs using
   the same exact-reference and real-machine timing gates. Implement proper
   sleep/wake islands; the Molla adapter still has no-op sleep/wake methods.
2. Continue full-plan performance/islands/sleeping, generic/GPU joint semantics,
   runtime/lifecycle, calibrated tyre/terrain and rendered acceptance gates.

### Relevant Files

- `crates/molla-math/src/dense_solve.rs` — contiguous Cholesky row access.
- `crates/molla-math/tests/dense_chol.rs` — bitwise reference and scaling tests.
- `/tmp/molla-dense-chol-before-{trace.json,profile.tsv,profile.log}` — fresh
  real-machine baseline profiling; trace instrumentation adds overhead.
- `/tmp/molla-dense-chol-{before,after}` — retained math benchmark executables.
- `/tmp/molla-dense-chol-scaling-{1..4}.log` — factor-only A/B/B/A measurements.
- `/tmp/gearbox-after-island-scratch`, `/tmp/gearbox-after-dense-chol` — retained
  real-machine baseline and candidate test executables.
- `/tmp/molla-dense-chol-machine-abba-{1..12}.log` — paired Kubota timings.
- `/tmp/molla-dense-chol-full.log` — combined CPU regression lane.
- `/tmp/molla-dense-chol-{molla-binary,rapier-binary,imported}.log` — final
  ordinary backend and imported-machine regression checks.
- `/tmp/molla-dense-chol-after-{trace.json,profile.tsv,profile.log}` — candidate
  profiling and warm-state verification.

## Next implementation steps

1. Finish runtime descriptor/integration semantics; retain the tested
   re-rooting, mutation, contact/readout/filtering paths while extending the world.
2. Implement sleep/wake/islands; verify continuous-force singular-inertia
   dynamics and benchmark queries with tyres.
3. Finish soft angular phase semantics and GPU joint-control parity; validate
   compliant hitches at real tractor/trailer scale.
4. Complete and validate the Gearbox backend in its existing physics worktree.
5. Validate real Gearbox tyre/terrain behavior and field-friction authoring,
   then sleeping/deformation/MPM as specified.
6. Run and record every original acceptance scene, performance and deterministic
   replay gate. Do not mark the objective complete from library tests alone.
