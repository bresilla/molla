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

## Batched closure whitening checkpoint (2026-09-20)

### Goal

Continue full-plan performance work by reducing repeated triangular solves in
closure responses, without changing physics equations or numerical thresholds.

### Instructions

- Original Gearbox, assets and user processes stay untouched; only isolated
  physics-worktree tests exercise the integration. No owned viewers are launched.
- Engram tools remain unavailable; this checkpoint preserves the session handoff.

### Discoveries

- The previous turn was progress: contiguous Cholesky access shipped with
  bitwise-reference coverage and repeated real-machine timing evidence.
- The closure response whitens many projected wrenches against the same factor.
  Four independent RHS lanes can share each coefficient load while keeping each
  lane's subtraction/division order identical to the existing scalar solve.
- Initial batching improves representative 39-DOF/22-row Gram assembly, but
  six-DOF/64-row measurements regress. The revised candidate retains the old
  scalar path for at most six DOFs or fewer than four RHS rows. This is a work
  scheduling cutoff, not a numerical rank or physics threshold.

### Accomplished

- Added four-RHS lower-triangular tiles with scalar handling of incomplete tails.
  Cholesky construction, spectral fallback, Gram accumulation and the global
  bounded constraint solve remain unchanged. Workspace is scoped to one call;
  no state is cached across substeps or topology mutations.
- Added bitwise comparisons against the old scalar Gram implementation across
  independent row counts, complete/partial batches, full-rank/singular/zero mass,
  three mass scales and repeated accumulation into nonzero response matrices.
  Invalid row lengths/non-finite inputs must leave the output unchanged.
- Initial candidate CPU lane: **390 passed, seven ignored**, 89 binaries;
  ordinary Molla binary lane: **87 passed, eight ignored**.
- Three additional factor-response A/B/B/A cycles give 39 DOFs/22 rows
  **13.863833 -> 8.077 us** (41.74% lower); six DOFs/64 rows increase
  **9.560333 -> 10.441833 us** (9.22%). These are response-only synthetic
  timings, not whole-machine performance, and motivate the scalar cutoff.
- Initial real-machine three-cycle A/B/B/A: Molla mean medians
  **0.888016250 -> 0.853618333 ms** (3.874% lower), Rapier controls
  **0.227858 -> 0.227716458 ms** (0.062% lower). Logged mechanical states
  match exactly across all 12 runs.
- Final scalar-cutoff candidate CPU lane: **390 passed, seven ignored**;
  ordinary Molla/Rapier binary lanes: **87 passed, eight ignored each**.
  Exact-reference cases additionally cover seven/eight DOFs at the cutoff;
  invalid-input atomicity exercises the eight-DOF/eight-row batch path.
- Final real-machine two-cycle A/B/B/A: Molla mean medians
  **0.892427813 -> 0.852627687 ms** (4.460% lower); individual cycle reductions
  are 4.253% and 4.665%. Rapier controls also shift
  **0.230770750 -> 0.226835062 ms** (1.705% lower), so preserve that host-noise
  caveat. The Molla/Rapier ratio remains about **3.76x**, not parity. All eight
  paired tests pass, with exact matching logged pressure/load/deflection/pose.
- Final unrestricted microbenchmarks remain noisy, including smaller cases.
  A separate three-cycle microbenchmark with only the test process pinned to
  allowed CPU 2 gives 39 DOFs/22 rows **11.658333 -> 5.663167 us**, and six
  DOFs/64 rows **7.701 -> 7.187833 us**, not reproducing the larger small-matrix
  regression. Six-DOF one/four-row calls remain 2.5/9.5 ns slower in that sample.
  This is not a universal speedup claim. Real-machine timings were NOT pinned.
- No owned builds or other tests overlapped paired/pinned timing runs. User
  processes remained untouched. Final imported-machine checks pass **all seven
  tests** at unchanged thresholds, including pressure extremes/replay, parked
  slopes, straight driving/turning, trailer support and hitch/PTO controllers.
- Final profiling passes. Whitening measures **0.035627992 ms/frame**, versus
  **0.073313317 ms/frame** in the prior post-Cholesky trace. Other phases also
  shift; traces are diagnostic, not latency acceptance. Paired uninstrumented
  results above remain the performance evidence. All owned handles are terminal.
- This checkpoint does not complete `FULL_COMPACT.md`: sleep/wake islands,
  fleet scaling, lifecycle/calibration, generic/GPU joint and rendered/soil
  acceptance remain incomplete or unverified.

### Next Steps

1. Start proper runtime sleep/wake state and connected islands. Existing adapter
   sleep/wake methods are no-ops; parked/fleet performance needs real integration
   and wake tests.
2. Continue full-plan sleep/wake islands and remaining runtime/joint/tyre/terrain
   acceptance. No fleet, GPU, rendered or full-plan completion claim follows from
   these CPU response tests.

### Relevant Files

- `crates/molla-solvers/src/rigid_impulse.rs` — batched whitening candidate.
- `crates/molla-solvers/src/gram_batch_tests.rs` — bitwise reference, invalid
  input atomicity and ignored response benchmark.
- `/tmp/molla-gram-batch-scaling-{1..16}.log` — initial response timings.
- `/tmp/molla-gram-batch-machine-abba-{1..12}.log` — initial paired Kubota timing.
- `/tmp/molla-gram-batch-full.log` — initial CPU regression lane.
- `/tmp/gearbox-after-dense-chol`, `/tmp/gearbox-after-gram-batch` — retained
  baseline and initial candidate real-machine test executables.
- `/tmp/gearbox-after-gram-batch-guard` — final candidate real-machine binary.
- `/tmp/molla-gram-batch-guard-*` — final CPU/backend checks and paired/micro
  timings, including `abba-{1..8}.log` and `scaling-{1..12}.log`.
- `/tmp/molla-gram-batch-pinned-scaling-{1..12}.log` — CPU-2 microbenchmarks.
- `/tmp/molla-gram-batch-imported.log` — final imported-machine regressions.
- `/tmp/molla-gram-batch-{trace.json,profile.tsv,profile.log}` — candidate
  profiling and warm-state verification.

## Rigid island dependency checkpoint (2026-09-20)

### Goal

Start full-plan sleeping/islands with deterministic dynamic-body connectivity
and wake dependencies, before integrating actual dormant solver execution.

### Instructions

- Preserve original Gearbox, assets and unrelated processes. No owned viewers
  are launched. Engram tools remain unavailable; this is the session handoff.
- The previous turn was progress: batched whitening is committed and validated.

### Discoveries

- Solver ownership is a whole Featherstone articulation. Dynamic branches of
  one fixed-root tree remain grouped conservatively; separate articulations
  must not merge merely because they contact the same fixed/kinematic body.
- Active loop joints and physical contacts connect dynamic groups. Immovable
  endpoints are separate wake dependencies: editing the ground can affect both
  nearby groups without waking one machine merely because another wakes.
- Tyres bypass rigid ground contacts. Their ground samples and heading-body
  dependencies must participate separately; removed wheel/heading references
  must not poison queries before the next wheel `prepare` call.
- Island snapshots expose stable body handles and transient articulation
  indices. They do not update themselves after scene edits or steps. Future
  activation must use freshly detected contacts before each substep, not just
  the latest-step diagnostic snapshot returned by `RigidWorld::islands()`.

### Accomplished

- Added `RigidIslands::build` for whole-articulation groups, active loop joints
  and supplied body-force/contact links. Disabled endpoints are excluded;
  unknown handles are rejected. Ordering is deterministic under reversed or
  duplicate edges. Input scenes are not mutated.
- Added dynamic-body ownership and reverse boundary ownership for affected-body
  expansion without scanning every island on each wake request.
- Added on-demand `RigidWorld::islands()` using valid nonsensor contact samples,
  active wheel heading references and published tyre ground samples. Stale
  samples after scene edits or failed steps do not supply contact links.
- Eight new focused tests pass: tree/loop activation, removal/reindexing,
  fixed/kinematic dependencies, fixed-root ownership, deterministic links,
  filtered/sensor contacts, stale samples, tyre-only support and stale heading
  cleanup. Reverse boundary ownership is covered by the final full regression.
- Final CPU solver lane: **398 passed, seven ignored**, across 90 test binaries.
  Ordinary Molla and Rapier binary lanes each pass **87 tests, eight ignored**.
  The final imported-machine suite passes **all seven tests**. Those imported
  timings are not paired performance evidence and no speedup is claimed here.
- Final focused rerun passes all eight tests; formatting checks pass and all
  owned build/test handles are terminal.
- No stepping path calls the new query automatically. This is dependency
  infrastructure, NOT implemented sleeping or a performance improvement.
  Backend `sleep`/`wake_up` remain no-ops and reporting remains truthful.

### Next Steps

1. Add actual per-body activation state and whole-island solver skipping.
   Connect it to fresh substep contacts and motion/force/control/pressure events before
   changing the backend's sleeping readout or honoring start-asleep requests.
2. Parentless ground has no body handle; collider/terrain edits require explicit
   dependency/invalidation handling. Arbitrary external force callbacks need
   supplied coupling edges or conservative activation. Raw state/control mutation
   paths also need a defined wake contract. Do not infer these from this graph.
3. Validate parked/mixed-active fleets, collision wakes, runtime merge/split,
   pressure changes, kinematic contacts and lifecycle preservation, then continue
   the remaining full-plan calibration/GPU/rendered/soil acceptance work.

### Relevant Files

- `crates/molla-solvers/src/rigid_islands.rs` — dependency graph and wake expansion.
- `crates/molla-solvers/src/rigid_world.rs` — on-demand built-in island snapshot.
- `crates/molla-solvers/src/lib.rs` — public solver-layer module export.
- `crates/molla-solvers/tests/rigid_islands.rs` — dependency/lifecycle/contact tests.
- `/tmp/molla-rigid-islands-{build,focused}.log` — initial focused checks.
- `/tmp/molla-rigid-islands-{full,final}.log` — final CPU and focused checks.
- `/tmp/molla-rigid-islands-{molla-binary,rapier-binary,imported}.log` — ordinary
  backend and imported-machine regressions.

## Selected articulation execution checkpoint (2026-09-20)

### Goal

Continue the full replacement plan with an actual CPU execution primitive for
inactive whole-articulation groups. This is not automatic sleep/wake integration.

### Instructions

- Original Gearbox, shared assets and unrelated user processes remain untouched.
- Keep the existing eight-substep machine configuration and acceptance tolerances.
- The user confirmed interacting with `molla-gram-perf`; its stopping is not an
  unattended failure observation. That answer does not explain the window closing.

### Discoveries

- Whole-articulation selection can omit drive-inertia factorization, ABA and
  integration while retaining inactive coordinates, velocities, poses and persistent
  forces exactly. Global FK, contact evaluation and scratch allocation still run.
- Dynamic contacts and enabled mimic links cannot cross the active/inactive
  boundary. Fixed/kinematic contacts do not require matching activation; sensor,
  disabled-collider and nonpenetrating contacts do not physically couple groups.
- A selected solver step is not a sleep policy: retained inactive velocities may
  be nonzero. The caller remains responsible for arbitrary force dependencies,
  wake events and fresh substep collision detection.

### Accomplished

- Added CPU `FeatherstoneSolver::step_active_articulations`, with explicit GPU
  rejection, mask/dependency validation and substep-underflow rejection. The normal
  stepping path still passes no selection and changes no physics settings.
- Inactive-only contact samples report no acted force or normal impulse. Active
  samples preserve the ordinary solver's force and substep impulse accounting.
- Twelve new functional tests pass. All-active trajectories match the ordinary
  solver exactly; mixed selections match separately paused/resumed simulations for
  120 steps with one, two and eight substeps, both rigorous solve settings, limits,
  disabled joints, motors and persistent wrenches. Tests cover contact readout,
  mimic dependencies, malformed inputs, immovable boundaries, sensors, empty and
  implicit articulations, and selection from `RigidIslands` snapshots.
- Final full CPU solver lane: **410 passed, eight ignored**, across 90 test
  binaries. Final focused lane: **13 passed, two ignored**. Ordinary isolated
  Gearbox binary lanes: **87 passed, eight ignored** on each backend. GPU-feature
  compilation passes; this is not GPU activation implementation or parity proof.
- Added an interleaved, alternating-order ignored scaling benchmark using the
  same rigorous solve and stable-feedback settings for ordinary/all/one/no-active
  execution. At 512 independent two-body articulations, ordinary stepping measured
  **1.438456 ms**, selected all-active **1.480955 ms**, selected one-active
  **0.502024 ms**, and selected none-active **0.502692 ms** median. These are
  solver-only synthetic timings, not real parked fleets or Rapier comparisons.
- Two retained-binary A/B/B/A cycles leave normal real-machine timing effectively
  unchanged: Molla's average of parked/driving medians is **0.851901938 →
  0.850407625 ms** (-0.175%, individual cycles +0.248% and -0.594%). Rapier control
  is **0.226564500 → 0.227791813 ms** (+0.542%). This is noise-scale variation,
  not a machine speedup; Molla remains about **3.73×** Rapier on this fixture.
  All recorded speeds/sleep counts match between binaries. No owned builds or
  tests ran concurrently with the timing cycles. Renderer and controller work
  remain outside the timed physics step.
- All **seven imported-machine regressions pass**, including pressure replay,
  controller actuation, slope parking, trailer parking and straight/turning track
  behavior. Implementation/tests are committed as `3970617`. All owned build/test
  handles are terminal; no viewer was launched in this checkpoint.

### Next Steps

1. Connect selection to fresh collision detection and external-force dependency
   closure before each substep, then implement runtime sleep/wake state and events.
2. Handle control/raw-state mutations, pressure transitions, static terrain edits,
   kinematic motion, contact wakes and topology merge/split. Do not change backend
   `is_sleeping` or honor start-asleep by merely freezing individual bodies.
3. Benchmark actual parked and mixed-active fleets, and continue the remaining
   calibration, GPU, visual, lifecycle, service and soil acceptance requirements.

### Relevant Files

- `crates/molla-solvers/src/featherstone_activation.rs` — opt-in CPU selection.
- `crates/molla-solvers/src/featherstone.rs` — skip selected dynamics and restore
  inactive state; ordinary stepping remains unselected.
- `crates/molla-solvers/tests/featherstone_islands.rs` — shared tree/mimic fixture.
- `crates/molla-solvers/tests/featherstone_islands/activation.rs` — selection
  correctness, dependency validation and synthetic scaling tests.
- `/tmp/molla-activation-{full-final,focused-final,gpu-check}.log` — solver gates.
- `/tmp/molla-activation-{molla-final,rapier-final}.log` — backend binary gates.
- `/tmp/molla-activation-scaling-final.log` — alternating synthetic benchmark.
- `/tmp/molla-activation-abba-{1..8}.log` — two real-machine timing cycles.
- `/tmp/molla-activation-imported.log` — full imported-machine regressions.
- `/tmp/gearbox-after-activation` — retained final machine test binary.

## Explicit runtime sleep/wake checkpoint (2026-09-20)

### Goal

Connect selected articulation execution to real runtime sleep/wake and the isolated
Gearbox body API. Automatic settling detection and the complete replacement plan
remain unfinished.

### Instructions

- Preserve original Gearbox and all shared assets. Only Molla and isolated
  `gearbox-physics` are being edited; no viewers were launched.
- Keep eight machine substeps and existing acceptance limits.

### Discoveries

- Runtime motor writes retain `SceneStamp`; gravity and raw mutable access also
  need independent activation invalidation. Added per-body mutation versions and
  conservative global versions for raw access, topology edits and public state
  commits. Successful world-owned steps acknowledge their own commits.
- Both scene handles and mutation counts can repeat after replacing a public
  `world.scene`. Activation stamps also retain a unique scene identity. Wheel
  storage uses its own retained identity plus revision, not a bare counter.
- A sleeping group must wake when contact pairs appear or disappear, including
  pair-filter changes, and when a cached tyre support is excluded. Fresh contacts
  are inspected before every substep; graph connectivity is reused only while
  those substep edges stay identical.
- Frozen tyres retain pressure/shear state and cached support/footprint readout.
  Held samples use `dt = 0`, so they do not invent integrated impulses. The backend
  explicitly exposes their static support rather than treating them as airborne.

### Accomplished

- Added `RigidWorld::{sleep_body,wake_body,is_sleeping}` for complete dynamic
  dependency groups, with zeroed generalized velocities and selected CPU stepping.
  No automatic sleep thresholds or timers are enabled.
- Added fresh-contact selected substeps. Collision/moving-kinematic wakes, motor
  and body-input changes, pressure transitions and surface edits resume dynamics.
  Unknown external-force callbacks run fully awake on successful steps; failures
  preserve sleep and rigid/tyre state. Structural/raw edits conservatively wake all.
- Connected isolated Gearbox's sleep/readout/wake methods and start-asleep request.
  Existing `_wake` mutation arguments still conservatively wake on actual edits;
  honoring explicit non-waking edits and strong/weak automatic-wake policy remains.
- New tests cover exact inactive poses, independent active motion/damping, tree
  and loop wake propagation, later-substep collision wakes, sensor/exclusion cases,
  fixed/kinematic boundaries, rollback, raw access, pressure/shear resume parity,
  retained support, storage replacement, backend start asleep and motor wake.
- Final checks passed **461 simulation/solver tests, nine ignored**, in
  95 binaries, and **90 ordinary backend binary tests, nine ignored**, under each
  backend environment. Final GPU-feature compilation and new-file formatting
  checks pass. GPU sleep execution/parity is not implemented.
- The final collider-free 128-body explicit-sleep microbenchmark measured
  **0.410540 ms** all asleep, **0.420646 ms** one awake and **1.640728 ms** all
  awake. These sequential synthetic samples are not parked-machine or Rapier
  acceptance. Substep connectivity caching avoids repeated identical graph builds.
- Two retained-binary A/B/B/A cycles on the ordinary Kubota path measured Molla
  **0.839948938 → 0.848724063 ms** (+1.045%) and Rapier control **0.225196688 →
  0.228182688 ms** (+1.326%), averaging parked/driving medians. Both move upward
  slightly; no normal-machine speedup is claimed. Molla remains **3.72×** Rapier.
  All logged Molla poses, deflections, loads and speeds match across binaries.
  No owned build/test ran concurrently with these cycles.
- All **eight imported-machine tests pass**, including the new explicit-sleep
  diagnostic and pressure/drive recovery. The real Kubota accepts explicit sleep,
  but remains asleep for **0/240 subsequent full-controller ticks** (median
  0.842015 ms). There is no live tractor sleep performance benefit yet.
- Source inspection identifies a concrete repeated wake path: Ackermann applies
  wheel velocity motors, then `ParkingBrakes::apply` overrides them with position
  holds. Both writes reach `RigidScene::set_joint_motor`; input-change activation
  is currently conservative even when that frame's final motor equals its prior
  value. Resolve this alongside the explicit non-waking update contract rather
  than suppressing genuine command or pressure wakes.
- All owned build/test handles are terminal. Core implementation and isolated
  integration are committed as `f981627` and `f567c44` respectively; no original
  Gearbox changes or GUI claims.

### Next Steps

1. Resolve controller velocity/parking-hold rewrite wakes and define non-waking
   motor/body edits. Require the full-controller diagnostic to demonstrate useful
   sleep persistence without breaking command/pressure/contact wakes.
2. Add automatic quiet detection and granular wake policy; preserve scene-edit,
   contact, pressure and failure semantics. Optimize remaining sleeping-loop/global
   scratch work, and measure real parked/mixed-active fleets against Rapier.
3. Continue every outstanding full-plan calibration, GPU, rendered, lifecycle,
   hitch/PTO service, terrain and soil acceptance requirement.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — activation identities and mutation versions.
- `crates/molla-solvers/src/rigid_sleep.rs` — explicit sleeping groups and wakes.
- `crates/molla-solvers/src/rigid_world.rs` — selected runtime routing.
- `crates/molla-solvers/src/featherstone_activation.rs` — fresh-contact selected steps.
- `crates/molla-solvers/src/wheel_forces.rs` — frozen tyre state and held support.
- `crates/molla-solvers/tests/rigid_sleep.rs` and `tests/wheel_forces/sleep.rs` — tests.
- Isolated `bin/gearbox/src/physics/molla/{body,mod,wheel,tests}.rs` — backend wiring.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — explicit-sleep diagnostic.
- `/tmp/molla-sleep-cache-{full,backend,rapier,gpu}.log` — final checks.
- `/tmp/molla-sleep-fleet-{initial,final}.log` — collider-free scaling diagnostics.
- `/tmp/molla-sleep-abba-{1..8}.log` — two normal-path timing guard cycles.
- `/tmp/molla-sleep-imported.log` — eight imported-machine tests and sleep diagnostic.
- `/tmp/gearbox-after-sleep` — retained final candidate binary.
- `/tmp/gearbox-after-activation` — ordinary-path pre-sleep baseline.

## Automatic settling and controller wake work (2026-09-20)

### Goal

Continue `FULL_COMPACT.md` sleep/performance work without changing the original
Gearbox checkout or weakening the existing machine-dynamics acceptance gates.

### Instructions

- The user confirmed interacting with the earlier `molla-gram-perf` window.
  Its stopping is not evidence of an unattended failure; closure is unexplained.
- Keep all integration in `robolibs/gearbox-physics`, not original Gearbox.

### Discoveries

- Parking rewrote velocity then position motors every frame. Selecting the final
  hold before writing controls eliminates those artificial activation changes.
- Real Kubota quiet detection cannot use chassis speed alone: four-/two-kilogram
  articulated bodies still rotate while its chassis appears stationary. At the
  conservative per-body thresholds, first automatic sleep occurs at tick 5383
  (about 44.87 seconds). Do not relax thresholds merely to hide that motion.
- Quiet-window activation stamps must be checked before the solver commits its
  own next state. Post-commit comparisons otherwise reset every timer each tick.
- Engram tools are unavailable in this session; this journal holds the handoff.

### Accomplished

- Added opt-in CPU `SleepSettings` and whole-island quiet windows; defaults are
  0.0001 m/s, 0.0001 rad/s and one continuous quiet second. Isolated Molla backend
  enables them; generic `RigidWorld` remains opt-in.
- Dynamic/boundary input changes and explicit wake reset affected windows.
  Pressure transitions, moving boundaries, persistent body forces, joint
  feed-forward, and nonzero velocity/effort commands prevent automatic sleep.
  Both compiled tree controls and authored loop motors are inspected.
- Invalid settings and failed steps preserve quiet time. Successful arbitrary
  force-callback steps clear it. Optional policy inspection failure logs and
  keeps normal committed dynamics successful rather than falsely rolling back.
- Batch sleep validates all handles before mutation and commits all selected
  groups together. Fully sleeping groups skip unused loop-force computation.
- Added non-waking joint motor updates and wired the adapter's mutation flag;
  the original motor method still wakes on input changes. Body mutation flags
  and strong/weak wake distinctions remain separate outstanding work.
- Parking keeps original force caps/hold compliance while avoiding intermediate
  velocity writes. The full-controller explicit-sleep diagnostic now requires
  all 240 measured ticks to stay asleep and verifies pressure/drive recovery.
- Added automatic-pressure/drive acceptance and core quiet-window, dependency,
  failure, tiny-command, pressure and batch-atomicity tests.
- Final CPU gates pass **473 simulation/solver tests, nine ignored**, in 95
  binaries. Both backend environments pass **91 ordinary binary tests, ten
  ignored** each. GPU-feature compilation and new-file formatting checks pass;
  GPU automatic-sleep execution/parity is not implemented.
- All **nine imported-machine tests pass**, including unchanged slope,
  pressure-replay and driving tolerances. Automatic sleep first occurs at tick
  5383; pressure reduction wakes immediately, re-settles after another 1534 ticks
  (about 12.78 seconds), and drive commands wake and reach the expected speed.
  The final automatic-sleep sample measured **0.163554 ms median**, **0.222162 ms
  p95** over 240 ticks; explicit sleep stays asleep **240/240**, median 0.192356
  ms. These are parked single-machine results, not general Rapier parity.
- Two retained-binary A/B/B/A cycles on the ordinary Kubota path measured Molla
  **0.863419000 → 0.853635688 ms** (-1.133%), with Rapier control **0.229341375 →
  0.229080438 ms** (-0.114%), averaging parked/driving medians. This is a small
  normal-path change, not parity: Molla remains **3.73×** Rapier. The short
  ordinary parked lane does not run long enough for automatic sleep. All logged
  poses, deflections, loads and speeds match byte-for-byte across all eight runs.
  No owned builds/tests ran concurrently with the final imported-machine lane or
  comparison runs.
- `/tmp` refused the candidate copy with a user-quota error despite free space.
  The partial copy was removed; the retained candidate is verified byte-identical
  under the isolated checkout's ignored `target/perf/gearbox-after-settling`.
- Core implementation is committed as `346c2ed`; isolated integration is committed
  as `f672656`. All owned build/test handles are terminal. No original Gearbox
  edits, merges, launches or rendered acceptance claims.

### Next Steps

1. Reduce parked stepping overhead and diagnose long internal settling without
   freezing real motion; benchmark real parked/mixed-active fleets.
2. Complete non-waking body mutations and strong/weak wake semantics; preserve
   controller, contact, pressure and failure coverage.
3. Continue all remaining full-plan dynamics, GPU, calibration, rendered tyre,
   lifecycle, service propagation, terrain and soil acceptance requirements.

### Relevant Files

- `crates/molla-solvers/src/rigid_settling.rs` — automatic quiet policy.
- `crates/molla-solvers/src/rigid_sleep.rs` — atomic batch sleep and wake reset.
- `crates/molla-solvers/src/rigid_world.rs` — policy step routing.
- `crates/molla-sim/src/runtime.rs` — non-waking motor update API.
- `crates/molla-solvers/tests/rigid_settling/mod.rs` — quiet-policy tests.
- `crates/molla-solvers/tests/wheel_forces/sleep.rs` — automatic pressure wake.
- Isolated `bin/gearbox/src/controller{,/parking,/benchmark}.rs` — final holds
  and full-controller imported tests.
- Isolated `bin/gearbox/src/physics/molla/{mod,joint,tests}.rs` — backend wiring.
- `/tmp/molla-auto-*` — test/diagnostic logs; diagnostic builds are not timing
  baselines. Temporary diagnostic printing was removed before final validation.
- Isolated `target/perf/gearbox-after-settling` — retained final candidate;
  `/tmp/gearbox-after-sleep` — pre-change normal-path baseline.

## Fully inactive CPU-step overhead (2026-09-20)

### Goal

Continue the full compatibility/performance plan after the verified automatic
settling checkpoint. The previous goal turn made progress (`346c2ed`, `f672656`,
`3266638`); current checkouts and plan were re-read before this work.

### Instructions

- Original Gearbox remains out of scope; use only its isolated physics checkout.
- Keep active-physics substeps and the contact/pressure/wake contracts intact.

### Discoveries

- Even with every articulation inactive, CPU Featherstone allocated all global
  ABA scratch arrays and repeated final FK before restoring inactive state.
- Skipping the remainder after validated contact readout avoids those unused
  allocations without skipping any collision query or per-substep callback.
  A callback may still wake the whole world on a later substep.
- Mixed-active worlds still allocate per-joint ABA arrays for total joint count;
  this fast return only removes that cost when every joint is inactive. The
  collision pipeline still performs SAP/filter/narrow-phase work each substep.
- Engram tools remain unavailable; this journal is the persistent handoff.

### Accomplished

- Fully inactive steps restore rigid state and persistent forces immediately
  after normal contact evaluation/filtering. Mixed/active dynamics are unchanged.
- Added regression coverage for later-substep callback wake and exact contact
  geometry/material metadata with zero inactive forces/impulses. Existing frozen
  state, mimic, fixed/kinematic, pressure and wake tests remain enabled.
- Initial retained-binary A/B/B/A on the real automatic-sleep fixture measured
  **0.1559455 → 0.1185420 ms** average parked median (about 24% less). Sleep tick
  5383, pressure resettle tick 1533, and post-wake driving/tyre state match exactly.
- Added an isolated paired long-parked benchmark with the same real Kubota,
  7200 warmup ticks and 1200 measured ticks on each backend, in R/M/M/R order.
- Final gates pass **475 simulation/solver tests, nine ignored**, in 95 binaries;
  **91 ordinary backend binary tests, eleven ignored** under each backend
  environment; and all **ten imported-machine tests**. GPU-feature compilation
  and activation-test formatting checks pass. This remains CPU activation work,
  not a GPU execution/parity result.
- Across both retained-binary A/B/B/A cycles, parked medians average
  **0.155761250 → 0.131412250 ms** (**15.632% less**). All samples are retained,
  including one candidate run at 0.168253 ms (p95 0.408111 ms); the other three
  candidate medians are 0.117497–0.120312 ms. The initial 24% result above is one
  cycle, not the full-run aggregate. Sleep/pressure timing and logged post-wake
  driving/tyre state match across all eight runs.
- The new paired long-parked lane measured average medians **0.145525 ms Molla**
  and **0.226860 ms Rapier**. Molla is asleep **26/26** and Rapier remains awake
  **0/26** after identical 60-second warmups. This is a narrow parked-policy
  comparison, not equal-active-state or general performance parity.
- One ordinary-path A/B/B/A guard cycle measured Molla **0.852211750 →
  0.846100125 ms** (-0.717%), Rapier control **0.230603750 → 0.231464250 ms**.
  All logged ordinary-path poses, deflections, loads and speeds match across
  binaries. No material active-path speedup is claimed; Molla remains **3.66×**
  Rapier in this lane. No owned build/test ran concurrently with timed lanes.
- Core change is committed as `61f53ff`; the isolated benchmark is committed as
  `7c35eb6`. All owned command handles are terminal. Original Gearbox is untouched;
  no live viewer was launched and no new rendered acceptance is claimed.

### Next Steps

1. Benchmark actual parked/mixed-active fleets and remove remaining mixed-world
   scratch/collision overhead while preserving per-substep callbacks and wakes.
2. Finish non-waking body mutations and strong/weak wake
   policy semantics; do not confuse a frozen single machine with driving parity.
3. Continue every remaining full-plan dynamics, GPU, calibrated tyre, rendered,
   lifecycle, service, terrain and soil requirement.

### Relevant Files

- `crates/molla-solvers/src/featherstone.rs` — inactive CPU early return.
- `crates/molla-solvers/tests/featherstone_islands/activation.rs` — regressions.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — long-parked paired timing.
- `/tmp/molla-inactive-*` — test and retained-binary timing logs.
- Isolated `target/perf/gearbox-after-settling` — prior checkpoint baseline.
- Isolated `target/perf/gearbox-after-inactive` — initial optimized candidate.
- Isolated `target/perf/gearbox-after-inactive-validated` — final candidate with
  the added long-parked benchmark; verified copy of the tested binary.
- `/tmp/molla-inactive-auto-{1..8}.log` — paired sleep/pressure/drive comparisons.
- `/tmp/molla-inactive-normal-{1..4}.log` — ordinary-path regression comparison.
- `/tmp/molla-inactive-imported-final.log` — ten imported tests and paired parking.

## Articulation-local ABA storage (2026-09-20)

### Goal

Continue mixed-scene performance work in `FULL_COMPACT.md`. The previous turn
made verified progress (`61f53ff`, `7c35eb6`, `75a6353`); current code, journal,
plan and isolated checkout were re-read before changing storage.

### Instructions

- Do not touch original Gearbox or weaken active dynamics/contact substeps.
- Keep shared solver work in Molla and machine integration in the isolated tree.

### Discoveries

- ABA's per-joint temporary arrays are only read within the current articulation;
  parent searches in the backward/forward passes stay within its joint range.
- `q_dd_local` duplicated acceleration writes but had no readers; generalized
  acceleration still lives in the separate world-wide `q_dd` array.
- Cargo produces different test executable hashes for the focused solver lane
  and the combined sim/solver lane. An initial scaling comparison accidentally
  reused the old combined-lane executable for both arms; matching checksums
  exposed it. Those logs are renamed `molla-local-scratch-stale-alias-*` and are
  not performance evidence. Final comparisons must use the exact executable
  named by the completed matching-lane build and verify distinct checksums.
- Engram tools remain unavailable; this journal stores the handoff.

### Accomplished

- Added bounded global-index views over reusable articulation-local arrays.
  Capacity is the largest active articulation, not total world joint count;
  every visited articulation resets its own range before the three ABA passes.
- Removed the write-only acceleration duplicate. Generalized state, motor
  inertia, reaction forces and joint integration math are unchanged.
- Added storage reuse/bounds tests and an exact separate-world trajectory oracle
  with heterogeneous 1/7/3/11/2-body articulations, fixed roots, changing active
  masks, scalar stops, both solve modes, and one/eight substeps.
- Initial full gates pass **478 simulation/solver tests, nine ignored**, in 95 binaries.
  Both backend environments pass **91 ordinary binary tests, eleven ignored**;
  GPU-feature compilation passes. These are CPU storage tests, not GPU execution
  or automatic-sleep parity evidence.
- Correctly paired combined-lane A/B/B/A measurements on the collider-free,
  two-body-articulation fixture show 512 articulations / one selected active
  articulation at **0.4973840 → 0.2533255 ms** average median (about 49% less).
  Ordinary all-active stepping is **1.4323935 → 1.2444850 ms** (about 13% less);
  selected all-active is **1.4761440 → 1.2937080 ms**. These are synthetic scaling
  results, not real multi-machine fleet acceptance.
- The first implementation passes all ten imported-machine tests and matches all
  logged dynamics, but two ordinary-path A/B/B/A cycles show a small consistent
  single-machine cost: Molla **0.836715813 → 0.843777687 ms** (+0.844%), Rapier
  control **0.227594688 → 0.226832438 ms**. It is not accepted as the final version.
- Added compile-time flat/local storage specializations: single-articulation
  worlds keep direct global indexing and one initialization; multi-articulation
  worlds use bounded reset views. A sole articulation cannot benefit from the
  smaller capacity, so remapping overhead is unnecessary there. The solver's
  calculations remain shared, without separate equations or tolerances.
- Final flat/local candidate passes **479 simulation/solver tests, nine ignored**
  in 95 binaries, both backend binary lanes (**91 passed, eleven ignored** each),
  all **ten imported-machine tests**, GPU-feature compilation and new-file
  formatting checks. The heterogeneous oracle now compares local-storage worlds
  against separate flat-storage worlds, exactly, through changing wake masks.
- Final matching-lane retained-binary A/B/B/A on 512 synthetic articulations:
  one selected active articulation **0.5022765 → 0.2628465 ms** (-47.669%);
  ordinary all-active **1.4382990 → 1.2412385 ms** (-13.701%); selected all-active
  **1.4852050 → 1.3036625 ms**. This does not establish real fleet performance.
- Two final real-machine A/B/B/A cycles measure Molla **0.857194188 →
  0.857813625 ms** (+0.072%), Rapier control **0.232007250 → 0.232260250 ms**
  (+0.109%). The single-machine path is effectively unchanged, and Molla remains
  **3.69×** Rapier. All logged poses, deflections, loads and speeds match across
  all eight binaries/runs. No owned builds/tests ran concurrently with timings.
- Implementation committed as `142bb47`; isolated integration source is unchanged
  this turn. All owned command handles are terminal. Original Gearbox is untouched
  and no viewer was launched. Final evidence is in the `*-final-*` logs; earlier
  unspecialized and stale-alias results above are diagnostic history only.

### Next Steps

1. Benchmark actual same-world parked/mixed-active machine fleets. Remaining
   world-wide FK, collision and state-buffer work is not removed by local ABA
   storage; preserve contact/filter callbacks and kinematic wake dependencies.
2. Continue
   body wake semantics, active-path optimization and remaining compatibility work.
3. Complete every original GPU, tyre calibration/visual, lifecycle, service,
   terrain and soil requirement; storage changes are not full-plan acceptance.

### Relevant Files

- `crates/molla-solvers/src/featherstone_scratch.rs` — bounded reusable storage.
- `crates/molla-solvers/src/featherstone.rs` — per-articulation storage routing.
- `crates/molla-solvers/tests/featherstone_islands/scratch.rs` — mixed-size oracle.
- `target/perf/featherstone-before-local-scratch` — matching combined-lane baseline.
- `target/perf/sleep-before-local-scratch` — prior rigid sleep test binary.
- Isolated `target/perf/gearbox-after-inactive-validated` — real-machine baseline.
- `/tmp/molla-local-scratch-*` — build/check/test evidence and diagnostics.
- `target/perf/featherstone-after-local-scratch-validated` — final combined-lane
  solver test binary; distinct checksum and matching build invocation verified.
- Isolated `target/perf/gearbox-after-local-scratch-validated` — final tested
  machine binary, verified byte-identical to the build output before timing.
- `/tmp/molla-local-scratch-final-{full,backend,rapier,gpu,imported,format}.log` —
  final gates; `final-scale-{1..4}` and `final-normal-{1..8}` logs — final timings.

## Mixed-fleet sleeping loop-force checkpoint (2026-09-20)

### Goal

Continue the complete `FULL_COMPACT.md` objective by measuring real same-world
fleets and removing loop-constraint work belonging to sleeping machines.

### Instructions

- Original Gearbox remains untouched; integration changes are confined to
  `robolibs/gearbox-physics`. Shared machine assets are read-only.
- Engram tools are unavailable; this tracked checkpoint preserves the handoff.
- No viewer was launched or stopped. User interaction with the earlier
  `molla-gram-perf` run is not unattended-failure evidence.

### Discoveries

- Synthetic articulation scaling did not expose the whole real-fleet cost.
  Loop forces still assembled and solved rows for sleeping machines whenever
  another machine was awake, although inactive ABA discarded their forces.
- The new fixture composes 1/2/4 real Kubota references into one USD stage and
  one physics world, 25 m apart, sharing ground. It checks unique IDs,
  per-machine body/joint/collider counts, mass, pressure, support and speed.
  Controllers/services run outside the measured physics step. It warms for
  60 s, then measures all parked, one driving and all driving, with 5 s phase
  warmup and 1,200 measured 120 Hz steps. No explicit sleep calls are used.
- The initial Rapier per-tick 0.15 m/s speed assertion failed at 2.166 m/s.
  Rapier diagnostics now follow the existing paired benchmark's endpoint
  0.2 m/s tolerance and log the complete measured speed range and excursion
  flag. Molla retains 0.15 m/s validation on every measured driving tick and
  verifies every parked body's sleep state on every measured tick.
- Rapier's four-machine run reproducibly includes a roughly 44 m/s transient
  on the tractor at x=50 m, despite recovering by the endpoint. This is an
  unresolved reference stability issue, not clean stable-fleet parity evidence.
  It is recorded rather than hidden by changing spacing or weakening Molla.
- Selecting only awake loop rows also prevents unrelated sleeping masses from
  influencing the active system's numerical conditioning. The new regression
  checks exact force independence across zero and 1e-15 through 1e15 sleeping
  mass/inertia scales, including physical singular inertias `(0, I, I)`.
  The first test used invalid `(0, I, 2I)` inertia and was corrected after the
  existing mass validation rejected its triangle-inequality violation.

### Accomplished

- Added an internal selected loop-force path. Disabled joints are unchanged;
  selected dynamic endpoints determine whether a row participates. Fixed and
  kinematic boundaries do not activate unrelated sleeping loops. Malformed
  masks and split dynamic loop dependencies fail before force-buffer writes.
  The world continues fresh collision/filter and wake propagation each substep.
- Added fixed/kinematic shared-boundary, bounded motor, active-response,
  unchanged sleeping-force, invalid-selection atomicity and extreme sleeping
  mass/singular-inertia regression tests. Three focused selection tests pass.
- Final complete sim/solver gate passed **482 tests, nine ignored, 95 binaries**;
  both ordinary backend lanes passed **91 tests, 12 ignored each**; all **ten**
  prior imported-machine gates passed. GPU-feature compilation and new-file
  formatting checks passed, not GPU runtime parity. The earlier failed
  `final-full` log contains the invalid test inertia; `validated-full` is the
  final passing gate, including actual mass-readback assertions.
- Same-harness real-fleet A/B/B/A, mean of medians (ms per physics step):

  | Fleet / active | Before | After | Change |
  | --- | ---: | ---: | ---: |
  | 1 / 1 | 0.8525205 | 0.8517910 | -0.086% |
  | 2 / 1 | 1.4630095 | 1.0671495 | -27.058% |
  | 4 / 1 | 3.7799685 | 1.5569930 | -58.809% |
  | 2 / 2 | 1.9515850 | 1.9518135 | +0.012% |
  | 4 / 4 | 5.3900285 | 5.2427970 | -2.732% |

- All four complete fleet runs passed; parked machines stayed asleep when
  another drove. Maximum difference across corresponding logged mechanical
  state values was 2.14e-14. Rapier medians shifted about 1.5–2.7%; the all-active
  4-machine change is similar to that control drift, not a claimed speedup.
  Four-machine/one-active p95 mean fell 5.0188915 to 3.2095255 ms, still far from
  latency parity. Parked timings were noisy: 1-machine mean medians rose
  0.1236115 to 0.1427790 ms and 4-machine 0.7542765 to 0.8521150 ms; no parked
  speedup is claimed. No owned builds/tests overlapped these timing runs.
- The matched A binary uses the final benchmark and unselected loop call;
  B uses the selected call. A second true pre-change baseline is retained for
  additional single-machine before/after checks. Restoring source formatting
  changed binary bytes, so the final rebuilt B artifact was retained separately,
  checked against build output, and used for all four fleet measurements.
- Two additional true pre-change A/B/B/A single-machine cycles passed. Mean
  medians across normal parked/driving phases: Molla **0.868609625 →
  0.853958250 ms** (-1.687%), Rapier control **0.227466813 → 0.226177250 ms**
  (-0.567%). All logged mechanical states match exactly across all eight runs.
  This guards against a single-machine regression; Molla remains **3.78×**
  Rapier, and quiet-host tail/real-time parity is not established.
- Committed solver change as `40989de` and isolated fleet fixture as `1eab2d4`.
  All owned build/test handles are terminal. Original Gearbox and the user's
  untracked `FULL_COMPACT.md` are unchanged. The complete goal remains active.

### Next Steps

1. Optimize the remaining all-active global closure solve/FK/collision work.
   Revisit the rejected compact-closure checkpoint before repeating it; preserve
   numerical rank/normalization semantics across genuinely coupled articulations.
2. Investigate the Rapier reference's transient separately. This test does not
   prove stable Rapier fleet behavior or broad backend performance parity.
3. Complete every original lifecycle, wake, joint/GPU, towing replay, service,
   pressure UI/CLI persistence, calibrated tyre/terrain and rendered soil gate.
   Fleet timing and sleeping-loop selection are not full-plan acceptance.

### Relevant Files

- `crates/molla-solvers/src/loop_forces.rs` — selected loop rows and validation.
- `crates/molla-solvers/src/loop_forces/selection.rs` — force/mass-scale oracles.
- `crates/molla-solvers/src/rigid_sleep.rs` — substep activation-mask forwarding.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — shared imported fixture.
- Isolated `bin/gearbox/src/controller/benchmark/fleet.rs` — same-world fleet gate.
- Isolated `target/perf/gearbox-fleet-{before,matched-before,after}-selected-loops`
  and `gearbox-fleet-after-selected-loops-final` — retained timing artifacts.
- `/tmp/molla-fleet-{initial,diagnostic,range}.log` — fixture diagnostics.
- `/tmp/molla-selected-loops-*` — tests/builds; `fleet-{1..4}.log` is the final
  same-harness A/B/B/A evidence; imported gate timing is correctness-only because
  its run overlapped other validation/build work.
- `/tmp/molla-selected-loops-normal-{1..8}.log` — true pre-change normal-machine
  comparisons; `validated-{full,gpu,format}.log` — final core validation.

## Active-fleet response compaction checkpoint (2026-09-20)

### Goal

Continue full compatibility/performance work by reducing the all-active fleet's
global loop-response work. Previous checkpoint was verified implementation
progress, not a wait or completion of the full objective.

### Instructions

- Same worktree, original Gearbox and shared-asset boundaries remain in force.
- Engram tools remain unavailable. No original checkout edits or viewer actions.

### Discoveries

- A block-Cholesky-only candidate passed the real 1/2/4-machine fleet gate but
  showed no clear total-step improvement: initial four-active median 5.363281 ms.
- Added an ignored profile-feature fleet trace in the isolated test fixture.
  Its first 120-step trace recorded 773.318 ms inside `rigid_step`, 407.353 ms
  in loop forces, 353.767 ms in loop response construction, but only 21.901 ms
  in the bounded solve. Response Gram construction was 130.207 ms, projection
  87.463 ms, whitening 49.676 ms. These are instrumented inclusive durations,
  not normal step-time acceptance measurements. No rank/Cholesky fallback
  events occurred in that trace.

### Accomplished

- Implemented exact-zero disconnected matrix components
  inside each global active-set linear solve. It retains the global matrix
  normalizer and pivot cutoff; any singular/near-cutoff/indefinite block falls
  back to the original complete solve. Active-set clamp/release order is not
  split. Dense/reference tests pass bit-exactly, including interleaved blocks,
  scales 1e-100–1e100 and arbitrarily small nonzero cross-block couplings.
- Second candidate compacts exactly zero projected Gram rows only for
  multi-articulation models. It scatters the small response back into the same
  global matrix and retains the original single-articulation route. Tests cover
  non-diagonal full-rank/singular matrices, empty/sparse/dense rows, scale and
  whiten-path cutovers, with exact equality to the dense path and atomic invalid
  input rejection. A final no-zero-row fast path delegates directly to the old
  Gram routine. Final validation and timing evidence is recorded below.
- Baseline is isolated `target/perf/gearbox-fleet-after-selected-loops-final`
  from the last committed checkpoint. Do not replace it with an older pre-sleep
  baseline. Initial block-only artifact is `gearbox-after-loop-blocks-initial`;
  first compact candidate is `gearbox-after-loop-blocks-compact`; final candidate
  build targets `gearbox-after-loop-blocks-final`. Verify build logs and byte
  comparisons before using an artifact.
- First combined-candidate fleet A/B/B/A passed with bit-identical logged states:
  4 active **5.322799 → 4.155651 ms** (-21.927%); 2 active **1.962706 →
  1.877601 ms** (-4.336%). Rapier corresponding controls changed +0.051% and
  +0.210%. Normal single-machine A/B/B/A **0.85804425 → 0.864436375 ms**
  (+0.745%), Rapier **0.2267145 → 0.22768825 ms** (+0.430%). However, one active
  in four regressed **1.5512545 → 1.5948885 ms** (+2.813%). No owned validation
  work overlapped these timings.
- Final revision counts articulations actually contributing to
  the loop response and invokes block solving only when more than one
  contributes, rather than using total model articulation count. This avoids
  partition-search overhead for a lone active tractor beside sleepers. Latest
  artifact is `gearbox-after-loop-blocks-active`; all final measurements below
  use that artifact, not the earlier `*-final` combined candidate.
- Final same-world fleet A/B/B/A mean medians (ms per physics step):

  | Fleet / active | Before | After | Change |
  | --- | ---: | ---: | ---: |
  | 1 / 1 | 0.8416895 | 0.8534365 | +1.396% |
  | 2 / 1 | 1.0557885 | 1.0747825 | +1.799% |
  | 4 / 1 | 1.5276820 | 1.5409375 | +0.868% |
  | 2 / 2 | 1.9373775 | 1.8430790 | -4.867% |
  | 4 / 4 | 5.2665860 | 4.0705825 | -22.709% |

- Corresponding Rapier all-active controls changed +0.235% (two machines) and
  +0.265% (four). Four-machine/one-active Rapier changed +1.340%; the small
  Molla mixed-fleet variation is not claimed as a speedup. Parked measurements
  remain noisy (two-machine mean medians +10.589%, four-machine -1.255%).
  Four-active p95 mean fell **5.9448975 → 4.6727190 ms**. Single-machine and
  quiet-host latency parity are not established by these fleet improvements.
- Final original single-machine paired A/B/B/A: Molla **0.843792750 →
  0.847208750 ms** (+0.405%); Rapier **0.224941625 → 0.223544750 ms** (-0.621%).
  This is a small measured single-machine cost, not a single-machine speedup;
  retain the optimization for the demonstrated real all-active fleet benefit.
  Molla remains about **3.79×** Rapier on that ordinary fixture.
- All four final fleet runs and all four final normal runs passed. Every logged
  mechanical state matches exactly across corresponding runs. The fleet gate
  checks all parked bodies remain asleep while others drive. The pre-existing
  Rapier 44 m/s transient remains flagged; no broad stable-backend-parity claim.
  No owned builds/tests overlapped final timings; user processes were not stopped.
- Final solver/simulation gate: **488 passed, nine ignored, 95 binaries**.
  Ordinary Molla/Rapier lanes each pass **91 tests, 12 ignored**; profile-feature
  binary passes **91 tests, 14 ignored**; all **ten** prior imported-machine
  tests pass. GPU feature compilation and new-file formatting checks pass.
  Imported validation ran alongside other correctness/build work, so its timing
  numbers are not used as performance evidence. GPU runtime parity remains open.
- Final four-active profile capture passes. Its 120-step inclusive durations
  were `rigid_step` **632.633 ms**, `loop_response` **204.200 ms**, Gram
  **9.307 ms**, whitening **15.439 ms**, and projection **88.196 ms**. Compared
  with the initial block-only trace, Gram/whitening work is substantially smaller;
  projection remains. These traces are diagnostic, not uninstrumented timings.
- Solver changes committed as `45f1e60`; isolated fleet trace as `bb8a47c`.
  All owned command handles are terminal. Original Gearbox and the user's
  untracked `FULL_COMPACT.md` remain unchanged. The complete goal remains active.

### Next Steps

1. Reduce remaining loop projection/global allocation work, without repeating
   the earlier compact-response attempt's single-machine regression. Consider
   repeated articulated mass computation only after checking its distinct
   drive/force conditioning semantics. Require actual-machine before/after gates.
2. Profile end-to-end viewer performance separately from headless CPU stepping;
   do not equate the latter with rendered performance or visual tyre acceptance.
3. Complete remaining body wake semantics, joints/GPU parity, towing replay,
   pressure lifecycle/persistence, service propagation, calibrated tyre/terrain
   mechanics and rendered soil requirements. This is not full-plan completion.

### Relevant Files

- `crates/molla-solvers/src/rigid_impulse/{blocks,compact}.rs` — candidates/tests.
- `crates/molla-solvers/src/loop_forces{.rs,/blocks.rs}` — routing/bounded oracle.
- `crates/molla-solvers/src/articulated_response.rs` — compact response entry.
- Isolated `bin/gearbox/src/controller/benchmark/fleet.rs` — added fleet trace.
- `/tmp/molla-loop-blocks-*` — diagnostic/build/test logs.
- Isolated `target/perf/molla-loop-blocks-fleet-trace.json` — first profile trace.
- Isolated `target/perf/molla-loop-blocks-active-trace.json` — final profile trace.
- `/tmp/molla-loop-blocks-active-{full,gpu,format,backend,rapier,profile-build,
  imported,trace}.log` — final validation; `active-fleet-{1..4}.log` and
  `active-normal-{1..4}.log` — final A/B/B/A measurements.

### 2026-09-20 viewer terrain profiling and candidate bounds reuse

#### Goal

- Continue the full replacement plan; investigate real viewer performance.

#### Instructions

- User requested a live launch while validation was in progress. Leave
  `molla-terrain-fix` open for their interaction; do not treat that window as an
  unattended benchmark or stop it as profiling cleanup. Original Gearbox is also
  running under user control and remains untouched.

#### Discoveries

- The isolated viewer needed an actual release executable; existing retained
  artifacts were test harnesses. Added `build-release-profile` to its Oslo recipes
  and `GEARBOX_LOG_FILE` support so launches do not truncate the original's log.
- First release viewer trace, 30–80 s: 1340 main frames, physics 31.352 s
  inclusive (23.397 ms/frame), render schedule 17.537 s inclusive. Tyre update
  systems totaled 232.266 ms across 2680 invocations; all 28 rubber meshes used
  GPU deformation. Inclusive/thread-overlapping durations are not additive.
- Targeted solver trace, 15–30 s: 1780 rigid steps totaled 9954.531 ms;
  collision totaled 7620.671 ms across 14241 substeps, about 77% of step time.
  Full terrain is an 801x801 heightfield; `local_aabb()` scans every height on
  each broad-phase substep. Headless cuboid fixtures did not expose that cost.
- Both diagnostic viewers were stopped by explicit instance name. Startup logged
  one Bevy slab allocator use-after-free warning; no conclusion or fix yet.

#### Accomplished

- Candidate `PreparedShapeBounds` borrows the immutable model, computes each
  heightfield's local bounds once, and transforms them using fresh body poses.
  `CollisionPipeline::collide_prepared` derives its model from that borrow;
  ordinary collision remains an uncached oracle. Both normal and active-set
  Featherstone stepping prepare bounds once per step, refresh contacts every
  substep, and retain all existing filters/resolution/solver iterations.
- Focused exact-bound/contact/filter/mutation tests pass. Combined geometry,
  simulation and solver lane: 682 passed, 10 ignored, 120 test binaries.
  Ordinary isolated backend: 91 passed, 12 ignored. GPU feature check passes.
  These changes are not committed yet; performance acceptance remains pending.
- Launched candidate instance `molla-terrain-fix`, confirmed one Kubota, PLAYING,
  and inspected `/tmp/molla-terrain-fix.png` showing rear tyres on the ground.

#### Next Steps

- Poll imported validation handle 72892 (`/tmp/molla-prepared-imported.log`),
  which was still running when the user requested launch. Its timings overlap
  the live viewer, so use them for correctness only.
- Leave user viewer handle 48050 alive. Its launch log is
  `/tmp/molla-terrain-fix-launch.log`; dedicated log `/tmp/molla-terrain-fix.log`.
- Run the ignored full-resolution collision timing and before/after viewer
  trials when not competing with owned builds/tests. The prepared A/B/B/A script
  has NOT run. Do not claim a measured terrain speedup from the implementation.
- Audit candidate, add any missing edge coverage, then commit if real evidence
  supports retention. Full plan remains incomplete and active.

#### Relevant Files

- `crates/molla-geometry/src/{shape_world,broad_phase,pipeline}.rs` — bounds reuse.
- `crates/molla-geometry/tests/prepared_bounds.rs` — exact oracles and ignored timing.
- `crates/molla-solvers/src/{featherstone,featherstone_activation}.rs` — step routing.
- Isolated `.make.lua`, `bin/gearbox/src/main.rs` — release trace build and log isolation.
- Isolated `target/perf/gearbox-viewer-profile-before` SHA256
  `42518d0f7517aadbfeb22b5afa30df238fddb893784446c545364dff8fcff3b3`.
- Isolated `target/perf/gearbox-viewer-profile-after` SHA256
  `1d12cb135c1a3c4552685508f50000ef989af0a39196f3bf120c7a6d0a9ce0d6`.
- Isolated `target/perf/run-terrain-viewer.sh` — unexecuted viewer A/B/B/A script.
- Isolated `target/perf/molla-{viewer,world}-profile.json` — diagnostic traces;
  broad viewer trace is approximately 7 GB, targeted solver trace 330 MB.
- `/tmp/molla-{viewer,world}-systems.json` — per-thread stack summaries.
- `/tmp/molla-prepared-{bounds-focused,bounds-full,backend,gpu-check,imported}.log`
  and `/tmp/molla-prepared-viewer-build.log` — validation/build evidence.
- Engram APIs remain unavailable in discovered tools; this tracked handoff
  records session context without editing the protected memory folder.

### 2026-09-20 matched meadow validation

#### Goal

- Validate and retain the terrain bounds optimization against the complete
  replacement plan's physics correctness and real-machine performance gates.

#### Instructions

- Keep the original Gearbox untouched. The user-facing `molla-terrain-fix`
  process subsequently exited normally (handle 48050 returned exit 0); no
  panic/quarantine/step-rejection appeared in its log. It was not stopped by
  benchmark cleanup, and it has not been automatically relaunched.

#### Discoveries

- The bounds microbenchmark with two instances of the same 801x801 field and
  eight fresh collision substeps averaged 8.629263 ms uncached versus
  1.239330 ms prepared, an 85.64% reduction. This is collision-only, not a
  whole-machine or viewer-FPS result.
- `RigidScene::stamp()` is a compiled-scene identity retained by simulation
  steps and motor updates; `scene.model()` is immutable. A future scoped
  runtime cache can potentially eliminate the remaining once-per-step scan.
  Do not cache public mutable `Model` geometry by pointer identity alone.

#### Accomplished

- Added a headless real-Kubota fixture using the viewer's actual meadow
  sampling/heightfield conversion, explicitly 801x801 over 800x800 m. It
  preserves full collision resolution, 120 Hz and solver settings, verifies
  finite/enabled bodies and tyre support, and drives within the flat spawn
  region. It excludes rendering, field visuals and the distant safety floor.
- Built a matched reference by temporarily routing both Featherstone entry
  points through fresh bounds, retaining all other candidate code and the
  same new fixture. Restored the prepared routes before the final build.
  The worktree is the candidate, not the temporary reference.
- Sequential reference/candidate/candidate/reference runs each passed. Every
  fixture runs Rapier/Molla/Molla/Rapier internally; four medians per table cell:

  | Full meadow physics | Reference ms | Prepared ms | Change |
  | --- | ---: | ---: | ---: |
  | Molla parked | 4.994811250 | 1.483867250 | -70.292% |
  | Molla driving | 5.020794750 | 1.494853250 | -70.227% |
  | Rapier parked control | 0.280457750 | 0.281478000 | +0.364% |
  | Rapier driving control | 0.290884500 | 0.292275250 | +0.478% |

- Driving p95 means were 6.104681500 -> 1.817279750 ms. All logged Molla tyre
  and chassis states match exactly across all four executions; this comparison
  covers the logged values, not a full-world state hash. Prepared meadow Molla
  remains 5.11x Rapier and above the plan's 1 ms CPU target.
- Matched original cuboid-ground A/B/B/A also passed: Molla parked medians
  0.851540750 -> 0.848152250 ms, driving 0.864012000 -> 0.862633750 ms.
  No substantive single-machine speedup/regression is established by these
  tiny differences. Rapier driving control shifted +2.29%; host activity exists.
  All logged Molla states match exactly across these runs too.
- No owned builds or correctness tests overlapped those matched timings.
  User-controlled GUI activity was not interrupted, so these are not quiet-host
  tail-latency acceptance and not an end-to-end viewer A/B/B/A comparison.
- All ten existing imported-machine gates passed (130.36 s; timings overlapped
  the user viewer and are not performance evidence). Final ordinary backend
  lanes pass 91 tests each for Molla/default and Rapier, with 13 ignored after
  adding the meadow fixture. Earlier combined geometry/sim/solver gate passed
  682 tests; final post-restoration repetition also passes 682 tests, ten ignored,
  120 binaries. Final GPU feature compilation and new-file formatting pass.
- Isolated profiling support committed as `9946da9`; meadow fixture `c7910cb`.
- Library optimization committed as `88c2ae7`. All owned validation and viewer
  handles are terminal; the original user's Gearbox was not stopped or changed.
  `FULL_COMPACT.md` remains untracked and untouched; the full goal stays active.

#### Next Steps

- Consider a typed scene-identity cache for terrain bounds, retaining exact
  query/contact oracles and all geometry/pose/rebuild invalidation semantics.
- End-to-end viewer comparison is still pending: the prepared viewer A/B/B/A
  script has never run. Preserve all remaining full-plan requirements.

#### Relevant Files

- Isolated `bin/gearbox/src/controller/benchmark/terrain.rs` — new real meadow gate.
- Isolated `bin/gearbox/src/terrain.rs` — test-only native meadow collider helper.
- Isolated `target/perf/gearbox-meadow-reference` SHA256
  `7980751b7dafb8343896def65eeade334486c2b8de14d8163a02b9cf52c297ab`.
- Isolated `target/perf/gearbox-meadow-prepared-final` SHA256
  `fd59d7ff72f7f607126317a0b6bfea30c89b8461b2e3856ac02e40b1a9ba373f`.
- `/tmp/molla-meadow-abba-{1,2,3,4}.log` — matched meadow executions.
- `/tmp/molla-meadow-normal-{1,2,3,4}.log` — matched original fixture executions.
- `/tmp/molla-meadow-{reference-build,prepared-final-build,prepared-first}.log`
  — builds and initial candidate diagnostic.
- `/tmp/molla-prepared-{terrain-timing,rapier,final-full}.log` — supplemental gates.
- Matched timing handle 19573 and imported validation 72892 are terminal, exit 0.
- The first final combined repetition (25496) returned exit 2 with its `/tmp`
  log abruptly ending at 4096 bytes and no failure summary. The cause was not
  established; a later append probe succeeded. Retained the truncated log as
  `target/perf/prepared-bounds-truncated.log` and reran the same gate with output
  in `target/perf/prepared-bounds-final-full.log` (handle 87993).
- That rerun completed successfully with the full 682-test result above. Final
  check logs: `target/perf/prepared-bounds-final-gpu.log` and
  `target/perf/prepared-bounds-format.log`; handles 87993 and 78971 exit 0.

### 2026-09-20 persistent scene-bound terrain cache

#### Goal

- Remove the remaining per-step heightfield scan without weakening geometry
  mutation correctness, substep contacts, pressure mechanics or the full plan.

#### Instructions

- Original Gearbox remains user-owned and untouched. No new viewer was launched
  this turn. Its latest release executable was built only in the isolated tree.

#### Discoveries

- A cache keyed by the compiled `SceneStamp` is safe for `RigidScene`'s immutable
  model access; raw `Model` geometry remains publicly mutable and must not use an
  address-only persistent cache. Scene changes/replacements refresh the cache;
  state commits retain it, while world-space bounds still use current body poses.
- One candidate meadow driving median was 1.001204 ms. A mean of medians below
  1 ms does not establish a consistently sub-millisecond step or tail acceptance.
- A user-owned `cargo test --all-targets` in original Gearbox was observed during
  the later normal-fixture timings (PID 1496227 cwd verified). It was not stopped.

#### Accomplished

- `SceneShapeBounds::prepare` produces a borrowed `PreparedShapeBounds` backed
  by retained local bounds. Standalone mutable-Model calls still own their
  per-call bounds. Normal and sleeping rigid-world steps both use the scene
  cache; private solver entry points reject mismatched model/bounds identities.
- Added exact contact/AABB oracles across body motion without a scene rebuild,
  collider pose edits, heightfield replacement, invalid atomic edits, enable
  toggles, heightfield/cuboid swaps, body removal and entire-scene replacement.
  Storage-reuse coverage verifies steady state does not replace the bounds vector.
- A 600-step rigid-world oracle compares retained versus forcibly rebuilt bounds
  through terrain edits, removal/reinsertion, explicit sleep/wake and a failed
  transient-force callback. Body/joint state and contact arrays match exactly;
  the test requires at least ten genuinely sleeping steps.
- Final combined geometry/sim/solver gate: **685 passed, ten ignored, 120
  binaries**. Focused final storage test and formatting pass. GPU feature
  compilation passes, not GPU runtime parity. Ordinary backend lanes each pass
  91 tests, 13 ignored; all ten prior imported-machine gates pass (126.06 s).
- Matched sequential A/B/B/A uses the previous per-step cache as reference and
  the persistent-scene cache as candidate. Each execution contains R/M/M/R;
  every full-meadow and original-cuboid execution passed:

  | Full meadow physics | Per-step ms | Per-scene ms | Change |
  | --- | ---: | ---: | ---: |
  | Molla parked | 1.487369000 | 0.977754500 | -34.263% |
  | Molla driving | 1.495692500 | 0.993062250 | -33.605% |
  | Rapier parked control | 0.279459250 | 0.281061250 | +0.573% |
  | Rapier driving control | 0.289816250 | 0.290686750 | +0.300% |

- Meadow driving mean p95: 1.999104250 -> 1.673707750 ms. Candidate remains
  **3.42x Rapier**. All logged Molla tyre/chassis states match exactly in the four
  corresponding runs. These are CPU step measurements, not viewer FPS.
- Original cuboid fixture parked medians 0.872897250 -> 0.869863000 ms; driving
  0.884168250 -> 0.870963750 ms. Rapier driving control 0.227768500 -> 0.230612250
  ms; tiny original-fixture differences are not claimed as a meaningful speedup.
  Logged Molla states match exactly across all four runs.
- No owned build/test jobs overlapped final timings; user activity continued.
  The diagnostic first meadow run overlapped correctness work and is not used
  for the accepted timing table. No quiet-host or full-viewer parity claim.
- Committed as `59e14a8 perf(geometry): cache terrain bounds by scene`. All owned
  build/test/timing handles are terminal. Full goal remains active and incomplete.

#### Next Steps

- Run the outstanding end-to-end viewer comparison with explicit isolated names,
  controlled camera/drive setup, logging and screenshots; do not stop the user's
  original app. The existing `run-terrain-viewer.sh` still points at older arms
  and has NEVER run: update its binary mapping before using it.
- Re-profile remaining active physics work (loop projections, repeated mass
  computation and FK) rather than changing stiffness, resolution or substeps.
- Retain the entire outstanding wake/joint/GPU/towing/pressure-lifecycle,
  calibrated tyre/field/soil and rendered acceptance requirements.

#### Relevant Files

- `crates/molla-geometry/src/shape_world.rs` — scene-stamped borrowed bounds.
- `crates/molla-geometry/src/scene_bounds_tests.rs` — storage/scene identity oracle.
- `crates/molla-geometry/tests/prepared_bounds.rs` — runtime mutation/contact oracle.
- `crates/molla-solvers/src/{featherstone,featherstone_activation}.rs` — private
  prepared-bound entry points with unchanged public stepping APIs.
- `crates/molla-solvers/src/{rigid_world,rigid_sleep}.rs` — retained cache routing.
- `crates/molla-solvers/src/rigid_bounds_tests.rs` — warm/cold full-step oracle.
- Molla `target/perf/scene-bounds-{focused,full,final-full,check,gpu,runtime,
  storage-final,format}.log` — validation evidence.
- Isolated `target/perf/scene-bounds-{backend-build,imported,meadow-first,rapier,
  viewer-build}.log` — imported/build evidence.
- Isolated `target/perf/scene-bounds-{meadow,normal}-{1,2,3,4}.log` — final A/B/B/A.
- Isolated `target/perf/gearbox-meadow-scene-bounds` SHA256
  `1915dc16de87e6c3bac830283ad2e730a07a729e318df1ecca84fc54d70899e8`.
- Reference remains `target/perf/gearbox-meadow-prepared-final`, SHA256
  `fd59d7ff72f7f607126317a0b6bfea30c89b8461b2e3856ac02e40b1a9ba373f`.
- Latest GUI `target/perf/gearbox-viewer-scene-bounds` SHA256
  `2cbc13d61fb7f17dfa3952b8b231b1b04114df644a0a11c77ddf384eee005abb`;
  `target/release/gearbox` is the same build, not launched after this change.
- Engram tool discovery still returns no memory API; this tracked summary
  preserves the handoff without editing protected memory files.

## Viewer launch and wake-strength checkpoint (2026-09-20)

### Goal

Continue the complete `FULL_COMPACT.md` integration and performance work, keep
the user's requested interactive preview open, and close the explicit strong/
weak wake gap without changing solver settings or the original Gearbox checkout.
The preceding launch turn made progress: the latest terrain-cache GUI was
launched and its rendered Kubota frame was inspected successfully.

### Instructions

- Leave the user's `molla-preview` window available for interaction. Do not
  drive it, close it, or use its user-controlled activity as a benchmark.
- Work only in Molla and isolated `gearbox-physics`; preserve the original
  Gearbox checkout, shared dependencies/assets, and untracked `FULL_COMPACT.md`.

### Discoveries

- Fresh viewer CLI clients can fail to resolve `/gearbox/info` while the
  renderer and physics keep running. `Ctx::client` uses the registry DID but
  ignores its concrete `addr`; the current directory/discovery path is a
  suspect, not an established root cause. No networking source was changed.
- The automated viewer attempts were not a completed A/B/B/A: one failed
  Nix startup after XDG isolation, one Rapier control drove successfully then
  failed its status query, and the next Rapier control repeatedly failed
  readiness. Only the owned benchmark windows were stopped. The early control
  also overlapped an accidentally broad read-only source scan; no controlled
  performance acceptance is claimed from it.
- Molla's `BodyMut::wake_up` ignored `strong`. The pinned Rapier 0.32 source
  resets its quiet timer only for strong wakes; weak wakes can sleep again on
  the next step if still eligible. Molla now implements that distinction.
- The first three-body explicit-sleep reference test exposed an unconditional
  Rapier assertion in `island_manager/sleep.rs:124` when some bodies were woken
  before the next step. The exact case passes on Molla. An explicitly ignored
  reference reproducer remains in the isolated tests; it fails with exit 101
  when invoked directly. This is a known reference failure, not a passing gate.

### Accomplished

- Launched the retained `gearbox-viewer-scene-bounds` GUI as `molla-preview`
  with `GEARBOX_PHYSICS=molla`, loaded the real Kubota, and moved only its camera
  to frame the tractor. Inspected `target/perf/molla-preview.png`: the Kubota
  and terrain render and the title reads `molla-preview | molla | PLAYING`.
  This is launch/render evidence, not driving or pressure visual acceptance.
- Added `RigidWorld::wake_body_with_strength(body, strong)`; the existing
  `wake_body` remains a strong wake. Both wake the full dependency group,
  including watched immovable boundaries. Strong wakes reset affected quiet
  windows; weak wakes preserve existing windows and restore eligibility only
  for currently valid sleeping groups. Ordinary motion/input checks still
  decide whether the next step may sleep. Automatic sleep remains opt-in.
- Added seven sleep tests covering awake quiet history, explicit/automatic
  sleepers, trees/loops/static boundaries, unrelated sleepers, invalid handles,
  failed steps, disabled automatic sleep, input edits, gravity and persistent
  forces. Wired the isolated backend's `wake_up(strong)` to the new API.
- Backend comparison checks automatic-sleep weak/strong transitions on both
  Molla and Rapier, and the explicit-sleep transition on Molla. Kept a separate
  ignored reproducer for the failing Rapier explicit-sleep case.
- Focused final sleep gate: **31 passed, one ignored**. GPU-feature compile
  passes, not runtime parity. Ordinary isolated Molla and Rapier lanes each
  pass **92 tests, 14 ignored**, including the newly recorded reference failure.
- The final combined geometry/sim/solver run passed **692 tests, ten ignored,
  120 binaries**, including all seven new wake tests. All **ten real imported-
  machine gates pass** (131.91 s), including pressure replay, steering/tracks,
  towing, slopes and controller exercise. These are correctness results;
  their incidental timings overlapped other tests and an interactive viewer.
- Committed Molla as `63c797b fix(solvers): distinguish weak and strong wakes`
  and isolated integration as `25197e1 fix(physics): honor wake strength in molla`.
- All owned build/test handles are terminal. Only the requested interactive
  preview is deliberately left running. No full-viewer performance or final
  goal completion claim is made.

### Next Steps

- Finish the remaining non-waking body mutation flags (pose/velocity/kind/
  mass/force/impulse), including failure atomicity and dependency behavior.
- Diagnose viewer directory discovery in the isolated integration only, then
  resume controlled full-viewer comparison in separate owned windows. The
  interactive preview uses the prior scene-bounds binary, not this new wake fix.
- Keep the complete joint/GPU/towing/service/pressure-lifecycle, calibrated
  tyre/surface/soil, rendered acceptance and Rapier-parity work open. The full
  objective is not complete.

### Relevant Files

- `crates/molla-solvers/src/{rigid_sleep,rigid_settling}.rs` — strength-aware
  wake propagation and validated quiet-window retention.
- `crates/molla-solvers/tests/rigid_settling/{mod,wake}.rs` — sleep regression
  wiring and seven new wake/failure/input tests.
- Isolated `bin/gearbox/src/physics/molla/{body,tests}.rs` — backend forwarding,
  cross-backend comparison and known failing Rapier reference reproducer.
- Molla `target/perf/wake-strength-{focused,full,final,full-final}.log` — CPU
  correctness, focused final regression and GPU compile evidence.
- Isolated `target/perf/wake-strength-{backend,backend-final,molla,rapier,
  rapier-reproducer,imported}.log` — backend/reference and imported evidence.
- Isolated `target/perf/viewer-check-{20260920T101150,20260920T101419}/` — failed
  control-channel viewer attempts; not final viewer performance evidence.
- Isolated `target/perf/run-terrain-viewer.sh` — ignored benchmark runner now
  uses latest scene-bounds candidate, explicit Nix features, and waits for the
  fly-camera request to be consumed before follow; no full comparison completed.
- Isolated `target/perf/molla-preview.{log,png}` — interactive launch/render
  evidence. Preview PID 1574460 was confirmed live during this checkpoint;
  revalidate process identity before any future action.
- Engram tool discovery still exposes no memory API. This tracked summary
  preserves the handoff without editing protected memory files.

## Non-waking body input checkpoint (2026-09-20)

### Goal

Continue `FULL_COMPACT.md` body/sleep compatibility without disturbing the
interactive preview. The preceding turn made verified progress: strong/weak
wakes were implemented, tested and committed in both allowed checkouts.

### Instructions

- Keep the requested preview open and leave original Gearbox untouched.
- Use the Molla Makefile, isolated backend test lane and unsigned title-only
  Conventional Commits. Preserve untracked `FULL_COMPACT.md`.

### Discoveries

- Non-waking edits need to retain pending inputs, not discard them or put the
  body back to sleep. The active-articulation solver already preserves inactive
  generalized rates and persistent wrenches; suppressing only the requested
  body's activation stamp lets these inputs take effect on a subsequent wake.
- Geometry/query invalidation is separate from activation. A velocity edit
  still commits validated state and changes the scene stamp, even when it does
  not wake the body. Existing pending wakes must never be acknowledged away.
- Do not blindly apply the same stamp suppression to pose, kind or mass edits:
  pose edits interact with held sleeping tyre samples, and kind/mass edits
  rebuild the model. Those remaining paths need their own lifecycle checks.

### Accomplished

- Added explicit `RigidScene` wake-policy variants for COM velocity, persistent
  force/torque, force at a world point, and force reset. Existing entry points
  retain their waking behavior. Finite-value, overflow, handle and fixed-body
  validation remains before mutation/activation.
- Wired isolated `BodyMut::{set_linvel,set_angvel,add_force,add_torque,
  reset_forces}` to the caller's wake flag instead of discarding it.
- Added three runtime tests for per-body activation isolation, point-force
  torque, finite/overflow/removed-handle failure atomicity, scene stamp
  invalidation, fixed bodies and zero wrench/reset behavior.
- Added six sleep integration tests for stored velocity/wrench retention,
  exact post-wake response, whole-tree rates, reset with/without wake,
  later contact wake, preserving prior wake requests, and quiet-window behavior.
- Added two backend tests: real Rapier/Molla non-waking input comparisons,
  and Molla waking/invalid-input/unrelated-sleeper behavior. The reference
  comparison starts from automatic sleep; the separately recorded Rapier
  explicit-sleep assertion remains unresolved and is not hidden by this gate.
- Final combined geometry/sim/solver gate: **701 passed, ten ignored, 121
  binaries**, including all three new runtime and six new sleep cases. Both
  ordinary backend lanes pass **94 tests, 14 ignored**. GPU-feature compilation
  passes; no GPU runtime parity is claimed. All **ten imported-machine gates
  pass** (132.02 s), covering pressure replay, slopes, towing, controller motion
  and steering/tracks. Incidental timing output is not a controlled benchmark.
- Committed Molla as `138c52f fix(sim): honor nonwaking velocity and force edits`
  and isolated integration as
  `399c3f6 fix(physics): forward nonwaking body input flags`.
- All owned test/build handles are terminal. The user-requested interactive
  preview remains running on its retained scene-bounds binary, not this newer
  body-input implementation. No final compatibility/performance claim is made.

### Next Steps

- Finish non-waking pose/kind/mass/impulse behavior, preserving dependency and
  failure semantics rather than globally acknowledging unrelated edits.
- Continue the controlled viewer/performance work and all remaining joint/GPU,
  towing/service, tyre calibration/lifecycle, terrain/soil and visual acceptance
  requirements. The complete objective remains active and incomplete.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — explicit wake-policy mutation variants.
- `crates/molla-sim/tests/runtime_nonwaking.rs` — activation and atomicity tests.
- `crates/molla-solvers/tests/rigid_sleep.rs` and
  `crates/molla-solvers/tests/rigid_settling/nonwaking.rs` — stepping/dependency/
  quiet-time tests.
- Isolated `bin/gearbox/src/physics/molla/{body,tests,nonwaking_tests}.rs` — flag
  forwarding and backend reference comparisons.
- Molla `target/perf/nonwaking-{focused,full}.log` — CPU and GPU compile gates.
- Isolated `target/perf/nonwaking-{backend-focused,molla,rapier,imported}.log` —
  backend and real-machine correctness evidence.
- Preview PID 1574460 was confirmed live during this work and was not driven
  or closed. Revalidate identity before any future interaction.
- Engram tool discovery still returns no memory API; this tracked summary is
  the available session handoff and does not alter protected memory files.

## Impulse activation checkpoint (2026-09-20)

### Goal

Continue the complete `FULL_COMPACT.md` body/sleep integration by implementing
impulse wake flags and eliminating unrelated impulse-driven wakeups. The prior
goal turn made verified progress: non-waking velocity/force edits were committed
and passed the full CPU, backend and imported-machine lanes.

### Instructions

- Leave the interactive preview and original Gearbox untouched; continue only
  in Molla and isolated `gearbox-physics`. Preserve untracked `FULL_COMPACT.md`.
- Keep physical impulse response in the reusable libraries, not the adapter.
  Use repo-native validation and unsigned title-only Conventional Commits.

### Discoveries

- `RigidWorld::apply_body_impulse` used `RigidScene::commit_step`, which
  invalidated every body's activation stamp, including unrelated sleepers and
  quiet windows. The adapter also discarded all three impulse wake flags.
- A validated dense-rate update is sufficient to commit the impulse response:
  poses and persistent forces are unchanged. Derive activation from the joints
  whose rates actually changed instead of accepting a caller-supplied list of
  affected bodies or acknowledging all previous sleep stamps after a raw commit.
- A nonzero impulse can have zero velocity response when a joint fully blocks
  it. Its explicit waking request still needs to activate that body's group;
  a zero input impulse remains a no-op. Both cases are covered separately.

### Accomplished

- Added `RigidScene::set_generalized_velocities_with_wake` for the current
  compiled model layout. It checks input and existing rate-buffer dimensions,
  finite rates and finite derived body state before committing; no-op updates
  retain scene/activation identity. Only changed joint children are activated
  when requested, and normal dependency grouping propagates their wake.
- Added waking-policy variants for COM/angular and world-point impulses in
  `RigidWorld`; retained the existing methods as waking entry points. The
  mass-matrix response remains unchanged. Non-waking impulses retain latent
  generalized rates without integrating asleep bodies or waking other groups.
- Forwarded the isolated adapter's linear, torque and point impulse flags.
- Added three dense-rate tests for selective activation/no-op identity, invalid
  input and finite-rate overflow atomicity, and malformed existing buffers.
- Added five rigid-world impulse tests covering all three impulse forms, latent
  response, whole-tree activation, unrelated sleepers/quiet windows, prior
  pending edits, zero/invalid/overflow inputs and fully constrained impulses.
- Extended the existing cross-backend tests to cover all three impulse forms
  and their exact stored linear/angular rates before wake; added invalid and
  waking impulse cases to the adapter isolation test.
- Final combined geometry/sim/solver gate: **709 passed, ten ignored, 122
  binaries**, including all three dense-rate and five impulse wake tests. Both
  ordinary backend lanes pass **94 tests, 14 ignored**. GPU-feature compilation
  passes, not GPU runtime parity. All **ten imported-machine checks pass**
  (127.53 s): pressure replay, slopes, towing, controls and steering/tracks.
  Incidental timing output overlapped correctness work and the live preview;
  it is not a controlled performance result.
- Committed reusable implementation as `6d64f36 fix(solvers): scope impulse
  activation by body`, and isolated adapter as `918398f fix(physics): honor
  impulse wake flags`.
- All owned build/test handles are terminal. The requested interactive preview
  remains live on its retained binary, and was not driven or closed. Neither
  full-plan completion nor end-to-end performance parity is established.

### Next Steps

- Finish non-waking pose/kind/mass lifecycle behavior, including held tyre
  sample refresh and dependency changes, without hiding prior pending wakes.
- Retain all end-to-end performance, joint/GPU/towing/service/pressure,
  calibration/terrain/soil and rendered acceptance requirements. The complete
  goal remains active; this checkpoint is not final implementation or parity.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — validated dense-rate mutation and scoped
  activation rather than globally activating a committed impulse result.
- `crates/molla-sim/tests/runtime_generalized_velocity.rs` — transactional
  rate/state validation and activation isolation.
- `crates/molla-solvers/src/rigid_world.rs` — explicit impulse wake policies.
- `crates/molla-solvers/tests/body_mutation{.rs,/impulse_wake.rs}` — real-step
  impulse, sleep, atomicity and unrelated-window regression tests.
- Isolated `bin/gearbox/src/physics/molla/{body,nonwaking_tests}.rs` — impulse
  wake forwarding and expanded Rapier/Molla comparisons.
- Molla `target/perf/impulse-wake-{focused,full}.log` — correctness/compile gates.
- Isolated `target/perf/impulse-wake-{backend,molla,rapier,imported}.log` —
  backend and imported-machine evidence.
- Preview PID 1574460 was verified live and left untouched; revalidate its
  identity before future interaction. It still uses the retained scene-bounds
  GUI, not the newer body-input changes.
- Engram tool discovery remains empty; this tracked summary preserves session
  continuity without writing protected memory files.

## Mass-edit activation checkpoint (2026-09-20)

### Goal

Continue full compatibility by implementing mass-edit wake flags, without
globally waking independent parked machines. The previous turn made progress:
impulse wake handling and scoped generalized-rate commits were implemented,
validated and committed in both permitted checkouts.

### Instructions

- Leave original Gearbox and the interactive preview untouched. Keep reusable
  physics in Molla and integration changes in isolated `gearbox-physics`.
- Preserve untracked `FULL_COMPACT.md`; use repo-native gates, patch tools and
  unsigned title-only Conventional Commits.

### Discoveries

- Mass edits previously used the same global activation path as topology edits.
  Mass-only rebuilds retain handles/joint layout, so their wake policy can be
  scoped to the edited body without acknowledging unrelated pending changes.
- The existing compiler already preserves the velocity field and force/torque
  application points when COM changes, and rejects overflow before committing.
  The new policy retains that transaction; it does not bypass recompilation or
  patch only inverse-mass fields.
- Identical total/additional mass values need neither rebuilding nor activation.
  This also prevents unchanged explicit mass recomputation from waking sleepers.
- Sleeping tyre samples remain held reactions (`dt = 0`), and pressure state is
  retained across a non-waking mass rebuild. Once woken, samples are recomputed
  with the new model; no new load-calibration claim is made from this test.

### Accomplished

- Factored private transactional `rebuild` from globally activating `edit`.
  Structural edits retain their existing activation behavior. Total mass and
  total/additional-component setters have explicit wake variants, activate only
  the edited body when requested, and detect unchanged values before rebuilding.
- Threaded optional activation through geometry's collider-plus-additional
  mass recomputation, `RigidWorld`, and isolated `BodyMut::set_additional_mass`.
- Added three runtime tests covering targeted/no-op activation, COM velocity/
  wrench invariance, invalid handles/values and rebuild-overflow atomicity.
- Added a geometry test for a noncentral collider plus authored additional
  mass, including correct total/COM, no-op identity and rejected-update atomicity.
- Added three rigid-sleep tests covering trees and loops, unrelated sleepers/
  quiet windows, prior pending wakes, mass recomputation and post-wake impulse
  response. Added a tyre sleep/rebuild/resume lifecycle test.
- Added two adapter tests: Rapier/Molla comparison for waking and non-waking
  additional mass edits and subsequent impulse response, plus Molla invalid/
  unchanged-mass behavior. Both focused adapter tests pass.
- Focused runtime/geometry gates pass; final focused sleep and wheel suites pass
  after correcting a test-only `[f64; 3]` finiteness check. Final combined CPU
  gate: **717 passed, ten ignored, 124 binaries**. Both ordinary backend lanes
  pass **96 tests, 14 ignored**. GPU-feature compilation passes, not GPU runtime
  parity. All **ten imported-machine gates pass** (133.29 s), covering pressure
  replay, slopes, towing, controls and steering/tracks. Incidental timings are
  not controlled performance evidence.
- Committed Molla as `a8aec06 fix(sim): scope activation for mass edits` and
  isolated integration as `4ae4f39 fix(physics): honor additional mass wake flag`.
- All owned build/test handles are terminal. The requested preview is still
  live on the retained scene-bounds GUI and was not driven or closed. Full-goal
  completion and end-to-end performance parity remain unproven.

### Next Steps

- Finish non-waking pose/kind behavior with correct held-tyre refresh and
  dependency reconstruction; do not apply mass-only activation suppression
  indiscriminately to structural or pose changes.
- Retain every remaining full-viewer/performance, joint/GPU, towing/service,
  pressure calibration/lifecycle, surface/soil and visual acceptance requirement.
  Full implementation and Rapier performance parity are still unproven.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — mass-specific wake policy and no-op guards.
- `crates/molla-geometry/src/mass_properties.rs` — wake-aware component summation.
- `crates/molla-solvers/src/rigid_world.rs` — public additional-mass wake variant.
- `crates/molla-sim/tests/runtime_mass_wake.rs` and
  `crates/molla-geometry/tests/collider_mass_wake.rs` — property/atomicity tests.
- `crates/molla-solvers/tests/rigid_settling/mass.rs` and
  `crates/molla-solvers/tests/wheel_forces/sleep.rs` — group and sleeping-tyre
  lifecycle tests; `crates/molla-solvers/tests/rigid_sleep.rs` wires the module.
- Isolated `bin/gearbox/src/physics/molla/{body,tests,mass_wake_tests}.rs` —
  adapter forwarding and reference comparisons.
- Molla `target/perf/mass-wake-{focused,focused-final,full}.log` — focused/final
  CPU and GPU-feature compilation evidence.
- Isolated `target/perf/mass-wake-{backend,molla,rapier,imported}.log` — backend
  and imported-machine gates.
- Preview PID 1574460 was verified live during this work and left untouched.
  It still runs the retained scene-bounds GUI, not these later body-input edits.
- Engram discovery still exposes no memory API. This tracked handoff preserves
  continuity without modifying protected memory files.

## 2026-09-20: Teleport activation and Oxbo viewer regression

### Goal

Continue FULL_COMPACT.md; validate non-waking teleports and investigate the
user-reported Oxbo instability when pressing Play.

### Discoveries

- Sleeping tyre observations must refresh after a non-waking teleport without
  integrating pressure, shear, latent velocity or rigid motion. Static terrain
  movement also changes the observation. Pair exclusions must be applied before
  evaluating a potentially degenerate tyre heading.
- The retained scene-bounds viewer loads the six-wheel Oxbo, but the user reports
  violent motion on Play. This is not accepted machine compatibility. Initial
  headless Oxbo tests, including the viewer's terrain/alignment helper, remain
  stable; they have not reproduced the full viewer failure.
- The original Kubota preview and both initial Oxbo windows have exited. Cause
  of closure has not been established. No user/original Gearbox process was killed.

### Accomplished

- Molla `d67134e fix(solvers): refresh sleeping tyres on teleport` adds optional
  pose activation, exact no-op preservation, shared ground sampling and geometry-
  keyed sleeping tyre observations. Isolated integration `1d40a8d` forwards the
  teleport wake flag. Failed steps retain sleeping/pressure state.
- Focused gates pass 43 sleep tests (one ignored) and 29 wheel tests. Combined
  CPU gate passes 724 tests (ten ignored, 124 binaries); GPU-feature compilation
  passes, not GPU runtime parity. Both backend lanes pass 97 tests (14 ignored).
  All ten imported Kubota/Krampe gates pass in 128.87 seconds. Incidental timings
  overlap correctness work and are not performance acceptance.
- Added uncommitted isolated Oxbo fixture coverage for 40 bodies, six wheels,
  roughly 20.38 tonnes. Flat-ground and viewer-terrain/alignment settling tests
  each pass both backends for ten simulated seconds. Attachment-order and live
  frame-path investigation remains ongoing; no Oxbo fix is claimed.

### Next Steps

- Reproduce the user's Oxbo Play failure with actual viewer ordering/lifecycle,
  fix the demonstrated cause and verify rendered tyre pressure changes before
  offering another playing preview. Do not equate headless settling with success.
- Check live build/test handles before starting any replacement jobs. Preserve
  every outstanding body-kind, GPU, performance, lifecycle and soil requirement.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — pose wake control.
- `crates/molla-solvers/src/{rigid_sleep,wheel_forces}.rs` and
  `crates/molla-solvers/src/wheel_forces/sleeping.rs` — held observation refresh.
- `crates/molla-solvers/tests/{rigid_settling,wheel_forces}/pose.rs` — regressions.
- Isolated `bin/gearbox/src/physics/molla/pose_wake_tests.rs` — reference check.
- Isolated `bin/gearbox/src/controller/benchmark{.rs,/oxbo.rs,/fleet.rs}`,
  `bin/gearbox/src/physics/benchmark.rs` and `bin/gearbox/src/load.rs` — pending
  real-Oxbo diagnostic fixture and actual alignment helper.
- Molla `target/perf/pose-wake-{focused-final,full}.log`; isolated
  `target/perf/pose-wake-{molla,rapier,imported}.log` — completed validation.
- Isolated `target/perf/oxbo-*.log` and `target/perf/molla-oxbo-*.log` — diagnostic
  runs, not a completed fix. `target/perf/oxbo-paused.toml` loads without auto-play.
- Engram memory APIs remain unavailable; this handoff records continuity.

## 2026-09-20: Fixed the Oxbo launch-on-Play failure

### Goal

Stop the real Oxbo from flying away on Play, then leave its pressure-enabled
preview available to the user while continuing FULL_COMPACT.md.

### Instructions

- The user confirmed they were closing the unstable test windows themselves;
  their disappearance was not evidence of a renderer crash.
- Continue using only the isolated Gearbox checkout. Do not relaunch the older
  retained scene-bounds viewer for user tests: it contains this teleport bug.

### Discoveries

- Captured the live failure: after 53 physics steps the reported machine height
  changed from about 1.93 m to 326.29 m. Render/physics transform disagreement was
  only about 0.000016 m, ruling out a camera-only/render-only explanation.
- Physics snapshots of headless and live initial states showed antipodal
  quaternions on the hopper and hydraulic linkage bodies. These represent the
  same physical rotations, but individual body teleports reseeded revolute
  coordinates through `atan2` without retaining their previous winding. Existing
  batch teleports already preserved winding. A spurious full revolution violated
  joint limits and caused explosive corrective forces.
- A focused runtime regression failed before the fix. The imported Oxbo with
  equivalent antipodal pose edits also failed before the fix on its first Molla
  step (over 3000 m/s vertical speed), while Rapier remained stable. The same
  regression passes after the fix without reducing forces, substeps or limits.
- Sequential spawn alignment also produced an approximately 18.5 mm difference
  at the middle suspension relative to the headless fixture; that separate
  uniform-alignment issue was not changed or claimed fixed here.

### Accomplished

- `f10b0b7 fix(sim): preserve joint winding on teleport` shares nearest-phase
  reseeding between single-body and batch teleports. Regression covers both wake
  flags, existing multi-turn coordinates, equivalent quaternion signs and a real
  small rotation. Combined CPU gate: 725 passed, ten ignored, 124 binaries.
  GPU-feature compilation passes, not GPU runtime parity.
- Isolated `45c263d test(physics): cover Oxbo teleport stability` adds strict
  real-Oxbo fixture dimensions (40 bodies, 46 joints, 13 machine colliders,
  six tyres, about 20.38 tonnes), real viewer-alignment test support, and three
  ten-second settling tests. Flat terrain, viewer meadow/alignment and antipodal
  teleports pass both backends. Ordinary backend lanes each pass 97 tests with
  17 ignored; the three new ignored Oxbo tests were explicitly run and passed.
- Rebuilt the actual GUI with Oslo and visually inspected the corrected playing
  Oxbo. Live readback settled at height 1.85088256 m and approximately 1.1e-13 m/s
  forward speed. Six tyres reported 1.8 bar with deflections around 0.100 m front,
  0.087 m middle and 0.069 m rear. This proves the launch regression is fixed in
  that run, not complete cross-machine tyre calibration or performance acceptance.
- Removed temporary physics snapshot instrumentation and attachment-order hooks
  from production/test code after collecting evidence. Original Gearbox and
  shared machine assets were not edited. No user process was killed.

### Next Steps

- Isolated validation session 56994 finished successfully: all ten imported
  Kubota/Krampe gates passed in 126.04 seconds. All owned build/test handles are
  terminal. These correctness timings are not controlled performance evidence.
- The corrected Oxbo preview has since exited (PID 2235934 is absent); do not
  claim it is still open or treat window closure alone as a physics failure.
  Pressure sweeps/visual acceptance across axles, surfaces
  and drive states still require their full specified verification.
- Retain all remaining FULL_COMPACT body-kind, GPU, performance, lifecycle,
  towing/service and soil requirements. Full-goal completion is unproven.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — nearest-phase reseeding on teleports.
- `crates/molla-sim/tests/runtime_pose_batch.rs` — winding/sign regression.
- Isolated `bin/gearbox/src/controller/benchmark{.rs,/oxbo.rs,/fleet.rs}` and
  `bin/gearbox/src/load.rs` — real machine and viewer-alignment regression tests.
- Molla `target/perf/pose-winding-{repro,final,full}.log`; isolated
  `target/perf/oxbo-antipodal-repro.log`, `oxbo-settling-final.log` and
  `winding-{molla,rapier,imported}.log` — failure, fix and validation evidence.
- Isolated `target/perf/oxbo-{headless,viewer}-snapshot.txt` and
  `oxbo-diagnostic-{before,after}.json` — captured initial state and live failure.
- Isolated `target/perf/oxbo-fixed2-{playing.png,playing.json,settled.json}` —
  inspected corrected GUI and readbacks; `molla-oxbo-fixed2.log` is its live log.
- Last verified user preview: `molla-oxbo | molla | PLAYING`, PID 2235934,
  isolated `target/release/gearbox launch`. It exited after the successful live
  readbacks and screenshot; a future launch must use the corrected binary below.
- Retained corrected GUI `target/perf/gearbox-viewer-oxbo-winding`, SHA256
  `b37947b5c5be8b5e1e912a5266bef41ca03e56afd05ea3b7f61f73b893769779`.
- Engram memory APIs remain unavailable; this tracked summary preserves handoff.

## 2026-09-20: Batched articulated spawn alignment

### Goal

Continue FULL_COMPACT.md by correcting the separate spawn-alignment distortion
observed while diagnosing the Oxbo, then verify the real pressure-enabled machine.

### Instructions

- Original Gearbox and shared assets remain untouched. No user process was killed.
- The user is interacting with the new `molla-alignment-check` window, including
  pressure controls and driving. Stop automation against that window and leave it
  available; do not overwrite their settings or close it as a disposable test.

### Discoveries

- The loader applied terrain translation to each body's *current* pose in a
  mutable loop. Molla's parent teleport also moves descendants, so a prismatic
  suspension could receive the same shift twice. The regression placed a knuckle
  at -0.44 m instead of the intended +0.53 m before the fix.
- Invalid projected transforms were skipped individually while other bodies
  moved and the pending marker was cleared. Missing colliders also allowed an
  initial partial relocation before the loader waited. These are separate from
  the already-fixed quaternion/winding failure.
- A validated pose batch already exists behind PhysicsBackend; use it for
  articulated group relocation rather than sequential body mutations. Collecting
  descendants once also avoids per-body repeated hierarchy traversal.

### Accomplished

- Changed isolated `load.rs` to collect the complete projected pose batch, wait
  for transforms/colliders, then apply poses atomically with velocities reset.
  Terrain adjustment captures all current target poses before one second batch;
  the scene-root translation updates only after that batch succeeds. Invalid
  sources keep the machine pending rather than partially activating it.
- Updated imported benchmark initial alignment and slope rotation to use the
  same batch API. The benchmark no longer relies on order-sensitive incremental
  descendant relocation.
- Three new loader regressions pass on Molla and Rapier: whole-suspension
  translation with both body insertion orders, invalid-transform atomicity and
  recovery/activation, and missing-transform/collider waiting without mutation.
  The first two were observed failing before the fix.
- Both ordinary backend lanes pass 100 tests (17 ignored). All three explicit
  real-Oxbo gates pass (28.77 s), and all ten imported Kubota/Krampe gates pass
  (134.34 s). The GUI builds through Oslo. All owned build/test handles are
  terminal. Incidental benchmark timings overlap correctness work and are not
  controlled performance acceptance. Molla core code was unchanged this turn.
- Inspected the real Oxbo playing after the corrected alignment. Paused readback
  showed height 1.850870575 m and effectively zero velocity. A front-axle command
  while paused accepted exactly two 0.5-bar targets while applied pressures stayed
  at 1.8 bar; the other four tyres retained their original targets at that read.
- Subsequent captures showed live UI pressure transitions and driving without
  runaway motion, but the user changed all-wheel pressure and drove during the
  attempted low/high comparison. Those captures are therefore **not** a matched
  pressure sweep: the nominal low capture had all tyres transitioning around
  3.46 bar, and the high capture had fronts at 4 bar, other axles about 0.606 bar,
  and speed about 3.8 m/s. Do not mislabel these as controlled low/high acceptance.

### Next Steps

- Keep the user's interactive Oxbo untouched. A controlled pressure sweep still
  needs isolated inputs and matched load/surface; complete every pressure UI/CLI,
  lifecycle, calibration and rendered acceptance requirement, not just this run.
- Continue remaining body-kind activation, GPU parity, end-to-end performance,
  towing/service, and Bekker/MPM work. FULL_COMPACT.md is still incomplete.

### Relevant Files

- Isolated `bin/gearbox/src/load.rs` — batched spawn/terrain alignment and wait
  behavior; `load/alignment_tests.rs` — regression fixture and recovery tests.
- Isolated `bin/gearbox/src/controller/benchmark.rs` — batched fixture relocation.
- Isolated `target/perf/spawn-alignment-{repro,final,molla,rapier,oxbo,imported}.log`
  — failing-before and passing-after evidence.
- Isolated `target/perf/alignment-oxbo-{reference,paused-target,low,high}.json`
  and reference/low/high PNGs — inspected live results, with interactive caveat.
- Isolated `target/perf/spawn-alignment-viewer.log` — active GUI log.
- Last verified live PID 2328595: `molla-alignment-check | molla | PLAYING`,
  isolated `target/release/gearbox launch`; revalidate identity before interaction.
- Retained GUI `target/perf/gearbox-viewer-batch-alignment`, SHA256
  `a1ea51fdb9174da170f30bddc6d385b044dfe68cf51448315c3597e7c77faacd`.
- Engram APIs remain unavailable; this tracked handoff preserves session context.

## Session checkpoint: tyre wrench telemetry

### Goal

- Continue the whole FULL_COMPACT.md objective, especially §6.1 physical
  readout. The previous acknowledgement-only turn made no implementation
  progress; this turn revalidated the worktrees and implemented a missing
  requirement. The full goal remains active and incomplete.

### Instructions

- Original Gearbox and shared machine assets remain untouched. Integration
  changes are only in isolated `gearbox-physics` on `feat/physics-backend`.
- The user closes Oxbo test windows manually; closures are not crash evidence.
  Prior preview PID 2328595 was absent when checked. No GUI was launched,
  closed, or manipulated in this turn.

### Discoveries

- Core wheel samples already contain the post-limiter resultant force and
  aligning moment actually used by Featherstone. The adapter previously
  dropped these, and exposed rolling resistance from only the final substep.
- Active telemetry needs duration-weighted forces/moments, whereas slip and
  contact point use normal-impulse weights. Sleeping zero-duration samples
  retain reactions without applying a fresh wrench; these must be labelled.
- The first test fixture produced zero longitudinal force in the simple-tyre
  limiter's positive-power case. The final fixture uses braking and an oblique
  guide to exercise nonzero longitudinal/lateral force and aligning moment.

### Accomplished

- Isolated commit `266ac18 feat(tyres): expose solver wrench telemetry` adds
  world-space force and aligning moment to `WheelForceOutput`, with explicit
  `held_support`. Rolling moment now has the same completed-step averaging as
  force. Retained sleeping reactions are reported once, not reintegrated.
- Controller LinkValues / machine-state telemetry now includes signed XYZ
  force, aligning and rolling moments, normal load, grip budget, slip ratio,
  slip angle, contact and held-support flags. Existing pressure telemetry and
  track properties remain available. Documented units, Y-up world frame,
  averaging, pressure/geometry timing, and sleep semantics in TYRE_PRESSURE.md.
- Three new regressions cover actual substep integration for simple/pressure
  tyres, changing timesteps, stale/contact-off clearing, sleeping/waking,
  signed LinkValues and units. Existing controller integration also checks
  paused telemetry against physical readout while a target edit is pending.
- Both ordinary backend lanes pass **103 tests, 17 ignored**. All three
  imported Oxbo gates pass (21.70 s), and all ten Kubota/Krampe imported gates
  pass (125.01 s). GUI build through Oslo succeeds (33.69 s). This is not
  new rendered acceptance, a calibrated pressure sweep, or performance parity.
- No core dynamics changes; no new Molla core/GPU test or parity claim.
  All owned process handles are terminal: focused 73359 (initial fixture
  assertion failure), focused 42422, full lanes 27760, imported/build 26850.

### Next Steps

- Finish controlled/calibrated pressure sweeps and lifecycle/UI/CLI/visual
  acceptance, now recording the actual force/moment outputs. Complete explicit
  machine metadata validation, not just heuristic legacy import defaults.
- Body-kind optional wake remains missing: core `runtime.rs::set_body_kind`
  calls global-activating `edit` even on same-kind input; isolated body.rs
  still ignores `_wake`. Preserve structural/dependency safety when fixing it.
- End-to-end performance, GPU parity, towing/replay/PTO and Bekker/MPM scope
  remain unfinished. No objective reduction or completion claim.

### Relevant Files

- Isolated `physics/backend.rs`, `physics/molla/wheel.rs` — wrench readout.
- Isolated `physics/molla/wheel_readout_tests.rs` — substep/sleep regressions.
- Isolated `controller/wheel_forces.rs` — physical telemetry and mapping tests.
- Isolated `specs/TYRE_PRESSURE.md` — telemetry contract and remaining gaps.
- Isolated `target/perf/tyre-wrench-{focused,molla,rapier,oxbo,imported}.log`
  and `tyre-wrench-viewer-build.log` — validation evidence.
- Current GUI `target/release/gearbox` SHA256
  `533afd2979c80b067244d1cd1b066b80922657b5b95072bb414b8bfdf5c30712`.
- Engram APIs are still absent from available tool metadata; this tracked
  handoff records the session instead. FULL_COMPACT.md remains unmodified
  and untracked.

## Session checkpoint: body-kind wake policy

### Goal

- Continue FULL_COMPACT.md body mutation and sleeping semantics. The previous
  goal turn was progress (committed and validated tyre telemetry). This turn
  closes the previously recorded ignored body-kind wake argument; the whole
  objective remains active and incomplete.

### Instructions

- Only Molla and isolated Gearbox were edited/built. Original Gearbox, shared
  machine assets and FULL_COMPACT.md were not modified. No GUI interaction or
  new preview launch occurred. Commits are unsigned, title-only Conventional.

### Discoveries

- `set_body_kind` rebuilt even on same-kind input and used global activation.
  Both defects were reproduced by failing core tests before the fix.
- Local Rapier 0.32 source makes same-kind input a no-op and only applies the
  explicit wake request when the new type is dynamic or kinematic. Fixing a
  body zeroes velocity without that explicit activation.
- A non-waking change does not promise indefinite sleep: a newly dynamic,
  awake boundary joins its dependents on the next step; a moving kinematic
  boundary also wakes dependent bodies. Unrelated sleepers stay asleep.

### Accomplished

- Molla `27a14b2 fix(sim): honor body kind wake policy` adds
  `RigidScene::set_body_kind_with_wake`. Same-kind calls preserve revision,
  snapshot and activation identity. Real edits rebuild atomically and activate
  only the edited non-fixed body when requested. The existing method delegates
  with wake=true. Fixed velocity clearing and physical mass/pose preservation
  remain intact.
- Isolated `559d055 fix(physics): pass body kind wake policy` connects the
  trait argument instead of ignoring it. Backend regression compares automatic
  sleepers through fixed/kinematic/dynamic round-trips against Rapier, including
  same-kind calls, subsequent stepping, impulses and unrelated sleepers.
- Core tests cover all kind pairs and wake flags, unknown-handle rejection,
  fixed zero velocity, retained pose/mass, re-rooted trees, compliant loops,
  retained sleep, awake/moving boundary propagation, prior wake preservation
  and unrelated automatic quiet windows.
- Full selected core lane: **734 passed, 0 failed, 10 ignored**, across 125
  test binaries. GPU-feature compilation succeeds; no GPU runtime-parity claim.
  Both ordinary Gearbox backend lanes: **104 passed, 17 ignored** each.
- All three explicit real-Oxbo gates pass (21.08 s). All ten Kubota/Krampe
  imported gates pass (123.07 s). GUI Oslo build succeeds (30.67 s). These
  correctness runs overlap core validation; incidental timing values are not
  controlled performance acceptance or a Rapier-parity claim.
- All owned handles are terminal: search 2611, failing repro 47925, core
  focused 22640, adapter focused 53879, core/GPU 95786, backend lanes 76247,
  imported/build 5099. A late attempt to stop the owned broad dependency search
  found it already absent; no user processes were stopped.

### Next Steps

- Continue the remaining full tyre contract: explicit metadata validation,
  controlled/calibrated pressure sweeps, combined slip/braking/coast-down,
  lifecycle and rendered UI/CLI acceptance on supported machine/surface sets.
- Complete end-to-end performance acceptance and consistent sub-1-ms/Rapier
  targets, GPU joint/tyre/sleep parity, deterministic towing/service/PTO gates,
  per-cell terrain coupling, Bekker and MPM. This body API fix is not completion
  of those requirements.

### Relevant Files

- `crates/molla-sim/src/runtime.rs` — kind mutation API and activation policy.
- `crates/molla-sim/tests/runtime_kind_wake.rs` — mutation/no-op/atomic tests.
- `crates/molla-solvers/tests/rigid_settling/kind.rs`, `rigid_sleep.rs` —
  dependency/sleep/wake tests.
- Isolated `physics/molla/body.rs`, `kind_wake_tests.rs`, `tests.rs` — adapter
  forwarding and Rapier comparison.
- Molla `target/perf/kind-wake-{repro,focused,full,gpu-check}.log` and isolated
  `target/perf/kind-wake-{focused,molla,rapier,oxbo,imported,viewer-build}.log`
  — validation evidence.
- Current isolated GUI `target/release/gearbox` SHA256
  `4b71e09086e1ad9df6a40ea53f7dfd1ef24c4e9690adca6f2b8b78553bd383fb`.
- Engram tools remain unavailable in metadata; this tracked checkpoint records
  the session. FULL_COMPACT.md remains untracked and unchanged.

## Session checkpoint: footprint-cell friction sampling

### Goal

- Continue the entire FULL_COMPACT.md objective, specifically §6.1 terrain
  coupling. The preceding user-requested divergence audit was read-only; this
  continuation implements the identified centre-only friction sampling gap.
  The full objective remains active and incomplete.

### Instructions

- No Gearbox source files changed this turn: isolated HEAD remains `559d055`,
  clean. Original Gearbox and shared assets remain untouched. Isolated builds
  and tests were run; no GUI was launched or manipulated.
- User asked how far the isolated worktree diverged. Git HEAD reflog confirms
  its origin at `6f2db09`. At audit time, isolated `559d055` had 51 commits and
  70 touched/net-changed paths: 37 modified, 30 added, 3 deleted; +12,222/-1,897
  lines, including tests, refactoring and docs. Original develop `d45154b`
  independently had 154 commits since that split. Do not characterize this as
  a small adapter-only patch or claim it was merged into the original checkout.

### Discoveries

- Vehicle brush evaluation already queries `Terrain::friction_at` per tread
  cell. The solver adapter flattened the selected ground material to one
  centre-probe scalar and the persistent entrypoint required FlatTerrain.
- A material reference based on the registered grid maximum plus collider
  fallback avoids zero-centre/high-friction-neighbour loss and ratio clipping.
  Cache the grid maximum on registration, not by scanning a whole field per
  wheel/substep. No per-cell raycasts are needed for this planar patch model.
- The current ideal brush has one centre strip: each rolling sample represents
  the whole width and has local_z=0. Width-resolved surface mixtures remain
  missing. Likewise live isolated terrain/world registration currently passes
  `None` for grids; this does not complete field authoring integration.

### Accomplished

- Molla `03fd9a8 feat(tyres): sample footprint surface friction` broadens the
  persistent planar brush to the existing Terrain trait. Pressure contact maps
  each brush cell from the local tangent plane through world space into the
  selected ground collider's local friction grid. Outside-grid lookup retains
  the collider's authored scalar friction. Simple non-pressure tyres keep their
  existing centre-probe model.
- Active and sleeping geometry-refresh paths use the same material lookup.
  Forces, aligning moment and available grip still come from the existing
  brush solve; no duplicate force path or tyre/chassis integrator was added.
- A failing-before mixed zero/high-friction regression reported 152.72 N on a
  zero-friction cell. It now passes with zero force there and the expected
  sum of load-sensitive budgets on the other cells.
- Three new tests cover that boundary, a rotated/translated/tilted collider,
  a zero-friction centre with nonzero outside-grid fallback, high-friction
  contrast, sleeping pose refresh, uniform-grid/scalar equivalence at mu=0,
  0.2 and 1.2, and invalid grid replacement without losing live output/state.
- Full selected CPU lane including molla-vehicle: **979 passed, 10 ignored,
  0 failed**, across 126 test binaries (242 vehicle tests included). GPU-feature
  compilation passes; this is not a GPU runtime-parity result.
- Both ordinary Gearbox backend lanes: **104 passed, 17 ignored** each.
  All three imported Oxbo gates pass (21.34 s); all ten Kubota/Krampe gates pass
  (121.67 s). Isolated GUI builds through Oslo (30.41 s). Incidental timings
  overlap correctness work and are not performance acceptance.
- All owned handles are terminal: repro 56326, focused 35306, 88031/79378
  (test strip initially missed all non-centre samples), 8791 (test API typo),
  final focused 85569, core/GPU 96706, backend lanes 29919, imported/build 30613.

### Next Steps

- Finish width-resolved footprint material response and live field/material
  authoring-to-grid integration; handle full surface parameters and calibrated
  pressure/load trends rather than pretending scalar friction is that model.
- Continue explicit machine metadata validation, controlled pressure sweeps,
  lifecycle/rendered UI/CLI acceptance, sub-1-ms/Rapier/end-to-end performance,
  GPU parity, towing/replay/PTO gates and Bekker/MPM coupling. None are complete
  merely because the planar friction-cell tests pass.

### Relevant Files

- `crates/molla-vehicle/src/real_tire/rig/runners.rs` — persistent brush terrain
  parameter; `real_tire` is now a module directory, not a single source file.
- `crates/molla-solvers/src/pressure_tire.rs` — tangent-plane material adapter.
- `crates/molla-solvers/src/wheel_forces.rs`, `wheel_forces/sleeping.rs` —
  registered grid reference cache and shared active/sleeping surface lookup.
- `crates/molla-solvers/tests/wheel_forces/footprint_material.rs` — regressions.
- Molla `target/perf/footprint-friction-{repro,focused,full,gpu-check}.log`;
  isolated `target/perf/footprint-friction-{molla,rapier,oxbo,imported,viewer-build}.log`.
- Current isolated GUI SHA256:
  `89950472a9035fa75eb4d3e189d00689e4454519db21717fab74b92fc394c515`.
- Engram APIs remain absent from tool metadata; this tracked checkpoint records
  the session. FULL_COMPACT.md remains unmodified and untracked.

## Session checkpoint: width-resolved brush and geometry cache

### Goal

- Continue FULL_COMPACT.md's footprint/material response without replacing its
  full scope. Previous goal turn was progress (footprint friction integration).
  This turn adds real lateral brush strips and measures/mitigates their CPU
  cost. FULL_COMPACT remains active and incomplete.

### Instructions

- Original Gearbox, shared assets and FULL_COMPACT.md remain untouched.
  Isolated Gearbox source remains clean at `559d055`; only binaries/tests were
  rebuilt there. No test window was launched, manipulated or closed.
- Do not count incomplete/narrow GPU or performance checks as full acceptance.

### Discoveries

- Brush contact had one full-width cell per circumferential sample. Its yaw
  moment only included lateral traction along the rolling coordinate; split
  left/right friction could not generate longitudinal-traction yaw leverage.
- Naively repeating the geometry per strip raised full-meadow driving median
  cost by 10.61%. Caching all reference cell coordinates (including lateral
  positions, not just trigonometry) reduced the remaining increase to 3.41%.
- Geometry depends on unloaded radius, width and both sample counts, not on
  inflation/load. Cache immutable reference coordinates; retain independent
  bristles and recompute penetration, loads and forces at each step.
- Existing GPU brush inputs still have no width-resolution field. Their seven
  existing tests pass, but they do not prove width-varying-material/yaw parity.
  Do not silently claim the new CPU feature is implemented by that shader.

### Accomplished

- Molla `f718780 feat(tyres): resolve brush contact across width` adds public
  `RealTireParams::width_segments`, default 3. Standalone brush sanitizes to
  1..=16; solver configuration rejects outside that range atomically.
  Circumferential and width samples form equal-area cells at strip midpoints.
  The true midpoint of odd-width grids is exactly zero in local coordinates.
- Persistent shear has one state per cell; geometry identity includes width
  resolution. Material, film and slip queries use both rolling and lateral
  positions. Aligning torque includes `z * longitudinal_force - x * lateral_force`
  (and the longitudinal rolling term where the reusable brush applies it).
  The existing Featherstone force path applies that torque; no extra body or
  wheel-spin integration was added.
- Airborne cells skip force/material calculations. Parameter copies/presets,
  cell telemetry docs and GPU-test parameter literals were updated. Obsolete
  one-strip telemetry assertions now verify four requested lateral bins,
  nonzero width and conserved lateral load, rather than requiring width=0.
- Molla `f6abd31 perf(tyres): cache brush cell geometry` stores immutable
  reference coordinates in an Arc shared across step-history clones. Geometry
  changes rebuild/reset it; pressure-only changes retain it. Invalid steps
  preserve state/cache identity; explicit reset clears both.
- A failing-before test demonstrated all brush cells were at local_z=0.
  Four vehicle regressions now cover both shoulders, independent split-surface
  shear/yaw, uniform-surface load/area/force conservation at 1/2/3/8/16 strips,
  and resolution/cache/rollback semantics. One world integration regression
  verifies asymmetric friction causes actual chassis yaw without duplicate
  ground contacts, and rejects invalid resolution without partial mutation.
- Final selected CPU lane (including vehicle): **984 passed, 10 ignored,
  0 failed**, 126 binaries. GPU-feature check passes. The seven explicit
  existing GPU brush tests pass through nixVulkan, with no adapter-skip messages.
  These remain narrower than complete new width-resolved GPU parity.
- Final isolated backend lanes: **104 passed, 17 ignored** each. All three
  Oxbo gates pass (21.90 s); all ten Kubota/Krampe gates pass (121.59 s).
  GUI Oslo build passes (29.22 s). No rendered acceptance was performed.

### Controlled CPU comparison

- Two separate A/B/B/A comparisons used retained binaries, the same 801×801
  meadow, full imported Kubota, solver settings unchanged, 600 warmup and 1200
  samples per parked/driving lane. All owned builds/tests were terminal before
  each comparison; no competing owned validation ran during timing. Each suite
  also runs Rapier/Molla/Molla/Rapier internally. All eight suites passed.
- Initial width feature: mean Molla driving medians **0.963860 → 1.066130 ms**,
  +10.61%; Rapier **0.2859775 → 0.2867435 ms**. This regression prompted caching.
- Final cached feature vs one-strip baseline (means of four medians each):

| Lane | Baseline ms | Cached three-strip ms |
|---|---:|---:|
| Molla parked | 0.959766500 | 0.991887750 |
| Molla driving | 0.967456750 | 1.000471250 |
| Rapier parked | 0.278610000 | 0.280224500 |
| Rapier driving | 0.287113750 | 0.288903000 |

- Final Molla driving cost is +3.41%, **3.463× Rapier**. Mean driving p95 is
  1.285393 ms (baseline 1.300234 ms); tail variability does not establish a
  speedup. Highest driving median is 1.004739 ms. Consistent sub-1-ms stepping,
  tails, Rapier parity and full-viewer frame performance are NOT achieved.

### Next Steps

- Implement/test GPU width sampling and asymmetric-material yaw explicitly;
  the old shader/input lacks the new sample dimension despite legacy tests passing.
- Wire authored field material grids into the live isolated ground registration
  (currently None), then complete calibrated pressure/material response, full
  footprint/terrain geometry and normal-load distribution as required.
- Continue explicit machine compatibility metadata, controlled lifecycle/UI/CLI
  and rendered pressure acceptance, CPU/frame performance, joint/GPU/sleep parity,
  towing/replay/PTO and Bekker/MPM coupling. Do not shrink the original objective.

### Relevant Files

- `crates/molla-vehicle/src/real_tire/{params,brush,bristle,contact}.rs` — width
  resolution, bounded sampling and persistent/cache state.
- `real_tire/rig/runners.rs`, `rig/sweeps.rs`, `reports/vehicle.rs` — brush solve
  and parameter propagation; `width_tests.rs`, `tests.rs` — regression coverage.
- `crates/molla-solvers/src/pressure_tire.rs` — resolution validation;
  `tests/wheel_forces/footprint_material.rs` — actual rigid-body yaw test.
- `crates/molla-kernels/tests/tire_brush_gpu.rs` — updated parameter literals;
  kernel/shader width implementation remains outstanding.
- Molla `target/perf/brush-width-{repro,focused,integration,full-initial,full,gpu-*}.log`
  and `brush-width-cache-{focused,full,gpu-*}.log` — validation history.
- Isolated `target/perf/brush-width-{cache-,}{molla,rapier,oxbo,imported,viewer-build}.log`
  plus `brush-width-meadow-*.log` and `brush-width-cached-meadow-*.log` — tests/timings.
- Retained isolated test binaries `gearbox-brush-width-before` SHA256
  `3b24ee0542576c61e4807b2f1b8a9f172fba3c5953689e9837b3c26d99db755d`,
  `gearbox-brush-width-after` SHA256
  `827a7103b2d4d52ffd8afa721bea1e207305527e9b646b4e77ff995eba200a9d`,
  and final `gearbox-brush-width-cached` SHA256
  `8bc7226c22384843c6c4832da58513c2c730c010efdc2d06b317bb3a54e3edfa`.
- Current isolated GUI SHA256
  `b6f65e23496c0533e7f1f0aba12ff514fd9783a36d081b89dea0dee046d3b6e0`.
- All owned handles are terminal. Final core/GPU handle 46040, final imported/
  viewer-build 91048, timing handles 40295 and 50865; earlier focused/build/test
  handles also returned terminal results. Engram APIs remain unavailable; this
  tracked checkpoint records session context.

## 2026-09-20 — GPU width sampling and stopped-wheel resistance

### Goal

Continue FULL_COMPACT.md without reducing its scope. This turn addressed the
GPU brush width gap identified by the preceding CPU width implementation.
The full objective remains active, not complete.

### Instructions

- Original Gearbox remains read-only; no source changes in either Gearbox
  worktree this turn. Isolated Gearbox was rebuilt/tested against changed Molla.
- Preserve make-based Molla validation, unsigned title-only commits, and
  apply_patch edits. FULL_COMPACT.md remains unchanged/untracked.

### Discoveries

- New regressions failed on the old GPU shader: shoulder-specific grip gave
  zero yaw moment, and contact counts omitted width strips. The seven older
  passing brush tests did not prove width parity.
- CPU `f64::signum()` at zero produced a directional rolling-resistance force.
  Initially mirroring it on GPU broke a resident FEM/vehicle substep test.
  Both paths now return zero rolling resistance at zero longitudinal speed;
  positive and negative zero are tested, including persistent CPU state.
- Rapier is not an exclusively impulse-joint engine. Original Gearbox at the
  fork point and its current checkout both contain MultibodyJointSet insertion
  with impulse fallback; original activation is opt-in via
  `GEARBOX_MULTIBODY=1`, otherwise joints use the impulse path. Isolated Rapier
  retains multibody support. Molla's RigidWorld uses
  FeatherstoneSolver::new_cpu() with eight substeps. Solver family alone does
  not establish superiority. Official reference:
  https://rapier.rs/docs/user_guides/rust/joint_constraints/

### Accomplished

- `85faa98 fix(tyres): stop rolling force at zero speed`.
- `4dae6c5 feat(gpu): sample tyre contact across width`.
- GPU width strips sample geometry/material independently, conserve cell area,
  and accumulate `z*(longitudinal-rolling)-x*lateral` yaw moment. Rolling
  resistance uses each cell's tangent instead of a single global subtraction.
- Kept BrushWheelInput at seven vec4 rows: hub_pos.w carries width resolution,
  integers 1..16, zero selecting three strips. Host inputs reject invalid
  values; the shader bounds GPU-resident counts. Scratch is per workgroup lane,
  not per circumferential sample. Existing circumferential GPU maximum is 256.
- Direct vehicle row builders expose per-wheel width_segments; all four
  resident spin/suspension/chassis builders forward wheel_controls.w. Updated
  Molla viewer packing. No alternative Gearbox chassis integrator was added.
- Final validation: core **985 passed, 10 ignored**, 126 binaries; kernel
  library **8 passed**; vehicle host/reference **15 passed, 21 GPU ignored**;
  explicit GPU brush **12 passed**; GPU vehicle **21 passed**; selected FEM
  brush consumers **3 passed** and resident FEM owner **1 passed**. No GPU
  adapter skips. Viewer/kernels all-targets check passed.
- GPU tests cover 1/2/3/4/8/16 strips, 8/48/96/256 circumferential samples,
  airborne wheels, asymmetric grip with reversed travel/heading/material sides,
  80/300 kPa, shoulder heights and rolling moment, zero speed, invalid counts,
  and GPU-resident row generation through brush output/readback.
- Isolated ordinary backends: **104 passed, 17 ignored each**. Imported Oxbo:
  **3 passed**, 21.35 s. Kubota/Krampe: **10 passed**, 126.66 s.
- All owned handles terminal (73005 core/GPU/FEM, 2061 final kernel checks,
  24461 isolated backends/imports; earlier handles also terminal). No GUI launch
  or rebuild this turn, no rendered acceptance and no new performance timing.
  Engram APIs remain unavailable; this checkpoint persists the session.

### Next Steps

- Full GPU parity remains unproven: persistent shear, general terrain tangent
  frames/camber, the CPU 512-vs-GPU 256 circumferential limit and full rigid
  GPU integration still need coverage/work. Width tests are not blanket parity.
- Publish authored field friction grids into live isolated ground registration,
  finish calibrated pressure/material response and full footprint geometry.
- Continue compatibility metadata, lifecycle/UI/CLI and rendered acceptance,
  joints/towing/replay/PTO, sleep/GPU integration, Bekker/MPM and performance.
- Last controlled CPU timing remains the preceding cached-width comparison:
  Molla driving 1.000471 ms vs Rapier 0.288903 ms, 3.463x. It predates this
  zero-speed fix; no newer timing or claim of Rapier parity is established.

### Relevant Files

- `crates/molla-kernels/{src/tire_brush.rs,shaders/tire_brush.wgsl}` — width
  contract, geometry/force sampling and per-cell rolling/moment reduction.
- `crates/molla-kernels/{src/tire_vehicle.rs,shaders/tire_vehicle_yaw.wgsl}` —
  CPU and GPU-resident width propagation.
- `crates/molla-kernels/tests/tire_brush_gpu/width.rs`, `tire_brush_gpu.rs`,
  `tire_vehicle_gpu.rs` — width/material/geometry/motion/validation regressions.
- `crates/molla-vehicle/src/real_tire/{rig/runners.rs,width_tests.rs}` — stopped
  rolling-force correction and stateless/persistent regression.
- `bin/molla_viewer/src/examples/121_tire_traction.rs` — width packing.
- Molla `target/perf/gpu-brush-width{,-repro,-unit,-vehicle,-check,-core,-fem,
  -fem-fixed}.log`; isolated `target/perf/gpu-brush-width-{molla,rapier,oxbo,
  imported}.log` — final and deliberately retained failing evidence.

## 2026-09-20 — Field friction publication in progress

### Goal

Continue FULL_COMPACT.md by connecting authored surface friction to the live
isolated Gearbox tyre backend. Previous goal turn made concrete progress on
GPU width and zero-speed resistance. The full goal remains active.

### Instructions

Original Gearbox and shared assets remain read-only. Changes below are ONLY in
gearbox-physics, currently uncommitted. No subagents, Python edits, or GUI launch.

### Discoveries

- Contrary to the plan's historical wording, field layouts/profile structs in
  this worktree had appearance and region metadata but no physical friction.
  All live ground registrations passed None despite Molla supporting a grid.
- Field rendering supports mixed regions on procedural terrain only. Existing
  USD adapter is material-only and flat ground has no published field regions.
- Runtime field geometry is startup-only (`ensure_fields` returns when root is
  unchanged). The new publisher therefore permits friction-value edits but
  rejects geometry/profile changes until terrain reload, avoiding visual/physics
  disagreement.
- Original Gearbox's opt-in multibody comment records a Rapier 0.32 Jacobian
  indexing failure during multi-frame assembly and missing two-axis joints.
  These are recorded reasons, not independently reproduced in this turn.

### Accomplished

- Added optional FieldLayout.default_friction and FieldSpec.friction, finite
  nonnegative validation, authored zero preservation, inheritance to collider
  friction, deterministic half-open region boundaries and row-major samples.
  No guessed grass/stubble coefficients: bundled layouts still use scalar
  collider friction without allocating a map.
- Added TerrainPlugin publisher after CoverUpdates/before physics stepping.
  It registers collider-local XZ maps on generated world-aligned procedural
  terrain, caches terrain/grid identity and change ticks, retains valid maps on
  rejected edits, clears overrides on removal, and skips Rapier.
- Added real Molla pressure-wheel ECS integration tests proving zero vs nonzero
  grip with normal support, no duplicate ground contacts, target-pressure
  preservation after invalid edits, reset/removal fallback and no republishing
  after unrelated spawns. Documented limits in fields README/TYRE_PRESSURE spec.
- Fields library: **11 passed** (`cargo test --release -p gearbox-fields --lib`).
  Live publisher: **4 passed** (`cargo test --release -p gearbox-sim --bin gearbox
  terrain::friction:: -- --nocapture --test-threads=1`). Commands used the Nix
  environment and CARGO_BUILD_JOBS=4. Initial publisher compile used the wrong
  PhysicsWorld.step signature; corrected to step(), rerun passed.
- All handles terminal: 49896 initial layout/compile (compile error), 95988
  corrected publisher tests (success). No live owned jobs or timing runs.
  Engram tools remain unavailable; this checkpoint preserves work in progress.

### Next Steps

- Review and finish publication lifecycle tests (replacement ground/grid and
  fallback material changes), then run ordinary isolated backend tests, imported
  Oxbo/Kubota/Krampe gates and native Oslo GUI build. New modules may be formatted
  individually; do not format whole legacy source files.
- Benchmark enabled authored grids on full-resolution terrain without concurrent
  owned builds/tests. No performance evidence for this new path yet.
- Commit validated isolated changes unsigned/title-only. Update this checkpoint
  with final outcomes. Current isolated HEAD is still 559d055.
- USD/flat authoring publication, calibrated pressure/surface trends, explicit
  compatibility metadata, full GPU parity, rendered/UI/lifecycle acceptance,
  CPU/frame performance, towing/replay/PTO and Bekker/MPM remain incomplete.

### Relevant Files

- Isolated `crates/gearbox-fields/src/layout.rs` and new
  `layout/friction_tests.rs` — authoring, validation, grid samples and tests.
- Isolated `bin/gearbox/src/terrain.rs` and new `terrain/friction.rs`,
  `terrain/friction/tests.rs` — live publication/cache and real backend tests.
- Isolated `crates/gearbox-fields/README.md`, `specs/TYRE_PRESSURE.md` — schema,
  interpolation, inheritance, update behavior and explicit limitations.
- Isolated `target/perf/field-friction-layout.log` and
  `field-friction-publication.log` — successful focused validation.

### Field publication follow-up before multibody clarification

- Added recovery after replacement terrain initially has invalid metadata,
  inherited collider-friction updates and height-grid replacement tests.
  Accepted layout identity now carries its terrain entity independently of
  attempted publication, so rejected edits do not poison replacement recovery.
- Isolated ordinary lanes now **110 passed, 18 ignored each** (Molla/Rapier).
  Includes six live publication tests. Native Oslo GUI build passed in 39.05 s.
  Handle 62915 terminal; earlier 96410 ended with missing remove_collider wake
  argument, corrected before the successful full lanes. No live owned jobs.
- Added ignored `controller::benchmark::terrain::imported_kubota_meadow_field_friction_timing`.
  Same full-resolution 801x801 terrain, warmup/sample counts and actual drive
  assertions as scalar timing; uses a uniform authored grid matching baseline
  friction to measure sampling cost without changing the motion case. Not run
  yet. Imported Oxbo/Kubota/Krampe gates and controlled timings remain next.
- New modules formatted individually via rustfmt; original Gearbox untouched.
  All isolated edits still uncommitted; core source unchanged this turn.
- Molla multibody clarification: RigidWorld always constructs FeatherstoneSolver;
  its articulated-tree path is already active. The optional reduced flag selects
  Rapier's joint implementation, not whether Molla uses articulated dynamics.

### User scope update: Molla replaces Rapier

- User explicitly instructed: do not spend time developing Rapier. Once Molla
  is comparable in performance, Rapier is to be removed completely.
- Concentrate implementation, debugging and optimization on Molla. Retain
  Rapier only as the performance comparison baseline until the user's removal
  condition is met. Do not add Rapier features/fixes or repeat separate Rapier
  validation lanes unnecessarily. Do not remove it prematurely or touch the
  original Gearbox checkout.
- Latest goal progress: imported Oxbo (3) and Kubota/Krampe (10) gates passed;
  handle 16618 terminal. Those gates were initiated before this scope update.
  No owned jobs currently live. Another user-owned Oslo build/run was observed
  and left alone; check competing load before accepting performance timings.
- Review added an unchanged-value layout cache guard and moved friction-grid
  size validation before the no-map fast path. These final changes still need
  the Molla/fields test lane and native viewer rebuild before committing.
  Do not claim the earlier binary validated those later edits.
- The field-friction scalar-vs-grid timing comparison is still pending. All
  isolated field changes remain uncommitted. Engram APIs are unavailable.

### Field publication final lifecycle validation

- Added a resource-removal regression covering both FieldLayout and FieldProfiles.
  Removing authoring resources restores scalar friction but must not forget the
  rendered terrain's accepted region geometry. Re-adding changed regions is
  rejected until terrain reload; restoring the original regions must recover.
- The first run caught a cache bug: comparing a restored layout to the last
  accepted layout could skip publication after a rejected attempt. The attempt
  cache now stores its own layout value, independent of accepted region identity.
  This preserves identical-write fast paths without skipping recovery.
- Fields library final lane: 11 passed. Isolated Molla final ordinary lane:
  111 passed, 18 ignored, including seven live field-publication regressions.
  Initial handle 46099 failed the new regression (110 passed, one failed);
  corrected tests passed in handle 87167, whose native GUI build is still running.
  No separate Rapier correctness lane was repeated after the scope update.
- Logs: isolated target/perf/field-friction-{layout,molla,viewer}-final.log.
  User-owned original Gearbox GUI/build activity remains untouched; performance
  timings must account for competing user load and wait for owned builds to end.
  Engram APIs remain unavailable; this file is the project handoff.

### Session summary: authored field friction delivered

#### Goal

- Continue FULL_COMPACT.md, including actual terrain-material coupling to the
  pressure tyre model and its enabled-model CPU performance gate.

#### Instructions

- Spend development effort on Molla; Rapier is only the performance baseline
  until comparable performance permits removal. Original Gearbox remains
  untouched. No premature backend removal or separate Rapier feature work.

#### Discoveries

- Rejected publication attempts and accepted render-region identity must have
  separate layout caches. Otherwise restoring valid settings after resource
  removal can incorrectly retain scalar fallback. The new regression failed
  before this correction and passes afterwards.
- Uniform authored maps preserve the scalar benchmark's logged pressure,
  deflections, normal loads, speed and chassis position exactly in these runs.
  Their enabled sampling cost is small compared with the remaining solver cost;
  this does not establish heterogeneous-surface calibration or GPU parity.

#### Accomplished

- Isolated commit `bee6719 feat(tyres): publish authored field friction` includes
  schema/validation, live procedural publication, lifecycle caching, seven live
  Molla regressions, three layout tests, documentation and the timing fixture.
  Isolated worktree clean after commit. No original-checkout edits.
- Final fields: 11 passed. Molla binary suite: 111 passed, 18 ignored.
  Native Oslo release/profile viewer build passed in 40.85 s. Handle 87167
  terminal. Viewer rebuilt, not launched; no new rendered acceptance claim.
- Timing handle 22235 terminal: sequential scalar/grid/grid/scalar, each with
  R/M/M/R control order; 600 warmup and 1200 samples per parked/driving phase.
  Full 801x801 terrain, 120 Hz, existing 16 solver / 4 internal iteration settings;
  no resolution, accuracy or substep reduction. All four runs passed.

  | Mean of four medians, ms/step | Scalar | Authored uniform grid |
  | --- | ---: | ---: |
  | Molla parked | 0.993225500 | 1.009362000 |
  | Molla driving | 1.002627500 | 1.022466500 |
  | Rapier parked control | 0.280942000 | 0.281805250 |
  | Rapier driving control | 0.291173000 | 0.289433000 |

- Driving grid overhead measured about 1.98%; grid Molla remains about 3.53x
  its scalar Rapier control. Molla driving mean p95: 1.507243750 scalar,
  1.658911500 grid. No performance parity or consistent sub-1-ms claim.
- No owned builds/tests overlapped timings. The user's original debug viewer
  remained active (roughly 176-181% lifetime CPU in process snapshots), untouched.
  Treat these as loaded-host comparisons, not quiet-host tail acceptance.
- Timing executable SHA256:
  `5927b0f4e04754414d27260f5b4c38305f8292e0edac34442fe7e60aabf52de6`.
  All owned handles terminal; no owned viewer. Engram tools unavailable.

#### Next Steps

- Re-profile current Molla on the full meadow to select the next material CPU
  optimization (loop projection, mass/FK work), preserving correctness and
  unchanged solver quality; do not optimize Rapier or remove it yet.
- Procedural field friction is implemented, but USD/flat publication, physical
  surface calibration, full footprint/load-distribution realism, GPU parity,
  rendered pressure/lifecycle acceptance, towing/replay/PTO and Bekker/MPM
  remain outstanding. FULL_COMPACT.md is not complete; goal remains active.

#### Relevant Files

- Isolated `bin/gearbox/src/terrain/friction.rs`, `terrain/friction/tests.rs` —
  live publication and lifecycle regressions.
- Isolated `crates/gearbox-fields/src/layout.rs`, `layout/friction_tests.rs` —
  validated field-friction schema and rasterization.
- Isolated `bin/gearbox/src/controller/benchmark/terrain.rs` — matched map timing.
- Isolated `crates/gearbox-fields/README.md`, `specs/TYRE_PRESSURE.md` — contract.
- Isolated `target/perf/field-friction-*-final.log`,
  `field-friction-timing-{scalar-a,grid-a,grid-b,scalar-b}.log`,
  `field-friction-load-{before,after}.txt`, `field-friction-timing-binary.sha256`
  — final validation, timings and competing-load evidence.

### Profiling checkpoint before Oxbo asset clarification

- Goal: continue Molla-only performance work without changing solver quality.
- Added an ignored profile-feature `imported_kubota_meadow_trace` in isolated
  controller/benchmark/terrain.rs (uncommitted). Uses current full meadow,
  Molla only, 600 settling + 600 driving warmup, 120 traced steps, real tyre and
  speed assertions. Handle 9778 terminal: trace test passed; profile build 19.53 s.
- Current trace inclusive ms/frame: rigid_step 1.308964, external_forces 0.554715,
  Featherstone substeps 0.471727, collision 0.250936, loop_forces 0.419029,
  drive_inertias 0.119581, wheel_forces 0.116060. Tracing adds overhead: these
  are hotspot diagnostics, not replacement acceptance timings. Loop projection,
  repeated mass construction and ABA remain optimization candidates; no new
  solver optimization or weakened settings has been applied this turn.
- Artifacts: isolated target/perf/molla-meadow-current-{trace.json,trace.log,
  profile.tsv}. No live owned jobs. Next: choose and validate a measured
  optimization, then commit the trace fixture with its evidence.
- User asked whether earlier Oxbo fixes changed USD assets. Verified commits
  b9421af and f10b0b7 modify loader alignment and Molla joint winding only.
  Prior checkpoints explicitly record shared assets untouched. Machines directory
  is not a Git repository, so Git cannot independently prove its historical state.
  No asset writes performed during this work. Engram APIs unavailable.

### Session summary: stack-backed Featherstone joint solves

#### Goal

- Continue FULL_COMPACT.md performance work using the current full meadow,
  without weakening solver accuracy, tyre resolution or substeps.

#### Instructions

- Only Molla development; Rapier stays a timing control. Original Gearbox,
  shared assets and user-owned processes remain untouched.

#### Discoveries

- Rigorous multi-DOF ABA allocated heap vectors for blocks bounded by six DOFs,
  including six separate inertia-response right-hand sides per joint/substep.
  Fixed-size stack storage removes that overhead without changing arithmetic.
- Traced phase durations include instrumentation and changing host load. Use
  uninstrumented matched runs for speed claims, not total traced frame duration.

#### Accomplished

- Core `757896f perf(solver): stack-allocate joint block solves`: stack-backed
  packed matrices, RHS vectors and triangular substitutions in both rigorous
  ABA passes. Existing singular fallback, pivot threshold and summation order
  retained. Large articulation mass solves keep their existing dynamic storage.
- Added bit-exact reference comparison for widths 0 through 6, regular/negative/
  tiny/zero pivots, cutoff equality, infinities/NaNs, and poisoned unused lanes.
- Solver release suite: **495 passed, 9 ignored, 91 binaries**. Isolated Molla:
  **111 passed, 18 ignored**. Oxbo: **3 passed**. Kubota/Krampe imported gates:
  **10 passed**. GPU-feature all-targets check passed (compilation, not GPU parity).
  Native Oslo viewer build passed in 28.39 s; not launched.
- Isolated `6ae3b7d test(perf): trace Molla on full meadow` adds the Molla-only
  profile fixture. Final candidate trace passed with matching before/after
  logged machine state. Profiling is diagnostic, not viewer FPS acceptance.
- Sequential A/B/B/A for scalar and enabled uniform-map terrain, each execution
  containing R/M/M/R controls. Full 801x801 meadow, 120 Hz, unchanged settings,
  600 warmup and 1200 samples per parked/driving phase. Eight runs passed.

  | Mean of four medians, ms/step | Before | Stack blocks |
  | --- | ---: | ---: |
  | Scalar Molla parked | 0.980823750 | 0.956727250 |
  | Scalar Molla driving | 0.994046250 | 0.964394500 |
  | Scalar Rapier driving control | 0.286014750 | 0.287320500 |
  | Grid Molla parked | 1.002469250 | 0.990606500 |
  | Grid Molla driving | 1.007298500 | 0.986846750 |
  | Grid Rapier driving control | 0.286130750 | 0.285825250 |

- Driving reduction: about **2.98% scalar / 2.03% grid**. Logged Molla pressure,
  load, deflections, speed and position match exactly between each corresponding
  before/after run. All eight candidate driving medians were below 1 ms, but
  tails were not: mean driving p95 1.192419 ms scalar / 1.385355 ms grid.
  Molla remains about 3.36x / 3.45x the respective Rapier control; not comparable.
- No owned builds/tests overlapped timings. Process snapshot before timings
  contained no original viewer; afterwards the user's original debug viewer
  was running. It was not stopped. No quiet-host/tail guarantee is claimed.
- All owned handles terminal: 3731 solver, 18221 imported validation, 49798 GPU
  check/viewer build, 57775 timings, 47064 candidate trace. No owned viewer.
  Engram APIs unavailable; this checkpoint persists the handoff.

#### Next Steps

- Further reduce actual solver/collision cost using measured phases, preserving
  exact dynamics where possible. Candidate: avoid root-only articulated-inertia
  propagation that has no parent, after verifying caches needed by forward solve.
  This candidate is not implemented or accepted yet.
- Continue remaining complete-plan requirements: joint/GPU parity, calibrated
  pressure/surface/footprint physics, lifecycle/UI/rendered acceptance, towing,
  replay/PTO, USD/flat material publication, Bekker and MPM. Goal remains active.

#### Relevant Files

- Core `crates/molla-solvers/src/featherstone.rs` — bounded stack solves and oracle.
- Core `target/perf/joint-block-stack-{solvers,gpu-check}.log` — validation.
- Isolated `bin/gearbox/src/controller/benchmark/terrain.rs` — trace fixture.
- Isolated `target/perf/joint-block-stack-{molla,oxbo,imported,viewer}.log`.
- Isolated `target/perf/joint-block-stack-{scalar,grid}-{before-a,after-a,after-b,before-b}.log`,
  `joint-block-stack-timing-summary.txt`, `joint-block-stack-load-{before,after}.txt`.
- Isolated retained `target/perf/gearbox-small-block-{before,after}`; SHA256s in
  `joint-block-stack-binaries.sha256`: before 5927b0f4e04754414d27260f5b4c38305f8292e0edac34442fe7e60aabf52de6,
  after 5dc39342c53890c07b496931e4025fe17242f023580e4f1e53a4f2dfaca7d192.
- Isolated `target/perf/joint-block-stack-{trace.json,trace.log,profile.tsv}` —
  candidate diagnostics, alongside molla-meadow-current-* baseline artifacts.

### Root-propagation candidate checkpoint

- Uncommitted candidate skips successful rigorous root-block inertia/bias
  propagation only after U and Cholesky caches required by the forward pass
  are populated. A private const-generic reference mode retains the old work.
  No joint quality, stiffness, timestep or substep settings changed.
- Added full state oracle for 120 substeps, single/global and three/local
  articulations, free/ball/D6 roots, internal multi-DOF links, partial locks,
  gravity and applied wrenches. Valid positive-inertia chains and centred
  zero-inertia point masses match the old path exactly; focused cases pass.
- Important separate finding: the initial zero-inertia fixture with offset
  frames becomes non-finite in BOTH optimized and original propagation paths.
  Reducing to one offset point mass still fails by step 10. This is NOT fixed
  by eliding root propagation. The accepted singular oracle uses coincident
  root/COM frames instead; do not claim general offset-singular stability.
  Reproducer: featherstone_root_tests.rs singular=true with frame restored to
  Transform::new(Vec3::new(root*2,0.4,-0.2),Quat::from_rotation_x(0.2)) and
  child_frame translate(0.03,0.1,-0.02), zero inertia and nonzero applied torque.
  Evidence: target/perf/root-propagation-offset-singular-failure.log. Investigate
  this separately against rigid-world singular-response semantics before full
  plan acceptance; do not weaken finite-state requirements to hide it.
- Solver suite handle 74653 and isolated validation/build handle 72178 are live
  at this checkpoint. Earlier focused handles 96596/50015 terminal (both failed
  on the offset-singular fixture). Candidate is not committed or timed yet.
  Engram APIs unavailable; this file preserves the discovery and next work.

### Session summary: root-propagation candidate rejected

#### Goal

- Continue FULL_COMPACT.md with measured Molla performance work, preserving
  physics fidelity and the original Gearbox/assets boundary.

#### Discoveries

- The rigorous solver can omit unused root-to-parent propagation without
  changing finite motion in the tested cases, but its full-machine timing
  effect is too small/noisy to justify the additional production branch and
  reference specialization. Do not repeat this candidate as a proven speedup.
- Offset zero-inertia point-mass instability occurs in both old and candidate
  raw Featherstone paths. See the preceding checkpoint and retained failure log.
  Centred singular fixtures passing is not proof of the offset case. This is
  a newly recorded unresolved correctness issue, not caused/fixed by this work.

#### Accomplished

- Candidate verification: solver **497 passed, 9 ignored**, 91 binaries;
  isolated Molla **111 passed**, Oxbo **3 passed**, Kubota/Krampe **10 passed**.
  GPU-feature all-targets compilation and native viewer build passed.
- Eight matched before/after timing runs passed, scalar and uniform-map full
  801x801 terrain, 120 Hz and all settings unchanged, A/B/B/A with internal
  R/M/M/R controls. Candidate and reference logged states match exactly.

  | Mean of four medians, ms/step | Stack-block baseline | Root candidate |
  | --- | ---: | ---: |
  | Scalar Molla parked | 0.975387250 | 0.971429500 |
  | Scalar Molla driving | 0.982362000 | 0.985930500 |
  | Scalar Rapier driving control | 0.294857500 | 0.295612000 |
  | Grid Molla parked | 0.996839000 | 0.996225000 |
  | Grid Molla driving | 1.005938750 | 1.003258250 |
  | Grid Rapier driving control | 0.291685000 | 0.291841500 |

- Scalar driving regressed 0.36%; grid improved 0.27%. These are not a reliable
  benefit. User-owned original GUI launches overlapped the host environment,
  while no owned builds/tests overlapped timings. User processes untouched.
- Reverted ONLY this turn's candidate source and new oracle module via
  apply_patch. `git diff --exit-code -- crates/molla-solvers/src/featherstone.rs`
  confirms exact return to committed stack-block implementation 757896f.
  Candidate patch and oracle retained as ignored diagnostics, not production.
- All candidate handles terminal: 96596/50015 failed focused offset-singular
  oracles; 74653 corrected/full solver, 72178 integration/viewer, 76912 GPU check,
  17511 timing. Restored-source native viewer rebuild handle 41493 still live
  at this checkpoint. No owned GUI. Engram APIs unavailable.

#### Next Steps

- Finish/poll restored viewer build and refresh ordinary Molla test binary so
  target/release outputs do not remain the rejected candidate.
- Investigate offset singular-inertia behavior with a minimal reproducible case
  in raw Featherstone and RigidWorld, including compatibility of applied torque
  with locked/singular angular modes. Do not assume the public adapter shares
  the raw-path failure without reproducing it.
- Continue larger measured solver/collision optimizations and the full remaining
  pressure/material/GPU/lifecycle/rendering/towing/replay/soil requirements.
  Goal remains active and incomplete; no performance-parity claim.

#### Relevant Files

- Core `target/perf/root-propagation-candidate.patch`,
  `root-propagation-oracle.rs`, `root-propagation-offset-singular-failure.log` —
  rejected candidate and unresolved edge-case evidence.
- Core `target/perf/root-propagation-{solvers,gpu-check}.log` — validation.
- Isolated `target/perf/root-propagation-{molla,oxbo,imported,viewer}.log`.
- Isolated `target/perf/root-propagation-{scalar,grid}-{before-a,after-a,after-b,before-b}.log`,
  `root-propagation-timing-summary.txt`, `root-propagation-load-{before,after}.txt`.
- Isolated `target/perf/gearbox-root-propagation-after` SHA256
  64de5bd3693fb9b582fd93e88735063606267d4d370d86e088af2ecbffdd7f1b;
  this is a rejected experimental binary, not the current implementation.
- Isolated `target/perf/root-propagation-restored-viewer.log` — restoration build.

### Status clarification after candidate restoration

- Restored-source native viewer rebuild handle 41493 is terminal: passed in
  35.50 s. No owned jobs remain. Ordinary non-profile test executable still
  needs refreshing before it is reused; retained accepted stack-block binary
  remains the correct performance baseline, not the root candidate.
- User requested overall progress: working integration, not replacement-ready.
  Latest accepted optimization is stack-backed joint solves, not the rejected
  root candidate. Typical one-machine physics remains roughly 1 ms versus
  0.29 ms baseline; tails/full-viewer performance and full-plan acceptance are
  incomplete. Do not provide an invented completion percentage.

### Singular continuous-force correction checkpoint

- Confirmed the offset zero-inertia failure through public RigidWorld, not only
  raw Featherstone. With mass 2, COM (0.03,0.1,-0.02), zero inertia and a small
  applied wrench, frame 2 produced COM/angular velocities around 1e22/1e23.
  BodyParams explicitly documents Mat3::ZERO as skipping angular dynamics;
  this is a real supported-case defect, not merely an invalid test fixture.
  Failure log: target/perf/singular-force-runtime-repro.log. Initial compile
  used nonexistent with_rotation; corrected to BodyParams::with_pose first.
- Candidate inside Featherstone handles isolated world-root Free/Distance/D6
  articulations with positive mass and singular rotational inertia. It reuses
  the impulse path's constrained reduced mass solve, projects COM wrenches,
  includes applied joint torques and RNEA velocity/gravity bias, and restricts
  zero principal angular modes. Existing Featherstone integration and stepping
  remain unchanged. Normal positive-definite/multi-link paths are not replaced.
- Three focused regressions pass: scale sweep 1e-6/1/1e6, full/partial zero
  moments, zero/offset COM, rotated pose, sustained locked-axis wrench; allowed
  spin with initial velocity at translated origins; exact raw offset-frame
  point-mass reproducer. Tests require finite COM acceleration and locked-axis
  behavior, not merely absence of panic/quarantine.
- Important scope: this is NOT a general constrained-inertia solve for linked
  singular bodies, spherical roots, or GPU. Those remain unverified/open.
  Positive-definite fast path uses a determinant gate before principal axes.
- Source is uncommitted. New modules formatted individually. Full solver handle
  51587 and isolated Molla/Oxbo/imported/native-viewer chain 74264 are live.
  Focused handles 96080 (compile failure), 12752 (reproduced physical failure),
  25249 and 91501 (candidate passes) are terminal. Engram APIs unavailable.
- Files: crates/molla-solvers/src/featherstone.rs, featherstone_singular_root.rs,
  rigid_impulse.rs (crate-private constrained solve reuse), and
  tests/rigid_singular_forces.rs. Original Gearbox and shared assets untouched.

### Session summary: isolated singular-root force fix accepted

#### Goal

- Implement FULL_COMPACT.md; resolve the newly reproduced continuous-force
  singular-inertia failure without changing the normal machine integration.

#### Instructions

- Molla development only. Original Gearbox, machine USD assets and user-owned
  builds/viewers remain untouched. Rapier retained only as comparison control.

#### Discoveries

- Public RigidWorld reproduces the raw solver failure for a supported zero
  inertia body with offset COM. This is not an asset issue. Impulse handling
  already constrains zero principal angular response; continuous-force ABA
  lacked equivalent handling for this isolated free-root case.
- The corrected reduced-coordinate RHS must include COM wrench projection,
  applied joint torque, and RNEA gravity/velocity bias. The existing integrator
  remains the single update path; no extra chassis/spin integration was added.

#### Accomplished

- Core commit `9078f35 fix(solver): constrain singular root response` reuses the
  constrained mass response for isolated world-root Free/Distance/D6 bodies
  whose principal rotational inertia has null modes. Positive-definite and
  multi-link articulations retain the existing dynamics path.
- Solver release suite: **498 passed, 9 ignored, 92 binaries**. Final focused
  three-test file additionally covers zero/gravity cases, 1/8 substeps and
  mass/inertia scales 1e-6/1/1e6. Offset-frame raw and public regressions now
  remain finite with correct COM acceleration and principal-axis response;
  an allowed-spin case checks existing angular velocity is not frozen.
- Isolated Molla ordinary **111 passed, 18 ignored**; Oxbo **3 passed**;
  Kubota/Krampe imported **10 passed**. GPU-feature all-targets compilation
  passed (NOT runtime GPU parity). Native viewer build passed in 29.37 s,
  not launched. Ordinary test executable refreshed from the rejected candidate.
- Four grid-enabled full-meadow A/B/B/A runs passed; logged tyre/chassis states
  exactly match between corresponding before/after runs. No owned builds/tests
  overlapped timings. However, a user-owned original Gearbox release build was
  observed in the initial load snapshot, so these are diagnostic timings only.
  Do NOT claim the apparent baseline change is a speedup from this correctness fix.
- Diagnostic means of four medians: Molla parked 0.983287500 -> 0.976375750 ms,
  driving 1.164836500 -> 0.986713250 ms; Rapier driving control 0.285903500 ->
  0.290181250 ms. Baseline driving was contaminated by competing build load.
  Candidate mean driving p95 1.293255500 ms. No performance-parity claim.
- All owned handles terminal: 51587 solver, 74264 integration/viewer, 63827
  final focused/GPU check, 25261 timing. Earlier focused handles 96080/12752
  reproduce compile/physical failures; 25249/91501 pass candidate cases.
  Engram APIs unavailable. Source fix committed; no owned GUI or asset edits.

#### Next Steps

- General linked singular bodies, spherical roots and GPU singular-force parity
  remain open; do not generalize this isolated free-root fix to those cases.
  Add targeted cases and extend constrained response where required.
- Resume larger measured Molla optimizations; use a new controlled comparison
  rather than the contaminated baseline from this turn. Keep full 801x801
  terrain and enabled pressure model/solver settings.
- Full-plan joint/lifecycle/rendered/calibrated tyre/GPU/towing/replay/soil
  acceptance remains incomplete. Goal active, not complete or blocked.

#### Relevant Files

- `crates/molla-solvers/src/featherstone_singular_root.rs` — constrained root RHS.
- `crates/molla-solvers/src/featherstone.rs` — force capture and acceleration handoff.
- `crates/molla-solvers/src/rigid_impulse.rs` — shared crate-private constrained solve.
- `crates/molla-solvers/tests/rigid_singular_forces.rs` — three regression families.
- Core `target/perf/singular-force-{runtime-repro,runtime-candidate,expanded,
  final-focused,solvers,gpu-check}.log` — failure and final validation evidence.
- Isolated `target/perf/singular-force-{molla,oxbo,imported,viewer}.log`,
  `singular-force-grid-{before-a,after-a,after-b,before-b}.log`,
  `singular-force-timing-summary.txt`, `singular-force-load-{before,after}.txt`.
- Isolated retained `target/perf/gearbox-singular-force-after` SHA256
  f1cd2555411944323c4c6d26935bbd0c6a1aad587552e361c95d4a81ba1f98f6;
  reference remains gearbox-small-block-after. Hash file singular-force-binaries.sha256.

### Local heightfield bounds candidate checkpoint

- Added debug tracing to CPU broad phase, pair filtering, contact output and
  individual shape pairs. Geometry now uses the existing tracing 0.1 dependency;
  core and isolated Cargo.lock add that dependency edge. Isolated meadow trace
  filter also enables molla_geometry. No new physics engine dependency.
- Full-meadow trace passed (handle 23372 terminal). All 13 surviving non-wheel
  pairs were machine shapes versus ground shape 17, 960 calls each over 120
  frames. Individual span overhead is substantial; traced durations are not an
  acceptance timing. Artifact: isolated target/perf/molla-collision-breakdown
  {-trace.json,.log,.tsv}. Global terrain bounds include distant hills and leave
  flat-patch parts in detailed collision detection even when locally airborne.
- Candidate caches maximum heights in 8x8-cell tiles alongside scene-stamped
  global heightfield bounds. Each tile includes shared boundary vertices and
  partial final tiles. The CPU prepared path transforms its existing broad-phase
  shape AABB into field coordinates and skips narrow phase only above all covered
  tiles. Bounds expand conservatively for floating-point transforms; non-finite
  or invalid/missing data falls back to normal collision. No resolution/substep
  reduction, no changed contact equations, no approximation of accepted contacts.
- Broad-phase pair/filter callback behavior and reported candidate counts stay
  unchanged. CPU narrow-phase call counts decrease. GPU broad/full-narrow paths
  retain their original behavior; no new GPU culling/parity claim.
- Focused verification passed: geometry lib 120 tests and prepared bounds 4
  tests (one timing ignored). New exact-contact oracle covers six shapes, both
  pair orders, tilted/translated terrain, non-square cells, peaks on tile edges,
  corners and partially filled tiles, both airborne and penetrating positions.
  Existing scene-edit/removal/replacement oracle still passes; storage-reuse
  test now checks tile cache allocation as well. New files formatted separately.
- Initial focused attempts had a test-only missing direct glam dependency
  (replaced EulerRot with quaternion products) and a wrong test point that was
  inside a peak-containing tile (corrected to a genuinely flat tile). No physics
  assertions were relaxed. Handles 4358/56203/31569/85425 all terminal.
- Source uncommitted at this checkpoint. Full geometry/solver handle 87446
  subsequently passed 697 tests (10 ignored, 118 binaries). Isolated handle
  90869 passed 111 ordinary tests (18 ignored), three Oxbo tests, ten imported
  tractor/trailer tests, and the native profile-viewer build. GPU-feature
  all-targets geometry/solver compilation passed (handle 48291); this is not
  new GPU runtime parity. All these handles are terminal.
- Molla-only full-meadow trace passed (58674 terminal). Its 120-frame segment
  rejects all 13 non-wheel ground pairs before detailed dispatch, with logged
  before/after machine states identical. Instrumented collision time changes
  from 0.439835 to 0.121920 ms/frame, but much of that is removed per-pair trace
  overhead: use the uninstrumented measurements below for performance claims.

### Session summary: local terrain culling validation and timing

#### Goal

- Continue the full FULL_COMPACT.md goal, prioritizing Molla performance without
  changing tyre mechanics, terrain resolution, substeps, or the original checkout.
- Previous user-facing status turn was no implementation progress; this turn
  resumes source review, adds cache-refresh coverage and measures the candidate.

#### Instructions

- Original Gearbox, shared assets and other dependencies remain read-only.
  Rapier is a timing control only; no Rapier development or removal in this work.
- Engram context/save/session-summary APIs are unavailable in the tool registry;
  this project file retains the handoff. No persistent memory-folder edits.

#### Discoveries

- The cached local terrain-height bound can skip all non-wheel ground narrow
  phases in the measured tractor segment without changing any logged states.
- Terrain replacements must invalidate tile maxima even when global extrema,
  dimensions and collider identity stay the same. Added an explicit regression
  that toggles local surface contact while a distant peak preserves global bounds.
- Workstation activity affects tails: original Gearbox viewers remain running,
  and an external Cargo test was observed briefly during the first scalar block.
  No user process was stopped or changed. No owned build/test overlapped timing.

#### Accomplished

- First complete scalar and field-grid A/B/B/A run passed (59035 terminal), each
  fixture internally using R/M/M/R, 600 warmup + 1200 samples for parked/driving,
  full 801x801 meadow, 120 Hz and enabled pressure model at unchanged settings.
- Mean of four medians per cell, milliseconds per physics step:

  | mode / backend | before parked | after parked | before driving | after driving |
  | --- | ---: | ---: | ---: | ---: |
  | scalar Molla | 0.982812250 | 0.814927750 | 0.986504750 | 0.834251000 |
  | scalar control | 0.281612250 | 0.290394500 | 0.293469000 | 0.305380000 |
  | field-grid Molla | 0.993381500 | 0.832526500 | 1.000986250 | 0.841129250 |
  | field-grid control | 0.291841750 | 0.282838750 | 0.292514250 | 0.295424500 |

- First-run driving medians improve about 15.4% scalar and 16.0% field-grid.
  Driving mean p95 remains 1.554327 ms scalar / 1.3265605 ms grid. This does NOT
  meet full under-1-ms/tail or Rapier-parity acceptance. All logged state lines
  compare exactly across before-a/after-a/after-b/before-b in both modes.
- Additional cache-refresh regression passed with all prepared-bounds tests:
  five passed, one timing ignored (85743 terminal). No production source changed
  after the full validation and candidate build; the extra change is test-only.
- A second complete scalar/grid A/B/B/A passed (95707 terminal). All eight repeat
  logs have exactly the same state lines as the corresponding first baseline.
  Mean of four medians per cell for the independent repeat:

  | mode / backend | before parked | after parked | before driving | after driving |
  | --- | ---: | ---: | ---: | ---: |
  | scalar Molla | 0.976688000 | 0.810530750 | 0.984374000 | 0.816371500 |
  | scalar control | 0.284607500 | 0.281572500 | 0.299135750 | 0.296365250 |
  | field-grid Molla | 1.000283250 | 0.834697000 | 1.011874250 | 0.845121750 |
  | field-grid control | 0.284119250 | 0.281543750 | 0.296535000 | 0.294005750 |

- Repeat driving reduction is 17.1% scalar / 16.5% grid, with the unchanged
  control moving less than 1%. Keep this optimization: both independent batches
  show a substantial median reduction and preserve logged machine states.
  Repeat candidate driving mean p95 is still 1.172439 / 1.6161695 ms; no stable
  tail reduction or parity claim. Candidate is still about 2.75x / 2.87x the
  repeat control. Original user viewer remained active in every repeat load
  snapshot; no competing build appeared in those snapshots. These are loaded-
  workstation physics timings, not full-render frame times or isolated-host tails.
- All owned validation/timing handles are terminal. No owned GUI launched and
  no shared USD/asset changes. Only Molla and isolated Gearbox source changed.
- Accepted source commit: core `16dfe4c` (local terrain bounds/culling, tracing
  and regression coverage). Isolated `e8292b1` enables geometry trace capture
  and updates the dependency lockfile. Both are unsigned, title-only commits;
  no merge, push or original-checkout operation was performed.

#### Next Steps

- Retain the measured local terrain culling and its exact-contact/cache tests;
  use `gearbox-heightfield-tile-after` as the next accepted timing reference.
- Next measured hotspots are constrained-joint response and Featherstone work;
  do not reduce physical fidelity to reach a smaller number.
- Full-plan lifecycle, calibrated tyre/surface, visual, replay, joint/GPU, soil
  and performance acceptance remains incomplete. Goal remains active.

#### Relevant Files

- `crates/molla-geometry/src/heightfield_bounds.rs` — shared-vertex tile maxima.
- `crates/molla-geometry/src/shape_world.rs` — scene-stamped tile cache and
  conservative field-local shape bounds.
- `crates/molla-geometry/src/{broad_phase,pipeline}.rs` — cached AABB reuse,
  prepared CPU narrow-phase culling and geometry tracing.
- `crates/molla-geometry/tests/prepared_bounds/tiles.rs` — shape/pose/edge contact
  oracle and unchanged-global-bounds terrain-edit regression.
- Core `target/perf/heightfield-tile-{core,gpu-check,final-cache}.log`.
- Isolated `target/perf/heightfield-tile-{molla,oxbo,imported,viewer,trace}.log`,
  `heightfield-tile-trace.json`, `heightfield-tile-profile.tsv`,
  `heightfield-tile-{scalar,grid}-{before-a,after-a,after-b,before-b}.log`,
  `heightfield-tile-timing-summary.txt`, `heightfield-tile-load-{before,after}.txt`.
- Repeat logs/load snapshots `heightfield-tile-repeat-{scalar,grid}-
  {before-a,after-a,after-b,before-b}{.log,-load.txt}` and
  `heightfield-tile-repeat-summary.txt`.
- Retained candidate `target/perf/gearbox-heightfield-tile-after`, SHA256
  40ae96b5a4b3925e1ece27c3f77eb4e9627bb9d92364c8c9eb316c7c61c6a924.
- Retained baseline `target/perf/gearbox-singular-force-after`, SHA256
  f1cd2555411944323c4c6d26935bbd0c6a1aad587552e361c95d4a81ba1f98f6.

### Sparse wrench-projection candidate checkpoint

- Previous goal turn made verified progress: accepted terrain optimization in
  `16dfe4c`, isolated trace update in `e8292b1`, timing handoff in `43503e4`.
- Next profile target is loop-response wrench projection. A body's COM Jacobian
  has all-zero columns for non-ancestor or locked joints, but each closure row
  currently scans all six full dense rows. Candidate caches packed nonzero
  columns alongside the existing per-state dense Jacobian. It retains exact
  six-term product order and dense fallback for non-finite wrenches. No columns
  are dropped by tolerance, and coefficients containing NaN/Inf remain present.
- Loop-force projection and wheel quadratic response share this representation
  through ArticulatedResponse, whose lifetime is one immutable state. Public
  Jacobian API, mass response, constraint equations and integration are unchanged.
- Focused cache/projection tests passed (56593 terminal), including sizes
  0/1/6/39/120, dense/sparse patterns, negative zero, subnormal values and
  non-finite wrench propagation compared to the original dense projection.
- Core solver full tests passed 499 tests (nine ignored, 92 binaries), followed
  by 111 isolated ordinary tests (18 ignored); handle 14546 is terminal.
  These runs preceded the final additional non-finite-coefficient test, which
  was archived without execution when the candidate was rejected below.
- Source: `crates/molla-solvers/src/articulated_response.rs`,
  `articulated_response/projection.rs`, `loop_forces.rs`. Logs:
  core `target/perf/sparse-projection-{focused,solvers}.log`, isolated
  `target/perf/sparse-projection-molla.log`. Accepted retained baseline remains
  isolated `target/perf/gearbox-heightfield-tile-after` (40ae96b5...).

### Session summary: sparse projection rejected after machine timing

#### Goal

- Continue FULL_COMPACT.md and reduce enabled full-machine physics cost without
  changing physical equations, accuracy settings, tyre features or terrain detail.

#### Instructions

- Work remains limited to Molla and isolated Gearbox. Original Gearbox and
  shared assets remain untouched. No Rapier development, merges or pushes.
- Engram save/context/session APIs are unavailable; this file is the handoff.

#### Discoveries

- Sparse packed wrench projection is not a win in this machine workload.
  Saving zero-column multiply/add operations did not offset cache preparation
  and indirect access costs sufficiently; that cost explanation is an inference,
  while the measured regression is direct evidence. Do not re-add this candidate
  merely because the local operation appears cheaper.
- The last accepted trace computes mass_matrix 1920 times over 120 frames at
  eight substeps (two per substep). Source shows separate caches for external
  tyre/loop response (`ArticulatedResponse`) and post-FK motor/constraint response
  (`JointMass`). This is a better next investigation, not a proven speedup yet.
- Mass-cache reuse across those phases needs exact state validity: Featherstone
  applies mimic constraints and recomputes FK before motor effective inertias.
  Blindly reusing pre-force response matrices would be wrong when that changes
  joint coordinates/body poses. Preserve distinct factorization/fallback semantics
  for singular matrices and the existing motor inertia thresholds.

#### Accomplished

- Completed scalar + grid A/B/B/A, each internally R/M/M/R, unchanged 801x801
  terrain/120 Hz/pressure model/warmup/sample counts; handle 52764 terminal.
  No owned validation/build ran concurrently. Per-run host snapshots retained;
  original user viewer remained active. All before/after logged state lines
  matched exactly within both modes.
- Mean of four medians, milliseconds per physics step:

  | mode / backend | before parked | candidate parked | before driving | candidate driving |
  | --- | ---: | ---: | ---: | ---: |
  | scalar Molla | 0.805666500 | 0.819102750 | 0.816810750 | 0.822317500 |
  | scalar control | 0.281246000 | 0.280673750 | 0.289212000 | 0.286829500 |
  | field-grid Molla | 0.823996750 | 0.847306750 | 0.834846750 | 0.865910500 |
  | field-grid control | 0.280562000 | 0.284119250 | 0.295803250 | 0.293933500 |

- Candidate driving medians are 0.67% / 3.72% slower; parked medians also worsen.
  **Rejected the candidate.** Restored both tracked source files byte-for-byte
  to HEAD through apply_patch and removed the new module. Only this progress
  document remains modified. No candidate GPU/imported/Oxbo/viewer acceptance
  is claimed or needed for an optimization that is not being retained.
- Retained experiment artifacts under ignored target/perf, never as release
  code. The additional non-finite-coefficient test is only in that archive and
  was not compiled/executed; do not count it as a passing regression.
- Restored isolated ordinary tests passed 111 (18 ignored) and native profile
  viewer build passed in 29.28 seconds (95576 terminal). All owned handles are
  terminal; no owned GUI launched. Do not use `gearbox-sparse-projection-after`
  as the next performance reference or launch it as accepted code.

#### Next Steps

- Investigate sharing the unchanged mass matrix between external force response
  and post-FK motor calculations within a single immutable-model substep. Use
  explicit coordinate/pose validity checks or a provably shared post-FK state;
  fall back after mimic/state changes. Do not cache across model edits by pointer
  alone, or reuse numerically different factorization semantics unconditionally.
- Keep accepted terrain-culling baseline. Full physics/tyre/lifecycle/visual/
  calibrated-surface/GPU/replay/soil and performance acceptance remains open.
  Goal active, not complete, paused or blocked.

#### Relevant Files

- Core ignored `target/perf/sparse-projection-{candidate.patch,candidate.rs,
  focused.log,solvers.log}` — rejected source and successful exercised tests.
- Isolated `target/perf/sparse-projection-{molla,restored,restored-viewer}.log`,
  `sparse-projection-{scalar,grid}-{before-a,after-a,after-b,before-b}{.log,-load.txt}`,
  `sparse-projection-summary.txt`, `sparse-projection-binaries.sha256`.
- Rejected retained binary `target/perf/gearbox-sparse-projection-after`, SHA256
  efb262c66faca19eba13210327172c10726fa655936e06f65f96983cf4ac7f75.
- Accepted baseline remains `target/perf/gearbox-heightfield-tile-after`, SHA256
  40ae96b5a4b3925e1ece27c3f77eb4e9627bb9d92364c8c9eb316c7c61c6a924.
- Next source investigation: `articulated_response.rs`, `joint_mass.rs`,
  `featherstone.rs` force callback/step and post-FK effective-drive inertia path.

### Shared substep mass-matrix candidate checkpoint

- Previous goal turn was progress: sparse projection was measured, rejected,
  restored exactly and documented in `5f9a3df`; accepted baseline unchanged.
- Candidate retains matrices already computed for tyre/loop articulated response
  and moves them into the CPU motor/constraint mass cache within the same substep.
  It shares matrices, NOT factors: the existing drive and singular-response
  factorization thresholds/fallbacks remain unchanged.
- `MassSeed` borrows the immutable Model, snapshots joint coordinates/body poses
  only when a response matrix exists, and transfers owned matrix allocations.
  The post-FK cache accepts them only for pointer-identical Model and bit-identical
  finite coordinates/poses. Changed mimic coordinates, FK poses, signed zero,
  non-finite states or different model fall back to fresh matrices. The Model
  borrow prevents mutation/pointer reuse during the seed lifetime.
- Ordinary CPU RigidWorld stepping publishes the seed after built-in/custom
  forces. Public Featherstone force callbacks retain their existing signature
  through a wrapper; raw stepping, GPU and selected-island stepping do not seed
  mass caches. No claim of a new GPU or sleeping-path optimization.
- Initial all-targets check passed (96300 terminal); removed its newly reported
  unused-mut warning. Focused seed/allocation and full-step oracle tests passed
  (59204 terminal). Extended the oracle afterward with an explicitly locked
  hinge and zero mass-matrix diagonal to exercise singular response, alongside
  mimic/no-mimic, zero/nonzero tip inertia, two articulations and two substeps.
- Full solver tests passed 500 (nine ignored, 92 binaries), followed by 111
  isolated ordinary tests (18 ignored); handle 28875 terminal. This includes
  the final explicitly singular/mimic oracle. Source remains uncommitted.
- First scalar/grid A/B/B/A completed (77251 terminal) without overlapping owned
  builds/tests; all logged state lines match exactly across before/after runs.
  Full terrain, model and controller settings unchanged. Mean of four driving
  medians: scalar 0.831122250 -> 0.793371750 ms (4.54% reduction), field-grid
  0.850799750 -> 0.825107000 ms (3.02%). Rapier controls: scalar 0.291135500 ->
  0.297510000, grid 0.297713750 -> 0.291354000. Host load is variable; repeat
  timing is still needed before accepting this optimization. Candidate driving
  mean p95 1.1355655 / 1.33171025 ms, so neither tail nor parity target is met.
- Oxbo/imported/native-viewer/Molla-only trace/GPU-check sequence passed (41868
  terminal): three Oxbo tests, ten imported tests, native profile viewer build,
  one full-meadow trace fixture and GPU-feature all-targets solver compilation.
  No new GPU runtime parity or rendered acceptance claim.
- Trace confirms 960 mass_matrix spans versus 1920 in the accepted baseline
  over 120 frames at eight substeps: one matrix build rather than two per
  substep. Trace before/after logged state lines also compare exactly. Use this
  for work-count evidence, not traced-duration speedup claims.
- Second scalar/grid A/B/B/A passed (16785 terminal); all owned validation/builds
  finished before it started. Every repeat state log matches the corresponding
  first-batch baseline exactly. Engram APIs unavailable; memory registry has no
  hits for JointMass, ArticulatedResponse or shared mass-matrix cache.
- Source: `articulated_response.rs`, `joint_mass.rs`, `joint_mass/seed.rs`,
  `featherstone.rs`, `featherstone_mass_seed.rs`, `featherstone_activation.rs`,
  `rigid_world.rs`. Core logs `target/perf/shared-mass-{check,focused,solvers}.log`;
  isolated `target/perf/shared-mass-molla.log`. Accepted baseline remains
  `target/perf/gearbox-heightfield-tile-after`, SHA256 40ae96b5... .
- Candidate retained executable `target/perf/gearbox-shared-mass-after`, SHA256
  fbc35a267e7f252f719516bcc95f7a16f07c397250717dd9d0b1638d05fa0803.
  First batch evidence: isolated `shared-mass-{scalar,grid}-
  {before-a,after-a,after-b,before-b}{.log,-load.txt}`, `shared-mass-summary.txt`,
  `shared-mass-binaries.sha256` under target/perf.

### Session summary: guarded substep mass reuse accepted

#### Goal

- Implement the full FULL_COMPACT.md objective, continuing the user-prioritized
  full-machine Molla performance work without reducing physical fidelity.

#### Instructions

- Only Molla and its isolated Gearbox integration are writable; original Gearbox
  and shared machines/assets remain untouched. Rapier stays a timing control.
- No new environment files or asset edits, no pushes/merges, no owned GUI launch.
  Make/Oslo native gates and unsigned title-only Conventional Commits retained.
- Engram APIs absent; this project handoff substitutes for unavailable session
  persistence. No writes to the persistent memory folder.

#### Discoveries

- Exact-configuration guards allow ordinary CPU RigidWorld external-response
  matrices to be reused after FK without changing physics results. Model lifetime
  protection plus finite bitwise coordinate/pose checks avoid stale matrices.
- Trace confirms halved mass-matrix construction count; this is not a claim of
  halving total physics cost. Factorization and solver work still dominate.
- Mimic/FK changes must invalidate reuse. Velocity/force changes do not invalidate
  configuration-dependent matrices. Factors remain independently recomputed so
  drive-inertia and singular-response thresholds are unchanged.

#### Accomplished

- Accepted core source in `6b55781` (`perf(solver): reuse substep mass matrices`).
  Public force callback API preserved, selected-island/raw/GPU paths unseeded.
- Validation: 500 solver tests (nine ignored), 111 ordinary integration tests
  (18 ignored), three Oxbo tests, ten tractor/trailer imported tests, one Molla
  full-meadow trace, GPU-feature all-targets compile, native profile-viewer build,
  new-file rustfmt check and diff check all passed. GPU runtime/visual acceptance
  was not rerun or claimed. All owned handles are terminal, including 50088.
- Two complete scalar/grid A/B/B/A batches passed with enabled pressure model,
  unchanged 120 Hz/settings and full 801x801 terrain. Logged machine states match
  exactly across all before/after/repeat runs and the Molla trace.
- Repeat means of four medians, milliseconds per step:

  | mode / backend | before parked | after parked | before driving | after driving |
  | --- | ---: | ---: | ---: | ---: |
  | scalar Molla | 0.812377750 | 0.792201750 | 0.820156250 | 0.799049250 |
  | scalar control | 0.280930500 | 0.283043500 | 0.288321750 | 0.291852750 |
  | field-grid Molla | 0.830178000 | 0.815205750 | 0.837955750 | 0.821221250 |
  | field-grid control | 0.280747250 | 0.282360500 | 0.291022750 | 0.290445750 |

- Repeat driving improvement: 2.57% scalar / 2.00% grid. Both independent
  batches improve both modes (first 4.54% / 3.02%). Retain this modest gain,
  not a broader claim of performance parity. Repeat candidate/control ratios
  remain about 2.74x / 2.83x; mean driving p95 1.2471815 / 1.2622905 ms.
- No owned builds/tests overlapped timing. Original user viewer remained active;
  per-run snapshots did not show competing builds in the repeat. These remain
  loaded-workstation physics timings, not rendering/frame-time acceptance.
- Accepted next retained baseline is `gearbox-shared-mass-after`; the rejected
  sparse-projection artifact remains only diagnostic and must not be launched.

#### Next Steps

- Investigate remaining factorization/Featherstone costs. Reusing a normalized
  Cholesky factor may be possible only after proving identical dimensions,
  normalization and pivot/fallback eligibility; do not silently change the
  spectral/locked-axis/drive-inertia semantics. This is an investigation, not a
  decided implementation or proven additional gain.
- Keep complete FULL_COMPACT.md scope: performance tails/parity and rendered
  acceptance, tyre lifecycle/calibration/replay, general joints/singular bodies,
  GPU limitations and soil coupling still require work. Goal active, not complete.

#### Relevant Files

- `crates/molla-solvers/src/joint_mass/seed.rs` — model-borrowed exact-state seed
  and reuse/invalidation tests. `joint_mass.rs` imports only matching matrices.
- `articulated_response.rs` — retains/exports already computed mass matrices.
- `featherstone.rs`, `rigid_world.rs` — private seeded substep handoff.
- `featherstone_activation.rs` — explicit unseeded selected-island calls.
- `featherstone_mass_seed.rs` — fresh/cached full-step oracle, including mimic,
  locked-axis singular matrices, multiple articulations and substeps.
- Core `target/perf/shared-mass-{check,focused,solvers,gpu-check,format}.log`.
- Isolated `target/perf/shared-mass-{molla,oxbo,imported,viewer,trace}.log`,
  `shared-mass-trace.json`, `shared-mass-span-counts.txt`.
- Isolated `target/perf/shared-mass-repeat-{scalar,grid}-
  {before-a,after-a,after-b,before-b}{.log,-load.txt}` and
  `shared-mass-repeat-summary.txt`, plus first-batch artifacts listed above.
- Accepted `target/perf/gearbox-shared-mass-after`, SHA256
  fbc35a267e7f252f719516bcc95f7a16f07c397250717dd9d0b1638d05fa0803.

### Shared full-rank factor candidate checkpoint

- Previous goal turn made verified progress: guarded matrix reuse accepted in
  `6b55781`, validation/timing handoff in `31e7769`. Next baseline is retained
  `gearbox-shared-mass-after` (fbc35a26...).
- Candidate extends the same exact-model/configuration seed to carry eligible
  normalized Cholesky factors. Only finite positive-scale full-rank Cholesky
  with squared pivots >1e-12 is exported. Spectral/zero/weak-pivot decompositions
  are discarded and retain the fresh motor fallback path.
- A compatible cached factor computes requested drive inverse diagonals with
  the existing `unit_inverse_diagonal` operation order. Positive squared pivots
  >1e-12 imply every normalized mass diagonal exceeds the old active-column
  selection cutoff 1e-14, so this path has the full original column ordering.
  Dimensions and finite factor contents are checked; partial/locked-axis cases
  stay on the unchanged active-submatrix path. Exact seed validity checks still
  gate both matrices and factors after mimic/FK changes.
- Focused tests passed (23299 terminal): bitwise fresh/cached drive inertias
  across scales 1e-60/1/1e60, full-rank/weak/zero angular modes, offset COM,
  stale poses, requested-axis subsets, force/acceleration/no-feedback modes;
  allocation identity confirms eligible factors survive rather than refactor.
  Full-step seeded oracle now includes factors and still passes mimic/locked
  singular cases; existing seed invalidation oracle passes.
- The last accepted full-meadow trace has 960 `loop_mass_rhs` calls with
  cholesky=true, so this workload has eligible factor decompositions. This is
  eligibility evidence, not a measured speedup from the new candidate.
- Full solver tests passed 501 (nine ignored, 92 binaries), followed by 111
  ordinary isolated tests (18 ignored); 43441 terminal. Source uncommitted.
- First scalar/grid A/B/B/A passed physically with identical logged states
  (88883 terminal), but timing is contaminated by independently running Nerv
  scrolling/store tests and workspace compilation. Load snapshots directly
  captured these processes. One candidate scalar median was 1.452282 ms while
  adjacent candidate medians were 0.75-0.82 ms; do not remove just that sample
  or claim the first grid improvement while ignoring the control's load drift.
  Preserve the entire first batch as diagnostic only and rerun after validation.
- Oxbo/imported/native-viewer/Molla trace/GPU compile sequence passed (33091
  terminal): three Oxbo tests, ten imported tests, native profile-viewer build,
  one full-meadow trace and GPU-feature all-targets compilation. Trace state
  logs exactly match the prior matrix-only baseline. No GPU runtime or new
  rendered-acceptance claim. No external process was changed/stopped.
- Two new scalar/grid A/B/B/A batches (`repeat`, `confirm`) passed (89193 terminal)
  after all owned validation/builds completed. Every run's logged state lines
  matched the original baseline exactly. No competing build/test appeared in
  their per-run host snapshots; original user viewer remained active. Both
  independent batches improve median Molla timings in both modes; accept this
  candidate based on those batches, not the contaminated first run. Engram APIs
  unavailable; memory registry has no hits.
- Rechecked the imported fixture's 4913.449268 versus 4916.010223 kg mass logs:
  these are explicit backend-specific fixture expectations in benchmark.rs,
  also present in prior logs, not an asset/cache regression. Shared USDZ mtimes
  remain September 14. No asset copy/edit was made during that check.
- Source: `rigid_impulse.rs`, `joint_mass.rs`, `joint_mass/seed.rs`,
  `articulated_response.rs`, `featherstone.rs`, `featherstone_mass_seed.rs`, new
  `featherstone_factor_seed.rs`. Core `target/perf/shared-factor-{focused,solvers}.log`;
  isolated `target/perf/shared-factor-molla.log`.
- Candidate `target/perf/gearbox-shared-factor-after` SHA256
  b364462764fba8da9f50de952cfdd35374ca094a3cf0e867764eb6053e08d622;
  baseline `gearbox-shared-mass-after` SHA256
  fbc35a267e7f252f719516bcc95f7a16f07c397250717dd9d0b1638d05fa0803.
  Isolated first-batch logs: `shared-factor-{scalar,grid}-
  {before-a,after-a,after-b,before-b}{.log,-load.txt}`, `shared-factor-summary.txt`,
  `shared-factor-external-load.txt`, `shared-factor-binaries.sha256` in target/perf.

### Session summary: compatible mass-factor reuse accepted

#### Goal

- Continue full FULL_COMPACT.md implementation, prioritizing measured full-machine
  CPU performance with the pressure-dependent tyre system enabled.

#### Instructions

- Original Gearbox and shared assets remain read-only. No Rapier feature work,
  removed physics features, changed solver settings, merges, pushes or GUI launches.
- Use native Make/Oslo validation and unsigned title-only Conventional Commits.
- Engram APIs remain unavailable; this file is the project handoff, without
  writing the persistent memory folder.

#### Discoveries

- The strict full-rank Cholesky path can reuse the existing normalized factor
  for motor inverse diagonals. Its pivot threshold is stricter than the motor
  active-column threshold, preserving the full column ordering. Existing
  spectral/weak/locked-axis fallback behavior remains unchanged.
- Eligibility and immutable exact-state checks are both required: matching
  dimensions alone is not sufficient to reuse a factor across a pose change.
- First timing batch was contaminated by external tests/builds. Kept all its
  observations but excluded the whole batch from acceptance rather than trimming
  the slow candidate sample. Two subsequent independent batches confirm gains.

#### Accomplished

- Accepted source in `ff4badb` (`perf(solver): reuse full-rank mass factors`).
  This extends `6b55781` matrix reuse; no change to penalty/tyre equations,
  integration, substeps, terrain detail or controller settings.
- Passed 501 solver tests (nine ignored), 111 ordinary integration tests
  (18 ignored), three Oxbo tests, ten imported tractor/trailer tests, one Molla
  meadow trace, GPU-feature all-targets compile, native profile-viewer build,
  new/owned test-module rustfmt check and git diff check. GPU runtime parity and
  new rendered acceptance are not claimed. All owned handles terminal, including
  23299, 43441, 88883, 33091, 89193 and 83605.
- Two accepted A/B/B/A batches per scalar/grid mode, each fixture internally
  R/M/M/R with unchanged warmup/samples/120 Hz/full 801x801 terrain and tyres.
  All logged before/after/repeat/confirm machine state lines and trace state lines
  compare exactly. Bitwise unit oracles cover factor reuse and fallback cases.
- Mean of four medians per cell, milliseconds per physics step:

  | batch / mode / backend | before parked | after parked | before driving | after driving |
  | --- | ---: | ---: | ---: | ---: |
  | repeat scalar Molla | 0.794166250 | 0.745492750 | 0.798725000 | 0.760282500 |
  | repeat scalar control | 0.282141250 | 0.287179000 | 0.296574250 | 0.289062250 |
  | repeat grid Molla | 0.807199500 | 0.769026000 | 0.821797000 | 0.769045000 |
  | repeat grid control | 0.283660250 | 0.280832750 | 0.294478000 | 0.294236750 |
  | confirm scalar Molla | 0.792944750 | 0.746063250 | 0.801595000 | 0.750745500 |
  | confirm scalar control | 0.291371000 | 0.281581500 | 0.294663000 | 0.289350500 |
  | confirm grid Molla | 0.817433500 | 0.773528750 | 0.825827250 | 0.779399000 |
  | confirm grid control | 0.283726000 | 0.286837500 | 0.296462750 | 0.294007000 |

- Driving median reductions: repeat 4.81% scalar / 6.42% grid; confirm 6.34%
  scalar / 5.62% grid. Retain this repeatable improvement. Confirm candidate
  mean driving p95 remains 1.2750735 / 1.4516885 ms and candidate/control ratio
  about 2.59x / 2.65x: no tail or Rapier-parity claim. Absolute timings still
  include the loaded workstation; original viewer stayed running throughout.
- Accepted repeat/confirm snapshots showed no competing build/test. No owned
  build/test overlapped any timing. Nothing was killed to obtain the result.
- Next accepted retained baseline: `gearbox-shared-factor-after`, replacing
  `gearbox-shared-mass-after` for subsequent before/after comparisons.

#### Next Steps

- Profile remaining work using the latest Molla full-meadow trace before choosing
  the next solver optimization. Preserve exact/reference checks and full-machine
  timing; local operation savings alone are insufficient (see sparse rejection).
- Full-plan performance parity/tails, rendered acceptance, tyre calibration and
  lifecycle/replay, remaining general joint/singular/GPU issues and soil coupling
  remain open. Keep the goal active; these commits do not complete FULL_COMPACT.md.

#### Relevant Files

- `rigid_impulse.rs` — finite full-rank normalized-factor eligibility/access.
- `joint_mass.rs`, `joint_mass/seed.rs`, `articulated_response.rs` — move eligible
  factors with matching matrices under existing model/configuration guards.
- `featherstone.rs` — cached full-rank requested inverse-diagonal path.
- `featherstone_factor_seed.rs`, `featherstone_mass_seed.rs` — bitwise/fallback,
  allocation, stale-pose and complete stepping coverage.
- Core `target/perf/shared-factor-{focused,solvers,gpu-check,format}.log`.
- Isolated `target/perf/shared-factor-{molla,oxbo,imported,viewer,trace}.log`,
  `shared-factor-trace.json`, `shared-factor-{repeat,confirm}-{scalar,grid}-
  {before-a,after-a,after-b,before-b}{.log,-load.txt}`,
  `shared-factor-repeat-summary.txt`, `shared-factor-binaries.sha256`.
- Contaminated first batch remains in the unsuffixed scalar/grid artifacts and
  `shared-factor-summary.txt`; do not use it as accepted timing evidence.
- Accepted binary `target/perf/gearbox-shared-factor-after`, SHA256
  b364462764fba8da9f50de952cfdd35374ca094a3cf0e867764eb6053e08d622.

### 2026-09-20 — Batched scalar joint-stop responses

#### Goal

- Continue FULL_COMPACT.md with the user's Molla performance priority. Keep
  tyres, terrain, fixed-step settings and machine behavior unchanged; retain
  Rapier only as the benchmark reference until the replacement gates pass.

#### Instructions

- Original Gearbox and shared machine assets remain untouched. Integration
  builds/tests run only in `robolibs/gearbox-physics`; no preview was launched.
- Use the Molla Makefile through Nix, native Oslo viewer build in the isolated
  worktree, patch edits and unsigned title-only Conventional Commits.

#### Discoveries

- Added debug spans for ABA setup, kinematics, backward/acceleration passes,
  constraints, integration and finalization. Added scalar-stop response/Gram/
  impulse spans. These distinguish joint-limit work from Featherstone's passes.
- Before the change, the Molla-only 120-frame full-meadow trace had 26 scalar
  stops on every one of 960 substeps. Response solves took 0.195608 ms/frame,
  versus 0.027641 ms/frame for the earlier trace's ABA backward pass. This
  directed optimization toward independent response solves, not ABA mechanics.
- Four-lane triangular solves preserve each RHS's division, subtraction and
  traversal order. Sharing arithmetic across opposite-sign bounds was not used;
  no signed-zero reflection assumption or omitted coupled stop was introduced.
- Profile-only after diagnostics: response solves 0.087476 ms/frame; scalar
  stops total 0.131440 versus 0.242054 ms/frame before. These instrumented
  measurements explain the target; the non-profile machine runs below determine
  acceptance. Remaining large trace regions include external/loop forces,
  projected wrenches/Jacobians, collision and wheel forces.

#### Accomplished

- Source commit `6a95c3f` (`perf(solver): batch joint stop responses`).
- `MassFactor::solve_batch` batches full four-RHS groups for Cholesky factors,
  reusing a transposed scratch buffer through forward/back substitution. Scalar
  tails and spectral/zero factors retain the existing solve path. Every RHS is
  dimension/finite checked. Scalar-stop construction, Gram matrix, unilateral
  active-set solution and reaction application remain unchanged.
- New exact-bit reference tests cover dimensions 0/1/2/6/7/39/78, batch sizes
  0/1/2/3/4/5/7/8/26/33, scales 1e-60/1/1e60, full-rank/weak/singular matrices,
  general RHSs and signed unit pairs. Additional tests cover zero factors,
  signed zero/subnormal RHSs and invalid inputs in every full-group/tail lane.
- Solver lane: **504 passed, 9 ignored across 92 binaries**. Isolated ordinary
  integration: **111 passed, 18 ignored**. Oxbo: **3 passed**. Imported tractor/
  trailer: **10 passed**, including pressure replay/extremes, slope parking,
  sleep, hitch/PTO and straight/turning/track checks. Profile trace test, native
  profile viewer build, GPU-feature all-targets compile, owned-module formatting
  and diff checks passed. No GPU runtime or rendered acceptance claim.
- All test/build handles reached exit 0: 58724, 5682, 8692, 35308 and 19147.
  Timing handles 1452 and 41500 also reached exit 0. No owned build/test ran
  concurrently with the accepted timing batches.
- Two A/B/B/A batches, each internally R/M/M/R, use the full 801x801 meadow,
  enabled pressure-dependent tyres, 120 Hz and unchanged solver settings.
  Mean of four medians per cell, milliseconds per physics step:

  | batch / mode | Molla before driving | Molla after driving | control before driving | control after driving |
  | --- | ---: | ---: | ---: | ---: |
  | repeat scalar | 0.754580000 | 0.673450750 | 0.291359000 | 0.293285500 |
  | repeat grid | 0.771751000 | 0.676393500 | 0.292895250 | 0.296697250 |
  | confirm scalar | 0.765072500 | 0.660467000 | 0.297021750 | 0.298653000 |
  | confirm grid | 0.772422000 | 0.674475500 | 0.297909500 | 0.292949750 |

- Driving reduction is 10.75%/12.36% in repeat and 13.67%/12.68% in confirm
  scalar/grid. Retain the repeatable gain. Confirm Molla/reference ratio remains
  **2.21x/2.30x**, with mean driving p95 **1.3260625/1.1191995 ms**. Repeat
  scalar p95 did not improve (1.42478475 before, 1.4665365 after). Thus no
  parity, tail-target or complete-plan claim.
- All four logged state lines in each of 16 before/after timing runs match
  exactly within/across batches. The two new trace state lines match both the
  previous shared-factor trace and the pre-batch scalar-stop trace exactly.
- Before/after load snapshots show no competing cargo/rustc activity; the
  user's original viewer stayed running. These are loaded-workstation physics
  timings, not full viewer-frame timings. No outside processes were stopped.
- Retained accepted baseline `target/perf/gearbox-stop-batch-after`, SHA256
  `8fbebbc7ac3cbcc49edc746946dbfe3aae8c9757ed5550801cda68199525fbc8`, supersedes
  `gearbox-shared-factor-after` for future comparisons. Isolated Gearbox stays
  clean at `e8292b1`; no integration source edits were needed for this change.

#### Next Steps

- Continue profiling/optimizing remaining loop-force/response and collision
  work against this accepted binary, retaining exact-state checks and full
  machine A/B/B/A gates. Local speedups alone still do not justify acceptance.
- Keep FULL_COMPACT.md active: performance parity/tails, full-frame/rendered
  evidence, calibration/pressure lifecycle/reference acceptance, general
  joint/singular/GPU issues and soil coupling remain open. Do not remove Rapier
  or modify the original checkout before the user's replacement condition.

#### Relevant Files

- `crates/molla-solvers/src/featherstone.rs` — ABA stage profiling.
- `crates/molla-solvers/src/joint_limits.rs` — batched stop responses/profiling.
- `crates/molla-solvers/src/rigid_impulse.rs` — response-batch module wiring.
- `crates/molla-solvers/src/rigid_impulse/response_batch{.rs,/tests.rs}` — batched
  triangular solves and bitwise/fallback/input validation coverage.
- Core `target/perf/stop-batch-{focused,solvers,gpu-check,format}.log`.
- Isolated `target/perf/{aba-stages,scalar-stops}-trace.{json,log}` and
  corresponding `*-profile.tsv` — pre-optimization diagnostics.
- Isolated `target/perf/stop-batch-{molla,oxbo,imported,trace,viewer}.log`,
  `stop-batch-trace.json`, `stop-batch-profile.tsv`, `stop-batch-binaries.sha256`,
  `stop-batch-{repeat,confirm}-summary.txt`, `run-stop-batch-timing.sh` and
  `stop-batch-{repeat,confirm}-{scalar,grid}-{before-a,after-a,after-b,before-b}`
  `.log`/`-load.txt` — validation and reproducible timing evidence.
- Engram tools remain unavailable in the configured tool registry; this
  project handoff records the session findings instead of claiming a memory
  save or session-summary tool call succeeded.

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
