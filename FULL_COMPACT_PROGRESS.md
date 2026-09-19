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
  frame, coordinate, rate, limit and motor signs remain unchanged. Closed loops,
  including multiple independent anchors in one component, remain unsupported.
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

## Requirement ledger

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
