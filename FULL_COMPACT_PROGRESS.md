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
  capsule, Y cylinder, round cylinder, convex hull, triangle mesh and the
  existing square-spacing heightfield, with local pose, friction, restitution
  and tags. Indexed-mesh convex decomposition and independently scaled
  row-major heightfields remain.
- Endpoint capsules retain the authored collider frame and lower to a posed
  Y capsule; coincident endpoints lower to a sphere. Collider descriptor,
  local/world pose reads and local/world pose writes are available without
  stepping. Geometry-frame, world and arbitrary-pose AABBs use the same shape
  bounds as broad phase, including the capsule's canonical-frame offset.
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
| 2: bodies | Pose/mass/COM/inertia/initial velocity/tag available in model/state/descriptors. Full read/write API, enabled/quarantine, damping, sleep/wake, additional-mass replacement, force/impulse helpers, CCD option, collider mass recomputation and no-step synchronization remain. |
| 3: colliders | Endpoint capsules, round cylinders, primitive/convex/other views, descriptor/pose reads, local/world pose writes and local/world/arbitrary-pose AABBs added. Material mutation, friction rules, two-way masks and enabled/sensor flags remain tested. Convex decomposition, rectangular scaled heightfields and density/mass remain. |
| 4: joints | Stable-ID insert/remove and directed-tree merge/split implemented for revolute/prismatic/fixed/ball/free/D6. Explicit locked-axis API, limits/motors/force caps, acceleration-based gains, mutable frames, softness, collision enable, enabled flag and full body-pair queries remain. |
| 5: world | Gravity access, CPU world step with configurable substeps, and per-step body-pair callback implemented. Solver-setting mapping for Gearbox and handle-aware CPU raycast remain. Existing `molla-sensors::Raycaster` is GPU-only; it cannot directly fill the CPU-backend requirement. |
| 6.1-6.3: tyres | Pending Featherstone wheel force elements, ground masking, scene terrain adapter with per-cell friction, wheel output readback. |
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

## Next implementation steps

1. Finish fixed-body/re-rooting and runtime descriptor semantics; retain the
   now-tested contact/readout/filtering path while extending the live world.
2. Complete convex decomposition, independently scaled heightfields and CPU
   raycast, then collider/body editing and mass/impulse APIs.
3. Finish joint editing, locking, motors and softness, including capture frames.
4. Implement the Gearbox backend in its existing physics worktree.
5. Connect tyre/terrain outputs, then sleeping/deformation/MPM as specified.
6. Run and record every original acceptance scene, performance and deterministic
   replay gate. Do not mark the objective complete from library tests alone.
