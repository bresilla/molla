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
- Runtime collider replacement supports the existing ball, cuboid, Y capsule,
  Y cylinder, convex hull, triangle mesh and heightfield builder paths, with
  local pose, friction, restitution and tags. This is not the complete collider
  contract in section 3.
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

## Requirement ledger

| Plan section | Status / remaining evidence |
| --- | --- |
| 0: solver and force architecture | Retained Featherstone for rigid runtime scenes; no XPBD replacement. Contacts/tyres must still feed `state.body_f`. |
| 1.1: penalty contacts | Pending Newton contact port, material stiffness/damping, refreshed detection per substep. |
| 1.2: live mutation | Foundation implemented and tested; re-rooting and complete mutable descriptors remain. |
| 1.3: contact force readout | Pending per-point force/impulse, friction, point, penetration, acted flag and pair normal. |
| 1.4: fixed bodies | Root transitions tested; non-root freezing/re-rooting and complete GPU Featherstone support remain. |
| 2: bodies | Pose/mass/COM/inertia/initial velocity/tag available in model/state/descriptors. Full read/write API, enabled/quarantine, damping, sleep/wake, additional-mass replacement, force/impulse helpers, CCD option, collider mass recomputation and no-step synchronization remain. |
| 3: colliders | Existing-shape mutation foundation implemented. Endpoint capsule, round cylinder, convex decomposition, rectangular scaled heightfields, density/mass, combine rules, two-way membership/filter masks, enabled/sensor flags, full mutable/read/query API and AABBs remain. |
| 4: joints | Stable-ID insert/remove and directed-tree merge/split implemented for revolute/prismatic/fixed/ball/free/D6. Explicit locked-axis API, limits/motors/force caps, acceleration-based gains, mutable frames, softness, collision enable, enabled flag and full body-pair queries remain. |
| 5: world | Gravity access and low-level step usable. World orchestration, iteration/substep settings, per-step pair callback and handle-aware raycast remain. |
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
  on `feat/physics-backend`, initially clean at `8358c2a`. The sibling
  `gearbox` worktree is on `develop` with user changes in `terrain.rs` and
  `crates/gearbox-fields/src/heights.rs`; do not switch or overwrite that tree.
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

## Next implementation steps

1. Finish fixed-body/re-rooting and runtime descriptor semantics while adding
   penalty contacts/readout, friction combine rules and pair filtering.
2. Complete shapes, collider/body editing and mass/impulse APIs.
3. Finish joint editing, locking, motors and softness, including capture frames.
4. Implement the Gearbox backend in its existing physics worktree.
5. Connect tyre/terrain outputs, then sleeping/deformation/MPM as specified.
6. Run and record every original acceptance scene, performance and deterministic
   replay gate. Do not mark the objective complete from library tests alone.
