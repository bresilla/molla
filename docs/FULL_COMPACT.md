# What molla needs to replace rapier in gearbox

Gearbox's physics now sits behind one trait, `PhysicsBackend`
(`robolibs/gearbox`, branch `feat/physics-backend`,
`bin/gearbox/src/physics/backend.rs`, 680 lines). Rapier is the reference
implementation (`physics/rapier/mod.rs`, 930 lines). Molla replaces rapier by
implementing that trait. This file lists what molla must be able to do for
that implementation to exist, in the order to build it.

Conventions gearbox assumes: SI, f64, glam (`DVec3`/`DQuat`/`DMat3`), Y up,
gravity `(0,-9.81,0)`, fixed step 120 Hz (`GEARBOX_PHYSICS_HZ`), up to 12
steps per frame. Handles are opaque `u64` ids (`BodyId`, `ColliderId`,
`JointId`).

Integration changes belong in the isolated `robolibs/gearbox-physics`
worktree on `feat/physics-backend`. Do not edit or merge into the user's
original `robolibs/gearbox` checkout while it is under separate development.

## 0. Decisions already made

- One rigid solver: **Featherstone** (reduced coordinates). XPBD, VBD, MPM,
  SPH stay for soft/particle scenes and are not part of this.
- Everything else reaches the solver as **forces in `state.body_f`**: penalty
  contacts, the tyre model, later soil impulses. Featherstone already consumes
  `body_f` (`featherstone.rs:593,723`).
- Tyres are force elements (`molla-vehicle::wheel_ground_wrench`), not
  colliders against the ground.
- Soil later: Bekker `DeformableTerrain` first (it is a `Terrain`), MPM
  two-way coupling last, Newton style (soil sees bodies as colliders, returns
  impulses, they go into `body_f`).

## 1. Blockers (nothing works without these)

1. **Contacts in Featherstone.** `featherstone.rs:521` takes `_contacts` and
   ignores it. Port Newton's `eval_body_contact`
   (`xtra/newton/.../semi_implicit/kernels_contact.py`, launched from
   `solver_featherstone.py:574`): penalty normal `ke/kd`, friction `kf/mu`
   with smoothing, written into `body_f` before ABA. CPU path first.
2. **Runtime scene mutation.** `Model` is built once by `ModelBuilder`.
   Gearbox inserts and removes bodies, colliders and joints while running
   (spawn/despawn machines, USD variant swaps, hitch attach/detach, static
   props that freeze to fixed). Needed: add/remove body, shape, joint on a
   live model, with stable ids that survive other removals, without losing
   the state of everything else. Rebuild-and-copy is acceptable if ids and
   state are preserved.
3. **Contact force readout.** `molla-sensors/contact.rs:22` says forces are
   not measured. Gearbox's traction budget and wheel stamping need, per
   contact point of the last step: normal impulse (or force), friction
   coefficient used, world point, penetration distance, and whether the
   solver acted on it; grouped per collider pair with a world normal.
   With penalty contacts this is the force you already computed; keep it.
4. **Fixed bodies.** There is `DYNAMIC` and `KINEMATIC` only. Needed: a fixed
   (static) body kind, and changing a body's kind at runtime
   (dynamic → fixed for settled props, body pinned by a world-anchored joint).

## 2. Bodies

Create from: kind, pose, linear/angular velocity, linear/angular damping,
start asleep, additional mass (mass, local COM, principal inertia **or** full
3×3 tensor), user tag (`u64` entity bits, returned on read).

Read: pose, linvel, angvel, mass, world COM, local COM, principal inertia,
inertia tensor in body frame, kind, enabled, sleeping, its colliders, tag.

Write: pose (teleport, with wake), linvel, angvel, kind, enabled/disabled
(gearbox disables a body whose state went NaN instead of crashing), wake,
sleep, damping, replace additional mass, add force/torque, reset forces,
impulse / torque impulse / impulse at point, CCD on/off (may be a no-op).

Also: recompute a body's mass from its colliders after shapes or densities
change; move all colliders to their bodies' current poses without stepping.

Sleeping/islands: molla has none. Acceptable to report "never sleeping" at
first; needed later for performance with many parked machines and bales.

## 3. Colliders and shapes

Shapes to build: cuboid, ball, capsule (two endpoints + radius), cylinder
(along Y), **round cylinder** (tyres: cylinder with border radius), convex
hull from points, **convex decomposition** from an indexed mesh, static
**trimesh**, **heightfield** (row-major `rows×cols`, rows along Z, cols along
X, centred, `scale.y` multiplies heights).

Per collider: pose in parent body frame (or world when parentless), friction,
restitution, **friction combine rule (gearbox uses Min on tyres so the ground
decides; molla hard-codes max)**, density or mass, collision groups
(`memberships & other.filter`, both ways, 32 bits), enabled flag, sensor flag,
user tag. All of those writable at runtime, plus replacing the shape
(cylinder → round cylinder on machine discovery).

Read: world pose, local pose, parent, a shape view (enough for gizmos:
primitive dims, convex points+edges, "other"), local AABB, world AABB, AABB
of the shape at an arbitrary pose.

GPU note: molla's trimesh/convex/heightfield narrow phase is CPU only today.
Fine for gearbox, which is CPU.

## 4. Joints

Kinds: revolute about an axis, prismatic along an axis, fixed, spherical,
generic 6-DOF with an explicit set of locked axes (hitches use it:
drawbar = lin + roll locked, ball = lin locked with pitch/roll limits).
Two local frames (pose in each body). The free axis of revolute/prismatic is
always addressed as `AngX`/`LinX` whatever the authored axis was.

Per axis: limits `[lo,hi]`; motor with target position, target velocity,
stiffness, damping, max force, and model (force-based vs acceleration-based
gains). Molla's `JointTargetMode` + `kp/kd` + effort clamp covers this; the
acceleration-based model needs gains scaled by the joint's effective inertia.

Runtime: read and **write** frames (the hitch capture slides `frame2` over
~1 s), limits, motor, softness (constraint spring: natural frequency Hz +
damping ratio, used to soften the hitch while capturing), contacts between
the two jointed bodies on/off, joint enabled. Insert/remove joints live.
Queries: joint's two bodies, all joints, joint(s) between two bodies.

Trees: a tractor with a hitched trailer is one tree, so hitching means
**merging two articulations at runtime** and splitting on detach. No loop
closures are required today. Rapier's fallback for that case was a soft
constraint joint; in Featherstone the clean answer is the merge.

## 5. World

Gravity get/set. Step by `dt` with solver iteration settings. A **pair
filter callback** per step: `excluded(bodyA, bodyB) -> bool` drops every
contact between two bodies (USD `FilteredPairsAPI`, machine self-collision,
tractor↔hitched trailer). Molla has a pair exclusion list; it must be
changeable every step.

`cast_ray(origin, dir, max) -> (collider, distance)`; used by tests and
tools, not per frame. Molla's `Raycaster` covers it (dynamic trimesh skipped
is fine).

Not needed by gearbox: shape casts, point queries, contact events,
character controller.

## 6. Vehicle layer (molla's advantage; rapier has none of it)

1. Tyre force element on Featherstone: per wheel body build `WheelContact`
   (hub pose/velocity, heading from the knuckle, spin = wheel joint velocity,
   radius), call `wheel_ground_wrench`, add the wrench to `body_f[wheel]`.
   Wheel shapes are masked out of ground contact. Drive torque stays the
   wheel joint motor. `Vehicle::step` (3-DOF island) becomes redundant.
2. `Terrain` from the scene heightfield/trimesh shape, with per-cell
   friction. Gearbox already authors ground friction per field.
3. Expose per-wheel `TireOutput` (`fz`, slip ratio, slip angle, contact
   point): gearbox's traction and wheel stamping read this instead of contact
   impulses.
4. Integrate the pressure-dependent tyre system described below. Adjustable
   inflation pressure must affect tyre mechanics and traction, not just a
   displayed value or a global friction multiplier.
5. Later: `DeformableTerrain` (ruts and soil resistance), then MPM coupling
   around tools.

### 6.1. Pressure-dependent tyre integration

This is required integration work, not an optional viewer demonstration.
Reuse and extend `molla-vehicle::real_tire` rather than maintaining an
unrelated Gearbox-only tyre model. Existing pressure/carcass/brush code is a
starting point; its presence alone does not prove live integration.

- **Authored tyre properties:** per-wheel inflation pressure, reference
  pressure, supported pressure range, unloaded radius, width, carcass
  stiffness/damping, and tread/material parameters. Specify whether each
  pressure is gauge or absolute and convert explicitly at model boundaries.
  Use Pa internally; expose bar with explicit units in user-facing controls.
  Assets without tyre metadata need documented, inspectable defaults.
- **Runtime adjustment:** set and read pressure per wheel, per axle, or for
  the whole machine through the isolated Gearbox UI and command interface.
  Edits work while paused and while driving, persist through unrelated scene
  rebuilds, and round-trip through the supported machine/scene save path.
  Reject non-finite, negative and out-of-range values without partial updates.
  Define bounded pressure transitions so an edit cannot teleport the chassis,
  reset wheel speed or inject an uncontrolled force spike.
- **Deformation and footprint:** current pressure and load determine carcass
  deflection, loaded radius, vertical stiffness/damping, contact-patch shape
  and area, and contact-pressure distribution. Use those outputs in actual
  wheel/terrain contact and slip calculations. Provide matching visible tyre
  flattening/bulging and pressure-dependent track footprints, not merely a
  scaled rigid wheel mesh unrelated to the physics.
- **Grip and resistance:** longitudinal and lateral force, combined-slip
  saturation, aligning moment and tyre rolling resistance must respond to
  pressure, load, carcass/tread properties and the contacted surface. Feed the
  resulting traction budget to the drive controller. Do not assume lower
  pressure always means more grip on every surface, or manufacture the effect
  by changing ground friction. Document and validate the model's calibrated
  operating range and expected pressure trends for each supported surface.
- **One rigid dynamics path:** apply the resulting force and moment through
  `body_f` in the current Featherstone substep; drive torque remains the wheel
  joint motor. Tyre deformation may have internal state, but must not introduce
  another wheel-spin or chassis integrator. Continue masking tyre shapes from
  ground contact, retain obstacle contacts, and avoid duplicate normal support,
  traction or rolling-resistance forces.
- **Readout and lifecycle:** expose pressure, deflection, loaded radius,
  footprint dimensions/area, normal load, slip ratio/angle, resultant tyre
  forces and rolling-resistance moment. Preserve per-wheel state across
  unrelated topology edits, clear it on removal, and refresh derived outputs
  after pressure changes. Telemetry, visual deformation, traction control and
  terrain stamping must consume the same physical state.
- **Terrain coupling:** integrate footprint pressure with scene surface
  materials and per-cell friction first. Later Bekker/MPM soil coupling must
  use the same footprint/load distribution for sinkage and rutting, without
  counting tyre and soil resistance twice.
- **Verification:** cover pressure sweeps at matched load/surface, combined
  slip, braking, coast-down resistance, rolling radius, and pressure changes
  at rest and in motion. Verify deterministic replay, finite bounded forces,
  lifecycle persistence and timestep convergence. Benchmark the enabled model
  against the existing 120 Hz / under-1-ms CPU target. If a GPU tyre path is
  provided, require CPU/GPU parity for these pressure-dependent outputs too;
  do not treat CPU integration as proof of GPU parity.

## 7. Stability targets

A 2.7–5 t tractor on four tyres with 26 bodies, steering knuckles of a few
kg next to a 3.8 t chassis (mass ratio ~1000:1), hitched 2–10 t trailers,
parked on a slope without creeping, 120 Hz, under 1 ms per step for one
machine on CPU. Penalty contacts will need `ke/kd` per material and
Featherstone substeps; contacts should be re-detected per substep (XPBD's
notes call out that not doing so amplifies impacts).

Determinism across runs on one machine is expected (scripted scenarios).

## 8. Build order

1. Fixed bodies + runtime add/remove with stable ids (§1.2, §1.4).
2. Penalty contacts in Featherstone + force readout (§1.1, §1.3), friction
   combine rule, pair filter callback.
3. Missing shapes: round cylinder, convex decomposition. Runtime shape and
   material edits.
4. Generic locked-axes joint, runtime frame/limit/motor/softness edits,
   articulation merge/split for hitches.
5. `impl PhysicsBackend for MollaBackend` in gearbox; add the arm in
   `backend_by_name` (`physics/world.rs`); run with `GEARBOX_PHYSICS=molla`.
6. Tyre force element + `Terrain` adapter (§6 items 1–3), then the complete
   pressure-dependent tyre integration (§6.1), including controls, readout,
   deformation, grip/resistance and regression coverage.
7. Sleeping/islands, then deformable terrain, then MPM coupling.

## 9. Acceptance

Same scenes on both backends (`GEARBOX_PHYSICS=rapier|molla`):

- gearbox's 43 binary tests pass, including the two joint-motor tests
  (constraint and reduced), the heightfield ray test and steering limits.
- `gearbox spawn machine tractor.usd`; `machine move --forward 2 --turn 0.4`
  drives and steers (rapier: heading +0.60 rad after 4 s); tyres leave tracks.
- `machine tools attach trailer1 --hitch rear_drawbar --teleport`, tow
  through a turn, detach; no NaN, no quarantined bodies.
- Kubota asset (5 controllers: drive, two hitches, two PTOs): hitch lift and
  PTO motors move their joints.
- Tractor parked on a 10° slope stays put for 60 s.

Additional Molla tyre acceptance (Rapier is a baseline, not required to
reproduce Molla's pressure-dependent tyre physics):

- On the real Kubota, change front/rear tyre pressure independently through
  both the UI and command interface, paused and driving. Read back the applied
  values and confirm they survive an unrelated spawn/despawn, hitch
  attach/detach and a supported save/reload cycle. Invalid edits are atomic.
- At fixed vehicle load and surface, compare low/reference/high pressures
  within the tyre's supported range. Record pressure, loaded radius,
  deflection, contact area/distribution, normal load, slip, tyre forces,
  acceleration, braking, cornering response and coast-down resistance.
  Compare results against documented model/reference tolerances rather than
  merely asserting that the numbers differ.
- Show matching visible tyre deformation and track-footprint changes on the
  machine. Repeat on firm ground and a supported field surface; verify that
  pressure-dependent grip uses surface/load response rather than a universal
  lower-pressure grip bonus.
- Repeat straight driving, turning, towing and the 60 s slope-parking test
  at the supported pressure extremes. No NaNs, quarantined bodies, uncontrolled
  impulses, duplicate support forces or return of the straight-drive stall.
  Include a scripted mid-drive pressure transition and deterministic replay.
- Record CPU step timings with pressure-dependent tyres enabled on the full
  machine; the existing stability and performance targets still apply.
