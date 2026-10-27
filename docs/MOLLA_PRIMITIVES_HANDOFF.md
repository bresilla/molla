# Molla primitives: Gearbox backend handoff

## Status and scope

Source inventory checked on 2026-10-01. Molla is on `develop` at base
`da495a0971c666c7230dd49e0b2a48d0307fc7a4` with an uncommitted implementation.
The inspected Gearbox checkout is
`/home/bresilla/data/code/robolibs/gearbox`, revision
`af6ae060c5763e15c16df2ea2a7905d48e3d7682`.

**No Gearbox files were changed and no Gearbox build/test was run.** Signatures
below are proposed changes for its next session, not APIs already available
there. Molla's original primitives gates pass as of2026-10-03; this does not
certify Gearbox or USD conformance. See [the report](MOLLA_PRIMITIVES_REPORT.md) and
[the performance ledger](MOLLA_PRIMITIVES_PERFORMANCE.tsv) for evidence scope.

Paths below are relative to Gearbox unless explicitly labelled Molla.

## 1. Change the neutral backend contract

`bin/gearbox/src/physics/backend.rs` currently has Revolute, Prismatic, Fixed,
Spherical and Generic joint kinds. Its `JointDesc` carries frames, paired scalar
limits, motors, softness, contacts_enabled and loop_closure, but no distance,
cone or general break settings. Add neutral value types with these exact fields:

```rust,ignore
struct DistanceLimits { min: Option<f64>, max: Option<f64> }
struct ConeLimits { angle0: Option<f64>, angle1: Option<f64> }
struct BreakLimits { force: Option<f64>, torque: Option<f64> }
struct FrictionCoefficients { mu_static: f64, mu_dynamic: f64 }
```

Add `JointKind::Distance`, change `JointKind::Spherical` to
`Spherical { axis: DVec3 }`, and add optional `distance_limits`, `cone_limits` and
`break_limits` fields to `JointDesc`. Default them to None; apply each only to a
compatible primitive and return Molla's validation error rather than discarding
it. None on the descriptor means no explicitly authored setting. Both optional
fields inside a bound type remain independent.

Existing spherical callers use axis X to preserve the present backend mapping;
new importer callers forward their authored primary axis. Revolute/Prismatic
already carry an axis. Generic remains joint-frame X with its locked-axis mask.

| Existing backend signature | Proposed replacement/addition |
|---|---|
| `insert_joint(&mut self, body1: BodyId, body2: BodyId, desc: JointDesc) -> JointId` | `insert_joint(&mut self, parent: Option<BodyId>, child: BodyId, desc: JointDesc) -> Result<JointId, String>` |
| `joint_bodies(&self, id: JointId) -> Option<(BodyId, BodyId)>` | `joint_bodies(&self, id: JointId) -> Option<(Option<BodyId>, BodyId)>` |
| `JointMut::set_limits(&mut self, axis: JointAxis, limits: [f64; 2])` | `set_limits(&mut self, axis: JointAxis, limits: Option<[f64; 2]>) -> Result<(), String>` |
| No radial/cone access | `Joint::distance_limits(&self) -> Result<DistanceLimits, String>`; `cone_limits(&self) -> Result<ConeLimits, String>` |
| No radial/cone mutation | `JointMut::set_distance_limits(&mut self, limits: DistanceLimits) -> Result<(), String>`; `set_cone_limits(&mut self, limits: ConeLimits) -> Result<(), String>` |
| No general break access | `Joint::break_limits(&self) -> Result<BreakLimits, String>`; `is_broken(&self) -> Result<bool, String>` |
| No general break mutation | `JointMut::set_break_limits(&mut self, limits: BreakLimits) -> Result<(), String>`; `repair(&mut self) -> Result<(), String>` |
| `Collider::friction(&self) -> f64` only | Add `friction_coefficients(&self) -> Option<FrictionCoefficients>` for the authored pair; None preserves legacy scalar authoring |
| `ColliderMut::set_friction(&mut self, friction: f64)` only | Replace with `set_friction(&mut self, friction: f64) -> Result<(), String>` and add `set_friction_coefficients(&mut self, value: Option<FrictionCoefficients>) -> Result<(), String>` |
| `PhysicsBackend::step(&mut self, excluded: PairExcluded<'_>)` | `step(&mut self, excluded: PairExcluded<'_>) -> Result<(), String>` |

Migrate callers and mocks along with implementations. Changing `joint_bodies`
also affects `joint_between`, all-joint pair queries, removal and entity lookup;
compare `Some(body)` for a body endpoint and do not fabricate a world BodyId.
Update `JointAccess.bodies` in `physics/molla/joint.rs` to the optional-parent
shape. Invalid handles must produce a caller-visible error rather than an index
panic. Joint creation plus its requested settings must not leave a partially
configured live joint after rejection.

Existing frame, motor, softness, contacts and enable setters also forward
fallible Molla operations. Make their errors observable through Result or an
explicit error channel; do not claim success merely because `apply` logged an
error. Preserve both motor targets, gains, caps and Force/Acceleration model.
`Joint::motor_position` currently reads only revolute AngX and prismatic LinX;
do not extend it to D6 by indexing quaternion coordinates as Euler angles.

## 2. Forward to existing Molla APIs

These are actual Molla methods, not proposed names. Authoring is on
`molla_sim::runtime::RigidScene`; readout is on
`molla_solvers::rigid_world::RigidWorld`.

| Backend operation | Molla mapping |
|---|---|
| Tree/world joint | `insert_joint(JointDesc { parent, child, kind, axis, frame_parent, frame_child })` |
| Rigid external link | `insert_rigid_loop_joint(JointDesc)` |
| Compliant external link | `insert_loop_joint(JointDesc, JointSoftness)` |
| Distance | `JointType::Distance`; `set_distance_limits`, `distance_limits`; `world.distance_reading` |
| Cone | `JointType::Ball` with the authored primary axis; `set_cone_limits`, `cone_limits`; `world.cone_reading` |
| Generic axes | `JointType::D6`; `set_joint_locked_axes`, `set_joint_limits`, `joint_axis_settings` |
| Motor | `set_joint_motor_with_wake(handle, axis, motor, wake)` |
| Compliance | `set_joint_softness(handle, Option<JointSoftness>)`; None selects rigid |
| Break | `set_joint_break_limits`, `joint_break_limits`, `joint_broken`, `repair_joint` |
| Friction pair | Read/replace `ColliderDesc.contact_material.friction` through `collider`/`replace_collider` |
| Pair exclusion | `set_body_pair_excluded(a, b, excluded)` and `set_collider_pair_excluded(a, b, excluded)` |

Current Gearbox `physics/molla/mod.rs::insert_joint` always sets
`parent: Some(parent)`, calls `.expect` for insertion, and supplies default
compliance `(30.0, 1.0)` for every loop_closure. Replace those defaults with the
explicit parent and rigid/compliant selection above. A world anchor uses
`parent: None` and a world-space parent frame. It must not freeze the child body
or silently weld its free DOFs.

Distance binds anchor separation, not three Cartesian coordinates or rotation.
Its bounds are independent, finite, nonnegative metres with min <= max when
both exist. Zero equality pins the anchors without locking twist.
Cone half-extents are independent finite radians in [0, pi] and describe
geodesic swing in the cyclic transverse basis of the joint's primary axis;
they are not independent Euler-angle clamps. Preserve the authored frames and
axis conversion rather than rotating only one endpoint to fit the adapter.

Break limits are nonnegative and not NaN, in N and Nm. None or positive infinity
disables a threshold. Comparison is inclusive, and a zero threshold can break on
a valid zero-load sample. Decisions use physical substep samples, not the mean
motor effort. Repair is explicit and draining events does not repair a joint.

## 3. Owned readouts and events

Add backend methods with owned, neutral DTOs:

```rust,ignore
fn set_reaction_readout(&mut self, enabled: bool) -> Result<(), String>;
fn joint_reaction(&self, id: JointId) -> Option<JointReaction>;
fn joint_reaction_samples(&self) -> Option<Vec<JointReactionSample>>;
fn distance_reading(&self, id: JointId) -> Option<DistanceReading>;
fn cone_reading(&self, id: JointId) -> Option<ConeReading>;
fn joint_break_events(&self) -> Vec<JointBreakEvent>;
fn drain_joint_break_events(&mut self) -> Vec<JointBreakEvent>;
```

Forward to identically named RigidWorld methods. Copy/clone under the existing
`Shared` mutex; never return a reference into the world after releasing its
guard. Preserve these fields rather than narrowing every load to a scalar:

- `JointReaction`: child/parent world force and torque, sample_dt, peak_force,
  peak_torque. Torques are about the respective authored anchors, not COM.
- `JointReactionSample`: stable joint id, child/parent world wrenches,
  parent/child anchor positions, substep, time_offset and sample_dt in seconds.
- `DistanceReading`: separation, violation, active bound, world child force,
  sample_dt. `ConeReading`: swing pair, violation, active, world child torque,
  sample_dt. Retain the active-bound enum rather than a guessed sign convention.
- `JointBreakEvent`: successful outer frame index, stable joint id, optional
  parent/body child ids, trigger, limits and full reaction sample.

Missing, disabled, unsolved or invalidated frame readouts remain None, not zero.
Legacy `mass_matrix`/`inverse_dynamics` use locked-column analysis semantics;
zero efforts there are not proof of zero physical support. `response_mass_matrix`
retains partial angular columns for physical response, and the transmitted-wrench
API preserves their support loads. Use the runtime reaction/event APIs above for
importer telemetry and break decisions rather than legacy analysis effort slots.
Hot dynamics edits invalidate solved constraint/reaction readings even if the
compiled scene stamp does not change. Events persist until drained, with
zero-based successful outer-frame and substep-relative timing. Translate handles
to stable Gearbox ids before returning; never expose cached dense solver indices.

## 4. Friction, materials and exclusions

Molla validates finite `0 <= mu_dynamic <= mu_static`; invalid authoring errors
rather than clamping. An explicit pair overrides the legacy scalar. Gearbox's
current `physics/molla/collider.rs::set_friction` changes only that scalar, and
even returns early if the scalar already matches. Once pairs are exposed, choose
this compatibility contract: `set_friction(mu)` validates mu, writes the scalar
and clears the explicit pair, even when the previous scalar equals mu. The
effective pair is then (mu, mu). Setting None on the pair restores scalar use.

Retain per-material combine rules; priority is Average < Min < Multiply < Max,
applied independently to static and dynamic coefficients. Static response holds
zero contact slip within the static disk; sliding uses the dynamic disk. Rolling
speed is not contact slip. Specialized wheel/track laws remain their own APIs.

Add fallible backend body/collider pair exclusion setters matching the Molla
methods above. Bit masks and contacts_enabled do not replace explicit pair
permissions. Resolve USD collections/relationships in the importer, then pass
resolved ids; do not add stage composition or collection evaluation to Molla.

## 5. Explicit execution selection and acceptance

Gearbox's current `MollaBackend::default` constructs a CPU `RigidWorld::new`.
Add explicit fallible construction/selection for CPU versus a shared
`Arc<molla_compute::Engine>`, using `RigidWorld::from_scene` or
`from_scene_gpu`. No implicit CPU fallback is acceptable on GPU rejection.
Share immutable programs through the Engine, not mutable world state.

The live GPU runtime retains explicit unsupported-feature errors, including
registered actuator/wheel/track stepping and articulations above 32 joints.
Automatic CPU breakage with registered devices is also rejected. Do not advertise
these combinations as supported just because isolated kernels or imported assets
exist. Runtime Cable remains unsupported. The 32-joint guard is a live-world
capability boundary, not a limit on independent scene instances or every raw
Featherstone API.

Future Gearbox acceptance must exercise optional world parents, min-only/max-only
distance, cone free twist, D6 simultaneous targets/models, offset-frame wrenches,
single break event and repair, friction legacy/pair edits, resolved collision
permissions, full tensor/COM mass and caller-visible rejected mutations. Use the
six [Molla examples](crates/molla-solvers/examples/README.md) as API illustrations,
not substitutes for USD fixture and real-machine runtime tests. Units, schema
defaults, authored mass precedence, scene ownership and collection resolution
remain importer responsibilities from section 12 of `MOLLA_PRIMITIVES.md`.

## Key Learnings:

1. Gearbox world anchors require an optional parent contract, not fixed children.
2. Readouts need owned copies across its world lock and explicit stale None values.
3. Legacy friction writes must clear an explicit pair, including same-scalar writes.
