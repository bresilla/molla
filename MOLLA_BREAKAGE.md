# General joint breakage

## Status

Threshold authoring, explicit batched detachment, persistent broken state and
explicit repair/re-enable are implemented. Automatic CPU and GPU breaking evaluate
solved substep loads, detach before the next substep, continue the frame and
publish events only after successful completion. GPU comparison and compact event
generation run on-device; host topology transactions occur only at actual break
boundaries. Both backends currently reject combinations with registered actuator
devices, wheels or tracks. CPU joint axis motors, contacts and force callbacks are
supported; existing GPU contact/compliance/control capability limits still apply.
This is not the full P3b acceptance gate or production GPU completion.

## Configuration and state

`molla_sim::runtime::JointBreakLimits` contains independent optional `force` and
`torque` thresholds. They apply to magnitudes of the world load on the authored
child, with torque about the child joint anchor. Units are N and N·m, not impulse
units. `None` and positive infinity disable a channel. Negative values, negative
infinity and NaN are rejected transactionally. Zero is inclusive: the first valid
solved sample reaches a zero threshold even when that sample's load is zero.
Sleeping/unsolved joints have no valid sample and must not trigger from stale data.

`JointBreakLimits::trigger(force, torque)` validates finite nonnegative sample
magnitudes and returns `Force`, `Torque`, `Both` or no trigger. It is a stateless
threshold evaluator, not an event emitter. Comparisons are inclusive `>=`.

`RigidScene::set_joint_break_limits` stores configuration by stable joint handle,
invalidates the dynamics-input stamp and activates endpoint dependencies without
rebuilding topology. `joint_break_limits` queries it; `breakable_joints` iterates
enabled finite-threshold joints with enabled endpoints. Removal of a joint or an
endpoint removes its configuration and broken state.

`break_joints(&[JointHandle])` is an explicit topology transaction. It validates
the entire batch before changing anything, deduplicates handles, and returns the
newly broken handles in stable order. It disables those joints but retains their
descriptors, limits, drives and handles. It preserves current body poses, COM
velocities and angular velocities while newly free components acquire implicit
free roots. Breaking an external loop does not detach another tree edge.
Repeating the call for already-broken joints makes no edit. The method does not
emit threshold events or invent a measured load.

`joint_broken` remains queryable through unrelated compaction. `repair_joint` or
explicit `set_joint_settings` with `enabled=true` clears broken state and restores
the authored joint. Authoring another setting with `enabled=false` does not
relatch it. Repair uses the retained authored frames; it does not automatically
refit anchors to separated bodies. Broken edges no longer reserve authored tree
connectivity or world anchors: replacement edges can be inserted, and a repair
that would create a cycle or duplicate world anchor fails atomically.

Only affected joint components lose structural/state dependency identities.
Subsequent edits do not propagate through broken edges. Unrelated sleeping bodies
remain asleep. GPU resources are recompiled when the topology changes, not on
every warm frame. Unsupported *active* compliant loops still fail explicitly on
GPU; stored broken/disabled loops are not executed and do not block the backend.

## Collision policy

A broken joint no longer contributes its joint-specific collision exclusion.
Independent user exclusions remain independent:

- `set_body_pair_excluded` stores an unordered stable body pair and expands every
  current collider/shape combination on those bodies during compilation.
- `set_collider_pair_excluded` stores an unordered stable collider pair and expands
  all its compiled shapes, including compound colliders and world colliders.
- Their corresponding query methods report the explicit policy, not the union
  with joint or mask filtering. Removing one explicit policy cannot remove another.
- Deletion prunes incident pairs; dense compaction remaps their compiled shape IDs.
- Runtime `step_filtered` exclusions remain an additional independent filter.

These exact sets do not encode pairs into a fixed-width group hash. Existing
membership/filter masks continue to apply alongside them. Normal joint disabling
retains its previous collision policy; persistent *broken* state specifically
removes that joint's restriction.

## Automatic CPU execution

`RigidWorld::step` and `step_with_forces` select the breakable CPU coordinator when
the scene has active finite thresholds. The coordinator invokes the existing
contact-refreshed or sleeping solver path for one configured substep at a time.
It samples total solved reactions, collects every joint reaching either threshold,
then detaches that batch before the next solver substep. Sleeping/unsolved owners
have no samples. Threshold evaluation does not depend on public reaction readout
being enabled, and a subthreshold frame retains its existing compiled model.

A host-state checkpoint covers solver-only commits within the unchanged model.
The first actual break forks the completed boundary state and retains the exact
original scene for rollback. Further detachments occur on the working scene.
A later failed substep discards provisional events and restores the original
topology, state, validity stamps, activation and sleep/settling state. A failure
before any detachment restores the host checkpoint without compiling a new model.
As in ordinary stepping, invalid-input quarantine occurs before this transaction.
External side effects performed by a user force callback are not rolled back.

Contact samples retain their real substep durations. Reaction means use the full
frame duration, so unsampled post-break time contributes zero. Legacy loop-force
means are aggregated too. Surviving joint drive/brake efforts are accumulated by
stable body/joint/local-DOF keys and mapped to the final compiled layout rather
than copied across changed dense offsets. Events and readouts are published only
after the entire frame succeeds.

`joint_break_events()` returns committed events retained until
`drain_joint_break_events()` consumes them. Each event includes stable joint and
endpoint handles, trigger channels, thresholds and the actual child/parent wrench
sample with anchor positions, substep duration/index and frame-relative end time.
Its `frame` is the zero-based successful outer-step index;
`completed_steps()` advances once per successful outer call, not per substep or
failed attempt. There is one event per intact-to-broken transition. Explicit
repair permits a new transition. Events are historical records: deleting a joint
or endpoint later does not erase previously committed events, and a handle lookup
for a deleted object returns no object.

The checkpoint API is for solver-only commits with unchanged authoring/controls;
it is not a general undo API for arbitrary scene edits. Model identity or
dynamics-input mismatches are rejected. `RigidScene::fork()` independently compiles
the current host scene and preserves its live kinematics/controls without sharing
mutable state buffers.

## Automatic GPU execution

The reaction store kernel compares the instantaneous child force/torque norm with
uploaded limits and appends triggering samples to a device event buffer. A separate
ordered single-invocation pass publishes the stop bit after every owner has finished
sampling. Bit `1 << 30` signals successful early termination; all other solver-status
bits remain faults. Host event ordering within a simultaneous batch is by stable
joint handle, not nondeterministic atomic append order.

Queued remaining Featherstone passes copy the halted boundary state through the
ping-pong buffers but do not integrate it. Constraint solves also skip on the stop
bit. Redrift keeps the corrected boundary velocities while copying unchanged
coordinates/poses. Reaction stores skip halted substeps, and sample buffers are
cleared at the start of each segment so old samples cannot become new events.

With active thresholds, one 12-byte notification is read after the queued segment:
solver status, completed substeps and event count. No body state is transferred
per substep. The regular boundary state and exactly `112 * event_count` event bytes
are then read. Without public reaction readout there is no full sample history
allocation or transfer. A nonbreaking one-body fixed scene reads 80 bytes total
(12 notification, 4 status, 64 pose/velocity), independent of substep count.

An actual break uses the same lazy scene fork and rollback checkpoint as the CPU
path. It detaches the complete batch, recompiles supported topology and resumes
the remaining substeps with their original duration. Segment means are weighted
by their requested duration over the full frame; samples/events receive global
frame-relative substep indices. Only the complete successful frame publishes
events, state validity and frame diagnostics. A later failure restores the original
scene and invalidates compiled GPU state; historical compilation counts are retained.

GPU channels use inclusive comparison after conversion to f32. Disabled channels
remain positive infinity; explicit zero remains zero. Positive finite thresholds
that overflow f32 or become subnormal/zero are rejected rather than silently
changing semantics. No epsilon is added to the physical decision. CPU/GPU exact
event timing is not promised inside a fixture's measured f32 load-error band;
parity gates must use thresholds outside that band. The controlled support/stop
fixtures use absolute load tolerances of 1e-3 N and 0.05 N·m respectively, not a
universal tolerance for arbitrary mass, length or load scale.

## Remaining integration boundary

Automatic breaking must evaluate each actual substep reaction before averaging,
collect all simultaneous triggers, and cease affected constraint action before
the next substep. A frame-end test of the mean or a delayed frame-end detach is
not a replacement for this behavior.

Extend CPU execution to actuator-device/wheel/track combinations with their
controller state, per-frame hooks, specialised connector meanings and readouts
included in the transaction. Measure CPU checkpoint/reconstruction cost and extend
moving-chain, mass/length-scale, repair and failure coverage.

Extend GPU coverage to mixed moving chains, scaling, capacity/resource gates and
the runtime's missing contact/compliance/control features. Measure notification,
halted queued dispatch and rare reconstruction costs over the required workloads.
Preserve specialised connector tensile/shear behavior rather than replacing it
with generic norm thresholds.

## Validation entry points

- `make test-primitives-breakage-cpu`: configuration/transaction and live CPU
  automatic-break, detachment and contact-policy tests.
- `make test-primitives-breakage-gpu`: explicit and automatic live GPU detachment,
  inactive metadata, compact transfer and capability-boundary tests.
- `runtime_break_configuration.rs`: exact threshold channels/zero/infinity,
  invalid batches, state preservation, compaction, replacement/repair topology,
  independent collision policies and dependency identities.
- `runtime_break_detachment.rs`: fixed, hinge, slider, ball, D6, Free, radial and
  external-ball detachment, newly free acceleration, loop/tree load sharing,
  unrelated sleep and actual CPU contact suppression/response.
- `runtime_joint_breakage.rs`: gravity/time-step oracles, per-substep transient and
  scalar-stop peaks, channel/equality/disabled semantics, simultaneous/sequential
  detachment, post-break failure rollback, sleep selection, optional readout,
  model retention below threshold, contact continuation, compliant-loop motion and
  legacy reactions, and surviving motor diagnostics across new root DOFs.

- `runtime_joint_breakage_gpu.rs`: support/free-motion oracles, delayed stop peaks
  and sequential boundaries, independent channels, a 130-owner simultaneous batch,
  sleeping zero thresholds, compact warm transfers, repair, tree/loop kinds and
  rollback after newly free motion overflows.

These fixtures establish the tested CPU/GPU behavior, not the complete P3b
acceptance gate or the required broad performance certification.
