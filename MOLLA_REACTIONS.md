# Joint reaction reconstruction

## Contract and current status

P3 reactions are total transmitted loads, not commanded motor efforts. A child
sample uses world force in N and world torque in N·m about its joint anchor.
Impulse and angular impulse are distinct quantities; substep loads must be
sampled before frame averaging or evaluating break thresholds.

`molla_solvers::rnea::inverse_dynamics_wrenches` implements the CPU inverse-dynamics
building block. Given one articulation's generalized accelerations and one world
COM external wrench per model body, it returns generalized efforts and a wrench
for every compiled joint, including zero-DOF fixed joints. Both output arrays use
articulation-local ordering. State body poses must agree with generalized
coordinates. Gravity is included by the evaluator and must not also be included
in the external array. `state.body_f` is not implicitly read.

This function computes the loads required for its supplied motion. The CPU runtime
now uses it for opt-in solved reaction readout. The GPU runtime uses a retained
general device evaluator and device-generated substep samples. Explicit detachment
and automatic CPU/GPU substep breaking are described in `MOLLA_BREAKAGE.md`;
actuator-device integration remains unimplemented.

## CPU runtime readout

`RigidWorld::set_reaction_readout(true)` enables reconstruction after every CPU
substep. It is disabled by default. `joint_reaction(handle)` returns the completed
frame's weighted mean child/parent wrenches, sampled duration and maximum child
force/torque magnitudes. `joint_reaction_samples()` returns individual samples,
their authored anchor positions, substep index, duration and end time **relative
to this frame**, not an absolute simulation clock. Wholly sleeping joints have
no samples. Means are weighted by substep duration divided by the full frame
duration, including any unsampled sleeping interval as zero contribution.

The discrete acceleration is `(solved_qdot - input_qdot) / substep_dt`; inertia,
velocity bias, load reference shifts and anchor positions use the input pose.
The solved velocity includes hard stops, motors and distance/cone corrections.
External loads include the force callback's actual persistent/transient COM
wrenches, acted penalty-contact forces, and external constraint-row loads.
Compliant loop forces already present in the force callback are not added twice;
their separately solved cone torques are added. Tree drives/compliance/cones are
internal and are not subtracted from the inverse-dynamics load.

Compiled tree endpoint direction is mapped back to authored child/parent handles.
Rigid ball, radial and compliant loops use the same public sample representation.
Radial/rigid-ball rows act at each endpoint's own anchor. Compliant loop forces
act at the child anchor on both endpoints, requiring a parent moment shift when
anchors differ. These are actual application-point conventions, not an assumed
coincident-anchor idealization.

Publication occurs only after a successful step and device post-step processing.
A failed step attempt invalidates the previous frame. The separate
`RigidScene::dynamics_stamp()` invalidates reactions after state, force, control,
gravity and dynamics-description edits, including non-waking force/motor edits.
The existing model/scene stamps retain their narrower meanings. Stable handles
remain valid through unrelated dense compaction, but a new solve is required.

Both readout backends support kinematic boundaries under the runtime's
piecewise-constant prescribed world COM
and angular velocities. Kinematic poses remain externally authored and are held
during a step; enabling readout does not advance them. Changes between authored
frames are not interpreted as an unrequested acceleration/impulse estimate.

For a prescribed boundary with COM C and fixed recursion origin O,
`v_O = (v_C - omega cross (C-O), omega)`. Zero prescribed COM/angular acceleration
gives `a_O = (-omega cross v_C, 0)`, before the inverse-dynamics gravity offset.
Setting this spatial acceleration to zero would instead create the spurious COM
acceleration `omega cross v_C`. Both CPU and GPU Featherstone now use the corrected
boundary term; CPU RNEA uses the same convention in its translated origin.
An explicit nonzero prescribed-acceleration input is not yet available and remains
a separate capability gap for arbitrary accelerating boundary trajectories.

## Derivation

At a fixed recursion origin O, each body contributes

```
v_i = v_parent + S_i qdot_i
a_i = a_parent + S_i qddot_i + v_i cross (S_i qdot_i)
f_i = I_i a_i + v_i cross_dual (I_i v_i) - f_external_i
```

Root acceleration includes `-gravity`. The existing Free/D6 translation
subspace derivative correction is retained. External COM torque shifts to O as
`torque_O = torque_COM + (COM - O) cross force`.

The backward pass adds descendant wrenches before sampling each joint. Its
generalized effort is `S_i transpose f_i`; this projection may have no entries
for a fixed joint even though the unprojected wrench is nonzero. At child anchor B,

```
force_B = f_i.linear
torque_B = f_i.angular - (B - O) cross f_i.linear
```

All spatial vectors in one recursion use the same fixed origin. Choosing the
current first child's body origin as O removes irrelevant large world
translations from inertia and torque arithmetic. O is held fixed during that
evaluation; it is not a moving-frame derivative convention. Legacy
`inverse_dynamics` retains its effort-only interface and does not allocate the
additional joint-wrench output vector.

For an ideal action/reaction pair, shifting the negative child wrench from B to
parent anchor A gives `F_parent = -F_child` and
`T_parent = -T_child + (B-A) cross (-F_child)`. Runtime integration must preserve
the actual endpoints and application points of external constraint rows. It must
not silently replace separately applied row forces with an ideal coincident-
anchor assumption when numerical or compliant anchor separation is nonzero.

## Validation

`crates/molla-solvers/tests/reaction_wrenches.rs` currently checks:

- A fixed chain under gravity and external COM loads, including lever-arm torque,
  zero generalized DOFs, and a million-metre common translation.
- A moving ball joint with rotated local frames, off-centre COM, full inertia,
  angular acceleration, centripetal acceleration and gyroscopic torque against
  direct Newton–Euler formulas, including a translated copy.
- A free body whose applied acceleration needs no transmitted joint wrench.
- A solved hard scalar stop transmitting torque between floating bodies while
  every commanded drive effort is zero.
- Articulation-local ordering and malformed load/acceleration rejection.

Existing inverse-dynamics/mass decomposition and reversed-frame tests also pass.
These are component oracles, not the complete P3 acceptance gate.

`runtime_joint_reactions.rs` adds eleven live-runtime checks: fixed support and
transient peaks, uncommanded scalar-stop load, Ball/D6 transmitted components,
reversed endpoint/anchor signs, rigid/compliant loop load accounting, cone-only
soft loops, radial sleep selection, contact subtraction, edit/compaction
invalidation, failed/disabled frames and an empty kinematic-only scene.

`runtime_kinematic_reactions.rs` checks translating/rotating kinematic parents,
off-centre COM centripetal loads through fixed/hinge/ball trees, full-inertia gyro
torque with external loads, reversed authored endpoints, translated/rotated frames
and different substep counts.
The GPU cases check both the corrected zero-acceleration hinge oracle and live
centripetal reaction readout for fixed/hinge/ball children in translated/rotated
frames with 1/4/8 substeps. The pre-fix CPU fixture produced a spurious -0.002 rad/s hinge rate in
one millisecond; the corrected fixture passes a 1e-10 rad/s bound. GPU accumulated
rate remains below 2e-5 rad/s across twenty steps in the tested frames.

The existing GPU RNEA shader is scalar-only: it previously assigned zero width
to every non-hinge/non-slider kind and silently returned for oversized chains.
Its constructor and dispatch now reject unsupported multi-DOF/reversed/locked
models, prescribed body motion and articulations above its 64-joint scratch limit. Four host-side
capability tests cover those errors and the supported boundary. This compatibility
kernel is not used for runtime reactions; the separate general implementation
described below supplies multi-DOF motion subspaces and full wrench output.

## Integration and acceptance still required

1. Extend device load assembly to the remaining runtime capabilities, particularly
   contacts, compliant loops and control-body loads. Keep unsupported paths explicit.
2. Add explicit nonzero prescribed boundary acceleration and validate all joint kinds,
   moving/off-centre chains, finite-state failures, contact coupling and CPU/GPU
   agreement across timestep, mass and length scales.
3. Implement break thresholds/events/topology transactions using substep samples,
   never frame means, and validate peak-load break decisions.
4. Measure opt-in CPU reconstruction cost and retained GPU resource/latency gates.

The current CPU Featherstone path does not read `control.body_external_f`; its
applied body loads come through `state.body_f` and its force/contact machinery.
The GPU runtime rejects nonzero control body wrenches. A reaction implementation
must not subtract a control load that the dynamics did not actually apply; this
control-surface gap also needs resolution before the full handoff.

## General retained GPU component

`molla_kernels::ReactionWrenchKernel` is the general wrench evaluator, separate
from the compatibility scalar-only `InverseDynamicsKernel` above. It uses the
shared articulated-response motion frames/columns for Fixed, Revolute, Prismatic,
Ball, Free, D6, Distance and Cable, including reversed traversal, equality locks,
disabled subspaces and prescribed kinematic boundaries. Like CPU RNEA it uses a
translated common origin and subtracts world COM external loads. Inertia-vector
products are evaluated directly; no dense 6-by-6 scratch matrix or stored array of
six columns per joint is needed. Joint columns are reevaluated in the backward
pass before projection to generalized effort.

`create_buffers` retains global-DOF acceleration/effort buffers, world COM loads,
compiled-joint child-anchor world wrenches, an articulation selection mask and
fault status. `prepare` retains bindings to uploaded model/state buffers.
`ReactionWrenchDispatch::encode` clears its status and encodes evaluation without
creating buffers, groups or pipelines, and without uploads or readbacks. Unselected
articulations explicitly clear their outputs. Bit 0 flags nonfinite output; bit 1
flags a capacity violation. A caller must consume/accumulate status before another
evaluation clears it and must reject all output when status is nonzero.

The component supports at most 64 compiled joints per articulation and requires
27 storage-buffer bindings. Host validation checks layout/dimensions, dispatch
capacity and output-buffer size/usage. Empty scenes encode only a status clear;
zero-DOF fixed trees retain real wrench output using dummy bindings for empty
DOF arrays. Replacing model/state buffers requires preparing bindings again.
This contract is not the live GPU runtime capability surface, which remains
limited by Featherstone and its other runtime passes.

The GPU oracle covers 144 articulations, 432 joints and 738 DOFs across all eight
joint kinds, floating/fixed/kinematic roots, reversed traversal, locked and
disabled motion subspaces, rotated frames, off-centre COM and full inertia.
Peak scaled wrench error is `2.4120215e-6` against a `3e-4` gate. Generalized
efforts also pass comparison. Selection clears stale output; nonfinite input sets
fault status and corrected input recovers. Empty, one-joint and 64-joint branching
fixed trees pass. Forty retained evaluations keep resources unchanged at
`[59 buffers, 773680 bytes, 4 groups, 3 pipelines, 3 shaders]` in the test fixture.
These counters include its model/state and internal wgpu resources, not only the
reaction pipeline. CPU RNEA now also masks disabled subspaces; an independent
static-support test verifies that latent rates/accelerations do not leak into them.

`make test-primitives-reactions-gpu` runs the component, live runtime, kinematic
GPU and legacy scalar RNEA tests. This is not the full P3 acceptance gate.

## Live GPU runtime readout

`set_reaction_readout(true)` also enables GPU runtime reconstruction. At each
substep the constraint pass retains instantaneous row loads and the complete
predictor-plus-constraint generalized velocity increment. Acceleration is formed
on device from that increment divided by substep duration, avoiding subtraction
of two rounded velocities. External radial/ball-loop rows are assembled at each
endpoint's own anchor into world COM loads. Tree cone/stop reactions remain
internal and are not subtracted. Compiled body-to-external-joint incidence ranges
avoid scanning every joint for every body during load assembly. The evaluator
uses the substep input state.

A second device pass maps compiled results to authored endpoints, stores anchor
positions and per-substep wrenches, and reduces weighted means and force/torque
peaks. The host only decodes these results at the existing single frame readback;
it does not solve reactions or reduce loads. Model/state bindings and sample
storage are retained across warm frames. Changing readout mode or sample count
rebuilds the cached resources. Selection masks are uploaded at frame boundaries;
sleeping owners publish no samples.

Evaluator or sample faults are accumulated into runtime status bit `16384` before
the next evaluator dispatch clears its local status. Any fault discards the entire
frame without committing state or publishing readout. Peak magnitude calculation
rescales large/small vectors before normalization and checks overflow explicitly.
On the tested GPU, directly dividing a vector near `2e38` by its largest component
produced zero through reciprocal underflow, hiding an overflowing peak; the
scaled calculation rejects that frame instead.

The eight live tests cover fixed support with off-centre COM and applied torque,
an 800 N·m instantaneous stop impulse versus a 200 N·m frame mean, moving Ball/D6
transmitted components and tree cones, capped motors against stops, external
loop/tree load accounting for world and dynamic parents, sleeping/waking,
dense compaction, stale samples, reaction-fault rollback/recovery, empty scenes
and readout/substep reconfiguration/capacity rejection. A moving floating chain
with rotated joint frames, off-centre COM and full inertia matches CPU substep
wrenches with peak absolute error `1.5672723e-4` against a `3e-3` gate over twenty
four-substep frames. Peak-norm fixtures span `1e-30` through `1e38` force components
before the explicit overflow case. Ten warm fixed-support frames retain
`[71 buffers, 528160 bytes, 19 groups, 6 pipelines, 5 shaders]` including model and
internal wgpu resources. Each reads 1044 bytes: status, one body's pose/velocity,
one 80-byte mean and eight 112-byte samples. These are bounded fixture measurements,
not a general performance certification. These resource counts predate the
additional GPU break buffers and ordered halt pipeline.

Warm external-loop cases also retain their resources: 109 buffers, 35 groups,
9 pipelines and 8 shaders; storage ranges from 533616 to 537024 bytes for the
tested radial/ball links with world/dynamic parents. The complete named GPU
reaction lane passes 15 tests. An additional 39 GPU distance/cone/runtime
regression tests pass, including the unchanged loaded-twist precision gate.
Evidence is in `benchmarks/primitives/reactions-2026-09-28/runtime-gpu-complete.log`
and `runtime-gpu-regressions.log` in the same directory.
