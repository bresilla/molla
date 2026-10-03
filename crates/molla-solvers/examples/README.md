# Live primitive API examples

## Analysis and physical response

`mass_matrix` and legacy `inverse_dynamics` omit finite equality Ball/D6 motion
columns: locked slots have zero matrix rows/columns and generalized efforts.
`response_mass_matrix` retains partially constrained angular columns for the
coupled response, matching the GPU mass kernel. Physical support loads come from
`RigidWorld::joint_reaction` or `rnea::inverse_dynamics_wrenches`, not zero legacy
effort slots. The wrench API retains partial angular support components and
reports world torque about the authored anchor. Independent translated/rotated
Ball/D6 full-inertia/COM oracles and the original D6/revolute analysis oracle are
in `analysis_response_contract.rs` and `multidof_oracle.rs`.

## Runtime examples

`runtime_primitives.rs` uses public `RigidScene` and `RigidWorld` APIs without a
USD importer, model assets, or benchmark helpers. Each named function constructs
an independent world and prints physical readout. These examples demonstrate API
usage; they do not replace the plan's numerical, resource, or performance gates.
The primitive implementation and its support matrix are still in progress; see
the repository's `MOLLA_PRIMITIVES_REPORT.md` for current limitations.

## Simultaneous drives and constrained inertia

`make test-primitives-mixed-drives-cpu` checks 24 world-anchored D6 bodies with
simultaneous linear effort/angular velocity drives and real sliding contacts,
plus 48 two-body Ball/D6 partial-angular-lock chains on CPU. Masses, full
off-diagonal inertia, COM offset, rotated root frames, Force/Acceleration scaling,
zero/finite/infinite caps and live angular targets are exercised. The independent
one-second rate/pose/reaction oracles retain CPU1e-8/GPU1e-3 gates; quaternion
increments follow normalized Euler, not exact exponential integration.

`make test-primitives-mixed-drives-gpu` checks the 24 root cases against both the
independent oracle and the actual CPU trajectory on hardware. Both lanes require
exact advancement, finite state/loads, zero quarantine/break events and sliding
impulses. GPU model compilation must remain unchanged after initial readout
setup. Bounds are CPU120s/GPU180s; the latter includes observed146.01s pipeline
preparation plus the full240-frame/two-substep trajectory, not reduced coverage.
Current maximum GPU pose/parity error is7.565e-6; reaction error is3.548e-6 when
normalized by mass/inertia. See `primitives-articulated-drives-all-axis-frames-mixed-regression-gpu.log`
in `benchmarks/primitives/impact-history-gpu-2026-09-29/`.

Partial angular locks change effective drive mobility. CPU feedback uses the
bilateral angular Jacobian projection across the articulation. GPU feedback now
uses the same constrained mobility across the tree, retaining the isolated-root
fast path. `make test-primitives-locked-drives-gpu` checks 48 independent two-body
articulations: Ball/D6, X/Y/Z free axes, identity/rotated signed endpoint frames,
all four Force/Acceleration pairs, full inertia/COM and one-second trajectories.
An independent two-by-two pivot-inertia oracle and actual CPU trace both pass
the original1e-3 gate. Maximum normalized state/drive-reaction error is5.501e-6;
CPU/GPU pose/velocity parity is5.159e-7. No extra GPU buffer/submission or CPU
solve is introduced. Private Gram scratch is bounded by two angular rows per
joint; the articulated projection checks every constraint row after rank
selection/refinement. The full
unprojected GPU ABA/stop/contact response is retained for constraint recovery,
rather than making late pose corrections impossible through a zero mobility.
This lane does not certify every loop/contact/lifecycle combination or P5 as a
whole. Earlier release benchmark source manifests remain historical snapshots
after this shader change; the final-source performance matrix is still open.

Current chain evidence is in `primitives-articulated-drives-all-axis-frames-mixed-regression-gpu.log`
under `benchmarks/primitives/impact-history-gpu-2026-09-29/`. The native180-second
GPU limit includes observed117.83-second pipeline preparation plus all100 frames,
two substeps and both reaction endpoints. CPU checks keep the1e-8 gate.

`make test-primitives-locked-drive-planes-cpu` and
`make test-primitives-locked-drive-planes-gpu` extend that coverage to24 noncoaxial
Ball/D6 chains with one locked and two free angular axes per joint. Different
root/descendant lock axes, signed rotated endpoint frames, full tensor/COM,
mass2/17 and all Force/Acceleration pairs exercise passive-axis motion over1.005 s.
An independent world-space pivot-inertia projection checks initial angular rates
(CPU1e-6/GPU1e-3) and support torque (1%). Subsequent reversal/disable/re-enable
checks retain anchors/locks and actual CPU/GPU trajectory parity within1e-3.
Current GPU maximum locked-angle error is1.722e-6 rad; parity is1.475e-7.
The native180-second limit includes114.40-second preparation and the full101
steps. State/reactions remain finite, no body is quarantined, and model
compilation is retained through hot controls. Evidence is
`primitives-noncoaxial-drive-planes-hot-controls-cpu-gpu.log` in the same directory.
These checks are not final all-combination P5 or performance certification.

From the repository root:

```sh
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
nix develop --impure -c make example-primitives
nix develop --impure -c make example-primitives A=distance
nix develop --impure -c make example-primitives-gpu A=cone
nix develop --impure -c make example-primitives-gpu
```

Cases are `distance`, `cone`, `reaction`, `break`, `friction`, `d6`, or `all`
(the default). CPU and GPU are explicit choices; requesting `--gpu` in a binary
compiled without the GPU feature returns an error instead of executing on CPU.
GPU examples require hardware adapter information and reject software adapters.
One hardware Engine is created per invocation and shared by the selected GPU
examples. Each example still constructs its own scene, model, buffers and state;
only immutable programs are reusable across these independent worlds. Selecting
`all` no longer creates a separate device and cold program cache for every case.
The GPU command uses the repository's `nixVulkan` wrapper. Serialize it with other
GPU validation, and do not run either command during performance measurements.

| Function | Authoring calls | Readout |
|---|---|---|
| `distance` | `insert_joint(Distance)`, `set_distance_limits` | `distance_reading`: separation, violation, active bound and child force |
| `cone` | `insert_joint(Ball)`, `set_cone_limits` | `cone_reading`: swing coordinates, violation and child torque |
| `reaction` | `insert_joint(Fixed)`, `set_reaction_readout(true)` | `joint_reaction`: frame mean and peak support loads |
| `breaking` | `set_joint_break_limits` | `drain_joint_break_events`, then a second step without a repeated event |
| `friction` | per-collider `ContactMaterial.friction` | `contact_pairs` and body velocity after sliding/rolling contact |
| `d6` | `set_joint_locked_axes`, `set_joint_motor(PositionVelocity)` | body pose after simultaneous position/velocity feedback |

## Frames, units and lifecycle

- Scene gravity and positions use metres and seconds. Mass is in kilograms,
  inertia in kg·m², forces in N, torques in N·m, and cone angles in radians.
- `parent: None` is a world endpoint, not a request to freeze the child. The
  authored endpoint frames locate each anchor. Distance leaves transverse motion
  free; the ball/cone example preserves ball anchoring and cone twist freedom.
- Distance bounds are independent `Option<f64>` values. This example uses equal
  one-metre bounds; use `None` for an absent endpoint, not a fabricated zero.
  Cone extents are separate geodesic swing half-extents, not rectangular Euler
  clamps. The loaded example uses 0.25 and 0.4 radians.
- General reactions require opt-in readout and a successful solved frame. Loads
  are in world coordinates, with each endpoint torque taken about that endpoint's
  authored anchor. They are forces/torques, not impulses; `sample_dt` is seconds.
  Missing or invalidated readout is `None`, not a physical zero wrench. Read it
  before changing the scene. Sleeping joints can have no solved sample.
- Distance, cone and ball-anchor samples also invalidate on hot force, gravity,
  control or mutable-state access, even when the compiled scene is retained.
  A successful subsequent frame publishes fresh readings; rejected and no-op
  edits preserve the existing sample.
- Featherstone adds `Control::body_external_f` to `State::body_f` once per
  substep. Both use world axes and torques about the body's COM; control loads
  are not accumulated back into state forces. `add_body_wrench` accumulates the
  state channel until `reset_body_forces`; replacing a control entry updates the
  separate control channel. Resolve `scene.body_index(handle)` when writing a
  control entry, rather than retaining a dense index across topology edits.
  Reaction readout and break thresholds include both channels.
- The fixed two-kilogram reaction example supports 20 N against gravity. The
  break example instead sets a 5 N threshold. Threshold evaluation and detachment
  occur during stepping; committed events remain queued until drained. The event
  contains its sample, trigger, original handle and frame/substep timing. The
  detached body's handle remains usable. Draining is not joint repair.
- Friction sets both static and dynamic coefficients explicitly on both contact
  colliders. The ball is free to rotate: nonzero rolling speed does not mean the
  tangential contact is slipping. This example is not a static block-holding test.
  Its collider mass is zero because the body's mass/inertia are supplied explicitly.
- The D6 example frees linear X, locks the other five axes and requests a 0.25 m
  position with zero target speed, capped at 5 N. Its printed one-second pose is
  a response sample, not a promise that the transient has fully settled.

The GPU live-runtime route retains device substeps but currently synchronizes
host state and diagnostics at frame boundaries. These examples do not certify
arbitrary loop, compliant, sleeping-contact, actuator, wheel or track combinations.
Unsupported combinations must continue to return explicit errors. Gearbox must
handle USD units, defaults, collections and schema mapping separately.

## Verification status

All six CPU examples and all six actual-GPU examples pass on the engine-local
program-cache source on 2026-10-01. Current logs are
`benchmarks/primitives/engine-program-cache-2026-10-01/examples-shared-engine-cpu-clippy.log`
and `examples-shared-engine-gpu.log`. GPU execution uses RTX 4080 / Vulkan /
NVIDIA 595.91.07 and one Engine across the six independent worlds. Feature-disabled
GPU selection also returns the explicit no-CPU-fallback error; see
`examples-no-gpu-fail-closed.log`. Full workspace formatting, all-target check and
strict all-target/all-feature Clippy pass in `post-example-workspace-static.log`.
Earlier example runs remain historical evidence, not current certification.
These examples demonstrate usage, not complete physical or
scale certification. The current benchmark partition, source identity and
remaining original 56-case matrix are recorded in `MOLLA_PRIMITIVES_REPORT.md`;
older distance measurements are not silently counted as current-source results.
Existing integration tests remain the numerical evidence:
`runtime_distance*`, `runtime_cone*`, `runtime_joint_reactions`,
`runtime_reactions_gpu`, `runtime_joint_breakage*`, `runtime_friction`,
`runtime_mixed_contacts*`, and `runtime_multidof*` / `runtime_d6_axis_certification`.

## Bounded implementation wiring

```sh
nix develop --impure -c make check-primitives-shader-wiring
nix develop --impure -c make check-primitives-gpu-wiring
nix develop --impure -c make check-primitives-gpu-step-wiring
```

The shader check parses and validates five production WGSL modules without
requesting an adapter. GPU preparation constructs two reversed-D6 worlds but
advances no physics. The optional step check compares one frame of three worlds
against matching CPU references, including an asserted active compliant angular
limit. Both GPU targets have a 120-second process deadline and ten-second kill
grace. The fixture retains nonidentity frames, full tensor/COM, a soft external
loop, motors, brakes, speed caps, damping and break thresholds. Hardware execution
is required; there is no software-adapter or CPU fallback.

The three one-frame comparisons passed on 2026-09-30 after correcting reversed
joint momentum transport, with maximum angular-rate error 2.956379141e-7 against
the declared 1e-3 tolerance. See `primitives-one-frame-loaded-compliance.log` in
the log directory above. This is not sustained numerical, contact, pole-chart,
mutation, resource or performance certification.

Active rigid XYZ phase rows have separate short gates:

```sh
nix develop --impure -c make test-primitives-singular-wiring-cpu
nix develop --impure -c make test-primitives-singular-wiring-gpu
```

These check both exact Y poles for one frame, using independently computed
combined X/Z phase intervals. They retain hard bounds and do not substitute
compliance. The CPU task covers both poles in one process; the GPU task composes
two original per-pole processes, each with a 120-second deadline. Current full
16-contact-reservation GPU checks pass in
`primitives-xyz-original-positive-current-gpu.log` and
`primitives-xyz-original-negative-current-gpu.log`. Preparation takes about
83–85 seconds per world, versus about 0.033 seconds for its one physics step.
These are preparation diagnostics, not performance-matrix timings.

```sh
nix develop --impure -c make test-primitives-chart-wiring-cpu
nix develop --impure -c make test-primitives-chart-wiring-gpu
```

These execute 1,000 one-millisecond frames across each pitch pole with bounded
but unlocked angular axes and zero gravity. The original sixteen-frame crossing
is retained, then motion continues to the authored +/-1.9 rad pitch bound and
stops. Independent orientation/rate checks include the fractional boundary step
and remaining stationary frames. Both CPU cases pass in
`primitives-xyz-sustained-crossing-current-cpu-clippy.log`; current hardware cases
pass in `primitives-xyz-sustained-positive-current-gpu.log` and
`primitives-xyz-sustained-negative-current-gpu.log`. The first CPU log also records
a progress-condition Clippy failure, corrected in the positive GPU log.
The GPU task composes both per-pole processes with 120-second deadlines.
Individual jobs can be selected without changing either physical fixture:

```sh
nix develop --impure -c make test-primitives-singular-pole-gpu PRIMITIVE_XYZ_POLE=positive
nix develop --impure -c make test-primitives-chart-pole-gpu PRIMITIVE_XYZ_POLE=negative
```

D6 angular bounds select an equivalent
intrinsic-XYZ chart using the authored intervals, including periodic angle lifts
and the alternate pitch branch; they do not expose an accumulated revolution
counter. Mixed angular locks, general crossing trajectories and active contact,
energy and resource acceptance remain open.

```sh
nix develop --impure -c make test-primitives-velocity-wiring-cpu
nix develop --impure -c make test-primitives-velocity-wiring-gpu
```

These check one-frame coupled force-model velocity drives against an independent
implicit-rate oracle. The xy inertia has principal moments 0.001/1.999, with a
physical orthogonal moment 2.0. Six cases cover rotated frames and force caps
zero/20/infinity. Both CPU and hardware GPU pass in
`primitives-velocity-active-set-physical-inertia.log`. The feedback solve uses a
bounded active set rather than a fixed count of relaxation sweeps; failed pivots
or exhaustion remain errors.

```sh
nix develop --impure -c make test-primitives-velocity-sustained-cpu
nix develop --impure -c make test-primitives-velocity-sustained-gpu
nix develop --impure -c make test-primitives-velocity-raw-gpu
```

The sustained fixture retains the original Ball cases and adds D6/Free: eighteen
closures, 100 frames at dt=0.01 with one substep, simultaneous x/y force drives
at damping 1e6 and the same physical inertia/caps/rotated frames. An independent
common-eigenvector oracle checks bounded implicit rates and normalized-Euler
orientation every frame. Targets reverse after 0.5 s, disable after 0.75 s and
re-enable after 0.9 s without layout changes. CPU gates remain 1e-8 and GPU gates
1e-3; finite state, no quarantine/breaks, exact step count and retained GPU model
compilation are checked. The raw target separately checks all eighteen roots
without coordinate compensation. Its actual hardware run passes in 47.14 s,
with peak rate error 4.269676746e-5 rad/s under the original 1e-3 rate gate;
see `primitives-velocity-raw-fallback-current-gpu.log`. This is one-step raw
coverage, not the one-second live-runtime trajectory.

`primitives-velocity-sustained-current-cpu-clippy.log` proves the CPU trajectory
in 0.07 s. `primitives-velocity-paired-response-corrected-gpu.log` proves actual
RTX 4080/Vulkan execution in 48.27 s: all-frame peak rate error
7.191560496e-4 rad/s and orientation error 3.732427834e-5 rad. Earlier failed
hardware attempts are preserved, not passing evidence. Isolated quaternion
roots retain paired motion-frame mobility through generalized response and use
matching gyroscopic/COM dynamics; stable no-brake force velocity rows defer to
the coupled feedback solve. No per-step CPU solve or GPU pipeline creation is
introduced. This does not certify arbitrary mixed drive/brake/contact scenes,
all dt/mass sweeps, or the complete performance matrix.

Affected hardware regressions also pass: original 540 off-centre Ball pivot
closures with physical motor caps/model scaling/brakes/reactions (82.73 s), and
the original 144 D6 axis cases over one second with full COM/tensor/mass sweeps
(84.49 s). See `primitives-velocity-paired-pivot-regression-gpu.log` and
`primitives-velocity-paired-all-axis-regression-gpu.log`. These are scoped
regressions, not aggregate acceptance. The new sustained trajectory's observed
GPU preparation was 47.14 s versus 23.99 s before repair; those uncontrolled
setup timings are not a throughput benchmark. Performance measurement remains
required before any performance or whole-plan completion claim.

## Coupled hinge and slider motor models

```sh
nix develop --impure -c make test-primitives-motor-coupling-cpu
nix develop --impure -c make test-primitives-motor-coupling-gpu
```

The existing two-body slider fixture now also covers coaxial hinge chains.
Seventy-two cases per feedback mode preserve the original mass, Force/Acceleration
model combinations, capped/uncapped controls and loaded/unloaded configurations.
An independent two-by-two inverse mass/inertia oracle checks joint coordinates,
physical body rates/poses and child reactions. Four substeps and forty frames
remain unchanged: explicit dt=.0004 spans .016 s, stable dt=.02 spans .8 s.
This is not a one-second trajectory claim. Original CPU 1e-8 and GPU 2e-4 scaled
gates remain unchanged, with finite state/reactions, no quarantine/breaks and
exact completed steps. GPU compilation is retained after the initial explicit
reaction-readout configuration boundary, not compared against its constructor.

CPU default both-mode test passes in 1.67 s. Actual RTX 4080/Vulkan explicit and
stable runs pass in 49.09/47.49 s, with coordinate maxima
2.374711344e-8/6.623242264e-7, rate 2.939390508e-6/3.710364803e-6 and reaction
7.035145036e-5/4.604191384e-5. Logs are
`primitives-coupled-hinge-slider-current-cpu-clippy.log`,
`primitives-coupled-hinge-slider-explicit-current-gpu.log` and
`primitives-coupled-hinge-slider-stable-readout-gpu.log`. The first GPU attempt
failed a newly added compilation-baseline assertion, not a physical oracle;
preserve `primitives-coupled-hinge-slider-stable-current-gpu.log` as failed.

The canonical GPU task composes both original mode jobs with 120-second deadlines.
`test-primitives-motor-feedback-gpu PRIMITIVE_MOTOR_FEEDBACK=stable` selects one;
the underlying test still defaults to both when the environment variable is absent.

## Mixed D6 angular point locks

Partially locked Euler angular motion uses physical equality rows while keeping
Cartesian anchor-frame angular rates. Linear locks and fully locked angular
motion retain structural zero columns. A pitch point at an XYZ pole uses two
swing-alignment equalities. Two angular points define an exact hinge with two
alignment rows and a stable twist coordinate for the remaining axis's bounds.
No dummy colliders are needed. Standalone CPU dynamics/drift and GPU dynamics
use coupled mass response for partial angular locks, bounds and speed caps;
retained GPU capacity changes require recreating the solver.

```sh
nix develop --impure -c make test-primitives-angular-lock-wiring-cpu
nix develop --impure -c make test-primitives-angular-lock-wiring-gpu
```

Both backends check 84 regular/pole, rotated-frame and one/two-lock fixtures
against independent physical pose/rate oracles, through runtime and standalone
routes, including bounds on all three remaining hinge axes. Runtime checks two
frames; standalone additionally corrects deliberately injected forbidden rates
on a third public drift frame. CPU checks pass in
`primitives-mixed-domain-drift-wiring.log`; actual RTX 4080/Vulkan checks pass in
`primitives-mixed-domain-integration-gpu.log` (24.36 seconds for the angular file).
Standalone GPU status must remain zero. Maximum GPU rate error is 3.86e-5 and
rotation error 6.48e-7 against 1e-3 gates. No software fallback or tolerance
relaxation is used.
Earlier memory and physical failure logs are preserved.
Floating/loaded-contact, mutation/break and sustained acceptance remain open;
these short fixtures are not full mixed-lock certification.

Independent mass/frame lanes cover 64 single X, single Z, pole-pitch and two-point
locks with off-diagonal inertia, COM/anchor offsets, nonidentity endpoint frames,
both pole signs, both tree traversals and allowed/forbidden loads. Runtime checks
analytic authored reaction wrenches as well as body motion. CPU/GPU runtime also
check dt=1e-4 without relaxing the 1% reaction gate. Authored fixed reference
quaternions retain their high/low components through the GPU residual evaluation;
small position-rounding errors must not create spurious reaction loads.

```sh
nix develop --impure -c make test-primitives-angular-mass-wiring-cpu
nix develop --impure -c make test-primitives-angular-mass-wiring-gpu
```

Fresh CPU/shader checks pass in `primitives-angular-point-reference-checks.log`;
the three actual RTX 4080/Vulkan tests pass in
`primitives-angular-point-reference-gpu.log` (79.27 seconds). These are focused
one-step oracles, not long-duration or loaded-contact certification.

Finite-bound runtime lanes add 32 full-inertia/COM fixtures at dt=1e-4:
lower/upper stops, inward/outward torque, both pitch poles, rotated frames and
reversed traversal. Upper-stop fixtures retain negative quaternion coordinates;
equivalent quaternion signs must not change the physical stop reaction.
Near a stop, the GPU forms a compensated local twist gap before atan2 instead of
subtracting rounded absolute angles. The authored interval lift still selects
the global branch. An undefined twist at an opposed-swing antipode remains an
explicit error; these checks do not certify that recovery case.

```sh
nix develop --impure -c make test-primitives-angular-bound-wiring-cpu
nix develop --impure -c make test-primitives-angular-bound-wiring-gpu
```

Actual GPU evidence passes in `primitives-angular-bound-reference-gpu.log`:
32 reactions meet the unchanged 1% gate, maximum torque error 2.80e-6 N m.
The GPU mass lane excludes the separately named bound/regular checks to stay
within its 120-second cap; the full baseline includes every test in the file.

Regular XYZ point lanes cover 128 full-tensor/COM/frame/reversed fixtures at
dt=1e-4, including principal and alternate pitch charts. Selected-chart matrix
components retain their tails through the near-stop angle difference; pitch
cosine uses a compensated square root with staged rounding barriers.

```sh
nix develop --impure -c make test-primitives-angular-regular-wiring-cpu
nix develop --impure -c make test-primitives-angular-regular-wiring-gpu
```

`primitives-angular-regular-reference-charts.log` passes CPU and actual GPU
with unchanged 1% reaction gates; GPU maximum torque error is 4.94e-6 N m.
These one-step checks do not certify singular phase precision or sustained
contact-loaded motion.

Narrow finite intervals retain their authored point/interval classification.
Euler rows carry exact point bits; `Model::joint_motion_flags` also carries
authored axis-point bits for ABA, mass, Jacobian and reaction response masks.
Full model upload refreshes these derived flags after limit/enable edits.
This adds one persistent 4-byte/joint host/device buffer, using existing enable
bindings. It adds no dispatch or per-frame readback; the public enable mirror
remains boolean.

```sh
nix develop --impure -c make test-primitives-angular-narrow-wiring-cpu
nix develop --impure -c make test-primitives-angular-narrow-wiring-gpu
```

`primitives-motion-metadata-gpu.log` passes 32 runtime full-tensor/COM/frame/
reversed fixtures at dt=5e-5 with width=1e-8 and equal f32 endpoint casts.
Inward motion remains free inside the authored interval; outward load reacts.
The unchanged 1% gate passes, maximum torque error 8.14e-6 N m. Existing angular,
mixed Distance and disabled-joint GPU checks also pass. Standalone sub-f32
stop-value accuracy remains separate acceptance work.

Singular phase rows now retain authored combined endpoints and sine/cosine low
tails. Physical pole sign is selected before authored pitch winding; no extra
row, buffer or dispatch is added. Focused lanes are:

```sh
nix develop --impure -c make test-primitives-angular-phase-wiring-cpu
nix develop --impure -c make test-primitives-angular-phase-wiring-gpu
```

The 64-case full-tensor/COM/frame/reversed physical oracle covers +/-pi/2,
+/-3pi/2 and negative quaternion representatives at dt=1e-5, width=1e-9.
CPU and actual RTX 4080 GPU pass; the latter takes 38.01 seconds with maximum
torque error 1.53e-4 N m against the unchanged 1% gate. Its combined command
later times out in a separate legacy regression. That regression's negative
pole passes but positive pole remains unverified after three bounded attempts;
the original 16-contact reservation is restored. This is not complete sustained,
contact-loaded or performance acceptance.

Finite two-point angular locks use a deterministic one-sided twist chart at
opposed swing. With cyclic basis `(p,b0,b1)`, the existing zero-extent cone
recovers along positive swing about b1, giving twist `2 atan2(q.b0,q.b1)` at the
cut. The analytic left-rotation derivative is -p there. The ordinary primary
twist chart is unchanged away from its singularity; both charts preserve
quaternion-sign equivalence and authored winding. Retained runtime references
select matching transverse components and low tails. No bound is silently
retired and no buffer/binding/row count changes.

```sh
nix develop --impure -c make test-primitives-hinge-antipode-wiring-cpu
nix develop --impure -c make test-primitives-hinge-antipode-wiring-gpu
```

The authored physical fixture covers all three free axes, two world frames,
forward/reversed trees, negative quaternion representatives and interior/both
violated finite stops. It uses full off-diagonal inertia, rotated local frames,
independent first-step velocity/reaction predictions and 32-step recovery gates.
The 72-case fixture passes CPU in 0.88 seconds and actual hardware GPU in 39.53
seconds. First-step reactions pass the unchanged 1% gate; all cases recover
within 1e-3 swing/bound error after 32 steps without quarantine or break events.
Strict Clippy, eight exact shader compositions and analytic derivative/recovery
unit checks also pass. This centered-anchor/zero-COM fixture does not certify
contact-loaded, floating or offset-COM cut-locus behavior. The separate 72-case
standalone CPU first-step oracle passes in 0.73 seconds. The original standalone
GPU fault 32 came from final projection feasibility/complementarity checks:
radial residual 6.10e-5 exceeded the unchanged 2e-5 internal gate. Refinement
now retains high/low mass-response corrections and evaluates residuals with
staged paired products/sums, preserving coordinate-residue scratch. Signed
bilateral corrections remain free; unilateral corrections remain nonnegative.
No row/buffer/binding/dispatch is added, but private per-articulation scratch
grows by one scalar tail per local response component. Working-precision gates
are unchanged; published velocities remain f32 and pass the physical 1e-3 gate.
All 72 standalone hardware fixtures now pass, maximum angular-rate error
1.17e-4 rad/s. Final-source Clippy and shader checks pass; runtime/standalone GPU
fixtures pass together in 57.74 seconds. Original failures/diagnostics remain
preserved. Standalone sustained/contact/floating/COM acceptance remains open.
The separate compensated-state 72-case first-step lane passes in 14.76 seconds.
Related mixed Distance and all 84 normal-chart runtime/raw/public-drift cases
also pass on CPU/GPU; GPU Distance takes 11.77 seconds and the two angular tests
57.36 seconds. These checks do not substitute for full sustained/performance
certification or standalone sub-f32 authored stop accuracy.

The standalone Make lanes include explicit compensated-state coverage:

```sh
nix develop --impure -c make test-primitives-hinge-antipode-standalone-cpu
nix develop --impure -c make test-primitives-hinge-antipode-standalone-gpu
```

The same Make lanes include a mixed partial-D6/Distance chain. A slack radial
maximum leaves the child free, while an active maximum reacts its outward load;
the independent legacy Distance projection does not override this coupled solve.
Two bodies under a disabled tree joint retain their frozen relative transform
and transmit child inertia/load to a floating parent. Focused native lanes are:

```sh
nix develop --impure -c make test-primitives-disabled-wiring-cpu
nix develop --impure -c make test-primitives-disabled-wiring-gpu
```

### Standalone authored narrow hinges

Full model upload derives `joint_limit_references` from authored f64 bounds:
high/low endpoints and full/half-angle sine/cosine references, 80 bytes per DOF.
It retains buffer identity at unchanged DOF count and refreshes after bound edits;
always use full model upload after changing raw limits. This adds one persistent
Featherstone binding (51 across five storage groups), no dispatch or per-step
metadata upload. Finite two-point hinges consume the half-angle references and
paired hard-frame removal. Raw regular XYZ rows use full-angle references and
the shared runtime paired-matrix geometry. Singular pitch-point rows compose
paired X/Z phase references and remove pitch/point frames with half-angle tails.
Prismatic/revolute and mixed translation row gaps use paired authored bounds;
scalar point classification uses authored motion flags, and endpoint publication
preserves low parts when compensation is enabled. Generic bounded-D6 activation
now includes nonstructural finite limits and velocity caps without requiring an
angular point axis; CPU activation and retained GPU row sizing share one predicate.
Structural translations publish paired authored endpoints; fully fixed D6 angular
motion publishes paired authored rotations and zero rates. Ball, Distance, D6 and
Free compensated quaternion advancement updates both high values and tails.
External runtime scalar-row ownership remains unchanged. Nonpoint near-pole pitch
rows now use paired bound references and retained sine/cosine values in raw and
runtime paths, including resolvable principal/alternate pitch tangents.
Rigorous Distance radial rows use a scaled paired vector norm and authored bound
tails; narrow radial intervals no longer collapse to a rounded equality. Ball
full-angular points now publish paired authored rotations and zero incoming
forbidden rates, matching CPU full-point semantics. Partial Ball angular limits
and Cartesian caps use coupled angular rows at quaternion/DOF offset zero; all
three Cartesian columns remain active unless all three angular axes are points.
CPU/GPU body-rate publication, shared GPU mass/Jacobian response and momentum
metadata follow that same structural rule. Public runtime external row export
includes Ball bounds/caps without changing rigid external Ball's cone-only limit
contract. Default standalone Distance also uses coupled
anchor effective mass, without requiring a D6 activator. Unbounded/slack radial
motion stays free, independent minimum/maximum bounds are unilateral, and zero
length uses three coincident-anchor rows while retaining rotational freedom.
Explicit `with_rigorous_distance(false)` retains standalone position clamping;
that approximation is not the production live-runtime distance path. Public
runtime external constraint ownership is unchanged. Retained row capacity follows
authored bounds/caps; no new binding, buffer, dispatch or per-step metadata upload.

Prepare and explicitly seed uploaded host state before initial integration:

```rust
solver.prepare_gpu_coordinate_compensation(&[&initial, &next])?;
solver.seed_gpu_coordinate_compensation(&initial)?;
```

Preparation still zeros residues. Seeding initializes quaternion/coordinate and
body-translation low parts from current host data; it rejects stale device-owned
host mirrors or host edits not uploaded. Do not reseed evolving device state from
old host data. Use an explicit import/edit boundary for a new seed/reset.

```sh
nix develop --impure -c make test-primitives-limit-reference-cpu
nix develop --impure -c make test-primitives-angular-narrow-standalone-cpu
nix develop --impure -c make test-primitives-angular-narrow-standalone-gpu
nix develop --impure -c make test-primitives-scalar-precision-cpu
nix develop --impure -c make test-primitives-scalar-precision-gpu
nix develop --impure -c make test-primitives-angular-regular-standalone-cpu
nix develop --impure -c make test-primitives-angular-regular-standalone-gpu
nix develop --impure -c make test-primitives-angular-phase-standalone-cpu
nix develop --impure -c make test-primitives-angular-phase-standalone-gpu
nix develop --impure -c make test-primitives-d6-activation-cpu
nix develop --impure -c make test-primitives-d6-interval-standalone-cpu
nix develop --impure -c make test-primitives-d6-interval-standalone-gpu
nix develop --impure -c make test-primitives-d6-interval-runtime-cpu
nix develop --impure -c make test-primitives-d6-interval-runtime-gpu
nix develop --impure -c make test-primitives-d6-point-publication-cpu
nix develop --impure -c make test-primitives-d6-point-publication-gpu
nix develop --impure -c make test-primitives-pitch-pole-cpu
nix develop --impure -c make test-primitives-pitch-pole-gpu
nix develop --impure -c make test-primitives-radial-precision-cpu
nix develop --impure -c make test-primitives-radial-precision-gpu
nix develop --impure -c make test-primitives-distance-activation-cpu
nix develop --impure -c make test-primitives-ball-point-publication-cpu
nix develop --impure -c make test-primitives-ball-point-publication-gpu
nix develop --impure -c make test-primitives-quaternion-motion-cpu
nix develop --impure -c make test-primitives-quaternion-motion-gpu
```

The 32-case dt=5e-5, width=1e-8 full-tensor/COM/frame/reversed/lower-upper/
inward-outward GPU fixture passes its independent 1% angular-acceleration oracle
with maximum error 8.95e-7 rad/s^2. Distinct authored endpoints cast identically
to f32. Strengthened retained-coordinate checks also pass: no bound violation at
1e-12 and hard-axis quaternion components below 1e-10. Reference refresh/resize,
retained buffer identity and seeding error-boundary checks pass. This fixture's
free axis is Y; wider all-axis/sustained/contact/floating acceptance remains open.
The unseeded antipodal fault is fixed: paired arithmetic now has rounding scratch
for every joint coordinate, not a four-byte fallback. That existing fallback
grows to 4*max(coordinate_count,1) bytes, without a new buffer/dispatch. Signed
bilateral rows remain active through the box solve while unilateral impulses
remain nonnegative. Both 72-case antipodal modes pass in 36.91 seconds. The
strengthened narrow test passes in 17.94 seconds, maximum acceleration error
9.43e-7 rad/s^2 and zero retained bound violations. Historical all-index internal
rounding acceptance before the scratch fix must not be assumed from body-only
physical passes; the corrected indexing is covered by these fresh checks.

Fresh standalone additions pass on RTX 4080/Vulkan/NVIDIA 595.91.07:
24 scalar cases (51.44 seconds, maximum acceleration error 1.93e-7), 128 regular
XYZ point cases (29.20 seconds, maximum acceleration error 1.07e-6), and 64 narrow
singular-phase cases (12.84 seconds). All use independent 1% acceleration gates;
scalar and phase retained bounds are checked at 1e-12. Scalar coverage includes
both joint kinds, full tensor/COM/frames, actual points and width=1e-8 intervals,
lower/upper endpoints/interiors and inward/outward effort. Regular points cover
principal/alternate charts, full tensor/COM/frames and reverse orientation at
dt=1e-4. Phase cases use width=1e-9, dt=1e-5, both poles, +/-3pi/2 winding and
negative quaternions; pitch transverse components are checked at 1e-10. Matching
CPU fixtures, shader wiring and strict Clippy pass. These are first-step proofs,
not full sustained/contact/floating/mutation or performance certification.

Bounded D6 without angular point locks also passes the independent anchored
full-inertia oracle: 192 regular width=1e-9 intervals plus 192 cap-only cases at
dt=1e-5 on each raw/public CPU/GPU route. The actual RTX 4080 raw lane finishes
in 34.31 seconds, maximum angular acceleration error 3.92e-5 rad/s^2 under the
unchanged 1% gate. Carried regular interval violations stay below 8.89e-16 with
a 1e-12 gate; cap-zero generalized rates stay below 1e-9. The public GPU runtime
finishes in 35.48 seconds; maximum reaction force/torque errors are 8.24e-5 N /
2.43e-4 N*m under independent 1% gates. Both principal/alternate charts, all
angular axes, full COM/tensor/frames, forward/reverse, lower/upper, negative
quaternions and inward/outward loads are covered. These regular interval/cap
cases do not substitute for contact/floating/sustained
acceptance. Corresponding raw/public CPU checks finish in 9.45/9.49 seconds;
pure activation, shader validation and strict Clippy pass.

Fresh publication and near-pole evidence (2026-09-30): the 96-case, four-step
CPU/GPU point fixture publishes exact translations/rotations and zero locked rates,
including perturbed starts, negative quaternions, authored frames and full bound
edits after two steps. GPU finishes in 15.64 seconds. Strengthened raw narrow-hinge
GPU checks also prove actual retained rotation advancement at 1e-12, rather than
only bound containment, in 14.11 seconds. Earlier quaternion-bound checks did not
detect stale tails and must not be interpreted as sustained-motion acceptance.

The 64-case nonpoint near-pole pitch fixture uses width=1e-9, dt=1e-5 and offsets
of +/-1e-7 about both poles, principal/alternate charts, full COM/tensor/frames,
forward/reverse, negative quaternions and both load directions. Raw/public CPU
checks pass in 0.58 seconds combined; actual GPU raw/public checks pass in 35.85
seconds combined. Raw acceleration and public force/torque use independent 1%
gates; retained pitch bounds and actual quaternion advancement use 1e-12 gates.
This is focused near-pole acceptance, not exact-pole nonpoint recovery, all-axis
sustained/contact/floating certification, or the required performance matrix.

The narrow standalone targets now also exercise eight steps without reseeding:
inward loads reach the opposite endpoint and hold; outward loads stay at their
initial stop. The two tests pass CPU/GPU in 0.15/27.98 seconds; GPU maximum final
endpoint error is 3.56e-15 under a 1e-12 gate and angular rate 1.47e-10 rad/s
under a 1e-7 gate. This proves sustained free-Y narrow motion, not all-axis,
moving-parent, floating/contact-loaded or full numerical certification.

Mixed Distance radial precision has 48 independent full-COM/off-diagonal-inertia
cases: points, width=1e-8 intervals with identical f32 endpoint casts,
lower/upper/interior starts, both load directions, axial/nonaxial displacement,
authored frames and negative quaternions. The cap-constrained D6 parent activates
the coupled solver; the oracle derives anchor/COM acceleration and angular
response from the world inertia and off-centre radial reaction, not another
backend. CPU checks plus the slack/maximum/drift regression pass in 0.01 seconds;
actual GPU counterparts pass in 28.47 seconds. Independent acceleration gates
remain 1%, and retained position/rotation/bounds gates remain 1e-12. Maximum GPU
errors are 4.30e-5 m/s^2 / 1.91e-5 rad/s^2, carried position 6.23e-15,
rotation 1.41e-15 rad and radial violation 6.00e-15. This proves first-step mixed
radial response/Distance quaternion motion, not long-run radial drift, legacy
standalone projection, mutation/contact-loaded or full performance acceptance.

The radial precision targets additionally exercise the same 48-case independent
COM/inertia oracle with standalone Distance and no D6 parent, plus 32 unbounded,
slack minimum-only/maximum-only/interval, boundary inward/outward, nonzero equality
and zero-equality cases under authored frames and free nonzero angular velocity.
CPU selected counts are four tests (including the existing mixed/slack/drift
regression); activation has a separate pure unit check. Four matching actual GPU
tests pass in 66.85 seconds. Semantic maximum retained position error is 2.67e-13,
rotation error 4.55e-16 and velocity error 2.66e-8 under unchanged gates. Both
200-step CPU legacy distance companions also pass with zero hanging-length error.
Long-horizon public-runtime acceptance, all GPU regressions and performance remain
separate requirements; these focused additions do not certify the full plan.

Ball full-point publication now has the same 96-case, four-step/edit fixture as
D6, with incoming rates and valid/perturbed quaternion starts, full COM/inertia
and authored frames. CPU passes; actual GPU Ball/D6 checks pass in 16.24/16.22
seconds, retained rotation error <=3.44e-15 (gate 1e-12), forbidden generalized
rates exactly zero and body rates within 1e-9. A separate 64-case, eight-step
constant-rate fixture proves Ball/Distance/Free/D6 retained quaternion advancement
for X/Y/Z/nonaxial rates, authored frames and negative quaternions, without
reseeding evolving state. CPU/GPU pass in 0.01/16.25 seconds; GPU maximum retained
rotation error 5.30e-15 (gate 1e-12). Its spherical inertia/zero COM is an exact
constant-rate oracle, not a substitute for loaded/contact/floating acceptance.

`test-primitives-ball-angular-cpu/gpu` now checks 384 independent raw Ball cases
and 384 public-runtime cases per backend: one/two angular points, all-axis width
1e-9 intervals and Cartesian zero caps, principal/alternate pitch, negative
quaternions, authored frames, off-centre COM and full off-diagonal inertia.
The independent anchored-inertia oracle gates COM/angular accelerations at 1%;
public reactions also have a 1% analytic gate. Raw retained quaternion motion and
Euler violations retain the 1e-12 gate; zero-cap residuals retain 1e-9. Public GPU
host snapshots are rounded f32 and are not used as 1e-12 retained-coordinate proof.
CPU passes two selected tests in 5.99 seconds plus the activation unit. Matching
actual RTX 4080/Vulkan/NVIDIA 595.91.07 GPU tests pass in 67.88 seconds. Raw GPU
maximum rotation error is 6.46e-15 rad, Euler violation 6.22e-15 rad, angular
acceleration error 5.77e-5 rad/s^2 and COM error 5.06e-5 m/s^2. Public GPU maximum
reaction force/torque errors are 9.87e-5 N / 3.31e-4 Nm. Logs are
`primitives-ball-angular-runtime-cpu-corrected.log` and
`primitives-ball-angular-gpu-final.log`. Earlier failed/diagnostic logs remain.
This is first-step tree response/reaction proof, not sustained/contact/floating,
redundant-row, mutation/breakage, full-plan numerical or performance acceptance.

These low-level disabled checks cover six joint kinds with fixed/floating roots;
they do not replace runtime enable/disable, detachment or reaction acceptance.

### Paired chain frames and sustained all-axis motion

`test-primitives-frame-reference-gpu` validates four seven-body mixed chains and
two all-fixed/zero-coordinate chains on hardware GPU. Three authored frame
uploads, reversed/disabled joints, signed quaternion tails and all four revolute
half-angle quadrants pass retained position/paired-component gates of 1e-12.
Only initial state is seeded; frame uploads do not reseed device poses. Worst
position/component errors are 1.23e-13/1.47e-14 (90.64 seconds). This is retained
kinematic/mutation evidence, not dynamic/contact acceptance.

Shared axis-angle FK now evaluates argument reduction and trigonometric Horner
polynomials using paired scalar arithmetic. Shared quaternion FK explicitly
rounds two_sum's error tail, matching Featherstone's retained arithmetic boundary.
No new buffers are introduced by those arithmetic repairs. Their extra private
work still requires the full performance matrix.

`test-primitives-all-axis-sustained-cpu/gpu` retains the original 36 simple D6
cases and adds 108 full-COM/off-diagonal-inertia cases at 1/17/83 kg. Every axis
is free/bounded/locked under both effort signs, with rotated and offset frames.
Independent projected anchored inertia sizes efforts so all bounded axes reach
their stops. The one-second, 240-frame/two-substep trajectory checks pose and COM
velocity against its analytic oracle. CPU/GPU peaks are 1.83e-7/1.65e-5 pose and
1.24e-7/4.02e-5 rate, under unchanged 1e-6/1e-3 gates (2.54/60.62 seconds).
This does not replace floating/contact trajectories or the full 10-second,
three-step-size numerical matrix.

Runtime scalar projection stops at an exactly bit-identical full sweep, then
applies its original finite/KKT/cap publication checks; physical tolerances and
row domains are unchanged. The worst-case changing-iterate bound remains 32768.

### Redundant point and zero-cap rows

`test-primitives-redundant-limits-cpu/gpu` includes independent Ball/D6/Distance
oracles: 144 loaded cases held for eight steps and 18 cases with nonzero permitted
motion. Raw CPU and hardware GPU pass; the GPU pair takes 36.44 seconds on RTX
4080/Vulkan/NVIDIA 595.91.07. Zero joint/body rates are exact in the hold fixture;
retained coordinate error is <=1.12e-16. Permitted-motion acceleration gates are
1%, and retained coordinate/rate gates are 1e-12/1e-9. No fixture gates relaxed.

`test-primitives-redundant-runtime-cpu/gpu` exercises actual RigidScene/RigidWorld.
Runtime Distance is an external radial link, not a Cartesian-axis control joint;
its cap cases use an explicit free-axis D6 tree plus the external Distance link
at matching frames. Holding reactions sum the physical contributions at their
shared anchor. Scene wrenches persist, so tests reset before assigning each load.
CPU's three selected fixtures pass in 2.98 seconds: eight-step loaded holds,
permitted motion, and an incompatible radial point/cap problem rejected without
advancing state or the completed-step counter. A separate equality-arithmetic
unit passes. Equality geometry below a scale-derived f64 roundoff bound is
resolved before division by dt; finite unilateral gaps and resolvable errors
are retained, and the common solver's convergence/allocation contract is unchanged.

The original 18-case public-runtime GPU motion fixture passes in 25.22 seconds,
and the 144-case eight-step loaded/reaction fixture passes in 40.61 seconds on
RTX 4080/Vulkan/NVIDIA 595.91.07. Exact zero-cap rates remain zero. Independent
motion/reaction gates are unchanged. The demonstrated radial/cap fault arose
when runtime FK refresh reconstructed body positions using rounded authored
frames while radial anchors used authored pairs. Compensated FK now retains
frame/axis tails and propagates paired rotations through the articulation.
The clean staged refresh regression measures 4.44e-15 m retained position error.

Frame/axis references use 80 bytes per joint, refreshed on model upload rather
than per step. Featherstone binds 52 buffers; paired FK publication adds private
work but no extra dispatch or body storage. Existing paired radial quaternion-FK
resources/dispatches and bounded rank-deficient solver work still need full
performance-matrix evidence. External radial/cone/loop rows now solve scalar
caps/bounds together even without a partial Ball/D6 angular lock; the selected
significantly incompatible radial/cap GPU fixture rejects with status 512 without
advancing pose, rates, completed steps or break events (25.65 seconds).

These selected gates do not certify the full P5 numerical/integrated scope or
complete P6/performance acceptance. No CPU substitution or hidden compliance is
claimed. Fresh evidence is in primitives-runtime-*-fk-*.log and earlier
primitives-redundant-* logs under
`benchmarks/primitives/impact-history-gpu-2026-09-29/`.

## Antipodal cone recovery

The cone evaluator now accepts constrained 180-degree swing using a deterministic
one-sided chart; see `MOLLA_CONE_GEOMETRY.md`. A disabled/pi-wide extent also
provides a short return-to-pole branch instead of a remote radial stop.
Retained GPU rows retire and change equality mode with the live chart.

`primitives-antipodal-retained-modes-gpu.log` records GPU recovery evidence;
the CPU counterpart is `primitives-antipodal-live-cpu.log`. Tree, rigid external
and compliant external routes recover in 32 short frames. Admissible-pole
checks preserve twist/free swing without angular-rate
spikes. The GPU counterpart used RTX 4080/Vulkan/NVIDIA 595.91.07 and finished
within its 120-second process cap. General rotated/floating/moving, zero-extent,
contact and sustained energy/scale acceptance remain open.
