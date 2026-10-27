# Rigid static/dynamic friction

## Status

The CPU Featherstone contact-only path solves coupled tangential impulses after
the unconstrained/normal-penalty predictor and before final coordinate drift.
When runtime colliders coexist with distance, rigid-ball-loop, cone or scalar
hinge/slider position constraints, a common f64 row solve now replaces sequential
contact/joint correction. Ordinary and selected/sleeping stepping both use it.
This is a partial P4 implementation, not CPU/GPU certification.
The GPU rigid runtime now dispatches retained collision, normal-force prediction,
coupled tangential response and redrift for supported contact scenes, and publishes
substep forces through `contact_pairs()`. Opt-in GPU joint reactions include normal
and tangential contact loads. GPU break thresholds also consume these loads in
supported tree-contact scenes, with detachment and continued contact stepping.
Distance, rigid ball-loop, cone and scalar hinge/slider position-stop impulses now
share the GPU contact solve. Contact sleep/wake, nonzero restitution, finite speed
caps and non-equality Free/D6 axis bounds remain explicit errors.
XPBD, FEM and tyre friction laws have not been replaced.

The standalone GPU `ResidentCoupledConstraints` component now supports one shared
Gram solve for friction disks and bilateral/nonnegative scalar impulses, with
global physical KKT verification before friction-mode promotion. Its ten GPU
tests pass, including analytic mixed response and retained device sequencing.
`ResidentContactResponse::with_constraints` now assembles the common articulated
Gram matrix and applies mixed disk/scalar solutions, including joint-only islands.
The distance/cone GPU geometry path can supply these rows directly on device, and
a retained publisher converts their common multipliers into joint reaction samples.
These components now feed live runtime scheduling and reaction/break reconstruction
without a subsequent independent joint correction. Eight runtime tests cover
analytic response, loaded stop release/editing, capacity/rollback, endpoint exchange
and break continuation. Six component tests include retained-increment precision.
An earlier shared-runtime checkpoint passed 61 targeted GPU and 37 CPU tests, including existing contact,
reaction, break and scalar-stop regressions. This is not full P4/P5 certification.
The latest allocation checkpoint passes 247 unique CPU and 24 targeted GPU tests.
It resolves a reproduced redundant cone/contact torque-allocation mismatch that
previously changed the break decision at `0.902 N m` despite matching motion.
The common minimum-norm allocation now passes capped/interior reaction parity,
an analytic interior torque oracle and break/no-break event checks at `0.898` and
`0.902 N m`. The full mixed GPU lane is green again, including loaded stop edits
and rollback. Near-dependent/near-tangent primary solves, larger islands and
performance certification remain open. See the allocation checkpoint and exact
logs in `MOLLA_PRIMITIVES_REPORT.md`; the earlier 61-test GPU set was not all rerun
for this checkpoint.

`RigidWorld::from_scene_gpu_with_contact_capacity(scene, engine, capacity)` accepts
`GpuContactCapacity { pairs, contacts_per_island }`; `from_scene_gpu` uses 64/64.
`gpu_contact_capacity()` returns the configured capacities.
Capacity overflow aborts and restores the frame rather than truncating contacts.
Internal scalar-row reservations do not enlarge the configured contact limit.
Model/material/permission edits rebuild retained bindings. Host filter callbacks
are rejected on GPU contact scenes; use authored scene collision permissions.
Normal forces consume a separate external-wrench buffer each substep. Coordinate
and FK-position residues survive frame uploads, reset after coordinate/pose edits,
and restore from device checkpoints after a failed contact frame. Capture-only and
reaction substep-count recompilations preserve these residues when the model and
published coordinates are unchanged. Contact readback occurs only at the frame
boundary; substeps still use multiple queue submissions.

Reaction reconstruction uses retained predictor plus common constraint velocity increments
and complete external COM wrenches, including contact loads on fixed endpoints.
Those boundary loads are not applied as predictor forces to immovable bodies.
Structurally immovable bodies do not merge otherwise independent contact-response
islands, including fixed ancestors of movable descendants. Scalar hinge/slider
position bounds share the contact response and consume coordinate residues in
their gap calculation. Finite speed caps and non-equality Free/D6 axis bounds
remain guarded; exact equality locks remain supported.

Automatic detachment retains the original frame's GPU solver, bindings and
precision checkpoint until the entire frame succeeds. A later segment fault
restores that original topology and its coordinate residues and discards provisional
events. Successful detachment transfers unchanged joint-coordinate residues and
seeds new free-root translations from world-position residues on the device.
Collision permissions are rebuilt without removing explicit user exclusions.

The public GPU runtime tests cover five-second holding (peak CPU position error
`3.47e-7 m`), one-second kinetic motion (error `1.41e-5 m`, acceleration
`3.999958 m/s^2` versus `4 m/s^2`), material/permission/topology edits, capacity and
nonfinite-input rollback/recovery, static-only scenes and substep-capacity changes.
Six additional GPU tests cover contact reaction balance and COM moments, both
load directions, capture-toggle precision, fixed-boundary islands, axis-bound
rejection, immovable-endpoint bookkeeping and capture-enabled fault recovery.
Contact-break fixtures also exercise contact-triggered release, collision-policy
refresh, post-detachment capacity rollback and translation-residue transfer to new
free roots. A two-break frame checks global event times and accumulated reaction
means against the CPU reference.
These do not certify the unsupported combinations or the full P4 scope.

## Coefficient route

`ContactMaterial::friction: Option<FrictionCoefficients>` stores `mu_static` and
`mu_dynamic`. Both must be finite, with `0 <= mu_dynamic <= mu_static`; reversed
pairs are rejected, not clamped. `None` maps the existing `ColliderDesc::friction`
or builder shape coefficient to both values. Explicit pairs override that scalar.
This preserves legacy coefficient authoring, not the former viscous creep law.

Each coefficient uses the existing symmetric pair-rule priority
`Average < Min < Multiply < Max`. The model resolves each shape's explicit/legacy
values before combining. Overflow is an error; the runtime collision pipeline
checks candidate materials before narrow phase. `ContactMaterial::averaged`
combines its normal/viscous parameters but cannot resolve a missing shape scalar;
use `Model::combined_shape_friction_coefficients` for actual contact pairs.

The collision pipeline writes the dynamic value to `Contacts::friction` and the
static value to `Contacts::static_friction`. Capacity growth and contact reduction
preserve both. Manually authored legacy contacts leave `static_friction` as NaN,
which means use `friction` for both channels. The narrow-phase manifold retains
its historical dynamic-only field; the complete pair is resolved when generating
`Contacts`. Low-level manifold users must resolve the pair explicitly.

`friction_stiffness` and `friction_smoothing` retain their original meanings for
the standalone regularized penalty evaluator. They are not reinterpreted as
static/dynamic coefficients. The simplified track path clears both contact
coefficients when it owns tangential traction.

## CPU law and numerical policy

Mixed runtime scenes use the same input-pose/contact geometry with shared scalar
rows and a separate f64 block-coordinate solve, followed by one unconstrained
scalar drift. Its physical KKT gate is `1e-9`; it includes scalar-only and
immovable-endpoint groups. The contact-only algorithm below is unchanged.
Primary KKT convergence does not fix reaction allocation in redundant mixed rows;
the common solvers now select the minimum Euclidean norm of physical SI row
impulses among feasible primary minimizers. This convention uses the existing
unit-direction Jacobian rows, not arbitrary row rescalings or a compliance law.
The secondary solve preserves `G * impulse` and `free dot impulse`, retains
strictly active disk faces and exact inactive scalar zeros, and projects onto the
remaining domains. Tangent disk faces whose radial direction lies in the affine
row space are reduced to fixed coordinates. The GPU stage reuses existing
retained scratch; publication still requires the original physical KKT gate.
Final cross-backend runtime validation is recorded in the allocation checkpoint
in `MOLLA_PRIMITIVES_REPORT.md`; broader conditioning certification remains open.

Normal loads still use the existing spring/compression-damping penalty law. The
new stage removes the old tangential viscous force from the Featherstone predictor.
For each contact it constructs two orthogonal tangential COM-wrench rows and
projects them through the articulation Jacobians. Connected contact/articulation
groups share the Gram response `G = J M^-1 J^T`, including off-diagonal contact
coupling and full rotational inertia. Fixed endpoints contribute no response;
kinematic endpoint and conveyor velocities remain in the relative-slip bias.

For fixed regime limits, impulses minimize

```
0.5 * lambda^T G lambda + v_predicted^T lambda
```

over circular two-dimensional Coulomb disks of radius `mu * normal_force * dt`.
This is not independent per-axis clipping. An unconstrained spectral solve handles
feasible sticking sets. Block coordinate solves handle the bounded problem; an
interior-point disk solve handles nearly redundant contact sets when block updates
stall. The accepted tangential KKT velocity residual is at most `1e-7 m/s`.
Nonconvergence or nonfinite impulses fail the step rather than publish a partial
contact solve. Generalized impulse responses are applied once, then coordinates
are redrifted with the corrected rates.

The capture band is `1e-4 m/s` of contact-relative tangential speed, independent of
the old smoothing parameter. Contacts within that band may use the static disk.
A static trial leaving slip above the band switches to the dynamic disk; an
already sliding contact uses the dynamic disk for stopping, then may return to
the static regime on the next substep. The band permits tiny point-velocity
changes while compliant normal support redistributes across a manifold. It is a
documented numerical capture policy, not a promise of exact zero velocity at every
individual point. Sustained displacement is tested independently of regime labels.

No tangential spring or warm-start state is cached. Material edits, separation,
new manifold points, collider deletion, normal changes and topology rebuilds
therefore use freshly generated rows and cannot retain an old tangential spring.
There is no cached tangential energy to transport or discard. Relative-slip energy
is minimized for the fixed normal-load predictor; this is not a proof of total
energy stability for arbitrary penalty stiffness or a coupled normal/tangent solve.

## Readout

`ContactForce` includes the static and dynamic coefficients, `friction_regime`,
`applied_friction` (the disk coefficient used), and world `tangent_impulse` on the
second endpoint. `force_on_b` includes that impulse divided by the actual substep
duration. The opposite endpoint receives the opposite impulse at the same point.
`Inactive`, `Regularized`, `Sticking`, and `Sliding` distinguish no friction,
standalone legacy evaluation, and the new CPU regimes. `Sticking` includes the
capture band above. Existing normal forces/impulses and stable-pair runtime readout
remain available. Sum impulses across substeps; do not sum instantaneous forces.

## Evidence and open gates

`make test-primitives-friction-cpu` covers material combination/validation,
generated/reduced contact arrays, old penalty readouts, the simplified tracked
vehicle path, and live static/dynamic fixtures in `runtime_friction.rs`.

The loaded block fixture is a free rigid compound body with four spherical corner
pads, not a pinned body. It uses 2 kg, normal stiffness `1e5 N/m` per point,
compression damping `300 N s/m`, four substeps, a five-second run and no sleeping.
At slope 0.3 rad, `mu_static=0.6`, `mu_dynamic=0.2`, peak tangential displacement
is about `3.687e-5 m`, below the unchanged `2e-4 m` bound. Post-settling four-second
drift is below `1.061e-7 m` in the translated/rotated fixture, against `1e-6 m`.
An exploratory `1e4 N/m` support gave about `3.69e-4 m` initial elastic deflection;
it was not reported as satisfying the tighter displacement gate.

At slope 0.5 rad with coefficients 0.4/0.2, measured sliding acceleration is
`3.039090262261243 m/s^2` versus the analytic `3.0390902622612845 m/s^2`.
Additional cases cover stop/re-stick, force reversal, zero/equal coefficients,
conveyors, atomic invalid edits, separation/removal, collider-endpoint exchange
and dense compaction, plus sleep and conveyor-edit wake. A separate cuboid with
box contact manifolds holds the same slope without sleeping, with peak displacement
`2.360e-5 m`. Logs are under `benchmarks/primitives/friction-2026-09-28/`.

The retained GPU path now generates filtered manifolds and normal-contact COM
wrenches. `ResidentNormalContacts` combines static/dynamic coefficients, retains
restitution and penalty parameters, and computes body-relative point and conveyor
velocities in 128-byte device rows. It applies only spring/compression-damping
normal forces, not tangential or restitution impulses. Its three retained passes
reset per-body incidence heads, generate rows/links and gather body forces. The
base external-force buffer is separate from generated forces and must not be
replaced by the previous step's accumulated result.

`make test-primitives-contact-pipeline-gpu` includes CPU force/material parity,
fixed/kinematic/sensor/compound endpoints, rebuilt metadata, fault/retirement gates,
a 100-sequence device bounds/manifold/force chain and a 1,000-step GPU Featherstone
drop-and-settle test. The latter has no intermediate readback, but is a test-wired
normal-only path with separate queue submissions, not `RigidWorld` GPU collider
support. Its horizontal frictionless drop is not the loaded incline friction gate.

`ResidentCoulombDisks` now provides a retained GPU coupled tangential solver core.
It minimizes the full device-provided Gram quadratic over circular impulse disks,
using separate static/dynamic radii and the same pre-predictor capture policy.
Accelerated projected iterations use velocity KKT convergence, a relative-slip
energy guard and sticky input/nonfinite/PSD/nonconvergence faults. Inputs, island
counts, scratch and results stay on device; encoding allocates or transfers nothing.
The caller must supply `J M^-1 J^T`, free point velocities and impulse radii
`mu * normal_force * dt`. This component does not construct those rows, apply the
impulses or advance any body.

`make test-primitives-friction-gpu` checks coupled CPU parity, singular/redundant
rows, static/dynamic transitions, queued count retirement, fault/reset handling,
tangent-basis rotations, contact permutations and reciprocal Gram/cap scaling
from `1e-30` through `1e30`. Published impulses receive an independent f64 KKT check
at the unchanged `1e-7 m/s` limit. Rounded physical impulses and compensated slip
sums use atomic bit storage. Boundary impulses are reconstructed at their physical
radius; compensated high/low slip and radial/angular residuals evaluate the actual
published impulse. Near-converged stalls receive bounded direction/adjacent-float
refinement. Acceptance remains `1e-7 m/s` (with a one-part-per-million internal
rounding guard), not the earlier blanket 75% target. These sampled checks are not
an all-input f32 error proof. The 100 queued regime
fixtures use supplied velocities, not a GPU integrated holding/sliding trajectory.

`ResidentContactIslands` now builds the disk solver's device counts and row mapping
from normal-contact rows. It groups whole caller-described articulations and
explicit dependency edges; immovable/world endpoints do not merge independent
dynamic components. Positive-load normal links join response groups even when
their friction coefficient is zero. Only positive-load, positive-static-coefficient
rows occupy tangential solve slots. Components are ordered by their minimum
articulation index and contacts by original normal-row index using a retained
stable radix sort. Capacity excess is a sticky fault, never silent truncation.
The normal producer's 100-sequence device test now consumes these counts and
mappings without an intermediate readback. A deterministic articulation-member
CSR now accompanies each island for response assembly.

`ResidentContactResponse` compiles model articulation ownership and COM Jacobian
requests, retains the existing GPU CRBA/Jacobian dispatches, builds point-wrench
tangent rows, and assembles `J M^-1 J^T` from whitened rows. Free slip includes the
pre-predictor relative/conveyor velocity plus the predicted generalized-rate
increment. It invokes the disk solver and produces generalized velocity deltas,
corrected rates and world tangent impulses in original normal-row order. Encoding
requires no host contact extraction, upload, allocation, submission or readback.
Callers still own state publication, redrift and fault/rollback transactions.

`ResidentContactPublication` retains status-guarded copies for normal-force and
corrected-rate prefixes. Bind the same status used by the producer and its
consumer, then encode publication between those passes. Every nonzero status,
including runtime break-stop, leaves the destination unchanged. Publication does
not reset status or roll back outputs accepted before a later fault; frame
checkpoint/rollback remains the runtime's responsibility. Buffer aliases, invalid
prefix sizes, missing storage usage and device-capacity excess fail at preparation.

The current mass response supports positive pivots and coordinate-null rows.
Nonzero normalized pivots at or below `1e-6`, including oblique rank deficiency,
fail explicitly; there is no silent CPU or independent-body fallback. This cutoff
also limits ill-conditioned/scale-stressed articulated cases and needs further work.

Test-wired geometry -> normal forces -> Featherstone predictor -> articulated
friction -> corrected-rate redrift now holds a loaded box for ten seconds and
matches a one-second kinetic-slide acceleration. The holding fixture uses 2 kg,
four box corners, per-point `1e5 N/m` stiffness and `300 N s/m` damping, 1 ms steps,
and gravity resolved for a 0.3 rad incline. Peak tangential displacement is about
`2.28e-5 m`, with about `5.61e-7 m` creep after the first second. The kinetic case
uses a 0.5 rad incline, coefficients 0.4/0.2 and initial tangential speed 0.1 m/s;
measured acceleration is about `2.9813668 m/s^2` versus `2.9813475 m/s^2` analytically.
No intermediate readback occurs, but these tests use separate queue submissions.
They now share one geometry/contact/Featherstone status and guard both normal-force
and corrected-rate publication on the device. They are not
production `RigidWorld` collider support or joint/contact coupling certification.

A 2.5-second stop/re-stick/force-reversal trajectory also passes, including
per-step independently reconstructed KKT/energy checks. Sub-threshold reversal
displacement is about `7.82e-6 m` with a bounded 0.25-second compliant settling
window; CPU position error remains about `3.13e-6 m`. The disk radial residual now
uses compensated squared-radius differences; a 585-case direct GPU probe covers
adjacent-float boundaries, angles and reciprocal scales.

The gradually loaded breakaway test now **passes**: CPU and GPU first slide at
step 1763 with 1 ms steps, and peak position error is about `7.04e-6 m`. A stricter
curvature-scaled radial convergence check removes the previously accepted static
creep near disk capacity. Force history, coefficients, capture band and trajectory
gates are unchanged. See `MOLLA_PRIMITIVES_REPORT.md` for the reproducer and logs.

The test-wired resident trajectories now prepare Featherstone scalar/translation
rounding residues, preserving increments smaller than an f32 coordinate ULP across
substeps. Redrift starts from the input state's residues, not the provisional
predictor's. Callers must checkpoint residues with coordinates, explicitly reset
after coordinate edits/foreign writers, and reprepare after topology changes.
Quaternion integration remains single precision. Dedicated
drift tests cover three coordinate scales, repeated redrift and edit/reset cycles.
This improves coordinate accumulation; the disk convergence correction was also
needed to pass breakaway. Neither capture nor trajectory tolerances were relaxed.

Coordinate accumulation now retains both rounding errors and renormalizes after
large reversals; its regression recovers the original f32 coordinates exactly
after repeated `+1e12/-1e12` translations. This is an arithmetic test, not a physical
large-world contact certification. Breakaway timestep refinement preserves the
same 2.5-second forcing history at 1, 2, 4 and 8 substeps per millisecond. Current
peak position errors are 7.04, 8.02, 13.34 and 20.32 micrometres; all are below the
unchanged 30 micrometre gate, and CPU/GPU sliding onset matches at every resolution.
The latest focused lanes pass 37 CPU and 14 GPU tests, including both previously
failing breakaway regressions; `check-all` also passes. These counts do not include
the unchanged geometry, drift and runtime lanes verified in prior checkpoints.

Opt-in compensation now also carries translation residues through Featherstone FK.
Callers can bind `gpu_position_compensation_buffer` outputs through
`ResidentCollisionOptions::plane_position_residues`; analytic plane depths consume
the low translations before rounding. Both coordinate and body residue buffers
must be checkpointed/reset with their corresponding state. Dedicated regressions
cover sub-ULP fixed-frame motion and plane penetration at three scales with both
endpoint orders. This does not add two-part rotations, general convex GJK/EPA
precision. The production contact bridge now owns these checkpoints and resets;
analytic sphere-box depths also consume the low translations.

The disk solver now checks both speed-weighted and curvature-scaled radial
complementarity, in addition to the angular residual. A 36-case feasible static
fixture across load, direction and reciprocal-scale variations reduced peak GPU
slip from `4.29e-6 m/s` to `6.72e-8 m/s`, below the unchanged `1e-7 m/s` limit.
The CPU barrier fallback uses eleven continuation stages and validates the same
additional residual. Rounded GPU Gram matrices can still acquire small indefinite
components; general rank/scale certification remains open.

Remaining P4 work includes broader production contact
fault/rollback and resource certification, scale/material/step-size
and energy gates, broader joint-stop
and external-loop coupling, changing normals/manifolds and sleeping certification,
and the full mixed-scene/performance matrix. The normal penalty/restitution path
was not replaced. CPU and GPU now share supported scalar/distance/cone impulses
with friction within each backend. Their redundant-row reaction allocation still
differs in the reproduced cone/contact case, preventing full reaction/break parity.
