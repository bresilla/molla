# Molla physics primitives: implementation handoff

## 0. Goal and scope

Implement and validate the engine-level capabilities needed before Gearbox can
consume the complete standard UsdPhysics schema family. Molla must expose generic
physics APIs; Gearbox will later translate USD into those APIs.

**This is an implementation plan, not a claim that these capabilities are done.**
Prepared on 2026-09-27 by inspecting source. No builds, physics tests, or performance
measurements were run while preparing this document.

| Repository | Inspected revision | Role |
|---|---|---|
| `/home/bresilla/data/code/OUSD/molla` | `da495a0971c666c7230dd49e0b2a48d0307fc7a4`, `develop` | Implementation target; working tree was clean |
| `/home/bresilla/data/code/robolibs/gearbox` | `af6ae06` plus working-tree changes | Future consumer; do not edit during this phase |
| `/home/bresilla/data/code/OUSD/bevy_openusd` | `499d4e2` plus possible working-tree changes | USD projection and vendored schema library |

The four main work packages are:

1. Finish **distance constraints through the live runtime**, reusing existing solvers.
2. Add **spherical swing/cone limits**.
3. Add **general joint reaction readout and force/torque break thresholds**.
4. Add **separate static and dynamic friction to general rigid contacts**.

An additional certification package verifies capabilities already present: D6
locks/limits/drives, world anchors, frames, mass/inertia, collision filtering, and
runtime mutation. Do not rewrite existing primitives merely because Gearbox does
not currently import them correctly.

### Boundaries

- Work in canonical Molla `develop`; do not create branches or worktrees.
- Do not touch Gearbox, Blender files, published machine USDs, or dependency pins.
- Do not introduce USD schemas, prim paths, Bevy entities, or machine controllers
  into Molla's core physics API.
- GPU is the intended production compute path; CPU remains the reference oracle.
- Preserve the existing simplified physics and FEM options. This project is not
  the completion of tracked FEM propulsion, nor a rewrite of rubber or tyre models.
- Use the repository Makefile and Nix environment. Use patch/edit tools, not Python
  to edit files. Keep code comments descriptive, without change-history rationale.
- No commits/pushes are implied by this document. If separately requested, use
  unsigned, title-only conventional commits and the user's existing branch policy.
- Old instructions in `FULL_COMPACT.md` about experimental branches/worktrees are
  historical. Reuse technical context, not that obsolete workflow.

## 1. Engine versus USD importer responsibilities

All 17 standard schemas have reading/authoring views in upstream
[`openusd-schemas::physics`](https://github.com/mxpv/openusd/blob/main/crates/openusd-schemas/src/physics/mod.rs).
That library is a data layer, not a physics solver.

| USD schema | Physics Molla must provide | Later Gearbox responsibility |
|---|---|---|
| `PhysicsJoint` | D6 constraint with six configurable axes | Decode generic joint, limits, drives, frames |
| `PhysicsFixedJoint` | Weld preserving both local frames | Stop discarding frame rotations |
| `PhysicsRevoluteJoint` | Hinge, angular limits, motors | Axis/frame/unit mapping |
| `PhysicsPrismaticJoint` | Slider, linear limits, motors | Axis/frame/unit mapping |
| `PhysicsSphericalJoint` | Ball joint plus cone limits | Read primary axis and independent cone bounds |
| `PhysicsDistanceJoint` | Radial minimum/maximum anchor separation | Convert bounds and disabled-bound sentinels |
| `PhysicsRigidBodyAPI` | Dynamic/kinematic/fixed bodies, velocities, sleeping | Schema flags, scene ownership, unit conversion |
| `PhysicsMassAPI` | COM, full inertia, density-based mass aggregation | USD mass precedence and principal-axis rotation |
| `PhysicsCollisionAPI` | Colliders, enable state, filtering | Collider geometry and authored flags |
| `PhysicsMeshCollisionAPI` | Supported collision representations | Approximation selection and explicit unsupported-mode policy |
| `PhysicsMaterialAPI` | Contact friction, restitution, density properties | Resolve bindings and authored/default values |
| `PhysicsCollisionGroup` | Explicit effective collision permissions | Evaluate collections, merging, inversion and group rules |
| `PhysicsFilteredPairsAPI` | Pair exclusions at required collider/body scope | Resolve targets and expand relationships |
| `PhysicsDriveAPI` | Position/velocity/effort actuation, force/acceleration models | Map USD drive parameters to existing motors |
| `PhysicsLimitAPI` | Per-axis free/locked/bounded behaviour | Decode API instances and USD limit conventions |
| `PhysicsArticulationRootAPI` | Articulated dynamics and external/loop constraints | Identify topology without imposing blanket self-collision suppression |
| `PhysicsScene` | A simulation world with its own gravity | Scene ownership and multiple-world routing |

Schema presence is not proof that a feature runs. Maintain separate labels for
**parsed**, **runtime-configurable**, **CPU-solved**, **GPU-solved**, and **validated**.

## 2. Verified starting points

Paths below are relative to the Molla root unless stated otherwise. Search by
symbol when lines drift; the indicated behaviour was inspected at the revision above.

### 2.1 Distance already exists below the runtime

- `crates/molla-sim/src/joint.rs`: `JointType::Distance` exists.
- `crates/molla-sim/src/builder.rs`: `ModelBuilder::add_joint_distance` exists;
  its model layout uses seven coordinates and six velocity DOFs.
- `crates/molla-solvers/src/featherstone_integrate.rs`: distance projection exists,
  including the `rigorous_distance` branch.
- `crates/molla-kernels/shaders/featherstone.wgsl`: GPU distance handling exists too.
- `crates/molla-solvers/tests/featherstone_polish_cpu.rs` and
  `featherstone_polish_gpu.rs`: existing distance fixtures.
- `crates/molla-solvers/tests/featherstone_multidof_gpu.rs`: related multi-DOF coverage.
- **Blocker:** `crates/molla-sim/src/runtime/tree.rs::compile_joints` and
  `crates/molla-sim/src/runtime.rs::add_joint` reject distance joints:

```rust
JointType::Distance | JointType::Cable => {
    Err(Error::Build("runtime joint kind is not supported".into()))
}
```

Removing that rejection alone is not completion. Audit interval semantics, two-body
response, off-centre anchors, loop constraints, reactions and GPU runtime access.

### 2.2 Existing axis controls and runtime topology

- `crates/molla-sim/src/joint_control.rs`: `JointAxis`, `JointAxes`,
  `JointAxisSettings`, `JointMotor`, `MotorModel::{Force, Acceleration}`.
- `crates/molla-sim/src/runtime.rs`: `set_joint_limits`, `set_joint_locked_axes`,
  `set_joint_motor`, `set_joint_settings`, `set_joint_frames`, loop-joint insertion.
- `JointDesc::parent: Option<BodyHandle>` already represents a world anchor.
- `JointSettings` currently exposes `enabled` and `contacts_enabled`, not general
  force/torque break thresholds.
- `crates/molla-solvers/tests/joint_control.rs` is an API/test-pattern exemplar:
  stable handles, `Result` validation, explicit motor caps, frame-aware assertions.
- `crates/molla-solvers/tests/runtime_tree.rs`, `runtime_scene.rs`,
  `runtime_mass.rs`, `runtime_limit_reactions.rs`, `runtime_softness.rs`, and
  `loop_forces.rs` are regression entry points.

### 2.3 Cone limits and joint breakage

- Ball joints exist. No dedicated spherical cone-limit configuration was found in
  the inspected runtime/control API. Scalar angular bounds are not an elliptical cone.
- `crates/molla-solvers/src/loop_forces.rs::LoopReaction` reports world-space force
  on the child and torque about its joint anchor for loop constraints.
- `crates/molla-solvers/src/devices/connectors.rs` has breakable docking welds using
  tensile/shear thresholds. This is useful groundwork, not general USD-style joint
  force/torque breakage for all tree and loop joints.
- General reaction recovery must include constrained directions. Motor effort alone
  cannot measure the load in a fixed joint, ball joint, or locked D6 axis.

### 2.4 Rigid friction and the runtime/GPU boundary

- `crates/molla-sim/src/runtime.rs::ColliderDesc` has a single `friction` value.
- `crates/molla-sim/src/contact_material.rs::ContactMaterial` configures stiffness,
  damping, friction regularization and coefficient-combination policy.
- `crates/molla-solvers/src/penalty_contact.rs::eval_penalty_contacts` uses a single
  contact friction coefficient and regularized Coulomb response.
- `crates/molla-solvers/src/rigid_world.rs::RigidWorld` is explicitly a **CPU**
  rigid-world orchestrator; construction selects `FeatherstoneSolver::new_cpu()`.
- GPU solver kernels existing elsewhere do not prove that the live runtime takes
  a GPU path. The handoff must identify and test the complete execution route.
- Tyre/FEM friction facilities are separate. For example,
  `crates/molla-fem/src/signorini.rs` has static/kinetic friction parameters.
  Do not duplicate or silently change those specialised models.

## 3. Implementation order and completion ledger

| ID | Work package | Dependency | Effort/risk | Initial status |
|---|---|---|---|---|
| P0 | Baseline, execution-path matrix, runtime contracts | None | Medium | Not run |
| P1 | Runtime distance constraints | P0 | Large/high | Lower-level CPU/GPU code exists |
| P2 | Spherical cone limits | P0 | Large/high | Dedicated primitive not identified |
| P3a | General reaction-wrench readout | P0; cover P1/P2 once available | Large/high | Loop groundwork exists |
| P3b | General breakable joints | P3a | Large/high | Connector-specific groundwork exists |
| P4 | Separate rigid static/dynamic friction | P0 | Large/high | General rigid runtime has one coefficient |
| P5 | Existing-primitive certification and integrated GPU gates | P1–P4 | Large/high | Not run |
| P6 | Public examples, measurements and Gearbox handoff | P5 | Medium | Not written |

These are scope/risk estimates, not elapsed-time promises. Implement one vertical
slice at a time: public runtime API, CPU oracle, GPU path, mutation behaviour, tests.
Do not finish all CPU work and quietly leave GPU as an unspecified future task.

## 4. P0 — Establish contracts and executable baselines

### Steps

1. Record current commit, dirty state, toolchain, GPU, driver and backend.
2. Compare this plan's symbols to current source. Preserve unrelated work.
3. Run the baseline lanes in section 11 and record failures before edits.
4. Map the CPU and GPU routes from public scene insertion to integration, including
   contacts, constraints, reaction recovery, mutations and readback.
5. Identify which existing GPU functions accept compiled `Model`/`State`, and how
   live `RigidScene` settings and topology reach that representation.
6. Reuse the existing runtime/compiler/solver structure. If GPU runtime dispatch
   is absent, make the required dispatch/compilation bridge an explicit dependency.
   Do not invent an unrelated second world API or label CPU execution GPU support.
7. Add the tests first for unsupported/incorrect cases. Do not remove guards until
   the replacement path and validation exist.

### Shared API rules

- Engine units are metres, kilograms, seconds, radians, newtons and newton-metres.
- Use existing `Real`, vector/matrix types, opaque handles and `Result` errors.
- New proposed names below are design sketches, **not APIs that currently exist**.
- Optional bounds use `Option<Real>` or an equivalent explicit enabled flag.
  USD's negative sentinels must not leak into Molla's public physical parameters.
- Validate finite values, allowed ranges, endpoint handles and applicable joint
  types. Rejected mutations leave scene state, settings and usable handles intact.
- Preserve existing revision/stamp distinctions for parameter edits versus topology
  rebuilds. Unrelated handles, poses, velocities and controllers must survive.
- Document wake behaviour. Unsupported combinations return clear errors, never
  silently skip constraints or change a dynamic body to fixed.
- Provide settings readback and per-step diagnostics, not setters alone.
- Defaults preserve existing behaviour unless a documented corrective change is
  required. Test legacy callers before changing shared structures or GPU layouts.

**Gate:** a written execution-path matrix and reproducible baseline logs exist;
new CPU/GPU work has named entry points rather than an assumed backend.

## 5. P1 — Runtime distance constraints

### Physical contract

Constrain the distance between the two **joint anchors**, not body origins or COMs.
For world anchor positions `a` and `b`, let `d = length(b - a)`.

- No bounds: no radial constraint.
- Minimum only: prevent `d < min`; no attraction when farther apart.
- Maximum only: prevent `d > max`; no repulsion when closer together.
- Both bounds: free inside the interval; unilateral response at either boundary.
- Equal nonzero bounds: bilateral fixed-length constraint; rotations remain free.
- Zero-length equality: coincident anchors, without arbitrarily welding rotation.
- Bounds must be nonnegative; reject `min > max`, NaN and invalid finite values.

USD disables each negative distance bound independently. Gearbox will translate
that convention later; it must not require both bounds to be authored.
[USD distance semantics](https://openusd.org/release/api/class_usd_physics_distance_joint.html)

### Proposed runtime surface

```rust
pub struct DistanceLimits {
    pub min: Option<Real>,
    pub max: Option<Real>,
}
// Proposed methods on the existing runtime, names subject to local conventions:
// set_distance_limits(joint, limits) -> Result<()>
// distance_limits(joint) -> Option<DistanceLimits>
```

Do not represent radial bounds to callers as `LinX` limits. Any use of the first
internal DOF slot belongs inside the implementation and must be documented there.

### Implementation sequence

1. Characterise existing CPU/GPU distance algorithms and the tests cited in 2.1.
   Establish which cases currently use positional clamping versus a physical solve.
2. Add validated settings storage, readback, runtime compilation and state copying.
   Keep cable support separate; do not remove its rejection as collateral work.
3. Apply distance response through the coupled body/articulation effective mass.
   For separated anchors use `n = (b-a)/d`; each endpoint includes its lever arm
   from COM and corresponding angular Jacobian. A world/fixed endpoint has zero
   inverse response. Two movable endpoints must share response, not just move child.
4. Use complementarity/one-sided activation at interval boundaries. Include
   outward velocity at an active boundary; do not wait for gross penetration.
5. Handle `d≈0` explicitly. A normalised zero vector must never create NaNs.
   A deterministic limiting direction or special coincident-anchor formulation
   must be documented and CPU/GPU equivalent.
6. Support distance links as external/loop constraints between existing trees,
   not only as reduced-coordinate tree edges. Reuse loop infrastructure where
   appropriate without incorrectly converting a radial constraint to six axis locks.
7. Preserve rotations, world-anchor semantics, enabled state and pair permissions.
8. Expose distance residual, active bound and reaction contribution for diagnostics
   and later breakage. Extend the GPU data path and runtime dispatch in the same slice.

### Required tests

- World/body and body/body; equal and unequal masses, including 1:1000 ratio.
- Offset anchors, rotated local frames and non-diagonal body inertia.
- Minimum-only, maximum-only, interval, equality, unbounded, zero-length cases.
- Start inside/outside/on boundary; inward/outward velocities and tangential motion.
- No force within slack region; no locked rotation; no false damping of free motion.
- Free two-body momentum conservation and endpoint-exchange symmetry.
- Link closing a loop and link between two otherwise independent articulations.
- Enable/disable, limits change, deletion, unrelated compaction and handle survival.
- CPU/GPU trajectory, activation and reaction comparison; repeated GPU execution.

**Gate:** all cases run through public runtime-created scenes, not just a manually
built `Model`; 10-second constrained fixtures satisfy section 10 tolerances.

## 6. P2 — Spherical swing/cone limits

### Physical contract

Keep ball-joint anchors together while limiting relative **swing** about a primary
axis. Twist about that axis stays free unless independently constrained.

- Support unequal angular extents in the two transverse directions.
- Support disabled limits independently and a completely unrestricted ball joint.
- Respect both authored local frames and arbitrary primary-axis orientation.
- A pair of independent Euler-angle clamps is not a cone constraint.
- Do not constrain twist accidentally or depend on quaternion sign.

USD defines two angular limits relative to the primary axis and its cyclically
ordered transverse axes; negative values disable the corresponding limit.
Angles are degrees in USD, radians in Molla.
[USD spherical-joint semantics](https://openusd.org/release/api/class_usd_physics_spherical_joint.html)

### Proposed runtime surface

`SphericalConeLimits { angle0: Option<Real>, angle1: Option<Real> }`, associated
with the existing ball joint and a documented joint-frame axis convention.
Expose setter, getter, residual/activation and reaction readback.

### Implementation sequence

1. Write down the exact swing representation and boundary before coding it.
   For a symmetric cone, the result must reduce to the angular separation between
   the primary directions being at most the chosen half-angle.
2. For unequal extents, select and document an elliptical swing parameterisation
   consistent with the USD axis convention. Derive its Jacobian. Do not assume
   OpenUSD prescribes a unique numerical solver or ellipse parameterisation.
3. Define zero extent, one disabled extent, very wide angles and near-antipodal
   orientations. If an angular range cannot be solved reliably, reject it explicitly
   and mark it unsupported; a complete handoff must resolve that restriction.
4. Add validated per-joint settings and persistent CPU/GPU representations.
5. Couple active cone rows to the articulated solve and contact response. Preserve
   translational ball constraints; the angular stop must not detach anchors.
6. Handle arbitrary local rotations and quaternion `q`/`-q` equivalence.
7. Support both tree and loop/external ball constraints. Port behaviour to GPU and
   report reaction contributions through P3a's common representation.

### Required tests

- Symmetric and asymmetric cones; simultaneous diagonal swing proves the boundary
  is not a rectangular pair of angle stops.
- X, Y and Z primary axes and arbitrary rotated frames.
- Unlimited twist while the cone is active and inactive.
- Zero, disabled and large extents; antipodal stress case; quaternion sign changes.
- Gravity-loaded ball pendulum and externally torqued two-body joint.
- Repeated impact on the cone, damping without spontaneous energy growth, and
  10-second bounded motion at three step sizes.
- Runtime edits/deletion/compaction, sleeping/wake transitions and GPU parity.

**Gate:** cone residual and reaction tests pass while anchor error and free twist
remain within their independent acceptance bounds.

## 7. P3 — Reaction wrenches and general breakable joints

### P3a: reaction contract first

Provide a common reaction sample for tree and loop joints. A useful proposed record
contains joint handle, force, torque, sample time/duration, validity and frame.

Define these conventions explicitly:

- Force is the transmitted load acting on the child; torque is measured about its
  joint anchor, not silently about COM. Report frame consistently, preferably world.
- The corresponding parent wrench must be recoverable with the correct reference
  point shift. Off-centre equal/opposite forces produce COM torques.
- Distinguish force in N from impulse in N·s and torque from angular impulse.
- Include the physical contributions required to enforce constraints, active limits
  and drives. Document any decomposition; never count the same contribution twice.
- Report substep samples/peaks for break decisions separately from frame averages.
- A fixed joint under load must report a wrench despite having zero free DOFs.

Reuse `LoopReaction` and existing solver effort diagnostics where valid. Tree-joint
reaction reconstruction needs its own derivation and tests. Do not substitute
commanded motor torque or stiffness times pose error for actual total reaction.

### P3b: breakage contract

Proposed configuration: `JointBreakLimits { force: Option<Real>, torque: Option<Real> }`.
It applies to all supported joint types, not just docking welds.

- Optional thresholds disabled by default; reject negative/NaN parameters.
- Break on either configured threshold being reached by the physical reaction.
- Record force-only, torque-only or both as the trigger.
- Specify exact zero/equality semantics, especially an unloaded zero-threshold joint.
- Preserve pose and velocity on break. No teleportation, freezing or replacement
  with a hidden fixed constraint.
- Emit one event with stable handles, time/substep, thresholds and measured load.
- Keep broken state queryable; require explicit repair/re-enable, not auto-relatching.

USD exposes separate force and torque thresholds, defaulting to no finite break
limit. Molla chooses and documents numerical sampling/event policy; USD does not
specify a complete breakage algorithm.
[USD joint break fields](https://openusd.org/release/api/class_usd_physics_joint.html)

### Implementation sequence

1. Derive and validate reaction readout before adding threshold decisions.
2. Store break settings/state separately from transient dense solver indices.
3. Evaluate thresholds each substep from actual solved reactions, before averaging.
4. Disable further constraint action at the next safe solver boundary. Batch topology
   changes transactionally; preserve state and invalidate affected caches only.
5. Detach broken tree edges correctly, including newly free components. Breaking a
   loop removes only that loop, not an unrelated tree edge or articulation.
6. Preserve collision policy: removal of this joint's collision restriction must not
   remove explicit user pair exclusions. Rebuild affected collision/sleep dependencies.
7. Implement GPU reaction reduction and threshold detection. Compact event transfer
   to host for topology management is allowed; reading every body's state each
   substep just to compute breakage on CPU is not the production GPU solution.
8. Preserve specialised connector semantics. Reuse common infrastructure only when
   tensile/shear connector thresholds retain their intended meaning.

### Required tests

- Static suspended load gives the expected support force; lever arm gives expected
  torque; motor against a hard stop produces the expected transmitted load.
- Below/at/above thresholds, force-only, torque-only, both, disabled and zero cases.
- Same physical load at different step/substep sizes gives the same break decision;
  impulses must not be mistaken for forces.
- Weld, hinge, slider, ball, D6 and distance, including tree and loop constraints.
- One break event only, explicit repair, no pose/velocity reset, no dangling handles.
- Simultaneous breaks, unrelated scene edits, sleeping bodies and deleted endpoints.
- Correct substep peak detection even when frame-average load is below threshold.
- CPU/GPU event agreement away from floating-point threshold ties; documented
  near-threshold tolerance rather than an unexplained exact-frame requirement.

**Gate:** analytic reaction fixtures pass before threshold tests; event and topology
tests pass through the runtime on CPU and GPU.

## 8. P4 — Separate static/dynamic friction for rigid contacts

### Physical contract

General rigid contacts need separate nonnegative coefficients `mu_static` and
`mu_dynamic`, with static holding and kinetic sliding behaviour. Preserve Coulomb
normal-load dependence, contact frame, relative surface velocity and restitution.

- Static friction can resist tangential load up to `mu_static * normal_load`.
- Sliding friction opposes tangential slip using the dynamic coefficient.
- Returning to rest must permit sticking again without perpetual chatter or creep.
- A pure velocity-proportional force is insufficient to balance load at zero slip.
- Combine each coefficient across the two materials using the documented pair rule.
- USD provides both coefficients; exact contact-solving and transition algorithms
  are engine choices, not dictated by the schema.
  [USD physics material fields](https://openusd.org/release/api/class_usd_physics_material_a_p_i.html)

### Implementation sequence

1. Inventory the coefficient route through `ColliderDesc`, model shape fields,
   `Contacts`, contact generation, solver kernels and contact readout.
2. Add an explicit coefficient pair without silently repurposing unrelated friction
   stiffness/smoothing parameters. Preserve old single-coefficient callers through
   a documented compatibility mapping, preferably setting both values equal.
3. Choose and document behaviour for `mu_dynamic > mu_static`. Do not silently clamp
   authored values; either support them or validate/reject explicitly and expose
   the limitation to future importers.
4. Implement a static-capable law in the rigid contact path. Candidate approaches
   are persistent tangential state or a coupled stick/slip solve using effective mass.
   Pick a concrete method, specify state/energy handling and test it before rollout.
5. If state is cached, key by stable contact identity rather than transient contact
   array index. Define invalidation/transport on normal changes, separation, material
   edits, collider removal, topology rebuild and contact manifold replacement.
6. Preserve moving-surface/conveyor-relative velocity and pair-combination rules.
   Do not break the simplified tracked-vehicle path while improving rigid contacts.
7. Report the active friction regime, applied coefficient and actual force/impulse.
8. Extend GPU contact state, layout, kernels and runtime dispatch. Retain buffers
   across steps and verify capacity growth and resource reuse.
9. Reuse conceptual work from specialised FEM/tyre friction, but leave their physics
   unchanged unless explicitly required by a shared API migration.

### Required tests

- Inclined block: `tan(slope) < mu_static` holds at rest under gravity.
- Increasing slope/load crosses the static bound and begins sliding.
- Kinetic slide acceleration on a plane matches
  `g * (sin(slope) - mu_dynamic * cos(slope))` in the controlled fixture.
- Stop/re-stick and force reversal; zero friction; equal coefficients (legacy case).
- Material-pair coefficient combination, body/endpoint exchange and arbitrary plane
  orientation; translation must not change physical response.
- Moving surface contact uses relative slip and transfers force to the body.
- No friction without normal contact; no stale tangential force after separation.
- Changed/deleted materials, changing manifolds, sleeping/wake and compaction.
- CPU/GPU regime and trajectory parity, energy dissipation and sustained holding.

**Gate:** static holding is demonstrated by measured displacement under load, not
just by two new coefficient fields or a stationary zero-load screenshot.

## 9. P5 — Certify existing primitives; avoid redundant implementation

For each item, add missing coverage or repair a demonstrated engine failure.
Do not rewrite a passing capability to compensate for a Gearbox importer bug.

| Existing capability | Certification required |
|---|---|
| D6 | Every axis independently free/locked/bounded; mixed axes and simultaneous drives; rotated frames; CPU/GPU |
| Hinge/slider motors | Position, velocity, combined targets and effort; stiffness/damping/caps; force versus acceleration scaling with mass/inertia |
| World anchors | All supported joint kinds with `parent=None`; body remains movable in permitted DOFs |
| Fixed joint frames | Non-identity endpoint rotations preserve intended relative pose |
| Ball joint | Translational anchoring, unrestricted rotation without cone settings |
| Runtime topology | Tree, external/loop links, disconnected components, enable/disable and safe deletion |
| Mass/inertia | Full off-diagonal tensor, COM offset, collider-density aggregation and runtime updates |
| Collision state | Collider enable/disable and body/joint contact settings affect actual contacts |
| Pair/group filtering | Effective per-collider/body permissions survive edits; scene size not silently limited by hashed group bits |
| Contact material | Restitution response measured, not merely stored; density and friction paths actually consumed |
| Geometry | Required primitive/hull/mesh representations and mass properties are documented; no silent shape substitution |

The group/collection logic itself remains Gearbox work. If Molla's current mask
width or runtime pair API cannot express the resolved permissions, extend its
generic filter representation; do not put USD collections into the engine.

Likewise, multiple USD scenes require importer/world routing, not a new gravity
primitive. Test isolation of two independently created worlds instead.

### Important scope honesty

Existing CPU tests are not GPU certification. Existing GPU `ModelBuilder` tests
are not live-runtime certification. Record both dimensions. If a backend or solver
does not support a feature, expose that fact explicitly rather than falling back
silently. These packages do not certify every FEM/XPBD/VBD solver combination.

## 10. GPU requirements and numerical acceptance

### Execution requirements

- Constraint evaluation, response, material updates requiring per-contact work,
  reaction accumulation and threshold detection execute on GPU in the production lane.
- Use batched dispatch, persistent device buffers, deterministic indexing and bounded
  scratch allocation. Avoid per-joint submissions and per-step pipeline creation.
- CPU orchestration, validation, rare topology compilation and compact event/UI
  readback are acceptable. Document any unavoidable synchronization boundary.
- Expose errors from shader compilation/validation. A missing GPU adapter is a failed
  or blocked GPU gate, never a passing test through CPU substitution.
- Preserve the CPU oracle and test difficult cases there before comparing GPU.
- Report which live-runtime-to-GPU path is used. Do not describe today's CPU
  `RigidWorld` orchestrator as GPU merely because a shader exists in the repository.

### Proposed acceptance thresholds

These are initial project gates, not measured results or universal error bounds.
Use metre-scale, approximately 1–100 kg fixtures first. Write the fixture parameters,
solver mode and tolerances into tests before tuning. Changes need an explanation
and convergence evidence, not silent tolerance relaxation.

| Measurement | Initial gate |
|---|---|
| Finite-state checks | No NaN/Inf poses, velocities, forces or active constraint parameters |
| One-metre rigid distance/anchor error | Maximum violation <= 1e-3 m over 10 s |
| Cone boundary error | Maximum active violation <= 1e-3 rad over 10 s |
| CPU/GPU short controlled trajectory | Position <= 1e-3 m and orientation <= 1e-3 rad over 1 s |
| Analytic quasi-static reaction | Within 1% after explicitly bounded settling window |
| Analytic sliding acceleration | Within 2% after contact transient |
| Static holding | Tangential drift <= 1e-3 m over 10 s under sub-threshold load |
| Free-system momentum | Relative error <= 1e-6 CPU, <= 1e-3 GPU, with stated absolute floor |
| Break event | Exactly one event; boundary-safe detachment; no repeated events without explicit repair |

For compliant constraints, distinguish intentional deflection from numerical error;
use a separately derived equilibrium oracle and declared compliance. Do not weaken
the rigid-limit gate by silently switching to a soft spring. Run step sizes
1/60, 1/120 and 1/240 s with explicit substeps, plus mass/length scale sweeps.

For chaotic/contact-rich scenes, compare invariants, residuals and aggregate metrics
rather than demanding long-horizon identical trajectories. Near break/activation
thresholds, test clear below/above margins and quantify boundary sensitivity.

### Performance evidence

Benchmark each primitive at 1, 100, 1,000 and 10,000 instances where memory permits.
Also run many independent articulations; do not report only an easily parallel
particle fixture as an articulated-runtime result.

Record warm-up, 1,000 measured steps, GPU timestamps and end-to-end wall time,
dispatch count, host/device bytes, peak allocations, device/driver, step/substep
settings and CPU reference timing. Report OOM/capacity limits explicitly.
Include a mixed scene with distance, cone, D6 drive, contact friction and one break
event. No fixed speedup claim is required; measurements must expose regressions.

## 11. Repository-native verification commands

Run from `/home/bresilla/data/code/OUSD/molla`. These commands follow the inspected
Makefile recipes but were **not executed during planning**.

The Nix flake requires `NVIDIA_VERSION`. Use the currently loaded driver value,
not a number copied from an old session:

```bash
cd /home/bresilla/data/code/OUSD/molla
git status --short
git branch --show-current
git rev-parse HEAD
git diff da495a0971c666c7230dd49e0b2a48d0307fc7a4 -- crates Makefile
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
```

If the driver query or development shell fails, fix/report environment setup first.
Do not alter global dependencies or switch the test to a different backend silently.

### Existing CPU baselines

```bash
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test joint_control'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test runtime_scene'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test runtime_tree'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test runtime_mass'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test runtime_limit_reactions'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test runtime_softness'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test loop_forces'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test featherstone_polish_cpu'
nix develop --impure -c make test APP_TARGET='-p molla-solvers --test contact_friction_modes'
```

### Existing GPU baselines

```bash
nix develop --impure -c make test-gpu P=molla-solvers T=featherstone_polish_gpu A='--nocapture --test-threads=1'
nix develop --impure -c make test-gpu P=molla-solvers T=featherstone_multidof_gpu A='--nocapture --test-threads=1'
nix develop --impure -c make test-gpu P=molla-solvers T=featherstone_motor_caps_gpu A='--nocapture --test-threads=1'
nix develop --impure -c make test-gpu P=molla-solvers T=contact_friction_modes_gpu A='--nocapture --test-threads=1'
```

`test-gpu` wraps `nixVulkan` and selects ignored tests. Confirm the expected test
names/counts actually ran: exit code 0 with zero selected tests is not acceptance.
The existing friction GPU lane is XPBD-related coverage, not proof of the new
general rigid contact/runtime implementation.

### New lanes to implement

Create focused integration test files under `crates/molla-solvers/tests/`, for example:

| CPU file | GPU file | Work package |
|---|---|---|
| `runtime_distance.rs` | `runtime_distance_gpu.rs` | P1 |
| `runtime_cone_limits.rs` | `runtime_cone_limits_gpu.rs` | P2 |
| `runtime_joint_reactions.rs` | `runtime_joint_reactions_gpu.rs` | P3a |
| `runtime_joint_breakage.rs` | `runtime_joint_breakage_gpu.rs` | P3b |
| `runtime_static_dynamic_friction.rs` | `runtime_static_dynamic_friction_gpu.rs` | P4 |
| `runtime_primitives_contract.rs` | `runtime_primitives_contract_gpu.rs` | P5/mixed fixture |

These file names and aggregate Make targets below are **proposed, not existing**.
Add `test-primitives`, `test-primitives-gpu` and `bench-primitives` to the Makefile,
composing existing recipes and including both new tests and relevant baselines.
Keep GPU tests serialized where the existing harness requires it.

```bash
# After implementing the new lanes:
nix develop --impure -c make test-primitives
nix develop --impure -c make test-primitives-gpu
nix develop --impure -c make bench-primitives
nix develop --impure -c make check-all
nix develop --impure -c make test-all
nix develop --impure -c make harden
```

All selected tests must pass with nonzero expected counts. `test-all` alone does
not run ignored GPU acceptance tests. Record pre-existing failures separately;
never describe a failed or unrun aggregate as a pass. A clean focused test is not
proof of unrelated FEM or full Gearbox correctness.

## 12. What remains in Gearbox after Molla is ready

This is a checklist for the next session, **not authorisation to edit Gearbox now**.
Paths are relative to `/home/bresilla/data/code/robolibs/gearbox`.

- `bin/gearbox/src/physics/joints.rs`: connect generic D6 and distance instead of
  skipping them; consume generic limits and cone settings; preserve fixed frames;
  stop freezing a body for every world-anchored joint; map actual drive type and
  simultaneous targets; pass collision and break settings.
- `bin/gearbox/src/physics/attach.rs`: correctly translate independent optional
  bounds, frames, axis conventions, schema defaults and units. Existing conversion
  requiring both distance or both cone fields needs review.
- `bin/gearbox/src/physics/bodies.rs`: full tensor/principal-axis mapping, density
  aggregation and authored mass precedence.
- `bin/gearbox/src/physics/colliders.rs`: honour collision enable state; material
  coefficient pairs/density; consume resolved collision groups and all supported
  filtered-pair target kinds; correct approximation selection.
- `bin/gearbox/src/physics/scene.rs`: correct gravity units and multi-scene ownership.
- `bin/gearbox/src/physics/molla/`: expose the completed APIs through the backend,
  including world anchors, reactions and break events, without lossy defaults.
- `bevy_openusd/crates/usd_bevy/src/route/physics.rs` in the sibling repository:
  ensure authored values/defaults/relationships needed by those mappings survive.
- Add small USD fixtures covering all 17 schemas and actual runtime behaviour.
  Then test real machines. Merely loading a file or seeing a TF tree is insufficient.

Do not claim this Molla plan alone gives full USD conformance. Collection evaluation,
stage composition, defaults, unit conversion and controller mapping remain importer work.

## 13. Stop/report conditions

- An inspected rejection or API no longer matches the checkout: re-audit that path
  and update the plan before implementing a duplicate solution.
- GPU runtime dispatch cannot reach the primitive: report the missing bridge and
  implement it explicitly; do not count standalone kernel tests as completion.
- Distance or cone behaviour is implemented by teleporting only one movable body,
  freezing free DOFs, or injecting unexplained energy: fail the physical gate.
- General breakage can only observe motor effort: finish reaction reconstruction
  before claiming break support.
- A fix requires rewriting the FEM architecture or changing model assets: stop at
  the scope boundary and explain the dependency rather than expanding silently.
- An API/layout change affects unrelated crates: inventory and test all consumers;
  do not leave a supposedly additive setting breaking legacy construction paths.
- Performance testing shares a saturated GPU with another process: report it and
  ask before stopping anything. Do not terminate the user's simulator unprompted.
- Tests fail: preserve the first failing command, seed, parameters and numerical
  evidence. Do not hide it by disabling a constraint, test or GPU feature.

## 14. Return package and definition of done

Return with a companion `MOLLA_PRIMITIVES_REPORT.md` in the Molla checkout containing:

1. Exact revision and dirty-state summary; modified files and any known regressions.
2. Final public types/methods with one minimal runtime example for each work package.
3. A CPU/GPU/runtime support matrix, including explicit unsupported combinations.
4. Test commands, selected test counts, logs, numerical maxima and performance data.
5. Reaction/frame/unit conventions, break event lifecycle, friction transition law
   and legacy-parameter compatibility policy.
6. Any required Gearbox backend signature changes, without implementing them there.
7. A precise list of remaining work. Separate component evidence from integration.

### Acceptance checklist

- [x] P0 baseline and CPU/GPU execution routes recorded.
- [x] Distance constraints are live-runtime accessible and physically tested.
- [x] Spherical cones are implemented beyond independent scalar angle clamps.
- [x] General tree/loop reaction wrenches have analytic validation.
- [x] Break thresholds produce one event and preserve physical state on detachment.
- [x] General rigid contacts demonstrably hold with static friction and slide with dynamic friction.
- [x] Existing D6/motor/world-anchor/frame/mass/filter contracts remain valid.
- [x] CPU and actual GPU runtime lanes pass, including mutation and mixed-scene tests.
- [x] No silent unsupported-feature fallback and no whole-scene per-step CPU solve disguised as GPU.
- [x] Regression results and benchmark evidence are attached; failures are explicit.
- [x] No Gearbox/model edits or unintended branch/worktree changes were made.
- [x] Public API examples and the return report are sufficient to implement the USD adapter.

## Key Learnings:

1. Distance solver code already exists on CPU and GPU; the live runtime currently rejects it.
2. D6 controls, force/acceleration motors and world anchors already exist in Molla;
   Gearbox does not yet map all of them correctly.
3. General reactions/breakage, spherical cone limits and separate rigid friction
   coefficients need work beyond merely recognising USD fields.
4. GPU kernel availability, GPU runtime integration and full machine validation
   are separate acceptance gates.
