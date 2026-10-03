# Molla primitives implementation report

**Status: complete for the original Molla primitives scope; all original gates pass.**
Accepted 2026-10-03. The [plan](MOLLA_PRIMITIVES.md) retains its original scope;
its acceptance checklist is complete. No Gearbox/USD conformance is claimed.

## Session checkpoint

### Goal
Implement the complete original MOLLA_PRIMITIVES.md P0–P6 scope, including
public CPU/GPU runtime APIs, physical contracts, original gates and handoff.

### Instructions
Use Make/Nix and patch tools. Preserve physical thresholds and workloads.
No commits, branches, worktrees, sibling/model edits, dependency-pin changes
or termination of external applications.

### Discoveries
- Published high/low pose reconstruction must not add the old residue twice.
- Floating momentum transport must subtract relative COM terms before reducing
  them; independent analytic motion exposed a bias hidden by trajectory parity.
- Separating impact contacts remain graph edges but do not consume solve capacity.
- Prescribed boundary quaternions are normalized high/low pairs, not zero-tail inputs.
- LBVH timing failed under heavy external GPU load, then passed unchanged at
  low load. Both failed attempts and telemetry are retained.

### Accomplished
- All original gates pass on the same 1482-path source snapshot:
  static/check/strict Clippy/shaders; six CPU and six GPU examples; CPU primitives
  586 passes; workspace test-all 2364 passes; harden 4774 passes across both
  configurations; original full GPU aggregate 413 passes, zero failures/ignored.
- Full GPU run completed with exit0 at2026-10-03 17:54:55+02, with97 nonempty
  test summaries and all1482 source hashes verified afterward. Its100k LBVH
  check passed at21.92ms against30ms, without test or threshold changes.
- Original requirement map, public API, lifecycle, physical maxima, support
  boundaries and56-row performance ledger audited. Performance rows retain
  their source snapshots; unavailable native/transient peaks are disclosed.
- Final report, public examples and proposed Gearbox signatures are delivered.
  Prior chronological report is archived as acceptance-history-2026-10-03.md.
- Final documentation reconciliation completed at2026-10-03 18:05+02:
  all12 original checklist items are checked,all1482 physical source hashes
  still verify,and git diff --check passes. The final expanded dirty inventory
  contains3333 paths,including the preserved pre-existing deletion.

### Next Steps
No remaining implementation or acceptance work within this plan.
Gearbox/USD adapter implementation remains a
separate project; the handoff describes proposed changes without making them.

### Relevant Files
- MOLLA_PRIMITIVES.md — original scope and completed acceptance checklist.
- MOLLA_PRIMITIVES_REPORT.md — final evidence and explicit support boundaries.
- MOLLA_PRIMITIVES_HANDOFF.md — actual Molla APIs and proposed Gearbox signatures.
- MOLLA_PRIMITIVES_PERFORMANCE.tsv —56 source-labelled performance attempts.
- MOLLA_CONE_GEOMETRY.md — swing representation, Jacobian and frame contract.
- crates/molla-solvers/examples/runtime_primitives.rs —six CPU/GPU public examples.
- benchmarks/primitives/engine-program-cache-2026-10-01/low-load-full-gpu-acceptance.log —original full GPU terminal pass.
- benchmarks/primitives/engine-program-cache-2026-10-01/quaternion-boundary-source-sha256.txt —accepted physical source identity.

## 1. Revision,environment and provenance

| Item | Recorded state |
|---|---|
| Checkout | /home/bresilla/data/code/OUSD/molla |
| Branch/base | develop,da495a0971c666c7230dd49e0b2a48d0307fc7a4 |
| Worktrees | One existing develop worktree; no new branch/worktree/commit |
| Native toolchain | Nix Cargo/rustc1.94.1 |
| Actual adapter | NVIDIA GeForce RTX4080,Vulkan,driver595.91.07; hardware assertions enabled |
| Source identity | quaternion-boundary-source-sha256.txt,1482 source paths |
| Final dirty snapshot2026-10-03 18:05+02 | 3333 expanded paths:248 modified,3084 untracked,one pre-existing deletion |
| Scope | Molla only; unrelated existing changes preserved |

Artifact paths below are relative to
`benchmarks/primitives/engine-program-cache-2026-10-01/` unless another path is
given. `low-load-full-gpu-acceptance-source-after.log` verifies the final
1482-path physical source snapshot. The manifest covers source,
fixtures,Makefile and dependency inputs; changing this report does not change
the recorded physical source. Before/after guards surround every runner stage.
`compliant-repair-source-delta.diff` proves the sole change since the paired-rate
manifest is runtime_break_detachment.rs; production source,other fixtures,
Makefile and dependency inputs are unchanged.

The subsequent paired-position-source-delta.diff records exactly two changed
production paths:rigid_contact_precision_gpu.rs and featherstone.wgsl. Older
aggregate results below are labelled historical and are not promoted to current
shader certification. The five-test precision file is accepted on this source.

The historical05:44+02 dirty snapshot contains2953 expanded paths:248 modified,
2704 untracked,one pre-existing deletion of
`crates/molla-geometry/shaders/lbvh.wgsl`. It is preserved in
`cloud-native-mode-final-dirty-files.txt`; it is not a fresh final path count.
`paired-rates-next-dirty-files.txt` contains the newer10:21:53+02 progress snapshot.
`acceptance-audit-2026-10-03-dirty-files.txt` is the current expanded inventory.
Refresh it after any further changes; do not use a grouped-directory status
count or silently attribute all pre-existing changes to this task.

Baseline environment/commands/failures are preserved under
`benchmarks/primitives/baseline-2026-09-28/`. Historical evidence is not silently
promoted to current-source certification.

## 2. Public API and runnable examples

Authoring uses `molla_sim::runtime::RigidScene`,opaque handles and fallible
mutations. Physical values use metres,kilograms,seconds,radians,N and Nm.

| Capability | Public surface |
|---|---|
| Radial limits | DistanceLimits { min: Option<Real>,max: Option<Real> }; set_distance_limits / distance_limits |
| Ball cones | SphericalConeLimits { angle0: Option<Real>,angle1: Option<Real> }; set_cone_limits / cone_limits |
| Break settings | JointBreakLimits { force: Option<Real>,torque: Option<Real> }; set_joint_break_limits / joint_break_limits |
| Break state | joint_broken / repair_joint |
| Explicit friction | FrictionCoefficients { mu_static,mu_dynamic }; ContactMaterial.friction: Option<FrictionCoefficients> |
| CPU world | RigidWorld::from_scene(scene) |
| GPU world | RigidWorld::from_scene_gpu(scene,Arc<Engine>) |
| Solved readout | set_reaction_readout,joint_reaction,joint_reaction_samples,distance_reading,cone_reading |
| Events | joint_break_events / drain_joint_break_events |

Scene settings getters return `Result`; solved world readouts return `Option`
because disabled,unsolved or stale data is not a fabricated zero sample.

The complete compiling minimal public-runtime examples are in
[runtime_primitives.rs](crates/molla-solvers/examples/runtime_primitives.rs).
Its `distance`,`cone`,`reaction`,`breaking`,`friction` and `d6` functions create
scenes,bodies,joints/materials and execute through the selected world backend.

```sh
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
nix develop --impure -c make -j1 example-primitives A=all
nix develop --impure -c make -j1 example-primitives-gpu A=all
```

These authoring fragments use a valid scene/joint of the stated kind; the linked
example supplies complete imports,body insertion,world construction and stepping:

```rust,ignore
// Distance joint.
scene.set_distance_limits(joint, DistanceLimits {
    min: Some(1.0), max: Some(1.0),
})?;

// Ball joint.
scene.set_cone_limits(joint, SphericalConeLimits {
    angle0: Some(0.25), angle1: Some(0.4),
})?;

// Any supported breakable joint.
scene.set_joint_break_limits(joint, JointBreakLimits {
    force: Some(5.0), torque: None,
})?;

// Collider contact material.
collider.contact_material.friction = Some(FrictionCoefficients {
    mu_static: 0.8, mu_dynamic: 0.4,
});

let mut world = RigidWorld::from_scene(scene)?;
world.set_reaction_readout(true)?;
world.step(1.0 / 240.0)?;
let reaction = world.joint_reaction(joint);
let events = world.drain_joint_break_events();
```

The snippets are additive authoring examples,not a single scene with mutually
incompatible joint kinds. GPU selection in the executable requires a hardware
adapter and errors when unavailable; it never substitutes CPU execution.

## 3. Execution routes and support boundaries

CPU: scene authoring/compiler -> compiled Model/State -> Featherstone and
runtime coupled constraint/contact response -> reaction recovery and substep
break decisions -> boundary-safe topology changes -> scene publication.

GPU: authoring/compiler -> model/control upload on rebuild -> retained
Featherstone ping-pong state -> device distance/cone/loop/contact response ->
device reaction accumulation and break reduction -> compact notification ->
boundary-safe host topology rebuild -> checked frame readout -> scene commit.

Host validation,sleep selection,topology orchestration and frame publication
are explicit. They are not a whole-scene per-step CPU physical solve.

| Capability | CPU live runtime | GPU live runtime | Certification scope |
|---|---|---|---|
| Distance,optional bounds,world/body/external endpoints | Implemented | Device response/readout | Runtime,contract,sleep,COM,10-second and mixed-contact gates pass in the final aggregate |
| Ball anchors,elliptical swing/free twist | Implemented | Device anchor/cone rows | Geometry/capability/limits,Ball-loop10-second and mixed-contact gates pass in the final aggregate |
| Fixed/Revolute/Prismatic/Ball/D6 external rigid/compliant links | Implemented | Retained device response | Current full GPU baseline loop/drive files pass |
| Tree/loop reactions and automatic break | Implemented | Device wrench/peak/threshold solve; host topology boundary | Current reaction/break files pass; contact breakage5/5 in source-identical remainder |
| Static/dynamic contacts and restitution | Implemented | Resident contacts/material/impact response | Current static10s/three-rate,kinetic/reversal/contact lifecycle and restitution6/6 pass |
| D6 axes,motors,frames,COM/full tensor | Implemented | Device coupled bounds/drives | Current full baseline and144-case axis certification pass |
| Density aggregation/live mass edits | Implemented | Host validation/recompile,device force response | Current CPU1/1 and GPU1/1 pass |
| Edits,stable handles,sleep,pair permissions and faults | Implemented | Atomic frame publication/retained buffers | Current CPU aggregate,public GPU contract12/12,contact sleep7/7 and resident filtering/lifecycle pass |
| Primitive/hull/mesh authoring | Explicit CPU representations | Packed primitive contact kinds0–6 | GPU unsupported admitted pairs error,not CPU proxies |
| Registered wheel/track/actuator stepping | Legacy CPU path; automatic break with devices rejected | Explicitly rejected | Not certified by this package |
| Runtime Cable | Explicitly unsupported | Explicitly unsupported | Lower-level Cable/FEM is separate |

GPU articulations above32 joints are rejected. Unknown motor models,unsupported
external kinds and unsupported coupled bounds return errors. Supported scalar
bound kinds include Revolute,Prismatic,D6 and Ball. Capacity errors abort rather
than truncate; unsupported or unrepresentable positive f32 parameters are not
silently dropped.

Resident GPU narrowphase supports sphere,plane,box,capsule,cylinder,cone and
ellipsoid. Rounded-cylinder,hull/convex-parts,triangle-mesh and heightfield
authoring is not a GPU collision substitute. Unsupported packed shapes use
sentinel0xff and an admitted pair faults; disabled/excluded/nonoverlapping
pairs need not generate a fault. Separate legacy geometry fallback behavior
does not certify this production lane.

Mass computation uses analytic primitive/mesh integration and additive convex
pieces. Heightfields have zero volume; rounded-cylinder mass uses the documented
inner-cylinder approximation,not a claim of exact rounded volume.

Immutable program initialization scopes Validation,OutOfMemory and Internal
errors before cache publication. Models/state remain world-owned. Checked
capacity recovery is tested separately from actual driver OOM; a genuine late
native-OOM recovery is not claimed by the existing capacity fixture.

## 4. Conventions and lifecycle

- Distance constrains authored anchors,not origins/COM. Optional bounds are
  independent; equal bounds are bilateral; zero equality does not weld rotation.
  Bounds are finite,nonnegative and ordered. Radial internal slots are not public
  LinX semantics. Coincident-anchor limiting response is deterministic.
- Cones use geodesic swing in the cyclic transverse basis of the authored primary
  axis,not independent Euler clamps. Extents are independent radians in[0,pi].
  Symmetric extents reduce to primary-direction angular separation; authored
  frames,quaternion sign and free twist are preserved. The exact representation
  and Jacobian are documented in [MOLLA_CONE_GEOMETRY.md](MOLLA_CONE_GEOMETRY.md).
- Reactions are transmitted child/parent world-space force and torque about their
  respective authored anchors. Reference-point shifts recover COM torques.
  Samples use N/Nm,not impulses; sample duration,substep peaks and frame means
  are distinct. Fixed support is not inferred from commanded motor effort.
- None or positive infinity disables a break threshold. Negative/NaN values error.
  Comparison is inclusive; a valid unloaded zero sample can reach a zero threshold.
  Device decisions use physical substep samples. One event is emitted per episode;
  detachment preserves pose/velocity; draining events does not repair the joint.
  Repair is explicit. Explicit user pair exclusions survive removal of joint policy.
- Friction is a coupled stick/slip impulse solve using articulated tangential
  effective mass. It does not use a persistent tangent-displacement spring.
  Impact episode state is separate. Static response stays inside the static disk;
  sliding uses the dynamic disk and relative surface velocity,not rolling speed.
- Authoring requires finite0<=mu_dynamic<=mu_static; invalid values error rather
  than clamp. Explicit pairs override legacy scalar friction; None maps the scalar
  to equal coefficients. Combine priority is Average<Min<Multiply<Max,applied
  independently to both coefficients. Specialized tyre/track laws remain separate.
- Hot dynamics edits invalidate solved readings even when compilation stamps do
  not change. Disabled,stale or unsolved readouts remain None. Rejected mutations
  are atomic; stable handles and unrelated state survive valid compilation.
- Legacy mass_matrix/inverse_dynamics locked-column analysis differs from physical
  response_mass_matrix and transmitted-wrench APIs. Zero analysis effort is not
  proof of zero support load.

## 5. Source-labelled acceptance evidence

All commands used Make under Nix, with the loaded NVIDIA driver and actual
RTX4080/Vulkan adapter. The final GPU aggregate uses the original Make lane list,
not an override. Each runner preserves raw output, terminal exit and source guards.
All1482 source hashes pass after the final GPU run.

| Original gate | Terminal result | Raw artifact prefix |
|---|---|---|
| fmt-check,check-all,strict clippy-check,check-primitives-shader-wiring | PASS0; ten composed shader modules | quaternion-boundary-acceptance-static-shaders |
| Six CPU and six hardware GPU public examples | PASS0 | quaternion-boundary-acceptance-examples |
| make test-primitives | PASS0;586 passed,zero failed,10 ignored;90 summaries | quaternion-boundary-acceptance-cpu-primitives |
| make test-all TEST_ALL_ARGS=--no-fail-fast | PASS0;2364 passed,zero failed,1212 ignored;462 summaries | quaternion-boundary-acceptance-test-all |
| make harden TEST_ALL_ARGS=--no-fail-fast | PASS0;4774 passed,zero failed,2439 ignored;937 summaries across no-default and all-feature configurations | quaternion-boundary-acceptance-harden |
| make test-primitives-gpu | PASS0 at2026-10-03 17:54:55+02;413 passed,zero failed,zero ignored;97 nonempty summaries | low-load-full-gpu-acceptance |
| LBVH lower-load diagnostic,unchanged whole file | PASS8/8;100k22.022ms versus30ms | lbvh-low-load-gpu |

Prefixes resolve to .log and .exit files in the artifact directory from section1.
Ignored GPU tests in native/harden lanes are not counted as GPU acceptance;
the explicit full GPU gate selected and passed them. Each native aggregate also
reports464 Criterion Success lines: these are benchmark smoke,not performance
measurements. Full original workloads include chain128 and million-body XPBD.

### Physical acceptance on the final source

| Measurement | Observed evidence | Original gate |
|---|---|---|
| Distance pendulum10s,60/120/240Hz,eight substeps | Peak anchor4.367491819e-5/1.091913809e-5/2.729811238e-6m | <=1e-3m |
| Distance short CPU/GPU trajectory | Peak position8.907788264e-7m | <=1e-3m over1s |
| Distance scale sweep,40 cases/rate | Relative length2.331603355e-9;analytic position1.052800337e-5m;relative reaction1.951173826e-7 | Declared scale oracle and1% reaction |
| Free distance momentum | CPU7.894477612e-13,GPU1.554844129e-4 relative;stated1e-3 absolute floor | CPU<=1e-6,GPU<=1e-3 |
| Cone loaded boundary/reaction | Peak violation below1.9e-7rad;torque error below1.2e-5Nm under3Nm | <=1e-3rad and1% reaction |
| Cone pendulum10s,three rates | Active impacts,zero reported violation/anchor error;bounded energy/free twist assertions pass | Independent rigid-bound/anchor/energy checks |
| Rigid Ball loop10s,three rates | Peak anchor1.838465481e-5/4.742270755e-6/1.246070243e-6m | <=1e-3m |
| Ball closed-chain1s CPU/GPU | Position7.519814721e-8m,orientation4.848347164e-4rad | Both<=1e-3 |
| Static rigid holding10s,three rates | Worst tangential drift3.607929576e-5m | <=1e-3m |
| Kinetic contact acceleration | 2.981366962 versus analytic2.981347547;relative error6.51e-6 | <=2% |
| Public runtime kinetic acceleration | 3.999957815 versus4;CPU position error1.407441567e-5m | <=2%,<=1e-3m |
| Contact detachment translation | Peak analytic8.792966355e-13m at1/4/8 substeps;exact boundary,velocity,rotation and event checks pass | Boundary-safe state,one event |
| Full-run LBVH100k/10k wall time | 21.92ms/3.47ms;10k CPU108.95ms | <=30ms/<=8ms and>=2x CPU |

The full GPU log also contains independent reaction,all-axis D6,force/acceleration
drive,mutation/sleep,filter,break/repair,fault rollback,resource-retention and
dispatch assertions mapped in section7. These are live-runtime fixtures where
required; raw-kernel geometry tests are labelled separately.

### Preserved failures and superseded evidence

- Earlier original aggregate:256 passes and one100k LBVH timing failure at34.247ms.
  One diagnostic repeat failed at32.132ms. External GPU utilization was94–99%;
  Astrocraft dominated SM use. Nothing external was stopped by the agent.
- The same source/test/threshold passed at low observed load,then passed inside
  the complete original aggregate. No performance assertion was relaxed.
- Original numerical/oracle failures,fix rationale,fixtures and segmented results
  remain in acceptance-history-2026-10-03.md and implementation-history.md,with
  their named raw logs/manifests. They are not presented as final aggregate passes.

## 6. Performance evidence

[MOLLA_PRIMITIVES_PERFORMANCE.tsv](MOLLA_PRIMITIVES_PERFORMANCE.tsv) has56 rows:
distance,cone,D6,reaction,break,friction and mixed,CPU/GPU,1/100/1000/10000 instances.
55 rows are collected snapshots; GPU mixed10000 is an explicit scalar-Jacobian
binding-capacity rejection,not a successful measurement or a native-OOM event.
Its measured workload fields are blank,not fabricated zero timing.

Successful rows use20 warmups,1000 measured steps,dt1/240,four substeps and
reaction readout. Fields preserve setup and wall times,device timeline,dispatch
counts,transfer/buffer/host allocation accounting,break events,source manifest
and raw log. Unavailable native/transient device peaks remain unavailable,not0.

Examples of preserved outcomes:

| Case | Measured wall | Other evidence |
|---|---|---|
| GPU mixed1000 | 153.672224360s | Setup158.956916940s; one break; four compilations;489429 dispatches |
| CPU mixed10000,60000 bodies/articulations | 1716.064103869s | One break; host peak1028237284B |
| GPU mixed10000,60000 bodies/articulations | Not measured | Checked dense scalar-Jacobian binding capacity error |

Rows refer to several source snapshots and a heavily shared RTX4080. They are
not final-current-source isolated speedup evidence. Device timeline includes
queue/host idle intervals and is not silently described as pure kernel time.
No external desktop/simulator application was stopped for measurements.

## 7. Original requirement audit

This final-scope audit maps original requirements to inspected source and
accepted tests. Paths are under `crates/molla-solvers/tests/` unless another
crate/source is stated. CPU,native and the full actual-GPU gates pass on the
same physical source snapshot; concrete maxima and terminal logs are in section5.

| Requirement | Concrete coverage and acceptance evidence |
|---|---|
| P0 baseline/environment/routes | baseline-2026-09-28; runtime.rs/compiler,rigid_world.rs,rigid_world_gpu.rs; final provenance/source guards |
| P0 atomic settings/stamps/wake/handles | runtime scene/configuration/kind-wake/nonwaking suites; full native gate |
| P1 independent bounds,equality,zero/slack | runtime_distance*,runtime_distance_contract; public runtime,not setters alone |
| P1 world/two-body,1:1000 masses,offsets/frames/tensor | runtime_distance_gpu,contract,COM roots and coupled response fixtures |
| P1 boundary velocity/tangent/free rotation,exchange/momentum | endpoint exchange/coincident fixtures,free-body momentum,mixed-distance semantics |
| P1 loops,disconnected articulations,edits/deletion/compaction/sleep | distance contract/sleep and mixed-contact runtime fixtures |
| P1 residual/reactions/scales,10s at60/120/240Hz | distance contract40 cases/rate,eight substeps; independent reaction/momentum oracles |
| P2 ellipse/Jacobian,symmetric reduction/diagonal swing | MOLLA_CONE_GEOMETRY.md,molla-sim spherical_cone,cone_geometry_gpu |
| P2 axes/frames/free twist,q/-q,zero/disabled/wide/antipodal | runtime_cone_capability,configuration,limits and coupled angular geometry suites |
| P2 tree/external/contact anchors,reactions,energy and10s | runtime_cone_limits pendulum/loaded/two-body cases,runtime_ball_loop,mixed contacts |
| P2 mutation/sleep/compaction | cone mutation_sleep; native and actual GPU execution |
| P3a world child/parent anchors,means/peaks,fixed/motor/limit loads | reaction_wrenches*,runtime_joint_reactions,runtime_reactions_gpu; no double-counting/COM shift oracles |
| P3a stale/disabled/invalidated readout and retention | joint reaction/runtime reaction input/fault/resource fixtures |
| P3b inclusive force/torque/both/zero/disabled thresholds | runtime_joint_breakage*,substep peak and clear below/above oracles |
| P3b all six kinds,tree/loop,detachment/motion/repair | tree_and_loop_joint_kinds_detach_at_the_same_support_load,runtime_break_detachment |
| P3b simultaneous breaks,sleep/deletion,collision policy,rollback | breakage_gpu,contact_breakage_gpu,sequential boundary and explicit exclusion fixtures |
| P4 pair/legacy/combine rules and diagnostics | molla-geometry contact_materials,contact_friction,coulomb_disk_gpu; full native consumers |
| P4 inclined loaded10s,kinetic2%,re-stick/reversal/energy | runtime_friction_contract,public runtime_contacts_gpu,contact_response_gpu_tests and runtime_rigid_loop_drives |
| P4 moving surfaces,edits/separation/manifolds,sleep,compaction | runtime_friction/contact_sleep,public contact material/topology tests and device contact trajectory/material response |
| P5 D6 every axis,mixed/simultaneous drives and rotated frames | runtime_d6_axis_certification,mixed_drives,locked_drive_chains/planes,D6 angular and Ball interval suites |
| P5 hinge/slider motor models,gains,caps,mass/inertia scaling | runtime_motor_models/coupling,acceleration_drives,Ball drives and overflow validation |
| P5 all world-anchor kinds,fixed frames,free Ball and topology | joint_control,tree/loop/scene,public runtime contract and primitive-specific tests |
| P5 COM/full tensor,density/live edits | runtime_mass,COM roots,runtime_density_mass; independently derived force/torque response |
| P5 collision enable/permissions/materials/restitution/geometry | runtime_contacts_gpu,runtime_restitution*,runtime_shapes,collider_mass,resident geometry; declared unsupported boundaries |
| P5 independent worlds | runtime shared-engine gravity/state/mutation isolation and CPU counterpart |
| GPU execution/retention/dispatch/faults/no fallback | resource/dispatch/program-cache/rollback fixtures and inspected batched device route |
| P6 examples,measurements,conventions,adapter signatures | Six runnable public examples,56-row ledger,this report,MOLLA_PRIMITIVES_HANDOFF.md |
| Section11 all selected tests/nonzero original commands | All original commands pass;full GPU aggregate413 passes,zero failures/ignored,97 nonempty summaries;LBVH timing passes unchanged |
| Section14 exact return package/scope preservation | Report/history/provenance/support matrix/handoff,56-row ledger and final expanded dirty inventory inspected;no required Molla work remains |

### Public surface audit,2026-10-02 20:18+02

Inspected current declarations and implementations,not only test names:

- `molla-sim/src/joint_control.rs` validates independent finite nonnegative
  distance bounds,independent cone extents in[0,pi],and nonnegative non-NaN
  break thresholds. Its inclusive trigger accepts valid zero-load samples;
  positive infinity cannot trigger a finite physical sample.
- `molla-sim/src/runtime.rs:1338–1516` provides the handoff's getters,setters,
  broken-state query and explicit repair. Distance/cone edits rebuild
  transactionally; break-limit edits touch dynamics without a topology rebuild.
  Successful changed settings activate endpoint dependencies. Batched breaks
  disable the authored joints and preserve stable handles via scene rebuild.
- `molla-solvers/src/rigid_world.rs:125–186` checks dynamics stamps before
  exposing reaction,distance or cone readings. Joint reactions additionally
  require enabled readout. Returned data is borrowed; the proposed Gearbox
  owned DTOs therefore correctly require copying before its mutex is released.
- `joint_reactions.rs`,`rigid_breakage.rs`,`distance_constraints.rs` and
  `cone_constraints.rs` match the documented public readout/event fields.
  `molla-sim/src/contact_material.rs` confirms explicit friction pairs,legacy
  uniform coefficients,and Average<Min<Multiply<Max precedence.
- `rigid_world_gpu.rs:209–242` validates and compiles the GPU backend before
  installing it on the existing RigidWorld; errors propagate without CPU
  fallback. `runtime_primitives.rs` explicitly rejects a software adapter.
- The current example log contains all six cases on both backends and hardware
  RTX4080/Vulkan identification. Its analytic fixed-load sample is20N on both
  backends,its distance equality reads1m with zero violation,and each break
  example checks exactly one event for the authored handle. The whole example
  gate has terminal exit0; these examples do not replace the long physical gates.

Branch/base/worktree were rechecked:develop at
da495a0971c666c7230dd49e0b2a48d0307fc7a4,one existing worktree. The subsequent
full GPU aggregate passed on this same source,including the unchanged timing
gate; its terminal result is separate from the public-surface inspection.

### Original acceptance checklist disposition,2026-10-03

The plan's original requirement wording is preserved; its checklist is now
complete. These dispositions combine physical and command-level evidence.

| Original checklist item | Inspected disposition |
|---|---|
| P0 baseline/routes | Recorded baseline environment/logs exist; native1.94.1,base revision and driver agree; CPU/GPU public entry points and production route inspected |
| Runtime distance | Current public runtime,sleep,contract,COM and scale tests pass; three-rate10s errors and independent reaction/momentum oracles satisfy the original gates |
| Spherical cones | Geometry contract/Jacobian,axes and sign tests,loaded ellipse/free twist,10s impacts and tree/loop anchors pass; not Euler clamps |
| General reactions | Current reaction/boundary/kinematic and contact-reaction files pass; independent loads,anchor shifts,means/peaks and stale/faulted readout assertions inspected |
| Break/detachment | Current tree/loop kind matrix,threshold channels,one-event lifecycle,rollback/repair and contact detachment pass; independent translation refinement peaks below9e-13m |
| Static/dynamic rigid friction | Loaded10s three-rate hold and live runtime hold/slide pass; slide2.981366962 versus analytic2.981347547,breakaway refinement,re-stick/reversal and material/filter edits pass |
| Existing D6/motor/frame/mass/filter contracts | Complete37-file GPU baseline,144-case D6 axis certification,public contract12/12,resident geometry/filter17/17 and native consumers pass |
| CPU and actual GPU lanes | CPU primitives,workspace,harden and original full GPU aggregate pass;GPU413/413 on hardware RTX4080,including mutation/mixed-scene cases |
| No silent fallback | Existing RigidWorld GPU construction validates/compiles before publication; hardware assertions,unsupported mutation/atomic rollback,retained resources and batched dispatch tests pass |
| Regression/benchmark evidence | Original high-load failures preserved;final full GPU413/413 with97 nonempty summaries;56-row performance ledger has valid references and explicit measurement limits |
| Scope preserved | develop/base/one worktree unchanged,no commits or dependency-lock changes;final expanded inventory distinguishes the pre-existing deletion;no sibling/model edits performed |
| Examples/adapter return package | All six CPU and six hardware GPU examples pass; handoff matches current public types/methods,owned readout policy and explicitly proposed sibling signatures |

CPU distance-scale momentum peaks7.894477612e-13 relative with its stated floor,
below the original1e-6 CPU gate; the GPU counterpart peaks1.554844129e-4,
below1e-3. The unchanged test assertions establish the physical limits; benchmark
smoke Success lines are not substituted for measured performance. No ignored or
zero-selected GPU result is used as acceptance. The LBVH threshold and complete
original aggregate both pass; no requirement is waived because of earlier failures.

## 8. Gearbox handoff and remaining work

[MOLLA_PRIMITIVES_HANDOFF.md](MOLLA_PRIMITIVES_HANDOFF.md) records actual Molla
methods and proposed neutral Gearbox signatures. In particular:optional world
parent,fallible insertion/mutation/step,independent bound fields,owned reaction
and event DTOs,explicit friction pairs and a documented legacy setter policy.
These are proposed sibling changes,not implemented Gearbox APIs.

Gearbox still owns schema reading/defaults,sentinels,degrees/unit conversion,
collection/filter resolution,scene routing,principal-axis mapping and USD/model
acceptance fixtures. No Gearbox,Blender or published machine assets were edited.

Remaining Molla work within this plan: **none**. Declared unsupported backend
combinations and measurement limitations remain explicit in sections3 and6;
they are not hidden fallbacks or claims of universal solver/USD conformance.
Future source changes require fresh relevant verification. No commit was made.

## Key Learnings:

1. Focused fixes can pass while full acceptance remains open; ignored/queued tests
   and shader-only validation are not actual GPU certification.
2. Source-labelled history must stay distinct from the current return package.
3. Matching GPU trajectories can share a real bias; independent analytic
   substep refinement exposed cancellation in floating momentum COM transport.
4. Quaternion FK normalizes prescribed boundary rotations into high/low pairs;
   a zero-tail or unnormalized oracle contradicts that retained-precision contract.
