# Deformable reinforced rubber track

## Goal and acceptance boundary

Add a volumetric deformable belt alongside the ideal continuous-belt drive.
Motor torque must reach belt nodes through sprocket/lug contact; belt-ground
contact must generate vehicle motion with equal/opposite reactions. No imposed
belt phase, prescribed belt velocity, or chassis velocity override qualifies.
Retain the current reduced model as a separately named, usable baseline.

## User-selectable simulation modes

This is a machine-wide tyre/track fidelity contract, not a CEOL-only feature.
All supported wheeled and tracked machines retain their current simplified
physics alongside an actual FEM option. Gearbox's per-machine physics menu
uses **Simplified / fast** and **FEM / experimental** consistently: pneumatic
tyres select a deformable tyre model; tracked machines select a deformable
reinforced belt model. Neither setting implies FEM for the entire chassis.
Different machines may use different modes in the same scene. The simplified
mode remains the default and is not removed when FEM becomes available.
An asset without a validated FEM implementation shows that option unavailable;
visual tyre deformation alone must never be labelled FEM physics.

Keep both implementations permanently. Gearbox will expose a per-machine menu:

- **Simple / fast**: current torque-driven ideal continuous belt; default.
- **Deformable rubber / experimental**: volumetric reinforced belt and physical
  sprocket/lug contact; higher cost. Unavailable until the complete drive path
  passes its contact and vehicle gates, not a silent fallback to simple mode.

Simulation fidelity is separate from visual quality. Both modes use the same
machine/controller interface and authored machine identity. Mode changes must
occur paused and rebuild the affected physics safely; preserve chassis pose
but explicitly reset incompatible belt deformation/tension state. Only one
mode owns traction at a time. Gearbox owns menu/configuration; Molla owns both
physical implementations. Runtime mode switching/menu implementation is pending.

## Verified starting point (2026-09-21)

- `molla-solvers/src/vbd.rs` and GPU VBD kernels already solve tetrahedral
  stable Neo-Hookean elasticity. Some module introductions describe older
  versions and must not be mistaken for the present implementation.
- Existing VBD soft-rigid shape contact is not the complete dynamic articulated
  coupling required here. `soft_rigid_dynamic_oracle.rs` tests a standalone
  impulse calculation, not production sprocket/belt/chassis integration.
- The tyre-specific FEM model includes hub/frame/contact assumptions; reuse
  generic material/solver machinery instead of stretching a tyre into a track.
- Existing Gearbox CEOL baseline is committed as aeade1d; its imported-machine
  regression covers drive/reverse, braking, both pivot signs, measured tread
  transforms, airborne momentum and controller removal. It remains ideal-belt
  physics, not deformable rubber.

## Implementation sequence

1. **Material bench (initial gates pass):** periodic watertight tetrahedral rubber
   loop, density-based nodal masses, positive volumes, stress-free rigid-motion
   invariance, deformation/recovery and CPU/GPU numerical comparison. No motor
   claim at this stage. Initial fixture: 288 nodes, 768 tetrahedra, density
   1100 kg/m³, E=2 MPa, nu=0.45, dt=1/480 s, 16 CPU iterations. Broader
   near-incompressibility, refinement and timestep convergence remain open.
2. **Reinforcement (initial axial gates pass):** embedded longitudinal tensile
   cords share the rubber midpoint-layer nodes. EA/L axial elements carry
   tension only; width quadrature preserves total EA as the mesh refines.
   Reference line density adds lumped nodal mass, and optional prestrain
   shortens cord rest length. Material shear/bending remains volumetric rubber.
Full mesh/timestep convergence, hysteresis and a physical tensioner remain
   open; cord bending, slip and delamination are not implemented.
3. **Contact bench:** one powered rigid sprocket, unpowered rollers and a
   supported belt. Resolve actual tooth/lug contact and two-way linear/angular
   reaction with the rigid solver. Validate energy and momentum; no rotation
   ratios or follower motors in the deformable path.
4. **Ground and chassis:** couple the rig to a free carrier and ground, then
   two tracks. Test air, freewheel, stall, reverse, frictionless ground, pivot,
   obstacle crossing, tension loss and belt retention.
5. **Native delivery:** GPU-resident state/contact scheduling, measured mesh
   deformation in Gearbox, USD material/topology metadata, matched Blender
   asset and labelled controls. Do not switch CEOL's default until it passes.

## Model limits and evidence

### GPU spherical closures (2026-09-22)

`SoftRigidBallJoint` supplies two body-local anchors to persistent GPU rows.
The sampled FEM constructor accepts these closures without changing existing
scene literals. They participate in the same contact sweeps through signed
XYZ impulses and the articulated mass response; no body pose is snapped and
no per-node runtime readback is added. Separation recovery uses the contact
recovery fraction and a speed cap per world axis. Exactly zero response axes
with a nonzero residual fail explicitly rather than silently discarding it.

`fem_ball_joint_gpu` checks a dense impulse oracle, rotated frames with nonzero
COM offsets, linear/angular momentum, soft contact coupling, planar redundancy,
invalid inputs and an incompatible locked direction. A driven floating
four-bar mechanism over128 substeps closes to about0.081 micrometres versus
6.14 mm without its closure; final momentum is about0.000100 kg m/s.
This includes repeated primitive gather before the persistent loop tail.

The integration fixture exposed inconsistent startup body transforms copied
from `Model::state` before FK. The FEM constructor now evaluates rigid FK and
body velocities from generalized state before its initial upload. Otherwise
the first closure used inconsistent poses and articulated mass response.

Gearbox505115b binds both authored CEOL hitch closures into this GPU path.
Its isolated mechanism test loads the actual22-body/29-DOF USD articulation,
applies100 Nm at the hitch lift, and accepts128 substeps at1/4096 s without
CPU body-pose updates. Open-closure gaps are8.347 mm; closed gaps are0.075 and
0.127 micrometres. All9 native GPU tests, actual asset binding,6 CPU tests and
the all-target check pass. This uses a small separate tetrahedron for the FEM
island, not the actual rubber tracks or full-machine runtime ownership.

These are solver and mechanism gates, not actual CEOL delivery. Actual track
geometry, exclusive backend ownership, pose/TF updates, menu selection, pivot
retention and the remaining full-vehicle gates are still pending.

### Positive-pivot retention failure (2026-09-22)

The full two-track positive pivot completed all five drive intervals but failed
the unchanged 20 mm belt-envelope gate: side 0 reached 30.782 mm. Its circulation
mean was 183.042 mm (range 181.022–189.607 mm), and lifetime minJ was 0.55167913.
The later floor/momentum/yaw assertions were not executed after this failure.
Runtime was 1631.21 seconds for 0.7 simulated seconds; this is not realtime.

Postprocessing the captured nodes in the chassis frame separates side 0's
30.773 mm in-plane distortion from only 1.578 mm maximum axial drift. Side 1
has 24.142 mm in-plane distortion and 4.347 mm axial drift. The primary problem
is belt-shape distortion during loaded turning, not lateral derailment.
These chord-based diagnostics locate the problem; they do not replace the
analytic acceptance gate. Tensioning and numerical convergence remain open;
no retention threshold or material parameter has been changed to hide it.

The full vehicle runner accepts `MOLLA_FEM_ELASTIC_ITERATIONS=128..2048` for
stronger-solve convergence comparisons; the default remains128, with unchanged
timestep, contact sweeps and acceptance gates. Captures now include solver
settings, track grid layout and exact reinforcement rest lengths/compliances.
This is investigation tooling, not evidence that a higher sweep count fixes
the pivot. The baseline/configuration CPU gates pass; full convergence runs
remain pending.

### Elastic convergence comparison (2026-09-22)

The controlled 256-sweep positive-pivot run also fails the unchanged 20 mm
retention gate: side 0 reaches 30.786309 mm, less than 0.004 mm different from
the 128-sweep result. Mean circulation is 183.041457 mm and lifetime minJ is
0.55169. Material, timestep, motor settings and contact sweeps were unchanged.
The later floor/momentum/yaw assertions again did not execute. Runtime was
1829.96 seconds for 0.7 simulated seconds. Doubling elastic iterations does
not resolve this failure; its physical/numerical cause is not yet established.

The authored CEOL is not the test rig: Gearbox's asset-binding regression finds
22 rigid bodies, 21 tree joints and 2 closed-loop hitch constraints, plus two
powered sprockets and ten passive wheels. The test fixture has 13 bodies.
Native integration must preserve the authored constraints rather than substitute
the test articulation or silently discard the closed loops.

### Renderer-paced submissions (2026-09-22)

`try_step_substep` submits at most one configured substep and refuses new work
until the previous queue-completion callback has fired. `poll_completion` uses
a non-waiting device poll; neither API reads simulated state back to the host.
An accepted submission is not completed simulation time. Completion does not
prove device health or valid physics; the host still owns those checks.
The blocking `step` API retains its 32-substep fences and can alternate with
the renderer-paced API without dropping or repeating integration steps.

Three GPU submission tests pass: existing window equivalence, 65 asynchronous
substeps versus the blocking result with stale CPU state, and mixed API calls
across a window boundary. These tests validate scheduling, not interactive
latency or realtime performance. Native Gearbox scheduling remains pending.

### Externally managed device and queue (2026-09-22)

`request_fem_device_limits` raises the caller's requested limits to the kernel
minimums before shared-device creation, preserving larger renderer limits.
Unsupported adapters leave the request unchanged and return an error. Boundary
tests cover every minimum and preservation of unrelated limits; a real GPU test
constructs a device using the negotiated limits, advances65 asynchronous
substeps, and renders the resident moving FEM positions directly.

FEM now accepts caller-owned matching device/queue handles, with additive
VBD, Featherstone, mass-matrix and Jacobian entry points. Existing Engine
constructors delegate; no Engine/device-health ownership changes are needed.
The caller owns device-loss handling and submission scheduling. Construction
does not replace callbacks, and dropping physics does not destroy the device.
Minimum kernel limits are checked before resource allocation.

The shared-device GPU test binds live FEM positions directly in a vertex
shader. After 65 substeps the rendered triangle moves three pixels, retaining
1326 covered pixels, while host particle shadows remain stale. Engine-backed,
external-device and sampled external-device states are identical. Rendering
still works after all physics systems are dropped, and the owner's loss
callback fires exactly once on explicit owner destruction. A real default-limit
device is rejected before physics allocation and remains usable afterward.
The submission regression and each static device-limit boundary also pass.
This is offscreen interoperability evidence, not Gearbox/native CEOL delivery.

### Renderer-compatible GPU API (2026-09-22)

Physics crates and the viewer's direct physics handles use wgpu 29, matching
Gearbox/Bevy 0.19. Explicit pipeline layouts use optional layout slots and
zero immediate bytes; headless compute uses the new instance constructor.
The standalone Molla viewer retains Bevy 0.18 and its independent wgpu 27
renderer. No renderer/device sharing or simulator FEM delivery is implied.

The isolated migration passes workspace/all-target checks with GPU enabled,
two cache-construction CPU tests, and eleven GPU cache, submission, barycentric
contact, articulated-dynamics, VBD-tet and two-track smoke gates. The 32-step
vehicle smoke retains lifetime minJ 0.83006966. Native shared-device ownership,
render deformation and full vehicle validation remain required.

### Immutable VBD binding reuse (2026-09-22)

VBD keeps separate 64-entry LRU caches for storage groups and immutable
uniform groups across substeps. Storage keys include layout, binding index,
buffer identity, offset and size; uniform keys include layout and exact bytes.
Eviction does not mutate resources referenced by already encoded GPU work.
Cache hits avoid new GPU groups, uniform buffers and uniform uploads.

Focused GPU tests verify identity, layout/range separation, bounded eviction,
and encoded commands surviving cache destruction. Alternating two tetrahedral
models and ping-pong states with changing timestep, damping, gravity and hard
mode produces identical positions, velocities and plasticity state to freshly
created kernels. This establishes reuse correctness, not a throughput or VRAM
improvement. Long vehicle performance and native integration remain open.
The broader wgpu 27 regression suite passes 25 GPU tests across binding reuse,
submission, coupled contact/motors and cloth/cable/tetrahedral/plasticity paths.

### Two-track vehicle fixture

The single-track engineering fixture now shares its physical assembly with a
two-track test carrier: 1000 kg chassis, 0.82 m track spacing, 13 rigid bodies,
18 DOFs and independent motor DOFs 8/14. The two belts contain 8736 particles
and 26880 tetrahedra. CPU topology and sampled initial-clearance gates pass;
maximum initial penetration is 0.174284 mm per side. All passive rollers are
unpowered revolute joints. These test parameters are not measured CEOL data.

Explicit GPU cases cover forward/reverse, both differential pivot signs,
airborne conservation, frictionless circulation, sampled-surface forward drive
and a powered/passive comparison. Forward vertex-contact acceptance passes;
the other full-vehicle gates remain pending. Straight
cases retain the single-track 40 Nm cap per motor; pivot cases use 300 Nm and
higher velocity gain. No chassis or belt velocity is prescribed. Captures
include both belts, rigid poses and intermediate drive-interval snapshots.

The wgpu 27 forward run at commit 9542924 settles for 0.10 s and drives for
0.60 s: chassis travel 76.985 mm forward, 1.288 mm lateral, yaw -0.000126573 rad.
Mean belt circulation is 82.405/81.551 mm, with envelope departures
10.330/10.208 mm and lifetime minJ 0.61785346. Unchanged circulation, volume,
ground clearance, tilt and travel gates pass. Runtime is 1693.26 s, not real
time. This predates binding reuse and does not validate sampled full-surface
contact or establish a performance improvement. Offline Blender inspection
of exported measured states confirms both belts remain around their rollers;
the rendering is not a native simulator screenshot.

Native Gearbox currently uses a CPU RigidWorld adapter and the older d5476026
Molla pin. Engine has no externally supplied render-device constructor yet.
Native GPU deformation needs explicit device sharing and exclusive physics
ownership, rather than per-node CPU readback. Current main Engine device-loss
edits are unrelated work and must be preserved when adding that interface.
The local physics GPU API now matches Gearbox's wgpu 29, but the Gearbox
dependency pin still needs updating after native integration is validated.

GPU submission is now fenced every 32 substeps, including across separate
step calls. These are completion waits, not state downloads or CPU physics.
The 65-substep monolithic/split test gives bit-identical GPU states and the
analytic rotor acceleration; a two-track 32-substep smoke gate passes with
lifetime minJ 0.830070. Clean contact/motor/effort/airborne regressions pass.
This bounds queued substeps, not retained driver memory: the long two-track
run still reports about 6.3 GiB, so the fence does not establish a VRAM fix or
throughput improvement. Avoid concurrent long full-track cases while profiling.

### Current full-track boundary (2026-09-22)

The undamped free-carrier frictionless run now passes horizontal momentum:
0.000482 kg m/s after 0.70 s against the unchanged 0.02 limit. Its belt still
fails circulation (about -0.15 mm instead of one 56.45 mm pitch), so the
mechanism is not accepted. Preserving relative Newton increments in the GPU
tetrahedral stress calculation removed most remaining numerical drift; no
momentum projection or acceptance relaxation was used. The earlier 95 mm
grounded travel result used an older damped fixture and is not final acceptance.

The initial 112-segment mesh had clear vertices but intersecting surfaces:
sampled edge/face penetration reached 0.697 mm at an idler and 0.637 mm at a
drive tooth. The fixture now checks vertices, edge midpoints and triangle
centroids before GPU execution. A coarse regression reproduces the missed
intersection. Refinement to 224/448 longitudinal segments reduces maximum
sampled penetration to 0.174/0.044 mm, below the unchanged 0.5 mm gate.
The 224-segment fixture preserves physical lug width and 56-lug pitch and now
passes the full undamped frictionless case: 150.290 mm belt circulation after
0.10 s settling plus 0.60 s drive, range 148.655..153.175 mm, maximum envelope
departure 9.237 mm, drive speed 1.974 rad/s. Horizontal momentum magnitude is
0.001016 kg m/s, ground penetration zero, lifetime minJ 0.617943 and finalJ
0.633438. Carrier horizontal travel is only 0.0232 mm. This run uses vertex
contacts and predates the sampled-contact kernel extension. It took 802.67 s
and does not establish realtime performance. Sampling is not an exhaustive
clearance proof. Default runtime contact remains node-based.

An opt-in GPU surface-sampled composition now adds unique edge midpoints and
triangle centroids without adding particle mass. Contact point and velocity
use barycentric interpolation; soft effective inverse mass sums squared
weights times inverse masses. Impulses distribute through the same weights,
and articulated reaction acts at the interpolated point. Analytic GPU gates
cover one/two/three-node support, sphere/box/cylinder geometry, friction cones,
linear/angular momentum and malformed descriptor rejection. Edge/face cases
also verify that vertex-only gathering misses their contacts.

On the prestressed-ring bench, vertex-only contact leaves 0.587 mm sampled
surface penetration. The new path with 64 contact sweeps reduces it to
0.021/0.009/0.004 mm at substeps 1/2400, 1/4800 and 1/9600 s, below the fixed
0.05 mm gate. Twelve sweeps failed that gate at 1/2400 s (0.102 mm); shared
surface support needs more iterations. This is discrete sampled contact,
not exact triangle/edge closest-point detection or CCD. Full-track sampled
contact acceptance and performance remain unverified.

Direct GPU effort controls allow a genuinely unpowered rotor after settling.
The undamped 112-segment powered/passive comparison passes with 95.905 mm
driven travel versus 11.028 mm passive relaxation: 84.877 mm additional motion,
above the unchanged one-pitch requirement. Its coarse initial surface
intersections still preclude full mechanism acceptance. Full two-track
motion, obstacles, stalls, energy/convergence, performance and native Gearbox
FEM/menu delivery remain incomplete.

`TetVolumeMonitor` records the initial configuration and every accepted GPU
substep, retaining the lifetime minimum determinant rather than only inspecting
frame-end meshes. Invalid/nonpositive elements set sticky status bit 8;
nonfinite determinants set bit 4. A restored final mesh cannot erase an earlier
failure. Unit GPU checks cover transient compression, recovery, inversion,
collapse, NaN/Inf, invalid indices and preservation of other status flags.
The track gate now requires lifetime minJ > 0.5 without per-substep readbacks.
This detects invalid material states; it does not itself prevent inversion.

The 1/9600 s, 0.5%-prestrain, 12 mm tooth fixture passes the full forward
circulation/retention gate with lifetime minJ 0.672318 and 76.85 mm material
travel. Reverse at the same timestep fails material acceptance (lifetime minJ
0.403477 despite 73.54 mm travel). At 1/19200 s reverse passes: lifetime minJ
0.749723, travel 72.16 mm and maximum envelope departure 16.29 mm. This run
took 616.52 s for 0.26 simulated seconds, not real time. Neither direction
proves two-track vehicle traction, long-run robustness or native delivery.

The free-carrier/ground fixture adds actual gravity, a 500 kg free chassis,
unmotorized support rollers and primitive ground contact. It measures belt
circulation relative to the moving chassis, requires chassis displacement and
ground traction impulse, and includes a zero-friction horizontal-momentum
control. In the first 0.24 s powered trial the carrier advances 27.15 mm,
ground penetration is zero, lifetime minJ is 0.598409 and final minJ is
0.903480. The trial fails the unchanged one-pitch circulation gate: material
travel is only 28.66 mm versus 56.45 mm required. The loaded drive window is
extended to 0.60 s with the same torque cap, timestep and acceptance limits.
That longer loaded case passes: carrier travel 95.17 mm, belt circulation 95.97 mm,
departure 12.07 mm, ground penetration zero, lifetime minJ 0.598409 and final
minJ 0.788392. It took 701.54 s for 0.70 simulated seconds, not real time.
The initial 0.24 s frictionless control
fails its unchanged horizontal momentum gate: |Pxy|=0.02944 kg m/s versus
0.02 allowed. Carrier displacement is only 0.053 mm longitudinally and ground
penetration is zero, but numerical drift is not accepted. Its material
circulation is also insufficient (2.44 mm); lifetime minJ is 0.668554.
Free-vehicle acceptance still requires resolving these failures and separating
motor-driven motion from passive prestress relaxation with an unpowered control.

The full-loop fixture was applying 1/s global velocity damping to the soft
particles only. This is external drag, not internal material damping or a
unity velocity-retention factor, so the frictionless case was not a closed
horizontal momentum system. The fixture now sets damping to zero; the same
gates are being rerun. Earlier results above describe the damped fixture.
Optional GPU diagnostics record separate soft/rigid momentum at input, elastic
prediction, rigid prediction, contact and accepted drift, plus cumulative phase
deltas. The rigid predictor is measured using current Jacobians and generalized
velocities rather than stale body-velocity caches. Synthetic mixed-size GPU
tests verify phase totals and repeated-step deltas with poisoned body caches.

The undamped 0.70 s control still showed |Pxy|=0.023893 kg m/s, mainly from
FEM. An isolated unforced, rotated and deformed reinforced loop reproduced
0.000282 kg m/s drift in 128 substeps, failing its 0.0001 limit. GPU tetrahedron
evaluation now combines relative Newton increments with base edge displacement
instead of adding tiny increments to large world/rest offsets first. The same
gate passes at the origin and after translation (0.0000514/0.0000290 kg m/s).
Clean cord, plasticity, multimaterial, contact and airborne regressions pass.
At 0.22 s the full control's cumulative FEM Py error drops from 0.00410 to
0.0000902 kg m/s; its final conservation/circulation gates are still running.

GPU articulated dynamics now implements dense Cholesky joint blocks when
`rigorous_d_solve` is enabled; the FEM composition enables it. Fixed-child
inertias/wrenches propagate to their parent. Free/Distance/D6 subspaces use
the translated joint anchor and their translation bias term, with acceleration
bias included before solving joint accelerations. A carrier/offset-COM/fixed-
payload/driven-rotor GPU gate agrees with the CPU oracle within 1.1e-7 per step;
the moving 0.105 s case differs by 1.7e-6 in velocity and 7e-8 in coordinates.
This replaces a measured 0.0172 velocity error for the fixed-payload case.

Contact correction now precedes the accepted position drift on both the GPU
runtime and CPU oracle. Elastic/rigid predictors provide velocities; contact
geometry and articulated responses use the beginning-of-step configuration;
corrected velocities integrate particles and joint coordinates exactly once.
GPU FK and final-pose Jacobians refresh body velocities without host readback.
This is first-order split integration, not a fully implicit coupled contact
energy solve or an intersection-free guarantee.

A prestressed ring exposed the former velocity-only-after-drift defect: it sank
2.423 mm into its cylindrical support, and raising the recovery-speed cap still
left 1.969 mm penetration. The new ring gate measures 1.49e-8 m penetration at
1/2400, 1/4800 and 1/9600 s, with positive material volumes. Drift parity covers
all eight joint types, including locked D6 axes and active force inputs that
must not be applied again. Existing articulated impulse and free-carrier
momentum gates retain their 0.001 tolerances. The updated anchored CPU tooth
speeds are 0.275111/0.274986 rad/s; the GPU negative case is 0.275179 rad/s.

Earlier coarse installed full-loop runs failed. With 2% prestrain, material
circulates 67.1 mm (pitch 56.5 mm) and envelope departure is 10.7 mm, but a
tetrahedron inverts (minJ -0.0169). At 0.5% prestrain the belt circulates
78.8 mm with 17.3 mm departure, but lug elements compress excessively
(minJ 0.2251; settled minJ already 0.4743). No volume or circulation acceptance
threshold was weakened. Contact/elastic coupling and geometric resolution
remain required work before robust full-track or native-simulator acceptance.

Installed track geometry now has an analytic common-tangent/arc envelope for
convex ordered pulleys, periodic arc-length sampling and diagnostic nearest
projection. `installed_positions` initializes the deformed state without
changing the circular elastic reference mesh; it is not a runtime path-following
constraint. Shared-face tetrahedral inner lugs add their actual volume and mass.
The 56-lug, 112-cell fixture is one connected watertight volume (2016 nodes,
6048 tets); installation at multiple phases retains positive element volumes.
The complete powered-loop acceptance gate must demonstrate material circulation
and retention, not just sprocket rotation or vertex movement. Its first run
failed: after 0.24 s of motor drive, mean material travel was only 34.2 mm versus
56.5 mm pitch, with a 60.4 mm envelope departure. The new gate remains required;
idler groove support, tension and engagement are under investigation.

GPU VBD reuses per-colour and dual uniforms across iterations. On the unchanged
reinforced-tooth gate this reduced observed test duration from 62.60 s to 24.66 s
with identical numerical output. Hard AVBD and plasticity regressions pass.
This eliminates redundant host setup; it does not make the full FEM track real-time.

Two-way contact now has a reusable CPU oracle in `soft_rigid_contact` and a
substep composition in `fem_rigid`. It uses articulated mass responses, not
independent-body inertia, for equal/opposite contact impulses and Coulomb
friction. A powered frictionless tooth transfers torque into the tetrahedral
loop in both directions: at 0.1 s the loaded rotor reaches about 0.259 rad/s
versus 1.775 rad/s unloaded, rubber angular momentum is about 0.448 kg m²/s,
and peak penetration is below 0.165 mm. This is one tooth pressing rubber,
not yet a circulating sprocket/lug mechanism or reinforced track.

GPU primitive contact generation reads live device FEM positions and rigid
transforms directly. The first real-GPU test checks 768 candidates and 39
active contacts against the CPU oracle, including ball/box/cylinder surfaces.
Only test acceptance downloads the resulting state. GPU articulated contact
response and resident substep scheduling now have a one-articulation prototype:
`FemRigidGpuSystem` dispatches GPU FEM, rigid integration, CRBA/Jacobian,
primitive gather, mass factorization, contact response and Coulomb iterations.
Its powered-tooth test runs 0.1 s without substep state readbacks in both
directions. At 128 elastic iterations, loaded speed is about 0.259 rad/s,
unloaded 1.779 rad/s, and belt angular momentum about 0.448 kg m²/s.
The CPU oracle is not the delivery path.
GPU response parity covers floating 25-DOF trees, reversed frames and locks;
contact parity covers friction 0/0.4/1, locked axes and rejected invalid inputs.
Contact sweeps remain serialized on device, parallelism is currently in FEM
and response assembly, and no real-time performance claim is made. GPU-only
stable compaction now builds an ascending active-candidate index list using
parallel 64-lane counts, block offsets and scatter. Contact sweeps no longer
scan inactive node/shape pairs on every iteration; the numerical ordering and
contact law are unchanged. Sparse 129/131/65536-candidate checks match dense
results bit-for-bit, including workgroup boundaries, reset-to-empty and zero
rows. Clean motor/contact/airborne regressions pass; the full forward track
retains identical measured circulation and volume results. Timings were taken
with concurrent GPU jobs and do not establish a performance improvement.
Primitive gathering is presently
dense node/shape pairing, not a production broad phase or continuous collision
detection implementation.

The resident composition currently accepts one unreversed dynamic articulation
up to 32 joints and 64 DOFs and explicit primitive contact shapes. It exposes sticky GPU
failure status; callers must check it before accepting results. Existing GPU
Featherstone rejects reversed joints and acceleration motors rather than
silently delegating them to CPU. Motor-only force caps execute on GPU before
feedforward, friction, cable and muscle forces are added; total joint effort
limits still apply to the resulting sum. NaN and negative motor caps are
rejected. Zero and positive-infinite caps retain their zero-drive and unlimited
semantics. Dense joint blocks execute on GPU.

`featherstone_motor_caps_gpu` checks all four driven modes against CPU and
analytic unit-inertia results, including both signs, zero/infinite motor caps,
separate feedforward and total-effort clipping. A coupled FEM island regression
also verifies capped rotor acceleration and zero GPU status. The extra control
binding raises Featherstone to35 storage buffers, below the FEM composition's
existing42-buffer minimum; shared-device minimum-limit rendering tests pass.
The driven bench uses the physical joint effort limit for its 5 Nm cap.
The same tooth bench now includes a free 750 kg carrier and its 2 kg rotor.
Both directions pass airborne conservation without prescribed motion or a
momentum projection: total linear momentum stays below 0.00045 kg m/s and
angular momentum below 0.000071 kg m²/s after 0.1 s, against unchanged 0.001
limits. Loaded relative rotor speed is about 0.266 rad/s. This coaxial test
does not validate arbitrary multi-DOF GPU acceleration coupling.
Reinforced circulating belt with sprocket/lug/roller engagement, ground
traction and native CEOL delivery are still required.

That gate exposed two f32 VBD issues. Rest-state matrix multiplication and
stress cancellation generated motion in untouched rubber; displacement-form
deformation and cancellation-free stress now give exact zero rest velocity.
World-position rounding also erased substep motion, so GPU VBD now iterates
on displacement from the input position and recovers velocity from it.
Sub-ULP velocity, rigid rotation, cloth, tetrahedral and plasticity GPU gates
pass; CPU/GPU strained-loop position difference is about 2.52 micrometres.
An isolated impulse convergence gate separates precision from iteration error:
16 sweeps lose significant momentum on both CPU and GPU, while 64 sweeps
reduce GPU drift below 0.0002 kg m/s over 48 steps. Coupled FEM defaults to
128 sweeps; automatic residual stopping and performance work remain open.

The first rigid-motion invariance test exposed a pre-existing CPU/GPU VBD
error: `cofactor3` returned the adjugate instead of the determinant gradient.
A stress-free rotated loop moved 0.124 m in one 1/480 s step. Both paths now
use cross products of matrix columns for the cofactor. CPU rotation invariance,
free-loop recovery and 78 existing solver library tests pass. GPU deformation
parity over 48 rotated-deformation steps agrees within 5.82 micrometres. The old cube-on-sphere test
had no floor: with objective rotation restored, the cube tips off the sphere
and falls to y=-10.05 m rather than remaining artificially balanced. The
settling fixture now includes a floor and retains sphere nonpenetration checks.
This is a material-solver correction, not belt contact work.

Rubber constants and reinforcement are initially estimates, not calibrated
CEOL measurements. Near-incompressibility, thin-element conditioning, contact
tunnelling, cord/rubber stiffness contrast and substep cost are explicit risks.
A coarse tetra mesh is a continuum approximation, not molecular rubber. No
claim of tear, heat, wear, fatigue or experimentally validated traction is made.

`RubberBeltMesh::append_reinforced_to` defaults to total axial rigidity
200 kN and additional line density 0.1 kg/m, with zero prestrain. The initial
reinforced bench uses 96 tension elements sharing the 288 tetrahedral nodes.
Its 2% radial expansion produces about 15 times the initial radial restoring
impulse of bare rubber. After 48 GPU steps at 1/4800 s, CPU/GPU positions agree
within 1.10 micrometres and momentum stays below 0.00034 kg m/s. Total energy
falls from 188.71 J to 110.79 J under implicit integration; this is numerical
dissipation, not a calibrated rubber hysteresis law. Pure tensile element tests
match implicit-Euler axial response, including stiffness above the bilateral
spring cap; slack compression produces no force. GPU VBD and XPBD both support
the tension flag; the barrier solver explicitly rejects it. Analytic closed-loop
mass, strain energy and prestrain gates span longitudinal/width refinement.
The reinforced loop also passes the floating powered-tooth test in both
directions: loaded relative rotor speed about 0.294 rad/s versus 1.780 unloaded,
rubber angular momentum about 0.442 kg m²/s, and displacement about 8.18 mm.
Composite mass is included in conservation checks; total airborne linear
momentum stays below 0.00028 kg m/s and angular momentum below 0.000057 kg m²/s
after 0.1 s. Unloaded reinforced controls have zero measured velocity.
This is not full circulating sprocket engagement or vehicle acceptance.
No imposed motion or hard strain cap is used for the cords.

GPU-first architecture with the CPU as correctness oracle remains required.
Keep physical laws in Molla; Gearbox owns rendering and machine binding.
The user reaffirmed that every GPU-suitable stage must execute on GPU, including
contact generation and two-way coupling, not only the elastic material update.
CPU coupling benches are correctness oracles and do not qualify as native FEM
delivery. Keep deformable positions, velocities and iterative contact state
device-resident; no per-node host round trips inside substeps.

### Contact response launch regression

Contact preparation uses disjoint batches of eight 64-thread workgroups, with
an immutable `base_row` uniform per batch. Every candidate row is assembled;
mass factorization, stable contact ordering, Coulomb sweeps and recovery are
unchanged. Batching introduces no CPU physics or simulated-state readback.

On RTX 4080 / NVIDIA 595.84 / wgpu 29.0.4, constructing a Featherstone pipeline
was sufficient to make a subsequent 385683-row preparation launch time out.
No Featherstone dispatch was necessary. The same captured CEOL inputs completed
on a cold device, and a 65-row preparation completed after construction. Full
response-buffer allocation with a single workgroup also completed. Moving the
private triangular-solve scratch into storage and reducing workgroup width did
not resolve the full launch. Eight-workgroup batches completed the full captured
input; 64-workgroup batches still timed out. The driver-internal cause is not
established, and this is not evidence of a physical material or joint-stop error.

`make test-gpu P=molla-solvers T=fem_contact_prepare_gpu` constructs Featherstone
before checking all sparse and dense native-scale response rows against an
independent double-precision oracle, including ground and spherical-loop rows.
`MOLLA_CONTACT_FIXTURE=<directory> make test-gpu P=molla-solvers T=fem_contact_replay_gpu`
replays the ten `.bin` inputs exported by Gearbox's native contact preparation
test. Capture and response readback occur only in tests. A completed preparation
kernel is not full vehicle traction, retention, energy or real-time acceptance.

### Fixed-world contact response

Contacts against the immovable world have no generalized rigid Jacobian.
Their effective inverse mass is the barycentric particle inverse mass in all
three orthonormal contact directions. Preparation clears any prior rigid row
response and bypasses the triangular solves; contact sweeps update only the
participating particle velocities. Dynamic-body and spherical-loop rows retain
their coupled articulated response. Generalized velocities are checked for
nonfinite values once during mass factorization, including world-only scenes.

`make test-gpu T=fem_static_contact_gpu` checks sliding, sticking, normal
recovery and a finite positive gap against an independent barycentric Coulomb
oracle at 1, 29, 32, 33 and 64 DOFs. It reuses a formerly dynamic contact row, requires
bitwise-unchanged rigid velocities and rejects nonfinite generalized velocity
without changing particle velocities. The existing mixed CPU/GPU, native-scale
preparation and spherical-loop gates remain required.

The full-scale CEOL loaded/passive trajectory captures use unchanged 512-step
startup windows. Both reproduce their previous uninstrumented endpoints exactly.
Their COM travel differs by less than 0.1 micrometre through 13.333 ms, then by
88.123 micrometres at 26.667 ms. Motor-added forward contact impulse reaches
14.678 N s, but the 100-micrometre causal displacement gate still fails. Both
cases are still settling vertically. This is not settled driving acceptance,
and the fixed-world optimization does not change those physical requirements.

### GPU scalar feedback stabilization

`FeatherstoneSolver::with_stable_motor_feedback(true)` supports scalar force
drives on the GPU with rigorous dynamics. It uses cached articulated inverse
mass responses to compute each drive's effective inertia, stabilizes feedback
with the same per-DOF denominator as the CPU implementation, then applies the
combined torque correction to every affected articulation coordinate before
joint-stop reactions. This is not a fully coupled implicit multi-motor solve.
Motor caps, feedforward and total effort limits retain their separate roles.
Each unit inertia probe is explicitly cleared before the next drive. Invalid
or nonfinite feedback solves latch the shared scalar-solve status bit32.

`FemRigidGpuSystem::with_stable_motor_feedback(true)` opts its rigid predictor
into the same method. The contact-corrected drift does not apply motor torque
again. Both APIs retain default-off behavior; state remains GPU-authoritative.

`make test-gpu T=featherstone_feedback_gpu` checks96 cap/mode/stop combinations,
configuration/fault rejection and512 steps of two sibling rotors against both
CPU feedback and an independent constant-mass analytical recurrence, including
root reaction and angular momentum. `make test-gpu T=fem_feedback_gpu` checks
the predictor/contact/drift schedule against a scalar analytical recurrence.
These are solver gates, not full CEOL settling or traction acceptance. Native
CEOL equilibrium, causal ground propulsion, sustained maneuvers and simulator
delivery remain required before claiming tracked FEM completion.

### Contact GPU pass profiling

`SoftRigidSolveProfile` records14 timestamp ticks for the seven full contact
passes on devices explicitly created with `TIMESTAMP_QUERY`. Ordinary solver
encoding remains uninstrumented. `encode_profiled` resolves ticks into a GPU
buffer; it does not transfer simulation state to the host.

`MOLLA_CONTACT_FIXTURE=<directory> make test-gpu T=fem_contact_replay_gpu`
retains the warmed preparation regression and compares a complete12-sweep
captured solve with its timestamped replay. Particle, generalized and body
velocities plus contact rows must be bitwise identical. The2026-09-23 RTX4080
capture with29DOFs,29568particles and385683candidate rows measured291.795ms
in `solve` of294.643ms total GPU pass time; preparation was1.817ms. These are
fixture measurements, not real-time vehicle acceptance or a speedup claim.
The serial contact sweep accounts for approximately99% of this contact profile.

### Cooperative contact sweep

`SoftRigidSolveKernel::new` selects64-lane loading of projected
contact Jacobians and inverse-mass responses, with generalized velocities cached
in workgroup memory. Lane0 retains the original contact order, normal/friction
updates and scalar accumulation order. Particle updates remain on that lane.
Faults stop subsequent row corrections without skipping required workgroup
barriers; the partial generalized state is flushed and the shared status prevents
body-velocity recovery. `new_serial` retains the single-invocation reference;
`new_cooperative` explicitly selects the same sweep as the default constructor.

The captured CEOL replay compares all particle/generalized/body velocities and
contact rows bitwise between ordinary serial, profiled serial and cooperative
solves. The first RTX4080 comparison measured294.646ms versus198.023ms total
contact GPU time, with the sweep reduced from291.792ms to195.010ms. This is a
single captured-state benchmark, not full-vehicle performance acceptance.

`fem_static_contact_gpu` runs both paths against independent world-contact and
dynamic barycentric impulse oracles through64DOFs, including row reuse and
nonfinite input rejection. `fem_contact_solve_gpu` runs both paths against the
articulated CPU oracle with friction, locked coordinates, sparse/empty rows and
invalid inputs. `fem_ball_joint_gpu` compares every output word across paths
before checking its dense impulse and momentum oracle, mixed contact/ball rows,
singular responses, partial-row failures and finite-input overflow. The replay
lane also checks reduced workgroup limits and invalid dimensions.

The default is exercised through full FEM substeps by the existing four-bar
closure, motor-to-rubber, reinforced airborne momentum and stabilized feedback
tests. Submission-window, shared-renderer-device and bounded-contact failure
tests also use the default path. Serial and cooperative references remain
explicit in the independent contact and spherical-loop oracle comparisons.

Full native settling and powered traction remain unaccepted. Native trajectory
comparison is still required when updating Gearbox to the cooperative default.

Primary references:
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-trusses.htm
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-embeddedelement.htm
- https://graphics.cs.utah.edu/research/projects/vbd/
- https://graphics.cs.utah.edu/research/projects/avbd/
- https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf
- https://mmacklin.com/smallsteps.pdf
