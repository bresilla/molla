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

### GPU previous-gradient cache (2026-09-24)

VBD computes the previous-position deformation-gradient high/low parts once per
tet on the GPU, retaining them across valid material continuations. A new step,
begin-material or force-only evaluation refreshes the cache. Current gradients
and plastic state remain uncached. Six vec4 columns per tet occupy a checked tail
of existing displacement scratch; the43storage-binding contract is unchanged.
The CEOL regression adds9,805,824bytes of cache, without host mirrors/readback.

Two102144-tet fixed-state replays retain bitwise-identical displacement and force
buffers at0/1/8/32/128/512 iterations versus the uncached shader. Warm material-only
submission/readback timings improve1.49x and1.47x, with1.50/2.76ms preparation.
These measurements are not full coupled-simulation acceleration or acceptance.
Lifecycle tests compare reused/fresh kernels under position, rest-basis, history,
timestep and topology changes, including zero tets and force-only nonmutation.
Validation passes29 GPU tests (material9, cache lifecycle1, precision2, plasticity3,
kinematic hardening3 and coupled contact11), the scratch-limit unit test and
all-target GPU checks. Independent f64 replay gates pass without tolerance
changes; the previously rejected state42 initial iterate remains rejected.
Main-checkout material9 and cache lifecycle1 GPU revalidation also pass.
The cache does not establish sustained settling, torque-driven travel, energy
balance, refinement or practical simulator performance; those remain open.

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

The native512-step passive CEOL trajectory with the cooperative default matches
all eight prior stable-feedback snapshots: particle positions/velocities,
generalized velocities and momentum packets match float32 bits; normalized
poses and COM travel match exactly. Retention15.902mm, minJ0.653345 and negligible
floor penetration pass the unchanged short-run gates. This covers26.667ms of
simulated time, not settled or powered driving. Full native settling, sustained
traction, simulator ownership/menu integration and wheel FEM remain unfinished.

### Objective rubber viscosity

`vbd::viscosity` and `tet_viscosity.wgsl` evaluate an isothermal finite-step
Green-strain dissipation potential in CPU and GPU VBD. `MaterialNeoHookean`
stores per-element `TetViscosity { shear, bulk }` in Pa s; constructors default
both to zero. Builder and upload validation reject negative, nonfinite,
overflowing and subnormal GPU coefficients. GPU storage uses one vec2 binding;
FEM device negotiation now requires 43 storage buffers. Existing CEOL material
settings remain unchanged: no native damping or settled-operation claim yet.

For previous deformation F0 and increment A, define
`dE=(F0^T*A+A^T*F0+A^T*A)/2` and
`D=V/h*(eta*||dev(dE)||^2+kappa/2*tr(dE)^2)` with nonnegative viscosity
coefficients in Pa s. CPU finite differences check the force and exact local
Hessian; the solver candidate uses its positive-semidefinite Gauss-Newton part,
with the geometric term tested separately. Independent rotations of current
and previous configurations leave the potential unchanged. Internal force and
torque balance, zero viscosity and scalar dilation have independent checks.
GPU force/tangent/potential comparisons cover256 cases, including strain
increments down to1e-10 with a relative force check. Incremental evaluation
avoids cancellation between near-equal right-Cauchy-Green tensors.

The frame-indifference construction follows the squared C-tensor difference
in [Badal et al., section2.1, equations2.8–2.9](https://arxiv.org/html/2604.27923v1).
The local isotropic blocks and Gauss-Newton approximation are independently
derived here; the paper's full thermoviscoelastic PDE/heat model is not included.
Runtime tests exercise free tetrahedral timesteps with zero, shear-only,
bulk-only and combined viscosity. CPU/GPU position and velocity tolerances are
20 micrometres and 0.5 mm/s. Finite rigid rotation and free translation do not
acquire material drag. A deformed free tet with E=2000 Pa and test viscosities
30/80 Pa s loses mechanical energy monotonically; after 0.2 s its CPU energy is
0.0639495 J versus 1.2111966 J without viscosity. These are test coefficients,
not a CEOL material calibration.

CPU and GPU tests halve the timestep from 2 ms through 0.25 ms and double VBD
iterations from 32 to 64. GPU endpoint errors against the finest timestep are
1.02176, 0.443857 and 0.149035 mm; the finest CPU/GPU positions agree within
20 micrometres. Invalid CPU blocks error; GPU viscous arithmetic failure marks
the state nonfinite and the existing volume monitor latches status bit 4,
minimum J=0 and the offending tet index. Tests read back nodes; runtime adds
no node readback or CPU physics fallback.

The zero-material control exposed dummy origin-shape contact when the authored
shape count was zero. VBD now disables shape contact in that case; nonempty
shape contact and FEM's explicitly disabled VBD contact are unchanged.
Clean verification passes 26 simulation-model tests, 86 solver-library tests,
four CPU timestep tests and 21 GPU tests across viscosity, shared-device,
coupled motor/contact, spherical-loop, tetrahedral and plasticity lanes. The
GPU-feature solver all-target check also passes. Constitutive calibration,
refined native load settling, sustained torque-to-ground traction and production
performance remain required. Small free-tet convergence is not full
tracked-machine acceptance.

### Opt-in material/contact iteration

`FemRigidGpuSystem::configure_material_contact` enables bounded GPU re-solves
with original step positions and strain history held fixed. Contact impulses
accumulate into a separate inertial-input velocity buffer. Prepared contact
responses and multipliers persist between re-solves; rigid generalized velocity
receives the opposite impulse. Positions drift once at the end. The existing
uncoupled path remains the default and allocates no iteration scratch buffers.
This path currently accepts only elastic tets and zero world-velocity damping.

The final soft-linear, rigid-linear and rigid-angular velocity corrections are
checked on device. A separate projected contact residual checks normal
complementarity, the Coulomb disk and spherical loops. Net velocity correction
alone is insufficient: inconsistent alternating constraints can produce zero
net correction while violating an individual contact. Status bit64 rejects
excess residuals; bit4 rejects nonfinite arithmetic. No runtime node readback
or CPU physics path is introduced. Configuration changes require an idle queue.

The actual GPU integration is checked against a constrained viscous tetrahedron
CPU oracle and a dynamically translating support with an independent reduced-mass
reference. Tests cover insufficient iteration budgets, configuration preservation,
unchanged disabled results, momentum transfer and two timesteps. Kernel tests
include cyclic constraints, sticking/complementarity errors, valid Coulomb sliding
and nonfinite inputs. See `fem_viscous_contact_iteration_gpu.rs`.

The final candidate also receives an independent assembled-force residual check:
each free vertex evaluates `h * |M^-1 (M/h^2 (y-delta) + f_internal)|` using
`delta = h * v_contact` and the original position/strain history. The inertial
target includes the accumulated contact impulse. The same spring, cable and tet
force assembly supplies the solve and the residual; this is not the norm of a
Newton correction. GPU reduction rejects a maximum above the linear tolerance
with bit64, or nonfinite force arithmetic with bit4. The separate scalar buffer
is exposed by `material_force_diagnostics`; the three existing correction
diagnostics retain their layout. No particle state or history is changed by the
residual pass. Pinned vertices are excluded from the free-DOF force equation.

`fem_material_force_gpu.rs` checks analytic axial elastic/viscous force balance,
previous deformation, timestep changes, bitwise state preservation, nonfinite
faults, and reduction tails. A runtime regression has exactly zero contact
corrections but a0.006145m/s internal residual at one inner iteration: it now
rejects, whereas256 inner iterations pass the same1e-4m/s tolerance.

This is not native CEOL acceptance. The earlier quarter/eighth timestep trials
finish13.333ms without volume faults but disagree in local deformation and do
not settle. One full CEOL coupled substep passed the previous contact/volume
guards, taking104.80s including setup for52.083us simulated. The stronger native
trial (cbfa2b3, same256outer/256inner budget) rejects with status64:
force residual0.000771915m/s exceeds1e-4m/s by7.72x although the largest contact
correction is1.102982e-5m/s. All29568 candidate node velocities and all three
coupling corrections match the previous candidate bitwise; independently
reconstructed candidate minimumJ remains0.669218 across102144tets. No timestep
is accepted by the new gate. Total test time112.62s includes setup and rejection
capture. Both speed and full-candidate force convergence remain unresolved.
Native multi-step/moving-contact validation, settling and traction, refinement,
energy accounting and production performance remain open before simulator exposure.

### Retained material iterates

After configuring material contact, `configure_material_contact_reuse(Some(n))`
selects retained VBD iterates with n elastic sweeps per contact correction.
The first solve keeps the configured full elastic budget. Subsequent solves
update the inertial velocity target without resetting displacement or the
original strain history. None retains the full-restart reference. Configuration
is idle-only and does not change the MaterialContactConfig struct. New material
contact configuration resets reuse to None; defaults remain unchanged.

The kernel validates bind-group identity and GPU timestep within a sequence.
An ordinary solve, force residual evaluation or stiffness update invalidates it.
The low-level caller must keep original positions, model data and history fixed;
only the inertial velocity input changes. All iteration work stays on device,
and final contact, internal-force, nonfinite and volume gates remain mandatory.

GPU tests verify17+111 retained sweeps match128 uninterrupted sweeps bitwise,
changed inertial targets, invalid lifecycle/buffer/timestep rejection, and
static/dynamic-support CPU references with1 or8 retained sweeps. A three-step
test using CEOL's ground friction0.85 verifies cold/retained switching, state
agreement, decreasing mechanical energy and the original residual/volume gates.

This is bounded iteration, not a universal convergence guarantee. The artificial
friction1000 sticking fixture passes the first substep but can fail later after
tiny state differences; more iterations are not monotonically better. It remains
an explicit rejection regression (force0.0014995 >1e-4 with256outer/128inner),
not a claimed solved trajectory.

Native CEOL at09f77f6 with8retained sweeps and256outer iterations takes89.05s
including setup, versus112.62s for the full-restart test. It still rejects:
force0.000740895m/s, contact correction1.010057e-5m/s. At512outer iterations,
the test takes165.15s and rejects force0.000264319m/s despite contact correction
4.96953e-7m/s. All force limits remain1e-4m/s. Independent capture comparison
confirms identical materials, masses and initial geometry; candidate minimumJ
remains0.669218, with maximum position difference5.96e-8m and velocity
difference3.37e-5m/s from the full-restart candidate. These are rejected states.
Force-component/precision analysis and GPU timing are needed before more costly
iteration increases; neither a roundoff floor nor a converged native step is
established. Sustained tracked motion and simulator activation remain unaccepted.

Native force-sample reconstruction at c63932f found a precision problem in
the per-node displacement-gradient evaluation. The captured current/rest
offset reaches1.018m. Against float64 forces on the same projected inputs,
float32 displacement-first geometry contributes up to0.002535m/s force error;
edge-first geometry reduces this estimate to9.16e-5m/s. The GPU-worst node
still fails the independent residual test, so under-convergence is also real.

GPU VBD elastic/viscous forces, plasticity gradients and volume diagnostics
now difference current/rest edges before assembling the displacement gradient.
A translated thin-tet regression fails before this change (GPU0 versus
oracle0.143237m/s), then matches within3.54e-5m/s on projected coordinates.
This does not eliminate coordinate-upload quantization or guarantee native
force convergence. The artificial friction1000 case now passes two steps but
rejects its third retained step (force0.119683m/s, correction0.002782m/s);
the regression checks that rejection rather than accepting the trajectory.

Native validation at c2a419f (256outer,8retained) still rejects after84.73s:
GPU force0.000739600m/s and independent float64 force0.000748168m/s identify
the same worst node12972. Maximum nodewise scalar disagreement falls from
0.002543063 to7.70176e-5m/s (33x smaller), but that remaining discrepancy is
still significant against1e-4. GPU/oracle counts above tolerance are282/294,
versus266/19994 before the fix. This confirms improved precision and genuine
remaining under-convergence, not an accepted native substep or practical speed.

GPU force diagnostics now expose the existing per-node target/residual scratch
and projected spring stiffness buffers. These are optional diagnostic handles,
not runtime CPU readbacks. Samples are unavailable before evaluation or after
a subsequent solve, invalid evaluation or stiffness change. Tests verify
sample identity, pinned-node zeros, targets, force reduction and capped versus
tension-only spring stiffness. Native force reconstruction remains test-only.

Optional `configure_gpu_timing(true)` records2*N+3 GPU timeline boundaries
for each substep: initial material, rigid prediction/gather, first contact,
alternating material/contact continuations, and final force checks/drift.
It requires timestamp-query support and an idle island; material-contact
reconfiguration disables it. Queries are paged in groups of256. Runtime
does not read timestamps back; diagnostic consumers wait for completion.
Intervals include inter-submission idle time, not just shader execution.
Bitwise state equivalence, missing-feature rejection, idle restrictions,
multi-page130iteration capture and4096iteration allocation are GPU-tested.

Native phase capture at027c2a4 exactly preserves the unprofiled candidate and
force samples. GPU timeline75.7136s contains74.6644s of continued contact
solves (98.614%) and0.5493s of continued material solves. This measures GPU
timeline spans, including queue idle, not isolated shader busy time.

`configure_material_contact_sweeps(Some(n))` optionally selects1..4096 contact
sweeps after each material correction. None preserves the initial solve count;
the first full contact solve is unchanged. Configuration is idle-only and
requires material coupling. Final contact, force and volume gates still apply.
One continuation sweep passes static/dynamic CPU references and the three-step
CEOL-friction cold/retained comparison. Native tests remain rejected:

| Outer iterations | GPU timeline | GPU force residual | Float64 force residual |
|---|---|---|---|
|256|8.9945s|0.0259891m/s|0.0259865m/s|
|1024|33.3008s|0.000221572m/s|0.000228551m/s|
|2048|70.4467s|0.000213391m/s|0.000220476m/s|

At1024outer, error is lower than the original256outer/12sweep candidate at
less than half its GPU timeline cost. Doubling again barely lowers the force
residual; contact corrections already fall below6e-7m/s. The residual plateau
requires material/precision diagnosis, not another blind iteration increase.
At2048outer the worst node5332 is a cord node with gravity-only inertial target,
not a node accumulating contact impulse. That revision established no accepted
native step, settling, sustained motion or practical simulator performance.

### Cord extension precision

Spring forces resolve length increments using
`dL = (2 dot(e,du) + dot(du,du)) / (length(e+du)+length(e))`.
The extension is `(length(e)-rest)+dL`; tension activation uses this extension.
Edges shorter than1e-8 retain the existing regularized force. This changes
neither material coefficients nor force/contact acceptance thresholds.

The GPU regression covers bilateral/tension-only springs, sub-position-ULP
increments, slack/preloaded states, compression, collapse and reversal. The
previous shader fails the same projected-float64 force reference.
Validation passes the GPU all-target check,7material-force GPU tests,
8coupling GPU tests and87CPU library tests (4ignored).

At the rejected native2048outer candidate, individual fixed-neighbor GPU
updates reduce node5332 residual from2.13391e-4 to3.05207e-7. Whole-mesh
fixed-target sweeps with the original shader instead oscillate:512sweeps
leave2.62992e-4 GPU and2.85476e-4 independent force residual. Incremental
spring lengths reduce the512sweep values to7.40579e-5 and9.17758e-5.
The128sweep independent value remains1.05065e-4: convergence margin is small.
These are material-only replays, not accepted coupled dynamics. Native
settling, sustained drive/turn/brake and practical performance remain open.

With the spring fix at e239ae1, the actual CEOL1024outer/1contact-sweep step
passes the unchanged GPU guard and independent force/volume/mass/momentum/
retention checks. Force is8.31871e-5 GPU and9.23243e-5 float64. The GPU
timeline is34.6208s for1/19200s simulated: contact still consumes88.878%.

A four-consecutive-step probe passes the GPU guard throughout, but the
independent material-force sequence gate FAILS at step2:

| Step | GPU force m/s | Independent force m/s | Independent nodes above1e-4 |
|---|---|---|---|
|1|8.31871e-5|9.23243e-5|0|
|2|7.53080e-5|1.37307e-4|3|
|3|7.79857e-5|9.13297e-5|0|
|4|6.68653e-5|9.57039e-5|0|

The oracle uses each immediately preceding accepted position buffer, current
inertial targets and projected material inputs. Step2's worst node15122 is
carcass material. Thus the native sequence is not independently accepted;
tet force/geometry precision requires further work before longer trajectories.
The4steps cover only0.000208333s and are not settled or performant.

### Tet gradient and cofactor precision

The elastic force now assembles the previous and incremental deformation
gradients separately. Explicit fused products include the identity term in
the previous gradient. Cofactors use compensated differences of products;
the volumetric stress evaluates determinant-minus-one without first rounding
the determinant near1. The elastic law, viscosity, Hessian and tolerances
are unchanged; all operations remain on the GPU.

At the captured step2 node15122, diagnostic gradient/stress arithmetic
reduced GPU/oracle force disagreement from1.33179e-4 to1.16426e-5 before
the final fused-identity refinement. A material-only full-mesh replay has
zero independent force violations after8..512sweeps. This is not native
trajectory acceptance. A self-contained regression covers rotated thin/skew
tets about principal and oblique axes, with and without viscosity, using
40nodal comparisons against projected-input float64 forces at3e-5 accuracy.
The old shader fails that regression; fused identity assembly passes it.
The extreme-friction1000 negative regression now requires a genuine force/
correction rejection within three steps, rather than on one exact third-step
solver variant. More accurate arithmetic triggers its guard on the second
step's cold solve. Finite residuals above the unchanged1e-4 thresholds and
status64 remain mandatory; real-friction accepted-trajectory gates are intact.
Validation passes the GPU all-target check,8material-force tests,3V3plastic/
multimaterial tests,3V4kinematic-hardening tests,8coupling tests and87CPU
library tests (4ignored).

Native3fa3e1d passes the GPU four-step probe and the independent sequence
gate with unchanged1e-4 force tolerance:

| Step | GPU force m/s | Independent force m/s | Minimum independent J |
|---|---|---|---|
|1|7.63633e-5|6.78361e-5|0.669221|
|2|8.44963e-5|8.38412e-5|0.687116|
|3|6.46574e-5|6.81780e-5|0.703812|
|4|7.04056e-5|9.30258e-5|0.718663|

All29568nodes pass independent force checks in each sample. The sequence
also checks total750kg, per-step momentum/state consistency and volume,
final retention0.000488279m and penetration9.61e-9m. Its duration is only
0.000208333s; quiet load-balanced time remains zero, so it is not settled
or sustained drive/turn/brake acceptance.

The first-step GPU timeline is26.2387s, including24.6537s continued contact
(93.96%) and1.07459s continued material. This is a measured run, not a
controlled performance comparison. Contact/coupling cost still prevents
practical simulator use and longer physical validation remains necessary.

### Ordered contact specialization

The cooperative contact pipeline specializes its DOF bounds/strides and
uses32lanes at1..32DOFs,64lanes above32. Preparation still uses64lanes;
device-limit and dimension validation remain unchanged. Independent rigid
velocity updates are emitted as constant-index GPU statements. Ordered
relative-velocity accumulation and contact order remain unchanged.

The captured29DOF native fixture has385683candidate rows and2386active
contacts. Original cooperative12-sweep solve time was196.047ms; the
specialized update replay measured119.935ms with particle/generalized/body
velocities and row impulses bitwise identical to the serial reference.
These are replay measurements, not full native or practical performance
acceptance. Unrolling the dot-product accumulation was rejected: it matched
the native fixture but changed rounding in a mixed-contact29DOF regression.

A self-contained sparse fixture exercises point/edge/triangle contacts,
static ground, two dynamic bodies and a ball loop at1/29/31/32/33/64DOFs.
It requires bitwise serial/cooperative equivalence and4+8continued sweeps
equal to12uninterrupted sweeps. Constructor boundary/limit gates are also
expanded. Validation passes the GPU all-target check,4replay/boundary tests,
5contact/motor tests,2static-contact tests,3ball-loop tests,8material/contact
coupling tests,1timing test and87CPU library tests (4ignored). Native
trajectory and practical performance acceptance remain pending.

Native752d5f9 now passes8consecutive GPU steps and the independent force,
volume,750kg mass and per-step momentum/state checks. Every physical field
in the first4packets exactly matches3fa3e1d. The additional steps have:

| Step | GPU force m/s | Independent force m/s | Minimum independent J |
|---|---|---|---|
|5|6.81369e-5|8.48641e-5|0.731768|
|6|6.86268e-5|8.27206e-5|0.743375|
|7|6.47577e-5|7.96821e-5|0.753767|
|8|6.65585e-5|7.60340e-5|0.763193|

Final retention is0.000810114m and penetration1.52e-8m. The run covers
only0.000416667s, with maximum particle speed3.01087m/s and zero sustained
quiet time. It is not settling or motor-driven trajectory acceptance.

The faster isolated replay does NOT establish native end-to-end speedup:
the new first-step timeline is30.5302s versus the historical26.2387s,
including25.2492s continued contact and4.86279s continued material.
These phase spans include capture/recovery/accumulation or submission idle,
not just individual kernel busy time. Finer native profiling is required;
practical performance and full simulator integration remain incomplete.

### Native contact subphase timestamps

Opt-in GPU timing also exposes N timestamp pairs after predictor copies and
after solve/recovery. The first pair includes contact preparation; later
pairs cover continued solve/recovery. Together with the existing2*N+3
boundaries these separate copies, solving and feedback/final-residual work.
Both streams use256-query pages and remain device-resident; no runtime
readback or physics changes are introduced. Timing-off behavior is unchanged.

GPU regression passes uncoupled,2outer,130outer/page-crossing and native-style
momentum-fence cases over2substeps, checking nested timestamps and bitwise
physical/momentum state. Missing feature, pending/disabled/reconfigured
states and4096iteration allocation are covered. GPU all-target check and
87CPU library tests (4ignored) pass. Nativef8870c9 first-step physics matches
752d5f9 exactly and passes independent force/volume/mass/momentum checks.
Continued copies take0.008868s, solve/recovery26.601653s and feedback/final
residual0.020513s: solving is99.89% of26.631034s contact time within a
32.953980s total timeline. Copies/feedback are not the dominant cost.

Batched response-prefetch trials preserve bitwise state but do not establish
a speedup and are reverted. Twelve-sweep replay grows from119..125ms to
140..156ms; single-sweep comparisons vary with the serial control. A fixed
single-sweep regression now repeats alternating serial/cooperative profiles
three times, retaining bitwise state checks and the original12sweep and4+8
continuation gates. All5replay tests pass. Archived fixture positions/masses
match current initial inputs; its2386active rows comprise2384dynamic and2ball
rows, with no static rows. A read-only active-contact buffer accessor and
timing parity test support checking the current native workload directly.

Native1594cd1 reports8086active rows, not the archived2386. Soft initial
positions/masses match, but the extra5700rows' type/distribution is not yet
captured; the archived replay cannot establish native performance. The
instrumented step exactly matchesf8870c9 physical state and passes independent
force/volume/mass/momentum checks. Its timeline is22.344843s, with20.338754s
continued solve/recovery,0.009072s copies and0.017175s feedback.110telemetry
samples at100% GPU utilization show2775MHzSM/10801MHzmemory/P2; clocks were
not changed. Earlier timing variation remains unexplained, so this is not
a controlled speedup claim. Capture the actual native workload before the
next contact-parallelism change; full tracked acceptance remains incomplete.

### Native input capture and separated static layers

The opt-in snapshot API published in2536e29 copies the ten contact inputs on
device immediately before the first solve. Gearbox test-only binary export
preserves settings and provenance without runtime node readback. The native
first step exactly matches1594cd1, including all force samples, and independently
passes force, volume,750kg mass and momentum checks. Its8086contacts contain
5700ground,2384dynamic and2ball rows. Ground support touches1368nodes and
dynamic support648nodes, with no shared nodes in this captured step.

The separated-static candidate builds its schedule entirely on GPU. It retains
dynamic row order and assigns static rows to at most64dependency layers,
preserving ascending row order at every node. One64-lane workgroup solves
independent rows in each layer with storage barriers between layers and sweeps.
Any overlap with dynamic support or layer-cap exhaustion selects the original
whole-island ordered GPU solve. Constraints, impulses and physical tolerances
are unchanged; continuation retains the schedule for its unchanged geometry.

The full native fixture matches the serial solver bitwise, including4+8versus12
sweeps. A paired12-sweep replay measures208.954ms cooperative versus123.371ms
separated solve, with7.967ms one-time scheduling. Repeated single-sweep solves
also improve, although clocks/timings vary and fresh preparation overhead can
outweigh a one-sweep saving. Native material continuation amortizes scheduling;
Published2c4c83d passes the native control one-step and eight-step gates below.
Six replay tests cover the CPU schedule oracle, wide point/edge/triangle layers,
overlap,64/65layer boundary, static-only/dynamic-only, sparse mixed1/29/31/32/33/64
DOF fixtures and constructor/dimension rejection. Snapshot and timing lifecycle
regressions pass with the integrated system constructor.
Additional gates pass87CPU library tests (4ignored),5contact/motor tests,
2static-contact tests,3ball-loop tests,8material/contact tests and1motor-feedback
test. These are solver regressions, not full-machine driving acceptance.

The first control step preserves every physical packet field, including forces,
against2536e29. Its GPU timeline measures12.0837s versus19.3809s; continued
solve/recovery measures10.5723s versus17.7637s. The eight-step control sequence
matches752d5f9 exactly and independently passes force, volume,750kg mass and
momentum checks. These timings are not practical-performance acceptance.

### Authored wheel-friction acceptance

The earlier native trajectories deliberately used `frictionless_drive_contacts`,
which clears friction on every retained wheel shape, including passive rollers.
Their ground friction remained enabled. They are causal controls, not evidence
for the full authored wheel-contact model. Gearbox now retains those controls
and adds explicit one/eight-step probes using the unmodified prepared shapes;
captures label this profile `authored_wheel_friction`.

The authored first-contact snapshot has2384dynamic rows with friction0.6,
5700ground rows and2ball rows. Positions/inverse masses match native inputs;
static/dynamic support remains disjoint. All six replay regressions pass against
this fixture, with serial/cooperative/separated outputs and4+8continuations
bitwise identical. Paired12-sweep solve timings are213.890ms cooperative and
128.245ms separated; these are replay timings, not a full trajectory speedup.

At2c4c83d the authored8step trajectory passes device and independent material
force, volume, mass and momentum checks without changing budgets or tolerances:

| Step | GPU force m/s | Independent force m/s | Minimum independent J |
|---|---|---|---|
|1|6.66971e-5|6.92646e-5|0.669221|
|2|6.83102e-5|8.16208e-5|0.687116|
|3|6.60377e-5|7.64838e-5|0.703812|
|4|6.74232e-5|7.66750e-5|0.718663|
|5|7.31414e-5|9.56847e-5|0.731768|
|6|6.86624e-5|7.23965e-5|0.743375|
|7|7.09902e-5|8.14230e-5|0.753767|
|8|7.76133e-5|7.77538e-5|0.763193|

All29568nodes pass each independent force gate. Final retention0.000810106m,
penetration1.52e-8m and max particle speed3.01087m/s are measured, but quiet
load-balanced time is zero. The sequence spans only0.000416667s. The separately
captured authored first-step timeline is18.1993s, including15.5492s continued
solve/recovery. Do not compare changed friction as an exact-state speedup.

Evidence is under Machines `.work/ceol/authored-friction-{native,eight-step}-evidence`
and `authored-friction-inputs`. Gearbox commitf8bac72 adds the profile probes;
the independent sequence verifier rejects mislabeled control profiles.

Full settling, sustained traction, practical performance, selectable simulator
ownership/menu/TF and wheel FEM remain incomplete. No canonical asset or
simulator FEM mode has been promoted from these short-step gates.

### Guarded final material polish

The authored64step probe at2c4c83d rejects step10 after9accepted steps:
status64, GPU material force4.447519e-4m/s versus the unchanged1e-4limit.
Independent reconstruction finds4.4839997e-4 at tread node10925 with20nodes
over tolerance. Contact velocity corrections remain1.28212e-5m/s or less.
This is genuine constitutive under-convergence, not volume failure or a
float representability floor; the longer trajectory is not accepted.

Fixed-target GPU replay of that rejected state converges after8additional
material sweeps: GPU7.478728e-5, independent7.094515e-5m/s, zero independent
violations and maximum velocity change1.11376e-5m/s. This replay excludes
coupled contact acceptance and cannot substitute for a native rerun.

An opt-in idle-only `configure_material_contact_polish(Some(sweeps))` adds
bounded retained material sweeps after the final contact correction. It keeps
the accumulated contact input unchanged and then rechecks velocity changes,
normal/friction/loop residuals and material force before drift. Existing
sticky faults and all tolerances are retained. Retained material context is
required; disabling reuse requires first disabling polish. Reconfiguration
resets polish, and default None preserves the original path. All solve work
remains GPU resident, with no node readback or imposed motion.

Validation passes87CPU tests,9material/contact GPU tests including dynamic
support momentum, real-friction energy/lifecycle and exhausted-budget rejection,
the immutable-input/nonfinite check-only kernel, snapshot replay, timing parity
including polish and motor-feedback regression. Gearboxf1c454b pinsc2566ed
and opts into8polish sweeps for authored-friction probes. Native step10 now
passes, with independent force7.521611e-5 versus GPU8.105209e-5m/s. However,
the64step probe rejects step12 after11accepted steps (222.85s wall time):
status64 while GPU force6.804205e-5, independent force9.243336e-5 and final
polish correction5.929813e-5 all remain below1e-4. The next diagnostic must
separate pre-polish sticky rejection from final contact complementarity.
The longer trajectory is NOT accepted, and no driveability or performance
claim follows from the accepted0.000572917s prefix.

Read-only contact-row diagnostics now expose the existing per-row residuals,
and a16byte device checkpoint preserves pre-polish status and three correction
channels before the final polish overwrites them. No status or physical state
is altered; the checkpoint is cleared each substep. Real-GPU coverage passes
all9coupling tests (197.32s), including low-budget sticky rejection and repeated
polish disable/re-enable, plus timing/state parity (26.84s). Capturing these
channels on the native step12 rejection remains the next gate.

That diagnostic rerun at62d2de6 now reproduces the same step12 rejection
(233.54s). Its11accepted samples are bitwise equal to the c2566ed baseline
after excluding revision/timestamps, and pass the same independent force,
volume, mass and momentum prefix checks. Pre-polish status is zero, with
largest contact correction6.357460e-5. The subsequent material polish creates
two ground normal-complementarity violations: GPUmaximum1.03494836e-4,
independent1.03496450e-4m/s. Rows3822/5228 exceed1e-4; friction residuals
remain about3.1e-6. This is a genuine final-contact failure, not stale status
or constitutive residual. Both contact and material equilibrium must converge
together; clearing status or loosening the limit would be incorrect.

Evidence is under Machines `.work/ceol/polish-rejection-evidence` and the
test-only exact first-contact inputs in `rejected-contact-inputs`. Gearbox
6625710 records the rejection channels and generalized velocities. A coupled
fixed-geometry replay fixture is being prepared to investigate continuation
without repeatedly integrating the accepted prefix. This fixture is not a
physical acceptance result or a replacement for a fresh native trajectory.

Gearbox CPU energy-reference export now includes partitioned rigid COM inertias
and joint-servo metadata. It matches native mass/geometry/spring data, but the
three572957.75stiffness servos require sampled joint coordinates and servo work
in the eventual energy audit. Test-only joint-coordinate sidecars now pass
step/time/revision/length/finiteness checks. An independent offline diagnostic
includes COM gravity, rotated anisotropic inertia, rubber/reinforcement elastic
energy and unsaturated static-target servo potential. The known component sum
decreases3869.102971 to3797.017704J through step11; this excludes viscous
incremental potential from stored energy. Contact/motor/limit work, dissipation
balance, intervening saturation and time/mesh refinement remain unverified.
These component measurements are not complete energy or dynamics acceptance.

### Coupled final polish cycles

A fixed-geometry GPU replay restores the rejected step12 material target,
velocities, contact impulses and generalized state after preparing responses.
Its zero-extra contact rows, material target and velocities are exact; the
material maximum is exact, while482per-node scalar residuals differ by at
most1.46678e-7. Four additional contact1/material8 cycles bring GPU/f64
material residual to7.20003e-5/8.94663e-5 and contact residual to
9.84235e-5/9.84241e-5. Momentum imbalance is about5.8e-8kgm/s after accounting
for changed ground impulses.8/16/32/64/128cycles also pass the simultaneous
residual checks. Replay3.71s and independent analysis pass; this is not a
native trajectory, volume/energy audit or practical-performance result.

The opt-in `configure_material_contact_polish_cycles(cycles)` implementation
interleaves retained contact corrections with final material passes. Default1
preserves existing polish behavior;1..4096requires idle enabled polish, and
polish reconfiguration resets1. Contact corrections update the accumulated
material input, not prescribed motion. Existing pre-loop sticky faults and
final contact, constitutive, velocity-change and volume guards are retained;
there is no CPU runtime readback/fallback or tolerance change. The diagnostic
replay's status restoration is test-only and is NOT used by this solver path.
Runtime validation passes all10GPU coupling tests194.55s, timing/state parity
28.83s and expanded dynamic-support momentum/CPU-oracle coverage42.81s,
including17cycles at both1/19200s and1/38400s. Low-budget sticky rejection,
real-friction energy/lifecycle and invalid/pending configuration remain covered.
Native17cycle first-step acceptance passes22.64s and independent force
6.645742e-5. Gearboxbe5ae8b pins67fd3ac and records the explicit cycle setting.
The longer trajectory now accepts12steps, all independently verified for
force/volume/mass/momentum, including step12force8.833364e-5 and minJ0.790282.
It rejects step13 after181.10s: status64, material6.842328e-5 but six final
ground normal residuals up to GPU3.604538e-4/f643.604647e-4. The64step
trajectory is NOT accepted; accepted physical time remains only0.000625s.

Captured step13 continuation through128cycles does not resolve the contact
plateau; its independent simultaneous gate correctly fails. A contact-only
control also plateaus near3.50377e-4. Planar geometry analysis finds2497sampled
rows with all vertex supports present and550stronger-than-implied bounds.
Edge3831 reports zero gap while its two vertex gaps imply2.9802322e-8m;
the resulting target discrepancy5.7220459e-4m/s exceeds the tolerance.
Symmetric edge5237 behaves identically. Maximum planar gap disagreement is
5.9604645e-8m. The box distance calculation subtracts its0.5m half-height from
local coordinates near0.5m to recover nanometer gaps. A stable face-relative
sampled distance needs GPU/f64 verification; no geometry fix or constraint
pruning has been implemented. Simply adding cycles or shifting the convergence
endpoint is not a demonstrated remedy for this plateau.

### Face-relative box geometry validation (2026-09-23)

The GPU gather now accumulates support offsets relative to the three signed
box faces, retaining the box edge/corner distance and normal calculation.
A failing real-GPU regression reproduced a represented gap4.8428774e-8m being
rounded to5.9604645e-8m. Three geometry tests now pass: signed faces,
static/dynamic translation, barycentric vertices/edges/triangles, radii,
dense/compact parity, rotated faces/edges/corners and composed body/shape poses
against f64 oracles. Existing surface2/compact4/static2/two-way drive5 GPU
tests pass, as do CPU84 (3ignored) and all-target GPU compilation.

A test-only captured step13 control re-gathers5700ground rows using this
shader, preserves all old normals/supports/friction/impulses, and recomputes
the gap-dependent velocity targets. Maximum f64 ground-gap error is1.80e-11m.
After128contact/material cycles, material passes (GPU8.58e-5/f648.00e-5),
but contact remains GPU2.069e-4/f642.069e-4. The independent simultaneous
gate correctly FAILS; contact-only continuation also plateaus near2.064e-4.
This is a validated geometry correction, NOT a complete contact convergence
fix or native trajectory acceptance. Worst remaining point rows3100/3784
have tiny penetrating gaps and positive normal impulses; remaining sampled
constraint/recovery convergence still needs diagnosis. No physical tolerances
or constraints were relaxed, and no CPU runtime fallback was introduced.

### Compensated material gradients (2026-09-24)

The stable-box native probe completed 64 substeps in 898.33 seconds, but its
independent material audit rejected steps 10 and 34: residuals 1.07241734e-4
and 1.01548442e-4 exceeded the unchanged 1e-4 velocity-equivalent force limit.
GPU status alone did not detect these violations. The trajectory covers only
3.333 milliseconds and establishes neither settling nor usable performance.

Gradient matrix products now accumulate product residuals and error-free sum
corrections, including the identity/origin term. This preserves f32 storage,
material laws, iteration budgets and physical tolerances without optional GPU
features or CPU runtime work. A self-contained production-API GPU regression
uses two recorded seven-node, four-tet neighborhoods and an independent f64
force oracle. The published shader fails with scalar error 5.4765163e-5;
compensated accumulation passes with errors 7.9024e-6 and 2.2911e-5 against
the 3e-5 arithmetic-parity limit. This is not exact f64 equivalence.

The previous high-friction negative trajectory now genuinely converges: all
six cold/reused candidates pass independent force, contact-row, energy and
state-parity checks. It remains a positive regression. A separate two-outer
case verifies both cold and reused paths reject actual non-equilibrium with
status 64; independent residuals are 6.232366 and 7.187426, not near a threshold.
All 11 coupling tests pass, with independent all-node force checks on the
accepted trajectories. Existing material-force 8, plasticity/multimaterial 3,
kinematic 3 and new gradient 1 GPU tests also pass.

Whole-mesh fixed-target replays of steps 10 and 34 preserve the initial force
failures. After one compensated material sweep, all 29568 nodes satisfy the
unchanged GPU and f64 force limits through checkpoints 1, 8, 32, 128 and 512.
At eight sweeps their maximum independent residuals are 8.759651e-5 and
8.008606e-5. These replay gates do not re-solve contact and are NOT fresh native
trajectory acceptance. Fresh coupled validation, settling, torque-driven
traction, energy/refinement, performance and selectable simulator ownership
remain required before enabling full FEM for users.

### Preserved rounding corrections (2026-09-24)

The next compensated native trajectory passed 64 GPU steps in 997.77 seconds,
but independent verification rejected step 42/node 20199: 1.056395e-4 versus
the unchanged 1e-4 force limit. The other 63 states passed. This remains failed
native acceptance and covers only 3.333 milliseconds of physical time.

Recorded GPU intermediates showed ordinary TwoSum corrections returning zero
despite nonzero exact rounding remainders. Product residuals were correct.
WGSL permits reassociation; FMA-subtraction, sign-bit subtraction and ordered
FastTwoSum formulations reproduced the missing corrections on this backend.
The sum remainder now uses integer significand/exponent alignment. The base
gradient retains high/low parts until the current displacement increment is
added. Material laws, storage, timestep, budgets and tolerances are unchanged.
No optional GPU feature, CPU runtime work or runtime readback was added.

A direct GPU regression checks 67206 arithmetic lanes against exact projected
sum remainders. The old helper fails; the new helper passes. Keeping the base
gradient correction through the increment also preserves the original force
regressions: errors 6.95924e-6 and 2.47731e-5 remain below 3e-5. The step 42 local
fixed-neighbor replay reaches independent 7.32628e-5 after 8 sweeps rather than
the old 1.12505e-4 plateau. This does not prove a fresh coupled trajectory.

Isolated validation passed 27 GPU tests: gradient 2, material-force 8,
plasticity/multimaterial 3, kinematic 3 and coupling 11 (230.62 seconds).
Both high-friction exhaustion controls still reject status 64 with independent
forces above 6. CPU 84 tests pass/3 ignored; all-target GPU compilation passes.
Fresh native verification, full physical scenarios and performance remain
required before simulator FEM activation.

### Strict material force precision (2026-09-24)

The next native trajectory rejected step16 after15 accepted states. Independent
force reconstruction passed every node (maximum6.70948e-5), while the GPU guard
reported1.01796e-4. The threshold remains1e-4. Four captured tet neighborhoods
now test both scalar parity and acceptance-side agreement, including the older
step42 false acceptance. Strict force evaluation carries gradient and cofactor
low parts through stress and compensates determinant-minus-one. Their errors
are5.87e-7,7.73e-7,1.57e-6 and4.26e-7; ordinary Newton evaluation is unchanged.

Captured cord16660 independently exposed another4.05899e-5 velocity-equivalent
error from two spring forces. Strict spring evaluation uses compensated
squared-length minus squared-rest and displacement low parts; the same captured
error falls to4.46564e-7. A production-API rotated/prestressed cord regression
fails before the correction (1.09177e-5 error) and passes afterward, including
tension-only slack and compression. No material law, budget or tolerance changes.

Validation:28 GPU tests pass (gradient2, material9, plasticity3, kinematic3,
coupling11);84 CPU tests pass/3 ignored, and all-target GPU compilation passes.
The coupling suite retains rejection of independently unconverged high-friction
states. Recorded whole-mesh material replays pass GPU/f64 gates from sweep0 at
step16 and sweep1 at step42; step42/sweep0 correctly rejects in both. Newton
delta buffers are bitwise unchanged at0/1/8/32/128/512 in both replays. These are
material-only controls, not fresh native/contact trajectory acceptance.

Remaining whole-mesh scalar disagreements reach4.34e-5 away from the maximum
residual. Uniform force accuracy is not proven. Fresh native acceptance, current
asset mass/inertia integration, settling, drive/reverse/turn/brake, obstacles,
energy/refinement, practical GPU performance and simulator ownership/menu remain
required. No simulator FEM activation or FEM-wheel completion is claimed.

Primary references:
- https://www.w3.org/TR/WGSL/#reassociation-fusion
- https://www.w3.org/TR/WGSL/#synchronization-builtin-functions
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-trusses.htm
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-embeddedelement.htm
- https://graphics.cs.utah.edu/research/projects/vbd/
- https://graphics.cs.utah.edu/research/projects/avbd/
- https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf
- https://mmacklin.com/smallsteps.pdf
