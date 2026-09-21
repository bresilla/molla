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

`TetVolumeMonitor` records the initial configuration and every accepted GPU
substep, retaining the lifetime minimum determinant rather than only inspecting
frame-end meshes. Invalid/nonpositive elements set sticky status bit 8;
nonfinite determinants set bit 4. A restored final mesh cannot erase an earlier
failure. Unit GPU checks cover transient compression, recovery, inversion,
collapse, NaN/Inf, invalid indices and preservation of other status flags.
The track gate now requires lifetime minJ > 0.5 without per-substep readbacks.
This detects invalid material states; it does not itself prevent inversion.

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

The installed full-loop gate is still failing. With 2% prestrain, material
circulates 67.1 mm (pitch 56.5 mm) and envelope departure is 10.7 mm, but a
tetrahedron inverts (minJ -0.0169). At 0.5% prestrain the belt circulates
78.8 mm with 17.3 mm departure, but lug elements compress excessively
(minJ 0.2251; settled minJ already 0.4743). No volume or circulation acceptance
threshold was weakened. Contact/elastic coupling and geometric resolution
remain required work before any full-track or native-simulator claim.

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
and response assembly, and no real-time performance claim is made.
Primitive gathering is presently
dense node/shape pairing, not a production broad phase or continuous collision
detection implementation.

The resident composition currently accepts one unreversed dynamic articulation
up to 32 joints and 64 DOFs and explicit primitive contact shapes. It exposes sticky GPU
failure status; callers must check it before accepting results. Existing GPU
Featherstone does not implement reversed joints, motor-only caps or the CPU
dense multi-DOF acceleration solve; those are not silently delegated to CPU.
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

Primary references:
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-trusses.htm
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-embeddedelement.htm
- https://graphics.cs.utah.edu/research/projects/vbd/
- https://graphics.cs.utah.edu/research/projects/avbd/
- https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf
- https://mmacklin.com/smallsteps.pdf
