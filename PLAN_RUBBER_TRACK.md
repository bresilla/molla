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
2. **Reinforcement:** embedded longitudinal tensile cords, bending and shear
   response, volume preservation, hysteresis/damping, prestretch and tensioner.
   Establish resolution/timestep convergence before increasing stiffness.
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

Two-way contact now has a reusable CPU oracle in `soft_rigid_contact` and a
substep composition in `fem_rigid`. It uses articulated mass responses, not
independent-body inertia, for equal/opposite contact impulses and Coulomb
friction. A powered frictionless tooth transfers torque into the tetrahedral
loop in both directions: at 0.1 s the loaded rotor reaches about 0.235 rad/s
versus 1.775 rad/s unloaded, rubber angular momentum is about 0.453 kg m²/s,
and peak penetration is below 0.165 mm. This is one tooth pressing rubber,
not yet a circulating sprocket/lug mechanism or reinforced track.

GPU primitive contact generation reads live device FEM positions and rigid
transforms directly. The first real-GPU test checks 768 candidates and 39
active contacts against the CPU oracle, including ball/box/cylinder surfaces.
Only test acceptance downloads the resulting state. GPU articulated contact
response/solve and complete resident substep scheduling remain pending; the
CPU coupled bench is not the delivery path. Primitive gathering is presently
dense node/shape pairing, not a production broad phase or continuous collision
detection implementation.

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

GPU-first architecture with the CPU as correctness oracle remains required.
Keep physical laws in Molla; Gearbox owns rendering and machine binding.
The user reaffirmed that every GPU-suitable stage must execute on GPU, including
contact generation and two-way coupling, not only the elastic material update.
CPU coupling benches are correctness oracles and do not qualify as native FEM
delivery. Keep deformable positions, velocities and iterative contact state
device-resident; no per-node host round trips inside substeps.

Primary references:
- https://graphics.cs.utah.edu/research/projects/vbd/
- https://graphics.cs.utah.edu/research/projects/avbd/
- https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf
- https://mmacklin.com/smallsteps.pdf
