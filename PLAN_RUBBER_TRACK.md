# Deformable reinforced rubber track

## Goal and acceptance boundary

Replace the ideal continuous-belt drive with a volumetric deformable belt.
Motor torque must reach belt nodes through sprocket/lug contact; belt-ground
contact must generate vehicle motion with equal/opposite reactions. No imposed
belt phase, prescribed belt velocity, or chassis velocity override qualifies.
Retain the current reduced model as a separately named, usable baseline.

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

Primary references:
- https://graphics.cs.utah.edu/research/projects/vbd/
- https://graphics.cs.utah.edu/research/projects/avbd/
- https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf
- https://mmacklin.com/smallsteps.pdf
