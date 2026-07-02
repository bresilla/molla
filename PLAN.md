# molla viewer parity plan

This plan replaces the previous `PLAN.md` content. It is only about **missing
physics viewer examples** found by scanning:

```text
xtra/warp/warp/examples/
xtra/newton/newton/examples/
bin/molla_viewer/src/main.rs
```

## Scope rules

- Physics examples only.
- No MJCF, USD, URDF, importer/exporter, asset pipeline, FFI, Python, Torch,
  JAX, DLPack, benchmark-only, distributed, allocator, graph-capture,
  work-queue, OpenGL-render, recording, replay, or external-policy work.
- Keep reusable implementation in functional crates under `crates/`.
- Keep the viewer as the single executable under `bin/molla_viewer`.
- GPU should be used wherever the physics path is GPU-feasible. If a GPU path
  is not meaningful yet, the scene must say why and keep a CPU oracle/test.
- All examples must be procedural or use in-repo generated geometry; do not
  require external robot/assets/import files.

## Current scan result

The viewer already covers most core Warp/Newton physics examples:

- Warp core raycast/raymarch/mesh/intersect/sample/marching-cubes/NVDB-like
  sparse grid/DEM/fluid/SPH/wave.
- Warp tile FFT/MLP/N-body/Cholesky/filter/compaction/matmul/convolution/MCGP.
- Warp FEM diffusion, 3D diffusion, convection-diffusion, Taylor-Green,
  Kelvin-Helmholtz, streamlines, deformed geometry, distortion, elastic shape
  optimization, FFT-Poisson/NS, APIC, stable fluid.
- Newton basic pendulum/joints/heightfield/conveyor/shapes, pyramid/brick
  contacts, cloth bending/twist/poker/hanging, cable pile/Y/twist, MPM
  snow/two-way/viscous/Bingham, diffsim ball/spring/drone, IK Franka/custom,
  sensors, softbody-on-cloth, and robot/cartpole-style procedural cases.

The remaining useful gaps are below.

## Priority 1 — highest value missing viewer examples

### 1. `MpmMultiMaterial`

Reference:

```text
xtra/newton/newton/examples/mpm/example_mpm_multi_material.py
```

Status: **partial viewer scene added**. `Scene::MpmMultiMaterial` exists in
`bin/molla_viewer/src/main.rs` with three procedural sand/snow/viscous visual
regions and an honest scene audit pointing at the existing reduced
multi-material CPU oracle. The current runtime MPM solver is still
single-rheology per simulation; true per-particle material IDs in the GPU MPM
path remain the required follow-up. Verified with the viewer unit suite,
`reduced_multi_material_oracle_dispatches_distinct_branches`, `cargo fmt`, and
`make check`. (Historical note: `make fmt` / per-family `make test-*-gpu`
targets were removed when the Makefile was simplified; GPU lanes now run via
`make test-gpu P=<pkg> T=<test>`.)

Add a procedural viewer scene that mixes at least sand, snow, and viscous/Bingham
material regions in the same MPM run.

Acceptance:

- Scene has distinct material groups visible in the viewer.
- CPU oracle proves material branch dispatch stays finite and distinct.
- GPU lane runs the material branches where available.
- Mass stays bounded/conserved within the existing MPM tolerances.

GPU:

- Expected GPU path: existing MPM material kernels.
- If a branch is CPU-only, document that branch explicitly in the scene audit.

### 2. `CableBundleHysteresis`

Reference:

```text
xtra/newton/newton/examples/cable/example_cable_bundle_hysteresis.py
```

Status: **partial viewer scene added**. `Scene::CableBundleHysteresis` exists
with seven procedural pinned bundle strands, a model/topology test, and a
Dahl-loop CPU oracle test proving positive hysteresis work plus delayed force
reversal on all strands. Full contact-coupled bundled Dahl integration remains
future work; the scene audit points at the Dahl/cable GPU lane (now run via
`make test-gpu P=molla-solvers T=<vbd/dahl test>`). Verified with the viewer
unit suite, the Dahl/cable GPU lane, `cargo fmt`, and `make check`.

Add a cable-bundle scene with several coupled CABLE/VBD strands and Dahl-style
hysteresis/friction response.

Acceptance:

- Bundle remains finite under repeated loading/unloading.
- Hysteresis loop shows path dependence, not just viscous damping.
- Existing Dahl oracle is connected to the scene proof.
- GPU cable/friction lane is named where applicable.

GPU:

- Expected partial GPU support through existing VBD/cable/Dahl lanes.
- Full bundled hysteresis kernel may stay future work if documented.

### 3. `NutBoltSdf`

Reference:

```text
xtra/newton/newton/examples/contacts/example_nut_bolt_sdf.py
```

Status: **partial viewer scene added**. `Scene::NutBoltSdf` exists with a
procedural six-wall socket, a dynamic cylinder bolt, and analytic torus SDF
rings used as thread-like contact lips. This intentionally avoids imported
threaded meshes/assets. The scene audit points at the existing dense-contact
CPU oracle plus the viewer SDF-sign test. Full threaded SDF narrow phase on GPU
is still future work; current GPU references are the contact-reduction and
sparse-grid lanes, not a claim that analytic SDF contact itself is GPU-resident.

Add a procedural nut/bolt-style dense contact scene using SDF or analytic
thread-like surfaces without importing meshes.

Acceptance:

- Dense caged contact remains finite.
- Contact count is high enough to exercise manifold reduction.
- The bolt cannot laterally escape the socket/nut.
- CPU SDF contact oracle proves expected normal/depth signs.

GPU:

- Use existing SDF/sparse-grid/contact-reduction GPU lanes where feasible.
- If the procedural SDF contact remains CPU-only, document why.

### 4. `NutBoltHydro`

Reference:

```text
xtra/newton/newton/examples/contacts/example_nut_bolt_hydro.py
```

Status: **partial viewer scene added**. `Scene::NutBoltHydro` exists as the
hydroelastic-audit variant of the procedural nut/bolt contact case. It reuses
the asset-free socket/bolt/SDF-thread geometry and points at reduced
hydroelastic patch oracles for penetration force growth and force/moment
preservation, plus `make test-hydroelastic-gpu` for the GPU kernel lane.
Hydroelastic forces are not yet coupled into the live viewer solver, so this
stays partial until the scene applies the distributed/anchored hydro contacts
during simulation.

Add a hydroelastic variant of the procedural nut/bolt contact case.

Acceptance:

- Hydroelastic patch force grows with penetration.
- Offset contacts preserve force and moment.
- Dense nut/bolt-style contact stays bounded.
- Scene audit points to the hydroelastic CPU/GPU proof.

GPU:

- Expected GPU path: existing hydroelastic GPU lane if compatible with the
  procedural geometry representation.

### 5. `DiffsimCloth`

Reference:

```text
xtra/newton/newton/examples/diffsim/example_diffsim_cloth.py
```

Status: **partial viewer scene added**. `Scene::DiffsimCloth` exists with a
procedural pinned cloth patch using the existing cloth/VBD topology. The
correctness proof is a reduced CPU differentiable oracle that checks a
low-dimensional pinned-edge control gradient against finite differences and
verifies optimization decreases loss. Forward GPU cloth coverage is named via
`make test-vbd-cloth-gpu`; full cloth adjoints/GPU adjoints remain future work.

Add a reduced differentiable cloth viewer example, e.g. optimize pin positions,
rest lengths, or a low-dimensional force/control vector to match a cloth target.

Acceptance:

- Loss decreases over optimization iterations.
- Gradient matches finite difference on a small cloth.
- Replay is deterministic.
- Cloth state remains finite and strain-bounded.

GPU:

- Forward GPU cloth path should be used where practical.
- CPU adjoint/finite-difference oracle is acceptable until GPU adjoints exist.

### 6. `DiffsimSoftBody`

Reference:

```text
xtra/newton/newton/examples/diffsim/example_diffsim_soft_body.py
```

Status: **partial viewer scene added**. `Scene::DiffsimSoftBody` exists with a
small procedural tetrahedral soft block. The correctness proof is a reduced CPU
differentiable oracle for low-dimensional initial-velocity/material response:
gradient matches finite differences and optimization decreases loss. Forward
tet GPU coverage is named via `make test-vbd-tets-gpu`; full tet adjoints,
material optimization, and GPU adjoints remain future work.

Add a reduced differentiable tet-softbody scene, e.g. optimize material,
actuation, or initial velocity to hit a target deformation/position.

Acceptance:

- Loss decreases.
- Gradient matches finite difference on a tiny tet mesh.
- Tet state remains finite and non-inverted/bounded.
- Scene audit distinguishes reduced differentiable softbody from full Newton
  parity.

GPU:

- Use existing VBD tet GPU forward path if practical.
- CPU gradient oracle is acceptable until a GPU adjoint exists.

## Priority 2 — useful FEM/IK parity gaps

### 7. `WarpFemConvectionDiffusionDg`

Reference:

```text
xtra/warp/warp/examples/fem/example_convection_diffusion_dg.py
```

Status: **partial viewer scene added**. `Scene::WarpFemConvDiffDg` exists as a
separate reduced DG-P0/FV advection-diffusion viewer scene, distinct from the
continuous SUPG `WarpFemConvDiff` scene. It includes limiter telemetry and CPU
tests for bounded/monotone behavior plus limiter clamping. The GPU audit points
at reusable DG flux/limiter coverage through `make test-fem-dg-swe-gpu`; a
dedicated scalar convection-diffusion DG GPU kernel remains future work.

Add a DG convection-diffusion viewer scene separate from the current continuous
`WarpFemConvDiff` scene.

Acceptance:

- Advection-dominated case stays monotone/bounded.
- DG limiter effect is visible or reported.
- CPU DG tests cover the same operator.

GPU:

- Use DG/SWE GPU-style flux lanes where reusable.
- If no dedicated convection-diffusion DG GPU kernel exists, document as CPU
  oracle plus future GPU flux work.

### 8. `WarpFemMixedElasticity`

Reference:

```text
xtra/warp/warp/examples/fem/example_mixed_elasticity.py
```

Status: **partial viewer scene added**. `Scene::WarpFemMixedElasticity` exists
as a reduced pressure/displacement split viewer scene comparing a
displacement-only locking proxy against a mixed pressure proxy for
near-incompressible material. CPU tests verify finite near-incompressible
behavior and that the mixed response avoids the severe locking of the
displacement-only proxy. GPU audit points at `make test-fem-elasticity-gpu` for
the existing displacement elasticity lane; full mixed pressure/displacement FE
solve and GPU parity remain future work.

Add a mixed-elasticity viewer scene if it can be expressed without external
assets.

Acceptance:

- Mixed formulation stays stable on near-incompressible material.
- Compare against the existing displacement-only elasticity scene.
- Stress/displacement remain finite.

GPU:

- Use FEM elasticity GPU assembly/solve where compatible.

### 9. `WarpFemNonconformingContact`

Reference:

```text
xtra/warp/warp/examples/fem/example_nonconforming_contact.py
```

Status: **partial viewer scene added**. `Scene::WarpFemNonconformingContact`
exists as a reduced nonmatching-surface load-transfer scene. It visualizes a
coarse source patch and nonmatching target triangle, transfers the contact load
with barycentric weights, and tests that resultant force and moment are
preserved. The audit points at existing nonconforming projection/penalty CPU
tests and FEM surface-contact transfer tests. GPU parity is not claimed because
there is no nonconforming contact GPU kernel yet.

Add a viewer scene for nonconforming FEM contact/load transfer.

Acceptance:

- Contact/load transfer preserves resultant force and moment.
- Nonmatching meshes remain finite.
- Scene points to nonconforming/contact CPU tests.

GPU:

- CPU oracle is acceptable unless/until nonconforming contact kernels exist.

### 10. `WarpFemStokesTransfer`

Reference:

```text
xtra/warp/warp/examples/fem/example_stokes_transfer.py
```

Status: **partial viewer scene added**. `Scene::WarpFemStokesTransfer` exists
as a reduced Stokes/FEM velocity-grid transfer scene: a procedural bilinear
velocity grid is sampled onto particle tracers, with constant-field,
linear/rotational boundedness, and tracer-Euler-step CPU tests. The GPU audit
names `make test-fem-apic-gpu` for the reusable APIC/field-transfer lane; a
full coupled Stokes-to-particle GPU transfer pipeline remains future work.

Add a viewer scene for transferring a Stokes/FEM velocity field to particles or
another representation.

Acceptance:

- Constant field transfer is preserved.
- Linear/rotational field transfer is bounded.
- Particle tracers follow the transferred velocity field.

GPU:

- Use APIC/transfer GPU lanes where possible.

### 11. `IkCubeStacking`

Reference:

```text
xtra/newton/newton/examples/ik/example_ik_cube_stacking.py
```

Status: **partial viewer scene added**. `Scene::IkCubeStacking` exists as a
reduced procedural pick/place scene: the existing GPU IK arm follows
pick/lift/place waypoints while procedural cubes are shown stacking in the
viewer. CPU tests cover waypoint reachability/audit linkage,
finite/supporting stack layout, and honest obstacle-residual telemetry. The GPU audit names
`make test-articulation-gpu` for the reusable GPU IK/articulation lane; full
gripper contact, coupled stack dynamics, and obstacle-aware IK objective parity
remain future work.

Add an IK-controlled stacking scene using procedural cubes.

Acceptance:

- IK target reaches pick/place poses.
- Stack remains finite and supported.
- Collision/obstacle residuals are reported honestly.

GPU:

- Use existing GPU IK lane for the arm solve where practical.
- Contact/stacking can remain CPU if no full coupled GPU path is available.

## Priority 3 — lower priority / reduced variants

### 12. `ClothRollers`

Reference:

```text
xtra/newton/newton/examples/cloth/example_cloth_rollers.py
```

Status: **partial viewer scene added**. `Scene::ClothRollers` exists with
procedural VBD cloth and three kinematic roller bodies. The viewer drives roller
angular velocity and reports tangential surface speed/support-gap telemetry.
CPU tests cover kinematic roller/cloth topology, the `ωr` surface-velocity
oracle, and finite initial support gap. The GPU audit names
`make test-vbd-cloth-gpu` for forward cloth; full frictional
cloth/roller tangential-velocity coupling remains future solver-side work.

Add a cloth-over-rollers scene using procedural kinematic cylinders.

Acceptance:

- Cloth remains finite and does not tunnel through rollers.
- Friction/kinematic motion transfers tangential velocity.
- Stretch remains bounded.

GPU:

- Use VBD/cloth GPU lane if roller contact is supported; otherwise CPU oracle.

### 13. `MpmBeamTwist`

Reference:

```text
xtra/newton/newton/examples/mpm/example_mpm_beam_twist.py
```

Status: **partial viewer scene added**. `Scene::MpmBeamTwist` exists as a
procedural twisted beam initialization over the current MPM runtime, with
torsion-colored particles in the viewer. CPU tests cover bounded twist
initialization, angle/color monotonicity, and audit honesty. Existing forward
MPM GPU lanes are named through `make test-mpm-gpu` / `make test-mpm-v3-gpu`;
a dedicated torsional beam constitutive/material setup remains future work.

Add a procedural MPM beam twist scene.

Acceptance:

- Twisted beam remains finite.
- Material response is visibly different from granular/fluid examples.
- Angular/torsional response is bounded.

GPU:

- Prefer GPU MPM material path; CPU oracle acceptable if this needs a new
  constitutive/twist setup first.

### 14. `FluidCheckpointOptimization`

Reference:

```text
xtra/warp/warp/examples/optim/example_fluid_checkpoint.py
```

Status: **partial viewer scene added**. `Scene::FluidCheckpointOptimization`
exists as a reduced scalar fluid-like checkpoint optimization. It saves bounded
rollout checkpoints, optimizes a single source/control amplitude against a
target field, and renders the optimized field versus target. CPU tests prove
checkpoint count is bounded, replay is deterministic, and loss decreases.
`make test-differentiable` is the named checkpoint/autograd verification lane;
related forward fluid/PDE GPU coverage is named through `make test-fem-ns-gpu`
and `make test-fft-gpu`. Full GPU fluid adjoint/checkpoint replay remains
future work.

Add a reduced viewer example demonstrating checkpointed differentiable rollout
for a fluid-like simulation.

Acceptance:

- Checkpointing bounds tape/window memory.
- Optimization loss decreases.
- Replay is deterministic.

GPU:

- GPU forward path where available; CPU adjoint/checkpointing is acceptable.

### 15. `NavierStokesPerturbationOptimization`

Reference:

```text
xtra/warp/warp/examples/optim/example_navier_stokes_perturbation.py
```

Status: **partial viewer scene added**.
`Scene::NavierStokesPerturbationOptimization` exists as a reduced procedural
flow-response optimizer over a scalar perturbation amplitude. CPU tests prove
the perturbation changes the flow response in the expected direction, the
analytic scalar gradient matches finite differences, and optimization decreases
loss. Forward Navier-Stokes GPU lanes are named through `make test-fem-ns-gpu`
and `make test-fem-fft-ns-gpu`; full GPU NS adjoint/perturbation optimization
remains future work.

Add a reduced NS perturbation optimization scene.

Acceptance:

- Perturbation parameter changes the flow in the expected direction.
- Gradient or finite-difference sensitivity is checked.
- Loss decreases in a small optimization loop.

GPU:

- Prefer existing FFT/NS GPU lanes for the forward solve.

## Explicitly skipped examples

Do not add viewer scenes for these categories:

- Importer/asset examples: URDF, MJCF, USD, XML, external robot assets.
- External policy examples requiring `.pt` or other learned-policy assets.
- JAX/Torch/CuPy/Paddle/Numba/Taichi interop examples.
- FFI/C++ examples.
- Allocator, graph-capture, work-queue, spin-lock, benchmark-only, distributed,
  OpenGL renderer, plotting, recording, and replay examples.
- Viewer/backend examples whose only purpose is infrastructure rather than
  physics behavior.

## Suggested implementation order

1. `MpmMultiMaterial`
2. `CableBundleHysteresis`
3. `NutBoltSdf`
4. `NutBoltHydro`
5. `DiffsimCloth`
6. `DiffsimSoftBody`
7. `WarpFemNonconformingContact`
8. `WarpFemConvectionDiffusionDg`
9. `WarpFemStokesTransfer`
10. `IkCubeStacking`
11. `WarpFemMixedElasticity`
12. `ClothRollers`
13. `MpmBeamTwist`
14. `FluidCheckpointOptimization`
15. `NavierStokesPerturbationOptimization`

## Completion definition

This plan is complete when:

- every listed scene exists in `bin/molla_viewer/src/main.rs`,
- every added scene has a `ScenePhysicsAudit` entry,
- every added scene has a CPU correctness test or a named existing CPU oracle,
- every GPU-feasible scene has a named GPU make target or a documented reason
  why GPU is not implemented yet,
- no scene depends on MJCF/USD/URDF/importer/exporter/FFI/external-asset scope,
- `nix develop --impure -c make check`,
  `nix develop --impure -c make test`, and
  `nix develop --impure -c make test-examples` pass after the additions.
