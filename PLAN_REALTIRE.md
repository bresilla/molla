# Real Tire + Tire/Ground Contact Plan

## Goal

Replace the viewer-only "patch factor" tire behavior with reusable tire physics in
`molla-vehicle`: a pressure-dependent deformable tire contact patch, a
material-aware ground contact law, and a brush/bristle traction model that can
distinguish asphalt, dirt, sand, and ice.

The viewer examples may stay simplified, but they must call the real model
instead of inventing fake traction multipliers locally.

## Current implementation status

- `crates/molla-vehicle/src/real_tire.rs` now contains the reusable tire work:
  - pressure/material brush contact (`evaluate_brush_tire`);
  - legacy dynamic shell-style carcass (`DynamicTireCarcass`);
  - true volumetric tetrahedral tire FEM (`TetraTireFem`).
- `Scene::TireTraction` uses the brush contact model for pressure/material
  dependent force and slip.
- `Scene::TireDynamics` now uses `TetraTireFem`: an annular tire volume with
  circumferential × radial × width nodes, hexahedral parameter cells split
  into tetrahedra, finite-deformation compressible Neo-Hookean internal rubber
  forces, bead/rim coupling, floor contact, and pressure-dependent membrane
  pre-stress / tangent stiffness.
- The tetra mesh now has first-pass material regions:
  - softer sidewall rubber;
  - stiffer tread/belt rubber;
  - stiff bead/rim-adjacent elements;
  - carcass fallback material for the remaining body.
- Direct pressure traction on the coarse exterior tread defaults to zero. The
  pressure effect is currently represented as pressure-dependent tangent
  stiffening because direct exterior pressure on this coarse mesh drove the
  lower tread into the floor. The physically better next step is an internal
  cavity pressure/volume constraint.
- `crates/molla-kernels/shaders/tire_fem.wgsl` now provides the first GPU
  tire-FEM node-step kernel. The Rust wrapper is `TireFemGpuKernel`, with raw
  snapshot export from `TetraTireFem::gpu_snapshot()`. It uses the same
  per-element material Lamé parameters and Neo-Hookean update as the CPU path.
  The GPU test lane is: `make test-gpu P=molla-kernels T=tire_fem_gpu`.
- `Scene::TireDynamics` defaults to the GPU tire-FEM node kernel through the
  viewer right rail; CPU is retained as a fallback/debug toggle while hub
  reaction reduction is still CPU-side.
- `Scene::TireTraction` has a dirt→ice slipperiness slider so wheel spin is
  easy to force and inspect.
- CPU tests cover brush contact, dynamic carcass stability, volumetric mesh
  creation, tetra FEM floor contact, and lower-pressure wider patches.
- The remaining GPU work is GPU hub/contact reduction, persistent buffers, and
  a faster adjacency/reduction layout for many tires.

## What is wrong today

- The remaining gap is not "fake viewer traction" anymore; it is solver depth:
  - the tire FEM is explicit Neo-Hookean tetra FEM, not an implicit tire
    solver;
  - inflation pressure is a membrane pre-stress approximation, not yet a true
    cavity pressure / enclosed-volume constraint;
  - floor contact is still a penalty contact, not a zero-gap
    Lagrange-multiplier surface contact;
  - brush traction and tetra carcass deformation are still separate layers;
    the tread contact nodes are not yet feeding a distributed bristle state;
  - tire heat, wear, anisotropic cord plies, lateral brush forces, and full
    deformable-terrain coupling are future extensions.

## Real model architecture

### 1. Surface material model

Add a real surface/contact material structure:

```rust
SurfaceMaterial {
    mu_static,
    mu_dynamic,
    bristle_damping,
    rolling_resistance,
    cohesion,
}
```

Initial built-ins:

- dry asphalt/concrete: high static and dynamic friction;
- dirt/gravel: medium friction, more damping;
- ice: very low friction;
- sand/soil: medium/low friction plus higher rolling resistance.

The existing `Terrain::friction()` remains as a compatibility scalar, but the
real tire APIs accept a `SurfaceMaterial` so the model can distinguish static
vs dynamic friction and rolling loss.

### 2. Pressure-dependent contact patch

Represent the tire contact patch by sampling the tire circumference into
segments. For a given hub pose, radius, width, pressure, and terrain height:

- compute each segment's ground penetration;
- compute per-segment normal load;
- solve hub height for a requested wheel load;
- report:
  - total normal load,
  - patch length,
  - contact cell count,
  - per-cell normal pressure.

Pressure affects vertical compliance:

- high pressure -> less deflection, shorter patch, higher local pressure;
- lower pressure -> more deflection, longer patch, lower local pressure;
- too low pressure -> high rolling resistance / carcass loss.

### 3. Brush/bristle longitudinal traction

For each contact segment:

- compute local slip speed from wheel spin vs hub velocity;
- convert slip speed into bristle shear displacement;
- compute trial tangential force from tread shear stiffness;
- cap by material friction:
  - static region: `|Ft| <= mu_static * Fn`;
  - sliding region: `Ft = mu_dynamic * Fn` opposite slip.

This gives the important behavior:

- asphalt: even small patch can transmit useful torque before slipping;
- ice: low `mu` caps force even if many segments touch;
- low pressure increases patch length and contact stiffness, improving
  low-frip traction up to the point where rolling resistance dominates.

### 4. Wheel spin dynamics

Driven wheels should integrate angular velocity:

```text
I * omega_dot = motor_torque - Fx * radius - bearing_loss
```

Vehicle longitudinal motion integrates:

```text
m * v_dot = sum(Fx_wheels) - rolling_resistance - drag
```

Slip is therefore an outcome: if motor torque exceeds what the tire/ground can
transmit, wheel angular velocity climbs while the car barely accelerates.

### 5. Viewer integration path

First integration target:

- `Scene::TireTraction`
  - red fixed car: high pressure;
  - blue fixed car: partial pressure;
  - green user car: pressure + payload sliders.

Replace local patch-factor fake with calls into `molla_vehicle::real_tire`:

- solve wheel hub height from payload/load;
- evaluate brush contact for the driven wheel;
- integrate wheel spin and chassis velocity;
- draw patch bars/readouts from actual contact output.

Second integration target:

- `Scene::TireDynamics`
  - now uses `TetraTireFem` instead of quasi-static shape generation;
  - continue improving toward implicit/nonlinear FEM and GPU kernels.

## Validation / tests

Add crate-level tests in `molla-vehicle`:

1. **Pressure changes patch length**
   - same load on flat terrain;
   - low pressure produces more contact cells / longer patch than high pressure.

2. **Surface material dominates ultimate grip**
   - asphalt can transmit more longitudinal force than ice for the same load
     and pressure.

3. **Lower pressure improves low-friction tractive stiffness**
   - same low-friction material and same slip speed;
   - partial pressure produces more brush force than over-inflated pressure,
     before the friction cap dominates.

4. **Torque-induced slip is an outcome**
   - with the same motor torque/load:
   - high pressure on low-friction ground accumulates higher slip;
   - partial pressure moves farther over the same simulated interval.

5. **Volumetric tetra FEM mesh and contact**
   - the tire mesh has circumferential/radial/width nodes and many tetrahedra;
   - loaded tetra FEM tires stay finite and above the floor;
   - lower pressure gives a wider tetra FEM footprint than high pressure.

## Implementation stages

### Stage A — reusable real-tire module

- Add `crates/molla-vehicle/src/real_tire.rs`.
- Export it from `molla-vehicle`.
- Implement:
  - `SurfaceMaterial`;
  - `RealTireParams`;
  - `RealWheelState`;
  - `RealTireContact`;
  - `solve_loaded_tire_hub_height`;
  - `evaluate_brush_tire`.
- Add the first crate tests.

### Stage B — use it in `Scene::TireTraction`

- Replace local `patch_factor()` force calculation.
- Store wheel angular velocity per car.
- Let tire force and slip come from `evaluate_brush_tire`.
- Keep the right-rail pressure/weight sliders.

### Stage C — improve tire shape demo

- Replace the old shape demo with `TetraTireFem`.
- Draw the actual tetra FEM surface wireframe and floor-contact tread nodes.
- Keep pressure bars and fixed/animated pressure comparison.

### Stage C2 — nonlinear tire FEM pass

- Replace the previous small-strain linear tetra model with compressible
  Neo-Hookean/hyperelastic tetra elements. **Done.**
- Add first-pass scalar material regions for sidewall, tread/belt, bead, and
  carcass. **Done.**
- Export per-element Lamé parameters to GPU and use them in WGSL. **Done.**
- Move from explicit time integration to implicit, semi-implicit Newton, or
  XPBD FEM constraints for larger stable time steps. **Open.**
- Replace pressure-as-stiffening with a true internal cavity gas law:
  pressure = f(volume, gas amount), forces from the gradient of enclosed
  volume, and optional mounting/inflation initialization. **Open.**
- Add anisotropic cord/belt material directions rather than scalar stiffness
  regions. **Open.**
- Couple the FEM tread nodes directly to the brush/terrain material contact
  model instead of the current flat-floor viewer contact. **Open.**

### Stage D — GPU path

The first GPU node-step path now exists in `molla-kernels`:

- `shaders/tire_fem.wgsl`
  - one invocation per tire FEM node;
  - each node loops over tetrahedra and accumulates only its own force
    contribution, avoiding float atomics/races;
  - applies compressible Neo-Hookean tetra stress, bead/rim coupling, optional
    pressure traction, and flat-floor contact;
  - writes next positions, velocities, and force diagnostics.
- `TireFemGpuKernel`
  - blocking wrapper for tests and small batches;
  - consumes f32 raw buffers exported by `TetraTireFem::gpu_snapshot()`;
  - validated by the ignored hardware lane
    `make test-gpu P=molla-kernels T=tire_fem_gpu`.
- Viewer wiring:
  - `Scene::TireDynamics` defaults to GPU node stepping;
  - right rail exposes a `Use GPU tire FEM kernel` toggle;
  - the first kernel still sends node buffers through a blocking wrapper, so it
    is correctness-first rather than throughput-first.

Follow-up GPU scaling work:

- replace the O(nodes × tets) race-free reference layout with node/tet CSR
  adjacency or per-tet force reduction;
- add GPU reductions for hub reaction force/contact patch diagnostics;
- add brush/contact GPU kernels for `Scene::TireTraction`.

## External modeling notes shaping the next pass

Realistic tire work usually combines a structural tire model with a local
tread/road contact model:

- Brush models remain a useful physics-grounded basis for tire/road contact,
  and even advanced tire formulations commonly keep brush-like local contact
  behavior.
- Real-time physical tire models often discretize tire geometry and couple
  bristle kinematics to carcass deformation instead of using a single point
  friction coefficient.
- Validated FEM tire studies use nonlinear tire/rim and tire/road contact,
  load-deflection curves, pressure-distribution checks, and experimental
  comparison. For full FEM tire-terrain interaction, the serious path includes
  rim/bead contact, inflation setup, surface-to-surface tread/road contact,
  material regions/reinforcements, and validation against static/dynamic test
  data.
