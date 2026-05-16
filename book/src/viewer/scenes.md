# Scene catalog

The `molla_viewer` ships with 101 self-contained scenes spanning every
solver, kernel, sensor, and FEM stepper the engine implements.
Pick one from the dropdown in the top-left of the viewer, or jump
to a marquee scene via a hotkey.

This chapter is the index. The scenes are grouped by topic; for the
strictly numerical ordering see `Scene::label()` in
`bin/molla_viewer/src/main.rs`.

## Group A — basics & rigid (XPBD)

| #  | Scene             | Hotkey | One-liner                                                       |
|----|-------------------|--------|-----------------------------------------------------------------|
| 01 | Pile              |        | Sphere shower onto a plane — XPBD floor + sphere narrow phase   |
| 02 | Rope              |        | Pinned 32-particle distance-constraint rope                     |
| 03 | Tower             |        | Stacked box tower                                               |
| 04 | Cloth             |        | XPBD cloth pinned at corners, rectangle drape                   |
| 30 | Pyramid           |        | Tall sphere/box pyramid struck by a heavy projectile            |
| 42 | BrickWall         |        | 6×10 brick wall (boxes) struck by a heavy projectile            |

## Group B — sensors (Stage 14)

| #  | Scene    | Hotkey | One-liner                                                |
|----|----------|--------|----------------------------------------------------------|
| 05 | Lidar    |        | GPU `Raycaster` — rays drawn as gizmo lines, hits as dots |
| 06 | Imu      |        | Bouncing sphere with a GPU `ImuSensor`; live accel + gyro |
| 07 | Contact  |        | Stacked bodies with GPU `ContactSensor`; per-body arrows  |
| 12 | Camera   |        | GPU 5-channel `TiledCamera` — color / depth / normal / albedo / shape-id |

Frame transform is exercised in [Group F #07] and in `tests/sensor_frame_transform_gpu.rs`.

## Group C — articulated rigid (Featherstone)

| #  | Scene         | Hotkey | One-liner                                                                |
|----|---------------|--------|--------------------------------------------------------------------------|
| 08 | Arm           |        | 3-link revolute chain + GPU PD actuator                                  |
| 10 | GpuIkArm      |        | 7-DOF Franka chases a draggable target via `IkSolver::solve_gpu`         |
| 13 | Actuator      |        | Single revolute joint driven by `Delay → ControllerPipelinePd → DcMotor` |
| 14 | Quadrotor     | `Q`    | FREE-root + 4 fixed rotor bodies; PD altitude + attitude control         |
| 17 | NativePendulum| `J`    | Procedural double pendulum stepped by the native Featherstone solver     |
| 26 | Pendulum      |        | 1-DoF revolute + 3-link chain pendulum side by side                      |
| 27 | JointSampler  |        | Revolute / prismatic / ball / distance joints, side by side              |
| 29 | Conveyor      |        | Kinematic conveyor belt + dropped boxes that slide                       |
| 37 | CartPole      |        | Cart on prismatic + pole on revolute; PD balance                         |
| 38 | CartPoleSwarm |        | 16 cart-poles in a row, each independently balanced                      |
| 39 | FourBar       |        | Closed kinematic loop via `EqualityType::JointMimic` for closure         |

## Group D — GPU stages (broad / narrow phase, IK, broadphase)

| #  | Scene          | Hotkey | One-liner                                                          |
|----|----------------|--------|--------------------------------------------------------------------|
| 11 | GpuNarrowPhase |        | Mixed-primitive pile with CPU↔GPU narrow-phase toggle (Stage 15)   |
| 15 | BroadPhase     |        | 4096-sphere shower with SAP↔LBVH toggle (Stage 19)                 |

## Group E — soft bodies (VBD)

| #  | Scene           | Hotkey | One-liner                                                                  |
|----|-----------------|--------|----------------------------------------------------------------------------|
| 16 | VbdCloth        | `V`    | 32×32 VBD cloth draping over a kinematic sphere (Stage 20)                 |
| 23 | Cable           | `C`    | 32-particle CABLE polyline yanked over a static cylinder (Stage 20.1)      |
| 25 | Jelly           | `Y`    | 8³ tet jelly cube + plane + kinematic puller (Stage 20.2)                  |
| 31 | ClothBending    |        | Side-by-side cloth strips with adjustable bending stiffness                |
| 32 | ClothTwist      |        | Cloth pinned top + bottom; bottom rotates to twist                         |
| 33 | CablePile       |        | Several cables dropped into a bowl, pile via CABLE bending                 |
| 34 | CableYJunction  |        | Three cables meeting at a Y junction (CONNECT equality)                    |
| 43 | SoftbodyDrop    |        | Tet soft cube onto a hanging VBD cloth (soft-on-soft)                      |
| 44 | ClothCollar     |        | Pinned cloth tube collapsing onto itself (Stage 33.3 self-collision)       |
| 45 | FastSoftImpact  |        | Fast cube into pinned cloth strip — CCD demo (Stage 33.4)                  |
| 46 | SoftBrake       |        | Tet block sliding down a tet ramp; friction sweep (Stage 33.5)             |
| 84 | ClothPokerCards |        | 52 stiff VBD cloth cards, kinematic sphere knocks them over                |
| 85 | SoftbodyHanging |        | 4 tet soft bodies hanging from pinned faces, E-modulus sweep               |
| 86 | CableTwist      |        | Single cable pinned at both ends; one endpoint rotates to inject twist     |
| 95 | JellyDampingFall|        | 4 jelly cubes (tet) with different damping slots                            |

## Group F — MPM (Stage 24, 39A, 40)

| #  | Scene             | Hotkey | One-liner                                                                |
|----|-------------------|--------|--------------------------------------------------------------------------|
| 24 | Mpm               | `P`    | Granular sand column poured into a bin (Stage 24d v1 explicit)           |
| 35 | MpmSnowBall       |        | Snow ball drop with `MpmRheology::Snow` (Stomakhin)                      |
| 36 | MpmTwoWay         |        | Heavy rigid sphere drops into sand pit (two-way coupling)                |
| 59 | WarpDem           |        | Warp port — MPM sand at ~45° friction angle                              |
| 60 | WarpFluid         |        | Warp port — MPM at ~5° friction angle (fluid-like)                       |
| 61 | WarpSph           |        | Warp port — MPM-as-fluid alternate layout (no real SPH solver)           |
| 93 | ImplicitMpmBingham|        | Implicit MPM with Bingham rheology (Stage 39A)                           |
| 98 | ImplicitMpmFull27 |        | Implicit MPM with full 125-point Newton Hessian (Stage 40 v4)            |

## Group G — differentiable / autograd

| #  | Scene             | Hotkey | One-liner                                                              |
|----|-------------------|--------|------------------------------------------------------------------------|
| 18 | DiffArm           | `G`    | 2-link planar arm reaches a target via gradient descent (Stage 23)     |
| 19 | DiffArm6          | `H`    | 6-DOF Franka reaches target via reverse-mode AD (Stage 23.1)           |
| 40 | DiffBall          |        | Ball (v, angle) initial condition → hoop; gradient on miss             |
| 41 | DiffSpring        |        | Particle in a spring cage; gradient on rest lengths                    |
| 51 | FemDiffBeam       |        | Differentiable FEM beam (gradient descent, Stage 36.4)                 |
| 66 | WarpParticleRepulsion |    | 32 particles gradient-descent repulsion via autograd tape              |
| 75 | WarpFemElasticOpt |        | FEM elastic shape optimization (beam height adjoint)                   |
| 80 | WarpDiffRay       |        | Differentiable raycasting — single sphere position via depth match     |
| 83 | AdamLandscape     |        | Adam on a 2D Rosenbrock landscape with loss heatmap (Stage 38c)        |
| 88 | DiffsimDrone      |        | MPC via FD gradients of a 50-step quadrotor rollout                    |
| 99 | SplatPoseRecovery |        | Differentiable 3DGS — 1024-splat camera-pose recovery (Stage 44 v2)    |

## Group H — neural controllers

| #  | Scene         | Hotkey | One-liner                                                            |
|----|---------------|--------|----------------------------------------------------------------------|
| 21 | NeuralPolicy  | `N`    | Cart-pole MLP (4 → 32 → 32 → 1) balance controller (Stage 18.1)      |
| 22 | NeuralLstm    | `L`    | Cart-pole LSTM (4 → 16 → 1) balance controller (Stage 31)            |
| 63 | WarpTileMlp   |        | Single MLP forward pass on 16 inputs (Warp tile port)                |

## Group I — Shapes and mesh utilities

| #  | Scene         | Hotkey | One-liner                                                                |
|----|---------------|--------|--------------------------------------------------------------------------|
| 09 | Shapes        |        | Cones / hulls / spheres dropped onto a wavy heightfield + SDF torus     |
| 20 | MeshTools     | `M`    | Decimate / SDF-bake / marching-cubes / convex-hull (Stage 21, no sim)    |
| 28 | Heightfield   |        | 32×32 sinusoidal heightfield + 30 mixed primitives dropped on top        |
| 56 | WarpSampleMesh |       | Uniformly random barycentric samples on a TriMesh sphere                 |
| 87 | IkCustom      |        | Collision-avoidance IK on the GpuIkArm Franka with sphere obstacle       |

## Group J — Warp core primitive ports

| #  | Scene             | One-liner                                                                |
|----|-------------------|--------------------------------------------------------------------------|
| 52 | WarpRaycast       | Fire a grid of rays at a kinematic primitive scene                       |
| 53 | WarpRaymarch      | Host CPU SDF raymarch into a 128×128 image                               |
| 54 | WarpMesh          | Animated `TriMesh` cube with procedural rotate + breathe scale           |
| 55 | WarpMeshIntersect | Two cubes whose overlap is reported via GJK                              |
| 57 | WarpMarchingCubes | Bake animated SDF (torus + box smooth-min), extract via MC               |
| 58 | WarpNvdb          | Sparse-grid insert + lookup at random voxels; latency in pane            |

## Group K — Warp tile compute primitives

| #  | Scene             | One-liner                                                                |
|----|-------------------|--------------------------------------------------------------------------|
| 62 | WarpTileFft       | GPU FFT on a 256-sample sinusoid                                         |
| 64 | WarpTileNbody     | Host-driven N-body with inverse-square gravity, 64 particles             |
| 65 | WarpTileCholesky  | Dense CPU Cholesky on a 16×16 SPD matrix                                 |
| 81 | WarpTileFilter    | `tile_convolution` + `tile_filtering` demo (Stage 37.8)                  |
| 82 | WarpTileCompaction| `tile_stream_compaction` demo (Stage 37.8)                               |
| 89 | WarpFftPoisson2d  | 2D FFT-Poisson on 32×32 periodic grid with moving Gaussian source        |
| 90 | WarpTileMatmul    | Standalone GEMM `C = A·B` demo, M=64 N=32 K=128 (Stage 37.8)             |
| 91 | WarpTileConvolution| Separable 2D convolution with 5-tap kernel (Stage 37.8)                 |
| 92 | WarpTileMcgp      | Monte Carlo Laplace via walk-on-spheres on the unit square               |

## Group L — Warp FEM port scenes

| #  | Scene                       | One-liner                                                                  |
|----|-----------------------------|----------------------------------------------------------------------------|
| 67 | WarpFemDiffusion            | 2D tri-mesh heat equation with Dirichlet BCs on vertical edges             |
| 68 | WarpFemDiffusion3D          | 3D tet-mesh diffusion with **time-varying** sinusoidal Dirichlet BCs       |
| 69 | WarpFemConvDiff             | Steady advection-diffusion `u·∇T = α∇²T` on a 2D tri mesh                  |
| 70 | WarpFemTaylorGreen          | Stokes Taylor-Green vortex initial condition; analytic decay in pane       |
| 71 | WarpFemKelvinHelmholtz      | Stokes shear layer with sinusoidal perturbation                            |
| 72 | WarpFemStreamlines          | Stokes lid-driven cavity with tracer particles                             |
| 73 | WarpFemDeformedGeometry     | Heat equation on a curved (annular) tri mesh                               |
| 74 | WarpFemDistortion           | Linear elasticity with distortion energy as loss; strain heatmap           |
| 76 | WarpFftFluid                | 2D Stokes solve as a fallback for FFT-Poisson NS coupling                  |
| 77 | WarpDistortion3D            | Bonus 3D distortion-energy example                                         |
| 79 | WarpApicFluid               | APIC FEM-MPM hybrid coupling demo (Stage 37.6)                             |
| 78 | WarpWave                    | Wave PDE `u_tt = c²∇²u` on 32×32 P1 tri mesh, Newmark-β (Stage 37.1)       |
| 94 | WarpStableFluid             | 2D Stam stable-fluid solver on periodic grid (Stage 39C)                   |

## Group M — molla FEM steppers

| #  | Scene         | One-liner                                                                  |
|----|---------------|----------------------------------------------------------------------------|
| 47 | FemHeat       | 16³ tet cube heat equation, backward-Euler (Stage 36 v1)                    |
| 48 | FemBeam       | 20×4×4 cantilever beam, linear + SVK elasticity (Stage 36.1)               |
| 49 | FemHeatP2     | P2 (quadratic-nodal) heat equation (Stage 36.3)                            |
| 50 | FemStokes     | Lid-driven cavity, Taylor-Hood + Uzawa-CG (Stage 36.2)                     |

## Group N — DG and shocks (Stage 43)

| #  | Scene                | One-liner                                                                    |
|----|----------------------|------------------------------------------------------------------------------|
| 96 | DgBurgersShock       | 1D Burgers' equation Lax-Friedrichs + minmod, shock at t ≈ 1/π               |
| 97 | ShallowWaterDamBreak | 2D radial dam break, Lax-Friedrichs flux on 48×48 grid                       |

## Group O — AMR, magnetostatics, splat (Stages 44–46)

| #   | Scene             | One-liner                                                                |
|-----|-------------------|--------------------------------------------------------------------------|
| 100 | AmrHeat           | AMR heat equation with moving Gaussian source (Stage 45 v2)              |
| 101 | Magnetostatics    | Short current loop drives a steady B-field via H(curl) Nedelec (Stage 46)|

## Catalog summary

- **101 scenes total**, numbered #01–#101 in the dropdown.
- 12 marquee scenes have single-letter hotkeys.
- ~30 are direct ports of Warp examples (prefix `Warp*`).
- ~25 are Newton-port scenes covering the Newton example tree.
- ~12 are FEM steppers (Stages 36–46).
- The remainder are original molla demos built around specific
  Stage deliverables.

Each scene's source lives in `bin/molla_viewer/src/main.rs`.
Search for the variant name (e.g. `Scene::ImplicitMpmBingham`) to
find the `build_scene` arm, the per-scene state struct, the update
system (if any), and the control_pane arm.

See [`viewer/overview.md`](./overview.md) for the architecture
behind the catalog, and [`roadmap/stages.md`](../roadmap/stages.md)
for the chronological stage history that produced each scene.
