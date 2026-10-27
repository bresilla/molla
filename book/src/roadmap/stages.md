# Stage history

molla was built in numbered stages. Each stage is a self-contained
deliverable with tests + benches + (usually) a viewer scene. This
chapter is the chronological roll-up grouped by family. The
authoritative source is `PLAN.md` at the repository root; this is
the human-readable index.

## Family 0 — scaffolding (Stage 0–2)

| Stage | Title                                      | Commit                                |
|-------|--------------------------------------------|---------------------------------------|
| 0     | Engine scaffold and PLAN                   | `d2531d3`                             |
| 1     | Semi-implicit Euler on CPU                 | `4fdfcce`                             |
| 2     | GPU semi-implicit Euler                    | `b88597c`                             |

## Family A — rigid bodies (Stage 3–7)

| Stage | Title                                              | Commit       |
|-------|----------------------------------------------------|--------------|
| 3.0   | Joint data model + FK                              | `c76cb60`    |
| 3     | Featherstone CPU+GPU                               | `a564b53`    |
| 4.0   | XPBD + sphere/plane contacts                       | `49550ee`    |
| 4     | Boxes — SAT, SAP, jacobi XPBD                      | `421e4c0`    |
| 5     | Joint limits, drives, multi-articulation, substeps | `bb0772a`    |
| 6.1   | Capsule + cylinder shapes, capsule narrow phase    | `933743f`    |
| 6     | GJK + EPA + BVH + trimesh                          | `29c32b9`    |
| 7a    | Multi-world per-entity, joint metadata, collision groups | `60ecc56` |
| 7     | Joint types, equality + mimic constraints, jacobian | `348ba33`   |
| 7     | Humanoid ball-joint demo via CONNECT               | `42d7e83`    |

## Family B — retired external-format experiments

| Stage | Title                                              | Commit       |
|-------|----------------------------------------------------|--------------|
| 5–8   | Retired scene-format bridge scaffold               | `5e5b033`, `152bc6f` |
| 29    | Retired multi-schema resolver experiments          | `36f305c`, `43b5524` |
| —     | Remove external-format bridge crate                | `8c81b13`    |

## Family C — distance constraints + cloth (Stage 12)

| Stage | Title                                          | Commit       |
|-------|------------------------------------------------|--------------|
| —     | XPBD distance constraints and cloth simulation | `02e617e`    |
| 12    | Cloth-vs-rigid SDF collision pass on GPU       | `adc39fa`    |
| 12    | Parallel GPU particle XPBD via colouring       | `277d1ec`    |

## Family D — sensors (Stage 14)

| Stage | Title                          | Commit       |
|-------|--------------------------------|--------------|
| 14    | GPU ray-cast sensor + Lidar    | `f6cbb06`    |
| 14    | GPU IMU sensor                 | `1c0a3f1`    |
| 14    | GPU contact sensor             | `5ac4635`    |
| 14    | GPU frame-transform sensor     | `296d886`    |
| 14    | Multi-channel raytraced camera | `Camera scene` |

## Family E — actuators (Stage 18, 18.1, 31)

| Stage | Title                                              | Commit       |
|-------|----------------------------------------------------|--------------|
| 18    | GPU PD/PID actuator controllers                    | `bcbc872`    |
| 18    | Composable actuator pipeline (Delay → PD → DcMotor)| `Actuator scene` |
| 18.1  | Neural-MLP cart-pole controller                    | `NeuralPolicy scene` |
| 31    | Neural LSTM controller                             | `NeuralLstm scene` |
| 42    | Hill-type muscle for CABLE joints                  | `1272b5d`    |
| 42 v3 | GPU Hill kernel + cable ABA                        | `bbd5c61`    |
| 42 v4 | SEE Newton GPU/CPU parity                          | `0411d0b`    |

## Family F — GPU narrow + broad + IK (Stage 15, 16, 19)

| Stage | Title                                | Commit       |
|-------|--------------------------------------|--------------|
| 15    | GPU narrow phase                     | `GpuNarrowPhase scene` |
| 16    | GPU IK on Franka                     | `GpuIkArm scene` |
| 17    | Multi-DOF Featherstone (quadrotor)   | `Quadrotor scene` |
| 19    | GPU LBVH broad phase                 | `BroadPhase scene` |

## Family G — VBD soft bodies (Stage 20, 20.1, 20.2, 20.3, 20.4)

| Stage   | Title                                       | Commit       |
|---------|---------------------------------------------|--------------|
| 20      | High-resolution VBD cloth                   | `VbdCloth scene` |
| 20.1    | CABLE polyline + Dahl friction              | `Cable scene` |
| 20.2    | JELLY tetrahedral soft body                 | `Jelly scene` |
| 20.3    | Multi-material tets + von Mises plasticity  | (wave-4)     |

## Family H — mesh utilities (Stage 21)

| Stage   | Title                                            | Commit       |
|---------|--------------------------------------------------|--------------|
| 21      | Decimate / SDF bake / marching cubes / hull      | `MeshTools scene` |
| 21.1    | Rayon-parallel mesh-utility tools (Cargo dep)    | —            |

## Family I — hydroelastic (Stage 22.1)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 22.1    | Hydroelastic contact (CPU + GPU)                   | `f3febbe`    |

## Family J — autograd (Stage 23, 23.1)

| Stage     | Title                                              | Commit       |
|-----------|----------------------------------------------------|--------------|
| 23        | Differentiable arm (open-loop torque grad descent) | `DiffArm scene` |
| 23.1      | Differentiable Featherstone (6-DOF reverse AD)     | `DiffArm6 scene` |
| 23.1.1    | GPU dispatcher parity + closed-form perf           | —            |
| 23.1.2    | Pure-analytic CPU adjoint + GPU multi-arti kernel  | (wave-4)     |
| 23.1.3    | Distance + D6 joint adjoints                       | (wave-4)     |

## Family K — small-matrix decomps (Stage 24a)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 24a     | SVD3 / QR3 / Eig3 / polar                          | (2026-05-09) |
| 38b     | Adjoint SVD3 + polar decomposition                 | `f3325b5`    |

## Family L — FFT, MPM, sparse grid (Stage 24c, 24d, 24e)

| Stage     | Title                                            | Commit       |
|-----------|--------------------------------------------------|--------------|
| 24c       | GPU FFT kernel                                   | —            |
| 24d v1    | Explicit MPM (sand)                              | `Mpm scene`  |
| 24d.1     | Semi-implicit MPM + Snow + 2-way coupling        | (wave-4)     |
| 24d.2     | Persistent-buffer MPM dispatcher (6.9× speedup)  | (wave-4)     |
| 24e v1    | Sparse grid (NanoVDB) mutable insert + lookup    | —            |
| 24e.1     | Murmur3-based GPU root hashing                   | (wave-4)     |

## Family M — Stage 28 contact reduction + MPR

| Stage   | Title                                          | Commit       |
|---------|------------------------------------------------|--------------|
| 28.1    | GPU MPR portal-loop kernel                     | `b40dc99`    |
| 28.1    | GPU contact reduction kernel                   | `6c78d67`    |
| 28.1    | MPR + contact reduction parity                 | `0606233`    |

## Family N — soft contact (Stage 33, 33.2, 33.2.1, 33.3, 33.4, 33.5)

| Stage   | Title                                          | Commit       |
|---------|------------------------------------------------|--------------|
| 33      | Soft-on-soft VT contact pipeline               | `SoftbodyDrop scene` |
| 33.2    | GPU soft VT contact                            | (GPU port)   |
| 33.2.1  | GPU Coulomb friction + restitution             | (GPU port)   |
| 33.3    | Cloth-collar self-collision                    | `ClothCollar scene` |
| 33.4    | Fast-impact CCD                                | `FastSoftImpact scene` |
| 33.5    | Soft brake (friction sweep)                    | `SoftBrake scene` |

## Family O — FEM v1 + variants (Stage 36 family)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 36      | FEM framework v1 (P1 nodal scalar)                 | `FemHeat scene` |
| 36.1    | FEM elasticity (linear + SVK)                      | `FemBeam scene` |
| 36.2    | FEM Stokes (Taylor-Hood + Uzawa-CG)                | `FemStokes scene` |
| 36.3    | P2 (quadratic-nodal) elements                      | `FemHeatP2 scene` |
| 36.4    | Adjoint FEM solver + grad helpers                  | `FemDiffBeam scene` |
| 36.5    | GPU FEM v1 (assembly + CSR matvec + resident CG)   | `697b85a`, `85c1345` |
| 36.7    | Direct solver scaffolding (russell feature)        | `65f90ec`    |

## Family P — Stage 37 (Phase 2 Warp engine work)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 37.1    | Wave PDE + Newmark-β + scene #78                   | `c4ea941`    |
| 37.2 + 37.4 | Convection / SUPG + distortion Newton          | `f30c6fb`    |
| 37.3 + 37.5 | NS time-stepping + FFT-NS coupling             | `6acddf2`    |
| 37.6    | APIC FEM-MPM hybrid coupling                       | `4ae31f7`    |
| 37.7    | Raycast FD adjoint                                 | `3087ef3`    |
| 37.8    | Tile primitives bundle (matmul / cholesky / conv / stream-compaction / MCGP) | `8b93493` |

## Family Q — Stage 38 cluster (GPU sparse + autograd)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 38a     | GPU sparse BSR algebra (matvec/axpy/transpose/spgemm) | `e99d93f` |
| 38b     | Adjoint SVD3 + polar decomposition                 | `f3325b5`    |
| 38c     | Adam + SGD with CPU + GPU fused-update kernels     | `0eaaabd`    |
| 38d.1   | Route particle + spring coloring through geometry  | `4a559e2`, `dc70cd2` |
| 38e     | Device-wide scan + 64-bit segmented radix sort     | `bb17a41`    |
| 38.1    | Differentiable MPM with SVD adjoint chain          | `6934075`    |
| 38.2    | Differentiable FEM Adam loop fits Young's modulus  | `93341f3`    |

## Family R — Stage 39 (DG, MPM Bingham, SpGEMM, CABLE, stable fluid)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 39A     | Implicit MPM with Bingham rheology (CPU)           | `e02f180`    |
| 39B     | DG (Discontinuous Galerkin) P0/P1 basis            | `d97b5c1`    |
| 39C     | 2D Stam stable fluid solver                        | `75156f7`    |
| 39D     | Hydroelastic anchor contact + bucket reduction     | `0ac863c`    |
| 39E     | CABLE joint type with 2 DoFs (stretch + bend)      | `09e4868`    |
| 39F     | Full-GPU SpGEMM with bitmap symbolic phase         | `399a1f8`    |

## Family S — Stage 40 (implicit MPM full Newton Hessian)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 40      | Full 27-point Newton Hessian (CPU)                 | `2abb95b`    |
| 40 v2   | GPU P2G + predict + G2P                            | `0172745`    |
| 40 v3   | GPU Bingham + advect, resident p_C                 | `ae3eabc`    |
| 40 v4   | Full 125-point Newton Hessian                      | `6cb417d`    |

## Family T — Stage 41 (GPU FEM)

| Stage    | Title                                              | Commit       |
|----------|----------------------------------------------------|--------------|
| 41.1     | GPU linear elasticity assembly + resident CG       | `72ce86f`    |
| 41 v2    | GPU Stokes steady-state + Uzawa-CG                 | `de9d43b`    |
| 41 v2    | GPU APIC FEM-MPM coupling                          | `5fb4d7d`    |
| 41 v2    | GPU hydroelastic contact + anchor reduction        | `f3febbe`    |
| 41 v3    | GPU IMEX Navier-Stokes time stepper                | `f2ae80f`    |
| 41 v4    | Fused NS-IMEX RHS axpy on GPU                      | `ccbe5c6`    |
| 41 v5    | Async-dot fusion for Uzawa inner CG                | `3169fac`    |
| 41 v6    | Fused-kernel pipelined CG (single dispatch per iter) | `2bde182`  |

## Family U — Stage 43 (DG SWE / Burgers)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 43      | DG P2 + numerical-flux library + BJ limiter        | `7bb5402`    |
| 43 v2   | DG infra expansion                                 | `00f36c1`    |
| 43 v3   | HLL/HLLC/Roe SWE flux + characteristic limiter     | `d37995e`    |
| 43 v4   | HLL/HLLC/Roe SWE flux GPU port                     | `bed9c43`    |

## Family V — Stage 44 (Gaussian splats)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 44 v1   | Forward 3DGS raytrace renderer                     | `d4583a7`    |
| 44 v2   | 3DGS reverse adjoint + pose recovery               | `237caa2`    |
| 44 v3   | 3DGS spherical-harmonic colour                     | `00d8083`    |
| 44 v4   | SH band 3 + diffuse/view split                     | `121c9d3`    |

## Family W — Stage 45 (AMR)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 45 v1   | AmrMesh foundation + uniform refinement            | `1e2c5d5`    |
| 45 v2   | AMR heat scene #100                                | `571934a`    |
| 45      | AMR hanging-node constraints + 2:1 balance         | `a7c51f6`    |
| 45      | AMR P2 hanging + DG-AMR faces + state transfer    | `4442e3d`    |
| 45 v4   | AMR coarsening + Krylov restart after re-mesh     | `66d976a`    |

## Family X — Stage 46 (magnetostatics)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 46      | H(curl)/Nedelec edge elements + magnetostatics     | `e4eab58`    |

## Family Y — Stages 47–51 (advanced FEM)

| Stage   | Title                                              | Commit       |
|---------|----------------------------------------------------|--------------|
| 47 v1   | NonconformingField L² projection                   | `8a756ab`    |
| 47 v2   | Vector projection + penalty                        | `d2599bd`    |
| 47 v3   | Surface-only cross-mesh M_st quadrature           | `0f5a27e`    |
| 47 v4   | Coupled penalty contact Newton-Raphson             | `0f9e57e`    |
| 48      | Time-harmonic Maxwell / eddy currents              | `35976c8`    |
| 48 v1+v2 | Eddy-current GMRES + single-level AMS              | `f6fe6a1`    |
| 48 v3   | Geometric multigrid V-cycle for AMS               | `fa3616e`    |
| 48 v4   | Algebraic multigrid (Ruge-Stüben) for AMS          | `c0cb1a6`    |
| 49      | Signorini fixed-active-set Lagrange contact        | `60a7792`    |
| 50      | Coulomb friction at Signorini contact              | `e751d17`    |
| 51      | Signorini active-set update loop                   | `4a4fc64`    |

## Reading the table

Each row is a single brick on the wall. The total count is **597
commits** on `main` at the time of this snapshot, spanning Stage 0
through Stage 51. 132 viewer scenes (numbered through #124, plus
unnumbered extras) correspond to a subset of these stages (some
stages don't add a scene, e.g. infra work).

For the chronological commit log run:

```bash
git log --oneline main
```

For the per-stage long-form rationale, design rationale,
deliverable list, and DoD (definition-of-done), open `PLAN.md`.

## Open items and out-of-scope

See [open.md](./open.md) and [oos.md](./oos.md) for the things
that are *not* on this list.
