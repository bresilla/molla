# GPU port status matrix

A comprehensive table of which solvers, kernels, sensors,
actuators, and FEM steppers have GPU paths today, which have
CPU-only implementations, and which have parity tests.

Notation:
- **GPU** — full GPU dispatch path exists.
- **CPU** — CPU reference implementation (oracle).
- **Parity** — `tests/<name>_gpu.rs` exists and asserts CPU/GPU
  agreement.
- **CPU-only** — only the Rust path ships; no WGSL kernel.

## Solvers

| Solver                  | CPU | GPU | Parity test                       | Notes                                          |
|-------------------------|-----|-----|-----------------------------------|------------------------------------------------|
| `SemiImplicitSolver`    | yes | yes | `cpu_gpu_parity.rs`               | Stage 1–2; the baseline                        |
| `XpbdSolver` (rigid)    | yes | yes | `xpbd_gpu_parity.rs`              | Stage 4–5                                      |
| `XpbdSolver` (particle) | yes | yes | `xpbd_particles_gpu.rs`           | Stage 12                                       |
| `XpbdSolver` (equality) | yes | yes | `xpbd_equality_gpu.rs`            | CONNECT / weld / distance                      |
| `XpbdSolver` (mimic)    | yes | yes | `equality_joint_mimic.rs`         | Stage 32                                       |
| `FeatherstoneSolver`    | yes | yes | `featherstone_gpu_parity.rs`      | Stage 3                                        |
| `FeatherstoneSolver` (multi-DOF) | yes | yes | `featherstone_multidof_gpu.rs` | Stage 17                                   |
| `VbdSolver` (cloth)     | yes | yes | `vbd_cloth_gpu.rs`                | Stage 20                                       |
| `VbdSolver` (tets)      | yes | yes | `vbd_tets_gpu.rs` + v3 + v4       | Stage 20.2 + 20.3 + 20.4                       |
| `VbdSolver` (soft VT contact) | yes | yes | `soft_vt_contact_gpu.rs`    | Stage 33.2                                     |
| `VbdSolver` (soft friction) | yes | yes | `soft_friction_gpu.rs`        | Stage 33.2.1                                   |
| `BarrierSolver` (barrier-Newton, C0) | yes | yes | `barrier_newton_gpu.rs` (7 tests: kernel + engine path + pair friction) | Invariant + energy-parity gates incl. on-device pair friction |

## Sensors

| Sensor                  | CPU | GPU | Parity test                       | Notes                                |
|-------------------------|-----|-----|-----------------------------------|--------------------------------------|
| `Raycaster`             | yes | yes | `sensor_raycast_gpu.rs`           | Stage 14                             |
| `TiledCamera` (5-ch)    | yes | yes | `sensor_tiled_camera_gpu.rs`      | Stage 14                             |
| `ImuSensor`             | yes | yes | `sensor_imu_gpu.rs`               | Stage 14                             |
| `ContactSensor`         | yes | yes | `sensor_contact_gpu.rs`           | Stage 14                             |
| `FrameTransformSensor`  | yes | yes | `sensor_frame_transform_gpu.rs`   | Stage 14                             |
| Raycast adjoint         | yes | (FD-only) | `raycast_adjoint.rs`        | Stage 37.7 — FD-based, no analytic   |

## Actuators

| Actuator                  | CPU | GPU | Parity test                       | Notes                              |
|---------------------------|-----|-----|-----------------------------------|------------------------------------|
| `ControllerPd`            | yes | yes | `actuator_pd_gpu.rs`              | Stage 18                           |
| `ControllerPid`           | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `ControllerPipelinePd`    | yes | yes | `actuator_pipeline_gpu.rs`        | Stage 18                           |
| `ControllerPipelinePid`   | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `Delay`                   | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `ClampingDcMotor`         | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `ClampingMaxEffort`       | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `ClampingPositionBased`   | yes | yes | (covered by pipeline tests)       | Stage 18                           |
| `ControllerNeuralMlp`     | yes | yes | (covered by Scene::NeuralPolicy)  | Stage 18.1                         |
| `ControllerNeuralLstm`    | yes | yes | `actuator_neural_lstm_gpu.rs`     | Stage 31                           |
| Hill-type muscle          | yes | yes | (covered by `joint_muscle_gpu.rs`)| Stage 42                           |
| Dahl friction             | yes | yes | (covered by cable tests)          | Stage 20.1                         |

## Kernels

### Math / sparse

| Kernel                      | CPU | GPU | Parity test                | Notes                          |
|-----------------------------|-----|-----|----------------------------|--------------------------------|
| `BsrMatrix::mv`             | yes | yes | `bsr_gpu_parity.rs`        | Stage 38a                      |
| `BsrMatrix::axpy`           | yes | yes | `bsr_gpu_parity.rs`        | Stage 38a                      |
| `BsrMatrix::transpose`      | yes | yes | `bsr_gpu_parity.rs`        | Stage 38a                      |
| `BsrMatrix::spgemm`         | yes | yes | `spgemm_symbolic_gpu.rs`   | Stage 39F                      |
| CG / BiCGStab / GMRES / PCG | yes | yes (via FEM) | `iterative.rs`    | CPU runs everywhere; GPU through FEM stepper |
| SVD3 / QR3 / Eig3           | yes | yes | `dense3_gpu.rs`            | Stage 24a                      |
| SVD3 adjoint                | yes | yes | `dense3_adj_gpu.rs`        | Stage 38b                      |
| FFT (1D + 2D)               | yes | yes | `fft_gpu.rs`               | Stage 24c                      |
| Scan + radix sort (32-bit)  | yes | yes | `scan_gpu.rs` + `radix_sort_gpu.rs` | Stage 38e             |
| Radix sort (64-bit)         | yes | yes | `radix_sort_64_gpu.rs`     | Stage 38e                      |

### Geometry

| Kernel                      | CPU | GPU | Parity test                | Notes                          |
|-----------------------------|-----|-----|----------------------------|--------------------------------|
| GJK                         | yes | yes | `gjk_gpu.rs`               |                                |
| EPA                         | yes | yes | (CPU oracle in narrow_phase) | Stage 6                      |
| MPR                         | yes | yes | `mpr_gpu.rs`               | Stage 28.1                     |
| Contact reduction           | yes | yes | `contact_reduction_gpu.rs` | Stage 28.1                     |
| LBVH (build + refit)        | (CPU oracle) | yes | `broadphase_lbvh_gpu.rs` | GPU-first                  |
| Mesh-BVH raycast            | yes | yes | (covered by sensor tests)  |                                |
| Sparse grid                 | yes | yes | `sparse_grid_v3.rs`, `v4.rs` | Stage 24e                    |
| Heightfield + SDF           | yes | yes | `heightfield_and_sdf.rs`   |                                |
| Convex hull (incremental)   | yes | CPU-only | `cone_and_hull.rs`     | Authoring-time, not a hot kernel |
| Marching cubes              | yes | (via MeshTools scene) | `mesh_utils.rs` | Authoring-time              |
| Mesh decimation (QEM)       | yes | CPU-only | `mesh_utils_hardened.rs` | Authoring-time                |

### FEM

| Kernel / Stepper            | CPU | GPU | Parity test                    | Notes                          |
|-----------------------------|-----|-----|--------------------------------|--------------------------------|
| P1 assembly                 | yes | yes | `fem_gpu.rs`                   | Stage 36.5                     |
| P1 matvec (CSR)             | yes | yes | `fem_gpu.rs`                   |                                |
| P2 elements                 | yes | partial | `fem_p2.rs`                | Stage 36.3                     |
| Elasticity (linear + SVK)   | yes | yes | `fem_elasticity_gpu.rs`        | Stage 41.1                     |
| Stokes (Taylor-Hood + Uzawa) | yes | yes | `fem_stokes_gpu.rs`           | Stage 41 v2                    |
| Navier-Stokes (IMEX)        | yes | yes | `fem_ns_gpu_lazy_cg.rs`, `fem_ns_gpu.rs` | Stage 41 v3, v4, v5, v6 |
| Wave (Newmark-β)            | yes | yes | `fem_wave_gpu.rs`              | Stage 37.1                     |
| Transient heat              | yes | yes | `fem_transient_gpu.rs`         | Stage 36                       |
| DG (P0 / P1 / P2)           | yes | yes | `fem_dg_swe_gpu.rs`, `fem_dg.rs`, `fem_dg_v2.rs`, `fem_dg_v3.rs`, `fem_dg_advanced.rs` | Stage 39B, 43 |
| APIC FEM-MPM hybrid         | yes | yes | `fem_apic_gpu.rs`              | Stage 37.6, 41 v2              |
| Adjoint FEM                 | yes | partial | `fem_adjoint.rs`           | Stage 36.4, 38.2               |
| AMR                         | yes | (CPU oracle) | `fem_amr*.rs` (×4)    | Stage 45                       |
| Magnetostatics (Nedelec)    | yes | (CPU oracle) | `fem_magnetostatics.rs` | Stage 46                      |
| Eddy currents (Maxwell)     | yes | partial | `fem_eddy_*.rs` (×4)       | Stage 48                       |
| Signorini contact           | yes | partial | `fem_signorini*.rs` (×3)   | Stage 49–51                    |
| Convection (SUPG)           | yes | yes | `fem_convection.rs`            | Stage 37                       |
| Distortion energy           | yes | (Newton-CPU) | `fem_distortion.rs`   | Stage 37.4                     |
| Nonconforming meshes        | yes | partial | `fem_nonconforming*.rs` (×4)| Stage 47                       |

### MPM

| Kernel                      | CPU | GPU | Parity test                | Notes                          |
|-----------------------------|-----|-----|----------------------------|--------------------------------|
| Explicit MPM                | yes | yes | `mpm_gpu.rs`               | Stage 24d v1                   |
| Semi-implicit + Snow + 2-way| yes | yes | `mpm_v3_gpu.rs`            | Stage 24d.1                    |
| Implicit Bingham            | yes | partial | `implicit_mpm_gpu.rs`  | Stage 39A, 40                  |
| Diff MPM                    | yes | (CPU oracle) | `mpm_diff_optim.rs`  | Stage 38.1                     |

### Splat (Stage 44)

| Kernel                      | CPU | GPU | Parity test                  | Notes                        |
|-----------------------------|-----|-----|------------------------------|------------------------------|
| Forward raytrace            | yes | yes | `splat_raytrace_forward.rs`  | v1                           |
| Backward raytrace           | yes | yes | `splat_raytrace_backward.rs` | v2                           |
| SH band 3                   | yes | yes | `splat_sh_*.rs` (×3)         | v3                           |
| Pose recovery (end-to-end)  | yes | (GPU FD) | `splat_pose_recovery.rs` | v2                          |

### Optim

| Kernel                      | CPU | GPU | Parity test                | Notes                          |
|-----------------------------|-----|-----|----------------------------|--------------------------------|
| Adam                        | yes | yes | `optim_gpu_parity.rs`      | Stage 38c                      |
| SGD                         | yes | yes | `optim_gpu_parity.rs`      | Stage 38c                      |
| Robust losses               | yes | yes | `robust_loss_gpu.rs`       |                                |

### Tile compute (Stage 37.8)

| Kernel                      | CPU | GPU | Parity test                | Notes                          |
|-----------------------------|-----|-----|----------------------------|--------------------------------|
| `tile_matmul`               | yes | yes | `tile_primitives_gpu.rs`   |                                |
| `tile_convolution`          | yes | yes | `tile_primitives_gpu.rs`   |                                |
| `tile_filtering`            | yes | yes | `tile_primitives_gpu.rs`   |                                |
| `tile_block_cholesky`       | yes | yes | (covered in `tile_primitives_gpu.rs`) | Stage 37.4 / 37.5    |
| `tile_stream_compaction`    | yes | yes | (covered in `tile_primitives_gpu.rs`) | Stage 37.8           |
| `tile_mcgp` (SPSA)          | yes | yes | (covered in `tile_primitives_gpu.rs`) |                       |
| Tile FFT                    | yes | yes | (covered in `fft_gpu.rs`)             |                       |

## Summary

- **Every solver in `prelude` has a GPU path and a parity test.**
- **Every sensor has a GPU path and a parity test.**
- **Every actuator has a GPU path; most have dedicated parity
  tests, the rest are covered by pipeline tests.**
- **The vast majority of FEM steppers have GPU paths.** A handful
  (AMR, magnetostatics, full Signorini, some adjoint paths) ship
  CPU-only with the GPU port still on the roadmap.
- **Compute primitives (tile, FFT, sort, scan, BSR, splat) are GPU
  with CPU oracles.**

The CPU paths are not deprecated — they remain the debugging path
and the oracle for parity testing. The "GPU port still on the
roadmap" status applies to a small number of FEM and MPM
sub-paths and is tracked in `PLAN.md` and `GAP_v2.md`.
