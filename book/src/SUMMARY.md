# Summary

[Introduction](./introduction.md)

---

# Foundations

- [Architecture overview](./architecture/overview.md)
  - [Workspace layout](./architecture/workspace.md)
  - [Crate boundaries](./architecture/crates.md)
  - [Feature flags](./architecture/features.md)
  - [Build & test commands](./architecture/build.md)
  - [Design rules](./architecture/design_rules.md)

---

# Math & Numerics

- [Vectors, matrices, transforms](./math/types.md)
- [Spatial algebra](./math/spatial.md)
- [Small-matrix decompositions](./math/dense3.md)
- [Sparse linear algebra](./math/sparse.md)
  - [BsrMatrix (CPU)](./math/bsr_cpu.md)
  - [BsrGpu (GPU)](./math/bsr_gpu.md)
  - [SpGEMM design](./math/spgemm.md)
- [Iterative solvers](./math/iterative.md)
- [Optimizers](./math/optim.md)
- [Graph coloring](./math/coloring.md)
- [Scan & 64-bit radix sort](./math/scan_sort.md)

---

# Compute & GPU

- [GPU runtime](./compute/runtime.md)
- [Storage abstractions](./compute/storage.md)
- [Kernel infrastructure](./compute/kernels.md)
- [Tile compute primitives](./compute/tile.md)
- [Persistent-buffer pattern](./compute/persistent.md)

---

# Geometry & Collision

- [Shape primitives](./geometry/shapes.md)
- [Convex hull builder](./geometry/hull.md)
- [BVH](./geometry/bvh.md)
- [Spatial hash](./geometry/spatial_hash.md)
- [Broad phase](./geometry/broad_phase.md)
- [Narrow phase](./geometry/narrow_phase.md)
  - [GJK](./geometry/gjk.md)
  - [EPA](./geometry/epa.md)
  - [MPR](./geometry/mpr.md)
- [Contact reduction](./geometry/contact_reduction.md)
- [Sparse grids (NanoVDB-equivalent)](./geometry/sparse_grid.md)
- [Marching cubes & SDF baking](./geometry/sdf.md)
- [Mesh decimation](./geometry/decimate.md)
- [Gaussian splats](./geometry/gaussian_splat.md)
- [Soft contact pipeline](./geometry/soft_contact.md)
- [Hydroelastic contact](./geometry/hydroelastic.md)

---

# Simulation Core

- [Data model](./sim/model.md)
- [ModelBuilder](./sim/builder.md)
- [Joints](./sim/joints.md)
  - [Revolute / Prismatic / Fixed / Free](./sim/joints_basic.md)
  - [Ball / D6 / Distance](./sim/joints_multi_dof.md)
  - [CABLE joint](./sim/joints_cable.md)
  - [Joint mimic](./sim/joints_mimic.md)
- [Articulation (Featherstone ABA)](./sim/featherstone.md)
- [CRBA / RNEA / Jacobian](./sim/spatial_kernels.md)
- [Inverse kinematics](./sim/ik.md)

---

# Solvers

- [Solvers overview](./solvers/overview.md)
- [Semi-implicit integrator](./solvers/semi_implicit.md)
- [XPBD](./solvers/xpbd.md)
- [VBD](./solvers/vbd.md)
- [MPM](./solvers/mpm.md)
  - [Explicit](./solvers/mpm_explicit.md)
  - [Semi-implicit](./solvers/mpm_semi_implicit.md)
  - [Implicit Bingham](./solvers/mpm_implicit.md)
- [Barrier-Newton contact (IPC)](./solvers/barrier_newton.md)

---

# Finite-Element Method

- [FEM overview](./fem/overview.md)
- [Mesh & function spaces](./fem/mesh.md)
- [Quadrature rules](./fem/quadrature.md)
- [Element matrices](./fem/elements.md)
- [Assembly](./fem/assembly.md)
- [Boundary conditions](./fem/bc.md)
- [Solvers](./fem/solvers.md)
- [Higher-order elements (P2)](./fem/p2.md)
- [Discontinuous Galerkin](./fem/dg.md)
- [Stokes & Navier-Stokes](./fem/stokes_ns.md)
- [Mixed (u, p) elasticity](./fem/mixed_elasticity.md)
- [APIC FEM-MPM coupling](./fem/apic.md)
- [Adjoint / differentiable FEM](./fem/adjoint.md)
- [Adaptive mesh refinement (AMR)](./fem/amr.md)
- [GPU FEM elasticity](./fem/elasticity_gpu.md)
- [Magnetostatics — H(curl) Nedelec](./fem/magneto.md)
- [Eddy currents (time-harmonic Maxwell)](./fem/eddy.md)
- [Multigrid (geometric, AMG, AMS)](./fem/multigrid.md)
- [Nonconforming meshes](./fem/nonconforming.md)
- [Signorini hard contact](./fem/signorini.md)
- [Coulomb friction](./fem/friction.md)
- [Phase-field fracture](./fem/phase_field.md)

---

# Sensors

- [Raycaster](./sensors/raycaster.md)
- [Sensor geometry snapshots](./sensors/scene.md)
- [Device-resident LiDAR batches](./sensors/lidar_batch.md)
- [TiledCamera](./sensors/tiled_camera.md)
- [GPU image processing](./sensors/image_processing.md)
- [GPU point-cloud processing](./sensors/cloud_processing.md)
- [Selected point-cloud export](./sensors/cloud_export.md)
- [GPU spatial bins and reductions](./sensors/cloud_grid.md)
- [GPU class and instance annotation](./sensors/annotation.md)
- [IMU](./sensors/imu.md)
- [Asynchronous sensor readback](./sensors/readback.md)
- [Sensor device loss and recovery boundaries](./sensors/device_lifecycle.md)
- [Leased GPU snapshots](./sensors/snapshot.md)
- [Direct leased sensor outputs](./sensors/device_results.md)
- [Simulation-time scheduling](./sensors/scheduler.md)
- [Contact sensor](./sensors/contact_sensor.md)
- [Frame transform](./sensors/frame_transform.md)

---

# Actuators & Control

- [Composable actuator pipeline](./actuators/pipeline.md)
- [PD / PID controllers](./actuators/pd_pid.md)
- [Neural controllers](./actuators/neural.md)
- [DC motor clamping](./actuators/dc_motor.md)
- [Muscle (Hill-type + SEE)](./actuators/muscle.md)

---

# Vehicle

- [molla-vehicle overview](./vehicle/overview.md)

---

# Differentiable Physics

- [Tape-based autograd](./autograd/tape.md)
- [Featherstone reverse adjoint](./autograd/featherstone.md)
- [XPBD / VBD adjoint](./autograd/xpbd_vbd.md)
- [MPM adjoint](./autograd/mpm.md)
- [FEM adjoint](./autograd/fem.md)
- [Raycast adjoint](./autograd/raycast.md)
- [Differentiable Gaussian splat renderer](./autograd/splat.md)
- [GPU cloth & softbody adjoints](./autograd/cloth_softbody_gpu.md)
- [Eulerian fluid adjoint](./autograd/stam_fluid.md)

---

# Viewer

- [Bevy adapter](./adapters/bevy.md)
- [Viewer overview](./viewer/overview.md)
- [Scene catalog (#1-124)](./viewer/scenes.md)

---

# Testing & Performance

- [Test strategy & Newton oracle](./testing/strategy.md)
- [CPU/GPU parity testing](./testing/parity.md)
- [Finite-difference gradient checks](./testing/fd_gradients.md)
- [Benchmarks](./testing/benchmarks.md)
- [GPU port status matrix](./testing/gpu_status.md)

---

# Roadmap

- [Stage history](./roadmap/stages.md)
- [Open items](./roadmap/open.md)
- [Out-of-scope](./roadmap/oos.md)

---

[About this book](./about.md)
