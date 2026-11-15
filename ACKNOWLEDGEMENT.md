# Acknowledgement and References

Molla builds on the algorithms, designs, reference implementations, and tests
of the projects cited below. **ZOZO's Contact Solver (`ppf-contact-solver`) is
the primary source acknowledged by this project**, particularly for its
barrier-based contact and deformable-simulation work.

Molla adapts and extends these ideas in Rust, with GPU implementations written
in WGSL for the `wgpu` runtime and CPU reference implementations. Its data
layouts, numerical methods, precision choices, resource management, and
integration interfaces differ from the upstream projects. These citations
credit that lineage; they do not imply a complete port, equivalent results,
or that Molla links or embeds each upstream engine.

## Primary source

### [1] ZOZO's Contact Solver / PPF

**Ando, Ryoichi. (2024). _ZOZO's Contact Solver_ [Software]. ZOZO, Inc.**
[Source repository](https://github.com/st-tech/ppf-contact-solver) ·
[Upstream citation metadata](https://github.com/st-tech/ppf-contact-solver/blob/main/CITATION.cff)

Primary reference for Molla's barrier-Newton contact backend, CCD-constrained
stepping, strain-limiting approach, and contact-rich deformable examples.
Molla's adaptations are in
[`barrier_contact.rs`](crates/molla-solvers/src/barrier_contact.rs),
[`barrier_solver.rs`](crates/molla-solvers/src/barrier_solver.rs), and
[`barrier_newton.rs`](crates/molla-kernels/src/barrier_newton.rs).
The implementation is PPF-inspired, not an exact reproduction of upstream's
material models or stiffness and strain formulations.

**Related paper:** Ando, Ryoichi. (2024). _A Cubic Barrier with
Elasticity-Inclusive Dynamic Stiffness_. ACM Transactions on Graphics, 43(6).
[DOI: 10.1145/3687908](https://doi.org/10.1145/3687908).

## Physics and simulation sources

### [2] NVIDIA PhysX

**NVIDIA Corporation and contributors. _NVIDIA PhysX SDK_ [Software].**
[Source repository](https://github.com/NVIDIA-Omniverse/PhysX)

Reference for rigid-body simulation and constraint/contact solver behavior.
For example, Molla's [`XPBD solver`](crates/molla-solvers/src/xpbd.rs)
explicitly references Newton/PhysX impulse-capping behavior. This citation
credits PhysX as a physics reference, not as an embedded Molla dependency
or a claim that the entire SDK has been ported to WGSL.

### [3] Newton

**The Newton Contributors. (2025). _Newton: GPU-accelerated physics
simulation for robotics and simulation research_ [Software].**
[Source repository](https://github.com/newton-physics/newton) ·
[Upstream citation metadata](https://github.com/newton-physics/newton/blob/main/CITATION.cff)

Newton, developed with contributions from NVIDIA and its collaborators, is
a major source for Molla's simulation architecture, solver organization,
joint conventions, and test fixtures. In particular, Molla's
`Model` / `State` / `Control` / `Contacts` organization follows Newton's
design. See the [architecture introduction](book/src/introduction.md) and
[`oracle_newton.rs`](crates/molla/tests/oracle_newton.rs).

### [4] NVIDIA Warp

**Macklin, Miles. (2022). _Warp: A High-performance Python Framework for GPU
Simulation and Graphics_. NVIDIA GPU Technology Conference (GTC).**
[Source repository](https://github.com/NVIDIA/warp) ·
[Upstream citation metadata](https://github.com/NVIDIA/warp/blob/main/CITATION.cff)

Source for compute-kernel patterns, spatial algebra, simulation examples,
and correctness fixtures adapted to Rust and WGSL. Molla does not require
Warp's Python/CUDA runtime for these implementations. See
[`featherstone.rs`](crates/molla-solvers/src/featherstone.rs),
[`oracle_newton.rs`](crates/molla/tests/oracle_newton.rs), and the
[existing lineage notes](book/src/about.md).

### [5] Helios / Helios3D

**Bailey, Brian N., and the Plant Simulation Lab contributors.
_Helios: 3D Plant and Environmental Modeling Framework_ [Software].**
[Source repository](https://github.com/PlantSimulationLab/Helios) ·
[Documentation](https://plantsimulationlab.github.io/Helios/)

Source for terrestrial and aerial LiDAR metadata conventions, point-cloud
records and PTX layout, synthetic annotation interfaces, and solar-position
API organization. Molla adapts these interfaces and conventions to its own
sensor and compute infrastructure. See
[`lidar.rs`](crates/molla-sensors/src/lidar.rs),
[`pointcloud.rs`](crates/molla-sensors/src/pointcloud.rs),
[`dataset.rs`](crates/molla-sensors/src/dataset.rs), and
[`solar.rs`](crates/molla-math/src/solar.rs).
This acknowledgement does not claim that Molla implements Helios's full
plant or environmental modeling suite.

**Related paper:** Bailey, Brian N. (2019). _Helios: A Scalable 3D Plant and
Environmental Biophysical Modeling Framework_. Frontiers in Plant Science,
10, 1185. [DOI: 10.3389/fpls.2019.01185](https://doi.org/10.3389/fpls.2019.01185).

### [6] Project Chrono

**The Project Chrono contributors. _Project Chrono_ [Software].**
[Source repository](https://github.com/projectchrono/chrono) ·
[Project website](https://projectchrono.org/)

Additional reference codebase for multibody, vehicle, deformable-body, and
terrain simulation, available in this development checkout under
`xtra/chrono`. It is credited here as a reference, not as evidence of a
specific port or a linked runtime dependency.

## Additional implementation and interface references

### [7] Webots

**Cyberbotics and contributors. _Webots_ [Software and documentation].**
[Sensor guide](https://cyberbotics.com/doc/guide/sensors) ·
[Actuator guide](https://cyberbotics.com/doc/guide/actuators)

Reference for sensor and actuator coverage and device interfaces. The
relationship is documented in Molla's
[generic sensors](book/src/sensors/generic_sensors.md) and
[device actuators](book/src/actuators/devices.md) chapters.

### [8] Paul Bourke's marching-cubes reference

**Bourke, Paul. _Polygonising a scalar field_ [Technical reference].**
[Reference and tables](https://paulbourke.net/geometry/polygonise/)

The source explicitly credited for the marching-cubes edge and triangle
lookup tables in
[`marching_cubes.rs`](crates/molla-geometry/src/marching_cubes.rs).

### [9] Point Cloud Library (PCL)

**The Point Cloud Library contributors. _Point Cloud Library_ [Software].**
[Source repository](https://github.com/PointCloudLibrary/pcl) ·
[PLY implementation](https://github.com/PointCloudLibrary/pcl/blob/master/io/include/pcl/io/ply/ply.h)

Format-compatibility reference for PLY scalar types and binary byte ordering,
as cited in Molla's [point-cloud export documentation](book/src/sensors/cloud_export.md).
This is an interoperability reference, not a claim that Molla embeds PCL.

## Portable compute foundation

### [10] wgpu

**The gfx-rs and wgpu contributors. _wgpu_ [Software].**
[Source repository](https://github.com/gfx-rs/wgpu)

Molla's GPU runtime dependency. `wgpu` and WGSL provide the execution layer
for Molla's portable adaptations; they are distinct from the upstream
physics algorithms credited above.

## Citation and attribution notes

Suggested short attribution:

> Molla is based on and extends ideas, algorithms, reference implementations,
> and tests from ZOZO's Contact Solver (PPF), NVIDIA PhysX, Newton, NVIDIA
> Warp, and Helios, with additional references acknowledged in this document.
> PPF is the primary acknowledged source for the barrier-contact work.
> The implementations are adapted for Rust, WGSL, and wgpu rather than being
> unchanged versions of the upstream engines.

For research using a particular method, cite its original paper as well as
the relevant software. Software titles and authors above follow upstream
citation metadata where available. Record the version or commit actually
used in an experiment; the bibliography does not pin Molla to those
projects' current default branches. Source links were reviewed on
2026-11-15.

These acknowledgements do not replace upstream copyright notices, license
texts, or attribution requirements for redistributed code or assets.
References, dependencies, and any copied or modified material retain their
respective licensing requirements; this document does not relicense them.
