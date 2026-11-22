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

## Attribution policy

Keep this bibliography updated whenever an external paper, implementation,
asset, or technical reference informs new work. Identify what was used and
distinguish conceptual inspiration, dependencies, and copied or adapted
material. Record relevant revisions and preserve all required copyright,
license, and attribution notices. An acknowledgement alone is not permission
to reuse protected material.

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
[`coloring.rs`](crates/molla-math/src/coloring.rs), reused for disjoint
fiber-constraint dispatch batches,
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
physics algorithms credited above. The strand solver uses retained storage
buffers, color-batched compute dispatches, error-scoped resource setup,
and guarded state publication on this runtime.

## Elastic rods, fibers, cutting, and fracture

### [11] Stressful Tree Modeling / EvoEngine

**Li, Bosheng; Schwarz, Nikolas A.; Pałubicki, Wojtek; Pirk, Sören;
Michels, Dominik L.; and Benes, Bedrich. (2025).
_Stressful Tree Modeling: Breaking Branches with Strands_. SIGGRAPH
Conference Papers, article 157, 1–11.**
[DOI: 10.1145/3721238.3730745](https://doi.org/10.1145/3721238.3730745) ·
[Authors' project page](https://storage.googleapis.com/pirk.io/projects/stressful/index.html)

Research reference for representing woody material as coupled elastic
strands, progressive connection failure, partial attachment, and surfaces
that change with fracture. The requested Molla work distinguishes clean
blade-driven cuts from stress-driven breakage; this citation is not a claim
of reproducing the paper's complete model or performance.

The reduced apple-tree example distinguishes prescribed rough breaks from
load-driven breaking while dragging a material point. Its local convex
decomposition and staggered surfaces are an independent, bounded-detail
approximation, not the paper's fracture morphology predictor. GPU bond damage
selects loaded-joint failure; geometry refinement follows that event.

**Li, Bosheng, and contributors. _EvoEngine_ [Software].**
[Repository](https://github.com/edisonlee0212/EvoEngine) ·
[Reviewed license](https://github.com/edisonlee0212/EvoEngine/blob/4e06139a4245336fe635d337863e77cb72cb59b1/LICENSE)

User-supplied visual and conceptual reference for interactive tree fracture.
Reviewed revision: `4e06139a4245336fe635d337863e77cb72cb59b1`.
That revision is source-available under restrictive terms, not a permissively
licensed dependency. No EvoEngine code, shaders, assets, or documentation
are incorporated in Molla's strand implementation. Molla implements the
published concepts independently in Rust/WGSL; using protected upstream
material would require a separately verified permission/license basis.

### [12] Position and Orientation Based Cosserat Rods

**Kugelstadt, Tassilo, and Schoemer, Elmar. (2016).
_Position and Orientation Based Cosserat Rods_. Symposium on Computer
Animation.**
[Authors' publication page](https://animation.rwth-aachen.de/publication/0550/)

Reference for local material orientations, authored rest bend/twist, and
position-orientation coupling in elastic rods. Material-relative rest
quantities distinguish shape recovery from attraction to fixed world points.

### [13] XPBD

**Macklin, Miles; Müller, Matthias; and Chentanez, Nuttapong. (2016).
_XPBD: Position-Based Simulation of Compliant Constrained Dynamics_.
Motion in Games.**
[DOI: 10.1145/2994258.2994272](https://doi.org/10.1145/2994258.2994272) ·
[Authors' paper](https://matthias-research.github.io/pages/publications/XPBD.pdf)

Reference for compliance-scaled constraint solves and accumulated multipliers,
including force estimates used to qualify progressive failure. Finite
iteration counts still require convergence checks; the citation does not
establish timestep- or resolution-independent results for Molla.

The apple-tree Grab / bend tool also uses an independently implemented
compliant point constraint with accumulated multipliers, solved in WGSL with
a matching CPU oracle. It applies forces through the existing strand model,
not by teleporting the grabbed geometry.
The grab's three rows participate in the coupled body solve. Relative-target
viscous damping uses the implicit compliant formulation discussed in section
5, with an analytic backward-Euler spring/damper regression. The demo's
velocity/acceleration-limited mouse handle and per-substep target interpolation
are Molla-specific interaction choices, not claimed upstream implementations.
Its bounded-force extension increases effective point compliance when the
combined elastic/damping load reaches the configured grip limit. CPU and GPU
force-limit tests accompany this independently authored interaction policy.

### [14] Direct Position-Based Solver for Stiff Rods

**Deul, Crispin; Kugelstadt, Tassilo; Weiler, Marcel; and Bender, Jan.
(2018). _Direct Position-Based Solver for Stiff Rods_. Computer Graphics Forum.**
[Authors' publication page](https://animation.rwth-aachen.de/publication/0557/)

Reference for oriented rigid rod segments, compliant connections, and the
need to solve long stiff chains beyond local iteration alone. Its direct
solver is a reference, not an already integrated Molla implementation.

### [15] Parry

**Dimforge and contributors. _Parry_ [Software].**
[Source repository](https://github.com/dimforge/parry)

Existing geometry dependency used by Molla's collision and general collider
mass-property infrastructure. Convex fiber cells use independently implemented
tetrahedral moment integration to retain full inertia tensors during cuts;
neither this integration nor plane clipping is copied from Parry.

### [16] SO(3) rotation differentials

**Solà, Joan; Deray, Jeremie; and Atchuthan, Dinesh. (2018, revised 2021).
_A micro Lie theory for state estimation in robotics_.**
[arXiv:1812.01537](https://arxiv.org/abs/1812.01537)

Mathematical reference for rotation logarithms and inverse left Jacobians
used in the oriented fiber connections. Molla's implementation uses principal
relative rotation vectors and verifies their derivatives numerically; no
upstream C++ library implementation is incorporated.

### [17] Polygon clipping

**Sutherland, Ivan E., and Hodgman, Gary W. (1974).
_Reentrant Polygon Clipping_. Communications of the ACM, 17(1), 32–42.**
[DOI: 10.1145/360767.360802](https://doi.org/10.1145/360767.360802)

Algorithmic reference for successive half-space clipping of ordered polygon
vertices. Molla independently implements clipping for convex cell faces and
paired material-interface polygons, with its own cap construction, validation,
state transfer, and topology transactions. The apple-tree renderer also uses
half-space clipping to remove buried faces in intact branch junctions while
retaining exposed cut and fracture caps.

### [18] Polyhedral mass properties

**Mirtich, Brian. (1996). _Fast and Accurate Computation of Polyhedral
Mass Properties_. Journal of Graphics Tools, 1(2).**
[Author's publication and implementation page](https://people.eecs.berkeley.edu/~jfc/mirtich/massProps.html)

Related reference consulted for polyhedral volume, center-of-mass, and full
inertia integration. Molla's cut cells use independently derived tetrahedral
moments, not a port of Mirtich's boundary-integral reduction or C implementation.

### [19] Preconditioned conjugate gradients

**Shewchuk, Jonathan Richard. (1994). _An Introduction to the Conjugate
Gradient Method Without the Agonizing Pain_. Carnegie Mellon University.**
[Author's report](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

Algorithmic reference for the block-preconditioned conjugate-gradient solve,
stopping criteria, and explicit residual checks in the coupled strand solver.
Molla independently implements the matrix-free body-space system in WGSL;
the report and its illustrative code are not incorporated or redistributed.

### [20] glam

**The glam contributors. _glam: a simple and fast 3D math library for Rust_
[Software].**
[Source repository](https://github.com/bitshifter/glam-rs)

Existing vector, matrix, and quaternion dependency beneath `molla-math`,
including the oriented-fiber geometry and CPU reference solver. Molla's
small-angle rotation-log evaluation and constraint Jacobians are independent
implementations of the mathematics referenced in [16], not copied glam code.

### [21] egui and eframe

**Ernerfeldt, Emil, and contributors. _egui / eframe_ [Software].**
[Source repository](https://github.com/emilk/egui)

Cable Lab's immediate-mode controls, native window/event integration and
wgpu-backed display framework. The example uses the published Rust APIs;
its finite-cell rendering and blade interaction are Molla-specific code.
The showcase exporter also uses egui-wgpu's public offscreen renderer APIs
with the same studio geometry, lighting and frame labels. No upstream demo
code or artwork is copied into the capture implementation.

### [22] image

**The image-rs contributors. _image_ [Software].**
[Source repository](https://github.com/image-rs/image)

PNG encoding dependency for captures of the examples' actual rendered
framebuffers. Cable Lab is procedural; the separate apple-tree example uses
the skeleton data credited in [25] and [26].

### [23] Separating-axis tests

**Eberly, David. (2001; revised 2022). _Intersection of Convex Objects:
The Method of Separating Axes_. Geometric Tools.**
[Author's report](https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf)

Mathematical reference for discrete convex-cell contact using face normals
and edge cross products. The report is CC BY 4.0. Molla independently
implements the tests in Rust and WGSL; no report text or source code is
incorporated. This reference does not imply continuous collision detection
has been implemented for strands.

### [24] Contact manifolds

**Catto, Erin. (2007). _Contact Manifolds_. Game Developers Conference.**
[Author's slides](https://box2d.org/files/ErinCatto_ContactManifolds_GDC2007.pdf)

Algorithmic reference for reference/incident-face clipping and a small set
of surface witnesses. Molla reuses its existing Newton-inspired CPU contact
reducer and independently implements the corresponding WGSL operations.
No Box2D code, slide text, or artwork is incorporated or redistributed.

### [25] AppleQSM orchard data and method

**Qiu, Tian. (2024). _AppleQSM-Row13_, version 1 [Dataset]. Figshare.**
[Dataset DOI](https://doi.org/10.6084/m9.figshare.26215094.v1)
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

**Qiu, Tian, Wang, Tao, Han, Tao, Kuehn, Kaspar, Cheng, Lailiang, Meng, Cheng,
Xu, Xiangtao, Xu, Kenong, and Yu, Jiang. (2024). _AppleQSM: Geometry-Based 3D
Characterization of Apple Tree Architecture in Orchards_. Plant Phenomics,
6, Article 0179.** [Paper](https://doi.org/10.34133/plantphenomics.0179)

The separate `molla_apple_tree` example includes a derived `tree1` skeleton
reconstructed by Appleskel from this dataset. The original dataset remains
CC BY 4.0; Molla's code license does not replace that attribution obligation.
The example converts Z-up to Y-up, translates the root, smooths and resamples
centerlines, selects a bounded branch hierarchy, and constructs tapered
convex physics cells. Radius taper, elastic constants, and joint behavior are
illustrative rather than scientifically validated AppleQSM measurements.
No endorsement by the authors is implied.

### [26] Appleskel reconstruction

**Bresilla, Trim. _Appleskel_ [Software and derived reconstruction].**
[Source repository](https://github.com/bresilla/appleskel)

The `tree1_characterization.json` geometry in the local Appleskel pipeline
output is the apple-tree example's source. The local checkout was at
`9731e7ebd7ccf8d20c6fe761490fbcdd8e2fe83a`; the output itself is identified by
SHA-256 in `bin/molla_cables/data/ATTRIBUTION.md`. Its nearest-parent
centerline attachment, smoothing, and radius-taper approach is adapted in
the Rust example builder. No Blender/USD runtime, renderer assets, or Python
source files are redistributed or required by this example.

The separate leafy-tree example adapts the procedural ovate blade,
serrated margin, cupping/droop and alternate 144-degree leaf layout from
`examples/qsm_to_blender.py` (`_leaf_blade` and `generate_leaves`) in that
checkout. Its tapering leafy shoots on every retained branch and along
the trunk through the upper leader, leaf point masses, individual
petiole strengths, GPU spring/damper attachment failure and two-way force
feedback are Molla additions. The foliage is procedural, not measured
AppleQSM data. No generated raster leaf images or external texture assets
are included. The reduced leaf model has gravity, drag and ground contact;
it is not a calibrated botanical abscission or shell-physics model.

The attachment formulation uses standard mass/spring/damper mechanics;
NVIDIA Warp's [spring API documentation](https://nvidia.github.io/warp/v1.7/modules/sim.html)
and [cloth benchmark](https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth_warp.py)
were consulted as references. Molla's implicit attachment update, CPU
oracle and WGSL kernel are independently implemented rather than copied
from those examples.

### [27] WGSL screen-space derivatives

**W3C GPU for the Web Working Group. _WebGPU Shading Language_, `fwidth`.**
[Specification](https://www.w3.org/TR/WGSL/#fwidth-builtin)

API reference for pixel-footprint filtering of the studio renderer's procedural
wood grain. The WGSL implementation is independent; no texture assets are used.

### [28] Midpoint rigid-body rotation and Cayley reconstruction

**Abdulle, Assyr; Cohen, David; Vilmart, Gilles; and Zygalakis, Konstantinos C.
(2011). _High weak order methods for stochastic differential equations based
on modified equations_. MATHICSE Technical Report 25.2011, pp. 18–19.**
[Authors' report](https://www.epfl.ch/labs/mathicse/wp-content/uploads/2018/10/25.2011_AA-DC-GV-KZ.pdf)

Reference for the body-momentum midpoint equation and matching quaternion
Cayley update. Molla uses the deterministic, zero-noise equations, solved by
Newton iterations in Rust and WGSL for free cell rotation; it does not
implement the paper's stochastic or higher-order modified-equation methods.
No source code or report text is incorporated.

### [29] Compensated sums and dot products

**Ogita, Takeshi; Rump, Siegfried M.; and Oishi, Shin'ichi. (2005).
_Accurate Sum and Dot Product_. SIAM Journal on Scientific Computing,
26(6), 1955–1988.** [DOI](https://doi.org/10.1137/030601818) ·
[Authors' paper](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)

Numerical reference for two-component sums and fused-multiply-add product
remainders. The strand PCG implementation independently uses these building
blocks in WGSL for solution accumulation, matrix products and true residuals.
This is compensated f32 arithmetic, not a claim of native f64 availability
or of the paper's IEEE error bounds on every WebGPU implementation.

### [30] FFmpeg and x264

**The FFmpeg developers. _FFmpeg_ [Software].**
[Project](https://ffmpeg.org/) ·
[Raw-video and MP4 format documentation](https://ffmpeg.org/ffmpeg-formats.html)

**VideoLAN and x264 developers. _x264_ [Software].**
[Project and source](https://www.videolan.org/developers/x264.html)

The apple-tree showcase pipes rendered RGBA frames to the installed FFmpeg
executable and selects its libx264 encoder for H.264/yuv420p MP4 output with
fast-start metadata. Neither tool is incorporated into Molla's physics or
copied into this repository. Encoding tools retain their own licenses.

### [31] Three-dimensional airflow and rotor-wake references

**Bridson, Robert, and Müller-Fischer, Matthias. (2007). _Fluid Simulation_.
SIGGRAPH course notes.**
[Course notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)

Numerical reference for the independent Rust/WGSL staggered-grid airflow
solver: semi-Lagrangian advection, pressure projection and fluid boundaries.
No course source code is incorporated.

**Bangura, Moses, Melega, Marco, Naldi, Roberto, and Mahony, Robert. (2016).
_Aerodynamics of Rotor Blades for Quadrotors_.**
[Paper](https://doi.org/10.48550/arXiv.1601.00733)

Reference for rotor momentum, thrust and torque modeling. The orchard work
uses regularized actuator disks and existing propeller devices, not a full
implementation of this paper's blade-element models.

**Bauersfeld, Leonard, Muller, Koen, Ziegler, Dominic, Coletti, Filippo, and
Scaramuzza, Davide. (2024). _Robotics meets Fluid Dynamics: A Characterization
of the Induced Airflow below a Quadrotor as a Turbulent Jet_. IEEE Robotics
and Automation Letters.** [Paper](https://arxiv.org/abs/2403.13321)

Experimental reference for rotor downwash and the limits of time-averaged
far-field jet approximations. Molla does not claim their experimental
validation or use that far-field approximation as a near-canopy solver.

**Fu, Shenghui, et al. (2025). _An Improved Hierarchical Leaf Density Model
for Spatio-Temporal Distribution Characteristic Analysis of UAV Downwash
Air-Flow in a Fruit Tree Canopy_. Agronomy, 15(8), 1867.**
[DOI: 10.3390/agronomy15081867](https://doi.org/10.3390/agronomy15081867)

Research context for canopy-dependent airflow, not a reproduction of the
paper's CFD or field measurements. The implementation and acceptance scope
are recorded in [`plans/DRONE_ORCHARD.md`](plans/DRONE_ORCHARD.md).
These additional references were reviewed on 2026-11-18.

The orchard's exterior particle queries add an original Molla approximation:
the sampled boundary velocity minus its background field is continued outside
the grid with a Gaussian distance decay, then added to the local background
gusts. This removes the former one-cell wake cutoff without extending the
pressure solve. It is not a published wake model or a divergence-free exterior
CFD solution. The in-grid solver and its source attribution remain unchanged.

### [32] Innerscene quadcopter visualization model (legacy)

**Innerscene Corp. (2026). _Quadcopter with flight electronics_.**
[Original model and CC0 dedication](https://www.innerscene.com/tools/library/3d-parts/quadcopter-with-flight-electronics-a25968c1).

The repository retains the original GLB under CC0 1.0 Universal.
The legacy loader adapts axis orientation and visual scale, aligns the propeller
disks with the simulated rotors, and drives their animation from motor speeds.
The current orchard displays original Molla geometry instead; see [33].
This is a visualization asset, not manufacturer CAD or a validated source
of aerodynamic coefficients. Full provenance and checksum are recorded in
[`data/drone/ATTRIBUTION.md`](bin/molla_cables/data/drone/ATTRIBUTION.md).

### [33] Freely falling plates and leaf aerodynamics

**A. Andersen, U. Pesavento and Z. Jane Wang (2005). _Unsteady aerodynamics
of fluttering and tumbling plates._ Journal of Fluid Mechanics 541, 65–90.**

- DOI: <https://doi.org/10.1017/S002211200500594X>
- Authors' paper: <https://dragonfly.tam.cornell.edu/publications/2005_JFM_Andersen_Pesavento_Wang_a.pdf>

**A. Andersen, U. Pesavento and Z. Jane Wang (2005). _Analysis of transitions
between fluttering, tumbling and steady descent of falling cards._ Journal of
Fluid Mechanics 541, 91–104.**

- DOI: <https://doi.org/10.1017/S0022112005005847>
- Authors' paper: <https://dragonfly.tam.cornell.edu/publications/2005_JFM_Andersen_Pesavento_Wang_b.pdf>

The detached-leaf CPU/WGSL solver implements the latter paper's thin-plate
Kirchhoff added mass/inertia, translational and rotational circulation,
angle-dependent drag and dissipative torque (equations 3.1–3.9). The planar
limit is checked against independently coded dimensionless equations 4.7–4.9
and 5.1–5.3. Coefficients are the paper's illustrative values, not measured
apple-leaf parameters. The three-dimensional extension uses a blade-normal
added-mass tensor and two in-plane rotational circulation components; it is
Molla's reduced approximation, not a published or validated 3D leaf model.
RK4 integration, elliptical blade inertia and frictional ground contacts
replace the earlier purely resistive surface-panel approximation. No source
code or figures were copied; no DNS, wake-resolved flutter or experimental
calibration is claimed.

The revised agricultural quadcopter mesh is original procedural Molla geometry
in `bin/molla_cables/src/drone_asset.rs`; it replaces the [32] visualization,
whose source, loader and attribution remain available. PX4's
[Gazebo model catalogue](https://docs.px4.io/main/en/sim_gazebo_gz/vehicles)
and [X500 model](https://github.com/PX4/PX4-gazebo-models/tree/main/models/x500_base)
were consulted while evaluating alternatives; no PX4 mesh was imported.
The new mesh is not manufacturer CAD and does not calibrate flight coefficients.

### [34] Shader hash and soft-vapor visualization

**David Hoskins (2014), *Hash without Sine*.**
[Original shader](https://www.shadertoy.com/view/4djSRW).
The [LYGIA random-function source](https://github.com/patriciogonzalezvivo/lygia/blob/main/generative/random.glsl)
credits David Hoskins, Patricio Gonzalez Vivo and Inigo Quilez and identifies
the Hoskins hash as MIT licensed, copyright 2014 David Hoskins.

The hash in `bin/molla_cables/src/studio_vapor.wgsl` adapts the sinless
fractional/dot-product hash construction to WGSL with different constants.
The retained [license notice](bin/molla_cables/data/HASH_NOISE_LICENSE.md)
covers that attribution. The surrounding vapor renderer is Molla display code:
animated value noise and Gaussian alpha falloff on depth-tested ribbons of
sampled air trajectories. It is not a reproduced vapor, condensation, or
volumetric-fluid model; the airflow and leaf-motion references above remain
the physics sources.

### [35] Boundary-aware curl fields for mixed winds

**Robert Bridson, Jim Hourihan and Marcus Nordenstam (2007).
_Curl-Noise for Procedural Fluid Flow_. ACM Transactions on Graphics 26(3), 46.**
[Authors' paper](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf).

Molla adapts the curl-of-a-vector-potential construction and boundary modulation
to six analytically differentiated, evolving trigonometric modes in CPU Rust
and WGSL. The orchard adds this synthetic subgrid velocity to its resolved
near-field queries; it is not the paper's Perlin implementation, resolved
turbulence or calibrated atmospheric data. No source code or figures were copied.
The curved rigid-leaf contact geometry and offset aerodynamic center are
original reduced Molla extensions to [33], not results validated by those papers.

### [36] Multirotor trajectory tracking and attitude control

**PX4 contributors. *Controller Diagrams* and *Jerk-limited Type Trajectory for
Multicopters*.** [Control architecture](https://docs.px4.io/main/en/flight_stack/controller_diagrams),
[trajectory documentation](https://docs.px4.io/main/en/config_mc/mc_jerk_limited_type_trajectory).
Reviewed 2026-11-18. These describe cascaded position/velocity/attitude control,
acceleration feed-forward and bounded-jerk motion references.
The [PX4 parameter reference](https://docs.px4.io/v1.13/en/advanced_config/parameter_reference#multicopter-attitude-control)
also documents reduced yaw priority. Molla's original bounded mixer preserves
collective thrust, then roll/pitch torque, then uses remaining capacity for yaw;
it is not PX4's sequential-desaturation implementation.

**Taeyoung Lee, Melvin Leok and N. Harris McClamroch (2010; revised 2011).
*Control of Complex Maneuvers for a Quadrotor UAV using Geometric Methods on
SE(3).*** [Author manuscript, arXiv:1003.2005](https://arxiv.org/abs/1003.2005).
Reference for rigid-body force/attitude geometry and SO(3) attitude error.

Molla's `airflow/trajectory.rs` implements an original seven-phase, rest-to-rest
S-curve and feeds position, velocity and acceleration into `devices/copter.rs`.
Individual rate-limited propellers apply rigid-body forces and reaction torques;
the airflow receives the opposite measured rotor wrench in its actual orientation.
This is not PX4 code/SITL or a full reproduction of Lee et al.'s controller or
stability guarantees. Motor/thrust/drag/inertia values remain illustrative rather
than identified from flight data. The revised survey-drone mesh is original
procedural Molla geometry, not manufacturer CAD. No upstream code/assets copied.

### [37] DJI Mavic 3 visualization asset

**llirikslon. *DJI Mavic 3*.**
[Original model](https://sketchfab.com/3d-models/dji-mavic-3-c5a5abae1dea468ab73b1bdc7d616fa6),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The 12,701-triangle derivative was reduced and converted by **Lexik and Krutnet**
and distributed in [Skyfront, pinned source and attribution](https://github.com/Lexik-mqslll/Skyfront/tree/31f4791441329e969968d3793fd3f41b5e8c516b).
Only that separately licensed model is imported, not Skyfront software.

Molla preserves UV textures and smooth normals, fits the visual hubs and blade
radii to the existing simulated rotors, and drives animation from actual motor
phase. This replaces the procedural mesh described in [33]/[36]; it does not
change the flight model or claim calibrated DJI flight performance. Full credits,
checksum, exact source path and modifications are retained in
[MAVIC_ATTRIBUTION.md](bin/molla_cables/data/drone/MAVIC_ATTRIBUTION.md).

### [38] Agricultural sprayer layout and droplet drag

**DJI Agriculture. *AGRAS T25: product overview and specifications*.**
[Product](https://ag.dji.com/t25), [specifications](https://ag.dji.com/t25/specs),
reviewed 2026-11-18. Used as a visual reference for a central spray tank, folding
arms, landing frame and downward application hardware. The AG-mode geometry is
original procedural Molla artwork, not downloaded DJI CAD or a T25 replica. It
uses the existing compact four-rotor dynamics, not the manufacturer's payload,
dimensions or application-rate specifications. No DJI endorsement is implied.

**OpenFOAM Foundation. *SchillerNaumannDragForce*, OpenFOAM 7 (2019).**
[Primary implementation reference](https://cpp.openfoam.org/v7/SchillerNaumannDragForce_8C_source.html).
Consulted for the standard Schiller–Naumann spherical-particle drag correlation
and its high-Reynolds-number continuation. Molla's water droplet integrator is an
independent Rust implementation of the correlation, not copied OpenFOAM code.
It integrates gravity and prescribed local air drag with exponential substeps.
No evaporation, breakup, coalescence, pesticide chemistry, tank slosh, nozzle
recoil or deposition-induced leaf loading is modeled. Particle parcels represent
many droplets; their visible billboard footprint is enlarged for readability.
Spray deposition uses swept intersections with moving elliptical leaf patches
and the ground. These are illustrative water-spray visuals, not calibrated crop
coverage, dosing advice or an application recommendation. Flight control and
shared GPU air references remain [32]–[36].

### [39] Nozzle pressure and flow scaling

**TeeJet Technologies. *Technical Information*, “Information About Spray Pressure”.**
[Manufacturer reference](https://www.teejet.com/en-us/-/media/dam/agricultural/usa/sales-material/catalog/technical_information.pdf),
reviewed 2026-11-19. The AG example uses square-root pressure scaling for nozzle
flow and exit speed, with a relative pressure multiplier of 27 and relative nozzle
flow coefficient of 1/3. The initial
30–60 micrometre diameter distribution and increased parcel sampling rate are
explicit demo settings, not a calibrated droplet-size law or manufacturer's
nozzle recommendation. No hardware pressure in bar/psi is specified. Existing
one-way water coupling and fixed aircraft-mass limitations remain unchanged.
The additional 2x nozzle-axis velocity gain is an authored directional injection
condition, independent of this base pressure scaling; it does not specify a
calibrated pressure increase. Lateral injection velocity and total flow stay fixed.
The manufacturer's [droplet size and drift information](https://www.teejet.com/en/-/media/dam/agricultural/usa/sales-material/catalog/cat52-m_14-technical_information.pdf)
also informs the distinction between fine, air-carried mist and coarse droplets.
Motion remains computed from gravity and sampled local-air drag, not animation;
Gaussian parcel density footprints are an illustrative rendering choice, not an
optical-scattering or experimentally validated atomization model.
The subsequent settling preset uses 160–240 micrometre droplets with the same
gravity, water density, shared wind, flow and fine density-kernel rendering.
This increases gravitational settling relative to drag without suppressing
updrafts. It is an authored size distribution, not a manufacturer recommendation
or a claim that visible fog consists of physically 30–60 micrometre drops.
The primary drag reference in [38] was rechecked on 2026-11-19; no new upstream
code was copied for this adjustment.

### [40] Stochastic turbulent dispersion of spray

**Ansys. *Fluent Beta Features Manual*, 2025 R1, section 14.11,
“Continuous Random Walk Dispersion Model”.**
[Primary theory reference](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/fluent_beta_doc/flu_beta_dpm_crw.html).
**OpenFOAM Foundation. *StochasticDispersionRAS*, OpenFOAM 12 (2024).**
[Primary implementation reference](https://cpp.openfoam.org/v12/StochasticDispersionRAS_8C_source.html).
Reviewed 2026-11-19 for the use of fluctuating local air velocity in the particle
drag force balance. Molla independently implements an isotropic Ornstein–Uhlenbeck
velocity process with exponential correlation, not either product's full model.
No source code was copied. The demo prescribes per-component RMS fluctuation at
12.5% of sampled air speed and a 0.12 s correlation time; it does not solve for
turbulent kinetic energy/dissipation or claim calibrated subgrid turbulence.
The 84-degree base cone and subsequent axial focusing are authored injection conditions. Droplet
trajectories still integrate air drag and gravity; renderer size is unchanged.

### [41] Agricultural aircraft proportions and construction references

**DJI Agriculture. *AGRAS T25*, product photographs and specifications.**
[Product](https://ag.dji.com/t25), [specifications](https://ag.dji.com/t25/specs).
**XAG. *P100 Pro Agricultural Drone*, product photographs and specifications.**
[Product](https://www.xa.com/en/p100pro),
[specifications](https://www.xa.com/en/p100pro/p100prospecs).
Reviewed 2026-11-19. Both are four-rotor agricultural platforms. Their low-profile
chassis, modular battery/tank arrangement, exposed arm joints, motor housings,
landing frames and plumbing informed the original procedural model in
`bin/molla_cables/src/ag_drone.rs`. The new mesh uses restrained industrial colors,
separate matte and metallic shading, tapered propellers and smaller mechanical
details instead of the earlier domed canopy and decorative green stripe.
No manufacturer CAD, texture or photograph is redistributed as a Molla asset.
This is a compact reference-inspired design, not a dimensional replica, branded
product model or calibrated simulation of either aircraft. Existing physical
rotor centers, radii, motor phases, nozzle locations, mass, flight controller and
spray/leaf parameters are preserved. Rotor number is not merely a visual choice:
the four visible rotors correspond to the four simulated propulsion devices.
The subsequent square-boom option is an original user-requested layout, not a
DJI/XAG feature reproduction: four nozzles at the corners of a centered 0.56 m
square. Its mesh and physical emission share the same body-space coordinates.
Total flow, nozzle velocity and droplet parameters are unchanged; the spatial
spray footprint changes with the relocated outlets.

### [42] Scanned leaf and bark textures

**ambientCG (Lennart Demes). *Leaf001*, *LeafSet001*, *LeafSet003*, *LeafSet014*,
*LeafSet022*, *LeafSet024*, *Bark003*, *Bark004* and *Bark005*.**
[ambientCG](https://ambientcg.com),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), downloaded
2026-11-19. The 33 photoscanned leaves are cut out of their atlases and
resampled onto the simulated blade outline; color, opacity and OpenGL normal
maps are kept. The three bark scans (scaly trunk, fissured limb, smooth young
shoot) are blended by wood-cell radius with cylindrical coordinates that follow
the bond graph. Each leaf draws one scanned layer, optionally mirrored, and
takes part of its palette from a second layer. Textures affect rendering only;
blade area, contact, droplet capture and wood mechanics are unchanged. Hashes,
the exact files and processing steps are recorded in
[ATTRIBUTION.md](bin/molla_cables/data/foliage/ATTRIBUTION.md).

### [43] Attached leaves on elastic petioles

**Vogel, S. (1992). *Twist-to-bend ratios and cross-sectional shapes of petioles
and stems*. Journal of Experimental Botany 43(11), 1527–1532.**
[doi:10.1093/jxb/43.11.1527](https://doi.org/10.1093/jxb/43.11.1527).
Consulted for petioles being much softer in torsion than in bending (single-digit
EI/GJ ratios); Molla uses EI/GJ = 4.

**Tadrist, L., Saudreau, M., Hémon, P., Amandolese, X., Marquier, A., Leclercq, T.
and de Langre, E. (2018). *Foliage motion under wind, from leaf flutter to branch
buffeting*. Journal of the Royal Society Interface 15, 20180010.**
[doi:10.1098/rsif.2018.0010](https://doi.org/10.1098/rsif.2018.0010).
Consulted for individual leaf flutter and reconfiguration as the dominant foliage
motions in moderate wind.

**Shaw, R. H. and Schumann, U. (1992). *Large-eddy simulation of turbulent flow
above and within a forest*. Boundary-Layer Meteorology 61, 47–64.**
[doi:10.1007/BF02033994](https://doi.org/10.1007/BF02033994).
Consulted for representing vegetation as a drag layer per unit leaf area; later
canopy simulations commonly use drag coefficients near 0.2, far below isolated
plates. Molla's sheltering fraction applies that reduction to attached blades.

Attached orchard leaves are six-degree-of-freedom rigid blades. The existing
translational spring-damper ties the petiole joint to its twig and carries the
abscission load; elastic bending (2.5–4 Hz about the joint), torsion, structural
damping and a gravity preload act relative to the twig-carried rest frame. Four
blade elements use the APW plate coefficients of [33] with a centre of pressure
moving from quarter chord to mid-chord with incidence, scaled by a 0.3 sheltering
fraction. Free leaves keep the unsheltered free-flight model. This is an original
Molla implementation (CPU oracle and WGSL kernel); no upstream code is copied.
Frequencies, sheltering and the abscission strength scale are calibrated choices,
not apple-leaf measurements; the strength scale reproduces the previous mission's
shed count.

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
